#!/usr/bin/env python
"""Train Stage-2 compactified flow maps with LESD/EESD/PESD objectives.

The default recipe is the pretrained-anchor variant from EqFM Eq. 76:
freeze a Stage-1 autonomous EqM checkpoint as ``b_pre`` and train a new
SiT-style network ``v_sigma`` such that ``X_sigma(y)=y+sigma*v_sigma(y)``.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.elastic.multiprocessing.errors import record
from torch.optim import AdamW

import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from eqfm.eval_utils import autonomous_forward, load_stage1_model, maybe_autocast  # noqa: E402
from eqfm.flowmap import flow_map  # noqa: E402
from eqfm.models.sit.models import SiT_models  # noqa: E402
from eqfm.sampling import save_grid  # noqa: E402
from eqfm.stage2_losses import Stage2FlowMapLoss, Stage2LossConfig  # noqa: E402
from eqfm.training import EMA  # noqa: E402
from eqfm.lr_plateau import LossPlateau  # noqa: E402
from eqfm.stage2_wandb import (  # noqa: E402
    S_OPT, S_VAL, WindowStats, add_glossary, define_wandb_metrics, eval_payload, load_logged,
    next_eval_group, pool_stage2, save_logged,
)
from eqfm.vae import LatentEncoder  # noqa: E402


DEFAULT_TRAIN_SHARDS = (
    "pipe:gcloud storage cat "
    "gs://cmu-gpucloud-jerryhua/imagenet/wds/train/train-{000000..001280}.tar"
)


def setup_distributed() -> tuple[int, int, int]:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        # Ranks can reach the first collective (DDP param broadcast) many minutes
        # apart when each loads two 5 GB checkpoints from contended NFS; the
        # default 10-min NCCL watchdog then SIGABRTs the job; but a hung communicator
        # must not hold GPUs for hours either. Allow 30 min.
        from datetime import timedelta
        dist.init_process_group(backend="nccl", timeout=timedelta(minutes=30))
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        torch.cuda.set_device(local_rank)
        return rank, local_rank, world_size
    return 0, 0, 1


def cleanup_distributed() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def unwrap(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DDP) else model


def cycle(loader):
    while True:
        for batch in loader:
            yield batch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])

    p.add_argument("--results-dir", type=Path, required=True)
    p.add_argument("--run-name", default=None)
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--keep-last-checkpoints", type=int, default=5)

    p.add_argument("--shards", default=DEFAULT_TRAIN_SHARDS)
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=4,
                   help="Per-GPU batch size.")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--shuffle-buffer", type=int, default=2000)
    p.add_argument("--synthetic-data", action="store_true", default=False,
                   help="Use random latents/labels instead of ImageNet+VAE. "
                        "Useful for L40S correctness/perf smoke tests.")

    p.add_argument("--teacher-ckpt", type=Path, default=None,
                   help="Stage-1 EqM checkpoint used as frozen b_pre. "
                        "Required unless --fake-teacher is set.")
    p.add_argument("--teacher-weights", choices=["ema", "raw"], default="ema")
    p.add_argument("--fake-teacher", action="store_true", default=False,
                   help="Use b_pre(x)=-fake_teacher_scale*x for synthetic smoke.")
    p.add_argument("--fake-teacher-scale", type=float, default=0.25)

    p.add_argument("--model", choices=sorted(SiT_models.keys()), default="SiT-XL/2")
    p.add_argument("--latent-size", type=int, default=32)
    p.add_argument("--vae-id", default="stabilityai/sd-vae-ft-ema")
    p.add_argument("--vae-cache-dir", default=None)

    p.add_argument("--objective",
                   choices=["lagrangian", "eulerian", "semigroup", "meanflow", "scaled_meanflow", "lag_semi",
                            "shifted_lagrangian", "shifted_meanflow"],
                   default="lagrangian",
                   help="lag_semi: Lagrangian (LESD) off-diagonal + --semi-weight * PESD "
                        "semigroup term, the MNIST-best recipe.")
    p.add_argument("--semi-weight", type=float, default=1.0,
                   help="lag_semi only: weight of the PESD semigroup term.")
    p.add_argument("--sigma-sampler", choices=["uniform", "canonical"], default="uniform",
                   help="Distribution of the off-diagonal flow time sigma in [sigma_min, sigma_max].")
    p.add_argument("--lr-decay-from", type=int, default=None,
                   help="Step from which the lr decays (cosine) from --lr to --lr-final at --steps.")
    p.add_argument("--lr-final", type=float, default=0.0)
    p.add_argument("--interp-time-sampler",
                   choices=["uniform", "canonical", "importance"],
                   default="canonical",
                   help="Distribution for interpolant points I_t used as "
                        "inputs to the diagonal/off-diagonal losses. "
                        "'canonical' is the paper-faithful sampler form "
                        "p(t) proportional to 1/(1-t), truncated by "
                        "--eps-train. 'uniform' is only an ablation.")
    p.add_argument("--eps-train", type=float, default=1e-3)
    p.add_argument("--importance-k", type=float, default=2.0)
    p.add_argument("--sigma-min", type=float, default=1e-4)
    p.add_argument("--sigma-max", type=float, default=1.0 - 1e-3)
    p.add_argument("--diag-weight", type=float, default=1.0)
    p.add_argument("--distill-weight", type=float, default=1.0)
    p.add_argument("--terminal-weight", type=float, default=1.0)
    p.add_argument("--terminal-teacher-norm-weight", type=float, default=0.0,
                   help="Weight for ||b_teacher(X_1_student(z), y)||^2. "
                        "This is a cheap terminal stationarity regularizer "
                        "that uses a frozen teacher with input gradients.")
    p.add_argument("--terminal-student-norm-weight", type=float, default=0.0,
                   help="Weight for ||v_student(X_1_student(z), sigma=0, y)||^2. "
                        "Usually keep this <= the teacher norm weight.")
    p.add_argument("--terminal-norm-max-batch", type=int, default=0,
                   help="If >0, compute terminal norm losses on at most this "
                        "many samples per microbatch to save memory.")
    p.add_argument("--selfcorr-weight", type=float, default=0.0,
                   help="Weight for || X_1(sg X_1(z)) - sg C(z) ||^2 on the same noise inputs z the "
                        "terminal-norm terms use. Teacher-free replacement for "
                        "--terminal-teacher-norm-weight: instead of asking ||b(X_1(z))||->0 it asks a "
                        "second application of X_1 to land on the reference chain C(z).")
    p.add_argument("--selfcorr-chain-sigma", type=float, default=1.0,
                   help="Compactified step of each chain jump. 1.0 => C = X_1^k (fixed-point target); "
                        "<1 => C = X_s^k, the multi-step sampler.")
    p.add_argument("--selfcorr-chain-steps", type=int, default=4, help="k, the number of chain jumps")
    p.add_argument("--selfcorr-chain-final-x1", action="store_true",
                   help="append a final X_1 after the chain, i.e. the paper K-schedule X_1(X_s0^k(z))")
    p.add_argument("--selfcorr-max-batch", type=int, default=0,
                   help="If >0, compute the self-correction term on at most this many samples per "
                        "microbatch (the teacher-norm term it replaces used 4 of 8).")
    p.add_argument("--terminal-invariance-weight", type=float, default=0.0,
                   help="For --objective scaled_meanflow, weight on the stopped "
                        "boundary "
                        "condition ||P(x)-P(x+delta*b(x))||^2.")
    p.add_argument("--terminal-invariance-step", type=float, default=0.1,
                   help="Physical autonomous-time Euler step delta used by the "
                        "terminal invariance target.")
    p.add_argument("--terminal-invariance-max-batch", type=int, default=0,
                   help="If >0, compute terminal invariance on at most this many "
                        "samples per microbatch to save memory.")
    p.add_argument("--adaptive-weight-p", type=float, default=0.0,
                   help="MeanFlow-style adaptive loss weight exponent. 0 disables.")
    p.add_argument("--adaptive-weight-eps", type=float, default=1e-3)

    # Auxiliary time-prediction head (self-supervised, NO DINO/REPA features).
    p.add_argument("--time-pred-weight", type=float, default=0.0,
                   help="Weight on the auxiliary time-prediction loss. When >0, "
                        "the student's diagonal forward emits a t-prediction "
                        "logit and an MSE loss pins it to the interpolant time t "
                        "of x0, forcing the flow-map student's features to encode "
                        "trajectory position. 0 disables (default).")
    p.add_argument("--time-pred-layer", type=int, default=14,
                   help="0-indexed transformer block whose mean-pooled output "
                        "feeds the time-prediction head. Default 14 (mid-depth "
                        "of the 28-block SiT-XL).")
    p.add_argument("--time-pred-target", choices=["t", "s_norm"], default="t",
                   help="Aux head regression target. 't': raw interpolant time; "
                        "'s_norm': normalized exponential-clock coordinate "
                        "-log(1-t)/log(1/eps).")

    p.add_argument("--guidance-mode", choices=["branch", "guided"], default="branch",
                   help="'branch' trains cond/null branches separately for external "
                        "CFG. 'guided' bakes teacher CFG into the conditional branch.")
    p.add_argument("--teacher-cfg-scale", type=float, default=1.0)
    p.add_argument("--cfg-mode", choices=["all", "first3"], default="all")
    p.add_argument("--cfg-match-uncond-norm", action="store_true", default=False,
                   help="For guided teacher CFG, rescale the unconditional "
                        "teacher field per sample to match the conditional "
                        "field norm before forming b_u + s*(b_c-b_u).")
    p.add_argument("--cfg-match-guided-norm", action="store_true", default=False,
                   help="For guided teacher CFG, rescale the final guided "
                        "vector per sample back to the conditional field norm "
                        "(direction-only guidance). In an autonomous field the "
                        "field magnitude sets the landing rate, i.e. the clock; "
                        "position-dependent rescaling by CFG decalibrates "
                        "sigma and breaks multi-step composition. Mirrors the "
                        "sampler's --cfg-match-guided-norm.")
    p.add_argument("--cfg-rule", choices=["standard", "perp", "perp_matched"], default="standard",
                   help="Teacher guidance rule. 'perp' projects the perturbation off "
                        "b_c so the landing rate is untouched (cos(u,b_c)=0).")
    p.add_argument("--loss-reduction", choices=["sum", "mean"], default="mean",
                   help="Per-sample residual reduction in eqfm/stage2_losses.py: mean over entries "
                        "(default) or sum (original convention; pass it to reproduce older runs).")
    p.add_argument("--cfg-cnorm", action="store_true", default=False,
                   help="With --cfg-rule perp, set ||u|| = (scale-1)*||b_c|| so the "
                        "perturbation decays with the field near the manifold. Can blow up "
                        "(division by ||b_u_perp||); prefer --cfg-rule perp_matched.")
    p.add_argument("--cfg-dropout-prob", type=float, default=0.1)
    p.add_argument("--null-class", type=int, default=1000)
    p.add_argument("--init-stage1-ckpt", type=Path, default=None,
                   help="Warm-start the Stage-2 student from a Stage-1 EqM "
                        "checkpoint before training. Ignored by --resume "
                        "after latest.pt is loaded.")
    p.add_argument("--init-stage1-weights", choices=["ema", "raw"], default="ema")
    p.add_argument("--init-student-ckpt", type=Path, default=None,
                   help="Start the student from the weights of a Stage-2 checkpoint (fresh optimizer, step 0). "
                        "Ignored when --resume finds <run>/checkpoints/latest.pt.")
    p.add_argument("--init-student-weights", choices=["raw", "ema"], default="raw")
    p.add_argument("--mf-single-jvp", action="store_true",
                   help="scaled_meanflow: one JVP with tangents (s b0, -k s) instead of two (same target).")
    p.add_argument("--lr-plateau-window", type=int, default=0,
                   help="lr slowdown on a stalled UNWEIGHTED training loss: window length in steps; the window mean "
                        "(all ranks, all samples) of sum_i w_i * raw_i is recorded and the mean of the last PATIENCE "
                        "windows is compared with the mean of the PATIENCE windows before (eqfm/lr_plateau.py); 0 = off.")
    p.add_argument("--lr-plateau-patience", type=int, default=5, help="windows per comparison block")
    p.add_argument("--lr-plateau-threshold", type=float, default=0.01,
                   help="relative drop between consecutive blocks below which the loss counts as stalled")
    p.add_argument("--lr-plateau-factor", type=float, default=0.5, help="lr scale multiplier at a cut")
    p.add_argument("--lr-plateau-min-scale", type=float, default=0.05)
    p.add_argument("--lr-plateau-cooldown", type=int, default=1, help="windows ignored after a cut")
    p.add_argument("--lr-plateau-stat", choices=["geomean", "mean"], default="geomean",
                   # geomean of the GLOBAL per-step mean (all ranks) since 2026-09-15: stored as "geomean_global"
                   help="window statistic of the per-step unweighted loss: geomean = exp(mean log) (default; the MeanFlow "
                        "residual is heavy-tailed: per-step values 0.15-52 gave +-7%% noise on 1000-step arithmetic means "
                        "vs ~1.5%% for the geometric mean) or mean.")

    p.add_argument("--steps", type=int, default=10_000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--betas", type=float, nargs=2, default=(0.9, 0.999))
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--grad-skip-norm", type=float, default=1e5,
                   help="Skip the optimizer step when the pre-clip grad norm "
                        "exceeds this (or is non-finite). Healthy Stage-2 norms "
                        "are 1e2-3e3; run 3070963 died on a single 6e16 step. "
                        "0 disables the magnitude check (non-finite still skips).")
    p.add_argument("--grad-accum-steps", type=int, default=1,
                   help="Number of microbatches to accumulate before each "
                        "optimizer step. Effective global batch is "
                        "batch_size * world_size * grad_accum_steps.")
    p.add_argument("--warmup-steps", type=int, default=500)
    p.add_argument("--ema-decay", type=float, default=0.9999)
    p.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--stage2-loss-fp32", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="Compute the student Stage-2 loss outside autocast. "
                        "Forward-mode AD through SiT attention is fragile "
                        "under bf16 autocast; VAE and frozen teacher still "
                        "use --bf16 where safe.")
    p.add_argument("--student-bf16-jvp", action=argparse.BooleanOptionalAction, default=False,
                   help="With --bf16: also run the Lagrangian JVP pass (forward-mode AD through the "
                        "student, and its backward) under bf16 autocast. Off by default; see "
                        "scripts/s2_speed_check.py for the gradient check.")
    p.add_argument("--student-bf16-nonjvp", action=argparse.BooleanOptionalAction,
                   default=False,
                   help="With --stage2-loss-fp32: run the student passes that "
                        "are not differentiated in forward mode (diagonal, "
                        "terminal, terminal-norm, semigroup) under bf16 "
                        "autocast; the JVP pass stays fp32. Off by default.")
    p.add_argument("--teacher-bf16", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="Run the frozen Stage-1 teacher target under bf16 "
                        "autocast when --bf16 is enabled. This is faster and "
                        "matches Stage-1 eval practice; pass "
                        "--no-teacher-bf16 to compute fp32 distillation "
                        "targets while keeping the student JVP loss fp32.")
    p.add_argument("--allow-tf32", action=argparse.BooleanOptionalAction, default=True)

    p.add_argument("--save-every", type=int, default=5000)
    p.add_argument("--save-every-minutes", type=float, default=0.0,
                   help="Also save when this many wall-clock minutes passed since the last save (rank 0 only, "
                        "no collective). Bounds the work lost to a preemption independently of GPU speed. 0 = off.")
    p.add_argument("--keep-every", type=int, default=0,
                   help="Hardlink every checkpoint whose step is a multiple of this into <run>/kept/ so pruning "
                        "never removes it (FID checkpoints). 0 = off.")
    p.add_argument("--on-keep-cmd", type=str, default=None,
                   help="Shell command run (non-blocking, rank 0) after a checkpoint is kept; {ckpt} and {step} "
                        "are substituted, e.g. an sbatch of the FID job.")
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--val-every", type=int, default=1000)
    p.add_argument("--val-n-samples", type=int, default=16)
    p.add_argument("--val-seed", type=int, default=0,
                   help="Seed for fixed validation noise and, when "
                        "--val-classes is omitted, validation labels.")
    p.add_argument("--val-classes", type=int, nargs="+", default=None,
                   help="Explicit ImageNet class indices for validation. If "
                        "omitted, labels are drawn once from the val-seed RNG "
                        "to match Stage-1 validation behavior.")
    p.add_argument("--val-diag-num-steps", type=int, default=100,
                   help="Euler steps for validation samples using the "
                        "diagonal field v_sigma=0.")
    p.add_argument("--val-eps-stop", type=float, default=1e-3,
                   help="Residual horizon for diagonal Euler and finite "
                        "off-diagonal compositions.")
    p.add_argument("--val-offdiag-s0", type=float, nargs="+", default=None,
                   help="Paper jump schedule for the validation grids: one s0 per --val-offdiag-steps entry; K "
                        "jumps = K-1 jumps of s0 then X_1. Default: equal jumps.")
    p.add_argument("--val-offdiag-steps", type=int, nargs="+", default=[1, 2, 4],
                   help="K values for K-fold composition of an equal "
                        "compactified off-diagonal step. K=1 here means "
                        "sigma=1-val_eps_stop, while terminal is logged "
                        "separately as sigma=1.")
    p.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", "eqfm"),
                   help="default: $WANDB_PROJECT or 'eqfm'")
    p.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY") or None,
                   help="default: $WANDB_ENTITY (None = the key's default entity)")
    p.add_argument("--wandb-mode", choices=["online", "offline", "disabled"],
                   default=os.environ.get("WANDB_MODE") or ("online" if os.environ.get("WANDB_API_KEY") else "offline"),
                   help="default: $WANDB_MODE, else online when WANDB_API_KEY is set, else offline "
                        "(sync later with `wandb sync`)")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def linear_warmup_lr(step: int, warmup_steps: int, base_lr: float) -> float:
    if step >= warmup_steps:
        return base_lr
    return base_lr * (step + 1) / max(1, warmup_steps)


def _atomic_save(obj: dict, path: Path, retries: int = 2) -> None:
    """Serialize to a sibling temp file, fsync, then rename into place.

    A partial write -- transient Lustre I/O error, full disk -- would otherwise
    leave a truncated ``.pt`` that ``torch.load`` cannot open. Renaming only
    after a complete write keeps the previous checkpoint usable.
    """
    tmp = path.with_name(path.name + ".tmp")
    last: Optional[BaseException] = None
    for attempt in range(retries + 1):
        try:
            with open(tmp, "wb") as f:
                torch.save(obj, f)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
            return
        except (RuntimeError, OSError) as e:
            last = e
            print(
                f"[stage2] WARN write of {path.name} failed "
                f"(attempt {attempt + 1}/{retries + 1}): {e}",
                flush=True,
            )
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass
    raise RuntimeError(f"could not write checkpoint {path}") from last


def copy_to_latest(src: Path) -> Path:
    """Publish an already-written checkpoint as ``latest.pt``.

    Copying the finished file is cheaper than re-serializing the state dicts
    off the GPU a second time, and the rename keeps ``latest.pt`` atomic.
    """
    dst = src.with_name("latest.pt")
    tmp = dst.with_name(dst.name + ".tmp")
    try:
        # Hardlink first (free; step_*.pt and latest.pt then share one inode,
        # and cleanup_old_checkpoints unlinking step_*.pt leaves latest.pt
        # intact). Copy only where the filesystem refuses links.
        try:
            tmp.unlink(missing_ok=True)
            os.link(src, tmp)
        except OSError:
            shutil.copyfile(src, tmp)
        os.replace(tmp, dst)
    except (OSError, RuntimeError):
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    return dst


def save_checkpoint(
    run_dir: Path,
    step: int,
    model: nn.Module,
    ema: Optional[EMA],
    optimizer: AdamW,
    args: argparse.Namespace,
    tag: Optional[str] = None,
) -> Path:
    ckpt = {
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "args": {k: (str(v) if isinstance(v, Path) else v)
                 for k, v in vars(args).items()},
    }
    if ema is not None:
        ckpt["ema"] = ema.state_dict()
    if getattr(save_checkpoint, "lr_plateau", None) is not None:
        ckpt["lr_plateau"] = save_checkpoint.lr_plateau.state_dict()
    name = tag or f"step_{step:07d}"
    path = run_dir / "checkpoints" / f"{name}.pt"
    _atomic_save(ckpt, path)
    return path


def maybe_resume(
    run_dir: Path,
    model: nn.Module,
    ema: Optional[EMA],
    optimizer: AdamW,
    device: torch.device,
    do_resume: bool,
    is_main: bool,
    loss_reduction: Optional[str] = None,
) -> int:
    if not do_resume:
        return 0
    latest = run_dir / "checkpoints" / "latest.pt"
    if not latest.exists():
        if is_main:
            print(f"[stage2] --resume requested but {latest} not found; starting fresh")
        return 0
    if is_main:
        print(f"[stage2] resuming from {latest}")
    state = torch.load(latest, map_location=device, weights_only=False)
    # The --loss-reduction default changed from sum to mean; never switch convention mid-run.
    # Checkpoints written before the option existed were trained with sum.
    saved_reduction = (state.get("args") or {}).get("loss_reduction", "sum")
    if loss_reduction is not None and saved_reduction != loss_reduction:
        raise ValueError(
            f"{latest} was trained with --loss-reduction {saved_reduction} but this launch uses "
            f"{loss_reduction}; pass --loss-reduction {saved_reduction} to continue the run "
            "(or use a new --run-name to change convention)."
        )
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    if ema is not None and "ema" in state:
        ema.load_state_dict(state["ema"])
    maybe_resume.lr_plateau = state.get("lr_plateau")
    return int(state.get("step", 0))


def warm_start_from_stage1(
    model: nn.Module,
    ckpt_path: Path,
    *,
    weights: str,
    device: torch.device,
    is_main: bool,
) -> None:
    """Partially initialize a Stage-2 student from a Stage-1 EqM checkpoint.

    Stage 1 and Stage 2 use the same SiT velocity architecture.  The only
    common schema mismatch is the class embedding table when the Stage-2 run
    does not allocate a null CFG row; in that case we copy the overlapping
    class rows and leave any extra target rows at their initialized values.
    """
    if is_main:
        print(f"[stage2] warm-starting student from Stage-1: {ckpt_path}")
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    source = dict(state["model"])
    if weights == "ema":
        if "ema" not in state:
            if is_main:
                print("[stage2] WARN Stage-1 checkpoint has no EMA; using raw model")
        else:
            for name, tensor in state["ema"].items():
                if name == "decay":
                    continue
                if name in source and source[name].shape == tensor.shape:
                    source[name] = tensor

    target = model.state_dict()
    copied = 0
    partial = 0
    skipped = []
    for name, src in source.items():
        if name not in target:
            skipped.append(name)
            continue
        dst = target[name]
        if dst.shape == src.shape:
            dst.copy_(src.to(device=dst.device, dtype=dst.dtype))
            copied += 1
            continue
        if (
            name == "y_embedder.embedding_table.weight"
            and dst.ndim == 2 and src.ndim == 2
            and dst.shape[1] == src.shape[1]
        ):
            n = min(dst.shape[0], src.shape[0])
            dst[:n].copy_(src[:n].to(device=dst.device, dtype=dst.dtype))
            partial += 1
            continue
        skipped.append(name)
    model.load_state_dict(target, strict=True)
    if is_main:
        step = state.get("step", state.get("steps", -1))
        print(
            f"[stage2] warm-start loaded step={step} weights={weights} "
            f"copied={copied} partial={partial} skipped={len(skipped)}"
        )
        if skipped:
            print(f"[stage2] warm-start skipped examples: {skipped[:5]}")


def cleanup_old_checkpoints(run_dir: Path, keep_last: int) -> None:
    if keep_last <= 0:
        return
    files = sorted(
        (run_dir / "checkpoints").glob("step_*.pt"),
        key=lambda p: int(p.stem.split("_")[1]),
    )
    for old in files[:-keep_last]:
        if int(old.stem.split("_")[1]) % 10_000 == 0:
            continue  # keep every 10k-step checkpoint for FID evaluation and sampling
        try:
            old.unlink()
            print(f"[stage2] removed old checkpoint {old.name}")
        except OSError as e:
            print(f"[stage2] WARN could not remove {old.name}: {e}")


def _mean_norm(x: torch.Tensor) -> float:
    return float(x.float().flatten(1).norm(dim=1).mean().item())


@torch.no_grad()
def validation_sample(
    *,
    model: nn.Module,
    ema: Optional[EMA],
    vae: Optional[LatentEncoder],
    run_dir: Path,
    step: int,
    labels: torch.Tensor,
    latent_size: int,
    device: torch.device,
    compute_dtype: torch.dtype,
    bf16: bool,
    val_seed: int,
    diag_num_steps: int,
    eps_stop: float,
    offdiag_steps: list[int],
    wandb_run,
    target_fn=None,
    map_err_sigmas: tuple = (0.5, 0.9, 0.97, 0.999),
    offdiag_s0: Optional[list] = None,
) -> None:
    if vae is None:
        return
    gen = torch.Generator(device=device).manual_seed(val_seed)
    z = torch.randn(labels.shape[0], 4, latent_size, latent_size,
                    device=device, generator=gen)

    # MNIST-style map-error diagnostics against the (guided) teacher ODE.
    # target_fn(x, y) is the exact field the student is distilled from.
    map_metrics: dict = {}
    if target_fn is not None:
        cache = validation_sample.__dict__.setdefault("_ode_ref", {})
        if not cache:
            n_ode = 100
            for s in map_err_sigmas:
                tau = -math.log(1.0 - s)
                h = tau / n_ode
                x = z.clone()
                for _ in range(n_ode):
                    with maybe_autocast(device, compute_dtype, bf16):
                        x = x + h * target_fn(x, labels).float()
                cache[s] = x
        def map_errors(tag: str) -> None:
            for s in map_err_sigmas:
                sig = torch.full((z.shape[0],), s, device=device, dtype=z.dtype)
                with maybe_autocast(device, compute_dtype, bf16):
                    xs = flow_map(model, z, sig, labels).float()
                map_metrics[f"map_err_s{s:.3f}{tag}"] = _mean_norm(xs - cache[s])
            sigma1 = torch.ones(z.shape[0], device=device, dtype=z.dtype)
            with maybe_autocast(device, compute_dtype, bf16):
                x1 = flow_map(model, z, sigma1, labels).float()
                map_metrics[f"endpoint_b_norm{tag}"] = _mean_norm(target_fn(x1, labels).float())

        map_errors("_raw")
        if ema is not None:
            with ema.apply_to(model):
                map_errors("")
        print(f"[stage2] step {step:>7d} | VAL " +
              " ".join(f"{k}={v:.3f}" for k, v in map_metrics.items()), flush=True)
    samples_dir = run_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    def decode_save(name: str, latents: torch.Tensor) -> Path:
        with maybe_autocast(device, compute_dtype, bf16):
            images = vae.decode(latents)
        grid_path = samples_dir / f"step_{step:07d}_{name}.png"
        save_grid(images.float().cpu(), grid_path)
        return grid_path

    def diagonal_euler(z0: torch.Tensor) -> tuple[torch.Tensor, dict]:
        h = -math.log(eps_stop) / max(1, diag_num_steps)
        x = z0.clone()
        norms = []
        sigma0 = torch.zeros(x.shape[0], device=device, dtype=x.dtype)
        for _ in range(diag_num_steps):
            with maybe_autocast(device, compute_dtype, bf16):
                b = model(x, sigma0, labels).float()
            norms.append(_mean_norm(b))
            x = x + h * b
        return x, {
            "velocity_norm_first": norms[0] if norms else 0.0,
            "velocity_norm_last": norms[-1] if norms else 0.0,
            "latent_norm_last": _mean_norm(x),
        }

    def offdiag_compose(z0: torch.Tensor, k: int, s0: Optional[float] = None) -> torch.Tensor:
        if k <= 0:
            raise ValueError(f"val offdiag steps must be positive, got {k}")
        steps = [1.0 - eps_stop ** (1.0 / k)] * k if s0 is None else [s0] * (k - 1) + [1.0]
        x = z0.clone()
        for sigma_step in steps:
            sigma = torch.full((z0.shape[0],), sigma_step, device=device, dtype=z0.dtype)
            with maybe_autocast(device, compute_dtype, bf16):
                x = flow_map(model, x, sigma, labels).float()
        return x

    ctx = ema.apply_to(model) if ema is not None else torch.no_grad()
    log_payload = {}
    with ctx:
        diag_latents, diag_summary = diagonal_euler(z)
        diag_path = decode_save("diag_euler", diag_latents)

        sigma1 = torch.ones(labels.shape[0], device=device, dtype=z.dtype)
        with maybe_autocast(device, compute_dtype, bf16):
            terminal_latents = flow_map(model, z, sigma1, labels).float()
        terminal_path = decode_save("terminal", terminal_latents)

        offdiag_paths = {}
        for i, k in enumerate(offdiag_steps):
            s0 = offdiag_s0[i] if offdiag_s0 is not None else None
            latents = offdiag_compose(z, k, s0)
            offdiag_paths[(k, s0)] = decode_save(f"offdiag_K{k}" + ("" if s0 is None else f"_s0{s0:g}"), latents)

    print(f"[stage2] step {step:>7d} | val diag -> {diag_path.name} "
          f"terminal -> {terminal_path.name} "
          f"offdiag K={offdiag_steps}")
    print(f"[stage2] step {step:>7d} | val diag norms: "
          f"||b||_first={diag_summary['velocity_norm_first']:.3f} "
          f"||b||_last={diag_summary['velocity_norm_last']:.3f} "
          f"||z||_last={diag_summary['latent_norm_last']:.3f}")
    if wandb_run is not None:
        import wandb
        log_payload.update({
            f"{S_VAL} samples/Euler on v(y,0), {diag_num_steps} steps": wandb.Image(str(diag_path)),
            f"{S_VAL} samples/K=1  X_1(z)": wandb.Image(str(terminal_path)),
            f"{S_VAL}/Euler on v(y,0): ‖v‖ at first step": diag_summary["velocity_norm_first"],
            f"{S_VAL}/Euler on v(y,0): ‖v‖ at last step (→ 0)": diag_summary["velocity_norm_last"],
            f"{S_VAL}/Euler on v(y,0): ‖y‖ at last step": diag_summary["latent_norm_last"],
        })
        for (k, s0), path in offdiag_paths.items():
            name = f"K={k}  equal jumps" if s0 is None else f"K={k}  X_1∘X_{s0:g}^{k - 1}"
            log_payload[f"{S_VAL} samples/{name}"] = wandb.Image(str(path))
        for k, v in map_metrics.items():
            w = "raw" if k.endswith("_raw") else "EMA"
            base = k[:-4] if k.endswith("_raw") else k
            if base.startswith("map_err_s"):
                key = f"map error ‖X_s(z) − ODE_s(z)‖  s={float(base[9:]):g}  {w}"
            else:
                key = f"endpoint ‖b(X_1(z))‖  {w}"
            log_payload[f"{S_VAL}/{key}"] = v
        try:
            wandb_run.log(log_payload, step=step)
        except Exception as e:  # logging must never kill training
            print(f"[stage2] WARN wandb val log failed: {e!r}", flush=True)


@record
def main() -> None:
    args = parse_args()
    if args.grad_accum_steps < 1:
        raise ValueError(f"--grad-accum-steps must be >= 1, got {args.grad_accum_steps}")
    rank, local_rank, world_size = setup_distributed()
    is_main = rank == 0

    if args.run_name is None:
        run_name = datetime.now().strftime("stage2_flowmap_%Y%m%d_%H%M%S") if is_main else ""
        if world_size > 1:
            obj = [run_name]
            dist.broadcast_object_list(obj, src=0)
            run_name = obj[0]
    else:
        run_name = args.run_name
    run_dir = args.results_dir / run_name

    if is_main:
        (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        with (run_dir / "args.json").open("w") as f:
            json.dump({k: (str(v) if isinstance(v, Path) else v)
                       for k, v in vars(args).items()}, f, indent=2, sort_keys=True)
        print(f"[stage2] run_dir = {run_dir}")
        print(f"[stage2] world_size = {world_size} (DDP={world_size > 1})")

    wandb_run = None
    if is_main and args.wandb_mode != "disabled":
        import uuid
        import wandb
        # One wandb run per run_dir: the id lives next to the checkpoints so a
        # requeued/resumed job appends to the same wandb run.
        id_file = run_dir / "wandb_id.txt"
        if args.resume and id_file.exists():
            wandb_id = id_file.read_text().strip()
        else:
            wandb_id = uuid.uuid4().hex[:8]
            id_file.write_text(wandb_id + "\n")
        wandb_kwargs = dict(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_dir.name,
            id=wandb_id,
            resume="allow",
            dir=str(run_dir),
            config={k: (str(v) if isinstance(v, Path) else v)
                    for k, v in vars(args).items()},
        )
        try:
            wandb_run = wandb.init(mode=args.wandb_mode, **wandb_kwargs)
        except Exception as e:  # a wandb outage must not kill a requeued job
            print(f"[stage2] wandb.init failed ({e!r}); logging offline in {run_dir}")
            wandb_run = wandb.init(mode="offline", **wandb_kwargs)
        try:
            define_wandb_metrics(wandb_run)
            add_glossary(wandb_run)
        except Exception as e:  # cosmetics must never kill training
            print(f"[stage2] WARN wandb metric setup failed: {e!r}", flush=True)

    torch.manual_seed(args.seed + rank)
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    if not torch.cuda.is_available():
        raise RuntimeError("Stage-2 training requires CUDA.")
    device = torch.device(f"cuda:{local_rank}")
    compute_dtype = torch.bfloat16 if args.bf16 else torch.float32

    vae = None
    data_iter = None
    if not args.synthetic_data:
        from eqfm.data import build_imagenet_loader, is_latent_spec

        latents_mode = is_latent_spec(args.shards)
        if is_main:
            print("[stage2] loading SD VAE" + (" (validation decode only; training reads cached latents)" if latents_mode else ""))
        vae = LatentEncoder(args.vae_id, cache_dir=args.vae_cache_dir).to(device).eval()
        # The loader restarts at epoch 0 with its seed on every (re)start, so each preemption used to replay the
        # same first shards and samples (s2mf run, 2026-09-15: the 1000-step training loss dropped after every
        # restart and climbed as the loader reached less-seen data). Offset the seed by the step being resumed:
        # each restart draws a fresh shard order and in-stream shuffle. Same on every rank; unchanged for step 0.
        data_seed = args.seed
        latest_ckpt = run_dir / "checkpoints" / "latest.pt"
        if args.resume and latest_ckpt.exists():
            try:
                data_seed = args.seed + int(torch.load(latest_ckpt, map_location="cpu", mmap=True,
                                                       weights_only=False).get("step", 0))
            except Exception as e:  # unreadable latest.pt: resume logic reports it; keep the base seed
                print(f"[stage2] WARN could not read resume step for the data seed: {e!r}", flush=True)
        if is_main:
            print(f"[stage2] data loader seed {data_seed} (base {args.seed} + resume step)", flush=True)
        loader = build_imagenet_loader(
            args.shards,
            image_size=args.image_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle_buffer=args.shuffle_buffer,
            # Same seed on every rank: the loader shuffles the shard list with
            # it *before* striding by rank, so a per-rank seed made ranks read
            # overlapping shard subsets. The in-stream shuffle already mixes
            # in self.rank.
            seed=data_seed,
            distributed=world_size > 1,
        )
        data_iter = cycle(loader)

    if args.fake_teacher:
        if is_main:
            print(f"[stage2] using fake teacher b(x)=-{args.fake_teacher_scale}x")

        def teacher_fn(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return -args.fake_teacher_scale * x

        def teacher_grad_fn(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return -args.fake_teacher_scale * x

    else:
        if args.teacher_ckpt is None:
            raise ValueError("--teacher-ckpt is required unless --fake-teacher is set")
        if is_main:
            print(f"[stage2] loading Stage-1 teacher: {args.teacher_ckpt}")
        teacher, teacher_step, teacher_weights = load_stage1_model(
            args.teacher_ckpt,
            latent_size=args.latent_size,
            device=device,
            use_ema=args.teacher_weights == "ema",
        )
        for p in teacher.parameters():
            p.requires_grad_(False)
        teacher.eval()
        teacher_model_fn = autonomous_forward(teacher)
        if is_main:
            print(f"[stage2] teacher step={teacher_step} weights={teacher_weights}")

        def teacher_fn(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            with torch.no_grad():
                # The student Stage-2 loss can run in fp32 for forward-AD
                # stability while the frozen teacher target is still evaluated
                # under bf16 autocast for speed. Use --no-teacher-bf16 if the
                # target precision itself becomes a suspected bottleneck.
                with maybe_autocast(
                    device, compute_dtype, args.bf16 and args.teacher_bf16
                ):
                    return teacher_model_fn(x, y).float()

        def teacher_grad_fn(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            # Frozen teacher parameters have requires_grad=False, but this
            # path deliberately keeps autograd enabled so endpoint norm losses
            # can backprop through the teacher input X_1_student(z).
            with maybe_autocast(
                device, compute_dtype, args.bf16 and args.teacher_bf16
            ):
                return teacher_model_fn(x, y).float()

    if is_main:
        print(f"[stage2] instantiating student {args.model} latent_size={args.latent_size}")
    base_model = SiT_models[args.model](
        input_size=args.latent_size,
        class_dropout_prob=args.cfg_dropout_prob,
    ).to(device)
    if args.init_stage1_ckpt is not None:
        warm_start_from_stage1(
            base_model,
            args.init_stage1_ckpt,
            weights=args.init_stage1_weights,
            device=device,
            is_main=is_main,
        )
    latest_exists = (run_dir / "checkpoints" / "latest.pt").exists()
    if args.init_student_ckpt is not None and not (args.resume and latest_exists):
        st = torch.load(args.init_student_ckpt, map_location="cpu", mmap=True, weights_only=False)
        key = "model" if args.init_student_weights == "raw" or "ema" not in st else "ema"
        sd = st[key] if key == "model" else {k: v for k, v in st["ema"].get("shadow", st["ema"]).items()}
        missing, unexpected = base_model.load_state_dict(sd, strict=False)
        if is_main:
            print(f"[stage2] student initialized from {args.init_student_ckpt} ({key} weights, checkpoint step "
                  f"{st.get('step')}); missing={len(missing)} unexpected={len(unexpected)}; fresh optimizer", flush=True)
        if missing or unexpected:
            raise RuntimeError(f"--init-student-ckpt mismatch: missing {missing[:5]} unexpected {unexpected[:5]}")
        del st, sd
    with torch.no_grad():
        probe_x = torch.zeros(1, 4, args.latent_size, args.latent_size, device=device)
        probe_sigma = torch.zeros(1, device=device)
        probe_y = torch.zeros(1, dtype=torch.long, device=device)
        probe_v = base_model(probe_x, probe_sigma, probe_y)
        if probe_v.shape != probe_x.shape:
            raise RuntimeError(
                f"Stage-2 student must return velocity shape {tuple(probe_x.shape)}, "
                f"got {tuple(probe_v.shape)} from {args.model}. Check learn_sigma/2C "
                "or model-output configuration."
            )
    n_params = sum(p.numel() for p in base_model.parameters())
    if is_main:
        print(f"[stage2] student params: {n_params/1e6:.1f}M")

    # Attach the auxiliary time-prediction head BEFORE DDP wrap / optimizer /
    # EMA so its params are broadcast, optimized, EMA-tracked, and checkpointed.
    # It is invoked only in the diagonal forward (return_time_pred=True); the
    # JVP distill paths never touch it.
    if args.time_pred_weight > 0.0:
        base_model.enable_time_prediction(args.time_pred_layer)
        base_model.to(device)
        if is_main:
            n_head = sum(p.numel() for p in base_model.t_pred_head.parameters())
            print(f"[stage2] time-pred head: layer={args.time_pred_layer} "
                  f"target={args.time_pred_target} weight={args.time_pred_weight:g} "
                  f"({n_head/1e3:.1f}k params)")

    model: nn.Module = base_model
    if world_size > 1:
        model = DDP(base_model, device_ids=[local_rank], output_device=local_rank)

    optimizer = AdamW(base_model.parameters(), lr=args.lr,
                      betas=tuple(args.betas), weight_decay=args.weight_decay,
                      fused=True)
    ema = EMA(base_model, decay=args.ema_decay) if is_main else None
    loss_cfg = Stage2LossConfig(
        objective="lagrangian" if args.objective == "lag_semi" else args.objective,
        interp_time_sampler=args.interp_time_sampler,
        eps_train=args.eps_train,
        importance_k=args.importance_k,
        sigma_min=args.sigma_min,
        sigma_max=args.sigma_max,
        sigma_sampler=args.sigma_sampler,
        diag_weight=args.diag_weight,
        distill_weight=args.distill_weight,
        terminal_weight=args.terminal_weight,
        terminal_teacher_norm_weight=args.terminal_teacher_norm_weight,
        loss_reduction=args.loss_reduction,
        terminal_student_norm_weight=args.terminal_student_norm_weight,
        terminal_norm_max_batch=args.terminal_norm_max_batch,
        selfcorr_weight=args.selfcorr_weight,
        selfcorr_chain_sigma=args.selfcorr_chain_sigma,
        selfcorr_chain_steps=args.selfcorr_chain_steps,
        selfcorr_chain_final_x1=args.selfcorr_chain_final_x1,
        selfcorr_max_batch=args.selfcorr_max_batch,
        student_bf16_nonjvp=bool(args.student_bf16_nonjvp and args.bf16),
        student_bf16_jvp=bool(args.student_bf16_jvp and args.bf16),
        terminal_invariance_weight=args.terminal_invariance_weight,
        terminal_invariance_step=args.terminal_invariance_step,
        terminal_invariance_max_batch=args.terminal_invariance_max_batch,
        guidance_mode=args.guidance_mode,
        teacher_cfg_scale=args.teacher_cfg_scale,
        cfg_mode=args.cfg_mode,
        cfg_match_uncond_norm=args.cfg_match_uncond_norm,
        cfg_match_guided_norm=args.cfg_match_guided_norm,
        cfg_rule=args.cfg_rule,
        cfg_cnorm=args.cfg_cnorm,
        cfg_dropout_prob=args.cfg_dropout_prob,
        null_class=args.null_class,
        adaptive_weight_p=args.adaptive_weight_p,
        adaptive_weight_eps=args.adaptive_weight_eps,
        time_pred_weight=args.time_pred_weight,
        time_pred_target=args.time_pred_target,
        # generic PESD add-on for every objective except lag_semi (its wrapper adds it) and pure semigroup
        semigroup_weight=args.semi_weight if args.objective not in ("lag_semi", "semigroup") else 0.0,
        mf_single_jvp=args.mf_single_jvp,
    )
    if args.objective == "lag_semi":
        semi_weight = args.semi_weight

        class LagSemi(Stage2FlowMapLoss):
            """Lagrangian (LESD) wrapper adding semi_weight * PESD to the off-diagonal term.

            Same construction as the MNIST-best recipe (train_stage2_flowmap_mnist.py).
            """
            def lagrangian_loss(self, student, teacher_fn, x0, labels, labels_in, drop_mask, sigma):
                lag = super().lagrangian_loss(student, teacher_fn, x0, labels, labels_in, drop_mask, sigma)
                sg, sgp = self.sample_semigroup_pair(x0.shape[0], x0.device, x0.dtype)
                with self.nonjvp_autocast(x0.device):  # no-op unless --student-bf16-nonjvp
                    semi = self.semigroup_loss(student, x0, labels_in, sg, sgp)
                self.last_semi = semi.detach()
                self.last_lag = lag.detach()
                return lag + semi_weight * semi

        loss_fn = LagSemi(loss_cfg)
    else:
        loss_fn = Stage2FlowMapLoss(loss_cfg)
    # Exact pooled wandb metrics over each logging window (rank 0 only): see eqfm/stage2_wandb.py.
    wstats = WindowStats() if wandb_run is not None else None
    if wstats is not None:
        loss_fn.collect_samples = True
    evals_state = run_dir / "wandb_logged_evals.json"
    evals_logged = load_logged(evals_state) if wstats is not None else set()
    n_log_points = 0

    def val_target_fn(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """The exact (possibly guided, norm-matched) field the student is distilled from."""
        no_drop = torch.zeros_like(y, dtype=torch.bool)
        return loss_fn.teacher_target_for_conditioning(teacher_fn, x, y, y, no_drop)

    start_step = maybe_resume(
        run_dir, base_model, ema, optimizer, device, args.resume, is_main,
        loss_reduction=args.loss_reduction,
    )
    plateau = None
    if args.lr_plateau_window > 0:
        plateau = LossPlateau(args.lr_plateau_window, args.lr_plateau_patience, args.lr_plateau_threshold,
                              args.lr_plateau_factor, args.lr_plateau_min_scale, args.lr_plateau_cooldown,
                              start_step=args.warmup_steps,
                              stat=args.lr_plateau_stat + ("_global" if args.lr_plateau_stat == "geomean" else ""))
        if getattr(maybe_resume, "lr_plateau", None):
            plateau.load_state_dict(maybe_resume.lr_plateau)
        save_checkpoint.lr_plateau = plateau
        if is_main:
            print(f"[stage2] lr plateau rule: window={args.lr_plateau_window} patience={args.lr_plateau_patience} "
                  f"threshold={args.lr_plateau_threshold:g} factor={args.lr_plateau_factor:g} "
                  f"min_scale={args.lr_plateau_min_scale:g} cooldown={args.lr_plateau_cooldown}; "
                  f"scale now {plateau.scale:g}, {plateau.bad} windows in the current block", flush=True)
    plateau_sum = torch.zeros((), device=device)
    plateau_n = 0
    semi_w_mon = args.semi_weight if args.objective != "semigroup" else 0.0
    # One-time EMA reset requested out-of-band: if <run>/checkpoints/EMA_RESET_REQUEST exists,
    # re-seed the EMA shadow from the current (raw) weights and delete the marker. Used to
    # purge the initialization from a slow (0.9999) EMA once training has moved far from it.
    reset_marker = run_dir / "checkpoints" / "EMA_RESET_REQUEST"
    # Idempotent until persisted: the marker records the step at which the reset was applied.
    # If the resumed checkpoint is not past that step (the resetting copy was preempted before
    # saving), apply the reset again; once a later checkpoint exists, drop the marker.
    if ema is not None and reset_marker.exists():
        try:
            applied_at = int(reset_marker.read_text().strip() or -1)
        except (OSError, ValueError):
            applied_at = -1
        if applied_at < 0 or start_step <= applied_at:  # never applied, or applied but not yet persisted
            with torch.no_grad():
                for name, p in base_model.named_parameters():
                    if name in ema.shadow:
                        ema.shadow[name].copy_(p.detach().float())
            try:
                reset_marker.write_text(str(start_step))
            except OSError:
                pass
            print(f"[stage2] EMA shadow reset to raw weights at step {start_step} "
                  f"(marker kept until a later checkpoint is saved)", flush=True)
        else:
            try:
                reset_marker.unlink()
            except OSError:
                pass
            print(f"[stage2] EMA reset at step {applied_at} is persisted in the checkpoint; marker removed", flush=True)
    # The checkpoint restores the EMA decay it was saved with; let the CLI value win so the
    # decay can be changed on resume (e.g. 0.9999 -> 0.999 once the init has been purged).
    if ema is not None and abs(ema.decay - args.ema_decay) > 1e-6:  # checkpoint stores decay as float32
        print(f"[stage2] EMA decay {ema.decay} (from checkpoint) -> {args.ema_decay} (CLI)", flush=True)
        ema.decay = args.ema_decay
    if is_main:
        print(f"[stage2] objective={args.objective} "
              f"guidance={args.guidance_mode} omega={args.teacher_cfg_scale:g} "
              f"cfg_match_uncond_norm={args.cfg_match_uncond_norm} "
              f"cfg_match_guided_norm={args.cfg_match_guided_norm} "
              f"interp_t={args.interp_time_sampler}")
        print(f"[stage2] terminal norm: "
              f"teacher_w={args.terminal_teacher_norm_weight:g} "
              f"student_w={args.terminal_student_norm_weight:g} "
              f"max_batch={args.terminal_norm_max_batch}")
        if args.objective == "scaled_meanflow":
            print(f"[stage2] scaled stopped MeanFlow: "
                  f"p={args.adaptive_weight_p:g} "
                  f"eps={args.adaptive_weight_eps:g} "
                  f"invariance_w={args.terminal_invariance_weight:g} "
                  f"delta={args.terminal_invariance_step:g} "
                  f"invariance_max_batch={args.terminal_invariance_max_batch}")
        print(f"[stage2] batch: per_gpu={args.batch_size} world={world_size} "
              f"accum={args.grad_accum_steps} "
              f"effective_global={args.batch_size * world_size * args.grad_accum_steps}")
        print(f"[stage2] teacher targets: "
              f"{'bf16 autocast' if args.bf16 and args.teacher_bf16 else 'fp32'}; "
              f"student non-JVP passes: {'bf16 autocast' if args.student_bf16_nonjvp and args.bf16 else 'as loss'}; "
              f"student JVP pass: {'bf16 autocast' if args.student_bf16_jvp and args.bf16 else 'as loss'}")
        print(f"[stage2] starting at step {start_step} target={args.steps}")

    if args.val_classes is not None:
        if len(args.val_classes) != args.val_n_samples:
            if is_main:
                print(f"[stage2] --val-classes had {len(args.val_classes)} entries, "
                      f"overriding --val-n-samples to match")
            args.val_n_samples = len(args.val_classes)
        val_labels = torch.tensor(args.val_classes, device=device, dtype=torch.long)
    else:
        label_gen = torch.Generator(device="cpu").manual_seed(args.val_seed)
        val_labels = torch.randint(
            0, 1000, (args.val_n_samples,), generator=label_gen,
        ).to(device=device, dtype=torch.long)
    if is_main:
        print(f"[stage2] validation: every {args.val_every} steps, "
              f"seed={args.val_seed}, labels={val_labels.cpu().tolist()[:8]}"
              f"{'...' if args.val_n_samples > 8 else ''}")

    t0 = time.time()
    tw = time.time()
    last_save_time = time.time()
    samples_w = 0
    n_grad_skips = 0

    for step in range(start_step, args.steps):
        optimizer.zero_grad(set_to_none=True)
        metrics_accum = None
        for micro_step in range(args.grad_accum_steps):
            if args.synthetic_data:
                latents = torch.randn(
                    args.batch_size, 4, args.latent_size, args.latent_size,
                    device=device,
                )
                labels = torch.randint(0, 1000, (args.batch_size,), device=device)
            else:
                assert data_iter is not None and vae is not None
                batch = next(data_iter)
                if latents_mode:
                    # Cached SD-VAE posterior stats: draw the same posterior
                    # sample encode() would have drawn, no pixels involved.
                    mean, std, labels = batch
                    labels = labels.to(device, non_blocking=True)
                    latents = vae.sample_from_stats(
                        mean.to(device, non_blocking=True), std.to(device, non_blocking=True)
                    )
                else:
                    images, labels = batch
                    images = images.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True) if torch.is_tensor(labels) \
                        else torch.as_tensor(labels, device=device)
                    with torch.amp.autocast(device.type, dtype=compute_dtype, enabled=args.bf16):
                        latents = vae.encode(images)

            sync_grads = micro_step == args.grad_accum_steps - 1
            sync_ctx = (
                model.no_sync()
                if isinstance(model, DDP) and not sync_grads
                else contextlib.nullcontext()
            )
            with sync_ctx:
                if args.stage2_loss_fp32:
                    loss, metrics = loss_fn(
                        model,
                        teacher_fn,
                        latents.float(),
                        labels,
                        teacher_grad_fn=teacher_grad_fn,
                    )
                else:
                    with torch.amp.autocast(device.type, dtype=compute_dtype, enabled=args.bf16):
                        loss, metrics = loss_fn(
                            model,
                            teacher_fn,
                            latents,
                            labels,
                            teacher_grad_fn=teacher_grad_fn,
                        )
                (loss / args.grad_accum_steps).backward()

            if metrics_accum is None:
                metrics_accum = {
                    k: v.detach().float().clone()
                    for k, v in metrics.items()
                }
            else:
                for k, v in metrics.items():
                    metrics_accum[k].add_(v.detach().float())
            if wstats is not None:
                pool_stage2(wstats, metrics, loss_fn._samples,
                            getattr(loss_fn, "last_lag", None), getattr(loss_fn, "last_semi", None),
                            objective=args.objective)
            samples_w += latents.shape[0] * world_size

        assert metrics_accum is not None
        metrics = {
            k: v / args.grad_accum_steps
            for k, v in metrics_accum.items()
        }
        grad_norm = torch.nn.utils.clip_grad_norm_(base_model.parameters(), args.grad_clip)
        if wstats is not None:  # no host sync here; resolved at the log point
            wstats.add_scalar(f"{S_OPT}/‖grad‖ before clip at {args.grad_clip:g}", grad_norm, track_max=True)
        # Live schedule override: drop <run_dir>/lr_override.json with any of
        # {"lr", "lr_decay_from", "lr_final", "steps"} to change the schedule
        # without restarting (polled every 50 steps, all ranks read the same file).
        if step % 50 == 0:
            _ov = run_dir / "lr_override.json"
            try:
                _m = _ov.stat().st_mtime if _ov.exists() else None
            except OSError:
                _m = None
            if _m is not None and _m != getattr(args, "_lr_override_mtime", None):
                try:
                    _d = json.loads(_ov.read_text())
                    for _k in ("lr", "lr_decay_from", "lr_final", "steps"):
                        if _k in _d and _d[_k] is not None:
                            setattr(args, _k, type(getattr(args, _k))(_d[_k]) if getattr(args, _k) is not None else _d[_k])
                    args._lr_override_mtime = _m
                    if _d.get("lr_plateau", True) is False and plateau is not None:
                        # e.g. a fixed annealing phase decided by hand: the stall rule stops acting (scale 1)
                        plateau = None
                        if is_main:
                            print("[stage2] lr plateau rule disabled by lr_override.json", flush=True)
                    if is_main:
                        print(f"[stage2] lr schedule override applied at step {step}: "
                              f"lr={args.lr:g} decay_from={args.lr_decay_from} lr_final={args.lr_final:g} steps={args.steps}",
                              flush=True)
                except Exception as _e:  # never let a bad override file kill the run
                    if is_main:
                        print(f"[stage2] WARN ignoring lr_override.json: {_e!r}", flush=True)
        if step >= args.steps:  # an override may have shortened the run
            if is_main:
                print(f"[stage2] reached overridden --steps={args.steps}; stopping", flush=True)
            break
        lr = linear_warmup_lr(step, args.warmup_steps, args.lr)
        if args.lr_decay_from is not None and step >= args.lr_decay_from:
            # Cosine decay from --lr at --lr-decay-from to --lr-final at --steps (MNIST-best schedule).
            frac = min(1.0, (step - args.lr_decay_from) / max(1, args.steps - args.lr_decay_from))
            lr = args.lr_final + 0.5 * (args.lr - args.lr_final) * (1.0 + math.cos(math.pi * frac))
        if plateau is not None:
            lr = lr * plateau.scale
            # Unweighted training loss, the quantity the slowdown rule watches: sum_i w_i * raw_i with the
            # loss's own term weights, raw = per-sample error without the adaptive weight.
            z0 = metrics["total_loss"].new_zeros(())
            mon = (args.diag_weight * metrics.get("diag_raw", z0)
                   + args.distill_weight * metrics.get("distill_raw", z0)
                   + semi_w_mon * metrics.get("semi_raw", z0)
                   + args.terminal_weight * metrics.get("terminal_loss", z0)
                   + args.terminal_teacher_norm_weight * metrics.get("terminal_teacher_norm_raw", z0))
            mon = mon.detach().float().reshape(1)
            if world_size > 1:
                # Global per-step mean over all 128 samples (every rank's micro-batches) BEFORE the log: a per-rank
                # mean covers 128/world samples, and E[log(mean of a heavy-tailed sample)] depends on that count, so
                # a per-rank log made the statistic jump whenever a copy with a different GPU count took over.
                dist.all_reduce(mon, op=dist.ReduceOp.AVG)
            mon = mon.reshape(())
            plateau_sum = plateau_sum + (torch.log(mon.clamp_min(1e-12)) if plateau.stat.startswith("geomean") else mon)
            plateau_n += 1
        for group in optimizer.param_groups:
            group["lr"] = lr
        # Skip pathological steps rather than letting them into the optimizer.
        # Run 3070963 died at step 7920 on a single update with grad norm 6e16
        # and never recovered; healthy norms there were 1e2-3e3. The decision is
        # all-reduced so every rank stays in lockstep.
        skip = not torch.isfinite(grad_norm) or (
            args.grad_skip_norm > 0.0 and float(grad_norm) > args.grad_skip_norm
        )
        if world_size > 1:
            flag = torch.tensor(
                [1.0 if skip else 0.0], device=device, dtype=torch.float32
            )
            dist.all_reduce(flag, op=dist.ReduceOp.MAX)
            skip = bool(flag.item() > 0.0)
        if skip:
            n_grad_skips += 1
            optimizer.zero_grad(set_to_none=True)
            if is_main:
                print(f"[stage2] step {step + 1}: SKIPPED, grad_norm="
                      f"{float(grad_norm):.3e} (total skips {n_grad_skips})",
                      flush=True)
        else:
            optimizer.step()
            if ema is not None:
                ema.update(base_model)

        if is_main and ((step + 1) % args.log_every == 0 or step == 0):
            dt = time.time() - tw
            sps = samples_w / max(dt, 1e-6)
            tw = time.time()
            samples_w = 0
            scalars = {
                "train/loss": float(metrics["total_loss"].item()),
                "train/diag_loss": float(metrics["diag_loss"].item()),
                "train/distill_loss": float(metrics["distill_loss"].item()),
                "train/terminal_loss": float(metrics["terminal_loss"].item()),
                "train/terminal_teacher_norm_loss": float(
                    metrics["terminal_teacher_norm_loss"].item()
                ),
                "train/terminal_teacher_endpoint_norm": float(
                    metrics["terminal_teacher_endpoint_norm"].item()
                ),
                "train/terminal_student_norm_loss": float(
                    metrics["terminal_student_norm_loss"].item()
                ),
                "train/terminal_student_endpoint_norm": float(
                    metrics["terminal_student_endpoint_norm"].item()
                ),
                "train/lr": lr,
                "train/grad_norm": float(grad_norm.item()),
                "train/samples_per_s": sps,
                "train/grad_accum_steps": args.grad_accum_steps,
                "train/effective_global_batch": args.batch_size * world_size * args.grad_accum_steps,
                "train/cfg_drop_frac": float(metrics["cfg_drop_frac"].item()),
                "train/sigma_mean": float(metrics["sigma_mean"].item()),
                "train/grad_skips": n_grad_skips,
                "train/interp_t_mean": float(metrics["interp_t_mean"].item()),
            }
            for k, v in metrics.items():
                if "_raw" in k:
                    scalars[f"train/{k}"] = float(v.item())
            for k in (
                "terminal_invariance_loss",
                "terminal_invariance_endpoint_delta",
                "meanflow_target_abs_max",
                "meanflow_target_norm",
                "meanflow_jvp_abs_max",
                "meanflow_clock_min",
            ):
                if k in metrics:
                    scalars[f"train/{k}"] = float(metrics[k].item())
            draw = scalars.get("train/distill_raw")
            raw_str = ""
            if draw is not None:
                b = [scalars.get(f"train/distill_raw_s{lo:g}_{hi:g}", 0.0)
                     for lo, hi in ((0, 0.5), (0.5, 0.9), (0.9, 0.99), (0.99, 1))]
                raw_str = (f"Draw {draw:.4g} "
                           f"[s<.5 {b[0]:.4g} | .5-.9 {b[1]:.4g} | "
                           f".9-.99 {b[2]:.4g} | >.99 {b[3]:.4g}] ")
            tp_str = ""
            if hasattr(loss_fn, "last_semi"):
                scalars["train/semi_loss"] = float(loss_fn.last_semi.item())
                scalars["train/lag_loss"] = float(loss_fn.last_lag.item())
                tp_str += f"lag {scalars['train/lag_loss']:.4f} semi {scalars['train/semi_loss']:.4f} "
            if "time_pred_loss" in metrics:
                scalars["train/time_pred_loss"] = float(metrics["time_pred_loss"].item())
                scalars["train/time_pred_mae"] = float(metrics["time_pred_mae"].item())
                tp_str = (f"tp {scalars['train/time_pred_loss']:.4f}"
                          f"(mae {scalars['train/time_pred_mae']:.3f}) ")
            inv_str = ""
            if args.terminal_invariance_weight > 0.0:
                inv_str = (
                    f"inv {scalars['train/terminal_invariance_loss']:.4f} "
                    f"(delta {scalars['train/terminal_invariance_endpoint_delta']:.3f}) "
                )
            print(f"[stage2] step {step + 1:>7d} | "
                  f"loss {scalars['train/loss']:.4f} "
                  f"diag {scalars['train/diag_loss']:.4f} "
                  f"D {scalars['train/distill_loss']:.4f} "
                  f"term {scalars['train/terminal_loss']:.4f} "
                  f"tnorm {scalars['train/terminal_teacher_endpoint_norm']:.3f} "
                  f"{raw_str}{inv_str}{tp_str}"
                  f"lr {lr:.2e} grad {scalars['train/grad_norm']:.2f} "
                  f"{sps:.1f} samples/s elapsed {time.time() - t0:.0f}s",
                  flush=True)
            if wandb_run is not None:
                payload = wstats.result() if wstats is not None else {}
                if wstats is not None:
                    wstats.reset()
                gb = args.batch_size * world_size * args.grad_accum_steps
                payload.update({
                    f"{S_OPT}/learning rate": lr,
                    f"{S_OPT}/skipped steps since restart": n_grad_skips,
                    f"{S_OPT}/samples per s (global, incl. val and saves)": sps,
                    f"{S_OPT}/seconds per step": gb / max(sps, 1e-9),
                    f"{S_OPT}/GPUs": world_size,
                    f"{S_OPT}/global batch = per-GPU × GPUs × accum": gb,
                    f"{S_OPT}/Slurm job id of the carrier": float(os.environ.get("SLURM_JOB_ID", 0) or 0),
                })
                logged_ids = []
                n_log_points += 1
                if n_log_points % 5 == 1:  # scan finished FID evals every 5th log point (cheap NFS glob)
                    try:
                        estep, group = next_eval_group(run_dir.name, evals_logged)
                        if group:
                            payload.update(eval_payload(estep, group))
                            logged_ids = [r["id"] for r in group]
                            print(f"[stage2] wandb: logging FID of step {estep} "
                                  f"({', '.join(r['weights'] + '_' + r['sampler'] for r in group)})", flush=True)
                    except Exception as e:
                        print(f"[stage2] WARN FID scan failed: {e!r}", flush=True)
                try:
                    wandb_run.log(payload, step=step + 1)
                except Exception as e:  # logging must never kill training
                    print(f"[stage2] WARN wandb log failed: {e!r}", flush=True)
                    logged_ids = []
                if logged_ids:
                    evals_logged.update(logged_ids)
                    try:
                        save_logged(evals_state, evals_logged)
                    except OSError as e:
                        print(f"[stage2] WARN could not save {evals_state.name}: {e!r}", flush=True)

        if plateau is not None and plateau.is_window_end(step + 1) and plateau_n > 0:
            buf = torch.stack([plateau_sum, torch.tensor(float(plateau_n), device=device)])
            # plateau_sum is already identical on every rank (global per-step means); nothing to reduce.
            win_mean = float(buf[0] / buf[1].clamp_min(1.0))
            if plateau.stat.startswith("geomean"):
                win_mean = math.exp(win_mean)
            plateau_sum = torch.zeros((), device=device)
            plateau_n = 0
            event = plateau.end_window(step + 1, win_mean)
            if is_main:
                print(f"[stage2] step {step + 1:>7d} | lr plateau: window unweighted loss {win_mean:.5g} "
                      f"last-{args.lr_plateau_patience} mean {plateau.recent:.5g} previous-{args.lr_plateau_patience} "
                      f"mean {plateau.previous:.5g} block {plateau.bad}/{2 * args.lr_plateau_patience} "
                      f"lr scale {plateau.scale:g}" + (f" | {event}" if event else ""), flush=True)
                try:
                    (run_dir / "lr_plateau.json").write_text(json.dumps(plateau.state_dict(), indent=1))
                except OSError:
                    pass
                if wandb_run is not None:
                    try:
                        wandb_run.log({
                            f"{S_OPT}/lr plateau: unweighted loss, {args.lr_plateau_window}-step {plateau.stat}": win_mean,
                            f"{S_OPT}/lr plateau: mean of last {args.lr_plateau_patience} windows": plateau.recent,
                            f"{S_OPT}/lr plateau: mean of previous {args.lr_plateau_patience} windows": plateau.previous,
                            f"{S_OPT}/lr plateau: lr scale (×)": plateau.scale,
                        }, step=step + 1)
                    except Exception as e:
                        print(f"[stage2] WARN wandb plateau log failed: {e!r}", flush=True)

        if (
            is_main and vae is not None and args.val_every > 0
            and ((step + 1) % args.val_every == 0 or (step + 1) == args.steps)
        ):
            validation_sample(
                model=base_model,
                ema=ema,
                vae=vae,
                run_dir=run_dir,
                step=step + 1,
                labels=val_labels,
                latent_size=args.latent_size,
                device=device,
                compute_dtype=compute_dtype,
                bf16=args.bf16,
                val_seed=args.val_seed,
                diag_num_steps=args.val_diag_num_steps,
                eps_stop=args.val_eps_stop,
                offdiag_steps=args.val_offdiag_steps,
                wandb_run=wandb_run,
                target_fn=val_target_fn,
                offdiag_s0=args.val_offdiag_s0,
            )

        time_save = (args.save_every_minutes > 0
                     and time.time() - last_save_time >= 60.0 * args.save_every_minutes)
        keep_now = args.keep_every > 0 and (step + 1) % args.keep_every == 0
        if is_main and ((step + 1) % args.save_every == 0 or (step + 1) == args.steps or time_save or keep_now):
            # A checkpoint write must never take down a multi-day run: the
            # other ranks are blocked in collectives and would be SIGTERMed.
            try:
                path = save_checkpoint(run_dir, step + 1, base_model, ema, optimizer, args)
                latest = copy_to_latest(path)
                print(f"[stage2] saved checkpoint -> {path}")
                print(f"[stage2] updated latest -> {latest}")
                last_save_time = time.time()
                if keep_now:
                    kept = run_dir / "kept" / path.name
                    kept.parent.mkdir(parents=True, exist_ok=True)
                    if not kept.exists():
                        os.link(path, kept)
                    print(f"[stage2] kept (hardlink) -> {kept}", flush=True)
                    if args.on_keep_cmd:
                        import subprocess
                        cmd = args.on_keep_cmd.format(ckpt=kept, step=step + 1)
                        try:
                            subprocess.Popen(cmd, shell=True, start_new_session=True)
                            print(f"[stage2] on-keep: {cmd}", flush=True)
                        except OSError as e:
                            print(f"[stage2] WARN on-keep command failed: {e!r}", flush=True)
                cleanup_old_checkpoints(run_dir, args.keep_last_checkpoints)
            except (RuntimeError, OSError) as e:
                print(
                    f"[stage2] WARN checkpoint at step {step + 1} failed, "
                    f"continuing training: {e}",
                    flush=True,
                )

    if is_main:
        print(f"[stage2] done in {time.time() - t0:.0f}s")
        if wandb_run is not None:
            wandb_run.finish()
    cleanup_distributed()


if __name__ == "__main__":
    main()
