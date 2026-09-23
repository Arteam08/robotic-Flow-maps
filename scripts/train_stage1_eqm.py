"""Stage-1 fine-tune SiT-XL/2 to the equilibrium field via the EqM loss.

This is the first stage of the three-stage Appendix-D pipeline (pretrain ->
fine-tune-to-equilibrium -> distill into compactified flow map). We start
from the pretrained SiT-XL/2 ImageNet 256 checkpoint, ablate time
conditioning by passing t=0 to the transformer on every step, and minimize
the canonical sampler-form EqM loss:

    L = E_{t ~ p, z, x} || b_hat(I_t) - (1 - t)(x - z) ||^2

where x are latent codes from the SD VAE and t is drawn from p(t) ∝ 1/(1-t)
(the inverse-CDF sampler). Optional time-importance sampling (t = 1 - U^k)
and the data-marginal anchor lambda_eq * ||b_hat(x_1)||^2 are exposed as
CLI flags. ``--time-sampler uniform`` is an ablation for testing whether the
canonical sampler over-concentrates near t=1; it is not the default
paper-faithful objective.

Single-GPU only in this version; DDP/FSDP for 8x H100 is a follow-up.

Usage:
    python scripts/train_stage1_eqm.py --results-dir /scratch/$USER/eqfm/runs

Disable wandb for quick local iteration:
    python scripts/train_stage1_eqm.py --results-dir /tmp/eqfm-test \
        --wandb-mode disabled --steps 50 --log-every 5 --save-every 50
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Iterator, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from eqfm.data import build_imagenet_loader, is_latent_spec, DEFAULT_TRAIN_SHARDS  # noqa: E402
from eqfm.losses import EqMLoss, EqMLossConfig  # noqa: E402
from eqfm.models.sit import SiT_XL_2, find_model  # noqa: E402
from eqfm.sampling import (  # noqa: E402
    SamplerDiagnostics, as_field_fn, sample_euler, save_grid, stop_time_from_eps,
    summarize_diagnostics,
)
from eqfm.training import EMA  # noqa: E402
from eqfm.eval_utils import (  # noqa: E402
    collect_fixed_batches, fixed_batch_loss, scale_output_head, search_output_scale,
)
from eqfm.metric_window import WindowStats, pool_eqm_metrics  # noqa: E402
from eqfm.wandb_glossary import (  # noqa: E402
    IMAGENET_TRAIN_IMAGES, NOTATION_STAGE1, STAGE1, SUMMARIES_STAGE1, ReadableLogger,
    stage1_context, stage1_header,
)
from eqfm.vae import LatentEncoder  # noqa: E402


# ---------------------------------------------------------------------------
# Distributed helpers
# ---------------------------------------------------------------------------

def setup_distributed() -> tuple[int, int, int]:
    """Initialise ``torch.distributed`` if launched under torchrun.

    Returns ``(rank, local_rank, world_size)``. When no torchrun env vars
    are present, behaves as a single-process run (rank 0 of 1) and leaves
    ``dist`` un-initialised.
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
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


def barrier_if_distributed() -> None:
    if dist.is_initialized():
        dist.barrier()


def unwrap(model: nn.Module) -> nn.Module:
    """Strip a DDP wrapper if present."""
    return model.module if isinstance(model, DDP) else model


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])

    # I/O
    p.add_argument("--results-dir", type=Path, required=True,
                   help="Base directory under which a per-run subfolder will be "
                        "created (checkpoints, args.json, wandb logs).")
    p.add_argument("--run-name", default=None,
                   help="Subfolder name under results-dir. Defaults to a "
                        "timestamped slug. Pin this across multiple sessions "
                        "(e.g. Modal's 24h timeout cap, or continuing a run) "
                        "so --resume can find the previous checkpoints.")
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True,
                   help="If <results-dir>/<run-name>/checkpoints/latest.pt "
                        "exists, restore model/EMA/optimizer + step counter "
                        "from it. No-op when the file is absent.")
    p.add_argument("--keep-last-checkpoints", type=int, default=5,
                   help="Sliding window: keep this many periodic step_*.pt "
                        "files; older ones are deleted after each save. "
                        "latest.pt is always kept. 0 retains every "
                        "checkpoint (no cleanup).")

    # Data
    p.add_argument("--shards", default=DEFAULT_TRAIN_SHARDS,
                   help="WebDataset shard pattern (pipe:gcloud ... .tar).")
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--shuffle-buffer", type=int, default=2000)
    p.add_argument("--hflip", action=argparse.BooleanOptionalAction, default=False,
                   help="Random horizontal flip (the DiT/SiT/EqM recipe).")

    # Model
    p.add_argument("--ckpt", default="SiT-XL-2-256x256.pt",
                   help="Pretrained SiT checkpoint name or path.")
    p.add_argument("--init-from-scratch", action="store_true",
                   help="Randomly initialize the SiT model instead of loading "
                        "--ckpt. Use a fresh --run-name or --no-resume for a "
                        "true scratch-control run.")
    p.add_argument("--ckpt-cache-dir", default=None,
                   help="Cache dir for the SiT checkpoint. Overrides "
                        "$EQFM_CACHE_DIR.")
    p.add_argument("--ckpt-key", choices=["auto", "model", "ema"], default="auto",
                   help="Which weights to take from a trainer checkpoint given "
                        "as --ckpt. 'auto' uses 'ema' only when it covers the "
                        "full model (a head-only run's EMA covers just the head).")
    p.add_argument("--init-output-scale", default="1",
                   help="Multiply final_layer.linear (weight+bias) by this once "
                        "at init, or 'auto' to pick the value from "
                        "--init-scale-candidates that minimises the training "
                        "loss on --val-loss-batches fixed batches. Used to warm-"
                        "start from a field with a different clock (EqM).")
    p.add_argument("--init-scale-candidates",
                   default="1,0.5,0.25,0.2,0.167,0.143,0.125,0.111,0.1,0.0909,0.0833,0.0714,0.0625",
                   help="Comma-separated candidates for --init-output-scale auto.")
    p.add_argument("--val-loss-batches", type=int, default=4,
                   help="Fixed training batches (per rank) on which the uw loss "
                        "and ||b(x_data)|| are probed at init and every "
                        "--val-every steps. 0 disables the probe.")
    p.add_argument("--grad-comm-bf16", action=argparse.BooleanOptionalAction, default=False,
                   help="All-reduce DDP gradients in bf16 (torch's bf16_compress_hook); "
                        "halves inter-GPU traffic on PCIe/cross-socket nodes. Weights, "
                        "optimizer state and clipping stay fp32.")
    p.add_argument("--trainable", default=None,
                   help="Regex; parameters whose name does not match are frozen "
                        r"(e.g. '^final_layer\.' for head-only, "
                        r"'^(?!t_embedder\.)' for all but the t_embedder).")
    p.add_argument("--vae-id", default="stabilityai/sd-vae-ft-ema")
    p.add_argument("--latent-size", type=int, default=32,
                   help="Spatial size of VAE latents (= image_size / 8).")
    p.add_argument("--cfg-dropout-prob", type=float, default=0.1,
                   help="Probability of replacing an ImageNet label with "
                        "the null class during EqM training. This trains "
                        "the unconditional branch used by CFG sampling. "
                        "Default 0.1 matches the pretrained SiT convention, "
                        "but is applied explicitly here for logging and "
                        "resume-time control.")
    p.add_argument("--null-class", type=int, default=1000,
                   help="Classifier-free guidance null label. SiT ImageNet "
                        "uses 1000 for the extra unconditional embedding.")

    # Optimization
    p.add_argument("--steps", type=int, default=50_000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--betas", type=float, nargs=2, default=(0.9, 0.999))
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--grad-accum-steps", type=int, default=1,
                   help="Micro-batches per optimizer step; the effective global "
                        "batch is batch_size * world_size * grad_accum_steps.")
    p.add_argument("--warmup-steps", type=int, default=500)
    p.add_argument("--lr-schedule", choices=["constant", "cosine"], default="constant",
                   help="After warmup: 'constant', or 'cosine' decay from --lr to --lr-final between "
                        "--lr-decay-start and --steps.")
    p.add_argument("--lr-decay-start", type=int, default=0,
                   help="Optimizer step at which the cosine decay begins (e.g. the resume step).")
    p.add_argument("--lr-final", type=float, default=0.0)
    p.add_argument("--ema-decay", type=float, default=0.9999)
    p.add_argument("--bf16", action="store_true", default=True,
                   help="bf16 autocast for compute (weights stay fp32).")
    p.add_argument("--no-bf16", dest="bf16", action="store_false",
                   help="Disable bf16 autocast (use fp32 throughout).")

    # Time-conditioning ablation
    p.add_argument("--time-conditioning", choices=["zero", "strict"],
                   default="strict",
                   help="How to ablate time conditioning. 'strict' zeros + "
                        "freezes the SiT t_embedder so it contributes nothing "
                        "to AdaLN (matches Appendix D / DroPE semantics). "
                        "'zero' keeps the t_embedder trainable and just "
                        "passes t=0 every forward (the legacy v0 behaviour, "
                        "leaves a constant bias for y_emb to absorb).")

    # Loss
    p.add_argument("--parameterization", choices=["velocity", "denoiser"],
                   default="velocity",
                   help="What the network predicts. 'velocity' (default): the "
                        "network outputs the autonomous field b and regresses "
                        "U_t = (1-t)(x-z); the sampler integrates dot{x}=b. "
                        "'denoiser' (x-pred): the network outputs the denoiser "
                        "D and regresses the clean data x; the sampler field is "
                        "dot{x} = D(x) - x with fixed point D(x)=x. The two "
                        "share the same EqM residual/time weighting but differ "
                        "in the raw regression target (x0- vs v-prediction).")
    p.add_argument("--time-sampler",
                   choices=["canonical", "importance", "uniform", "uniform_weighted", "uniform_clock_weighted"],
                   default="canonical")
    p.add_argument("--clock", default="linear",
                   help="Clock c(t) of the target c(t)(x1 - z): linear (1 - t) | trunc:A | power:P | logref | logsnr.")
    p.add_argument("--clock-a", type=float, default=0.8)
    p.add_argument("--clock-lambda", type=float, default=1.0)
    p.add_argument("--clock-weight-eps", type=float, default=1e-3,
                   help="eps in the 1/(c(t) + eps) weight of --time-sampler uniform_clock_weighted.")
    p.add_argument("--importance-k", type=float, default=2.0)
    p.add_argument("--eps-train", type=float, default=1e-3)
    p.add_argument("--data-anchor", type=float, default=0.0,
                   help="lambda_eq for the data-marginal anchor "
                        "||b_hat(x_1)||^2. 0 disables it (default).")

    # Auxiliary time-prediction head (REPA-style *self*-supervision, no DINO)
    p.add_argument("--time-pred-weight", type=float, default=0.0,
                   help="Weight on the auxiliary time-prediction loss. When "
                        ">0, a small head reads a pooled intermediate block and "
                        "regresses the interpolant time t (MSE). Forces the "
                        "autonomous field's features to encode where on the "
                        "trajectory the input sits, even though the field "
                        "output stays time-agnostic. 0 disables (default).")
    p.add_argument("--time-pred-layer", type=int, default=14,
                   help="0-indexed transformer block whose output is mean-"
                        "pooled and fed to the time-prediction head. Default 14 "
                        "(mid-depth of the 28-block SiT-XL). Earlier taps shape "
                        "more of the network; later taps probe the field's own "
                        "belief about its position.")
    p.add_argument("--time-pred-target", choices=["t", "s_norm"], default="t",
                   help="Aux head regression target. 't': raw interpolant time "
                        "(well-conditioned under uniform/uniform_weighted "
                        "sampling). 's_norm': normalized exponential-clock "
                        "coordinate -log(1-t)/log(1/eps), better when t "
                        "concentrates near 1 (canonical sampler).")

    # Logging / checkpointing
    p.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", "eqfm"),
                   help="default: $WANDB_PROJECT or 'eqfm'")
    p.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY") or None,
                   help="default: $WANDB_ENTITY (None = the key's default entity)")
    p.add_argument("--wandb-mode", choices=["online", "offline", "disabled"],
                   default=os.environ.get("WANDB_MODE") or ("online" if os.environ.get("WANDB_API_KEY") else "offline"),
                   help="default: $WANDB_MODE, else online when WANDB_API_KEY is set, else offline "
                        "(sync later with `wandb sync`)")
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--save-every", type=int, default=5000)

    # Validation sampling (EMA, fixed seed + classes so samples track over time)
    p.add_argument("--val-every", type=int, default=1000,
                   help="Run validation sampling every N steps. 0 disables.")
    p.add_argument("--on-keep-cmd", type=str, default=None,
                   help="Shell command run (detached) after every kept checkpoint; {ckpt} and {step} are substituted.")
    p.add_argument("--keep-every", type=int, default=0,
                   help="Never prune step_*.pt whose step is a multiple of "
                        "this (0: prune purely by --keep-last-checkpoints).")
    p.add_argument("--val-n-samples", type=int, default=16,
                   help="Number of images to generate per validation. "
                        "Defaults to a 4x4 grid.")
    p.add_argument("--val-num-steps", type=int, default=100,
                   help="Euler steps per sample at validation time. Fewer "
                        "than the standalone sampler default (250) to keep "
                        "validation cheap; bump for final-quality checks.")
    p.add_argument("--val-eps-stop", type=float, default=1e-3,
                   help="Residual to the manifold at the integration "
                        "terminus; S_stop = log(1/eps_stop). Keep "
                        ">= --eps-train.")
    p.add_argument("--val-cfg-scale", type=float, default=1.0,
                   help="CFG scale for validation samples. Use >1 only if "
                        "--cfg-dropout-prob trains the null branch.")
    p.add_argument("--val-seed", type=int, default=0,
                   help="Seed for the fixed validation noise + class draw, "
                        "kept constant across training so successive grids "
                        "are directly comparable.")
    p.add_argument("--val-classes", type=int, nargs="+", default=None,
                   help="Optional explicit class IDs for validation. If "
                        "omitted, drawn once from the val-seed RNG and "
                        "reused every validation step.")

    p.add_argument("--seed", type=int, default=42)

    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def cycle(loader: Iterator):
    """Wrap a finite iterable to yield indefinitely (WebDataset is iterable)."""
    while True:
        for batch in loader:
            yield batch


def linear_warmup_lr(step: int, warmup_steps: int, base_lr: float) -> float:
    if step >= warmup_steps:
        return base_lr
    return base_lr * (step + 1) / max(1, warmup_steps)


def scheduled_lr(step: int, *, warmup_steps: int, base_lr: float, schedule: str = "constant",
                 decay_start: int = 0, total_steps: int = 1, final_lr: float = 0.0) -> float:
    """Linear warmup, then constant or cosine decay base_lr -> final_lr over [decay_start, total_steps]."""
    lr = linear_warmup_lr(step, warmup_steps, base_lr)
    if schedule == "cosine" and step >= max(decay_start, warmup_steps):
        span = max(1, total_steps - decay_start)
        frac = min(1.0, (step - decay_start) / span)
        lr = final_lr + 0.5 * (base_lr - final_lr) * (1.0 + math.cos(math.pi * frac))
    return lr


def strip_time_conditioning(model: nn.Module) -> int:
    """Zero + freeze the SiT t_embedder so it contributes nothing to AdaLN.

    With the final Linear of ``t_embedder.mlp`` set to zero (and the whole
    submodule frozen), ``c = t_emb + y_emb`` reduces to ``c = y_emb`` for
    any ``t``. This matches Appendix D's "remove time conditioning" prompt
    and the DroPE analogy of literally removing the positional scaffold
    rather than pinning it. Returns the number of parameters frozen.
    """
    final = model.t_embedder.mlp[2]  # Linear -> SiLU -> Linear
    nn.init.zeros_(final.weight)
    if final.bias is not None:
        nn.init.zeros_(final.bias)
    frozen = 0
    for p in model.t_embedder.parameters():
        p.requires_grad_(False)
        frozen += p.numel()
    return frozen


def autonomous_forward(model: nn.Module):
    """Closure that calls SiT with the time argument pinned at 0.

    The ``t = 0`` we pass is irrelevant if ``strip_time_conditioning`` has
    been applied (the t_embedder outputs zeros regardless), but we still
    feed *something* of the right shape because SiT.forward requires the
    positional argument. In ``--time-conditioning zero`` mode the t_embedder
    is still trainable and the constant ``t = 0`` contributes a learnable
    bias to AdaLN that ``y_embedder`` will absorb over training.
    """
    def forward(z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        t = torch.zeros(z.shape[0], device=z.device, dtype=z.dtype)
        return model(z, t, y)
    return forward


def autonomous_forward_tpred(model: nn.Module):
    """Like ``autonomous_forward`` but requests the aux time-prediction logit.

    Returns a closure that yields ``(field, t_pred_logit)`` so ``EqMLoss`` can
    add the auxiliary time-prediction loss from a single forward pass. Routes
    through the (possibly DDP-wrapped) module so head gradients are synced.
    """
    def forward(z: torch.Tensor, y: torch.Tensor):
        t = torch.zeros(z.shape[0], device=z.device, dtype=z.dtype)
        return model(z, t, y, return_time_pred=True)
    return forward


def apply_cfg_dropout(
    labels: torch.Tensor,
    *,
    dropout_prob: float,
    null_class: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Randomly replace labels with the CFG null class.

    We do this explicitly instead of relying on ``LabelEmbedder``'s internal
    train-mode dropout so the exact probability is controlled by CLI args,
    logged, and preserved across resume runs. The model's internal
    ``dropout_prob`` is set to 0 in ``main`` to avoid double-dropping.
    """
    if dropout_prob < 0.0 or dropout_prob > 1.0:
        raise ValueError(f"cfg_dropout_prob must lie in [0, 1], got {dropout_prob}")
    if dropout_prob == 0.0:
        mask = torch.zeros_like(labels, dtype=torch.bool)
        return labels, mask
    mask = torch.rand(labels.shape, device=labels.device) < dropout_prob
    dropped = torch.where(mask, torch.full_like(labels, null_class), labels)
    return dropped, mask


@torch.no_grad()
def run_validation(
    step: int,
    *,
    model: nn.Module,
    vae: LatentEncoder,
    ema: EMA,
    val_labels: torch.Tensor,
    val_seed: int,
    n_samples: int,
    latent_size: int,
    num_steps: int,
    eps_stop: float,
    cfg_scale: float,
    null_class: int,
    device: torch.device,
    compute_dtype: torch.dtype,
    bf16: bool,
    run_dir: Path,
    wlog,
    parameterization: str = "velocity",
) -> dict:
    """Sample under EMA, decode, save grid, log scalars + image to wandb.

    Uses a fixed RNG seed for the initial Gaussian noise so successive
    validation steps are sampling the *same* latent trajectory and the
    grid PNGs are directly comparable as training progresses. Pair with
    ``val_labels`` that was also drawn once at startup.
    """
    was_training = model.training
    model.eval()
    samples_dir = run_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    # Per-validation generator pinned to val_seed so noise is fixed.
    val_generator = torch.Generator(device=device).manual_seed(val_seed)
    diag = SamplerDiagnostics()

    with ema.apply_to(model):
        # Raw SiT output is the field b (velocity) or the denoiser D; wrap it
        # so the sampler always integrates the autonomous field b(x).
        model_fn = as_field_fn(autonomous_forward(model), parameterization)
        latents = sample_euler(
            model_fn,
            shape=(n_samples, 4, latent_size, latent_size),
            class_labels=val_labels,
            num_steps=num_steps,
            stop_time=stop_time_from_eps(eps_stop),
            cfg_scale=cfg_scale,
            null_class=null_class,
            device=device,
            dtype=torch.float32,
            autocast_dtype=compute_dtype if bf16 else None,
            generator=val_generator,
            diagnostics=diag,
        )
        with torch.amp.autocast(device.type, dtype=compute_dtype, enabled=bf16):
            images = vae.decode(latents)
    images = images.float().cpu()

    grid_path = samples_dir / f"step_{step:07d}.png"
    save_grid(images, grid_path)

    summary = summarize_diagnostics(diag)
    print(f"[stage1] step {step:>7d} | val: "
          f"||b||_first={summary['velocity_norm_first']:.3f} "
          f"||b||_last={summary['velocity_norm_last']:.3f} "
          f"||z||_last={summary['latent_norm_last']:.3f} "
          f"cfg={cfg_scale:g} "
          f"-> {grid_path.name}")

    if wlog is not None and wlog.run is not None:
        import wandb
        wlog.log({
            "samples/grid":         wandb.Image(str(grid_path)),
            "samples/b_norm_first": summary["velocity_norm_first"],
            "samples/b_norm_last":  summary["velocity_norm_last"],
            "samples/b_norm_max":   summary["velocity_norm_max"],
            "samples/x_norm_last":  summary["latent_norm_last"],
            "samples/cfg_scale":    cfg_scale,
        }, step=step)

    if was_training:
        model.train()
    return summary


def save_checkpoint(
    run_dir: Path,
    step: int,
    model: nn.Module,
    ema: EMA,
    optimizer: AdamW,
    args: argparse.Namespace,
    *,
    tag: str = None,
) -> Path:
    ckpt = {
        "step": step,
        "model": model.state_dict(),
        "ema": ema.state_dict(),
        "optimizer": optimizer.state_dict(),
        "args": {k: (str(v) if isinstance(v, Path) else v)
                 for k, v in vars(args).items()},
    }
    name = tag or f"step_{step:07d}"
    path = run_dir / "checkpoints" / f"{name}.pt"
    # Atomic: a preemption mid-write must never leave a truncated latest.pt.
    tmp = path.with_suffix(".pt.tmp")
    torch.save(ckpt, tmp)
    os.replace(tmp, path)
    return path


def _cleanup_old_checkpoints(run_dir: Path, keep_last: int, keep_every: int = 0) -> None:
    """Drop oldest ``step_*.pt`` files past the keep-last window.

    Only periodic snapshots are pruned. ``latest.pt`` is always kept --
    it is the file ``--resume`` looks for next session. Snapshots whose
    step is a multiple of ``keep_every`` (if > 0) are never pruned.
    """
    if keep_last <= 0:
        return
    ckpts_dir = run_dir / "checkpoints"
    step_files = sorted(
        ckpts_dir.glob("step_*.pt"),
        key=lambda p: int(p.stem.split("_")[1]),
    )
    if keep_every > 0:
        step_files = [p for p in step_files if int(p.stem.split("_")[1]) % keep_every != 0]
    for old in step_files[:-keep_last]:
        try:
            old.unlink()
            print(f"[stage1] removed old checkpoint {old.name}")
        except OSError as e:
            print(f"[stage1] WARN failed to remove {old.name}: {e}")


def maybe_resume(
    *,
    base_model: nn.Module,
    ema: Optional[EMA],
    optimizer: AdamW,
    run_dir: Path,
    device: torch.device,
    is_main: bool,
    do_resume: bool,
) -> int:
    """Restore model/EMA/optimizer state from latest.pt; return starting step.

    Called *after* the model is loaded from the pretrained SiT checkpoint
    and wrapped in DDP, and *after* the optimizer + EMA are built. The
    resume overwrites the pretrained weights with the previously trained
    state. ``ema`` is rank-0-only, so it's ``None`` on non-main ranks --
    we skip that key there.

    Returns 0 if ``--no-resume`` or no ``latest.pt`` exists, otherwise the
    saved step count so the training loop picks up from there.
    """
    if not do_resume:
        return 0
    latest = run_dir / "checkpoints" / "latest.pt"
    if not latest.exists():
        if is_main:
            print(f"[stage1] --resume requested but {latest} not found; "
                  f"starting fresh from the pretrained SiT.")
        return 0

    if is_main:
        print(f"[stage1] resuming from {latest}")
    state = torch.load(latest, map_location=device, weights_only=False)

    base_model.load_state_dict(state["model"])
    if ema is not None and "ema" in state:
        ema.load_state_dict(state["ema"])
    optimizer.load_state_dict(state["optimizer"])

    start_step = int(state.get("step", 0))
    if is_main:
        print(f"[stage1] resumed at step {start_step}")
    return start_step


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # ----- distributed setup -----
    rank, local_rank, world_size = setup_distributed()
    is_main = rank == 0

    # ----- run dir (consistent across ranks so all ranks can resume) ------
    # When --run-name is None we generate a timestamp on rank 0 and
    # broadcast so every rank ends up with the same path. When --run-name
    # is supplied (the common resume case), no broadcast is needed.
    if args.run_name is None:
        run_name = (
            datetime.now().strftime("stage1_eqm_%Y%m%d_%H%M%S")
            if is_main else ""
        )
        if world_size > 1:
            obj_list = [run_name]
            dist.broadcast_object_list(obj_list, src=0)
            run_name = obj_list[0]
    else:
        run_name = args.run_name
    run_dir = args.results_dir / run_name

    # Only rank 0 creates the dir + writes args.json. Non-main ranks just
    # need the path for finding latest.pt during --resume.
    if is_main:
        (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        with (run_dir / "args.json").open("w") as f:
            json.dump(
                {k: (str(v) if isinstance(v, Path) else v)
                 for k, v in vars(args).items()},
                f, indent=2, sort_keys=True,
            )
        print(f"[stage1] run_dir = {run_dir}")
        print(f"[stage1] world_size = {world_size} (DDP={world_size > 1})")

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
            print(f"[stage1] wandb.init failed ({e!r}); logging offline in {run_dir}")
            wandb_run = wandb.init(mode="offline", **wandb_kwargs)

    # Readable wandb keys + glossary (run notes, a table, run_dir/wandb_glossary.md).
    # On non-main ranks run=None: key resolution only, nothing is logged or written.
    wlog = ReadableLogger(
        wandb_run, STAGE1, stage1_context(args, world_size),
        notation=NOTATION_STAGE1, header=stage1_header(args, world_size),
        run_dir=run_dir if is_main else None, summaries=SUMMARIES_STAGE1,
        printer=print if is_main else (lambda *_a, **_k: None),
    )

    # ----- determinism (different seed per rank for data + time sampling) -----
    torch.manual_seed(args.seed + rank)

    # ----- device + dtype -----
    if not torch.cuda.is_available():
        raise RuntimeError("Stage-1 EqM training requires CUDA. "
                           "Got torch.cuda.is_available() = False.")
    device = torch.device(f"cuda:{local_rank}")
    compute_dtype = torch.bfloat16 if args.bf16 else torch.float32

    # ----- VAE (frozen) -----
    if is_main:
        print("[stage1] loading SD VAE")
    vae = LatentEncoder(vae_id=args.vae_id).to(device)
    vae.eval()

    # ----- SiT (trainable) -----
    if is_main:
        print(f"[stage1] instantiating SiT-XL/2 at latent_size={args.latent_size}")
    if args.init_from_scratch:
        # Keep the random model initialization identical across ranks. The
        # per-rank seed is restored below for data/time/dropout randomness.
        torch.manual_seed(args.seed)
    model = SiT_XL_2(input_size=args.latent_size).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    if is_main:
        print(f"[stage1] parameter count: {n_params/1e6:.1f}M")
    if args.init_from_scratch:
        if is_main:
            print("[stage1] init-from-scratch: random SiT initialization; "
                  "skipping pretrained checkpoint load")
        torch.manual_seed(args.seed + rank)
    else:
        if is_main:
            print(f"[stage1] loading SiT checkpoint: {args.ckpt} "
                  f"(cache_dir={args.ckpt_cache_dir})")
        state = find_model(args.ckpt, cache_dir=args.ckpt_cache_dir, key=args.ckpt_key)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if is_main and missing:
            print(f"[stage1] WARN missing keys: {missing[:5]}"
                  f"{' ...' if len(missing) > 5 else ''}")
        if is_main and unexpected:
            print(f"[stage1] WARN unexpected keys: {unexpected[:5]}"
                  f"{' ...' if len(unexpected) > 5 else ''}")

    # Strip time conditioning BEFORE wrapping in DDP so frozen params aren't
    # registered as participants in the gradient sync.
    if args.time_conditioning == "strict":
        n_frozen = strip_time_conditioning(model)
        if is_main:
            print(f"[stage1] time-conditioning=strict: zeroed + froze "
                  f"t_embedder ({n_frozen/1e3:.1f}k params)")
    else:
        if is_main:
            print("[stage1] time-conditioning=zero: t_embedder still trainable")

    # Optional parameter freezing (before the DDP wrap so frozen params are
    # never registered with the reducer).
    if args.trainable:
        pat = re.compile(args.trainable)
        kept, frozen = [], 0
        for name, param in model.named_parameters():
            if pat.search(name):
                kept.append(name)
            else:
                param.requires_grad_(False)
                frozen += param.numel()
        if not kept:
            raise ValueError(f"--trainable {args.trainable!r} matches no parameter")
        if is_main:
            print(f"[stage1] --trainable {args.trainable!r}: {len(kept)} tensors "
                  f"trainable ({kept[0]} ... {kept[-1]}), {frozen/1e6:.1f}M params frozen")

    # SiT's LabelEmbedder has its own train-mode dropout default (0.1).
    # Disable that hidden source of randomness and apply CFG dropout
    # explicitly in the training loop so it is CLI-controlled + logged.
    internal_cfg_dropout = float(model.y_embedder.dropout_prob)
    model.y_embedder.dropout_prob = 0.0
    if is_main:
        print(f"[stage1] cfg dropout: explicit p={args.cfg_dropout_prob:.3f}, "
              f"null_class={args.null_class} "
              f"(disabled internal LabelEmbedder dropout p={internal_cfg_dropout:.3f})")

    # Attach the auxiliary time-prediction head BEFORE DDP wrap so its params
    # are broadcast from rank 0 and participate in gradient sync / EMA /
    # checkpoints. Warm-started backbones (--ckpt pointing at a Stage-1 ckpt)
    # will not carry head weights; strict=False loading leaves the fresh head.
    if args.time_pred_weight > 0.0:
        model.enable_time_prediction(args.time_pred_layer)
        if is_main:
            n_head = sum(p.numel() for p in model.t_pred_head.parameters())
            print(f"[stage1] time-pred head: layer={args.time_pred_layer} "
                  f"target={args.time_pred_target} weight={args.time_pred_weight:g} "
                  f"({n_head/1e3:.1f}k params)")

    model.train()

    # ----- DDP wrap -----
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                    find_unused_parameters=False)
        if args.grad_comm_bf16:
            from torch.distributed.algorithms.ddp_comm_hooks import default_hooks
            model.register_comm_hook(state=None, hook=default_hooks.bf16_compress_hook)
            if is_main:
                print("[stage1] DDP gradient all-reduce in bf16 (bf16_compress_hook)")
    base_model = unwrap(model)
    model_fn = (autonomous_forward_tpred(model) if args.time_pred_weight > 0.0
                else autonomous_forward(model))

    # ----- data -----
    latents_mode = is_latent_spec(args.shards)   # "latents:<dir>" = cached SD-VAE posteriors, no pixels, no VAE encode
    if latents_mode and args.hflip:
        if is_main:
            print("[stage1] --hflip ignored: the latent cache holds unflipped posteriors only", flush=True)
        args.hflip = False
    global_batch = args.batch_size * world_size * max(1, args.grad_accum_steps)
    if is_main:
        print(f"[stage1] building loader (batch_size={args.batch_size} per GPU, "
              f"global={global_batch}, num_workers={args.num_workers})")
    loader = build_imagenet_loader(
        shards=args.shards,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle_buffer=args.shuffle_buffer,
        shardshuffle=100,
        distributed=world_size > 1,
        hflip=args.hflip,
    )
    data_iter = cycle(loader)

    # ----- loss -----
    loss_fn = EqMLoss(EqMLossConfig(
        eps_train=args.eps_train,
        time_sampler=args.time_sampler,
        importance_k=args.importance_k,
        data_anchor_weight=args.data_anchor,
        parameterization=args.parameterization,
        time_pred_weight=args.time_pred_weight,
        time_pred_target=args.time_pred_target,
        clock=args.clock, clock_a=args.clock_a, clock_lambda=args.clock_lambda,
        clock_weight_eps=args.clock_weight_eps,
    ))

    # ----- fixed-batch probe + one-off output rescale (warm starts) -----
    # Both happen BEFORE the EMA shadow is created so the shadow starts from
    # the rescaled head. Skipped when resuming from latest.pt: the head is
    # then restored from the checkpoint anyway.
    fixed_batches = []
    if args.val_loss_batches > 0:
        fixed_batches = collect_fixed_batches(
            data_iter, vae, args.val_loss_batches, device=device, bf16=args.bf16)
    latest_exists = (run_dir / "checkpoints" / "latest.pt").exists()
    resuming = bool(args.resume and latest_exists)
    init_probe = {}
    if not args.init_from_scratch and not resuming:
        if args.init_output_scale == "auto":
            if not fixed_batches:
                raise ValueError("--init-output-scale auto needs --val-loss-batches > 0")
            cands = [float(c) for c in args.init_scale_candidates.split(",") if c.strip()]
            model.eval()
            scale, table = search_output_scale(
                base_model, model_fn, loss_fn, fixed_batches, cands,
                seed=args.val_seed, device=device, bf16=args.bf16)
            model.train()
            if is_main:
                print("[stage1] init output-scale search (scale, loss): "
                      + ", ".join(f"({s_:.4g}, {l_:.4f})" for s_, l_ in table))
            init_probe["init/scale_table"] = table
        else:
            scale = float(args.init_output_scale)
        if scale != 1.0:
            scale_output_head(base_model, scale)
        init_probe["init/scale"] = scale
        if is_main:
            print(f"[stage1] final_layer.linear scaled by {scale:.5g}")
    if fixed_batches and not resuming:
        model.eval()
        res = fixed_batch_loss(model_fn, loss_fn, fixed_batches,
                               seed=args.val_seed, device=device, bf16=args.bf16)
        model.train()
        init_probe["init/uw_loss"] = res["loss"]
        init_probe["init/b_data_rms"] = res["b_data_rms"]
        init_probe["init/x1_norm"] = res["x1_norm"]
        if is_main:
            print(f"[stage1] init probe: loss {res['loss']:.4f} | "
                  f"b_data_rms {res['b_data_rms']:.4f} | x1_norm {res['x1_norm']:.2f} (rank-0 batches)")
    if is_main and init_probe:
        with (run_dir / "init_probe.json").open("w") as f:
            json.dump(init_probe, f, indent=2)
        wlog.log({k: v for k, v in {
            "init/scale": init_probe.get("init/scale"),
            "init/objective": init_probe.get("init/uw_loss"),
            "init/b_data_rms": init_probe.get("init/b_data_rms"),
            "init/x1_norm": init_probe.get("init/x1_norm"),
        }.items() if v is not None}, step=0)

    # Reference copy of the starting trainable weights (rank 0, CPU) for the
    # "distance from start" diagnostic. On resume the head rescale was not
    # re-applied in this process, so re-apply the recorded scale to the copy.
    start_params = None
    if is_main and args.val_every > 0:
        start_params = {n: prm.detach().to("cpu", dtype=torch.float32, copy=True)
                        for n, prm in base_model.named_parameters() if prm.requires_grad}
        if resuming:
            try:
                with (run_dir / "init_probe.json").open() as f:
                    rec_scale = float(json.load(f).get("init/scale", 1.0))
            except (OSError, ValueError):
                rec_scale = 1.0
            if rec_scale != 1.0:
                for n in ("final_layer.linear.weight", "final_layer.linear.bias"):
                    if n in start_params:
                        start_params[n].mul_(rec_scale)

    # ----- optimizer + EMA -----
    # Filter to trainable params so AdamW doesn't allocate state for the
    # frozen t_embedder under --time-conditioning strict.
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable_params)
    if is_main:
        print(f"[stage1] trainable params: {n_trainable/1e6:.1f}M "
              f"({100.0 * n_trainable / n_params:.2f}% of total)")
    optimizer = AdamW(trainable_params, lr=args.lr,
                      betas=tuple(args.betas), weight_decay=args.weight_decay)
    # EMA only on rank 0: all ranks have identical params after each step
    # (DDP all-reduces gradients before optimizer.step), so a single shadow
    # is canonical.
    ema = EMA(base_model, decay=args.ema_decay) if is_main else None
    if is_main:
        print(f"[stage1] parameterization = {args.parameterization} "
              f"({'denoiser/x-pred: net outputs D, field = D(x)-x' if args.parameterization == 'denoiser' else 'velocity: net outputs b, field = b(x)'})")

    # ----- maybe resume (called on ALL ranks) -----------------------------
    # Each rank loads latest.pt from the shared Modal Volume. Loading on
    # every rank (rather than loading on rank 0 and broadcasting) ensures
    # the per-rank optimizer state dicts -- AdamW momenta -- are restored
    # consistently. Otherwise non-main ranks would step from zero momenta
    # and diverge from rank 0 after a single optimizer.step().
    start_step = maybe_resume(
        base_model=base_model,
        ema=ema,
        optimizer=optimizer,
        run_dir=run_dir,
        device=device,
        is_main=is_main,
        do_resume=args.resume,
    )
    barrier_if_distributed()

    # ----- fixed validation labels (rank 0 only; only rank 0 validates) ----
    val_labels = None
    if is_main:
        if args.val_classes is not None:
            if len(args.val_classes) != args.val_n_samples:
                print(f"[stage1] --val-classes had {len(args.val_classes)} entries, "
                      f"overriding --val-n-samples to match")
                args.val_n_samples = len(args.val_classes)
            val_labels = torch.tensor(args.val_classes, device=device, dtype=torch.long)
        else:
            label_gen = torch.Generator(device="cpu").manual_seed(args.val_seed)
            val_labels = torch.randint(
                0, 1000, (args.val_n_samples,), generator=label_gen,
            ).to(device=device, dtype=torch.long)
        print(f"[stage1] validation: every {args.val_every} steps, "
              f"{args.val_n_samples} samples x {args.val_num_steps} Euler steps, "
              f"cfg={args.val_cfg_scale:g}, "
              f"labels={val_labels.cpu().tolist()[:8]}"
              f"{'...' if args.val_n_samples > 8 else ''}")

    # ----- training loop -----
    if is_main:
        if start_step > 0:
            print(f"[stage1] starting training at step {start_step} (target {args.steps})")
        else:
            print("[stage1] starting training")
        if start_step >= args.steps:
            print(f"[stage1] start_step={start_step} >= --steps={args.steps}; "
                  f"nothing to do. Bump --steps to continue.")
    t_start = time.time()
    t_window = time.time()
    samples_window = 0

    import contextlib
    accum = max(1, args.grad_accum_steps)
    stats = WindowStats()                       # pooled over every rank-0 micro-batch between logs
    clip_count = torch.zeros((), device=device)
    steps_in_window = 0      # optimizer steps pooled in `stats` / clip_count
    timed_steps = 0          # optimizer steps inside the timing window (reset after validation too)
    data_wait = 0.0
    images_seen = start_step * args.batch_size * world_size * accum

    def _global_norm(tensors):
        return torch.stack([x.detach().float().norm() for x in tensors]).norm()

    def _rel_gap(named_ref):
        """||theta - ref|| / ||ref|| over trainable params; ref: name -> tensor (any device)."""
        num = torch.zeros((), device=device)
        den = torch.zeros((), device=device)
        with torch.no_grad():
            for n, prm in base_model.named_parameters():
                if n in named_ref:
                    ref = named_ref[n].to(device=device, dtype=torch.float32, non_blocking=True)
                    num += (prm.detach().float() - ref).pow(2).sum()
                    den += ref.pow(2).sum()
        return float((num / den.clamp_min(1e-30)).sqrt().item())

    for step in range(start_step, args.steps):
        optimizer.zero_grad(set_to_none=True)
        for micro in range(accum):
            t_wait = time.perf_counter()
            batch = next(data_iter)
            data_wait += time.perf_counter() - t_wait
            if latents_mode:
                # Cached posterior stats: draw the same posterior sample encode() would have drawn.
                mean, std, labels = batch
                cached_latents = vae.sample_from_stats(
                    mean.to(device, non_blocking=True), std.to(device, non_blocking=True)
                )
                n_in = cached_latents.shape[0]
            else:
                images, labels = batch
                images = images.to(device, non_blocking=True)
                cached_latents = None
                n_in = images.shape[0]
            labels = labels.to(device, non_blocking=True) if torch.is_tensor(labels) \
                else torch.as_tensor(labels, device=device)
            train_labels, cfg_drop_mask = apply_cfg_dropout(
                labels,
                dropout_prob=args.cfg_dropout_prob,
                null_class=args.null_class,
            )
            # Skip DDP gradient sync on all but the last micro-batch.
            sync_ctx = (model.no_sync() if isinstance(model, DDP) and micro < accum - 1
                        else contextlib.nullcontext())
            # Encode pixels -> latents under autocast; VAE runs no_grad internally.
            with sync_ctx, torch.amp.autocast("cuda", dtype=compute_dtype, enabled=args.bf16):
                latents = cached_latents if latents_mode else vae.encode(images)
                loss, metrics = loss_fn(model_fn, latents, train_labels)
                (loss / accum).backward()
            if is_main:
                pool_eqm_metrics(stats, metrics, drop_mask=cfg_drop_mask)
                stats.add_scalar("cfg_drop_frac", cfg_drop_mask.float().mean())
            images_seen += n_in * world_size
            if micro < accum - 1:
                samples_window += n_in * world_size
        # Clip only trainable params (frozen t_embedder has no grad anyway).
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clip)
        if is_main:
            stats.add_scalar("grad_norm", grad_norm)
            stats.add_scalar("grad_norm_max", grad_norm)
            clip_count += (grad_norm > args.grad_clip).float()
        steps_in_window += 1
        timed_steps += 1

        # Warmup-only LR schedule (constant after warmup).
        lr = scheduled_lr(step, warmup_steps=args.warmup_steps, base_lr=args.lr, schedule=args.lr_schedule,
                          decay_start=args.lr_decay_start, total_steps=args.steps, final_lr=args.lr_final)
        for g in optimizer.param_groups:
            g["lr"] = lr

        optimizer.step()
        if ema is not None:
            ema.update(base_model)
        # Throughput counter tracks the global mini-batch so the printed
        # samples/s is system-level, not per-GPU.
        samples_window += n_in * world_size

        if is_main and ((step + 1) % args.log_every == 0 or step == 0):
            now = time.time()
            dt = max(now - t_window, 1e-6)
            samples_per_sec = samples_window / dt
            m = stats.result()   # one device sync for everything pooled since the last log
            m.update({
                "lr": lr,
                "clip_frac": float(clip_count.item()) / max(steps_in_window, 1),
                "images_per_s": samples_per_sec,
                "sec_per_step": dt / max(timed_steps, 1),
                "data_wait_frac": min(1.0, data_wait / dt),
                "images_seen": images_seen,
                "epoch": images_seen / IMAGENET_TRAIN_IMAGES,
                "param_norm": float(_global_norm(trainable_params).item()),
            })
            if ema is not None:
                m["ema_gap"] = _rel_gap(ema.shadow)
            if device.type == "cuda":
                m["gpu_mem_gb"] = torch.cuda.max_memory_allocated(device) / 1e9
                torch.cuda.reset_peak_memory_stats(device)
            elapsed = now - t_start
            tp_str = (f"tp {m['time_pred_loss']:.4f}(mae {m['time_pred_mae']:.3f}) | "
                      if "time_pred_loss" in m else "")
            print(f"[stage1] step {step+1:>7d} | "
                  f"loss {m.get('total_loss', float('nan')):.4f} | "
                  f"{tp_str}"
                  f"lr {lr:.2e} | "
                  f"grad_norm {m.get('grad_norm', float('nan')):.2f} | "
                  f"{samples_per_sec:.1f} samples/s | "
                  f"data wait {100 * m['data_wait_frac']:.0f}% | "
                  f"elapsed {elapsed:.0f}s")
            wlog.log(m, step=step + 1)
            stats.reset()
            clip_count.zero_()
            steps_in_window = 0
            timed_steps = 0
            data_wait = 0.0
            t_window = now
            samples_window = 0

        if args.val_every > 0 and (
            (step + 1) % args.val_every == 0 or (step + 1) == args.steps
        ):
            if is_main:
                probe = {}
                if fixed_batches:
                    def _subset(res):
                        return {k: v for k, v in res.items()
                                if k in ("loss", "b_data_rms")
                                or k.startswith("eqm_raw_t") or k.startswith("eqm_relerr_t")}
                    base_model.eval()
                    raw = fixed_batch_loss(model_fn, loss_fn, fixed_batches,
                                           seed=args.val_seed, device=device, bf16=args.bf16)
                    probe.update({f"probe_raw/{k}": v for k, v in _subset(raw).items()})
                    msg = f"objective raw {raw['loss']:.4f}"
                    msg_b = f"RMS b(x1) raw {raw['b_data_rms']:.4f}"
                    if ema is not None:
                        with ema.apply_to(base_model):
                            em = fixed_batch_loss(model_fn, loss_fn, fixed_batches,
                                                  seed=args.val_seed, device=device, bf16=args.bf16)
                        probe.update({f"probe_ema/{k}": v for k, v in _subset(em).items()})
                        msg += f" ema {em['loss']:.4f}"
                        msg_b += f" ema {em['b_data_rms']:.4f}"
                    base_model.train()
                    print(f"[stage1] val probe step {step+1}: {msg} | {msg_b}")
                if start_params is not None:
                    probe["init_gap"] = _rel_gap(start_params)
                    print(f"[stage1] step {step+1}: ||theta - theta_start|| / ||theta_start|| = "
                          f"{probe['init_gap']:.4g}")
                if probe:
                    wlog.log(probe, step=step + 1)
                run_validation(
                    step=step + 1,
                    model=base_model,
                    vae=vae,
                    ema=ema,
                    val_labels=val_labels,
                    val_seed=args.val_seed,
                    n_samples=args.val_n_samples,
                    latent_size=args.latent_size,
                    num_steps=args.val_num_steps,
                    eps_stop=args.val_eps_stop,
                    cfg_scale=args.val_cfg_scale,
                    null_class=args.null_class,
                    device=device,
                    compute_dtype=compute_dtype,
                    bf16=args.bf16,
                    run_dir=run_dir,
                    wlog=wlog,
                    parameterization=args.parameterization,
                )
                # Reset the timing window so validation cost does not
                # contaminate the next throughput / data-wait readings.
                t_window = time.time()
                samples_window = 0
                timed_steps = 0
                data_wait = 0.0
            # All ranks must reach the barrier together.
            barrier_if_distributed()

        if (step + 1) % args.save_every == 0 or (step + 1) == args.steps:
            if is_main:
                path = save_checkpoint(run_dir, step + 1, base_model, ema, optimizer, args)
                if args.on_keep_cmd and args.keep_every > 0 and (step + 1) % args.keep_every == 0:
                    import subprocess
                    cmd = args.on_keep_cmd.format(ckpt=path, step=step + 1)
                    try:
                        subprocess.Popen(cmd, shell=True, start_new_session=True)
                        print(f"[stage1] on-keep: {cmd}", flush=True)
                    except OSError as e:
                        print(f"[stage1] WARN on-keep command failed: {e!r}", flush=True)
                print(f"[stage1] saved checkpoint -> {path}")
                save_checkpoint(run_dir, step + 1, base_model, ema, optimizer, args, tag="latest")
                _cleanup_old_checkpoints(run_dir, args.keep_last_checkpoints, args.keep_every)
            barrier_if_distributed()

    if is_main:
        print(f"[stage1] done in {time.time() - t_start:.0f}s")
        if wandb_run is not None:
            wandb_run.finish()

    cleanup_distributed()


if __name__ == "__main__":
    main()
