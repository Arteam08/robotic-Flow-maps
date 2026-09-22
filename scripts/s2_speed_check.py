"""Speed and gradient check of Stage-2 precision options on a trained student.

Loads the Stage-1 teacher and a Stage-2 student checkpoint (raw weights), draws real latents from the
SD-VAE cache and compares three precision settings of the lag_semi loss:

  fp32   production: student loss in fp32, teacher in bf16
  nonjvp --student-bf16-nonjvp (field, anchor, endpoint, semigroup student passes in bf16)
  jvp    --student-bf16-nonjvp --student-bf16-jvp (the Lagrangian JVP pass in bf16 too)

(1) Full-batch gradient (``--micro`` micro-batches accumulated, as the trainer does) of each setting vs fp32:
    cosine and |g - g_fp32| / |g_fp32|. Baseline = the same statistics between the fp32 gradients of two
    different batches (the optimizer's own noise).
(2) Per-term attribution on single micro-batches: |g_bf16 - g_fp32| / |g_fp32| for each loss term alone.
(3) Single-GPU throughput (forward + backward) and peak memory vs per-GPU batch for fp32 and all-bf16.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eqfm.eval_utils import autonomous_forward, load_stage1_model, maybe_autocast  # noqa: E402
from eqfm.models.sit.models import SiT_models  # noqa: E402
from eqfm.stage2_losses import Stage2FlowMapLoss, Stage2LossConfig  # noqa: E402
from eqfm.vae import LatentEncoder  # noqa: E402


def make_loss(cfg: Stage2LossConfig, semi_weight: float = 1.0, lag_scale: float = 1.0):
    class LagSemi(Stage2FlowMapLoss):  # same wrapper as train_stage2_flowmap.main
        def lagrangian_loss(self, student, teacher_fn, x0, labels, labels_in, drop_mask, sigma):
            lag = super().lagrangian_loss(student, teacher_fn, x0, labels, labels_in, drop_mask, sigma)
            sg, sgp = self.sample_semigroup_pair(x0.shape[0], x0.device, x0.dtype)
            with self.nonjvp_autocast(x0.device):
                semi = self.semigroup_loss(student, x0, labels_in, sg, sgp)
            return lag_scale * lag + semi_weight * semi
    return LagSemi(cfg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default=None)
    ap.add_argument("--student", default=None)
    ap.add_argument("--latents", default=None)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--micro", type=int, default=16, help="micro-batches per full-batch gradient")
    ap.add_argument("--big-batches", type=int, nargs="*", default=[8, 12, 16, 24, 32])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    dev = torch.device("cuda")

    teacher, step, w = load_stage1_model(Path(args.teacher), latent_size=32, device=dev, use_ema=True)
    for p in teacher.parameters():
        p.requires_grad_(False)
    teacher.eval()
    tm = autonomous_forward(teacher)

    def teacher_fn(x, y):
        with torch.no_grad(), maybe_autocast(dev, torch.bfloat16, True):
            return tm(x, y).float()

    def teacher_grad_fn(x, y):
        with maybe_autocast(dev, torch.bfloat16, True):
            return tm(x, y).float()

    student = SiT_models["SiT-XL/2"](input_size=32, class_dropout_prob=0.0).to(dev)
    state = torch.load(args.student, map_location="cpu", mmap=True, weights_only=False)
    student.load_state_dict(state["model"])
    s_step = int(state.get("step", -1))
    del state
    student.train()
    print(f"teacher step={step}; student {args.student} step={s_step}; gpu={torch.cuda.get_device_name(0)}", flush=True)

    kw = dict(objective="lagrangian", interp_time_sampler="uniform", sigma_sampler="uniform",
              sigma_min=1e-4, sigma_max=0.97, adaptive_weight_p=0.5, adaptive_weight_eps=1e-3,
              diag_weight=1.0, distill_weight=1.0, terminal_weight=1.0,
              terminal_teacher_norm_weight=1.0, terminal_norm_max_batch=4,
              guidance_mode="guided", teacher_cfg_scale=1.5,
              cfg_match_uncond_norm=True, cfg_match_guided_norm=True,
              cfg_mode="all", cfg_dropout_prob=0.0, null_class=1000)
    settings = {
        "fp32": make_loss(Stage2LossConfig(**kw)),
        "nonjvp": make_loss(Stage2LossConfig(**kw, student_bf16_nonjvp=True)),
        "jvp": make_loss(Stage2LossConfig(**kw, student_bf16_nonjvp=True, student_bf16_jvp=True)),
    }

    shard = sorted(glob.glob(f"{args.latents}/shard_*.npz"))[7]
    with np.load(shard) as z:
        mean, std, label = z["mean"], z["std"], z["label"]
    gen = torch.Generator(device="cpu").manual_seed(0)
    idx = torch.randperm(len(label), generator=gen)

    def latents(n, off):
        i = idx[off:off + n].numpy()
        m = torch.from_numpy(mean[i]).to(dev)
        s = torch.from_numpy(std[i]).to(dev)
        return LatentEncoder.sample_from_stats(m, s), torch.from_numpy(label[i].astype(np.int64)).to(dev)

    def run(loss_fn, x, y, seed, want_grad=True):
        student.zero_grad(set_to_none=True)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.synchronize()
        t0 = time.time()
        loss, metrics = loss_fn(student, teacher_fn, x, y, teacher_grad_fn=teacher_grad_fn)
        loss.backward()
        torch.cuda.synchronize()
        dt = time.time() - t0
        g = torch.cat([p.grad.detach().flatten().float().cpu() for p in student.parameters()
                       if p.grad is not None]) if want_grad else None
        student.zero_grad(set_to_none=True)
        return float(loss), dt, g

    def cos(a, b):
        a, b = a.double(), b.double()
        return float((a @ b) / (a.norm() * b.norm()))

    def rel(a, b):
        return float((a.double() - b.double()).norm() / b.double().norm())

    n_rows = len(label)

    def grad_of(fn, micro, off, seed0):
        """Accumulated gradient (sum over ``micro`` micro-batches / micro) as the trainer builds it."""
        acc = None
        for j in range(micro):
            x, y = latents(args.batch_size, (off + j * args.batch_size) % (n_rows - args.batch_size))
            _, _, g = run(fn, x, y, seed0 + j)
            acc = g if acc is None else acc.add_(g)
        return acc / micro

    summary = {"gpu": torch.cuda.get_device_name(0), "batch_size": args.batch_size, "student_step": s_step,
               "micro_batches_per_full_grad": args.micro}
    rows = []
    # (1) full-batch gradients (micro x batch samples): precision settings vs fp32, noise floor = another batch.
    g32 = grad_of(settings["fp32"], args.micro, 0, 100)
    g32_other = grad_of(settings["fp32"], args.micro, 10_000, 900)
    summary["baseline_noise_full_grad_cos_between_batches"] = cos(g32_other, g32)
    summary["baseline_noise_full_grad_rel_between_batches"] = rel(g32_other, g32)
    del g32_other
    for name in ("nonjvp", "jvp"):
        g = grad_of(settings[name], args.micro, 0, 100)
        summary[f"{name}_full_grad_cos_vs_fp32"] = cos(g, g32)
        summary[f"{name}_full_grad_rel_vs_fp32"] = rel(g, g32)
        del g
    print("FULL-BATCH " + json.dumps(summary), flush=True)
    del g32

    # (2) per-term attribution on single micro-batches: which term carries the bf16 deviation.
    terms = {"field": "diag_weight", "anchor": "terminal_weight", "endpoint": "terminal_teacher_norm_weight",
             "lagrangian": None, "semigroup": None}
    for i in range(args.k):
        x, y = latents(args.batch_size, (20_000 + i * args.batch_size) % (n_rows - args.batch_size))
        r = {"batch": i}
        for term, wkey in terms.items():
            tkw = dict(kw)
            for k2 in ("diag_weight", "terminal_weight", "terminal_teacher_norm_weight"):
                tkw[k2] = 1.0 if k2 == wkey else 1e-12   # tiny, not 0: every pass (and RNG draw) still runs
            tkw["distill_weight"] = 1.0 if term in ("lagrangian", "semigroup") else 1e-12
            semi_w = 1.0 if term == "semigroup" else 1e-12
            lag_scale = 1e-12 if term == "semigroup" else 1.0
            gs = {}
            for name, extra in (("fp32", {}), ("jvp", dict(student_bf16_nonjvp=True, student_bf16_jvp=True))):
                fn = make_loss(Stage2LossConfig(**tkw, **extra), semi_w, lag_scale)
                _, _, gs[name] = run(fn, x, y, 3000 + i)
            r[f"{term}_gradnorm_fp32"] = float(gs["fp32"].double().norm())
            r[f"{term}_rel_bf16_vs_fp32"] = rel(gs["jvp"], gs["fp32"])
            del gs
        rows.append(r)
        print(json.dumps(r), flush=True)
    summary["per_term_rows"] = rows

    # (3) throughput vs per-GPU batch (forward + backward, no optimizer), fp32 and all-bf16.
    big = {}
    for name in ("fp32", "jvp"):
        for bs in args.big_batches:
            try:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                ts = []
                for j in range(3):
                    x, y = latents(bs, (30_000 + j * bs) % (n_rows - bs))
                    _, dt, _ = run(settings[name], x, y, 5000 + j, want_grad=False)
                    ts.append(dt)
                big[f"{name}_bs{bs}"] = {"samples_per_s": bs / (sum(ts[1:]) / 2),
                                         "peak_GB": torch.cuda.max_memory_allocated() / 2**30}
                print(name, bs, big[f"{name}_bs{bs}"], flush=True)
            except torch.OutOfMemoryError:
                print(name, bs, "OOM", flush=True)
                student.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                break
    summary["throughput_single_gpu"] = big
    print("SUMMARY " + json.dumps(summary, indent=1), flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"rows": rows, "summary": summary}, indent=1))


if __name__ == "__main__":
    main()
