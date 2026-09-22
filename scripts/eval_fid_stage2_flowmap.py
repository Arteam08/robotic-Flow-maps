#!/usr/bin/env python
"""Generate Stage-2 flow-map samples and evaluate/save FID batches.

This mirrors ``scripts/eval_fid.py`` for Stage-1 EqM, but samples a trained
compactified flow-map student using one of:

  * offdiag: K-fold compactified composition; ``--offdiag-schedule equal``
    uses K jumps of 1 - eps^(1/K), ``paper`` uses K-1 jumps of s0 then X_1
  * diag_euler: 250-step autonomous Euler sampling with v_{sigma=0}
  * terminal: one-shot X_1(z), kept as an optional diagnostic

Pass ``--terminal-project`` to apply the learned terminal projector
``P(x) = X_1(x)`` after an offdiag/diag sample, i.e. to project the finite-time
sample to the learned infinite-time manifold.

It can stream generated samples to an OpenAI evaluator-compatible .npz and can
optionally run the OpenAI guided-diffusion evaluator immediately afterward.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from eqfm.eval_utils import maybe_autocast  # noqa: E402
from eqfm.flowmap import flow_map  # noqa: E402
from eqfm.metrics import InceptionFeatureExtractor, fid_from_stats, stats_from_features  # noqa: E402
from eqfm.models.sit.models import SiT_models  # noqa: E402
from eqfm.training import EMA  # noqa: E402
from eqfm.vae import LatentEncoder  # noqa: E402

from eval_fid import (  # noqa: E402
    SampleNpzWriter,
    _images_to_uint8_nhwc,
    _to_unit,
    compute_reference_stats,
    run_oai_evaluator,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--weights", choices=["ema", "raw"], default="ema")
    p.add_argument("--model", choices=sorted(SiT_models.keys()), default=None,
                   help="Defaults to the model saved in checkpoint args.")
    p.add_argument("--latent-size", type=int, default=None,
                   help="Defaults to checkpoint args latent_size, else 32.")

    p.add_argument("--sampler", choices=["offdiag", "diag_euler", "terminal"],
                   required=True)
    p.add_argument("--terminal-repeats", type=int, default=1,
                   help="terminal sampler only: apply X_1 this many times in a row "
                        "(2 = X_1(X_1(z)), the 'X1^2' protocol).")
    p.add_argument("--offdiag-steps", type=int, default=1,
                   help="K for K-fold compactified composition when "
                        "--sampler offdiag.")
    p.add_argument("--offdiag-schedule", choices=["equal", "paper"], default="equal",
                   help="equal: K jumps of 1 - eps_stop^(1/K). paper: K-1 jumps "
                        "of --offdiag-s0, then one terminal jump X_1.")
    p.add_argument("--offdiag-s0", type=float, default=0.5,
                   help="Small-jump size for --offdiag-schedule paper.")
    p.add_argument("--diag-num-steps", type=int, default=250,
                   help="Euler steps when --sampler diag_euler.")
    p.add_argument("--eps-stop", type=float, default=1e-3)
    p.add_argument("--terminal-project", action=argparse.BooleanOptionalAction,
                   default=False,
                   help="After offdiag/diag sampling, apply X_1 once as a "
                        "terminal manifold projection. No-op for "
                        "--sampler terminal, which is already X_1(z).")

    p.add_argument("--num-samples", type=int, default=50000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--log-gen-every", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--save-samples-npz", type=Path, default=None,
                   help="Save generated samples as arr_0 NHWC uint8 .npz.")
    p.add_argument("--overwrite-samples-npz", action="store_true", default=False)
    p.add_argument("--skip-pytorch-fid", action="store_true", default=False)
    p.add_argument("--oai-evaluator", type=Path, default=None)
    p.add_argument("--oai-ref", type=Path, default=None)
    p.add_argument("--oai-python", default=sys.executable)

    p.add_argument("--ref-stats", type=Path, default=None)
    p.add_argument("--ref-raw", type=Path, default=None)
    p.add_argument("--ref-shards", default=None)
    p.add_argument("--ref-num", type=int, default=None)
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--ref-num-workers", type=int, default=4)
    p.add_argument("--log-ref-every", type=int, default=20)

    p.add_argument("--vae-id", default="stabilityai/sd-vae-ft-ema")
    p.add_argument("--vae-cache-dir", default=None)
    p.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def load_stage2_model(args: argparse.Namespace, device: torch.device):
    print(f"[stage2-fid] ckpt = {args.ckpt}", flush=True)
    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    ckpt_args = state.get("args", {}) if isinstance(state, dict) else {}
    model_name = args.model or ckpt_args.get("model") or "SiT-XL/2"
    latent_size = args.latent_size or int(ckpt_args.get("latent_size", 32))
    embed_rows = state["model"]["y_embedder.embedding_table.weight"].shape[0]
    if embed_rows == 1000:
        class_dropout_prob = 0.0
    elif embed_rows == 1001:
        class_dropout_prob = 0.1
    else:
        raise ValueError(f"unexpected class embedding rows: {embed_rows}")

    print(f"[stage2-fid] checkpoint step = {state.get('step', state.get('steps', -1))}",
          flush=True)
    print(f"[stage2-fid] model={model_name} latent_size={latent_size} "
          f"class_embed_rows={embed_rows}", flush=True)
    model = SiT_models[model_name](
        input_size=latent_size,
        class_dropout_prob=class_dropout_prob,
    ).to(device)
    model.load_state_dict(state["model"], strict=True)
    model.eval()

    ema = None
    if args.weights == "ema":
        if "ema" not in state:
            raise KeyError(f"{args.ckpt} has no EMA state")
        ema = EMA(model)
        ema.load_state_dict(state["ema"])
        print("[stage2-fid] using EMA weights", flush=True)
    else:
        print("[stage2-fid] using raw weights", flush=True)
    return model, ema, latent_size, state.get("step", state.get("steps", -1))


@torch.no_grad()
def sample_stage2_batch(
    model: torch.nn.Module,
    z0: torch.Tensor,
    labels: torch.Tensor,
    *,
    sampler: str,
    offdiag_steps: int,
    diag_num_steps: int,
    offdiag_schedule: str = "equal",
    offdiag_s0: float = 0.5,
    terminal_repeats: int = 1,
    eps_stop: float,
    device: torch.device,
    compute_dtype: torch.dtype,
    bf16: bool,
) -> torch.Tensor:
    if sampler == "terminal":
        if terminal_repeats <= 0:
            raise ValueError("--terminal-repeats must be positive")
        sigma = torch.ones(z0.shape[0], device=device, dtype=z0.dtype)
        x = z0
        for _ in range(terminal_repeats):
            with maybe_autocast(device, compute_dtype, bf16):
                x = flow_map(model, x, sigma, labels).float()
        return x

    if sampler == "offdiag":
        if offdiag_steps <= 0:
            raise ValueError("--offdiag-steps must be positive")
        if offdiag_schedule == "equal":
            steps = [1.0 - eps_stop ** (1.0 / offdiag_steps)] * offdiag_steps
        elif offdiag_schedule == "paper":
            steps = [offdiag_s0] * (offdiag_steps - 1) + [1.0]
        else:
            raise ValueError(f"unknown offdiag schedule: {offdiag_schedule}")
        x = z0.float()
        for sigma_step in steps:
            sigma = torch.full((z0.shape[0],), sigma_step, device=device, dtype=z0.dtype)
            with maybe_autocast(device, compute_dtype, bf16):
                x = flow_map(model, x, sigma, labels).float()
        return x

    if sampler == "diag_euler":
        h = -math.log(eps_stop) / diag_num_steps
        sigma0 = torch.zeros(z0.shape[0], device=device, dtype=z0.dtype)
        x = z0.float()
        for _ in range(diag_num_steps):
            with maybe_autocast(device, compute_dtype, bf16):
                b = model(x, sigma0, labels).float()
            x = x + h * b
        return x

    raise ValueError(f"unknown sampler: {sampler}")


@torch.no_grad()
def compute_stage2_gen_stats(args, model, ema, vae, incep, device, compute_dtype, latent_size):
    n = args.num_samples
    labels_all = torch.arange(n, dtype=torch.long) % 1000
    writer = (
        SampleNpzWriter(args.save_samples_npz, n, args.overwrite_samples_npz)
        if args.save_samples_npz is not None else None
    )
    feats = [] if incep is not None else None
    ctx = ema.apply_to(model) if ema is not None else contextlib.nullcontext()

    produced = 0
    batch_idx = 0
    t0 = time.time()
    print(f"[stage2-fid] generating {n} samples sampler={args.sampler} terminal_repeats={args.terminal_repeats} "
          f"offdiag_K={args.offdiag_steps} schedule={args.offdiag_schedule} "
          f"s0={args.offdiag_s0} diag_K={args.diag_num_steps} "
          f"terminal_project={args.terminal_project} batch={args.batch_size}",
          flush=True)
    with ctx:
        gen = torch.Generator(device=device)
        while produced < n:
            batch_idx += 1
            bsz = min(args.batch_size, n - produced)
            y = labels_all[produced:produced + bsz].to(device)
            gen.manual_seed(args.seed + produced)
            z0 = torch.randn(
                bsz, 4, latent_size, latent_size, device=device, generator=gen
            )
            latents = sample_stage2_batch(
                model,
                z0,
                y,
                sampler=args.sampler,
                terminal_repeats=args.terminal_repeats,
                offdiag_steps=args.offdiag_steps,
                diag_num_steps=args.diag_num_steps,
                offdiag_schedule=args.offdiag_schedule,
                offdiag_s0=args.offdiag_s0,
                eps_stop=args.eps_stop,
                device=device,
                compute_dtype=compute_dtype,
                bf16=args.bf16,
            )
            if args.terminal_project and args.sampler != "terminal":
                sigma1 = torch.ones(bsz, device=device, dtype=latents.dtype)
                with maybe_autocast(device, compute_dtype, args.bf16):
                    latents = flow_map(model, latents, sigma1, y).float()
            with torch.amp.autocast(device.type, dtype=compute_dtype, enabled=args.bf16):
                images = vae.decode(latents)
            if writer is not None:
                writer.write(_images_to_uint8_nhwc(images), produced)
            if incep is not None:
                feats.append(incep(_to_unit(images.float())))
            produced += bsz
            if (
                batch_idx == 1
                or batch_idx % max(1, args.log_gen_every) == 0
                or produced >= n
            ):
                print(f"[stage2-fid]   gen {produced}/{n} "
                      f"({produced / max(time.time() - t0, 1e-6):.2f} img/s)",
                      flush=True)
    if writer is not None:
        writer.close()
    if incep is None:
        return None, None
    return stats_from_features(np.concatenate(feats, axis=0)[:n])


def main() -> None:
    args = parse_args()
    if args.skip_pytorch_fid and args.save_samples_npz is None:
        raise ValueError("--skip-pytorch-fid is only useful with --save-samples-npz")
    if args.oai_evaluator is not None and args.save_samples_npz is None:
        raise ValueError("--oai-evaluator requires --save-samples-npz")
    if args.eps_stop <= 0.0 or args.eps_stop >= 1.0:
        raise ValueError("--eps-stop must lie in (0, 1)")
    if not 0.0 < args.offdiag_s0 < 1.0:
        raise ValueError("--offdiag-s0 must lie in (0, 1)")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    compute_dtype = torch.bfloat16 if args.bf16 and device.type == "cuda" else torch.float32
    torch.manual_seed(args.seed)

    model, ema, latent_size, ckpt_step = load_stage2_model(args, device)
    print(f"[stage2-fid] loading VAE: {args.vae_id}", flush=True)
    vae = LatentEncoder(args.vae_id, cache_dir=args.vae_cache_dir).to(device).eval()

    incep = None
    mu_ref = sigma_ref = None
    ref_path = None
    actual_ref_num = None
    fid = None
    if not args.skip_pytorch_fid:
        print("[stage2-fid] loading Inception feature extractor", flush=True)
        incep = InceptionFeatureExtractor(device)
        mu_ref, sigma_ref, ref_path, actual_ref_num = compute_reference_stats(
            args, incep, device
        )

    mu_gen, sigma_gen = compute_stage2_gen_stats(
        args, model, ema, vae, incep, device, compute_dtype, latent_size
    )
    if not args.skip_pytorch_fid:
        fid = fid_from_stats(mu_gen, sigma_gen, mu_ref, sigma_ref)
        print(f"[stage2-fid] FID = {fid:.4f}", flush=True)

    out = {
        "fid": fid,
        "ckpt": str(args.ckpt),
        "ckpt_step": ckpt_step,
        "weights": args.weights,
        "sampler": args.sampler,
        "terminal_repeats": args.terminal_repeats,
        "offdiag_steps": args.offdiag_steps,
        "offdiag_schedule": args.offdiag_schedule,
        "offdiag_s0": args.offdiag_s0,
        "diag_num_steps": args.diag_num_steps,
        "eps_stop": args.eps_stop,
        "terminal_project": args.terminal_project,
        "num_samples": args.num_samples,
        "ref_num": actual_ref_num,
        "ref_stats_path": str(ref_path) if ref_path is not None else None,
        "samples_npz": str(args.save_samples_npz) if args.save_samples_npz else None,
        "skip_pytorch_fid": args.skip_pytorch_fid,
        "oai_evaluator": str(args.oai_evaluator) if args.oai_evaluator else None,
        "oai_ref": str(args.oai_ref) if args.oai_ref else None,
        "seed": args.seed,
    }
    fid_json = args.output_dir / "fid.json"
    with fid_json.open("w") as f:
        json.dump(out, f, indent=2, sort_keys=True)
    print(f"[stage2-fid] wrote {fid_json}", flush=True)
    run_oai_evaluator(args)


if __name__ == "__main__":
    main()
