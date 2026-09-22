#!/usr/bin/env python
"""Sample a trained Stage-2 compactified flow map.

This is the standalone version of the Stage-2 validation sampler. It compares:

  * diagonal Euler sampling with v_{sigma=0}, K steps in autonomous time
  * off-diagonal K-fold compactified flow-map composition
  * optional terminal one-shot projector X_1
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from eqfm.eval_utils import maybe_autocast  # noqa: E402
from eqfm.flowmap import flow_map  # noqa: E402
from eqfm.models.sit.models import SiT_models  # noqa: E402
from eqfm.sampling import save_grid  # noqa: E402
from eqfm.training import EMA  # noqa: E402
from eqfm.vae import LatentEncoder  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--weights", choices=["ema", "raw"], default="ema")
    p.add_argument("--model", choices=sorted(SiT_models.keys()), default=None,
                   help="Defaults to the model saved in the checkpoint args.")
    p.add_argument("--latent-size", type=int, default=None,
                   help="Defaults to the latent_size saved in the checkpoint args.")
    p.add_argument("--n-samples", type=int, default=16)
    p.add_argument("--val-seed", type=int, default=0,
                   help="Seed for fixed noise and default label draw.")
    p.add_argument("--val-classes", type=int, nargs="+", default=None,
                   help="Explicit ImageNet class indices. If omitted, labels "
                        "are drawn from torch.randint with --val-seed, matching "
                        "Stage-1/Stage-2 validation.")
    p.add_argument("--diag-num-steps", type=int, default=250)
    p.add_argument("--offdiag-steps", type=int, nargs="+", default=[1, 2, 4, 8])
    p.add_argument("--eps-stop", type=float, default=1e-3)
    p.add_argument("--save-terminal", action=argparse.BooleanOptionalAction,
                   default=True)
    p.add_argument("--vae-id", default="stabilityai/sd-vae-ft-ema")
    p.add_argument("--vae-cache-dir", default=None)
    p.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def mean_norm(x: torch.Tensor) -> float:
    return float(x.float().flatten(1).norm(dim=1).mean().item())


def load_model(args: argparse.Namespace, device: torch.device):
    print(f"[stage2-sample] loading checkpoint: {args.ckpt}", flush=True)
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    ckpt_args = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}
    model_name = args.model or ckpt_args.get("model") or "SiT-XL/2"
    latent_size = args.latent_size or int(ckpt_args.get("latent_size", 32))
    embed_rows = ckpt["model"]["y_embedder.embedding_table.weight"].shape[0]
    if embed_rows == 1000:
        class_dropout_prob = 0.0
    elif embed_rows == 1001:
        class_dropout_prob = 0.1
    else:
        raise ValueError(
            "Unexpected class embedding table size in checkpoint: "
            f"{embed_rows}. Expected 1000 or 1001 rows."
        )

    print(f"[stage2-sample] model={model_name} latent_size={latent_size} "
          f"class_embed_rows={embed_rows}", flush=True)
    model = SiT_models[model_name](
        input_size=latent_size,
        class_dropout_prob=class_dropout_prob,
    )
    msg = model.load_state_dict(ckpt["model"], strict=True)
    print(f"[stage2-sample] loaded raw model: {msg}", flush=True)
    model = model.to(device).eval()

    ema = None
    if args.weights == "ema":
        if "ema" not in ckpt:
            raise KeyError(f"{args.ckpt} has no EMA state, but --weights ema was requested")
        ema = EMA(model)
        ema.load_state_dict(ckpt["ema"])
        print("[stage2-sample] using EMA weights", flush=True)
    else:
        print("[stage2-sample] using raw weights", flush=True)

    step = ckpt.get("step", ckpt.get("steps", None))
    return model, ema, latent_size, step


def validation_labels(args: argparse.Namespace, device: torch.device) -> torch.Tensor:
    if args.val_classes is not None:
        labels = torch.tensor(args.val_classes, dtype=torch.long)
        if labels.numel() != args.n_samples:
            print(f"[stage2-sample] --val-classes has {labels.numel()} entries; "
                  f"overriding --n-samples", flush=True)
            args.n_samples = int(labels.numel())
    else:
        gen = torch.Generator(device="cpu").manual_seed(args.val_seed)
        labels = torch.randint(0, 1000, (args.n_samples,), generator=gen)
    return labels.to(device=device, dtype=torch.long)


@torch.no_grad()
def main() -> None:
    args = parse_args()
    if args.eps_stop <= 0.0 or args.eps_stop >= 1.0:
        raise ValueError(f"--eps-stop must be in (0, 1), got {args.eps_stop}")
    for k in args.offdiag_steps:
        if k <= 0:
            raise ValueError(f"--offdiag-steps must be positive, got {k}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    compute_dtype = torch.bfloat16 if args.bf16 and device.type == "cuda" else torch.float32
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model, ema, latent_size, ckpt_step = load_model(args, device)
    labels = validation_labels(args, device)
    print(f"[stage2-sample] labels={labels.cpu().tolist()}", flush=True)

    print(f"[stage2-sample] loading VAE: {args.vae_id}", flush=True)
    vae = LatentEncoder(args.vae_id, cache_dir=args.vae_cache_dir).to(device).eval()

    gen = torch.Generator(device=device).manual_seed(args.val_seed)
    z0 = torch.randn(args.n_samples, 4, latent_size, latent_size,
                     device=device, generator=gen)

    def decode_save(name: str, latents: torch.Tensor) -> Path:
        with maybe_autocast(device, compute_dtype, args.bf16):
            images = vae.decode(latents)
        path = args.output_dir / f"{name}.png"
        save_grid(images.float().cpu(), path)
        print(f"[stage2-sample] wrote {path}", flush=True)
        return path

    def diagonal_euler() -> tuple[torch.Tensor, dict[str, float]]:
        h = -math.log(args.eps_stop) / max(1, args.diag_num_steps)
        x = z0.clone()
        norms = []
        sigma0 = torch.zeros(x.shape[0], device=device, dtype=x.dtype)
        for _ in range(args.diag_num_steps):
            with maybe_autocast(device, compute_dtype, args.bf16):
                b = model(x, sigma0, labels).float()
            norms.append(mean_norm(b))
            x = x + h * b
        return x, {
            "velocity_norm_first": norms[0] if norms else 0.0,
            "velocity_norm_last": norms[-1] if norms else 0.0,
            "velocity_norm_max": max(norms) if norms else 0.0,
            "latent_norm_first": mean_norm(z0),
            "latent_norm_last": mean_norm(x),
        }

    def offdiag_compose(k: int) -> torch.Tensor:
        sigma_step = 1.0 - args.eps_stop ** (1.0 / k)
        sigma = torch.full((z0.shape[0],), sigma_step, device=device, dtype=z0.dtype)
        x = z0.clone()
        for _ in range(k):
            with maybe_autocast(device, compute_dtype, args.bf16):
                x = flow_map(model, x, sigma, labels).float()
        return x

    results = {
        "ckpt": str(args.ckpt),
        "ckpt_step": ckpt_step,
        "weights": args.weights,
        "n_samples": args.n_samples,
        "labels": labels.cpu().tolist(),
        "val_seed": args.val_seed,
        "eps_stop": args.eps_stop,
        "diag_num_steps": args.diag_num_steps,
        "offdiag_steps": args.offdiag_steps,
        "paths": {},
        "diagnostics": {},
    }

    ctx = ema.apply_to(model) if ema is not None else contextlib.nullcontext()
    with ctx:
        diag_latents, diag_summary = diagonal_euler()
        results["paths"]["diag_euler"] = str(
            decode_save(f"diag_euler_K{args.diag_num_steps}", diag_latents)
        )
        results["diagnostics"]["diag_euler"] = diag_summary

        if args.save_terminal:
            sigma1 = torch.ones(z0.shape[0], device=device, dtype=z0.dtype)
            with maybe_autocast(device, compute_dtype, args.bf16):
                terminal = flow_map(model, z0, sigma1, labels).float()
            results["paths"]["terminal"] = str(decode_save("terminal_X1", terminal))
            results["diagnostics"]["terminal"] = {
                "latent_norm_first": mean_norm(z0),
                "latent_norm_last": mean_norm(terminal),
            }

        for k in args.offdiag_steps:
            latents = offdiag_compose(k)
            key = f"offdiag_K{k}"
            results["paths"][key] = str(decode_save(key, latents))
            results["diagnostics"][key] = {
                "latent_norm_first": mean_norm(z0),
                "latent_norm_last": mean_norm(latents),
            }

    json_path = args.output_dir / "sample_summary.json"
    with json_path.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"[stage2-sample] wrote {json_path}", flush=True)


if __name__ == "__main__":
    main()
