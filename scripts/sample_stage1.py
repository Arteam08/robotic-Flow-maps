"""Sample images from a Stage-1 EqM checkpoint.

Loads SiT-XL/2 weights from a checkpoint produced by
``scripts/train_stage1_eqm.py``, optionally swaps in the EMA shadow,
forward-Euler integrates the autonomous ODE on Gaussian noise, decodes
through the SD VAE, and saves a grid PNG plus per-sample PNGs.

Usage:
    python scripts/sample_stage1.py \
        --ckpt /scratch/$USER/eqfm/runs/<name>/checkpoints/latest.pt \
        --output-dir /tmp/eqfm-samples \
        --n-samples 16 --num-steps 250 --eps-stop 1e-3
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from eqfm.models.sit import SiT_XL_2  # noqa: E402
from eqfm.sampling import (  # noqa: E402
    SamplerDiagnostics, as_field_fn, sample_euler, save_grid, stop_time_from_eps,
    summarize_diagnostics,
)
from eqfm.training import EMA  # noqa: E402
from eqfm.vae import LatentEncoder  # noqa: E402


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    p.add_argument("--ckpt", type=Path, required=True,
                   help="Path to a Stage-1 checkpoint .pt file.")
    p.add_argument("--output-dir", type=Path, required=True,
                   help="Directory to write grid.png, sample_*.png, "
                        "labels.json, and diagnostics.json into.")

    p.add_argument("--n-samples", type=int, default=16,
                   help="Number of images to generate.")
    p.add_argument("--classes", type=int, nargs="+", default=None,
                   help="Optional explicit class IDs (0--999). If omitted, "
                        "draws random labels.")
    p.add_argument("--num-steps", type=int, default=250)
    p.add_argument("--eps-stop", type=float, default=1e-3,
                   help="Manifold-residual at the integration terminus; "
                        "S_stop = log(1/eps_stop) under the exponential "
                        "clock. Recommended >= eps-train used at training.")
    p.add_argument("--stop-time", type=float, default=None,
                   help="Explicit autonomous-time horizon S_stop. Overrides "
                        "--eps-stop if given.")
    p.add_argument("--cfg-scale", type=float, default=1.0,
                   help="Classifier-free guidance scale. 1.0 = off. >1 "
                        "doubles per-step NFE.")
    p.add_argument("--cfg-mode", choices=["all", "first3"], default="all",
                   help="Which latent channels to guide. first3 matches "
                        "the released SiT/DiT forward_with_cfg convention.")
    p.add_argument("--cfg-schedule", choices=["constant", "anneal_to_one"],
                   default="constant",
                   help="constant = fixed CFG. anneal_to_one decays CFG "
                        "toward 1 near the terminal fixed point.")
    p.add_argument("--cfg-schedule-power", type=float, default=1.0,
                   help="Power for anneal_to_one: cfg(s)=1+(cfg-1)*(1-t)^p.")
    p.add_argument("--cfg-match-uncond-norm", action="store_true", default=False,
                   help="Before CFG, rescale the unconditional field per "
                        "sample so its L2 norm matches the conditional "
                        "field norm.")
    p.add_argument("--null-class", type=int, default=1000,
                   help="SiT unconditional embedding index (1000 for "
                        "ImageNet-1k; embedding table is size 1001).")
    p.add_argument("--momentum", type=float, default=0.0,
                   help="Heavy-ball / Nesterov coefficient on the Euler "
                        "step. 0.0 = plain Euler. Accelerates approach to "
                        "the fixed point; can overshoot near it. Try "
                        "0.5-0.9.")
    p.add_argument("--nesterov", action="store_true", default=False,
                   help="Evaluate the field at the momentum look-ahead "
                        "point (Nesterov) instead of at x (heavy-ball). "
                        "Same NFE.")

    p.add_argument("--parameterization", choices=["auto", "velocity", "denoiser"],
                   default="auto",
                   help="How to interpret the network output. 'auto' reads it "
                        "from the checkpoint's saved training args (falling "
                        "back to 'velocity'). 'denoiser' reconstructs the field "
                        "as D(x)-x; 'velocity' uses the output as b directly.")
    p.add_argument("--latent-size", type=int, default=32,
                   help="Spatial size of the VAE latent (32 for ImageNet "
                        "256 / SD VAE).")
    p.add_argument("--vae-id", default="stabilityai/sd-vae-ft-ema")

    p.add_argument("--use-ema", action="store_true", default=True,
                   help="Sample with EMA weights (the standard convention).")
    p.add_argument("--no-ema", dest="use_ema", action="store_false",
                   help="Sample with live (non-EMA) weights.")

    p.add_argument("--bf16", action="store_true", default=True,
                   help="bf16 autocast on the model forward.")
    p.add_argument("--no-bf16", dest="bf16", action="store_false")

    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def autonomous_forward(model: nn.Module):
    """Same closure as in training: pin t=0 every forward."""
    def forward(z, y):
        t = torch.zeros(z.shape[0], device=z.device, dtype=z.dtype)
        return model(z, t, y)
    return forward


def save_individuals(images: torch.Tensor, out_dir: Path) -> None:
    from torchvision.utils import save_image
    for i in range(images.shape[0]):
        save_image(images[i].clamp(-1, 1), out_dir / f"sample_{i:03d}.png",
                   normalize=True, value_range=(-1, 1))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[sample] ckpt = {args.ckpt}")
    print(f"[sample] output_dir = {output_dir}")

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    compute_dtype = torch.bfloat16 if args.bf16 else torch.float32

    # ----- load checkpoint -----
    print(f"[sample] loading checkpoint")
    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    train_step = state.get("step", -1)
    print(f"[sample] checkpoint step = {train_step}")

    parameterization = args.parameterization
    if parameterization == "auto":
        ckpt_args = state.get("args", {}) or {}
        parameterization = ckpt_args.get("parameterization", "velocity")
        print(f"[sample] parameterization = {parameterization} (auto-detected "
              f"from checkpoint args)")
    else:
        print(f"[sample] parameterization = {parameterization} (CLI override)")

    # ----- model -----
    print(f"[sample] instantiating SiT-XL/2 (latent_size={args.latent_size})")
    model = SiT_XL_2(input_size=args.latent_size).to(device)
    model.load_state_dict(state["model"])

    # ----- optional EMA swap -----
    if args.use_ema and "ema" in state:
        print(f"[sample] applying EMA shadow")
        # Rebuild EMA from the current (trainable) params, then load the
        # checkpoint's shadow into it. This works regardless of whether
        # training used --time-conditioning strict (frozen t_embedder is
        # simply not in the shadow).
        ema = EMA(model, decay=0.9999)
        ema.load_state_dict(state["ema"])
        ema_ctx = ema.apply_to(model)
    else:
        if args.use_ema:
            print(f"[sample] WARN no 'ema' key in checkpoint; using live weights")
        import contextlib
        ema_ctx = contextlib.nullcontext()

    model.eval()

    # ----- VAE -----
    print(f"[sample] loading SD VAE: {args.vae_id}")
    vae = LatentEncoder(vae_id=args.vae_id).to(device)
    vae.eval()

    # ----- class labels -----
    if args.classes is not None:
        if len(args.classes) != args.n_samples:
            print(f"[sample] note: --classes had {len(args.classes)} entries, "
                  f"overriding --n-samples to match")
            args.n_samples = len(args.classes)
        labels = torch.tensor(args.classes, device=device, dtype=torch.long)
    else:
        labels = torch.randint(0, 1000, (args.n_samples,), device=device,
                               dtype=torch.long)

    # Persist for later reference.
    with (output_dir / "labels.json").open("w") as f:
        json.dump({"classes": labels.tolist(),
                   "ckpt_step": train_step,
                   "args": {k: (str(v) if isinstance(v, Path) else v)
                            for k, v in vars(args).items()}},
                  f, indent=2)

    # ----- sample -----
    if args.stop_time is None:
        s_stop = stop_time_from_eps(args.eps_stop)
    else:
        s_stop = args.stop_time
    print(f"[sample] integrating: K={args.num_steps} S_stop={s_stop:.3f} "
          f"(step={s_stop/args.num_steps:.4f}) "
          f"cfg={args.cfg_scale} mode={args.cfg_mode} schedule={args.cfg_schedule} "
          f"match_uncond={args.cfg_match_uncond_norm} "
          f"momentum={args.momentum}{' nesterov' if args.nesterov else ''}")
    if args.cfg_scale != 1.0:
        print("[sample] cfg-scale != 1: this assumes the checkpoint trained "
              "the null label via --cfg-dropout-prob. Older checkpoints may "
              "only have SiT's implicit/default label dropout; verify with "
              "sampling/FID sweeps.")

    diag = SamplerDiagnostics()
    with ema_ctx:
        model_fn = as_field_fn(autonomous_forward(model), parameterization)
        latents = sample_euler(
            model_fn,
            shape=(args.n_samples, 4, args.latent_size, args.latent_size),
            class_labels=labels,
            num_steps=args.num_steps,
            stop_time=s_stop,
            cfg_scale=args.cfg_scale,
            cfg_mode=args.cfg_mode,
            cfg_schedule=args.cfg_schedule,
            cfg_schedule_power=args.cfg_schedule_power,
            cfg_match_uncond_norm=args.cfg_match_uncond_norm,
            null_class=args.null_class,
            momentum=args.momentum,
            nesterov=args.nesterov,
            device=device,
            dtype=torch.float32,           # fp32 z, bf16 model under autocast
            autocast_dtype=compute_dtype if args.bf16 else None,
            diagnostics=diag,
        )

    diag_summary = summarize_diagnostics(diag)
    print(f"[sample] diagnostics: "
          f"||b||_first={diag_summary['velocity_norm_first']:.3f}, "
          f"||b||_last={diag_summary['velocity_norm_last']:.3f}, "
          f"||z||_last={diag_summary['latent_norm_last']:.3f}")
    with (output_dir / "diagnostics.json").open("w") as f:
        json.dump(
            {"summary": diag_summary,
             "per_step_velocity_norm": diag.velocity_norms,
             "per_step_latent_norm": diag.latent_norms},
            f, indent=2,
        )

    # ----- decode + save -----
    print(f"[sample] decoding latents through VAE")
    with torch.amp.autocast(device.type, dtype=compute_dtype, enabled=args.bf16):
        images = vae.decode(latents)
    images = images.float().cpu()

    grid_path = output_dir / "grid.png"
    save_grid(images, grid_path)
    save_individuals(images, output_dir)
    print(f"[sample] wrote {grid_path} (+ {args.n_samples} sample_*.png)")
    print("[sample] ok")


if __name__ == "__main__":
    main()
