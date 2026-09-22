"""Stage-0 smoke test.

Downloads the pretrained SiT-XL/2 ImageNet-256 checkpoint, loads it into the
vendored SiT model, and runs a forward pass on a dummy latent batch. This is
the L40S debug entry point: if this script runs end-to-end on your dev box,
the checkpoint, the model definition, and the basic environment are all wired
up correctly.

Note on precision: SiT's vendored TimestepEmbedder hardcodes its sinusoidal
output to fp32, so casting the whole model to bf16/fp16 produces a dtype
mismatch on the first Linear of the timestep MLP. We instead keep weights in
fp32 (only ~2.7 GB for SiT-XL) and wrap the forward call in autocast for the
chosen compute precision -- the SiT upstream convention.

Usage:
    python scripts/load_sit_smoke.py
    python scripts/load_sit_smoke.py --device cuda --batch-size 4 --dtype bf16
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from eqfm.models.sit import SiT_XL_2, find_model  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="SiT-XL-2-256x256.pt",
                   help="Checkpoint name (auto-downloaded) or path to a local .pt file.")
    p.add_argument("--cache-dir", default=None,
                   help="Directory to download/cache the pretrained checkpoint. "
                        "Overrides $EQFM_CACHE_DIR. Defaults to ./pretrained_models. "
                        "Ignored when --ckpt is an explicit file path.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", default="bf16", choices=["fp32", "fp16", "bf16"],
                   help="Compute dtype (via autocast). Weights stay in fp32.")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--latent-size", type=int, default=32,
                   help="Latent spatial size. SD VAE on ImageNet-256 gives 32.")
    return p.parse_args()


DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


def main() -> None:
    args = parse_args()
    compute_dtype = DTYPES[args.dtype]
    device = torch.device(args.device)

    print(f"[smoke] device={device} compute_dtype={args.dtype} batch={args.batch_size}")
    print(f"[smoke] instantiating SiT-XL/2 at latent size {args.latent_size}")
    model = SiT_XL_2(input_size=args.latent_size).to(device=device)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[smoke] parameter count: {n_params/1e6:.1f}M")

    print(f"[smoke] loading checkpoint: {args.ckpt} (cache_dir={args.cache_dir})")
    state = find_model(args.ckpt, cache_dir=args.cache_dir)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[smoke] WARNING missing keys: {missing[:5]}{' ...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"[smoke] WARNING unexpected keys: {unexpected[:5]}{' ...' if len(unexpected) > 5 else ''}")

    B, C, H = args.batch_size, 4, args.latent_size
    x = torch.randn(B, C, H, H, device=device)
    t = torch.full((B,), 0.5, device=device)
    y = torch.randint(0, 1000, (B,), device=device)

    print(f"[smoke] forward pass: x={tuple(x.shape)} t={tuple(t.shape)} y={tuple(y.shape)}")
    autocast_enabled = compute_dtype != torch.float32 and device.type in ("cuda", "cpu")
    with torch.no_grad(), torch.amp.autocast(
        device_type=device.type, dtype=compute_dtype, enabled=autocast_enabled
    ):
        out = model(x, t, y)
    print(f"[smoke] output shape: {tuple(out.shape)} dtype={out.dtype}")
    print(f"[smoke] output stats: mean={out.float().mean().item():+.4f} "
          f"std={out.float().std().item():.4f}")
    print("[smoke] ok")


if __name__ == "__main__":
    main()
