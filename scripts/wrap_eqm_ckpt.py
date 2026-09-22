#!/usr/bin/env python
"""Wrap the released EqM-XL/2 state_dict into our Stage-1 checkpoint format.

The raw EqM file (raywang4/EqM Google-Drive checkpoint, EMA weights only) is a
flat state_dict whose keys/shapes match ``SiT_XL_2`` exactly. Our trainer
(``--ckpt``) and ``scripts/eval_fid.py`` expect ``{"model": sd, "step", "args"}``.

EqM regresses ``4*min(1, 5(1-t)) * (x1 - x0)``; our Stage-1 loss regresses
``(1-t) * (x1 - x0)`` with a ``1/(1-t)`` weight. The fields differ mostly by a
gain in [4, 20], so ``--scale`` multiplies ``final_layer.linear`` once;
``--scale auto`` picks the candidate minimising our loss on a few fixed
ImageNet batches (needs a GPU, the VAE, and ``--shards``).

``--eval-only`` skips writing and just reports the fixed-batch loss and
``||b(x_data)||`` for any checkpoint (flat or wrapped), e.g. to get the same
numbers for the uw450k teacher.
"""
from __future__ import annotations

import argparse
import json
import sys
from itertools import cycle
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from eqfm.data import build_imagenet_loader  # noqa: E402
from eqfm.eval_utils import (  # noqa: E402
    autonomous_forward, collect_fixed_batches, fixed_batch_loss, scale_output_head,
    search_output_scale,
)
from eqfm.losses import EqMLoss, EqMLossConfig  # noqa: E402
from eqfm.models.sit import SiT_XL_2, select_weights  # noqa: E402
from eqfm.vae import LatentEncoder  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", type=Path, required=True)
    p.add_argument("--dst", type=Path, default=None, help="Output .pt (required unless --eval-only).")
    p.add_argument("--scale", default="1", help="float or 'auto'")
    p.add_argument("--candidates",
                   default="1,0.5,0.25,0.2,0.167,0.143,0.125,0.111,0.1,0.0909,0.0833,0.0714,0.0625")
    p.add_argument("--ckpt-key", choices=["auto", "model", "ema"], default="auto")
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--shards", default=None, help="parquet:... spec; required for auto / eval-only")
    p.add_argument("--batches", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=3)
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--latent-size", type=int, default=32)
    p.add_argument("--vae-id", default="stabilityai/sd-vae-ft-ema")
    p.add_argument("--eps-train", type=float, default=1e-3)
    p.add_argument("--time-sampler", default="uniform_weighted")
    p.add_argument("--data-anchor", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.eval_only and args.dst is None:
        raise SystemExit("--dst is required unless --eval-only")
    need_data = args.eval_only or args.scale == "auto"
    if need_data and not args.shards:
        raise SystemExit("--shards is required for --scale auto / --eval-only")

    raw = torch.load(args.src, map_location="cpu", weights_only=False)
    sd = select_weights(raw, key=args.ckpt_key)
    wrapped = isinstance(raw, dict) and ({"model", "ema"} & set(raw))
    print(f"[wrap] src={args.src} ({'wrapped' if wrapped else 'flat'}, {len(sd)} tensors, "
          f"step={raw.get('step', 'n/a') if wrapped else 'n/a'})")

    device = torch.device(args.device)
    model = SiT_XL_2(input_size=args.latent_size)
    model.load_state_dict(sd, strict=True)  # verifies keys + shapes
    model.to(device).eval()
    model_fn = autonomous_forward(model)

    report: dict = {"source": str(args.src), "ckpt_key": args.ckpt_key}
    scale = 1.0
    if need_data:
        vae = LatentEncoder(vae_id=args.vae_id).to(device).eval()
        loader = build_imagenet_loader(
            shards=args.shards, image_size=args.image_size, batch_size=args.batch_size,
            num_workers=args.num_workers, shuffle_buffer=2000, shardshuffle=100,
            distributed=False, hflip=False)
        batches = collect_fixed_batches(cycle(loader), vae, args.batches, device=device, bf16=args.bf16)
        loss_fn = EqMLoss(EqMLossConfig(eps_train=args.eps_train, time_sampler=args.time_sampler,
                                        data_anchor_weight=args.data_anchor))
        before = fixed_batch_loss(model_fn, loss_fn, batches, seed=args.seed, device=device, bf16=args.bf16)
        print(f"[wrap] unscaled: loss {before['loss']:.4f} | b_data_rms {before['b_data_rms']:.4f}")
        report["unscaled"] = before
        if args.scale == "auto":
            cands = [float(c) for c in args.candidates.split(",") if c.strip()]
            scale, table = search_output_scale(model, model_fn, loss_fn, batches, cands,
                                               seed=args.seed, device=device, bf16=args.bf16)
            print("[wrap] scale search (scale, loss): " + ", ".join(f"({s:.4g}, {l:.4f})" for s, l in table))
            report["scale_table"] = table
        elif not args.eval_only:
            scale = float(args.scale)
        if scale != 1.0:
            scale_output_head(model, scale)
            after = fixed_batch_loss(model_fn, loss_fn, batches, seed=args.seed, device=device, bf16=args.bf16)
            print(f"[wrap] scaled x{scale:.5g}: loss {after['loss']:.4f} | b_data_rms {after['b_data_rms']:.4f}")
            report["scaled"] = after
    elif not args.eval_only:
        scale = float(args.scale)
        if scale != 1.0:
            scale_output_head(model, scale)
    report["scale"] = scale

    if args.eval_only:
        print(json.dumps(report, indent=2))
        return

    out = {
        "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "step": 0,
        "args": {
            "parameterization": "velocity",
            "time_conditioning": "zero",
            "time_sampler": args.time_sampler,
            "eps_train": args.eps_train,
            "init_output_scale": scale,
            "source": str(args.src),
        },
        "note": (f"Released EqM-XL/2 EMA weights (raywang4/EqM) with final_layer.linear "
                 f"scaled by {scale:.6g} for the EqFM (1-t) clock. No 'ema' key: "
                 f"eval_fid.py and find_model(key=auto) use 'model'."),
    }
    args.dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.dst.with_suffix(".pt.tmp")
    torch.save(out, tmp)
    tmp.replace(args.dst)
    with open(str(args.dst) + ".scale.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"[wrap] wrote {args.dst} (scale {scale:.6g})")


if __name__ == "__main__":
    main()
