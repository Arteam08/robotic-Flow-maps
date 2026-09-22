#!/usr/bin/env python
"""Compute and cache FID reference stats from ImageNet WebDataset shards."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from eqfm.data import DEFAULT_TRAIN_SHARDS, build_imagenet_loader  # noqa: E402
from eqfm.metrics import InceptionFeatureExtractor, stats_from_features  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True,
                   help="Destination .npz containing FID stats arrays mu/sigma.")
    p.add_argument("--num-images", type=int, default=50_000)
    p.add_argument("--shards", default=DEFAULT_TRAIN_SHARDS)
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--shuffle-buffer", type=int, default=1000)
    p.add_argument("--shardshuffle", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--overwrite", action="store_true", default=False)
    return p.parse_args()


def to_unit(images: torch.Tensor) -> torch.Tensor:
    """Convert ImageNet loader tensors from [-1, 1] to [0, 1]."""
    return ((images + 1.0) / 2.0).clamp(0.0, 1.0)


@torch.no_grad()
def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(
            f"{args.output} already exists. Pass --overwrite to recompute."
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    print(f"[fid-ref] output={args.output}", flush=True)
    print(f"[fid-ref] num_images={args.num_images} batch={args.batch_size} "
          f"workers={args.num_workers}", flush=True)
    print(f"[fid-ref] shards={args.shards}", flush=True)

    print("[fid-ref] loading Inception feature extractor", flush=True)
    incep = InceptionFeatureExtractor(device)

    loader = build_imagenet_loader(
        shards=args.shards,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle_buffer=args.shuffle_buffer,
        shardshuffle=args.shardshuffle,
        seed=args.seed,
        distributed=False,
    )

    feats = []
    seen = 0
    batch_idx = 0
    t0 = time.time()
    for images, _ in loader:
        batch_idx += 1
        images = images.to(device, non_blocking=True)
        feats.append(incep(to_unit(images)))
        seen += images.shape[0]
        if (
            batch_idx == 1
            or batch_idx % max(1, args.log_every) == 0
            or seen >= args.num_images
        ):
            print(f"[fid-ref]   ref {min(seen, args.num_images)}/{args.num_images} "
                  f"({seen / max(time.time() - t0, 1e-6):.1f} img/s)",
                  flush=True)
        if seen >= args.num_images:
            break

    if seen < args.num_images:
        raise RuntimeError(
            f"Only saw {seen} images before the loader ended; requested {args.num_images}."
        )

    feat_arr = np.concatenate(feats, axis=0)[:args.num_images]
    mu, sigma = stats_from_features(feat_arr)
    np.savez(
        args.output,
        mu=mu,
        sigma=sigma,
        num_images=np.array(args.num_images, dtype=np.int64),
        image_size=np.array(args.image_size, dtype=np.int64),
        seed=np.array(args.seed, dtype=np.int64),
        shards=np.array(str(args.shards)),
    )

    meta_path = args.output.with_suffix(".json")
    with meta_path.open("w") as f:
        json.dump(
            {
                "output": str(args.output),
                "num_images": args.num_images,
                "image_size": args.image_size,
                "batch_size": args.batch_size,
                "num_workers": args.num_workers,
                "shuffle_buffer": args.shuffle_buffer,
                "shardshuffle": args.shardshuffle,
                "seed": args.seed,
                "shards": str(args.shards),
            },
            f,
            indent=2,
        )
    print(f"[fid-ref] wrote {args.output}", flush=True)
    print(f"[fid-ref] wrote {meta_path}", flush=True)


if __name__ == "__main__":
    main()
