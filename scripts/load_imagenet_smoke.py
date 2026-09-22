"""Stream a few batches of ImageNet from the GCS WebDataset bucket.

End-to-end check that:
  * gcloud is authenticated on this host,
  * WebDataset can shell out to `gcloud storage cat` for each shard,
  * decoding + transform produce the expected tensor shape and value range.

Each phase prints its elapsed time so a hang is easy to localize:
  * `[t=...]` after the imports tells you whether torch + webdataset loaded
    quickly.
  * `[t=...] loader built` confirms WebDataset accepted the shard pattern.
  * `[t=...] first sample out` measures the cold-start cost: gcloud
    subprocess startup + TLS/OAuth + initial bytes + decode.
  * `batch N` lines are steady-state throughput.

Usage:
    python scripts/load_imagenet_smoke.py            # single shard, 1 batch
    python scripts/load_imagenet_smoke.py --batches 4 --batch-size 16
    python scripts/load_imagenet_smoke.py --full-shards --num-workers 4
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_t0 = time.time()
import torch  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from eqfm.data import build_imagenet_loader, DEFAULT_TRAIN_SHARDS  # noqa: E402
print(f"[t={time.time() - _t0:5.2f}s] imports complete")


SINGLE_SHARD = (
    "pipe:gcloud storage cat "
    "gs://cmu-gpucloud-jerryhua/imagenet/wds/train/train-000000.tar"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--shards", default=None,
                   help="Override shard pattern entirely. Useful for pointing "
                        "at a single shard for fast iteration.")
    p.add_argument("--full-shards", action="store_true",
                   help="Use the full 1281-shard training split. Default is a "
                        "single shard so the smoke test stays fast.")
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=0,
                   help="0 keeps the loader on the main process so gcloud auth "
                        "errors are visible. Bump to 4+ for actual training.")
    p.add_argument("--shuffle-buffer", type=int, default=1,
                   help="Default 1 == no shuffle, fastest cold start. Real "
                        "training should use ~5000+.")
    p.add_argument("--batches", type=int, default=1)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.shards is not None:
        shards = args.shards
    elif args.full_shards:
        shards = DEFAULT_TRAIN_SHARDS
    else:
        shards = SINGLE_SHARD

    print(f"[smoke] shards: {shards[:140]}{'...' if len(shards) > 140 else ''}")
    print(f"[smoke] image_size={args.image_size} batch_size={args.batch_size} "
          f"num_workers={args.num_workers} shuffle_buffer={args.shuffle_buffer}")

    t_build = time.time()
    loader = build_imagenet_loader(
        shards=shards,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle_buffer=args.shuffle_buffer,
        shardshuffle=False,                         # smoke = deterministic
        distributed=False,
    )
    print(f"[t={time.time() - t_build:5.2f}s] loader built")

    t_iter = time.time()
    it = iter(loader)
    print(f"[t={time.time() - t_iter:5.2f}s] iterator created (no network yet)")

    t_first = time.time()
    first_batch = next(it)
    print(f"[t={time.time() - t_first:5.2f}s] first batch received "
          f"(cold start: gcloud subprocess + TLS + initial decode)")

    images, labels = first_batch
    if not torch.is_tensor(labels):
        labels = torch.as_tensor(labels)
    print(f"[smoke] batch 1: images={tuple(images.shape)} dtype={images.dtype} "
          f"range=[{images.min().item():+.3f}, {images.max().item():+.3f}] "
          f"mean={images.float().mean().item():+.3f} | "
          f"labels={tuple(labels.shape)} dtype={labels.dtype} "
          f"range=[{int(labels.min())}, {int(labels.max())}]")

    for i in range(2, args.batches + 1):
        t_b = time.time()
        images, labels = next(it)
        if not torch.is_tensor(labels):
            labels = torch.as_tensor(labels)
        print(f"[t={time.time() - t_b:5.2f}s] batch {i}: "
              f"images={tuple(images.shape)} "
              f"labels={tuple(labels.shape)} "
              f"range=[{int(labels.min())}, {int(labels.max())}]")

    print("[smoke] ok")


if __name__ == "__main__":
    main()
