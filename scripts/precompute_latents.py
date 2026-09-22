"""Encode ImageNet parquet shards to SD-VAE posterior statistics once.

For every parquet shard ``train-XXXXX-of-NNNNN.parquet`` this writes
``<out>/shard_XXXXX.npz`` with float16 ``mean``/``std`` ``(N, 4, 32, 32)``
(unscaled encoder posterior) and int16 ``label`` ``(N,)``, rows in parquet
order. Preprocessing is exactly the training transform
(``eqfm.data.imagenet_wds._build_train_transform``: resize 256, center crop,
[-1, 1]) and the encoder runs under bf16 autocast, as in the trainers.

Idempotent: shards whose output exists are skipped, so an array job can be
requeued. ``--verify N`` re-encodes the first N rows of each processed shard
in a fresh pass and checks the stored statistics and labels against them.

Usage (Slurm array): ``--task $SLURM_ARRAY_TASK_ID --ntasks $SLURM_ARRAY_TASK_COUNT``
handles shards ``task::ntasks``.
"""
from __future__ import annotations

import argparse
import io
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, IterableDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))



def expand_parquet_files(spec: str):
    """``parquet:/path/*.parquet`` or comma-separated globs -> sorted file list."""
    import glob
    if spec.startswith("parquet:"):
        spec = spec[len("parquet:"):]
    files = []
    for pattern in spec.split(","):
        if pattern.strip():
            files.extend(sorted(glob.glob(pattern.strip())))
    if not files:
        raise FileNotFoundError(f"no parquet files match {spec!r}")
    return files

from eqfm.data.imagenet_wds import _build_train_transform, _parse_label  # noqa: E402
from eqfm.vae import LatentEncoder  # noqa: E402


class ShardRows(IterableDataset):
    """All rows of one parquet shard, in order, decoded + transformed."""

    def __init__(self, path: str, image_size: int, limit: int | None = None):
        self.path, self.transform, self.limit = path, _build_train_transform(image_size), limit

    def __iter__(self):
        import pyarrow.parquet as pq
        from torch.utils.data import get_worker_info
        info = get_worker_info()
        nw, wid = (info.num_workers, info.id) if info else (1, 0)
        pf = pq.ParquetFile(self.path)
        idx = 0
        for rg in range(pf.num_row_groups):
            table = pf.read_row_group(rg, columns=["image", "label"])
            imgs = table.column("image").to_pylist()
            labels = table.column("label").to_pylist()
            for im, lb in zip(imgs, labels):
                if self.limit is not None and idx >= self.limit:
                    return
                if idx % nw == wid:  # round-robin rows over workers; index restores order
                    raw = im["bytes"] if isinstance(im, dict) else im
                    try:
                        x = self.transform(Image.open(io.BytesIO(raw)).convert("RGB"))
                        ok = True
                    except Exception:
                        x, ok = torch.zeros(3, 256, 256), False  # keep row alignment; flagged
                    yield idx, x, int(_parse_label(lb)), ok
                idx += 1


def encode_shard(vae, path, out_path, *, batch_size, num_workers, device, limit=None):
    ds = ShardRows(path, 256, limit)
    dl = DataLoader(ds, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                    prefetch_factor=4 if num_workers > 0 else None)
    idxs, means, stds, labels, bad = [], [], [], [], []
    for idx, x, y, ok in dl:
        # Full fp32 (no autocast, TF32 off): the cache is a persistent artifact,
        # so it stores the exact encoder posterior; bf16 encoding (what the
        # on-the-fly path does) deviates from it by ~1e-2 relative per element.
        m, s = vae.encode_stats(x.to(device, non_blocking=True))
        idxs.append(idx); labels.append(y)
        means.append(m.to(torch.float16).cpu()); stds.append(s.to(torch.float16).cpu())
        bad.extend(int(i) for i, o in zip(idx, ok) if not bool(o))
    idx = torch.cat(idxs); order = torch.argsort(idx)
    assert torch.equal(idx[order], torch.arange(len(idx))), "row index gap"
    mean = torch.cat(means)[order].numpy(); std = torch.cat(stds)[order].numpy()
    label = torch.cat(labels)[order].numpy().astype(np.int16)
    if bad:
        keep = np.ones(len(label), dtype=bool); keep[bad] = False
        print(f"  dropping {len(bad)} undecodable rows", flush=True)
        mean, std, label = mean[keep], std[keep], label[keep]
    if out_path is not None:
        tmp = out_path.with_suffix(".npz.tmp.npz")
        np.savez(tmp, mean=mean, std=std, label=label, meta=np.array([CACHE_FORMAT]))
        os.replace(tmp, out_path)
    return mean, std, label


CACHE_FORMAT = "sdvae-ft-ema-posterior-fp32-v1"


def is_complete(out_path: Path) -> bool:
    """True if ``out_path`` exists and was written with the current CACHE_FORMAT."""
    if not out_path.exists():
        return False
    try:
        with np.load(out_path) as z:
            return "meta" in z and str(z["meta"][0]) == CACHE_FORMAT
    except Exception:
        return False


def verify_shard(mean, std, label, m2, s2, l2):
    """Per-row agreement between the stored stats and a fresh fp32 re-encode
    (different batch composition, so this also bounds fp32 kernel nondeterminism).
    Row misalignment would show as O(1) relative error even inside a
    class-sorted shard where labels alone would not catch it."""
    n = len(l2)
    a, b = mean[:n].astype(np.float32), m2.astype(np.float32)
    per_row = np.linalg.norm((a - b).reshape(n, -1), axis=1) / np.linalg.norm(b.reshape(n, -1), axis=1)
    rel_rms = float(np.linalg.norm(a - b) / np.linalg.norm(b))
    std_rel = float(np.linalg.norm(std[:n].astype(np.float32) - s2) / np.linalg.norm(s2))
    labels_equal = bool((label[:n] == l2).all())
    p99 = float(np.percentile(per_row, 99)); worst = float(per_row.max())
    print(f"  verify first {n}: mean rel-RMS {rel_rms:.2e}  per-row rel err p99 {p99:.2e} max {worst:.2e}  "
          f"std rel-RMS {std_rel:.2e}  labels_equal={labels_equal}", flush=True)
    return labels_equal and rel_rms < 5e-3 and p99 < 1e-2 and std_rel < 5e-3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", required=True, help="parquet:/path/train-*.parquet")
    ap.add_argument("--out", required=True)
    ap.add_argument("--task", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    ap.add_argument("--ntasks", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1)))
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=6)
    ap.add_argument("--vae-id", default="stabilityai/sd-vae-ft-ema")
    ap.add_argument("--vae-cache-dir", default=None)
    ap.add_argument("--verify", type=int, default=0, help="re-encode the first N rows and compare")
    ap.add_argument("--max-shards", type=int, default=None)
    args = ap.parse_args()

    files = expand_parquet_files(args.shards)[args.task::args.ntasks]
    if args.max_shards:
        files = files[: args.max_shards]
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = True
    vae = LatentEncoder(args.vae_id, cache_dir=args.vae_cache_dir).to(device).eval()
    print(f"task {args.task}/{args.ntasks}: {len(files)} shards -> {out}", flush=True)
    for path in files:
        stem = Path(path).stem  # train-00012-of-00294
        num = stem.split("-")[1] if "-" in stem else stem
        out_path = out / f"shard_{num}.npz"
        if is_complete(out_path):
            print(f"skip {out_path.name} (complete, {CACHE_FORMAT})", flush=True); continue
        t0 = time.time()
        mean, std, label = encode_shard(vae, path, out_path, batch_size=args.batch_size,
                                        num_workers=args.num_workers, device=device)
        dt = time.time() - t0
        print(f"{out_path.name}: {len(label)} rows in {dt:.0f}s ({len(label)/dt:.0f} img/s) "
              f"mean|{np.abs(mean).mean():.3f}| std {std.mean():.3f} labels [{label.min()},{label.max()}]", flush=True)
        if args.verify > 0:
            # fresh pass with a different batch composition (batch 48, 3 workers)
            m2, s2, l2 = encode_shard(vae, path, None, batch_size=48, num_workers=3,
                                      device=device, limit=args.verify)
            if not verify_shard(mean, std, label, m2, s2, l2):
                out_path.unlink(missing_ok=True)
                raise AssertionError(f"cache verification failed for {out_path.name}; file removed")


if __name__ == "__main__":
    main()
