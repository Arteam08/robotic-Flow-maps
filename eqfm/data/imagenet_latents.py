"""ImageNet-1k as a cache of SD-VAE posterior statistics (``latents:`` prefix).

Each shard ``shard_XXXX.npz`` (written by ``scripts/precompute_latents.py``
from one parquet shard) holds ``mean`` and ``std`` of the SD-VAE encoder
posterior, unscaled, as float16 ``(N, 4, 32, 32)`` arrays, plus ``label``
``(N,)`` int16. Training draws ``(mean + std * eps) * 0.18215`` per step via
:meth:`eqfm.vae.LatentEncoder.sample_from_stats`, i.e. the same posterior
sample the on-the-fly path took, so the data distribution is unchanged and
the VAE encoder and JPEG decoding drop out of the training step.

Sharding follows :mod:`eqfm.data.imagenet_parquet`: the shard list is
shuffled with a seed shared by all ranks, then strided over
``world_size * num_workers``; rows are permuted inside each shard and mixed
through an in-stream shuffle buffer seeded per (epoch, rank, worker).

Yields ``(mean, std, label)`` batches: float16 ``(B, 4, 32, 32)`` x2 and
int64 ``(B,)``.
"""
from __future__ import annotations

import glob
import random
import time
from typing import Iterable, Iterator, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

LATENTS_PREFIX = "latents:"


def expand_latent_files(spec: str) -> List[str]:
    """``latents:/dir/shard_*.npz`` (or a directory) or comma-separated globs."""
    if spec.startswith(LATENTS_PREFIX):
        spec = spec[len(LATENTS_PREFIX):]
    files: List[str] = []
    for attempt in range(20):  # autofs on Babel: retry an empty first glob
        files = []
        for pattern in spec.split(","):
            pattern = pattern.strip()
            if not pattern:
                continue
            if not any(ch in pattern for ch in "*?[") and not pattern.endswith(".npz"):
                pattern = pattern.rstrip("/") + "/shard_*.npz"
            files.extend(sorted(glob.glob(pattern)))
        if files:
            return files
        time.sleep(3)
    raise FileNotFoundError(f"no latent shards match {spec!r}")


class LatentImageNet(IterableDataset):
    def __init__(
        self,
        files: Sequence[str],
        *,
        shuffle_buffer: int,
        seed: int,
        rank: int,
        world_size: int,
        batch_size: int = 32,
    ) -> None:
        super().__init__()
        self.files = list(files)
        self.shuffle_buffer = shuffle_buffer
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.batch_size = batch_size

    def _my_files(self, epoch: int) -> List[str]:
        rng = random.Random(self.seed + 1000 * epoch)  # same on every rank
        files = list(self.files)
        rng.shuffle(files)
        info = get_worker_info()
        n_workers = info.num_workers if info is not None else 1
        worker_id = info.id if info is not None else 0
        stride = self.world_size * n_workers
        offset = self.rank * n_workers + worker_id
        return files[offset::stride]

    @staticmethod
    def _load(path: str):
        for attempt in range(6):  # NFS ESTALE: retry
            try:
                with np.load(path) as z:
                    return z["mean"], z["std"], z["label"]
            except OSError as e:
                print(f"[latents] load failed on {path} ({e}); retry {attempt + 1}/6 in 30 s", flush=True)
                time.sleep(30)
        print(f"[latents] skipping unreadable shard {path}", flush=True)
        return None

    def _samples(self, epoch: int) -> Iterator:
        info = get_worker_info()
        worker_id = info.id if info is not None else 0
        rng = random.Random(self.seed + 7919 * epoch + 31 * self.rank + worker_id)
        buf: list = []
        for path in self._my_files(epoch):
            loaded = self._load(path)
            if loaded is None:
                continue
            mean, std, label = loaded
            order = np.random.default_rng(rng.getrandbits(64)).permutation(len(label))
            for i in order:
                sample = (mean[i], std[i], int(label[i]))
                if self.shuffle_buffer > 1:
                    buf.append(sample)
                    if len(buf) >= self.shuffle_buffer:
                        j = rng.randrange(len(buf))
                        buf[j], buf[-1] = buf[-1], buf[j]
                        yield buf.pop()
                else:
                    yield sample
        rng.shuffle(buf)
        yield from buf

    def __iter__(self) -> Iterator:
        epoch = 0
        while True:
            ms: list = []
            ss: list = []
            ys: list = []
            for m, s, y in self._samples(epoch):
                ms.append(m)
                ss.append(s)
                ys.append(y)
                if len(ms) == self.batch_size:
                    yield (
                        torch.from_numpy(np.stack(ms)),
                        torch.from_numpy(np.stack(ss)),
                        torch.tensor(ys, dtype=torch.long),
                    )
                    ms, ss, ys = [], [], []
            epoch += 1


def build_imagenet_latent_loader(
    spec: str,
    *,
    batch_size: int = 32,
    num_workers: int = 1,
    shuffle_buffer: int = 1000,
    seed: int = 0,
    distributed: bool = False,
    persistent_workers: Optional[bool] = None,
    **_ignored,
) -> Iterable:
    files = expand_latent_files(spec)
    rank, world = 0, 1
    if distributed:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            rank, world = dist.get_rank(), dist.get_world_size()
    ds = LatentImageNet(
        files, shuffle_buffer=shuffle_buffer, seed=seed, rank=rank, world_size=world,
        batch_size=batch_size,
    )
    if persistent_workers is None:
        persistent_workers = num_workers > 0
    return DataLoader(
        ds, batch_size=None, num_workers=num_workers,
        persistent_workers=persistent_workers, pin_memory=True,
        prefetch_factor=4 if num_workers > 0 else None,
    )
