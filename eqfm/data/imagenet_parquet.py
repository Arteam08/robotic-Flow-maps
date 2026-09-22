"""ImageNet-1k from local parquet shards (HF ``imagenet-1k`` layout).

Each parquet file holds rows with an ``image`` struct column (``bytes``,
``path``) and an integer ``label`` column in the canonical sorted-WNID order
used by torchvision and the SiT-XL/2 checkpoint. Shards are split across DDP
ranks and DataLoader workers; row groups are read one at a time so memory
stays bounded, and an in-stream shuffle buffer mixes samples across shards.

Yields ``(image, label)`` batches in the same format as
:mod:`eqfm.data.imagenet_wds`: float32 images in ``[-1, 1]`` of shape
``(B, 3, H, H)`` and int64 labels of shape ``(B,)``.
"""
from __future__ import annotations

import glob
import io
import random
import time
from typing import Iterable, Iterator, List, Optional, Sequence

import torch
from PIL import Image
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from .imagenet_wds import _build_train_transform, _parse_label

PARQUET_PREFIX = "parquet:"


def expand_parquet_files(spec: str) -> List[str]:
    """``parquet:/path/*.parquet`` or a comma-separated list of globs."""
    if spec.startswith(PARQUET_PREFIX):
        spec = spec[len(PARQUET_PREFIX):]
    files: List[str] = []
    # /data on Babel is autofs-mounted on first access; several DDP ranks
    # globbing at once can see an empty directory while the mount is in
    # progress, so retry before giving up.
    for attempt in range(20):
        files = []
        for pattern in spec.split(","):
            pattern = pattern.strip()
            if not pattern:
                continue
            files.extend(sorted(glob.glob(pattern)))
        if files:
            return files
        time.sleep(3)
    raise FileNotFoundError(f"no parquet files match {spec!r}")


class ParquetImageNet(IterableDataset):
    def __init__(
        self,
        files: Sequence[str],
        *,
        image_size: int,
        shuffle_buffer: int,
        seed: int,
        rank: int,
        world_size: int,
        image_col: str = "image",
        label_col: str = "label",
        batch_size: int = 32,
        hflip: bool = False,
    ) -> None:
        super().__init__()
        self.files = list(files)
        self.transform = _build_train_transform(image_size, hflip=hflip)
        self.shuffle_buffer = shuffle_buffer
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.image_col = image_col
        self.label_col = label_col
        self.batch_size = batch_size

    def _my_files(self, epoch: int) -> List[str]:
        rng = random.Random(self.seed + 1000 * epoch)
        files = list(self.files)
        rng.shuffle(files)
        info = get_worker_info()
        n_workers = info.num_workers if info is not None else 1
        worker_id = info.id if info is not None else 0
        stride = self.world_size * n_workers
        offset = self.rank * n_workers + worker_id
        return files[offset::stride]

    def _decode(self, img_field, label_field):
        raw = img_field["bytes"] if isinstance(img_field, dict) else img_field
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        return self.transform(img), int(_parse_label(label_field))

    def _samples(self, epoch: int) -> Iterator:
        import pyarrow.parquet as pq

        info = get_worker_info()
        worker_id = info.id if info is not None else 0
        rng = random.Random(self.seed + 7919 * epoch + 31 * self.rank + worker_id)
        buf: list = []
        for path in self._my_files(epoch):
            pf = None
            for attempt in range(6):  # NFS on Babel occasionally returns ESTALE; reopen and retry
                try:
                    pf = pq.ParquetFile(path)
                    break
                except OSError as e:
                    print(f"[parquet] open failed ({e}); retry {attempt + 1}/6 in 30 s", flush=True)
                    time.sleep(30)
            if pf is None:
                print(f"[parquet] skipping unreadable shard {path}", flush=True)
                continue
            for rg in range(pf.num_row_groups):
                table = None
                for attempt in range(6):
                    try:
                        table = pf.read_row_group(rg, columns=[self.image_col, self.label_col])
                        break
                    except OSError as e:
                        print(f"[parquet] read failed on {path} rg {rg} ({e}); retry {attempt + 1}/6 in 30 s", flush=True)
                        time.sleep(30)
                        try:
                            pf = pq.ParquetFile(path)
                        except OSError:
                            pass
                if table is None:
                    print(f"[parquet] skipping row group {rg} of {path}", flush=True)
                    continue
                imgs = table.column(self.image_col).to_pylist()
                labels = table.column(self.label_col).to_pylist()
                for im, lb in zip(imgs, labels):
                    try:
                        sample = self._decode(im, lb)
                    except Exception:  # corrupt image: skip, like warn_and_continue
                        continue
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
        while True:  # infinite stream; the trainer cycles anyway
            xs: list = []
            ys: list = []
            for x, y in self._samples(epoch):
                xs.append(x)
                ys.append(y)
                if len(xs) == self.batch_size:
                    yield torch.stack(xs), torch.tensor(ys, dtype=torch.long)
                    xs, ys = [], []
            epoch += 1


def build_imagenet_parquet_loader(
    spec: str,
    *,
    image_size: int = 256,
    batch_size: int = 32,
    num_workers: int = 4,
    shuffle_buffer: int = 1000,
    seed: int = 0,
    distributed: bool = False,
    image_col: str = "image",
    label_col: str = "label",
    persistent_workers: Optional[bool] = None,
    hflip: bool = False,
) -> Iterable:
    files = expand_parquet_files(spec)
    rank, world = 0, 1
    if distributed:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            rank, world = dist.get_rank(), dist.get_world_size()
    ds = ParquetImageNet(
        files, image_size=image_size, shuffle_buffer=shuffle_buffer, seed=seed,
        rank=rank, world_size=world, image_col=image_col, label_col=label_col,
        batch_size=batch_size, hflip=hflip,
    )
    if persistent_workers is None:
        persistent_workers = num_workers > 0
    return DataLoader(
        ds, batch_size=None, num_workers=num_workers,
        persistent_workers=persistent_workers, pin_memory=True,
        prefetch_factor=4 if num_workers > 0 else None,
    )
