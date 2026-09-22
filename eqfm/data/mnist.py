"""MNIST as a fully GPU-resident tensor dataset for the small-scale EqFM ablation.

Images are padded 28 -> 32 (so SiT patch sizes 2/4 tile evenly), scaled to
[-1, 1], and kept on-device as a single ``(N, 1, 32, 32)`` tensor. 60k images
are 245 MB in fp32, so there is no need for a DataLoader.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator, Tuple

import torch
import torch.nn.functional as F
from torchvision.datasets import MNIST


def load_mnist(root: str | Path, train: bool, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return ``(x, y)`` with ``x`` in [-1, 1], shape ``(N, 1, 32, 32)``, on ``device``."""
    ds = MNIST(str(root), train=train, download=False)
    x = ds.data.to(torch.float32).div_(255.0).mul_(2.0).sub_(1.0)  # (N, 28, 28)
    x = F.pad(x, (2, 2, 2, 2), value=-1.0).unsqueeze(1)             # (N, 1, 32, 32)
    y = ds.targets.to(torch.long)
    return x.to(device), y.to(device)


class InfiniteBatches:
    """Endless shuffled minibatches from GPU-resident tensors (reshuffles each epoch)."""

    def __init__(self, x: torch.Tensor, y: torch.Tensor, batch_size: int, generator: torch.Generator):
        self.x, self.y, self.bs, self.gen = x, y, batch_size, generator
        self.n = x.shape[0]
        self._perm = None
        self._pos = self.n  # force a shuffle on first call
        self.epoch = 0

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        return self

    def __next__(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._pos + self.bs > self.n:
            self._perm = torch.randperm(self.n, device=self.x.device, generator=self.gen)
            self._pos = 0
            self.epoch += 1
        idx = self._perm[self._pos:self._pos + self.bs]
        self._pos += self.bs
        return self.x[idx], self.y[idx]
