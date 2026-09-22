"""Exact pooled training statistics over a logging window.

Logging the metrics of the last micro-batch is noisy, and per-t-bucket means of a
small micro-batch read 0 whenever no sample falls in a bucket. :class:`WindowStats`
instead keeps GPU-side running sums and counts over every micro-batch since the
last log point and resolves them with a single device sync at log time:

- scalar metrics: mean over micro-batches (keys ending in ``_max``/``_min``: max/min);
- per-sample quantities: pooled mean over all samples, overall and per t bucket;
  buckets with no sample in the whole window are omitted instead of reported as 0;
- shares: fraction of a per-sample total (e.g. w·mse) contributed by each bucket.
"""
from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence, Tuple

import torch

T_BUCKETS: Tuple[Tuple[float, float], ...] = ((0.0, 0.1), (0.1, 0.5), (0.5, 0.9), (0.9, 1.0))


def _bucket_masks(t: torch.Tensor, buckets) -> Iterable[Tuple[str, torch.Tensor]]:
    last = buckets[-1][1]
    for lo, hi in buckets:
        m = (t >= lo) & ((t <= hi) if hi == last else (t < hi))
        yield f"t{lo:g}_{hi:g}", m


class WindowStats:
    def __init__(self, buckets: Sequence[Tuple[float, float]] = T_BUCKETS):
        self.buckets = tuple(buckets)
        self.reset()

    def reset(self) -> None:
        self._sum: Dict[str, torch.Tensor] = {}
        self._cnt: Dict[str, torch.Tensor] = {}
        self._ext: Dict[str, Tuple[str, torch.Tensor]] = {}   # key -> ("max"|"min", value)
        self._share: Dict[str, Dict[str, torch.Tensor]] = {}  # name -> {bucket: sum, "": total}

    def _acc(self, key: str, s: torch.Tensor, n) -> None:
        s = s.detach().float()
        n = n.detach().float() if torch.is_tensor(n) else torch.tensor(float(n), device=s.device)
        if key in self._sum:
            self._sum[key] = self._sum[key] + s
            self._cnt[key] = self._cnt[key] + n
        else:
            self._sum[key], self._cnt[key] = s, n

    def add_scalar(self, key: str, value) -> None:
        v = value.detach().float() if torch.is_tensor(value) else torch.tensor(float(value))
        if key.endswith("_max") or key.endswith("_min"):
            how = "max" if key.endswith("_max") else "min"
            if key in self._ext:
                prev = self._ext[key][1]
                v = torch.maximum(prev, v.to(prev.device)) if how == "max" else torch.minimum(prev, v.to(prev.device))
            self._ext[key] = (how, v)
        else:
            self._acc(key, v, 1.0)

    def add_scalars(self, metrics: Dict[str, object], skip: Iterable[str] = ()) -> None:
        skip = set(skip)
        for k, v in metrics.items():
            if k.startswith("_") or k in skip:
                continue
            if torch.is_tensor(v) and v.numel() != 1:
                continue
            if not (torch.is_tensor(v) or isinstance(v, (int, float))):
                continue
            self.add_scalar(k, v)

    def add_samples(self, t: torch.Tensor, values: Dict[str, torch.Tensor],
                    mask: Optional[torch.Tensor] = None, bucketed: bool = True) -> None:
        """Pool per-sample ``values`` (same length as ``t``); optional boolean ``mask`` selects samples."""
        t = t.detach().float().reshape(-1)
        keep = torch.ones_like(t, dtype=torch.bool) if mask is None else mask.reshape(-1).to(torch.bool)
        for name, v in values.items():
            v = v.detach().float().reshape(-1)
            self._acc(name, (v * keep).sum(), keep.sum())
            if bucketed:
                for tag, m in _bucket_masks(t, self.buckets):
                    mm = m & keep
                    self._acc(f"{name}_{tag}", (v * mm).sum(), mm.sum())

    def add_share(self, name: str, t: torch.Tensor, contrib: torch.Tensor) -> None:
        t = t.detach().float().reshape(-1)
        c = contrib.detach().float().reshape(-1)
        d = self._share.setdefault(name, {})
        d[""] = d[""] + c.sum() if "" in d else c.sum()
        for tag, m in _bucket_masks(t, self.buckets):
            key = f"{name}_{tag}"
            d[key] = d[key] + (c * m).sum() if key in d else (c * m).sum()

    def result(self) -> Dict[str, float]:
        """Resolve everything with one host transfer; keys with a zero count are omitted."""
        keys, tensors = [], []
        for k in self._sum:
            keys.append(("mean", k)); tensors += [self._sum[k].reshape(()), self._cnt[k].reshape(())]
        for k, (_, v) in self._ext.items():
            keys.append(("ext", k)); tensors += [v.reshape(()), v.reshape(())]
        for name, d in self._share.items():
            for k, v in d.items():
                if k:
                    keys.append(("share", k)); tensors += [v.reshape(()), d[""].reshape(())]
        if not tensors:
            return {}
        dev = tensors[0].device
        flat = torch.stack([x.to(dev) for x in tensors]).tolist()
        out: Dict[str, float] = {}
        for i, (kind, k) in enumerate(keys):
            a, b = flat[2 * i], flat[2 * i + 1]
            if kind == "mean":
                if b > 0:
                    out[k] = a / b
            elif kind == "ext":
                out[k] = a
            else:
                out[k] = a / b if b > 0 else 0.0
        return out


_POOLED_BASES = ("eqm_raw", "eqm_pred_norm", "eqm_target_norm")


def _is_loss_bucket_key(k: str) -> bool:
    """Scalar keys the Stage-1 loss emits whose exact pooled versions replace them."""
    return any(k == b or k.startswith(b + "_t") for b in _POOLED_BASES)


def pool_eqm_metrics(stats: WindowStats, metrics: Dict[str, object],
                     drop_mask: Optional[torch.Tensor] = None) -> None:
    """Add one EqMLoss micro-batch to ``stats``: scalars, pooled per-sample buckets, objective shares,
    and (given the CFG ``drop_mask``) mse split into class-label and null-label samples."""
    stats.add_scalars({k: v for k, v in metrics.items() if not _is_loss_bucket_key(k)})
    if "_t" not in metrics:
        return
    t = metrics["_t"]
    stats.add_samples(t, {name: metrics["_" + name] for name in
                          ("eqm_raw", "eqm_pred_norm", "eqm_target_norm", "eqm_cos", "eqm_relerr")
                          if "_" + name in metrics})
    if "_eqm_contrib" in metrics:
        stats.add_share("eqm_share", t, metrics["_eqm_contrib"])
    if drop_mask is not None and "_eqm_raw" in metrics:
        dm = drop_mask.reshape(-1).to(torch.bool)
        stats.add_samples(t, {"mse_null": metrics["_eqm_raw"]}, mask=dm, bucketed=False)
        stats.add_samples(t, {"mse_cond": metrics["_eqm_raw"]}, mask=~dm, bucketed=False)
