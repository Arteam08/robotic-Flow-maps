"""Shared logging helpers for the MNIST ablation trainers (Stage 1 and Stage 2).

* :func:`init_wandb` -- one wandb run per run_dir, id persisted in ``wandb_id.txt`` so a
  requeued Slurm job appends to the same run (same pattern as scripts/train_stage2_flowmap.py).
* :class:`StepStats` -- per-window instability statistics: gradient-norm mean / max / p95,
  fraction of steps where clipping fired, non-finite steps, max loss, Adam update ratio,
  EMA distance.
* :func:`param_norm`, :func:`param_delta_norm`, :func:`ema_distance` -- cheap global norms.
"""
from __future__ import annotations

import math
import uuid
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import torch
import torch.nn as nn


def init_wandb(args, run_dir: Path, tags: Optional[List[str]] = None, stage: str = "stage2"):
    """Return a wandb run (or None when ``args.wandb_mode == "disabled"`` or import fails)."""
    mode = getattr(args, "wandb_mode", "disabled")
    if mode == "disabled":
        return None
    try:
        import wandb
    except Exception as e:  # noqa: BLE001
        print(f"[{stage}] wandb import failed ({e!r}); logging to log.jsonl only", flush=True)
        return None
    id_file = run_dir / "wandb_id.txt"
    if getattr(args, "resume", True) and id_file.exists():
        wandb_id = id_file.read_text().strip()
    else:
        wandb_id = uuid.uuid4().hex[:8]
        id_file.write_text(wandb_id + "\n")
    kwargs = dict(
        project=getattr(args, "wandb_project", "eqfm-mnist"),
        entity=getattr(args, "wandb_entity", None),
        name=run_dir.name,
        id=wandb_id,
        resume="allow",
        dir=str(run_dir),
        tags=[stage] + list(tags or []),
        config={k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
    )
    try:
        return wandb.init(mode=mode, **kwargs)
    except Exception as e:  # noqa: BLE001  (a wandb outage must not kill a requeued job)
        print(f"[{stage}] wandb.init failed ({e!r}); falling back to offline", flush=True)
        try:
            return wandb.init(mode="offline", **kwargs)
        except Exception as e2:  # noqa: BLE001
            print(f"[{stage}] wandb offline init failed too ({e2!r}); log.jsonl only", flush=True)
            return None


def wandb_log(run, rec: Dict[str, float], step: int, prefix: str = "train") -> None:
    if run is None:
        return
    try:
        run.log({f"{prefix}/{k}": v for k, v in rec.items() if isinstance(v, (int, float)) and k != "step"}, step=step)
    except Exception as e:  # noqa: BLE001
        print(f"[wandb] log failed at step {step}: {e!r}", flush=True)


@torch.no_grad()
def param_norm(params: Iterable[torch.Tensor]) -> float:
    return math.sqrt(sum(float(p.detach().float().pow(2).sum()) for p in params))


@torch.no_grad()
def snapshot(params: Iterable[torch.Tensor]) -> List[torch.Tensor]:
    return [p.detach().clone() for p in params]


@torch.no_grad()
def param_delta_norm(params: Iterable[torch.Tensor], before: List[torch.Tensor]) -> float:
    return math.sqrt(sum(float((p.detach() - b).float().pow(2).sum()) for p, b in zip(params, before)))


@torch.no_grad()
def ema_distance(model: nn.Module, shadow: Dict[str, torch.Tensor]) -> float:
    """||theta_ema - theta|| / ||theta|| over the parameters tracked by the EMA."""
    num = 0.0
    den = 0.0
    for n, p in model.named_parameters():
        if n in shadow:
            num += float((shadow[n].to(p.dtype) - p.detach()).float().pow(2).sum())
            den += float(p.detach().float().pow(2).sum())
    return math.sqrt(num) / max(math.sqrt(den), 1e-12)


class StepStats:
    """Accumulates per-step numbers between two log records."""

    def __init__(self, grad_clip: float):
        self.grad_clip = grad_clip
        self.reset()
        self.nonfinite_total = 0

    def reset(self) -> None:
        self.gnorms: List[float] = []
        self.losses: List[float] = []
        self.update_ratios: List[float] = []
        self.nonfinite_window = 0

    def add(self, gnorm: float, loss: float, update_ratio: Optional[float] = None) -> bool:
        """Record one step. Returns True when the step is finite (i.e. should be applied)."""
        finite = math.isfinite(gnorm) and math.isfinite(loss)
        if not finite:
            self.nonfinite_window += 1
            self.nonfinite_total += 1
            return False
        self.gnorms.append(gnorm)
        self.losses.append(loss)
        if update_ratio is not None and math.isfinite(update_ratio):
            self.update_ratios.append(update_ratio)
        return True

    def summary(self) -> Dict[str, float]:
        g = sorted(self.gnorms)
        out: Dict[str, float] = {
            "nonfinite_steps": float(self.nonfinite_total),
            "nonfinite_window": float(self.nonfinite_window),
        }
        if g:
            out.update({
                "grad_norm_mean": sum(g) / len(g),
                "grad_norm_max": g[-1],
                "grad_norm_p95": g[min(len(g) - 1, int(0.95 * len(g)))],
                "clip_frac": sum(1.0 for x in g if x > self.grad_clip) / len(g),
                "loss_max": max(self.losses),
                "loss_mean": sum(self.losses) / len(self.losses),
            })
        if self.update_ratios:
            out["update_ratio"] = sum(self.update_ratios) / len(self.update_ratios)
        return out
