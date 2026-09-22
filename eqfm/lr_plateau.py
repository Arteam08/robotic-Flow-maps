"""Loss-plateau learning-rate slowdown for long Stage-2 runs.

The trainer's lr is ``lr(step) = scheduled_lr(step) * scale``. Every ``window`` optimizer steps the trainer
passes the windowed mean of a monitored loss (the unweighted training loss, averaged over every sample of
every rank) to :meth:`LossPlateau.end_window`.

Stall test (block comparison, robust to noise): once ``2 * patience`` windows have been collected since the
start or since the last cut, compare the mean of the last ``patience`` windows (``recent``) with the mean of
the ``patience`` windows before them (``previous``). The loss has stalled when

    recent > previous * (1 - threshold)

i.e. it improved by less than ``threshold`` (relative) over ``patience`` windows. Then ``scale`` is multiplied
by ``factor`` (never below ``min_scale``) and a fresh block of ``2 * patience`` windows is needed before the
next test. Comparing block means (not the best single window) avoids the downward bias of a running minimum of
noisy window means, which would declare stalls while the loss is still improving. The test runs once per block
(every ``patience`` windows), so consecutive tests do not reuse the same windows. The state is small and goes
into every checkpoint.

Noise budget (ImageNet, measured per-step coefficient of variation of the unweighted loss ~50% at 128 samples):
a 1000-step window mean has ~1.6% noise, a 5-window block ~0.7%, the block difference ~1.0%; a true 3% drop per
block is then declared a stall with probability ~2% per test.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional


class LossPlateau:
    def __init__(self, window: int, patience: int, threshold: float, factor: float,
                 min_scale: float, cooldown: int = 0, start_step: int = 0, stat: str = "mean") -> None:
        if window <= 0 or patience <= 0:
            raise ValueError("window and patience must be positive")
        if not 0.0 < factor < 1.0:
            raise ValueError("factor must lie in (0, 1)")
        self.window, self.patience, self.threshold = window, patience, threshold
        self.factor, self.min_scale, self.cooldown, self.start_step = factor, min_scale, cooldown, start_step
        self.stat = stat                          # how the trainer summarises a window (geomean or mean)
        self.scale = 1.0
        self.block: List[float] = []            # window means since the start or the last cut (after cooldown)
        self.cool = 0
        self.recent = math.nan
        self.previous = math.nan
        self.history: List[Dict[str, float]] = []

    # compatibility with the logging code: "best" = previous block mean, "bad" = windows collected in the block
    @property
    def best(self) -> float:
        return self.previous

    @property
    def bad(self) -> int:
        return len(self.block)

    def is_window_end(self, step_done: int) -> bool:
        """``step_done`` = number of optimizer steps completed (step + 1)."""
        return step_done > self.start_step and step_done % self.window == 0

    def end_window(self, step_done: int, mean: float) -> Optional[str]:
        """Record a finished window; returns a message when the lr scale changed."""
        event = None
        if not math.isfinite(mean):
            pass
        elif self.cool > 0:
            self.cool -= 1
        else:
            self.block.append(mean)
            p = self.patience
            if len(self.block) >= 2 * p and len(self.block) % p == 0:   # non-overlapping blocks: fewer false stalls
                self.recent = sum(self.block[-p:]) / p
                self.previous = sum(self.block[-2 * p:-p]) / p
                if self.recent > self.previous * (1.0 - self.threshold) and self.scale > self.min_scale:
                    old = self.scale
                    self.scale = max(self.min_scale, self.scale * self.factor)
                    event = (f"unweighted loss stalled: mean of the last {p} windows {self.recent:.5g} vs "
                             f"previous {p} windows {self.previous:.5g} (needed a {100 * self.threshold:g}% drop): "
                             f"lr scale {old:g} -> {self.scale:g}")
                    self.block = []
                    self.cool = self.cooldown
        self.history.append({"step": step_done, "mean": mean, "recent": self.recent, "previous": self.previous,
                             "scale": self.scale})
        return event

    def state_dict(self) -> dict:
        return {"scale": self.scale, "block": list(self.block), "cool": self.cool, "recent": self.recent,
                "previous": self.previous, "history": self.history[-500:], "stat": self.stat}

    def load_state_dict(self, d: dict) -> None:
        self.scale = float(d.get("scale", 1.0))
        self.history = list(d.get("history", []))
        if d.get("stat", "mean") != self.stat:
            # window values of another statistic are not comparable: keep the lr scale, restart the block
            self.block, self.cool, self.recent, self.previous = [], 0, math.nan, math.nan
            return
        self.block = [float(v) for v in d.get("block", [])]
        self.cool = int(d.get("cool", 0))
        self.recent = float(d.get("recent", math.nan))
        self.previous = float(d.get("previous", math.nan))
