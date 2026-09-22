"""Clocks c(t) for the autonomous field b (and the flow-map semigroup built on it).

Along the interpolant I_t = (1 - t) z + t x a clock sets the regression target and the
autonomous time in which the field b traverses the path:

    U_t = c(t) (x - z),      dtau = dt / c(t),      dx/dtau = b(x).

    power p   c = (1-t)^p               tau = -log(1-t)                  (p = 1)
                                        tau = ((1-t)^(1-p) - 1) / (p-1)  (p > 1)
    logref    c = (1-t) log(1/(1-t))    tau = log log(1/(1-t)),   1 - t = exp(-e^tau)
    trunc a   c = min(1, (1-t)/(1-a))   tau = t (t <= a),   a + (1-a) log((1-a)/(1-t))
    logsnr    c = t (1-t)               tau = log(t/(1-t))
    lam * c                             tau -> tau / lam

Semigroup law, only when tau(0) is finite (power, trunc):

    sigma1 (+) sigma2 = t( tau(sigma1) + tau(sigma2) - tau(0) ).

For power p > 1 this is (1 - s1(+)s2)^(1-p) = (1-s1)^(1-p) + (1-s2)^(1-p) - 1, so K equal
steps give (1 - sigma_K)^(1-p) = K (1-sigma)^(1-p) - (K-1); only p = 1 has the
multiplicative form 1 - s1(+)s2 = (1-s1)(1-s2).  logref and logsnr put noise at
tau = -inf (c(0) = 0): no identity element, so the law is not defined there.

Grids for Euler on dx/dtau = b over t in [t_start, 1 - eps_stop] (all clocks integrate the
same interpolant path; they differ in where the steps go):

    geo   1 - t_k geometric      (uniform tau of the p = 1 clock; the historical default)
    t     t_k uniform
    tau   uniform in this clock's own tau
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List

import torch

KINDS = ("power", "logref", "trunc", "logsnr")
GRIDS = ("geo", "t", "tau")


@dataclass(frozen=True)
class Clock:
    kind: str = "power"
    p: float = 1.0
    a: float = 0.8
    lam: float = 1.0

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"unknown clock kind {self.kind!r}; expected one of {KINDS}")
        if self.kind == "power" and self.p < 1.0:
            raise ValueError(f"power clock needs p >= 1, got {self.p}")
        if self.kind == "trunc" and not 0.0 <= self.a < 1.0:
            raise ValueError(f"trunc clock needs 0 <= a < 1, got {self.a}")
        if self.lam <= 0.0:
            raise ValueError(f"clock scale lam must be > 0, got {self.lam}")

    @property
    def name(self) -> str:
        base = {"power": f"power{self.p:g}", "logref": "logref", "trunc": f"trunc{self.a:g}", "logsnr": "logsnr"}[self.kind]
        return base if self.lam == 1.0 else f"{base}_lam{self.lam:g}"

    @property
    def is_linear(self) -> bool:
        return self.kind == "power" and self.p == 1.0 and self.lam == 1.0

    # ---- c(t) on tensors (training targets) ----
    def c(self, t: torch.Tensor) -> torch.Tensor:
        k = (1.0 - t).clamp_min(0.0)
        if self.kind == "power":
            out = k if self.p == 1.0 else k.pow(self.p)
        elif self.kind == "logref":
            out = k * (-torch.log1p(-t.clamp_max(1.0 - 1e-12)))
        elif self.kind == "trunc":
            out = torch.where(t <= self.a, torch.ones_like(t), k / (1.0 - self.a))
        else:  # logsnr
            out = t * k
        return self.lam * out

    # ---- tau(t) and its inverse on python floats (float64; grids can span 1e11) ----
    def tau(self, t: float) -> float:
        if self.kind == "power":
            if t >= 1.0:
                return math.inf
            v = -math.log1p(-t) if self.p == 1.0 else ((1.0 - t) ** (1.0 - self.p) - 1.0) / (self.p - 1.0)
        elif self.kind == "logref":
            if t <= 0.0:
                return -math.inf
            if t >= 1.0:
                return math.inf
            v = math.log(-math.log1p(-t))
        elif self.kind == "trunc":
            if t >= 1.0:
                return math.inf
            v = t if t <= self.a else self.a + (1.0 - self.a) * math.log((1.0 - self.a) / (1.0 - t))
        else:  # logsnr
            if t <= 0.0:
                return -math.inf
            if t >= 1.0:
                return math.inf
            v = math.log(t) - math.log1p(-t)
        return v / self.lam

    def t_of_tau(self, tau: float) -> float:
        s = tau * self.lam
        if self.kind == "power":
            if self.p == 1.0:
                return -math.expm1(-s)
            return 1.0 - (1.0 + (self.p - 1.0) * s) ** (-1.0 / (self.p - 1.0))
        if self.kind == "logref":
            return -math.expm1(-math.exp(s))
        if self.kind == "trunc":
            return s if s <= self.a else 1.0 - (1.0 - self.a) * math.exp(-(s - self.a) / (1.0 - self.a))
        return 1.0 / (1.0 + math.exp(-s))  # logsnr

    @property
    def tau0(self) -> float:
        return self.tau(0.0)

    @property
    def has_identity(self) -> bool:
        return math.isfinite(self.tau0)

    def compose(self, s1: float, s2: float) -> float:
        """sigma1 (+) sigma2 via tau additivity; needs a finite tau(0)."""
        if not self.has_identity:
            raise ValueError(f"clock {self.name} has tau(0) = -inf: no semigroup identity, (+) undefined")
        return self.t_of_tau(self.tau(s1) + self.tau(s2) - self.tau0)

    def iterate(self, sigma: float, K: int) -> float:
        """sigma_K = sigma (+) ... (+) sigma (K times)."""
        return self.t_of_tau(self.tau0 + K * (self.tau(sigma) - self.tau0))


def inv_c_normalizer(clock: Clock, eps: float) -> float:
    """Z = int_0^1 dt / (c(t) + eps) (trapezoid on a grid refined geometrically at both ends).

    uniform t with weight 1/((c + eps) Z) is the d tau = dt / c measure regularised by eps, normalised to mean 1."""
    n = 200_000
    ts = torch.cat([torch.linspace(0.0, 1.0, n + 1, dtype=torch.float64),
                    1.0 - torch.logspace(0.0, -15.0, n // 10, dtype=torch.float64),
                    torch.logspace(-15.0, 0.0, n // 10, dtype=torch.float64)])
    ts = torch.unique(ts.clamp(0.0, 1.0))
    f = 1.0 / (clock.c(ts) + eps)
    return float(torch.trapezoid(f, ts))


def make_clock(spec: str, clock_a: float = 0.8, clock_lambda: float = 1.0) -> Clock:
    """``linear`` | ``power:P`` | ``logref`` | ``trunc`` / ``trunc:A`` | ``logsnr`` (scale from ``clock_lambda``)."""
    kind, _, arg = spec.partition(":")
    if kind == "linear":
        return Clock("power", p=1.0, lam=clock_lambda)
    if kind == "power":
        return Clock("power", p=float(arg or 1.0), lam=clock_lambda)
    if kind == "trunc":
        return Clock("trunc", a=float(arg) if arg else clock_a, lam=clock_lambda)
    if kind in ("logref", "logsnr"):
        if arg:
            raise ValueError(f"clock {kind} takes no parameter, got {spec!r}")
        return Clock(kind, lam=clock_lambda)
    raise ValueError(f"unknown clock spec {spec!r}")


def default_t_start(clock: Clock) -> float:
    """0 when noise sits at finite tau; else start just off the noise end (tau(0) = -inf)."""
    return 0.0 if clock.has_identity else 1e-3


def t_grid(clock: Clock, K: int, grid: str, eps_stop: float, t_start: float) -> List[float]:
    """K + 1 interpolant times from t_start to 1 - eps_stop (see module docstring)."""
    t_end = 1.0 - eps_stop
    if not 0.0 <= t_start < t_end:
        raise ValueError(f"need 0 <= t_start < 1 - eps_stop, got {t_start}, {t_end}")
    if grid == "geo":
        r0, r1 = 1.0 - t_start, eps_stop
        return [1.0 - r0 * (r1 / r0) ** (k / K) for k in range(K + 1)]
    if grid == "t":
        return [t_start + (t_end - t_start) * k / K for k in range(K + 1)]
    if grid == "tau":
        a, b = clock.tau(t_start), clock.tau(t_end)
        if not (math.isfinite(a) and math.isfinite(b)):
            raise ValueError(f"tau grid needs finite endpoints; clock {clock.name}, t_start={t_start}")
        ts = [clock.t_of_tau(a + (b - a) * k / K) for k in range(K + 1)]
        ts[0], ts[-1] = t_start, t_end
        return ts
    raise ValueError(f"unknown grid {grid!r}; expected one of {GRIDS}")


def tau_steps(clock: Clock, K: int, grid: str, eps_stop: float, t_start: float) -> List[float]:
    """Euler step sizes h_k = tau(t_{k+1}) - tau(t_k) for x <- x + h_k b(x)."""
    if grid == "tau":
        a, b = clock.tau(t_start), clock.tau(1.0 - eps_stop)
        return [(b - a) / K] * K
    ts = t_grid(clock, K, grid, eps_stop, t_start)
    taus = [clock.tau(t) for t in ts]
    return [taus[k + 1] - taus[k] for k in range(K)]
