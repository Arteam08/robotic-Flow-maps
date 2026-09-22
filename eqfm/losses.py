"""Equilibrium Matching loss + time samplers (Boffi 2026, Sec. 3).

We use the *sampler form* of the autonomous-time L2 loss (Eq. 22):

    L_EqM(b_hat) = E_{t ~ p, z, x} [|| b_hat(I_t) - U_t ||^2]

where p(t) absorbs the 1/c(t) reweighting so the Monte Carlo loss carries
no explicit per-sample weight. Under the canonical exponential clock
c(t) = 1 - t, the sampler is:

    s ~ Unif(0, log(1 / eps_train)),    t = 1 - exp(-s)

so t is concentrated near 1 where the data manifold lives. The truncation
eps_train is required because 1/(1 - t) is not Lebesgue-integrable.

The uniform sampler is exposed as an ablation for testing whether the canonical
objective over-concentrates training near the data end. The unweighted variant
changes the Monte Carlo objective, so treat it as diagnostic rather than the
paper-faithful default. The ``uniform_weighted`` variant samples compactified
time uniformly but applies the density-ratio weight back to the canonical
objective.

    An alternative sampler from Section 9.7 oversamples t -> 1 with a power
law t = 1 - U^k (U ~ Unif[0, 1]); k=2 is the default heuristic.

Optionally adds the data-marginal anchor lambda_eq * E_{x1} ||b_hat(x1)||^2
(Section 9.7) which pins the fixed-point set of b directly on the data.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import torch

from .clocks import inv_c_normalizer, make_clock
from .interpolant import canonical_interpolant, denoiser_target, equilibrium_target


def sample_time_canonical(
    batch_size: int,
    eps_train: float = 1e-3,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Inverse-CDF sampler for ``p(t) ∝ 1/(1 - t)`` on ``[0, 1 - eps_train]``.

    Equivalent to ``s ~ Unif(0, log(1/eps_train))``, ``t = 1 - exp(-s)``.
    """
    log_max = -math.log(eps_train)
    s = torch.rand(batch_size, device=device, dtype=dtype) * log_max
    return 1.0 - torch.exp(-s)


def sample_time_importance(
    batch_size: int,
    k: float = 2.0,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Power-law time-importance sampler ``t = 1 - U^k`` for ``U ~ Unif[0, 1]``.

    With ``k > 1`` the density concentrates near ``t = 1``. ``k = 1`` reduces
    to uniform; ``k = 2`` is the cheap heuristic from Section 9.7.
    """
    u = torch.rand(batch_size, device=device, dtype=dtype)
    return 1.0 - u.pow(k)


def sample_time_uniform(
    batch_size: int,
    eps_train: float = 0.0,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Uniform compactified-time sampler.

    If ``eps_train > 0``, samples on ``[0, 1 - eps_train]`` so importance
    weights against the canonical density stay finite.
    """
    hi = 1.0 - eps_train
    if hi <= 0.0 or hi > 1.0:
        raise ValueError(f"eps_train must lie in [0, 1); got {eps_train}.")
    return torch.rand(batch_size, device=device, dtype=dtype) * hi


def uniform_to_canonical_weight(
    t: torch.Tensor,
    eps_train: float = 1e-3,
) -> torch.Tensor:
    """Density-ratio weight for uniform ``t`` samples.

    Canonical EqM samples ``p(t) = 1 / (log(1/eps) * (1 - t))`` on
    ``[0, 1 - eps]``. Uniform samples use ``q(t) = 1 / (1 - eps)`` over the
    same interval, so ``p/q = (1 - eps) / (log(1/eps) * (1 - t))``.
    """
    if eps_train <= 0.0 or eps_train >= 1.0:
        raise ValueError(f"eps_train must lie in (0, 1); got {eps_train}.")
    log_max = -math.log(eps_train)
    denom = (1.0 - t).clamp_min(eps_train)
    return (1.0 - eps_train) / (log_max * denom)


# ---------------------------------------------------------------------------
# General interval samplers + importance weights (MNIST ablation, 2026-09-11).
# Every sampler draws on [lo, hi) and has a closed-form density so a run can
# sample from one law q and reweight to a reference law p_ref (weight p_ref/q).
# ---------------------------------------------------------------------------

TimeSampler = str  # "uniform" | "canonical" | "power" | "logitnormal"

# Interpolant-time buckets for per-time diagnostics (raw loss, target norm, ...).
T_BUCKETS = ((0.0, 0.1), (0.1, 0.5), (0.5, 0.9), (0.9, 1.0))


def _check_interval(lo: float, hi: float) -> None:
    if not (0.0 <= lo < hi <= 1.0):
        raise ValueError(f"need 0 <= lo < hi <= 1; got lo={lo}, hi={hi}")


def sample_time_on_interval(
    batch_size: int,
    lo: float,
    hi: float,
    sampler: TimeSampler,
    *,
    power_k: float = 2.0,
    ln_mean: float = -0.4,
    ln_std: float = 1.0,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Draw ``t`` in ``[lo, hi)`` from one of the ablation samplers.

    uniform      t = lo + D U
    canonical    tau ~ U[-log(1-lo), -log(1-hi)],  t = 1 - exp(-tau)     (p ∝ 1/(1-t); needs hi < 1)
    power        r = 1 - U^k,                        t = lo + D r          (mass near hi for k > 1)
    logitnormal  r = sigmoid(m + s N(0,1)),           t = lo + D r          (MeanFlow / SD3 style)
    with D = hi - lo.
    """
    _check_interval(lo, hi)
    D = hi - lo
    if sampler == "uniform":
        u = torch.rand(batch_size, device=device, dtype=dtype)
        return lo + D * u
    if sampler == "canonical":
        if hi >= 1.0:
            raise ValueError("canonical sampler needs hi < 1 (1/(1-t) is not integrable)")
        tau_lo, tau_hi = -math.log1p(-lo), -math.log1p(-hi)
        u = torch.rand(batch_size, device=device, dtype=dtype)
        return 1.0 - torch.exp(-(tau_lo + (tau_hi - tau_lo) * u))
    if sampler == "power":
        u = torch.rand(batch_size, device=device, dtype=dtype)
        return lo + D * (1.0 - u.pow(power_k))
    if sampler == "logitnormal":
        n = torch.randn(batch_size, device=device, dtype=dtype)
        return lo + D * torch.sigmoid(ln_mean + ln_std * n)
    raise ValueError(f"unknown sampler {sampler!r}")


def time_density(
    t: torch.Tensor,
    lo: float,
    hi: float,
    sampler: TimeSampler,
    *,
    power_k: float = 2.0,
    ln_mean: float = -0.4,
    ln_std: float = 1.0,
) -> torch.Tensor:
    """Closed-form density ``q(t)`` on ``[lo, hi)`` of :func:`sample_time_on_interval`.

    uniform      1 / D
    canonical    1 / ((1-t) log((1-lo)/(1-hi)))
    power        (1-r)^(1/k - 1) / (k D)
    logitnormal  phi((logit r - m)/s) / (s r (1-r) D)
    with r = (t - lo)/D.  Values are clamped away from the interval ends so
    weights stay finite for samples that land exactly on a boundary.
    """
    _check_interval(lo, hi)
    D = hi - lo
    t = t.float()
    if sampler == "uniform":
        return torch.full_like(t, 1.0 / D)
    if sampler == "canonical":
        if hi >= 1.0:
            raise ValueError("canonical density needs hi < 1")
        z = math.log((1.0 - lo) / (1.0 - hi))
        return 1.0 / ((1.0 - t).clamp_min(1.0 - hi) * z)
    r = ((t - lo) / D).clamp(1e-6, 1.0 - 1e-6)
    if sampler == "power":
        return (1.0 - r).pow(1.0 / power_k - 1.0) / (power_k * D)
    if sampler == "logitnormal":
        x = (torch.log(r) - torch.log1p(-r) - ln_mean) / ln_std
        phi = torch.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)
        return phi / (ln_std * r * (1.0 - r) * D)
    raise ValueError(f"unknown sampler {sampler!r}")


def importance_weight(
    t: torch.Tensor,
    lo: float,
    hi: float,
    sampler: TimeSampler,
    ref: TimeSampler,
    *,
    clip: float = 0.0,
    **density_kwargs,
) -> torch.Tensor:
    """``p_ref(t) / q_sampler(t)``: reweights samples from ``sampler`` to the objective of ``ref``.

    ``sampler == "uniform", ref == "canonical"`` on ``[0, 1-eps)`` reproduces
    :func:`uniform_to_canonical_weight`.  ``clip > 0`` caps the weight.
    """
    if sampler == ref:
        return torch.ones_like(t.float())
    w = time_density(t, lo, hi, ref, **density_kwargs) / time_density(t, lo, hi, sampler, **density_kwargs)
    if clip > 0.0:
        w = w.clamp_max(clip)
    return w


def bucket_stats(
    per: torch.Tensor,
    t: torch.Tensor,
    prefix: str,
    buckets=T_BUCKETS,
    tag: str = "t",
) -> Dict[str, torch.Tensor]:
    """Mean of a per-sample quantity inside each ``[lo, hi)`` bucket of ``t`` (last bucket closed)."""
    per = per.detach().float()
    tt = t.detach().float().reshape(-1)
    out = {f"{prefix}": per.mean()}
    last = buckets[-1][1]
    for lo, hi in buckets:
        m = (tt >= lo) & ((tt < hi) if hi < last else (tt <= hi))
        # Masked mean without a host sync: sum over the bucket divided by its
        # count; an empty bucket gives 0/1 = 0, exactly as the old
        # ``per[m].mean() if m.any() else 0`` did, but with no ``.any()``
        # (device->host copy) and no boolean indexing (data-dependent shape).
        mf = m.to(per.dtype)
        out[f"{prefix}_{tag}{lo:g}_{hi:g}"] = (per * mf).sum() / mf.sum().clamp_min(1.0)
    return out


def weight_stats(w: torch.Tensor, prefix: str) -> Dict[str, torch.Tensor]:
    """mean / max / effective sample size of a per-sample weight vector."""
    w = w.detach().float().reshape(-1)
    ess = w.sum().square() / w.square().sum().clamp_min(1e-30)
    return {f"{prefix}_mean": w.mean(), f"{prefix}_max": w.max(), f"{prefix}_ess": ess}


@dataclass
class EqMLossConfig:
    eps_train: float = 1e-3
    # "canonical" | "importance" | "uniform" | "uniform_weighted"
    time_sampler: str = "canonical"
    importance_k: float = 2.0
    data_anchor_weight: float = 0.0      # lambda_eq for ||b(x1)||^2 anchor
    # "velocity": net outputs the autonomous field b, regresses U_t.
    # "denoiser": net outputs the denoiser D, regresses the clean data x;
    #             the sampler field is reconstructed as b(x) = D(x) - x.
    parameterization: str = "velocity"
    # Auxiliary time-prediction head weight. When > 0, ``model_fn`` is expected
    # to return ``(field, t_pred_logit)`` and an MSE loss pins the head's
    # sigmoid output to the interpolant time so the field's features become
    # time-aware. 0 disables the head entirely (default).
    time_pred_weight: float = 0.0
    # Regression target for the aux head: "t" predicts the raw interpolant time
    # (well-conditioned when t is sampled uniformly); "s_norm" predicts the
    # normalized exponential-clock coordinate -log(1-t)/log(1/eps) in [0,1]
    # (spreads out targets when t concentrates near 1, e.g. canonical sampler).
    time_pred_target: str = "t"
    # Clock c(t) of the target U_t = c(t) (x - z) (spec for eqfm.clocks.make_clock): "linear" = c = 1 - t (default),
    # "trunc:A" = min(1, (1-t)/(1-A)) (Wang & Du 2025, Eq. 5; they use A = 0.8, lambda = 4), "power:P", ...
    # Every clock is multiplied by clock_lambda.
    clock: str = "linear"
    clock_a: float = 0.8
    clock_lambda: float = 1.0
    # time_sampler="uniform_clock_weighted": t ~ U[0,1), per-sample weight 1 / ((c(t) + clock_weight_eps) Z),
    # Z = int_0^1 dt / (c + eps), i.e. the d tau = dt / c measure regularised by eps and normalised to mean 1.
    clock_weight_eps: float = 1e-3


ModelFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
"""Signature for the autonomous model wrapper: ``(latent, class_label) -> velocity``.

The training script supplies this closure so the loss module never needs to
know about time conditioning -- the closure passes ``t = 0`` to SiT.
"""


class EqMLoss:
    """Equilibrium Matching loss with time-sampling and an optional anchor."""

    def __init__(self, config: EqMLossConfig):
        self.config = config

    def sample_time(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if self.config.time_sampler == "canonical":
            return sample_time_canonical(batch_size, self.config.eps_train, device, dtype)
        if self.config.time_sampler == "importance":
            return sample_time_importance(batch_size, self.config.importance_k, device, dtype)
        if self.config.time_sampler in ("uniform", "uniform_clock_weighted"):
            return sample_time_uniform(batch_size, device=device, dtype=dtype)
        if self.config.time_sampler == "uniform_weighted":
            return sample_time_uniform(
                batch_size,
                eps_train=self.config.eps_train,
                device=device,
                dtype=dtype,
            )
        raise ValueError(f"Unknown time_sampler: {self.config.time_sampler!r}")

    def __call__(
        self,
        model_fn: ModelFn,
        x: torch.Tensor,
        y: torch.Tensor,
        *,
        z: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute the EqM loss for one batch.

        Args:
            model_fn: Closure returning the autonomous velocity for a latent
                batch. The training script binds the time argument inside.
            x: Data batch in latent space, shape ``(B, *S)``.
            y: Class labels, shape ``(B,)``.
            z: Optional noise. If ``None`` drawn from ``N(0, I)`` with
                ``torch.randn_like(x)``.

        Returns:
            ``(total_loss, metrics)``. ``metrics`` keys: ``eqm_loss``,
            ``total_loss``, and ``data_anchor`` if the anchor is active.
            All metric tensors are detached.
        """
        B = x.shape[0]
        device, dtype = x.device, x.dtype
        if z is None:
            z = torch.randn_like(x)
        t = self.sample_time(B, device, dtype)
        I_t = canonical_interpolant(z, x, t)
        # In the denoiser parameterization the network outputs D(I_t) and
        # regresses the clean data x; in the velocity parameterization it
        # outputs b(I_t) and regresses U_t. Both give the same per-sample
        # residual (||D - x|| == ||b - U||), so the time sampling/weighting
        # below is shared.
        if self.config.parameterization == "denoiser":
            target = denoiser_target(z, x, t)
        elif self.config.parameterization == "velocity":
            clock = make_clock(self.config.clock, self.config.clock_a, self.config.clock_lambda)
            if clock.is_linear:
                target = equilibrium_target(z, x, t)
            else:
                if self.config.time_sampler == "uniform_weighted":
                    raise ValueError("uniform_weighted assumes the linear clock; use uniform_clock_weighted")
                c = clock.c(t.float()).to(x.dtype)
                target = c.view(-1, *([1] * (x.ndim - 1))) * (x - z)
        else:
            raise ValueError(
                f"Unknown parameterization: {self.config.parameterization!r}"
            )
        out = model_fn(I_t, y)
        # When the aux time-prediction head is on, ``model_fn`` returns a
        # ``(field, t_pred_logit)`` tuple from a single forward pass.
        if isinstance(out, tuple):
            pred, t_pred_logit = out
        else:
            pred, t_pred_logit = out, None
        sq_err = (pred - target) ** 2
        metrics: Dict[str, torch.Tensor]
        per_sample = sq_err.flatten(1).mean(dim=1)
        if self.config.time_sampler == "uniform_clock_weighted":
            clock = make_clock(self.config.clock, self.config.clock_a, self.config.clock_lambda)
            if getattr(self, "_inv_c_Z", None) is None:
                self._inv_c_Z = inv_c_normalizer(clock, self.config.clock_weight_eps)
            weight = (1.0 / ((clock.c(t.float()) + self.config.clock_weight_eps) * self._inv_c_Z)).to(per_sample.dtype)
            eqm = (per_sample * weight).mean()
            metrics = {
                "eqm_loss": eqm.detach(),
                "time_weight_mean": weight.mean().detach(),
                "time_weight_max": weight.max().detach(),
                "time_weight_min": weight.min().detach(),
                "time_weight_ess": weight_stats(weight, "w")["w_ess"],
            }
        elif self.config.time_sampler == "uniform_weighted":
            weight = uniform_to_canonical_weight(t, self.config.eps_train).to(per_sample.dtype)
            eqm = (per_sample * weight).mean()
            metrics = {
                "eqm_loss": eqm.detach(),
                "time_weight_mean": weight.mean().detach(),
                "time_weight_max": weight.max().detach(),
                "time_weight_min": weight.min().detach(),
                "time_weight_ess": weight_stats(weight, "w")["w_ess"],
            }
        else:
            eqm = sq_err.mean()
            metrics = {"eqm_loss": eqm.detach()}
        # Instability diagnostics: unweighted per-sample loss, target and prediction
        # magnitude, each bucketed by interpolant time t (see T_BUCKETS).
        metrics.update(bucket_stats(per_sample, t, "eqm_raw"))
        metrics.update(bucket_stats(target.detach().float().flatten(1).norm(dim=1), t, "eqm_target_norm"))
        metrics.update(bucket_stats(pred.detach().float().flatten(1).norm(dim=1), t, "eqm_pred_norm"))
        metrics["eqm_loss_max"] = per_sample.detach().max()
        # Per-sample tensors (underscore keys are never logged directly): the trainer pools them over
        # every micro-batch between log points into exact per-t-bucket means (eqfm.metric_window).
        with torch.no_grad():
            p_flat = pred.detach().float().flatten(1)
            g_flat = target.detach().float().flatten(1)
            p_norm, g_norm = p_flat.norm(dim=1), g_flat.norm(dim=1)
            metrics["_t"] = t.detach().float().reshape(-1)
            metrics["_eqm_raw"] = per_sample.detach().float()
            metrics["_eqm_pred_norm"] = p_norm
            metrics["_eqm_target_norm"] = g_norm
            metrics["_eqm_cos"] = (p_flat * g_flat).sum(dim=1) / (p_norm * g_norm).clamp_min(1e-12)
            metrics["_eqm_relerr"] = (p_flat - g_flat).norm(dim=1) / g_norm.clamp_min(1e-12)
            contrib = per_sample.detach().float()
            if self.config.time_sampler in ("uniform_weighted", "uniform_clock_weighted"):
                contrib = contrib * weight.detach().float()
            metrics["_eqm_contrib"] = contrib

        total = eqm

        if self.config.time_pred_weight > 0.0:
            if t_pred_logit is None:
                raise RuntimeError(
                    "time_pred_weight > 0 but model_fn returned no time "
                    "prediction; build the closure with return_time_pred=True."
                )
            t_hat = torch.sigmoid(t_pred_logit)
            if self.config.time_pred_target == "s_norm":
                log_max = -math.log(self.config.eps_train)
                t_tgt = -(1.0 - t).clamp_min(self.config.eps_train).log() / log_max
            elif self.config.time_pred_target == "t":
                t_tgt = t
            else:
                raise ValueError(
                    f"Unknown time_pred_target: {self.config.time_pred_target!r}"
                )
            t_tgt = t_tgt.to(t_hat.dtype)
            tp_loss = ((t_hat - t_tgt) ** 2).mean()
            total = total + self.config.time_pred_weight * tp_loss
            metrics["time_pred_loss"] = tp_loss.detach()
            metrics["time_pred_mae"] = (t_hat - t_tgt).abs().mean().detach()

        if self.config.data_anchor_weight > 0.0:
            # Anchor pins the autonomous field b to zero on the data, i.e.
            # x is a fixed point. With the denoiser parameterization the
            # field at the data is b(x) = D(x) - x, so the anchor becomes
            # the denoising-identity residual ||D(x) - x||^2.
            out_at_data = model_fn(x, y)
            if isinstance(out_at_data, tuple):
                out_at_data = out_at_data[0]
            b_at_data = out_at_data - x if self.config.parameterization == "denoiser" else out_at_data
            anchor = (b_at_data ** 2).mean()
            total = total + self.config.data_anchor_weight * anchor
            metrics["data_anchor"] = anchor.detach()

        metrics["total_loss"] = total.detach()
        return total, metrics
