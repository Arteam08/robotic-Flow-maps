"""Stage-2 compactified flow-map distillation losses.

The implementation follows the pretrained-anchor recipe from EqFM Section 6.4:

    X_sigma(y) = y + sigma * v_sigma(y)

is trained with a frozen Stage-1 autonomous teacher ``b_pre`` at the diagonal,
one of the compactified off-diagonal objectives (LESD/EESD/PESD), and a
terminal data anchor ``X_1(x_data) = x_data``.
"""
from __future__ import annotations

import contextlib
import math
from dataclasses import dataclass
from typing import Callable, Dict, Literal, Optional, Tuple

import torch
import torch.nn as nn

from .flowmap import (
    broadcast_time,
    compactified_add,
    compactified_sub,
    flow_map,
    sigma_derivative,
    spatial_jvp,
    velocity_jvp,
)
from .cfg_rules import CFG_RULES, perp_guided, warn_if_perp_cnorm
from .interpolant import canonical_interpolant
from .losses import (
    T_BUCKETS,
    bucket_stats,
    importance_weight,
    sample_time_canonical,
    sample_time_importance,
    sample_time_on_interval,
    sample_time_uniform,
    weight_stats,
)


Objective = Literal[
    "lagrangian", "eulerian", "semigroup", "meanflow", "scaled_meanflow", "consistency",
    "shifted_lagrangian", "shifted_meanflow"
]
CfgRule = Literal["standard", "perp", "perp_matched"]
InterpSampler = Literal["uniform", "canonical", "importance", "power", "logitnormal"]
SigmaSampler = Literal["uniform", "canonical", "power", "logitnormal"]
TimeRef = Literal["uniform", "canonical"]
MeanFlowClock = Literal["clamp", "add"]
GuidanceMode = Literal["branch", "guided"]
CfgMode = Literal["all", "first3"]
TeacherFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


@dataclass
class Stage2LossConfig:
    objective: Objective = "lagrangian"
    # Paper-faithful input distribution for the autonomous source-sink field:
    # sample interpolant points from autonomous occupation,
    # p(t) proportional to 1 / (1 - t), truncated by eps_train.
    # Uniform t is kept only as an ablation.
    interp_time_sampler: InterpSampler = "canonical"
    eps_train: float = 1e-3
    importance_k: float = 2.0
    sigma_min: float = 1e-4
    sigma_max: float = 1.0 - 1e-3
    # "uniform": sigma ~ Unif[sigma_min, sigma_max] (default, all ImageNet runs).
    # "canonical": uniform in autonomous time tau = -log(1 - sigma) on the same interval,
    #              i.e. p(sigma) ∝ 1/(1 - sigma): the occupation density of the exponential clock.
    sigma_sampler: str = "uniform"
    # consistency objective (EqFM Sec. 5.3 item 4, teacher form): small compactified step delta along the
    # teacher field, target = EMA student at the shorter duration:
    #     || X_theta(s (+) delta, x0) - sg X_ema(s, Phi_h(x0)) ||^2,   Phi_h = cd_substeps Euler steps of b, h = -log(1-delta)
    cd_delta: float = 0.05
    cd_substeps: int = 1
    # meanflow: floor on (1 - sigma) in the target's division, so the bootstrap gain sigma/(1-sigma) is bounded
    meanflow_eps: float = 1e-2
    diag_weight: float = 1.0
    distill_weight: float = 1.0
    terminal_weight: float = 1.0
    terminal_teacher_norm_weight: float = 0.0
    terminal_student_norm_weight: float = 0.0
    terminal_norm_max_batch: int = 0
    # ---- self-correction endpoint term (drop-in replacement for terminal_teacher_norm) ----
    # Both terms score the endpoint of X_1 on the SAME pure-noise inputs z. The teacher-norm term
    # asks ||b(X_1(z))|| -> 0, i.e. "the endpoint is an equilibrium of the teacher field", and needs
    # the teacher (with grad through its input) at sampling-distribution points. This one instead asks
    #
    #     || X_1( sg X_1(z) ) - sg C(z) ||^2 ,      C(z) = the reference chain below
    #
    # i.e. "a second application of X_1 carries the one-shot output onto the reference sample", which
    # needs no teacher at all. Only the OUTER X_1 carries gradient, so the term trains the second
    # application as a corrector rather than moving the one-shot.
    selfcorr_weight: float = 0.0
    # C(z) = selfcorr_chain_steps jumps of selfcorr_chain_sigma applied to z, then X_1 if
    # selfcorr_chain_final_x1. So (sigma=1, steps=k, final=False) is X_1^k(z), and
    # (sigma=0.1, steps=8, final=True) is the 9-NFE sampler X_1(X_0.1^8(z)).
    selfcorr_chain_sigma: float = 1.0
    selfcorr_chain_steps: int = 4
    selfcorr_chain_final_x1: bool = False
    # Cap the term to this many samples per rank (the teacher-norm term it replaces used 4 of 8).
    selfcorr_max_batch: int = 0
    # Run the student passes that are NOT differentiated in forward mode
    # (diagonal, terminal, terminal-norm, semigroup) under bf16 autocast while
    # the caller keeps the JVP pass (lagrangian/eulerian) in fp32. Loss
    # residuals are formed against fp32 targets, so reductions stay fp32.
    student_bf16_nonjvp: bool = False
    # Also run the forward-mode JVP pass of the Lagrangian objective (and its
    # backward) under bf16 autocast. Opt-in; checked against fp32 with
    # scripts/s2_speed_check.py (gradient cosine vs mini-batch noise).
    student_bf16_jvp: bool = False
    # scaled_meanflow: compute the target's two JVP terms  s (D_x v) b0 - k s d_s v  in ONE forward-mode pass
    # with tangents (s b0, -k s) instead of two (identical value; ported from branch meanflow-lagrangian).
    mf_single_jvp: bool = False
    guidance_mode: GuidanceMode = "branch"
    teacher_cfg_scale: float = 1.0
    cfg_mode: CfgMode = "all"
    cfg_match_uncond_norm: bool = False
    # Rescale the *final* guided vector back to ||b_cond||, mirroring the
    # sampler's --cfg-match-guided-norm. In an autonomous field the magnitude
    # of b IS the clock: near data b ~ -L(y - x1) and the compactified clock
    # k(sigma) = 1 - sigma is calibrated to L = 1, so a guidance rule that
    # returns lambda(y) * b_cond makes the residual decay like exp(-lambda*L*t).
    # A constant lambda is an admissible reparameterization (Prop. G.2), but
    # CFG's lambda varies with position and class, which decalibrates the clock
    # and breaks the composition law 1-(s1 (+) s2) = (1-s1)(1-s2) that
    # multi-step sampling relies on. Forcing ||b_guided|| = ||b_cond|| keeps
    # guidance purely angular and preserves the landing rate. This is the
    # setting the best Stage-1 FID sampler uses (FID 2.08); without it the
    # Stage-2 student distills a clock-distorted teacher.
    cfg_match_guided_norm: bool = False
    # Perpendicular guidance. b_c owns the landing (and the convergence
    # guarantee); guidance may only relocate WHICH equilibrium we reach, so the
    # perturbation must carry no b_c-parallel component:
    #     u = -s * b_u^perp,   b_u^perp = b_u - (b_u . b_c^) b_c^,   s = scale - 1
    # The b_c-parallel part of the guided field is then EXACTLY b_c, so the
    # landing rate is untouched by construction (measured cos(u, b_c) = 0.000
    # across every toy geometry, vs 0.21-0.53 for the naive
    # bc + s*(bc - bu_perp) form, which is really (1+s)bc - s*bu_perp and
    # multiplies the landing rate by 1+s).
    cfg_rule: CfgRule = "standard"
    # Scale the perturbation to ||u|| = s * ||b_c||, so s is dimensionless and
    # the guidance inherits b_c's own decay toward the manifold -- annealing
    # tied to the current point rather than hand-tuned. Best rule in the toy
    # study: purity 0.952 at angular entropy 0.301 where plain perp gets 0.900.
    cfg_cnorm: bool = False
    cfg_dropout_prob: float = 0.1
    null_class: int = 1000
    adaptive_weight_p: float = 0.0
    adaptive_weight_eps: float = 1e-3
    # Endpoint trace of the compact MeanFlow equation.  Stationarity
    # b(P(x))=0 only says the output lies on the equilibrium set; this term
    # preserves which teacher orbit/equilibrium the input belongs to via
    # P(x)=P(Phi_delta(x)).
    terminal_invariance_weight: float = 0.0
    terminal_invariance_step: float = 0.1
    terminal_invariance_max_batch: int = 0
    # Auxiliary time-prediction head (self-supervised, NO DINO/REPA features).
    # When > 0, the student's diagonal forward also emits a t-prediction logit
    # (from the head attached via ``student.enable_time_prediction``) and an
    # MSE loss pins its sigmoid to the interpolant time t of x0. This forces
    # the flow-map student's features to encode *where on the z->x_data
    # trajectory* the input sits, rather than relying only on the teacher/sigma
    # signal. 0 disables (default). Target: "t" (raw) or "s_norm".
    time_pred_weight: float = 0.0
    time_pred_target: str = "t"
    # ---- MNIST ablation additions (2026-09-11); defaults reproduce the previous behaviour ----
    # Importance-sampling correction of the interpolant time t: sample from
    # ``interp_time_sampler`` on [0, 1-eps_train) and multiply the per-sample
    # loss by p_ref(t)/q(t) so the objective is that of ``interp_ref``.
    # (Stage-2 losses are pointwise identities, so this is emphasis, not bias.)
    interp_is_correct: bool = False
    interp_ref: TimeRef = "canonical"
    interp_logitnormal_mean: float = -0.4
    interp_logitnormal_std: float = 1.0
    # Same for the jump sigma on [sigma_min, sigma_max).
    sigma_is_correct: bool = False
    sigma_ref: TimeRef = "canonical"
    sigma_power_k: float = 2.0
    sigma_logitnormal_mean: float = -0.4
    sigma_logitnormal_std: float = 1.0
    # Cap on either importance weight (0 = no cap). Heavy-tailed proposals
    # (logit-normal) need it.
    time_weight_clip: float = 0.0
    # meanflow (divided) target clock: "clamp" -> max(1-sigma, eps); "add" -> (1-sigma) + eps.
    meanflow_clock: MeanFlowClock = "clamp"
    # Generic PESD add-on: semigroup_weight * L_PESD added to ANY objective
    # (replaces the trainer-side lag_semi subclass).
    semigroup_weight: float = 0.0
    # Adaptive-weight granularity: "sample" -> w_i = (L_i + eps)^-p (per sample, the original);
    # "batch" -> w = (mean_i L_i + eps)^-p shared by the whole batch (pure scale normalisation:
    # per-sample gradients stay proportional to their residual).
    adaptive_weight_mode: str = "sample"
    # Per-sample loss reduction over the D entries: "sum" (original) or "mean" (per-entry, D-invariant).
    # Default "mean" since 2026-09-20 (was "sum"); pass "sum" to reproduce runs trained before that.
    loss_reduction: str = "mean"


def _per_sample_mse(x: torch.Tensor, reduction: str = "sum") -> torch.Tensor:
    """Per-sample squared residual. ``sum``: ||r_i||^2 over all entries (original convention, units of a
    squared distance in sample space). ``mean``: (1/D) ||r_i||^2, the flow-matching / MeanFlow convention in
    which eps = 1e-3 and grad clip 1.0 are calibrated; the gradient then scales as D^(p-1) relative to sum."""
    sq = x.float().flatten(1).pow(2)
    return sq.mean(dim=1) if reduction == "mean" else sq.sum(dim=1)


def _adaptive_factor(
    loss_per_sample: torch.Tensor,
    p: float,
    eps: float | torch.Tensor,
) -> torch.Tensor:
    if p <= 0.0:
        return torch.ones_like(loss_per_sample.detach())
    return (loss_per_sample.detach() + eps).pow(-p)


def _weighted_mean(
    loss_per_sample: torch.Tensor,
    p: float,
    eps: float | torch.Tensor,
    sample_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """``mean_i[ w_i * L_i * (sg L_i + eps)^-p ]``.

    The adaptive factor is computed from the UNWEIGHTED loss; ``sample_weight``
    (importance weight) multiplies it afterwards. Folding ``w`` into ``L`` first
    would give ``wL/(wL+eps) ~ 1`` at p=1 and silently cancel the reweighting.
    """
    weight = _adaptive_factor(loss_per_sample, p, eps)
    if sample_weight is not None:
        weight = weight * sample_weight.detach().to(weight.dtype)
    return (loss_per_sample * weight).mean()


# Buckets chosen to line up with the K-step validation rollouts, whose per-step
# sigma is 1 - eps^(1/K): K=8 -> 0.578, K=4 -> 0.822, K=2 -> 0.968, K=1 -> 0.999.
_SIGMA_BUCKETS = ((0.0, 0.5), (0.5, 0.9), (0.9, 0.99), (0.99, 1.0))


def _raw_sigma_stats(
    per: torch.Tensor, sigma: torch.Tensor, prefix: str
) -> Dict[str, torch.Tensor]:
    """Unweighted per-sample error, overall and bucketed by sigma.

    The adaptive weight makes the reported loss uninformative at p -> 1: the
    per-sample contribution is ``e/(e+eps)``, which pins at 1 whenever
    ``e >> eps``. These raw numbers are what actually shows whether the run is
    learning, and the buckets show *where* in the clock it is or is not.
    """
    per = per.detach().float()
    sig = sigma.detach().float().reshape(-1)
    out = {f"{prefix}_raw": per.mean()}
    for lo, hi in _SIGMA_BUCKETS:
        m = (sig >= lo) & (sig < hi) if hi < 1.0 else (sig >= lo)
        key = f"{prefix}_raw_s{lo:g}_{hi:g}"
        # Sync-free masked mean (see losses.bucket_stats): empty bucket -> 0.
        mf = m.to(per.dtype)
        out[key] = (per * mf).sum() / mf.sum().clamp_min(1.0)
    return out


def _time_pred_terms(
    t_logit: torch.Tensor,
    t: torch.Tensor,
    target: str,
    eps_train: float,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """MSE loss pinning the head's sigmoid output to the interpolant time.

    ``target="t"`` regresses the raw interpolant time (well-conditioned under
    uniform sampling); ``"s_norm"`` regresses the normalized exponential-clock
    coordinate ``-log(1-t)/log(1/eps)`` in [0, 1] (spreads targets out when t
    concentrates near 1, e.g. the canonical sampler).
    """
    t_hat = torch.sigmoid(t_logit)
    if target == "s_norm":
        log_max = -math.log(eps_train)
        t_tgt = -(1.0 - t).clamp_min(eps_train).log() / log_max
    elif target == "t":
        t_tgt = t
    else:
        raise ValueError(f"Unknown time_pred_target: {target!r}")
    t_tgt = t_tgt.to(t_hat.dtype)
    tp_loss = ((t_hat - t_tgt) ** 2).mean()
    return tp_loss, {
        "time_pred_loss": tp_loss.detach(),
        "time_pred_mae": (t_hat - t_tgt).abs().mean().detach(),
    }


def _combine_cfg(
    b_cond: torch.Tensor,
    b_uncond: torch.Tensor,
    cfg_scale: float,
    cfg_mode: CfgMode,
) -> torch.Tensor:
    guided = b_uncond + cfg_scale * (b_cond - b_uncond)
    if cfg_mode == "all":
        return guided
    out = b_cond.clone()
    out[:, :3] = guided[:, :3]
    return out


def _perp_guided(
    b_cond: torch.Tensor,
    b_uncond: torch.Tensor,
    cfg_scale: float,
    cnorm: bool,
) -> torch.Tensor:
    """b_c + s*(b_c - b_u)^perp  ==  b_c - s*b_u^perp,  with s = cfg_scale - 1."""
    s = cfg_scale - 1.0
    flat_c = b_cond.float().flatten(1)
    nc = flat_c.norm(dim=1).clamp_min(1e-12)
    shape = (-1, *([1] * (b_cond.ndim - 1)))
    ch = (flat_c / nc.unsqueeze(1)).view_as(b_cond)
    dot = (b_uncond.float().flatten(1) * ch.flatten(1)).sum(dim=1).view(shape)
    u = -(b_uncond.float() - dot * ch)
    if cnorm:
        un = u.flatten(1).norm(dim=1).clamp_min(1e-12).view(shape)
        u = u / un * (s * nc.view(shape))
    else:
        u = s * u
    return (b_cond.float() + u).to(dtype=b_cond.dtype)


def _match_uncond_norm(b_cond: torch.Tensor, b_uncond: torch.Tensor) -> torch.Tensor:
    cond_norm = b_cond.float().flatten(1).norm(dim=1)
    uncond_norm = b_uncond.float().flatten(1).norm(dim=1).clamp_min(1e-12)
    scale = (cond_norm / uncond_norm).view(-1, *([1] * (b_uncond.ndim - 1)))
    return b_uncond * scale.to(dtype=b_uncond.dtype)


class Stage2FlowMapLoss:
    """Compute Stage-2 losses for a compactified flow-map student."""

    _aux: Dict[str, torch.Tensor]

    def __init__(self, config: Stage2LossConfig):
        self._aux = {}
        # Opt-in: keep detached per-sample tensors of the last call in ``self._samples`` so a trainer can
        # pool exact per-bucket statistics (eqfm/stage2_wandb.py). Off by default; no effect on the loss.
        self.collect_samples = False
        self._samples: Dict[str, torch.Tensor] = {}
        if config.objective not in {
            "lagrangian", "eulerian", "semigroup", "meanflow", "scaled_meanflow", "consistency",
            "shifted_lagrangian", "shifted_meanflow"
        }:
            raise ValueError(f"unknown Stage-2 objective: {config.objective!r}")
        if config.sigma_sampler not in {"uniform", "canonical", "power", "logitnormal"}:
            raise ValueError(f"unknown sigma sampler: {config.sigma_sampler!r}")
        if config.interp_time_sampler not in {"uniform", "canonical", "importance", "power", "logitnormal"}:
            raise ValueError(f"unknown interpolant sampler: {config.interp_time_sampler!r}")
        if config.interp_ref not in {"uniform", "canonical"}:
            raise ValueError(f"unknown interp_ref: {config.interp_ref!r}")
        if config.sigma_ref not in {"uniform", "canonical"}:
            raise ValueError(f"unknown sigma_ref: {config.sigma_ref!r}")
        if config.interp_is_correct and config.interp_time_sampler == "importance":
            raise ValueError("use interp_time_sampler='power' for a corrected power-law sampler")
        if config.sigma_is_correct and config.sigma_ref == "canonical" and config.sigma_max >= 1.0:
            raise ValueError("sigma_ref='canonical' needs sigma_max < 1")
        if config.meanflow_clock not in {"clamp", "add"}:
            raise ValueError(f"unknown meanflow_clock: {config.meanflow_clock!r}")
        if config.semigroup_weight < 0.0:
            raise ValueError("semigroup_weight must be non-negative")
        if config.time_weight_clip < 0.0:
            raise ValueError("time_weight_clip must be non-negative")
        if config.adaptive_weight_mode not in {"sample", "batch"}:
            raise ValueError(f"unknown adaptive_weight_mode: {config.adaptive_weight_mode!r}")
        if config.loss_reduction not in {"sum", "mean"}:
            raise ValueError(f"unknown loss_reduction: {config.loss_reduction!r}")
        if config.guidance_mode not in {"branch", "guided"}:
            raise ValueError(f"unknown guidance mode: {config.guidance_mode!r}")
        if config.cfg_rule not in CFG_RULES:
            raise ValueError(f"unknown cfg rule: {config.cfg_rule!r}; expected one of {CFG_RULES}")
        warn_if_perp_cnorm(config.cfg_rule, config.cfg_cnorm)
        if config.cfg_mode not in {"all", "first3"}:
            raise ValueError(f"unknown cfg mode: {config.cfg_mode!r}")
        if not 0.0 <= config.cfg_dropout_prob <= 1.0:
            raise ValueError("cfg_dropout_prob must lie in [0, 1]")
        if not 0.0 <= config.sigma_min < config.sigma_max <= 1.0:
            raise ValueError("need 0 <= sigma_min < sigma_max <= 1")
        if config.terminal_teacher_norm_weight < 0.0:
            raise ValueError("terminal_teacher_norm_weight must be non-negative")
        if config.terminal_student_norm_weight < 0.0:
            raise ValueError("terminal_student_norm_weight must be non-negative")
        if config.terminal_norm_max_batch < 0:
            raise ValueError("terminal_norm_max_batch must be non-negative")
        if config.terminal_invariance_weight < 0.0:
            raise ValueError("terminal_invariance_weight must be non-negative")
        if config.terminal_invariance_step <= 0.0:
            raise ValueError("terminal_invariance_step must be positive")
        if config.terminal_invariance_max_batch < 0:
            raise ValueError("terminal_invariance_max_batch must be non-negative")
        # terminal_invariance_weight > 0 is allowed with any objective: for
        # objectives other than scaled_meanflow the frozen teacher field at x0
        # is evaluated once more under no_grad (see __call__).
        self.config = config
        self._w_t: Optional[torch.Tensor] = None   # importance weight of t (None = off)
        self._w_s: Optional[torch.Tensor] = None   # importance weight of sigma (None = off)
        self._t: Optional[torch.Tensor] = None     # interpolant time of the current batch (diagnostics)

    def _psm(self, x: torch.Tensor) -> torch.Tensor:
        return _per_sample_mse(x, self.config.loss_reduction)

    _sq = _psm  # name used on the martin branch

    def _keep(self, **tensors: torch.Tensor) -> None:
        if self.collect_samples:
            for k, v in tensors.items():
                self._samples[k] = v.detach().float().reshape(-1)

    @staticmethod
    def _norm(x: torch.Tensor) -> torch.Tensor:
        return x.detach().float().flatten(1).norm(dim=1)

    def sample_interpolant_time(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        cfg = self.config
        if cfg.interp_time_sampler == "uniform":
            if cfg.interp_is_correct:
                # truncated like the canonical reference so p_ref/q stays finite
                return sample_time_uniform(batch_size, eps_train=cfg.eps_train, device=device, dtype=dtype)
            return torch.rand(batch_size, device=device, dtype=dtype)
        if cfg.interp_time_sampler == "canonical":
            return sample_time_canonical(batch_size, cfg.eps_train, device, dtype)
        if cfg.interp_time_sampler == "importance":
            return sample_time_importance(batch_size, cfg.importance_k, device, dtype)
        return sample_time_on_interval(
            batch_size, 0.0, 1.0 - cfg.eps_train, cfg.interp_time_sampler,
            power_k=cfg.importance_k, ln_mean=cfg.interp_logitnormal_mean, ln_std=cfg.interp_logitnormal_std,
            device=device, dtype=dtype,
        )

    def interp_time_weight(self, t: torch.Tensor) -> Optional[torch.Tensor]:
        """p_ref(t)/q(t) on [0, 1-eps_train), or None when the correction is off."""
        cfg = self.config
        if not cfg.interp_is_correct:
            return None
        return importance_weight(
            t, 0.0, 1.0 - cfg.eps_train, cfg.interp_time_sampler, cfg.interp_ref,
            clip=cfg.time_weight_clip, power_k=cfg.importance_k,
            ln_mean=cfg.interp_logitnormal_mean, ln_std=cfg.interp_logitnormal_std,
        )

    def sigma_weight(self, sigma: torch.Tensor) -> Optional[torch.Tensor]:
        """p_ref(sigma)/q(sigma) on [sigma_min, sigma_max), or None when off."""
        cfg = self.config
        if not cfg.sigma_is_correct:
            return None
        return importance_weight(
            sigma, cfg.sigma_min, cfg.sigma_max, cfg.sigma_sampler, cfg.sigma_ref,
            clip=cfg.time_weight_clip, power_k=cfg.sigma_power_k,
            ln_mean=cfg.sigma_logitnormal_mean, ln_std=cfg.sigma_logitnormal_std,
        )

    def _w_ts(self) -> Optional[torch.Tensor]:
        """Combined importance weight for terms that draw both t and sigma."""
        if self._w_t is None:
            return self._w_s
        if self._w_s is None:
            return self._w_t
        return self._w_t * self._w_s

    def _wmean(self, per: torch.Tensor, sample_weight: Optional[torch.Tensor], key: str,
               eps: float | torch.Tensor | None = None) -> torch.Tensor:
        """Adaptive-weighted mean + diagnostics (adaptive factor stats, ESS of the total weight)."""
        cfg = self.config
        e = cfg.adaptive_weight_eps if eps is None else eps
        if cfg.adaptive_weight_mode == "batch" and cfg.adaptive_weight_p > 0.0:
            e_b = e.mean() if isinstance(e, torch.Tensor) else e
            aw = (per.detach().mean() + e_b).pow(-cfg.adaptive_weight_p).expand_as(per)
        else:
            aw = _adaptive_factor(per, cfg.adaptive_weight_p, e)
        w = aw if sample_weight is None else aw * sample_weight.detach().to(aw.dtype)
        self._aux.update({f"{key}_aw_mean": aw.mean(), f"{key}_aw_max": aw.max()})
        self._aux[f"{key}_ess"] = weight_stats(w, "w")["w_ess"]
        return (per * w).mean()

    def sample_sigma(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        cfg = self.config
        if cfg.sigma_sampler in ("power", "logitnormal"):
            return sample_time_on_interval(
                batch_size, cfg.sigma_min, cfg.sigma_max, cfg.sigma_sampler,
                power_k=cfg.sigma_power_k, ln_mean=cfg.sigma_logitnormal_mean, ln_std=cfg.sigma_logitnormal_std,
                device=device, dtype=dtype,
            )
        u = torch.rand(batch_size, device=device, dtype=dtype)
        if cfg.sigma_sampler == "canonical":
            tau_lo = -math.log(1.0 - cfg.sigma_min)
            tau_hi = -math.log(1.0 - cfg.sigma_max)
            return 1.0 - torch.exp(-(tau_lo + (tau_hi - tau_lo) * u))
        return cfg.sigma_min + (cfg.sigma_max - cfg.sigma_min) * u

    def sample_semigroup_pair(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample ``(sigma, sigma_prime)`` for the PESD triangular integral.

        Eq. (73) integrates ``d sigma' d sigma`` over
        ``0 <= sigma' <= sigma <= 1``.  A normalized Monte Carlo estimator
        therefore samples outer ``sigma`` with density proportional to
        ``sigma`` and then samples ``sigma_prime | sigma`` uniformly on the
        lower interval.
        """
        cfg = self.config
        u = torch.rand(batch_size, device=device, dtype=dtype)
        lo2 = cfg.sigma_min ** 2
        hi2 = cfg.sigma_max ** 2
        sigma = torch.sqrt(lo2 + (hi2 - lo2) * u)
        sigma_prime = torch.rand_like(sigma) * sigma
        return sigma, sigma_prime

    def drop_labels(self, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        cfg = self.config
        mask = torch.rand(labels.shape, device=labels.device) < cfg.cfg_dropout_prob
        labels_in = labels.clone()
        labels_in[mask] = cfg.null_class
        return labels_in, mask

    def teacher_target_for_conditioning(
        self,
        teacher_fn: TeacherFn,
        x: torch.Tensor,
        labels: torch.Tensor,
        labels_in: torch.Tensor,
        drop_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate the frozen teacher for the already-sampled branch labels.

        A single ``labels_in`` / ``drop_mask`` pair is shared across the
        diagonal, off-diagonal, and terminal terms.  In branch mode, this is
        just the teacher evaluated at the same conditional/null label that the
        student sees.  In guided mode, non-dropped examples target the guided
        teacher while dropped examples target the null teacher.
        """
        cfg = self.config
        if cfg.guidance_mode == "branch":
            return teacher_fn(x, labels_in)

        null = torch.full_like(labels, cfg.null_class)
        if cfg.teacher_cfg_scale == 1.0 and cfg.cfg_dropout_prob <= 0.0:
            return teacher_fn(x, labels)

        # One teacher forward on the concatenated (cond, uncond) batch instead
        # of two on the same x. SiT is per-sample (adaLN conditioning, no
        # batch statistics), so the two halves are exactly the separate
        # results; this halves the launches and doubles the GEMM sizes.
        both = teacher_fn(torch.cat([x, x], dim=0), torch.cat([labels, null], dim=0))
        b_cond, b_uncond = both.chunk(2, dim=0)
        if cfg.teacher_cfg_scale == 1.0:
            return torch.where(
                drop_mask.view(-1, *([1] * (x.ndim - 1))),
                b_uncond,
                b_cond,
            )

        if cfg.cfg_match_uncond_norm:
            b_uncond = _match_uncond_norm(b_cond, b_uncond)
        if cfg.cfg_rule == "perp":
            guided = _perp_guided(b_cond, b_uncond, cfg.teacher_cfg_scale, cfg.cfg_cnorm)
        elif cfg.cfg_rule == "perp_matched":
            # normalise b_u to ||b_c|| first, then remove the parallel part: ||g - b_c|| <= s ||b_c||
            guided = perp_guided(b_cond, b_uncond, cfg.teacher_cfg_scale, match_uncond_norm_first=True)
        else:
            guided = _combine_cfg(b_cond, b_uncond, cfg.teacher_cfg_scale, cfg.cfg_mode)
        if cfg.cfg_match_guided_norm:
            # Direction-only guidance: preserve the conditional landing rate so
            # the sigma <-> tau clock calibration survives distillation.
            guided = _match_uncond_norm(b_cond, guided)
        return torch.where(
            drop_mask.view(-1, *([1] * (x.ndim - 1))),
            b_uncond,
            guided,
        )

    def diagonal_loss(
        self,
        student: nn.Module,
        teacher_fn: TeacherFn,
        x0: torch.Tensor,
        labels: torch.Tensor,
        labels_in: torch.Tensor,
        drop_mask: torch.Tensor,
        t: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, torch.Tensor]]:
        cfg = self.config
        target = self.teacher_target_for_conditioning(
            teacher_fn, x0, labels, labels_in, drop_mask
        )
        sigma0 = torch.zeros(x0.shape[0], device=x0.device, dtype=x0.dtype)
        # When the aux head is on, request the time-prediction logit from the
        # same diagonal forward (no extra pass). x0 is the interpolant at time
        # t, so the head is trained to recover t from the student's features.
        tp_loss: Optional[torch.Tensor] = None
        tp_metrics: Dict[str, torch.Tensor] = {}
        if cfg.time_pred_weight > 0.0:
            if t is None:
                raise ValueError("time_pred_weight > 0 requires the interpolant time t")
            pred, t_logit = student(x0, sigma0, labels_in, return_time_pred=True)
            tp_loss, tp_metrics = _time_pred_terms(
                t_logit, t, cfg.time_pred_target, cfg.eps_train
            )
        else:
            pred = student(x0, sigma0, labels_in)
        per = self._psm(pred - target.detach())
        self._b0_cache = (x0, target.detach())  # reused by scaled_meanflow_loss: same x0, labels, teacher call
        if self.collect_samples:
            self._keep(diag_raw=per, diag_target_norm=self._norm(target))
        loss = self._wmean(per, self._w_t, "diag")
        if t is not None:
            self._aux.update(bucket_stats(per, t, "diag_raw"))
            self._aux.update(bucket_stats(target.detach().float().flatten(1).norm(dim=1), t, "diag_target_norm"))
        metrics = {
            "diag_loss": loss.detach(),
            "diag_raw": per.detach().mean(),
            "cfg_drop_frac": drop_mask.float().mean().detach(),
            "diag_target_norm": target.detach().float().flatten(1).norm(dim=1).mean(),
            "diag_pred_norm": pred.detach().float().flatten(1).norm(dim=1).mean(),
        }
        metrics.update(tp_metrics)
        return loss, tp_loss, metrics

    def nonjvp_autocast(self, device: torch.device):
        """bf16 autocast for the student passes that are not differentiated in
        forward mode (diagonal, terminal, terminal-norm, semigroup), when
        ``config.student_bf16_nonjvp`` is set; a no-op context otherwise.
        JVP passes must not be wrapped in it."""
        if not self.config.student_bf16_nonjvp:
            return contextlib.nullcontext()
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16)

    def terminal_loss(
        self,
        student: nn.Module,
        x_data: torch.Tensor,
        labels_in: torch.Tensor,
    ) -> torch.Tensor:
        sigma1 = torch.ones(x_data.shape[0], device=x_data.device, dtype=x_data.dtype)
        terminal = flow_map(student, x_data, sigma1, labels_in)
        per = self._psm(terminal - x_data)
        self._keep(anchor_raw=per)
        return per.mean()

    def terminal_norm_inputs(
        self,
        z: torch.Tensor,
        labels: torch.Tensor,
        labels_in: torch.Tensor,
        drop_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Select the noise inputs used by terminal norm regularizers."""
        cfg = self.config
        if cfg.terminal_norm_max_batch <= 0 or z.shape[0] <= cfg.terminal_norm_max_batch:
            return z, labels, labels_in, drop_mask
        idx = torch.randperm(z.shape[0], device=z.device)[:cfg.terminal_norm_max_batch]
        return z[idx], labels[idx], labels_in[idx], drop_mask[idx]

    def terminal_teacher_norm_loss(
        self,
        student: nn.Module,
        teacher_grad_fn: TeacherFn,
        z: torch.Tensor,
        labels: torch.Tensor,
        labels_in: torch.Tensor,
        drop_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Penalize teacher field norm at the student's terminal endpoint.

        Unlike the ordinary teacher target path, ``teacher_grad_fn`` must not
        wrap its forward pass in ``torch.no_grad()``. Teacher parameters remain
        frozen, but gradients need to flow through the teacher input back into
        ``X_1_student(z)``.
        """
        z, labels, labels_in, drop_mask = self.terminal_norm_inputs(
            z, labels, labels_in, drop_mask
        )
        sigma1 = torch.ones(z.shape[0], device=z.device, dtype=z.dtype)
        terminal = flow_map(student, z, sigma1, labels_in)
        field = self.teacher_target_for_conditioning(
            teacher_grad_fn, terminal, labels, labels_in, drop_mask
        )
        per = self._psm(field)
        loss = _weighted_mean(per, self.config.adaptive_weight_p, self.config.adaptive_weight_eps)
        self._aux["terminal_teacher_norm_raw"] = per.detach().mean()
        if self.collect_samples:
            self._keep(endpoint_field_norm=self._norm(field))
        norm = field.detach().float().flatten(1).norm(dim=1).mean()
        return loss, norm

    def terminal_student_norm_loss(
        self,
        student: nn.Module,
        z: torch.Tensor,
        labels_in: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Penalize student diagonal field norm at its terminal endpoint."""
        cfg = self.config
        if cfg.terminal_norm_max_batch > 0 and z.shape[0] > cfg.terminal_norm_max_batch:
            idx = torch.randperm(z.shape[0], device=z.device)[:cfg.terminal_norm_max_batch]
            z = z[idx]
            labels_in = labels_in[idx]
        sigma1 = torch.ones(z.shape[0], device=z.device, dtype=z.dtype)
        sigma0 = torch.zeros(z.shape[0], device=z.device, dtype=z.dtype)
        terminal = flow_map(student, z, sigma1, labels_in)
        field = student(terminal, sigma0, labels_in)
        per = self._psm(field)
        loss = _weighted_mean(per, cfg.adaptive_weight_p, cfg.adaptive_weight_eps)
        norm = field.detach().float().flatten(1).norm(dim=1).mean()
        return loss, norm

    def lagrangian_loss(
        self,
        student: nn.Module,
        teacher_fn: TeacherFn,
        x0: torch.Tensor,
        labels: torch.Tensor,
        labels_in: torch.Tensor,
        drop_mask: torch.Tensor,
        sigma: torch.Tensor,
    ) -> torch.Tensor:
        cfg = self.config
        jvp_ctx = (torch.autocast(device_type=x0.device.type, dtype=torch.bfloat16)
                   if cfg.student_bf16_jvp else contextlib.nullcontext())
        with jvp_ctx:
            x_sigma, d_x = sigma_derivative(student, x0, sigma, labels_in)
        x_sigma, d_x = x_sigma.float(), d_x.float()
        with torch.no_grad():
            target = self.teacher_target_for_conditioning(
                teacher_fn, x_sigma.detach(), labels, labels_in, drop_mask
            )
        pred = (1.0 - sigma).view(-1, *([1] * (x0.ndim - 1))) * d_x
        residual = pred - target
        per = self._psm(residual)
        self._distill_diagnostics(per, sigma, target, pred)
        return self._wmean(per, self._w_ts(), "distill")

    def _distill_diagnostics(self, per, sigma, target, pred) -> None:
        """Raw per-sample error by sigma and t bucket, target/pred magnitude by sigma bucket."""
        if self.collect_samples:
            self._keep(distill_raw=per, distill_sigma=sigma, distill_target_norm=self._norm(target))
        self._aux.update(_raw_sigma_stats(per, sigma, "distill"))
        if self._t is not None:
            self._aux.update(bucket_stats(per, self._t, "distill_raw", tag="t"))
        tn = target.detach().float().flatten(1).norm(dim=1)
        pn = pred.detach().float().flatten(1).norm(dim=1)
        self._aux.update(bucket_stats(tn, sigma, "distill_target_norm", buckets=_SIGMA_BUCKETS, tag="s"))
        self._aux.update(bucket_stats(pn, sigma, "distill_pred_norm", buckets=_SIGMA_BUCKETS, tag="s"))
        self._aux["distill_residual_ratio"] = (per.detach().sqrt() / tn.clamp_min(1e-8)).mean()
        self._aux["distill_raw_max"] = per.detach().max()

    def shifted_lagrangian_loss(
        self,
        student: nn.Module,
        teacher_fn: TeacherFn,
        x0: torch.Tensor,
        labels: torch.Tensor,
        labels_in: torch.Tensor,
        drop_mask: torch.Tensor,
        sigma: torch.Tensor,
    ) -> torch.Tensor:
        """Shifted Lagrangian: regress ``v`` directly on a fully stopped target.

        The Lagrangian equation ``k d/dsigma X_sigma = b(X_sigma)`` with
        ``X_sigma = x + sigma v`` and ``k = 1 - sigma`` reads
        ``k (v + sigma d_sigma v) = b(x + sigma v)``.  Solving for the ``v``
        that is not multiplied by sigma gives the regression

            v = sg[ b(x + sigma v) + sigma (v - k d_sigma v) ].

        The forward residual equals the one of ``lagrangian_loss``; the
        gradient flows only through the prediction ``v`` (nothing through the
        sigma-JVP or through the point where the teacher is evaluated), so no
        double backward is needed (about 1.6x faster per step).  As
        sigma -> 1 the residual becomes ``-b(X_1)``: a semi-gradient
        equilibrium loss on the landing point.
        """
        cfg = self.config
        pred = student(x0, sigma, labels_in)
        with torch.no_grad():
            sig = broadcast_time(sigma, x0)
            b_end = self.teacher_target_for_conditioning(
                teacher_fn, x0 + sig * pred.detach(), labels, labels_in, drop_mask
            )
            _, jvp_sigma = velocity_jvp(
                student, x0, sigma, labels_in,
                torch.zeros_like(x0), torch.ones_like(sigma),
            )
            target = b_end + sig * (pred.detach() - (1.0 - sig) * jvp_sigma)
        per = self._psm(pred - target)
        if self.collect_samples:
            self._keep(distill_raw=per, distill_sigma=sigma, distill_target_norm=self._norm(target))
        return _weighted_mean(per, cfg.adaptive_weight_p, cfg.adaptive_weight_eps, self._w_s)

    def shifted_meanflow_loss(
        self,
        student: nn.Module,
        teacher_fn: TeacherFn,
        x0: torch.Tensor,
        labels: torch.Tensor,
        labels_in: torch.Tensor,
        drop_mask: torch.Tensor,
        sigma: torch.Tensor,
    ) -> torch.Tensor:
        """Shifted MeanFlow: the compact Eulerian identity solved for ``v``.

        ``k (v + sigma d_sigma v) = b + sigma (D_x v) b`` with ``k = 1 - sigma``
        rearranged as

            v = sg[ b(x) + sigma (v + (D_x v) b(x) - k d_sigma v) ].

        Same forward residual as ``scaled_meanflow_loss`` (= k times the
        divided MeanFlow residual), gradient through ``v`` instead of ``k v``,
        nothing divided by ``k``.
        """
        cfg = self.config
        pred = student(x0, sigma, labels_in)
        sig = broadcast_time(sigma, x0)
        with torch.no_grad():
            cache = getattr(self, "_b0_cache", None)
            if cache is not None and cache[0] is x0:
                b0 = cache[1]   # the diagonal term already evaluated the guided teacher at this x0
            else:
                b0 = self.teacher_target_for_conditioning(
                    teacher_fn, x0, labels, labels_in, drop_mask
                )
            if cfg.mf_single_jvp:
                # one JVP with tangents (s b0, -k s): D v . (s b0, -k s) = s (D_x v) b0 - k s d_s v
                _, jvp_both = velocity_jvp(
                    student, x0, sigma, labels_in, sig * b0, -(1.0 - sigma) * sigma
                )
                target = b0 + sig * pred.detach() + jvp_both
            else:
                _, jvp_spatial = velocity_jvp(
                    student, x0, sigma, labels_in, b0, torch.zeros_like(sigma)
                )
                _, jvp_sigma = velocity_jvp(
                    student, x0, sigma, labels_in,
                    torch.zeros_like(x0), torch.ones_like(sigma),
                )
                target = b0 + sig * (pred.detach() + jvp_spatial - (1.0 - sig) * jvp_sigma)
        per = self._psm(pred - target)
        if self.collect_samples:
            self._keep(distill_raw=per, distill_sigma=sigma, distill_target_norm=self._norm(target))
        self._meanflow_b0 = b0
        return _weighted_mean(per, cfg.adaptive_weight_p, cfg.adaptive_weight_eps, self._w_s)

    def eulerian_loss(
        self,
        student: nn.Module,
        teacher_fn: TeacherFn,
        x0: torch.Tensor,
        labels: torch.Tensor,
        labels_in: torch.Tensor,
        drop_mask: torch.Tensor,
        sigma: torch.Tensor,
    ) -> torch.Tensor:
        cfg = self.config
        b0 = self.teacher_target_for_conditioning(
            teacher_fn, x0, labels, labels_in, drop_mask
        )
        _, d_sigma = sigma_derivative(student, x0, sigma, labels_in)
        _, pushed = spatial_jvp(student, x0, sigma, labels_in, b0.detach())
        pred = (1.0 - sigma).view(-1, *([1] * (x0.ndim - 1))) * d_sigma
        residual = pred - pushed
        per = self._psm(residual)
        self._distill_diagnostics(per, sigma, pushed, pred)
        return self._wmean(per, self._w_ts(), "distill")

    def meanflow_loss(
        self,
        student: nn.Module,
        teacher_fn: TeacherFn,
        x0: torch.Tensor,
        labels: torch.Tensor,
        labels_in: torch.Tensor,
        drop_mask: torch.Tensor,
        sigma: torch.Tensor,
    ) -> torch.Tensor:
        """Historical divided MeanFlow target under the exponential clock.

        This objective is retained for checkpoint and ablation compatibility.
        New infinite-time runs should use scaled_meanflow, which has the same
        fixed point and p=1 semi-gradient without constructing 1/(1-sigma).
        """
        cfg = self.config
        pred = student(x0, sigma, labels_in)
        with torch.no_grad():
            b0 = self.teacher_target_for_conditioning(
                teacher_fn, x0, labels, labels_in, drop_mask
            )
            zero_x = torch.zeros_like(x0)
            zero_sigma = torch.zeros_like(sigma)
            one_sigma = torch.ones_like(sigma)
            _, jvp_spatial = velocity_jvp(
                student, x0, sigma, labels_in, b0, zero_sigma
            )
            _, jvp_sigma = velocity_jvp(
                student, x0, sigma, labels_in, zero_x, one_sigma
            )
            sig = broadcast_time(sigma, x0)
            # bound the bootstrap gain sigma/(1-sigma): clamp -> max(1-sigma, eps); add -> (1-sigma) + eps
            if cfg.meanflow_clock == "add":
                clock = (1.0 - sig) + cfg.meanflow_eps
            else:
                clock = (1.0 - sig).clamp_min(cfg.meanflow_eps)
            target = (b0 + sig * jvp_spatial) / clock - sig * jvp_sigma
        per = self._psm(pred - target)
        self._distill_diagnostics(per, sigma, target, pred)
        self._aux.update({
            "meanflow_target_abs_max": target.detach().float().abs().amax(),
            "meanflow_jvp_abs_max": torch.maximum(
                jvp_spatial.detach().float().abs().amax(), jvp_sigma.detach().float().abs().amax()
            ),
            "meanflow_clock_min": clock.detach().float().amin(),
        })
        return self._wmean(per, self._w_ts(), "distill")

    def scaled_meanflow_loss(
        self,
        student: nn.Module,
        teacher_fn: TeacherFn,
        x0: torch.Tensor,
        labels: torch.Tensor,
        labels_in: torch.Tensor,
        drop_mask: torch.Tensor,
        sigma: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pole-free stopped MeanFlow under the exponential clock.

        For ``X_sigma = x + sigma v_sigma`` and ``k = 1 - sigma``, expanding
        the compact Eulerian equation gives

            k (v + sigma partial_sigma v) = b + sigma (D_x v) b.

        Instead of solving this equation for ``v`` and creating a ``1/k``
        target, regress the scaled prediction onto the fully stopped remainder:

            k v = sg[b + sigma (D_x v)b - k sigma partial_sigma v].

        The forward residual is exactly the EESD residual, but gradients flow
        only through ``k v``.  Both velocity JVPs and the guided teacher are
        part of the stopped target.  For adaptive p=1, scaling epsilon by
        ``k^2`` makes this semi-gradient exactly identical to the divided
        MeanFlow update while never forming the pole.

        The frozen guided ``b`` is returned for the terminal-invariance term,
        ensuring it uses the same standard/perp/cnorm CFG construction without
        an additional teacher forward.
        """
        cfg = self.config
        velocity = student(x0, sigma, labels_in)
        sig = broadcast_time(sigma, x0)
        clock = 1.0 - sig
        pred_scaled = clock * velocity

        with torch.no_grad():
            cache = getattr(self, "_b0_cache", None)
            if cache is not None and cache[0] is x0:
                b0 = cache[1]   # the diagonal term already evaluated the guided teacher at this x0
            else:
                b0 = self.teacher_target_for_conditioning(
                    teacher_fn, x0, labels, labels_in, drop_mask
                )
            if cfg.mf_single_jvp:
                # one JVP with tangents (s b0, -k s): D v . (s b0, -k s) = s (D_x v) b0 - k s d_s v
                _, jvp_both = velocity_jvp(
                    student, x0, sigma, labels_in, sig * b0, -(1.0 - sigma) * sigma
                )
                jvp_spatial, jvp_sigma = jvp_both, jvp_both  # diagnostics only
                target_scaled = b0 + jvp_both
            else:
                zero_x = torch.zeros_like(x0)
                zero_sigma = torch.zeros_like(sigma)
                one_sigma = torch.ones_like(sigma)
                # (D_x v_sigma) . b0  -- spatial JVP only.
                _, jvp_spatial = velocity_jvp(
                    student, x0, sigma, labels_in, b0, zero_sigma
                )
                # partial_sigma v_sigma  -- sigma JVP only.
                _, jvp_sigma = velocity_jvp(
                    student, x0, sigma, labels_in, zero_x, one_sigma
                )
                target_scaled = (
                    b0 + sig * jvp_spatial - clock * sig * jvp_sigma
                )

        per = self._psm(pred_scaled - target_scaled)
        self._distill_diagnostics(per, sigma, target_scaled, pred_scaled)
        clock_per_sample = (1.0 - sigma).float()
        scaled_eps = (
            cfg.adaptive_weight_eps * clock_per_sample.square()
        ).clamp_min(torch.finfo(per.dtype).tiny)
        self._aux.update({
            "meanflow_target_abs_max": target_scaled.detach().float().abs().amax(),
            "meanflow_target_norm": (
                target_scaled.detach().float().flatten(1).norm(dim=1).mean()
            ),
            "meanflow_jvp_abs_max": torch.maximum(
                jvp_spatial.detach().float().abs().amax(),
                jvp_sigma.detach().float().abs().amax(),
            ),
            "meanflow_clock_min": clock_per_sample.detach().amin(),
        })
        loss = self._wmean(per, self._w_ts(), "distill", eps=scaled_eps)
        return loss, b0.detach()

    def selfcorr_loss(
        self,
        student: nn.Module,
        z: torch.Tensor,
        labels_in: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Self-correction endpoint term (see ``Stage2LossConfig.selfcorr_weight``).

            p1  = X_1(z)                                   [no grad]  the one-shot sample
            C   = X_1^{k}(z) | X_1(X_s^{k}(z))             [no grad]  the reference chain
            L   = || X_1(p1) - sg C ||^2                              only the outer X_1 has grad

        Returns ``(loss, residual_norm, ratio)`` where ``ratio = ||X_1(p1) - C|| / ||p1 - C||``.
        A ratio of 1 means X_1 is idempotent on its own output and the second call did nothing;
        below 1 means it moved the one-shot toward the reference. This is the term's only honest
        progress signal, so it is logged every step.
        """
        cfg = self.config
        if cfg.selfcorr_max_batch > 0 and z.shape[0] > cfg.selfcorr_max_batch:
            idx = torch.randperm(z.shape[0], device=z.device)[:cfg.selfcorr_max_batch]
            z, labels_in = z[idx], labels_in[idx]
        sigma1 = torch.ones(z.shape[0], device=z.device, dtype=z.dtype)
        with torch.no_grad():
            p1 = flow_map(student, z, sigma1, labels_in)
            chain = p1
            steps = cfg.selfcorr_chain_steps
            if cfg.selfcorr_chain_sigma >= 1.0:
                # each chain jump IS X_1, so the first one is p1 and we only need steps-1 more
                for _ in range(max(0, steps - 1)):
                    chain = flow_map(student, chain, sigma1, labels_in)
            else:
                sigma_s = torch.full_like(sigma1, cfg.selfcorr_chain_sigma)
                chain = z
                for _ in range(steps):
                    chain = flow_map(student, chain, sigma_s, labels_in)
            if cfg.selfcorr_chain_final_x1:
                chain = flow_map(student, chain, sigma1, labels_in)
            gap = self._psm(p1 - chain)
        residual = flow_map(student, p1, sigma1, labels_in) - chain
        per = self._psm(residual)
        loss = _weighted_mean(per, cfg.adaptive_weight_p, cfg.adaptive_weight_eps)
        self._aux["selfcorr_raw"] = per.detach().mean()
        self._aux["selfcorr_gap_raw"] = gap.mean()
        if self.collect_samples:
            self._keep(selfcorr_raw=per, selfcorr_residual_norm=self._norm(residual))
        norm = residual.detach().float().flatten(1).norm(dim=1).mean()
        ratio = (per.detach().sqrt() / gap.sqrt().clamp_min(1e-12)).mean()
        return loss, norm, ratio

    def terminal_invariance_loss(
        self,
        student: nn.Module,
        x0: torch.Tensor,
        b0: torch.Tensor,
        labels_in: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Enforce ``P(x) = P(Phi_delta(x))`` with a stopped endpoint target.

        The short teacher flow is approximated by ``x + delta*b(x)``.  Only
        the first endpoint receives gradients, preventing two student branches
        from chasing one another.
        """
        cfg = self.config
        if (
            cfg.terminal_invariance_max_batch > 0
            and x0.shape[0] > cfg.terminal_invariance_max_batch
        ):
            idx = torch.randperm(x0.shape[0], device=x0.device)[
                :cfg.terminal_invariance_max_batch
            ]
            x0 = x0[idx]
            b0 = b0[idx]
            labels_in = labels_in[idx]

        sigma1 = torch.ones(x0.shape[0], device=x0.device, dtype=x0.dtype)
        endpoint = flow_map(student, x0, sigma1, labels_in)
        with torch.no_grad():
            stepped = x0 + cfg.terminal_invariance_step * b0
            target_endpoint = flow_map(student, stepped, sigma1, labels_in)
        per = self._psm(endpoint - target_endpoint)
        loss = _weighted_mean(
            per, cfg.adaptive_weight_p, cfg.adaptive_weight_eps
        )
        endpoint_delta = per.detach().clamp_min(0.0).sqrt().mean()
        return loss, endpoint_delta

    def consistency_loss(
        self,
        student: nn.Module,
        target_student: nn.Module,
        teacher_fn: TeacherFn,
        x0: torch.Tensor,
        labels: torch.Tensor,
        labels_in: torch.Tensor,
        drop_mask: torch.Tensor,
        sigma: torch.Tensor,
    ) -> torch.Tensor:
        """Consistency along the equilibrium flow with a teacher-driven small step.

        Fixed point (h -> 0): X(s (+) delta, x) = X(s, x + h b(x)) i.e. the Eulerian equation with the
        clock, and at s (+) delta = 1 the projector invariance P(x) = P(x + h b(x)). Bootstraps from the
        EMA copy ``target_student`` (stop-grad), as in consistency distillation.
        """
        cfg = self.config
        delta = cfg.cd_delta
        h = -math.log(1.0 - delta)
        sigma_long = compactified_add(sigma, torch.full_like(sigma, delta)).clamp(max=1.0)
        x_long = flow_map(student, x0, sigma_long, labels_in)
        with torch.no_grad():
            x_step = x0
            for _ in range(max(1, cfg.cd_substeps)):
                b = self.teacher_target_for_conditioning(teacher_fn, x_step, labels, labels_in, drop_mask)
                x_step = x_step + (h / max(1, cfg.cd_substeps)) * b
            target = flow_map(target_student, x_step, sigma, labels_in)
        per = self._psm(x_long - target)
        self._distill_diagnostics(per, sigma, target - x0, x_long - x0)
        return self._wmean(per, self._w_ts(), "distill")

    def semigroup_loss(
        self,
        student: nn.Module,
        x0: torch.Tensor,
        labels_in: torch.Tensor,
        sigma: torch.Tensor,
        sigma_prime: torch.Tensor,
    ) -> torch.Tensor:
        cfg = self.config
        sigma_minus = compactified_sub(sigma, sigma_prime).clamp(0.0, 1.0)
        x_sigma = flow_map(student, x0, sigma, labels_in)
        with torch.no_grad():
            x_prime = flow_map(student, x0, sigma_prime, labels_in)
            target = flow_map(student, x_prime, sigma_minus, labels_in)
        per = self._psm(x_sigma - target.detach())
        if self.collect_samples:
            self._keep(semi_raw=per, semi_sigma=sigma, semi_step_norm=self._norm(x_sigma - x0))
        self._aux.update(_raw_sigma_stats(per, sigma, "semi"))
        return self._wmean(per, self._w_t, "semi")

    def __call__(
        self,
        student: nn.Module,
        teacher_fn: TeacherFn,
        x_data: torch.Tensor,
        labels: torch.Tensor,
        *,
        z: Optional[torch.Tensor] = None,
        teacher_grad_fn: Optional[TeacherFn] = None,
        target_student: Optional[nn.Module] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        cfg = self.config
        self._aux = {}
        self._b0_cache = None
        self._samples = {}
        if z is None:
            z = torch.randn_like(x_data)
        t = self.sample_interpolant_time(x_data.shape[0], x_data.device, x_data.dtype)
        self._keep(t=t)
        x0 = canonical_interpolant(z, x_data, t)
        sigma = self.sample_sigma(x_data.shape[0], x_data.device, x_data.dtype)
        labels_in, drop_mask = self.drop_labels(labels)
        self._t = t
        self._w_t = self.interp_time_weight(t)
        self._w_s = self.sigma_weight(sigma)
        if self._w_t is not None:
            self._aux.update(weight_stats(self._w_t, "time_weight_t"))
        if self._w_s is not None:
            self._aux.update(weight_stats(self._w_s, "time_weight_s"))

        def nonjvp_ctx():
            return self.nonjvp_autocast(x_data.device)

        with nonjvp_ctx():
            diag, tp_loss, metrics = self.diagonal_loss(
                student, teacher_fn, x0, labels, labels_in, drop_mask, t
            )
            term = self.terminal_loss(student, x_data, labels_in)
        zero = term.detach().new_zeros(())
        teacher_norm_term = zero
        teacher_endpoint_norm = zero
        student_norm_term = zero
        student_endpoint_norm = zero
        terminal_invariance_term = zero
        terminal_invariance_endpoint_delta = zero
        selfcorr_term = zero
        selfcorr_norm = zero
        selfcorr_ratio = zero

        if cfg.terminal_teacher_norm_weight > 0.0:
            if teacher_grad_fn is None:
                raise ValueError(
                    "terminal_teacher_norm_weight > 0 requires teacher_grad_fn"
                )
            with nonjvp_ctx():
                teacher_norm_term, teacher_endpoint_norm = self.terminal_teacher_norm_loss(
                    student, teacher_grad_fn, z, labels, labels_in, drop_mask
                )

        if cfg.terminal_student_norm_weight > 0.0:
            with nonjvp_ctx():
                student_norm_term, student_endpoint_norm = self.terminal_student_norm_loss(
                    student, z, labels_in
                )

        if cfg.selfcorr_weight > 0.0:
            with nonjvp_ctx():
                selfcorr_term, selfcorr_norm, selfcorr_ratio = self.selfcorr_loss(
                    student, z, labels_in
                )

        if cfg.objective == "lagrangian":
            distill = self.lagrangian_loss(
                student, teacher_fn, x0, labels, labels_in, drop_mask, sigma
            )
        elif cfg.objective == "eulerian":
            distill = self.eulerian_loss(
                student, teacher_fn, x0, labels, labels_in, drop_mask, sigma
            )
        elif cfg.objective == "meanflow":
            distill = self.meanflow_loss(
                student, teacher_fn, x0, labels, labels_in, drop_mask, sigma
            )
        elif cfg.objective == "shifted_lagrangian":
            distill = self.shifted_lagrangian_loss(
                student, teacher_fn, x0, labels, labels_in, drop_mask, sigma
            )
        elif cfg.objective == "shifted_meanflow":
            distill = self.shifted_meanflow_loss(
                student, teacher_fn, x0, labels, labels_in, drop_mask, sigma
            )
        elif cfg.objective == "scaled_meanflow":
            distill, meanflow_b0 = self.scaled_meanflow_loss(
                student, teacher_fn, x0, labels, labels_in, drop_mask, sigma
            )
            if cfg.terminal_invariance_weight > 0.0:
                (
                    terminal_invariance_term,
                    terminal_invariance_endpoint_delta,
                ) = self.terminal_invariance_loss(
                    student, x0, meanflow_b0, labels_in
                )
        elif cfg.objective == "consistency":
            if target_student is None:
                raise ValueError("objective 'consistency' requires target_student (EMA copy)")
            distill = self.consistency_loss(
                student, target_student, teacher_fn, x0, labels, labels_in, drop_mask, sigma
            )
        else:
            sigma, sigma_prime = self.sample_semigroup_pair(
                x_data.shape[0], x_data.device, x_data.dtype
            )
            distill = self.semigroup_loss(student, x0, labels_in, sigma, sigma_prime)
            metrics["sigma_prime_mean"] = sigma_prime.detach().mean()

        semi_add = zero
        if cfg.semigroup_weight > 0.0 and cfg.objective != "semigroup":
            sg, sgp = self.sample_semigroup_pair(x_data.shape[0], x_data.device, x_data.dtype)
            with nonjvp_ctx():
                semi_add = self.semigroup_loss(student, x0, labels_in, sg, sgp)
            metrics["semigroup_add_loss"] = semi_add.detach()

        if cfg.terminal_invariance_weight > 0.0 and cfg.objective != "scaled_meanflow":
            with torch.no_grad():
                b0_inv = self.teacher_target_for_conditioning(
                    teacher_fn, x0, labels, labels_in, drop_mask
                )
            (
                terminal_invariance_term,
                terminal_invariance_endpoint_delta,
            ) = self.terminal_invariance_loss(student, x0, b0_inv, labels_in)

        total = (
            cfg.diag_weight * diag
            + cfg.distill_weight * distill
            + cfg.semigroup_weight * semi_add
            + cfg.terminal_weight * term
            + cfg.terminal_teacher_norm_weight * teacher_norm_term
            + cfg.terminal_student_norm_weight * student_norm_term
            + cfg.terminal_invariance_weight * terminal_invariance_term
            + cfg.selfcorr_weight * selfcorr_term
        )
        if tp_loss is not None:
            total = total + cfg.time_pred_weight * tp_loss
        metrics.update({
            "distill_loss": distill.detach(),
            "terminal_loss": term.detach(),
            "terminal_teacher_norm_loss": teacher_norm_term.detach(),
            "terminal_teacher_endpoint_norm": teacher_endpoint_norm.detach(),
            "terminal_student_norm_loss": student_norm_term.detach(),
            "terminal_student_endpoint_norm": student_endpoint_norm.detach(),
            "terminal_invariance_loss": terminal_invariance_term.detach(),
            "terminal_invariance_endpoint_delta": (
                terminal_invariance_endpoint_delta.detach()
            ),
            "selfcorr_loss": selfcorr_term.detach(),
            "selfcorr_residual_norm": selfcorr_norm.detach(),
            "selfcorr_ratio": selfcorr_ratio.detach(),
            "total_loss": total.detach(),
            "sigma_mean": sigma.detach().mean(),
            "interp_t_mean": t.detach().mean(),
        })
        metrics.update(self._aux)
        return total, metrics
