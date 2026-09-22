"""Forward-Euler sampler for the autonomous equilibrium field.

After Stage-1 fine-tuning, ``b_hat: R^d -> R^d`` is a time-independent
vector field whose fixed points are (approximately) the data manifold.
Sample generation is forward integration of the autonomous ODE

    dot{x}_s = b_hat(x_s, y),   x_0 ~ N(0, I),   s in [0, S_stop],

with ``S_stop = log(1 / eps_stop)`` under the exponential clock (paper
Eq. 42--44). For a step budget ``K``, the natural grid is uniform in
autonomous time ``s``, which corresponds to a *geometric* grid in the
compactified time ``sigma`` -- each Euler step closes a fixed
multiplicative fraction of the remaining gap to the manifold.

This is the right sampler for Stage 1. Stage 2 will use the compactified
flow map and its multi-step semigroup composition (sigma_eff =
1 - (1 - sigma)^K), which gives one-step generation at sigma=1 and a
clean test-time-compute knob.
"""
from __future__ import annotations

import math
from contextlib import nullcontext as _nullcontext
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import torch

from .cfg_rules import CFG_RULES, warn_if_perp_cnorm
from .cfg_rules import combine_cfg as combine_cfg_rule


ModelFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def as_field_fn(model_fn: ModelFn, parameterization: str = "velocity") -> ModelFn:
    """Wrap a raw model closure into the autonomous field ``b(x)``.

    The sampler integrates ``dot{x}_s = b(x)``. Depending on how the model
    was trained, its raw output is interpreted differently:

      * ``"velocity"``: the network directly outputs the field ``b``, so the
        wrapper is the identity.
      * ``"denoiser"``: the network outputs the denoiser ``D`` (x-prediction),
        and the field is reconstructed as ``b(x) = D(x) - x``. The fixed-point
        set is ``D(x) = x``.

    Returning the field (rather than handling the parameterization inside
    :func:`sample_euler`) keeps all CFG / momentum logic unchanged: classifier
    -free guidance acts on ``b``, and the ``-x`` offset cancels in the
    conditional/unconditional difference, so ``b_u + s (b_c - b_u)`` is exactly
    ``(D_u - x) + s (D_c - D_u)``.
    """
    if parameterization == "velocity":
        return model_fn
    if parameterization == "denoiser":
        def field_fn(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return model_fn(x, y) - x
        return field_fn
    raise ValueError(f"Unknown parameterization: {parameterization!r}")


@dataclass
class SamplerDiagnostics:
    """Per-step bookkeeping for convergence inspection."""
    velocity_norms: List[float] = field(default_factory=list)
    latent_norms: List[float] = field(default_factory=list)

    def append(self, velocity: torch.Tensor, latent: torch.Tensor) -> None:
        # Mean L2 norm across the batch, in fp32 to avoid bf16 overflow.
        self.velocity_norms.append(
            float(velocity.float().flatten(1).norm(dim=1).mean().item())
        )
        self.latent_norms.append(
            float(latent.float().flatten(1).norm(dim=1).mean().item())
        )


def stop_time_from_eps(eps_stop: float = 1e-3) -> float:
    """``S_stop = log(1 / eps_stop)`` under the canonical exponential clock."""
    if eps_stop <= 0.0 or eps_stop >= 1.0:
        raise ValueError(f"eps_stop must lie in (0, 1); got {eps_stop}.")
    return -math.log(eps_stop)


@torch.no_grad()
def sample_euler(
    model_fn: ModelFn,
    *,
    shape,
    class_labels: torch.Tensor,
    num_steps: int = 250,
    stop_time: Optional[float] = None,
    eps_stop: float = 1e-3,
    cfg_scale: float = 1.0,
    cfg_mode: str = "all",
    cfg_schedule: str = "constant",
    cfg_schedule_power: float = 1.0,
    cfg_early_frac: float = 0.5,
    cfg_match_uncond_norm: bool = False,
    cfg_match_guided_norm: bool = False,
    cfg_rule: str = "standard",
    cfg_cnorm: bool = False,
    null_class: int = 1000,
    momentum: float = 0.0,
    nesterov: bool = False,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
    autocast_dtype: Optional[torch.dtype] = None,
    generator: Optional[torch.Generator] = None,
    diagnostics: Optional[SamplerDiagnostics] = None,
) -> torch.Tensor:
    """Forward-Euler integrate the autonomous ODE from Gaussian noise.

    Args:
        model_fn: Closure ``(z, y) -> velocity`` evaluating the autonomous
            field. The training script's ``autonomous_forward`` is the
            canonical builder; it handles the (irrelevant) time argument
            under both ``strict`` and ``zero`` time-conditioning modes.
        shape: Tuple ``(B, C, H, W)`` for the latent batch.
        class_labels: ``(B,)`` long tensor on the same device as ``shape``.
        num_steps: Number of Euler steps ``K``.
        stop_time: Autonomous-time horizon ``S_stop``. If ``None``, derived
            from ``eps_stop`` as ``log(1/eps_stop)``.
        eps_stop: Residual to the manifold at the integration terminus.
            Recommended ``eps_stop >= eps_train`` to stay in-distribution.
        cfg_scale: Classifier-free guidance scale. ``1.0`` disables it
            (one model eval/step). For ``s != 1`` the field becomes
            ``b_u + s (b_c - b_u)`` where ``b_c = model_fn(x, y)`` and
            ``b_u = model_fn(x, null_class)``; this doubles the per-step
            NFE.
        cfg_mode: Which channels to guide. ``"all"`` is standard CFG over
            every latent channel. ``"first3"`` matches the released SiT/DiT
            convention of guiding channels 0:3 and leaving the remaining
            channel conditional.
        cfg_schedule: ``"constant"`` uses ``cfg_scale`` at every step.
            ``"anneal_to_one"`` uses
            ``1 + (cfg_scale - 1) * (1 - t(s))**cfg_schedule_power`` so
            guidance is strongest near noise and decays to 1 near the
            fixed-point/data end. ``"early"`` applies full ``cfg_scale`` for
            the first ``cfg_early_frac`` of the autonomous-time horizon and
            then turns guidance off exactly (scale 1) for the rest. This is
            the basin-selection schedule: guide the particle into the right
            class basin during early transport, then let the unguided
            equilibrium field land cleanly on the manifold (where the CFG
            residual is mostly branch-estimation error). On the toy circle
            this was the best-FID-analog EqM schedule.
        cfg_schedule_power: Exponent for ``"anneal_to_one"``.
        cfg_early_frac: For ``cfg_schedule="early"``, the fraction of the
            autonomous-time horizon ``S_stop`` over which full guidance is
            applied. ``0.5`` guides the first half of the trajectory.
        cfg_match_uncond_norm: If ``True``, rescale the unconditional
            field per sample so its L2 norm matches the conditional field's
            norm before forming the CFG combination. This preserves the null
            direction while removing relative cond/null magnitude mismatch
            from ``b_cond - b_uncond``.
        cfg_rule: ``"standard"`` (b_u + w (b_c - b_u)), ``"perp"`` (b_c - s b_u_perp,
            s = w - 1) or ``"perp_matched"`` (rescale b_u to ||b_c|| first, then
            subtract s times its perpendicular part; ||g - b_c|| <= s ||b_c||).
            See ``eqfm/cfg_rules.py``.
        cfg_cnorm: With ``"perp"``, rescale the perpendicular part to ||b_c||.
        cfg_match_guided_norm: If ``True``, rescale the *final* guided vector
            per sample so its L2 norm matches the conditional field's norm.
            This is direction-only guidance: CFG can turn the field but cannot
            increase autonomous speed near equilibrium (where ``||b_cond||``
            -> 0), which keeps the fixed point intact. Intended for
            ``cfg_mode="all"``. Setting both this and ``cfg_match_uncond_norm``
            reproduces the toy's strongest variant
            (``match_norm_guided_norm_match``): match the null branch norm
            first, then match the final guided vector norm. No-op at scale 1.
        null_class: Index of SiT's unconditional embedding. SiT-XL/2 on
            ImageNet uses ``num_classes = 1000``, so the null token is
            ``1000`` (embedding table is size 1001).
        momentum: Look-ahead / heavy-ball coefficient ``mu``. ``0.0`` is plain
            forward Euler. With ``nesterov=True`` this is the NAG-GD ``mu``
            from raywang4/EqM ``sample_gd.py``; the reference checkpoint uses
            ``mu = 0.3`` with ``stepsize = 0.0017``. With ``nesterov=False``
            it is the Polyak heavy-ball coefficient. NAG-GD is the recommended
            mode and its benefit over plain GD is largest at *low* step counts.
        nesterov: If ``True``, use the paper's NAG-GD update: evaluate the
            field at the look-ahead ``x + step_size * momentum * m`` where
            ``m`` is the *previous* field value, then take a plain Euler step
            ``x + step_size * b``. The position update carries no accumulated
            velocity, so it does not inflate the effective step size. If
            ``False`` (with ``momentum > 0``), use Polyak heavy-ball
            (accumulated velocity), which can overshoot the fixed point.
            Same NFE either way.
        device, dtype: Where + how to keep the integration variable ``x``.
            Use fp32 to avoid drift over many steps even under bf16
            autocast for the model forward.
        autocast_dtype: Mixed-precision compute dtype for the model
            evaluation (e.g. ``torch.bfloat16``). ``None`` disables
            autocast.
        generator: Optional ``torch.Generator`` for reproducible noise.
        diagnostics: If provided, ``velocity_norms`` and ``latent_norms``
            are appended each step for convergence inspection. The norm
            recorded is the (CFG-combined) field actually used to step.

    Returns:
        Final latents of shape ``shape`` (still in the VAE-scaled space;
        decode with :meth:`LatentEncoder.decode` to get pixel images).
    """
    if device is None:
        device = class_labels.device
    if stop_time is None:
        stop_time = stop_time_from_eps(eps_stop)
    if momentum < 0.0 or momentum >= 1.0:
        raise ValueError(f"momentum must lie in [0, 1); got {momentum}.")
    if cfg_mode not in {"all", "first3"}:
        raise ValueError(f"cfg_mode must be 'all' or 'first3'; got {cfg_mode!r}.")
    if cfg_rule not in CFG_RULES:
        raise ValueError(f"cfg_rule must be one of {CFG_RULES}; got {cfg_rule!r}.")
    warn_if_perp_cnorm(cfg_rule, cfg_cnorm)
    if cfg_schedule not in {"constant", "anneal_to_one", "early"}:
        raise ValueError(
            f"cfg_schedule must be 'constant', 'anneal_to_one', or 'early'; "
            f"got {cfg_schedule!r}."
        )
    if cfg_schedule_power <= 0.0:
        raise ValueError(f"cfg_schedule_power must be > 0; got {cfg_schedule_power}.")
    if not 0.0 <= cfg_early_frac <= 1.0:
        raise ValueError(f"cfg_early_frac must lie in [0, 1]; got {cfg_early_frac}.")
    step_size = stop_time / num_steps

    z = torch.randn(*shape, device=device, dtype=dtype, generator=generator)
    use_autocast = autocast_dtype is not None
    autocast_device_type = device.type if hasattr(device, "type") else "cuda"
    use_cfg = cfg_scale != 1.0
    y_null = (
        torch.full_like(class_labels, null_class) if use_cfg else None
    )

    def cfg_at_step(step_idx: int) -> float:
        if not use_cfg:
            return 1.0
        if cfg_schedule == "constant":
            return float(cfg_scale)
        # Midpoint in autonomous time, consistent with the Euler step.
        s_mid = (step_idx + 0.5) * step_size
        if cfg_schedule == "early":
            # Full guidance for the first cfg_early_frac of the horizon, then
            # turn it off exactly so the unguided field lands on the manifold.
            return float(cfg_scale) if s_mid <= cfg_early_frac * stop_time else 1.0
        one_minus_t = math.exp(-s_mid)
        return 1.0 + (float(cfg_scale) - 1.0) * (one_minus_t ** cfg_schedule_power)

    def combine_cfg(b_cond: torch.Tensor, b_uncond: torch.Tensor, scale: float) -> torch.Tensor:
        guided = b_uncond + scale * (b_cond - b_uncond)
        if cfg_mode == "all":
            return guided
        out = b_cond.clone()
        out[:, :3] = guided[:, :3]
        return out

    def match_uncond_norm(b_cond: torch.Tensor, b_uncond: torch.Tensor) -> torch.Tensor:
        cond_norm = b_cond.float().flatten(1).norm(dim=1)
        uncond_norm = b_uncond.float().flatten(1).norm(dim=1).clamp_min(1e-12)
        scale = cond_norm / uncond_norm
        scale = scale.view(-1, *([1] * (b_uncond.ndim - 1)))
        return b_uncond * scale.to(dtype=b_uncond.dtype)

    def eval_field(x: torch.Tensor, step_idx: int) -> torch.Tensor:
        """Autonomous field at ``x``, with CFG if enabled. Returns dtype(z)."""
        ctx = (
            torch.amp.autocast(autocast_device_type, dtype=autocast_dtype)
            if use_autocast else _nullcontext()
        )
        with ctx:
            b_c = model_fn(x, class_labels)
            if use_cfg:
                b_u = model_fn(x, y_null)
                if cfg_match_uncond_norm:
                    b_u = match_uncond_norm(b_c, b_u)
                b = combine_cfg_rule(b_c, b_u, cfg_at_step(step_idx), cfg_rule, cnorm=cfg_cnorm, cfg_mode=cfg_mode)
                if cfg_match_guided_norm and cfg_at_step(step_idx) != 1.0:
                    # Rescale the final guided vector to ||b_cond|| per sample,
                    # so guidance can turn the field but not increase its speed
                    # near equilibrium (where ||b_cond|| -> 0). Combine with
                    # cfg_match_uncond_norm to reproduce the toy's best variant.
                    b = match_uncond_norm(b_c, b)
            else:
                b = b_c
        return b.to(z.dtype)

    # ``m`` holds the previous field value for the Nesterov look-ahead, or the
    # accumulated velocity for Polyak heavy-ball. Unused for plain Euler.
    m = torch.zeros_like(z) if momentum > 0.0 else None

    for i in range(num_steps):
        if momentum > 0.0 and nesterov:
            # NAG-GD, matching raywang4/EqM ``sample_gd.py``: evaluate the
            # field at a look-ahead extrapolated a fraction ``momentum`` of one
            # step along the previous field value, then take a *plain* Euler
            # step. The position update carries no accumulated velocity, so
            # this cannot inflate the effective step size (unlike heavy-ball);
            # the look-ahead acts as a cheap predictor that mainly helps at low
            # step counts.
            x_eval = z + step_size * momentum * m
            b = eval_field(x_eval, i)
            if diagnostics is not None:
                diagnostics.append(b, z)
            m = b
            z = z + step_size * b
        elif momentum > 0.0:
            # Polyak heavy-ball: accumulate velocity. Accelerates the approach
            # but can overshoot the fixed point since ``b`` is not a clean
            # convex gradient. Prefer ``nesterov=True`` to match the paper.
            b = eval_field(z, i)
            if diagnostics is not None:
                diagnostics.append(b, z)
            m = momentum * m + step_size * b
            z = z + m
        else:
            b = eval_field(z, i)
            if diagnostics is not None:
                diagnostics.append(b, z)
            z = z + step_size * b

    return z


def summarize_diagnostics(diag: SamplerDiagnostics) -> Dict[str, float]:
    """Convergence summary from per-step diagnostics.

    ``velocity_norm_first`` vs ``velocity_norm_last`` gives a direct read
    on how close the integration finished to a fixed point of ``b_hat``:
    for a well-trained model, ``last`` should be << ``first``.
    """
    if not diag.velocity_norms:
        return {}
    return {
        "velocity_norm_first": diag.velocity_norms[0],
        "velocity_norm_last":  diag.velocity_norms[-1],
        "velocity_norm_max":   max(diag.velocity_norms),
        "latent_norm_first":   diag.latent_norms[0],
        "latent_norm_last":    diag.latent_norms[-1],
    }


def save_grid(images: torch.Tensor, path, *, nrow: Optional[int] = None) -> None:
    """Save an ``(N, 3, H, W)`` tensor in ``[-1, 1]`` as a single grid PNG.

    ``nrow`` defaults to ``sqrt(N)`` so a 16-image batch lays out as 4x4.
    """
    from torchvision.utils import make_grid, save_image
    if nrow is None:
        nrow = max(1, int(images.shape[0] ** 0.5))
    grid = make_grid(images.clamp(-1, 1), nrow=nrow)
    save_image(grid, path, normalize=True, value_range=(-1, 1))
