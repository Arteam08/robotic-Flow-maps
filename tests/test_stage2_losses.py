"""Offline smoke tests for Stage-2 compactified flow-map losses."""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from eqfm.flowmap import compactified_add, compactified_sub, flow_map  # noqa: E402
from eqfm.stage2_losses import (  # noqa: E402
    Stage2FlowMapLoss,
    Stage2LossConfig,
    _weighted_mean,
)


class ZeroFlow(nn.Module):
    def forward(self, x, sigma, y):
        return torch.zeros_like(x)


class BadFlow(nn.Module):
    def forward(self, x, sigma, y):
        return torch.zeros(x.shape[0], x.shape[1] * 2, *x.shape[2:], device=x.device)


class TinyFlow(nn.Module):
    def __init__(self, channels: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels + 1, 8, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(8, channels, kernel_size=1),
        )

    def forward(self, x, sigma, y):
        s = sigma.view(-1, 1, 1, 1).expand(x.shape[0], 1, x.shape[2], x.shape[3])
        return self.net(torch.cat([x, s], dim=1))


class ScalarLinearVelocity(nn.Module):
    """v(x, sigma) = scale*x, with a visible target-JVP parameter."""

    def __init__(self, scale: float):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(scale))

    def forward(self, x, sigma, y):
        return self.scale * x


class ExactCompactLinearFlow(nn.Module):
    """Exact compact average velocity for b(x)=-rate*x."""

    def __init__(self, rate: float):
        super().__init__()
        self.rate = rate

    def forward(self, x, sigma, y):
        sig = sigma.view(-1, *([1] * (x.ndim - 1)))
        log_clock = torch.log1p(-sig)
        numerator = torch.expm1(self.rate * log_clock)
        coeff = torch.where(
            sig > 0.0,
            numerator / sig.clamp_min(torch.finfo(sig.dtype).tiny),
            torch.full_like(sig, -self.rate),
        )
        return coeff * x


def _zero_teacher(x, y):
    return torch.zeros_like(x)


def _linear_teacher(x, y):
    return -0.25 * x


class CountingTeacher:
    def __init__(self):
        self.calls = 0

    def __call__(self, x, y):
        self.calls += 1
        return torch.zeros_like(x)


def test_compactified_add_sub_are_inverse():
    sigma = torch.tensor([0.25, 0.5, 0.9])
    sigma_prime = torch.tensor([0.1, 0.2, 0.3])
    sigma_minus = compactified_sub(sigma, sigma_prime)
    recovered = compactified_add(sigma_prime, sigma_minus)
    assert torch.allclose(recovered, sigma, atol=1e-6)


def test_zero_flow_has_zero_stage2_loss_against_zero_teacher():
    x = torch.randn(4, 2, 4, 4)
    y = torch.zeros(4, dtype=torch.long)
    for objective in (
        "lagrangian", "eulerian", "semigroup", "meanflow", "scaled_meanflow"
    ):
        loss_fn = Stage2FlowMapLoss(Stage2LossConfig(
            objective=objective,
            cfg_dropout_prob=0.0,
            sigma_min=0.1,
            sigma_max=0.9,
        ))
        loss, metrics = loss_fn(ZeroFlow(), _zero_teacher, x, y, z=torch.randn_like(x))
        assert loss.item() < 1e-8, (objective, metrics)


def test_stage2_defaults_use_canonical_interpolant_sampling():
    assert Stage2LossConfig().interp_time_sampler == "canonical"


def test_semigroup_pair_samples_lower_triangle():
    loss_fn = Stage2FlowMapLoss(Stage2LossConfig(
        objective="semigroup",
        sigma_min=0.05,
        sigma_max=0.95,
    ))
    sigma, sigma_prime = loss_fn.sample_semigroup_pair(
        4096, torch.device("cpu"), torch.float32
    )
    assert torch.all(sigma >= 0.05)
    assert torch.all(sigma <= 0.95)
    assert torch.all(sigma_prime >= 0.0)
    assert torch.all(sigma_prime <= sigma)
    # Uniform area measure on a lower triangle has outer density
    # proportional to sigma, so its mean is 2/3 on [0, 1].  The truncated
    # case is close to the same value for this broad range and, importantly,
    # well above the midpoint that a uniform outer sampler would produce.
    assert sigma.mean().item() > 0.60


def test_lagrangian_loss_does_not_waste_teacher_forward():
    x = torch.randn(4, 2, 4, 4)
    y = torch.arange(4, dtype=torch.long)
    labels_in = y.clone()
    drop_mask = torch.zeros_like(y, dtype=torch.bool)
    sigma = torch.full((4,), 0.5)
    teacher = CountingTeacher()
    loss_fn = Stage2FlowMapLoss(Stage2LossConfig(
        objective="lagrangian",
        cfg_dropout_prob=0.0,
    ))
    loss = loss_fn.lagrangian_loss(
        ZeroFlow(), teacher, x, y, labels_in, drop_mask, sigma
    )
    assert torch.isfinite(loss).item()
    assert teacher.calls == 1


def test_stage2_objectives_are_finite_and_backpropagate():
    torch.manual_seed(0)
    x = torch.randn(3, 2, 4, 4)
    y = torch.tensor([0, 1, 2], dtype=torch.long)
    for objective in (
        "lagrangian", "eulerian", "semigroup", "meanflow", "scaled_meanflow",
        "shifted_lagrangian", "shifted_meanflow",
    ):
        model = TinyFlow(channels=2)
        loss_fn = Stage2FlowMapLoss(Stage2LossConfig(
            objective=objective,
            cfg_dropout_prob=0.0,
            terminal_weight=0.1,
            sigma_min=0.05,
            sigma_max=0.8,
        ))
        loss, metrics = loss_fn(model, _linear_teacher, x, y, z=torch.randn_like(x))
        assert torch.isfinite(loss).item(), objective
        assert "distill_loss" in metrics
        loss.backward()
        grad_norm = sum(
            p.grad.detach().abs().sum().item()
            for p in model.parameters()
            if p.grad is not None
        )
        assert grad_norm > 0.0, objective


def test_scaled_meanflow_exact_compact_linear_flow_is_a_root():
    rate = 0.25
    x = torch.randn(4, 2, 4, 4)
    y = torch.zeros(4, dtype=torch.long)
    sigma = torch.tensor([0.1, 0.5, 0.9, 1.0 - 1e-7])
    drop_mask = torch.zeros_like(y, dtype=torch.bool)
    loss_fn = Stage2FlowMapLoss(Stage2LossConfig(
        objective="scaled_meanflow",
        adaptive_weight_p=0.0,
        cfg_dropout_prob=0.0,
    ))
    loss, _ = loss_fn.scaled_meanflow_loss(
        ExactCompactLinearFlow(rate),
        _linear_teacher,
        x,
        y,
        y,
        drop_mask,
        sigma,
    )
    assert torch.isfinite(loss)
    assert loss.item() < 1e-9


def test_shifted_losses_share_root_and_residual_with_their_base_forms():
    torch.manual_seed(0)
    x = torch.randn(4, 2, 4, 4)
    y = torch.zeros(4, dtype=torch.long)
    sigma = torch.tensor([0.1, 0.5, 0.9, 1.0 - 1e-7])
    drop_mask = torch.zeros_like(y, dtype=torch.bool)

    def fn(objective):
        return Stage2FlowMapLoss(Stage2LossConfig(
            objective=objective, adaptive_weight_p=0.0, cfg_dropout_prob=0.0,
        ))

    exact = ExactCompactLinearFlow(0.25)
    for name in ("shifted_lagrangian_loss", "shifted_meanflow_loss"):
        root = getattr(fn(name[:-5]), name)(exact, _linear_teacher, x, y, y, drop_mask, sigma)
        assert root.item() < 1e-9, name

    model = TinyFlow(channels=2)
    shifted_lag = fn("shifted_lagrangian").shifted_lagrangian_loss(model, _linear_teacher, x, y, y, drop_mask, sigma)
    lag = fn("lagrangian").lagrangian_loss(model, _linear_teacher, x, y, y, drop_mask, sigma)
    assert torch.allclose(shifted_lag, lag, atol=1e-6, rtol=1e-4)
    shifted_mf = fn("shifted_meanflow").shifted_meanflow_loss(model, _linear_teacher, x, y, y, drop_mask, sigma)
    scaled, _ = fn("scaled_meanflow").scaled_meanflow_loss(model, _linear_teacher, x, y, y, drop_mask, sigma)
    assert torch.allclose(shifted_mf, scaled, atol=1e-6, rtol=1e-4)


def test_shifted_lagrangian_stops_the_whole_target():
    torch.manual_seed(0)
    x = torch.randn(3, 2, 4, 4)
    y = torch.zeros(3, dtype=torch.long)
    sigma = torch.tensor([0.2, 0.6, 0.95])
    drop_mask = torch.zeros_like(y, dtype=torch.bool)
    model = TinyFlow(channels=2)
    loss_fn = Stage2FlowMapLoss(Stage2LossConfig(
        objective="shifted_lagrangian", adaptive_weight_p=0.0, cfg_dropout_prob=0.0,
        loss_reduction="sum",  # the hand-written reference below sums over entries
    ))
    loss = loss_fn.shifted_lagrangian_loss(model, _linear_teacher, x, y, y, drop_mask, sigma)
    grads = torch.autograd.grad(loss, list(model.parameters()), allow_unused=True)
    # reference: the same regression with the target computed beforehand and detached
    with torch.no_grad():
        pred0 = model(x, sigma, y)
    from eqfm.flowmap import velocity_jvp
    with torch.no_grad():
        _, dv = velocity_jvp(model, x, sigma, y, torch.zeros_like(x), torch.ones_like(sigma))
        sig = sigma.view(-1, 1, 1, 1)
        target = _linear_teacher(x + sig * pred0, y) + sig * (pred0 - (1.0 - sig) * dv)
    ref = (model(x, sigma, y) - target).flatten(1).pow(2).sum(1).mean()
    ref_grads = torch.autograd.grad(ref, list(model.parameters()), allow_unused=True)
    for g, r in zip(grads, ref_grads):
        if g is not None:
            assert torch.allclose(g, r, atol=1e-6, rtol=1e-4)


def test_scaled_meanflow_p1_matches_divided_semigradient():
    clock = torch.tensor([0.8, 0.2, 1e-3])
    feature = torch.tensor([1.0, -2.0, 0.5])
    target = torch.tensor([0.3, 0.7, -0.2])
    eps = 1e-3
    theta_divided = torch.tensor(0.4, requires_grad=True)
    theta_scaled = theta_divided.detach().clone().requires_grad_(True)

    residual_divided = theta_divided * feature - target
    divided_per = residual_divided.square()
    divided_loss = _weighted_mean(divided_per, 1.0, eps)

    residual_scaled = clock * (theta_scaled * feature - target)
    scaled_per = residual_scaled.square()
    scaled_loss = _weighted_mean(scaled_per, 1.0, eps * clock.square())

    divided_grad, = torch.autograd.grad(divided_loss, theta_divided)
    scaled_grad, = torch.autograd.grad(scaled_loss, theta_scaled)
    assert torch.allclose(scaled_loss, divided_loss, atol=1e-7, rtol=1e-6)
    assert torch.allclose(scaled_grad, divided_grad, atol=1e-7, rtol=1e-6)


def test_scaled_meanflow_p1_is_finite_at_compact_endpoint():
    x = torch.randn(2, 2, 2, 2)
    y = torch.zeros(2, dtype=torch.long)
    sigma = torch.tensor([1.0, 1.0 - 1e-7])
    drop_mask = torch.zeros_like(y, dtype=torch.bool)
    loss_fn = Stage2FlowMapLoss(Stage2LossConfig(
        objective="scaled_meanflow",
        adaptive_weight_p=1.0,
        adaptive_weight_eps=1e-3,
        cfg_dropout_prob=0.0,
    ))
    loss, _ = loss_fn.scaled_meanflow_loss(
        TinyFlow(channels=2), _linear_teacher, x, y, y, drop_mask, sigma
    )
    assert torch.isfinite(loss)
    assert loss.item() <= 1.0


def test_scaled_meanflow_stops_entire_jvp_target():
    rate = 0.25
    x = torch.randn(3, 2, 2, 2)
    y = torch.zeros(3, dtype=torch.long)
    sigma = torch.tensor([0.2, 0.6, 0.9])
    drop_mask = torch.zeros_like(y, dtype=torch.bool)
    model = ScalarLinearVelocity(scale=0.4)
    loss_fn = Stage2FlowMapLoss(Stage2LossConfig(
        objective="scaled_meanflow",
        adaptive_weight_p=0.0,
        cfg_dropout_prob=0.0,
    ))
    loss, _ = loss_fn.scaled_meanflow_loss(
        model, _linear_teacher, x, y, y, drop_mask, sigma
    )
    actual_grad, = torch.autograd.grad(loss, model.scale)

    sig = sigma.view(-1, 1, 1, 1)
    clock = 1.0 - sig
    theta = model.scale.detach()
    target_scaled = -rate * (1.0 + sig * theta) * x
    residual = clock * theta * x - target_scaled
    expected_grad = (
        2.0 * residual * (clock * x)
    ).flatten(1).mean(dim=1).mean()  # default loss_reduction="mean": per-entry average
    assert torch.allclose(actual_grad, expected_grad, atol=1e-6, rtol=1e-6)


def test_terminal_invariance_uses_one_sided_stopped_target():
    rate = 0.25
    delta = 0.1
    x = torch.randn(3, 2, 2, 2)
    y = torch.zeros(3, dtype=torch.long)
    b = -rate * x
    model = ScalarLinearVelocity(scale=0.4)
    loss_fn = Stage2FlowMapLoss(Stage2LossConfig(
        objective="scaled_meanflow",
        adaptive_weight_p=0.0,
        cfg_dropout_prob=0.0,
        terminal_invariance_step=delta,
    ))
    loss, _ = loss_fn.terminal_invariance_loss(model, x, b, y)
    actual_grad, = torch.autograd.grad(loss, model.scale)

    theta = model.scale.detach()
    residual = (1.0 + theta) * rate * delta * x
    expected_grad = (2.0 * residual * x).flatten(1).mean(dim=1).mean()  # default loss_reduction="mean"
    assert torch.allclose(actual_grad, expected_grad, atol=1e-6, rtol=1e-6)


def test_scaled_meanflow_invariance_reuses_guided_teacher_field():
    x = torch.randn(3, 2, 2, 2)
    y = torch.zeros(3, dtype=torch.long)
    teacher = CountingTeacher()
    loss_fn = Stage2FlowMapLoss(Stage2LossConfig(
        objective="scaled_meanflow",
        adaptive_weight_p=1.0,
        cfg_dropout_prob=0.0,
        terminal_invariance_weight=0.5,
        sigma_min=0.1,
        sigma_max=0.9,
    ))
    loss, metrics = loss_fn(
        TinyFlow(channels=2), teacher, x, y, z=torch.randn_like(x)
    )
    assert torch.isfinite(loss)
    assert "terminal_invariance_loss" in metrics
    # One teacher call in total: MeanFlow reuses the diagonal term's field at the same x0
    # (exact, see tests/test_s2mf.py) and invariance reuses MeanFlow's b0.
    assert teacher.calls == 1


def test_flow_map_terminal_identity_for_zero_velocity():
    x = torch.randn(2, 2, 4, 4)
    y = torch.zeros(2, dtype=torch.long)
    sigma = torch.ones(2)
    out = flow_map(ZeroFlow(), x, sigma, y)
    assert torch.allclose(out, x)


def test_flow_map_rejects_channel_mismatch():
    x = torch.randn(2, 2, 4, 4)
    y = torch.zeros(2, dtype=torch.long)
    sigma = torch.ones(2)
    try:
        flow_map(BadFlow(), x, sigma, y)
    except ValueError as e:
        assert "velocity-only" in str(e)
    else:
        raise AssertionError("flow_map accepted a 2C velocity output")


# ---------------------------------------------------------------------------
# Defaults after merging cfg-perp-matched-and-loss-reduction: mean reduction, perp+cnorm warns
# ---------------------------------------------------------------------------
def test_defaults_are_mean_reduction_and_standard_cfg():
    cfg = Stage2LossConfig()
    assert cfg.loss_reduction == "mean" and cfg.cfg_rule == "standard" and cfg.cfg_cnorm is False


def test_perp_cnorm_warns_and_perp_matched_does_not():
    import warnings

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        Stage2FlowMapLoss(Stage2LossConfig(cfg_rule="perp_matched"))
        Stage2FlowMapLoss(Stage2LossConfig(cfg_rule="perp"))
        assert not caught
        Stage2FlowMapLoss(Stage2LossConfig(cfg_rule="perp", cfg_cnorm=True))
        assert len(caught) == 1 and "perp_matched" in str(caught[0].message)


def test_perp_matched_is_bounded_where_cnorm_is_not():
    from eqfm.cfg_rules import combine_cfg

    torch.manual_seed(0)
    b_c = torch.randn(6, 2, 4, 4)
    b_u = 1.001 * b_c + 1e-4 * torch.randn(6, 2, 4, 4)      # nearly parallel: tiny perpendicular remainder
    s = 2.0
    nc = b_c.flatten(1).norm(dim=1)
    pm = (combine_cfg(b_c, b_u, 1.0 + s, "perp_matched") - b_c).flatten(1).norm(dim=1)
    cn = (combine_cfg(b_c, b_u, 1.0 + s, "perp", cnorm=True) - b_c).flatten(1).norm(dim=1)
    assert torch.all(pm <= s * nc + 1e-5) and torch.all(pm < 1e-2 * nc)
    assert torch.allclose(cn, s * nc, rtol=1e-4)               # cnorm inflates the remainder to full size


if __name__ == "__main__":
    tests = [
        test_compactified_add_sub_are_inverse,
        test_zero_flow_has_zero_stage2_loss_against_zero_teacher,
        test_stage2_defaults_use_canonical_interpolant_sampling,
        test_semigroup_pair_samples_lower_triangle,
        test_lagrangian_loss_does_not_waste_teacher_forward,
        test_stage2_objectives_are_finite_and_backpropagate,
        test_scaled_meanflow_exact_compact_linear_flow_is_a_root,
        test_shifted_losses_share_root_and_residual_with_their_base_forms,
        test_shifted_lagrangian_stops_the_whole_target,
        test_scaled_meanflow_p1_matches_divided_semigradient,
        test_scaled_meanflow_p1_is_finite_at_compact_endpoint,
        test_scaled_meanflow_stops_entire_jvp_target,
        test_terminal_invariance_uses_one_sided_stopped_target,
        test_scaled_meanflow_invariance_reuses_guided_teacher_field,
        test_flow_map_terminal_identity_for_zero_velocity,
        test_flow_map_rejects_channel_mismatch,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")


# ---------------------------------------------------------------------------
# MNIST ablation additions (2026-09-11): importance-corrected samplers, meanflow
# clock variant, generic PESD add-on, weighted mean with sample weights.
# ---------------------------------------------------------------------------
from eqfm.losses import (  # noqa: E402
    importance_weight,
    sample_time_on_interval,
    uniform_to_canonical_weight,
)


def test_importance_weights_integrate_to_one():
    torch.manual_seed(0)
    n = 400_000
    lo, hi = 0.0, 0.999
    for sampler, ref, tol in (
        ("uniform", "canonical", 0.03),
        ("canonical", "uniform", 0.03),
        ("power", "uniform", 0.03),
        ("power", "canonical", 0.03),
        ("logitnormal", "uniform", 0.08),   # heavy-tailed ratio, looser tolerance
    ):
        t = sample_time_on_interval(n, lo, hi, sampler)
        w = importance_weight(t, lo, hi, sampler, ref)
        assert torch.isfinite(w).all(), (sampler, ref)
        assert abs(w.mean().item() - 1.0) < tol, (sampler, ref, w.mean().item())
    t = torch.rand(1000) * 0.999
    w_new = importance_weight(t, 0.0, 0.999, "uniform", "canonical")
    w_old = uniform_to_canonical_weight(t, 1e-3)
    assert torch.allclose(w_new, w_old, atol=1e-5)


def test_weighted_mean_sample_weight_one_matches_old_path():
    torch.manual_seed(0)
    per = torch.rand(16) * 3.0
    for p in (0.0, 0.5, 1.0):
        a = _weighted_mean(per, p, 1e-3)
        b = _weighted_mean(per, p, 1e-3, sample_weight=torch.ones_like(per))
        assert torch.allclose(a, b)
    # the adaptive factor comes from the unweighted loss: scaling w by 2 doubles the value at p = 1
    w2 = torch.full_like(per, 2.0)
    assert torch.allclose(_weighted_mean(per, 1.0, 1e-3, sample_weight=w2), 2.0 * _weighted_mean(per, 1.0, 1e-3))


def test_stage2_config_defaults_unchanged():
    cfg = Stage2LossConfig()
    assert cfg.interp_is_correct is False and cfg.sigma_is_correct is False
    assert cfg.meanflow_clock == "clamp" and cfg.semigroup_weight == 0.0 and cfg.time_weight_clip == 0.0
    assert cfg.interp_time_sampler == "canonical" and cfg.sigma_sampler == "uniform"


def test_meanflow_add_clock_is_finite_at_compact_endpoint():
    torch.manual_seed(0)
    x = torch.randn(4, 2, 4, 4)
    y = torch.zeros(4, dtype=torch.long)
    for clock in ("clamp", "add"):
        model = TinyFlow(channels=2)
        loss_fn = Stage2FlowMapLoss(Stage2LossConfig(
            objective="meanflow", meanflow_clock=clock, meanflow_eps=1e-3,
            sigma_min=0.5, sigma_max=1.0, cfg_dropout_prob=0.0,
        ))
        sigma = torch.tensor([0.5, 0.9, 0.999, 1.0])
        drop_mask = torch.zeros_like(y, dtype=torch.bool)
        loss = loss_fn.meanflow_loss(model, _linear_teacher, x, y, y, drop_mask, sigma)
        assert torch.isfinite(loss).item(), clock
        assert loss_fn._aux["meanflow_clock_min"].item() >= (1e-3 if clock == "clamp" else 1e-3 - 1e-9)


def test_semigroup_weight_adds_finite_term_for_each_objective():
    torch.manual_seed(0)
    x = torch.randn(3, 2, 4, 4)
    y = torch.tensor([0, 1, 2], dtype=torch.long)
    for objective in ("lagrangian", "eulerian", "meanflow", "scaled_meanflow"):
        model = TinyFlow(channels=2)
        loss_fn = Stage2FlowMapLoss(Stage2LossConfig(
            objective=objective, semigroup_weight=1.0, cfg_dropout_prob=0.0,
            sigma_min=0.05, sigma_max=0.8, terminal_invariance_weight=0.5,
        ))
        loss, metrics = loss_fn(model, _linear_teacher, x, y, z=torch.randn_like(x))
        assert torch.isfinite(loss).item(), objective
        assert "semigroup_add_loss" in metrics and torch.isfinite(metrics["semigroup_add_loss"]).item()
        assert torch.isfinite(metrics["terminal_invariance_loss"]).item()
        assert "distill_raw_t0_0.1" in metrics and "distill_ess" in metrics
        loss.backward()


def test_corrected_samplers_run_and_report_weight_stats():
    torch.manual_seed(0)
    x = torch.randn(6, 2, 4, 4)
    y = torch.zeros(6, dtype=torch.long)
    for ts, ss in (("uniform", "uniform"), ("power", "logitnormal"), ("logitnormal", "power")):
        model = TinyFlow(channels=2)
        loss_fn = Stage2FlowMapLoss(Stage2LossConfig(
            objective="lagrangian", interp_time_sampler=ts, interp_is_correct=True, interp_ref="canonical",
            sigma_sampler=ss, sigma_is_correct=True, sigma_ref="uniform", sigma_min=0.05, sigma_max=0.9,
            time_weight_clip=50.0, cfg_dropout_prob=0.0,
        ))
        loss, metrics = loss_fn(model, _linear_teacher, x, y, z=torch.randn_like(x))
        assert torch.isfinite(loss).item(), (ts, ss)
        assert metrics["time_weight_t_max"].item() <= 50.0 + 1e-6
        assert "time_weight_s_ess" in metrics

# loss_reduction and the perp_matched CFG rule (branch cfg-perp-matched-and-loss-reduction)
# ---------------------------------------------------------------------------
from eqfm.losses import (  # noqa: E402
    importance_weight,
    sample_time_on_interval,
    uniform_to_canonical_weight,
)


def test_importance_weights_integrate_to_one():
    torch.manual_seed(0)
    n = 400_000
    lo, hi = 0.0, 0.999
    for sampler, ref, tol in (
        ("uniform", "canonical", 0.03),
        ("canonical", "uniform", 0.03),
        ("power", "uniform", 0.03),
        ("power", "canonical", 0.03),
        ("logitnormal", "uniform", 0.08),   # heavy-tailed ratio, looser tolerance
    ):
        t = sample_time_on_interval(n, lo, hi, sampler)
        w = importance_weight(t, lo, hi, sampler, ref)
        assert torch.isfinite(w).all(), (sampler, ref)
        assert abs(w.mean().item() - 1.0) < tol, (sampler, ref, w.mean().item())
    t = torch.rand(1000) * 0.999
    w_new = importance_weight(t, 0.0, 0.999, "uniform", "canonical")
    w_old = uniform_to_canonical_weight(t, 1e-3)
    assert torch.allclose(w_new, w_old, atol=1e-5)


def test_weighted_mean_sample_weight_one_matches_old_path():
    torch.manual_seed(0)
    per = torch.rand(16) * 3.0
    for p in (0.0, 0.5, 1.0):
        a = _weighted_mean(per, p, 1e-3)
        b = _weighted_mean(per, p, 1e-3, sample_weight=torch.ones_like(per))
        assert torch.allclose(a, b)
    # the adaptive factor comes from the unweighted loss: scaling w by 2 doubles the value at p = 1
    w2 = torch.full_like(per, 2.0)
    assert torch.allclose(_weighted_mean(per, 1.0, 1e-3, sample_weight=w2), 2.0 * _weighted_mean(per, 1.0, 1e-3))


def test_stage2_config_defaults_unchanged():
    cfg = Stage2LossConfig()
    assert cfg.interp_is_correct is False and cfg.sigma_is_correct is False
    assert cfg.meanflow_clock == "clamp" and cfg.semigroup_weight == 0.0 and cfg.time_weight_clip == 0.0
    assert cfg.interp_time_sampler == "canonical" and cfg.sigma_sampler == "uniform"


def test_meanflow_add_clock_is_finite_at_compact_endpoint():
    torch.manual_seed(0)
    x = torch.randn(4, 2, 4, 4)
    y = torch.zeros(4, dtype=torch.long)
    for clock in ("clamp", "add"):
        model = TinyFlow(channels=2)
        loss_fn = Stage2FlowMapLoss(Stage2LossConfig(
            objective="meanflow", meanflow_clock=clock, meanflow_eps=1e-3,
            sigma_min=0.5, sigma_max=1.0, cfg_dropout_prob=0.0,
        ))
        sigma = torch.tensor([0.5, 0.9, 0.999, 1.0])
        drop_mask = torch.zeros_like(y, dtype=torch.bool)
        loss = loss_fn.meanflow_loss(model, _linear_teacher, x, y, y, drop_mask, sigma)
        assert torch.isfinite(loss).item(), clock
        assert loss_fn._aux["meanflow_clock_min"].item() >= (1e-3 if clock == "clamp" else 1e-3 - 1e-9)


def test_semigroup_weight_adds_finite_term_for_each_objective():
    torch.manual_seed(0)
    x = torch.randn(3, 2, 4, 4)
    y = torch.tensor([0, 1, 2], dtype=torch.long)
    for objective in ("lagrangian", "eulerian", "meanflow", "scaled_meanflow"):
        model = TinyFlow(channels=2)
        loss_fn = Stage2FlowMapLoss(Stage2LossConfig(
            objective=objective, semigroup_weight=1.0, cfg_dropout_prob=0.0,
            sigma_min=0.05, sigma_max=0.8, terminal_invariance_weight=0.5,
        ))
        loss, metrics = loss_fn(model, _linear_teacher, x, y, z=torch.randn_like(x))
        assert torch.isfinite(loss).item(), objective
        assert "semigroup_add_loss" in metrics and torch.isfinite(metrics["semigroup_add_loss"]).item()
        assert torch.isfinite(metrics["terminal_invariance_loss"]).item()
        assert "distill_raw_t0_0.1" in metrics and "distill_ess" in metrics
        loss.backward()


def test_corrected_samplers_run_and_report_weight_stats():
    torch.manual_seed(0)
    x = torch.randn(6, 2, 4, 4)
    y = torch.zeros(6, dtype=torch.long)
    for ts, ss in (("uniform", "uniform"), ("power", "logitnormal"), ("logitnormal", "power")):
        model = TinyFlow(channels=2)
        loss_fn = Stage2FlowMapLoss(Stage2LossConfig(
            objective="lagrangian", interp_time_sampler=ts, interp_is_correct=True, interp_ref="canonical",
            sigma_sampler=ss, sigma_is_correct=True, sigma_ref="uniform", sigma_min=0.05, sigma_max=0.9,
            time_weight_clip=50.0, cfg_dropout_prob=0.0,
        ))
        loss, metrics = loss_fn(model, _linear_teacher, x, y, z=torch.randn_like(x))
        assert torch.isfinite(loss).item(), (ts, ss)
        assert metrics["time_weight_t_max"].item() <= 50.0 + 1e-6
        assert "time_weight_s_ess" in metrics


# ---------------------------------------------------------------------------
# Handoff setting: sigma ~ U[0, 1] and t ~ U[0, 1] with NO caps (sigma_max = 1, eps_train = 0).
# Every term used by the handoff runs must stay finite (loss and gradients) at the endpoints.
def test_handoff_full_range_sigma_and_t_are_finite():
    torch.manual_seed(0)
    x = torch.randn(4, 2, 4, 4)
    y = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    drop_mask = torch.zeros_like(y, dtype=torch.bool)
    sigma = torch.tensor([0.0, 0.5, 0.999, 1.0])
    for objective in ("shifted_meanflow", "shifted_lagrangian"):
        for semi, term in ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)):
            model = TinyFlow(channels=2)
            loss_fn = Stage2FlowMapLoss(Stage2LossConfig(
                objective=objective, adaptive_weight_p=1.0, adaptive_weight_eps=1e-3, loss_reduction="mean",
                sigma_min=0.0, sigma_max=1.0, sigma_sampler="uniform", interp_time_sampler="uniform", eps_train=0.0,
                semigroup_weight=semi, terminal_teacher_norm_weight=term, terminal_norm_max_batch=4,
                cfg_dropout_prob=0.0, mf_single_jvp=True,
            ))
            fn = getattr(loss_fn, f"{objective}_loss")
            distill = fn(model, _linear_teacher, x, y, y, drop_mask, sigma)
            assert torch.isfinite(distill).item(), (objective, "endpoint sigma")
            for seed in range(3):
                torch.manual_seed(seed)
                loss, metrics = loss_fn(model, _linear_teacher, x, y, z=torch.randn_like(x),
                                        teacher_grad_fn=_linear_teacher)
                assert torch.isfinite(loss).item(), (objective, semi, term, seed)
                grads = torch.autograd.grad(loss, [p for p in model.parameters() if p.requires_grad], allow_unused=True)
                assert all(torch.isfinite(g).all() for g in grads if g is not None), (objective, semi, term, seed)
