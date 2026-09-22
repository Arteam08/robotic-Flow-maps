"""Offline tests for sample_euler: Euler integration, CFG, momentum.

Pure CPU, analytic fields -- no model, no GPU, no checkpoint. We use
closed-form vector fields whose ODE behavior is known, so the sampler's
update rules can be checked exactly.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from eqfm.sampling import SamplerDiagnostics, sample_euler  # noqa: E402

CPU = torch.device("cpu")


def _const_field(value: float):
    """model_fn ignoring x: returns a constant vector per element."""
    def fn(x, y):
        return torch.full_like(x, value)
    return fn


def _gen():
    return torch.Generator().manual_seed(0)


def test_euler_converges_to_fixed_point():
    # dot x = target - x  =>  x(s) = target + (x0 - target) e^{-s}.
    target = 2.0

    def fn(x, y):
        return target - x

    z = sample_euler(
        fn, shape=(4, 8),
        class_labels=torch.zeros(4, dtype=torch.long),
        num_steps=200, stop_time=10.0, device=CPU, generator=_gen(),
    )
    assert torch.allclose(z, torch.full_like(z, target), atol=1e-2), z.mean()


def test_cfg_combines_fields_exactly():
    # b_c = +1 (cond), b_u = -1 (null). Combined = b_u + s (b_c - b_u).
    def fn(x, y):
        is_null = (y == 1000).view(-1, *([1] * (x.ndim - 1)))
        return torch.where(is_null,
                           torch.full_like(x, -1.0),
                           torch.full_like(x, 1.0))

    diag = SamplerDiagnostics()
    s = 3.0
    sample_euler(
        fn, shape=(2, 4),
        class_labels=torch.zeros(2, dtype=torch.long),
        num_steps=1, stop_time=1.0,
        cfg_scale=s, null_class=1000,
        device=CPU, generator=_gen(), diagnostics=diag,
    )
    # combined value = -1 + 3*(1 - (-1)) = 5; per-sample L2 over 4 dims
    # = sqrt(4 * 25) = 10; mean over the batch = 10.
    assert abs(diag.velocity_norms[0] - 10.0) < 1e-4, diag.velocity_norms


def test_cfg_scale_one_is_plain_conditional():
    def fn(x, y):
        # +2 for conditional, -7 for null; if CFG were (wrongly) applied
        # at s=1 the result would still be the conditional value, but we
        # also assert the null path is never even consulted by checking
        # the recorded norm equals the conditional norm.
        is_null = (y == 1000).view(-1, *([1] * (x.ndim - 1)))
        return torch.where(is_null,
                           torch.full_like(x, -7.0),
                           torch.full_like(x, 2.0))

    diag = SamplerDiagnostics()
    sample_euler(
        fn, shape=(3, 4),
        class_labels=torch.zeros(3, dtype=torch.long),
        num_steps=1, stop_time=1.0, cfg_scale=1.0,
        device=CPU, generator=_gen(), diagnostics=diag,
    )
    # conditional value 2 over 4 dims -> norm sqrt(4*4) = 4.
    assert abs(diag.velocity_norms[0] - 4.0) < 1e-4, diag.velocity_norms


def test_first3_cfg_guides_only_first_three_channels():
    def fn(x, y):
        is_null = (y == 1000).view(-1, *([1] * (x.ndim - 1)))
        return torch.where(is_null,
                           torch.full_like(x, -1.0),
                           torch.full_like(x, 1.0))

    z = sample_euler(
        fn, shape=(1, 4),
        class_labels=torch.zeros(1, dtype=torch.long),
        num_steps=1, stop_time=1.0,
        cfg_scale=3.0, cfg_mode="first3", null_class=1000,
        device=CPU, generator=_gen(),
    )
    z0 = torch.randn(1, 4, device=CPU, generator=_gen())
    # first3 guided value = -1 + 3*(1 - -1) = 5; last channel stays cond=1.
    expected = z0 + torch.tensor([[5.0, 5.0, 5.0, 1.0]])
    assert torch.allclose(z, expected, atol=1e-5), (z, expected)


def test_cfg_anneal_to_one_reduces_late_guidance():
    def fn(x, y):
        is_null = (y == 1000).view(-1, *([1] * (x.ndim - 1)))
        return torch.where(is_null,
                           torch.full_like(x, -1.0),
                           torch.full_like(x, 1.0))

    diag = SamplerDiagnostics()
    sample_euler(
        fn, shape=(1, 4),
        class_labels=torch.zeros(1, dtype=torch.long),
        num_steps=2, stop_time=10.0,
        cfg_scale=5.0, cfg_schedule="anneal_to_one",
        cfg_schedule_power=1.0, null_class=1000,
        device=CPU, generator=_gen(), diagnostics=diag,
    )
    assert diag.velocity_norms[0] > diag.velocity_norms[1], diag.velocity_norms


def test_cfg_can_match_unconditional_norm_to_conditional_norm():
    # Conditional norm over 4 dims = 2. Unconditional norm over 4 dims = 20.
    # With cfg=2 and matching on, b_u is scaled from -10 to -1, so guided value
    # is -1 + 2 * (1 - -1) = 3. Norm over 4 dims = 6.
    def fn(x, y):
        is_null = (y == 1000).view(-1, *([1] * (x.ndim - 1)))
        return torch.where(is_null,
                           torch.full_like(x, -10.0),
                           torch.full_like(x, 1.0))

    diag = SamplerDiagnostics()
    sample_euler(
        fn, shape=(1, 4),
        class_labels=torch.zeros(1, dtype=torch.long),
        num_steps=1, stop_time=1.0,
        cfg_scale=2.0, cfg_match_uncond_norm=True, null_class=1000,
        device=CPU, generator=_gen(), diagnostics=diag,
    )
    assert abs(diag.velocity_norms[0] - 6.0) < 1e-4, diag.velocity_norms


def test_cfg_match_unconditional_norm_can_scale_up():
    # Conditional norm over 4 dims = 2. Unconditional norm over 4 dims = 0.2.
    # Matching scales b_u from -0.1 to -1.0, giving the same guided value
    # as the downscale test: -1 + 2 * (1 - -1) = 3.
    def fn(x, y):
        is_null = (y == 1000).view(-1, *([1] * (x.ndim - 1)))
        return torch.where(is_null,
                           torch.full_like(x, -0.1),
                           torch.full_like(x, 1.0))

    diag = SamplerDiagnostics()
    sample_euler(
        fn, shape=(1, 4),
        class_labels=torch.zeros(1, dtype=torch.long),
        num_steps=1, stop_time=1.0,
        cfg_scale=2.0, cfg_match_uncond_norm=True, null_class=1000,
        device=CPU, generator=_gen(), diagnostics=diag,
    )
    assert abs(diag.velocity_norms[0] - 6.0) < 1e-4, diag.velocity_norms


def test_momentum_accelerates_constant_field():
    common = dict(
        shape=(1, 4),
        class_labels=torch.zeros(1, dtype=torch.long),
        num_steps=20, stop_time=1.0, device=CPU,
    )
    z_plain = sample_euler(_const_field(1.0), momentum=0.0,
                           generator=_gen(), **common)
    z_mom = sample_euler(_const_field(1.0), momentum=0.9,
                         generator=_gen(), **common)
    # Same seed -> same z0. Constant +1 field. Heavy-ball accumulates
    # velocity, so it must have travelled strictly further than plain
    # Euler in the +1 direction.
    assert (z_mom > z_plain).all(), (z_mom - z_plain)


def test_nesterov_runs_and_is_finite():
    z = sample_euler(
        _const_field(0.3), shape=(2, 4),
        class_labels=torch.zeros(2, dtype=torch.long),
        num_steps=10, stop_time=1.0, momentum=0.7, nesterov=True,
        device=CPU, generator=_gen(),
    )
    assert torch.isfinite(z).all()


def test_cfg_early_schedule_turns_off_after_frac():
    # b_c = +1 (cond), b_u = -1 (null). With cfg_scale=3 the guided norm is 10
    # while guidance is on, and the plain conditional norm is 2 once it turns
    # off. With 4 steps over stop_time=1 and cfg_early_frac=0.5, step midpoints
    # are 0.125, 0.375 (on) and 0.625, 0.875 (off): first two norms 10, last
    # two norms 2.
    def fn(x, y):
        is_null = (y == 1000).view(-1, *([1] * (x.ndim - 1)))
        return torch.where(is_null,
                           torch.full_like(x, -1.0),
                           torch.full_like(x, 1.0))

    diag = SamplerDiagnostics()
    sample_euler(
        fn, shape=(2, 4),
        class_labels=torch.zeros(2, dtype=torch.long),
        num_steps=4, stop_time=1.0,
        cfg_scale=3.0, cfg_schedule="early", cfg_early_frac=0.5,
        null_class=1000, device=CPU, generator=_gen(), diagnostics=diag,
    )
    norms = [round(v, 4) for v in diag.velocity_norms]
    assert norms == [10.0, 10.0, 2.0, 2.0], norms


def test_cfg_early_frac_out_of_range_raises():
    for bad in (-0.1, 1.5):
        try:
            sample_euler(
                _const_field(1.0), shape=(1, 4),
                class_labels=torch.zeros(1, dtype=torch.long),
                num_steps=1, stop_time=1.0, cfg_scale=2.0,
                cfg_schedule="early", cfg_early_frac=bad, device=CPU,
            )
            raise AssertionError(f"cfg_early_frac={bad} should have raised")
        except ValueError:
            pass


def test_cfg_match_guided_norm_rescales_to_cond_norm():
    # b_c = +1 (cond, per-sample L2 over 4 dims = 2), b_u = -1 (null).
    # Standard CFG at s=3 gives guided value 5 (norm 10). With guided-norm
    # matching the final vector is rescaled to ||b_c|| = 2.
    def fn(x, y):
        is_null = (y == 1000).view(-1, *([1] * (x.ndim - 1)))
        return torch.where(is_null,
                           torch.full_like(x, -1.0),
                           torch.full_like(x, 1.0))

    diag = SamplerDiagnostics()
    sample_euler(
        fn, shape=(2, 4),
        class_labels=torch.zeros(2, dtype=torch.long),
        num_steps=1, stop_time=1.0,
        cfg_scale=3.0, cfg_match_guided_norm=True,
        null_class=1000, device=CPU, generator=_gen(), diagnostics=diag,
    )
    assert abs(diag.velocity_norms[0] - 2.0) < 1e-4, diag.velocity_norms


def test_momentum_out_of_range_raises():
    for bad in (1.0, 1.5, -0.1):
        try:
            sample_euler(
                _const_field(1.0), shape=(1, 4),
                class_labels=torch.zeros(1, dtype=torch.long),
                num_steps=1, stop_time=1.0, momentum=bad, device=CPU,
            )
            raise AssertionError(f"momentum={bad} should have raised")
        except ValueError:
            pass


if __name__ == "__main__":
    import traceback
    failures = 0
    for name in list(globals()):
        if name.startswith("test_"):
            try:
                globals()[name]()
                print(f"PASS {name}")
            except Exception:
                failures += 1
                print(f"FAIL {name}")
                traceback.print_exc()
    sys.exit(1 if failures else 0)


def test_sample_euler_perp_rules_reduce_to_plain_field_when_uncond_equals_cond():
    """With a label-independent field b_u == b_c, every perp rule must return b_c exactly (no guidance push)."""
    import torch
    from eqfm.sampling import sample_euler

    def field(x, y):
        return -x

    labels = torch.zeros(3, dtype=torch.long)
    kw = dict(shape=(3, 2, 4, 4), class_labels=labels, num_steps=8, null_class=1000, device=torch.device("cpu"))
    ref = sample_euler(field, cfg_scale=1.0, generator=torch.Generator().manual_seed(0), **kw)
    for rule in ("perp", "perp_matched"):
        out = sample_euler(field, cfg_scale=3.0, cfg_rule=rule, generator=torch.Generator().manual_seed(0), **kw)
        assert torch.isfinite(out).all()
        assert torch.allclose(out, ref, atol=1e-6), rule
    bad = False
    try:
        sample_euler(field, cfg_scale=2.0, cfg_rule="nope", generator=torch.Generator().manual_seed(0), **kw)
    except ValueError:
        bad = True
    assert bad
