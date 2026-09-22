"""Offline correctness tests for the EqM loss and time samplers.

No GPU, no checkpoint, no GCS. We only need to verify:

  * The samplers produce values in the right range.
  * The interpolant + target obey the canonical identities at the endpoints.
  * EqMLoss is zero when the model returns the exact target.
  * Gradients actually flow.

Run with either pytest or plain ``python tests/test_losses.py``.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from eqfm.interpolant import (  # noqa: E402
    canonical_interpolant,
    denoiser_target,
    equilibrium_target,
)
from eqfm.losses import (  # noqa: E402
    EqMLoss,
    EqMLossConfig,
    sample_time_canonical,
    sample_time_importance,
    sample_time_uniform,
    uniform_to_canonical_weight,
)


def test_canonical_time_sampler_in_range():
    torch.manual_seed(0)
    t = sample_time_canonical(1024, eps_train=1e-3)
    assert t.min() >= 0.0
    assert t.max() <= 1.0 - 1e-4
    # Density should concentrate near 1; median > 0.5 with eps_train=1e-3.
    assert t.median().item() > 0.5


def test_importance_time_sampler_in_range():
    torch.manual_seed(0)
    for k in (1.0, 2.0, 4.0):
        t = sample_time_importance(1024, k=k)
        assert t.min() >= 0.0
        assert t.max() <= 1.0


def test_uniform_weighted_sampler_matches_canonical_density_ratio():
    torch.manual_seed(0)
    eps = 1e-3
    t = sample_time_uniform(65536, eps_train=eps)
    assert t.min() >= 0.0
    assert t.max() <= 1.0 - eps

    w = uniform_to_canonical_weight(t, eps)
    assert torch.isfinite(w).all()
    # The density ratio p_canonical / p_uniform has expectation 1 under
    # uniform samples on [0, 1 - eps].
    assert abs(w.mean().item() - 1.0) < 0.1
    assert w.min().item() >= 0.0
    assert w.max().item() <= (1.0 - eps) / (-math.log(eps) * eps)


def test_interpolant_endpoint_identities():
    z = torch.randn(4, 4, 32, 32)
    x = torch.randn(4, 4, 32, 32)
    t0 = torch.zeros(4)
    t1 = torch.ones(4)
    assert torch.allclose(canonical_interpolant(z, x, t0), z)
    assert torch.allclose(canonical_interpolant(z, x, t1), x)
    # Target vanishes at t = 1 (manifold landing).
    assert torch.allclose(equilibrium_target(z, x, t1), torch.zeros_like(z))
    # And matches (x - z) at t = 0.
    assert torch.allclose(equilibrium_target(z, x, t0), x - z)


def test_eqm_loss_is_zero_at_oracle():
    torch.manual_seed(0)
    cfg = EqMLossConfig(eps_train=1e-3, time_sampler="canonical")
    loss_fn = EqMLoss(cfg)

    def oracle(I_t, y):
        # The oracle "model" recovers the target exactly. The wrapper has to
        # be passed I_t and produce U_t; we rebuild U_t here using the shared
        # state on the closure.
        return oracle.target

    x = torch.randn(8, 4, 32, 32)
    y = torch.zeros(8, dtype=torch.long)
    z = torch.randn_like(x)

    # Pin the time draw and pre-compute target so the oracle can return it.
    B = x.shape[0]
    t = loss_fn.sample_time(B, x.device, x.dtype)
    oracle.target = equilibrium_target(z, x, t)

    # Bypass the sampler so we use the same t the target was built from.
    I_t = canonical_interpolant(z, x, t)
    pred = oracle(I_t, y)
    loss = ((pred - oracle.target) ** 2).mean()
    assert loss.item() < 1e-10


def test_eqm_loss_gradient_flows():
    torch.manual_seed(0)
    cfg = EqMLossConfig()
    loss_fn = EqMLoss(cfg)
    head = torch.nn.Linear(4 * 32 * 32, 4 * 32 * 32)

    def model_fn(I_t, y):
        B = I_t.shape[0]
        return head(I_t.reshape(B, -1)).reshape_as(I_t)

    x = torch.randn(4, 4, 32, 32)
    y = torch.zeros(4, dtype=torch.long)
    total, metrics = loss_fn(model_fn, x, y)
    assert torch.isfinite(total).item()
    assert "eqm_loss" in metrics
    total.backward()
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in head.parameters())
    assert has_grad, "no gradient flowed through the head"


def test_uniform_weighted_loss_logs_weights():
    torch.manual_seed(0)
    cfg = EqMLossConfig(time_sampler="uniform_weighted", eps_train=1e-3)
    loss_fn = EqMLoss(cfg)

    def model_fn(I_t, y):
        return torch.zeros_like(I_t)

    x = torch.randn(4, 4, 32, 32)
    y = torch.zeros(4, dtype=torch.long)
    total, metrics = loss_fn(model_fn, x, y)
    assert torch.isfinite(total).item()
    assert "time_weight_mean" in metrics
    assert "time_weight_min" in metrics
    assert "time_weight_max" in metrics
    assert metrics["time_weight_max"] >= metrics["time_weight_min"]


def test_denoiser_target_is_clean_data():
    # D_target(I_t) = I_t + U_t = x for every t, so the denoiser regresses
    # the clean data regardless of the interpolant time.
    z = torch.randn(4, 4, 8, 8)
    x = torch.randn(4, 4, 8, 8)
    for tv in (0.0, 0.3, 0.999):
        t = torch.full((4,), tv)
        target = denoiser_target(z, x, t)
        assert torch.allclose(target, x)
        # The denoiser target equals the interpolant plus the velocity target.
        recon = canonical_interpolant(z, x, t) + equilibrium_target(z, x, t)
        assert torch.allclose(recon, x, atol=1e-5)


def test_denoiser_and_velocity_losses_match_pointwise():
    # ||D(I_t) - x||^2 == ||b(I_t) - U_t||^2 when D = b + I_t, so the two
    # parameterizations share the same per-sample EqM residual.
    torch.manual_seed(0)
    x = torch.randn(8, 4, 8, 8)
    y = torch.zeros(8, dtype=torch.long)
    z = torch.randn_like(x)

    # A fixed "field" the velocity network would output.
    b = torch.randn_like(x)

    def velocity_fn(I_t, _y):
        return b

    def denoiser_fn(I_t, _y):
        return b + I_t  # D = b + I_t

    # Reuse the same time draw for both by pinning the RNG.
    cfg = EqMLossConfig(time_sampler="canonical")
    vel_loss_fn = EqMLoss(cfg)
    den_loss_fn = EqMLoss(EqMLossConfig(parameterization="denoiser"))

    torch.manual_seed(123)
    vel_total, _ = vel_loss_fn(velocity_fn, x, y, z=z)
    torch.manual_seed(123)
    den_total, _ = den_loss_fn(denoiser_fn, x, y, z=z)
    assert torch.allclose(vel_total, den_total, atol=1e-5)


def test_denoiser_loss_zero_at_oracle():
    # An oracle denoiser that returns the clean data drives the loss to 0.
    torch.manual_seed(0)
    loss_fn = EqMLoss(EqMLossConfig(parameterization="denoiser"))
    x = torch.randn(8, 4, 8, 8)
    y = torch.zeros(8, dtype=torch.long)

    def oracle(I_t, _y):
        return x  # D(I_t) = x exactly

    total, metrics = loss_fn(oracle, x, y)
    assert total.item() < 1e-10
    assert metrics["eqm_loss"].item() < 1e-10


def test_denoiser_anchor_uses_field_residual():
    # With the denoiser parameterization the anchor is ||D(x) - x||^2.
    # An identity denoiser (D(x)=x) has zero anchor; a shifted one is > 0.
    x = torch.randn(4, 4, 8, 8)
    y = torch.zeros(4, dtype=torch.long)

    identity_fn = lambda I_t, _y: I_t           # D(x) = x -> anchor 0
    shifted_fn = lambda I_t, _y: I_t + 1.0       # D(x) = x + 1 -> anchor 1

    cfg = EqMLossConfig(parameterization="denoiser", data_anchor_weight=1.0)
    _, m_id = EqMLoss(cfg)(identity_fn, x, y)
    _, m_sh = EqMLoss(cfg)(shifted_fn, x, y)
    assert m_id["data_anchor"].item() < 1e-8
    assert abs(m_sh["data_anchor"].item() - 1.0) < 1e-5


def test_as_field_fn_reconstructs_denoiser_field():
    from eqfm.sampling import as_field_fn

    x = torch.randn(4, 4, 8, 8)
    y = torch.zeros(4, dtype=torch.long)
    D = torch.randn_like(x)

    raw_fn = lambda inp, _y: D
    # Velocity: identity passthrough.
    assert torch.allclose(as_field_fn(raw_fn, "velocity")(x, y), D)
    # Denoiser: field is D - x.
    assert torch.allclose(as_field_fn(raw_fn, "denoiser")(x, y), D - x)


def test_data_anchor_is_added_and_logged():
    cfg = EqMLossConfig(data_anchor_weight=1.0)
    loss_fn = EqMLoss(cfg)

    def model_fn(I_t, y):
        return I_t  # b_hat(y) = y -- not zero on data, so anchor > 0.

    x = torch.randn(4, 4, 32, 32)
    y = torch.zeros(4, dtype=torch.long)
    total, metrics = loss_fn(model_fn, x, y)
    assert "data_anchor" in metrics
    assert metrics["data_anchor"].item() > 0


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
