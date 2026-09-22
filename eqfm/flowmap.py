"""Compactified equilibrium flow-map utilities.

Stage 2 learns the compactified family

    X_sigma(y) = y + sigma * v_sigma(y),    sigma in [0, 1],

where ``v_sigma`` is represented by a SiT-style network taking ``sigma`` as
its scalar time input.  The terminal projector is ``P(y) = X_1(y)``.
"""
from __future__ import annotations

import contextlib

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as _DDP


@contextlib.contextmanager
def _math_sdp_for_forward_ad():
    """Disable fused SDPA kernels while running forward-mode AD.

    PyTorch's flash/memory-efficient scaled-dot-product attention kernels do
    not currently implement forward AD. SiT/timm attention calls
    ``F.scaled_dot_product_attention`` and lets PyTorch choose the backend, so
    JVPs through SiT can accidentally select flash attention and crash.  The
    math backend is slower but supports the derivatives we need for Stage 2
    LESD/EESD.  This context is intentionally scoped only around JVP calls.

    Only the *selection* of the math backend is guarded by try/except (the new
    ``sdpa_kernel`` API vs. the legacy ``sdp_kernel`` fallback). The ``yield``
    happens exactly once, outside the try, so an exception raised inside the
    JVP body propagates cleanly instead of being masked by
    ``RuntimeError: generator didn't stop after throw()``.
    """
    if not torch.cuda.is_available():
        yield
        return
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
        cm = sdpa_kernel([SDPBackend.MATH])
    except Exception:
        cm = torch.backends.cuda.sdp_kernel(
            enable_flash=False,
            enable_mem_efficient=False,
            enable_math=True,
        )
    with cm:
        yield


@contextlib.contextmanager
def _jvp_attention(model: nn.Module, device_type: str):
    """Attention backend for a JVP through ``model``.

    fp32 (autocast off): the math SDPA backend, exactly as before. Under bf16
    autocast the math backend's ``_safe_softmax`` backward fails on the
    dual-dtype graph ("expected input and grad types to match"), so timm
    attention modules switch to their explicit ``softmax(q k^T) v`` path for
    the duration of the JVP (``fused_attn = False``); autocast runs that
    softmax in fp32 with a regular, dtype-consistent backward.
    """
    if not torch.is_autocast_enabled(device_type):
        with _math_sdp_for_forward_ad():
            yield
        return
    mods = [m for m in model.modules() if getattr(m, "fused_attn", None) is True]
    for m in mods:
        m.fused_attn = False
    try:
        yield
    finally:
        for m in mods:
            m.fused_attn = True


def _unwrap_for_jvp(model: nn.Module) -> nn.Module:
    """Return the underlying module for forward-mode AD under DDP.

    ``torch.func.jvp`` does not compose with ``DistributedDataParallel``'s
    ``forward`` wrapper -- putting a dual number on the input tensor that
    DDP.forward processes crashes (spatial_jvp under multi-GPU). We run the
    JVP on the wrapped module instead. Gradient synchronization is preserved:
    the DDP model and its ``.module`` share the same parameter tensors, and the
    diagonal/terminal passes (which DO go through ``DDP.forward`` every step)
    arm DDP's reducer, so the JVP's locally-accumulated gradient contributions
    are all-reduced together with the rest.
    """
    return model.module if isinstance(model, _DDP) else model


def broadcast_time(t: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Right-pad a batch vector so it broadcasts over ``ref``."""
    while t.ndim < ref.ndim:
        t = t.unsqueeze(-1)
    return t


def compactified_add(sigma_a: torch.Tensor, sigma_b: torch.Tensor) -> torch.Tensor:
    """Exponential-clock composition ``sigma_a ⊕ sigma_b``.

    With ``sigma = 1 - exp(-tau)``, adding autonomous durations gives
    ``sigma_a + sigma_b - sigma_a * sigma_b``.
    """
    return sigma_a + sigma_b - sigma_a * sigma_b


def compactified_sub(sigma: torch.Tensor, sigma_prime: torch.Tensor) -> torch.Tensor:
    """Return ``sigma_minus`` such that ``sigma_prime ⊕ sigma_minus = sigma``."""
    return (sigma - sigma_prime) / (1.0 - sigma_prime).clamp_min(1e-12)


def flow_map(model: nn.Module, x: torch.Tensor, sigma: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Evaluate ``X_sigma(x) = x + sigma * v_sigma(x, y)``."""
    v = model(x, sigma, y)
    if v.shape != x.shape:
        raise ValueError(
            "Stage-2 flow-map velocity must have the same shape as the input "
            f"latent; got velocity {tuple(v.shape)} for input {tuple(x.shape)}. "
            "Check that the student model is velocity-only, not learn_sigma/2C."
        )
    return x + broadcast_time(sigma, x) * v


def sigma_derivative(
    model: nn.Module,
    x: torch.Tensor,
    sigma: torch.Tensor,
    y: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(X_sigma(x), d/dsigma X_sigma(x))`` via forward-mode JVP."""
    model = _unwrap_for_jvp(model)

    def fn(sig: torch.Tensor) -> torch.Tensor:
        return flow_map(model, x, sig, y)

    tangent = torch.ones_like(sigma)
    with _jvp_attention(model, x.device.type):
        return torch.func.jvp(fn, (sigma,), (tangent,))


def spatial_jvp(
    model: nn.Module,
    x: torch.Tensor,
    sigma: torch.Tensor,
    y: torch.Tensor,
    tangent: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(X_sigma(x), D_x X_sigma(x) @ tangent)``."""
    model = _unwrap_for_jvp(model)

    def fn(inp: torch.Tensor) -> torch.Tensor:
        return flow_map(model, inp, sigma, y)

    with _math_sdp_for_forward_ad():
        return torch.func.jvp(fn, (x,), (tangent,))


def velocity_jvp(
    model: nn.Module,
    x: torch.Tensor,
    sigma: torch.Tensor,
    y: torch.Tensor,
    x_tangent: torch.Tensor,
    sigma_tangent: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(v_sigma(x), D v_sigma . (x_tangent, sigma_tangent))``.

    JVP through the *raw velocity* ``v_sigma(x) = model(x, sigma, y)`` (not the
    flow map ``X_sigma``). With ``x_tangent = b0`` (teacher velocity at x) and
    ``sigma_tangent = 1`` this yields the MeanFlow total derivative
    ``d/dsigma v = partial_sigma v + (D_x v) . b0`` in a single forward-AD pass,
    alongside the primal velocity. Callers that want a stop-gradient target
    should run this under ``torch.no_grad()``.
    """
    model = _unwrap_for_jvp(model)

    def fn(inp: torch.Tensor, sig: torch.Tensor) -> torch.Tensor:
        return model(inp, sig, y)

    with _math_sdp_for_forward_ad():
        return torch.func.jvp(fn, (x, sigma), (x_tangent, sigma_tangent))
