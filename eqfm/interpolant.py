"""Canonical manifold-landing interpolant (Boffi 2026, Def. 3.3).

We use the linear interpolant under an exponential clock, which is the
default the paper uses throughout the main text after Section 3:

    I_t = (1 - t) * z + t * x,        t in [0, 1)
    c(t) = 1 - t
    U_t  = (1 - t) * (x - z)

where ``z ~ rho_0`` (noise prior), ``x ~ pi_M`` (data on the manifold), and
``U_t = c(t) * dI_t/dt`` is the *autonomous-time* velocity used as the
regression target for the equilibrium field b. Note that ``U_t -> 0`` as
``t -> 1`` (the manifold-landing condition that bakes the data in as the
fixed-point set of b).
"""
from __future__ import annotations

import torch


def _broadcast_t(t: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Right-pad ``t`` with singleton axes to broadcast against ``ref``."""
    while t.ndim < ref.ndim:
        t = t.unsqueeze(-1)
    return t


def canonical_interpolant(
    z: torch.Tensor, x: torch.Tensor, t: torch.Tensor
) -> torch.Tensor:
    """Compute ``I_t = (1 - t) * z + t * x`` (Eq. 14).

    Args:
        z: Noise sample, shape ``(B, *S)``.
        x: Data sample (already in latent space if training in latents),
            shape ``(B, *S)``.
        t: Time in ``[0, 1]``, shape ``(B,)``.

    Returns:
        Tensor of shape ``(B, *S)``.
    """
    t = _broadcast_t(t, z)
    return (1.0 - t) * z + t * x


def equilibrium_target(
    z: torch.Tensor, x: torch.Tensor, t: torch.Tensor
) -> torch.Tensor:
    """Autonomous-time velocity target ``U_t = (1 - t) * (x - z)`` (Eq. 16).

    This is the regression target for the equilibrium field b in the
    sampler form of the EqM loss (Eq. 22 of the paper).
    """
    t = _broadcast_t(t, z)
    return (1.0 - t) * (x - z)


def denoiser_target(
    z: torch.Tensor, x: torch.Tensor, t: torch.Tensor
) -> torch.Tensor:
    """Denoiser / x-prediction target for the autonomous field.

    The autonomous field is reparameterized as ``dot{x}_t = D(x_t) - x_t``
    with the data manifold as the fixed-point set ``D(x) = x``. Equating
    this field with the velocity field ``b`` gives ``b(x) = D(x) - x``, so
    at the interpolant point the denoiser regression target is

        D_target(I_t) = I_t + U_t
                      = [(1 - t) z + t x] + (1 - t)(x - z)
                      = x,

    i.e. the *clean data* regardless of ``t``. The per-sample residual is
    numerically identical to the velocity loss, since

        || D(I_t) - x ||^2 = || (b(I_t) + I_t) - (U_t + I_t) ||^2
                           = || b(I_t) - U_t ||^2,

    so every time sampler / importance weight in :mod:`eqfm.losses` carries
    over unchanged. What differs is the *raw network output*: it learns the
    absolute clean target ``x`` (which stays O(1) near the manifold) instead
    of the velocity ``U_t`` (which collapses to 0), analogous to x0- vs
    v-prediction in diffusion.

    ``z`` and ``t`` are accepted for a signature consistent with
    :func:`equilibrium_target`; the target itself is just ``x``.
    """
    return x
