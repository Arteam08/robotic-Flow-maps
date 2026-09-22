"""Classifier-free-guidance combination rules for autonomous fields.

Single source of truth used by both the Stage-2 teacher (``eqfm/stage2_losses.py``)
and the Stage-1 sampler (``eqfm/sampling.py``).  Notation: ``b_c`` conditional
field, ``b_u`` unconditional field, ``w`` guidance scale, ``s = w - 1``.

    standard      g = b_u + w (b_c - b_u)
    perp          g = b_c - s * b_u_perp,                b_u_perp = b_u - (b_u . c^) c^,  c^ = b_c / ||b_c||
    perp + cnorm  g = b_c - s ||b_c|| * b_u_perp / ||b_u_perp||        (rescales a possibly tiny remainder: can blow up)
    perp_matched  g = b_c - s ||b_c|| * b_u_perp / ||b_u||             (normalise b_u to ||b_c|| FIRST, then take the
                                                                        perpendicular part; ||g - b_c|| <= s ||b_c|| always)

``perp_matched`` equals ``match_uncond_norm`` followed by plain ``perp``; it is exposed as its own rule so it can be
selected without the separate norm-matching flag and so the bound above holds by construction.
"""
from __future__ import annotations

import warnings

import torch

CFG_RULES = ("standard", "perp", "perp_matched")


def warn_if_perp_cnorm(rule: str, cnorm: bool) -> None:
    """``perp`` + ``cnorm`` divides by ||b_u_perp||, which blows up (ImageNet FID 28.6 at w=3); use ``perp_matched``."""
    if rule == "perp" and cnorm:
        warnings.warn(
            "cfg rule 'perp' with cnorm rescales a possibly tiny perpendicular remainder and can blow up; "
            "use cfg rule 'perp_matched' (bounded by (w-1)||b_c||) instead.",
            stacklevel=3,
        )


def match_uncond_norm(b_cond: torch.Tensor, b_uncond: torch.Tensor) -> torch.Tensor:
    """Rescale ``b_uncond`` per sample so that ``||b_uncond|| = ||b_cond||``."""
    cond_norm = b_cond.float().flatten(1).norm(dim=1)
    uncond_norm = b_uncond.float().flatten(1).norm(dim=1).clamp_min(1e-12)
    scale = (cond_norm / uncond_norm).view(-1, *([1] * (b_uncond.ndim - 1)))
    return b_uncond * scale.to(dtype=b_uncond.dtype)


def perp_guided(
    b_cond: torch.Tensor,
    b_uncond: torch.Tensor,
    cfg_scale: float,
    *,
    cnorm: bool = False,
    match_uncond_norm_first: bool = False,
) -> torch.Tensor:
    """``b_c - s * u`` with ``u`` the component of ``b_u`` perpendicular to ``b_c`` (``s = cfg_scale - 1``).

    ``match_uncond_norm_first``: rescale ``b_u`` to ``||b_c||`` before projecting (the ``perp_matched`` rule).
    ``cnorm``: rescale the perpendicular part itself to ``s * ||b_c||`` (the original ``--cfg-cnorm``).
    """
    s = cfg_scale - 1.0
    if match_uncond_norm_first:
        b_uncond = match_uncond_norm(b_cond, b_uncond)
    flat_c = b_cond.float().flatten(1)
    nc = flat_c.norm(dim=1).clamp_min(1e-12)
    shape = (-1, *([1] * (b_cond.ndim - 1)))
    c_hat = (flat_c / nc.unsqueeze(1)).view_as(b_cond)
    dot = (b_uncond.float().flatten(1) * c_hat.flatten(1)).sum(dim=1).view(shape)
    u = b_uncond.float() - dot * c_hat                       # perpendicular part of b_u
    if cnorm:
        un = u.flatten(1).norm(dim=1).clamp_min(1e-12).view(shape)
        u = u / un * nc.view(shape)                          # unit-free direction, length ||b_c||
    return (b_cond.float() - s * u).to(dtype=b_cond.dtype)


def combine_cfg(
    b_cond: torch.Tensor,
    b_uncond: torch.Tensor,
    cfg_scale: float,
    rule: str = "standard",
    *,
    cnorm: bool = False,
    cfg_mode: str = "all",
) -> torch.Tensor:
    """Apply one of ``CFG_RULES``. ``cfg_mode="first3"`` (standard rule only) guides channels ``[:3]``."""
    if rule == "standard":
        guided = b_uncond + cfg_scale * (b_cond - b_uncond)
        if cfg_mode == "all":
            return guided
        out = b_cond.clone()
        out[:, :3] = guided[:, :3]
        return out
    if rule == "perp":
        return perp_guided(b_cond, b_uncond, cfg_scale, cnorm=cnorm)
    if rule == "perp_matched":
        return perp_guided(b_cond, b_uncond, cfg_scale, cnorm=False, match_uncond_norm_first=True)
    raise ValueError(f"unknown cfg rule {rule!r}; expected one of {CFG_RULES}")
