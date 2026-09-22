"""Thin wrapper around the Stable Diffusion VAE used by SiT/DiT.

SiT-XL/2 was trained on 32x32x4 latents produced by ``stabilityai/sd-vae-ft-ema``
with the canonical scaling factor ``0.18215``. We keep the VAE frozen, run it
in inference mode, and only need the encoder for Stage-1 EqM training (decode
will be needed once we sample images).

The Huggingface download honours ``HF_HOME``; set that alongside
``EQFM_CACHE_DIR`` on hosts where ``$HOME`` has no room (the L40S box).
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

try:
    from diffusers import AutoencoderKL
except ImportError as e:  # pragma: no cover - environment misconfiguration
    raise ImportError(
        "diffusers is required for the VAE wrapper. "
        "Install it via `pip install -r requirements.txt`."
    ) from e


DEFAULT_VAE_ID = "stabilityai/sd-vae-ft-ema"
DEFAULT_SCALING_FACTOR = 0.18215


class LatentEncoder(nn.Module):
    """Frozen SD VAE; encodes pixel-space images to scaled latents.

    Latents follow the standard convention: ``z = vae.encode(x).latent_dist.sample()
    * scaling_factor``. The scaling factor is bundled with the VAE checkpoint and
    is what the pretrained SiT was trained against.
    """

    def __init__(
        self,
        vae_id: str = DEFAULT_VAE_ID,
        scaling_factor: float = DEFAULT_SCALING_FACTOR,
        cache_dir: Optional[str] = None,
    ):
        super().__init__()
        self.vae = AutoencoderKL.from_pretrained(vae_id, cache_dir=cache_dir)
        self.vae.eval()
        for p in self.vae.parameters():
            p.requires_grad_(False)
        self.scaling_factor = scaling_factor

    @torch.no_grad()
    def encode(
        self,
        images: torch.Tensor,
        *,
        sample: bool = True,
    ) -> torch.Tensor:
        """Encode pixel images in ``[-1, 1]`` to scaled latents.

        Args:
            images: ``(B, 3, H, W)`` float tensor in ``[-1, 1]``.
            sample: If ``True``, draw from the encoder's posterior (training
                convention). If ``False``, use the posterior mean (deterministic
                eval / sampling).

        Returns:
            ``(B, 4, H/8, W/8)`` latents on the same device/dtype as ``images``.
        """
        dist = self.vae.encode(images).latent_dist
        z = dist.sample() if sample else dist.mean
        return z * self.scaling_factor

    @torch.no_grad()
    def encode_stats(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Posterior ``(mean, std)`` of the encoder, *unscaled* (no 0.18215).

        Used to build the latent cache: ``sample_from_stats(mean, std)`` then
        reproduces ``encode(images, sample=True)`` exactly in distribution.
        """
        dist = self.vae.encode(images).latent_dist
        return dist.mean, dist.std

    @staticmethod
    def sample_from_stats(
        mean: torch.Tensor,
        std: torch.Tensor,
        *,
        scaling_factor: float = DEFAULT_SCALING_FACTOR,
        sample: bool = True,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """``(mean + std * eps) * scaling_factor`` with ``eps ~ N(0, I)`` in fp32.

        Mirrors :meth:`encode`: same posterior sample, same scaling. ``sample=False``
        returns the scaled posterior mean (eval convention).
        """
        mean = mean.float()
        if not sample:
            return mean * scaling_factor
        eps = torch.randn(mean.shape, device=mean.device, dtype=mean.dtype, generator=generator)
        return (mean + std.float() * eps) * scaling_factor

    @torch.no_grad()
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode latents back to pixel images in ``[-1, 1]``."""
        return self.vae.decode(latents / self.scaling_factor).sample
