"""Frechet Inception Distance primitives.

The Frechet distance + stats helpers are pure numpy/scipy so they're
unit-testable without a GPU or the Inception weights. The Inception
feature extractor lazily imports ``pytorch-fid`` only when actually
constructed (i.e. on the eval host), so importing this module is cheap
and dependency-light.

Caveats this module does NOT hide from you:

  * Numbers are only comparable against a reference computed with the
    *same* Inception + the *same* reference image set. We compute our
    reference from the project's own ImageNet WebDataset shards, so our
    FID is internally consistent (good for tracking training and
    comparing our own checkpoints) but is NOT the number SiT/DiT
    report -- those use the ADM TensorFlow evaluator and the
    VIRTUAL_imagenet256 reference. Plug an ADM ``.npz`` into the eval
    script's ``--ref-stats`` if you need that parity.
  * FID is biased upward at small sample counts. Use a fixed N when
    comparing checkpoints; only the 50k number is "report-grade".
"""
from __future__ import annotations

from typing import Tuple

import numpy as np


def _frechet_distance(
    mu1: np.ndarray, sigma1: np.ndarray,
    mu2: np.ndarray, sigma2: np.ndarray,
    eps: float = 1e-6,
) -> float:
    """Frechet distance between N(mu1, sigma1) and N(mu2, sigma2).

    Identical to the formula pytorch-fid / the original FID paper use:
    ||mu1 - mu2||^2 + Tr(C1 + C2 - 2 (C1 C2)^{1/2}), with the standard
    eps fallback when the matrix sqrt picks up imaginary components from
    numerical error.
    """
    from scipy import linalg

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    return float(
        diff.dot(diff)
        + np.trace(sigma1)
        + np.trace(sigma2)
        - 2.0 * np.trace(covmean)
    )


class InceptionFeatureExtractor:
    """Wraps the FID InceptionV3 to emit 2048-d pool features.

    Input images are expected in ``[0, 1]`` of shape ``(B, 3, H, W)``.
    ``pytorch-fid``'s InceptionV3 resizes to 299x299 and applies its own
    [0,1] -> [-1,1] normalization internally, so we only need to hand it
    clamped [0,1] tensors. ``pytorch-fid`` is imported here, lazily, so
    this module imports fine on hosts without it.
    """

    DIM = 2048

    def __init__(self, device):
        import torch  # local: keep module import torch-free for the math path

        try:
            from pytorch_fid.inception import InceptionV3
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "pytorch-fid is required for Inception feature extraction. "
                "Install it via `pip install -r requirements.txt`."
            ) from e

        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[self.DIM]
        self.model = InceptionV3([block_idx]).to(device).eval()
        self.device = device
        self._torch = torch

    def __call__(self, images) -> np.ndarray:
        """Return ``(B, 2048)`` float64 features for a ``[0, 1]`` image batch."""
        torch = self._torch
        with torch.no_grad():
            images = images.to(self.device).clamp(0.0, 1.0)
            feat = self.model(images)[0]            # (B, 2048, 1, 1)
            feat = feat.squeeze(3).squeeze(2)       # (B, 2048)
            return feat.double().cpu().numpy()


def stats_from_features(feats: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the activation Gaussian ``(mu, sigma)`` from ``(N, 2048)``."""
    if feats.ndim != 2:
        raise ValueError(f"Expected (N, D) features, got shape {feats.shape}.")
    mu = feats.mean(axis=0)
    sigma = np.cov(feats, rowvar=False)
    return mu, sigma


def fid_from_stats(
    mu1: np.ndarray, sigma1: np.ndarray,
    mu2: np.ndarray, sigma2: np.ndarray,
) -> float:
    """Frechet distance between two activation Gaussians."""
    return _frechet_distance(mu1, sigma1, mu2, sigma2)
