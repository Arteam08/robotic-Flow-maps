"""Offline sanity tests for the FID math.

No GPU, no Inception download -- we only exercise the Frechet distance
and the stats helper, which are pure numpy/scipy.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from eqfm.metrics.fid import fid_from_stats, stats_from_features  # noqa: E402


def test_fid_zero_for_identical_stats():
    rng = np.random.default_rng(0)
    feats = rng.normal(size=(512, 64))
    mu, sigma = stats_from_features(feats)
    fid = fid_from_stats(mu, sigma, mu, sigma)
    assert abs(fid) < 1e-6, f"FID of identical stats should be ~0, got {fid}"


def test_fid_positive_for_shifted_mean():
    rng = np.random.default_rng(1)
    feats = rng.normal(size=(512, 64))
    mu, sigma = stats_from_features(feats)
    mu_shift = mu + 3.0
    fid = fid_from_stats(mu, sigma, mu_shift, sigma)
    # Pure mean shift by d in every dim of D dims -> FID == ||delta||^2.
    assert fid > 0
    assert abs(fid - (3.0 ** 2) * mu.shape[0]) < 1.0, (
        f"mean-shift FID should be ~||delta||^2 = {(3.0**2)*mu.shape[0]}, "
        f"got {fid}"
    )


def test_fid_monotone_in_mean_gap():
    rng = np.random.default_rng(2)
    feats = rng.normal(size=(512, 32))
    mu, sigma = stats_from_features(feats)
    f_small = fid_from_stats(mu, sigma, mu + 0.5, sigma)
    f_large = fid_from_stats(mu, sigma, mu + 2.0, sigma)
    assert f_large > f_small > 0


def test_stats_shape():
    rng = np.random.default_rng(3)
    feats = rng.normal(size=(100, 16))
    mu, sigma = stats_from_features(feats)
    assert mu.shape == (16,)
    assert sigma.shape == (16, 16)


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
