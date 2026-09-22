"""Compute FID from an existing folder or .npz of generated images.

This is useful when sampling was already run and you want to compare those
exact images against an ADM/ImageNet reference batch without regenerating.

Inputs:
  * --samples: folder of PNG/JPEG images, or .npz with arr_0/images.
  * --ref: either stats .npz with mu/sigma, or raw-image .npz with arr_0.

The feature extractor is still this repo's pytorch-fid Inception wrapper, so
the result is internally consistent with our scripts. It is not the original
ADM TensorFlow evaluator unless you run that evaluator separately.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Iterable, List

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from eqfm.metrics import InceptionFeatureExtractor, fid_from_stats, stats_from_features  # noqa: E402


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--samples", type=Path, required=True,
                   help="Generated sample folder or .npz.")
    p.add_argument("--ref", type=Path, required=True,
                   help="Reference .npz. Accepts mu/sigma stats or raw images in arr_0.")
    p.add_argument("--ref-kind", choices=["auto", "stats", "raw"], default="auto",
                   help="How to interpret --ref. auto prefers mu/sigma when "
                        "present. raw forces feature extraction from arr_0/images, "
                        "which is required for an exact same-file self-check.")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--num-samples", type=int, default=None,
                   help="Optional cap. Defaults to all generated images found.")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--log-every", type=int, default=20)
    return p.parse_args()


def _image_paths(root: Path) -> List[Path]:
    paths = sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
    if not paths:
        raise FileNotFoundError(f"No images found under {root}")
    return paths


def _pil_to_tensor(path: Path) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


def _np_images_to_tensor(arr: np.ndarray) -> torch.Tensor:
    """Convert NHWC uint8/float or NCHW arrays to BCHW [0,1]."""
    if arr.ndim != 4:
        raise ValueError(f"Expected image array with 4 dims, got {arr.shape}")
    x = torch.from_numpy(arr)
    if x.shape[-1] == 3:
        x = x.permute(0, 3, 1, 2)
    elif x.shape[1] != 3:
        raise ValueError(f"Could not infer channels for image array shape {arr.shape}")
    x = x.float()
    if x.max() > 2.0:
        x = x / 255.0
    return x.clamp(0.0, 1.0)


def _npz_image_array(path: Path) -> np.ndarray:
    d = np.load(path)
    if "arr_0" in d.files:
        return d["arr_0"]
    for key in ("images", "samples", "x"):
        if key in d.files:
            return d[key]
    raise KeyError(f"{path} has keys {d.files}, but no arr_0/images/samples/x")


@torch.no_grad()
def stats_from_image_folder(paths: List[Path], n: int, batch_size: int, incep) -> tuple[np.ndarray, np.ndarray]:
    feats = []
    t0 = time.time()
    for start in range(0, n, batch_size):
        chunk = paths[start:start + batch_size]
        images = torch.stack([_pil_to_tensor(p) for p in chunk], dim=0)
        feats.append(incep(images))
        seen = min(start + len(chunk), n)
        batch_idx = start // batch_size + 1
        if batch_idx == 1 or batch_idx % 20 == 0 or seen >= n:
            print(f"[fid-existing]   samples {seen}/{n} "
                  f"({seen/max(time.time()-t0,1e-6):.1f} img/s)", flush=True)
    return stats_from_features(np.concatenate(feats, axis=0)[:n])


@torch.no_grad()
def stats_from_image_npz(path: Path, n: int, batch_size: int, incep, *, tag: str) -> tuple[np.ndarray, np.ndarray]:
    arr = _npz_image_array(path)
    n = min(n, arr.shape[0])
    feats = []
    t0 = time.time()
    for start in range(0, n, batch_size):
        images = _np_images_to_tensor(arr[start:start + batch_size])
        feats.append(incep(images))
        seen = min(start + images.shape[0], n)
        batch_idx = start // batch_size + 1
        if batch_idx == 1 or batch_idx % 20 == 0 or seen >= n:
            print(f"[fid-existing]   {tag} {seen}/{n} "
                  f"({seen/max(time.time()-t0,1e-6):.1f} img/s)", flush=True)
    return stats_from_features(np.concatenate(feats, axis=0)[:n])


def load_ref_stats(path: Path, batch_size: int, incep, n_cap: int | None, ref_kind: str):
    d = np.load(path)
    if ref_kind in {"auto", "stats"} and "mu" in d.files and "sigma" in d.files:
        return d["mu"], d["sigma"], "stats"
    if ref_kind == "stats":
        raise KeyError(f"{path} has keys {d.files}, but no mu/sigma stats")
    n = d["arr_0"].shape[0] if "arr_0" in d.files else _npz_image_array(path).shape[0]
    if n_cap is not None:
        n = min(n, n_cap)
    mu, sigma = stats_from_image_npz(path, n, batch_size, incep, tag="ref")
    return mu, sigma, "raw_images"


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    incep = InceptionFeatureExtractor(device)

    print(f"[fid-existing] samples={args.samples}")
    if args.samples.is_dir():
        paths = _image_paths(args.samples)
        n = min(args.num_samples or len(paths), len(paths))
        print(f"[fid-existing] found {len(paths)} sample images, using {n}")
        mu_gen, sigma_gen = stats_from_image_folder(paths[:n], n, args.batch_size, incep)
        sample_source = "folder"
    else:
        arr = _npz_image_array(args.samples)
        n = min(args.num_samples or arr.shape[0], arr.shape[0])
        print(f"[fid-existing] found sample npz array {arr.shape}, using {n}")
        mu_gen, sigma_gen = stats_from_image_npz(args.samples, n, args.batch_size, incep, tag="samples")
        sample_source = "npz"

    print(f"[fid-existing] ref={args.ref}")
    mu_ref, sigma_ref, ref_kind = load_ref_stats(
        args.ref, args.batch_size, incep, args.num_samples, args.ref_kind
    )
    fid = fid_from_stats(mu_gen, sigma_gen, mu_ref, sigma_ref)
    print(f"[fid-existing] FID = {fid:.4f}")

    np.savez(args.output_dir / "generated_stats.npz", mu=mu_gen, sigma=sigma_gen)
    out = {
        "fid": fid,
        "samples": str(args.samples),
        "sample_source": sample_source,
        "ref": str(args.ref),
        "ref_kind": ref_kind,
        "num_samples": n,
        "batch_size": args.batch_size,
    }
    with (args.output_dir / "fid.json").open("w") as f:
        json.dump(out, f, indent=2, sort_keys=True)
    print(f"[fid-existing] wrote {args.output_dir / 'fid.json'}")


if __name__ == "__main__":
    main()
