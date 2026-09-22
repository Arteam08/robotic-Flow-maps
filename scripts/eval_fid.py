"""Measure FID of a Stage-1 EqM checkpoint.

Pipeline:
  1. Load SiT-XL/2 + EMA from a checkpoint, load the SD VAE.
  2. Reference stats: stream real ImageNet from the WebDataset shards,
     extract Inception features over --ref-num images, cache (mu, sigma)
     to an .npz so subsequent runs skip this. (Or pass an existing
     --ref-stats .npz, e.g. an ADM VIRTUAL_imagenet256 reference, for
     parity with published SiT/DiT numbers.)
  3. Generate --num-samples class-balanced samples via forward-Euler
     integration of the autonomous field, decode through the VAE.
  4. Inception features -> (mu, sigma) -> Frechet distance vs reference.
  5. Write fid.json.

Honest caveats (see eqfm/metrics/fid.py docstring for the full list):
  * Self-computed reference != ADM evaluator reference. Good for tracking
    our own training; not the same scale as SiT's reported 2.06.
  * --cfg-scale > 1 assumes the checkpoint was trained with label dropout
    to the null class. Current training exposes this as --cfg-dropout-prob.
    For older checkpoints, verify CFG behavior with a sweep before treating
    guided FID as more than a relative diagnostic.
  * FID is upward-biased at small N. Compare checkpoints at a *fixed* N;
    only --num-samples 50000 is report-grade.

Usage:
    python scripts/eval_fid.py \
        --ckpt /scratch/$USER/eqfm-ckpts/step_0060000.pt \
        --output-dir /scratch/$USER/eqfm-fid/step60k \
        --num-samples 10000 --num-steps 250
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from eqfm.data import build_imagenet_loader, DEFAULT_TRAIN_SHARDS  # noqa: E402
from eqfm.metrics import (  # noqa: E402
    InceptionFeatureExtractor, fid_from_stats, stats_from_features,
)
from eqfm.models.sit import SiT_XL_2  # noqa: E402
from eqfm.sampling import as_field_fn, sample_euler, stop_time_from_eps  # noqa: E402
from eqfm.training import EMA  # noqa: E402
from eqfm.vae import LatentEncoder  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)

    # Generation
    p.add_argument("--num-samples", type=int, default=10000,
                   help="Generated images for FID. 10k = fast tracking, "
                        "50k = report-grade. FID is biased upward at low N; "
                        "keep this fixed when comparing checkpoints.")
    p.add_argument("--save-samples-npz", type=Path, default=None,
                   help="Save generated samples as an OpenAI evaluator-compatible "
                        ".npz with key arr_0, shape NHWC uint8 in [0,255].")
    p.add_argument("--overwrite-samples-npz", action="store_true", default=False)
    p.add_argument("--skip-pytorch-fid", action="store_true", default=False,
                   help="Only generate/save samples, skipping this script's "
                        "pytorch-fid feature extraction.")
    p.add_argument("--oai-evaluator", type=Path, default=None,
                   help="Path to OpenAI guided-diffusion/evaluations/evaluator.py. "
                        "If set, runs it after saving --save-samples-npz.")
    p.add_argument("--oai-ref", type=Path, default=None,
                   help="OpenAI evaluator reference batch, e.g. "
                        "VIRTUAL_imagenet256_labeled.npz.")
    p.add_argument("--oai-python", default=sys.executable,
                   help="Python executable/env used to run --oai-evaluator.")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--log-gen-every", type=int, default=5,
                   help="Print generation progress every N batches.")
    p.add_argument("--log-ref-every", type=int, default=20,
                   help="Print reference-stat progress every N batches.")
    p.add_argument("--num-steps", type=int, default=250)
    p.add_argument("--eps-stop", type=float, default=1e-3)
    p.add_argument("--cfg-scale", type=float, default=1.0,
                   help="Classifier-free guidance scale. 1.0 = off. >1 "
                        "doubles per-step NFE. Requires a checkpoint whose "
                        "null branch was trained, e.g. via "
                        "--cfg-dropout-prob in train_stage1_eqm.py.")
    p.add_argument("--cfg-mode", choices=["all", "first3"], default="all")
    p.add_argument("--cfg-schedule", choices=["constant", "anneal_to_one", "early"],
                   default="constant")
    p.add_argument("--cfg-schedule-power", type=float, default=1.0)
    p.add_argument("--cfg-early-frac", type=float, default=0.5,
                   help="For --cfg-schedule early, apply full CFG over the "
                        "first this-fraction of the autonomous-time horizon, "
                        "then turn guidance off (scale 1). 0.5 = first half.")
    p.add_argument("--cfg-rule", choices=["standard", "perp", "perp_matched"], default="standard",
                   help="CFG combination rule (eqfm/cfg_rules.py). perp_matched: rescale b_u to ||b_c|| "
                        "then subtract (w-1) times its perpendicular part.")
    p.add_argument("--cfg-cnorm", action="store_true", default=False,
                   help="with --cfg-rule perp: rescale the perpendicular part to ||b_c||")
    p.add_argument("--cfg-match-uncond-norm", action="store_true", default=False,
                   help="Before CFG, rescale the unconditional field per "
                        "sample so its L2 norm matches the conditional "
                        "field norm.")
    p.add_argument("--cfg-match-guided-norm", action="store_true", default=False,
                   help="After CFG, rescale the final guided vector per sample "
                        "to the conditional field norm (direction-only "
                        "guidance). Combine with --cfg-match-uncond-norm for "
                        "the toy's best variant.")
    p.add_argument("--null-class", type=int, default=1000,
                   help="SiT unconditional embedding index (1000 for "
                        "ImageNet-1k).")
    p.add_argument("--momentum", type=float, default=0.0,
                   help="Heavy-ball / Nesterov coefficient on the Euler "
                        "step. 0.0 = plain Euler. Try 0.5-0.9 as a "
                        "test-time-compute knob.")
    p.add_argument("--nesterov", action="store_true", default=False,
                   help="Nesterov look-ahead variant of --momentum.")
    p.add_argument("--parameterization", choices=["auto", "velocity", "denoiser"],
                   default="auto",
                   help="Network output interpretation. 'auto' reads the "
                        "checkpoint's training args (default 'velocity'); "
                        "'denoiser' reconstructs the field as D(x)-x.")
    p.add_argument("--latent-size", type=int, default=32)
    p.add_argument("--vae-id", default="stabilityai/sd-vae-ft-ema")
    p.add_argument("--use-ema", action="store_true", default=True)
    p.add_argument("--no-ema", dest="use_ema", action="store_false")

    # Reference
    p.add_argument("--ref-stats", type=Path, default=None,
                   help="Path to a cached/external reference .npz with "
                        "arrays 'mu' and 'sigma'. If it exists it's loaded "
                        "directly. If omitted, defaults to "
                        "<output-dir>/../ref_stats_<ref-num>.npz and is "
                        "computed from --ref-shards on first run.")
    p.add_argument("--ref-raw", type=Path, default=None,
                   help="Path to a raw-image reference .npz, e.g. ADM "
                        "VIRTUAL_imagenet256_labeled.npz with arr_0/images. "
                        "Features are extracted with this script's Inception "
                        "wrapper and cached only in --output-dir.")
    p.add_argument("--ref-shards", default=DEFAULT_TRAIN_SHARDS,
                   help="WebDataset pattern for the real reference images.")
    p.add_argument("--ref-num", type=int, default=None,
                   help="Number of real images for the reference stats. "
                        "Defaults to --num-samples (match the gen count).")
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--ref-num-workers", type=int, default=4)

    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--bf16", action="store_true", default=True)
    p.add_argument("--no-bf16", dest="bf16", action="store_false")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def _to_unit(images: torch.Tensor) -> torch.Tensor:
    """[-1, 1] -> [0, 1] for Inception input."""
    return ((images + 1.0) / 2.0).clamp(0.0, 1.0)


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


def _images_to_uint8_nhwc(images: torch.Tensor) -> np.ndarray:
    """Convert decoded VAE images in [-1, 1] to NHWC uint8 for OpenAI eval."""
    x = (_to_unit(images.float()) * 255.0).round().clamp(0, 255)
    return x.to(torch.uint8).permute(0, 2, 3, 1).contiguous().cpu().numpy()


class SampleNpzWriter:
    """Streaming writer for OpenAI-compatible sample batches.

    We first write a temporary .npy memmap and then wrap it into an uncompressed
    .npz as arr_0.npy. This avoids holding 50k 256x256 RGB samples in RAM.
    """

    def __init__(self, path: Path, total: int, overwrite: bool = False):
        self.path = path
        self.total = total
        self.overwrite = overwrite
        self.tmp_npy = path.with_name(path.name + ".tmp.npy")
        self.tmp_npz = path.with_name(path.name + ".tmp")
        self.mmap = None

        if self.path.exists() and not self.overwrite:
            raise FileExistsError(
                f"{self.path} already exists. Pass --overwrite-samples-npz "
                "to replace it."
            )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        for tmp in (self.tmp_npy, self.tmp_npz):
            if tmp.exists():
                tmp.unlink()

    def write(self, batch: np.ndarray, start: int) -> None:
        if batch.dtype != np.uint8 or batch.ndim != 4 or batch.shape[-1] != 3:
            raise ValueError(
                "Expected generated sample batch as NHWC uint8 RGB, got "
                f"shape={batch.shape} dtype={batch.dtype}"
            )
        if self.mmap is None:
            shape = (self.total, *batch.shape[1:])
            self.mmap = np.lib.format.open_memmap(
                self.tmp_npy, mode="w+", dtype=np.uint8, shape=shape
            )
        self.mmap[start:start + batch.shape[0]] = batch

    def close(self) -> None:
        if self.mmap is None:
            raise RuntimeError("No samples were written to the NPZ writer.")
        self.mmap.flush()
        del self.mmap
        self.mmap = None
        with zipfile.ZipFile(
            self.tmp_npz, mode="w", compression=zipfile.ZIP_STORED, allowZip64=True
        ) as zf:
            zf.write(self.tmp_npy, arcname="arr_0.npy")
        os.replace(self.tmp_npz, self.path)
        self.tmp_npy.unlink(missing_ok=True)
        print(f"[fid] saved OpenAI sample batch -> {self.path}", flush=True)


@torch.no_grad()
def compute_reference_stats(args, incep, device):
    requested_ref_num = args.ref_num or args.num_samples
    if args.ref_stats is not None and args.ref_raw is not None:
        raise ValueError("Pass only one of --ref-stats or --ref-raw.")

    if args.ref_raw is not None:
        ref_path = args.ref_raw
        arr = _npz_image_array(ref_path)
        n = min(requested_ref_num, arr.shape[0])
        if requested_ref_num > arr.shape[0]:
            print(f"[fid] raw reference has only {arr.shape[0]} images; "
                  f"using {n} instead of requested {requested_ref_num}",
                  flush=True)
        cache_path = args.output_dir / f"{ref_path.stem}_stats_{n}.npz"
        if cache_path.exists():
            print(f"[fid] loading cached raw-reference stats from {cache_path}")
            d = np.load(cache_path)
            return d["mu"], d["sigma"], cache_path, n
        print(f"[fid] computing raw-reference stats from {ref_path} "
              f"over {n} images -> {cache_path}")
        feats = []
        t0 = time.time()
        for start in range(0, n, args.batch_size):
            images = _np_images_to_tensor(arr[start:start + args.batch_size])
            feats.append(incep(images))
            seen = min(start + images.shape[0], n)
            batch_idx = start // args.batch_size + 1
            if batch_idx == 1 or batch_idx % max(1, args.log_ref_every) == 0 or seen >= n:
                print(f"[fid]   raw ref {seen}/{n} "
                      f"({seen/max(time.time()-t0,1e-6):.1f} img/s)", flush=True)
        mu, sigma = stats_from_features(np.concatenate(feats, axis=0)[:n])
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache_path, mu=mu, sigma=sigma)
        print(f"[fid] raw-reference stats cached to {cache_path}")
        return mu, sigma, cache_path, n

    if args.ref_stats is not None:
        ref_path = args.ref_stats
    else:
        ref_path = args.output_dir.parent / f"ref_stats_{requested_ref_num}.npz"

    if ref_path.exists():
        print(f"[fid] loading cached reference stats from {ref_path}")
        d = np.load(ref_path)
        if "mu" not in d.files or "sigma" not in d.files:
            raise KeyError(
                f"{ref_path} exists but does not contain FID stats keys "
                f"'mu' and 'sigma' (found {d.files}). Use a precomputed "
                "stats .npz here, not a raw-image ADM npz."
            )
        return d["mu"], d["sigma"], ref_path, requested_ref_num
    if args.ref_stats is not None:
        raise FileNotFoundError(
            f"--ref-stats was passed explicitly, but {ref_path} does not exist. "
            "Refusing to compute and overwrite an explicit reference path."
        )

    print(f"[fid] computing reference stats over {requested_ref_num} real images "
          f"(this is one-time; cached to {ref_path})")
    loader = build_imagenet_loader(
        shards=args.ref_shards,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.ref_num_workers,
        shuffle_buffer=1000,
        shardshuffle=100,
        distributed=False,
    )
    feats = []
    seen = 0
    batch_idx = 0
    t0 = time.time()
    for images, _ in loader:
        batch_idx += 1
        images = images.to(device, non_blocking=True)
        feats.append(incep(_to_unit(images)))
        seen += images.shape[0]
        if batch_idx == 1 or batch_idx % max(1, args.log_ref_every) == 0 or seen >= requested_ref_num:
            print(f"[fid]   ref {seen}/{requested_ref_num} "
                  f"({seen/max(time.time()-t0,1e-6):.1f} img/s)", flush=True)
        if seen >= requested_ref_num:
            break
    feats = np.concatenate(feats, axis=0)[:requested_ref_num]
    mu, sigma = stats_from_features(feats)
    ref_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(ref_path, mu=mu, sigma=sigma)
    print(f"[fid] reference stats cached to {ref_path}")
    return mu, sigma, ref_path, requested_ref_num


def autonomous_forward(model):
    def forward(z, y):
        t = torch.zeros(z.shape[0], device=z.device, dtype=z.dtype)
        return model(z, t, y)
    return forward


@torch.no_grad()
def compute_gen_stats(args, model, ema, vae, incep, device, compute_dtype,
                      parameterization="velocity"):
    n = args.num_samples
    # Class-balanced labels: 0,1,...,999,0,1,... so every class is equally
    # represented (the standard convention for conditional ImageNet FID).
    labels_all = torch.arange(n, dtype=torch.long) % 1000

    ema_ctx = ema.apply_to(model) if (ema is not None and args.use_ema) \
        else contextlib.nullcontext()

    feats = [] if incep is not None else None
    writer = (
        SampleNpzWriter(args.save_samples_npz, n, args.overwrite_samples_npz)
        if args.save_samples_npz is not None else None
    )
    produced = 0
    batch_idx = 0
    t0 = time.time()
    print(f"[fid] generating {n} images "
          f"(batch={args.batch_size}, steps={args.num_steps}, "
          f"cfg={args.cfg_scale}, mode={args.cfg_mode}, "
          f"schedule={args.cfg_schedule}, "
          f"match_uncond={args.cfg_match_uncond_norm})",
          flush=True)
    with ema_ctx:
        model_fn = as_field_fn(autonomous_forward(model), parameterization)
        gen = torch.Generator(device=device)
        while produced < n:
            batch_idx += 1
            b = min(args.batch_size, n - produced)
            y = labels_all[produced:produced + b].to(device)
            gen.manual_seed(args.seed + produced)  # reproducible per-chunk
            latents = sample_euler(
                model_fn,
                shape=(b, 4, args.latent_size, args.latent_size),
                class_labels=y,
                num_steps=args.num_steps,
                stop_time=stop_time_from_eps(args.eps_stop),
                cfg_scale=args.cfg_scale,
                cfg_mode=args.cfg_mode,
                cfg_schedule=args.cfg_schedule,
                cfg_schedule_power=args.cfg_schedule_power,
                cfg_early_frac=args.cfg_early_frac,
                cfg_match_uncond_norm=args.cfg_match_uncond_norm,
                cfg_match_guided_norm=args.cfg_match_guided_norm,
                cfg_rule=args.cfg_rule,
                cfg_cnorm=args.cfg_cnorm,
                null_class=args.null_class,
                momentum=args.momentum,
                nesterov=args.nesterov,
                device=device,
                dtype=torch.float32,
                autocast_dtype=compute_dtype if args.bf16 else None,
                generator=gen,
            )
            with torch.amp.autocast(device.type, dtype=compute_dtype,
                                    enabled=args.bf16):
                images = vae.decode(latents)
            if writer is not None:
                writer.write(_images_to_uint8_nhwc(images), produced)
            if incep is not None:
                feats.append(incep(_to_unit(images.float())))
            produced += b
            if batch_idx == 1 or batch_idx % max(1, args.log_gen_every) == 0 or produced >= n:
                print(f"[fid]   gen {produced}/{n} "
                      f"({produced/max(time.time()-t0,1e-6):.2f} img/s)",
                      flush=True)
    if writer is not None:
        writer.close()
    if incep is None:
        return None, None
    feats = np.concatenate(feats, axis=0)[:n]
    return stats_from_features(feats)


def run_oai_evaluator(args) -> None:
    if args.oai_evaluator is None:
        return
    if args.save_samples_npz is None:
        raise ValueError("--oai-evaluator requires --save-samples-npz.")
    if args.oai_ref is None:
        raise ValueError("--oai-evaluator requires --oai-ref.")
    if not args.oai_evaluator.exists():
        raise FileNotFoundError(args.oai_evaluator)
    if not args.oai_ref.exists():
        raise FileNotFoundError(args.oai_ref)
    if not args.save_samples_npz.exists():
        raise FileNotFoundError(args.save_samples_npz)

    cmd = [
        args.oai_python,
        str(args.oai_evaluator),
        str(args.oai_ref),
        str(args.save_samples_npz),
    ]
    print(f"[fid] running OpenAI evaluator: {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, text=True, capture_output=True)
    report = args.output_dir / "oai_eval.txt"
    with report.open("w") as f:
        f.write("$ " + " ".join(cmd) + "\n\n")
        f.write(result.stdout)
        if result.stderr:
            f.write("\n[stderr]\n")
            f.write(result.stderr)
    if result.stdout:
        print(result.stdout, end="", flush=True)
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr, flush=True)
    print(f"[fid] wrote {report}", flush=True)
    result.check_returncode()


def main() -> None:
    args = parse_args()
    if args.cfg_scale != 1.0:
        print("[fid] cfg-scale != 1: requires a checkpoint whose null branch "
              "was trained, e.g. via SiT's label dropout or explicit "
              "--cfg-dropout-prob. Treat older checkpoints as diagnostic "
              "until verified with probe_cfg_dropout_training.py.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.skip_pytorch_fid and args.save_samples_npz is None:
        raise ValueError("--skip-pytorch-fid is only useful with --save-samples-npz.")
    if args.oai_evaluator is not None and args.save_samples_npz is None:
        raise ValueError("--oai-evaluator requires --save-samples-npz.")
    device = torch.device(args.device)
    compute_dtype = torch.bfloat16 if args.bf16 else torch.float32
    torch.manual_seed(args.seed)

    print(f"[fid] ckpt = {args.ckpt}")
    print("[fid] loading checkpoint", flush=True)
    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    ckpt_step = state.get("step", -1)
    print(f"[fid] checkpoint step = {ckpt_step}")

    parameterization = args.parameterization
    if parameterization == "auto":
        ckpt_args = state.get("args", {}) or {}
        parameterization = ckpt_args.get("parameterization", "velocity")
        print(f"[fid] parameterization = {parameterization} (auto-detected)")
    else:
        print(f"[fid] parameterization = {parameterization} (CLI override)")

    print("[fid] instantiating EqM model", flush=True)
    model = SiT_XL_2(input_size=args.latent_size).to(device)
    model.load_state_dict(state["model"])
    model.eval()

    ema = None
    if args.use_ema and "ema" in state:
        ema = EMA(model, decay=0.9999)
        ema.load_state_dict(state["ema"])

    print(f"[fid] loading VAE: {args.vae_id}", flush=True)
    vae = LatentEncoder(vae_id=args.vae_id).to(device)
    vae.eval()

    incep = None
    ref_path = None
    actual_ref_num = None
    fid = None
    mu_ref = sigma_ref = None
    if not args.skip_pytorch_fid:
        print("[fid] loading Inception feature extractor", flush=True)
        incep = InceptionFeatureExtractor(device)
        mu_ref, sigma_ref, ref_path, actual_ref_num = compute_reference_stats(
            args, incep, device
        )

    mu_gen, sigma_gen = compute_gen_stats(
        args, model, ema, vae, incep, device, compute_dtype, parameterization
    )

    if not args.skip_pytorch_fid:
        fid = fid_from_stats(mu_gen, sigma_gen, mu_ref, sigma_ref)
        print(f"[fid] FID = {fid:.4f}  "
              f"(N_gen={args.num_samples}, N_ref={actual_ref_num}, "
              f"steps={args.num_steps}, ckpt_step={ckpt_step})")

    out = {
        "fid": fid,
        "ckpt": str(args.ckpt),
        "ckpt_step": ckpt_step,
        "num_samples": args.num_samples,
        "ref_num": actual_ref_num,
        "ref_num_requested": args.ref_num or args.num_samples,
        "ref_stats_path": str(ref_path) if ref_path is not None else None,
        "samples_npz": str(args.save_samples_npz) if args.save_samples_npz else None,
        "skip_pytorch_fid": args.skip_pytorch_fid,
        "oai_evaluator": str(args.oai_evaluator) if args.oai_evaluator else None,
        "oai_ref": str(args.oai_ref) if args.oai_ref else None,
        "num_steps": args.num_steps,
        "eps_stop": args.eps_stop,
        "parameterization": parameterization,
        "use_ema": bool(args.use_ema and ema is not None),
        "cfg_scale": args.cfg_scale,
        "cfg_mode": args.cfg_mode,
        "cfg_schedule": args.cfg_schedule,
        "cfg_schedule_power": args.cfg_schedule_power,
        "cfg_early_frac": args.cfg_early_frac,
        "cfg_match_uncond_norm": args.cfg_match_uncond_norm,
        "cfg_match_guided_norm": args.cfg_match_guided_norm,
        "cfg_rule": args.cfg_rule,
        "cfg_cnorm": args.cfg_cnorm,
        "null_class": args.null_class,
        "momentum": args.momentum,
        "nesterov": args.nesterov,
        "seed": args.seed,
    }
    with (args.output_dir / "fid.json").open("w") as f:
        json.dump(out, f, indent=2, sort_keys=True)
    print(f"[fid] wrote {args.output_dir / 'fid.json'}")
    run_oai_evaluator(args)


if __name__ == "__main__":
    main()
