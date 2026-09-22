"""Shared helpers for checkpoint diagnostics and sampling scripts."""
from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn

from .models.sit import SiT_XL_2, find_model
from .training import EMA


def autonomous_forward(model: nn.Module):
    """Pin t=0, matching Stage-1 training and sampling."""
    def forward(z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        t = torch.zeros(z.shape[0], device=z.device, dtype=z.dtype)
        return model(z, t, y)
    return forward


def load_stage1_model(
    ckpt: Path,
    *,
    latent_size: int,
    device: torch.device,
    use_ema: bool = True,
) -> Tuple[nn.Module, int, str]:
    """Load a Stage-1 checkpoint into SiT-XL/2, optionally applying EMA."""
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    step = int(state.get("step", -1))
    model = SiT_XL_2(input_size=latent_size).to(device)
    model.load_state_dict(state["model"])
    model.eval()

    which = "raw"
    if use_ema and "ema" in state:
        ema = EMA(model, decay=0.9999)
        ema.load_state_dict(state["ema"])
        with ema.apply_to(model):
            # Make the swap permanent for diagnostics by cloning the applied
            # values back into the model before leaving the context.
            applied = {
                name: p.detach().clone()
                for name, p in model.named_parameters()
            }
        for name, p in model.named_parameters():
            p.data.copy_(applied[name].to(device=p.device, dtype=p.dtype))
        which = "ema"
    elif use_ema:
        print("[eval_utils] WARN checkpoint has no EMA state; using raw weights")
    return model, step, which


def load_sit_teacher(
    ckpt: str,
    *,
    cache_dir: str | None,
    latent_size: int,
    device: torch.device,
) -> nn.Module:
    """Load the pretrained time-conditioned SiT-XL/2 teacher."""
    model = SiT_XL_2(input_size=latent_size).to(device)
    state = find_model(ckpt, cache_dir=cache_dir)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


@contextlib.contextmanager
def maybe_autocast(device: torch.device, dtype: torch.dtype, enabled: bool):
    with torch.amp.autocast(device.type, dtype=dtype, enabled=enabled):
        yield


def per_sample_norm(x: torch.Tensor) -> torch.Tensor:
    return x.float().flatten(1).norm(dim=1)


def per_sample_cosine(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    a_flat = a.float().flatten(1)
    b_flat = b.float().flatten(1)
    denom = a_flat.norm(dim=1) * b_flat.norm(dim=1)
    return (a_flat * b_flat).sum(dim=1) / denom.clamp_min(eps)


# ---------------------------------------------------------------------------
# Fixed-batch loss probes (used by the Stage-1 trainer and wrap_eqm_ckpt.py)
# ---------------------------------------------------------------------------

def collect_fixed_batches(
    data_iter,
    vae,
    n_batches: int,
    *,
    device: torch.device,
    bf16: bool,
) -> list:
    """Pull ``n_batches`` (images, labels) batches and encode them once.

    Latents are taken at the posterior mean (``sample=False``) so repeated
    probes see identical inputs. Kept on ``device``; 4 x 32 latents of
    4x32x32 is ~2 MB.
    """
    batches = []
    for _ in range(n_batches):
        batch = next(data_iter)
        if len(batch) == 3:
            # cached SD-VAE posterior stats (``latents:`` shards): posterior mean, no pixels
            mean, std, labels = batch
            labels = labels.to(device)
            latents = vae.sample_from_stats(mean.to(device), std.to(device), sample=False)
        else:
            images, labels = batch
            images = images.to(device, non_blocking=True)
            labels = labels.to(device) if torch.is_tensor(labels) else torch.as_tensor(labels, device=device)
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
                latents = vae.encode(images, sample=False)
        batches.append((latents.detach().float(), labels))
    return batches


@torch.no_grad()
def fixed_batch_loss(
    model_fn,
    loss_fn,
    batches: list,
    *,
    seed: int,
    device: torch.device,
    bf16: bool,
) -> dict:
    """Evaluate ``loss_fn`` on fixed batches with fixed noise and times.

    The RNG is forked and reseeded so ``z`` and ``t`` are identical on every
    call. Returns the mean ``loss``, ``b_data_rms`` (RMS of the field on clean
    data: how far data are from being fixed points), ``x1_norm`` (mean norm of
    the clean latents), every scalar the loss emits, and exact pooled means of
    its per-sample quantities overall and per t bucket (``eqm_raw_t0.9_1``,
    ``eqm_relerr_t0.9_1``, ..., see :mod:`eqfm.metric_window`).
    """
    from .metric_window import WindowStats, pool_eqm_metrics

    stats = WindowStats()
    devices = [device] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        for x, y in batches:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
                loss, metrics = loss_fn(model_fn, x, y)
                b_data = model_fn(x, y)
                if isinstance(b_data, tuple):
                    b_data = b_data[0]
            stats.add_scalar("loss", loss)
            stats.add_scalar("b_data_rms", b_data.float().pow(2).mean().sqrt())
            stats.add_scalar("x1_norm", x.float().flatten(1).norm(dim=1).mean())
            pool_eqm_metrics(stats, metrics)
    return stats.result()


def _allreduce_mean(x: float, device: torch.device) -> float:
    import torch.distributed as dist
    if dist.is_available() and dist.is_initialized():
        t = torch.tensor([x], device=device, dtype=torch.float64)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        return float(t.item()) / dist.get_world_size()
    return x


def search_output_scale(
    model: nn.Module,
    model_fn,
    loss_fn,
    batches: list,
    candidates,
    *,
    seed: int,
    device: torch.device,
    bf16: bool,
) -> Tuple[float, list]:
    """Pick the global rescale of ``final_layer.linear`` minimising the loss.

    Used when warm-starting from a field trained with a different clock
    (e.g. EqM's ``4*min(1, 5(1-t))`` target vs our ``(1-t)`` target): a
    single scalar on the output head absorbs most of the mismatch. Each
    candidate is applied in place, evaluated on the fixed batches (mean over
    DDP ranks so every rank picks the same value) and reverted. Returns
    ``(best_scale, [(scale, loss), ...])``; the model is left unscaled.
    """
    head = model.final_layer.linear
    w0 = head.weight.detach().clone()
    b0 = head.bias.detach().clone() if head.bias is not None else None
    table = []
    for s in candidates:
        with torch.no_grad():
            head.weight.copy_(w0 * s)
            if b0 is not None:
                head.bias.copy_(b0 * s)
        res = fixed_batch_loss(model_fn, loss_fn, batches, seed=seed, device=device, bf16=bf16)
        table.append((float(s), _allreduce_mean(res["loss"], device)))
    with torch.no_grad():
        head.weight.copy_(w0)
        if b0 is not None:
            head.bias.copy_(b0)
    best = min(table, key=lambda r: r[1])[0]
    return best, table


@torch.no_grad()
def scale_output_head(model: nn.Module, scale: float) -> None:
    """Multiply ``final_layer.linear`` (weight and bias) by ``scale`` in place."""
    head = model.final_layer.linear
    head.weight.mul_(scale)
    if head.bias is not None:
        head.bias.mul_(scale)
