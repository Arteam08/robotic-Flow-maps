"""Exactness tests for the training-efficiency changes (branch train-efficiency).

Every optimisation on that branch must reproduce the previous numbers; each
test below compares the new code against an inline copy of the old one.
Run with ``pytest tests/test_train_efficiency.py`` or plain python.
"""
from __future__ import annotations

import math
import os
import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eqfm.losses import T_BUCKETS, bucket_stats  # noqa: E402
from eqfm.models.sit.models import SiT_models, TimestepEmbedder  # noqa: E402
from eqfm.stage2_losses import (  # noqa: E402
    Stage2FlowMapLoss,
    Stage2LossConfig,
    _SIGMA_BUCKETS,
    _raw_sigma_stats,
)
from eqfm.training.ema import EMA  # noqa: E402

torch.manual_seed(0)


# ---------------------------------------------------------------- bucket stats
def _old_bucket_stats(per, t, prefix, buckets=T_BUCKETS, tag="t"):
    per = per.detach().float()
    tt = t.detach().float().reshape(-1)
    out = {f"{prefix}": per.mean()}
    last = buckets[-1][1]
    for lo, hi in buckets:
        m = (tt >= lo) & ((tt < hi) if hi < last else (tt <= hi))
        out[f"{prefix}_{tag}{lo:g}_{hi:g}"] = per[m].mean() if bool(m.any()) else per.new_zeros(())
    return out


def _old_raw_sigma_stats(per, sigma, prefix):
    per = per.detach().float()
    sig = sigma.detach().float().reshape(-1)
    out = {f"{prefix}_raw": per.mean()}
    for lo, hi in _SIGMA_BUCKETS:
        m = (sig >= lo) & (sig < hi) if hi < 1.0 else (sig >= lo)
        out[f"{prefix}_raw_s{lo:g}_{hi:g}"] = per[m].mean() if bool(m.any()) else per.new_zeros(())
    return out


def test_bucket_stats_matches_old_including_empty_buckets():
    for trial in range(20):
        n = 1 + trial
        per = torch.rand(n) * 10
        # concentrate t so some buckets are empty
        t = torch.rand(n) * (0.05 if trial % 2 else 1.0)
        new, old = bucket_stats(per, t, "x"), _old_bucket_stats(per, t, "x")
        assert new.keys() == old.keys()
        for k in old:
            assert torch.allclose(new[k], old[k], atol=1e-6, rtol=1e-6), (k, new[k], old[k])
        s = torch.rand(n) * (0.3 if trial % 2 else 1.0)
        new, old = _raw_sigma_stats(per, s, "d"), _old_raw_sigma_stats(per, s, "d")
        assert new.keys() == old.keys()
        for k in old:
            assert torch.allclose(new[k], old[k], atol=1e-6, rtol=1e-6), (k, new[k], old[k])


# ------------------------------------------------------- batched teacher CFG
def _make_student_teacher():
    torch.manual_seed(1)
    teacher = SiT_models["SiT-S/2"](input_size=8, class_dropout_prob=0.0).eval()
    student = SiT_models["SiT-S/2"](input_size=8, class_dropout_prob=0.0).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    return student, teacher


def _teacher_fn(teacher):
    def fn(x, y):
        with torch.no_grad():
            return teacher(x, torch.zeros(x.shape[0], dtype=x.dtype), y).float()
    return fn


def _old_teacher_target(cfg, teacher_fn, x, labels, labels_in, drop_mask):
    from eqfm.stage2_losses import _combine_cfg, _match_uncond_norm, _perp_guided
    if cfg.guidance_mode == "branch":
        return teacher_fn(x, labels_in)
    null = torch.full_like(labels, cfg.null_class)
    if cfg.teacher_cfg_scale == 1.0:
        target = teacher_fn(x, labels)
        if cfg.cfg_dropout_prob <= 0.0:
            return target
        uncond = teacher_fn(x, null)
        return torch.where(drop_mask.view(-1, *([1] * (x.ndim - 1))), uncond, target)
    b_cond = teacher_fn(x, labels)
    b_uncond = teacher_fn(x, null)
    if cfg.cfg_match_uncond_norm:
        b_uncond = _match_uncond_norm(b_cond, b_uncond)
    if cfg.cfg_rule == "perp":
        guided = _perp_guided(b_cond, b_uncond, cfg.teacher_cfg_scale, cfg.cfg_cnorm)
    else:
        guided = _combine_cfg(b_cond, b_uncond, cfg.teacher_cfg_scale, cfg.cfg_mode)
    if cfg.cfg_match_guided_norm:
        guided = _match_uncond_norm(b_cond, guided)
    return torch.where(drop_mask.view(-1, *([1] * (x.ndim - 1))), b_uncond, guided)


def test_batched_teacher_target_equals_two_calls():
    student, teacher = _make_student_teacher()
    tfn = _teacher_fn(teacher)
    x = torch.randn(6, 4, 8, 8)
    labels = torch.randint(0, 10, (6,))
    for scale, drop in [(1.5, 0.0), (1.5, 0.5), (1.0, 0.5), (1.0, 0.0)]:
        for mu, mg in [(False, False), (True, False), (True, True)]:
            cfg = Stage2LossConfig(guidance_mode="guided", teacher_cfg_scale=scale,
                                   cfg_dropout_prob=drop, null_class=10,
                                   cfg_match_uncond_norm=mu, cfg_match_guided_norm=mg)
            loss = Stage2FlowMapLoss(cfg)
            torch.manual_seed(3)
            labels_in, drop_mask = loss.drop_labels(labels)
            new = loss.teacher_target_for_conditioning(tfn, x, labels, labels_in, drop_mask)
            old = _old_teacher_target(cfg, tfn, x, labels, labels_in, drop_mask)
            assert torch.allclose(new, old, atol=1e-5, rtol=1e-5), (scale, drop, mu, mg, (new - old).abs().max())


# -------------------------------------------------------------------- EMA
def test_foreach_ema_equals_loop_ema():
    torch.manual_seed(2)
    model = torch.nn.Sequential(torch.nn.Linear(5, 7), torch.nn.Linear(7, 3))
    ema = EMA(model, decay=0.9)
    ref = {n: p.detach().clone() for n, p in model.named_parameters()}
    for _ in range(10):
        for p in model.parameters():
            p.data.add_(torch.randn_like(p) * 0.1)
        ema.update(model)
        for n, p in model.named_parameters():
            ref[n].mul_(0.9).add_(p.detach(), alpha=0.1)
    for n in ref:
        assert torch.allclose(ema.shadow[n], ref[n], atol=1e-7), n


# ---------------------------------------------------------- timestep table
def test_cached_timestep_embedding_matches_formula():
    t = torch.rand(9) * 1000
    for dim in (256, 255):
        got = TimestepEmbedder.timestep_embedding(t, dim)
        got2 = TimestepEmbedder.timestep_embedding(t, dim)  # cached path
        half = dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(0, half, dtype=torch.float32) / half)
        args = t[:, None].float() * freqs[None]
        ref = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            ref = torch.cat([ref, torch.zeros_like(ref[:, :1])], dim=-1)
        assert torch.equal(got, got2)
        assert torch.allclose(got, ref, atol=1e-6)


# -------------------------------------------------------- latest.pt hardlink
def test_copy_to_latest_hardlinks_and_survives_source_removal():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from train_stage2_flowmap import copy_to_latest  # noqa: E402
    with tempfile.TemporaryDirectory() as d:
        src = Path(d) / "step_0000100.pt"
        torch.save({"step": 100}, src)
        latest = copy_to_latest(src)
        assert latest.name == "latest.pt"
        assert os.stat(src).st_ino == os.stat(latest).st_ino  # hardlink, no copy
        src.unlink()  # cleanup_old_checkpoints does this later
        assert torch.load(latest, weights_only=False)["step"] == 100
        # re-publishing a newer step replaces latest atomically
        src2 = Path(d) / "step_0000200.pt"
        torch.save({"step": 200}, src2)
        copy_to_latest(src2)
        assert torch.load(latest, weights_only=False)["step"] == 200


# ------------------------------------------- full loss: new code == old code
def test_full_lag_semi_loss_unchanged_vs_baseline_worktree():
    """Compare the whole lag_semi loss against the production tree at ~/EqFM
    (skipped when it is not present). Same seeds => same losses."""
    prod = Path.home() / "EqFM"
    if not (prod / "eqfm" / "stage2_losses.py").exists():
        return
    import importlib.util
    def load(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod
    # Load the production package under a different name.
    import importlib
    sys.path.insert(0, str(prod))
    saved = {k: v for k, v in sys.modules.items() if k == "eqfm" or k.startswith("eqfm.")}
    for k in saved:
        del sys.modules[k]
    try:
        prod_losses = importlib.import_module("eqfm.stage2_losses")
        prod_ema = importlib.import_module("eqfm.training.ema")
        prod_pkg_file = prod_losses.__file__
        assert str(prod) in prod_pkg_file, prod_pkg_file
        student, teacher = _make_student_teacher()
        tfn = _teacher_fn(teacher)
        x = torch.randn(8, 4, 8, 8); labels = torch.randint(0, 10, (8,))
        kw = dict(objective="lagrangian", guidance_mode="guided", teacher_cfg_scale=1.5,
                  null_class=10, cfg_match_uncond_norm=True, cfg_match_guided_norm=True,
                  semigroup_weight=1.0, terminal_weight=1.0,
                  terminal_teacher_norm_weight=1.0, terminal_norm_max_batch=4,
                  interp_time_sampler="uniform", sigma_sampler="uniform",
                  sigma_min=1e-4, sigma_max=0.97, adaptive_weight_p=0.5)
        old_loss = prod_losses.Stage2FlowMapLoss(prod_losses.Stage2LossConfig(**kw))
        # On CPU the flash kernel has no forward-AD rule; production only
        # forces the math backend on CUDA, so force it here for both trees.
        from torch.nn.attention import SDPBackend, sdpa_kernel
        with sdpa_kernel([SDPBackend.MATH]):
            torch.manual_seed(11); lo, mo = old_loss(student, tfn, x, labels, teacher_grad_fn=tfn)
    finally:
        for k in [k for k in sys.modules if k == "eqfm" or k.startswith("eqfm.")]:
            del sys.modules[k]
        sys.modules.update(saved)
        sys.path.remove(str(prod))
    new_loss = Stage2FlowMapLoss(Stage2LossConfig(**kw))
    from torch.nn.attention import SDPBackend, sdpa_kernel
    with sdpa_kernel([SDPBackend.MATH]):
        torch.manual_seed(11); ln, mn = new_loss(student, tfn, x, labels, teacher_grad_fn=tfn)
    assert torch.allclose(ln, lo, atol=1e-5, rtol=1e-5), (ln, lo)
    for k in mo:
        assert k in mn, k
        assert torch.allclose(mn[k], mo[k], atol=1e-5, rtol=1e-4), (k, mn[k], mo[k])




# ------------------------------------------------------------ latent cache
def test_latent_loader_partition_covers_every_row_once_and_sampling_matches_encode():
    import numpy as np
    from eqfm.data.imagenet_latents import LatentImageNet, expand_latent_files
    from eqfm.vae import LatentEncoder
    with tempfile.TemporaryDirectory() as d:
        rows = {}
        for s in range(5):  # 5 shards of unequal size
            n = 7 + 3 * s
            mean = np.random.randn(n, 4, 2, 2).astype(np.float16)
            std = np.random.rand(n, 4, 2, 2).astype(np.float16)
            label = (np.arange(n) + 100 * s).astype(np.int16)
            np.savez(Path(d) / f"shard_{s:05d}.npz", mean=mean, std=std, label=label)
            for i in range(n):
                rows[int(label[i])] = (mean[i], std[i])
        files = expand_latent_files(f"latents:{d}")
        assert len(files) == 5
        seen = []
        for rank in range(2):
            ds = LatentImageNet(files, shuffle_buffer=4, seed=3, rank=rank, world_size=2, batch_size=1)
            it = ds._samples(epoch=0)  # single worker per rank here
            for m, s, y in it:
                seen.append(y)
                assert np.array_equal(m, rows[y][0]) and np.array_equal(s, rows[y][1])
        assert sorted(seen) == sorted(rows)  # every row exactly once across ranks
        # sample_from_stats == (mean + std*eps)*0.18215 with the same eps; mean path is deterministic
        m = torch.tensor(rows[0][0]); s = torch.tensor(rows[0][1])
        g = torch.Generator().manual_seed(0)
        z = LatentEncoder.sample_from_stats(m[None], s[None], generator=g)
        eps = torch.randn(m[None].shape, generator=torch.Generator().manual_seed(0))
        assert torch.allclose(z, (m[None].float() + s[None].float() * eps) * 0.18215)
        assert torch.allclose(LatentEncoder.sample_from_stats(m[None], s[None], sample=False), m[None].float() * 0.18215)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print("ok", name)
