"""MeanFlow long-run additions: single-JVP target, cached teacher field, semigroup add-on, lr plateau rule."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eqfm.lr_plateau import LossPlateau  # noqa: E402
from eqfm.stage2_losses import Stage2FlowMapLoss, Stage2LossConfig  # noqa: E402


class TinyVelocity(nn.Module):
    """Nonlinear in x and sigma so both JVP directions matter."""

    def __init__(self):
        super().__init__()
        self.a = nn.Linear(16, 16)
        self.b = nn.Linear(16, 16)
        self.emb = nn.Embedding(1001, 16)

    def forward(self, x, sigma, y):
        h = x.flatten(1)
        h = torch.tanh(self.a(h) + self.emb(y)) * (1.0 + sigma.view(-1, 1)) + torch.sin(3.0 * sigma.view(-1, 1)) * h
        return self.b(h).view_as(x)


def _teacher():
    torch.manual_seed(7)
    t = TinyVelocity()
    return lambda x, y: t(x, torch.zeros(x.shape[0]), y).detach()


def _cfg(**kw):
    base = dict(objective="scaled_meanflow", adaptive_weight_p=1.0, adaptive_weight_eps=1e-3, loss_reduction="mean",
                sigma_max=0.97, guidance_mode="guided", teacher_cfg_scale=1.5, cfg_match_uncond_norm=True,
                cfg_match_guided_norm=True, null_class=1000, terminal_weight=1.0)
    base.update(kw)
    return Stage2LossConfig(**base)


def _grad(cfg, seed=0):
    torch.manual_seed(1)
    student = TinyVelocity()
    teacher = _teacher()
    x = torch.randn(6, 1, 4, 4)
    y = torch.randint(0, 1000, (6,))
    torch.manual_seed(seed)
    loss, metrics = Stage2FlowMapLoss(cfg)(student, teacher, x, y, teacher_grad_fn=teacher)
    loss.backward()
    return loss.detach(), torch.cat([p.grad.flatten() for p in student.parameters() if p.grad is not None]), metrics


def test_single_jvp_matches_two_jvps():
    l2, g2, m2 = _grad(_cfg(mf_single_jvp=False))
    l1, g1, m1 = _grad(_cfg(mf_single_jvp=True))
    assert torch.allclose(l1, l2, rtol=1e-5, atol=1e-7), (l1, l2)
    assert torch.allclose(g1, g2, rtol=1e-4, atol=1e-6)
    assert torch.allclose(m1["distill_raw"], m2["distill_raw"], rtol=1e-5, atol=1e-8)


def test_semigroup_addon_and_endpoint_raw_for_meanflow():
    _, _, m0 = _grad(_cfg())
    _, _, m1 = _grad(_cfg(semigroup_weight=1.0, terminal_teacher_norm_weight=1.0))
    assert "semi_raw" not in m0 and "semi_raw" in m1 and float(m1["semigroup_add_loss"]) > 0
    assert "terminal_teacher_norm_raw" in m1 and math.isfinite(float(m1["terminal_teacher_norm_raw"]))


def test_cached_teacher_field_equals_recomputed():
    cfg = _cfg()
    torch.manual_seed(1)
    student, teacher = TinyVelocity(), _teacher()
    x = torch.randn(6, 1, 4, 4)
    y = torch.randint(0, 1000, (6,))
    calls = {"n": 0}

    def counting_teacher(xx, yy):
        calls["n"] += 1
        return teacher(xx, yy)

    torch.manual_seed(0)
    loss_cached, _ = Stage2FlowMapLoss(cfg)(student, counting_teacher, x, y, teacher_grad_fn=teacher)
    n_cached = calls["n"]
    lf = Stage2FlowMapLoss(cfg)
    orig = lf.diagonal_loss

    def diag_no_cache(*a, **k):
        out = orig(*a, **k)
        lf._b0_cache = None
        return out

    lf.diagonal_loss = diag_no_cache
    calls["n"] = 0
    torch.manual_seed(0)
    loss_fresh, _ = lf(student, counting_teacher, x, y, teacher_grad_fn=teacher)
    assert torch.equal(loss_cached.detach(), loss_fresh.detach())
    assert n_cached == calls["n"] - 1  # one teacher call saved


def test_plateau_block_rule_cuts_restores_and_ignores_noise():
    import random
    p = LossPlateau(window=10, patience=3, threshold=0.01, factor=0.5, min_scale=0.2, cooldown=1, start_step=0)
    # improving 3% per window: never stalls
    v = 1.0
    for k in range(1, 13):
        assert p.end_window(10 * k, v) is None
        v *= 0.97
    assert p.scale == 1.0 and p.bad == 12
    # flat: the next full comparison of flat blocks cuts
    q = LossPlateau(window=10, patience=3, threshold=0.01, factor=0.5, min_scale=0.2, cooldown=1)
    events = [q.end_window(10 * k, 1.0) for k in range(1, 7)]
    assert events[:5] == [None] * 5 and events[5] is not None and q.scale == 0.5 and q.bad == 0 and q.cool == 1
    assert q.end_window(70, 1.0) is None and q.cool == 0 and q.bad == 0      # cooldown window
    st = q.state_dict()
    r = LossPlateau(window=10, patience=3, threshold=0.01, factor=0.5, min_scale=0.2)
    r.load_state_dict(st)
    assert (r.scale, r.block, r.cool) == (q.scale, q.block, q.cool)
    for k in range(8, 14):
        r.end_window(10 * k, 1.0)
    assert r.scale == 0.25
    for k in range(14, 40):
        r.end_window(10 * k, 1.0)
    assert r.scale == 0.2                                                    # floor
    # noisy but improving 0.6% per window with 1.6% window noise (ImageNet-like): rare cuts over 40 windows;
    # the same data under a flat loss is cut most of the time.
    random.seed(0)
    def cuts_per_trial(rate):
        cuts = 0
        for trial in range(200):
            n = LossPlateau(window=1, patience=5, threshold=0.01, factor=0.5, min_scale=0.05)
            for k in range(1, 41):
                cuts += n.end_window(k, ((1 - rate) ** k) * (1 + 0.016 * random.gauss(0, 1))) is not None
        return cuts / 200
    improving, flat = cuts_per_trial(0.006), cuts_per_trial(0.0)
    assert improving < 0.25 and flat > 2.0, (improving, flat)
    assert LossPlateau(window=1, patience=5, threshold=0.01, factor=0.5, min_scale=0.05).is_window_end(1)


def test_plateau_stat_change_resets_block_keeps_scale():
    old = LossPlateau(window=10, patience=2, threshold=0.01, factor=0.5, min_scale=0.1, stat="mean")
    for k in range(1, 5):
        old.end_window(10 * k, 1.0)
    assert old.scale == 0.5
    old.end_window(50, 1.0)
    st = old.state_dict()
    new = LossPlateau(window=10, patience=2, threshold=0.01, factor=0.5, min_scale=0.1, stat="geomean")
    new.load_state_dict(st)
    assert new.scale == 0.5 and new.block == [] and new.stat == "geomean"
    same = LossPlateau(window=10, patience=2, threshold=0.01, factor=0.5, min_scale=0.1, stat="mean")
    same.load_state_dict(st)
    assert same.block == old.block


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("s2mf tests passed")
