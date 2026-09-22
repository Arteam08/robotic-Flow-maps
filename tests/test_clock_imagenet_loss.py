"""CPU checks: trunc clock target + 1/(c+eps) weight in the ImageNet EqMLoss (python tests/test_clock_imagenet_loss.py)."""
import sys, math, types
from pathlib import Path
import torch
REPO = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO)); torch.set_num_threads(2)
from eqfm.losses import EqMLoss, EqMLossConfig
from eqfm.clocks import make_clock, inv_c_normalizer
from eqfm import wandb_glossary as wg

def test_trunc_target_and_weight():
    clock = make_clock("trunc:0.8")
    t = torch.tensor([0.0, 0.5, 0.8, 0.9, 0.99])
    assert torch.allclose(clock.c(t), torch.tensor([1.0, 1.0, 1.0, 0.5, 0.05]))
    Z = inv_c_normalizer(clock, 1e-3); assert abs(Z - 2.18) < 0.01, Z
    torch.manual_seed(0)
    x = torch.randn(512, 4, 8, 8); z = torch.randn_like(x); y = torch.zeros(512, dtype=torch.long)
    cfg = EqMLossConfig(time_sampler="uniform_clock_weighted", clock="trunc:0.8", clock_weight_eps=1e-3)
    loss_fn = EqMLoss(cfg)
    # oracle field: returns the exact target for the sampled t -> zero loss
    captured = {}
    def fn(xt, yy):
        return captured["target"]
    g = torch.Generator().manual_seed(1)
    torch.manual_seed(1); t_s = torch.rand(512)          # same draw the loss will make
    captured["target"] = clock.c(t_s).view(-1, 1, 1, 1) * (x - z)
    torch.manual_seed(1)
    loss, m = loss_fn(fn, x, y, z=z)
    assert float(loss) < 1e-10, float(loss)
    w_mean = float(m["time_weight_mean"])
    assert 0.8 < w_mean < 1.2, w_mean                   # normalised weight, MC mean ~1
    print(f"trunc target exact; Z = {Z:.3f}; weight mean {w_mean:.3f}, max {float(m['time_weight_max']):.1f}")

def test_linear_unchanged():
    torch.manual_seed(0)
    x = torch.randn(16, 4, 8, 8); z = torch.randn_like(x); y = torch.zeros(16, dtype=torch.long)
    fn = lambda xt, yy: 0.3 * xt
    a = EqMLoss(EqMLossConfig(time_sampler="uniform_weighted"))
    b = EqMLoss(EqMLossConfig(time_sampler="uniform_weighted", clock="linear"))
    torch.manual_seed(5); la, _ = a(fn, x, y, z=z)
    torch.manual_seed(5); lb, _ = b(fn, x, y, z=z)
    assert torch.equal(la, lb)
    print("linear clock path unchanged")

def test_glossary_context():
    args = types.SimpleNamespace(time_sampler="uniform_clock_weighted", clock="trunc:0.8", clock_lambda=1.0,
                                 clock_weight_eps=1e-3, parameterization="velocity", data_anchor=0.0)
    c = wg.stage1_context(args)
    assert "min(1, (1 − t)/0.2)" in c["target"] and c["weighted"] and "Z" in c["w_def"], c["target"]
    print("glossary:", c["target"], "|", c["tlaw"])

if __name__ == "__main__":
    test_trunc_target_and_weight(); test_linear_unchanged(); test_glossary_context(); print("ALL OK")
