"""Offdiag sampling schedules of scripts/eval_fid_stage2_flowmap.py (run: python tests/test_offdiag_schedule.py)."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from eval_fid_stage2_flowmap import sample_stage2_batch  # noqa: E402


class Recorder(torch.nn.Module):
    """v_sigma(x, y) = -x (exact flow map X_s(x) = (1 - s) x); records every sigma it sees."""

    def __init__(self):
        super().__init__()
        self.sigmas = []

    def forward(self, x, sigma, y):
        self.sigmas.append(float(sigma[0]))
        return -x


def run(**kw):
    m = Recorder()
    z = torch.randn(3, 4, 8, 8, generator=torch.Generator().manual_seed(0))
    out = sample_stage2_batch(m, z, torch.zeros(3, dtype=torch.long), sampler="offdiag", diag_num_steps=250,
                              eps_stop=1e-3, device=torch.device("cpu"), compute_dtype=torch.float32, bf16=False, **kw)
    return z, out, m.sigmas


def test_equal_matches_original_loop():
    for k in (1, 2, 4, 8):
        z, out, sig = run(offdiag_steps=k)  # default schedule
        s = 1.0 - 1e-3 ** (1.0 / k)
        assert len(sig) == k and all(abs(v - s) < 1e-6 for v in sig)
        ref = z.clone()
        for _ in range(k):
            ref = ref + s * (-ref)
        assert torch.equal(out, ref)


def test_paper_schedule():
    for k, s0 in ((2, 0.5), (4, 0.3), (8, 0.2)):
        z, out, sig = run(offdiag_steps=k, offdiag_schedule="paper", offdiag_s0=s0)
        assert len(sig) == k
        assert all(abs(v - s0) < 1e-6 for v in sig[:-1]) and sig[-1] == 1.0
        assert torch.count_nonzero(out) == 0  # last jump X_1 of the toy map lands exactly on 0
    z, out, sig = run(offdiag_steps=1, offdiag_schedule="paper", offdiag_s0=0.3)
    assert sig == [1.0]  # K = 1 is the terminal map X_1


if __name__ == "__main__":
    torch.set_num_threads(2)
    test_equal_matches_original_loop()
    test_paper_schedule()
    print("offdiag schedule tests passed")
