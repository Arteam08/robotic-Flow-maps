"""Run:  python -m pytest tests/test_tracking.py -q   (needs no network, no key)."""
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _run(env_extra, tmp_path):
    code = f"""
import sys; sys.path.insert(0, {str(ROOT)!r})
from rfm.tracking import Tracker
tr = Tracker({str(tmp_path / 'run0')!r}, config={{'lr': 1e-4}}, tags=['test'])
for s in range(3): tr.log(s, {{'loss': 1.0 / (s + 1)}}, prefix='train')
p = {str(tmp_path / 'ck.pt')!r}; open(p, 'wb').write(b'x' * 100)
tr.log_checkpoint(p, step=2)
tr.finish()
"""
    env = {k: v for k, v in os.environ.items() if not k.startswith("WANDB")}
    env.update(env_extra)
    subprocess.run([sys.executable, "-c", code], check=True, env=env, cwd=tmp_path)
    rows = [json.loads(l) for l in open(tmp_path / "run0" / "metrics.jsonl")]
    assert [r["step"] for r in rows] == [0, 1, 2] and rows[1]["train/loss"] == 0.5
    ck = json.loads(open(tmp_path / "run0" / "checkpoints.jsonl").readline())
    assert ck["bytes"] == 100 and len(ck["sha256"]) == 64


def test_disabled(tmp_path):
    _run({"WANDB_MODE": "disabled"}, tmp_path)
    assert not (tmp_path / "run0" / "wandb").exists()


def test_offline_without_key(tmp_path):
    _run({}, tmp_path)  # no key -> offline wandb run
    assert (tmp_path / "run0" / "wandb_id.txt").exists()
    assert list((tmp_path / "run0" / "wandb").glob("offline-run-*"))
