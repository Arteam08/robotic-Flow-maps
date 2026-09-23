#!/usr/bin/env python
"""Log every finished ADM-scale FID of a run into wandb, as its own run named ``<run>-fid`` (safe to call while the
trainer is still writing to the training run). Idempotent: keeps a ledger in ``$EQFM_FID_ROOT/<run>/wandb_fid_logged.json``.

    python scripts/log_fid_wandb.py --run s2_shifted_meanflow          # scans $EQFM_FID_ROOT/<run>/step<S>_<w>_n<N>/<sampler>/adm_fid.json

Metrics: ``fid/<sampler> [<weights>]`` against the axis ``ckpt_step`` (one point per checkpoint), plus the same values
as a table so the numbers can be read off without hovering. The runs/fid_*.sh scripts call this at the end.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rfm.tracking import load_dotenv  # noqa: E402


def scan(root: Path, run: str):
    recs = []
    for f in sorted((root / run).glob("step*_*_n*/*/adm_fid.json")):
        m = re.match(r"step(\d+)_(raw|ema)_n(\d+)$", f.parent.parent.name)
        if not m:
            continue
        d = json.load(open(f))
        fid = d.get("FID_N", d.get("fid", d.get("FID")))      # tools/fid_adm/score_npz.py writes FID_N, IS, KID_mean, FID_inf, top1
        if fid is None:
            continue
        recs.append({"id": f"{f.parent.parent.name}/{f.parent.name}", "step": int(m.group(1)), "weights": m.group(2),
                     "n": int(m.group(3)), "sampler": f.parent.name, "fid": float(fid),
                     "is": d.get("IS", d.get("inception_score")), "kid": d.get("KID_mean", d.get("kid")),
                     "fid_inf": d.get("FID_inf"), "top1": d.get("top1")})
    recs.sort(key=lambda r: (r["step"], r["weights"], r["sampler"]))
    return recs


def main() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--fid-root", default=os.environ.get("EQFM_FID_ROOT", "results/fid"))
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    root = Path(a.fid_root)
    recs = scan(root, a.run)
    if not recs:
        print(f"no adm_fid.json under {root / a.run}/step*_*_n*/*/"); return
    ledger = root / a.run / "wandb_fid_logged.json"
    logged = set(json.load(open(ledger))) if ledger.exists() else set()
    new = [r for r in recs if r["id"] not in logged]
    for r in recs:
        print(f"{'new ' if r in new else 'old '} step {r['step']:>7} {r['weights']} n={r['n']} {r['sampler']:14s} FID {r['fid']:.2f}")
    if a.dry_run or not new:
        return
    if not os.environ.get("WANDB_API_KEY") and os.environ.get("WANDB_MODE", "") not in ("offline",):
        print("WANDB_API_KEY not set: numbers printed above only (set the key or WANDB_MODE=offline)"); return
    import wandb
    os.environ.setdefault("WANDB_SILENT", "true")
    run = wandb.init(entity=os.environ.get("WANDB_ENTITY"), project=os.environ.get("WANDB_PROJECT", "robotic-flow-maps"),
                     name=f"{a.run}-fid", id=None, job_type="fid", resume="allow",
                     config={"run": a.run, "protocol": "ADM-2k unless n says otherwise"})
    # resume the same wandb run next time: keep its id next to the ledger
    idf = root / a.run / "wandb_fid_run_id.txt"
    if idf.exists() and idf.read_text().strip() != run.id:
        run.finish()
        run = wandb.init(entity=os.environ.get("WANDB_ENTITY"), project=os.environ.get("WANDB_PROJECT", "robotic-flow-maps"),
                         id=idf.read_text().strip(), resume="must")
    idf.write_text(run.id + "\n")
    run.define_metric("ckpt_step"); run.define_metric("fid/*", step_metric="ckpt_step", summary="min,last")
    table = wandb.Table(columns=["step", "weights", "n", "sampler", "fid", "fid_inf", "is", "kid", "top1"])
    for r in recs:
        table.add_data(r["step"], r["weights"], r["n"], r["sampler"], r["fid"], r["fid_inf"], r["is"], r["kid"], r["top1"])
    for step in sorted({r["step"] for r in new}):
        payload = {"ckpt_step": step, **{f"fid/{r['sampler']} [{r['weights']}]": r["fid"] for r in new if r["step"] == step}}
        run.log(payload)
    run.log({"fid_table": table})
    run.summary["url"] = run.url
    run.finish()
    json.dump(sorted(logged | {r["id"] for r in new}), open(ledger, "w"))
    print(f"logged {len(new)} FID values to {run.url}")


if __name__ == "__main__":
    main()
