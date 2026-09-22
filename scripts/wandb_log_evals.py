#!/usr/bin/env python
"""Log finished FID evaluations of a Stage-2 run into its wandb run, when no trainer is writing to it.

While training runs, the trainer itself logs new FID results (eqfm/stage2_wandb.py). Evaluations
that finish after training ends (e.g. the final checkpoint) need this script. It resumes the run
recorded in ``<run_dir>/wandb_id.txt`` and logs every not-yet-logged checkpoint, lowest step first,
against the ``0_fid/train_step`` axis. Already-logged results are tracked in
``<run_dir>/wandb_logged_evals.json`` (shared with the trainer).

    python scripts/wandb_log_evals.py --run-dir results/<run>

Run it only when no training copy of that run is alive (two writers to one wandb run are unsafe).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from eqfm.stage2_wandb import (  # noqa: E402
    define_wandb_metrics, eval_payload, load_logged, next_eval_group, save_logged,
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--project", default="eqfm")
    p.add_argument("--entity", default=None)
    p.add_argument("--fid-root", default=None, help="default: $EQFM_FID_ROOT or results/fid")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    id_file = args.run_dir / "wandb_id.txt"
    state = args.run_dir / "wandb_logged_evals.json"
    logged = load_logged(state)
    groups = []
    seen = set(logged)
    while True:
        step, group = next_eval_group(args.run_dir.name, seen, root=args.fid_root)
        if not group:
            break
        groups.append((step, group))
        seen |= {r["id"] for r in group}
    if not groups:
        print("[wandb-evals] nothing new to log")
        return
    for step, group in groups:
        print(f"[wandb-evals] step {step}: " + ", ".join(f"{r['weights']}_{r['sampler']}={r['fid']:.2f}" for r in group))
    if args.dry_run:
        return
    if not id_file.exists():
        raise SystemExit(f"no {id_file}; cannot find the wandb run")
    import wandb
    run = wandb.init(project=args.project, entity=args.entity, id=id_file.read_text().strip(),
                     resume="must", dir=str(args.run_dir))
    define_wandb_metrics(run)
    for step, group in groups:
        run.log(eval_payload(step, group))
        logged |= {r["id"] for r in group}
        save_logged(state, logged)
    run.finish()
    print(f"[wandb-evals] logged {len(groups)} checkpoint(s) to {run.url}")


if __name__ == "__main__":
    main()
