#!/usr/bin/env python
"""Download the released weights (no token needed for the public repo).

    python scripts/hf_download.py                 # everything in manifest.json -> weights/
    python scripts/hf_download.py --only mnist_fm_teacher.pt
    python scripts/hf_download.py --list
    python scripts/hf_download.py --repo EquilibriumMap/robotic-flow-maps-results --run exp01_seed0   # a run's checkpoints

Each file's sha256 is verified against manifest.json after download.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rfm.tracking import load_dotenv, sha256_of  # noqa: E402

DEFAULT_REPO = "EquilibriumMap/robotic-flow-maps-weights"


def main() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=os.environ.get("RFM_HF_REPO", DEFAULT_REPO))
    ap.add_argument("--dest", default="weights")
    ap.add_argument("--only", nargs="*", default=None, help="file names from the manifest")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--run", default=None, help="results repo: run folder whose manifest.json to use")
    args = ap.parse_args()

    from huggingface_hub import hf_hub_download

    token = os.environ.get("HF_TOKEN")  # only needed while the repo is private
    prefix = f"{args.run}/" if args.run else ""
    man_path = hf_hub_download(args.repo, f"{prefix}manifest.json", token=token)
    manifest = json.load(open(man_path))
    entries = manifest["files"]
    if args.list:
        for e in entries:
            print(f"{e['file']:45s} {e['bytes'] / 1e6:8.1f} MB  {e.get('note', '')}")
        return
    want = set(args.only) if args.only else {e["file"] for e in entries}
    missing = want - {e["file"] for e in entries}
    if missing:
        sys.exit(f"not in manifest: {sorted(missing)}")
    dest = Path(args.dest) / args.run if args.run else Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    for e in entries:
        if e["file"] not in want:
            continue
        out = dest / e["file"]
        if out.exists() and sha256_of(out) == e["sha256"]:
            print(f"ok      {out}")
            continue
        p = hf_hub_download(args.repo, f"{prefix}{e['file']}", token=token, local_dir=str(dest.parent if args.run else dest))
        got = sha256_of(Path(p))
        if got != e["sha256"]:
            sys.exit(f"sha256 mismatch for {e['file']}: {got} != {e['sha256']}")
        print(f"fetched {p}  ({e['bytes'] / 1e6:.1f} MB, sha256 ok)")


if __name__ == "__main__":
    main()
