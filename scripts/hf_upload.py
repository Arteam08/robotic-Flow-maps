#!/usr/bin/env python
"""(Maintainer side) Upload weights + manifest.json to the HF repo.

    python scripts/hf_upload.py --spec upload_spec.json [--repo ORG/NAME] [--public]

upload_spec.json = [{"src": "/path/to/ckpt.pt", "dest": "mnist_fm_teacher.pt",
                     "note": "Stage-0 flow-matching teacher, 60k steps, FID-10k 3.28", "step": 60000}, ...]

Needs a write token in HF_TOKEN (or `hf auth login`).  Existing manifest entries with the same
``dest`` are replaced; others are kept.  Files whose sha256 already matches are skipped.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rfm.tracking import load_dotenv, sha256_of  # noqa: E402

DEFAULT_REPO = "EquilibriumMap/robotic-flow-maps-weights"


def main() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--repo", default=os.environ.get("RFM_HF_REPO", DEFAULT_REPO))
    ap.add_argument("--public", action="store_true", help="create the repo public (default private)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    from huggingface_hub import HfApi, hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError

    api = HfApi(token=os.environ.get("HF_TOKEN"))
    try:
        api.repo_info(args.repo, repo_type="model")
    except RepositoryNotFoundError:
        print(f"creating {args.repo} ({'public' if args.public else 'private'})")
        if not args.dry_run:
            api.create_repo(args.repo, repo_type="model", private=not args.public)
    try:
        manifest = json.load(open(hf_hub_download(args.repo, "manifest.json", token=api.token)))
    except (EntryNotFoundError, RepositoryNotFoundError):
        manifest = {"repo": args.repo, "files": []}
    by_name = {e["file"]: e for e in manifest["files"]}

    spec = json.load(open(args.spec))
    for item in spec:
        src = Path(item["src"]).expanduser()
        dest = item.get("dest", src.name)
        digest = sha256_of(src)
        entry = {"file": dest, "bytes": src.stat().st_size, "sha256": digest,
                 "uploaded": time.strftime("%Y-%m-%d"), **{k: v for k, v in item.items() if k not in ("src", "dest")}}
        if by_name.get(dest, {}).get("sha256") == digest:
            print(f"skip    {dest} (same sha256 already in repo)")
            by_name[dest].update(entry)
            continue
        print(f"upload  {src} -> {args.repo}/{dest}  ({entry['bytes'] / 1e6:.1f} MB)")
        if not args.dry_run:
            api.upload_file(path_or_fileobj=str(src), path_in_repo=dest, repo_id=args.repo, repo_type="model",
                            commit_message=f"add {dest}")
        by_name[dest] = entry
    manifest["files"] = sorted(by_name.values(), key=lambda e: e["file"])
    manifest["repo"] = args.repo
    out = Path("manifest.json")
    out.write_text(json.dumps(manifest, indent=1))
    print(f"manifest: {len(manifest['files'])} files -> {out}")
    if not args.dry_run:
        api.upload_file(path_or_fileobj=str(out), path_in_repo="manifest.json", repo_id=args.repo, repo_type="model",
                        commit_message="update manifest")


if __name__ == "__main__":
    main()
