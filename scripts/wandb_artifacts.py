#!/usr/bin/env python
"""Move files through wandb Artifacts with the shared key (no other credential needed).

    python scripts/wandb_artifacts.py put  <path> --name <artifact name> --type model|dataset [--alias latest] [--note "..."]
    python scripts/wandb_artifacts.py get  <artifact name>[:<alias>] --dest <dir>
    python scripts/wandb_artifacts.py list [--type model]

<path> may be a file or a directory. Uses WANDB_API_KEY / WANDB_ENTITY / WANDB_PROJECT from .env or the environment.
Checkpoint convention for the handoff:  name = <run name>, alias = step-<k>, one file per put:
    python scripts/wandb_artifacts.py put results/s2_shifted_meanflow/kept/step_0020000_ema.pt \\
        --name s2_shifted_meanflow --type model --alias step-20000 --note "EMA, FID-2k K8 18.9"
Data convention: dataset artifact imagenet-latents-256 (294 npz shards):
    python scripts/wandb_artifacts.py get imagenet-latents-256:latest --dest data/imagenet-latents-256
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rfm.tracking import load_dotenv, sha256_of  # noqa: E402


def main() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("put"); p.add_argument("path"); p.add_argument("--name", required=True)
    p.add_argument("--type", default="model"); p.add_argument("--alias", action="append", default=[])
    p.add_argument("--note", default=""); p.add_argument("--step", type=int, default=None)
    g = sub.add_parser("get"); g.add_argument("name"); g.add_argument("--dest", required=True)
    ls = sub.add_parser("list"); ls.add_argument("--type", default=None)
    a = ap.parse_args()

    import wandb
    entity = os.environ.get("WANDB_ENTITY"); project = os.environ.get("WANDB_PROJECT", "robotic-flow-maps")
    if not os.environ.get("WANDB_API_KEY"):
        sys.exit("WANDB_API_KEY is not set (put it in .env)")
    os.environ.setdefault("WANDB_SILENT", "true")

    if a.cmd == "put":
        path = Path(a.path)
        if not path.exists():
            sys.exit(f"missing {path}")
        meta = {"note": a.note}
        if a.step is not None:
            meta["step"] = a.step
        if path.is_file():
            meta.update(bytes=path.stat().st_size, sha256=sha256_of(path))
        run = wandb.init(entity=entity, project=project, job_type="upload", name=f"upload-{a.name}", config=meta)
        art = wandb.Artifact(a.name, type=a.type, metadata=meta)
        if path.is_file():
            art.add_file(str(path), name=path.name)
        else:
            art.add_dir(str(path))
        run.log_artifact(art, aliases=list(dict.fromkeys(a.alias + ["latest"])))
        art.wait()
        print(f"uploaded {path} -> {entity}/{project}/{a.name}:{art.version}  aliases {art.aliases}  {meta}")
        run.finish()
    elif a.cmd == "get":
        api = wandb.Api()
        name = a.name if ":" in a.name else a.name + ":latest"
        art = api.artifact(f"{entity}/{project}/{name}")
        out = art.download(root=a.dest)
        print(f"downloaded {name} ({art.size / 1e9:.2f} GB, {len(art.manifest.entries)} files) -> {out}")
        print("metadata:", art.metadata)
    else:
        api = wandb.Api()
        for t in api.artifact_types(project=f"{entity}/{project}"):
            if a.type and t.name != a.type:
                continue
            for coll in t.collections():
                for v in coll.artifacts():
                    print(f"{t.name:8s} {coll.name}:{v.version:5s} {v.size / 1e9:7.2f} GB  aliases={v.aliases}  {v.metadata.get('note', '')}")


if __name__ == "__main__":
    main()
