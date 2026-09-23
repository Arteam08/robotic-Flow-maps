"""Experiment tracking that needs no personal credentials.

Every scalar is ALWAYS written to ``<run_dir>/metrics.jsonl`` (and ``config.json``), so the
experiments never depend on wandb being reachable.  On top of that, wandb is used when it can be:

  WANDB_MODE=disabled        -> local files only
  WANDB_API_KEY set          -> online run under WANDB_ENTITY / WANDB_PROJECT
  no key                     -> wandb *offline* run under <run_dir>/wandb (send it to us, we sync)
  wandb.init() fails         -> falls back to offline, never kills the job

Checkpoints: ``log_checkpoint`` records sha256/step locally, then uploads the file to the Hugging Face
results repo when ``RFM_HF_RESULTS_REPO`` and ``HF_TOKEN`` are set (folder ``<run name>/`` plus a
per-run ``manifest.json``), otherwise as a wandb Artifact when online, otherwise it stays local.

The key comes from the environment or from a git-ignored ``.env`` file next to the repo root
(``KEY=VALUE`` lines).  Nothing is ever written to ``~/.netrc`` or the keychain.

Typical use::

    tr = Tracker(run_dir="results/exp01_seed0", config=vars(args))
    for step in ...:
        tr.log(step, {"loss": loss.item(), "lr": lr})
    tr.log_checkpoint("results/exp01_seed0/ema_020000.pt", step=20000, aliases=["latest"])
    tr.finish()

The run id is persisted in ``<run_dir>/wandb_id.txt`` so restarting the script appends to the
same wandb run instead of creating a new one.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

DEFAULT_PROJECT = "robotic-flow-maps"


def load_dotenv(path: Optional[Path] = None) -> Dict[str, str]:
    """Read ``.env`` (repo root by default) into os.environ without overriding existing vars."""
    path = path or (Path(__file__).resolve().parent.parent / ".env")
    loaded: Dict[str, str] = {}
    if os.environ.get("RFM_NO_DOTENV") or not path.exists():   # RFM_NO_DOTENV=1: ignore .env (tests)
        return loaded
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k and v and k not in os.environ:
            os.environ[k] = v
            loaded[k] = v
    return loaded


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def _jsonable(v: Any) -> Any:
    if isinstance(v, Path):
        return str(v)
    if hasattr(v, "item"):
        try:
            return v.item()
        except Exception:  # noqa: BLE001
            return str(v)
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, dict):
        return {k: _jsonable(x) for k, x in v.items()}
    return v


class Tracker:
    def __init__(
        self,
        run_dir: str | Path,
        config: Optional[Dict[str, Any]] = None,
        name: Optional[str] = None,
        project: Optional[str] = None,
        entity: Optional[str] = None,
        tags: Optional[Iterable[str]] = None,
        mode: Optional[str] = None,
        resume: bool = True,
        group: Optional[str] = None,
    ) -> None:
        load_dotenv()
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.name = name or self.run_dir.name
        self.config = _jsonable(dict(config or {}))
        (self.run_dir / "config.json").write_text(json.dumps(self.config, indent=1, sort_keys=True))
        self._jsonl = open(self.run_dir / "metrics.jsonl", "a", buffering=1)
        self._t0 = time.time()

        self.mode = (mode or os.environ.get("WANDB_MODE") or ("online" if os.environ.get("WANDB_API_KEY") else "offline")).lower()
        self.project = project or os.environ.get("WANDB_PROJECT") or DEFAULT_PROJECT
        self.entity = entity or os.environ.get("WANDB_ENTITY") or None
        self.run = None
        self.wandb = None
        if self.mode != "disabled":
            self.run = self._init_wandb(list(tags or []), resume, group)
        where = "disabled" if self.run is None and self.mode == "disabled" else (
            f"{self.run.mode if hasattr(self.run, 'mode') else self.mode}: {getattr(self.run, 'url', None) or self.run_dir / 'wandb'}"
            if self.run is not None else "local only (wandb unavailable)")
        print(f"[tracker] run '{self.name}' -> {self.run_dir}  | wandb {where}", flush=True)

    # ------------------------------------------------------------------ wandb
    def _init_wandb(self, tags, resume, group):
        try:
            import wandb
        except Exception as e:  # noqa: BLE001
            print(f"[tracker] wandb import failed ({e!r}); local files only", flush=True)
            return None
        self.wandb = wandb
        id_file = self.run_dir / "wandb_id.txt"
        if resume and id_file.exists():
            run_id = id_file.read_text().strip()
        else:
            run_id = uuid.uuid4().hex[:8]
            id_file.write_text(run_id + "\n")
        kwargs = dict(project=self.project, entity=self.entity, name=self.name, id=run_id, resume="allow",
                      dir=str(self.run_dir), tags=tags, group=group, config=self.config)
        if self.mode == "online" and not os.environ.get("WANDB_API_KEY"):
            print("[tracker] WANDB_MODE=online but no WANDB_API_KEY; using offline", flush=True)
            self.mode = "offline"
        try:
            return wandb.init(mode=self.mode, **kwargs)
        except Exception as e:  # noqa: BLE001
            print(f"[tracker] wandb.init({self.mode}) failed ({e!r}); falling back to offline", flush=True)
            try:
                self.mode = "offline"
                return wandb.init(mode="offline", **kwargs)
            except Exception as e2:  # noqa: BLE001
                print(f"[tracker] offline init failed too ({e2!r}); local files only", flush=True)
                return None

    # ---------------------------------------------------------------- logging
    def log(self, step: int, metrics: Dict[str, Any], prefix: Optional[str] = None) -> None:
        rec = {(f"{prefix}/{k}" if prefix else k): _jsonable(v) for k, v in metrics.items()}
        self._jsonl.write(json.dumps({"step": int(step), "time": round(time.time() - self._t0, 1), **rec}) + "\n")
        if self.run is not None:
            try:
                self.run.log(rec, step=int(step))
            except Exception as e:  # noqa: BLE001
                print(f"[tracker] wandb.log failed ({e!r})", flush=True)

    def log_image(self, step: int, name: str, path: str | Path) -> None:
        """Log an image that is already saved on disk (PNG grid etc.)."""
        if self.run is not None and self.wandb is not None:
            try:
                self.run.log({name: self.wandb.Image(str(path))}, step=int(step))
            except Exception as e:  # noqa: BLE001
                print(f"[tracker] wandb image log failed ({e!r})", flush=True)

    def log_checkpoint(self, path: str | Path, step: int, aliases: Iterable[str] = ("latest",),
                       metadata: Optional[Dict[str, Any]] = None, artifact_name: Optional[str] = None) -> str:
        """Record a checkpoint: sha256 + size to <run_dir>/checkpoints.jsonl, and upload it as a
        wandb Artifact when online.  Returns the sha256 so it can be quoted in results."""
        path = Path(path)
        digest = sha256_of(path)
        rec = {"step": int(step), "file": str(path), "bytes": path.stat().st_size, "sha256": digest,
               "time": time.strftime("%Y-%m-%dT%H:%M:%S"), **_jsonable(metadata or {})}
        with open(self.run_dir / "checkpoints.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        hf_repo = os.environ.get("RFM_HF_RESULTS_REPO")
        if hf_repo and os.environ.get("HF_TOKEN"):
            self._hf_upload(path, rec, hf_repo)
        elif self.run is not None and self.wandb is not None and self.mode == "online":
            try:
                art = self.wandb.Artifact(artifact_name or f"{self.name}-ckpt", type="model", metadata=rec)
                art.add_file(str(path), name=path.name)
                self.run.log_artifact(art, aliases=list(aliases) + [f"step-{int(step)}"])
                print(f"[tracker] uploaded {path.name} ({rec['bytes'] / 1e6:.0f} MB) as wandb artifact", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"[tracker] artifact upload failed ({e!r}); file kept locally", flush=True)
        else:
            print(f"[tracker] checkpoint {path.name} recorded locally only (no HF token / offline)", flush=True)
        return digest

    def _hf_upload(self, path: Path, rec: Dict[str, Any], repo: str) -> None:
        """Upload <path> to <repo>/<run name>/<file> and merge <rec> into <repo>/<run name>/manifest.json.
        Synchronous (a few seconds for a 100 MB file); failures never stop training."""
        try:
            os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")  # keep training logs readable
            from huggingface_hub import HfApi, hf_hub_download
            from huggingface_hub.utils import EntryNotFoundError
            api = HfApi(token=os.environ["HF_TOKEN"])
            folder = self.name
            try:
                manifest = json.load(open(hf_hub_download(repo, f"{folder}/manifest.json", token=api.token)))
            except EntryNotFoundError:
                manifest = {"repo": repo, "run": folder, "files": []}
            entry = {"file": path.name, **{k: v for k, v in rec.items() if k != "file"}}
            manifest["files"] = [e for e in manifest["files"] if e["file"] != path.name] + [entry]
            manifest["config"] = self.config
            api.upload_file(path_or_fileobj=str(path), path_in_repo=f"{folder}/{path.name}", repo_id=repo,
                            commit_message=f"{folder}: {path.name} step {rec['step']} sha256 {rec['sha256'][:12]}")
            api.upload_file(path_or_fileobj=json.dumps(manifest, indent=1).encode(), path_in_repo=f"{folder}/manifest.json",
                            repo_id=repo, commit_message=f"{folder}: manifest")
            print(f"[tracker] uploaded {path.name} ({rec['bytes'] / 1e6:.0f} MB) to {repo}/{folder}/", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[tracker] HF upload failed ({e!r}); file kept locally", flush=True)

    def finish(self) -> None:
        self._jsonl.close()
        if self.run is not None:
            try:
                self.run.finish()
            except Exception as e:  # noqa: BLE001
                print(f"[tracker] wandb.finish failed ({e!r})", flush=True)
            if self.mode == "offline":
                print(f"[tracker] offline wandb run saved under {self.run_dir / 'wandb'}; "
                      f"pack it with scripts/pack_offline_runs.sh and send it, or run `wandb sync` with a key", flush=True)


def metrics_frame(run_dir: str | Path):
    """Load <run_dir>/metrics.jsonl as a pandas DataFrame (one row per log call)."""
    import pandas as pd
    rows = [json.loads(l) for l in open(Path(run_dir) / "metrics.jsonl") if l.strip()]
    return pd.DataFrame(rows)
