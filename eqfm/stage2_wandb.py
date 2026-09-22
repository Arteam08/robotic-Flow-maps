"""Readable, exactly pooled wandb metrics for Stage-2 flow-map training.

Why this exists. The original Stage-2 logging sent, every ``log_every`` steps, the
metrics of that single optimizer step, averaged over its micro-batches. Two problems:

* per-bucket means (by interpolant time t or flow time sigma) were computed per
  micro-batch and an empty bucket contributed 0 to the average, so narrow buckets
  read far too low, and by an amount that changed with the per-GPU batch size of
  whichever copy was running;
* one step of rank-0 samples per point is very noisy.

:class:`WindowStats` keeps GPU-side sums and counts over every rank-0 micro-batch
since the last log point and resolves them with one host sync at log time. Buckets
with no sample in the whole window are omitted, never reported as 0.

:func:`pool_stage2` feeds one micro-batch (weighted batch losses from ``metrics``,
unweighted per-sample tensors collected by ``Stage2FlowMapLoss`` when
``collect_samples`` is on). :func:`new_eval_record` / :func:`eval_payload` log
finished FID evaluations of the run's checkpoints against the training step.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch

Buckets = Sequence[Tuple[float, float]]

T_BUCKETS: Tuple[Tuple[float, float], ...] = ((0.0, 0.1), (0.1, 0.3), (0.3, 0.5), (0.5, 0.9), (0.9, 1.0))
SIGMA_BUCKETS: Tuple[Tuple[float, float], ...] = ((0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 0.9), (0.9, 1.0))

# Section names: wandb groups panels by the text before "/" and sorts sections alphabetically, so the
# leading digit fixes the order. Every key carries its formula; the notation is in GLOSSARY (run notes).
S_FID = "0 FID-10k ADM scale"               # FID of saved checkpoints, published-scale evaluator, raw weights
S_ERR = "1 error unweighted"                # plain per-sample squared errors, pooled over the window
S_LOSS = "2 loss optimized (weighted)"      # what the optimizer minimizes (adaptive per-sample weight)
S_DIAG_T = "3 field error by t"             # student velocity at s = 0 vs teacher field, by interpolant time t
S_LAG_S = "4 Lagrangian error by jump s"    # flow-map velocity residual, by jump size s
S_LAG_T = "5 Lagrangian error by start t"   # same residual, by the start point's interpolant time t
S_SEMI_S = "6 semigroup error by jump s"    # one jump vs two jumps, by total jump size s
S_FIELD = "7 teacher field size"            # size of the guided teacher field where errors are measured
S_OPT = "8 optim and system"
S_VAL = "9 val 16 fixed noises"             # validation map error vs the teacher ODE (EMA and raw weights)
FID_STEP = f"{S_FID}/train step"

GLOSSARY = """\
## Stage-2 wandb metrics: notation
- `x` data latent (4x32x32 = 4096 entries), `z` ~ N(0, I) noise, `t` in [0, 1] interpolation time,
  `x_t = (1-t) z + t x`.
- `b(y)` the distillation target: the frozen Stage-1 teacher field with cfg 1.5, unconditional and guided
  vectors norm-matched to the conditional one (guided mode, no label dropout).
- `v(y, s)` the student network; `X_s(y) = y + s v(y, s)` its flow map, a jump of size `s` in [0, 1]
  along dy/dtau = b(y) (tau = -log(1-s)); `X_1` jumps to the equilibrium (one-step sample).
- `s' <= s` a split point and `s- = (s - s') / (1 - s')`, so jumping `s'` then `s-` equals one jump `s`.
- `||.||^2` squared L2 norm summed over the 4096 entries (per sample), then averaged over samples.
- `AW(e) = e / sg(e + 1e-3)^0.5` the adaptive weight (p = 0.5): the optimized value is ~ ||r|| (not squared).

## Sections (all pooled over every rank-0 sample since the previous log point; empty buckets omitted)
- **0 FID-10k ADM scale/**: FID of 10k class-balanced samples from the checkpoint's RAW weights, torch-fidelity
  TF-Inception vs ADM VIRTUAL_imagenet256 (the published scale). x-axis = `0 FID-10k ADM scale/train step`.
  `K=1` = X_1(z); `K=k paper s0` = X_1 after k-1 jumps of s0; `K=k equal` = k jumps of 1 - 0.001^(1/k).
  Baseline in the same panel: the teacher, 250 Euler steps = 4.69 (2.18 at 50k samples).
- **1 error unweighted/**: the training terms without any weight, the numbers to read.
- **2 loss optimized (weighted)/**: the same terms after the adaptive weight, as minimized.
- **3 field error by t/**: `||v(x_t, 0) - b(x_t)||^2` and the scale-free `||v - b|| / ||b||`, by t.
- **4 Lagrangian error by jump s/**: `r = (1-s) dX_s/ds (x_t) - b(X_s(x_t))`, by s (few-step quality lives at large s).
  MeanFlow runs log **4 MeanFlow error by jump s/** instead: `r = (1-s) v_s(x_t) - sg[b + s (D_x v) b - (1-s) s dv/ds]`
  (per-entry mean, adaptive weight p = 1 with eps scaled by (1-s)^2).
- **5 Lagrangian (or MeanFlow) error by start t/**: same residual by the start point's t.
- **6 semigroup error by jump s/**: `||X_s(x_t) - X_s-(X_s'(x_t))||^2` (two-jump target stop-gradient) and relative
  to the jump length `||X_s(x_t) - x_t||`.
- **7 teacher field size/**: `||b||` where errors are measured, to read squared errors in context.
- **8 optim and system/**: lr, gradient norm before clipping (window mean and max), throughput, GPUs, Slurm job;
  `lr plateau …`: the windowed unweighted loss the lr-slowdown rule watches, the means of the last and previous
  `patience` windows, and the lr scale (halved when the last block is not at least `threshold` below the previous).
- **9 val 16 fixed noises/**: every val interval: `||X_s(z) - ODE_s(z)||` against 100 Euler steps of b over the same
  tau, for EMA and raw weights, and `||b(X_1(z))||`; sample grids.
"""


def _tag(lo: float, hi: float, var: str = "t") -> str:
    close = "]" if hi >= 1.0 else ")"
    return f"  {var}∈[{lo:g}, {hi:g}{close}"


class WindowStats:
    """Exact pooled means over a logging window; nothing leaves the GPU until :meth:`result`."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._sum: Dict[str, torch.Tensor] = {}
        self._cnt: Dict[str, torch.Tensor] = {}
        self._max: Dict[str, torch.Tensor] = {}

    def _acc(self, key: str, s: torch.Tensor, n: torch.Tensor) -> None:
        if key in self._sum:
            self._sum[key] = self._sum[key] + s
            self._cnt[key] = self._cnt[key] + n
        else:
            self._sum[key], self._cnt[key] = s, n

    def add_scalar(self, key: str, value, track_max: bool = False) -> None:
        v = value.detach().float().reshape(()) if torch.is_tensor(value) else torch.tensor(float(value))
        self._acc(key, v, torch.ones((), device=v.device))
        if track_max:
            k = key + " (window max)"
            self._max[k] = torch.maximum(self._max[k], v.to(self._max[k].device)) if k in self._max else v

    def add_samples(self, key: str, values: torch.Tensor,
                    axis: Optional[torch.Tensor] = None, buckets: Optional[Buckets] = None,
                    overall: bool = True, var: str = "t") -> None:
        """Pool a per-sample tensor overall (key) and per bucket of ``axis`` (key + bucket tag)."""
        v = values.detach().float().reshape(-1)
        ok = torch.isfinite(v)
        v = torch.where(ok, v, torch.zeros_like(v))
        if overall:
            self._acc(key, v.sum(), ok.sum().float())
        if axis is None or buckets is None:
            return
        a = axis.detach().float().reshape(-1)
        last = buckets[-1][1]
        for lo, hi in buckets:
            m = ((a >= lo) & ((a <= hi) if hi == last else (a < hi))) & ok
            self._acc(f"{key}{_tag(lo, hi, var)}", (v * m).sum(), m.sum().float())

    def result(self) -> Dict[str, float]:
        keys = list(self._sum)
        mkeys = list(self._max)
        if not keys and not mkeys:
            return {}
        parts = [self._sum[k].reshape(()) for k in keys] + [self._cnt[k].reshape(()) for k in keys] \
            + [self._max[k].reshape(()) for k in mkeys]
        dev = parts[0].device
        flat = torch.stack([p.to(dev) for p in parts]).tolist()
        n = len(keys)
        out: Dict[str, float] = {}
        for i, k in enumerate(keys):
            c = flat[n + i]
            if c > 0:
                out[k] = flat[i] / c
        for j, k in enumerate(mkeys):
            out[k] = flat[2 * n + j]
        return out


def _rel(err_sq: torch.Tensor, norm: torch.Tensor) -> torch.Tensor:
    return err_sq.clamp_min(0).sqrt() / norm.clamp_min(1e-8)


def pool_stage2(stats: WindowStats, metrics: Dict[str, torch.Tensor], samples: Dict[str, torch.Tensor],
                lag: Optional[torch.Tensor] = None, semi: Optional[torch.Tensor] = None,
                objective: str = "lag_semi") -> None:
    """Add one Stage-2 micro-batch. ``metrics`` are the loss's batch scalars (weighted), ``samples`` the
    unweighted per-sample tensors from ``Stage2FlowMapLoss._samples``, ``lag``/``semi`` the weighted
    lag_semi components of this micro-batch."""
    mf = objective in ("scaled_meanflow", "meanflow")
    dname = "MeanFlow" if mf else "Lagrangian"
    dform = ("‖(1−s)v_s(x_t) − sg[b + s(D_x v)b − (1−s)s∂ₛv]‖²" if mf
             else "‖(1−s)∂ₛX_s(x_t) − b(X_s(x_t))‖²")
    sec_s = f"4 {dname} error by jump s"
    sec_t = f"5 {dname} error by start t"
    tnorm = "‖MeanFlow target‖" if mf else "‖b(X_s(x_t))‖"
    rel = "relative ‖r‖ ÷ ‖target‖" if mf else "relative ‖r‖ ÷ ‖b(X_s)‖"
    if mf and semi is None and "semigroup_add_loss" in metrics:
        semi = metrics["semigroup_add_loss"]
    for src, dst in (("total_loss", "total = field + distill + semigroup + anchor + endpoint"),
                     ("diag_loss", "field  AW‖v(x_t,0) − b(x_t)‖²"),
                     ("distill_loss", "distill = MeanFlow" if mf else "distill = Lagrangian + semigroup"),
                     ("terminal_loss", "anchor  ‖X_1(x) − x‖²"),
                     ("terminal_teacher_norm_loss", "endpoint  AW‖b(X_1(z))‖²")):
        if src in metrics:
            stats.add_scalar(f"{S_LOSS}/{dst}", metrics[src])
    if lag is not None:
        stats.add_scalar(f"{S_LOSS}/Lagrangian  AW‖r‖²", lag)
    if semi is not None:
        stats.add_scalar(f"{S_LOSS}/semigroup  AW‖X_s − X_s⁻(X_s′)‖²", semi)

    t = samples.get("t")
    if "diag_raw" in samples:
        e, n = samples["diag_raw"], samples.get("diag_target_norm")
        stats.add_samples(f"{S_ERR}/field at s=0  ‖v(x_t,0) − b(x_t)‖²", e)
        if t is not None:
            stats.add_samples(f"{S_DIAG_T}/‖v(x_t,0) − b(x_t)‖²", e, t, T_BUCKETS, overall=False)
            if n is not None:
                stats.add_samples(f"{S_DIAG_T}/relative ‖v − b‖ ÷ ‖b‖", _rel(e, n), t, T_BUCKETS, overall=False)
                stats.add_samples(f"{S_FIELD}/‖b(x_t)‖", n, t, T_BUCKETS, overall=False)
    if "distill_raw" in samples:
        e, n, s = samples["distill_raw"], samples.get("distill_target_norm"), samples.get("distill_sigma")
        stats.add_samples(f"{S_ERR}/{dname}  {dform}", e)
        if s is not None:
            stats.add_samples(f"{sec_s}/‖r‖²", e, s, SIGMA_BUCKETS, overall=False, var="s")
            if n is not None:
                stats.add_samples(f"{sec_s}/{rel}", _rel(e, n), s, SIGMA_BUCKETS,
                                  overall=False, var="s")
                stats.add_samples(f"{S_FIELD}/{tnorm}", n, s, SIGMA_BUCKETS, overall=False, var="s")
        if t is not None:
            stats.add_samples(f"{sec_t}/‖r‖²", e, t, T_BUCKETS, overall=False)
            if n is not None:
                stats.add_samples(f"{sec_t}/{rel}", _rel(e, n), t, T_BUCKETS, overall=False)
    if "semi_raw" in samples:
        e, n, s = samples["semi_raw"], samples.get("semi_step_norm"), samples.get("semi_sigma")
        stats.add_samples(f"{S_ERR}/semigroup  ‖X_s(x_t) − X_s⁻(X_s′(x_t))‖²", e)
        if s is not None:
            stats.add_samples(f"{S_SEMI_S}/‖X_s − X_s⁻(X_s′)‖²", e, s, SIGMA_BUCKETS, overall=False, var="s")
            if n is not None:
                stats.add_samples(f"{S_SEMI_S}/relative ÷ ‖X_s(x_t) − x_t‖", _rel(e, n), s, SIGMA_BUCKETS,
                                  overall=False, var="s")
    if "anchor_raw" in samples:
        stats.add_samples(f"{S_ERR}/anchor  ‖X_1(x) − x‖²", samples["anchor_raw"])
    if "endpoint_field_norm" in samples:
        stats.add_samples(f"{S_ERR}/endpoint  ‖b(X_1(z))‖ (not squared)", samples["endpoint_field_norm"])


# ---------------------------------------------------------------------------------------------------
# FID of saved checkpoints -> wandb (logged against the training step of the evaluated checkpoint)
# ---------------------------------------------------------------------------------------------------

FID_ROOT_DEFAULT = os.environ.get("EQFM_FID_ROOT", "results/fid")
# Baselines drawn in the FID panel: Stage-1 teacher uw450k, cfg 1.5 + norm matching, 250 Euler steps, 10k samples,
# same ADM-scale evaluator (50k samples: 2.18).
TEACHER_REFS = {"baseline: teacher 250 Euler steps": 4.69}


def sampler_label(name: str) -> str:
    """Directory name of an evaluation -> readable label. X1 | K2_equal | K2_s00.3 (paper schedule) | K8 (old)."""
    import re
    if name == "X1":
        return "K=1  X_1(z)"
    m = re.fullmatch(r"K(\d+)_s0([0-9.]+)", name)
    if m:
        k, s0 = m.groups()
        return f"K={k}  X_1∘X_{s0}^{int(k) - 1} (paper s0={s0})"
    m = re.fullmatch(r"K(\d+)(?:_equal)?", name)
    if m:
        return f"K={m.group(1)}  equal jumps"
    return name


def scan_eval_records(run_name: str, root: Optional[str] = None) -> List[dict]:
    """Finished ADM-scale FIDs of a run: [{id, step, weights, sampler, n, fid}], sorted by step.

    Layout: <root>/<run>/step<S>_<weights>_n<N>/<sampler>/adm_fid.json (written by score_npz.py)."""
    base = Path(root or os.environ.get("EQFM_FID_ROOT", FID_ROOT_DEFAULT)) / run_name
    recs: List[dict] = []
    if not base.is_dir():
        return recs
    for f in sorted(base.glob("step*_*_n*/*/adm_fid.json")):
        try:
            d = json.loads(f.read_text())
            fid = d.get("FID_N")
            if fid is None or not math.isfinite(float(fid)):
                continue
            tag, sampler = f.parent.parent.name, f.parent.name       # step30000_raw_n10000, K8_s00.1
            step = int(tag.split("_")[0][4:])
            weights = tag.split("_")[1]
            n = int(d.get("N") or tag.split("_n")[-1])
            recs.append({"id": f"{tag}/{sampler}", "step": step, "weights": weights, "sampler": sampler,
                         "n": n, "fid": float(fid)})
        except Exception:
            continue
    recs.sort(key=lambda r: (r["step"], r["weights"], r["sampler"]))
    return recs


def load_logged(state_file: Path) -> set:
    try:
        return set(json.loads(state_file.read_text()))
    except Exception:
        return set()


def save_logged(state_file: Path, logged: set) -> None:
    tmp = state_file.with_name(state_file.name + ".tmp")
    tmp.write_text(json.dumps(sorted(logged)))
    os.replace(tmp, state_file)


def next_eval_group(run_name: str, logged: set, root: Optional[str] = None) -> Tuple[Optional[int], List[dict]]:
    """Unlogged results of the lowest not-yet-logged checkpoint step (all weights/samplers of that step)."""
    todo = [r for r in scan_eval_records(run_name, root) if r["id"] not in logged]
    if not todo:
        return None, []
    s = todo[0]["step"]
    return s, [r for r in todo if r["step"] == s]


def eval_payload(step: int, group: List[dict]) -> Dict[str, float]:
    p: Dict[str, float] = {FID_STEP: float(step)}
    for r in group:
        suffix = "" if r["n"] == 10000 else f"  (N={r['n']})"
        w = "" if r["weights"] == "raw" else f"  [{r['weights']} weights]"
        p[f"{S_FID}/{sampler_label(r['sampler'])}{w}{suffix}"] = r["fid"]
    for k, v in TEACHER_REFS.items():
        p[f"{S_FID}/{k} = {v}"] = v
    return p


def define_wandb_metrics(run) -> None:
    run.define_metric(FID_STEP)
    run.define_metric(f"{S_FID}/*", step_metric=FID_STEP, summary="min,last")
    for sec in (S_ERR, S_LOSS, S_DIAG_T, S_LAG_S, S_LAG_T, S_SEMI_S, S_VAL,
                "4 MeanFlow error by jump s", "5 MeanFlow error by start t"):
        run.define_metric(f"{sec}/*", summary="min,last")


def add_glossary(run) -> None:
    try:
        notes = run.notes or ""
        if "## Stage-2 wandb metrics: notation" not in notes:
            run.notes = (notes + "\n\n" + GLOSSARY).strip()
    except Exception:
        pass
