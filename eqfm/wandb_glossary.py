"""Self-describing wandb metric names and a per-run glossary.

Trainers compute metrics under short internal names (``eqm_raw_t0.9_1``).
:class:`ReadableLogger` renames each one to a wandb key a newcomer can read
without the code::

    "<n> <section>/<words>  <short formula>"      e.g.  "2 loss by t/mse  t∈[0.9, 1]"

The leading digit orders the sections in the wandb workspace (sections sort
alphabetically). At start-up the logger writes a glossary -- the run recipe,
the notation, and every key with its definition and good direction -- into
the run notes (wandb "Overview" tab), a wandb Table, and
``<run_dir>/wandb_glossary.md``.

Internal keys with no glossary entry are still logged, under
``9 other/<internal key>``, and reported once on stdout, so a gap in the
glossary is visible instead of silent.

Stage 2 (flow-map distillation) keys are registered under the names the
Stage-2 trainer logs today (``train/distill_raw_s0.9_0.99`` ...), so that
trainer can adopt the logger by wrapping its existing ``wandb_run.log`` call.

``python -m eqfm.wandb_glossary`` prints both glossaries as markdown with the
legacy key of every metric; ``docs/wandb_metrics.md`` is generated from it and
decodes runs logged before this module existed.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

IMAGENET_TRAIN_IMAGES = 1_281_167
T_BUCKETS = ((0.0, 0.1), (0.1, 0.5), (0.5, 0.9), (0.9, 1.0))
SIGMA_BUCKETS = ((0.0, 0.5), (0.5, 0.9), (0.9, 0.99), (0.99, 1.0))
_NUM = r"[0-9]+(?:\.[0-9]+)?"
B = rf"(?P<lo>{_NUM})_(?P<hi>{_NUM})"   # bucket suffix "lo_hi"


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Entry:
    """One metric family.

    ``internal`` is a regex matched against the whole internal key; named
    groups ``lo``/``hi`` mark a bucketed family and fill ``{b}`` (for example
    ``t∈[0.5, 0.9)``). Other named groups are available to the templates.
    ``key=None`` drops the metric (e.g. a duplicate of the wandb step).
    """

    internal: str
    key: Optional[str]
    meaning: str
    good: str = ""          # "↓" lower is better, "↑", "→ 0", "→ 1", "≈ <ref>", "" = context only
    legacy: str = ""        # key used by runs logged before this module
    bucket_var: str = "t"   # variable the lo/hi bucket is over
    buckets: Sequence[Tuple[float, float]] = T_BUCKETS
    when: Optional[Callable[[dict], bool]] = None   # list in the glossary only if true


def _fmt_num(s: str) -> str:
    x = float(s)
    return f"{x:g}"


def _bucket(var: str, lo: str, hi: str) -> str:
    closed = math.isclose(float(hi), 1.0)
    return f"{var}∈[{_fmt_num(lo)}, {_fmt_num(hi)}{']' if closed else ')'}"


def _bucket_list(var: str, buckets) -> str:
    return ", ".join(_bucket(var, str(lo), str(hi)) for lo, hi in buckets)


class _SafeDict(dict):
    def __missing__(self, key):   # leave unknown placeholders visible instead of crashing
        return "{" + key + "}"


def _fill(template: str, fields: dict) -> str:
    return template.format_map(_SafeDict(fields))


# ---------------------------------------------------------------------------
# Stage 1: EqM field training (scripts/train_stage1_eqm.py)
# ---------------------------------------------------------------------------

NOTATION_STAGE1: List[Tuple[str, str]] = [
    ("x1", "clean data latent: SD-VAE latent × 0.18215, shape 4×32×32, d = 4096 entries"),
    ("z", "noise ~ N(0, I), same shape; ‖z‖ ≈ √d = 64"),
    ("t", "interpolation time in [0, 1): t = 0 is pure noise, t → 1 is data"),
    ("x_t", "(1 − t)·z + t·x1, the network input"),
    ("U_t", "{target}, the regression target; it vanishes at data, so data are fixed points"),
    ("b(x)", "the network output: an autonomous field (no time input; the t-embedder is fed 0). "
             "Sampling integrates dx/dτ = b(x) from noise"),
    ("mse", "per-sample (1/d)·‖{resid}‖², the squared error averaged over the d latent entries"),
    ("w(t)", "{w_def}"),
    ("‖·‖", "L2 norm over the d entries of one sample, then averaged over the batch"),
    ("θ, θ_ema", "current weights; exponential moving average of the weights (decay {ema}), used for samples and FID"),
    ("λ", "weight of the data anchor E‖b(x1)‖²/d in the total loss (λ = {lam})"),
]


def _s1_weighted(c):
    return c.get("weighted", True)


STAGE1: List[Entry] = [
    # 0 init -- logged once at step 0 of a fresh run
    Entry("init/scale", "0 init/head scale s",
          "Output layer (final_layer.linear weight and bias) multiplied by s once before training. Converts a field "
          "trained with another clock to ours; the released EqM regresses 4·min(1, 5(1−t))·(x1 − z), hence s ≈ 1/9. "
          "1 = no rescale.", legacy="(init_probe.json only)"),
    Entry("init/objective", "0 init/objective at start  {obj}",
          "Probe objective (see 4 probe) of the starting weights, after the head rescale.", "↓",
          legacy="(init_probe.json: init/uw_loss)"),
    Entry("init/b_data_rms", "0 init/RMS b(x1) at start",
          "RMS of the starting field on clean probe latents; 0 = data are exact fixed points.", "→ 0",
          legacy="(init_probe.json: init/b_data_rms)"),
    Entry("init/x1_norm", "0 init/‖x1‖ data reference",
          "Mean norm of clean probe latents: where a sampler should end. Compare 5 samples/‖x‖ at last step.", ""),

    # 1 loss
    Entry("total_loss", "1 loss/total (optimized)",
          "The scalar that is back-propagated: {total_formula}.", "↓",
          legacy="train/loss, train/total_loss"),
    Entry("eqm_loss", "1 loss/objective  {obj}",
          "Batch estimate of the Stage-1 objective {obj}, {tlaw}. Compare runs by this (or by 4 probe/objective, "
          "which has no batch noise).", "↓", legacy="train/eqm_loss"),
    Entry("eqm_raw", "1 loss/unweighted  E[mse]",
          "Plain batch mean of mse without the time weight; dominated by small t, where mse is largest.", "↓",
          legacy="train/eqm_raw"),
    Entry("eqm_loss_max", "1 loss/worst sample  max mse",
          "Largest per-sample mse seen since the last log. Spikes flag outlier images or instability.", "↓",
          legacy="train/eqm_loss_max"),
    Entry("mse_cond", "1 loss/mse, class label",
          "Mean mse over samples that kept their class label: the conditional branch b(x | y) used by CFG.", "↓"),
    Entry("mse_null", "1 loss/mse, null label",
          "Mean mse over samples whose label was replaced by the null class: the unconditional branch CFG needs. "
          "Missing on steps where no label was dropped.", "↓", when=lambda c: c.get("cfg_p", 0) > 0),
    Entry("data_anchor", "1 loss/anchor  E‖b(x1)‖²/d",
          "Mean squared field on clean data; pins data as fixed points. Enters the total with weight λ.", "→ 0",
          legacy="train/data_anchor", when=lambda c: c.get("lam", 0) > 0),
    Entry("time_pred_loss", "1 loss/time head  E(t̂ − t)²",
          "Auxiliary head predicting the noise level from features; enters the total with weight μ.", "↓",
          legacy="train/time_pred_loss", when=lambda c: c.get("mu", 0) > 0),
    Entry("time_pred_mae", "1 loss/time head  E|t̂ − t|",
          "Mean absolute error of the auxiliary noise-level prediction.", "↓",
          legacy="train/time_pred_mae", when=lambda c: c.get("mu", 0) > 0),

    # 2 loss by t
    Entry(rf"eqm_raw_t{B}", "2 loss by t/mse  {b}",
          "Mean mse over the window's samples with t in the bucket.", "↓",
          legacy="train/eqm_raw_t{lo}_{hi}"),
    Entry(rf"eqm_share_t{B}", "2 loss by t/share of objective  {b}",
          "Fraction of the objective Σ w·mse contributed by samples with t in the bucket (the shares sum to 1): "
          "which noise levels the gradient actually trains.", ""),

    # 3 field by t
    Entry(rf"eqm_pred_norm(?:_t{B})?", "3 field by t/‖b(x_t)‖  {b}",
          "Mean norm of the network output. Should track ‖U_t‖ in the same bucket.", "≈ ‖U_t‖",
          legacy="train/eqm_pred_norm[_t{lo}_{hi}]"),
    Entry(rf"eqm_target_norm(?:_t{B})?", "3 field by t/‖U_t‖ target  {b}",
          "Mean norm of the regression target {target}. Fixed by the data and the time law, not by the model.", "",
          legacy="train/eqm_target_norm[_t{lo}_{hi}]"),
    Entry(rf"eqm_cos(?:_t{B})?", "3 field by t/cos(b, U_t)  {b}",
          "Mean cosine between output and target: direction agreement, independent of scale.", "→ 1"),
    Entry(rf"eqm_relerr(?:_t{B})?", "3 field by t/‖b − U_t‖ / ‖U_t‖  {b}",
          "Mean relative error per sample: 0 = exact, 1 = no better than predicting 0.", "→ 0"),

    # 4 probe -- fixed batches, fixed z and t, no gradient, every val step
    Entry(r"probe_(?P<w>raw|ema)/loss", "4 probe/objective  {obj}, {W} θ",
          "Objective on {nb} fixed training batches with fixed noise and fixed t, no gradient. Free of batch noise, "
          "so a change means the weights changed.", "↓", legacy="val/uw_loss_raw, val/uw_loss_ema"),
    Entry(r"probe_(?P<w>raw|ema)/b_data_rms", "4 probe/RMS b(x1)  {W} θ",
          "RMS of the field on the clean probe latents; 0 = data are exact fixed points.", "→ 0",
          legacy="val/b_data_rms_raw, val/b_data_rms (EMA)"),
    Entry(rf"probe_(?P<w>raw|ema)/eqm_raw_t{B}", "4 probe/mse  {b}, {W} θ",
          "Probe mse restricted to the fixed samples with t in the bucket.", "↓"),
    Entry(rf"probe_(?P<w>raw|ema)/eqm_relerr_t{B}", "4 probe/‖b − U_t‖ / ‖U_t‖  {b}, {W} θ",
          "Probe relative error restricted to the fixed samples with t in the bucket.", "→ 0"),

    # 5 samples -- EMA weights, fixed noise and classes
    Entry("samples/grid", "5 samples/grid (EMA θ)",
          "{n_val} samples: Euler on dx/dτ = b(x), K = {K} steps to ε_stop = {eps_stop}, CFG w = {cfg}, "
          "fixed noise and classes so grids are comparable across steps.", "", legacy="val/samples"),
    Entry("samples/b_norm_first", "5 samples/‖b‖ at first step",
          "Mean field norm at the start of the sampler (pure noise).", "", legacy="val/velocity_norm_first"),
    Entry("samples/b_norm_last", "5 samples/‖b‖ at last step",
          "Mean field norm where the sampler stops; → 0 means the trajectories ended at fixed points of b.", "→ 0",
          legacy="val/velocity_norm_last"),
    Entry("samples/b_norm_max", "5 samples/max ‖b‖ along path",
          "Largest mean field norm over the sampler steps.", "", legacy="val/velocity_norm_max"),
    Entry("samples/x_norm_last", "5 samples/‖x‖ at last step",
          "Mean norm of the final latents. Should match 0 init/‖x1‖ data reference.", "≈ ‖x1‖",
          legacy="val/latent_norm_last"),
    Entry("samples/cfg_scale", "5 samples/CFG scale w",
          "Guidance used for the grid: b_null + w·(b_class − b_null).", "", legacy="val/cfg_scale"),

    # 6 optim
    Entry("lr", "6 optim/learning rate", "AdamW learning rate: {lr_sched}.", "",
          legacy="train/lr"),
    Entry("grad_norm", "6 optim/‖grad‖ before clip",
          "Global L2 norm of the gradient over trainable parameters before clipping at {clip}, mean over the "
          "optimizer steps since the last log.", "", legacy="train/grad_norm"),
    Entry("grad_norm_max", "6 optim/max ‖grad‖ before clip",
          "Largest pre-clip gradient norm since the last log.", ""),
    Entry("clip_frac", "6 optim/fraction of steps clipped",
          "Fraction of optimizer steps since the last log whose gradient norm exceeded {clip}.", ""),
    Entry("param_norm", "6 optim/‖θ‖", "Global L2 norm of the trainable parameters.", ""),
    Entry("ema_gap", "6 optim/‖θ − θ_ema‖ / ‖θ‖",
          "Relative distance between current and EMA weights: large early while the EMA lags, then roughly flat.", ""),
    Entry("init_gap", "6 optim/‖θ − θ_start‖ / ‖θ_start‖",
          "Relative distance of the current weights from the starting checkpoint ({init}, after the head rescale): "
          "how far training has moved the model. Logged at probe steps.", ""),

    # 7 batch
    Entry("cfg_drop_frac", "7 batch/null-label fraction",
          "Fraction of the batch whose label was replaced by the null class (CFG dropout, p = {cfg_p}).", "≈ p",
          legacy="train/cfg_drop_frac"),
    Entry(r"time_weight_(?P<stat>mean|min|max)", "7 batch/w(t) {stat}",
          "{stat} of the time weight w(t) since the last log (mean: average of batch means).", "",
          legacy="train/time_weight_{stat}",
          when=_s1_weighted),
    Entry("time_weight_ess", "7 batch/effective samples  (Σw)²/Σw²",
          "Effective sample size of a weighted micro-batch of B = {bs}; B / ESS is the variance cost of w(t).",
          "↑", legacy="train/time_weight_ess", when=_s1_weighted),
    Entry("images_seen", "7 batch/images seen", "Training images processed, all GPUs and micro-batches.", ""),
    Entry("epoch", "7 batch/epochs", "images seen / 1,281,167 (ImageNet-1k train split).", ""),

    # 8 system
    Entry("images_per_s", "8 system/images per second",
          "Training throughput since the last log (or validation), all GPUs.", "↑",
          legacy="train/samples_per_s"),
    Entry("sec_per_step", "8 system/seconds per step", "Wall seconds per optimizer step since the last log.", "↓"),
    Entry("data_wait_frac", "8 system/data wait fraction",
          "Fraction of wall time spent waiting for the next batch; high = input pipeline is the bottleneck.", "↓"),
    Entry("gpu_mem_gb", "8 system/peak GPU memory GB",
          "Peak allocated CUDA memory on rank 0 since the last log.", ""),
    Entry("step", None, "Duplicate of the wandb step axis.", legacy="train/step"),
]

SUMMARIES_STAGE1 = {"1 loss": "min,last", "4 probe": "min,last", "2 loss by t": "last"}


def stage1_context(args, world_size: int = 1) -> dict:
    """Template values for the Stage-1 glossary, taken from the trainer args."""
    g = lambda k, d=None: getattr(args, k, d)
    tsamp = g("time_sampler", "uniform_weighted")
    weighted = tsamp in ("uniform_weighted", "uniform_clock_weighted")
    clock_spec = g("clock", "linear") or "linear"
    clock_lam = float(g("clock_lambda", 1.0) or 1.0)
    lam_txt = "" if clock_lam == 1.0 else f"{clock_lam:g}·"
    if clock_spec == "linear":
        c_txt = f"{lam_txt}(1 − t)"
    elif clock_spec.startswith("trunc"):
        a_clk = float(clock_spec.partition(":")[2] or g("clock_a", 0.8))
        c_txt = f"{lam_txt}min(1, (1 − t)/{1 - a_clk:g})"
    else:
        c_txt = f"{lam_txt}c(t) [{clock_spec}]"
    ceps = g("clock_weight_eps", 1e-3)
    eps = g("eps_train", 1e-3)
    velocity = g("parameterization", "velocity") == "velocity"
    lam = float(g("data_anchor", 0.0) or 0.0)
    mu = float(g("time_pred_weight", 0.0) or 0.0)
    obj = "E[w·mse]" if weighted else "E[mse]"
    laws = {
        "uniform_weighted": f"t ~ U[0, 1−ε] weighted by w(t), ε = {eps:g}",
        "uniform_clock_weighted": f"t ~ U[0, 1) weighted by w(t) = 1/((c(t) + ε)·Z), c(t) = {c_txt}, ε = {ceps:g}",
        "canonical": f"t ~ p(t) ∝ 1/(1 − t) on [0, 1−ε], ε = {eps:g}, no weight",
        "uniform": "t ~ U[0, 1), no weight",
        "importance": f"t = 1 − u^{g('importance_k', 2.0):g}, u ~ U[0, 1), no weight",
    }
    total = obj + (f" + λ·E‖b(x1)‖²/d (λ = {lam:g})" if lam > 0 else "") + \
        (f" + μ·E(t̂ − t)² (μ = {mu:g})" if mu > 0 else "")
    bs = g("batch_size", 0)
    return dict(
        weighted=weighted, obj=obj, tlaw=laws.get(g("time_sampler", ""), str(g("time_sampler"))),
        w_def=((f"1/((c(t) + ε)·Z), c(t) = {c_txt}, ε = {ceps:g}, Z = ∫₀¹ dt/(c + ε) normalises the mean weight to 1 "
                f"(the dτ = dt/c measure, regularised by ε)") if tsamp == "uniform_clock_weighted" else
               (f"(1 − ε)/(log(1/ε)·(1 − t)), ε = {eps:g}: importance weight that makes E_t[w·mse] with uniform t "
                f"equal the canonical objective with t ~ p(t) ∝ 1/(1 − t)") if weighted else "1 (no time weight)"),
        resid="b(x_t) − U_t" if velocity else "D(x_t) − x1  (denoiser output D, field b = D − x)",
        target=f"U_t = {c_txt}·(x1 − z)" if velocity else "x1 (denoiser parameterization)",
        lam=lam, mu=mu, total_formula=total, ema=g("ema_decay", 0.9999), clip=g("grad_clip", 1.0),
        cfg_p=g("cfg_dropout_prob", 0.0), warmup=g("warmup_steps", 0), K=g("val_num_steps", 0),
        lr_sched=(f"linear warmup over {g('warmup_steps', 0)} steps, then "
                  + (f"cosine decay {g('lr')} -> {g('lr_final', 0.0)} from step {g('lr_decay_start', 0)} "
                     f"to {g('steps')}" if g("lr_schedule", "constant") == "cosine" else "constant")),
        eps_stop=g("val_eps_stop", 1e-3), cfg=g("val_cfg_scale", 1.0), n_val=g("val_n_samples", 16),
        nb=g("val_loss_batches", 0), bs=bs, init=str(g("ckpt", "")),
        buckets=_bucket_list("t", T_BUCKETS),
    )


def stage1_header(args, world_size: int = 1) -> str:
    g = lambda k, d=None: getattr(args, k, d)
    c = stage1_context(args, world_size)
    gb = g("batch_size", 0) * world_size * max(1, g("grad_accum_steps", 1) or 1)
    start = "random init" if g("init_from_scratch", False) else f"`{g('ckpt')}` (weights: {g('ckpt_key', 'auto')})"
    return "\n".join([
        f"# Stage 1: equilibrium field b(x) · run `{Path(str(g('results_dir', '.'))).name}/{g('run_name', '')}`",
        "",
        f"- **Start**: {start}; output head × {g('init_output_scale', '1')}; trainable = `{g('trainable') or 'all'}`; "
        f"time conditioning `{g('time_conditioning')}`",
        f"- **Objective**: minimize {c['total_formula']}, {c['tlaw']}",
        f"- **Optimizer**: AdamW lr {g('lr')} ({c['lr_sched']}), clip {g('grad_clip')}, "
        f"global batch {gb} = {g('batch_size')} × {world_size} GPUs × {g('grad_accum_steps', 1)} accum, "
        f"EMA {g('ema_decay')}, bf16 {g('bf16')}",
        f"- **Data**: `{g('shards')}`, hflip {g('hflip', False)}, CFG label dropout p = {g('cfg_dropout_prob')}",
        f"- **Probe** (section 4): {g('val_loss_batches')} fixed batches, every {g('val_every')} steps · "
        f"**Samples** (section 5): EMA θ, Euler K = {g('val_num_steps')}, CFG w = {g('val_cfg_scale')}",
        f"- **Logging**: a point every {g('log_every')} steps. Sections 1-3 and 7 pool every rank-0 micro-batch "
        f"since the previous point, so per-t buckets are exact means over all samples in that window "
        f"(a bucket with no sample is skipped, never 0).",
    ])


# ---------------------------------------------------------------------------
# Stage 2: flow-map distillation (scripts/train_stage2_flowmap.py), keys as logged today
# ---------------------------------------------------------------------------

NOTATION_STAGE2: List[Tuple[str, str]] = [
    ("x1, z, t, x_t", "data latent (d = 4096 entries), noise, interpolation time, x_t = (1 − t)·z + t·x1"),
    ("T(x, c)", "frozen Stage-1 field (the teacher), EMA weights, class c or the null class"),
    ("b(x)", "teacher target field. Guided mode: CFG w with norm matching, so ‖b‖ = ‖T(x, y)‖; "
             "branch mode: T(x, y) with the student's label"),
    ("v(x, σ, y)", "the student network"),
    ("X_σ(x)", "x + σ·v(x, σ, y): the flow map, a jump of size σ along dx/dτ = b(x)"),
    ("σ", "jump size in [0, 1]; the jump covers ODE time τ = −log(1 − σ); X_1 jumps to the equilibrium"),
    ("a ⊕ b", "a + b − ab: composing jumps a then b gives a jump of a ⊕ b"),
    ("‖r‖²", "squared L2 norm summed over all d entries (divide by 4096 for a per-entry value)"),
    ("AW(r²)", "adaptive weighting mean_i r_i² / (sg(r_i²) + ε)^p; with p = 0.5 it is ≈ mean ‖r‖ (unsquared)"),
    ("raw", "unweighted batch mean of ‖r‖², no adaptive weight"),
]

S = rf"(?P<lo>{_NUM})_(?P<hi>{_NUM})"

STAGE2: List[Entry] = [
    # 1 loss
    Entry("train/loss", "1 loss/total (optimized)",
          "w_diag·diag + w_distill·distill + w_term·terminal + w_tnorm·terminal-at-teacher-eq + "
          "w_snorm·terminal-at-student-eq + w_inv·invariance (+ μ·time head).", "↓"),
    Entry("train/diag_loss", "1 loss/diagonal  AW‖v(x_t, 0) − b(x_t)‖²",
          "At σ = 0 the student's velocity must equal the teacher field (the map's derivative at the start).", "↓"),
    Entry("train/distill_loss", "1 loss/distill  (lag_semi: lag + λ_semi·semi)",
          "Off-diagonal distillation term of the chosen --objective, averaged over micro-batches.", "↓"),
    Entry("train/lag_loss", "1 loss/Lagrangian  AW‖(1−σ)∂σX_σ − b(X_σ)‖²",
          "The map moves along the teacher field: its σ-derivative (forward JVP) matches b at the map's output. "
          "Last micro-batch only, so it does not add up to distill.", "↓"),
    Entry("train/semi_loss", "1 loss/semigroup  AW‖X_σ − X_σ⁻(X_σ')‖²",
          "One long jump equals two shorter jumps (σ' ⊕ σ⁻ = σ); the two-jump target is stop-gradient. "
          "Last micro-batch only.", "↓"),
    Entry("train/terminal_loss", "1 loss/terminal keeps data  ‖X_1(x1) − x1‖²",
          "The full jump must leave clean data unchanged (data are equilibria). Unweighted, squared.", "→ 0"),
    Entry("train/terminal_teacher_norm_loss", "1 loss/terminal at teacher eq.  AW‖b(X_1(z))‖²",
          "The one-step output from noise must be an equilibrium of the teacher field (gradient flows through the "
          "frozen teacher). Small random subset of the batch.", "→ 0"),
    Entry("train/terminal_student_norm_loss", "1 loss/terminal at student eq.  AW‖v(X_1(z), 0)‖²",
          "Same, with the student's own σ = 0 velocity. 0 when its weight is 0.", "→ 0"),
    Entry("train/terminal_invariance_loss", "1 loss/terminal invariance  AW‖X_1(x_t) − X_1(x_t + δb)‖²",
          "Moving the start point along the teacher field must not change the endpoint. 0 when its weight is 0.",
          "→ 0"),
    Entry("train/diag_raw", "1 loss/diagonal raw  ‖v(x_t, 0) − b‖²", "Unweighted mean of the diagonal residual.", "↓"),
    Entry("train/distill_raw", "1 loss/distill raw  ‖r‖²",
          "Unweighted mean of the objective's residual (lag_semi: the Lagrangian part).", "↓"),
    Entry("train/distill_raw_max", "1 loss/distill worst sample  max ‖r‖²",
          "Largest per-sample distill residual (max per micro-batch, averaged).", "↓"),
    Entry("train/semi_raw", "1 loss/semigroup raw  ‖X_σ − X_σ⁻(X_σ')‖²", "Unweighted mean of the semigroup residual.",
          "↓"),
    Entry("train/time_pred_loss", "1 loss/time head  E(t̂ − t)²", "Auxiliary noise-level head (weight μ).", "↓"),
    Entry("train/time_pred_mae", "1 loss/time head  E|t̂ − t|", "Mean absolute error of the noise-level head.", "↓"),

    # 2 residual by t (base point) / 3 residual by σ (jump size)
    Entry(rf"train/diag_raw_t{B}", "2 residual by t/diagonal ‖v(x_t, 0) − b‖²  {b}",
          "Diagonal residual over samples whose base point has t in the bucket. Empty buckets count as 0 in the "
          "micro-batch mean (biased low at small batch).", "↓"),
    Entry(rf"train/distill_raw_t{B}", "2 residual by t/distill ‖r‖²  {b}",
          "Distill residual over samples whose base point has t in the bucket (empty buckets count as 0).", "↓"),
    Entry(rf"train/distill_raw_s{S}", "3 residual by σ/distill ‖r‖²  {b}",
          "Distill residual over samples whose jump size σ is in the bucket (lag_semi: the Lagrangian σ). "
          "A bucket above --sigma-max always reads 0.", "↓", bucket_var="σ", buckets=SIGMA_BUCKETS),
    Entry(rf"train/semi_raw_s{S}", "3 residual by σ/semigroup ‖X_σ − X_σ⁻(X_σ')‖²  {b}",
          "Semigroup residual over samples whose outer jump σ is in the bucket.", "↓",
          bucket_var="σ", buckets=SIGMA_BUCKETS),

    # 4 map vs teacher ODE
    Entry(rf"val/map_err_s(?P<s>{_NUM})", "4 map vs teacher ODE/‖X_σ(z) − ODE_σ(z)‖  σ={s_short}",
          "Norm (not squared, over 4096 entries) of one student jump from noise minus 100 Euler steps of the teacher "
          "ODE over the same τ; 16 fixed samples, EMA student. Has a floor from the reference's Euler error.", "↓"),
    Entry("val/endpoint_b_norm", "4 map vs teacher ODE/‖b(X_1(z))‖ one-step endpoint",
          "Teacher field at the one-step sample from noise: 0 = the map lands on a teacher equilibrium. "
          "16 fixed samples, EMA student.", "→ 0"),
    Entry("train/terminal_teacher_endpoint_norm", "4 map vs teacher ODE/‖b(X_1(z))‖ train batch",
          "Same quantity on the training subset used by the terminal-at-teacher-equilibrium loss (unsquared).", "→ 0"),
    Entry("train/terminal_student_endpoint_norm", "4 map vs teacher ODE/‖v(X_1(z), 0)‖ train batch",
          "Student σ = 0 velocity at the one-step endpoint (0 when that loss is off).", "→ 0"),
    Entry("train/terminal_invariance_endpoint_delta", "4 map vs teacher ODE/‖X_1(x_t) − X_1(x_t + δb)‖",
          "Endpoint change when the start point moves along b (0 when that loss is off).", "→ 0"),

    # 5 samples
    Entry("val/terminal_samples", "5 samples/1 jump  X_1(z)", "One network evaluation from noise, EMA student.", ""),
    Entry(r"val/offdiag_K(?P<K>[0-9]+)_samples", "5 samples/{K} jumps  X_s applied {K}×",
          "K equal jumps of size s = 1 − ε_stop^(1/K), composing to σ = 0.999; EMA student.", ""),
    Entry("val/diag_euler_samples", "5 samples/Euler on v(x, 0), 100 steps",
          "Student used as a field (σ = 0) integrated like Stage 1; no CFG at sampling (guidance is baked in).", ""),
    Entry("val/diag_velocity_norm_first", "5 samples/Euler ‖v‖ at first step", "Mean ‖v(z, 0)‖ at the start.", ""),
    Entry("val/diag_velocity_norm_last", "5 samples/Euler ‖v‖ at last step",
          "Mean ‖v(x, 0)‖ where the Euler sampler stops; → 0 = ended at a fixed point.", "→ 0"),
    Entry("val/diag_latent_norm_last", "5 samples/Euler ‖x‖ at last step", "Should sit at the data-latent norm.", ""),

    # 6 optim
    Entry("train/lr", "6 optim/learning rate", "Warmup, constant, optional cosine decay; lr_override.json can change it.",
          ""),
    Entry("train/grad_norm", "6 optim/‖grad‖ before clip", "Global gradient norm before clipping.", ""),
    Entry("train/grad_skips", "6 optim/skipped steps (since restart)",
          "Steps skipped because the gradient norm was non-finite or above --grad-skip-norm. Resets on restart.",
          "→ 0"),

    # 7 batch
    Entry("train/cfg_drop_frac", "7 batch/null-label fraction", "Fraction of labels replaced by the null class.", ""),
    Entry("train/sigma_mean", "7 batch/mean jump size σ", "Batch mean of the sampled σ (sanity check of the σ law).", ""),
    Entry("train/interp_t_mean", "7 batch/mean base time t", "Batch mean of the base-point t (uniform: 0.5).", ""),
    Entry("train/effective_global_batch", "7 batch/global batch", "batch × GPUs × accumulation (constant).", ""),
    Entry("train/grad_accum_steps", "7 batch/accumulation steps", "Micro-batches per optimizer step (constant).", ""),

    # 8 system
    Entry("train/samples_per_s", "8 system/images per second",
          "Global images per second since the last log (includes validation time).", "↑"),

    # 9 meanflow diagnostics
    Entry(r"train/meanflow_(?P<what>target_abs_max|jvp_abs_max|clock_min|target_norm)",
          "9 meanflow/{what}", "MeanFlow objective diagnostics (only for --objective meanflow/scaled_meanflow).", ""),
]

SUMMARIES_STAGE2 = {"1 loss": "min,last", "4 map vs teacher ODE": "min,last"}

GOTCHAS_STAGE2 = [
    "`1 loss/Lagrangian` and `1 loss/semigroup` come from the last micro-batch, `1 loss/distill` from all micro-batches, "
    "so they do not add up.",
    "Bucket means count empty buckets as 0 inside each micro-batch, so small buckets read low at micro-batch 8.",
    "σ buckets above --sigma-max (e.g. σ∈[0.99, 1] at σ_max = 0.97) are always 0.",
    "Only rank 0's micro-batches are logged; nothing is all-reduced.",
]


# ---------------------------------------------------------------------------
# Resolution and rendering
# ---------------------------------------------------------------------------

def resolve(registry: Sequence[Entry], internal: str, ctx: dict) -> Optional[Tuple[Optional[str], str, str, Entry]]:
    """Return ``(wandb_key, meaning, good, entry)`` for an internal key, or None if unregistered."""
    for e in registry:
        m = re.fullmatch(e.internal, internal)
        if m is None:
            continue
        gd = {k: v for k, v in m.groupdict().items() if v is not None}
        fields = dict(ctx)
        fields.update(gd)
        if "lo" in gd and "hi" in gd:
            fields["b"] = _bucket(e.bucket_var, gd["lo"], gd["hi"])
        else:
            fields["b"] = f"all {e.bucket_var}"
        if "w" in gd:
            fields["W"] = "EMA" if gd["w"] == "ema" else "raw"
        if "s" in gd:
            fields["s_short"] = _fmt_num(gd["s"])
        key = None if e.key is None else _fill(e.key, fields)
        return key, _fill(e.meaning, fields), e.good, e
    return None


def glossary_rows(registry: Sequence[Entry], ctx: dict, *, include_hidden: bool = False) -> List[Tuple[str, str, str, str]]:
    """One row per family: (wandb key, meaning, good, legacy). Bucketed families show a generic bucket."""
    rows = []
    for e in registry:
        if e.key is None and not include_hidden:
            continue
        if e.when is not None and not e.when(ctx) and not include_hidden:
            continue
        fields = dict(ctx)
        bucketed = "(?P<lo>" in e.internal
        optional_bucket = "(?:_t" in e.internal or "(?:_s" in e.internal
        if bucketed:
            fields["b"] = f"{e.bucket_var}∈[lo, hi)" + (f" or all {e.bucket_var}" if optional_bucket else "")
        else:
            fields["b"] = f"all {e.bucket_var}"
        fields.setdefault("W", "raw | EMA")
        fields.setdefault("s_short", "{0.5, 0.9, 0.97, 0.999}")
        fields.setdefault("stat", "mean | min | max")
        fields.setdefault("K", "K")
        fields.setdefault("what", "…")
        meaning = _fill(e.meaning, fields)
        if bucketed:
            meaning += f" Buckets: {_bucket_list(e.bucket_var, e.buckets)}."
        key = "(not logged)" if e.key is None else _fill(e.key, fields)
        rows.append((key, meaning, e.good, _fill(e.legacy, fields) if e.legacy else ""))
    return rows


def _md_escape(s: str) -> str:
    return s.replace("|", "\\|").replace("\n", " ")


def _readable_pattern(regex: str) -> str:
    """``train/distill_raw_s(?P<lo>...)_(?P<hi>...)`` -> ``train/distill_raw_s{lo}_{hi}``."""
    r = regex.replace(B, "{lo}_{hi}").replace(S, "{lo}_{hi}")
    r = re.sub(r"\(\?:_t\{lo\}_\{hi\}\)\?", "[_t{lo}_{hi}]", r)
    r = re.sub(r"\(\?P<(\w+)>[^()]*(?:\([^()]*\)[^()]*)*\)", r"{\1}", r)
    return r.replace("\\.", ".")


def render_markdown(registry: Sequence[Entry], notation: Sequence[Tuple[str, str]], ctx: dict, *,
                    header: str = "", legacy: bool = False, notes: Sequence[str] = (),
                    logged_as_internal: bool = False) -> str:
    out = [header.strip(), ""] if header else []
    out += ["## Notation", "", "| symbol | meaning |", "|---|---|"]
    out += [f"| {_md_escape(_fill(s, ctx))} | {_md_escape(_fill(m, ctx))} |" for s, m in notation]
    out += ["", "## Metrics", "",
            "Sections are numbered so the workspace lists them in reading order. "
            "Good: ↓ lower is better, ↑ higher, → target value, blank = context only.", ""]
    if logged_as_internal:
        out += ["| readable key | definition | good | key logged today |", "|---|---|---|---|"]
        out += [f"| `{_md_escape(k)}` | {_md_escape(m)} | {g} | `{_md_escape(_readable_pattern(e.internal))}` |"
                for (k, m, g, _), e in zip(glossary_rows(registry, ctx, include_hidden=True), registry)]
    elif legacy:
        out += ["| wandb key | definition | good | key in older runs |", "|---|---|---|---|"]
        out += [f"| `{_md_escape(k)}` | {_md_escape(m)} | {g} | {('`' + _md_escape(l) + '`') if l else ''} |"
                for k, m, g, l in glossary_rows(registry, ctx, include_hidden=True)]
    else:
        out += ["| wandb key | definition | good |", "|---|---|---|"]
        out += [f"| `{_md_escape(k)}` | {_md_escape(m)} | {g} |" for k, m, g, _ in glossary_rows(registry, ctx)]
    if notes:
        out += ["", "## Caveats", ""] + [f"- {n}" for n in notes]
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------

def _to_loggable(v):
    """float for scalar tensors / numbers, the object itself for wandb media, None otherwise."""
    if isinstance(v, bool):
        return float(v)
    if isinstance(v, (int, float)):
        return float(v)
    numel = getattr(v, "numel", None)
    if callable(numel) and hasattr(v, "item"):
        return float(v.item()) if numel() == 1 else None
    if type(v).__module__.startswith("wandb"):
        return v
    return None


class ReadableLogger:
    """Rename internal metric keys to self-describing wandb keys and publish a glossary.

    Safe to construct with ``run=None`` (non-main ranks, wandb disabled): ``log`` is then a no-op
    apart from key resolution, and the glossary file is still written when ``run_dir`` is given.
    """

    def __init__(self, run, registry: Sequence[Entry], ctx: dict, *,
                 notation: Sequence[Tuple[str, str]], header: str = "", run_dir: Optional[Path] = None,
                 summaries: Optional[Dict[str, str]] = None, notes: Sequence[str] = (),
                 printer: Callable[[str], None] = print):
        self.run = run
        self.registry = list(registry)
        self.ctx = dict(ctx)
        self.printer = printer
        self._cache: Dict[str, Optional[str]] = {}
        self.unregistered: set = set()
        self.markdown = render_markdown(self.registry, notation, self.ctx, header=header, notes=notes)
        self._rows = glossary_rows(self.registry, self.ctx)
        self._table_logged = False
        if run_dir is not None:
            try:
                Path(run_dir).mkdir(parents=True, exist_ok=True)
                (Path(run_dir) / "wandb_glossary.md").write_text(self.markdown)
            except OSError as e:
                printer(f"[wandb_glossary] could not write glossary file: {e}")
        if run is not None:
            try:
                run.notes = self.markdown
            except Exception as e:  # notes are a convenience; never fail a run over them
                printer(f"[wandb_glossary] could not set run notes: {e!r}")
            for section, how in (summaries or {}).items():
                try:
                    run.define_metric(f"{section}/*", summary=how)
                except Exception as e:
                    printer(f"[wandb_glossary] define_metric({section!r}) failed: {e!r}")

    def key(self, internal: str) -> Optional[str]:
        if internal not in self._cache:
            r = resolve(self.registry, internal, self.ctx)
            if r is None:
                self._cache[internal] = f"9 other/{internal}"
                if internal not in self.unregistered:
                    self.unregistered.add(internal)
                    self.printer(f"[wandb_glossary] no glossary entry for {internal!r}; logging as "
                                 f"'9 other/{internal}'")
            else:
                self._cache[internal] = r[0]
        return self._cache[internal]

    def translate(self, metrics: Dict[str, object], prefix: str = "") -> Dict[str, object]:
        out = {}
        for k, v in metrics.items():
            if k.startswith("_"):
                continue
            val = _to_loggable(v)
            if val is None:
                continue
            key = self.key(prefix + k)
            if key is None:
                continue
            out[key] = val
        return out

    def log(self, metrics: Dict[str, object], step: int, prefix: str = "") -> Dict[str, object]:
        out = self.translate(metrics, prefix)
        if self.run is not None and out:
            if not self._table_logged:
                try:
                    import wandb
                    out["0 init/glossary"] = wandb.Table(columns=["wandb key", "definition", "good"],
                                                         data=[list(r[:3]) for r in self._rows])
                except Exception as e:
                    self.printer(f"[wandb_glossary] glossary table not logged: {e!r}")
                self._table_logged = True
            self.run.log(out, step=step)
        return out


def _cli() -> None:
    import argparse
    p = argparse.ArgumentParser(description="Print the wandb metric glossaries as markdown.")
    p.add_argument("--stage", choices=["1", "2", "all"], default="all")
    a = p.parse_args()
    from types import SimpleNamespace
    parts = ["# wandb metric glossary", "",
             "Generated by `python -m eqfm.wandb_glossary`. Each run also carries its own filled-in copy in the "
             "wandb run notes (Overview tab) and in `<run_dir>/wandb_glossary.md`. The last column gives the key "
             "used by runs logged before the readable names, so older runs can be decoded too.", ""]
    if a.stage in ("1", "all"):
        ctx = stage1_context(SimpleNamespace(time_sampler="uniform_weighted", eps_train=1e-3, data_anchor=0.5,
                                             time_pred_weight=0.0, parameterization="velocity", ema_decay=0.9999,
                                             grad_clip=1.0, cfg_dropout_prob=0.1, warmup_steps=500,
                                             val_num_steps=100, val_eps_stop=1e-3, val_cfg_scale=1.0,
                                             val_n_samples=16, val_loss_batches=4, batch_size=32,
                                             ckpt="<--ckpt>"))
        ctx.update(lam=0.5)
        parts.append(render_markdown(STAGE1, NOTATION_STAGE1, ctx, legacy=True,
                                     header="# Stage 1: equilibrium field (`scripts/train_stage1_eqm.py`)\n\n"
                                            "Example values: uniform_weighted sampler, ε = 1e-3, λ = 0.5.")
                     .replace("## ", "### "))
    if a.stage in ("2", "all"):
        parts.append(render_markdown(
            STAGE2, NOTATION_STAGE2, {}, logged_as_internal=True, notes=GOTCHAS_STAGE2,
            header="# Stage 2: flow-map distillation (`scripts/train_stage2_flowmap.py`)\n\n"
                   "Registered under the keys the Stage-2 trainer logs today; the readable name is what the key "
            "becomes once the trainer logs through `ReadableLogger`.").replace("## ", "### "))
    print("\n".join(parts))


if __name__ == "__main__":
    _cli()
