# Best ImageNet FID Sampler (EqM Stage 1)

Recommended sampler for the lowest FID from a Stage-1 EqM checkpoint, following
Equilibrium Matching (Wang & Du, arXiv:2510.02300) and their reference
`raywang4/EqM` code: **250-step NAG-GD with `mu = 0.3` and CFG.**

## TL;DR command

```bash
python scripts/eval_fid.py \
  --ckpt /path/to/stage1_checkpoint.pt \
  --output-dir /scratch/$USER/eqfm-fid/naggd_cfg15 \
  --num-samples 50000 \
  --num-steps 250 \
  --momentum 0.3 --nesterov \
  --cfg-scale 1.5 \
  --cfg-match-uncond-norm \
  --eps-stop 1e-3 \
  --bf16
```

This integrates the autonomous field for 250 steps using the paper's NAG-GD
update (`--momentum 0.3 --nesterov`), with classifier-free guidance at scale
1.5. `--num-samples 50000` is report-grade; use `10000` for fast iteration.

## What each knob is and why

| Flag | Value | Why |
|---|---|---|
| `--num-steps` | `250` | Matches the paper's sampling budget. |
| `--momentum` + `--nesterov` | `0.3` + on | The paper's NAG-GD. `mu` is dimensionless, so 0.3 transfers directly. NAG-GD ≥ GD, gap largest at low step counts. |
| `--cfg-scale` | `1.5` | Paper's best CFG for EqM-XL/2. **Sweep this** (see below) — the optimum is checkpoint-specific. |
| `--cfg-match-uncond-norm` | on | Rescales the **unconditional** field to the conditional norm *before* CFG. Fixes the *direction* of the residual. |
| `--cfg-match-guided-norm` | on (see below) | Rescales the **final guided vector** to the conditional norm *after* CFG. Caps guidance *speed* near equilibrium. Best combined with `--cfg-match-uncond-norm`. |
| `--eps-stop` | `1e-3` | Sets the autonomous-time horizon `S_stop = log(1/eps_stop) ≈ 6.9`, i.e. step size `≈ 6.9/250 ≈ 0.028`. This is the repo's calibrated horizon — **do not** copy the paper's `eta = 0.0017` (that is specific to their field normalization, not this repo's flow-residual field). |
| `--bf16` | on | bf16 model forward; the integration variable stays fp32 internally. |

### Why NAG-GD here (and how it is implemented)

`eqfm/sampling.py` was corrected to match `raywang4/EqM/sample_gd.py`. With
`--nesterov` the per-step update is:

```text
x_lookahead = x + step_size * mu * m      # m = previous field value
b           = field(x_lookahead, y)        # (CFG applied here)
m           = b
x           = x + step_size * b            # plain Euler step, no accumulated velocity
```

The position step is identical to GD; `mu` only shifts *where* the field is
sampled. This is **not** heavy-ball — it never accumulates velocity, so it
cannot inflate the effective step or overshoot the fixed point. (Heavy-ball,
`--momentum 0.3` *without* `--nesterov`, accumulates velocity and underperforms
here — avoid it.)

## Confirm NAG-GD beats GD, then pick CFG

Run a small sweep at `--num-samples 10000` first. The NAG-GD > GD gap should be
clearest if you also try a tighter step budget.

```bash
for SAMPLER in "GD: --momentum 0.0" "NAGGD: --momentum 0.3 --nesterov"; do
  NAME=${SAMPLER%%:*}; FLAGS=${SAMPLER#*: }
  for CFG in 1.0 1.5 2.0; do
    python scripts/eval_fid.py \
      --ckpt /path/to/stage1_checkpoint.pt \
      --output-dir /scratch/$USER/eqfm-fid/${NAME}_cfg${CFG} \
      --num-samples 10000 --num-steps 250 \
      $FLAGS --cfg-scale $CFG --cfg-match-uncond-norm --bf16
  done
done
```

Compare `fid.json` across runs. Then rerun the winning `(sampler, cfg)` at
`--num-samples 50000` for the report-grade number.

## Norm-matching the guidance residual

The EqM field vanishes at the fixed point, so CFG has two distinct failure
modes near the manifold: wrong residual *direction* (cond/null magnitude
mismatch) and excess residual *speed* (pushing off the manifold). There are two
independent rescales, and they fix different things. On the toy circle
(constant CFG, conditional baseline `conc 0.939`):

| Variant | flags | best `conc` | off-circle |
|---|---|---|---|
| unconditional-matched | `--cfg-match-uncond-norm` | 0.951 | 0.000 |
| guided-vector-matched | `--cfg-match-guided-norm` | 0.948 | 0.047 |
| **both** | both flags | **0.963** | 0.009 |

Matching the **unconditional** beat matching the **entire guided vector**: the
guided-vector rescale caps speed but leaves a radial component, so off-circle
fraction climbs with scale (0.03 -> 0.20 from s=1.5 -> 6). But the two are
complementary -- **using both was the strongest variant overall** (the toy's
`match_norm_guided_norm_match`), because null-norm matching fixes direction and
final-vector matching caps speed. Recommended best-EqM command:

```bash
python scripts/eval_fid.py \
  --ckpt /path/to/stage1_checkpoint.pt \
  --output-dir /scratch/$USER/eqfm-fid/naggd_early_bothnorm_cfg2 \
  --num-samples 50000 --num-steps 250 \
  --momentum 0.3 --nesterov \
  --cfg-scale 2.0 --cfg-schedule early --cfg-early-frac 0.5 \
  --cfg-match-uncond-norm --cfg-match-guided-norm \
  --eps-stop 1e-3 --bf16
```

Whether the combined rescale that won on the toy also wins on ImageNet FID is
an experiment to run -- compare it against `--cfg-match-uncond-norm` alone at
the same `(sampler, schedule, cfg)`.

## Early-stopping CFG (best EqM schedule on the toy)

On the toy circle, the strongest *EqM* guidance was not constant CFG but
**early-stopping CFG**: apply full guidance only for the first part of the
trajectory (basin selection), then turn it off so the unguided equilibrium
field lands cleanly on the manifold. Constant CFG keeps pushing a guidance
residual into the fixed-point regime, where it is mostly branch-estimation
error; early stopping avoids that. Use `--cfg-schedule early` with
`--cfg-early-frac` (the fraction of the autonomous-time horizon to guide;
`0.5` = first half):

```bash
python scripts/eval_fid.py \
  --ckpt /path/to/stage1_checkpoint.pt \
  --output-dir /scratch/$USER/eqfm-fid/naggd_early0p5_cfg15 \
  --num-samples 50000 --num-steps 250 \
  --momentum 0.3 --nesterov \
  --cfg-scale 1.5 --cfg-schedule early --cfg-early-frac 0.5 \
  --cfg-match-uncond-norm --eps-stop 1e-3 --bf16
```

Worth sweeping the cutoff and a slightly higher scale (early stopping tolerates
more guidance because it never reaches the fixed point):

```bash
for FRAC in 0.25 0.5; do
  for CFG in 1.5 2.0 3.0; do
    python scripts/eval_fid.py \
      --ckpt /path/to/stage1_checkpoint.pt \
      --output-dir /scratch/$USER/eqfm-fid/early${FRAC}_cfg${CFG} \
      --num-samples 10000 --num-steps 250 \
      --momentum 0.3 --nesterov \
      --cfg-scale $CFG --cfg-schedule early --cfg-early-frac $FRAC \
      --cfg-match-uncond-norm --bf16
  done
done
```

Compare `early` against `constant` and `anneal_to_one` at the same `(sampler,
cfg)`. On the toy, `early` (frac 0.5, scale 3) gave the best on-manifold class
concentration; whether it wins on ImageNet FID is exactly the experiment to
run. Note the cutoff is in *autonomous time* (`s_mid <= frac * S_stop`), so it
is independent of `--num-steps`.

## Report-grade FID (parity with published SiT/DiT)

The self-computed reference is fine for tracking your own checkpoints, but is
**not** on the same scale as published numbers. For parity, evaluate against
the ADM `VIRTUAL_imagenet256_labeled.npz` reference:

```bash
python scripts/eval_fid.py \
  --ckpt /path/to/stage1_checkpoint.pt \
  --output-dir /scratch/$USER/eqfm-fid/report \
  --num-samples 50000 --num-steps 250 \
  --momentum 0.3 --nesterov --cfg-scale 1.5 --cfg-match-uncond-norm --bf16 \
  --ref-stats /path/to/VIRTUAL_imagenet256_labeled.npz
```

## Caveats (read before trusting guided FID)

- **CFG assumes a trained null branch.** `--cfg-scale > 1` is only principled
  if the checkpoint was Stage-1 trained with label dropout (`--cfg-dropout-prob
  > 0`). If it was not, `b_uncond` is SiT's *pretrained* unconditional velocity,
  not an autonomous-field uncond, and guided FID is a relative diagnostic only.
  Verify with the CFG sweep above before treating it as report-grade.
- **`eta` does not transfer from the paper.** Use `--num-steps` / `--eps-stop`,
  not a hand-set step size. `mu = 0.3` does transfer.
- **FID is upward-biased at small N.** Only compare checkpoints at a *fixed* N;
  only `--num-samples 50000` is report-grade.

## References

- Paper: https://arxiv.org/abs/2510.02300
- Reference code: https://github.com/raywang4/EqM (`sample_gd.py`)
