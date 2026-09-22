# Experiment handoff: ImageNet-256 flow maps from the 2.00-FID field

This document is prescriptive. Every setting is fixed unless a line says "you choose". If something here
cannot be done as written, stop and ask; do not substitute. Numbers to report are defined in section 5.

Contents: 0 rules, 1 setup and sanity checks, 2 fixed settings, 3 the runs in priority order, 4 what to log
and upload, 5 evaluation protocol and reference numbers, 6 compute, 7 abort criteria, 8 glossary.

## 0. Rules

1. **One experiment at a time.** Give every GPU you have to the run with the highest priority and finish it.
   Never run two experiments slowly in parallel. Evaluation jobs (section 5) may share the machine.
2. **Priority order:** run 1, run 2, run 3, then add-ons A, B, C (section 3). Do not start a lower one before
   the higher one has reached its last step, unless it was aborted under section 7.
3. **Never change:** global batch 128, EMA 0.9999, adaptive weight p = 1 with eps 1e-3, per-entry mean loss,
   fp32 with TF32 off, guidance 1.5 with unconditional + guided norm matching, sigma and t uniform on the
   full [0, 1], the sampler schedules of section 5, seed 0. Everything else that is not listed as a choice is
   also fixed.
4. **Report only FIDs tagged `ADM-2k`** as defined in section 5, never a number from another evaluator or
   another sample count. Always say raw or EMA weights.
5. **Launch only through `runs/*.sh`.** They contain the exact flags. If you must pass something extra,
   append it after the script name and write it in the run log.

## 1. Setup and sanity checks (do all of them before run 1)

```bash
git clone https://github.com/Arteam08/robotic-Flow-maps && cd robotic-Flow-maps
python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt
cp .env.example .env            # paste the credentials you received; see docs/SETUP.md
python -m pytest tests -q       # 86 tests, CPU only, ~30 s. All must pass.

# released weights + FID reference (about 3 GB), sha256-verified
python scripts/hf_download.py --dest weights
# public SiT-XL/2 weights (2.7 GB): download, then verify against manifest.json "external"
wget -O weights/SiT-XL-2-256x256.pt https://dl.fbaipublicfiles.com/sit/SiT-XL-2-256x256.pt
sha256sum weights/SiT-XL-2-256x256.pt   # must equal the sha256 printed by: python scripts/hf_download.py --list
# ImageNet-1k train as cached SD-VAE posteriors (294 shards, 21 GB, private repo, needs HF_TOKEN)
hf download EquilibriumMap/eqfm-imagenet-distill --include "latents/*" --local-dir data/imagenet-latents-256-tmp \
  && mv data/imagenet-latents-256-tmp/latents data/imagenet-latents-256
# (older huggingface_hub: replace `hf download` by `huggingface-cli download`)
ls data/imagenet-latents-256 | wc -l    # 294
```

Edit the four paths at the top of `runs/env.sh` (WEIGHTS, LATENTS, RESULTS, FIDOUT) and NGPU. With N GPUs
the per-GPU batch is 128/N; if it does not fit in memory set BS smaller and ACCUM so that BS*NGPU*ACCUM = 128.

**Sanity check 1, evaluator (must pass, about 20 min on one GPU):**
```bash
runs/fid_stage1.sh --ckpt weights/stage1_eqmft_B30k_ema.pt eqmft_B30k
```
Expected `ADM-2k`, legacy stop, raw = EMA (the file holds EMA weights only): **18.1 +- 0.4**. If you get a
number 3 to 4 points higher, you are scoring with the wrong evaluator. If you get something else, stop and ask.
The native-stop number for this linear-clock field is not meaningful; ignore it.

**Sanity check 2, trainer (about 10 min):** 30 steps of run 1 on the real model, then delete the run.
```bash
runs/s2_shifted_meanflow.sh --steps 30 --log-every 5 --val-every 0 --save-every 30 --run-name smoke_s2
rm -rf $RESULTS/smoke_s2
```
The log must print `loss_reduction=mean`, `objective=shifted_meanflow`, and finite losses of order 1 at every
line, with `grad` (pre-clip norm) between roughly 3 and 30. Note the printed samples/s: this is your
throughput; compare with section 6.

**Sanity check 3, field trainer (about 10 min):**
```bash
runs/s1_b_trunc09.sh --steps 30 --log-every 5 --val-every 0 --save-every 30 --run-name smoke_s1
rm -rf $RESULTS/smoke_s1
```
The log must show the latent cache being read (no VAE encode), `clock=trunc:0.9`, and a finite loss.

## 2. Fixed settings

| setting | Stage-2 runs (1, 2, A, B, C) | run 3 (field b) |
|---|---|---|
| network | SiT-XL/2, 32x32x4 latents | SiT-XL/2, time input zeroed and frozen (`--time-conditioning strict`) |
| init | `weights/stage1_eqmft_B30k_ema.pt` (EMA weights), s-embedding starts at zero so the map equals the field | public `SiT-XL-2-256x256.pt` |
| teacher | the same eqmft file, EMA weights, frozen | none (loss on data) |
| data | latent cache, no augmentation | latent cache, no augmentation |
| global batch | 128 | 128 |
| steps | 80 000 | 200 000 |
| optimizer | AdamW, lr 1e-4, betas default, weight decay 0, clip 1.0 | same |
| lr schedule | linear warmup 500 steps, constant to step 48 000, cosine to 0 at 80 000 | warmup 1 000, constant to 120 000, cosine to 0 at 200 000 |
| EMA | 0.9999 | 0.9999 |
| precision | fp32 everywhere, TF32 off (`--no-allow-tf32 --no-bf16 --no-teacher-bf16`) | fp32 (`--no-bf16`) |
| loss reduction | per-entry mean | per-entry mean (built in) |
| adaptive weight | w_i = (L_i + 1e-3)^-1 per sample, on every term | none; time weight 1/(c(t) + 1e-3), normalised |
| diagonal term | field loss against the teacher field b(x_t) on the interpolant, weight 1 | the field loss itself |
| off-diagonal term | objective of the run, weight 1 | none |
| sigma (map time) | uniform on [0, 1], no cap | not applicable |
| t (interpolant time) | uniform on [0, 1], no cap | uniform on [0, 1) |
| clock | linear, c(t) = 1 - t | truncated, c(t) = min(1, (1 - t)/0.1), i.e. a = 0.9, lambda = 1 |
| guidance | teacher field = guided field g at w = 1.5 with unconditional-norm and guided-norm matching; label dropout 0 | label dropout 0.1 (unconditional branch for sampling); no data anchor |
| checkpoints | every 1 000 steps (last 2 kept), permanent copy every 5 000 | same |
| validation | every 2 500 steps, 16 fixed noises: map error vs the teacher ODE at s = 0.5, 0.9, 0.97, 0.999, endpoint norm, sample grids | every 2 500 steps, 16 fixed noises, 250 Euler steps on the native stop |
| seed | 0 | 0 |

## 3. The runs

Names are the run directories and the wandb run names. Do not rename.

### Run 1 `s2_shifted_meanflow` (priority 1)
`runs/s2_shifted_meanflow.sh`. Objective `shifted_meanflow` with one JVP:
the student v(x, s) is regressed on the fully stopped target
`v = sg[ b(x) + s ( v + (D_x v) b(x) - (1 - s) d_s v ) ]`, where b is the guided teacher field at x, and
X_s(x) = x + s v(x, s) is the flow map. Nothing is divided by 1 - s. Semigroup weight 0, eq weight 0,
data anchor 0.

### Run 2 `s2_shifted_lagrangian` (priority 2)
`runs/s2_shifted_lagrangian.sh`. Objective `shifted_lagrangian`:
`v = sg[ b(x + s v) + s ( v - (1 - s) d_s v ) ]`, teacher evaluated at the landing point, one sigma-JVP under
no-grad, no spatial JVP. Same everything else as run 1.

### Run 3 `s1_b_trunc09` (priority 3)
`runs/s1_b_trunc09.sh`. Train an autonomous field b(x) from the public SiT-XL/2 flow-matching weights with the
truncated clock a = 0.9: target c(t)(x_data - z) on x_t = (1 - t) z + t x_data, loss
`E_t ||b(x_t) - c(t)(x_data - z)||^2 / (c(t) + 1e-3)` normalised to mean weight 1, t uniform. The SiT time
input is zeroed and frozen. Label dropout 0.1. No data anchor. 200k steps.

### Add-on A `s2_shifted_meanflow_eq` (priority 4)
Run 1 plus the eq loss: weight 1 times `||b_teacher(X_1(z))||^2` on 4 samples per step (`--terminal-norm-max-batch 4`),
adaptive-weighted like the other terms.

### Add-on B `s2_shifted_meanflow_semi` (priority 5)
Run 1 plus the semigroup term, weight 1: `||X_s(x) - X_{s ominus s'}(X_{s'}(x))||^2` with (s, s') drawn on the
triangle 0 <= s' <= s <= 1, the inner map under stop-gradient. It reuses the same x, labels and teacher call as
the MeanFlow term; only the extra student forwards are new.

### Add-on C `s2_shifted_meanflow_eq_semi` (priority 6)
Run 1 plus both.

Not in scope: teacher-free (self-distillation) variants. Do not run them.

## 4. What to log and upload

- **wandb** (automatic): project `robotic-flow-maps` under the team in `.env`. Training terms, gradient norm,
  lr, samples/s, the validation map errors and sample grids. Every run has its own wandb run; a restart
  continues the same run (`wandb_id` is stored in the run directory).
- **Local**: `$RESULTS/<run>/` holds `config.json`, `log.jsonl`, checkpoints. Keep the permanent checkpoints
  (`kept/` every 5k) until we say otherwise.
- **FID** (section 5): every 5 000 steps, both weights, all samplers. Paste the summary lines in the wandb
  run notes and keep `$FIDOUT/<run>/step<k>/.../adm_fid.json`.
- **Upload to `EquilibriumMap/robotic-flow-maps-results/<run>/`**: the EMA checkpoint at every 20 000 steps
  and raw + EMA at the final step, plus the FID json files. Command, after writing the spec (see
  `scripts/hf_upload.py` docstring):
  `python scripts/hf_upload.py --repo EquilibriumMap/robotic-flow-maps-results --spec spec_<run>.json`
  with `"dest": "<run>/<file>"`. Never upload the 1k-step rolling checkpoints.

## 5. Evaluation protocol and reference numbers

**Definition of `ADM-2k`:** 2 000 samples, labels i mod 1000 (class-balanced), noise seed 0, decoded with the
SD-VAE (`stabilityai/sd-vae-ft-ema`) in fp32, scored with `tools/fid_adm/score_npz.py` against
`weights/ref_inception.npz` (TF-Inception features, full-50k ImageNet-256 reference mu/sigma).
This is the only number you report. Its repeat noise on the same checkpoint is about 0.1; the spread across
independent 2k subsets is about 0.4. Differences below 0.5 are not conclusive.

**Stage-2 checkpoints** (`runs/fid_stage2.sh <run> <step>`, raw and EMA, every 5 000 steps):

| tag | sampler |
|---|---|
| `X1` | one call: X_1(z) |
| `X1x2` | X_1(X_1(z)) |
| `K2_s0.3` | X_1(X_0.3(z)) |
| `K4_s0.2` | X_1 after 3 jumps of s = 0.2 |
| `K8_s0.1` | X_1 after 7 jumps of s = 0.1 |

**Field checkpoints** (`runs/fid_stage1.sh <run> <step>`, raw and EMA, every 5 000 steps): 250 Euler steps,
guidance 1.5 with unconditional + guided norm matching, on two horizons:
`native` = stop at t = 1e-3 measured in the truncated clock's own time (`--eps-stop 0.2565`), the primary
number; `legacy` = the exponential-clock horizon tau = 6.9 (`--eps-stop 1e-3`), which pairs with the
reference fields below. Report both.

**Reference numbers on this evaluator (same seeds, same labels):**

| model | ADM-2k | ADM-10k | ADM-50k |
|---|---|---|---|
| eqmft B30k EMA field (your init and teacher), Euler 250 legacy | 18.1 | 4.55 | 2.00 |
| uw450k field (older teacher) | ~18.3 | 4.68 | 2.18 |
| Lagrangian-distilled student of uw450k, 40k steps, raw, K8 | ~18.6 | 4.94 | 2.47 |
| same student, X1 | ~32 | 19.4 | 16.8 |
| scaled-MeanFlow student of uw450k, 40k steps, raw: X1 / K2 / K4 / K8 | 30.0 / 21.4 / 19.0 / 18.6 | | |
| collaborator's MeanFlow + eq loss, no semigroup, 30k EMA (X1 hacked): X1 / K2 / K4 / K8 | 37.5 / 23.5 / 18.8 / 18.3 | | |

Conversion at this quality level: `ADM-10k ~ ADM-2k - 13.3`, `ADM-50k ~ ADM-10k - 2.5`. Use it only to read
your 2k numbers against the 10k/50k column; never report a converted number as measured.

**What a healthy run looks like.** Validation map error (EMA weights, s = 0.5 / 0.999) on our Lagrangian run:
14.6 / 44 at 2.5k, 7.0 / 18.3 at 17.5k, 5.5 / 18.6 at 40k. `K8_s0.1` ADM-2k should be below 20 by 20k steps
and approach the teacher's 18 by the end. `X1` above 40 after 20k steps, or a map error that rises for
10k steps, means something is wrong: apply section 7.

**Success criteria** (final checkpoint, EMA unless raw is better; state which):
run 1 and run 2: `K8_s0.1` <= 18.6 and `X1` < 30 beat the previous students; `X1` < 25 is a strong result.
Run 3: `native` <= 18.1 at any checkpoint ties the init field; below 17.5 is a gain. Add-ons: report the
delta to run 1 at the same step for every sampler.

## 6. Compute

Measured on one L40S at global batch 128, fp32, TF32 off (fill in from your sanity check 2 and 3):

| run | samples/s per GPU | 4 GPUs, wall time | 8 GPUs |
|---|---|---|---|
| runs 1, A, B, C (shifted MeanFlow) | pending (we fill this in) | | |
| run 2 (shifted Lagrangian) | | | |
| run 3 (field) | | | |

Rule of thumb until measured: Stage-2 about 5 samples/s per L40 -> 10.24M samples in about 6 days on 4 GPUs;
run 3 about 50 samples/s per L40 -> 25.6M samples in about 1.5 days on 4 GPUs.
FID: 15 min per Stage-2 checkpoint (10 sampler x weights combinations) and 20 min per field checkpoint on one GPU.

## 7. Abort criteria

Stop the run, keep the last permanent checkpoint, and report, if any of these happens:
1. a non-finite loss, or `grad` above 50 times its median over the previous 1 000 steps on 3 log lines in a row;
2. the validation map error at s = 0.999 (EMA) rises for 10 000 consecutive steps;
3. `X1` ADM-2k above 60 at 20 000 steps or later;
4. a restart that does not resume from `latest.pt` (the log prints the resumed step; it must not be 0).
For (1) and (2), restart from the last permanent checkpoint at half the learning rate only if we say so.

## 8. Glossary

b: autonomous (time-free) field on latents; the teacher. X_s(x) = x + s v(x, s): the flow map; s in [0, 1]
is the map time, s = 1 lands on the data manifold. Diagonal: the s = 0 field loss. Off-diagonal: the map
loss (MeanFlow / Lagrangian form). JVP: forward-mode derivative of v along a direction. Guided field:
g = b_u + w (b_c - b_u) with norm matching, w = 1.5. eq loss: `||b(X_1(z))||^2`, the field must vanish at
the one-shot endpoint. Semigroup: X_s = X_{s ominus s'} o X_{s'} with s ominus s' = (s - s')/(1 - s').
Native stop: the last integration time expressed in the clock of the field being sampled.
