# EqFM — handoff to a Claude session on another cluster (written 2026-09-15 on Babel/CMU)

You are picking up the **Equilibrium Flow Maps (EqFM)** project from a session that has been running it on the
CMU Babel Slurm cluster. You have none of Babel's filesystem. Everything you need — code, weights, data, the
evaluation protocol, and the accumulated findings — is reachable from this document.

Read in this order:
1. this file, sections 0–6 (setup) and 11 (what to work on);
2. `docs/handoff/PAPER_HANDOFF_2026-09-15_ICLR.md` — **every experiment and every number**, with experiment IDs
   (IN-1..13 ImageNet, MN-1..10 MNIST, TOY-1..2). It is the source of truth for results; this file is the source
   of truth for how to run things. Do not re-derive a number that is in there.
3. `docs/handoff/MNIST_DISTILLATION_ABLATION.md` — the long form of the MNIST element study (MN-4..MN-7).
4. `docs/handoff/SELFCORR_ENDPOINT_TERM.md` — the self-correction endpoint term (2026-09-16), a
   teacher-free replacement for `||b(X_1(z))||`; read it before running anything with `--selfcorr-*`.

Paths written as `~/...` are on Babel and are **not visible to you**; they appear only so a number can be traced
if the user asks the Babel session to re-dump it.

---------------------------------------------------------------------------------------------------------------------
## 0. The first 30 minutes (copy-paste)

```bash
# 1. code
git clone -b martin https://github.com/justinlin26/EqFM && cd EqFM
#    branch `martin` = ImageNet Stage-2 trainer + efficiency work + this handoff.  Other branches: section 3.

# 2. python env  (python 3.11; any torch >= 2.7 with forward-mode AD works; Babel runs 2.13+cu130)
pip install -r requirements.txt          # or: mamba env create -f environment.yml
pip install torch-fidelity              # published-scale evaluator, not in requirements.txt

# 3. weights (private HF repo; the user must add your HF account to the EquilibriumMap org)
hf auth login
hf download EquilibriumMap/eqfm-imagenet-distill --local-dir ./eqfm-share \
    --include 'ckpt/stage1_uw450k_ema.pt' 'ckpt/step_0040000.pt' 'ckpt/stage1_eqmft_B30k_ema.pt' 'README.md'

# 4. data for Stage-2: the pre-encoded ImageNet latent cache (20 GiB, no ImageNet download needed)
hf download EquilibriumMap/eqfm-imagenet-distill --local-dir ./eqfm-share --include 'latents/*'

# 5. FID reference (see tools/fid_adm/README.md)
wget https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/256/VIRTUAL_imagenet256_labeled.npz
python tools/fid_adm/eval_fid.py --ref-npz VIRTUAL_imagenet256_labeled.npz --ref-cache ref_inception.npz
export EQFM_REF_INCEPTION=$PWD/ref_inception.npz

# 6. smoke test: 64 samples from the released student, one-step map, no FID
python scripts/eval_fid_stage2_flowmap.py --ckpt eqfm-share/ckpt/step_0040000.pt --weights raw \
  --sampler terminal --num-samples 64 --batch-size 32 --skip-pytorch-fid \
  --output-dir /tmp/smoke --save-samples-npz /tmp/smoke/samples.npz
```
If step 6 produces recognisable ImageNet images, your stack is correct.

---------------------------------------------------------------------------------------------------------------------
## 1. What the project is, in one page

Two stages, both on 4x32x32 SD-VAE (`stabilityai/sd-vae-ft-ema`) latents of ImageNet-256, SiT-XL/2 backbone.

**Stage 1 — equilibrium field (the "EqM/EqFM field").** A *time-free* velocity `b(x, y)`: data are its fixed points
(`b(x_data) ~ 0`). Sampling = run the autonomous ODE `x <- x + h b(x)` to convergence (250 Euler steps, `h = ln(1000)/250`).
Training (`uw` = the teacher's loss), with `x_t = (1-t) z + t x1`, `t ~ U[0,1]`:

    L1 = E || b(x_t) - (1-t)(x1 - z) ||^2 / max(1-t, 1e-3)  +  0.5 || b(x1) ||^2      (second term = data anchor)

**Stage 2 — flow map (the "student").** `X_s(x, y) = x + s v_s(x, y)`, `s in [0,1]`, with the compactified clock
`tau = -log(1-s)` and the composition law `s (+) s' = s + s' - s s'` (i.e. `1 - s(+)s' = (1-s)(1-s')`).
`X_1` jumps straight onto an equilibrium in **one** network call. Distillation loss ("lag_semi"), `g` = guided teacher field:

    L_diag = || v_0(x) - g(x) ||^2                                            (match the field at s=0)
    L_LESD = || (1-s) d/ds X_s(x) - g(X_s(x)) ||^2                            (Lagrangian, one JVP in s)
    L_PESD = || X_s(x) - sg X_{s(-)s'}( sg X_{s'}(x) ) ||^2                   (semigroup / self-composition)
    L_term = || X_1(x_data) - x_data ||^2 ;   L_eq = || g(X_1(z)) ||^2        (anchor; "eq-norm"/tnorm)
    L = L_diag + L_LESD + 1.0 L_PESD + 1.0 L_term + 1.0 L_eq,  each term adaptively weighted
        L_i -> L_i / (sg L_i + 1e-3)^p  with  p = 0.5   (summed over dims; p=1 with a summed loss diverges, MN-5/MN-6)

**Guidance** lives *inside* the student (it distils the guided field, so sampling costs 1 NFE/step, not 2).
The teacher's guided field at `w = 1.5` with norm matching ("gnorm"):

    b_u' = b_u ||b_c|| / ||b_u|| ;   g0 = b_u' + w (b_c - b_u') ;   g = g0 ||b_c|| / ||g0||

Norm matching is not cosmetic: a constant-CFG autonomous field has **spurious off-manifold equilibria** (TOY-1);
matching the unconditional norm removes them at every `w`.

**Student sampler ("paper schedule", this is part of the method).** K NFE = apply `X_{s0}` K-1 times, then `X_1`.
ImageNet `s0 = 0.3 / 0.2 / 0.1` for K = 2 / 4 / 8; MNIST `0.5 / 0.3 / 0.2`. Equal compactified jumps are 6–14 FID
worse (IN-9) and produce the notorious "K=2 worse than K=1" inversion.

**Where it stands.** ImageNet-256, class-conditional, ADM-50k FID: teacher `uw450k` **2.18** at 500 NFE;
student `s2_lagsemi @40k raw` **2.47** at 8 NFE (16.78 at 1 NFE); a fine-tuned Stage-1 field `eqmft B30k` **2.00**
at 500 NFE. MNIST flow map C4@60k: 5.45 / 3.92 / 3.29 / 3.13 at 1/2/4/8 NFE vs its 100-step teacher 3.28.

---------------------------------------------------------------------------------------------------------------------
## 2. Results you must not re-derive

`docs/handoff/PAPER_HANDOFF_2026-09-15_ICLR.md`. Its section 0 defines every symbol, sampler and FID protocol tag;
section 1 is the headline table; sections 2–4 are the experiments; **section 5 is the list of dead ends** (read it
before proposing anything); section 6 lists pitfalls; section 9 maps claims to evidence.

Quick index of what is already settled:
- objective: Lagrangian + semigroup + eq-norm wins; Eulerian (EESD) diverges; MeanFlow variants are cheaper per
  step but worse, and regular MeanFlow collapses one-shot at 60k on MNIST (MN-8);
- weighting: `p = 0.5` with a summed loss (or per-entry mean with `p = 1`);
- student init: start from the teacher (MN-9);
- sampling: the paper schedule; iterating `X_1` is a no-op (residual 8e-4 after one call) and residual-thresholded
  adaptive NFE loses to a fixed step count;
- guidance: `w = 1.5`; standard / matchu / gnorm / perp_matched all tie at `w = 1.5`; `perp + cnorm` is broken;
- Stage-1 clock: truncated clock beats linear at a fixed t-stop (MNIST 2.88 vs 3.28), and the ranking flips at a
  fixed horizon `tau = 20` — always state the sampler protocol with a clock claim.

---------------------------------------------------------------------------------------------------------------------
## 3. Code: branches and what lives where

Repo `https://github.com/justinlin26/EqFM` (the user's collaborator owns it; `origin/main` is behind).

| branch | contents | use it for |
|---|---|---|
| **`martin`** | ImageNet Stage-2 trainer + launchers, exact-speedup work (PR #1), lr-plateau rule, wandb sections, this handoff | **default: base new ImageNet work here** |
| `mnist-ablation` | all MNIST code (`scripts/train_stage2_flowmap_mnist.py`, `scripts/mnist_ablation/*`, `eqfm/data/mnist.py`), MNIST result tables | MNIST-scale ablations |
| `eqmft-finetune` | Stage-1 fine-tune of released EqM-XL/2 (IN-4) | Stage-1 fine-tuning |
| `trunc-finetune` | truncated-clock Stage-1 fine-tune (IN-5) | clock work on ImageNet |
| `clock-study` | MNIST Stage-1 clock sweep (MN-3) | |
| `meanflow-lagrangian`, `meanflow-scratch`, `meanflow-scratch-recipe` | MeanFlow-family objectives (MN-8) | |
| `init-study`, `offdiag-schedule`, `sampler-study`, `sched4-search` | eval-only studies (MN-9, IN-9, IN-6b, MN-10) | sampler/schedule work |
| `cfg-perp-matched-and-loss-reduction` | `perp_matched` CFG rule + `--loss-reduction` (PR #2) | CFG rules |
| `gnorm-resume` | the 40k→60k resume of the main run (IN-12) | |

Key files: `scripts/train_stage2_flowmap.py` (Stage-2), `scripts/train_stage1_eqm.py` (Stage-1),
`scripts/eval_fid_stage2_flowmap.py` (student sampling+FID), `eqfm/stage2_losses.py` (all Stage-2 terms),
`eqfm/cfg_rules.py`, `eqfm/flowmap.py`, `eqfm/sampling.py`, `eqfm/data/{imagenet_parquet,imagenet_latents,mnist}.py`,
`eqfm/stage2_wandb.py`, `eqfm/lr_plateau.py`, `tools/fid_adm/` (published-scale evaluator, added for you).

The Babel `*.slurm` launchers at the repo root are readable as **recipes** (they carry the exact flag sets that
produced the numbers) but are Babel-specific (`/data/user_data/mdrieux`, `preempt` QOS, `/home/mdrieux/envs/glass`).
Copy the python command out of them, not the `#SBATCH` header.

---------------------------------------------------------------------------------------------------------------------
## 4. Weights: the private HF repo

`EquilibriumMap/eqfm-imagenet-distill` (private; ImageNet-derived — do not redistribute). Its `README.md` documents
every file, including load snippets and the exact sampler; read it after downloading.

| file | size | what |
|---|---|---|
| `ckpt/stage1_uw450k_ema.pt` | 2.7 GB | **Stage-1 teacher** of every student here. EMA weights. ADM-50k **2.18** |
| `ckpt/stage1_eqmft_B30k_ema.pt` | 2.7 GB | Stage-1 field from fine-tuning released EqM-XL/2 (IN-4). ADM-50k **2.00**. *Not yet distilled* |
| `ckpt/stage1_eqmft_B30k_trainer.pt` | 10.8 GB | same step, full trainer state (raw + ema + AdamW) |
| `ckpt/step_0040000.pt` | 10.8 GB | **main student**, end of the original run. raw K8 ADM-50k **2.47** |
| `ckpt/step_0045000.pt` | 10.8 GB | resumed run (`resume40k`), mid-anneal |
| `ckpt/step_0034800.pt` | 10.8 GB | earlier student checkpoint (first share) |
| `latents/shard_XXXXX.npz` | 20 GiB total, 294 files | ImageNet-1k train as SD-VAE posteriors (see section 5) |

Checkpoint layout: `model` (fp32 raw), `ema`, `optimizer` (AdamW), `step`, `args` (the full training config —
`torch.load(p, map_location="cpu", weights_only=False)["args"]` answers most "what flags did that run use?" questions).
The two Stage-1 `*_ema.pt` files are **EMA-only**: their `model` tensors already *are* the EMA weights and they have
no `ema` key. The loaders print `WARN checkpoint has no EMA state; using raw weights` and that is correct behaviour —
pass `--teacher-weights ema` anyway, it is a no-op for those files.

Not on HF: the MNIST checkpoints (124 MB each, on Babel), the full 5.4 GB uw450k trainer state, the released
EqM-XL/2 and DiT-XL/2 baselines (public: `EquilibriumMatching/EqM-XL-2` / `facebook/DiT-XL-2-256`), and the ADM
reference batch (public URL in section 0). Ask the user if you need one of those uploaded.

---------------------------------------------------------------------------------------------------------------------
## 5. Data

**Preferred: the latent cache.** `latents/shard_XXXXX.npz`, one per official ILSVRC train parquet shard, parquet row
order, 1,281,167 rows total. Keys: `mean`, `std` float16 `(N,4,32,32)` = **unscaled** SD-VAE posterior; `label` int16.
Per step the trainer draws `z = (mean + std * eps) * 0.18215`. Preprocessing already applied: resize 256, center crop,
`[-1,1]`. Pass it as

    --shards "latents:/path/to/eqfm-share/latents"

This is what you want: no ImageNet download, no VAE encode in the loop, and it is bit-identical to what produced the
numbers above. Caveat: the cache has **no horizontal flips** (Stage 2 does not use them; the Stage-1 fine-tunes did).

**Raw ImageNet** (only needed for Stage-1 training or for hflip augmentation): `--shards "parquet:/path/train-*.parquet"`
reads HF parquet shards (`image: struct<bytes,path>`, `label: int64`, sorted-WNID order). `ILSVRC/imagenet-1k` is gated
(manual approval, the user has access); ungated 256px mirrors that work with the same loader: `evanarlian/imagenet_1k_resized_256`,
`benjamin-paine/imagenet-1k-256x256`. To build your own latent cache: `scripts/precompute_latents.py` (see
`precompute_latents.slurm` for the flag set) — it is ~2 GPU-h for the full train set.

**MNIST** downloads itself (`eqfm/data/mnist.py`, `--data-root <dir>`); the whole MNIST loop runs on one GPU.

---------------------------------------------------------------------------------------------------------------------
## 6. Environment and hardware

- python 3.11, torch >= 2.7 (Babel: 2.13+cu130). The Stage-2 loss needs **forward-mode AD** (`torch.func.jvp`)
  through SiT; it works on every torch >= 2.4 we tried. `requirements.txt` pins cu126 wheels — adjust for your driver.
- The student trains in **fp32** (the JVP is fp32); the teacher runs in bf16. bf16 for the student's non-JVP passes
  is available on `train-efficiency` (gradient cosine 0.9999 vs fp32) and gives ~1.2x.
- Throughput, samples/s per GPU, production code, real data: RTX PRO 6000 7.9–8.7, 6000 Ada ~4.9, L40S 4.35,
  A6000 2.45–3.0; synthetic single-GPU A100-80GB 7.06 (8.93 with the efficiency branch).
  **40k steps x global batch 128 = 5.12M samples ~ 330 L40S GPU-h ~ 240–270 A100-80GB GPU-h** (~$2k / 2.5 days on an
  8xA100 cloud VM at list price).
- Memory: per-GPU batch 8 fits 48 GB with `PYTORCH_ALLOC_CONF=expandable_segments:True`. Keep the **global batch at
  128** = per_gpu x grad_accum x world_size (8 GPUs: `--batch-size 8 --grad-accum-steps 2`; 4 GPUs: accum 4).
- Multi-GPU: on shared L40S/L40 nodes DDP init hung until `NCCL_P2P_DISABLE=1` was set; harmless elsewhere. Keep a
  finite `init_process_group` timeout so a hang cannot hold GPUs for hours.
- Preemptible queues: every launcher is idempotent and resumable (`--resume` from `<run>/checkpoints/latest.pt`).
  Reproduce that habit on your cluster; the trainer offsets the data seed by the resume step so a requeue does not
  replay the same batches.

---------------------------------------------------------------------------------------------------------------------
## 7. Running Stage-2 distillation (the main result, IN-8)

Exact command that produced the 2.47 student (taken from `stage2_lagsemi_gnorm_imagenet_babel.slurm`, paths made
generic; `NGPU` x `--batch-size` x `--grad-accum-steps` must equal 128):

```bash
torchrun --standalone --nproc_per_node=$NGPU scripts/train_stage2_flowmap.py \
  --results-dir $RESULTS --run-name $RUN --resume \
  --shards "latents:$SHARE/latents" \
  --teacher-ckpt $SHARE/ckpt/stage1_uw450k_ema.pt --teacher-weights ema \
  --init-stage1-ckpt $SHARE/ckpt/stage1_uw450k_ema.pt --init-stage1-weights ema \
  --model SiT-XL/2 \
  --objective lag_semi --semi-weight 1.0 \
  --interp-time-sampler uniform --sigma-sampler uniform --sigma-min 1e-4 --sigma-max 0.97 \
  --adaptive-weight-p 0.5 --adaptive-weight-eps 1e-3 \
  --diag-weight 1.0 --distill-weight 1.0 --terminal-weight 1.0 \
  --terminal-teacher-norm-weight 1.0 --terminal-norm-max-batch 4 \
  --guidance-mode guided --teacher-cfg-scale 1.5 --cfg-match-uncond-norm --cfg-match-guided-norm \
  --cfg-mode all --cfg-dropout-prob 0.0 \
  --batch-size 8 --grad-accum-steps 2 --num-workers 3 \
  --steps 40000 --lr 1e-4 --warmup-steps 500 --lr-decay-from 25000 --lr-final 0 \
  --ema-decay 0.9999 --grad-clip 1.0 \
  --save-every 1000 --keep-last-checkpoints 3 \
  --val-every 2500 --val-n-samples 16 --val-diag-num-steps 100 --val-offdiag-steps 1 2 4 8 \
  --wandb-mode online
```

Things that matter, learned the hard way:
- **`--guidance-mode guided` + `--cfg-dropout-prob 0`** is the faithful CFG-distillation setting: the student learns
  the *guided* field, so it needs no null class at sampling time. (`branch` mode + dropout 0.1 distils `b(.,y)` and
  `b(.,null)` separately; most MNIST arms accidentally used it — see the caveat in section 3 of the paper handoff.)
- The **lr schedule is the single biggest lever on one-shot quality**: training loss is flat from ~4k steps, and
  ADM-10k K8 went 6.29 @27.3k -> 4.94 @40k almost entirely during the cosine decay. Budget the decay, do not extend
  the constant-lr phase.
- Raw weights beat EMA at the end of a decay (2.47 vs 2.59); evaluate both.
- `--terminal-norm-max-batch 4` bounds the cost of the eq-norm term (it needs a teacher call on `X_1(z)`).
- Live control without killing the job: write `<run_dir>/lr_override.json` (the trainer polls every 50 steps) to
  change `steps` / `decay_from` / disable the lr-plateau rule; `eqfm/lr_plateau.py` implements "halve the lr when the
  unweighted loss stalls" (geometric-mean window statistic, `"lr_plateau": false` turns it off).
- wandb sections (`eqfm/stage2_wandb.py`): `0_fid`, `1_loss_weighted`, `2_error_unweighted`, `3_diag_by_t`,
  `4_lagrangian_by_sigma`, `5_lagrangian_by_t`, `6_semigroup_by_sigma`, `7_teacher_field_norm`, `8_optim_system`;
  a glossary is written into the run notes. Watch `2_error_unweighted` (weighted losses are ~1 by construction) and
  the diagonal term — a rising diag drift was the early warning on the first run.
- Diagnostics that correlate with final FID: validation map error at `s = .5/.9/.97/.999` (final run: 5.54/15.71/17.69/18.65)
  and the endpoint norm `E||g(X_1(z))||` (0.074 at 40k).

## 8. Stage-1: fine-tuning a released EqM into an EqFM field (IN-4, cheap and reproducible)

30k steps (23.7 h on 2 x A6000) converts `EqM-XL/2` into a field scoring **2.00** at ADM-50k:
scale the output head by 1/9 (EqM regresses `4 min(1, 5(1-t))(x1 - z)`, we regress `(1-t)(x1-z)` with weight `1/(1-t)`),
1k head-only steps (phase A), then 30k steps of all weights **except the t-embedder**, AdamW lr 3e-5 constant,
EMA 0.9999, global batch 256, hflip, label dropout 0.1, no data anchor. Branch `eqmft-finetune`; the step-0 rescaled
model already scores 4.62 ADM-10k vs our 450k-step teacher's 4.67, i.e. **the rescale alone gives a usable teacher**.

## 9. The MNIST loop (use it for anything exploratory)

Branch `mnist-ablation`. SiT-S/4 (32.5M params) on MNIST padded to 32x32, fp32 (bf16 diverges ~12k steps), one GPU.
Costs: Stage-0 flow-matching teacher 50k steps ~1 h; Stage-1 field 25k steps ~0.5 h; the best Stage-2 recipe
(C4) 329 ms/step -> 60k steps = 5.5 GPU-h; a 10k-step screen is ~0.9 GPU-h. FID protocol: pytorch-fid vs 60k MNIST
train stats, 10k class-balanced samples ("MN-10k"); floor (test vs train) 1.03.

The best recipe (`s2p_C4_anneal60k`, FID 5.45 / 3.92 / 3.29 / 3.13 at 1/2/4/8 NFE) is
`scripts/train_stage2_flowmap_mnist.py` with: `--objective lagrangian --semi-weight 1 --terminal-weight 1
--terminal-teacher-norm-weight 1 --adaptive-weight-p 0.5 --loss-reduction sum --sigma-max 0.97 --batch-size 128
--lr 1e-4 --steps 60000 --lr-decay-from 20000 --ema-decay 0.999 --guidance-mode branch --cfg-dropout-prob 0.1`
(the last two are the branch-mode caveat; a fresh study should use `--guidance-mode guided --cfg-dropout-prob 0`),
teacher `stage1_eqm_mnist_sit_s4_uw/checkpoints/ema_step_0025000.pt`.
Sweeps are driven by `scripts/mnist_ablation/launch.sh <manifest>` with `DRY=1` to print the sbatch commands —
read the manifests (`manifest_stage2*.txt`) to see how an arm is specified as a delta from the star config.
MNIST checkpoints are not on HF; ask the user, or retrain (1.5 GPU-h for both teachers).

## 10. Evaluation: the protocol is part of the result

Full definitions in section 0.5 of the paper handoff. The short version:

| tag | evaluator | N | note |
|---|---|---|---|
| ADM-50k / ADM-10k / ADM-2k | torch-fidelity TF-Inception vs ADM `VIRTUAL_imagenet256_labeled` **full mu/sigma** | 50k / 10k / 2k | **paper numbers**; `tools/fid_adm/` |
| PT-10k | `eqfm.metrics` (pytorch-fid) | 10k | internal only; reads **+3.5 to +3.9 higher** |
| MN-10k | pytorch-fid vs 60k MNIST train | 10k | all MNIST numbers |

Rules: never compare across evaluators; never compare across N (at this quality `FID(50k) ~ FID(10k) - 2.5`);
always use class-balanced labels `i % 1000` (random labels cost ~2.8 FID at N=2048); always state the sampler
(`K`, `s0`, raw vs EMA). ADM-10k bootstrap sd is 0.08–0.33, ADM-50k 0.02–0.12, so a 0.2 difference at 10k is noise —
use a **paired** ADM-2k screen (same seeds, sd 0.06) for ranking, and confirm the winner at 10k/50k.

Generation + scoring commands are in `tools/fid_adm/README.md`. Sampler flags:
`--sampler terminal` (X1, 1 NFE), `--sampler offdiag --offdiag-steps K --offdiag-schedule paper --offdiag-s0 <s0>`
(K NFE), `--sampler diag_euler --diag-num-steps 250` (the field, teacher-style).

## 11. What to work on (the user's current priorities)

**P0 — distil the FID-2.00 field.** Every student here distils `uw450k` (2.18). The better Stage-1 field
`stage1_eqmft_B30k_ema.pt` (2.00, IN-4) **has never been distilled**, and the user asked for exactly this next.
It is a drop-in replacement in the section-7 command:

```bash
  --teacher-ckpt $SHARE/ckpt/stage1_eqmft_B30k_ema.pt --teacher-weights ema \
  --init-stage1-ckpt $SHARE/ckpt/stage1_eqmft_B30k_ema.pt --init-stage1-weights ema
```
Keep everything else identical (lag_semi, p 0.5, cfg 1.5 gnorm, guided mode, dropout 0, global batch 128,
lr 1e-4 warmup 500, cosine 25k -> 40k) so the comparison against the 2.47 student is clean. Notes: this field was
fine-tuned with its t-embedder frozen — keep feeding `t = 0` and never zero the embedder; it was trained with label
dropout 0.1, so the null class is healthy and `guided` mode works. Expected signal: the teacher is 0.18 FID better,
so a student that lands below ~2.4 at K8/ADM-50k reproduces the transfer; also report X1, K2, K4 and the same
checkpoints (30k, 37k, 40k) as IN-8 so the two runs are directly comparable.

**P1 — the numbers a reviewer will ask for** (all evaluation, no training): ADM-50k of the 40k student at
**K1 / K2 / K4** (only X1 and K8 exist); ADM-50k of the teacher **without guidance**; ADM-50k of **released EqM-XL/2**
on our evaluator (only a DiT-anchored estimate exists); per-run **GPU-hours**. These are ~16 L40S GPU-h per 50k setting.

**P2 — scale Stage-2.** The main run stopped at 40k steps because the lr schedule was compressed twice on a
plateau, not because it converged: the resume past 40k (IN-12) and the MeanFlow continuation (IN-11) were both still
improving when this handoff was written. On better GPUs the honest experiment is a from-scratch 80k-step run with a
single cosine decay to 0 at 80k, global batch 128, everything else as in section 7 — that is the "what would this be
with proper compute" number the paper needs. Larger global batch is untested; if you change it, re-tune lr.

**P3 — MNIST-scale ablations.** Cheap and high-yield; see section 9 and MN-4..MN-10 for what has been covered.
Open threads there: the 4-NFE schedule search (re-noising schedules looked ~0.7 better at N=2k, unconfirmed at 10k),
and a faithful-mode (guided, dropout 0) redo of the element ablation.

Before starting anything: check section 5 (dead ends) of the paper handoff, and section 7 of it (what was still
running here — those rows may already be answered by the time you read this; ask the user).

## 12. Pitfalls that cost this project time

1. **Evaluator mixing.** A peer session switched the ImageNet run from `gnorm` to `matchu` on a comparison that was
   really a +3.5 evaluator offset. Tag every number with its protocol.
2. **`p = 1` with a summed loss** silently starves the `s -> 1` end (gradient scales like `(1-s)^(2p-1)`); the
   semigroup term then diverges and K-step branches die in order of `1/(1-s)`. This killed the first ImageNet
   Stage-2 attempt on the collaborator's cluster (IN-10).
3. **Equal-jump sampling** makes a good map look broken (K2 worse than K1). Always use the paper schedule.
4. **EMA lag.** After an EMA reset or during a fast decay, EMA is *worse* than raw; report both.
5. **The `args` in a checkpoint are the ground truth** for what a run actually did — a resubmitted job that silently
   dropped an env var once reverted a 50k schedule to 80k and nobody noticed for hours. Verify the schedule printed
   in the run's first log line.
6. **`(1-t)` weighting needs an eps** (1e-3) and the eps interacts with `p`: with a summed loss over 4096 dims every
   sample sits in the normalising branch, which is why 100% of steps clip at grad-clip 1.0. That is expected, not a bug.
7. **Latents are unscaled posteriors.** Multiply by 0.18215 *after* sampling `mean + std * eps`, once.
8. **Class-balanced labels** (`i % 1000`) everywhere, seed 0 for the 10k runs, `10000*i` for 50k shards (shard 0 is
   then paired with the 10k run).

## 13. How the user works (match this)

- **Answer with the formula and the number, not prose.** The user is a CMU PhD-level researcher on this project;
  equations, notation and tables land better than explanation. Keep summaries short.
- **Fair baselines.** After exploratory work, the user expects the winner to be proved against a same-endpoint /
  same-NFE / same-GPU-hour simple baseline plus an ablation ladder, not just against the previous best.
- **Say what is running, what it costs, and what you will conclude from it** before burning GPU-days.
- Long jobs should be self-resubmitting, resumable and idempotent; the user does not want to babysit a queue.
- The user reads wandb: entity `mdrieux-carnegie-mellon-university`, projects `eqfm` (ImageNet) and `eqfm-mnist`.
  Use the same entity/projects if you have the key, and put the run name in the FID json paths so the two agree.
- Report FIDs as `<protocol tag> <N> <sampler> <weights>`: e.g. "ADM-10k, K8 paper s0=0.1, raw: 4.94".

## 14. What is running on Babel right now (2026-09-15), so you do not duplicate it

| run | what | state |
|---|---|---|
| `s2_lagsemi_cfg15_gnorm_babel_resume40k` | the main student resumed 40k -> 60k, lr re-warm 3e-5 cosine | running; 45k evaluated (ADM-10k EMA X1 18.27, K2 7.69, K4 5.45, K8 4.98 — already better than 40k) |
| `s2mf_p1_semi_anchor_endpoint_cfg15_gnorm_from_lag27k` | regular MeanFlow continuation from the lag map (IN-11) | running; +10k raw ADM-10k X1 16.51 (best one-shot so far), K2 8.16 |
| `eqmft_T_trunc08` | truncated-clock Stage-1 fine-tune (IN-5) | running; T-5k EMA already ties B-30k at ADM-50k (1.99 vs 2.00) |
| MNIST MeanFlow-from-scratch / sched4 searches | MN-8/MN-10 follow-ups | running |

The Babel session also holds the MNIST checkpoints, the full FID artefacts, and can re-dump any number on request.
Coordinate through the user; several Claude sessions share this repo and the same Slurm account, so if a branch or a
queue changes under you, that is a peer session, not corruption.

## 15. Glossary

`uw` uniform-weighted Stage-1 loss · `uw450k` the 450k-step teacher (2.18) · `eqmft` EqM-XL/2 fine-tuned to the EqFM
objective (2.00) · `lag_semi` the Stage-2 recipe (LESD + PESD + terminal + eq-norm) · `LESD/EESD` Lagrangian/Eulerian
self-distillation · `PESD` semigroup (path) term · `tnorm`/eq-norm `||g(X_1(z))||^2` · `matchu` match the
unconditional norm · `gnorm` matchu + match the guided norm · `perp_matched` / `cnorm` alternative CFG rules (cnorm is
broken) · `X1` one-shot terminal map · `K8 paper s0=0.1` seven jumps of `s_0` then `X_1` · `ADM-*` the published-scale
evaluator · `PT-10k` the internal pytorch-fid one · `MN-10k` the MNIST one · `map_err` distance to the teacher-ODE
endpoint at fixed interpolant points · `endpoint` `E||b(X_1(z))||`.
