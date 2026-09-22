# HANDOFF — read this first, then follow it top to bottom

You (the agent) are running a fixed list of ImageNet experiments for a collaboration. Everything is decided:
what to run, in which order, with which flags, how to evaluate, what to send back. Your job is to execute
exactly, verify each step against the stated expectation, and report. Where this file says "expected", a
different outcome means stop and ask the human, not improvise.

Files that matter, in reading order:
1. `HANDOFF.md` (this) — the sequence.
2. `docs/SETUP.md` — environment, one credential, weights, data.
3. `docs/EXPERIMENTS.md` — the runs, fixed settings, evaluation protocol, reference numbers, abort rules.
4. `runs/` — the launch scripts. Only launch through them.

Do not modify anything under `eqfm/`, `scripts/`, `tools/` or `runs/` except the four path variables and NGPU
at the top of `runs/env.sh`. If a script fails, fix the environment, not the code; if it still fails, stop and ask.

## Phase 0 — environment (30 min)
1. `python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt`
   (CUDA 12.6 torch wheels; if the driver is older than 560 or you need another CUDA, edit only the
   `--extra-index-url` line and the torch bound in `requirements.txt`, and say so in the run log).
2. `cp .env.example .env` and paste the `WANDB_API_KEY` you were given. Never run `wandb login` or any HF login.
3. `python -m pytest tests -q` — expected: 86 passed. Anything else: stop.
4. `nvidia-smi` — note GPU model, count and memory in `RUNLOG.md` (create it; append every action and result).

## Phase 1 — weights and data (1 h download + 2 to 4 GPU-hours, unattended)
Follow `docs/SETUP.md` section 3 exactly. Expected at the end:
- `weights/stage1_eqmft_B30k_ema.pt`, `weights/ref_inception.npz` (sha256 verified by the script),
  `weights/SiT-XL-2-256x256.pt` with sha256 `e645b5cd82d1d645cfd9b7c96041b56b2a0aa61faed90c0869e79be02ab55aca`.
- `data/imagenet-latents-256/shard_*.npz`: 52 files, 1 281 167 rows, labels 0..999.
Set `WEIGHTS`, `LATENTS`, `RESULTS`, `FIDOUT`, `NGPU` in `runs/env.sh`.

## Phase 2 — sanity checks (1 h, one GPU)
Run the three checks in `docs/EXPERIMENTS.md` section 1, in order. Record the three results in `RUNLOG.md`:
- check 1: the FID-2k of the released field on the legacy horizon, expected 18.1 +- 0.4;
- check 2: 30 steps of run 1, finite losses, and the printed samples/s;
- check 3: 30 steps of the field trainer reading the latent cache.
Do not start Phase 3 with a failed check.

## Phase 3 — the runs, one at a time, this order
| # | script | steps | evaluate every | send back |
|---|---|---|---|---|
| 1 | `runs/s2_shifted_meanflow.sh` | 80 000 | 5 000 (`runs/fid_stage2.sh`) | EMA ckpt at 20k/40k/60k/80k + raw at 80k |
| 2 | `runs/s2_shifted_lagrangian.sh` | 80 000 | same | same |
| 3 | `runs/s1_b_trunc09.sh` | 200 000 | 5 000 (`runs/fid_stage1.sh`) | EMA at every 40k + raw and EMA at 200k |
| 4 | `runs/s2_shifted_meanflow_eq.sh` | 80 000 | same as 1 | same as 1 |
| 5 | `runs/s2_shifted_meanflow_semi.sh` | 80 000 | same as 1 | same as 1 |
| 6 | `runs/s2_shifted_meanflow_eq_semi.sh` | 80 000 | same as 1 | same as 1 |

Rules while a run is going:
- All GPUs on the one run (`NGPU` in `runs/env.sh`); the FID script uses one GPU and may run alongside
  (`CUDA_VISIBLE_DEVICES=<idx> runs/fid_stage2.sh <run> <step>`), or after, if memory is tight.
- Every 5 000 steps a permanent checkpoint appears under `$RESULTS/<run>/kept/`. Evaluate it, paste the
  `FID-2k` summary lines into `RUNLOG.md` and into the wandb run notes.
- A crash or a machine restart: re-run the same script; it resumes from `latest.pt` and the log prints the
  resumed step. Note it in `RUNLOG.md`.
- Apply the abort criteria of `docs/EXPERIMENTS.md` section 7 literally.
- Send checkpoints back with `python scripts/wandb_artifacts.py put <file> --name <run> --type model --alias step-<k> --note "<weights>; FID-2k K8 <value>"`
  as soon as they are evaluated (Stage-2: X1 and K8 values in the note; field: native and legacy values).

## Phase 4 — report at the end of each run
Post in the wandb run notes and in `RUNLOG.md`, one table per run, rows = evaluated steps, columns = the
samplers of `docs/EXPERIMENTS.md` section 5, raw and EMA, plus GPU model, NGPU, samples/s, wall-clock hours,
number of restarts. Then start the next run.

## What is NOT your call
Learning rate, schedule, batch, precision, loss weights, sigma/t ranges, samplers, sample counts, seeds, the
order of runs, running two runs in parallel, using another evaluator, skipping a sanity check, self-distillation
variants. If you think one of these should change, write the reason in `RUNLOG.md` and ask; do not change it.
