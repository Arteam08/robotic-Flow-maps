# Orientation for an agent working in this repository

Start with `HANDOFF.md` and follow it in order. It points to `docs/SETUP.md` (environment, the single credential,
weights, data) and `docs/EXPERIMENTS.md` (the runs, fixed settings, evaluation, reference numbers, abort rules).
Launch only through `runs/*.sh`. Do not edit code; edit the path variables at the top of `runs/env.sh` only.
Keep `RUNLOG.md` up to date: every command run, its outcome, every FID line, every restart.

Non-negotiable conventions: per-entry mean loss, adaptive weight p = 1 with eps 1e-3, EMA 0.9999, fp32 with TF32 off,
global batch 128, sigma and t uniform on the full [0, 1], guidance 1.5 with unconditional + guided norm matching,
FIDs reported only as `ADM-2k` (2 000 samples, seed 0, labels i mod 1000, `tools/fid_adm`), always stating raw or EMA.
Never run `wandb login` or any Hugging Face login; credentials come from `.env` only.
