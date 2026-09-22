# robotic-Flow-maps

Lean, self-contained copy of the EqFM ImageNet code (SiT-XL/2 autonomous fields and flow maps). The
experiment plan is provided to collaborators separately and is not part of this repository. Everything runs from environment
variables and a `.env` file: no personal wandb or Hugging Face accounts are needed
(`docs/SETUP.md`).

What is here

| path | role |
|---|---|
| `eqfm/` | models (SiT-XL/2), losses (Stage-1 autonomous field with linear/truncated clocks; Stage-2 flow-map objectives incl. `shifted_meanflow`, `shifted_lagrangian`), samplers, data loaders (ImageNet parquet, cached SD-VAE latents) |
| `scripts/train_stage1_eqm.py` | train the autonomous field b (from the public SiT-XL/2 weights or a Stage-1 checkpoint) |
| `scripts/train_stage2_flowmap.py` | distil a flow map from a frozen Stage-1 field |
| `scripts/eval_fid_stage2_flowmap.py`, `scripts/eval_fid.py`, `tools/fid_adm/` | sampling + FID on the ADM reference scale (X1, X1^2, paper K2/K4/K8 schedules; Euler for the field) |
| `scripts/precompute_latents.py` | encode ImageNet parquet shards to SD-VAE posteriors once |
| `rfm/tracking.py` | credential-free tracking helper (wandb online/offline fallback, `metrics.jsonl`, HF checkpoint upload) |
| `scripts/hf_download.py`, `scripts/hf_upload.py` | weights in and out of the Hugging Face repos |
| `tests/` | 85 unit tests (`python -m pytest tests -q`), CPU only |

Conventions that must not change: per-entry **mean** loss reduction, adaptive weight p=1 with eps 1e-3,
EMA 0.9999, fp32 training (`--no-allow-tf32 --no-bf16 --no-teacher-bf16`), FIDs tagged with evaluator and N.
