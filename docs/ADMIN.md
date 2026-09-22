# Maintainer setup (what only we can do)

## wandb: one throwaway account (simple path)
1. Sign up at https://wandb.ai/signup with a throwaway mailbox, username e.g. `eqfm-handoff`, free plan.
   Copy the API key from https://wandb.ai/authorize.
2. On the account's home page: **Create new project** -> `robotic-flow-maps`, visibility **Public**.
3. Credentials for him: `WANDB_API_KEY=<that key>`, `WANDB_ENTITY=<the username>`. Rotate the key on the
   authorize page (or delete the account) when the collaboration ends. Never hand out your own key.
Alternative if live curves are not needed: leave `WANDB_API_KEY` empty; runs are logged offline and sent back
with `scripts/pack_offline_runs.sh`, then synced here with `wandb sync`.

## Hugging Face: nothing for him to set up
`EquilibriumMap/robotic-flow-maps-weights` is PUBLIC (made public 2026-09-22): `stage1_eqmft_B30k_ema.pt`,
`ref_inception.npz`, manifest with the SiT-XL/2 sha256. Add releases with `python scripts/hf_upload.py --spec ...`
(our token). The results repo `robotic-flow-maps-results` exists but is unused in the simple path.

## Data and results
* Training data: he builds the latent cache himself from the ungated mirror `evanarlian/imagenet_1k_resized_256`
  (docs/SETUP.md). We do not ship ImageNet-derived data. The private `eqfm-imagenet-distill` repo stays private.
* His checkpoints = wandb model artifacts named after the run, alias `step-<k>` (`scripts/wandb_artifacts.py put`).
  Pull one here: `python scripts/wandb_artifacts.py get s2_shifted_meanflow:step-20000 --dest /data/.../s2_shifted_meanflow`.
  Free-plan storage is 100 GB: about 6 EMA checkpoints per run (2.7 GB each) fits; delete old versions in the wandb UI if needed.

## Offline runs coming back
```bash
tar -xzf offline_runs_*.tar.gz
WANDB_API_KEY=<our key> wandb sync --entity <team> --project robotic-flow-maps results/**/wandb/offline-run-*
```
