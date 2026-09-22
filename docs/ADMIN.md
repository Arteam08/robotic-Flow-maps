# Maintainer setup (what only we can do)

## wandb: one throwaway account (simple path)
1. Sign up at https://wandb.ai/signup with a throwaway mailbox, username e.g. `eqfm-handoff`, free plan.
   Copy the API key from https://wandb.ai/authorize.
2. On the account's home page: **Create new project** -> `robotic-flow-maps`, visibility **Public**.
3. Credentials for him: `WANDB_API_KEY=<that key>`, `WANDB_ENTITY=<the username>`. Rotate the key on the
   authorize page (or delete the account) when the collaboration ends. Never hand out your own key.
Alternative if live curves are not needed: leave `WANDB_API_KEY` empty; runs are logged offline and sent back
with `scripts/pack_offline_runs.sh`, then synced here with `wandb sync`.

## Hugging Face: two repos, one token
Both exist under the org, private (created 2026-09-22):
* `EquilibriumMap/robotic-flow-maps-weights`  released weights, read-only for him. README + manifest.json.
* `EquilibriumMap/robotic-flow-maps-results`  his checkpoints, one folder per run name, per-run manifest.json
  written by `Tracker.log_checkpoint`.

1. Release weights with `python scripts/hf_upload.py --spec upload_spec.json` (spec format in the
   docstring; our own write token in `HF_TOKEN`). Only EMA weights, one file per milestone.
2. Token for him: huggingface.co -> Settings -> Access Tokens -> **Create new token** -> Fine-grained:
   Repositories -> select `robotic-flow-maps-weights` (read) and `robotic-flow-maps-results`
   (read + write). No user or org permissions. Name it after him so it can be revoked alone.
   If the org repos do not appear in the picker, create a bot HF account, add it to the org with
   write on the results repo, and issue the token from that account instead.
3. Fill `docs/CREDENTIALS_TEMPLATE.md` and send it out of band (not in git, not in an issue).
4. When the release list is final: weights repo Settings -> **Make public** (then the download
   needs no token at all). The results repo can stay private.
5. Never Git-LFS weights into the GitHub repo (1 GB/month bandwidth cap).

## Offline runs coming back
```bash
tar -xzf offline_runs_*.tar.gz
WANDB_API_KEY=<our key> wandb sync --entity <team> --project robotic-flow-maps results/**/wandb/offline-run-*
```
