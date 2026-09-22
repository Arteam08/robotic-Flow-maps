# Credentials for <friend> — SEND OUT OF BAND, NEVER COMMIT

Paste the block below into `.env` at the repo root (git-ignored). Nothing else is needed:
no `wandb login`, no `huggingface-cli login`, no account of your own.

```
WANDB_API_KEY=<service-account key>
WANDB_ENTITY=<team name>
WANDB_PROJECT=robotic-flow-maps
HF_TOKEN=<fine-grained token: read weights repo, write results repo>
RFM_HF_REPO=EquilibriumMap/robotic-flow-maps-weights
RFM_HF_RESULTS_REPO=EquilibriumMap/robotic-flow-maps-results
```

| credential | scope | what it lets you do | expires |
|---|---|---|---|
| `WANDB_API_KEY` | team `<team>` only | log curves and view runs in the team's projects | rotated at the end of the collaboration |
| `HF_TOKEN` | the two repos above only | read released weights, push your checkpoints under `<run name>/` | same |

Where things land
* curves: https://wandb.ai/<team>/robotic-flow-maps
* released weights: https://huggingface.co/EquilibriumMap/robotic-flow-maps-weights
* your checkpoints: https://huggingface.co/EquilibriumMap/robotic-flow-maps-results/tree/main/<run name>

If a token stops working, tell us; both are revocable and re-issuable in a minute.
