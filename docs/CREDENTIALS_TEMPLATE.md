# Credentials for <friend> — SEND OUT OF BAND, NEVER COMMIT

Paste the block below into `.env` at the repo root (git-ignored). That is the only credential:
no wandb account, no Hugging Face account, no `wandb login`, no `huggingface-cli login`.

```
WANDB_API_KEY=<key>
WANDB_ENTITY=eqfm-handoff-carnegie-mellon-university
WANDB_PROJECT=robotic-flow-maps
RFM_HF_REPO=EquilibriumMap/robotic-flow-maps-weights
```

| what | where | credential |
|---|---|---|
| released weights + FID reference | https://huggingface.co/EquilibriumMap/robotic-flow-maps-weights (public) | none |
| training data | built locally from the public mirror `evanarlian/imagenet_1k_resized_256` (docs/SETUP.md) | none |
| your curves | https://wandb.ai/eqfm-handoff-carnegie-mellon-university/robotic-flow-maps | the key (viewing may need no login if the project is public) |
| your checkpoints back to us | wandb artifacts named after the run, alias `step-<k>` | the key |

The key belongs to a throwaway account made for this collaboration and will be rotated when it ends.
