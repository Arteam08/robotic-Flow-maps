# Setup (5 minutes, no personal accounts)

## 1. Environment
```bash
git clone https://github.com/Arteam08/robotic-Flow-maps && cd robotic-Flow-maps
python -m venv .venv && source .venv/bin/activate      # or conda
pip install -r requirements.txt   # CUDA 12.6 torch wheels; see the file header for other CUDA versions
cp .env.example .env                                    # git-ignored
```

## 2. Credentials (all optional, all via `.env`)
| variable | what | who gives it |
|---|---|---|
| `WANDB_API_KEY` | shared **service-account** key, not tied to any person | us |
| `WANDB_ENTITY` | the team name the key belongs to | us |
| `WANDB_PROJECT` | leave `robotic-flow-maps` | preset |
| `HF_TOKEN` | fine-grained token: read the weights repo, write the results repo | us |

Do **not** run `wandb login` or `huggingface-cli login`: both write the token into your home
directory. The code reads `.env` and the environment only.

Nothing breaks without a key:
* no `WANDB_API_KEY` -> runs are logged **offline** under `results/<run>/wandb/`. Every scalar is
  also in `results/<run>/metrics.jsonl` regardless. Send offline runs back with
  `scripts/pack_offline_runs.sh`, we sync them.
* `WANDB_MODE=disabled` -> local files only.

## 3. Weights and data
```bash
python scripts/hf_download.py --list          # what is available
python scripts/hf_download.py                 # everything -> weights/, sha256-verified
wget -O weights/SiT-XL-2-256x256.pt https://dl.fbaipublicfiles.com/sit/SiT-XL-2-256x256.pt   # public, check sha256 vs --list
hf download EquilibriumMap/eqfm-imagenet-distill --include "latents/*" --local-dir data/tmp && mv data/tmp/latents data/imagenet-latents-256
```
The latent cache (ImageNet-1k train as SD-VAE posteriors, 21 GB) lives in a private repo: it is derived from
ImageNet and must not be redistributed. Your token has read access to it.
Checkpoints you produce are recorded in `results/<run>/checkpoints.jsonl` (sha256, step) and
uploaded by `Tracker.log_checkpoint` to `EquilibriumMap/robotic-flow-maps-results/<run name>/`
together with a per-run `manifest.json` (needs `HF_TOKEN`; without it they stay local, with only a
wandb key they go to wandb Artifacts instead). Only milestone EMA checkpoints are uploaded, never
every save. To pull a run back: `python scripts/hf_download.py --repo $RFM_HF_RESULTS_REPO --run <run name>`.

## 4. Check it works
```bash
python -m pytest tests/test_tracking.py -q     # offline + disabled paths, no network
python -c "from rfm.tracking import Tracker; t=Tracker('results/smoke'); t.log(0,{'x':1}); t.finish()"
```
The second command prints the wandb URL when the key is valid, otherwise the offline directory.
