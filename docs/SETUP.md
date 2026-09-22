# Setup (5 minutes, no personal accounts)

## 1. Environment
```bash
git clone https://github.com/Arteam08/robotic-Flow-maps && cd robotic-Flow-maps
python -m venv .venv && source .venv/bin/activate      # or conda
pip install -r requirements.txt   # CUDA 12.6 torch wheels; see the file header for other CUDA versions
cp .env.example .env                                    # git-ignored
```

## 2. Credentials (one line)
Put the key you received in `.env` as `WANDB_API_KEY`, keep the entity and project lines from `.env.example`.
Do **not** run `wandb login` or `huggingface-cli login`. Nothing else is needed: the weights repo is public and
the training data comes through the same wandb account.

Nothing breaks without the key: runs are logged **offline** under `results/<run>/wandb/` and every scalar is in
`results/<run>/metrics.jsonl`; send offline runs back with `scripts/pack_offline_runs.sh`.

## 3. Weights and data
```bash
python scripts/hf_download.py --list          # what is released
python scripts/hf_download.py                 # -> weights/, sha256-verified, no token
wget -O weights/SiT-XL-2-256x256.pt https://dl.fbaipublicfiles.com/sit/SiT-XL-2-256x256.pt   # public; sha256 in --list
python scripts/wandb_artifacts.py get imagenet-latents-256:latest --dest data/imagenet-latents-256   # 21 GB, needs the key
ls data/imagenet-latents-256 | wc -l          # 294
```
The latent cache is derived from ImageNet: do not redistribute it.
If the artifact is not available, build the cache yourself (about 3 GPU-hours plus a 150 GB download):
```bash
# ImageNet-1k train, 256 px, parquet (ungated mirror), then encode once with the SD-VAE
hf download benjamin-paine/imagenet-1k-256x256 --repo-type dataset --include "data/train-*" --local-dir data/imagenet-parquet
python scripts/precompute_latents.py --shards "parquet:data/imagenet-parquet/data/train-*.parquet" --out data/imagenet-latents-256
```

Checkpoints you produce go back the same way, one command per milestone (EMA every 20k steps, raw + EMA at the end):
```bash
python scripts/wandb_artifacts.py put results/<run>/kept/step_0020000.pt --name <run> --type model --alias step-20000 --note "EMA; FID-2k K8 18.9"
```

## 4. Check it works
```bash
python -m pytest tests/test_tracking.py -q     # offline + disabled paths, no network
python -c "from rfm.tracking import Tracker; t=Tracker('results/smoke'); t.log(0,{'x':1}); t.finish()"
```
The second command prints the wandb URL when the key is valid, otherwise the offline directory.
