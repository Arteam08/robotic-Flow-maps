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
```

**Training data: build the latent cache yourself (no credential, about 1 hour of download + 2 to 4 GPU-hours).**
ImageNet-1k train at 256 px from an ungated mirror (52 parquet shards, 22.5 GB, standard sorted-WNID labels),
then one pass through the SD-VAE encoder that stores the posterior mean/std per image:
```bash
hf download evanarlian/imagenet_1k_resized_256 --repo-type dataset --include "data/train-*" --local-dir data/imagenet-parquet
ls data/imagenet-parquet/data/train-*.parquet | wc -l        # 52
python scripts/precompute_latents.py --shards "parquet:data/imagenet-parquet/data/train-*.parquet" \
    --out data/imagenet-latents-256 --batch-size 64 --num-workers 6 --verify 64
ls data/imagenet-latents-256/shard_*.npz | wc -l              # 52
python - <<'PY'
import numpy as np, glob
n = 0
for f in sorted(glob.glob("data/imagenet-latents-256/shard_*.npz")):
    d = np.load(f); assert d["mean"].shape[1:] == (4, 32, 32) and d["mean"].dtype == np.float16; n += len(d["label"])
print("rows", n, "min label", min(int(np.load(f)["label"].min()) for f in glob.glob("data/imagenet-latents-256/*.npz")))
PY
# expected: rows 1281167, min label 0 (labels must span 0..999)
```
`(older huggingface_hub: use huggingface-cli download instead of hf download)`. The script is idempotent: rerun it to
resume; shards whose npz exists are skipped. `--verify 64` re-encodes the first 64 rows of each shard in a second pass
and checks them. The mirror's JPEGs are pre-resized, so these latents differ slightly from the ones our fields were
trained on; this is fine for distillation and is stated in the results. Do not redistribute the cache.

Checkpoints can also go to Hugging Face if you were given `HF_TOKEN` (write on `EquilibriumMap/robotic-flow-maps-results`):
put `HF_TOKEN` and `RFM_HF_RESULTS_REPO` in `.env`, then
`python scripts/hf_upload.py --repo $RFM_HF_RESULTS_REPO --spec spec.json` with `"dest": "<run>/<file>"` entries
(see the script docstring), or let `rfm.tracking.Tracker.log_checkpoint` do it. Otherwise checkpoints go back the same way, one command per milestone (EMA every 20k steps, raw + EMA at the end):
```bash
python scripts/wandb_artifacts.py put results/<run>/kept/step_0020000.pt --name <run> --type model --alias step-20000 --note "EMA; FID-2k K8 18.9"
```

## 4. Check it works
```bash
python -m pytest tests/test_tracking.py -q     # offline + disabled paths, no network
python -c "from rfm.tracking import Tracker; t=Tracker('results/smoke'); t.log(0,{'x':1}); t.finish()"
```
The second command prints the wandb URL when the key is valid, otherwise the offline directory.
