# Published-scale ("ADM") FID evaluator

Every ImageNet FID in the paper handoff tagged **ADM-50k / ADM-10k / ADM-2k** comes from this evaluator:
torch-fidelity's TF-compatible InceptionV3 (`inception-v3-compat`) scored against the **full mu/sigma**
carried inside the ADM reference batch `VIRTUAL_imagenet256_labeled.npz`.

It is NOT the same number as `eqfm.metrics` (pytorch-fid), which reads **+3.5 to +3.9 FID higher** on identical
samples (PT-10k tag). Never mix the two in one table.

## One-time setup

```bash
pip install torch-fidelity torchvision scipy          # plus torch
# ADM reference batch (~7 GB, 10k images + precomputed mu/sigma over the full 50k train set)
wget https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/256/VIRTUAL_imagenet256_labeled.npz
# builds ref_inception.npz (~115 MB): reference Inception features (for KID) + the ADM mu/sigma (for FID)
python tools/fid_adm/eval_fid.py --ref-npz VIRTUAL_imagenet256_labeled.npz --ref-cache ref_inception.npz
export EQFM_REF_INCEPTION=$PWD/ref_inception.npz
```

## Scoring samples

```bash
# 1) generate, class-balanced labels (i % 1000), seed 0  -- one L40S does ~10k samples/h at K=8
python scripts/eval_fid_stage2_flowmap.py --ckpt <ckpt.pt> --weights raw \
  --sampler offdiag --offdiag-steps 8 --offdiag-schedule paper --offdiag-s0 0.1 \
  --num-samples 10000 --batch-size 64 --seed 0 --skip-pytorch-fid \
  --save-samples-npz out/K8/samples.npz --output-dir out/K8
# 2) score (writes out/K8/adm_fid.json)
python tools/fid_adm/score_npz.py out/K8/samples.npz
```

50k runs are 5 shards of 10k with seeds 10000*i (shard 0 == the seed-0 10k run, so the 10k number is paired
with the 50k one); concatenate the `arr_0` arrays and score once. `~/EqFM/fid50k_flowmap_babel.slurm` +
`scripts/score_npz_shards.py` do that on Babel.

Reference points on this evaluator (class-balanced labels, seed 0): Stage-1 uw450k teacher 2.18 @50k / 4.68 @10k;
student `s2_lagsemi` 40k raw K8 2.47 @50k / 4.94 @10k; released EqM-XL/2 4.30 @10k; DiT-XL/2 4.57 @10k.
At this quality level, FID(50k) ~ FID(10k) - 2.5.
