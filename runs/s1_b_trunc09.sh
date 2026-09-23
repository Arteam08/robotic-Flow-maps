#!/usr/bin/env bash
# main run 3: autonomous field b from the public SiT-XL/2 weights, truncated clock a = 0.9, 200k steps.
# Resumable: re-running continues from $RESULTS/s1_b_trunc09/checkpoints/latest.pt.
. "$(dirname "$0")/env.sh"
exec $TORCHRUN scripts/train_stage1_eqm.py --results-dir "$RESULTS" --run-name s1_b_trunc09 --resume --on-keep-cmd "$PWD/runs/fid_on_keep.sh s1_b_trunc09 {step}" \
  --shards latents:$LATENTS --num-workers 2 \
  --ckpt "$SIT" --ckpt-key auto --init-output-scale 1 --vae-id stabilityai/sd-vae-ft-ema \
  --time-conditioning strict --parameterization velocity \
  --time-sampler uniform_clock_weighted --clock trunc:0.9 --clock-lambda 1 --clock-weight-eps 1e-3 \
  --data-anchor 0 --cfg-dropout-prob 0.1 \
  --batch-size $BS --grad-accum-steps $ACCUM --steps 200000 --lr 1e-4 --warmup-steps 1000 \
  --lr-schedule cosine --lr-decay-start 120000 --lr-final 0 --weight-decay 0 --grad-clip 1 --ema-decay 0.9999 \
  --no-bf16 \
  --save-every 1000 --keep-last-checkpoints 2 --keep-every 5000 \
  --log-every 20 --val-every 2500 --val-n-samples 16 --val-num-steps 250 --val-eps-stop 0.2565 --val-cfg-scale 1.5 \
  --seed 0 "$@"
