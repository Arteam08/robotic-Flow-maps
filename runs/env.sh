# Shared settings for every launch script. Edit the four paths, nothing else.
# Sourced by runs/*.sh. All scripts are plain bash + torchrun: wrap them in your scheduler's job file.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHONPATH=$PWD PYTHONUNBUFFERED=1
[ -f .env ] && set -a && . ./.env && set +a

# --- paths (edit) ---
export WEIGHTS=${WEIGHTS:-$PWD/weights}                       # scripts/hf_download.py --dest weights
export LATENTS=${LATENTS:-$PWD/data/imagenet-latents-256}     # 294 shard_*.npz from EquilibriumMap/eqfm-imagenet-distill/latents
export RESULTS=${RESULTS:-$PWD/results}                       # run dirs: $RESULTS/<run name>/
export FIDOUT=${FIDOUT:-$PWD/results/fid}                     # FID sample dirs
export EQFM_REF_INCEPTION=${EQFM_REF_INCEPTION:-$WEIGHTS/ref_inception.npz}

# --- hardware (edit) ---
export NGPU=${NGPU:-4}                       # GPUs on this machine used by torchrun
export GLOBAL_BATCH=128                      # fixed by the handoff; do not change
export BS=${BS:-$((GLOBAL_BATCH / NGPU))}    # per-GPU micro-batch
export ACCUM=${ACCUM:-1}                     # raise if BS does not fit: BS * NGPU * ACCUM must equal 128
[ $((BS * NGPU * ACCUM)) -eq $GLOBAL_BATCH ] || { echo "BS*NGPU*ACCUM must be $GLOBAL_BATCH"; exit 1; }
export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-1}   # multi-GPU L40/L40S nodes hang in DDP init without it
export PY=${PY:-python}
TORCHRUN="$PY -m torch.distributed.run --standalone --nproc_per_node=$NGPU"

# --- fixed checkpoints ---
export EQMFT=$WEIGHTS/stage1_eqmft_B30k_ema.pt              # the "2.00 FID" field: teacher AND init of every Stage-2 run
export SIT=$WEIGHTS/SiT-XL-2-256x256.pt                     # public SiT-XL/2 flow-matching weights: init of the b training

# --- identical for every Stage-2 run (handoff section 3) ---
STAGE2_COMMON="--shards latents:$LATENTS --num-workers 2 --model SiT-XL/2
  --teacher-ckpt $EQMFT --teacher-weights ema --init-stage1-ckpt $EQMFT --init-stage1-weights ema
  --interp-time-sampler uniform --eps-train 0 --sigma-sampler uniform --sigma-min 0 --sigma-max 1
  --adaptive-weight-p 1 --adaptive-weight-eps 1e-3 --loss-reduction mean --diag-weight 1 --distill-weight 1
  --semi-weight 0 --terminal-weight 0 --terminal-teacher-norm-weight 0
  --guidance-mode guided --teacher-cfg-scale 1.5 --cfg-match-uncond-norm --cfg-match-guided-norm --cfg-mode all --cfg-dropout-prob 0
  --batch-size $BS --grad-accum-steps $ACCUM --steps 80000 --lr 1e-4 --warmup-steps 500 --lr-decay-from 48000 --lr-final 0
  --weight-decay 0 --grad-clip 1 --ema-decay 0.9999
  --stage2-loss-fp32 --no-allow-tf32 --no-bf16 --no-teacher-bf16 --no-student-bf16-jvp --no-student-bf16-nonjvp
  --save-every 1000 --keep-last-checkpoints 2 --keep-every 5000
  --log-every 20 --val-every 2500 --val-n-samples 16 --val-diag-num-steps 100 --val-offdiag-steps 2 4 8 --val-offdiag-s0 0.3 0.2 0.1
  --seed 0"
