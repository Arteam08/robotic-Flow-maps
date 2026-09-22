#!/usr/bin/env bash
# FID-2k of one Stage-2 checkpoint, every sampler of the handoff, raw AND EMA weights, then ADM-scale scoring.
#   runs/fid_stage2.sh <run name> <step>            e.g. runs/fid_stage2.sh s2_shifted_meanflow 20000
# Output: $FIDOUT/<run>/step<step>/<weights>/<sampler>/adm_fid.json  and a one-line summary per sampler.
# Uses ONE GPU (set CUDA_VISIBLE_DEVICES). ~15 min per checkpoint on an L40S.
. "$(dirname "$0")/env.sh"
RUN=$1; STEP=$2; N=${N:-2000}
CKPT=$RESULTS/$RUN/checkpoints/step_$(printf %07d $STEP).pt; [ -f "$CKPT" ] || { echo "missing $CKPT"; exit 1; }
[ -f "$EQFM_REF_INCEPTION" ] || { echo "missing $EQFM_REF_INCEPTION (weights/ref_inception.npz)"; exit 1; }
SAMPLERS=(
  "X1:--sampler terminal"
  "X1x2:--sampler terminal --terminal-repeats 2"
  "K2_s0.3:--sampler offdiag --offdiag-steps 2 --offdiag-schedule paper --offdiag-s0 0.3"
  "K4_s0.2:--sampler offdiag --offdiag-steps 4 --offdiag-schedule paper --offdiag-s0 0.2"
  "K8_s0.1:--sampler offdiag --offdiag-steps 8 --offdiag-schedule paper --offdiag-s0 0.1"
)
for W in raw ema; do for spec in "${SAMPLERS[@]}"; do
  name=${spec%%:*}; extra=${spec#*:}; d=$FIDOUT/$RUN/step$STEP/$W/$name; mkdir -p $d
  [ -f $d/adm_fid.json ] && { echo "done    $RUN step $STEP $W $name: $(cat $d/adm_fid.json)"; continue; }
  $PY scripts/eval_fid_stage2_flowmap.py --ckpt "$CKPT" --weights $W --model SiT-XL/2 --output-dir $d \
    --num-samples $N --batch-size 100 --seed 0 --skip-pytorch-fid --save-samples-npz $d/samples.npz --no-bf16 $extra
  $PY tools/fid_adm/score_npz.py --ref-cache "$EQFM_REF_INCEPTION" $d/samples.npz
  echo "FID-2k  $RUN step $STEP $W $name: $(cat $d/adm_fid.json)"
done; done
