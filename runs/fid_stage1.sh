#!/usr/bin/env bash
# FID-2k of one Stage-1 (field b) checkpoint: Euler 250 steps, CFG 1.5 with uncond+guided norm matching, raw AND EMA,
# on the field's NATIVE stop (trunc a=0.9: eps-stop 0.2565 = t-stop 1e-3 in the clock's own time) and on the
# LEGACY horizon (eps-stop 1e-3, tau 6.9) that pairs with the 2.00 / 2.18 reference fields.
#   runs/fid_stage1.sh <run name> <step>        e.g. runs/fid_stage1.sh s1_b_trunc09 50000
#   runs/fid_stage1.sh --ckpt <file> <name>    any checkpoint file (e.g. weights/stage1_eqmft_B30k_ema.pt eqmft_B30k), logged as step 0
# Output: $FIDOUT/<run>/step<step>_<weights>_n<N>/euler250_<native|legacy>/adm_fid.json, then logged to wandb.
. "$(dirname "$0")/env.sh"
N=${N:-2000}
if [ "$1" = "--ckpt" ]; then CKPT=$2; RUN=$3; STEP=0; else CKPT=$RESULTS/$1/checkpoints/step_$(printf %07d $2).pt; RUN=$1; STEP=$2; fi
[ -f "$CKPT" ] || { echo "missing $CKPT"; exit 1; }
STOPS=${STOPS:-"native:0.2565 legacy:1e-3"}
for W in raw ema; do for st in $STOPS; do
  name=${st%%:*}; eps=${st#*:}; d=$FIDOUT/$RUN/step${STEP}_${W}_n${N}/euler250_$name; mkdir -p $d
  [ -f $d/adm_fid.json ] && { echo "done    $RUN step $STEP $W $name: $(cat $d/adm_fid.json)"; continue; }
  WFLAG=--use-ema; [ $W = raw ] && WFLAG=--no-ema
  $PY scripts/eval_fid.py --ckpt "$CKPT" $WFLAG --output-dir $d --num-samples $N --batch-size 50 --seed 0 \
    --num-steps 250 --eps-stop $eps --cfg-scale 1.5 --cfg-mode all --cfg-schedule constant \
    --cfg-match-uncond-norm --cfg-match-guided-norm --no-bf16 --skip-pytorch-fid --save-samples-npz $d/samples.npz
  $PY tools/fid_adm/score_npz.py --ref-cache "$EQFM_REF_INCEPTION" $d/samples.npz
  echo "FID-2k  $RUN step $STEP $W euler250 $name: $(cat $d/adm_fid.json)"
done; done
$PY scripts/log_fid_wandb.py --run "$RUN"
