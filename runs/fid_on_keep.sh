#!/usr/bin/env bash
# Called by the trainers at every kept checkpoint (--on-keep-cmd): runs the FID evaluation of that checkpoint in the
# background, one at a time (flock), on the GPU given by FID_GPU (default: the last GPU of the machine, shared with
# training at a reduced sample batch). Logs to $FIDOUT/<run>/on_keep.log. Nothing here can stop the training process.
#   runs/fid_on_keep.sh <run name> <step>
. "$(dirname "$0")/env.sh" >/dev/null 2>&1 || true
RUN=$1; STEP=$2; mkdir -p "$FIDOUT/$RUN"; LOG=$FIDOUT/$RUN/on_keep.log
NGPU_LOCAL=$(nvidia-smi -L 2>/dev/null | wc -l); export FID_GPU=${FID_GPU:-$((NGPU_LOCAL - 1))}
export FID_BATCH=${FID_BATCH:-25}     # small batch: the GPU is usually also training
case "$RUN" in s1_*) SCRIPT=runs/fid_stage1.sh;; *) SCRIPT=runs/fid_stage2.sh;; esac
(
  flock -w 86400 9 || exit 1
  echo "$(date -Is) start $RUN step $STEP on GPU $FID_GPU batch $FID_BATCH" >> "$LOG"
  CUDA_VISIBLE_DEVICES=$FID_GPU nice -n 10 $SCRIPT "$RUN" "$STEP" >> "$LOG" 2>&1
  echo "$(date -Is) done  $RUN step $STEP rc=$?" >> "$LOG"
) 9>"$FIDOUT/fid_on_keep.lock" </dev/null >/dev/null 2>&1 &
disown
