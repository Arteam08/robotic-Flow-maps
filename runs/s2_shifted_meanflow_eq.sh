#!/usr/bin/env bash
# add-on A: run 1 + eq loss ||b(X_1(z))||^2 (teacher field at the one-shot endpoint)
# Resumable: re-running this script continues from $RESULTS/s2_shifted_meanflow_eq/checkpoints/latest.pt.
# Flag order matters: STAGE2_COMMON first, the run's own flags after it (argparse keeps the last value).
. "$(dirname "$0")/env.sh"
exec $TORCHRUN scripts/train_stage2_flowmap.py --results-dir "$RESULTS" --run-name s2_shifted_meanflow_eq --resume --on-keep-cmd "$PWD/runs/fid_on_keep.sh s2_shifted_meanflow_eq {step}" \
  $STAGE2_COMMON \
  --objective shifted_meanflow --mf-single-jvp --terminal-teacher-norm-weight 1 --terminal-norm-max-batch 4 "$@"
