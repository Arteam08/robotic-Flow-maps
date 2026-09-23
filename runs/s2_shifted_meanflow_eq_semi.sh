#!/usr/bin/env bash
# add-on C: run 1 + eq loss + semigroup
# Resumable: re-running this script continues from $RESULTS/s2_shifted_meanflow_eq_semi/checkpoints/latest.pt.
# Flag order matters: STAGE2_COMMON first, the run's own flags after it (argparse keeps the last value).
. "$(dirname "$0")/env.sh"
exec $TORCHRUN scripts/train_stage2_flowmap.py --results-dir "$RESULTS" --run-name s2_shifted_meanflow_eq_semi --resume --on-keep-cmd "$PWD/runs/fid_on_keep.sh s2_shifted_meanflow_eq_semi {step}" \
  $STAGE2_COMMON \
  --objective shifted_meanflow --mf-single-jvp --semi-weight 1 --terminal-teacher-norm-weight 1 --terminal-norm-max-batch 4 "$@"
