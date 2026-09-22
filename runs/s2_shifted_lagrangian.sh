#!/usr/bin/env bash
# main run 2: shifted Lagrangian, teacher distillation
# Resumable: re-running this script continues from $RESULTS/s2_shifted_lagrangian/checkpoints/latest.pt.
# Flag order matters: STAGE2_COMMON first, the run's own flags after it (argparse keeps the last value).
. "$(dirname "$0")/env.sh"
exec $TORCHRUN scripts/train_stage2_flowmap.py --results-dir "$RESULTS" --run-name s2_shifted_lagrangian --resume \
  $STAGE2_COMMON \
  --objective shifted_lagrangian  "$@"
