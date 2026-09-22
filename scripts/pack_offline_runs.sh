#!/usr/bin/env bash
# Friend side, only when running WITHOUT a wandb key: bundle every offline wandb run under results/
# into one tarball to send back.  We sync it with `wandb sync` on our side.
set -euo pipefail
ROOT=${1:-results}
OUT=offline_runs_$(date +%Y%m%d_%H%M).tar.gz
find "$ROOT" -type d -path '*/wandb/offline-run-*' -print0 | tar -czvf "$OUT" --null -T -
echo "wrote $OUT  ($(du -h "$OUT" | cut -f1)); send this file"
