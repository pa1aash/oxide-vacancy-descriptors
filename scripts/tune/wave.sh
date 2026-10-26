#!/usr/bin/env bash
# Launch the four tuning studies of one anchor as detached processes (runs ON the pod).
#   bash scripts/tune/wave.sh <anchor> [extra run_study.py args, e.g. --dry-run --trials 2]
# Each study is one process (setsid + nohup, so an SSH drop does not kill it); logs go to
# /workspace/tuning/logs/. Start CUDA MPS first (bash scripts/pod/mps.sh start).
set -euo pipefail
ANCHOR="${1:?usage: wave.sh <anchor> [run_study args]}"; shift
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PY="${PY:-/workspace/miniforge3/envs/dftgnn-gpu/bin/python}"
LOGS=/workspace/tuning/logs
mkdir -p "$LOGS"
cd "$ROOT"
TAG=""; case " $* " in *" --dry-run "*) TAG="_dryrun" ;; esac
for m in S D-state D-late P1; do
  log="$LOGS/${m}_a${ANCHOR}${TAG}.log"
  setsid nohup "$PY" scripts/tune/run_study.py --model "$m" --anchor "$ANCHOR" --device cuda "$@" \
    >> "$log" 2>&1 < /dev/null &
  echo "$m a$ANCHOR pid $! log $log"
done
