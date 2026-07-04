#!/bin/bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: bash scripts/submit_rq_array.sh <config.jsonl>"
  echo "Optional env: CHUNK_SIZE=1 RUN_TAG=20260704_test"
  exit 1
fi

CONFIG="$1"
CHUNK_SIZE="${CHUNK_SIZE:-1}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"

if [ ! -f "$CONFIG" ]; then
  echo "Config file not found: $CONFIG"
  exit 1
fi

N_CONFIGS=$(grep -v '^[[:space:]]*$' "$CONFIG" | grep -v '^[[:space:]]*#' | wc -l)
if [ "$N_CONFIGS" -eq 0 ]; then
  echo "No configs found in $CONFIG"
  exit 1
fi

N_TASKS=$(( (N_CONFIGS + CHUNK_SIZE - 1) / CHUNK_SIZE ))
LAST_TASK=$(( N_TASKS - 1 ))

mkdir -p logs outputs

echo "Submitting RQ array"
echo "  config:      $CONFIG"
echo "  configs:     $N_CONFIGS"
echo "  chunk size:  $CHUNK_SIZE"
echo "  array tasks: 0-$LAST_TASK"
echo "  run tag:     $RUN_TAG"

sbatch \
  --array=0-${LAST_TASK} \
  --export=ALL,CONFIG="$CONFIG",CHUNK_SIZE="$CHUNK_SIZE",RUN_TAG="$RUN_TAG" \
  scripts/rq_array.slurm
