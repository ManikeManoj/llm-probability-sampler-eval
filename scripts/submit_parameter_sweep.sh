#!/usr/bin/env bash
set -euo pipefail

CONFIG=${1:?Usage: bash scripts/submit_parameter_sweep.sh CONFIG.jsonl [chunk_size] [max_parallel]}
CHUNK_SIZE=${2:-1}
MAX_PARALLEL=${3:-8}

if [[ ! -f "$CONFIG" ]]; then
  echo "Config file not found: $CONFIG" >&2
  exit 1
fi
if ! [[ "$CHUNK_SIZE" =~ ^[1-9][0-9]*$ ]]; then
  echo "chunk_size must be a positive integer." >&2
  exit 1
fi
if ! [[ "$MAX_PARALLEL" =~ ^[1-9][0-9]*$ ]]; then
  echo "max_parallel must be a positive integer." >&2
  exit 1
fi

CONFIG_ABS=$(realpath "$CONFIG")
REPO_DIR=$(pwd)
N_CONFIGS=$(grep -v '^[[:space:]]*$' "$CONFIG_ABS" | grep -v '^[[:space:]]*#' | wc -l)
if [[ "$N_CONFIGS" -eq 0 ]]; then
  echo "No configs found in $CONFIG_ABS" >&2
  exit 1
fi

N_TASKS=$(( (N_CONFIGS + CHUNK_SIZE - 1) / CHUNK_SIZE ))
LAST_TASK=$(( N_TASKS - 1 ))

mkdir -p logs outputs

echo "Submitting parameter sweep"
echo "  config:       $CONFIG_ABS"
echo "  configs:      $N_CONFIGS"
echo "  chunk size:   $CHUNK_SIZE"
echo "  array:        0-${LAST_TASK}%${MAX_PARALLEL}"
echo "  repository:   $REPO_DIR"

sbatch \
  --array="0-${LAST_TASK}%${MAX_PARALLEL}" \
  --export=ALL,CONFIG="$CONFIG_ABS",CHUNK_SIZE="$CHUNK_SIZE",REPO_DIR="$REPO_DIR" \
  scripts/parameter_sweep_array.slurm
