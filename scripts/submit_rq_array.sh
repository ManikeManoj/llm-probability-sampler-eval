#!/usr/bin/env bash
set -euo pipefail

CONFIG=${1:?Usage: bash scripts/submit_rq_array.sh configs/file.jsonl [chunk_size]}
CHUNK_SIZE=${2:-1}

N_LINES=$(grep -cve '^\s*$' "$CONFIG")
N_TASKS=$(( (N_LINES + CHUNK_SIZE - 1) / CHUNK_SIZE ))
LAST=$(( N_TASKS - 1 ))

echo "Submitting $N_LINES configs as $N_TASKS array tasks, CHUNK_SIZE=$CHUNK_SIZE"
export CONFIG
export CHUNK_SIZE
sbatch --array=0-${LAST} scripts/rq_array.slurm
