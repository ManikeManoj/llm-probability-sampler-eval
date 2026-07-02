#!/bin/bash
set -euo pipefail

mkdir -p logs outputs

# Replace these with the exact HuggingFace model IDs you are using on Helix.
# You can also export these variables in your shell before running this script.
export SMOKE_QWEN="${SMOKE_QWEN:-Qwen/Qwen3-4B}"
export SMOKE_GEMMA="${SMOKE_GEMMA:-google/gemma-3-12b-it}"
export SMOKE_MISTRAL="${SMOKE_MISTRAL:-mistralai/Mistral-7B-v0.3}"

echo "Submitting smoke tests with:"
echo "  Qwen   : $SMOKE_QWEN"
echo "  Gemma  : $SMOKE_GEMMA"
echo "  Mistral: $SMOKE_MISTRAL"

sbatch --array=0-2 scripts/smoke_array.slurm
