#!/bin/bash
#SBATCH --job-name=fwv2_smoke
#SBATCH --partition=gpu-single
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --array=0-5
#SBATCH --output=logs/fwv2_smoke_%A_%a.out
#SBATCH --error=logs/fwv2_smoke_%A_%a.err

set -euo pipefail

cd "$HOME/projects/llm-probability-sampler-eval"
source ~/.bashrc
conda activate llm-thesis

mkdir -p logs outputs

MODELS=(
  "Qwen/Qwen3-4B-Base"
  "Qwen/Qwen3-4B-Instruct-2507"
  "Qwen/Qwen3-4B-Instruct-2507"
  "google/gemma-4-E4B"
  "google/gemma-4-E4B-it"
  "google/gemma-4-E4B-it"
)

PROTOCOLS=(
  "raw_direct"
  "raw_direct"
  "chat_direct"
  "raw_direct"
  "raw_direct"
  "chat_direct"
)

TAGS=(
  "qwen4b_base_raw"
  "qwen4b_it_raw"
  "qwen4b_it_chat"
  "gemmae4b_base_raw"
  "gemmae4b_it_raw"
  "gemmae4b_it_chat"
)

MODEL="${MODELS[$SLURM_ARRAY_TASK_ID]}"
PROTOCOL="${PROTOCOLS[$SLURM_ARRAY_TASK_ID]}"
TAG="${TAGS[$SLURM_ARRAY_TASK_ID]}"

echo "MODEL=$MODEL"
echo "PROTOCOL=$PROTOCOL"
echo "TAG=$TAG"

PYTHONPATH=.:scripts:src python scripts/run_compare.py \
  --model-name "$MODEL" \
  --lm-scoring-method single_token \
  --prompt-protocol "$PROTOCOL" \
  --distribution normal \
  --params '{"mean": 0.0, "std": 1.0}' \
  --n-samples 100000 \
  --decimals 3 \
  --prefixes 'ROOT,-,-1,-0,0,1,-1.,-0.,0.,1.,-0.0,0.5,1.0' \
  --prompt-type plain \
  --run-id "fwv2_smoke_${TAG}"
