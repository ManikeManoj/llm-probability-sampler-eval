#!/bin/bash
#SBATCH --job-name=rq1rq2
#SBATCH --partition=gpu-single
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/rq1rq2_%A_%a.out
#SBATCH --error=logs/rq1rq2_%A_%a.err

set -euo pipefail

mkdir -p logs outputs

echo "Job ID: ${SLURM_JOB_ID:-NA}"
echo "Array task: ${SLURM_ARRAY_TASK_ID:-NA}"
echo "Node: ${SLURMD_NODENAME:-NA}"
echo "Config: ${CONFIG:?CONFIG variable not set}"
echo "Chunk size: ${CHUNK_SIZE:-1}"
echo "Run tag: ${RUN_TAG:-none}"
echo "Working directory: $(pwd)"

# Helix environment. Adjust module/env names if your cluster setup differs.
module load devel/miniforge/24.9.2 || true
if command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
fi
conda activate llm-thesis

# Keep HF cache out of the repo.
export HF_HOME="$HOME/hf_cache"
export HF_HUB_CACHE="$HOME/hf_cache/hub"
export TRANSFORMERS_CACHE="$HOME/hf_cache/transformers"
export HF_DATASETS_CACHE="$HOME/hf_cache/datasets"
export TOKENIZERS_PARALLELISM=false

python scripts/run_json_config.py \
  --config "$CONFIG" \
  --index "${SLURM_ARRAY_TASK_ID}" \
  --chunk-size "${CHUNK_SIZE:-1}"
