#!/bin/bash
#SBATCH --job-name=precision_v2
#SBATCH --partition=gpu-single
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --array=0-59%10
#SBATCH --output=logs/precision_v2_%A_%a.out
#SBATCH --error=logs/precision_v2_%A_%a.err

set -eo pipefail

cd "$HOME/projects/llm-probability-sampler-eval"

# Use the cache we already verified contains all required models.
export HF_HOME="$HOME/hf_cache"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

module load devel/miniforge/24.9.2
conda activate llm-thesis

# Avoid the /etc/bashrc + set -u issue we encountered previously.
set -u

echo "============================================================"
echo "PRECISION V2"
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "SLURM_ARRAY_JOB_ID=$SLURM_ARRAY_JOB_ID"
echo "SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"
echo "HOST=$(hostname)"
echo "DATE=$(date --iso-8601=seconds)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "============================================================"

PYTHONPATH=.:scripts:src \
python scripts/run_manifest_row_v2.py \
    --manifest manifests/precision_v2.jsonl \
    --index "$SLURM_ARRAY_TASK_ID"