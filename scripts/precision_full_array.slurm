#!/bin/bash
#SBATCH --job-name=precision_full
#SBATCH --partition=gpu-single
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --array=0-39%2
#SBATCH --output=logs/precision_full_%A_%a.out
#SBATCH --error=logs/precision_full_%A_%a.err

set -euo pipefail

module purge
module load devel/miniforge/24.9.2

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate llm-thesis

cd "$HOME/projects/llm-probability-sampler-eval"

export HF_HOME="$HOME/hf_cache"
export HF_HUB_CACHE="$HF_HOME/hub"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

CONFIG="configs/precision_full_small_qg_plain_n500000.jsonl"

mkdir -p logs outputs

echo "========================================"
echo "Job ID:       $SLURM_JOB_ID"
echo "Array task:  $SLURM_ARRAY_TASK_ID"
echo "Node:         $(hostname)"
echo "Config:       $CONFIG"
echo "========================================"

if [[ ! -f "$CONFIG" ]]; then
    echo "Config file not found: $CONFIG"
    exit 1
fi

nvidia-smi

python scripts/run_json_config.py \
    --config "$CONFIG" \
    --index "$SLURM_ARRAY_TASK_ID"

echo "========================================"
echo "Completed config index: $SLURM_ARRAY_TASK_ID"
echo "========================================"