#!/usr/bin/env bash
#SBATCH --job-name=param_sweep
#SBATCH --partition=gpu-single
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/param_sweep_%A_%a.out
#SBATCH --error=logs/param_sweep_%A_%a.err

set -euo pipefail

: "${CONFIG:?CONFIG must point to the parameter-sweep JSONL file.}"
CHUNK_SIZE="${CHUNK_SIZE:-1}"
REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR}}"

cd "$REPO_DIR"
mkdir -p logs outputs

module purge
module load devel/miniforge/24.9.2
eval "$(conda shell.bash hook)"
conda activate "$HOME/.conda/envs/llm-thesis"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-$HOME/hf_cache}"
export HF_XET_HIGH_PERFORMANCE=1

echo "Host:              $(hostname)"
echo "Repository:        $PWD"
echo "Config:            $CONFIG"
echo "Array task:        $SLURM_ARRAY_TASK_ID"
echo "Chunk size:        $CHUNK_SIZE"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi || true

python scripts/run_parameter_sweep_config.py \
  --config "$CONFIG" \
  --index "$SLURM_ARRAY_TASK_ID" \
  --chunk-size "$CHUNK_SIZE"
