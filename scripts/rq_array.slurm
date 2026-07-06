#!/bin/bash
#SBATCH --job-name=rq_grid
#SBATCH --partition=gpu-single
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/rq_grid_%A_%a.out
#SBATCH --error=logs/rq_grid_%A_%a.err

set -euo pipefail

CONFIG=${CONFIG:?CONFIG env var missing}
CHUNK_SIZE=${CHUNK_SIZE:-1}

cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs outputs

# Adjust module/env lines to your Helix setup if needed.
module purge || true
module load devel/miniforge/24.9.2 || true
source activate llm-thesis

export HF_HOME=${HF_HOME:-$HOME/hf_cache}
export HF_HUB_CACHE=${HF_HUB_CACHE:-$HF_HOME/hub}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-$HF_HOME/transformers}
export HF_HUB_ENABLE_HF_TRANSFER=1

START=$((SLURM_ARRAY_TASK_ID * CHUNK_SIZE))
END=$((START + CHUNK_SIZE))

echo "Job ID: ${SLURM_JOB_ID}"
echo "Array task: ${SLURM_ARRAY_TASK_ID}"
echo "Node: $(hostname)"
echo "Config: ${CONFIG}"
echo "Chunk size: ${CHUNK_SIZE}"
echo "Rows: ${START} to $((END-1))"
echo "Working directory: $(pwd)"
echo "HF_HOME: ${HF_HOME}"

python scripts/run_json_config.py --config "${CONFIG}" --start "${START}" --end "${END}"
