#!/bin/bash
#SBATCH --job-name=sg_pilot_s2_n1
#SBATCH --partition=gpu-single
#SBATCH --gres=gpu:A40:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/sample_generation/sg_pilot_s2_n1_%j.out
#SBATCH --error=logs/sample_generation/sg_pilot_s2_n1_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

PYTHON_EXE="${HOME}/.conda/envs/llm-thesis/bin/python"
export HF_HOME="${HOME}/hf_cache"
export TOKENIZERS_PARALLELISM=false

if [[ ! -x "${PYTHON_EXE}" ]]; then
    echo "Python environment not found: ${PYTHON_EXE}" >&2
    exit 1
fi

echo "Job ID: ${SLURM_JOB_ID}"
echo "Host: $(hostname)"
echo "Condition: S2 / Normal(0,1) / plain / raw"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

"${PYTHON_EXE}" src/sample_generation.py \
  --model-name Qwen/Qwen3-4B-Instruct-2507 \
  --model-label S2 \
  --distribution normal \
  --params '{"mean":0.0,"std":1.0}' \
  --parameter-id N1 \
  --prompt-type plain \
  --prompt-protocol raw_direct \
  --experiment-layer layer1 \
  --run-id l1_s2_n1_plain_raw \
  --sampling-seeds 100 101 102 103 104 \
  --samples-per-seed 200 \
  --batch-size 8 \
  --max-new-tokens 16 \
  --output-dir outputs/sample_generation_pilot_v2
