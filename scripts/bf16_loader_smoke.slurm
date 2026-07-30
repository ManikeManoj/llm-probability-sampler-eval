#!/bin/bash
#SBATCH --job-name=bf16_smoke
#SBATCH --partition=gpu-single
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --array=0-2%1
#SBATCH --output=logs/bf16_smoke_%A_%a.out
#SBATCH --error=logs/bf16_smoke_%A_%a.err

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

mkdir -p logs outputs

case "$SLURM_ARRAY_TASK_ID" in
    0)
        MODEL_NAME="Qwen/Qwen3-4B-Instruct-2507"
        MODEL_TAG="qwen4b_instruct"
        ;;
    1)
        MODEL_NAME="google/gemma-4-E4B"
        MODEL_TAG="gemma_e4b_base"
        ;;
    2)
        MODEL_NAME="google/gemma-4-E4B-it"
        MODEL_TAG="gemma_e4b_it"
        ;;
    *)
        echo "Unknown array index: $SLURM_ARRAY_TASK_ID"
        exit 1
        ;;
esac

RUN_ID="bf16_loader_smoke_${MODEL_TAG}"

echo "========================================"
echo "Job ID:       $SLURM_JOB_ID"
echo "Array task:  $SLURM_ARRAY_TASK_ID"
echo "Node:         $(hostname)"
echo "Model:        $MODEL_NAME"
echo "Precision:    BF16"
echo "Run ID:       $RUN_ID"
echo "========================================"

nvidia-smi

python src/run_compare.py \
    --model-name "$MODEL_NAME" \
    --lm-scoring-method auto \
    --no-4bit \
    --distribution normal \
    --params '{"mean": 0.0, "std": 1.0}' \
    --n-samples 10000 \
    --decimals 3 \
    --seed 42 \
    --prefixes 'ROOT,0.,1.' \
    --prompt-type plain \
    --run-id "$RUN_ID"

echo "========================================"
echo "Completed: $RUN_ID"
echo "========================================"