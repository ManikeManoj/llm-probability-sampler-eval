#!/bin/bash
#SBATCH --job-name=llmprob_smoke
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --array=0-2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00

# If Helix requires a partition/account, uncomment/edit these:
# #SBATCH --partition=gpu
# #SBATCH --account=<your_account>

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs outputs

module purge || true
module load devel/miniforge/24.9.2
conda activate llm-thesis

export HF_HOME="$HOME/hf_cache"
export HF_HUB_CACHE="$HOME/hf_cache"
export TRANSFORMERS_CACHE="$HF_HUB_CACHE"
export TOKENIZERS_PARALLELISM=false

# Edit these defaults in submit_smoke.sh or export them before sbatch.
QWEN_MODEL="${SMOKE_QWEN:-Qwen/Qwen3-4B}"
GEMMA_MODEL="${SMOKE_GEMMA:-google/gemma-3-12b-it}"
MISTRAL_MODEL="${SMOKE_MISTRAL:-mistralai/Mistral-7B-v0.3}"

MODELS=("$QWEN_MODEL" "$GEMMA_MODEL" "$MISTRAL_MODEL")
TAGS=("qwen" "gemma" "mistral")
SCORING=("auto" "auto" "auto")

IDX="$SLURM_ARRAY_TASK_ID"
MODEL_NAME="${MODELS[$IDX]}"
TAG="${TAGS[$IDX]}"
METHOD="${SCORING[$IDX]}"

RUN_ID="smoke_${TAG}_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

echo "================ SMOKE TEST ================"
echo "date       : $(date)"
echo "host       : $(hostname)"
echo "job id     : ${SLURM_JOB_ID}"
echo "array idx  : ${SLURM_ARRAY_TASK_ID}"
echo "model      : ${MODEL_NAME}"
echo "scoring    : ${METHOD}"
echo "run id     : ${RUN_ID}"
echo "============================================"

python src/run_compare.py \
  --model-name "$MODEL_NAME" \
  --lm-scoring-method "$METHOD" \
  --distribution normal \
  --mean 4.0 \
  --std 1.0 \
  --prompt-type plain \
  --support-mode agnostic \
  --prefixes ROOT,3,4,4.,5 \
  --n-samples 10000 \
  --decimals 3 \
  --seed 42 \
  --run-id "$RUN_ID"

echo "Smoke test finished for ${TAG} at $(date)"
