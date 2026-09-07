#!/bin/bash
#SBATCH --job-name=sg_smoke_suite
#SBATCH --partition=gpu-single
#SBATCH --gres=gpu:A40:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --array=0-3%4
#SBATCH --output=logs/sample_generation/sg_smoke_suite_%A_%a.out
#SBATCH --error=logs/sample_generation/sg_smoke_suite_%A_%a.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

PYTHON_EXE="${HOME}/.conda/envs/llm-thesis/bin/python"
export HF_HOME="${HOME}/hf_cache"
export TOKENIZERS_PARALLELISM=false

if [[ ! -x "${PYTHON_EXE}" ]]; then
    echo "Python environment not found: ${PYTHON_EXE}" >&2
    exit 1
fi

ICL_ARGS=()

case "${SLURM_ARRAY_TASK_ID}" in
  0)
    MODEL_NAME="Qwen/Qwen3-4B-Instruct-2507"
    MODEL_LABEL="S3"
    DISTRIBUTION="normal"
    PARAMS='{"mean":0.0,"std":1.0}'
    PARAMETER_ID="N1"
    PROMPT_TYPE="plain"
    PROMPT_PROTOCOL="chat_direct"
    EXPERIMENT_LAYER="layer1"
    RUN_ID="l1_s3_n1_plain_chat"
    OUTPUT_DIR="outputs/sample_generation_smoke_suite_v2"
    ;;
  1)
    MODEL_NAME="Qwen/Qwen3-4B-Instruct-2507"
    MODEL_LABEL="S2"
    DISTRIBUTION="normal"
    PARAMS='{"mean":0.0,"std":10.0}'
    PARAMETER_ID="N6"
    PROMPT_TYPE="plain"
    PROMPT_PROTOCOL="raw_direct"
    EXPERIMENT_LAYER="layer2"
    RUN_ID="l2_s2_n6_plain_raw"
    OUTPUT_DIR="outputs/sample_generation_smoke_suite_v2"
    ;;
  2)
    MODEL_NAME="google/gemma-4-E4B-it"
    MODEL_LABEL="S5"
    DISTRIBUTION="beta"
    PARAMS='{"alpha":2.0,"beta":2.0}'
    PARAMETER_ID="B2"
    PROMPT_TYPE="icl"
    PROMPT_PROTOCOL="raw_direct"
    EXPERIMENT_LAYER="layer3"
    RUN_ID="l3_s5_b2_icl_seed0_raw"
    OUTPUT_DIR="outputs/sample_generation_smoke_suite_v2"
    ICL_ARGS=(--icl-n-examples 5 --icl-seed 0)
    ;;
  3)
    MODEL_NAME="Qwen/Qwen3-4B-Instruct-2507"
    MODEL_LABEL="S2"
    DISTRIBUTION="normal"
    PARAMS='{"mean":0.0,"std":1.0}'
    PARAMETER_ID="N1"
    PROMPT_TYPE="plain"
    PROMPT_PROTOCOL="raw_direct"
    EXPERIMENT_LAYER="layer1"
    RUN_ID="l1_s2_n1_plain_raw"
    OUTPUT_DIR="outputs/sample_generation_smoke_repro_v2"
    ;;
  *)
    echo "Unexpected SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}" >&2
    exit 2
    ;;
esac

echo "Job ID: ${SLURM_JOB_ID}"
echo "Array task: ${SLURM_ARRAY_TASK_ID}"
echo "Host: $(hostname)"
echo "Condition: ${RUN_ID}"
echo "Model: ${MODEL_NAME}"
echo "Output directory: ${OUTPUT_DIR}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

"${PYTHON_EXE}" src/sample_generation.py \
  --model-name "${MODEL_NAME}" \
  --model-label "${MODEL_LABEL}" \
  --distribution "${DISTRIBUTION}" \
  --params "${PARAMS}" \
  --parameter-id "${PARAMETER_ID}" \
  --prompt-type "${PROMPT_TYPE}" \
  --prompt-protocol "${PROMPT_PROTOCOL}" \
  --experiment-layer "${EXPERIMENT_LAYER}" \
  --run-id "${RUN_ID}" \
  --output-dir "${OUTPUT_DIR}" \
  "${ICL_ARGS[@]}" \
  --smoke-test
