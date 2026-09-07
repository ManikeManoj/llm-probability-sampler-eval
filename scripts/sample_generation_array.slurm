#!/bin/bash
#SBATCH --job-name=sg_array
#SBATCH --partition=gpu-single
#SBATCH --gres=gpu:A40:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/sample_generation/sg_array_%A_%a.out
#SBATCH --error=logs/sample_generation/sg_array_%A_%a.err

# Generic manifest-driven executor for sample generation.
#
# Submit with both a manifest and an explicit array range, for example:
#
#   sbatch --array=0-59%4 \
#     --export=ALL,MANIFEST_PATH=manifests/sample_generation_v2/sample_generation_layer1_v2.jsonl \
#     scripts/sample_generation_array.slurm
#
# RUN_MODE may be full (default), smoke, or dry.  A smoke run writes eight
# samples into the generator's separate <run_id>__smoke directory.

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

PYTHON_EXE="${PYTHON_EXE:-${HOME}/.conda/envs/llm-thesis/bin/python}"
GENERATOR_PATH="${GENERATOR_PATH:-src/sample_generation.py}"
RUN_MODE="${RUN_MODE:-full}"
REPAIR_INCOMPLETE="${REPAIR_INCOMPLETE:-false}"

export HF_HOME="${HOME}/hf_cache"
export TOKENIZERS_PARALLELISM=false

if [[ ! -x "${PYTHON_EXE}" ]]; then
    echo "Python environment not found: ${PYTHON_EXE}" >&2
    exit 1
fi

if [[ ! -f "${GENERATOR_PATH}" ]]; then
    echo "Generator not found: ${GENERATOR_PATH}" >&2
    exit 1
fi

if [[ -z "${MANIFEST_PATH:-}" ]]; then
    echo "MANIFEST_PATH was not supplied." >&2
    echo "Submit with --export=ALL,MANIFEST_PATH=<manifest.jsonl>." >&2
    exit 2
fi

if [[ ! -f "${MANIFEST_PATH}" ]]; then
    echo "Manifest not found: ${MANIFEST_PATH}" >&2
    exit 2
fi

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    echo "This script must be submitted with --array=<indices>." >&2
    exit 2
fi

case "${RUN_MODE}" in
    full|smoke|dry)
        ;;
    *)
        echo "RUN_MODE must be full, smoke, or dry; received: ${RUN_MODE}" >&2
        exit 2
        ;;
esac

case "${REPAIR_INCOMPLETE}" in
    true|false)
        ;;
    *)
        echo "REPAIR_INCOMPLETE must be true or false." >&2
        exit 2
        ;;
esac

# Python validates and extracts exactly one JSONL row.  NUL separators avoid
# shell evaluation and preserve JSON strings safely.
mapfile -d '' -t FIELDS < <(
    "${PYTHON_EXE}" - "${MANIFEST_PATH}" "${SLURM_ARRAY_TASK_ID}" <<'PY'
import json
import pathlib
import sys

manifest_path = pathlib.Path(sys.argv[1])
task_index = int(sys.argv[2])

with manifest_path.open("r", encoding="utf-8") as handle:
    rows = [json.loads(line) for line in handle if line.strip()]

if task_index < 0 or task_index >= len(rows):
    raise SystemExit(
        f"Array index {task_index} is outside manifest range 0-{len(rows) - 1}."
    )

row = rows[task_index]
if row.get("manifest_schema_version") != "sample-generation-manifest-v1":
    raise SystemExit(
        "Unexpected manifest schema: "
        f"{row.get('manifest_schema_version')!r}"
    )
if row.get("generator_schema_version") != "sample-generation-v2":
    raise SystemExit(
        "Manifest does not target sample-generation-v2: "
        f"{row.get('generator_schema_version')!r}"
    )

# A layer manifest uses manifest_layer_index; the combined manifest uses
# manifest_global_index.  The selected row must match at least one.
valid_indices = {
    row.get("manifest_layer_index"),
    row.get("manifest_global_index"),
}
if task_index not in valid_indices:
    raise SystemExit(
        f"Manifest index mismatch for task {task_index}: {sorted(valid_indices)}"
    )

sampling_seeds = row.get("sampling_seeds")
samples_per_seed = row.get("samples_per_seed")
if not isinstance(sampling_seeds, list) or not sampling_seeds:
    raise SystemExit("Manifest row has no sampling seeds.")
if len(set(sampling_seeds)) != len(sampling_seeds):
    raise SystemExit("Manifest row contains duplicate sampling seeds.")
if row.get("expected_attempts") != len(sampling_seeds) * samples_per_seed:
    raise SystemExit("Manifest expected_attempts is inconsistent.")

required = (
    "model_name",
    "model_label",
    "distribution",
    "params",
    "parameter_id",
    "prompt_type",
    "prompt_protocol",
    "experiment_layer",
    "run_id",
    "support_mode",
    "icl_n_examples",
    "icl_seed",
    "uses_icl",
    "samples_per_seed",
    "batch_size",
    "max_new_tokens",
    "output_dir",
    "expected_attempts",
)
missing = [key for key in required if key not in row]
if missing:
    raise SystemExit(f"Manifest row is missing fields: {missing}")

values = (
    row["model_name"],
    row["model_label"],
    row["distribution"],
    json.dumps(row["params"], sort_keys=True, separators=(",", ":")),
    row["parameter_id"],
    row["prompt_type"],
    row["prompt_protocol"],
    row["experiment_layer"],
    row["run_id"],
    row["support_mode"],
    "" if row.get("lower") is None else row["lower"],
    "" if row.get("upper") is None else row["upper"],
    row["icl_n_examples"],
    row["icl_seed"],
    str(row["uses_icl"]).lower(),
    " ".join(str(seed) for seed in sampling_seeds),
    row["samples_per_seed"],
    row["batch_size"],
    row["max_new_tokens"],
    row["output_dir"],
    row["expected_attempts"],
)

for value in values:
    sys.stdout.write(str(value))
    sys.stdout.write("\0")
PY
)

if [[ "${#FIELDS[@]}" -ne 21 ]]; then
    echo "Manifest extraction returned ${#FIELDS[@]} fields; expected 21." >&2
    exit 2
fi

MODEL_NAME="${FIELDS[0]}"
MODEL_LABEL="${FIELDS[1]}"
DISTRIBUTION="${FIELDS[2]}"
PARAMS="${FIELDS[3]}"
PARAMETER_ID="${FIELDS[4]}"
PROMPT_TYPE="${FIELDS[5]}"
PROMPT_PROTOCOL="${FIELDS[6]}"
EXPERIMENT_LAYER="${FIELDS[7]}"
RUN_ID="${FIELDS[8]}"
SUPPORT_MODE="${FIELDS[9]}"
LOWER="${FIELDS[10]}"
UPPER="${FIELDS[11]}"
ICL_N_EXAMPLES="${FIELDS[12]}"
ICL_SEED="${FIELDS[13]}"
USES_ICL="${FIELDS[14]}"
SAMPLING_SEEDS_TEXT="${FIELDS[15]}"
SAMPLES_PER_SEED="${FIELDS[16]}"
BATCH_SIZE="${FIELDS[17]}"
MAX_NEW_TOKENS="${FIELDS[18]}"
OUTPUT_DIR="${FIELDS[19]}"
EXPECTED_ATTEMPTS="${FIELDS[20]}"

read -r -a SAMPLING_SEED_ARGS <<< "${SAMPLING_SEEDS_TEXT}"

OPTIONAL_ARGS=()
if [[ -n "${LOWER}" ]]; then
    OPTIONAL_ARGS+=(--lower "${LOWER}")
fi
if [[ -n "${UPPER}" ]]; then
    OPTIONAL_ARGS+=(--upper "${UPPER}")
fi
if [[ "${USES_ICL}" == "true" ]]; then
    OPTIONAL_ARGS+=(--icl-n-examples "${ICL_N_EXAMPLES}" --icl-seed "${ICL_SEED}")
fi
if [[ "${RUN_MODE}" == "smoke" ]]; then
    OPTIONAL_ARGS+=(--smoke-test)
elif [[ "${RUN_MODE}" == "dry" ]]; then
    OPTIONAL_ARGS+=(--dry-run)
fi
if [[ "${REPAIR_INCOMPLETE}" == "true" ]]; then
    OPTIONAL_ARGS+=(--repair-incomplete)
fi

echo "Job ID: ${SLURM_JOB_ID}"
echo "Array task: ${SLURM_ARRAY_TASK_ID}"
echo "Manifest: ${MANIFEST_PATH}"
echo "Run mode: ${RUN_MODE}"
echo "Host: $(hostname)"
echo "Condition: ${RUN_ID}"
echo "Model: ${MODEL_NAME} (${MODEL_LABEL})"
echo "Target: ${DISTRIBUTION} ${PARAMS} (${PARAMETER_ID})"
echo "Prompt: ${PROMPT_TYPE} / ${PROMPT_PROTOCOL} / ICL seed ${ICL_SEED}"
echo "Expected attempts: ${EXPECTED_ATTEMPTS}"
echo "Output directory: ${OUTPUT_DIR}"

if [[ "${RUN_MODE}" != "dry" ]]; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
fi

"${PYTHON_EXE}" "${GENERATOR_PATH}" \
    --model-name "${MODEL_NAME}" \
    --model-label "${MODEL_LABEL}" \
    --distribution "${DISTRIBUTION}" \
    --params "${PARAMS}" \
    --parameter-id "${PARAMETER_ID}" \
    --prompt-type "${PROMPT_TYPE}" \
    --prompt-protocol "${PROMPT_PROTOCOL}" \
    --experiment-layer "${EXPERIMENT_LAYER}" \
    --run-id "${RUN_ID}" \
    --support-mode "${SUPPORT_MODE}" \
    --sampling-seeds "${SAMPLING_SEED_ARGS[@]}" \
    --samples-per-seed "${SAMPLES_PER_SEED}" \
    --batch-size "${BATCH_SIZE}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --output-dir "${OUTPUT_DIR}" \
    "${OPTIONAL_ARGS[@]}"
