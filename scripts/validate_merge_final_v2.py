#!/usr/bin/env python3

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from final_v2_spec import (
    SMALL_MODEL_PROTOCOLS,
    MEDIUM_MODEL_PROTOCOLS,
    PARAMETER_CONFIGS,
    BASELINE_PARAMETER_IDS,
    ALL_PROMPT_TYPES,
    MAIN_PROMPT_TYPES,
)


# =============================================================================
# PATHS / SETTINGS
# =============================================================================

MASTER_MANIFEST = Path("manifests/final_v2_execution_all.jsonl")
OUTPUT_DIR = Path("outputs")
MERGED_DIR = Path("merged")
VALIDATION_DIR = Path("validation")

TOL = 1e-5


# =============================================================================
# HELPERS
# =============================================================================

def flatten_prefixes(parameter):
    out = []
    for prefixes in parameter["prefixes"].values():
        out.extend(prefixes)
    return out


def normalize_prefix(value):
    """
    Manifest uses ROOT. Output CSV may use ROOT or an empty string.
    Preserve all non-root representations exactly, including -0.00 etc.
    """
    if pd.isna(value):
        return "ROOT"

    value = str(value)

    if value in {"", "ROOT", "<ROOT>"}:
        return "ROOT"

    return value


def read_jsonl(path):
    rows = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    return rows


def read_csv(path):
    # Force prefix to remain textual so 0.50 does not become 0.5.
    return pd.read_csv(
        path,
        dtype={"prefix": "string"},
        keep_default_na=False,
    )


def require_columns(df, columns, path):
    missing = sorted(set(columns) - set(df.columns))

    if missing:
        raise AssertionError(
            f"{path}: missing required columns: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )


def find_column(df, candidates, label, path):
    for col in candidates:
        if col in df.columns:
            return col

    raise AssertionError(
        f"{path}: could not find {label}. "
        f"Tried {candidates}. "
        f"Available columns: {list(df.columns)}"
    )


def config_key(row):
    return (
        row["model_id"],
        row["parameter_id"],
        row["prompt_protocol"],
        row["prompt_type"],
        row["precision"],
    )


def expected_keys(models, parameters, prompts):
    out = []

    for model in models:
        for parameter in parameters:
            for prompt in prompts:
                out.append(
                    (
                        model["id"],
                        parameter["id"],
                        model["protocol"],
                        prompt,
                        "bf16",
                    )
                )

    return out


def assert_single_value(df, col, expected, path):
    if col not in df.columns:
        return

    values = {
        str(x)
        for x in df[col].unique()
    }

    if values != {str(expected)}:
        raise AssertionError(
            f"{path}: {col} mismatch. "
            f"expected={expected!r}, observed={values}"
        )


def add_manifest_metadata(df, manifest_row, logical_experiment):
    """
    Preserve raw result columns, but ensure merged files contain the
    metadata needed for later RQ analysis.
    """
    df = df.copy()

    metadata = {
        "model_id": manifest_row["model_id"],
        "model_name": manifest_row["model_name"],
        "parameter_id": manifest_row["parameter_id"],
        "distribution": manifest_row["distribution"],
        "prompt_protocol": manifest_row["prompt_protocol"],
        "prompt_type": manifest_row["prompt_type"],
        "precision": manifest_row["precision"],
        "lm_scoring_method": manifest_row["lm_scoring_method"],
        "n_samples": manifest_row["n_samples"],
        "decimals": manifest_row["decimals"],
        "seed": manifest_row["seed"],
        "icl_n_examples": manifest_row["icl_n_examples"],
        "icl_seed": manifest_row["icl_seed"],
    }

    for col, value in metadata.items():
        if col not in df.columns:
            df[col] = value

    df["experiment_membership"] = "|".join(
        sorted(manifest_row["experiment_membership"])
    )

    df["logical_experiment"] = logical_experiment

    return df


# =============================================================================
# LOAD FROZEN DESIGN
# =============================================================================

parameter_by_id = {
    x["id"]: x
    for x in PARAMETER_CONFIGS
}

model_by_id = {
    x["id"]: x
    for x in (
        SMALL_MODEL_PROTOCOLS
        + MEDIUM_MODEL_PROTOCOLS
    )
}

baseline_parameters = [
    parameter_by_id[param_id]
    for param_id in BASELINE_PARAMETER_IDS.values()
]


# =============================================================================
# LOAD MASTER EXECUTION MANIFEST
# =============================================================================

master_rows = read_jsonl(MASTER_MANIFEST)

assert len(master_rows) == 504, (
    f"Expected 504 physical runs, got {len(master_rows)}"
)

run_ids = [
    row["run_id"]
    for row in master_rows
]

assert len(run_ids) == len(set(run_ids)), (
    "Duplicate run IDs in master manifest"
)

master_by_run = {
    row["run_id"]: row
    for row in master_rows
}

master_by_key = {
    config_key(row): row
    for row in master_rows
}

assert len(master_by_key) == 504, (
    "Duplicate physical configuration keys in master manifest"
)


# =============================================================================
# VERIFY MANIFEST ITSELF
# =============================================================================

for row in master_rows:

    assert row["precision"] == "bf16"
    assert row["load_in_4bit"] is False
    assert row["lm_scoring_method"] == "single_token"
    assert row["n_samples"] == 500_000
    assert row["decimals"] == 3
    assert row["seed"] == 42
    assert row["mc_reliable_threshold"] == 1_000
    assert row["icl_n_examples"] == 5
    assert row["icl_seed"] == 0

    model = model_by_id[row["model_id"]]
    parameter = parameter_by_id[row["parameter_id"]]

    assert row["model_name"] == model["model_name"]
    assert row["prompt_protocol"] == model["protocol"]
    assert row["distribution"] == parameter["distribution"]
    assert row["params"] == parameter["params"]

    expected_prefixes = [
        normalize_prefix(x)
        for x in flatten_prefixes(parameter)
    ]

    manifest_prefixes = [
        normalize_prefix(x)
        for x in row["prefixes"]
    ]

    assert manifest_prefixes == expected_prefixes


# =============================================================================
# INDEPENDENTLY RECONSTRUCT EXPECTED LOGICAL GRIDS
# =============================================================================

prompt_keys = expected_keys(
    SMALL_MODEL_PROTOCOLS,
    baseline_parameters,
    ALL_PROMPT_TYPES,
)

parameter_keys = expected_keys(
    SMALL_MODEL_PROTOCOLS,
    PARAMETER_CONFIGS,
    ["plain"],
)

main_keys = expected_keys(
    SMALL_MODEL_PROTOCOLS + MEDIUM_MODEL_PROTOCOLS,
    baseline_parameters,
    MAIN_PROMPT_TYPES,
)

assert len(prompt_keys) == 330
assert len(parameter_keys) == 144
assert len(main_keys) == 120

assert len(set(prompt_keys)) == 330
assert len(set(parameter_keys)) == 144
assert len(set(main_keys)) == 120


# Every logical condition must map to a physical execution.
for key in prompt_keys + parameter_keys + main_keys:
    assert key in master_by_key, (
        f"Logical condition missing from physical execution registry: {key}"
    )


# =============================================================================
# CHECK MEMBERSHIP LABELS AGAINST THE INDEPENDENT LOGICAL GRIDS
# =============================================================================

expected_prompt_set = set(prompt_keys)
expected_parameter_set = set(parameter_keys)
expected_main_set = set(main_keys)

observed_prompt_set = {
    config_key(row)
    for row in master_rows
    if "prompt" in row["experiment_membership"]
}

observed_parameter_set = {
    config_key(row)
    for row in master_rows
    if "parameter" in row["experiment_membership"]
}

observed_main_set = {
    config_key(row)
    for row in master_rows
    if "main" in row["experiment_membership"]
}

assert observed_prompt_set == expected_prompt_set
assert observed_parameter_set == expected_parameter_set
assert observed_main_set == expected_main_set


membership_counts = Counter(
    frozenset(row["experiment_membership"])
    for row in master_rows
)

expected_membership_counts = {
    frozenset({"prompt"}): 270,
    frozenset({"main", "prompt"}): 30,
    frozenset({"main", "parameter", "prompt"}): 30,
    frozenset({"parameter"}): 114,
    frozenset({"main"}): 60,
}

assert membership_counts == expected_membership_counts, (
    f"Unexpected membership structure:\n{membership_counts}"
)


# =============================================================================
# FILE COMPLETENESS
# =============================================================================

expected_ids = set(run_ids)

token_files = {
    path.stem.removeprefix("token_level_"): path
    for path in OUTPUT_DIR.glob("token_level_v2_*.csv")
}

prefix_files = {
    path.stem.removeprefix("prefix_summary_"): path
    for path in OUTPUT_DIR.glob("prefix_summary_v2_*.csv")
}

assert set(token_files) == expected_ids, (
    "Token-level file set does not exactly match master manifest"
)

assert set(prefix_files) == expected_ids, (
    "Prefix-summary file set does not exactly match master manifest"
)


# =============================================================================
# VALIDATE ALL 504 PHYSICAL OUTPUTS
# =============================================================================

token_frames = {}
prefix_frames = {}

run_validation = []

worst = {
    "restricted_sum_error": 0.0,
    "analytic_sum_error": 0.0,
    "unconditional_sum_error": 0.0,
    "candidate_complement_error": 0.0,
    "reconstruction_error": 0.0,
    "prefix_token_mass_match_error": 0.0,
    "tvd_recompute_error": 0.0,
}

schema_reported = False


for i, manifest_row in enumerate(master_rows, start=1):

    run_id = manifest_row["run_id"]

    token_path = token_files[run_id]
    prefix_path = prefix_files[run_id]

    token_df = read_csv(token_path)
    prefix_df = read_csv(prefix_path)

    require_columns(
        token_df,
        ["run_id", "prefix"],
        token_path,
    )

    require_columns(
        prefix_df,
        ["run_id", "prefix"],
        prefix_path,
    )

    # -------------------------------------------------------------------------
    # Resolve probability columns once from actual output schema.
    # -------------------------------------------------------------------------

    lm_col = find_column(
        token_df,
        ["lm_prob"],
        "restricted LM probability",
        token_path,
    )

    analytic_col = find_column(
        token_df,
        [
            "analytic_prob",
            "truth_prob",
            "true_prob",
            "gt_prob",
        ],
        "analytic probability",
        token_path,
    )

    unconditional_col = find_column(
        token_df,
        ["lm_prob_unconditional"],
        "unconditional LM probability",
        token_path,
    )

    prefix_mass_col = find_column(
        prefix_df,
        ["valid_candidate_mass"],
        "valid candidate mass",
        prefix_path,
    )

    prefix_other_col = find_column(
        prefix_df,
        ["other_vocab_mass"],
        "other vocabulary mass",
        prefix_path,
    )

    tvd_col = find_column(
        prefix_df,
        [
            "tvd",
            "tvd_analytic",
            "total_variation_distance",
        ],
        "TVD",
        prefix_path,
    )

    if not schema_reported:
        print("=" * 80)
        print("DETECTED RESULT SCHEMA")
        print("=" * 80)
        print("restricted LM prob :", lm_col)
        print("analytic prob      :", analytic_col)
        print("unconditional prob :", unconditional_col)
        print("valid mass         :", prefix_mass_col)
        print("other-vocab mass   :", prefix_other_col)
        print("TVD                :", tvd_col)
        print()
        schema_reported = True

    # -------------------------------------------------------------------------
    # Run ID integrity
    # -------------------------------------------------------------------------

    token_run_ids = set(
        token_df["run_id"].astype(str).unique()
    )

    prefix_run_ids = set(
        prefix_df["run_id"].astype(str).unique()
    )

    assert token_run_ids == {run_id}, (
        f"{token_path}: run_id mismatch"
    )

    assert prefix_run_ids == {run_id}, (
        f"{prefix_path}: run_id mismatch"
    )

    # -------------------------------------------------------------------------
    # Metadata integrity where output stores metadata.
    # -------------------------------------------------------------------------

    metadata_checks = {
        "model_name": manifest_row["model_name"],
        "prompt_protocol": manifest_row["prompt_protocol"],
        "distribution": manifest_row["distribution"],
        "prompt_type": manifest_row["prompt_type"],
        "lm_scoring_method": manifest_row["lm_scoring_method"],
        "n_samples": manifest_row["n_samples"],
        "decimals": manifest_row["decimals"],
        "seed": manifest_row["seed"],
    }

    for col, expected in metadata_checks.items():
        assert_single_value(
            token_df,
            col,
            expected,
            token_path,
        )

        assert_single_value(
            prefix_df,
            col,
            expected,
            prefix_path,
        )

    # -------------------------------------------------------------------------
    # Prefix coverage
    # -------------------------------------------------------------------------

    expected_prefixes = [
        normalize_prefix(x)
        for x in manifest_row["prefixes"]
    ]

    prefix_df["_prefix_norm"] = (
        prefix_df["prefix"]
        .map(normalize_prefix)
    )

    token_df["_prefix_norm"] = (
        token_df["prefix"]
        .map(normalize_prefix)
    )

    actual_prefixes = list(
        prefix_df["_prefix_norm"]
    )

    assert len(actual_prefixes) == len(expected_prefixes), (
        f"{run_id}: prefix-row count mismatch. "
        f"expected={len(expected_prefixes)}, "
        f"observed={len(actual_prefixes)}"
    )

    assert not prefix_df["_prefix_norm"].duplicated().any(), (
        f"{run_id}: duplicate prefix-summary rows"
    )

    assert set(actual_prefixes) == set(expected_prefixes), (
        f"{run_id}: prefix-summary coverage mismatch\n"
        f"Missing: {sorted(set(expected_prefixes) - set(actual_prefixes))}\n"
        f"Extra: {sorted(set(actual_prefixes) - set(expected_prefixes))}"
    )

    assert (
        set(token_df["_prefix_norm"])
        == set(expected_prefixes)
    ), f"{run_id}: token-level prefix coverage mismatch"

    # -------------------------------------------------------------------------
    # Convert probability columns
    # -------------------------------------------------------------------------

    token_df[lm_col] = pd.to_numeric(
        token_df[lm_col],
        errors="raise",
    )

    token_df[analytic_col] = pd.to_numeric(
        token_df[analytic_col],
        errors="raise",
    )

    token_df[unconditional_col] = pd.to_numeric(
        token_df[unconditional_col],
        errors="raise",
    )

    prefix_df[prefix_mass_col] = pd.to_numeric(
        prefix_df[prefix_mass_col],
        errors="raise",
    )

    prefix_df[prefix_other_col] = pd.to_numeric(
        prefix_df[prefix_other_col],
        errors="raise",
    )

    prefix_df[tvd_col] = pd.to_numeric(
        prefix_df[tvd_col],
        errors="raise",
    )

    # -------------------------------------------------------------------------
    # Probability bounds
    # -------------------------------------------------------------------------

    for col in [
        lm_col,
        analytic_col,
        unconditional_col,
    ]:
        values = token_df[col].to_numpy(float)

        assert np.isfinite(values).all(), (
            f"{run_id}: non-finite values in {col}"
        )

        assert values.min() >= -TOL
        assert values.max() <= 1.0 + TOL

    for col in [
        prefix_mass_col,
        prefix_other_col,
        tvd_col,
    ]:
        values = prefix_df[col].to_numpy(float)

        assert np.isfinite(values).all(), (
            f"{run_id}: non-finite values in {col}"
        )

        assert values.min() >= -TOL
        assert values.max() <= 1.0 + TOL

    # -------------------------------------------------------------------------
    # Prefix → mass lookup
    # -------------------------------------------------------------------------

    mass_by_prefix = dict(
        zip(
            prefix_df["_prefix_norm"],
            prefix_df[prefix_mass_col],
        )
    )

    other_by_prefix = dict(
        zip(
            prefix_df["_prefix_norm"],
            prefix_df[prefix_other_col],
        )
    )

    tvd_by_prefix = dict(
        zip(
            prefix_df["_prefix_norm"],
            prefix_df[tvd_col],
        )
    )

    # Candidate complement:
    # valid_candidate_mass + other_vocab_mass = 1
    complement_error = float(
        np.max(
            np.abs(
                prefix_df[prefix_mass_col].to_numpy(float)
                + prefix_df[prefix_other_col].to_numpy(float)
                - 1.0
            )
        )
    )

    worst["candidate_complement_error"] = max(
        worst["candidate_complement_error"],
        complement_error,
    )

    # -------------------------------------------------------------------------
    # Token-level probability invariants, prefix by prefix
    # -------------------------------------------------------------------------

    for prefix, group in token_df.groupby(
        "_prefix_norm",
        sort=False,
    ):

        lm = group[lm_col].to_numpy(float)
        analytic = group[analytic_col].to_numpy(float)
        unconditional = group[unconditional_col].to_numpy(float)

        valid_mass = float(
            mass_by_prefix[prefix]
        )

        # Restricted LM distribution sums to 1.
        restricted_error = abs(
            float(lm.sum()) - 1.0
        )

        # Analytic distribution sums to 1.
        analytic_error = abs(
            float(analytic.sum()) - 1.0
        )

        # Unconditional candidate probabilities sum to valid mass.
        unconditional_sum_error = abs(
            float(unconditional.sum())
            - valid_mass
        )

        # Prefix/token representation of valid candidate mass agrees.
        prefix_token_mass_error = abs(
            float(unconditional.sum())
            - valid_mass
        )

        # Restricted probabilities should reconstruct from full-vocab mass.
        if valid_mass > TOL:
            reconstructed = (
                unconditional / valid_mass
            )

            reconstruction_error = float(
                np.max(
                    np.abs(
                        reconstructed - lm
                    )
                )
            )
        else:
            reconstruction_error = 0.0

        # Recompute TVD from token-level distributions.
        recomputed_tvd = 0.5 * float(
            np.abs(
                lm - analytic
            ).sum()
        )

        tvd_error = abs(
            recomputed_tvd
            - float(tvd_by_prefix[prefix])
        )

        worst["restricted_sum_error"] = max(
            worst["restricted_sum_error"],
            restricted_error,
        )

        worst["analytic_sum_error"] = max(
            worst["analytic_sum_error"],
            analytic_error,
        )

        worst["unconditional_sum_error"] = max(
            worst["unconditional_sum_error"],
            unconditional_sum_error,
        )

        worst["prefix_token_mass_match_error"] = max(
            worst["prefix_token_mass_match_error"],
            prefix_token_mass_error,
        )

        worst["reconstruction_error"] = max(
            worst["reconstruction_error"],
            reconstruction_error,
        )

        worst["tvd_recompute_error"] = max(
            worst["tvd_recompute_error"],
            tvd_error,
        )

    # -------------------------------------------------------------------------
    # Store validated frames
    # -------------------------------------------------------------------------

    token_frames[run_id] = token_df.drop(
        columns=["_prefix_norm"]
    )

    prefix_frames[run_id] = prefix_df.drop(
        columns=["_prefix_norm"]
    )

    run_validation.append({
        "run_id": run_id,
        "model_id": manifest_row["model_id"],
        "parameter_id": manifest_row["parameter_id"],
        "prompt_protocol": manifest_row["prompt_protocol"],
        "prompt_type": manifest_row["prompt_type"],
        "memberships": "|".join(
            sorted(manifest_row["experiment_membership"])
        ),
        "prefix_rows": len(prefix_df),
        "token_rows": len(token_df),
    })

    if i % 50 == 0 or i == len(master_rows):
        print(
            f"Validated {i:>3} / "
            f"{len(master_rows)} runs"
        )


# =============================================================================
# FINAL NUMERICAL INVARIANT CHECK
# =============================================================================

print()
print("=" * 80)
print("FINAL V2 NUMERICAL INVARIANTS")
print("=" * 80)

for key, value in worst.items():
    print(f"{key:35s}: {value:.3e}")

for key, value in worst.items():
    assert value <= TOL, (
        f"Invariant failed: {key}={value:.6e} > {TOL}"
    )


# =============================================================================
# PHYSICAL PREFIX ACCOUNTING
# =============================================================================

physical_prefix_rows = sum(
    len(frame)
    for frame in prefix_frames.values()
)

expected_physical_prefix_rows = sum(
    len(row["prefixes"])
    for row in master_rows
)

assert physical_prefix_rows == expected_physical_prefix_rows
assert physical_prefix_rows == 6942


# =============================================================================
# MAP LOGICAL CONDITIONS → PHYSICAL RUN IDS
# =============================================================================

def run_ids_for_keys(keys):
    return [
        master_by_key[key]["run_id"]
        for key in keys
    ]


prompt_run_ids = run_ids_for_keys(prompt_keys)
parameter_run_ids = run_ids_for_keys(parameter_keys)
main_run_ids = run_ids_for_keys(main_keys)

assert len(prompt_run_ids) == 330
assert len(parameter_run_ids) == 144
assert len(main_run_ids) == 120

assert len(set(prompt_run_ids)) == 330
assert len(set(parameter_run_ids)) == 144
assert len(set(main_run_ids)) == 120


# =============================================================================
# BUILD LOGICAL MERGES
# =============================================================================

def build_merge(run_ids, frames, logical_experiment):
    blocks = []

    for run_id in run_ids:
        manifest_row = master_by_run[run_id]

        blocks.append(
            add_manifest_metadata(
                frames[run_id],
                manifest_row,
                logical_experiment,
            )
        )

    return pd.concat(
        blocks,
        ignore_index=True,
    )


prompt_prefix = build_merge(
    prompt_run_ids,
    prefix_frames,
    "prompt",
)

prompt_token = build_merge(
    prompt_run_ids,
    token_frames,
    "prompt",
)

parameter_prefix = build_merge(
    parameter_run_ids,
    prefix_frames,
    "parameter",
)

parameter_token = build_merge(
    parameter_run_ids,
    token_frames,
    "parameter",
)

main_prefix = build_merge(
    main_run_ids,
    prefix_frames,
    "main",
)

main_token = build_merge(
    main_run_ids,
    token_frames,
    "main",
)


# =============================================================================
# LOGICAL DATASET VALIDATION
# =============================================================================

assert prompt_prefix["run_id"].nunique() == 330
assert prompt_token["run_id"].nunique() == 330

assert parameter_prefix["run_id"].nunique() == 144
assert parameter_token["run_id"].nunique() == 144

assert main_prefix["run_id"].nunique() == 120
assert main_token["run_id"].nunique() == 120


# Exact prefix-row counts from frozen design.
assert len(prompt_prefix) == 4422
assert len(parameter_prefix) == 2118
assert len(main_prefix) == 1608


# Physical overlap is intentional.
assert len(
    set(prompt_run_ids)
    & set(parameter_run_ids)
    & set(main_run_ids)
) == 30

assert len(
    (
        set(prompt_run_ids)
        & set(main_run_ids)
    )
    - set(parameter_run_ids)
) == 30


# =============================================================================
# WRITE ONLY AFTER ALL VALIDATION PASSES
# =============================================================================

MERGED_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

VALIDATION_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


prompt_prefix.to_csv(
    MERGED_DIR
    / "prefix_summary_prompt_sweep_v2_merged.csv",
    index=False,
)

prompt_token.to_csv(
    MERGED_DIR
    / "token_level_prompt_sweep_v2_merged.csv",
    index=False,
)

parameter_prefix.to_csv(
    MERGED_DIR
    / "prefix_summary_parameter_sweep_v2_merged.csv",
    index=False,
)

parameter_token.to_csv(
    MERGED_DIR
    / "token_level_parameter_sweep_v2_merged.csv",
    index=False,
)

main_prefix.to_csv(
    MERGED_DIR
    / "prefix_summary_main_grid_v2_merged.csv",
    index=False,
)

main_token.to_csv(
    MERGED_DIR
    / "token_level_main_grid_v2_merged.csv",
    index=False,
)


pd.DataFrame(
    run_validation
).to_csv(
    VALIDATION_DIR
    / "final_v2_run_validation.csv",
    index=False,
)


validation_summary = {
    "physical_execution": {
        "expected_runs": 504,
        "validated_runs": len(master_rows),
        "prefix_rows": physical_prefix_rows,
        "token_rows": int(
            sum(
                len(frame)
                for frame in token_frames.values()
            )
        ),
    },

    "logical_grids": {
        "prompt": {
            "runs": int(
                prompt_prefix["run_id"].nunique()
            ),
            "prefix_rows": len(prompt_prefix),
            "token_rows": len(prompt_token),
        },
        "parameter": {
            "runs": int(
                parameter_prefix["run_id"].nunique()
            ),
            "prefix_rows": len(parameter_prefix),
            "token_rows": len(parameter_token),
        },
        "main": {
            "runs": int(
                main_prefix["run_id"].nunique()
            ),
            "prefix_rows": len(main_prefix),
            "token_rows": len(main_token),
        },
    },

    "overlap": {
        "all_three_runs": 30,
        "main_prompt_only_runs": 30,
        "physical_runs_avoided": 90,
    },

    "precision": {
        "bf16_runs": 504,
        "four_bit_runs": 0,
    },

    "worst_invariant_errors": {
        key: float(value)
        for key, value in worst.items()
    },

    "tolerance": TOL,
}


with (
    VALIDATION_DIR
    / "final_v2_results_validation.json"
).open(
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        validation_summary,
        f,
        indent=2,
        sort_keys=True,
    )


# =============================================================================
# REPORT
# =============================================================================

print()
print("=" * 80)
print("FINAL V2 VALIDATION / MERGE")
print("=" * 80)

print()
print("PHYSICAL EXECUTION")
print(f"  Runs validated : {len(master_rows)} / 504")
print(f"  Prefix rows    : {physical_prefix_rows}")
print(
    "  Token rows     : "
    f"{sum(len(x) for x in token_frames.values())}"
)

print()
print("LOGICAL DATASETS")
print(
    f"  Prompt    : "
    f"{prompt_prefix['run_id'].nunique():>3} runs | "
    f"{len(prompt_prefix):>5} prefix rows | "
    f"{len(prompt_token):>6} token rows"
)
print(
    f"  Parameter : "
    f"{parameter_prefix['run_id'].nunique():>3} runs | "
    f"{len(parameter_prefix):>5} prefix rows | "
    f"{len(parameter_token):>6} token rows"
)
print(
    f"  Main      : "
    f"{main_prefix['run_id'].nunique():>3} runs | "
    f"{len(main_prefix):>5} prefix rows | "
    f"{len(main_token):>6} token rows"
)

print()
print("OVERLAP")
print("  Shared main + parameter + prompt : 30")
print("  Shared main + prompt only        : 30")
print("  Redundant GPU executions avoided : 90")

print()
print("OUTPUT FILES")
print(
    "  merged/"
    "prefix_summary_prompt_sweep_v2_merged.csv"
)
print(
    "  merged/"
    "token_level_prompt_sweep_v2_merged.csv"
)
print(
    "  merged/"
    "prefix_summary_parameter_sweep_v2_merged.csv"
)
print(
    "  merged/"
    "token_level_parameter_sweep_v2_merged.csv"
)
print(
    "  merged/"
    "prefix_summary_main_grid_v2_merged.csv"
)
print(
    "  merged/"
    "token_level_main_grid_v2_merged.csv"
)

print()
print("VALIDATION FILES")
print(
    "  validation/"
    "final_v2_run_validation.csv"
)
print(
    "  validation/"
    "final_v2_results_validation.json"
)

print()
print("=" * 80)
print("ALL FINAL-V2 RESULT CHECKS PASSED")
print("=" * 80)