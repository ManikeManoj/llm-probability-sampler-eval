#!/usr/bin/env python3

import csv
import json
from collections import Counter
from pathlib import Path

from final_v2_spec import (
    DECIMALS,
    N_SAMPLES,
    MC_SEED,
    MC_RELIABLE_THRESHOLD,
    ICL_N_EXAMPLES,
    ICL_SEED,
    LM_SCORING_METHOD,
    SMALL_MODEL_PROTOCOLS,
    MEDIUM_MODEL_PROTOCOLS,
    ALL_PROMPT_TYPES,
    MAIN_PROMPT_TYPES,
    BASELINE_PARAMETER_IDS,
    PARAMETER_CONFIGS,
)


MANIFEST_DIR = Path("manifests")

PROMPT_OUT = MANIFEST_DIR / "prompt_sweep_v2.jsonl"
PARAMETER_OUT = MANIFEST_DIR / "parameter_sweep_v2.jsonl"
MAIN_OUT = MANIFEST_DIR / "main_grid_v2.jsonl"

ALL_OUT = MANIFEST_DIR / "final_v2_execution_all.jsonl"
MEMBERSHIP_OUT = MANIFEST_DIR / "final_v2_run_membership.csv"


PROMPT_TAGS = {
    "short": "short",
    "plain": "plain",
    "formal": "formal",
    "explanatory_1": "exp1",
    "explanatory_2": "exp2",
    "explanatory_3": "exp3",
    "explanatory_4": "exp4",
    "cot": "cot",
    "icl": "icl",
    "icl_random": "iclrnd",
    "icl_cot": "iclcot",
}


ICL_PROMPTS = {
    "icl",
    "icl_random",
    "icl_cot",
}


def flatten_prefixes(config):
    out = []

    for prefixes in config["prefixes"].values():
        out.extend(prefixes)

    return out


def protocol_tag(protocol):
    return {
        "raw_direct": "raw",
        "chat_direct": "chat",
    }[protocol]


def make_run_id(
    model,
    parameter,
    prompt_type,
):
    prompt_tag = PROMPT_TAGS[prompt_type]

    if prompt_type in ICL_PROMPTS:
        prompt_tag = (
            f"{prompt_tag}_"
            f"n{ICL_N_EXAMPLES}_"
            f"s{ICL_SEED}"
        )

    return (
        f"v2_"
        f"{model['id']}_"
        f"{parameter['id']}_"
        f"{protocol_tag(model['protocol'])}_"
        f"{prompt_tag}_"
        f"bf16"
    )


def make_row(
    model,
    parameter,
    prompt_type,
    memberships,
):
    return {
        "experiment": "final_v2",

        "experiment_membership": sorted(
            memberships
        ),

        "run_id": make_run_id(
            model,
            parameter,
            prompt_type,
        ),

        "model_id": model["id"],
        "model_name": model["model_name"],
        "prompt_protocol": model["protocol"],

        "parameter_id": parameter["id"],
        "distribution": parameter["distribution"],
        "params": parameter["params"],
        "prefixes": flatten_prefixes(
            parameter
        ),

        "prompt_type": prompt_type,

        "precision": "bf16",
        "load_in_4bit": False,

        "lm_scoring_method":
            LM_SCORING_METHOD,

        "n_samples": N_SAMPLES,
        "decimals": DECIMALS,
        "seed": MC_SEED,
        "mc_reliable_threshold":
            MC_RELIABLE_THRESHOLD,

        "icl_n_examples":
            ICL_N_EXAMPLES,

        "icl_seed":
            ICL_SEED,
    }


def config_key(row):
    return (
        row["model_id"],
        row["parameter_id"],
        row["prompt_protocol"],
        row["prompt_type"],
        row["precision"],
    )


def write_jsonl(path, rows):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as f:
        for row in rows:
            f.write(
                json.dumps(
                    row,
                    sort_keys=True,
                )
                + "\n"
            )


def expected_logical_keys(
    models,
    parameters,
    prompts,
):
    keys = set()

    for model in models:
        for parameter in parameters:
            for prompt in prompts:
                keys.add(
                    (
                        model["id"],
                        parameter["id"],
                        model["protocol"],
                        prompt,
                        "bf16",
                    )
                )

    return keys


def validate_common(
    rows,
    parameter_by_id,
    model_by_id,
):
    for row in rows:

        model = model_by_id[
            row["model_id"]
        ]

        parameter = parameter_by_id[
            row["parameter_id"]
        ]

        assert (
            row["model_name"]
            == model["model_name"]
        )

        assert (
            row["prompt_protocol"]
            == model["protocol"]
        )

        assert (
            row["distribution"]
            == parameter["distribution"]
        )

        assert (
            row["params"]
            == parameter["params"]
        )

        assert (
            row["prefixes"]
            == flatten_prefixes(parameter)
        )

        assert row["precision"] == "bf16"
        assert row["load_in_4bit"] is False

        assert (
            row["lm_scoring_method"]
            == "single_token"
        )

        assert row["n_samples"] == 500_000
        assert row["decimals"] == 3
        assert row["seed"] == 42

        assert (
            row["mc_reliable_threshold"]
            == 1_000
        )

        assert (
            row["icl_n_examples"]
            == 5
        )

        assert row["icl_seed"] == 0


def main():

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
        for param_id
        in BASELINE_PARAMETER_IDS.values()
    ]

    baseline_ids = {
        p["id"]
        for p in baseline_parameters
    }

    # ==============================================================
    # 1. PROMPT SWEEP
    #
    # This becomes the canonical execution source for all
    # small-model baseline conditions.
    # ==============================================================

    prompt_rows = []

    for model in SMALL_MODEL_PROTOCOLS:
        for parameter in baseline_parameters:
            for prompt_type in ALL_PROMPT_TYPES:

                memberships = {
                    "prompt"
                }

                # The small-model portion of the main grid
                # already exists here.
                if prompt_type in MAIN_PROMPT_TYPES:
                    memberships.add(
                        "main"
                    )

                # Baseline/plain conditions are also part of
                # the parameter grid.
                if prompt_type == "plain":
                    memberships.add(
                        "parameter"
                    )

                prompt_rows.append(
                    make_row(
                        model,
                        parameter,
                        prompt_type,
                        memberships,
                    )
                )

    # ==============================================================
    # 2. PARAMETER SWEEP — ADDITIONAL RUNS ONLY
    #
    # Baseline/plain rows already exist in prompt_rows.
    # There are 19 non-baseline parameter settings:
    #
    # 24 total - 5 baseline = 19
    #
    # 19 × 6 = 114 executions.
    # ==============================================================

    parameter_rows = []

    nonbaseline_parameters = [
        p
        for p in PARAMETER_CONFIGS
        if p["id"] not in baseline_ids
    ]

    assert len(nonbaseline_parameters) == 19

    for model in SMALL_MODEL_PROTOCOLS:
        for parameter in nonbaseline_parameters:

            parameter_rows.append(
                make_row(
                    model,
                    parameter,
                    "plain",
                    {"parameter"},
                )
            )

    # ==============================================================
    # 3. MAIN GRID — MEDIUM RUNS ONLY
    #
    # All 60 small-model main configurations already exist
    # inside the prompt sweep.
    #
    # Medium:
    # 6 model/protocol × 5 baseline × 2 prompts
    # = 60 new executions.
    # ==============================================================

    main_rows = []

    for model in MEDIUM_MODEL_PROTOCOLS:
        for parameter in baseline_parameters:
            for prompt_type in MAIN_PROMPT_TYPES:

                main_rows.append(
                    make_row(
                        model,
                        parameter,
                        prompt_type,
                        {"main"},
                    )
                )

    # ==============================================================
    # BASIC COUNTS
    # ==============================================================

    assert len(prompt_rows) == 330
    assert len(parameter_rows) == 114
    assert len(main_rows) == 60

    all_rows = (
        prompt_rows
        + parameter_rows
        + main_rows
    )

    assert len(all_rows) == 504

    # ==============================================================
    # NO DUPLICATE EXECUTIONS
    # ==============================================================

    all_run_ids = [
        r["run_id"]
        for r in all_rows
    ]

    assert (
        len(all_run_ids)
        == len(set(all_run_ids))
    ), "Duplicate run IDs"

    all_keys = [
        config_key(r)
        for r in all_rows
    ]

    assert (
        len(all_keys)
        == len(set(all_keys))
    ), "Duplicate execution configurations"

    # ==============================================================
    # COMMON VALIDATION
    # ==============================================================

    validate_common(
        all_rows,
        parameter_by_id,
        model_by_id,
    )

    # ==============================================================
    # PROMPT EXECUTION BALANCE
    # ==============================================================

    assert Counter(
        r["model_id"]
        for r in prompt_rows
    ) == {
        x["id"]: 55
        for x in SMALL_MODEL_PROTOCOLS
    }

    assert Counter(
        r["parameter_id"]
        for r in prompt_rows
    ) == {
        parameter_id: 66
        for parameter_id
        in BASELINE_PARAMETER_IDS.values()
    }

    assert Counter(
        r["prompt_type"]
        for r in prompt_rows
    ) == {
        prompt: 30
        for prompt in ALL_PROMPT_TYPES
    }

    assert Counter(
        r["prompt_protocol"]
        for r in prompt_rows
    ) == {
        "raw_direct": 220,
        "chat_direct": 110,
    }

    # ==============================================================
    # PARAMETER EXECUTION BALANCE
    # ==============================================================

    assert Counter(
        r["model_id"]
        for r in parameter_rows
    ) == {
        x["id"]: 19
        for x in SMALL_MODEL_PROTOCOLS
    }

    assert Counter(
        r["parameter_id"]
        for r in parameter_rows
    ) == {
        p["id"]: 6
        for p in nonbaseline_parameters
    }

    assert Counter(
        r["prompt_protocol"]
        for r in parameter_rows
    ) == {
        "raw_direct": 76,
        "chat_direct": 38,
    }

    assert set(
        r["prompt_type"]
        for r in parameter_rows
    ) == {"plain"}

    # ==============================================================
    # MAIN EXECUTION BALANCE
    # ==============================================================

    assert Counter(
        r["model_id"]
        for r in main_rows
    ) == {
        x["id"]: 10
        for x in MEDIUM_MODEL_PROTOCOLS
    }

    assert Counter(
        r["prompt_type"]
        for r in main_rows
    ) == {
        "plain": 30,
        "explanatory_4": 30,
    }

    assert Counter(
        r["prompt_protocol"]
        for r in main_rows
    ) == {
        "raw_direct": 40,
        "chat_direct": 20,
    }

    # ==============================================================
    # LOGICAL GRID RECONSTRUCTION
    #
    # This is the important validation:
    # after reuse, do we STILL have exactly the locked
    # 120 / 144 / 330 scientific grids?
    # ==============================================================

    observed_main = {
        config_key(r)
        for r in all_rows
        if "main"
        in r["experiment_membership"]
    }

    observed_parameter = {
        config_key(r)
        for r in all_rows
        if "parameter"
        in r["experiment_membership"]
    }

    observed_prompt = {
        config_key(r)
        for r in all_rows
        if "prompt"
        in r["experiment_membership"]
    }

    expected_main = expected_logical_keys(
        SMALL_MODEL_PROTOCOLS
        + MEDIUM_MODEL_PROTOCOLS,
        baseline_parameters,
        MAIN_PROMPT_TYPES,
    )

    expected_parameter = expected_logical_keys(
        SMALL_MODEL_PROTOCOLS,
        PARAMETER_CONFIGS,
        ["plain"],
    )

    expected_prompt = expected_logical_keys(
        SMALL_MODEL_PROTOCOLS,
        baseline_parameters,
        ALL_PROMPT_TYPES,
    )

    assert len(expected_main) == 120
    assert len(expected_parameter) == 144
    assert len(expected_prompt) == 330

    assert observed_main == expected_main
    assert observed_parameter == expected_parameter
    assert observed_prompt == expected_prompt

    # ==============================================================
    # PREFIX-ROW EXPECTATIONS
    # ==============================================================

    baseline_prefixes = sum(
        len(flatten_prefixes(p))
        for p in baseline_parameters
    )

    all_parameter_prefixes = sum(
        len(flatten_prefixes(p))
        for p in PARAMETER_CONFIGS
    )

    nonbaseline_prefixes = sum(
        len(flatten_prefixes(p))
        for p in nonbaseline_parameters
    )

    assert baseline_prefixes == 67
    assert all_parameter_prefixes == 353
    assert nonbaseline_prefixes == 286

    expected_prompt_prefix_rows = (
        67 * 6 * 11
    )

    expected_parameter_exec_prefix_rows = (
        286 * 6
    )

    expected_main_exec_prefix_rows = (
        67 * 6 * 2
    )

    assert expected_prompt_prefix_rows == 4422
    assert expected_parameter_exec_prefix_rows == 1716
    assert expected_main_exec_prefix_rows == 804

    # ==============================================================
    # WRITE
    # ==============================================================

    MANIFEST_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_jsonl(
        PROMPT_OUT,
        prompt_rows,
    )

    write_jsonl(
        PARAMETER_OUT,
        parameter_rows,
    )

    write_jsonl(
        MAIN_OUT,
        main_rows,
    )

    write_jsonl(
        ALL_OUT,
        all_rows,
    )

    with MEMBERSHIP_OUT.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "run_id",
                "model_id",
                "parameter_id",
                "prompt_protocol",
                "prompt_type",
                "memberships",
            ],
        )

        writer.writeheader()

        for row in all_rows:
            writer.writerow({
                "run_id":
                    row["run_id"],

                "model_id":
                    row["model_id"],

                "parameter_id":
                    row["parameter_id"],

                "prompt_protocol":
                    row["prompt_protocol"],

                "prompt_type":
                    row["prompt_type"],

                "memberships":
                    "|".join(
                        row[
                            "experiment_membership"
                        ]
                    ),
            })

    # ==============================================================
    # REPORT
    # ==============================================================

    print("=" * 80)
    print("FINAL V2 MANIFEST GENERATION")
    print("=" * 80)

    print()
    print("LOCKED LOGICAL GRIDS")
    print("  Main grid:       120")
    print("  Parameter sweep: 144")
    print("  Prompt sweep:    330")
    print("  Logical total:   594")

    print()
    print("DEDUPLICATED EXECUTION")
    print(
        f"  Prompt jobs:            "
        f"{len(prompt_rows)}"
    )
    print(
        f"  Parameter extra jobs:   "
        f"{len(parameter_rows)}"
    )
    print(
        f"  Main medium jobs:       "
        f"{len(main_rows)}"
    )
    print(
        f"  Unique GPU jobs:        "
        f"{len(all_rows)}"
    )
    print(
        f"  Redundant jobs avoided: "
        f"{594 - len(all_rows)}"
    )

    print()
    print("EXECUTION PREFIX ROWS")
    print(
        f"  Prompt:    "
        f"{expected_prompt_prefix_rows}"
    )
    print(
        f"  Parameter: "
        f"{expected_parameter_exec_prefix_rows}"
    )
    print(
        f"  Main:      "
        f"{expected_main_exec_prefix_rows}"
    )

    print()
    print("LOGICAL RECONSTRUCTION")
    print(
        f"  Main recovered:      "
        f"{len(observed_main)} / 120"
    )
    print(
        f"  Parameter recovered: "
        f"{len(observed_parameter)} / 144"
    )
    print(
        f"  Prompt recovered:    "
        f"{len(observed_prompt)} / 330"
    )

    print()
    print("PRECISION")
    print("  BF16: 504 / 504")
    print("  4-bit: 0")

    print()
    print("FILES")
    print(f"  {PROMPT_OUT}")
    print(f"  {PARAMETER_OUT}")
    print(f"  {MAIN_OUT}")
    print(f"  {ALL_OUT}")
    print(f"  {MEMBERSHIP_OUT}")

    print()
    print("=" * 80)
    print("ALL FINAL-V2 MANIFEST CHECKS PASSED")
    print("=" * 80)


if __name__ == "__main__":
    main()