#!/usr/bin/env python3

import json
from collections import Counter, defaultdict
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
    BASELINE_PARAMETER_IDS,
    PARAMETER_CONFIGS,
)

OUT = Path("manifests/precision_v2.jsonl")

PRECISIONS = [
    ("4bit", True),
    ("bf16", False),
]


def flatten_prefixes(config):
    out = []

    for prefixes in config["prefixes"].values():
        out.extend(prefixes)

    return out


def protocol_short(protocol):
    return {
        "raw_direct": "raw",
        "chat_direct": "chat",
    }[protocol]


def main():
    by_id = {
        cfg["id"]: cfg
        for cfg in PARAMETER_CONFIGS
    }

    rows = []

    for model in SMALL_MODEL_PROTOCOLS:
        for distribution, parameter_id in BASELINE_PARAMETER_IDS.items():
            parameter = by_id[parameter_id]

            assert parameter["distribution"] == distribution

            prefixes = flatten_prefixes(parameter)

            for precision, load_in_4bit in PRECISIONS:

                run_id = (
                    f"precision_v2_"
                    f"{model['id']}_"
                    f"{parameter_id}_"
                    f"{protocol_short(model['protocol'])}_"
                    f"{precision}"
                )

                rows.append({
                    "experiment": "precision_v2",

                    "run_id": run_id,

                    "model_id": model["id"],
                    "model_name": model["model_name"],
                    "prompt_protocol": model["protocol"],

                    "parameter_id": parameter_id,
                    "distribution": distribution,
                    "params": parameter["params"],
                    "prefixes": prefixes,

                    "prompt_type": "plain",

                    "precision": precision,
                    "load_in_4bit": load_in_4bit,

                    "lm_scoring_method": LM_SCORING_METHOD,

                    "n_samples": N_SAMPLES,
                    "decimals": DECIMALS,
                    "seed": MC_SEED,
                    "mc_reliable_threshold": MC_RELIABLE_THRESHOLD,

                    "icl_n_examples": ICL_N_EXAMPLES,
                    "icl_seed": ICL_SEED,
                })

    # ==============================================================
    # HARD VALIDATION
    # ==============================================================

    assert len(rows) == 60, len(rows)

    run_ids = [r["run_id"] for r in rows]
    assert len(run_ids) == len(set(run_ids)), "Duplicate run IDs"

    precision_counts = Counter(
        r["precision"] for r in rows
    )

    assert precision_counts == {
        "4bit": 30,
        "bf16": 30,
    }, precision_counts

    model_counts = Counter(
        r["model_id"] for r in rows
    )

    assert model_counts == {
        model["id"]: 10
        for model in SMALL_MODEL_PROTOCOLS
    }, model_counts

    parameter_counts = Counter(
        r["parameter_id"] for r in rows
    )

    expected_parameter_counts = {
        parameter_id: 12
        for parameter_id in BASELINE_PARAMETER_IDS.values()
    }

    assert parameter_counts == expected_parameter_counts, parameter_counts

    protocol_counts = Counter(
        r["prompt_protocol"] for r in rows
    )

    assert protocol_counts == {
        "raw_direct": 40,
        "chat_direct": 20,
    }, protocol_counts

    # Every model × baseline pair must have exactly one 4bit + one BF16 run.
    paired = defaultdict(set)

    for row in rows:
        key = (
            row["model_id"],
            row["parameter_id"],
        )

        paired[key].add(row["precision"])

    assert len(paired) == 30

    for key, precisions in paired.items():
        assert precisions == {"4bit", "bf16"}, (
            key,
            precisions,
        )

    # Every row must exactly agree with frozen specification.
    model_by_id = {
        x["id"]: x
        for x in SMALL_MODEL_PROTOCOLS
    }

    for row in rows:
        model = model_by_id[row["model_id"]]
        parameter = by_id[row["parameter_id"]]

        assert row["model_name"] == model["model_name"]
        assert row["prompt_protocol"] == model["protocol"]

        assert row["distribution"] == parameter["distribution"]
        assert row["params"] == parameter["params"]
        assert row["prefixes"] == flatten_prefixes(parameter)

        assert row["prompt_type"] == "plain"
        assert row["lm_scoring_method"] == "single_token"

        assert row["n_samples"] == 500_000
        assert row["decimals"] == 3
        assert row["seed"] == 42
        assert row["mc_reliable_threshold"] == 1_000

        if row["precision"] == "4bit":
            assert row["load_in_4bit"] is True

        elif row["precision"] == "bf16":
            assert row["load_in_4bit"] is False

        else:
            raise AssertionError(row["precision"])

    # Expected prefix-summary rows after merging.
    baseline_prefix_count = sum(
        len(flatten_prefixes(by_id[parameter_id]))
        for parameter_id in BASELINE_PARAMETER_IDS.values()
    )

    expected_prefix_rows = (
        baseline_prefix_count
        * len(SMALL_MODEL_PROTOCOLS)
        * len(PRECISIONS)
    )

    # Current locked baseline:
    # N1=18, U1=9, E1=13, B2=9, L1=18
    # total=67 per model/protocol/precision
    assert baseline_prefix_count == 67
    assert expected_prefix_rows == 804

    # ==============================================================
    # WRITE
    # ==============================================================

    OUT.parent.mkdir(parents=True, exist_ok=True)

    with OUT.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(
                json.dumps(
                    row,
                    sort_keys=True,
                )
                + "\n"
            )

    print("=" * 80)
    print("PRECISION V2 MANIFEST VALIDATION PASSED")
    print("=" * 80)

    print(f"Manifest:                {OUT}")
    print(f"Runs:                    {len(rows)}")
    print(f"Unique run IDs:          {len(set(run_ids))}")
    print(f"4-bit:                   {precision_counts['4bit']}")
    print(f"BF16:                    {precision_counts['bf16']}")
    print(f"Raw protocol:            {protocol_counts['raw_direct']}")
    print(f"Chat protocol:           {protocol_counts['chat_direct']}")
    print(f"Model × baseline pairs:  {len(paired)}")
    print(f"Prefixes / full block:   {baseline_prefix_count}")
    print(f"Expected merged prefixes:{expected_prefix_rows}")

    print()
    print("MODEL COUNTS")
    for key in sorted(model_counts):
        print(f"  {key}: {model_counts[key]}")

    print()
    print("BASELINE COUNTS")
    for key in sorted(parameter_counts):
        print(f"  {key}: {parameter_counts[key]}")

    print()
    print("PRECISION MANIFEST: PASS")


if __name__ == "__main__":
    main()