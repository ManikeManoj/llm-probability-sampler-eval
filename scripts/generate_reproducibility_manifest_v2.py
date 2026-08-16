#!/usr/bin/env python3

import json
from pathlib import Path


MASTER = Path("manifests/final_v2_execution_all.jsonl")
OUT = Path("manifests/reproducibility_v2.jsonl")


# ---------------------------------------------------------------------
# Ten representative FINAL-v2 configurations.
#
# Each receives TWO new independent executions.
# Together with its existing final-v2 run, this gives n=3 executions
# per configuration.
# ---------------------------------------------------------------------

SELECTIONS = [
    {
        "repro_id": "R01",
        "model_id": "M1",
        "parameter_id": "N1",
        "prompt_type": "plain",
    },
    {
        "repro_id": "R02",
        "model_id": "M2",
        "parameter_id": "N1",
        "prompt_type": "plain",
    },
    {
        "repro_id": "R03",
        "model_id": "M3",
        "parameter_id": "N1",
        "prompt_type": "plain",
    },
    {
        "repro_id": "R04",
        "model_id": "M4",
        "parameter_id": "B2",
        "prompt_type": "plain",
    },
    {
        "repro_id": "R05",
        "model_id": "M5",
        "parameter_id": "B2",
        "prompt_type": "plain",
    },
    {
        "repro_id": "R06",
        "model_id": "M6",
        "parameter_id": "B2",
        "prompt_type": "plain",
    },
    {
        "repro_id": "R07",
        "model_id": "S1",
        "parameter_id": "E4",
        "prompt_type": "plain",
    },
    {
        "repro_id": "R08",
        "model_id": "S3",
        "parameter_id": "U1",
        "prompt_type": "icl_random",
    },
    {
        "repro_id": "R09",
        "model_id": "S4",
        "parameter_id": "L4",
        "prompt_type": "plain",
    },
    {
        "repro_id": "R10",
        "model_id": "S6",
        "parameter_id": "N1",
        "prompt_type": "explanatory_4",
    },
]


def read_jsonl(path):
    rows = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    return rows


master = read_jsonl(MASTER)

assert len(master) == 504


def matches(row, spec):
    return (
        row["model_id"] == spec["model_id"]
        and row["parameter_id"] == spec["parameter_id"]
        and row["prompt_type"] == spec["prompt_type"]
    )


output_rows = []


for spec in SELECTIONS:

    matches_found = [
        row
        for row in master
        if matches(row, spec)
    ]

    assert len(matches_found) == 1, (
        f"{spec['repro_id']}: expected exactly one "
        f"matching final-v2 run, got {len(matches_found)}"
    )

    original = matches_found[0]

    for replicate_id in (1, 2):

        row = dict(original)

        row["experiment"] = "reproducibility_v2"

        row["run_id"] = (
            f"repro_v2_"
            f"{spec['repro_id']}_"
            f"rep{replicate_id}"
        )

        # These extra fields make pairing unambiguous later.
        row["repro_id"] = spec["repro_id"]
        row["replicate_id"] = replicate_id
        row["origin_run_id"] = original["run_id"]

        # Reproducibility requires IDENTICAL inputs.
        assert row["precision"] == "bf16"
        assert row["load_in_4bit"] is False
        assert row["lm_scoring_method"] == "single_token"
        assert row["n_samples"] == 500_000
        assert row["decimals"] == 3
        assert row["seed"] == 42
        assert row["icl_seed"] == 0
        assert row["icl_n_examples"] == 5

        output_rows.append(row)


# ---------------------------------------------------------------------
# VALIDATION
# ---------------------------------------------------------------------

assert len(output_rows) == 20

assert len({
    row["run_id"]
    for row in output_rows
}) == 20

for spec in SELECTIONS:

    selected = [
        row
        for row in output_rows
        if row["repro_id"] == spec["repro_id"]
    ]

    assert len(selected) == 2

    assert {
        row["replicate_id"]
        for row in selected
    } == {1, 2}

    # Both duplicates must point to exactly the same original.
    assert len({
        row["origin_run_id"]
        for row in selected
    }) == 1


OUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)

with OUT.open(
    "w",
    encoding="utf-8",
) as f:

    for row in output_rows:
        f.write(
            json.dumps(
                row,
                sort_keys=True,
            )
            + "\n"
        )


print("=" * 80)
print("FINAL-V2 REPRODUCIBILITY MANIFEST")
print("=" * 80)

print()
print("Representative configurations :", len(SELECTIONS))
print("Fresh repeats per configuration:", 2)
print("New GPU executions             :", len(output_rows))
print("Total executions/config incl.")
print("existing final-v2 run          : 3")

print()

for spec in SELECTIONS:

    rows = [
        r
        for r in output_rows
        if r["repro_id"] == spec["repro_id"]
    ]

    r = rows[0]

    print(
        f"{spec['repro_id']} | "
        f"{r['model_id']:>2} | "
        f"{r['parameter_id']:>2} | "
        f"{r['prompt_protocol']:<11} | "
        f"{r['prompt_type']:<15} | "
        f"origin={r['origin_run_id']}"
    )

print()
print(f"Manifest: {OUT}")
print()
print("REPRODUCIBILITY MANIFEST: PASS")