#!/usr/bin/env python3

import json
from pathlib import Path

import numpy as np
import pandas as pd


MANIFEST = Path("manifests/reproducibility_v2.jsonl")
OUTPUT_DIR = Path("outputs")
VALIDATION_DIR = Path("validation")
MERGED_DIR = Path("merged")


def read_manifest():
    rows = []
    with MANIFEST.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def read_csv(path):
    return pd.read_csv(
        path,
        dtype={"prefix": "string", "token": "string"},
        keep_default_na=False,
    )


def load_token(run_id):
    return read_csv(
        OUTPUT_DIR / f"token_level_{run_id}.csv"
    )


def load_prefix(run_id):
    return read_csv(
        OUTPUT_DIR / f"prefix_summary_{run_id}.csv"
    )


def compare_three(dfs, keys, metric):
    """
    dfs:
      {
        "original": df,
        "rep1": df,
        "rep2": df,
      }

    Returns row-level ranges plus summary.
    """

    pieces = []

    for label, df in dfs.items():
        x = df[keys + [metric]].copy()
        x = x.rename(columns={metric: label})
        pieces.append(x)

    merged = pieces[0]

    for piece in pieces[1:]:
        merged = merged.merge(
            piece,
            on=keys,
            how="outer",
            validate="one_to_one",
        )

    assert not merged[
        ["original", "rep1", "rep2"]
    ].isna().any().any()

    values = merged[
        ["original", "rep1", "rep2"]
    ].to_numpy(float)

    merged["range"] = (
        np.max(values, axis=1)
        - np.min(values, axis=1)
    )

    merged["sd"] = np.std(
        values,
        axis=1,
        ddof=1,
    )

    merged["abs_rep1_minus_original"] = np.abs(
        merged["rep1"] - merged["original"]
    )

    merged["abs_rep2_minus_original"] = np.abs(
        merged["rep2"] - merged["original"]
    )

    merged["abs_rep1_minus_rep2"] = np.abs(
        merged["rep1"] - merged["rep2"]
    )

    summary = {
        "metric": metric,
        "n_rows": len(merged),

        "max_range": float(
            merged["range"].max()
        ),

        "mean_range": float(
            merged["range"].mean()
        ),

        "median_range": float(
            merged["range"].median()
        ),

        "p95_range": float(
            merged["range"].quantile(0.95)
        ),

        "max_sd": float(
            merged["sd"].max()
        ),

        "mean_sd": float(
            merged["sd"].mean()
        ),

        "max_abs_rep1_minus_original": float(
            merged[
                "abs_rep1_minus_original"
            ].max()
        ),

        "max_abs_rep2_minus_original": float(
            merged[
                "abs_rep2_minus_original"
            ].max()
        ),

        "max_abs_rep1_minus_rep2": float(
            merged[
                "abs_rep1_minus_rep2"
            ].max()
        ),
    }

    return merged, summary


rows = read_manifest()

assert len(rows) == 20


# ---------------------------------------------------------------------
# GROUP REPLICATES
# ---------------------------------------------------------------------

groups = {}

for row in rows:
    groups.setdefault(
        row["repro_id"],
        []
    ).append(row)

assert len(groups) == 10


all_summaries = []
all_detail = []


TOKEN_METRICS = [
    "lm_prob",
    "lm_prob_unconditional",
    "analytic_truth",
]

PREFIX_METRICS = [
    "tv_analytic_lm",
    "valid_candidate_mass",
    "other_vocab_mass",
]


for repro_id in sorted(groups):

    reps = sorted(
        groups[repro_id],
        key=lambda x: x["replicate_id"],
    )

    assert len(reps) == 2
    assert reps[0]["replicate_id"] == 1
    assert reps[1]["replicate_id"] == 2

    origin_run_id = reps[0]["origin_run_id"]

    assert (
        reps[1]["origin_run_id"]
        == origin_run_id
    )

    rep1_run_id = reps[0]["run_id"]
    rep2_run_id = reps[1]["run_id"]

    print("=" * 80)
    print(repro_id)
    print("origin:", origin_run_id)
    print("rep1  :", rep1_run_id)
    print("rep2  :", rep2_run_id)

    # ==============================================================
    # TOKEN LEVEL
    # ==============================================================

    token_dfs = {
        "original": load_token(
            origin_run_id
        ),
        "rep1": load_token(
            rep1_run_id
        ),
        "rep2": load_token(
            rep2_run_id
        ),
    }

    for metric in TOKEN_METRICS:

        detail, summary = compare_three(
            token_dfs,
            ["prefix", "token"],
            metric,
        )

        summary.update({
            "repro_id": repro_id,
            "level": "token",

            "origin_run_id":
                origin_run_id,

            "rep1_run_id":
                rep1_run_id,

            "rep2_run_id":
                rep2_run_id,

            "model_id":
                reps[0]["model_id"],

            "parameter_id":
                reps[0]["parameter_id"],

            "prompt_protocol":
                reps[0]["prompt_protocol"],

            "prompt_type":
                reps[0]["prompt_type"],
        })

        all_summaries.append(summary)

        detail["repro_id"] = repro_id
        detail["level"] = "token"
        detail["metric"] = metric

        all_detail.append(detail)

    # ==============================================================
    # PREFIX LEVEL
    # ==============================================================

    prefix_dfs = {
        "original": load_prefix(
            origin_run_id
        ),
        "rep1": load_prefix(
            rep1_run_id
        ),
        "rep2": load_prefix(
            rep2_run_id
        ),
    }

    for metric in PREFIX_METRICS:

        detail, summary = compare_three(
            prefix_dfs,
            ["prefix"],
            metric,
        )

        summary.update({
            "repro_id": repro_id,
            "level": "prefix",

            "origin_run_id":
                origin_run_id,

            "rep1_run_id":
                rep1_run_id,

            "rep2_run_id":
                rep2_run_id,

            "model_id":
                reps[0]["model_id"],

            "parameter_id":
                reps[0]["parameter_id"],

            "prompt_protocol":
                reps[0]["prompt_protocol"],

            "prompt_type":
                reps[0]["prompt_type"],
        })

        all_summaries.append(summary)

        detail["repro_id"] = repro_id
        detail["level"] = "prefix"
        detail["metric"] = metric

        all_detail.append(detail)


# ---------------------------------------------------------------------
# WRITE
# ---------------------------------------------------------------------

summary_df = pd.DataFrame(
    all_summaries
)

detail_df = pd.concat(
    all_detail,
    ignore_index=True,
)

MERGED_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

VALIDATION_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

summary_df.to_csv(
    MERGED_DIR
    / "reproducibility_v2_summary.csv",
    index=False,
)

detail_df.to_csv(
    MERGED_DIR
    / "reproducibility_v2_differences.csv",
    index=False,
)


# ---------------------------------------------------------------------
# GLOBAL SUMMARY
# ---------------------------------------------------------------------

print()
print("=" * 80)
print("FINAL V2 REPRODUCIBILITY ANALYSIS")
print("=" * 80)

for metric in (
    TOKEN_METRICS
    + PREFIX_METRICS
):

    x = summary_df[
        summary_df["metric"] == metric
    ]

    print()
    print(metric)

    print(
        "  configurations :",
        len(x)
    )

    print(
        "  maximum range  :",
        f"{x['max_range'].max():.12g}"
    )

    print(
        "  maximum SD     :",
        f"{x['max_sd'].max():.12g}"
    )

    print(
        "  max rep1-origin:",
        f"{x['max_abs_rep1_minus_original'].max():.12g}"
    )

    print(
        "  max rep2-origin:",
        f"{x['max_abs_rep2_minus_original'].max():.12g}"
    )

    print(
        "  max rep1-rep2  :",
        f"{x['max_abs_rep1_minus_rep2'].max():.12g}"
    )


# ---------------------------------------------------------------------
# IMPORTANT SANITY CHECK:
# analytic truth must not change between executions.
# ---------------------------------------------------------------------

analytic = summary_df[
    summary_df["metric"]
    == "analytic_truth"
]

analytic_max = float(
    analytic["max_range"].max()
)

assert analytic_max <= 1e-12, (
    "Analytic truth changed across repetitions: "
    f"{analytic_max}"
)


global_summary = {
    "configurations": 10,
    "fresh_repeats_per_configuration": 2,
    "executions_per_configuration_including_original": 3,

    "max_lm_prob_range": float(
        summary_df.loc[
            summary_df["metric"]
            == "lm_prob",
            "max_range",
        ].max()
    ),

    "max_tvd_range": float(
        summary_df.loc[
            summary_df["metric"]
            == "tv_analytic_lm",
            "max_range",
        ].max()
    ),

    "max_candidate_mass_range": float(
        summary_df.loc[
            summary_df["metric"]
            == "valid_candidate_mass",
            "max_range",
        ].max()
    ),

    "max_analytic_truth_range":
        analytic_max,
}


with (
    VALIDATION_DIR
    / "reproducibility_v2_summary.json"
).open(
    "w"
) as f:

    json.dump(
        global_summary,
        f,
        indent=2,
        sort_keys=True,
    )


print()
print("=" * 80)
print("KEY REPRODUCIBILITY NUMBERS")
print("=" * 80)

print(
    "max |range lm_prob|          :",
    f"{global_summary['max_lm_prob_range']:.12g}"
)

print(
    "max |range TVD|              :",
    f"{global_summary['max_tvd_range']:.12g}"
)

print(
    "max |range candidate mass|   :",
    f"{global_summary['max_candidate_mass_range']:.12g}"
)

print(
    "max |range analytic truth|   :",
    f"{global_summary['max_analytic_truth_range']:.12g}"
)

print()
print("REPRODUCIBILITY ANALYSIS COMPLETE")