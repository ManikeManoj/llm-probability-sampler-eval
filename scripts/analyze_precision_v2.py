#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import pandas as pd


INPUT = Path(
    "merged/precision_v2_pairwise_prefix_deltas.csv"
)

OUT_DIR = Path("merged")
OUT_DIR.mkdir(exist_ok=True)


def q(x, p):
    return float(np.quantile(x, p))


def summary_stats(df):
    signed = df["delta_bf16_minus_4bit"].astype(float)
    absolute = df["abs_tvd_delta"].astype(float)

    return {
        "n": len(df),

        "mean_delta_bf16_minus_4bit":
            signed.mean(),

        "median_delta_bf16_minus_4bit":
            signed.median(),

        "mean_abs_delta":
            absolute.mean(),

        "median_abs_delta":
            absolute.median(),

        "p90_abs_delta":
            q(absolute, 0.90),

        "p95_abs_delta":
            q(absolute, 0.95),

        "max_abs_delta":
            absolute.max(),

        "bf16_better_fraction":
            float((signed < 0).mean()),

        "4bit_better_fraction":
            float((signed > 0).mean()),

        "equal_fraction":
            float(
                np.isclose(
                    signed,
                    0.0,
                    atol=1e-8,
                ).mean()
            ),
    }


def grouped_summary(df, columns):
    rows = []

    for key, g in df.groupby(
        columns,
        dropna=False,
    ):
        if not isinstance(key, tuple):
            key = (key,)

        row = dict(zip(columns, key))
        row.update(summary_stats(g))

        row["mean_tvd_4bit"] = (
            g["tvd_4bit"].mean()
        )

        row["mean_tvd_bf16"] = (
            g["tvd_bf16"].mean()
        )

        rows.append(row)

    return pd.DataFrame(rows)


def aggregate_condition_ranking(df):
    """
    Compare whether model/protocol ordering changes
    under 4-bit vs BF16.
    """

    agg = (
        df.groupby(
            [
                "model_id",
                "model_name",
                "prompt_protocol",
            ],
            as_index=False,
        )
        .agg(
            mean_tvd_4bit=(
                "tvd_4bit",
                "mean",
            ),
            mean_tvd_bf16=(
                "tvd_bf16",
                "mean",
            ),
        )
    )

    agg["rank_4bit"] = (
        agg["mean_tvd_4bit"]
        .rank(
            method="average",
            ascending=True,
        )
    )

    agg["rank_bf16"] = (
        agg["mean_tvd_bf16"]
        .rank(
            method="average",
            ascending=True,
        )
    )

    agg["rank_shift"] = (
        agg["rank_bf16"]
        - agg["rank_4bit"]
    )

    spearman = (
        agg["mean_tvd_4bit"]
        .corr(
            agg["mean_tvd_bf16"],
            method="spearman",
        )
    )

    return agg, float(spearman)


def main():
    df = pd.read_csv(
        INPUT,
        keep_default_na=False,
    )

    print("=" * 80)
    print("PRECISION V2 EFFECT ANALYSIS")
    print("=" * 80)

    # --------------------------------------------------------------
    # Overall
    # --------------------------------------------------------------

    overall = summary_stats(df)

    print("\nOVERALL — PREFIX LEVEL")

    for key, value in overall.items():
        if key == "n":
            print(f"  {key:<32}: {value}")
        else:
            print(
                f"  {key:<32}: "
                f"{value:.6f}"
            )

    print()
    print(
        "Interpretation of signed delta:"
    )
    print(
        "  negative = BF16 has lower TVD"
    )
    print(
        "  positive = 4-bit has lower TVD"
    )

    # --------------------------------------------------------------
    # Stage
    # --------------------------------------------------------------

    stage = grouped_summary(
        df,
        ["prefix_kind"],
    )

    print("\n" + "=" * 80)
    print("BY PREFIX STAGE")
    print("=" * 80)

    print(
        stage[
            [
                "prefix_kind",
                "n",
                "mean_tvd_4bit",
                "mean_tvd_bf16",
                "mean_delta_bf16_minus_4bit",
                "median_delta_bf16_minus_4bit",
                "mean_abs_delta",
                "p95_abs_delta",
            ]
        ]
        .sort_values("prefix_kind")
        .to_string(index=False)
    )

    # --------------------------------------------------------------
    # Model / protocol
    # --------------------------------------------------------------

    model_protocol = grouped_summary(
        df,
        [
            "model_id",
            "model_name",
            "prompt_protocol",
        ],
    )

    print("\n" + "=" * 80)
    print("BY MODEL / PROTOCOL")
    print("=" * 80)

    print(
        model_protocol[
            [
                "model_id",
                "prompt_protocol",
                "n",
                "mean_tvd_4bit",
                "mean_tvd_bf16",
                "mean_delta_bf16_minus_4bit",
                "mean_abs_delta",
                "p95_abs_delta",
            ]
        ]
        .sort_values(
            [
                "model_id",
                "prompt_protocol",
            ]
        )
        .to_string(index=False)
    )

    # --------------------------------------------------------------
    # Distribution
    # --------------------------------------------------------------

    distribution = grouped_summary(
        df,
        [
            "parameter_id",
            "distribution",
        ],
    )

    print("\n" + "=" * 80)
    print("BY DISTRIBUTION")
    print("=" * 80)

    print(
        distribution[
            [
                "parameter_id",
                "distribution",
                "n",
                "mean_tvd_4bit",
                "mean_tvd_bf16",
                "mean_delta_bf16_minus_4bit",
                "mean_abs_delta",
                "p95_abs_delta",
            ]
        ]
        .sort_values("parameter_id")
        .to_string(index=False)
    )

    # --------------------------------------------------------------
    # Ranking stability
    # --------------------------------------------------------------

    rankings, spearman = (
        aggregate_condition_ranking(df)
    )

    print("\n" + "=" * 80)
    print("MODEL / PROTOCOL RANKING STABILITY")
    print("=" * 80)

    print(
        rankings[
            [
                "model_id",
                "prompt_protocol",
                "mean_tvd_4bit",
                "mean_tvd_bf16",
                "rank_4bit",
                "rank_bf16",
                "rank_shift",
            ]
        ]
        .sort_values("rank_4bit")
        .to_string(index=False)
    )

    print()
    print(
        "Spearman rank correlation "
        f"(4-bit vs BF16): {spearman:.6f}"
    )

    # --------------------------------------------------------------
    # Largest differences
    # --------------------------------------------------------------

    largest = (
        df.sort_values(
            "abs_tvd_delta",
            ascending=False,
        )
        .head(20)
        .copy()
    )

    print("\n" + "=" * 80)
    print("20 LARGEST ABSOLUTE TVD DIFFERENCES")
    print("=" * 80)

    print(
        largest[
            [
                "model_id",
                "prompt_protocol",
                "parameter_id",
                "prefix",
                "prefix_kind",
                "tvd_4bit",
                "tvd_bf16",
                "delta_bf16_minus_4bit",
                "abs_tvd_delta",
            ]
        ].to_string(index=False)
    )

    # --------------------------------------------------------------
    # Candidate mass sensitivity
    # --------------------------------------------------------------

    mass_delta = (
        df[
            "delta_valid_mass_bf16_minus_4bit"
        ]
        .astype(float)
        .abs()
    )

    print("\n" + "=" * 80)
    print("VALID-CANDIDATE-MASS PRECISION EFFECT")
    print("=" * 80)

    print(
        f"  mean |delta mass|   : "
        f"{mass_delta.mean():.6f}"
    )

    print(
        f"  median |delta mass| : "
        f"{mass_delta.median():.6f}"
    )

    print(
        f"  p95 |delta mass|    : "
        f"{q(mass_delta, 0.95):.6f}"
    )

    print(
        f"  max |delta mass|    : "
        f"{mass_delta.max():.6f}"
    )

    # --------------------------------------------------------------
    # Write summaries
    # --------------------------------------------------------------

    stage.to_csv(
        OUT_DIR
        / "precision_v2_stage_summary.csv",
        index=False,
    )

    model_protocol.to_csv(
        OUT_DIR
        / "precision_v2_model_protocol_summary.csv",
        index=False,
    )

    distribution.to_csv(
        OUT_DIR
        / "precision_v2_distribution_summary.csv",
        index=False,
    )

    rankings.to_csv(
        OUT_DIR
        / "precision_v2_ranking_stability.csv",
        index=False,
    )

    largest.to_csv(
        OUT_DIR
        / "precision_v2_largest_differences.csv",
        index=False,
    )

    print("\n" + "=" * 80)
    print("PRECISION V2 ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()