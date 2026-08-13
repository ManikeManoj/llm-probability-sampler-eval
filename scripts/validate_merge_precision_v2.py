#!/usr/bin/env python3

import json
from pathlib import Path

import numpy as np
import pandas as pd


MANIFEST = Path("manifests/precision_v2.jsonl")
OUTPUT_DIR = Path("outputs")
MERGED_DIR = Path("merged")
VALIDATION_DIR = Path("validation")

PREFIX_OUT = MERGED_DIR / "prefix_summary_precision_v2_merged.csv"
TOKEN_OUT = MERGED_DIR / "token_level_precision_v2_merged.csv"
PAIRS_OUT = MERGED_DIR / "precision_v2_pairwise_prefix_deltas.csv"
SUMMARY_OUT = VALIDATION_DIR / "precision_v2_results_validation.json"

TOL = 1e-6
TRUTH_TOL = 1e-12


def fail(msg):
    raise RuntimeError(msg)


def as_bool(x):
    if isinstance(x, bool):
        return x

    s = str(x).strip().lower()

    if s in {"true", "1", "yes"}:
        return True
    if s in {"false", "0", "no"}:
        return False

    fail(f"Cannot interpret boolean value: {x!r}")


def numeric(series):
    return pd.to_numeric(series, errors="raise").astype(float)


def normalize_prefixes(prefixes):
    return [
        "" if p == "ROOT" else p
        for p in prefixes
    ]


def read_manifest():
    with MANIFEST.open() as f:
        rows = [
            json.loads(line)
            for line in f
            if line.strip()
        ]

    if len(rows) != 60:
        fail(f"Expected 60 manifest rows, found {len(rows)}")

    ids = [r["run_id"] for r in rows]

    if len(ids) != len(set(ids)):
        fail("Duplicate run IDs in precision manifest")

    return rows


def load_output_files(pattern, expected_ids, filename_prefix):
    paths = sorted(OUTPUT_DIR.glob(pattern))

    if len(paths) != 60:
        fail(
            f"{pattern}: expected 60 files, "
            f"found {len(paths)}"
        )

    frames = []
    observed_ids = set()

    for path in paths:
        # keep_default_na=False is important:
        # ROOT is stored as an empty string prefix.
        df = pd.read_csv(
            path,
            keep_default_na=False,
        )

        if "run_id" not in df.columns:
            fail(f"{path}: missing run_id column")

        ids = set(df["run_id"].astype(str))

        if len(ids) != 1:
            fail(
                f"{path}: expected one run_id, "
                f"found {ids}"
            )

        run_id = next(iter(ids))

        if run_id not in expected_ids:
            fail(f"{path}: unexpected run_id {run_id}")

        expected_name = f"{filename_prefix}{run_id}.csv"

        if path.name != expected_name:
            fail(
                f"Filename/run_id mismatch:\n"
                f"  file={path.name}\n"
                f"  expected={expected_name}"
            )

        observed_ids.add(run_id)
        frames.append(df)

    if observed_ids != expected_ids:
        fail(
            "Output run IDs do not exactly match manifest"
        )

    return pd.concat(
        frames,
        ignore_index=True,
    )


def validate_run_metadata(df, row):
    run_id = row["run_id"]
    part = df[df["run_id"] == run_id]

    if part.empty:
        fail(f"{run_id}: no rows")

    checks = {
        "model_name": row["model_name"],
        "lm_scoring_method": row["lm_scoring_method"],
        "prompt_protocol": row["prompt_protocol"],
        "precision": row["precision"],
        "distribution": row["distribution"],
        "prompt_type": row["prompt_type"],
    }

    for col, expected in checks.items():
        values = set(part[col].astype(str))

        if values != {str(expected)}:
            fail(
                f"{run_id}: {col} mismatch: "
                f"{values} != {expected!r}"
            )

    bool_values = {
        as_bool(x)
        for x in part["load_in_4bit"].unique()
    }

    if bool_values != {bool(row["load_in_4bit"])}:
        fail(
            f"{run_id}: load_in_4bit mismatch "
            f"{bool_values}"
        )

    for col, expected in [
        ("n_samples", row["n_samples"]),
        ("decimals", row["decimals"]),
        ("icl_n_examples", row["icl_n_examples"]),
        ("icl_seed", row["icl_seed"]),
    ]:
        vals = set(
            pd.to_numeric(
                part[col],
                errors="raise",
            ).astype(int)
        )

        if vals != {int(expected)}:
            fail(
                f"{run_id}: {col} mismatch "
                f"{vals} != {expected}"
            )

    # Compare JSON structurally rather than as text.
    param_values = {
        json.dumps(
            json.loads(x),
            sort_keys=True,
        )
        for x in part["distribution_params"]
    }

    expected_params = json.dumps(
        row["params"],
        sort_keys=True,
    )

    if param_values != {expected_params}:
        fail(
            f"{run_id}: distribution params mismatch"
        )


def validate_prefix_coverage(prefix_df, manifest):
    if len(prefix_df) != 804:
        fail(
            f"Expected 804 prefix rows, "
            f"found {len(prefix_df)}"
        )

    duplicate_count = prefix_df.duplicated(
        subset=["run_id", "prefix"]
    ).sum()

    if duplicate_count:
        fail(
            f"Found {duplicate_count} duplicate "
            f"(run_id,prefix) rows"
        )

    for row in manifest:
        run_id = row["run_id"]

        expected = normalize_prefixes(
            row["prefixes"]
        )

        part = prefix_df[
            prefix_df["run_id"] == run_id
        ]

        observed = list(part["prefix"])

        if len(observed) != len(expected):
            fail(
                f"{run_id}: expected "
                f"{len(expected)} prefixes, "
                f"found {len(observed)}"
            )

        if set(observed) != set(expected):
            fail(
                f"{run_id}: prefix set mismatch\n"
                f"expected={expected}\n"
                f"observed={observed}"
            )


def validate_token_structure(token_df):
    duplicates = token_df.duplicated(
        subset=["run_id", "prefix", "token"]
    ).sum()

    if duplicates:
        fail(
            f"Found {duplicates} duplicate "
            f"(run_id,prefix,token) rows"
        )

    if token_df.empty:
        fail("Merged token table is empty")


def validate_candidate_mass(
    prefix_df,
    token_df,
):
    worst_restricted_sum = 0.0
    worst_unconditional_sum = 0.0
    worst_complement = 0.0
    worst_reconstruction = 0.0
    worst_prefix_mass_match = 0.0
    worst_analytic_sum = 0.0

    prefix_lookup = prefix_df.set_index(
        ["run_id", "prefix"]
    )

    for (run_id, prefix), g in token_df.groupby(
        ["run_id", "prefix"],
        sort=False,
    ):
        lm = numeric(g["lm_prob"]).to_numpy()
        unconditional = numeric(
            g["lm_prob_unconditional"]
        ).to_numpy()
        analytic = numeric(
            g["analytic_truth"]
        ).to_numpy()

        mass_values = numeric(
            g["valid_candidate_mass"]
        ).to_numpy()

        other_values = numeric(
            g["other_vocab_mass"]
        ).to_numpy()

        if (
            np.ptp(mass_values) > TOL
            or np.ptp(other_values) > TOL
        ):
            fail(
                f"{run_id} prefix={prefix!r}: "
                "candidate mass not constant "
                "across token rows"
            )

        mass = float(mass_values[0])
        other = float(other_values[0])

        restricted_err = abs(
            lm.sum() - 1.0
        )

        unconditional_err = abs(
            unconditional.sum() - mass
        )

        complement_err = abs(
            mass + other - 1.0
        )

        analytic_err = abs(
            analytic.sum() - 1.0
        )

        worst_restricted_sum = max(
            worst_restricted_sum,
            restricted_err,
        )

        worst_unconditional_sum = max(
            worst_unconditional_sum,
            unconditional_err,
        )

        worst_complement = max(
            worst_complement,
            complement_err,
        )

        worst_analytic_sum = max(
            worst_analytic_sum,
            analytic_err,
        )

        if restricted_err > TOL:
            fail(
                f"{run_id} {prefix!r}: "
                f"restricted sum error "
                f"{restricted_err}"
            )

        if unconditional_err > TOL:
            fail(
                f"{run_id} {prefix!r}: "
                f"unconditional sum error "
                f"{unconditional_err}"
            )

        if complement_err > TOL:
            fail(
                f"{run_id} {prefix!r}: "
                f"candidate complement error "
                f"{complement_err}"
            )

        if analytic_err > TOL:
            fail(
                f"{run_id} {prefix!r}: "
                f"analytic truth sum error "
                f"{analytic_err}"
            )

        if mass > 0:
            reconstructed = (
                unconditional / mass
            )

            reconstruction_err = float(
                np.max(
                    np.abs(
                        reconstructed - lm
                    )
                )
            )

            worst_reconstruction = max(
                worst_reconstruction,
                reconstruction_err,
            )

            if reconstruction_err > TOL:
                fail(
                    f"{run_id} {prefix!r}: "
                    f"reconstruction error "
                    f"{reconstruction_err}"
                )

        p_row = prefix_lookup.loc[
            (run_id, prefix)
        ]

        prefix_mass = float(
            p_row["valid_candidate_mass"]
        )

        prefix_other = float(
            p_row["other_vocab_mass"]
        )

        mass_match_err = max(
            abs(prefix_mass - mass),
            abs(prefix_other - other),
        )

        worst_prefix_mass_match = max(
            worst_prefix_mass_match,
            mass_match_err,
        )

        if mass_match_err > TOL:
            fail(
                f"{run_id} {prefix!r}: "
                "prefix/token candidate-mass mismatch"
            )

    return {
        "worst_restricted_sum_error":
            worst_restricted_sum,

        "worst_unconditional_sum_error":
            worst_unconditional_sum,

        "worst_candidate_complement_error":
            worst_complement,

        "worst_reconstruction_error":
            worst_reconstruction,

        "worst_prefix_token_mass_match_error":
            worst_prefix_mass_match,

        "worst_analytic_sum_error":
            worst_analytic_sum,
    }


def validate_precision_pairs(
    prefix_df,
    token_df,
    manifest,
):
    pairs = {}

    for row in manifest:
        key = (
            row["model_id"],
            row["parameter_id"],
            row["prompt_protocol"],
        )

        pairs.setdefault(
            key,
            {},
        )[row["precision"]] = row["run_id"]

    if len(pairs) != 30:
        fail(
            f"Expected 30 precision pairs, "
            f"found {len(pairs)}"
        )

    pair_rows = []

    worst_analytic_diff = 0.0
    worst_mc_diff = 0.0

    for key, runs in pairs.items():
        if set(runs) != {"4bit", "bf16"}:
            fail(
                f"Precision pair {key} incomplete: "
                f"{runs}"
            )

        rid4 = runs["4bit"]
        ridb = runs["bf16"]

        p4 = prefix_df[
            prefix_df["run_id"] == rid4
        ].copy()

        pb = prefix_df[
            prefix_df["run_id"] == ridb
        ].copy()

        merged = p4.merge(
            pb,
            on="prefix",
            suffixes=("_4bit", "_bf16"),
            validate="one_to_one",
        )

        if len(merged) != len(p4):
            fail(
                f"{key}: prefix pairing incomplete"
            )

        # Truth-side prefix quantities MUST be identical.
        truth_cols = [
            "mc_prefix_count",
            "tv_mc_analytic",
            "kl_mc_analytic",
            "spearman_mc_analytic",
            "entropy_analytic",
            "truth_top1_prob",
            "truth_top2_prob",
            "truth_top1_minus_top2",
            "truth_max_minus_min",
        ]

        for col in truth_cols:
            a = pd.to_numeric(
                merged[f"{col}_4bit"],
                errors="coerce",
            ).to_numpy()

            b = pd.to_numeric(
                merged[f"{col}_bf16"],
                errors="coerce",
            ).to_numpy()

            mask = np.isfinite(a) & np.isfinite(b)

            if mask.any():
                diff = float(
                    np.max(
                        np.abs(
                            a[mask] - b[mask]
                        )
                    )
                )

                worst_mc_diff = max(
                    worst_mc_diff,
                    diff,
                )

                if diff > TRUTH_TOL:
                    fail(
                        f"{key}: precision pair "
                        f"truth mismatch in {col}: "
                        f"{diff}"
                    )

        # Token-level analytic + MC truth must also match.
        t4 = token_df[
            token_df["run_id"] == rid4
        ][
            [
                "prefix",
                "token",
                "mc_truth",
                "analytic_truth",
            ]
        ]

        tb = token_df[
            token_df["run_id"] == ridb
        ][
            [
                "prefix",
                "token",
                "mc_truth",
                "analytic_truth",
            ]
        ]

        tm = t4.merge(
            tb,
            on=["prefix", "token"],
            suffixes=("_4bit", "_bf16"),
            validate="one_to_one",
        )

        if len(tm) != len(t4) or len(tm) != len(tb):
            fail(
                f"{key}: token-set precision pairing "
                "is incomplete"
            )

        for col in [
            "mc_truth",
            "analytic_truth",
        ]:
            a = numeric(
                tm[f"{col}_4bit"]
            ).to_numpy()

            b = numeric(
                tm[f"{col}_bf16"]
            ).to_numpy()

            diff = float(
                np.max(
                    np.abs(a - b)
                )
            )

            if col == "analytic_truth":
                worst_analytic_diff = max(
                    worst_analytic_diff,
                    diff,
                )

            if diff > TRUTH_TOL:
                fail(
                    f"{key}: token-level {col} "
                    f"diff={diff}"
                )

        model_id, parameter_id, protocol = key

        for _, r in merged.iterrows():
            tv4 = float(
                r["tv_analytic_lm_4bit"]
            )

            tvb = float(
                r["tv_analytic_lm_bf16"]
            )

            mass4 = float(
                r["valid_candidate_mass_4bit"]
            )

            massb = float(
                r["valid_candidate_mass_bf16"]
            )

            pair_rows.append(
                {
                    "model_id": model_id,
                    "model_name":
                        r["model_name_4bit"],
                    "parameter_id":
                        parameter_id,
                    "distribution":
                        r["distribution_4bit"],
                    "distribution_params":
                        r["distribution_params_4bit"],
                    "prompt_protocol":
                        protocol,
                    "prompt_type":
                        r["prompt_type_4bit"],
                    "prefix":
                        r["prefix"],
                    "prefix_kind":
                        r["prefix_kind_4bit"],

                    "tvd_4bit": tv4,
                    "tvd_bf16": tvb,

                    "delta_bf16_minus_4bit":
                        tvb - tv4,

                    "abs_tvd_delta":
                        abs(tvb - tv4),

                    "valid_candidate_mass_4bit":
                        mass4,

                    "valid_candidate_mass_bf16":
                        massb,

                    "delta_valid_mass_bf16_minus_4bit":
                        massb - mass4,
                }
            )

    pair_df = pd.DataFrame(pair_rows)

    if len(pair_df) != 402:
        # 67 baseline prefixes × 6 model/protocol conditions
        fail(
            f"Expected 402 paired prefix rows, "
            f"found {len(pair_df)}"
        )

    return (
        pair_df,
        {
            "precision_pairs": len(pairs),
            "paired_prefix_rows": len(pair_df),
            "worst_pairwise_analytic_truth_diff":
                worst_analytic_diff,
            "worst_pairwise_truth_metric_diff":
                worst_mc_diff,
        },
    )


def main():
    manifest = read_manifest()

    expected_ids = {
        r["run_id"]
        for r in manifest
    }

    print("=" * 80)
    print("PRECISION V2 RESULT VALIDATION")
    print("=" * 80)

    prefix_df = load_output_files(
        "prefix_summary_precision_v2_*.csv",
        expected_ids,
        "prefix_summary_",
    )

    token_df = load_output_files(
        "token_level_precision_v2_*.csv",
        expected_ids,
        "token_level_",
    )

    # Required v2 schema.
    prefix_required = {
        "run_id",
        "model_name",
        "lm_scoring_method",
        "prompt_protocol",
        "load_in_4bit",
        "precision",
        "distribution",
        "distribution_params",
        "prompt_type",
        "n_samples",
        "decimals",
        "prefix",
        "prefix_kind",
        "valid_candidate_mass",
        "other_vocab_mass",
        "tv_analytic_lm",
        "tv_mc_analytic",
        "entropy_analytic",
    }

    token_required = {
        "run_id",
        "model_name",
        "lm_scoring_method",
        "prompt_protocol",
        "load_in_4bit",
        "precision",
        "distribution",
        "distribution_params",
        "prompt_type",
        "prefix",
        "prefix_kind",
        "token",
        "mc_truth",
        "analytic_truth",
        "lm_prob",
        "lm_prob_unconditional",
        "valid_candidate_mass",
        "other_vocab_mass",
    }

    missing_prefix_cols = (
        prefix_required
        - set(prefix_df.columns)
    )

    missing_token_cols = (
        token_required
        - set(token_df.columns)
    )

    if missing_prefix_cols:
        fail(
            f"Prefix CSV missing columns: "
            f"{sorted(missing_prefix_cols)}"
        )

    if missing_token_cols:
        fail(
            f"Token CSV missing columns: "
            f"{sorted(missing_token_cols)}"
        )

    # Validate each run against the manifest.
    for row in manifest:
        validate_run_metadata(
            prefix_df,
            row,
        )

        validate_run_metadata(
            token_df,
            row,
        )

    validate_prefix_coverage(
        prefix_df,
        manifest,
    )

    validate_token_structure(
        token_df,
    )

    invariant_summary = validate_candidate_mass(
        prefix_df,
        token_df,
    )

    pair_df, pair_summary = (
        validate_precision_pairs(
            prefix_df,
            token_df,
            manifest,
        )
    )

    # Enrich merged tables with frozen-manifest IDs.
    model_id_map = {
        r["run_id"]: r["model_id"]
        for r in manifest
    }

    parameter_id_map = {
        r["run_id"]: r["parameter_id"]
        for r in manifest
    }

    prefix_df.insert(
        1,
        "model_id",
        prefix_df["run_id"].map(
            model_id_map
        ),
    )

    prefix_df.insert(
        2,
        "parameter_id",
        prefix_df["run_id"].map(
            parameter_id_map
        ),
    )

    token_df.insert(
        1,
        "model_id",
        token_df["run_id"].map(
            model_id_map
        ),
    )

    token_df.insert(
        2,
        "parameter_id",
        token_df["run_id"].map(
            parameter_id_map
        ),
    )

    MERGED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    VALIDATION_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    prefix_df.to_csv(
        PREFIX_OUT,
        index=False,
    )

    token_df.to_csv(
        TOKEN_OUT,
        index=False,
    )

    pair_df.to_csv(
        PAIRS_OUT,
        index=False,
    )

    precision_counts = (
        prefix_df[
            ["run_id", "precision"]
        ]
        .drop_duplicates()
        ["precision"]
        .value_counts()
        .to_dict()
    )

    summary = {
        "manifest_runs": 60,
        "prefix_files": 60,
        "token_files": 60,
        "prefix_rows": len(prefix_df),
        "token_rows": len(token_df),
        "unique_runs":
            prefix_df["run_id"].nunique(),
        "precision_run_counts":
            precision_counts,
        **invariant_summary,
        **pair_summary,
    }

    with SUMMARY_OUT.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
            sort_keys=True,
        )

    print()
    print("FILES / STRUCTURE")
    print(f"  Runs:              {summary['unique_runs']}")
    print(f"  Prefix rows:       {summary['prefix_rows']}")
    print(f"  Token rows:        {summary['token_rows']}")
    print(f"  Precision counts:  {precision_counts}")
    print(f"  Precision pairs:   {pair_summary['precision_pairs']}")

    print()
    print("CANDIDATE-MASS INVARIANTS")

    for key, value in invariant_summary.items():
        print(
            f"  {key}: {value:.3e}"
        )

    print()
    print("PAIRWISE TRUTH CONSISTENCY")
    print(
        "  max analytic-truth difference: "
        f"{pair_summary['worst_pairwise_analytic_truth_diff']:.3e}"
    )
    print(
        "  max truth-metric difference:   "
        f"{pair_summary['worst_pairwise_truth_metric_diff']:.3e}"
    )

    print()
    print("=" * 80)
    print("ALL PRECISION-V2 RESULT CHECKS PASSED")
    print("=" * 80)

    print(f"Prefix merge: {PREFIX_OUT}")
    print(f"Token merge:  {TOKEN_OUT}")
    print(f"Pair deltas:  {PAIRS_OUT}")
    print(f"Summary:      {SUMMARY_OUT}")


if __name__ == "__main__":
    main()