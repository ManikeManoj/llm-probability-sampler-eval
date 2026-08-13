# scripts/validate_prefix_grid_v2.py

import csv
import gc
import json
import math
import os
from collections import Counter

from distributions import (
    DistributionSpec,
    default_support_for_distribution,
)
from truth_model_analytic import (
    prefix_mass,
    next_token_truth_distribution,
)
from truth_model_mc import build_truth_model
from real_prefix_logic import (
    classify_prefix,
    valid_next_tokens,
)

from final_v2_spec import (
    DECIMALS,
    N_SAMPLES,
    MC_SEED,
    MC_RELIABLE_THRESHOLD,
    SMALL_MODEL_PROTOCOLS,
    MEDIUM_MODEL_PROTOCOLS,
    ALL_PROMPT_TYPES,
    MAIN_PROMPT_TYPES,
    BASELINE_PARAMETER_IDS,
    PARAMETER_CONFIGS,
    EXPECTED_PARAMETER_COUNTS,
    EXPECTED_RUN_COUNTS,
)


OUTPUT_DIR = "validation"
OUTPUT_CSV = os.path.join(
    OUTPUT_DIR,
    "prefix_grid_v2_validation.csv",
)
OUTPUT_SUMMARY = os.path.join(
    OUTPUT_DIR,
    "prefix_grid_v2_validation_summary.json",
)

SUM_TOL = 1e-10
MASS_TOL = 1e-15

# MC is only a stochastic cross-check.
# Large MC-vs-analytic TVD is reported, not treated as an automatic
# analytic-truth failure.
MC_TVD_WARN_THRESHOLD = 0.10

# All prefixes explicitly tagged "magnitude" were selected because they
# should create a genuinely nontrivial integer continuation decision.
MIN_MAGNITUDE_NORMALIZED_ENTROPY = 0.70


def normalize_prefix(prefix):
    return "" if prefix == "ROOT" else prefix


def display_prefix(prefix):
    return "ROOT" if prefix == "" else prefix


def entropy(dist, tokens):
    h = 0.0

    for tok in tokens:
        p = float(dist.get(tok, 0.0))

        if p > 0.0:
            h -= p * math.log(p)

    return h


def normalized_entropy(dist, tokens):
    if len(tokens) <= 1:
        return 0.0

    h = entropy(dist, tokens)
    h_max = math.log(len(tokens))

    if h_max <= 0.0:
        return 0.0

    return h / h_max


def tvd(dist_a, dist_b, tokens):
    return 0.5 * sum(
        abs(
            float(dist_a.get(tok, 0.0))
            - float(dist_b.get(tok, 0.0))
        )
        for tok in tokens
    )


def expected_stage(prefix):
    kind = classify_prefix(prefix)

    if kind == "start":
        return "root"

    if kind == "sign":
        return "sign"

    if kind == "integer":
        return "integer"

    if kind == "dot":
        return "dot"

    if kind == "fraction":
        frac_len = len(prefix.split(".", 1)[1])

        if frac_len == 1:
            return "fraction_d1"

        if frac_len == 2:
            return "fraction_d2"

        return f"fraction_d{frac_len}"

    raise ValueError(
        f"Unhandled prefix kind for {prefix!r}: {kind}"
    )


def flatten_prefixes(config):
    rows = []

    magnitude = {
        normalize_prefix(p)
        for p in config.get("magnitude_prefixes", [])
    }

    for declared_stage, prefixes in config["prefixes"].items():
        for raw_prefix in prefixes:
            prefix = normalize_prefix(raw_prefix)

            role = (
                "magnitude"
                if prefix in magnitude
                else "structural"
            )

            rows.append(
                {
                    "prefix": prefix,
                    "declared_stage": declared_stage,
                    "role": role,
                }
            )

    return rows


def validate_design_grid():
    print("=" * 80)
    print("VALIDATING LOCKED EXPERIMENT GRID")
    print("=" * 80)

    errors = []

    # --------------------------------------------------------------
    # Basic counts
    # --------------------------------------------------------------

    if len(SMALL_MODEL_PROTOCOLS) != 6:
        errors.append(
            f"Expected 6 small model/protocol conditions, "
            f"found {len(SMALL_MODEL_PROTOCOLS)}"
        )

    if len(MEDIUM_MODEL_PROTOCOLS) != 6:
        errors.append(
            f"Expected 6 medium model/protocol conditions, "
            f"found {len(MEDIUM_MODEL_PROTOCOLS)}"
        )

    if len(ALL_PROMPT_TYPES) != 11:
        errors.append(
            f"Expected 11 prompt types, found {len(ALL_PROMPT_TYPES)}"
        )

    if len(MAIN_PROMPT_TYPES) != 2:
        errors.append(
            f"Expected 2 main prompt types, "
            f"found {len(MAIN_PROMPT_TYPES)}"
        )

    if len(PARAMETER_CONFIGS) != 24:
        errors.append(
            f"Expected 24 parameter configurations, "
            f"found {len(PARAMETER_CONFIGS)}"
        )

    # --------------------------------------------------------------
    # Unique IDs
    # --------------------------------------------------------------

    small_ids = [x["id"] for x in SMALL_MODEL_PROTOCOLS]
    medium_ids = [x["id"] for x in MEDIUM_MODEL_PROTOCOLS]
    parameter_ids = [x["id"] for x in PARAMETER_CONFIGS]

    if len(small_ids) != len(set(small_ids)):
        errors.append("Duplicate small model/protocol IDs.")

    if len(medium_ids) != len(set(medium_ids)):
        errors.append("Duplicate medium model/protocol IDs.")

    if len(parameter_ids) != len(set(parameter_ids)):
        errors.append("Duplicate parameter configuration IDs.")

    if len(ALL_PROMPT_TYPES) != len(set(ALL_PROMPT_TYPES)):
        errors.append("Duplicate prompt types.")

    # --------------------------------------------------------------
    # Parameter-family counts
    # --------------------------------------------------------------

    observed_parameter_counts = Counter(
        x["distribution"]
        for x in PARAMETER_CONFIGS
    )

    for distribution, expected_count in EXPECTED_PARAMETER_COUNTS.items():
        observed = observed_parameter_counts.get(distribution, 0)

        if observed != expected_count:
            errors.append(
                f"{distribution}: expected {expected_count} "
                f"parameter configs, found {observed}"
            )

    # --------------------------------------------------------------
    # Baseline IDs exist and match distribution
    # --------------------------------------------------------------

    config_by_id = {
        x["id"]: x
        for x in PARAMETER_CONFIGS
    }

    for distribution, config_id in BASELINE_PARAMETER_IDS.items():
        if config_id not in config_by_id:
            errors.append(
                f"Missing baseline configuration {config_id}"
            )
            continue

        observed_distribution = (
            config_by_id[config_id]["distribution"]
        )

        if observed_distribution != distribution:
            errors.append(
                f"Baseline {config_id}: expected distribution "
                f"{distribution}, found {observed_distribution}"
            )

    # --------------------------------------------------------------
    # Combinatorial run counts
    # --------------------------------------------------------------

    n_small = len(SMALL_MODEL_PROTOCOLS)
    n_all_models = (
        len(SMALL_MODEL_PROTOCOLS)
        + len(MEDIUM_MODEL_PROTOCOLS)
    )
    n_baselines = len(BASELINE_PARAMETER_IDS)
    n_parameters = len(PARAMETER_CONFIGS)
    n_prompts = len(ALL_PROMPT_TYPES)

    computed = {
        "precision": n_small * n_baselines * 2,
        "main": n_all_models * n_baselines * len(MAIN_PROMPT_TYPES),
        "parameter": n_small * n_parameters,
        "prompt": n_small * n_baselines * n_prompts,
    }

    computed["total"] = sum(computed.values())

    for experiment, expected in EXPECTED_RUN_COUNTS.items():
        observed = computed[experiment]

        if observed != expected:
            errors.append(
                f"{experiment}: expected {expected} runs, "
                f"computed {observed}"
            )

    print("Small model/protocol conditions :", n_small)
    print("Medium model/protocol conditions:", len(MEDIUM_MODEL_PROTOCOLS))
    print("Baseline distributions          :", n_baselines)
    print("Parameter configurations        :", n_parameters)
    print("Prompt conditions               :", n_prompts)

    print()
    print("Expected/computed run counts:")

    for name in [
        "precision",
        "main",
        "parameter",
        "prompt",
        "total",
    ]:
        print(
            f"  {name:<10} "
            f"{computed[name]:>4}"
        )

    if errors:
        print("\nGRID VALIDATION FAILED")

        for error in errors:
            print("  ERROR:", error)

        raise RuntimeError(
            f"Locked grid validation failed with "
            f"{len(errors)} error(s)."
        )

    print("\nGRID VALIDATION PASSED")

    return computed


def validate_beta_prefix_identity():
    """
    B1-B6 must use exactly the same prefix grid.
    This is intentional so Beta shape is the only changing factor.
    """

    beta_configs = [
        x
        for x in PARAMETER_CONFIGS
        if x["distribution"] == "beta"
    ]

    reference = beta_configs[0]["prefixes"]

    for config in beta_configs[1:]:
        if config["prefixes"] != reference:
            raise RuntimeError(
                "Beta prefix identity check failed: "
                f"{config['id']} differs from {beta_configs[0]['id']}."
            )

    print("Beta B1-B6 identical-prefix check: PASS")


def validate_d1_d2_matching(config):
    d1 = {
        normalize_prefix(p)
        for p in config["prefixes"].get(
            "fraction_d1",
            [],
        )
    }

    d2 = {
        normalize_prefix(p)
        for p in config["prefixes"].get(
            "fraction_d2",
            [],
        )
    }

    errors = []

    for prefix_d2 in d2:
        parent_d1 = prefix_d2[:-1]

        if parent_d1 not in d1:
            errors.append(
                f"{config['id']}: d2 prefix "
                f"{prefix_d2!r} has missing d1 parent "
                f"{parent_d1!r}"
            )

    return errors


def validate_one_parameter_config(config):
    config_id = config["id"]
    distribution = config["distribution"]
    params = config["params"]

    spec = DistributionSpec(
        distribution,
        params,
    )

    (
        lower,
        upper,
        allow_negative,
        support_mode,
    ) = default_support_for_distribution(spec)

    prefix_entries = flatten_prefixes(config)

    errors = []
    warnings = []

    # --------------------------------------------------------------
    # No duplicated prefixes inside a parameter configuration
    # --------------------------------------------------------------

    raw_prefixes = [
        row["prefix"]
        for row in prefix_entries
    ]

    duplicates = [
        p
        for p, count in Counter(raw_prefixes).items()
        if count > 1
    ]

    if duplicates:
        errors.append(
            f"{config_id}: duplicate prefixes: "
            f"{[display_prefix(x) for x in duplicates]}"
        )

    # --------------------------------------------------------------
    # d1 -> d2 structural matching
    # --------------------------------------------------------------

    errors.extend(
        validate_d1_d2_matching(config)
    )

    # --------------------------------------------------------------
    # Build MC truth ONCE for this parameter configuration.
    # This is the exact same MC implementation used by the experiment.
    # --------------------------------------------------------------

    print()
    print("-" * 80)
    print(
        f"{config_id}: "
        f"{distribution} {params}"
    )
    print(
        f"support=({lower}, {upper}), "
        f"allow_negative={allow_negative}, "
        f"mode={support_mode}"
    )
    print("building 500k Monte Carlo truth...")

    (
        formatted_samples,
        mc_counts,
        mc_probs,
    ) = build_truth_model(
        distribution=distribution,
        params=params,
        n_samples=N_SAMPLES,
        decimals=DECIMALS,
        lower=lower,
        upper=upper,
        seed=MC_SEED,
    )

    output_rows = []

    for entry in prefix_entries:
        prefix = entry["prefix"]
        declared_stage = entry["declared_stage"]
        role = entry["role"]

        # ----------------------------------------------------------
        # Prefix syntax / stage
        # ----------------------------------------------------------

        try:
            observed_stage = expected_stage(prefix)
        except Exception as exc:
            errors.append(
                f"{config_id} {display_prefix(prefix)!r}: "
                f"invalid prefix: {exc}"
            )
            continue

        if observed_stage != declared_stage:
            errors.append(
                f"{config_id} {display_prefix(prefix)!r}: "
                f"declared stage={declared_stage}, "
                f"observed stage={observed_stage}"
            )

        # ----------------------------------------------------------
        # Grammar candidates
        # ----------------------------------------------------------

        allowed = valid_next_tokens(
            prefix=prefix,
            decimals=DECIMALS,
            allow_negative=allow_negative,
        )

        if not allowed:
            errors.append(
                f"{config_id} {display_prefix(prefix)!r}: "
                "empty allowed-next-token set"
            )
            continue

        # ----------------------------------------------------------
        # Analytic parent mass
        # ----------------------------------------------------------

        parent_mass = prefix_mass(
            prefix=prefix,
            distribution=distribution,
            params=params,
            decimals=DECIMALS,
            lower=lower,
            upper=upper,
        )

        if (
            not math.isfinite(parent_mass)
            or parent_mass <= MASS_TOL
        ):
            errors.append(
                f"{config_id} {display_prefix(prefix)!r}: "
                f"invalid/zero analytic prefix mass "
                f"{parent_mass}"
            )
            continue

        # ----------------------------------------------------------
        # Analytic next-symbol distribution
        # ----------------------------------------------------------

        analytic_dist = next_token_truth_distribution(
            prefix=prefix,
            distribution=distribution,
            params=params,
            decimals=DECIMALS,
            lower=lower,
            upper=upper,
            allow_negative=allow_negative,
        )

        if set(analytic_dist) != set(allowed):
            errors.append(
                f"{config_id} {display_prefix(prefix)!r}: "
                "analytic token set differs from grammar token set. "
                f"allowed={allowed}, "
                f"analytic={sorted(analytic_dist)}"
            )

        analytic_values = [
            float(analytic_dist.get(tok, 0.0))
            for tok in allowed
        ]

        if any(
            (not math.isfinite(p)) or p < -SUM_TOL
            for p in analytic_values
        ):
            errors.append(
                f"{config_id} {display_prefix(prefix)!r}: "
                f"invalid analytic probabilities "
                f"{analytic_dist}"
            )

        analytic_sum = sum(analytic_values)

        if abs(analytic_sum - 1.0) > SUM_TOL:
            errors.append(
                f"{config_id} {display_prefix(prefix)!r}: "
                f"analytic probabilities sum to "
                f"{analytic_sum:.16f}"
            )

        # ----------------------------------------------------------
        # Analytic entropy
        # ----------------------------------------------------------

        h = entropy(
            analytic_dist,
            allowed,
        )

        h_norm = normalized_entropy(
            analytic_dist,
            allowed,
        )

        if not math.isfinite(h):
            errors.append(
                f"{config_id} {display_prefix(prefix)!r}: "
                "non-finite truth entropy"
            )

        if not (
            -SUM_TOL
            <= h_norm
            <= 1.0 + SUM_TOL
        ):
            errors.append(
                f"{config_id} {display_prefix(prefix)!r}: "
                f"normalized entropy out of range: "
                f"{h_norm}"
            )

        # Magnitude prefixes are deliberately required to be
        # genuinely nontrivial.
        if role == "magnitude":
            if observed_stage != "integer":
                errors.append(
                    f"{config_id} {display_prefix(prefix)!r}: "
                    "magnitude prefix is not integer-stage"
                )

            if (
                h_norm
                < MIN_MAGNITUDE_NORMALIZED_ENTROPY
            ):
                errors.append(
                    f"{config_id} {display_prefix(prefix)!r}: "
                    f"magnitude prefix entropy too low: "
                    f"H={h:.6f}, Hnorm={h_norm:.6f}"
                )

        # ----------------------------------------------------------
        # Monte Carlo cross-check
        # ----------------------------------------------------------

        prefix_counter = mc_counts.get(
            prefix,
            {},
        )

        mc_prefix_count = sum(
            prefix_counter.values()
        )

        mc_reliable = (
            mc_prefix_count
            >= MC_RELIABLE_THRESHOLD
        )

        mc_dist = mc_probs.get(
            prefix,
            {},
        )

        # MC should never emit a symbol the grammar says is impossible.
        unexpected_mc_tokens = (
            set(mc_dist)
            - set(allowed)
        )

        if unexpected_mc_tokens:
            errors.append(
                f"{config_id} {display_prefix(prefix)!r}: "
                f"MC produced grammar-invalid next symbols "
                f"{sorted(unexpected_mc_tokens)}"
            )

        mc_tvd = None

        if mc_prefix_count > 0:
            mc_tvd = tvd(
                mc_dist,
                analytic_dist,
                allowed,
            )

            if (
                mc_reliable
                and mc_tvd
                > MC_TVD_WARN_THRESHOLD
            ):
                warnings.append(
                    f"{config_id} "
                    f"{display_prefix(prefix)}: "
                    f"reliable MC/analytic TVD "
                    f"{mc_tvd:.4f} exceeds warning "
                    f"threshold "
                    f"{MC_TVD_WARN_THRESHOLD}"
                )

        expected_mc_count = (
            N_SAMPLES
            * parent_mass
        )

        output_rows.append(
            {
                "parameter_id": config_id,
                "distribution": distribution,
                "params": json.dumps(
                    params,
                    sort_keys=True,
                ),
                "support_mode": support_mode,
                "allow_negative": allow_negative,
                "prefix": display_prefix(prefix),
                "stage": observed_stage,
                "role": role,
                "analytic_prefix_mass": parent_mass,
                "expected_mc_count": expected_mc_count,
                "mc_prefix_count": mc_prefix_count,
                "mc_reliable": mc_reliable,
                "allowed_tokens": json.dumps(
                    allowed,
                ),
                "analytic_next_probs": json.dumps(
                    analytic_dist,
                    sort_keys=True,
                ),
                "analytic_prob_sum": analytic_sum,
                "truth_entropy": h,
                "truth_entropy_normalized": h_norm,
                "mc_analytic_tvd": mc_tvd,
            }
        )

    del formatted_samples
    del mc_counts
    del mc_probs
    gc.collect()

    return output_rows, errors, warnings


def validate_beta1_uniform1_equivalence(all_rows):
    """
    Beta(1,1) and Uniform(0,1) are mathematically identical.
    For their identical prefix grid, analytic next-symbol distributions
    should match.
    """

    u1 = {
        row["prefix"]: row
        for row in all_rows
        if row["parameter_id"] == "U1"
    }

    b1 = {
        row["prefix"]: row
        for row in all_rows
        if row["parameter_id"] == "B1"
    }

    if set(u1) != set(b1):
        raise RuntimeError(
            "U1/B1 prefix sets differ. "
            "Beta(1,1) control is not matched."
        )

    for prefix in sorted(u1):
        u_dist = json.loads(
            u1[prefix]["analytic_next_probs"]
        )
        b_dist = json.loads(
            b1[prefix]["analytic_next_probs"]
        )

        tokens = sorted(
            set(u_dist)
            | set(b_dist)
        )

        difference = tvd(
            u_dist,
            b_dist,
            tokens,
        )

        if difference > 1e-10:
            raise RuntimeError(
                f"U1/B1 analytic equivalence failed "
                f"at prefix {prefix!r}: "
                f"TVD={difference}"
            )

    print(
        "Uniform(0,1) vs Beta(1,1) "
        "analytic-equivalence check: PASS"
    )


def write_validation_csv(rows):
    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )

    fieldnames = [
        "parameter_id",
        "distribution",
        "params",
        "support_mode",
        "allow_negative",
        "prefix",
        "stage",
        "role",
        "analytic_prefix_mass",
        "expected_mc_count",
        "mc_prefix_count",
        "mc_reliable",
        "allowed_tokens",
        "analytic_next_probs",
        "analytic_prob_sum",
        "truth_entropy",
        "truth_entropy_normalized",
        "mc_analytic_tvd",
    ]

    with open(
        OUTPUT_CSV,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)


def print_magnitude_summary(rows):
    magnitude_rows = [
        row
        for row in rows
        if row["role"] == "magnitude"
    ]

    print()
    print("=" * 80)
    print("MAGNITUDE-STRESS PREFIXES")
    print("=" * 80)

    for row in magnitude_rows:
        print(
            f"{row['parameter_id']:<4} "
            f"prefix={row['prefix']:<5} "
            f"H={row['truth_entropy']:.6f} "
            f"Hnorm="
            f"{row['truth_entropy_normalized']:.6f} "
            f"MC={row['mc_prefix_count']}"
        )


def print_mc_summary(rows):
    reliable_rows = [
        row
        for row in rows
        if (
            row["mc_reliable"]
            and row["mc_analytic_tvd"]
            is not None
        )
    ]

    unreliable_rows = [
        row
        for row in rows
        if not row["mc_reliable"]
    ]

    print()
    print("=" * 80)
    print("MONTE CARLO COVERAGE")
    print("=" * 80)

    print(
        f"Total validated prefixes : {len(rows)}"
    )
    print(
        f"MC reliable             : {len(reliable_rows)}"
    )
    print(
        f"MC below threshold      : {len(unreliable_rows)}"
    )

    if reliable_rows:
        worst = max(
            reliable_rows,
            key=lambda row: row["mc_analytic_tvd"],
        )

        print(
            "Worst reliable MC/analytic TVD: "
            f"{worst['mc_analytic_tvd']:.6f} "
            f"({worst['parameter_id']} "
            f"prefix={worst['prefix']})"
        )

    if unreliable_rows:
        print()
        print(
            "Prefixes below MC reliability threshold "
            "(NOT analytic failures):"
        )

        for row in unreliable_rows:
            print(
                f"  {row['parameter_id']:<4} "
                f"{row['prefix']:<7} "
                f"stage={row['stage']:<11} "
                f"MC={row['mc_prefix_count']:<6} "
                f"expected≈{row['expected_mc_count']:.1f}"
            )


def main():
    computed_runs = validate_design_grid()

    print()
    validate_beta_prefix_identity()

    all_rows = []
    all_errors = []
    all_warnings = []

    print()
    print("=" * 80)
    print("VALIDATING ANALYTIC + MC PREFIX GRID")
    print("=" * 80)

    for config in PARAMETER_CONFIGS:
        (
            rows,
            errors,
            warnings,
        ) = validate_one_parameter_config(
            config
        )

        all_rows.extend(rows)
        all_errors.extend(errors)
        all_warnings.extend(warnings)

    # Independent control:
    # Beta(1,1) must equal Uniform(0,1)
    validate_beta1_uniform1_equivalence(
        all_rows
    )

    write_validation_csv(
        all_rows
    )

    print_magnitude_summary(
        all_rows
    )

    print_mc_summary(
        all_rows
    )

    summary = {
        "decimals": DECIMALS,
        "n_samples": N_SAMPLES,
        "mc_seed": MC_SEED,
        "mc_reliable_threshold": MC_RELIABLE_THRESHOLD,
        "parameter_configurations": len(PARAMETER_CONFIGS),
        "validated_prefixes": len(all_rows),
        "mc_reliable_prefixes": sum(
            bool(row["mc_reliable"])
            for row in all_rows
        ),
        "mc_unreliable_prefixes": sum(
            not bool(row["mc_reliable"])
            for row in all_rows
        ),
        "errors": all_errors,
        "warnings": all_warnings,
        "computed_run_counts": computed_runs,
    }

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )

    with open(
        OUTPUT_SUMMARY,
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
    print("=" * 80)

    if all_warnings:
        print(
            f"WARNINGS: {len(all_warnings)}"
        )

        for warning in all_warnings:
            print("  WARNING:", warning)

    if all_errors:
        print(
            f"PREFIX GRID VALIDATION FAILED: "
            f"{len(all_errors)} error(s)"
        )

        for error in all_errors:
            print("  ERROR:", error)

        print("=" * 80)

        raise RuntimeError(
            "Final-v2 prefix validation failed."
        )

    print("ALL FINAL-V2 PREFIX/GRID CHECKS PASSED")
    print("=" * 80)
    print(
        f"CSV:     {OUTPUT_CSV}"
    )
    print(
        f"Summary: {OUTPUT_SUMMARY}"
    )


if __name__ == "__main__":
    main()