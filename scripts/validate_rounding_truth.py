#!/usr/bin/env python3
"""Validation suite for rounding-aware analytic prefix truth.

Run from the repository root after replacing src/truth_model_analytic.py:
    python scripts/validate_rounding_truth.py

This does not load any language model and does not need a GPU.
"""
from __future__ import annotations

import math

from real_prefix_logic import valid_next_tokens
from truth_model_analytic import (
    decimal_prefix_interval,
    next_token_truth_distribution,
    prefix_mass,
)
from truth_model_mc import build_truth_model


def assert_close(a: float, b: float, tol: float = 1e-12, msg: str = "") -> None:
    if not math.isclose(a, b, rel_tol=0.0, abs_tol=tol):
        raise AssertionError(f"{msg} expected {b}, got {a}")


def tvd(a: dict[str, float], b: dict[str, float]) -> float:
    keys = set(a) | set(b)
    return 0.5 * sum(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in keys)


def test_geometry() -> None:
    expected = {
        "0.5": (0.4995, 0.5995),
        "-0.5": (-0.5995, -0.4995),
        "10.": (9.9995, 10.9995),
        "-10.": (-10.9995, -9.9995),
        "0.": (0.0, 0.9995),
        "-0.": (-0.9995, 0.0),
        "0.500": (0.4995, 0.5005),
        "-0.500": (-0.5005, -0.4995),
    }

    for prefix, (want_l, want_r) in expected.items():
        got_l, got_r = decimal_prefix_interval(prefix, decimals=3)
        assert_close(got_l, want_l, msg=f"left interval for {prefix!r}")
        assert_close(got_r, want_r, msg=f"right interval for {prefix!r}")

    print("PASS geometry: rounding-aware decimal intervals and signed zero")


def test_uniform_boundary() -> None:
    dist = next_token_truth_distribution(
        prefix="",
        distribution="uniform",
        params={"low": 0.0, "high": 1.0},
        decimals=3,
        lower=0.0,
        upper=1.0,
        allow_negative=False,
    )

    assert_close(dist["0"], 0.9995, tol=1e-12, msg="Uniform ROOT->0")
    assert_close(dist["1"], 0.0005, tol=1e-12, msg="Uniform ROOT->1")

    one = next_token_truth_distribution(
        prefix="1",
        distribution="uniform",
        params={"low": 0.0, "high": 1.0},
        decimals=3,
        lower=0.0,
        upper=1.0,
        allow_negative=False,
    )
    assert_close(one["."], 1.0, msg="Uniform prefix '1' -> '.'")

    print("PASS boundary: Uniform(0,1) correctly assigns rounding mass to '1.000'")


def test_parent_child_partition() -> None:
    cases = [
        (
            "uniform", {"low": 0.0, "high": 1.0}, 0.0, 1.0, False,
            ["", "0", "0.", "0.0", "0.5", "0.99", "1"],
        ),
        (
            "normal", {"mean": 0.0, "std": 1.0}, None, None, True,
            ["", "-", "-0", "0", "-1", "1", "-0.", "0.", "-0.5", "0.5"],
        ),
        (
            "normal", {"mean": 10.0, "std": 0.5}, None, None, True,
            ["", "1", "9", "10", "11", "10.", "10.0", "10.5"],
        ),
        (
            "exponential", {"rate": 1.0}, 0.0, None, False,
            ["", "0", "1", "2", "0.", "1.", "0.5"],
        ),
        (
            "beta", {"alpha": 2.0, "beta": 2.0}, 0.0, 1.0, False,
            ["", "0", "1", "0.", "0.5", "0.9"],
        ),
        (
            "laplace", {"loc": 0.0, "scale": 1.0}, None, None, True,
            ["", "-", "-0", "0", "-1", "1", "-0.", "0.", "-0.5", "0.5"],
        ),
    ]

    worst = 0.0
    worst_case = None

    for distribution, params, lower, upper, allow_negative, prefixes in cases:
        for prefix in prefixes:
            allowed = valid_next_tokens(prefix, decimals=3, allow_negative=allow_negative)
            if not allowed:
                continue

            parent = prefix_mass(
                prefix,
                distribution=distribution,
                params=params,
                lower=lower,
                upper=upper,
                decimals=3,
            )
            children = sum(
                prefix_mass(
                    prefix + tok,
                    distribution=distribution,
                    params=params,
                    lower=lower,
                    upper=upper,
                    decimals=3,
                )
                for tok in allowed
            )

            err = abs(parent - children)
            if err > worst:
                worst = err
                worst_case = (distribution, params, prefix, parent, children)

            if err > 1e-12:
                raise AssertionError(
                    "Parent/child partition failed: "
                    f"distribution={distribution} params={params} prefix={prefix!r} "
                    f"parent={parent} children={children} error={err}"
                )

    print(f"PASS partition: max |parent-sum(children)| = {worst:.3e}")
    if worst_case:
        print("  worst case:", worst_case)


def test_mc_agreement(n_samples: int = 500_000) -> None:
    """A stochastic cross-check; threshold is deliberately loose for finite MC noise."""
    cases = [
        (
            "uniform", {"low": 0.0, "high": 1.0}, 0.0, 1.0, False,
            ["", "0.", "0.0", "0.5", "0.9"],
        ),
        (
            "normal", {"mean": 0.0, "std": 1.0}, None, None, True,
            ["", "-", "0.", "-0.", "1.", "-1.", "0.5", "-0.5"],
        ),
        (
            "normal", {"mean": 10.0, "std": 0.5}, None, None, True,
            ["", "1", "10.", "10.0", "10.5"],
        ),
        (
            "exponential", {"rate": 1.0}, 0.0, None, False,
            ["", "0.", "1.", "0.5"],
        ),
        (
            "beta", {"alpha": 2.0, "beta": 2.0}, 0.0, 1.0, False,
            ["", "0.", "0.5", "0.9"],
        ),
        (
            "laplace", {"loc": 0.0, "scale": 1.0}, None, None, True,
            ["", "-", "0.", "-0.", "0.5", "-0.5"],
        ),
    ]

    worst = 0.0
    worst_case = None

    for distribution, params, lower, upper, allow_negative, prefixes in cases:
        _, counts, mc_probs = build_truth_model(
            distribution=distribution,
            params=params,
            n_samples=n_samples,
            decimals=3,
            lower=lower,
            upper=upper,
            seed=123,
        )

        for prefix in prefixes:
            count = sum(counts.get(prefix, {}).values())
            if count < 5_000:
                # Avoid pretending a rare-prefix finite-MC sample is a precise validator.
                continue

            analytic = next_token_truth_distribution(
                prefix=prefix,
                distribution=distribution,
                params=params,
                decimals=3,
                lower=lower,
                upper=upper,
                allow_negative=allow_negative,
            )
            mc = mc_probs.get(prefix, {})
            d = tvd(analytic, mc)

            if d > worst:
                worst = d
                worst_case = (distribution, params, prefix, count, d)

    # At >=5k conditional samples, 0.03 is intentionally generous; typical observed
    # deviations are much smaller (~0.001-0.01) and should be MC noise.
    if worst > 0.03:
        raise AssertionError(f"Analytic/MC TVD unexpectedly large: {worst_case}")

    print(f"PASS MC cross-check: worst TVD = {worst:.6f}")
    print("  worst case:", worst_case)


def main() -> None:
    test_geometry()
    test_uniform_boundary()
    test_parent_child_partition()
    test_mc_agreement()
    print("\nALL ROUNDING-TRUTH VALIDATION TESTS PASSED")


if __name__ == "__main__":
    main()
