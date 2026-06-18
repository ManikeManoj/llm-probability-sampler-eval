from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class DistributionSpec:
    name: str
    params: dict[str, Any]

    def normalized_name(self) -> str:
        return self.name.lower().strip()

    def label(self) -> str:
        name = self.normalized_name()
        p = self.params

        if name == "normal":
            return f"Normal({p['mean']}, {p['std']})"

        if name == "uniform":
            return f"Uniform({p['low']}, {p['high']})"

        if name == "exponential":
            return f"Exponential(rate={p['rate']})"

        if name == "beta":
            return f"Beta({p['alpha']}, {p['beta']})"

        if name == "laplace":
            return f"Laplace({p['loc']}, {p['scale']})"

        if name == "lognormal":
            return f"Lognormal({p['meanlog']}, {p['sdlog']})"

        raise ValueError(f"Unknown distribution: {self.name!r}")


def _require_keys(params: dict[str, Any], keys: list[str], distribution_name: str) -> None:
    missing = [k for k in keys if k not in params]
    if missing:
        raise ValueError(
            f"Missing parameter(s) for {distribution_name!r}: {missing}. "
            f"Received params={params!r}"
        )


def validate_distribution(spec: DistributionSpec) -> None:
    """
    Checks whether the distribution name and parameters are valid.
    This prevents silent mistakes before analytic or MC truth is computed.
    """

    name = spec.normalized_name()
    p = spec.params

    if name == "normal":
        _require_keys(p, ["mean", "std"], name)
        if p["std"] <= 0:
            raise ValueError("Normal std must be positive.")
        return

    if name == "uniform":
        _require_keys(p, ["low", "high"], name)
        if p["high"] <= p["low"]:
            raise ValueError("Uniform high must be greater than low.")
        return

    if name == "exponential":
        _require_keys(p, ["rate"], name)
        if p["rate"] <= 0:
            raise ValueError("Exponential rate must be positive.")
        return

    if name == "beta":
        _require_keys(p, ["alpha", "beta"], name)
        if p["alpha"] <= 0 or p["beta"] <= 0:
            raise ValueError("Beta alpha and beta must be positive.")
        return

    if name == "laplace":
        _require_keys(p, ["loc", "scale"], name)
        if p["scale"] <= 0:
            raise ValueError("Laplace scale must be positive.")
        return

    if name == "lognormal":
        _require_keys(p, ["meanlog", "sdlog"], name)
        if p["sdlog"] <= 0:
            raise ValueError("Lognormal sdlog must be positive.")
        return

    raise ValueError(f"Unknown distribution: {spec.name!r}")


def distribution_cdf(x: float, spec: DistributionSpec) -> float:
    """
    CDF F(x) for the selected distribution.

    This is used by the analytic truth model:
        P(a <= X < b) = F(b) - F(a)
    """

    validate_distribution(spec)

    name = spec.normalized_name()
    p = spec.params

    if name == "normal":
        return float(
            stats.norm.cdf(
                x,
                loc=p["mean"],
                scale=p["std"],
            )
        )

    if name == "uniform":
        return float(
            stats.uniform.cdf(
                x,
                loc=p["low"],
                scale=p["high"] - p["low"],
            )
        )

    if name == "exponential":
        return float(
            stats.expon.cdf(
                x,
                scale=1.0 / p["rate"],
            )
        )

    if name == "beta":
        return float(
            stats.beta.cdf(
                x,
                a=p["alpha"],
                b=p["beta"],
            )
        )

    if name == "laplace":
        return float(
            stats.laplace.cdf(
                x,
                loc=p["loc"],
                scale=p["scale"],
            )
        )

    if name == "lognormal":
        return float(
            stats.lognorm.cdf(
                x,
                s=p["sdlog"],
                scale=np.exp(p["meanlog"]),
            )
        )

    raise ValueError(f"Unknown distribution: {spec.name!r}")


def interval_mass(
    a: float,
    b: float,
    spec: DistributionSpec,
    lower: float | None = None,
    upper: float | None = None,
) -> float:
    """
    Probability mass in interval [a, b).

    If lower/upper are given, this computes the conditional probability:

        P(a <= X < b | lower <= X <= upper)

    This is the generic replacement for the old Normal-only interval mass.
    """

    validate_distribution(spec)

    if b <= a:
        return 0.0

    left = a if lower is None else max(a, lower)
    right = b if upper is None else min(b, upper)

    if right <= left:
        return 0.0

    numerator = distribution_cdf(right, spec) - distribution_cdf(left, spec)

    # No truncation / conditioning
    if lower is None and upper is None:
        return max(0.0, float(numerator))

    cond_left = float("-inf") if lower is None else lower
    cond_right = float("inf") if upper is None else upper

    denominator = distribution_cdf(cond_right, spec) - distribution_cdf(cond_left, spec)

    if denominator <= 0.0:
        return 0.0

    return max(0.0, float(numerator / denominator))


def sample_distribution(
    spec: DistributionSpec,
    n_samples: int,
    lower: float | None = None,
    upper: float | None = None,
    seed: int = 42,
) -> np.ndarray:
    """
    Monte Carlo sampler for the selected distribution.

    Uses NumPy for sampling.
    If lower/upper are supplied, rejection sampling is used.
    """

    validate_distribution(spec)

    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")

    rng = np.random.default_rng(seed)

    name = spec.normalized_name()
    p = spec.params

    lo = float("-inf") if lower is None else lower
    hi = float("inf") if upper is None else upper

    if hi <= lo:
        raise ValueError(f"Invalid bounds: lower={lower}, upper={upper}")

    def draw_batch(size: int) -> np.ndarray:
        if name == "normal":
            return rng.normal(
                loc=p["mean"],
                scale=p["std"],
                size=size,
            )

        if name == "uniform":
            return rng.uniform(
                low=p["low"],
                high=p["high"],
                size=size,
            )

        if name == "exponential":
            return rng.exponential(
                scale=1.0 / p["rate"],
                size=size,
            )

        if name == "beta":
            return rng.beta(
                a=p["alpha"],
                b=p["beta"],
                size=size,
            )

        if name == "laplace":
            return rng.laplace(
                loc=p["loc"],
                scale=p["scale"],
                size=size,
            )

        if name == "lognormal":
            return rng.lognormal(
                mean=p["meanlog"],
                sigma=p["sdlog"],
                size=size,
            )

        raise ValueError(f"Unknown distribution: {spec.name!r}")

    accepted: list[float] = []

    while len(accepted) < n_samples:
        remaining = n_samples - len(accepted)
        batch_size = max(1000, int(remaining * 1.5))

        batch = draw_batch(batch_size)
        batch = batch[(batch >= lo) & (batch <= hi)]

        accepted.extend(batch.tolist())

    return np.asarray(accepted[:n_samples], dtype=float)


def default_support_for_distribution(
    spec: DistributionSpec,
) -> tuple[float | None, float | None, bool, str]:
    """
    Returns sensible default support settings.

    Returns:
        lower, upper, allow_negative, support_mode
    """

    validate_distribution(spec)

    name = spec.normalized_name()
    p = spec.params

    if name == "normal":
        return None, None, True, "agnostic"

    if name == "uniform":
        allow_negative = p["low"] < 0
        return p["low"], p["high"], allow_negative, "bounded"

    if name == "exponential":
        return 0.0, None, False, "positive"

    if name == "beta":
        return 0.0, 1.0, False, "bounded"

    if name == "laplace":
        return None, None, True, "agnostic"

    if name == "lognormal":
        return 0.0, None, False, "positive"

    raise ValueError(f"Unknown distribution: {spec.name!r}")


def format_distribution_params(params: dict[str, Any]) -> str:
    """
    Stable parameter string for run IDs or filenames.

    Example:
        {"low": 0.0, "high": 1.0}
        -> "high=1.0_low=0.0"
    """

    return "_".join(f"{k}={params[k]}" for k in sorted(params))


if __name__ == "__main__":
    specs = [
        DistributionSpec("normal", {"mean": 0.0, "std": 1.0}),
        DistributionSpec("uniform", {"low": 0.0, "high": 1.0}),
        DistributionSpec("exponential", {"rate": 1.0}),
        DistributionSpec("beta", {"alpha": 2.0, "beta": 2.0}),
        DistributionSpec("laplace", {"loc": 0.0, "scale": 1.0}),
        DistributionSpec("lognormal", {"meanlog": 0.0, "sdlog": 0.5}),
    ]

    for spec in specs:
        print("\n===", spec.label(), "===")
        print("default support:", default_support_for_distribution(spec))
        print("CDF(0):", distribution_cdf(0.0, spec))
        print("CDF(1):", distribution_cdf(1.0, spec))
        print("mass[0, 1]:", interval_mass(0.0, 1.0, spec))
        print("samples:", sample_distribution(spec, n_samples=5, seed=42))