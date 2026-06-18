import math

from distributions import DistributionSpec, interval_mass
from real_prefix_logic import (
    classify_prefix,
    valid_next_tokens,
    ordered_interval,
    successor_value,
)

# 1. Basic Normal / Truncated-Normal Probability Mass

def normal_cdf(x: float, mu: float, sigma: float) -> float:
    z = (x - mu) / (sigma * math.sqrt(2.0))
    return 0.5 * (1.0 + math.erf(z))


def normal_interval_mass(
    a: float,
    b: float,
    mu: float,
    sigma: float,
) -> float:

    if b <= a:
        return 0.0
    return normal_cdf(b, mu, sigma) - normal_cdf(a, mu, sigma)


def truncated_normal_interval_mass(
    a: float,
    b: float,
    mu: float,
    sigma: float,
    lower: float | None = None,
    upper: float | None = None,
) -> float:

    if lower is None:
        lower = float("-inf")
    if upper is None:
        upper = float("inf")

    left = max(a, lower)
    right = min(b, upper)

    if right <= left:
        return 0.0

    numerator = normal_interval_mass(left, right, mu, sigma)
    denominator = normal_interval_mass(lower, upper, mu, sigma)

    if denominator <= 0.0:
        return 0.0

    return numerator / denominator

# 2. Prefix Mass for Integer / Decimal Prefixes
def integer_prefix_mass(
    prefix: str,
    mu: float,
    sigma: float,
    lower: float | None,
    upper: float | None,
    max_extra_digits: int = 12,
    tol: float = 1e-15,
) -> float:

    if prefix == "0":
        return truncated_normal_interval_mass(0.0, 1.0, mu, sigma, lower, upper)

    if prefix == "-0":
        return truncated_normal_interval_mass(-1.0, 0.0, mu, sigma, lower, upper)

    base = float(prefix)
    succ = successor_value(prefix)

    total = 0.0
    tiny_count = 0

    for i in range(max_extra_digits + 1):
        a = base * (10 ** i)
        b = succ * (10 ** i)
        left, right = ordered_interval(a, b)

        mass = truncated_normal_interval_mass(left, right, mu, sigma, lower, upper)
        total += mass

        
        if mass < tol:
            tiny_count += 1
            if tiny_count >= 3:
                break
        else:
            tiny_count = 0

    return total


def decimal_prefix_interval(prefix: str) -> tuple[float, float]:

    base = float(prefix)
    succ = successor_value(prefix)
    return ordered_interval(base, succ)

def prefix_mass(
    prefix: str,
    distribution: str = "normal",
    params: dict | None = None,
    mu: float | None = None,
    sigma: float | None = None,
    lower: float | None = None,
    upper: float | None = None,
    max_extra_digits: int = 12,
) -> float:
    
    if params is None:
        if distribution == "normal":
            if mu is None or sigma is None:
                raise ValueError("Normal distribution requires mu and sigma.")
            params = {"mean": mu, "std": sigma}
        else:
            raise ValueError(
                f"Non-normal distribution {distribution!r} requires params."
            )

    spec = DistributionSpec(distribution, params)

    kind = classify_prefix(prefix)

    if kind == "start":
        return 1.0

    if kind == "sign":
        return interval_mass(
            float("-inf"),
            0.0,
            spec=spec,
            lower=lower,
            upper=upper,
        )

    if kind == "integer":
        return integer_prefix_mass_generic(
            prefix=prefix,
            spec=spec,
            lower=lower,
            upper=upper,
            max_extra_digits=max_extra_digits,
        )

    if kind in {"dot", "fraction"}:
        left, right = decimal_prefix_interval(prefix)
        return interval_mass(
            left,
            right,
            spec=spec,
            lower=lower,
            upper=upper,
        )

    raise ValueError(f"Unhandled prefix kind: {kind}")


def integer_prefix_mass_generic(
    prefix: str,
    spec: DistributionSpec,
    lower: float | None,
    upper: float | None,
    max_extra_digits: int = 12,
    tol: float = 1e-15,
) -> float:
    """
    Mass of all numbers whose integer/string prefix begins with `prefix`.

    Example:
        prefix "4" includes:
        [4,5), [40,50), [400,500), ...

        prefix "-1" includes:
        [-2,-1), [-20,-10), [-200,-100), ...

    This is generic across distributions because it uses interval_mass().
    """

    if prefix == "0":
        return interval_mass(0.0, 1.0, spec=spec, lower=lower, upper=upper)

    if prefix == "-0":
        return interval_mass(-1.0, 0.0, spec=spec, lower=lower, upper=upper)

    base = float(prefix)
    succ = successor_value(prefix)

    total = 0.0
    tiny_count = 0

    for i in range(max_extra_digits + 1):
        a = base * (10 ** i)
        b = succ * (10 ** i)

        left, right = ordered_interval(a, b)

        mass = interval_mass(
            left,
            right,
            spec=spec,
            lower=lower,
            upper=upper,
        )

        total += mass

        if mass < tol:
            tiny_count += 1
            if tiny_count >= 3:
                break
        else:
            tiny_count = 0

    return total
# 3. Recursive Next-Token Truth Distribution

def next_token_truth_distribution(
    prefix: str,
    distribution: str = "normal",
    params: dict | None = None,
    mu: float | None = None,
    sigma: float | None = None,
    decimals: int = 3,
    lower: float | None = None,
    upper: float | None = None,
    allow_negative: bool = True,
    max_extra_digits: int = 12,
):

    allowed = valid_next_tokens(
        prefix=prefix,
        decimals=decimals,
        allow_negative=allow_negative,
    )

    if len(allowed) == 0:
        return {}

    parent_mass = prefix_mass(
        prefix=prefix,
        distribution=distribution,
        params=params,
        mu=mu,
        sigma=sigma,
        lower=lower,
        upper=upper,
        max_extra_digits=max_extra_digits,
    )

    if parent_mass <= 0.0:
        return {tok: 0.0 for tok in allowed}

    dist = {}

    for tok in allowed:
        child_prefix = prefix + tok

        child_mass = prefix_mass(
            prefix=child_prefix,
            distribution=distribution,
            params=params,
            mu=mu,
            sigma=sigma,
            lower=lower,
            upper=upper,
            max_extra_digits=max_extra_digits,
        )

        dist[tok] = child_mass / parent_mass

    total = sum(dist.values())

    if total > 0.0:
        dist = {k: v / total for k, v in dist.items()}
    
    return dist



def pretty_print_distribution(prefix: str, dist: dict[str, float]):
    print(f"\nPrefix: {repr(prefix)}")
    for tok, prob in sorted(dist.items(), key=lambda x: x[1], reverse=True):
        print(f"  next='{tok}'  prob={prob:.6f}")



if __name__ == "__main__":
    print("=== Uniform(0,1) analytic example ===")

    DISTRIBUTION = "uniform"
    PARAMS = {"low": 0.0, "high": 1.0}
    LOWER = 0.0
    UPPER = 1.0
    DECIMALS = 3

    prefixes = ["", "0", "0.", "0.0", "0.5", "0.9"]

    for prefix in prefixes:
        dist = next_token_truth_distribution(
            prefix=prefix,
            distribution=DISTRIBUTION,
            params=PARAMS,
            decimals=DECIMALS,
            lower=LOWER,
            upper=UPPER,
            allow_negative=False,
        )
        pretty_print_distribution(prefix, dist)