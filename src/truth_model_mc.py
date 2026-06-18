from collections import Counter, defaultdict
from distributions import DistributionSpec, sample_distribution
import numpy as np


def sample_truncated_normal(
    mean: float,
    std: float,
    n_samples: int,
    lower: float | None = None,
    upper: float | None = None,
    seed: int = 42,
):

    rng = np.random.default_rng(seed)
    accepted = []

    if lower is None:
        lower = float("-inf")
    if upper is None:
        upper = float("inf")

    while len(accepted) < n_samples:
        batch_size = max(1000, int((n_samples - len(accepted)) * 1.5))
        batch = rng.normal(loc=mean, scale=std, size=batch_size)

        batch = batch[(batch >= lower) & (batch <= upper)]
        accepted.extend(batch.tolist())

    return np.array(accepted[:n_samples])


def format_sample(x: float, decimals: int = 3) -> str:

    return f"{x:.{decimals}f}"


def build_formatted_samples(
    distribution: str = "normal",
    params: dict | None = None,
    mean: float | None = None,
    std: float | None = None,
    n_samples: int = 100000,
    decimals: int = 3,
    lower: float | None = None,
    upper: float | None = None,
    seed: int = 42,
):
    """
    Build formatted numeric samples from a chosen distribution.

    Backward compatible:
    - For old Normal runs, you can still pass mean/std.
    - For new distribution-agnostic runs, pass distribution + params.
    """

    if params is None:
        if distribution == "normal":
            if mean is None or std is None:
                raise ValueError("Normal distribution requires mean and std.")
            params = {"mean": mean, "std": std}
        else:
            raise ValueError(
                f"Non-normal distribution {distribution!r} requires params."
            )

    spec = DistributionSpec(distribution, params)

    numeric_samples = sample_distribution(
        spec=spec,
        n_samples=n_samples,
        lower=lower,
        upper=upper,
        seed=seed,
    )

    string_samples = [format_sample(x, decimals=decimals) for x in numeric_samples]
    return string_samples


def collect_prefix_next_token_counts(strings: list[str]):
    """
    For each formatted number string s, count transitions:
      prefix = s[:i]
      next_token = s[i]
    """
    prefix_to_next_counter = defaultdict(Counter)

    for s in strings:
        for i in range(len(s)):
            prefix = s[:i]
            next_token = s[i]
            prefix_to_next_counter[prefix][next_token] += 1

    return prefix_to_next_counter


def normalize_counter(counter: Counter):
    total = sum(counter.values())
    if total == 0:
        return {}

    return {token: count / total for token, count in counter.items()}


def build_truth_model(
    distribution: str = "normal",
    params: dict | None = None,
    mean: float | None = None,
    std: float | None = None,
    n_samples: int = 100000,
    decimals: int = 3,
    lower: float | None = None,
    upper: float | None = None,
    seed: int = 42,
):
    formatted_samples = build_formatted_samples(
        distribution=distribution,
        params=params,
        mean=mean,
        std=std,
        n_samples=n_samples,
        decimals=decimals,
        lower=lower,
        upper=upper,
        seed=seed,
    )

    prefix_to_next_counts = collect_prefix_next_token_counts(formatted_samples)

    prefix_to_next_probs = {
        prefix: normalize_counter(counter)
        for prefix, counter in prefix_to_next_counts.items()
    }

    return formatted_samples, prefix_to_next_counts, prefix_to_next_probs


def pretty_print_distribution(prefix: str, dist: dict[str, float]):
    print(f"\nPrefix: {repr(prefix)}")
    for token, prob in sorted(dist.items(), key=lambda x: x[1], reverse=True):
        print(f"  next='{token}'  prob={prob:.6f}")


if __name__ == "__main__":
    formatted_samples, prefix_to_next_counts, prefix_to_next_probs = build_truth_model(
        distribution="uniform",
        params={"low": 0.0, "high": 1.0},
        n_samples=100000,
        decimals=3,
        lower=0.0,
        upper=1.0,
        seed=42,
    )

    print("First 10 formatted samples:")
    for s in formatted_samples[:10]:
        print(" ", s)

    for prefix in ["", "0", "0.", "0.0", "0.5", "0.9"]:
        dist = prefix_to_next_probs.get(prefix, {})
        pretty_print_distribution(prefix, dist)