from __future__ import annotations

from distributions import DistributionSpec, sample_distribution

ALL_PROMPT_TYPES = [
    "short",
    "plain",
    "formal",
    "explanatory_1",
    "explanatory_2",
    "explanatory_3",
    "explanatory_4",
    "cot",
    "icl",
    "icl_cot",
]

def _support_text(support_mode: str, lower, upper) -> str:
    """Returns a short support constraint clause, or empty string."""

    if support_mode == "positive":
        return "conditioned to be non-negative"

    if support_mode == "bounded":
        if lower is None or upper is None:
            raise ValueError("support_mode='bounded' requires both lower and upper")
        return f"conditioned to lie between {lower} and {upper}"

    if support_mode == "agnostic":
        return ""

    raise ValueError(f"Unknown support_mode: {support_mode!r}")

def _icl_examples(
    spec: DistributionSpec,
    decimals: int,
    n_examples: int,
    lower=None,
    upper=None,
    seed: int = 0,
) -> list[str]:
    """
    Generate in-context examples from the same distribution.
    """

    samples = sample_distribution(
        spec=spec,
        n_samples=n_examples,
        lower=lower,
        upper=upper,
        seed=seed,
    )

    return [f"{x:.{decimals}f}" for x in samples]

def build_prompt(
    distribution: str = "normal",
    params: dict | None = None,
    mean: float | None = None,
    std: float | None = None,
    decimals: int = 3,
    prompt_type: str = "plain",
    support_mode: str = "agnostic",
    lower: float | None = None,
    upper: float | None = None,
    icl_n_examples: int = 5,
    icl_seed: int = 0,
) -> str:
    """
    Distribution-aware prompt dispatcher.

    Backward compatible:
    - Old Normal runs can still pass mean/std.
    - New runs should pass distribution + params.
    """

    if params is None:
        if distribution == "normal":
            if mean is None or std is None:
                raise ValueError("Normal prompt requires mean and std.")
            params = {"mean": mean, "std": std}
        else:
            raise ValueError(f"Non-normal distribution {distribution!r} requires params.")

    spec = DistributionSpec(distribution, params)
    name = spec.normalized_name()

    if prompt_type not in ALL_PROMPT_TYPES:
        raise ValueError(f"Unknown prompt_type: {prompt_type!r}")

    if name == "normal":
        return build_normal_prompt(
            spec=spec,
            decimals=decimals,
            prompt_type=prompt_type,
            support_mode=support_mode,
            lower=lower,
            upper=upper,
            icl_n_examples=icl_n_examples,
            icl_seed=icl_seed,
        )

    if name == "uniform":
        return build_uniform_prompt(
            spec=spec,
            decimals=decimals,
            prompt_type=prompt_type,
            support_mode=support_mode,
            lower=lower,
            upper=upper,
            icl_n_examples=icl_n_examples,
            icl_seed=icl_seed,
        )
    
    if name == "exponential":
        return build_exponential_prompt(
        spec=spec,
        decimals=decimals,
        prompt_type=prompt_type,
        support_mode=support_mode,
        lower=lower,
        upper=upper,
        icl_n_examples=icl_n_examples,
        icl_seed=icl_seed,
        )
    
    if name == "beta":
        return build_beta_prompt(
        spec=spec,
        decimals=decimals,
        prompt_type=prompt_type,
        support_mode=support_mode,
        lower=lower,
        upper=upper,
        icl_n_examples=icl_n_examples,
        icl_seed=icl_seed,
        )
    
    if name == "laplace":
        return build_laplace_prompt(
        spec=spec,
        decimals=decimals,
        prompt_type=prompt_type,
        support_mode=support_mode,
        lower=lower,
        upper=upper,
        icl_n_examples=icl_n_examples,
        icl_seed=icl_seed,
        )
    """if name == "lognormal":
        return build_lognormal_prompt(
        spec=spec,
        decimals=decimals,
        prompt_type=prompt_type,
        support_mode=support_mode,
        lower=lower,
        upper=upper,
        icl_n_examples=icl_n_examples,
        icl_seed=icl_seed,
        )"""



    raise NotImplementedError(
        f"Prompt templates for distribution {distribution!r} are not implemented yet."
    )


def build_normal_prompt(
    spec: DistributionSpec,
    decimals: int,
    prompt_type: str,
    support_mode: str,
    lower,
    upper,
    icl_n_examples: int,
    icl_seed: int,
) -> str:
    """
    Normal prompt family.

    This preserves your old prompt ladder:
    short, plain, formal, explanatory_1..4, cot, icl, icl_cot.
    """

    mean = spec.params["mean"]
    std = spec.params["std"]

    support = _support_text(support_mode, lower, upper)
    sc = f" {support}" if support else ""

    if prompt_type == "short":
        return (
            f"Sample one number from N({mean}, {std}){sc}. "
            f"Use exactly {decimals} decimal places. "
            f"Output only the number."
        )

    if prompt_type == "plain":
        return (
            f"You are generating from a Normal distribution with mean {mean} "
            f"and standard deviation {std}{sc}. "
            f"Return one independent sample as a number with exactly {decimals} decimal places. "
            f"Output only the number."
        )

    if prompt_type == "formal":
        return (
            f"Draw one independent sample from a Normal distribution with mean {mean} "
            f"and standard deviation {std}{sc}. "
            f"Return the answer as a number with exactly {decimals} decimal places. "
            f"Output only the number."
        )

    if prompt_type == "explanatory_1":
        return (
            f"You are drawing a sample from a Normal (Gaussian) distribution "
            f"with mean {mean} and standard deviation {std}{sc}. "
            f"Output the sample as a number with exactly {decimals} decimal places. "
            f"Output only the number."
        )

    if prompt_type == "explanatory_2":
        lo1, hi1 = round(mean - std, 4), round(mean + std, 4)
        lo2, hi2 = round(mean - 2 * std, 4), round(mean + 2 * std, 4)
        lo3, hi3 = round(mean - 3 * std, 4), round(mean + 3 * std, 4)

        return (
            f"You are drawing a sample from a Normal (Gaussian) distribution "
            f"with mean {mean} and standard deviation {std}{sc}. "
            f"This distribution has the following coverage properties: "
            f"about 68% of values fall between {lo1} and {hi1}, "
            f"about 95% fall between {lo2} and {hi2}, "
            f"and about 99.7% fall between {lo3} and {hi3}. "
            f"Output the sample as a number with exactly {decimals} decimal places. "
            f"Output only the number."
        )

    if prompt_type == "explanatory_3":
        lo1, hi1 = round(mean - std, 4), round(mean + std, 4)
        lo2, hi2 = round(mean - 2 * std, 4), round(mean + 2 * std, 4)
        lo3, hi3 = round(mean - 3 * std, 4), round(mean + 3 * std, 4)

        return (
            f"You are drawing a sample from a Normal (Gaussian) distribution "
            f"with mean {mean} and standard deviation {std}{sc}. "
            f"The distribution is symmetric around {mean}. "
            f"A typical sample falls between {lo1} and {hi1} (68% probability). "
            f"Values between {lo2} and {hi2} are fairly common (95% probability), "
            f"while values outside {lo3} to {hi3} are rare (less than 0.3% probability). "
            f"Most samples will be close to {mean}; large deviations are possible but uncommon. "
            f"Output the sample as a number with exactly {decimals} decimal places. "
            f"Output only the number."
        )

    if prompt_type == "explanatory_4":
        lo1, hi1 = round(mean - std, 4), round(mean + std, 4)
        lo2, hi2 = round(mean - 2 * std, 4), round(mean + 2 * std, 4)
        lo3, hi3 = round(mean - 3 * std, 4), round(mean + 3 * std, 4)

        return (
            f"You are drawing one independent sample from a Normal (Gaussian) distribution "
            f"with mean {mean} and standard deviation {std}{sc}. "
            f"Key properties of this distribution:\n"
            f"- It is perfectly symmetric around its mean of {mean}, "
            f"so positive and negative deviations from the mean are equally likely.\n"
            f"- About 68% of samples fall between {lo1} and {hi1}.\n"
            f"- About 95% of samples fall between {lo2} and {hi2}.\n"
            f"- About 99.7% of samples fall between {lo3} and {hi3}.\n"
            f"- Values far from {mean} (beyond ±{3 * std} from the mean) are very rare "
            f"but not impossible.\n"
            f"- Each draw is independent; the value you produce should not be "
            f"influenced by any previous samples.\n"
            f"Output the sample as a number with exactly {decimals} decimal places. "
            f"Output only the number."
        )

    if prompt_type == "cot":
        return (
            f"You need to sample one number from a Normal distribution "
            f"with mean {mean} and standard deviation {std}{sc}. "
            f"Think step by step: consider the mean, the spread, and where "
            f"a typical sample would fall. Then output ONLY the sampled number "
            f"with exactly {decimals} decimal places on the final line. "
            f"Output only the number."
        )

    if prompt_type in {"icl", "icl_cot"}:
        examples = _icl_examples(
            spec=spec,
            decimals=decimals,
            n_examples=icl_n_examples,
            lower=lower,
            upper=upper,
            seed=icl_seed,
        )
        example_block = "\n".join(examples)

        if prompt_type == "icl":
            return (
                f"You are sampling from a Normal distribution with mean {mean} "
                f"and standard deviation {std}{sc}. "
                f"Here are {icl_n_examples} example samples from this distribution:\n"
                f"{example_block}\n"
                f"Now produce one new independent sample in the same format "
                f"(exactly {decimals} decimal places). Output only the number."
            )

        if prompt_type == "icl_cot":
            return (
                f"You are sampling from a Normal distribution with mean {mean} "
                f"and standard deviation {std}{sc}. "
                f"Here are {icl_n_examples} example samples from this distribution:\n"
                f"{example_block}\n"
                f"Think step by step about where the next sample should fall "
                f"relative to the mean and spread shown above. "
                f"Then output ONLY one new sample with exactly {decimals} decimal places. "
                f"Output only the number."
            )

    raise ValueError(f"Unknown prompt_type: {prompt_type!r}")

def build_uniform_prompt(
    spec: DistributionSpec,
    decimals: int,
    prompt_type: str,
    support_mode: str,
    lower,
    upper,
    icl_n_examples: int,
    icl_seed: int,
) -> str:
    """
    Uniform prompt family with the same information ladder as Normal.
    """

    low = spec.params["low"]
    high = spec.params["high"]

    support = _support_text(support_mode, lower, upper)
    sc = f" {support}" if support else ""

    if prompt_type == "short":
        return (
            f"Sample one number from Uniform({low}, {high}){sc}. "
            f"Use exactly {decimals} decimal places. "
            f"Output only the number."
        )

    if prompt_type == "plain":
        return (
            f"You are generating from a Uniform distribution between {low} and {high}{sc}. "
            f"Return one independent sample as a number with exactly {decimals} decimal places. "
            f"Output only the number."
        )

    if prompt_type == "formal":
        return (
            f"Draw one independent sample from a continuous Uniform distribution "
            f"on the interval [{low}, {high}]{sc}. "
            f"Return the answer as a number with exactly {decimals} decimal places. "
            f"Output only the number."
        )

    if prompt_type == "explanatory_1":
        return (
            f"You are drawing a sample from a continuous Uniform distribution "
            f"between {low} and {high}{sc}. "
            f"Every value in this interval is equally likely in terms of probability density. "
            f"Output the sample as a number with exactly {decimals} decimal places. "
            f"Output only the number."
        )

    if prompt_type == "explanatory_2":
        mid = (low + high) / 2

        return (
            f"You are drawing a sample from a continuous Uniform distribution "
            f"on [{low}, {high}]{sc}. "
            f"The probability density is constant across the whole interval. "
            f"Values near {low}, near {mid}, and near {high} are not preferred by the distribution; "
            f"equal-length subintervals have equal probability. "
            f"Output the sample as a number with exactly {decimals} decimal places. "
            f"Output only the number."
        )

    if prompt_type == "explanatory_3":
        mid = (low + high) / 2

        return (
            f"You are drawing a sample from a continuous Uniform distribution "
            f"on [{low}, {high}]{sc}. "
            f"The distribution has hard bounds: values below {low} and above {high} "
            f"have probability zero. "
            f"Within the interval, the density is flat: every interval of the same length "
            f"has the same probability. "
            f"There is no central peak around {mid}; the middle is not more likely than the edges. "
            f"Output the sample as a number with exactly {decimals} decimal places. "
            f"Output only the number."
        )

    if prompt_type == "explanatory_4":
        mid = (low + high) / 2
        q1 = low + 0.25 * (high - low)
        q3 = low + 0.75 * (high - low)

        return (
            f"You are drawing one independent sample from a continuous Uniform distribution "
            f"on the interval [{low}, {high}]{sc}. "
            f"Key properties of this distribution:\n"
            f"- The support is exactly [{low}, {high}]. Values outside this interval are impossible.\n"
            f"- The probability density is constant across the interval.\n"
            f"- Equal-length intervals have equal probability.\n"
            f"- The distribution does not concentrate around the centre {mid}; "
            f"values near the lower bound, middle, and upper bound are all generated according to the same flat density.\n"
            f"- About 25% of values fall below {q1}, about 50% fall below {mid}, "
            f"and about 75% fall below {q3}.\n"
            f"- Each draw is independent; the value you produce should not be influenced by previous samples.\n"
            f"Output the sample as a number with exactly {decimals} decimal places. "
            f"Output only the number."
        )

    if prompt_type == "cot":
        return (
            f"You need to sample one number from a continuous Uniform distribution "
            f"on [{low}, {high}]{sc}. "
            f"Think step by step: consider the hard lower and upper bounds, and remember that "
            f"all equal-length intervals inside the support have equal probability. "
            f"Then output ONLY the sampled number with exactly {decimals} decimal places on the final line. "
            f"Output only the number."
        )

    if prompt_type in {"icl", "icl_cot"}:
        examples = _icl_examples(
            spec=spec,
            decimals=decimals,
            n_examples=icl_n_examples,
            lower=lower,
            upper=upper,
            seed=icl_seed,
        )
        example_block = "\n".join(examples)

        if prompt_type == "icl":
            return (
                f"You are sampling from a continuous Uniform distribution "
                f"on [{low}, {high}]{sc}. "
                f"Here are {icl_n_examples} example samples from this distribution:\n"
                f"{example_block}\n"
                f"Now produce one new independent sample in the same format "
                f"(exactly {decimals} decimal places). Output only the number."
            )

        if prompt_type == "icl_cot":
            return (
                f"You are sampling from a continuous Uniform distribution "
                f"on [{low}, {high}]{sc}. "
                f"Here are {icl_n_examples} example samples from this distribution:\n"
                f"{example_block}\n"
                f"Think step by step about the support and the flat density of the distribution. "
                f"Then output ONLY one new independent sample with exactly {decimals} decimal places. "
                f"Output only the number."
            )

    raise ValueError(f"Unknown prompt_type: {prompt_type!r}")

def build_exponential_prompt(
    spec: DistributionSpec,
    decimals: int,
    prompt_type: str,
    support_mode: str,
    lower,
    upper,
    icl_n_examples: int,
    icl_seed: int,
) -> str:
    """
    Exponential prompt family with the same information ladder as Normal/Uniform.

    Exponential(rate) has:
    - support [0, infinity)
    - density highest near 0
    - right-skewed shape
    - mean = 1 / rate
    - standard deviation = 1 / rate
    """

    rate = spec.params["rate"]
    mean = 1.0 / rate

    support = _support_text(support_mode, lower, upper)
    sc = f" {support}" if support else ""

    if prompt_type == "short":
        return (
            f"Sample one number from Exponential(rate={rate}){sc}. "
            f"Use exactly {decimals} decimal places. "
            f"Output only the number."
        )

    if prompt_type == "plain":
        return (
            f"You are generating from an Exponential distribution with rate {rate}{sc}. "
            f"Return one independent sample as a number with exactly {decimals} decimal places. "
            f"Output only the number."
        )

    if prompt_type == "formal":
        return (
            f"Draw one independent sample from an Exponential distribution "
            f"with rate parameter {rate}{sc}. "
            f"Return the answer as a number with exactly {decimals} decimal places. "
            f"Output only the number."
        )

    if prompt_type == "explanatory_1":
        return (
            f"You are drawing a sample from an Exponential distribution "
            f"with rate {rate}{sc}. "
            f"This distribution is non-negative and right-skewed. "
            f"Output the sample as a number with exactly {decimals} decimal places. "
            f"Output only the number."
        )

    if prompt_type == "explanatory_2":
        return (
            f"You are drawing a sample from an Exponential distribution "
            f"with rate {rate}{sc}. "
            f"The support is non-negative: values below 0 are impossible. "
            f"The density is highest near 0 and decreases as the value becomes larger. "
            f"The mean of this distribution is {mean:.4f}. "
            f"Output the sample as a number with exactly {decimals} decimal places. "
            f"Output only the number."
        )

    if prompt_type == "explanatory_3":
        q50 = 0.6931471805599453 / rate
        q90 = 2.302585092994046 / rate
        q95 = 2.995732273553991 / rate

        return (
            f"You are drawing a sample from an Exponential distribution "
            f"with rate {rate}{sc}. "
            f"The distribution is right-skewed: small positive values are common, "
            f"and large values are possible but increasingly rare. "
            f"Values below 0 have probability zero. "
            f"The median is about {q50:.4f}, about 90% of values fall below {q90:.4f}, "
            f"and about 95% of values fall below {q95:.4f}. "
            f"Output the sample as a number with exactly {decimals} decimal places. "
            f"Output only the number."
        )

    if prompt_type == "explanatory_4":
        q50 = 0.6931471805599453 / rate
        q90 = 2.302585092994046 / rate
        q95 = 2.995732273553991 / rate
        q99 = 4.605170185988092 / rate

        return (
            f"You are drawing one independent sample from an Exponential distribution "
            f"with rate {rate}{sc}. "
            f"Key properties of this distribution:\n"
            f"- The support is [0, infinity). Values below 0 are impossible.\n"
            f"- The density is highest near 0 and decreases continuously as the value increases.\n"
            f"- The distribution is right-skewed: small values are common, while large values are rare but possible.\n"
            f"- The mean is {mean:.4f}, and the standard deviation is also {mean:.4f}.\n"
            f"- About 50% of values fall below {q50:.4f}.\n"
            f"- About 90% of values fall below {q90:.4f}.\n"
            f"- About 95% of values fall below {q95:.4f}.\n"
            f"- About 99% of values fall below {q99:.4f}.\n"
            f"- Each draw is independent; the value you produce should not be influenced by previous samples.\n"
            f"Output the sample as a number with exactly {decimals} decimal places. "
            f"Output only the number."
        )

    if prompt_type == "cot":
        return (
            f"You need to sample one number from an Exponential distribution "
            f"with rate {rate}{sc}. "
            f"Think step by step: values must be non-negative, small values are more common, "
            f"and larger values become increasingly rare. "
            f"Then output ONLY the sampled number with exactly {decimals} decimal places on the final line. "
            f"Output only the number."
        )

    if prompt_type in {"icl", "icl_cot"}:
        examples = _icl_examples(
            spec=spec,
            decimals=decimals,
            n_examples=icl_n_examples,
            lower=lower,
            upper=upper,
            seed=icl_seed,
        )
        example_block = "\n".join(examples)

        if prompt_type == "icl":
            return (
                f"You are sampling from an Exponential distribution "
                f"with rate {rate}{sc}. "
                f"Here are {icl_n_examples} example samples from this distribution:\n"
                f"{example_block}\n"
                f"Now produce one new independent sample in the same format "
                f"(exactly {decimals} decimal places). Output only the number."
            )

        if prompt_type == "icl_cot":
            return (
                f"You are sampling from an Exponential distribution "
                f"with rate {rate}{sc}. "
                f"Here are {icl_n_examples} example samples from this distribution:\n"
                f"{example_block}\n"
                f"Think step by step about the non-negative support and the right-skewed shape. "
                f"Small values are common, and large values are increasingly rare. "
                f"Then output ONLY one new independent sample with exactly {decimals} decimal places. "
                f"Output only the number."
            )

    raise ValueError(f"Unknown prompt_type: {prompt_type!r}")

def build_beta_prompt(
    spec: DistributionSpec,
    decimals: int,
    prompt_type: str,
    support_mode: str,
    lower,
    upper,
    icl_n_examples: int,
    icl_seed: int,
) -> str:
    """
    Beta prompt family with the same information ladder as Normal/Uniform/Exponential.

    Beta(alpha, beta) has:
    - support [0, 1]
    - shape controlled by alpha and beta
    - for Beta(2,2), it is symmetric and centre-heavy around 0.5
    """

    alpha = spec.params["alpha"]
    beta = spec.params["beta"]

    support = _support_text(support_mode, lower, upper)
    sc = f" {support}" if support else ""

    # Simple shape description for common cases
    if alpha == beta:
        if alpha > 1:
            shape_text = (
                f"It is symmetric around 0.5 and has more density near the centre "
                f"than near 0 or 1."
            )
        elif alpha == 1:
            shape_text = (
                f"It is equivalent to a Uniform distribution on [0, 1]."
            )
        else:
            shape_text = (
                f"It is symmetric and U-shaped, with more density near 0 and 1 "
                f"than near the centre."
            )
    elif alpha > beta:
        shape_text = (
            f"It is skewed toward 1, so larger values are generally more likely "
            f"than smaller values."
        )
    else:
        shape_text = (
            f"It is skewed toward 0, so smaller values are generally more likely "
            f"than larger values."
        )

    mean = alpha / (alpha + beta)

    if prompt_type == "short":
        return (
            f"Sample one number from Beta({alpha}, {beta}){sc}. "
            f"Use exactly {decimals} decimal places. "
            f"Output only the number."
        )

    if prompt_type == "plain":
        return (
            f"You are generating from a Beta distribution with alpha {alpha} "
            f"and beta {beta}{sc}. "
            f"Return one independent sample as a number with exactly {decimals} decimal places. "
            f"Output only the number."
        )

    if prompt_type == "formal":
        return (
            f"Draw one independent sample from a continuous Beta distribution "
            f"with shape parameters alpha={alpha} and beta={beta}{sc}. "
            f"Return the answer as a number with exactly {decimals} decimal places. "
            f"Output only the number."
        )

    if prompt_type == "explanatory_1":
        return (
            f"You are drawing a sample from a continuous Beta distribution "
            f"with alpha {alpha} and beta {beta}{sc}. "
            f"The distribution is supported on the interval [0, 1]. "
            f"Output the sample as a number with exactly {decimals} decimal places. "
            f"Output only the number."
        )

    if prompt_type == "explanatory_2":
        return (
            f"You are drawing a sample from a continuous Beta distribution "
            f"with alpha {alpha} and beta {beta}{sc}. "
            f"The support is [0, 1], so values below 0 or above 1 are impossible. "
            f"{shape_text} "
            f"The mean of this distribution is {mean:.4f}. "
            f"Output the sample as a number with exactly {decimals} decimal places. "
            f"Output only the number."
        )

    if prompt_type == "explanatory_3":
        return (
            f"You are drawing a sample from a continuous Beta distribution "
            f"with alpha {alpha} and beta {beta}{sc}. "
            f"The support is [0, 1]. "
            f"{shape_text} "
            f"For Beta({alpha}, {beta}), the parameters determine how much mass is near "
            f"0, near 1, or near the centre. "
            f"Because the support is bounded, samples must stay between 0 and 1. "
            f"Output the sample as a number with exactly {decimals} decimal places. "
            f"Output only the number."
        )

    if prompt_type == "explanatory_4":
        return (
            f"You are drawing one independent sample from a continuous Beta distribution "
            f"with shape parameters alpha={alpha} and beta={beta}{sc}. "
            f"Key properties of this distribution:\n"
            f"- The support is exactly [0, 1]. Values below 0 and above 1 are impossible.\n"
            f"- The alpha and beta parameters control the shape of the density.\n"
            f"- For this parameter setting, {shape_text}\n"
            f"- The mean is alpha / (alpha + beta) = {mean:.4f}.\n"
            f"- A draw should follow the density shape, not just choose any number uniformly from [0, 1].\n"
            f"- Each draw is independent; the value you produce should not be influenced by previous samples.\n"
            f"Output the sample as a number with exactly {decimals} decimal places. "
            f"Output only the number."
        )

    if prompt_type == "cot":
        return (
            f"You need to sample one number from a Beta distribution "
            f"with alpha {alpha} and beta {beta}{sc}. "
            f"Think step by step: values must lie between 0 and 1, and the shape of the "
            f"density is determined by alpha and beta. {shape_text} "
            f"Then output ONLY the sampled number with exactly {decimals} decimal places on the final line. "
            f"Output only the number."
        )

    if prompt_type in {"icl", "icl_cot"}:
        examples = _icl_examples(
            spec=spec,
            decimals=decimals,
            n_examples=icl_n_examples,
            lower=lower,
            upper=upper,
            seed=icl_seed,
        )
        example_block = "\n".join(examples)

        if prompt_type == "icl":
            return (
                f"You are sampling from a continuous Beta distribution "
                f"with alpha {alpha} and beta {beta}{sc}. "
                f"Here are {icl_n_examples} example samples from this distribution:\n"
                f"{example_block}\n"
                f"Now produce one new independent sample in the same format "
                f"(exactly {decimals} decimal places). Output only the number."
            )

        if prompt_type == "icl_cot":
            return (
                f"You are sampling from a continuous Beta distribution "
                f"with alpha {alpha} and beta {beta}{sc}. "
                f"Here are {icl_n_examples} example samples from this distribution:\n"
                f"{example_block}\n"
                f"Think step by step about the bounded support [0, 1] and the density shape. "
                f"{shape_text} "
                f"Then output ONLY one new independent sample with exactly {decimals} decimal places. "
                f"Output only the number."
            )

    raise ValueError(f"Unknown prompt_type: {prompt_type!r}")

def build_laplace_prompt(
    spec: DistributionSpec,
    decimals: int,
    prompt_type: str,
    support_mode: str,
    lower,
    upper,
    icl_n_examples: int,
    icl_seed: int,
) -> str:
    """
    Laplace prompt family with the same information ladder.

    Laplace(loc, scale) has:
    - support over all real numbers
    - symmetry around loc
    - sharper peak than Normal at the centre
    - heavier tails than Normal
    """

    loc = spec.params["loc"]
    scale = spec.params["scale"]

    support = _support_text(support_mode, lower, upper)
    sc = f" {support}" if support else ""

    q25 = loc - scale * 0.6931471805599453
    q50 = loc
    q75 = loc + scale * 0.6931471805599453

    if prompt_type == "short":
        return (
            f"Sample one number from Laplace({loc}, {scale}){sc}. "
            f"Use exactly {decimals} decimal places. "
            f"Output only the number."
        )

    if prompt_type == "plain":
        return (
            f"You are generating from a Laplace distribution with location {loc} "
            f"and scale {scale}{sc}. "
            f"Return one independent sample as a number with exactly {decimals} decimal places. "
            f"Output only the number."
        )

    if prompt_type == "formal":
        return (
            f"Draw one independent sample from a Laplace distribution "
            f"with location parameter {loc} and scale parameter {scale}{sc}. "
            f"Return the answer as a number with exactly {decimals} decimal places. "
            f"Output only the number."
        )

    if prompt_type == "explanatory_1":
        return (
            f"You are drawing a sample from a Laplace distribution "
            f"with location {loc} and scale {scale}{sc}. "
            f"The distribution is symmetric around {loc}. "
            f"Output the sample as a number with exactly {decimals} decimal places. "
            f"Output only the number."
        )

    if prompt_type == "explanatory_2":
        return (
            f"You are drawing a sample from a Laplace distribution "
            f"with location {loc} and scale {scale}{sc}. "
            f"The distribution is symmetric around {loc}, has its highest density at {loc}, "
            f"and decreases exponentially as values move away from {loc}. "
            f"Compared with a Normal distribution, it has a sharper peak and heavier tails. "
            f"Output the sample as a number with exactly {decimals} decimal places. "
            f"Output only the number."
        )

    if prompt_type == "explanatory_3":
        return (
            f"You are drawing a sample from a Laplace distribution "
            f"with location {loc} and scale {scale}{sc}. "
            f"The distribution is centred at {loc} and is symmetric: negative and positive "
            f"deviations from {loc} are equally likely. "
            f"It has a sharp peak at the centre and heavier tails than a Normal distribution, "
            f"so values close to {loc} are common, but larger deviations are also possible. "
            f"The median is {q50:.4f}, the lower quartile is about {q25:.4f}, "
            f"and the upper quartile is about {q75:.4f}. "
            f"Output the sample as a number with exactly {decimals} decimal places. "
            f"Output only the number."
        )

    if prompt_type == "explanatory_4":
        return (
            f"You are drawing one independent sample from a Laplace distribution "
            f"with location {loc} and scale {scale}{sc}. "
            f"Key properties of this distribution:\n"
            f"- The distribution is symmetric around its location parameter {loc}.\n"
            f"- The highest density occurs at {loc}.\n"
            f"- Probability decreases exponentially as values move away from {loc}.\n"
            f"- Compared with a Normal distribution, the Laplace distribution has a sharper central peak.\n"
            f"- Compared with a Normal distribution, it also has heavier tails, so larger deviations are more likely.\n"
            f"- The lower quartile is about {q25:.4f}, the median is {q50:.4f}, "
            f"and the upper quartile is about {q75:.4f}.\n"
            f"- Each draw is independent; the value you produce should not be influenced by previous samples.\n"
            f"Output the sample as a number with exactly {decimals} decimal places. "
            f"Output only the number."
        )

    if prompt_type == "cot":
        return (
            f"You need to sample one number from a Laplace distribution "
            f"with location {loc} and scale {scale}{sc}. "
            f"Think step by step: the distribution is symmetric around {loc}, "
            f"has a sharp central peak, and has heavier tails than a Normal distribution. "
            f"Then output ONLY the sampled number with exactly {decimals} decimal places on the final line. "
            f"Output only the number."
        )

    if prompt_type in {"icl", "icl_cot"}:
        examples = _icl_examples(
            spec=spec,
            decimals=decimals,
            n_examples=icl_n_examples,
            lower=lower,
            upper=upper,
            seed=icl_seed,
        )
        example_block = "\n".join(examples)

        if prompt_type == "icl":
            return (
                f"You are sampling from a Laplace distribution "
                f"with location {loc} and scale {scale}{sc}. "
                f"Here are {icl_n_examples} example samples from this distribution:\n"
                f"{example_block}\n"
                f"Now produce one new independent sample in the same format "
                f"(exactly {decimals} decimal places). Output only the number."
            )

        if prompt_type == "icl_cot":
            return (
                f"You are sampling from a Laplace distribution "
                f"with location {loc} and scale {scale}{sc}. "
                f"Here are {icl_n_examples} example samples from this distribution:\n"
                f"{example_block}\n"
                f"Think step by step about the symmetric shape, the sharp central peak, "
                f"and the heavier tails compared with a Normal distribution. "
                f"Then output ONLY one new independent sample with exactly {decimals} decimal places. "
                f"Output only the number."
            )

    raise ValueError(f"Unknown prompt_type: {prompt_type!r}")

if __name__ == "__main__":
    print("=== Normal prompt check ===")
    print(
        build_prompt(
            distribution="normal",
            params={"mean": 0.0, "std": 1.0},
            decimals=3,
            prompt_type="explanatory_4",
        )
    )

    print("\n=== Uniform prompt check ===")
    print(
        build_prompt(
            distribution="uniform",
            params={"low": 0.0, "high": 1.0},
            decimals=3,
            prompt_type="explanatory_4",
            support_mode="bounded",
            lower=0.0,
            upper=1.0,
        )
    )

    print("\n=== Exponential prompt check ===")
    print(
        build_prompt(
            distribution="exponential",
            params={"rate": 1.0},
            decimals=3,
            prompt_type="explanatory_4",
            support_mode="positive",
            lower=0.0,
        )
    )

    print("\n=== Beta prompt check ===")
    print(
        build_prompt(
        distribution="beta",
        params={"alpha": 2.0, "beta": 2.0},
        decimals=3,
        prompt_type="explanatory_4",
        support_mode="bounded",
        lower=0.0,
        upper=1.0,
        )
    )

    print("\n=== Laplace prompt check ===")
    print(
        build_prompt(
            distribution="laplace",
            params={"loc": 0.0, "scale": 1.0},
            decimals=3,
            prompt_type="explanatory_4",
            support_mode="agnostic",
            lower=None,
            upper=None,
        )
    )