from __future__ import annotations

import random
from scipy.stats import beta as beta_dist

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
   

    if n_examples < 1:
        raise ValueError("n_examples must be at least 1")

    pool_size = max(10_000, n_examples * 2_000)

    samples = sample_distribution(
        spec=spec,
        n_samples=pool_size,
        lower=lower,
        upper=upper,
        seed=seed,
    )

    ordered_samples = sorted(float(x) for x in samples)

    probabilities = [
        (i + 1) / (n_examples + 1)
        for i in range(n_examples)
    ]

    examples = [
        ordered_samples[
            round(probability * (len(ordered_samples) - 1))
        ]
        for probability in probabilities
    ]

    rng = random.Random(seed)
    rng.shuffle(examples)

    return [f"{x:.{decimals}f}" for x in examples]

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

    q25 = mean - 0.6744897501960817 * std
    q50 = mean
    q75 = mean + 0.6744897501960817 * std

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
            f"You are drawing one independent sample from a Normal (Gaussian) distribution "
            f"with mean {mean} and standard deviation {std}{sc}. "
            f"The natural support of this distribution is all real numbers, "
            f"so the sample may be negative, zero, or positive. "
            f"Output the sample as a number with exactly {decimals} decimal places. "
            f"Output only the number."
        )

    if prompt_type == "explanatory_2":
        return (
        f"You are drawing one independent sample from a Normal (Gaussian) distribution "
        f"with mean {mean} and standard deviation {std}{sc}. "
        f"The natural support of this distribution is all real numbers, "
        f"so the sample may be negative, zero, or positive. "
        f"The distribution is bell-shaped and symmetric around its mean of {mean}. "
        f"Values closer to the mean have higher probability density, "
        f"while values farther from the mean have lower probability density. "
        f"Output the sample as a number with exactly {decimals} decimal places. "
        f"Output only the number."
    )

    if prompt_type == "explanatory_3":
        return (
        f"You are drawing one independent sample from a Normal (Gaussian) distribution "
        f"with mean {mean} and standard deviation {std}{sc}. "
        f"The natural support of this distribution is all real numbers, "
        f"so the sample may be negative, zero, or positive. "
        f"The distribution is bell-shaped and symmetric around its mean of {mean}. "
        f"The mean {mean} represents the centre of the distribution, "
        f"and the standard deviation {std} describes the typical spread around that centre. "
        f"Values closer to the mean have higher probability density, "
        f"while values farther away are less likely but still possible. "
        f"Output the sample as a number with exactly {decimals} decimal places. "
        f"Output only the number."
    )

    if prompt_type == "explanatory_4":
        return (
        f"You are drawing one independent sample from a Normal (Gaussian) distribution "
        f"with mean {mean} and standard deviation {std}{sc}. "
        f"The natural support of this distribution is all real numbers, "
        f"so the sample may be negative, zero, or positive. "
        f"The distribution is bell-shaped and symmetric around its mean of {mean}. "
        f"The mean {mean} represents the centre of the distribution, "
        f"and the standard deviation {std} describes the typical spread around that centre. "
        f"Values closer to the mean have higher probability density, "
        f"while values farther away are less likely but still possible. "
        f"About 25% of values fall below {q25:.4f}, "
        f"about 50% fall below {q50:.4f}, "
        f"and about 75% fall below {q75:.4f}. "
        f"Output the sample as a number with exactly {decimals} decimal places. "
        f"Output only the number."
    )

    if prompt_type == "cot":
        return (
        f"You need to draw one independent sample from a Normal distribution "
        f"with mean {mean} and standard deviation {std}{sc}. "
        f"Before answering, reason internally step by step about the distribution's "
        f"support, shape, and allocation of probability mass. "
        f"Do not output the reasoning. "
        f"Output only one sampled number with exactly {decimals} decimal places."
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
                f"Here are {icl_n_examples} representative example values "
                f"from this distribution:\n"
                f"{example_block}\n"
                f"Now produce one new independent sample in the same format "
                f"(exactly {decimals} decimal places). Output only the number."
            )

        if prompt_type == "icl_cot":
            return (
                f"You are sampling from a Normal distribution with mean {mean} "
                f"and standard deviation {std}{sc}. "
                f"Here are {icl_n_examples} representative example values "
                f"from this distribution:\n"
                f"{example_block}\n"
                f"Before answering, reason internally step by step about the distribution's "
                f"support, shape, and allocation of probability mass, using the examples "
                f"as additional context. "
                f"Do not output the reasoning. "
                f"Output only one new independent sample with exactly "
                f"{decimals} decimal places."
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
    if (
        support_mode == "bounded"
        and lower == low
        and upper == high
    ):
        support = ""


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
            f"You are drawing one independent sample from a continuous Uniform distribution "
            f"on the interval [{low}, {high}]{sc}. "
            f"The natural support of this distribution is exactly [{low}, {high}], "
            f"so values outside this interval are impossible. "
            f"Output the sample as a number with exactly {decimals} decimal places. "
            f"Output only the number."
        )

    if prompt_type == "explanatory_2":
        mid = (low + high) / 2

        return (
        f"You are drawing one independent sample from a continuous Uniform distribution "
        f"on the interval [{low}, {high}]{sc}. "
        f"The natural support of this distribution is exactly [{low}, {high}], "
        f"so values outside this interval are impossible. "
        f"The probability density is constant across the entire interval. "
        f"Equal-length subintervals have equal probability, and values near the centre "
        f"{mid} are not more likely than values near either boundary. "
        f"Output the sample as a number with exactly {decimals} decimal places. "
        f"Output only the number."
    )

    if prompt_type == "explanatory_3":
        mid = (low + high) / 2
        spread = (high - low) / (12 ** 0.5)

        return (
        f"You are drawing one independent sample from a continuous Uniform distribution "
        f"on the interval [{low}, {high}]{sc}. "
        f"The natural support of this distribution is exactly [{low}, {high}], "
        f"so values outside this interval are impossible. "
        f"The probability density is constant across the entire interval. "
        f"Equal-length subintervals have equal probability, and the distribution "
        f"does not favour values near the centre over values near the boundaries. "
        f"The mean {mid:.4f} represents the centre of the interval, "
        f"and the standard deviation {spread:.4f} describes its spread. "
        f"Output the sample as a number with exactly {decimals} decimal places. "
        f"Output only the number."
    )

    if prompt_type == "explanatory_4":
        mid = (low + high) / 2
        spread = (high - low) / (12 ** 0.5)

        q25 = low + 0.25 * (high - low)
        q50 = mid
        q75 = low + 0.75 * (high - low)

        return (
        f"You are drawing one independent sample from a continuous Uniform distribution "
        f"on the interval [{low}, {high}]{sc}. "
        f"The natural support of this distribution is exactly [{low}, {high}], "
        f"so values outside this interval are impossible. "
        f"The probability density is constant across the entire interval. "
        f"Equal-length subintervals have equal probability, and the distribution "
        f"does not favour values near the centre over values near the boundaries. "
        f"The mean {mid:.4f} represents the centre of the interval, "
        f"and the standard deviation {spread:.4f} describes its spread. "
        f"About 25% of values fall below {q25:.4f}, "
        f"about 50% fall below {q50:.4f}, "
        f"and about 75% fall below {q75:.4f}. "
        f"Output the sample as a number with exactly {decimals} decimal places. "
        f"Output only the number."
    )

    if prompt_type == "cot":
        return (
        f"You need to draw one independent sample from a continuous Uniform distribution "
        f"on the interval [{low}, {high}]{sc}. "
        f"Before answering, reason internally step by step about the distribution's "
        f"support, shape, and allocation of probability mass. "
        f"Do not output the reasoning. "
        f"Output only one sampled number with exactly {decimals} decimal places."
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
                f"Here are {icl_n_examples} representative example values "
                f"from this distribution:\n"
                f"{example_block}\n"
                f"Now produce one new independent sample in the same format "
                f"(exactly {decimals} decimal places). Output only the number."
            )

        if prompt_type == "icl_cot":
            return (
                f"You are sampling from a continuous Uniform distribution "
                f"on [{low}, {high}]{sc}. "
                f"Here are {icl_n_examples} representative example values "
                f"from this distribution:\n"
                f"{example_block}\n"
                f"Before answering, reason internally step by step about the distribution's "
                f"support, shape, and allocation of probability mass, using the examples "
                f"as additional context. "
                f"Do not output the reasoning. "
                f"Output only one new independent sample with exactly "
                f"{decimals} decimal places."
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
    if (
        support_mode == "positive"
        and (lower is None or lower == 0)
        and upper is None
    ):
        support = ""

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
            f"You are drawing one independent sample from an Exponential distribution "
            f"with rate {rate}{sc}. "
            f"The natural support of this distribution is [0, infinity), "
            f"so negative values are impossible. "
            f"Output the sample as a number with exactly {decimals} decimal places. "
            f"Output only the number."
        )

    if prompt_type == "explanatory_2":
        return (
        f"You are drawing one independent sample from an Exponential distribution "
        f"with rate {rate}{sc}. "
        f"The natural support of this distribution is [0, infinity), "
        f"so negative values are impossible. "
        f"The distribution is right-skewed. "
        f"The probability density is highest near 0 and decreases continuously "
        f"as the value becomes larger. "
        f"Small positive values are therefore more common, while large values "
        f"are possible but increasingly unlikely. "
        f"Output the sample as a number with exactly {decimals} decimal places. "
        f"Output only the number."
    )

    if prompt_type == "explanatory_3":
        return (
        f"You are drawing one independent sample from an Exponential distribution "
        f"with rate {rate}{sc}. "
        f"The natural support of this distribution is [0, infinity), "
        f"so negative values are impossible. "
        f"The distribution is right-skewed. "
        f"The probability density is highest near 0 and decreases continuously "
        f"as the value becomes larger. "
        f"Small positive values are common, while large values are possible "
        f"but increasingly unlikely. "
        f"The mean is {mean:.4f}, and the standard deviation is also {mean:.4f}. "
        f"These values describe the average value and spread of the distribution. "
        f"Output the sample as a number with exactly {decimals} decimal places. "
        f"Output only the number."
    )

    if prompt_type == "explanatory_4":
        q25 = 0.2876820724517809 / rate
        q50 = 0.6931471805599453 / rate
        q75 = 1.3862943611198906 / rate

        return (
        f"You are drawing one independent sample from an Exponential distribution "
        f"with rate {rate}{sc}. "
        f"The natural support of this distribution is [0, infinity), "
        f"so negative values are impossible. "
        f"The distribution is right-skewed. "
        f"The probability density is highest near 0 and decreases continuously "
        f"as the value becomes larger. "
        f"Small positive values are common, while large values are possible "
        f"but increasingly unlikely. "
        f"The mean is {mean:.4f}, and the standard deviation is also {mean:.4f}. "
        f"These values describe the average value and spread of the distribution. "
        f"About 25% of values fall below {q25:.4f}, "
        f"about 50% fall below {q50:.4f}, "
        f"and about 75% fall below {q75:.4f}. "
        f"Output the sample as a number with exactly {decimals} decimal places. "
        f"Output only the number."
    )

    if prompt_type == "cot":
        return (
        f"You need to draw one independent sample from an Exponential distribution "
        f"with rate {rate}{sc}. "
        f"Before answering, reason internally step by step about the distribution's "
        f"support, shape, and allocation of probability mass. "
        f"Do not output the reasoning. "
        f"Output only one sampled number with exactly {decimals} decimal places."
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
                f"Here are {icl_n_examples} representative example values "
                f"from this distribution:\n"
                f"{example_block}\n"
                f"Now produce one new independent sample in the same format "
                f"(exactly {decimals} decimal places). Output only the number."
            )

        if prompt_type == "icl_cot":
            return (
                f"You are sampling from an Exponential distribution "
                f"with rate {rate}{sc}. "
                f"Here are {icl_n_examples} representative example values "
                f"from this distribution:\n"
                f"{example_block}\n"
                f"Before answering, reason internally step by step about the distribution's "
                f"support, shape, and allocation of probability mass, using the examples "
                f"as additional context. "
                f"Do not output the reasoning. "
                f"Output only one new independent sample with exactly "
                f"{decimals} decimal places."
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
    if (
        support_mode == "bounded"
        and lower == 0
        and upper == 1
    ):
        support = ""

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

    std = ( alpha * beta/ ((alpha + beta) ** 2 * (alpha + beta + 1))) ** 0.5

    q25 = beta_dist.ppf(0.25, alpha, beta)
    q50 = beta_dist.ppf(0.50, alpha, beta)
    q75 = beta_dist.ppf(0.75, alpha, beta)

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
        f"You are drawing one independent sample from a continuous Beta distribution "
        f"with alpha {alpha} and beta {beta}{sc}. "
        f"The natural support of this distribution is exactly [0, 1], "
        f"so values below 0 or above 1 are impossible. "
        f"Output the sample as a number with exactly {decimals} decimal places. "
        f"Output only the number."
    )
    if prompt_type == "explanatory_2":
        return (
        f"You are drawing one independent sample from a continuous Beta distribution "
        f"with alpha {alpha} and beta {beta}{sc}. "
        f"The natural support of this distribution is exactly [0, 1], "
        f"so values below 0 or above 1 are impossible. "
        f"{shape_text} "
        f"The probability density varies across the interval according to "
        f"the alpha and beta parameters. "
        f"Output the sample as a number with exactly {decimals} decimal places. "
        f"Output only the number."
    )

    if prompt_type == "explanatory_3":
        return (
        f"You are drawing one independent sample from a continuous Beta distribution "
        f"with alpha {alpha} and beta {beta}{sc}. "
        f"The natural support of this distribution is exactly [0, 1], "
        f"so values below 0 or above 1 are impossible. "
        f"{shape_text} "
        f"The probability density varies across the interval according to "
        f"the alpha and beta parameters. "
        f"The mean is {mean:.4f}, and the standard deviation is {std:.4f}. "
        f"These values describe the centre and spread of the distribution. "
        f"Output the sample as a number with exactly {decimals} decimal places. "
        f"Output only the number."
    )

    if prompt_type == "explanatory_4":
        return (
        f"You are drawing one independent sample from a continuous Beta distribution "
        f"with alpha {alpha} and beta {beta}{sc}. "
        f"The natural support of this distribution is exactly [0, 1], "
        f"so values below 0 or above 1 are impossible. "
        f"{shape_text} "
        f"The probability density varies across the interval according to "
        f"the alpha and beta parameters. "
        f"The mean is {mean:.4f}, and the standard deviation is {std:.4f}. "
        f"These values describe the centre and spread of the distribution. "
        f"About 25% of values fall below {q25:.4f}, "
        f"about 50% fall below {q50:.4f}, "
        f"and about 75% fall below {q75:.4f}. "
        f"Output the sample as a number with exactly {decimals} decimal places. "
        f"Output only the number."
    )

    if prompt_type == "cot":
        return (
        f"You need to draw one independent sample from a continuous Beta distribution "
        f"with alpha {alpha} and beta {beta}{sc}. "
        f"Before answering, reason internally step by step about the distribution's "
        f"support, shape, and allocation of probability mass. "
        f"Do not output the reasoning. "
        f"Output only one sampled number with exactly {decimals} decimal places."
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
                f"Here are {icl_n_examples} representative example values "
                f"from this distribution:\n"
                f"{example_block}\n"
                f"Now produce one new independent sample in the same format "
                f"(exactly {decimals} decimal places). Output only the number."
            )

        if prompt_type == "icl_cot":
            return (
                f"You are sampling from a continuous Beta distribution "
                f"with alpha {alpha} and beta {beta}{sc}. "
                f"Here are {icl_n_examples} representative example values "
                f"from this distribution:\n"
                f"{example_block}\n"
                f"Before answering, reason internally step by step about the distribution's "
                f"support, shape, and allocation of probability mass, using the examples "
                f"as additional context. "
                f"Do not output the reasoning. "
                f"Output only one new independent sample with exactly "
                f"{decimals} decimal places."
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
    std = scale * (2 ** 0.5)

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
        f"You are drawing one independent sample from a Laplace distribution "
        f"with location {loc} and scale {scale}{sc}. "
        f"The natural support of this distribution is all real numbers, "
        f"so the sample may be negative, zero, or positive. "
        f"Output the sample as a number with exactly {decimals} decimal places. "
        f"Output only the number."
    )

    if prompt_type == "explanatory_2":
        return (
        f"You are drawing one independent sample from a Laplace distribution "
        f"with location {loc} and scale {scale}{sc}. "
        f"The natural support of this distribution is all real numbers, "
        f"so the sample may be negative, zero, or positive. "
        f"The distribution is symmetric around its location parameter {loc}. "
        f"It has its highest probability density at {loc} and a sharp central peak. "
        f"The probability density decreases exponentially as values move away from {loc}. "
        f"Values farther from {loc} remain possible but become progressively less likely. "
        f"Output the sample as a number with exactly {decimals} decimal places. "
        f"Output only the number."
    )

    if prompt_type == "explanatory_3":
        return (
        f"You are drawing one independent sample from a Laplace distribution "
        f"with location {loc} and scale {scale}{sc}. "
        f"The natural support of this distribution is all real numbers, "
        f"so the sample may be negative, zero, or positive. "
        f"The distribution is symmetric around its location parameter {loc}. "
        f"It has its highest probability density at {loc} and a sharp central peak. "
        f"The probability density decreases exponentially as values move away from {loc}. "
        f"Values farther from {loc} remain possible but become progressively less likely. "
        f"The mean is {loc:.4f}, and the standard deviation is {std:.4f}. "
        f"These values describe the centre and spread of the distribution. "
        f"Output the sample as a number with exactly {decimals} decimal places. "
        f"Output only the number."
    )

    if prompt_type == "explanatory_4":
        return (
        f"You are drawing one independent sample from a Laplace distribution "
        f"with location {loc} and scale {scale}{sc}. "
        f"The natural support of this distribution is all real numbers, "
        f"so the sample may be negative, zero, or positive. "
        f"The distribution is symmetric around its location parameter {loc}. "
        f"It has its highest probability density at {loc} and a sharp central peak. "
        f"The probability density decreases exponentially as values move away from {loc}. "
        f"Values farther from {loc} remain possible but become progressively less likely. "
        f"The mean is {loc:.4f}, and the standard deviation is {std:.4f}. "
        f"These values describe the centre and spread of the distribution. "
        f"About 25% of values fall below {q25:.4f}, "
        f"about 50% fall below {q50:.4f}, "
        f"and about 75% fall below {q75:.4f}. "
        f"Output the sample as a number with exactly {decimals} decimal places. "
        f"Output only the number."
    )

    if prompt_type == "cot":
        return (
        f"You need to draw one independent sample from a Laplace distribution "
        f"with location {loc} and scale {scale}{sc}. "
        f"Before answering, reason internally step by step about the distribution's "
        f"support, shape, and allocation of probability mass. "
        f"Do not output the reasoning. "
        f"Output only one sampled number with exactly {decimals} decimal places."
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
                f"Here are {icl_n_examples} representative example values "
                f"from this distribution:\n"
                f"{example_block}\n"
                f"Now produce one new independent sample in the same format "
                f"(exactly {decimals} decimal places). Output only the number."
            )

        if prompt_type == "icl_cot":
            return (
                f"You are sampling from a Laplace distribution "
                f"with location {loc} and scale {scale}{sc}. "
                f"Here are {icl_n_examples} representative example values "
                f"from this distribution:\n"
                f"{example_block}\n"
                f"Before answering, reason internally step by step about the distribution's "
                f"support, shape, and allocation of probability mass, using the examples "
                f"as additional context. "
                f"Do not output the reasoning. "
                f"Output only one new independent sample with exactly "
                f"{decimals} decimal places."
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