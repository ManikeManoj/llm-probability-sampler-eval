import math
import csv
import os
import json

from dataclasses import dataclass
from typing import List
from truth_model_mc import build_truth_model
from truth_model_analytic import next_token_truth_distribution
from lm_next_token import build_prompt, next_token_distribution
from real_prefix_logic import valid_next_tokens, classify_prefix

@dataclass
class RunConfig:
    mean: float | None
    std: float | None
    lower: float | None
    upper: float | None
    n_samples: int
    decimals: int
    prefixes: List[str]
    mc_reliable_threshold: int
    token_level_csv: str
    prefix_summary_csv: str
    prompt_type: str
    support_mode: str = "agnostic"
    allow_negative: bool = True
    seed: int = 42
    run_id: str = "default_run"
    icl_n_examples: int = 5
    icl_seed: int = 0
    # New distribution-agnostic fields
    distribution: str = "normal"
    params: dict | None = None
    model_name: str = "Qwen/Qwen3-4B"

def align_three_distributions(
    mc_dist: dict[str, float],
    analytic_dist: dict[str, float],
    lm_dist: dict[str, float],
    token_set: list[str],
):

    aligned = []

    for tok in token_set:
        p_mc = mc_dist.get(tok, 0.0)
        p_analytic = analytic_dist.get(tok, 0.0)
        p_lm = lm_dist.get(tok, 0.0)
        aligned.append((tok, p_mc, p_analytic, p_lm))

    return aligned


def total_variation_distance_two_dicts(dist_a: dict[str, float], dist_b: dict[str, float], token_set: list[str]):
    """
    TV(P,Q) = 0.5 * sum |P - Q|
    """
    return 0.5 * sum(abs(dist_a.get(tok, 0.0) - dist_b.get(tok, 0.0)) for tok in token_set)


def kl_divergence(dist_p: dict[str, float], dist_q: dict[str, float], token_set: list[str], eps: float = 1e-12):
    """
    KL(P || Q) with epsilon smoothing.
    """
    kl = 0.0
    for tok in token_set:
        p = max(dist_p.get(tok, 0.0), eps)
        q = max(dist_q.get(tok, 0.0), eps)
        kl += p * math.log(p / q)
    return kl

def js_divergence(
    dist_a: dict[str, float],
    dist_b: dict[str, float],
    token_set: list[str],
    eps: float = 1e-12,
) -> float:

    m_dist = {}
    for tok in token_set:
        p = dist_a.get(tok, 0.0)
        q = dist_b.get(tok, 0.0)
        m_dist[tok] = 0.5 * (p + q)

    return 0.5 * kl_divergence(dist_a, m_dist, token_set, eps=eps) + 0.5 * kl_divergence(dist_b, m_dist, token_set, eps=eps)

def entropy(
    dist: dict[str, float],
    token_set: list[str],
    eps: float = 1e-12,
) -> float:

    h = 0.0
    for tok in token_set:
        p = max(dist.get(tok, 0.0), eps)
        h -= p * math.log(p)
    return h

def weighted_abs_rank_error(
    truth_dist: dict[str, float],
    candidate_dist: dict[str, float],
    token_set: list[str],
) -> float:
    
    truth_ranks = compute_token_ranks(truth_dist, token_set)
    candidate_ranks = compute_token_ranks(candidate_dist, token_set)

    error = 0.0
    for tok in token_set:
        w = truth_dist.get(tok, 0.0)
        error += w * abs(candidate_ranks[tok] - truth_ranks[tok])

    return error

def top_probability_summary(
    dist: dict[str, float],
    token_set: list[str],
) -> tuple[float, float, float, float]:

    probs = [dist.get(tok, 0.0) for tok in token_set]

    if len(probs) == 0:
        return 0.0, 0.0, 0.0, 0.0

    sorted_probs = sorted(probs, reverse=True)
    top1 = sorted_probs[0]
    top2 = sorted_probs[1] if len(sorted_probs) > 1 else 0.0
    max_minus_min = max(probs) - min(probs)

    return top1, top2, top1 - top2, max_minus_min

def classify_prefix_sharpness(
    mc_reliable: bool,
    truth_top1_minus_top2: float,
) -> str:

    if not mc_reliable:
        return "mc_unreliable"

    if truth_top1_minus_top2 >= 0.05:
        return "sharp"
    elif truth_top1_minus_top2 >= 0.005:
        return "semi_sharp"
    else:
        return "flat"

def average_ranks_from_scores(scores: list[float]) -> list[float]:

    indexed = list(enumerate(scores))
    indexed.sort(key=lambda x: x[1], reverse=True)  

    ranks = [0.0] * len(scores)
    i = 0

    while i < len(indexed):
        j = i
        while j + 1 < len(indexed) and indexed[j + 1][1] == indexed[i][1]:
            j += 1

        
        avg_rank = (i + 1 + j + 1) / 2.0

        for k in range(i, j + 1):
            original_idx = indexed[k][0]
            ranks[original_idx] = avg_rank

        i = j + 1

    return ranks


def pearson_correlation(x: list[float], y: list[float]) -> float:

    if len(x) != len(y) or len(x) == 0:
        return float("nan")

    mean_x = sum(x) / len(x)
    mean_y = sum(y) / len(y)

    num = sum((a - mean_x) * (b - mean_y) for a, b in zip(x, y))
    den_x = math.sqrt(sum((a - mean_x) ** 2 for a in x))
    den_y = math.sqrt(sum((b - mean_y) ** 2 for b in y))

    if den_x == 0.0 or den_y == 0.0:
        return float("nan")

    return num / (den_x * den_y)


def spearman_rank_correlation(
    dist_a: dict[str, float],
    dist_b: dict[str, float],
    token_set: list[str],
) -> float:

    scores_a = [dist_a.get(tok, 0.0) for tok in token_set]
    scores_b = [dist_b.get(tok, 0.0) for tok in token_set]

    ranks_a = average_ranks_from_scores(scores_a)
    ranks_b = average_ranks_from_scores(scores_b)

    return pearson_correlation(ranks_a, ranks_b)

def compute_token_ranks(dist: dict[str, float], token_set: list[str]) -> dict[str, float]:

    scores = [dist.get(tok, 0.0) for tok in token_set]
    ranks = average_ranks_from_scores(scores)
    return {tok: rank for tok, rank in zip(token_set, ranks)}


def pretty_print_comparison(prefix: str, aligned_rows):
    print(f"PREFIX: {repr(prefix)}")
    print(
        f"{'token':<8}"
        f"{'mc_truth':<15}"
        f"{'analytic_truth':<18}"
        f"{'lm_prob':<15}"
        f"{'rank_mc':<10}"
        f"{'rank_an':<10}"
        f"{'rank_lm':<10}"
        f"{'|mc-lm|':<15}"
        f"{'|an-lm|':<15}"
    )


    for tok, p_mc, p_analytic, p_lm, rank_mc, rank_analytic, rank_lm, _, _, _, _ in sorted(aligned_rows, key=lambda x: x[1], reverse=True):
        abs_diff_mc = abs(p_mc - p_lm)
        abs_diff_analytic = abs(p_analytic - p_lm)
        print(
            f"{tok:<8}"
            f"{p_mc:<15.6f}"
            f"{p_analytic:<18.6f}"
            f"{p_lm:<15.6f}"
            f"{rank_mc:<10.1f}"
            f"{rank_analytic:<10.1f}"
            f"{rank_lm:<10.1f}"
            f"{abs_diff_mc:<15.6f}"
            f"{abs_diff_analytic:<15.6f}"
        )


def compare_for_prefix(
    prefix: str,
    mc_truth_probs: dict[str, dict[str, float]],
    mc_truth_counts: dict[str, dict[str, int]],
    prompt: str,
    mean: float | None,
    std: float | None,
    lower: float | None,
    upper: float | None,
    mc_reliable_threshold: int,
    run_id: str,
    prompt_type: str,
    n_samples: int,
    decimals: int,
    allow_negative: bool,
    support_mode: str,
    icl_n_examples: int = 5,
    icl_seed: int = 0,
    distribution: str = "normal",
    params: dict | None = None,
    model_name: str = "Qwen/Qwen3-4B",
 ):
    """
    Compare:
    - Monte Carlo truth vs LM
    - Analytic truth vs LM
    - Monte Carlo truth vs Analytic truth

    Also records:
    - Monte Carlo support count for the prefix
    - reliability flag for Monte Carlo
    """
    token_set = valid_next_tokens(
    prefix=prefix,
    decimals=decimals,
    allow_negative=allow_negative,
    )

    prefix_kind = classify_prefix(prefix)

    mc_dist = mc_truth_probs.get(prefix, {})
    analytic_dist = next_token_truth_distribution(
    prefix=prefix,
    distribution=distribution,
    params=params,
    mu=mean,
    sigma=std,
    decimals=decimals,
    lower=lower,
    upper=upper,
    allow_negative=allow_negative,
    )


    lm_dist, _ = next_token_distribution(
        prompt=prompt,
        prefix=prefix,
        decimals=decimals,
        allow_negative=allow_negative,
        model_name=model_name,
    )




    rank_mc = compute_token_ranks(mc_dist, token_set)
    rank_analytic = compute_token_ranks(analytic_dist, token_set)
    rank_lm = compute_token_ranks(lm_dist, token_set)

    base_rows = align_three_distributions(mc_dist, analytic_dist, lm_dist, token_set)

    aligned_rows = []
    for tok, p_mc, p_analytic, p_lm in base_rows:
        aligned_rows.append((
            tok,
            p_mc,
            p_analytic,
            p_lm,
            rank_mc[tok],
            rank_analytic[tok],
            rank_lm[tok],
            rank_lm[tok] - rank_mc[tok],
            rank_lm[tok] - rank_analytic[tok],
            abs(rank_lm[tok] - rank_mc[tok]),
            abs(rank_lm[tok] - rank_analytic[tok]),
        ))


    # Monte Carlo support count = total number of transitions observed from this prefix
    prefix_counter = mc_truth_counts.get(prefix, {})
    mc_prefix_count = sum(prefix_counter.values())
    mc_reliable = mc_prefix_count >= mc_reliable_threshold

    # Summary metrics
    tv_mc_lm = total_variation_distance_two_dicts(mc_dist, lm_dist, token_set)
    kl_mc_lm = kl_divergence(mc_dist, lm_dist, token_set)

    tv_analytic_lm = total_variation_distance_two_dicts(analytic_dist, lm_dist, token_set)
    kl_analytic_lm = kl_divergence(analytic_dist, lm_dist, token_set)

    tv_mc_analytic = total_variation_distance_two_dicts(mc_dist, analytic_dist, token_set)
    kl_mc_analytic = kl_divergence(mc_dist, analytic_dist, token_set)

    spearman_mc_lm = spearman_rank_correlation(mc_dist, lm_dist, token_set)
    spearman_analytic_lm = spearman_rank_correlation(analytic_dist, lm_dist, token_set)
    spearman_mc_analytic = spearman_rank_correlation(mc_dist, analytic_dist, token_set)

    js_analytic_lm = js_divergence(analytic_dist, lm_dist, token_set)

    entropy_analytic = entropy(analytic_dist, token_set)
    entropy_lm = entropy(lm_dist, token_set)
    entropy_gap_lm_minus_analytic = entropy_lm - entropy_analytic

    weighted_abs_rank_error_analytic_lm = weighted_abs_rank_error(
        analytic_dist,
        lm_dist,
        token_set,
    )

    truth_top1_prob, truth_top2_prob, truth_top1_minus_top2, truth_max_minus_min = top_probability_summary(
        analytic_dist,
        token_set,
    )

    prefix_class = classify_prefix_sharpness(
    mc_reliable=mc_reliable,
    truth_top1_minus_top2=truth_top1_minus_top2,
    )



    pretty_print_comparison(prefix, aligned_rows)

    print(f"\nSummary for prefix {repr(prefix)}:")
    print(f"  MC prefix count        = {mc_prefix_count}")
    print(f"  MC reliable            = {mc_reliable}")
    print(f"  TV(MC, LM)             = {tv_mc_lm:.6f}")
    print(f"  KL(MC || LM)           = {kl_mc_lm:.6f}")
    print(f"  TV(Analytic, LM)       = {tv_analytic_lm:.6f}")
    print(f"  KL(Analytic || LM)     = {kl_analytic_lm:.6f}")
    print(f"  TV(MC, Analytic)       = {tv_mc_analytic:.6f}")
    print(f"  KL(MC || Analytic)     = {kl_mc_analytic:.6f}")
    print(f"  Spearman(MC, LM)       = {spearman_mc_lm:.6f}")
    print(f"  Spearman(Analytic, LM) = {spearman_analytic_lm:.6f}")
    print(f"  Spearman(MC, Analytic) = {spearman_mc_analytic:.6f}")
    print(f"  JS(Analytic, LM)       = {js_analytic_lm:.6f}")
    print(f"  H(Analytic)            = {entropy_analytic:.6f}")
    print(f"  H(LM)                  = {entropy_lm:.6f}")
    print(f"  H(LM)-H(Analytic)      = {entropy_gap_lm_minus_analytic:.6f}")
    print(f"  WeightedRankErr(AN,LM) = {weighted_abs_rank_error_analytic_lm:.6f}")
    print(f"  Truth top1 prob        = {truth_top1_prob:.6f}")
    print(f"  Truth top2 prob        = {truth_top2_prob:.6f}")
    print(f"  Truth top1-top2        = {truth_top1_minus_top2:.6f}")
    print(f"  Truth max-min          = {truth_max_minus_min:.6f}")
    print(f"  Prefix kind            = {prefix_kind}")
    print(f"  Prefix class           = {prefix_class}")    

    return {
        "run_id": run_id,
        "prompt_type": prompt_type,
        "support_mode": support_mode,
        "allow_negative": allow_negative,
        "mean": mean,
        "std": std,
        "lower": lower,
        "upper": upper,
        "distribution": distribution,
        "params": params,
        "model_name": model_name,
        "n_samples": n_samples,
        "decimals": decimals,
        "icl_n_examples": icl_n_examples,
        "icl_seed": icl_seed,
        "prefix": prefix,
        "aligned_rows": aligned_rows,
        "mc_prefix_count": mc_prefix_count,
        "mc_reliable": mc_reliable,
        "tv_mc_lm": tv_mc_lm,
        "kl_mc_lm": kl_mc_lm,
        "tv_analytic_lm": tv_analytic_lm,
        "kl_analytic_lm": kl_analytic_lm,
        "tv_mc_analytic": tv_mc_analytic,
        "kl_mc_analytic": kl_mc_analytic,
        "spearman_mc_lm": spearman_mc_lm,
        "spearman_analytic_lm": spearman_analytic_lm,
        "spearman_mc_analytic": spearman_mc_analytic,
        "js_analytic_lm": js_analytic_lm,
        "entropy_analytic": entropy_analytic,
        "entropy_lm": entropy_lm,
        "entropy_gap_lm_minus_analytic": entropy_gap_lm_minus_analytic,
        "weighted_abs_rank_error_analytic_lm": weighted_abs_rank_error_analytic_lm,
        "truth_top1_prob": truth_top1_prob,
        "truth_top2_prob": truth_top2_prob,
        "truth_top1_minus_top2": truth_top1_minus_top2,
        "truth_max_minus_min": truth_max_minus_min,
        "prefix_kind": prefix_kind,
        "prefix_class": prefix_class,        
    }


def export_token_level_csv(results, filepath):
    """
    One row per (prefix, token).
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
                "run_id",
                "model_name",
                "distribution",
                "distribution_params",
                "prompt_type",
                "support_mode",
                "allow_negative",
                "mean",
                "std",
                "lower",
                "upper",
                "n_samples",
                "decimals",
                "icl_n_examples",
                "icl_seed",
                "prefix",
                "prefix_kind",
                "token",
                "mc_truth",
                "analytic_truth",
                "lm_prob",
                "rank_mc",
                "rank_analytic",
                "rank_lm",
                "rank_diff_mc_lm",
                "rank_diff_analytic_lm",
                "abs_rank_diff_mc_lm",
                "abs_rank_diff_analytic_lm",
                "abs_diff_mc_lm",
                "abs_diff_analytic_lm",               
        ])

        for result in results:
            prefix = result["prefix"]
            for (
                tok,
                p_mc,
                p_analytic,
                p_lm,
                rank_mc,
                rank_analytic,
                rank_lm,
                rank_diff_mc_lm,
                rank_diff_analytic_lm,
                abs_rank_diff_mc_lm,
                abs_rank_diff_analytic_lm,
            ) in result["aligned_rows"]:
                writer.writerow([
                        result["run_id"],
                        result["model_name"],
                        result["distribution"],
                        json.dumps(result["params"], sort_keys=True),
                        result["prompt_type"],
                        result["support_mode"],
                        result["allow_negative"],
                        result["mean"],
                        result["std"],
                        result["lower"],
                        result["upper"],
                        result["n_samples"],
                        result["decimals"],
                        result["icl_n_examples"],
                        result["icl_seed"],
                        prefix,
                        result["prefix_kind"],
                        tok,
                        p_mc,
                        p_analytic,
                        p_lm,
                        rank_mc,
                        rank_analytic,
                        rank_lm,
                        rank_diff_mc_lm,
                        rank_diff_analytic_lm,
                        abs_rank_diff_mc_lm,
                        abs_rank_diff_analytic_lm,
                        abs(p_mc - p_lm),
                        abs(p_analytic - p_lm),
                ])


def export_prefix_summary_csv(results, filepath):
    """
    One row per prefix summary.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
                "run_id",
                "model_name",
                "distribution",
                "distribution_params",
                "prompt_type",
                "support_mode",
                "allow_negative",
                "mean",
                "std",
                "lower",
                "upper",
                "n_samples",
                "decimals",
                "icl_n_examples",
                "icl_seed",
                "prefix",
                "prefix_kind",
                "prefix_class",
                "mc_prefix_count",
                "mc_reliable",
                "tv_mc_lm",
                "kl_mc_lm",
                "tv_analytic_lm",
                "kl_analytic_lm",
                "tv_mc_analytic",
                "kl_mc_analytic",
                "spearman_mc_lm",
                "spearman_analytic_lm",
                "spearman_mc_analytic",
                "js_analytic_lm",
                "entropy_analytic",
                "entropy_lm",
                "entropy_gap_lm_minus_analytic",
                "weighted_abs_rank_error_analytic_lm",
                "truth_top1_prob",
                "truth_top2_prob",
                "truth_top1_minus_top2",
                "truth_max_minus_min",                

        ])

        for result in results:
            writer.writerow([
                    result["run_id"],
                    result["model_name"],
                    result["distribution"],
                    json.dumps(result["params"], sort_keys=True),
                    result["prompt_type"],
                    result["support_mode"],
                    result["allow_negative"],
                    result["mean"],
                    result["std"],
                    result["lower"],
                    result["upper"],
                    result["n_samples"],
                    result["decimals"],
                    result["icl_n_examples"],
                    result["icl_seed"],
                    result["prefix"],
                    result["prefix_kind"],
                    result["prefix_class"],
                    result["mc_prefix_count"],
                    result["mc_reliable"],
                    result["tv_mc_lm"],
                    result["kl_mc_lm"],
                    result["tv_analytic_lm"],
                    result["kl_analytic_lm"],
                    result["tv_mc_analytic"],
                    result["kl_mc_analytic"],
                    result["spearman_mc_lm"],
                    result["spearman_analytic_lm"],
                    result["spearman_mc_analytic"],
                    result["js_analytic_lm"],
                    result["entropy_analytic"],
                    result["entropy_lm"],
                    result["entropy_gap_lm_minus_analytic"],
                    result["weighted_abs_rank_error_analytic_lm"],
                    result["truth_top1_prob"],
                    result["truth_top2_prob"],
                    result["truth_top1_minus_top2"],
                    result["truth_max_minus_min"],
            ])

def run_experiment(config: RunConfig):

    formatted_samples, prefix_to_next_counts, prefix_to_next_probs = build_truth_model(
    distribution=config.distribution,
    params=config.params,
    mean=config.mean,
    std=config.std,
    n_samples=config.n_samples,
    decimals=config.decimals,
    lower=config.lower,
    upper=config.upper,
    seed=config.seed,
    )



    prompt = build_prompt(
    distribution=config.distribution,
    params=config.params,
    mean=config.mean,
    std=config.std,
    decimals=config.decimals,
    prompt_type=config.prompt_type,
    support_mode=config.support_mode,
    lower=config.lower,
    upper=config.upper,
    icl_n_examples=config.icl_n_examples,
    icl_seed=config.icl_seed,
    )




    all_results = []

    for prefix in config.prefixes:
        result = compare_for_prefix(
            prefix=prefix,
            mc_truth_probs=prefix_to_next_probs,
            mc_truth_counts=prefix_to_next_counts,
            prompt=prompt,
            mean=config.mean,
            std=config.std,
            lower=config.lower,
            upper=config.upper,
            distribution=config.distribution,
            params=config.params,
            model_name=config.model_name,
            mc_reliable_threshold=config.mc_reliable_threshold,
            run_id=config.run_id,
            prompt_type=config.prompt_type,
            n_samples=config.n_samples,
            decimals=config.decimals,
            allow_negative=config.allow_negative,
            support_mode=config.support_mode,
            icl_n_examples=config.icl_n_examples,
            icl_seed=config.icl_seed,
        )
        all_results.append(result)

    export_token_level_csv(all_results, config.token_level_csv)
    export_prefix_summary_csv(all_results, config.prefix_summary_csv)

    return all_results



from datetime import datetime

if __name__ == "__main__":
    distribution = "uniform"
    params = {"low": 0.0, "high": 1.0}

    lower = 0.0
    upper = 1.0
    n_samples = 500000
    decimals = 3
    prompt_type = "plain"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    RUN_ID = f"uniform_0_1_n{n_samples}_dec{decimals}_{prompt_type}_{timestamp}"

    config = RunConfig(
        distribution=distribution,
        params=params,
        model_name="Qwen/Qwen3-4B",

        mean=None,
        std=None,
        lower=lower,
        upper=upper,
        n_samples=n_samples,
        decimals=decimals,
        prefixes=["", "0", "0.", "0.0", "0.1", "0.5", "0.9"],
        mc_reliable_threshold=1000,
        token_level_csv=f"outputs/token_level_{RUN_ID}.csv",
        prefix_summary_csv=f"outputs/prefix_summary_{RUN_ID}.csv",
        prompt_type=prompt_type,
        support_mode="bounded",
        allow_negative=False,
        seed=42,
        run_id=RUN_ID,
    )

    run_experiment(config)


