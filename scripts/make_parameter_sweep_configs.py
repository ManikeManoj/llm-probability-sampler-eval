#!/usr/bin/env python3
"""
Generate a targeted parameter-sweep JSONL configuration for the thesis pipeline.

Design:
- Four small Qwen/Gemma checkpoints (base and instruct).
- Five continuous distribution families.
- Location, scale, support, skewness, and shape variations.
- Deterministic adaptive prefixes chosen from a fixed pilot sample so each
  parameter setting has reliable ROOT/sign/integer/dot/fraction coverage.
- Plain prompt by default. A smaller prompt-probe profile is available for
  explanatory_4 after the plain sweep is complete.

The output is consumed by scripts/run_parameter_sweep_config.py.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


MODEL_CATALOG = [
    {
        "model_alias": "qwen4b_base",
        "model_name": "Qwen/Qwen3-4B-Base",
        "model_family": "qwen",
        "model_size_class": "small",
        "model_stage": "base",
        "notes": "small Qwen base checkpoint",
    },
    {
        "model_alias": "qwen4b_instruct_2507",
        "model_name": "Qwen/Qwen3-4B-Instruct-2507",
        "model_family": "qwen",
        "model_size_class": "small",
        "model_stage": "instruct",
        "notes": "small Qwen instruction-tuned checkpoint",
    },
    {
        "model_alias": "gemma_e4b_base",
        "model_name": "google/gemma-4-E4B",
        "model_family": "gemma",
        "model_size_class": "small",
        "model_stage": "base",
        "notes": "small Gemma base checkpoint",
    },
    {
        "model_alias": "gemma_e4b_it",
        "model_name": "google/gemma-4-E4B-it",
        "model_family": "gemma",
        "model_size_class": "small",
        "model_stage": "instruct",
        "notes": "small Gemma instruction-tuned checkpoint",
    },
]


@dataclass(frozen=True)
class SweepSetting:
    setting_id: str
    distribution: str
    params: dict[str, float]
    sweep_axis: str
    parameter_label: str
    is_baseline: bool = False
    prompt_probe: bool = False


SETTINGS = [
    # Normal: location and scale, one factor at a time.
    SweepSetting("normal_mu_m1_sd1", "normal", {"mean": -1.0, "std": 1.0}, "location", "Normal(mu=-1, sigma=1)", prompt_probe=False),
    SweepSetting("normal_mu0_sd0p5", "normal", {"mean": 0.0, "std": 0.5}, "scale", "Normal(mu=0, sigma=0.5)", prompt_probe=False),
    SweepSetting("normal_mu0_sd1", "normal", {"mean": 0.0, "std": 1.0}, "baseline", "Normal(mu=0, sigma=1)", is_baseline=True),
    SweepSetting("normal_mu0_sd2", "normal", {"mean": 0.0, "std": 2.0}, "scale", "Normal(mu=0, sigma=2)", prompt_probe=True),
    SweepSetting("normal_mu_p1_sd1", "normal", {"mean": 1.0, "std": 1.0}, "location", "Normal(mu=1, sigma=1)", prompt_probe=True),

    # Laplace: matched location/scale design for a sharper, heavier-tailed family.
    SweepSetting("laplace_loc_m1_s1", "laplace", {"loc": -1.0, "scale": 1.0}, "location", "Laplace(loc=-1, scale=1)"),
    SweepSetting("laplace_loc0_s0p5", "laplace", {"loc": 0.0, "scale": 0.5}, "scale", "Laplace(loc=0, scale=0.5)"),
    SweepSetting("laplace_loc0_s1", "laplace", {"loc": 0.0, "scale": 1.0}, "baseline", "Laplace(loc=0, scale=1)", is_baseline=True),
    SweepSetting("laplace_loc0_s2", "laplace", {"loc": 0.0, "scale": 2.0}, "scale", "Laplace(loc=0, scale=2)", prompt_probe=True),
    SweepSetting("laplace_loc_p1_s1", "laplace", {"loc": 1.0, "scale": 1.0}, "location", "Laplace(loc=1, scale=1)"),

    # Uniform: same-width shift, signed support, narrow resolution, and wide support.
    SweepSetting("uniform_0_0p1", "uniform", {"low": 0.0, "high": 0.1}, "support_width", "Uniform(0, 0.1)"),
    SweepSetting("uniform_0_1", "uniform", {"low": 0.0, "high": 1.0}, "baseline", "Uniform(0, 1)", is_baseline=True),
    SweepSetting("uniform_1_2", "uniform", {"low": 1.0, "high": 2.0}, "location", "Uniform(1, 2)"),
    SweepSetting("uniform_m1_1", "uniform", {"low": -1.0, "high": 1.0}, "signed_support", "Uniform(-1, 1)", prompt_probe=True),
    SweepSetting("uniform_0_10", "uniform", {"low": 0.0, "high": 10.0}, "support_width", "Uniform(0, 10)"),

    # Exponential: rate controls concentration near zero and tail length.
    SweepSetting("exponential_rate0p5", "exponential", {"rate": 0.5}, "rate", "Exponential(rate=0.5)"),
    SweepSetting("exponential_rate1", "exponential", {"rate": 1.0}, "baseline", "Exponential(rate=1)", is_baseline=True),
    SweepSetting("exponential_rate2", "exponential", {"rate": 2.0}, "rate", "Exponential(rate=2)", prompt_probe=True),

    # Beta: edge-concentrated, flat equivalence control, canonical, and two skew directions.
    SweepSetting("beta_a0p5_b0p5", "beta", {"alpha": 0.5, "beta": 0.5}, "shape", "Beta(0.5, 0.5)"),
    SweepSetting("beta_a1_b1", "beta", {"alpha": 1.0, "beta": 1.0}, "equivalence_control", "Beta(1, 1) = Uniform(0, 1)"),
    SweepSetting("beta_a2_b2", "beta", {"alpha": 2.0, "beta": 2.0}, "baseline", "Beta(2, 2)", is_baseline=True),
    SweepSetting("beta_a2_b5", "beta", {"alpha": 2.0, "beta": 5.0}, "skew", "Beta(2, 5)", prompt_probe=True),
    SweepSetting("beta_a5_b2", "beta", {"alpha": 5.0, "beta": 2.0}, "skew", "Beta(5, 2)"),
]


def parse_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def sample_distribution(setting: SweepSetting, n: int, rng: np.random.Generator) -> np.ndarray:
    p = setting.params
    if setting.distribution == "normal":
        return rng.normal(p["mean"], p["std"], size=n)
    if setting.distribution == "laplace":
        return rng.laplace(p["loc"], p["scale"], size=n)
    if setting.distribution == "uniform":
        return rng.uniform(p["low"], p["high"], size=n)
    if setting.distribution == "exponential":
        return rng.exponential(1.0 / p["rate"], size=n)
    if setting.distribution == "beta":
        return rng.beta(p["alpha"], p["beta"], size=n)
    raise ValueError(f"Unsupported distribution: {setting.distribution}")


def format_number(value: float, decimals: int) -> str:
    # Avoid the visually distinct but numerically zero string "-0.000".
    rounded = round(float(value), decimals)
    if rounded == 0:
        rounded = 0.0
    return f"{rounded:.{decimals}f}"


def integer_prefix(formatted: str) -> str:
    return formatted.split(".", maxsplit=1)[0]


def fraction_prefix(formatted: str, depth: int) -> str:
    integer, fractional = formatted.split(".", maxsplit=1)
    if depth < 1 or depth > len(fractional):
        raise ValueError(f"Invalid fractional depth {depth} for {formatted!r}")
    return f"{integer}.{fractional[:depth]}"


def numeric_prefix_sort_key(prefix: str) -> tuple[float, str]:
    try:
        return (float(prefix), prefix)
    except ValueError:
        return (math.inf, prefix)


def choose_prefixes(
    setting: SweepSetting,
    *,
    pilot_n: int,
    pilot_seed: int,
    decimals: int,
    final_n_samples: int,
    mc_reliable_threshold: int,
    min_prefix_mass: float,
    max_integer_prefixes: int,
    fraction_depths: tuple[int, ...],
    max_fraction_prefixes_per_depth: int,
) -> tuple[list[str], list[dict[str, Any]]]:
    # Derive a stable per-setting seed without relying on Python's randomized hash().
    offset = sum((i + 1) * ord(ch) for i, ch in enumerate(setting.setting_id))
    rng = np.random.default_rng(pilot_seed + offset)
    samples = sample_distribution(setting, pilot_n, rng)
    formatted = [format_number(x, decimals) for x in samples]

    integer_counts = Counter(integer_prefix(x) for x in formatted)
    dot_counts = Counter(f"{integer_prefix(x)}." for x in formatted)
    fraction_counts_by_depth = {
        depth: Counter(fraction_prefix(x, depth) for x in formatted)
        for depth in fraction_depths
    }
    sign_count = sum(x.startswith("-") for x in formatted)

    # Require more than the bare MC threshold to reduce unreliable-prefix risk.
    scaled_threshold = math.ceil(
        2.0 * mc_reliable_threshold * pilot_n / max(final_n_samples, 1)
    )
    minimum_count = max(math.ceil(min_prefix_mass * pilot_n), scaled_threshold)

    selected_integers: list[str] = []

    # Quantile anchors give low/central/high coverage instead of only modal branches.
    for q in (0.10, 0.30, 0.50, 0.70, 0.90):
        candidate = integer_prefix(format_number(float(np.quantile(samples, q)), decimals))
        if integer_counts[candidate] >= minimum_count and candidate not in selected_integers:
            selected_integers.append(candidate)

    # Fill any remaining slots with high-mass integer branches.
    for candidate, count in integer_counts.most_common():
        if count < minimum_count:
            continue
        if candidate not in selected_integers:
            selected_integers.append(candidate)
        if len(selected_integers) >= max_integer_prefixes:
            break

    # Keep the strongest anchors if quantiles produced more than the requested cap.
    if len(selected_integers) > max_integer_prefixes:
        selected_integers = sorted(
            selected_integers,
            key=lambda p: (-integer_counts[p], numeric_prefix_sort_key(p)),
        )[:max_integer_prefixes]

    selected_integers = sorted(selected_integers, key=numeric_prefix_sort_key)

    selected_fractions: list[str] = []
    selected_fraction_depth: dict[str, int] = {}

    # Select interpretable lower/central/upper quantile anchors at each depth.
    for depth in fraction_depths:
        counts = fraction_counts_by_depth[depth]
        depth_selected: list[str] = []
        for q in (0.15, 0.50, 0.85):
            candidate = fraction_prefix(
                format_number(float(np.quantile(samples, q)), decimals),
                depth,
            )
            if counts[candidate] >= minimum_count and candidate not in depth_selected:
                depth_selected.append(candidate)

        # Fill missing positions with high-mass branches at this depth.
        for candidate, count in counts.most_common():
            if count < minimum_count:
                continue
            if candidate not in depth_selected:
                depth_selected.append(candidate)
            if len(depth_selected) >= max_fraction_prefixes_per_depth:
                break

        for candidate in depth_selected[:max_fraction_prefixes_per_depth]:
            if candidate not in selected_fractions:
                selected_fractions.append(candidate)
                selected_fraction_depth[candidate] = depth

    prefixes = ["ROOT"]
    if sign_count >= minimum_count:
        prefixes.append("-")
    prefixes.extend(selected_integers)
    prefixes.extend(f"{p}." for p in selected_integers if dot_counts[f"{p}."] >= minimum_count)
    prefixes.extend(selected_fractions)

    # Preserve order while preventing accidental duplicates.
    prefixes = list(dict.fromkeys(prefixes))

    manifest_rows: list[dict[str, Any]] = []
    for prefix in prefixes:
        if prefix == "ROOT":
            count = pilot_n
            kind = "start"
        elif prefix == "-":
            count = sign_count
            kind = "sign"
        elif prefix.endswith("."):
            count = dot_counts[prefix]
            kind = "dot"
        elif "." in prefix:
            depth = selected_fraction_depth[prefix]
            count = fraction_counts_by_depth[depth][prefix]
            kind = "fraction"
        else:
            count = integer_counts[prefix]
            kind = "integer"
        manifest_rows.append(
            {
                "setting_id": setting.setting_id,
                "distribution": setting.distribution,
                "parameter_label": setting.parameter_label,
                "sweep_axis": setting.sweep_axis,
                "is_baseline": setting.is_baseline,
                "prefix": prefix,
                "prefix_kind": kind,
                "fraction_depth": selected_fraction_depth.get(prefix, 0),
                "pilot_count": int(count),
                "pilot_mass": float(count / pilot_n),
                "expected_final_count": float(count / pilot_n * final_n_samples),
                "minimum_pilot_count": int(minimum_count),
            }
        )
    return prefixes, manifest_rows


def selected_settings(profile: str, include_baselines: bool) -> list[SweepSetting]:
    if profile == "smoke":
        settings = [s for s in SETTINGS if s.setting_id == "normal_mu_p1_sd1"]
    elif profile == "prompt_probe":
        settings = [s for s in SETTINGS if s.prompt_probe]
    elif profile == "full":
        settings = list(SETTINGS)
    else:
        raise ValueError(profile)

    if not include_baselines:
        settings = [s for s in settings if not s.is_baseline]
    return settings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, help="Output JSONL path.")
    parser.add_argument("--manifest", default=None, help="Prefix manifest CSV path.")
    parser.add_argument("--profile", choices=["smoke", "full", "prompt_probe"], default="full")
    parser.add_argument("--include-baselines", action="store_true",
                        help="Include canonical settings. Recommended for the full sweep because the adaptive prefix policy differs from the old main-grid prefixes.")
    parser.add_argument("--model-aliases", default="all",
                        help="Comma-separated aliases or 'all'.")
    parser.add_argument("--prompts", default="plain",
                        help="Comma-separated prompt types. Use plain for the full sweep.")
    parser.add_argument("--precision", choices=["4bit", "bf16"], default="4bit")
    parser.add_argument("--n-samples", type=int, default=500000)
    parser.add_argument("--decimals", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mc-reliable-threshold", type=int, default=1000)
    parser.add_argument("--lm-scoring-method", choices=["single_token", "sequence", "auto"], default="auto")
    parser.add_argument("--pilot-n", type=int, default=200000)
    parser.add_argument("--pilot-seed", type=int, default=20260731)
    parser.add_argument("--min-prefix-mass", type=float, default=0.004)
    parser.add_argument("--max-integer-prefixes", type=int, default=5)
    parser.add_argument("--fraction-depths", default="1,2",
                        help="Comma-separated fractional prefix depths.")
    parser.add_argument("--max-fraction-prefixes-per-depth", type=int, default=3)
    parser.add_argument("--tag", default="param_sweep")
    args = parser.parse_args()

    fraction_depths = tuple(int(x) for x in parse_csv(args.fraction_depths))
    if not fraction_depths or any(depth < 1 or depth > args.decimals for depth in fraction_depths):
        raise ValueError("--fraction-depths must be between 1 and --decimals.")

    aliases = {m["model_alias"] for m in MODEL_CATALOG}
    requested_aliases = aliases if args.model_aliases == "all" else set(parse_csv(args.model_aliases))
    unknown = requested_aliases - aliases
    if unknown:
        raise ValueError(f"Unknown model aliases: {sorted(unknown)}")

    models = [m for m in MODEL_CATALOG if m["model_alias"] in requested_aliases]
    prompts = parse_csv(args.prompts)
    settings = selected_settings(args.profile, args.include_baselines)
    if not settings:
        raise ValueError("No settings selected. For profile=full, consider --include-baselines.")

    prefix_map: dict[str, list[str]] = {}
    manifest_rows: list[dict[str, Any]] = []
    for setting in settings:
        prefixes, rows = choose_prefixes(
            setting,
            pilot_n=args.pilot_n,
            pilot_seed=args.pilot_seed,
            decimals=args.decimals,
            final_n_samples=args.n_samples,
            mc_reliable_threshold=args.mc_reliable_threshold,
            min_prefix_mass=args.min_prefix_mass,
            max_integer_prefixes=args.max_integer_prefixes,
            fraction_depths=fraction_depths,
            max_fraction_prefixes_per_depth=args.max_fraction_prefixes_per_depth,
        )
        prefix_map[setting.setting_id] = prefixes
        manifest_rows.extend(rows)

    load_in_4bit = args.precision == "4bit"
    rows: list[dict[str, Any]] = []
    for model in models:
        for setting in settings:
            for prompt in prompts:
                run_id = (
                    f"{args.tag}_{model['model_alias']}_{setting.setting_id}_{prompt}_"
                    f"n{args.n_samples}_d{args.decimals}_{args.precision}"
                )
                rows.append(
                    {
                        **model,
                        "run_id": run_id,
                        "experiment": "parameter_sweep",
                        "profile": args.profile,
                        "setting_id": setting.setting_id,
                        "sweep_axis": setting.sweep_axis,
                        "parameter_label": setting.parameter_label,
                        "is_baseline": setting.is_baseline,
                        "distribution": setting.distribution,
                        "params": setting.params,
                        "prompt_type": prompt,
                        "prompt_protocol": "raw_direct",
                        "quantization": args.precision,
                        "load_in_4bit": load_in_4bit,
                        "lm_scoring_method": args.lm_scoring_method,
                        "prefix_policy": "adaptive_quantile_mass_v1",
                        "prefixes": ",".join(prefix_map[setting.setting_id]),
                        "n_samples": args.n_samples,
                        "decimals": args.decimals,
                        "seed": args.seed,
                        "mc_reliable_threshold": args.mc_reliable_threshold,
                        "pilot_n": args.pilot_n,
                        "pilot_seed": args.pilot_seed,
                        "min_prefix_mass": args.min_prefix_mass,
                    }
                )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    manifest = Path(args.manifest) if args.manifest else out.with_suffix(".prefix_manifest.csv")
    manifest.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(manifest_rows[0].keys())
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"Wrote {len(rows)} configs to {out}")
    print(f"Wrote {len(manifest_rows)} prefix rows to {manifest}")
    print(f"Models: {len(models)} | Settings: {len(settings)} | Prompts: {len(prompts)}")
    for setting in settings:
        print(f"  {setting.setting_id:28s} {len(prefix_map[setting.setting_id]):2d} prefixes: {','.join(prefix_map[setting.setting_id])}")


if __name__ == "__main__":
    main()
