#!/usr/bin/env python3
"""
Create JSONL config files for the RQ1/RQ2 core grid.

Default design:
  3 non-thinking/instruct models × 5 distributions × selected prompts.

Edit MODEL_SPECS below if the exact Hugging Face IDs differ on Helix.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


# ---------------------------------------------------------------------
# EDIT THESE MODEL IDS IF NEEDED
# ---------------------------------------------------------------------
MODEL_SPECS: List[Dict[str, Any]] = [
    {
        "model_alias": "qwen30_a3b_instruct_2507",
        "model_family": "qwen",
        "model_variant": "instruct",
        "reasoning_mode": "none",
        "prompt_protocol": "raw_direct",
        "model_name": "Qwen/Qwen3-30B-A3B-Instruct-2507",
        "lm_scoring_method": "auto",
    },
    {
        "model_alias": "gemma4_31b_it_non_thinking",
        "model_family": "gemma",
        "model_variant": "instruct",
        "reasoning_mode": "off",
        "prompt_protocol": "raw_direct",
        "model_name": "google/gemma-4-31B-it",
        "lm_scoring_method": "auto",
    },
    {
        "model_alias": "mistral_small_31_24b_instruct_2503",
        "model_family": "mistral",
        "model_variant": "instruct",
        "reasoning_mode": "none",
        "prompt_protocol": "raw_direct",
        "model_name": "mistralai/Mistral-Small-3.1-24B-Instruct-2503",
        "lm_scoring_method": "auto",
    },
]


DISTRIBUTION_SPECS: List[Dict[str, Any]] = [
    {
        "distribution": "normal",
        "distribution_label": "Normal(0,1)",
        "params": {"mean": 0.0, "std": 1.0},
        "support_mode": "agnostic",
        "allow_negative": True,
        "lower": None,
        "upper": None,
        "prefixes": ["ROOT", "-", "-2", "-2.", "-1", "-1.", "0", "0.", "1", "1.", "2", "2."],
    },
    {
        "distribution": "laplace",
        "distribution_label": "Laplace(0,1)",
        "params": {"loc": 0.0, "scale": 1.0},
        "support_mode": "agnostic",
        "allow_negative": True,
        "lower": None,
        "upper": None,
        "prefixes": ["ROOT", "-", "-2", "-2.", "-1", "-1.", "0", "0.", "1", "1.", "2", "2."],
    },
    {
        "distribution": "uniform",
        "distribution_label": "Uniform(0,1)",
        "params": {"lower": 0.0, "upper": 1.0},
        "support_mode": "bounded",
        "allow_negative": False,
        "lower": 0.0,
        "upper": 1.0,
        "prefixes": ["ROOT", "0", "0.", "0.0", "0.1", "0.4", "0.5", "0.8", "0.9"],
    },
    {
        "distribution": "beta",
        "distribution_label": "Beta(2,2)",
        "params": {"alpha": 2.0, "beta": 2.0},
        "support_mode": "bounded",
        "allow_negative": False,
        "lower": 0.0,
        "upper": 1.0,
        "prefixes": ["ROOT", "0", "0.", "0.0", "0.1", "0.2", "0.4", "0.5", "0.8", "0.9"],
    },
    {
        "distribution": "exponential",
        "distribution_label": "Exponential(1)",
        "params": {"rate": 1.0},
        "support_mode": "positive",
        "allow_negative": False,
        "lower": None,
        "upper": None,
        "prefixes": ["ROOT", "0", "0.", "0.0", "0.1", "0.5", "1", "1.", "2", "2.", "3", "3."],
    },
]


def safe_name(text: str) -> str:
    return (
        text.lower()
        .replace("/", "_")
        .replace(".", "p")
        .replace("-", "_")
        .replace("(", "")
        .replace(")", "")
        .replace(",", "_")
        .replace(" ", "_")
    )


def build_configs(prompts: List[str], n_samples: int, decimals: int, seed: int, experiment: str) -> List[Dict[str, Any]]:
    configs: List[Dict[str, Any]] = []
    for model in MODEL_SPECS:
        for dist in DISTRIBUTION_SPECS:
            for prompt in prompts:
                run_id = f"{experiment}_{model['model_alias']}_{safe_name(dist['distribution_label'])}_{prompt}_n{n_samples}"
                cfg: Dict[str, Any] = {
                    "experiment": experiment,
                    "run_id": run_id,
                    "model_alias": model["model_alias"],
                    "model_family": model["model_family"],
                    "model_variant": model["model_variant"],
                    "reasoning_mode": model["reasoning_mode"],
                    "prompt_protocol": model["prompt_protocol"],
                    "model_name": model["model_name"],
                    "lm_scoring_method": model["lm_scoring_method"],
                    "distribution": dist["distribution"],
                    "distribution_label": dist["distribution_label"],
                    "params": dist["params"],
                    "support_mode": dist["support_mode"],
                    "allow_negative": dist["allow_negative"],
                    "lower": dist["lower"],
                    "upper": dist["upper"],
                    "prefixes": dist["prefixes"],
                    "prompt_type": prompt,
                    "n_samples": n_samples,
                    "decimals": decimals,
                    "seed": seed,
                }
                configs.append(cfg)
    return configs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, help="Output JSONL path, e.g. configs/rq1rq2_instruct_plain.jsonl")
    parser.add_argument("--prompts", default="plain", help="Comma-separated prompt types, e.g. plain or plain,explanatory_4")
    parser.add_argument("--n-samples", type=int, default=500000)
    parser.add_argument("--decimals", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--experiment", default="rq1rq2")
    args = parser.parse_args()

    prompts = [p.strip() for p in args.prompts.split(",") if p.strip()]
    configs = build_configs(prompts, args.n_samples, args.decimals, args.seed, args.experiment)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for cfg in configs:
            f.write(json.dumps(cfg, ensure_ascii=False, sort_keys=True) + "\n")

    print(f"Wrote {len(configs)} configs to {out}")
    print("Models:", ", ".join(m["model_alias"] for m in MODEL_SPECS))
    print("Distributions:", ", ".join(d["distribution_label"] for d in DISTRIBUTION_SPECS))
    print("Prompts:", ", ".join(prompts))


if __name__ == "__main__":
    main()
