#!/usr/bin/env python3
"""
Generate JSONL configs for the revised thesis model grid.

The generated config is consumed by scripts/run_json_config.py, which calls
src/run_compare.py with the current CLI. Extra metadata is kept in the JSONL
and later joined back into merged CSVs by scripts/combine_prefix_summaries.py.
"""
import argparse
import json
from pathlib import Path
from datetime import datetime

MODEL_CATALOG = [
    # SMALL / lower-cost set
    {
        "model_alias": "qwen4b_base",
        "model_name": "Qwen/Qwen3-4B-Base",
        "model_family": "qwen",
        "model_size_class": "small",
        "model_stage": "base",
        "reasoning_available": False,
        "notes": "small Qwen base checkpoint",
    },
    {
        "model_alias": "qwen4b_instruct_2507",
        "model_name": "Qwen/Qwen3-4B-Instruct-2507",
        "model_family": "qwen",
        "model_size_class": "small",
        "model_stage": "instruct",
        "reasoning_available": False,
        "notes": "small Qwen non-thinking instruct checkpoint",
    },
    {
        "model_alias": "qwen4b_thinking_2507",
        "model_name": "Qwen/Qwen3-4B-Thinking-2507",
        "model_family": "qwen",
        "model_size_class": "small",
        "model_stage": "reasoning",
        "reasoning_available": True,
        "notes": "small Qwen separate thinking checkpoint; direct numeric protocol only for now",
    },
    {
        "model_alias": "gemma_e4b_base",
        "model_name": "google/gemma-4-E4B",
        "model_family": "gemma",
        "model_size_class": "small",
        "model_stage": "base",
        "reasoning_available": False,
        "notes": "small Gemma base checkpoint",
    },
    {
        "model_alias": "gemma_e4b_it",
        "model_name": "google/gemma-4-E4B-it",
        "model_family": "gemma",
        "model_size_class": "small",
        "model_stage": "instruct",
        "reasoning_available": False,
        "notes": "small Gemma instruction tuned checkpoint; no separate reasoning checkpoint",
    },
    {
        "model_alias": "ministral3_3b_base_2512",
        "model_name": "mistralai/Ministral-3-3B-Base-2512",
        "model_family": "mistral",
        "model_size_class": "small",
        "model_stage": "base",
        "reasoning_available": False,
        "notes": "small Ministral base checkpoint",
    },
    {
        "model_alias": "ministral3_3b_instruct_2512",
        "model_name": "mistralai/Ministral-3-3B-Instruct-2512",
        "model_family": "mistral",
        "model_size_class": "small",
        "model_stage": "instruct",
        "reasoning_available": False,
        "notes": "small Ministral instruct checkpoint",
    },
    {
        "model_alias": "ministral3_3b_reasoning_2512",
        "model_name": "mistralai/Ministral-3-3B-Reasoning-2512",
        "model_family": "mistral",
        "model_size_class": "small",
        "model_stage": "reasoning",
        "reasoning_available": True,
        "notes": "small Ministral reasoning checkpoint; direct numeric protocol only for now",
    },

    # MID / main scale set
    {
        "model_alias": "qwen14b_base",
        "model_name": "Qwen/Qwen3-14B-Base",
        "model_family": "qwen",
        "model_size_class": "mid",
        "model_stage": "base",
        "reasoning_available": False,
        "notes": "mid Qwen base checkpoint",
    },
    {
        "model_alias": "qwen14b_hybrid",
        "model_name": "Qwen/Qwen3-14B",
        "model_family": "qwen",
        "model_size_class": "mid",
        "model_stage": "instruct",
        "reasoning_available": False,
        "notes": "mid Qwen hybrid/instruct-style checkpoint; no separate thinking weights in this grid",
    },
    {
        "model_alias": "gemma12b_base",
        "model_name": "google/gemma-4-12B",
        "model_family": "gemma",
        "model_size_class": "mid",
        "model_stage": "base",
        "reasoning_available": False,
        "notes": "mid Gemma base checkpoint",
    },
    {
        "model_alias": "gemma12b_it",
        "model_name": "google/gemma-4-12B-it",
        "model_family": "gemma",
        "model_size_class": "mid",
        "model_stage": "instruct",
        "reasoning_available": False,
        "notes": "mid Gemma instruction tuned checkpoint; no separate reasoning checkpoint",
    },
    {
        "model_alias": "ministral3_14b_base_2512",
        "model_name": "mistralai/Ministral-3-14B-Base-2512",
        "model_family": "mistral",
        "model_size_class": "mid",
        "model_stage": "base",
        "reasoning_available": False,
        "notes": "mid Ministral base checkpoint",
    },
    {
        "model_alias": "ministral3_14b_instruct_2512",
        "model_name": "mistralai/Ministral-3-14B-Instruct-2512",
        "model_family": "mistral",
        "model_size_class": "mid",
        "model_stage": "instruct",
        "reasoning_available": False,
        "notes": "mid Ministral instruct checkpoint",
    },
    {
        "model_alias": "ministral3_14b_reasoning_2512",
        "model_name": "mistralai/Ministral-3-14B-Reasoning-2512",
        "model_family": "mistral",
        "model_size_class": "mid",
        "model_stage": "reasoning",
        "reasoning_available": True,
        "notes": "mid Ministral reasoning checkpoint; direct numeric protocol only for now",
    },
]

ALLOWED_PROMPTS = {
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
}

DISTRIBUTIONS = {
    "normal": {
        "distribution": "normal",
        "params": {"mean": 0.0, "std": 1.0},
        "prefixes": "ROOT,-,-2,-2.,-1,-1.,0,0.,1,1.,2,2.",
    },
    "laplace": {
        "distribution": "laplace",
        "params": {"loc": 0.0, "scale": 1.0},
        "prefixes": "ROOT,-,-2,-2.,-1,-1.,0,0.,1,1.,2,2.",
    },
    "uniform": {
        "distribution": "uniform",
        "params": {"low": 0.0, "high": 1.0},
        "prefixes": "ROOT,0,0.,0.0,0.1,0.4,0.5,0.8,0.9",
    },
    "beta": {
        "distribution": "beta",
        "params": {"alpha": 2.0, "beta": 2.0},
        "prefixes": "ROOT,0,0.,0.0,0.1,0.2,0.4,0.5,0.8,0.9",
    },
    "exponential": {
        "distribution": "exponential",
        "params": {"rate": 1.0},
        "prefixes": "ROOT,0,0.,0.0,0.1,0.5,1,1.,2,2.,3,3.",
    },
}


def parse_csv_set(s: str | None, allowed: set[str], name: str) -> set[str]:
    if not s or s.lower() == "all":
        return allowed
    vals = {x.strip().lower() for x in s.split(",") if x.strip()}
    bad = vals - allowed
    if bad:
        raise ValueError(f"Unknown {name}: {sorted(bad)}. Allowed: {sorted(allowed)}")
    return vals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="Output JSONL path")
    ap.add_argument("--size-classes", default="all", help="small,mid,all")
    ap.add_argument("--families", default="all", help="qwen,gemma,mistral,all")
    ap.add_argument("--stages", default="base,instruct", help="base,instruct,reasoning,all")
    ap.add_argument("--distributions", default="all", help="normal,laplace,uniform,beta,exponential,all")
    ap.add_argument("--prompts", default="plain", help="comma-separated prompt types, e.g. plain,explanatory_4")
    ap.add_argument("--icl-n-examples",type=int,default=5,help="Number of representative values used for ICL prompts")
    ap.add_argument("--icl-seed",type=int,default=0,help="Seed used to construct and shuffle representative ICL values")
    ap.add_argument("--prompt-template-version",default="balanced_v1",help="Version label for the prompt template collection")
    ap.add_argument("--n-samples", type=int, default=500000)
    ap.add_argument("--decimals", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mc-reliable-threshold", type=int, default=1000)
    ap.add_argument("--lm-scoring-method", default="auto", choices=["single_token", "sequence", "auto"])
    ap.add_argument("--load-in-4bit", action="store_true", default=True, help="Default true; kept explicit for metadata")
    ap.add_argument("--no-4bit", action="store_true", help="Generate configs that pass --no-4bit to run_compare.py")
    ap.add_argument("--tag", default=None, help="Optional tag used in run_id")
    args = ap.parse_args()

    allowed_sizes = {"small", "mid"}
    allowed_families = {"qwen", "gemma", "mistral"}
    allowed_stages = {"base", "instruct", "reasoning"}
    allowed_dists = set(DISTRIBUTIONS.keys())

    size_filter = parse_csv_set(args.size_classes, allowed_sizes, "size class")
    family_filter = parse_csv_set(args.families, allowed_families, "family")
    stage_filter = parse_csv_set(args.stages, allowed_stages, "stage")
    dist_filter = parse_csv_set(args.distributions, allowed_dists, "distribution")
    prompts = [p.strip() for p in args.prompts.split(",") if p.strip()]

    unknown_prompts = set(prompts) - ALLOWED_PROMPTS

    if unknown_prompts:
        raise ValueError( f"Unknown prompt types: {sorted(unknown_prompts)}. "
                         f"Allowed: {sorted(ALLOWED_PROMPTS)}" )

    if args.icl_n_examples <1:
        raise ValueError("--icl-n-examples must be at least 1")

    quantization = "4bit" if not args.no_4bit else "bf16"
    tag = args.tag or f"rq_core_{quantization}"
    generated_at = datetime.now().strftime("%Y%m%d_%H%M%S")

    rows = []
    for model in MODEL_CATALOG:
        if model["model_size_class"] not in size_filter:
            continue
        if model["model_family"] not in family_filter:
            continue
        if model["model_stage"] not in stage_filter:
            continue
        for dist_name in sorted(dist_filter):
            dist = DISTRIBUTIONS[dist_name]
            for prompt in prompts:
                prompt_family, prompt_level, prompt_protocol = prompt_metadata(prompt)
                uses_internal_reasoning = prompt in {"cot", "icl_cot"}
                uses_icl = prompt in {"icl", "icl_cot"}


                run_id = (
                    f"{tag}_{model['model_alias']}_{dist_name}_{prompt}_"
                    f"n{args.n_samples}_d{args.decimals}_{quantization}"
                )
                rows.append({
                    **model,
                    "run_id": run_id,
                    "prompt_family": prompt_family,
                    "prompt_level": prompt_level,
                    "prompt_protocol": prompt_protocol,
                    "reasoning_protocol": ("internal_reasoning_cue"if uses_internal_reasoning else "none"),
                    "prompt_template_version": args.prompt_template_version,
                    "quantization": quantization,
                    "load_in_4bit": not args.no_4bit,
                    "lm_scoring_method": args.lm_scoring_method,
                    "distribution": dist["distribution"],
                    "params": dist["params"],
                    "prefixes": dist["prefixes"],
                    "prompt_type": prompt,
                    "n_samples": args.n_samples,
                    "decimals": args.decimals,
                    "seed": args.seed,
                    "mc_reliable_threshold": args.mc_reliable_threshold,
                    "generated_at": generated_at,
                    "icl_n_examples": args.icl_n_examples if uses_icl else None,
                    "icl_seed": args.icl_seed if uses_icl else None,
                    "icl_example_selection": ("shuffled_empirical_quantiles" if uses_icl else None),
                })

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")

    print(f"Wrote {len(rows)} configs to {out}")
    by_stage = {}
    for r in rows:
        key = (r["model_size_class"], r["model_family"], r["model_stage"])
        by_stage[key] = by_stage.get(key, 0) + 1
    for key, count in sorted(by_stage.items()):
        print(f"  {key}: {count}")

def prompt_metadata(prompt: str) -> tuple[str, int | None, str]:
    """
    Return:
    - prompt family
    - explanatory level, if applicable
    - prompt intervention/protocol
    """

    if prompt in {"short", "plain", "formal"}:
        return "direct", None, "direct_instruction"

    if prompt.startswith("explanatory_"):
        level = int(prompt.rsplit("_", 1)[1])
        return "explanatory", level, "explanatory_guidance"

    if prompt == "cot":
        return "reasoning_examples", None, "internal_reasoning_cue"

    if prompt == "icl":
        return "reasoning_examples", None, "representative_icl"

    if prompt == "icl_cot":
        return (
            "reasoning_examples",
            None,
            "representative_icl_with_internal_reasoning",
        )

    raise ValueError(f"Unknown prompt type: {prompt!r}")

if __name__ == "__main__":
    main()
