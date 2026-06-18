import argparse
from datetime import datetime
from unicodedata import name
from csv_compare import RunConfig, run_experiment
import json
from distributions import DistributionSpec, default_support_for_distribution, format_distribution_params



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen3-4B", help="HuggingFace model name to use for next-token predictions.")
    #distribution parameters
    parser.add_argument("--distribution", type=str, default="normal", choices=["normal", "uniform", "exponential", "beta", "laplace", "lognormal"], help="Distribution family to use for sampling(normal, uniform, exponential, beta, laplace).")
    parser.add_argument("--params", type=str, default=None, help="JSON string of distribution parameters. Example: '{\"mean\": 0.0, \"std\": 1.0}'")
    parser.add_argument("--mean", type=float, default=0.0, help="Mean for normal distribution.")
    parser.add_argument("--std", type=float, default=1.0, help="Standard deviation for normal distribution.")
    parser.add_argument("--lower", type=float, default= None)
    parser.add_argument("--upper", type=float, default= None)
    
    #sample parameters
    parser.add_argument("--n-samples", type=int, default=500000)
    parser.add_argument("--decimals", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    #prefixes and outputs
    parser.add_argument("--prefixes",type=str,required=True,help='Comma-separated prefixes. Use ROOT for the empty prefix, e.g. ROOT,-,-1,-1.,0,0.,1',)
    parser.add_argument("--mc-reliable-threshold", type=int, default=1000)
    parser.add_argument("--token-level-csv", type=str, default="outputs/token_level_comparison.csv")
    parser.add_argument("--prefix-summary-csv", type=str, default="outputs/prefix_summary.csv")
    #prompt and support types
    parser.add_argument("--prompt-type", type=str, default="plain", help="One of: short, plain, formal, "
                             "explanatory_1, explanatory_2, explanatory_3, explanatory_4, " 
                             "cot, icl, icl_cot")
    parser.add_argument("--support-mode",type=str,choices=["positive", "bounded", "agnostic"],default=None,help="How to handle distribution support: 'positive' means only positive outputs are valid, 'bounded' means only outputs within [lower, upper] are valid, and 'agnostic' means all outputs are valid. By default, this is inferred from the distribution choice (e.g. 'exponential' -> 'positive', 'beta' -> 'bounded', 'normal' -> 'agnostic').")
    parser.add_argument("--allow-negative",action="store_true",help="Allow '-' as a valid initial token / signed outputs.",)
    parser.add_argument("--force-no-negative",action="store_true",help="Force negative outputs to be disallowed, even if the distribution default allows them.",)
    parser.add_argument("--icl-n-examples", type=int, default=5,help="Number of in-context examples for icl / icl_cot prompts.")
    parser.add_argument("--icl-seed", type=int, default=0,help="RNG seed used to draw in-context examples. " 
                        "Change this to test sensitivity to example choice.")
    parser.add_argument("--run-id", type=str, default=None,help="Override the auto-generated run ID.")


    return parser.parse_args()



def short_num_tag(x: float) -> str:
    return str(x).replace(".", "p").replace("-", "m")


if __name__ == "__main__":
    args = parse_args()

    if args.params is not None:
        params = json.loads(args.params)
    else:
        if args.distribution == "normal":
            params = {"mean": args.mean, "std": args.std}
        else:
            raise ValueError( f"For distribution={args.distribution!r}, please pass --params as JSON.")
        
    dist_spec = DistributionSpec(name=args.distribution, params=params)
    default_lower, default_upper, default_allow_negative, default_support_mode = default_support_for_distribution(dist_spec)
    lower = args.lower if args.lower is not None else default_lower
    upper = args.upper if args.upper is not None else default_upper

    support_mode = args.support_mode if args.support_mode is not None else default_support_mode

    allow_negative = default_allow_negative
    if args.allow_negative:
        allow_negative = True
    if args.force_no_negative:
        allow_negative = False


    
    def prompt_tag(pt: str) -> str:
        replacements = {
            "explanatory_1": "exp1",
            "explanatory_2": "exp2",
            "explanatory_3": "exp3",
            "explanatory_4": "exp4",
            "icl_cot":       "iclcot",
            "plain":         "pln",
            "formal":        "frm",
            "short":         "sht",
            "cot":           "cot",
            "icl":           "icl",
        }
        return replacements.get(pt, pt)
    
    def short_model_tag(model_name: str) -> str:
        name = model_name.lower()

        if "qwen" in name:
            return "q"
        if "gemma" in name:
            return "g"
        if "mistral" in name or "ministral" in name:
            return "m"
        if "deepseek" in name:
            return "d"
        if "llama" in name:
            return "l"

        return "x"
    
    pt_tag = prompt_tag(args.prompt_type)
    model_tag = short_model_tag(args.model_name)
    
    timestamp = datetime.now().strftime("%m%d_%H%M%S")

    icl_tag = ""
    if args.prompt_type in {"icl", "icl_cot"}:
        icl_tag = f"_n{args.icl_n_examples}_s{args.icl_seed}"

    run_id = args.run_id or f"{timestamp}_{model_tag}_{args.distribution}_{pt_tag}{icl_tag}"

    raw_prefixes = [p.strip() for p in args.prefixes.split(",")]
    prefixes = ["" if p == "ROOT" else p for p in raw_prefixes]

    config = RunConfig(
        model_name=args.model_name,
        mean=args.mean,
        std=args.std,
        lower=lower,
        upper=upper,
        n_samples=args.n_samples,
        decimals=args.decimals,
        prefixes=prefixes,
        mc_reliable_threshold=args.mc_reliable_threshold,
        token_level_csv=f"outputs/token_level_{run_id}.csv",
        prefix_summary_csv=f"outputs/prefix_summary_{run_id}.csv",
        prompt_type=args.prompt_type,
        support_mode=support_mode,
        allow_negative=args.allow_negative,
        icl_n_examples=args.icl_n_examples,
        icl_seed=args.icl_seed,
        seed=args.seed,
        run_id=run_id,
        distribution=args.distribution,
        params=params,
    )


    results = run_experiment(config)

    print("\nFinished run:")
    for r in results:
        print(
            f"prefix={repr(r['prefix'])} "
            f"kind={r.get('prefix_kind', 'NA')} "
            f"count={r['mc_prefix_count']} "
            f"mc_reliable={r['mc_reliable']} "
            f"TV(AN,LM)={r['tv_analytic_lm']:.6f}"
        )