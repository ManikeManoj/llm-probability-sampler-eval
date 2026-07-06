#!/usr/bin/env python3
"""Run one config row from a JSONL file by invoking src/run_compare.py."""
import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path


def read_rows(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def command_from_config(row: dict) -> list[str]:
    cmd = [
        sys.executable,
        "src/run_compare.py",
        "--model-name", row["model_name"],
        "--lm-scoring-method", row.get("lm_scoring_method", "auto"),
        "--distribution", row["distribution"],
        "--params", json.dumps(row["params"]),
        "--prompt-type", row.get("prompt_type", "plain"),
        "--prefixes", row["prefixes"],
        "--n-samples", str(row.get("n_samples", 500000)),
        "--decimals", str(row.get("decimals", 3)),
        "--seed", str(row.get("seed", 42)),
        "--mc-reliable-threshold", str(row.get("mc_reliable_threshold", 1000)),
        "--run-id", row["run_id"],
    ]
    if not row.get("load_in_4bit", True):
        cmd.append("--no-4bit")
    return cmd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--index", type=int, default=None, help="Run a single zero-based row index")
    ap.add_argument("--start", type=int, default=None, help="Run rows in [start, end)")
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    rows = read_rows(args.config)
    if args.index is not None:
        selected = [(args.index, rows[args.index])]
    else:
        start = args.start or 0
        end = args.end if args.end is not None else len(rows)
        selected = list(enumerate(rows[start:end], start=start))

    Path("outputs").mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)

    for i, row in selected:
        cmd = command_from_config(row)
        print("=" * 90)
        print(f"Config index: {i}")
        print(f"run_id: {row.get('run_id')}")
        print(f"model_alias: {row.get('model_alias')} | family={row.get('model_family')} | size={row.get('model_size_class')} | stage={row.get('model_stage')}")
        print(f"distribution: {row.get('distribution')} | prompt={row.get('prompt_type')} | quantization={row.get('quantization')}")
        print("Command:")
        print(" ".join(shlex.quote(x) for x in cmd))
        if not args.dry_run:
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
