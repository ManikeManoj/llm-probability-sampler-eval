#!/usr/bin/env python3
"""Run one JSONL config row by calling src/run_compare.py.

This wrapper keeps SLURM scripts simple and preserves metadata in JSONL for plotting later.
It does not require run_compare.py to know model_family/model_variant yet.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_no} of {path}: {e}") from e
    return rows


def build_command(cfg: Dict[str, Any]) -> List[str]:
    prefixes = cfg["prefixes"]
    if not isinstance(prefixes, list) or not prefixes:
        raise ValueError("Config must contain non-empty list field 'prefixes'.")

    run_tag = os.environ.get("RUN_TAG", "").strip()
    run_id = cfg.get("run_id") or f"{cfg['model_alias']}_{cfg['distribution']}_{cfg['prompt_type']}"
    if run_tag:
        run_id = f"{run_id}_{run_tag}"

    cmd: List[str] = [
        sys.executable,
        "src/run_compare.py",
        "--model-name", str(cfg["model_name"]),
        "--lm-scoring-method", str(cfg.get("lm_scoring_method", "auto")),
        "--distribution", str(cfg["distribution"]),
        "--params", json.dumps(cfg.get("params", {})),
        "--prompt-type", str(cfg["prompt_type"]),
        "--support-mode", str(cfg["support_mode"]),
        "--prefixes", ",".join(prefixes),
        "--n-samples", str(cfg.get("n_samples", 500000)),
        "--decimals", str(cfg.get("decimals", 3)),
        "--seed", str(cfg.get("seed", 42)),
        "--run-id", run_id,
    ]

    if cfg.get("allow_negative", False):
        cmd.append("--allow-negative")
    else:
        # Only add this if your run_compare.py supports it.
        # It was present in the latest thesis code discussed.
        cmd.append("--force-no-negative")

    if cfg.get("lower") is not None:
        cmd.extend(["--lower", str(cfg["lower"])])
    if cfg.get("upper") is not None:
        cmd.extend(["--upper", str(cfg["upper"])])

    if cfg.get("icl_n_examples") is not None:
        cmd.extend(["--icl-n-examples", str(cfg["icl_n_examples"])])
    if cfg.get("icl_seed") is not None:
        cmd.extend(["--icl-seed", str(cfg["icl_seed"])])

    return cmd


def run_one(cfg: Dict[str, Any], dry_run: bool = False) -> None:
    print("=" * 80)
    print("RUN_ID:", cfg.get("run_id"))
    print("MODEL:", cfg.get("model_alias"), "=>", cfg.get("model_name"))
    print("DIST:", cfg.get("distribution_label", cfg.get("distribution")), cfg.get("params"))
    print("PROMPT:", cfg.get("prompt_type"))
    print("PREFIXES:", cfg.get("prefixes"))
    cmd = build_command(cfg)
    print("COMMAND:")
    print(" ".join(json.dumps(x) if " " in x else x for x in cmd))
    if not dry_run:
        subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="JSONL config file")
    parser.add_argument("--index", type=int, required=True, help="Config index or chunk index")
    parser.add_argument("--chunk-size", type=int, default=1, help="Number of consecutive configs to run per task")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    configs = load_jsonl(Path(args.config))
    if args.chunk_size < 1:
        raise ValueError("--chunk-size must be >= 1")

    start = args.index * args.chunk_size
    end = min(start + args.chunk_size, len(configs))
    if start >= len(configs):
        raise IndexError(f"Start index {start} is outside config range 0..{len(configs)-1}")

    print(f"Loaded {len(configs)} configs from {args.config}")
    print(f"Running configs [{start}:{end}) with chunk_size={args.chunk_size}")

    for i in range(start, end):
        print(f"\n### Config {i}/{len(configs)-1}")
        run_one(configs[i], dry_run=args.dry_run)


if __name__ == "__main__":
    main()
