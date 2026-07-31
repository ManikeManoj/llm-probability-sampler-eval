#!/usr/bin/env python3
"""Run one or more parameter-sweep JSONL rows through src/run_compare.py."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
    return rows


def prefixes_as_string(value: Any) -> str:
    if isinstance(value, str):
        prefixes = [part.strip() for part in value.split(",") if part.strip()]
    elif isinstance(value, list):
        prefixes = [str(part).strip() for part in value if str(part).strip()]
    else:
        raise TypeError("'prefixes' must be a comma-separated string or a list.")
    if not prefixes or prefixes[0] != "ROOT":
        raise ValueError("Every sweep row must begin prefixes with ROOT.")
    if len(prefixes) != len(set(prefixes)):
        raise ValueError(f"Duplicate prefixes detected: {prefixes}")
    return ",".join(prefixes)


def command_from_config(row: dict[str, Any]) -> list[str]:
    required = ["model_name", "distribution", "params", "prompt_type", "prefixes", "run_id"]
    missing = [field for field in required if field not in row]
    if missing:
        raise KeyError(f"Config row is missing fields: {missing}")

    cmd = [
        sys.executable,
        "src/run_compare.py",
        "--model-name", str(row["model_name"]),
        "--lm-scoring-method", str(row.get("lm_scoring_method", "auto")),
        "--distribution", str(row["distribution"]),
        "--params", json.dumps(row["params"], sort_keys=True),
        "--prompt-type", str(row.get("prompt_type", "plain")),
        "--prefixes", prefixes_as_string(row["prefixes"]),
        "--n-samples", str(row.get("n_samples", 500000)),
        "--decimals", str(row.get("decimals", 3)),
        "--seed", str(row.get("seed", 42)),
        "--mc-reliable-threshold", str(row.get("mc_reliable_threshold", 1000)),
        "--run-id", str(row["run_id"]),
    ]

    if not bool(row.get("load_in_4bit", True)):
        cmd.append("--no-4bit")

    if row.get("prompt_type") in {"icl", "icl_cot"}:
        cmd.extend(["--icl-n-examples", str(row.get("icl_n_examples", 5))])
        cmd.extend(["--icl-seed", str(row.get("icl_seed", 0))])

    return cmd


def outputs_exist(run_id: str) -> bool:
    return (
        Path(f"outputs/token_level_{run_id}.csv").is_file()
        and Path(f"outputs/prefix_summary_{run_id}.csv").is_file()
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--index", type=int, required=True,
                        help="Zero-based chunk index, normally SLURM_ARRAY_TASK_ID.")
    parser.add_argument("--chunk-size", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--rerun-existing", action="store_true")
    args = parser.parse_args()

    if args.index < 0:
        raise ValueError("--index must be >= 0")
    if args.chunk_size < 1:
        raise ValueError("--chunk-size must be >= 1")

    rows = load_jsonl(Path(args.config))
    start = args.index * args.chunk_size
    end = min(start + args.chunk_size, len(rows))
    if start >= len(rows):
        raise IndexError(f"Chunk starts at {start}, but config contains {len(rows)} rows.")

    Path("outputs").mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)

    print(f"Loaded {len(rows)} configs; executing rows [{start}:{end}).", flush=True)
    for row_index in range(start, end):
        row = rows[row_index]
        run_id = str(row["run_id"])
        print("=" * 100, flush=True)
        print(f"Config index: {row_index}", flush=True)
        print(f"Run ID:       {run_id}", flush=True)
        print(f"Model:        {row.get('model_alias')} -> {row.get('model_name')}", flush=True)
        print(f"Setting:      {row.get('parameter_label')} [{row.get('sweep_axis')}]", flush=True)
        print(f"Prompt:       {row.get('prompt_type')}", flush=True)
        print(f"Precision:    {row.get('quantization')}", flush=True)
        print(f"Prefixes:     {prefixes_as_string(row['prefixes'])}", flush=True)

        if outputs_exist(run_id) and not args.rerun_existing:
            print("Both output CSVs already exist; skipping.", flush=True)
            continue

        cmd = command_from_config(row)
        print("Command:", flush=True)
        print(" ".join(shlex.quote(arg) for arg in cmd), flush=True)
        if not args.dry_run:
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
