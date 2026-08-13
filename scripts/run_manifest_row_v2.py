#!/usr/bin/env python3

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path


def read_manifest(path):
    rows = []

    with Path(path).open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number}: {exc}"
                ) from exc

    return rows


def build_command(row):
    cmd = [
        sys.executable,
        "src/run_compare.py",

        "--model-name",
        row["model_name"],

        "--lm-scoring-method",
        row["lm_scoring_method"],

        "--prompt-protocol",
        row["prompt_protocol"],

        "--distribution",
        row["distribution"],

        "--params",
        json.dumps(row["params"]),

        "--prompt-type",
        row["prompt_type"],

        "--prefixes",
        ",".join(row["prefixes"]),

        "--n-samples",
        str(row["n_samples"]),

        "--decimals",
        str(row["decimals"]),

        "--seed",
        str(row["seed"]),

        "--mc-reliable-threshold",
        str(row["mc_reliable_threshold"]),

        "--icl-n-examples",
        str(row.get("icl_n_examples", 5)),

        "--icl-seed",
        str(row.get("icl_seed", 0)),

        "--run-id",
        row["run_id"],
    ]

    if not row["load_in_4bit"]:
        cmd.append("--no-4bit")

    return cmd


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--manifest",
        required=True,
    )

    parser.add_argument(
        "--index",
        required=True,
        type=int,
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
    )

    args = parser.parse_args()

    rows = read_manifest(args.manifest)

    if not 0 <= args.index < len(rows):
        raise IndexError(
            f"Index {args.index} outside "
            f"0..{len(rows)-1}"
        )

    row = rows[args.index]

    print("=" * 100)
    print(f"MANIFEST:   {args.manifest}")
    print(f"INDEX:      {args.index}")
    print(f"RUN ID:     {row['run_id']}")
    print(f"MODEL:      {row['model_name']}")
    print(f"PROTOCOL:   {row['prompt_protocol']}")
    print(f"DIST:       {row['distribution']} {row['params']}")
    print(f"PROMPT:     {row['prompt_type']}")
    print(f"PRECISION:  {row['precision']}")
    print(f"PREFIXES:   {row['prefixes']}")
    print("=" * 100)

    command = build_command(row)

    print()
    print("COMMAND")
    print(shlex.join(command))
    print()

    if args.dry_run:
        print("DRY RUN ONLY")
        return

    subprocess.run(
        command,
        check=True,
    )


if __name__ == "__main__":
    main()