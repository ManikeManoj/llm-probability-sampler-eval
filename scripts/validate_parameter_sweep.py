#!/usr/bin/env python3
"""Validate a generated parameter-sweep JSONL before submission."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()

    path = Path(args.config)
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not rows:
        raise SystemExit("Config is empty.")

    run_ids = [row["run_id"] for row in rows]
    duplicates = [run_id for run_id, count in Counter(run_ids).items() if count > 1]
    if duplicates:
        raise SystemExit(f"Duplicate run IDs: {duplicates[:10]}")

    errors: list[str] = []
    for index, row in enumerate(rows):
        prefixes = row["prefixes"]
        if isinstance(prefixes, str):
            prefixes = [part.strip() for part in prefixes.split(",") if part.strip()]
        if not prefixes or prefixes[0] != "ROOT":
            errors.append(f"row {index}: ROOT is missing or not first")
        if len(prefixes) != len(set(prefixes)):
            errors.append(f"row {index}: duplicate prefixes")
        if float(row["params"].get("std", 1.0)) <= 0:
            errors.append(f"row {index}: std must be positive")
        if float(row["params"].get("scale", 1.0)) <= 0:
            errors.append(f"row {index}: scale must be positive")
        if float(row["params"].get("rate", 1.0)) <= 0:
            errors.append(f"row {index}: rate must be positive")
        if row["distribution"] == "uniform" and not row["params"]["low"] < row["params"]["high"]:
            errors.append(f"row {index}: uniform low must be below high")
        if row["distribution"] == "beta" and (
            row["params"]["alpha"] <= 0 or row["params"]["beta"] <= 0
        ):
            errors.append(f"row {index}: beta parameters must be positive")

    if errors:
        raise SystemExit("\n".join(errors[:30]))

    print(f"Valid config: {path}")
    print(f"Rows:       {len(rows)}")
    print(f"Models:     {len(set(row['model_alias'] for row in rows))}")
    print(f"Settings:   {len(set(row['setting_id'] for row in rows))}")
    print(f"Prompts:    {sorted(set(row['prompt_type'] for row in rows))}")
    print(f"Precision:  {sorted(set(row['quantization'] for row in rows))}")
    print("\nRows by distribution:")
    counts = Counter(row["distribution"] for row in rows)
    for name, count in sorted(counts.items()):
        print(f"  {name:12s} {count}")
    print("\nRows by model:")
    counts = Counter(row["model_alias"] for row in rows)
    for name, count in sorted(counts.items()):
        print(f"  {name:28s} {count}")


if __name__ == "__main__":
    main()
