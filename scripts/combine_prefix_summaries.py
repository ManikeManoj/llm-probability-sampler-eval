#!/usr/bin/env python3
"""Merge prefix_summary CSVs and attach metadata from JSONL configs.

Use this after SLURM outputs exist. It is intentionally small and robust.
"""

from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


def load_configs(path: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                row = json.loads(line)
                rows.append({
                    "run_id_base": row.get("run_id"),
                    "model_alias": row.get("model_alias"),
                    "model_family": row.get("model_family"),
                    "model_variant": row.get("model_variant"),
                    "reasoning_mode": row.get("reasoning_mode"),
                    "prompt_protocol": row.get("prompt_protocol"),
                    "distribution_label": row.get("distribution_label"),
                    "distribution": row.get("distribution"),
                    "prompt_type_cfg": row.get("prompt_type"),
                    "params_json": json.dumps(row.get("params", {}), sort_keys=True),
                })
    return pd.DataFrame(rows)


def infer_run_id_base(filename: str, config_bases: List[str]) -> str | None:
    name = Path(filename).name
    # prefix_summary_<run_id>.csv
    if name.startswith("prefix_summary_") and name.endswith(".csv"):
        rid = name[len("prefix_summary_"):-len(".csv")]
    else:
        rid = Path(filename).stem

    # If RUN_TAG was appended, the filename run_id may be <base>_<timestamp>.
    # Choose the longest config base that is a prefix of the actual run id.
    matches = [base for base in config_bases if isinstance(base, str) and rid.startswith(base)]
    if not matches:
        return None
    return max(matches, key=len)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-glob", default="outputs/prefix_summary_*.csv")
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    config_df = load_configs(Path(args.config))
    config_bases = config_df["run_id_base"].dropna().tolist()

    frames = []
    for path in sorted(glob.glob(args.outputs_glob)):
        df = pd.read_csv(path, dtype={"prefix": str})
        df["prefix"] = df["prefix"].fillna("ROOT")
        df["source_file"] = Path(path).name
        df["run_id_base"] = infer_run_id_base(path, config_bases)
        frames.append(df)

    if not frames:
        raise SystemExit(f"No files matched: {args.outputs_glob}")

    merged = pd.concat(frames, ignore_index=True)
    merged = merged.merge(config_df, on="run_id_base", how="left")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out, index=False)
    print(f"Merged {len(frames)} files, {len(merged)} rows -> {out}")
    missing = merged["model_alias"].isna().sum()
    if missing:
        print(f"WARNING: {missing} rows could not be matched to config metadata. Check RUN_TAG/run_id naming.")


if __name__ == "__main__":
    main()
