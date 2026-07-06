#!/usr/bin/env python3
"""Merge prefix_summary CSVs and attach metadata from the JSONL config."""
import argparse
import glob
import json
from pathlib import Path
import pandas as pd

META_COLS = [
    "run_id", "model_alias", "model_family", "model_size_class", "model_stage",
    "reasoning_available", "prompt_protocol", "reasoning_protocol", "quantization", "notes",
]


def read_config(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="outputs/prefix_summary_*.csv", help="Glob for prefix summary CSVs")
    ap.add_argument("--config", required=True, help="JSONL config used to generate the runs")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    paths = sorted(glob.glob(args.glob))
    if not paths:
        raise SystemExit(f"No files matched: {args.glob}")

    dfs = []
    for p in paths:
        df = pd.read_csv(p, dtype={"prefix": "string", "run_id": "string"})
        df["source_file"] = Path(p).name
        dfs.append(df)
    merged = pd.concat(dfs, ignore_index=True)
    merged["prefix"] = merged["prefix"].fillna("ROOT")
    merged.loc[merged["prefix"].eq(""), "prefix"] = "ROOT"

    cfg = read_config(args.config)
    meta = cfg[[c for c in META_COLS if c in cfg.columns]].copy()
    out = merged.merge(meta, on="run_id", how="left", validate="many_to_one")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"Merged {len(paths)} files, {len(out)} rows -> {args.out}")
    missing = out["model_alias"].isna().sum() if "model_alias" in out.columns else len(out)
    if missing:
        print(f"WARNING: {missing} rows did not match config metadata by run_id")


if __name__ == "__main__":
    main()
