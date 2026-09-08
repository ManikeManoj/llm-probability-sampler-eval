#!/usr/bin/env python3
"""Audit and merge sample-generation JSONL shards.

The generator stores one condition directory per manifest row and five JSONL
seed blocks per condition.  This script treats those raw shards as immutable
source data.  It validates every condition, seed, sample index, configuration
hash, prompt hash, and record invariant before atomically publishing:

* one sample-level CSV;
* one seed-level summary CSV;
* one condition-level summary CSV;
* one condition-metadata JSONL file; and
* one merge-audit JSON file.

No Wasserstein distance or other target-comparison metric is calculated here.
Those belong in the analysis notebook after the merge has passed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


MANIFEST_SCHEMA_VERSION = "sample-generation-manifest-v1"
GENERATOR_SCHEMA_VERSION = "sample-generation-v2"
MERGE_SCHEMA_VERSION = "sample-generation-merge-v1"
DEFAULT_EXPECTED_CONDITIONS = 204

OUTPUT_NAMES = {
    "samples": "sample_generation_samples_v2.csv",
    "seeds": "sample_generation_seed_summary_v2.csv",
    "conditions": "sample_generation_condition_summary_v2.csv",
    "metadata": "sample_generation_condition_metadata_v2.jsonl",
    "audit": "sample_generation_merge_audit_v2.json",
}

TERMINATION_REASONS = {"eos", "newline_boundary", "max_new_tokens"}

SAMPLE_COLUMNS = (
    "sample_id",
    "manifest_global_index",
    "manifest_layer_index",
    "experiment_layer",
    "target_role",
    "model_label",
    "model_name",
    "model_display_name",
    "model_size_tier",
    "prompt_protocol",
    "protocol_short",
    "distribution",
    "parameter_id",
    "distribution_params_json",
    "support_mode",
    "lower",
    "upper",
    "prompt_type",
    "uses_icl",
    "icl_n_examples",
    "icl_seed",
    "sampling_seed",
    "sample_index",
    "global_sample_index",
    "raw_completion",
    "response_text",
    "normalized_text",
    "parsed_value",
    "numeric_valid",
    "canonical_format_valid",
    "support_valid",
    "strict_valid",
    "negative_zero",
    "extra_after_first_line",
    "termination_reason",
    "truncated",
    "error_type",
    "generated_token_count",
    "boundary_token_id",
    "generated_token_ids_json",
    "config_hash",
    "prompt_hash",
    "generator_schema_version",
)

SUMMARY_ID_COLUMNS = (
    "manifest_global_index",
    "manifest_layer_index",
    "experiment_layer",
    "run_id",
    "model_label",
    "model_name",
    "model_display_name",
    "model_size_tier",
    "prompt_protocol",
    "protocol_short",
    "distribution",
    "parameter_id",
    "distribution_params_json",
    "target_role",
    "support_mode",
    "lower",
    "upper",
    "prompt_type",
    "uses_icl",
    "icl_n_examples",
    "icl_seed",
)

STAT_COLUMNS = (
    "attempted_count",
    "numeric_count",
    "canonical_count",
    "support_count",
    "strict_count",
    "truncated_count",
    "negative_zero_count",
    "extra_after_first_line_count",
    "numeric_rate",
    "canonical_rate",
    "support_rate",
    "strict_rate",
    "truncated_rate",
    "negative_zero_rate",
    "extra_after_first_line_rate",
    "strict_n_unique",
    "strict_mean",
    "strict_sample_std",
    "strict_min",
    "strict_q25",
    "strict_median",
    "strict_q75",
    "strict_max",
    "error_counts_json",
    "termination_counts_json",
)

SEED_COLUMNS = SUMMARY_ID_COLUMNS + ("sampling_seed",) + STAT_COLUMNS

CONDITION_COLUMNS = SUMMARY_ID_COLUMNS + (
    "sampling_seeds_json",
    "samples_per_seed",
    "expected_seed_blocks",
    "expected_attempts",
    "batch_size",
    "max_new_tokens",
    "config_hash",
    "prompt_hash",
    "prompt_text",
    "context_hash",
    "model_backend",
    "generation_boundary_token_ids_json",
    "generation_boundary_tokens_json",
    "software_versions_json",
) + STAT_COLUMNS


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _pretty_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        indent=2,
    ) + "\n"


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_float(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric; received {value!r}.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite; received {value!r}.")
    return result


def _quantile(values: Sequence[float], probability: float) -> float | None:
    """NumPy-compatible default linear quantile for a small sorted sequence."""

    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return float(ordered[lower_index])
    weight = position - lower_index
    return float(
        ordered[lower_index] * (1.0 - weight)
        + ordered[upper_index] * weight
    )


@dataclass
class SampleAccumulator:
    """Accumulate compliance and strict-value summaries for one group."""

    keep_values: bool = True
    attempted_count: int = 0
    numeric_count: int = 0
    canonical_count: int = 0
    support_count: int = 0
    strict_count: int = 0
    truncated_count: int = 0
    negative_zero_count: int = 0
    extra_after_first_line_count: int = 0
    error_counts: Counter[str] = field(default_factory=Counter)
    termination_counts: Counter[str] = field(default_factory=Counter)
    strict_values: list[float] = field(default_factory=list)

    def add(self, row: Mapping[str, Any]) -> None:
        self.attempted_count += 1
        for field_name, count_name in (
            ("numeric_valid", "numeric_count"),
            ("canonical_format_valid", "canonical_count"),
            ("support_valid", "support_count"),
            ("strict_valid", "strict_count"),
            ("truncated", "truncated_count"),
            ("negative_zero", "negative_zero_count"),
            ("extra_after_first_line", "extra_after_first_line_count"),
        ):
            if bool(row[field_name]):
                setattr(self, count_name, getattr(self, count_name) + 1)

        error_type = row.get("error_type")
        if error_type is not None:
            self.error_counts[str(error_type)] += 1
        self.termination_counts[str(row["termination_reason"])] += 1

        if row["strict_valid"]:
            value = _finite_float(row["parsed_value"], "strict parsed_value")
            if self.keep_values:
                self.strict_values.append(value)

    def _rate(self, count: int) -> float | None:
        return None if self.attempted_count == 0 else count / self.attempted_count

    def to_dict(self) -> dict[str, Any]:
        values = self.strict_values
        if self.keep_values and len(values) != self.strict_count:
            raise AssertionError("Strict-value accumulator is inconsistent.")
        result: dict[str, Any] = {
            "attempted_count": self.attempted_count,
            "numeric_count": self.numeric_count,
            "canonical_count": self.canonical_count,
            "support_count": self.support_count,
            "strict_count": self.strict_count,
            "truncated_count": self.truncated_count,
            "negative_zero_count": self.negative_zero_count,
            "extra_after_first_line_count": self.extra_after_first_line_count,
            "numeric_rate": self._rate(self.numeric_count),
            "canonical_rate": self._rate(self.canonical_count),
            "support_rate": self._rate(self.support_count),
            "strict_rate": self._rate(self.strict_count),
            "truncated_rate": self._rate(self.truncated_count),
            "negative_zero_rate": self._rate(self.negative_zero_count),
            "extra_after_first_line_rate": self._rate(
                self.extra_after_first_line_count
            ),
            "strict_n_unique": len(set(values)) if self.keep_values else None,
            "strict_mean": statistics.fmean(values) if values else None,
            "strict_sample_std": statistics.stdev(values) if len(values) > 1 else None,
            "strict_min": min(values) if values else None,
            "strict_q25": _quantile(values, 0.25),
            "strict_median": _quantile(values, 0.50),
            "strict_q75": _quantile(values, 0.75),
            "strict_max": max(values) if values else None,
            "error_counts_json": _canonical_json(dict(sorted(self.error_counts.items()))),
            "termination_counts_json": _canonical_json(
                dict(sorted(self.termination_counts.items()))
            ),
        }
        return result


def _load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except FileNotFoundError as exc:
        raise ValueError(f"Missing required file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON file: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Expected one JSON object in {path}.")
    return value


def load_manifest(path: Path, expected_conditions: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    raise ValueError(f"Blank manifest line at {path}:{line_number}.")
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid manifest JSON at {path}:{line_number}."
                    ) from exc
                if not isinstance(row, dict):
                    raise ValueError(f"Manifest row {line_number} is not an object.")
                rows.append(row)
    except FileNotFoundError as exc:
        raise ValueError(f"Manifest not found: {path}") from exc

    if len(rows) != expected_conditions:
        raise ValueError(
            f"Manifest has {len(rows)} conditions; expected {expected_conditions}."
        )
    run_ids: set[str] = set()
    identities: set[tuple[Any, ...]] = set()
    for index, row in enumerate(rows):
        if row.get("manifest_schema_version") != MANIFEST_SCHEMA_VERSION:
            raise ValueError(f"Manifest schema mismatch at row {index}.")
        if row.get("generator_schema_version") != GENERATOR_SCHEMA_VERSION:
            raise ValueError(f"Generator schema mismatch at manifest row {index}.")
        if row.get("manifest_global_index") != index:
            raise ValueError(f"Non-contiguous manifest_global_index at row {index}.")
        run_id = row.get("run_id")
        if not isinstance(run_id, str) or not run_id:
            raise ValueError(f"Invalid run_id at manifest row {index}.")
        if run_id in run_ids:
            raise ValueError(f"Duplicate manifest run_id: {run_id}")
        run_ids.add(run_id)
        identity = (
            row.get("model_name"),
            row.get("prompt_protocol"),
            row.get("distribution"),
            _canonical_json(row.get("params")),
            row.get("prompt_type"),
            row.get("icl_seed") if row.get("uses_icl") else None,
        )
        if identity in identities:
            raise ValueError(f"Duplicate scientific condition: {run_id}")
        identities.add(identity)
    return rows


def _assert_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise ValueError(f"{label} mismatch: expected {expected!r}, found {actual!r}.")


def _assert_float_close(actual: Any, expected: Any, label: str) -> None:
    if actual is None or expected is None:
        if actual is not expected:
            raise ValueError(f"{label} mismatch: expected {expected!r}, found {actual!r}.")
        return
    actual_float = _finite_float(actual, label)
    expected_float = _finite_float(expected, label)
    if not math.isclose(actual_float, expected_float, rel_tol=1e-10, abs_tol=1e-12):
        raise ValueError(
            f"{label} mismatch: expected {expected_float!r}, found {actual_float!r}."
        )


def validate_condition_files(
    manifest_row: Mapping[str, Any],
    condition_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    metadata = _load_json(condition_dir / "condition_metadata.json")
    run_summary = _load_json(condition_dir / "run_summary.json")
    run_id = str(manifest_row["run_id"])

    _assert_equal(metadata.get("schema_version"), GENERATOR_SCHEMA_VERSION,
                  f"{run_id} metadata schema")
    condition = metadata.get("condition")
    if not isinstance(condition, dict):
        raise ValueError(f"{run_id} metadata has no condition object.")

    direct_fields = (
        "model_name",
        "model_label",
        "distribution",
        "parameter_id",
        "prompt_type",
        "prompt_protocol",
        "experiment_layer",
        "run_id",
        "support_mode",
        "lower",
        "upper",
        "icl_n_examples",
        "icl_seed",
        "sampling_seeds",
        "samples_per_seed",
        "batch_size",
        "max_new_tokens",
    )
    for key in direct_fields:
        _assert_equal(condition.get(key), manifest_row.get(key), f"{run_id} {key}")
    _assert_equal(condition.get("params"), manifest_row.get("params"), f"{run_id} params")
    _assert_equal(condition.get("total_samples"), manifest_row.get("expected_attempts"),
                  f"{run_id} total_samples")
    for key, value in manifest_row.get("decoding", {}).items():
        if key == "precision":
            _assert_equal(condition.get("precision"), value, f"{run_id} precision")
        else:
            _assert_equal(condition.get(key), value, f"{run_id} decoding.{key}")

    config_hash = condition.get("config_hash")
    prompt_hash = metadata.get("prompt_hash")
    if not isinstance(config_hash, str) or len(config_hash) != 64:
        raise ValueError(f"{run_id} has an invalid config hash.")
    if not isinstance(prompt_hash, str) or len(prompt_hash) != 64:
        raise ValueError(f"{run_id} has an invalid prompt hash.")
    _assert_equal(run_summary.get("schema_version"), GENERATOR_SCHEMA_VERSION,
                  f"{run_id} summary schema")
    _assert_equal(run_summary.get("status"), "complete", f"{run_id} status")
    _assert_equal(run_summary.get("run_id"), run_id, f"{run_id} summary run_id")
    _assert_equal(run_summary.get("config_hash"), config_hash,
                  f"{run_id} summary config_hash")
    _assert_equal(run_summary.get("expected_samples"), manifest_row["expected_attempts"],
                  f"{run_id} expected_samples")
    _assert_equal(run_summary.get("stored_samples"), manifest_row["expected_attempts"],
                  f"{run_id} stored_samples")
    _assert_equal(run_summary.get("sampling_seed_blocks"), manifest_row["sampling_seeds"],
                  f"{run_id} sampling_seed_blocks")
    if not isinstance(metadata.get("prompt"), str):
        raise ValueError(f"{run_id} metadata has no prompt text.")
    if not isinstance(metadata.get("rendered_context"), str):
        raise ValueError(f"{run_id} metadata has no rendered context.")
    if not metadata.get("model_backend"):
        raise ValueError(f"{run_id} metadata has no model backend.")
    return metadata, run_summary


def validate_sample_record(
    row: Mapping[str, Any],
    *,
    manifest_row: Mapping[str, Any],
    config_hash: str,
    prompt_hash: str,
    sampling_seed: int,
    seed_position: int,
    sample_index: int,
    source: str,
) -> None:
    required = (
        "schema_version", "config_hash", "prompt_hash", "run_id",
        "model_name", "model_label", "prompt_protocol", "distribution",
        "distribution_params", "parameter_id", "prompt_type", "icl_seed",
        "sampling_seed", "sample_index", "global_sample_index",
        "raw_completion", "response_text", "normalized_text",
        "generated_token_ids", "generated_token_count", "boundary_token_id",
        "parsed_value", "numeric_valid", "canonical_format_valid",
        "support_valid", "strict_valid", "negative_zero",
        "extra_after_first_line", "termination_reason", "truncated",
        "error_type",
    )
    missing = [key for key in required if key not in row]
    if missing:
        raise ValueError(f"{source} is missing fields: {missing}")

    expected_fields = {
        "schema_version": GENERATOR_SCHEMA_VERSION,
        "config_hash": config_hash,
        "prompt_hash": prompt_hash,
        "run_id": manifest_row["run_id"],
        "model_name": manifest_row["model_name"],
        "model_label": manifest_row["model_label"],
        "prompt_protocol": manifest_row["prompt_protocol"],
        "distribution": manifest_row["distribution"],
        "distribution_params": manifest_row["params"],
        "parameter_id": manifest_row["parameter_id"],
        "prompt_type": manifest_row["prompt_type"],
        "icl_seed": manifest_row["icl_seed"],
        "sampling_seed": sampling_seed,
        "sample_index": sample_index,
        "global_sample_index": (
            seed_position * manifest_row["samples_per_seed"] + sample_index
        ),
    }
    for key, expected in expected_fields.items():
        _assert_equal(row.get(key), expected, f"{source} {key}")

    for key in (
        "numeric_valid", "canonical_format_valid", "support_valid",
        "strict_valid", "negative_zero", "extra_after_first_line", "truncated",
    ):
        if not isinstance(row[key], bool):
            raise ValueError(f"{source} {key} is not Boolean.")
    for key in ("raw_completion", "response_text", "normalized_text"):
        if not isinstance(row[key], str):
            raise ValueError(f"{source} {key} is not text.")

    token_ids = row["generated_token_ids"]
    if not isinstance(token_ids, list) or any(
        isinstance(token_id, bool) or not isinstance(token_id, int)
        for token_id in token_ids
    ):
        raise ValueError(f"{source} has invalid generated_token_ids.")
    _assert_equal(row["generated_token_count"], len(token_ids),
                  f"{source} generated_token_count")

    parsed_value = row["parsed_value"]
    if parsed_value is not None:
        _finite_float(parsed_value, f"{source} parsed_value")
    _assert_equal(row["numeric_valid"], parsed_value is not None,
                  f"{source} numeric/parsed invariant")
    if row["canonical_format_valid"] and not row["numeric_valid"]:
        raise ValueError(f"{source} is canonical but not numeric.")
    if row["support_valid"] and not row["numeric_valid"]:
        raise ValueError(f"{source} is support-valid but not numeric.")
    expected_strict = bool(
        row["numeric_valid"]
        and row["canonical_format_valid"]
        and row["support_valid"]
        and not row["truncated"]
    )
    _assert_equal(row["strict_valid"], expected_strict,
                  f"{source} strict-valid invariant")

    termination = row["termination_reason"]
    if termination not in TERMINATION_REASONS:
        raise ValueError(f"{source} has unknown termination_reason {termination!r}.")
    _assert_equal(row["truncated"], termination == "max_new_tokens",
                  f"{source} truncation invariant")
    if row["error_type"] is not None and not isinstance(row["error_type"], str):
        raise ValueError(f"{source} has invalid error_type.")


def _summary_identifiers(manifest_row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "manifest_global_index": manifest_row["manifest_global_index"],
        "manifest_layer_index": manifest_row["manifest_layer_index"],
        "experiment_layer": manifest_row["experiment_layer"],
        "run_id": manifest_row["run_id"],
        "model_label": manifest_row["model_label"],
        "model_name": manifest_row["model_name"],
        "model_display_name": manifest_row["model_display_name"],
        "model_size_tier": manifest_row["model_size_tier"],
        "prompt_protocol": manifest_row["prompt_protocol"],
        "protocol_short": manifest_row["protocol_short"],
        "distribution": manifest_row["distribution"],
        "parameter_id": manifest_row["parameter_id"],
        "distribution_params_json": _canonical_json(manifest_row["params"]),
        "target_role": manifest_row["target_role"],
        "support_mode": manifest_row["support_mode"],
        "lower": manifest_row["lower"],
        "upper": manifest_row["upper"],
        "prompt_type": manifest_row["prompt_type"],
        "uses_icl": manifest_row["uses_icl"],
        "icl_n_examples": manifest_row["icl_n_examples"],
        "icl_seed": manifest_row["icl_seed"],
    }


def _sample_csv_row(
    manifest_row: Mapping[str, Any],
    record: Mapping[str, Any],
) -> dict[str, Any]:
    run_id = str(manifest_row["run_id"])
    sampling_seed = int(record["sampling_seed"])
    sample_index = int(record["sample_index"])
    return {
        "sample_id": f"{run_id}__seed{sampling_seed:010d}__sample{sample_index:06d}",
        "manifest_global_index": manifest_row["manifest_global_index"],
        "manifest_layer_index": manifest_row["manifest_layer_index"],
        "experiment_layer": manifest_row["experiment_layer"],
        "target_role": manifest_row["target_role"],
        "model_label": manifest_row["model_label"],
        "model_name": manifest_row["model_name"],
        "model_display_name": manifest_row["model_display_name"],
        "model_size_tier": manifest_row["model_size_tier"],
        "prompt_protocol": manifest_row["prompt_protocol"],
        "protocol_short": manifest_row["protocol_short"],
        "distribution": manifest_row["distribution"],
        "parameter_id": manifest_row["parameter_id"],
        "distribution_params_json": _canonical_json(manifest_row["params"]),
        "support_mode": manifest_row["support_mode"],
        "lower": manifest_row["lower"],
        "upper": manifest_row["upper"],
        "prompt_type": manifest_row["prompt_type"],
        "uses_icl": manifest_row["uses_icl"],
        "icl_n_examples": manifest_row["icl_n_examples"],
        "icl_seed": record["icl_seed"],
        "sampling_seed": sampling_seed,
        "sample_index": sample_index,
        "global_sample_index": record["global_sample_index"],
        "raw_completion": record["raw_completion"],
        "response_text": record["response_text"],
        "normalized_text": record["normalized_text"],
        "parsed_value": record["parsed_value"],
        "numeric_valid": record["numeric_valid"],
        "canonical_format_valid": record["canonical_format_valid"],
        "support_valid": record["support_valid"],
        "strict_valid": record["strict_valid"],
        "negative_zero": record["negative_zero"],
        "extra_after_first_line": record["extra_after_first_line"],
        "termination_reason": record["termination_reason"],
        "truncated": record["truncated"],
        "error_type": record["error_type"],
        "generated_token_count": record["generated_token_count"],
        "boundary_token_id": record["boundary_token_id"],
        "generated_token_ids_json": _canonical_json(record["generated_token_ids"]),
        "config_hash": record["config_hash"],
        "prompt_hash": record["prompt_hash"],
        "generator_schema_version": record["schema_version"],
    }


def _compare_accumulator_to_run_summary(
    accumulator: SampleAccumulator,
    run_summary: Mapping[str, Any],
    run_id: str,
) -> None:
    computed = accumulator.to_dict()
    _assert_equal(computed["attempted_count"], run_summary["stored_samples"],
                  f"{run_id} recomputed sample count")
    rate_map = {
        "numeric_valid": "numeric_rate",
        "canonical_format_valid": "canonical_rate",
        "support_valid": "support_rate",
        "strict_valid": "strict_rate",
        "truncated": "truncated_rate",
        "negative_zero": "negative_zero_rate",
        "extra_after_first_line": "extra_after_first_line_rate",
    }
    stored_rates = run_summary.get("rates", {})
    for stored_key, computed_key in rate_map.items():
        _assert_float_close(
            computed[computed_key],
            stored_rates.get(stored_key),
            f"{run_id} recomputed {stored_key} rate",
        )
    _assert_equal(
        json.loads(computed["error_counts_json"]),
        run_summary.get("error_counts", {}),
        f"{run_id} recomputed error counts",
    )
    _assert_equal(
        json.loads(computed["termination_counts_json"]),
        run_summary.get("termination_counts", {}),
        f"{run_id} recomputed termination counts",
    )

    stored_values = run_summary.get("strict_valid_value_summary", {})
    value_map = {
        "n": "strict_count",
        "n_unique": "strict_n_unique",
        "mean": "strict_mean",
        "sample_std": "strict_sample_std",
        "min": "strict_min",
        "q25": "strict_q25",
        "median": "strict_median",
        "q75": "strict_q75",
        "max": "strict_max",
    }
    for stored_key, computed_key in value_map.items():
        if stored_key not in stored_values:
            continue
        if stored_key in {"n", "n_unique"}:
            _assert_equal(computed[computed_key], stored_values[stored_key],
                          f"{run_id} recomputed {stored_key}")
        else:
            _assert_float_close(computed[computed_key], stored_values[stored_key],
                                f"{run_id} recomputed {stored_key}")


def _new_temp_path(output_dir: Path, final_name: str) -> Path:
    descriptor, name = tempfile.mkstemp(
        prefix=f".{final_name}.",
        suffix=".tmp",
        dir=output_dir,
    )
    os.close(descriptor)
    return Path(name)


def _flush_and_sync(handle: Any) -> None:
    handle.flush()
    os.fsync(handle.fileno())


def merge_outputs(
    *,
    manifest_path: Path,
    source_root: Path,
    output_dir: Path,
    expected_conditions: int,
) -> dict[str, Any]:
    manifest_rows = load_manifest(manifest_path, expected_conditions)
    output_dir.mkdir(parents=True, exist_ok=True)

    temp_paths = {
        key: _new_temp_path(output_dir, name)
        for key, name in OUTPUT_NAMES.items()
    }
    source_tree_digest = hashlib.sha256()
    global_accumulator = SampleAccumulator(keep_values=False)
    layer_counts: Counter[str] = Counter()
    model_counts: Counter[str] = Counter()
    prompt_counts: Counter[str] = Counter()
    seed_summary_rows = 0
    condition_summary_rows = 0

    try:
        with (
            temp_paths["samples"].open("w", encoding="utf-8", newline="") as sample_handle,
            temp_paths["seeds"].open("w", encoding="utf-8", newline="") as seed_handle,
            temp_paths["conditions"].open("w", encoding="utf-8", newline="") as condition_handle,
            temp_paths["metadata"].open("w", encoding="utf-8") as metadata_handle,
        ):
            sample_writer = csv.DictWriter(
                sample_handle, fieldnames=SAMPLE_COLUMNS, extrasaction="raise"
            )
            seed_writer = csv.DictWriter(
                seed_handle, fieldnames=SEED_COLUMNS, extrasaction="raise"
            )
            condition_writer = csv.DictWriter(
                condition_handle, fieldnames=CONDITION_COLUMNS, extrasaction="raise"
            )
            sample_writer.writeheader()
            seed_writer.writeheader()
            condition_writer.writeheader()

            for manifest_row in manifest_rows:
                run_id = str(manifest_row["run_id"])
                condition_dir = source_root / run_id
                metadata, run_summary = validate_condition_files(
                    manifest_row, condition_dir
                )
                config_hash = metadata["condition"]["config_hash"]
                prompt_hash = metadata["prompt_hash"]
                condition_accumulator = SampleAccumulator()
                summary_ids = _summary_identifiers(manifest_row)

                source_paths = [
                    condition_dir / "condition_metadata.json",
                    condition_dir / "run_summary.json",
                ]

                sampling_seeds = manifest_row["sampling_seeds"]
                for seed_position, sampling_seed in enumerate(sampling_seeds):
                    block_path = (
                        condition_dir
                        / "blocks"
                        / f"seed_{int(sampling_seed):010d}.jsonl"
                    )
                    source_paths.append(block_path)
                    seed_accumulator = SampleAccumulator()
                    records_in_block = 0
                    try:
                        with block_path.open("r", encoding="utf-8") as handle:
                            for line_number, line in enumerate(handle, start=1):
                                if not line.strip():
                                    raise ValueError(
                                        f"Blank source line at {block_path}:{line_number}."
                                    )
                                try:
                                    record = json.loads(line)
                                except json.JSONDecodeError as exc:
                                    raise ValueError(
                                        f"Invalid JSON at {block_path}:{line_number}."
                                    ) from exc
                                if not isinstance(record, dict):
                                    raise ValueError(
                                        f"Non-object record at {block_path}:{line_number}."
                                    )
                                validate_sample_record(
                                    record,
                                    manifest_row=manifest_row,
                                    config_hash=config_hash,
                                    prompt_hash=prompt_hash,
                                    sampling_seed=int(sampling_seed),
                                    seed_position=seed_position,
                                    sample_index=records_in_block,
                                    source=f"{block_path}:{line_number}",
                                )
                                sample_writer.writerow(
                                    _sample_csv_row(manifest_row, record)
                                )
                                seed_accumulator.add(record)
                                condition_accumulator.add(record)
                                global_accumulator.add(record)
                                records_in_block += 1
                    except FileNotFoundError as exc:
                        raise ValueError(f"Missing seed block: {block_path}") from exc

                    _assert_equal(
                        records_in_block,
                        manifest_row["samples_per_seed"],
                        f"{run_id} seed {sampling_seed} block length",
                    )
                    seed_row = dict(summary_ids)
                    seed_row["sampling_seed"] = sampling_seed
                    seed_row.update(seed_accumulator.to_dict())
                    seed_writer.writerow(seed_row)
                    seed_summary_rows += 1

                _assert_equal(
                    condition_accumulator.attempted_count,
                    manifest_row["expected_attempts"],
                    f"{run_id} merged attempts",
                )
                _compare_accumulator_to_run_summary(
                    condition_accumulator, run_summary, run_id
                )

                context = metadata["rendered_context"]
                condition_row = dict(summary_ids)
                condition_row.update(
                    {
                        "sampling_seeds_json": _canonical_json(sampling_seeds),
                        "samples_per_seed": manifest_row["samples_per_seed"],
                        "expected_seed_blocks": manifest_row["expected_seed_blocks"],
                        "expected_attempts": manifest_row["expected_attempts"],
                        "batch_size": manifest_row["batch_size"],
                        "max_new_tokens": manifest_row["max_new_tokens"],
                        "config_hash": config_hash,
                        "prompt_hash": prompt_hash,
                        "prompt_text": metadata["prompt"],
                        "context_hash": _sha256_text(context),
                        "model_backend": metadata["model_backend"],
                        "generation_boundary_token_ids_json": _canonical_json(
                            metadata.get("generation_boundary_token_ids")
                        ),
                        "generation_boundary_tokens_json": _canonical_json(
                            metadata.get("generation_boundary_tokens")
                        ),
                        "software_versions_json": _canonical_json(
                            metadata.get("software_versions", {})
                        ),
                    }
                )
                condition_row.update(condition_accumulator.to_dict())
                condition_writer.writerow(condition_row)
                condition_summary_rows += 1

                metadata_handle.write(
                    _canonical_json(
                        {
                            "merge_schema_version": MERGE_SCHEMA_VERSION,
                            "manifest": manifest_row,
                            "condition_metadata": metadata,
                            "run_summary": run_summary,
                        }
                    )
                    + "\n"
                )

                for source_path in source_paths:
                    relative_path = source_path.relative_to(source_root)
                    source_tree_digest.update(str(relative_path).encode("utf-8"))
                    source_tree_digest.update(b"\0")
                    source_tree_digest.update(_sha256_file(source_path).encode("ascii"))
                    source_tree_digest.update(b"\n")

                layer_counts[str(manifest_row["experiment_layer"])] += 1
                model_counts[str(manifest_row["model_label"])] += 1
                prompt_counts[str(manifest_row["prompt_type"])] += 1
                print(
                    f"[merge] {condition_summary_rows}/{len(manifest_rows)} "
                    f"{run_id}: {condition_accumulator.attempted_count} rows"
                )

            for handle in (
                sample_handle, seed_handle, condition_handle, metadata_handle
            ):
                _flush_and_sync(handle)

        _assert_equal(condition_summary_rows, len(manifest_rows),
                      "condition summary row count")
        _assert_equal(
            seed_summary_rows,
            sum(len(row["sampling_seeds"]) for row in manifest_rows),
            "seed summary row count",
        )
        _assert_equal(
            global_accumulator.attempted_count,
            sum(row["expected_attempts"] for row in manifest_rows),
            "global merged sample count",
        )

        published_file_details: dict[str, Any] = {}
        for key in ("samples", "seeds", "conditions", "metadata"):
            path = temp_paths[key]
            published_file_details[key] = {
                "name": OUTPUT_NAMES[key],
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }

        audit = {
            "merge_schema_version": MERGE_SCHEMA_VERSION,
            "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
            "generator_schema_version": GENERATOR_SCHEMA_VERSION,
            "status": "complete",
            "completed_at_utc": _utc_now(),
            "manifest_path": str(manifest_path),
            "manifest_sha256": _sha256_file(manifest_path),
            "source_root": str(source_root),
            "source_tree_sha256": source_tree_digest.hexdigest(),
            "counts": {
                "conditions": condition_summary_rows,
                "seed_blocks": seed_summary_rows,
                "samples": global_accumulator.attempted_count,
                "conditions_by_layer": dict(sorted(layer_counts.items())),
                "conditions_by_model": dict(sorted(model_counts.items())),
                "conditions_by_prompt": dict(sorted(prompt_counts.items())),
            },
            "global_compliance": global_accumulator.to_dict(),
            "outputs": published_file_details,
            "notes": {
                "wasserstein_calculated": False,
                "target_metrics_calculated": False,
                "strict_valid_definition": (
                    "numeric AND canonical-three-decimal AND in-support AND not-truncated"
                ),
                "raw_jsonl_shards_modified": False,
            },
        }
        with temp_paths["audit"].open("w", encoding="utf-8") as handle:
            handle.write(_pretty_json(audit))
            _flush_and_sync(handle)

        # Publish only after the entire source tree and every output pass.
        for key, final_name in OUTPUT_NAMES.items():
            os.replace(temp_paths[key], output_dir / final_name)

        return audit
    except BaseException:
        for path in temp_paths.values():
            path.unlink(missing_ok=True)
        raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit and merge sample-generation-v2 JSONL shards."
    )
    parser.add_argument(
        "--manifest",
        default=(
            "manifests/sample_generation_v2/"
            "sample_generation_all_v2.jsonl"
        ),
        help="Combined manifest used to generate the conditions.",
    )
    parser.add_argument(
        "--source-root",
        default="outputs/sample_generation_v2",
        help="Directory containing one subdirectory per run_id.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/sample_generation_merged_v2",
        help="Directory for the merged CSV/JSONL files and audit report.",
    )
    parser.add_argument(
        "--expected-conditions",
        type=int,
        default=DEFAULT_EXPECTED_CONDITIONS,
        help="Expected manifest condition count; defaults to the frozen full design.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        if args.expected_conditions < 1:
            raise ValueError("--expected-conditions must be positive.")
        audit = merge_outputs(
            manifest_path=Path(args.manifest),
            source_root=Path(args.source_root),
            output_dir=Path(args.output_dir),
            expected_conditions=args.expected_conditions,
        )
        print("Merge completed and audited")
        print(f"  Conditions: {audit['counts']['conditions']}")
        print(f"  Seed blocks: {audit['counts']['seed_blocks']}")
        print(f"  Samples: {audit['counts']['samples']}")
        print(f"  Output directory: {args.output_dir}")
        return 0
    except (AssertionError, KeyError, OSError, TypeError, ValueError) as exc:
        print(f"[error] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
