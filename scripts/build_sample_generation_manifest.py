#!/usr/bin/env python3
"""Build the frozen sample-generation experiment manifests.

This file defines *which* conditions will be generated.  It does not load an
LM and it does not generate samples.  Each JSONL row is one fixed scientific
condition and maps directly to one invocation of ``sample_generation.py``.

Frozen design
-------------
Layer 1
    12 model/protocol conditions x 5 canonical targets x plain prompt = 60.
Layer 2
    6 small model/protocol conditions x 4 predeclared target contrasts = 24.
Layer 3
    4 anchors x 5 canonical targets x explanatory_4 = 20, plus
    4 anchors x 5 canonical targets x structured ICL x 5 demo seeds = 100.

Total: 204 conditions.  Every condition has 1,000 attempted generations,
implemented as five independently seeded blocks of 200 attempts.

The generated files are deterministic: rebuilding the same design produces
byte-identical JSONL files and identical SHA-256 hashes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


MANIFEST_SCHEMA_VERSION = "sample-generation-manifest-v1"
GENERATOR_SCHEMA_VERSION = "sample-generation-v2"

SAMPLING_SEEDS = (100, 101, 102, 103, 104)
SAMPLES_PER_SEED = 200
EXPECTED_ATTEMPTS = len(SAMPLING_SEEDS) * SAMPLES_PER_SEED
BATCH_SIZE = 8
MAX_NEW_TOKENS = 16
ICL_N_EXAMPLES = 5
ICL_DEMO_SEEDS = (0, 1, 2, 3, 4)

EXPECTED_LAYER_COUNTS = {
    "layer1": 60,
    "layer2": 24,
    "layer3": 120,
}
EXPECTED_TOTAL_CONDITIONS = sum(EXPECTED_LAYER_COUNTS.values())

DEFAULT_RESULTS_DIR = "outputs/sample_generation_v2"
SAFE_RUN_ID = re.compile(r"^[a-z0-9][a-z0-9_]*$")


@dataclass(frozen=True)
class ModelCondition:
    """One model checkpoint evaluated under one prompt protocol."""

    model_label: str
    model_name: str
    prompt_protocol: str
    size_tier: str
    display_name: str

    @property
    def protocol_short(self) -> str:
        return {"raw_direct": "raw", "chat_direct": "chat"}[
            self.prompt_protocol
        ]


@dataclass(frozen=True)
class TargetCondition:
    """One target distribution and parameter setting."""

    parameter_id: str
    distribution: str
    params: Mapping[str, float]
    support_mode: str
    lower: float | None
    upper: float | None
    allow_negative: bool
    role: str


# Exact small-model IDs are preserved from the frozen parameter-grid data.
# The main-grid analysis artifact retained M1--M6 labels but not repository IDs;
# its checkpoint names are reconstructed consistently here and may be replaced
# explicitly with --model-overrides-json if a local mirror/path was used.
MODEL_CONDITIONS: tuple[ModelCondition, ...] = (
    ModelCondition(
        "S1",
        "Qwen/Qwen3-4B-Base",
        "raw_direct",
        "small",
        "Qwen3-4B Base - raw",
    ),
    ModelCondition(
        "S2",
        "Qwen/Qwen3-4B-Instruct-2507",
        "raw_direct",
        "small",
        "Qwen3-4B Instruct - raw",
    ),
    ModelCondition(
        "S3",
        "Qwen/Qwen3-4B-Instruct-2507",
        "chat_direct",
        "small",
        "Qwen3-4B Instruct - chat",
    ),
    ModelCondition(
        "S4",
        "google/gemma-4-E4B",
        "raw_direct",
        "small",
        "Gemma-4-E4B Base - raw",
    ),
    ModelCondition(
        "S5",
        "google/gemma-4-E4B-it",
        "raw_direct",
        "small",
        "Gemma-4-E4B IT - raw",
    ),
    ModelCondition(
        "S6",
        "google/gemma-4-E4B-it",
        "chat_direct",
        "small",
        "Gemma-4-E4B IT - chat",
    ),
    ModelCondition(
        "M1",
        "Qwen/Qwen3-14B-Base",
        "raw_direct",
        "medium",
        "Qwen3-14B Base - raw",
    ),
    ModelCondition(
        "M2",
        "Qwen/Qwen3-14B",
        "raw_direct",
        "medium",
        "Qwen3-14B - raw",
    ),
    ModelCondition(
        "M3",
        "Qwen/Qwen3-14B",
        "chat_direct",
        "medium",
        "Qwen3-14B - chat",
    ),
    ModelCondition(
        "M4",
        "google/gemma-4-12B",
        "raw_direct",
        "medium",
        "Gemma-4-12B Base - raw",
    ),
    ModelCondition(
        "M5",
        "google/gemma-4-12B-it",
        "raw_direct",
        "medium",
        "Gemma-4-12B IT - raw",
    ),
    ModelCondition(
        "M6",
        "google/gemma-4-12B-it",
        "chat_direct",
        "medium",
        "Gemma-4-12B IT - chat",
    ),
)


CANONICAL_TARGETS: tuple[TargetCondition, ...] = (
    TargetCondition(
        "N1",
        "normal",
        {"mean": 0.0, "std": 1.0},
        "agnostic",
        None,
        None,
        True,
        "canonical",
    ),
    TargetCondition(
        "U1",
        "uniform",
        {"low": 0.0, "high": 1.0},
        "bounded",
        0.0,
        1.0,
        False,
        "canonical",
    ),
    TargetCondition(
        "E1",
        "exponential",
        {"rate": 1.0},
        "positive",
        0.0,
        None,
        False,
        "canonical",
    ),
    TargetCondition(
        "B2",
        "beta",
        {"alpha": 2.0, "beta": 2.0},
        "bounded",
        0.0,
        1.0,
        False,
        "canonical",
    ),
    TargetCondition(
        "L1",
        "laplace",
        {"loc": 0.0, "scale": 1.0},
        "agnostic",
        None,
        None,
        True,
        "canonical",
    ),
)


LAYER2_TARGETS: tuple[TargetCondition, ...] = (
    TargetCondition(
        "N6",
        "normal",
        {"mean": 0.0, "std": 10.0},
        "agnostic",
        None,
        None,
        True,
        "extreme_scale_contrast",
    ),
    TargetCondition(
        "L4",
        "laplace",
        {"loc": 0.0, "scale": 10.0},
        "agnostic",
        None,
        None,
        True,
        "extreme_scale_contrast",
    ),
    TargetCondition(
        "E4",
        "exponential",
        {"rate": 0.1},
        "positive",
        0.0,
        None,
        False,
        "extreme_scale_contrast",
    ),
    TargetCondition(
        "B1",
        "beta",
        {"alpha": 1.0, "beta": 1.0},
        "bounded",
        0.0,
        1.0,
        False,
        "equivalent_target_control",
    ),
)

SMALL_MODEL_LABELS = ("S1", "S2", "S3", "S4", "S5", "S6")
LAYER3_ANCHOR_LABELS = ("S2", "S3", "S5", "S6")


def _json_dumps(value: Any, *, pretty: bool = False) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        indent=2 if pretty else None,
        separators=None if pretty else (",", ":"),
    )


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def _jsonl_bytes(rows: Iterable[Mapping[str, Any]]) -> bytes:
    return ("".join(_json_dumps(row) + "\n" for row in rows)).encode("utf-8")


def _parse_model_overrides(raw: str | None) -> dict[str, str]:
    if raw is None:
        return {}
    candidate = Path(raw)
    if candidate.is_file():
        value = json.loads(candidate.read_text(encoding="utf-8"))
    else:
        value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("Model overrides must be a JSON object mapping labels to IDs.")
    parsed: dict[str, str] = {}
    for label, model_name in value.items():
        if not isinstance(label, str) or not isinstance(model_name, str):
            raise ValueError("Every model override key and value must be a string.")
        parsed[label.strip().upper()] = model_name.strip()
    return parsed


def resolve_models(overrides: Mapping[str, str]) -> tuple[ModelCondition, ...]:
    known = {model.model_label for model in MODEL_CONDITIONS}
    unknown = sorted(set(overrides) - known)
    if unknown:
        raise ValueError(f"Unknown model override labels: {unknown}")
    return tuple(
        replace(model, model_name=overrides.get(model.model_label, model.model_name))
        for model in MODEL_CONDITIONS
    )


def _run_id(
    layer: str,
    model: ModelCondition,
    target: TargetCondition,
    prompt_type: str,
    icl_seed: int,
) -> str:
    prompt_tag = {
        "plain": "plain",
        "explanatory_4": "explanatory4",
        "icl": f"icl_seed{icl_seed}",
    }[prompt_type]
    value = "_".join(
        (
            layer.replace("layer", "l"),
            model.model_label.lower(),
            target.parameter_id.lower(),
            prompt_tag,
            model.protocol_short,
        )
    )
    if not SAFE_RUN_ID.fullmatch(value):
        raise AssertionError(f"Unsafe generated run_id: {value!r}")
    return value


def _condition_row(
    *,
    layer: str,
    model: ModelCondition,
    target: TargetCondition,
    prompt_type: str,
    icl_seed: int,
    results_dir: str,
) -> dict[str, Any]:
    uses_icl = prompt_type == "icl"
    run_id = _run_id(layer, model, target, prompt_type, icl_seed)
    return {
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "generator_schema_version": GENERATOR_SCHEMA_VERSION,
        "experiment_layer": layer,
        "run_id": run_id,
        "condition_id": run_id,
        "model_label": model.model_label,
        "model_name": model.model_name,
        "model_display_name": model.display_name,
        "model_size_tier": model.size_tier,
        "prompt_protocol": model.prompt_protocol,
        "protocol_short": model.protocol_short,
        "distribution": target.distribution,
        "parameter_id": target.parameter_id,
        "params": dict(target.params),
        "target_role": target.role,
        "support_mode": target.support_mode,
        "lower": target.lower,
        "upper": target.upper,
        "allow_negative": target.allow_negative,
        "prompt_type": prompt_type,
        "uses_icl": uses_icl,
        "icl_n_examples": ICL_N_EXAMPLES,
        "icl_seed": icl_seed,
        "sampling_seeds": list(SAMPLING_SEEDS),
        "samples_per_seed": SAMPLES_PER_SEED,
        "expected_seed_blocks": len(SAMPLING_SEEDS),
        "expected_attempts": EXPECTED_ATTEMPTS,
        "batch_size": BATCH_SIZE,
        "max_new_tokens": MAX_NEW_TOKENS,
        "output_dir": results_dir,
        "decoding": {
            "precision": "bf16",
            "load_in_4bit": False,
            "do_sample": True,
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": 0,
            "num_beams": 1,
            "repetition_penalty": 1.0,
        },
    }


def build_manifest_rows(
    models: Sequence[ModelCondition],
    results_dir: str,
) -> list[dict[str, Any]]:
    model_by_label = {model.model_label: model for model in models}
    rows: list[dict[str, Any]] = []

    # Layer 1: the complete canonical main-grid bridge.
    for model in models:
        for target in CANONICAL_TARGETS:
            rows.append(
                _condition_row(
                    layer="layer1",
                    model=model,
                    target=target,
                    prompt_type="plain",
                    icl_seed=0,
                    results_dir=results_dir,
                )
            )

    # Layer 2: only the four predeclared parameter mechanisms.  Their canonical
    # references already exist in Layer 1, so they are not repeated here.
    for label in SMALL_MODEL_LABELS:
        model = model_by_label[label]
        for target in LAYER2_TARGETS:
            rows.append(
                _condition_row(
                    layer="layer2",
                    model=model,
                    target=target,
                    prompt_type="plain",
                    icl_seed=0,
                    results_dir=results_dir,
                )
            )

    # Layer 3: explanatory prompt once per anchor/target, followed by five
    # genuinely different structured-ICL demonstration contexts.
    for label in LAYER3_ANCHOR_LABELS:
        model = model_by_label[label]
        for target in CANONICAL_TARGETS:
            rows.append(
                _condition_row(
                    layer="layer3",
                    model=model,
                    target=target,
                    prompt_type="explanatory_4",
                    icl_seed=0,
                    results_dir=results_dir,
                )
            )
            for icl_seed in ICL_DEMO_SEEDS:
                rows.append(
                    _condition_row(
                        layer="layer3",
                        model=model,
                        target=target,
                        prompt_type="icl",
                        icl_seed=icl_seed,
                        results_dir=results_dir,
                    )
                )

    layer_counters = {layer: 0 for layer in EXPECTED_LAYER_COUNTS}
    for global_index, row in enumerate(rows):
        layer = row["experiment_layer"]
        row["manifest_global_index"] = global_index
        row["manifest_layer_index"] = layer_counters[layer]
        layer_counters[layer] += 1
    return rows


def _identity(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        row["model_name"],
        row["prompt_protocol"],
        row["distribution"],
        _json_dumps(row["params"]),
        row["prompt_type"],
        row["icl_seed"] if row["uses_icl"] else None,
        _json_dumps(row["decoding"]),
    )


def validate_models(models: Sequence[ModelCondition]) -> None:
    if len(models) != 12:
        raise AssertionError(f"Expected 12 model/protocol rows, found {len(models)}.")
    labels = [model.model_label for model in models]
    expected_labels = [f"S{i}" for i in range(1, 7)] + [f"M{i}" for i in range(1, 7)]
    if labels != expected_labels:
        raise AssertionError(f"Unexpected model order: {labels}")
    for model in models:
        if model.prompt_protocol not in {"raw_direct", "chat_direct"}:
            raise AssertionError(f"Invalid protocol for {model.model_label}.")
        if not model.model_name or "/" not in model.model_name:
            raise AssertionError(f"Invalid model_name for {model.model_label}.")
        if model.size_tier not in {"small", "medium"}:
            raise AssertionError(f"Invalid size tier for {model.model_label}.")
    for raw_label, chat_label in (("S2", "S3"), ("S5", "S6"), ("M2", "M3"), ("M5", "M6")):
        by_label = {model.model_label: model for model in models}
        if by_label[raw_label].model_name != by_label[chat_label].model_name:
            raise AssertionError(f"{raw_label}/{chat_label} must share one checkpoint.")


def validate_manifest(rows: Sequence[Mapping[str, Any]]) -> None:
    if len(rows) != EXPECTED_TOTAL_CONDITIONS:
        raise AssertionError(
            f"Expected {EXPECTED_TOTAL_CONDITIONS} conditions, found {len(rows)}."
        )

    counts = {layer: 0 for layer in EXPECTED_LAYER_COUNTS}
    run_ids: set[str] = set()
    identities: set[tuple[Any, ...]] = set()
    for expected_global_index, row in enumerate(rows):
        layer = str(row["experiment_layer"])
        if layer not in counts:
            raise AssertionError(f"Unknown experiment layer: {layer}")
        if row["manifest_global_index"] != expected_global_index:
            raise AssertionError("Global manifest indices are not contiguous.")
        if row["manifest_layer_index"] != counts[layer]:
            raise AssertionError(f"Layer indices are not contiguous for {layer}.")
        counts[layer] += 1

        run_id = str(row["run_id"])
        if run_id in run_ids:
            raise AssertionError(f"Duplicate run_id: {run_id}")
        run_ids.add(run_id)
        identity = _identity(row)
        if identity in identities:
            raise AssertionError(f"Duplicate scientific condition: {run_id}")
        identities.add(identity)

        if row["condition_id"] != run_id or not SAFE_RUN_ID.fullmatch(run_id):
            raise AssertionError(f"Invalid condition/run identifier: {run_id}")
        if row["sampling_seeds"] != list(SAMPLING_SEEDS):
            raise AssertionError(f"Sampling seed drift in {run_id}.")
        if row["samples_per_seed"] != SAMPLES_PER_SEED:
            raise AssertionError(f"Sample-count drift in {run_id}.")
        if row["expected_attempts"] != EXPECTED_ATTEMPTS:
            raise AssertionError(f"Attempt-count drift in {run_id}.")
        if row["decoding"]["temperature"] != 1.0:
            raise AssertionError(f"Temperature drift in {run_id}.")
        if not row["decoding"]["do_sample"]:
            raise AssertionError(f"Sampling disabled in {run_id}.")
        if row["prompt_type"] == "icl":
            if not row["uses_icl"] or row["icl_seed"] not in ICL_DEMO_SEEDS:
                raise AssertionError(f"Invalid ICL configuration in {run_id}.")
        elif row["uses_icl"] or row["icl_seed"] != 0:
            raise AssertionError(f"Inactive ICL fields are inconsistent in {run_id}.")
        if row["prompt_type"] == "icl_random":
            raise AssertionError("icl_random is excluded from the frozen bridge.")

    if counts != EXPECTED_LAYER_COUNTS:
        raise AssertionError(
            f"Layer count mismatch: expected {EXPECTED_LAYER_COUNTS}, found {counts}."
        )

    layer1 = [row for row in rows if row["experiment_layer"] == "layer1"]
    expected_l1 = {
        (model.model_label, target.parameter_id)
        for model in MODEL_CONDITIONS
        for target in CANONICAL_TARGETS
    }
    actual_l1 = {(row["model_label"], row["parameter_id"]) for row in layer1}
    if actual_l1 != expected_l1 or any(row["prompt_type"] != "plain" for row in layer1):
        raise AssertionError("Layer 1 is not the complete canonical plain grid.")

    layer2 = [row for row in rows if row["experiment_layer"] == "layer2"]
    expected_l2 = {
        (label, target.parameter_id)
        for label in SMALL_MODEL_LABELS
        for target in LAYER2_TARGETS
    }
    actual_l2 = {(row["model_label"], row["parameter_id"]) for row in layer2}
    if actual_l2 != expected_l2 or any(row["prompt_type"] != "plain" for row in layer2):
        raise AssertionError("Layer 2 is not the frozen 6 x 4 contrast grid.")

    layer3 = [row for row in rows if row["experiment_layer"] == "layer3"]
    explanatory = [row for row in layer3 if row["prompt_type"] == "explanatory_4"]
    icl = [row for row in layer3 if row["prompt_type"] == "icl"]
    expected_anchor_targets = {
        (label, target.parameter_id)
        for label in LAYER3_ANCHOR_LABELS
        for target in CANONICAL_TARGETS
    }
    if {(row["model_label"], row["parameter_id"]) for row in explanatory} != expected_anchor_targets:
        raise AssertionError("Layer 3 explanatory grid is incomplete.")
    expected_icl = {
        (label, target.parameter_id, seed)
        for label in LAYER3_ANCHOR_LABELS
        for target in CANONICAL_TARGETS
        for seed in ICL_DEMO_SEEDS
    }
    actual_icl = {
        (row["model_label"], row["parameter_id"], row["icl_seed"])
        for row in icl
    }
    if actual_icl != expected_icl:
        raise AssertionError("Layer 3 structured-ICL grid is incomplete.")


def _file_names() -> dict[str, str]:
    return {
        "layer1": "sample_generation_layer1_v2.jsonl",
        "layer2": "sample_generation_layer2_v2.jsonl",
        "layer3": "sample_generation_layer3_v2.jsonl",
        "all": "sample_generation_all_v2.jsonl",
        "summary": "sample_generation_manifest_summary_v2.json",
    }


def build_summary(
    rows: Sequence[Mapping[str, Any]],
    models: Sequence[ModelCondition],
    results_dir: str,
    file_payloads: Mapping[str, bytes],
) -> dict[str, Any]:
    layer_counts = {
        layer: sum(row["experiment_layer"] == layer for row in rows)
        for layer in EXPECTED_LAYER_COUNTS
    }
    names = _file_names()
    return {
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "generator_schema_version": GENERATOR_SCHEMA_VERSION,
        "status": "validated",
        "design": {
            "layer_counts": layer_counts,
            "total_conditions": len(rows),
            "attempts_per_condition": EXPECTED_ATTEMPTS,
            "total_planned_attempts": len(rows) * EXPECTED_ATTEMPTS,
            "sampling_seeds": list(SAMPLING_SEEDS),
            "samples_per_seed": SAMPLES_PER_SEED,
            "icl_demo_seeds": list(ICL_DEMO_SEEDS),
            "layer3_anchors": list(LAYER3_ANCHOR_LABELS),
            "included_prompt_types": ["plain", "explanatory_4", "icl"],
            "excluded_prompt_types": ["icl_random"],
            "results_dir": results_dir,
        },
        "models": [
            {
                "model_label": model.model_label,
                "model_name": model.model_name,
                "prompt_protocol": model.prompt_protocol,
                "size_tier": model.size_tier,
                "display_name": model.display_name,
            }
            for model in models
        ],
        "canonical_parameter_ids": [target.parameter_id for target in CANONICAL_TARGETS],
        "layer2_parameter_ids": [target.parameter_id for target in LAYER2_TARGETS],
        "files": {
            key: {
                "name": names[key],
                "rows": len(rows)
                if key == "all"
                else sum(row["experiment_layer"] == key for row in rows),
                "sha256": _sha256_bytes(payload),
            }
            for key, payload in file_payloads.items()
        },
    }


def write_manifests(
    output_dir: Path,
    rows: Sequence[Mapping[str, Any]],
    models: Sequence[ModelCondition],
    results_dir: str,
) -> dict[str, Any]:
    names = _file_names()
    by_layer = {
        layer: [row for row in rows if row["experiment_layer"] == layer]
        for layer in EXPECTED_LAYER_COUNTS
    }
    payloads = {
        "layer1": _jsonl_bytes(by_layer["layer1"]),
        "layer2": _jsonl_bytes(by_layer["layer2"]),
        "layer3": _jsonl_bytes(by_layer["layer3"]),
        "all": _jsonl_bytes(rows),
    }
    summary = build_summary(rows, models, results_dir, payloads)
    for key, payload in payloads.items():
        _atomic_write_bytes(output_dir / names[key], payload)
    _atomic_write_bytes(
        output_dir / names["summary"],
        (_json_dumps(summary, pretty=True) + "\n").encode("utf-8"),
    )

    # Read-after-write checks catch truncation or unexpected filesystem issues.
    for key, expected_payload in payloads.items():
        path = output_dir / names[key]
        if _sha256_file(path) != _sha256_bytes(expected_payload):
            raise OSError(f"Post-write hash mismatch: {path}")
        with path.open("r", encoding="utf-8") as handle:
            parsed_rows = [json.loads(line) for line in handle if line.strip()]
        expected_rows = len(rows) if key == "all" else EXPECTED_LAYER_COUNTS[key]
        if len(parsed_rows) != expected_rows:
            raise OSError(f"Post-write row-count mismatch: {path}")
    return summary


def _print_design(summary: Mapping[str, Any], output_dir: Path | None) -> None:
    design = summary["design"]
    print("Validated sample-generation design")
    print(f"  Layer 1: {design['layer_counts']['layer1']} conditions")
    print(f"  Layer 2: {design['layer_counts']['layer2']} conditions")
    print(f"  Layer 3: {design['layer_counts']['layer3']} conditions")
    print(f"  Total:   {design['total_conditions']} conditions")
    print(f"  Attempts per condition: {design['attempts_per_condition']}")
    print(f"  Planned attempts:       {design['total_planned_attempts']}")
    if output_dir is not None:
        print(f"  Wrote manifests to:     {output_dir}")
        for details in summary["files"].values():
            print(
                f"    {details['name']}: {details['rows']} rows, "
                f"sha256={details['sha256'][:12]}..."
            )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build and validate the frozen 204-condition generation manifests."
    )
    parser.add_argument(
        "--output-dir",
        default="manifests",
        help="Directory for the four JSONL manifests and summary JSON.",
    )
    parser.add_argument(
        "--results-dir",
        default=DEFAULT_RESULTS_DIR,
        help="Output root embedded in every condition for sample_generation.py.",
    )
    parser.add_argument(
        "--model-overrides-json",
        default=None,
        help=(
            "Optional JSON object or JSON file mapping model labels to exact model IDs/paths, "
            "for example '{\"M4\":\"/path/to/checkpoint\"}'."
        ),
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate and print the design without writing manifest files.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        overrides = _parse_model_overrides(args.model_overrides_json)
        models = resolve_models(overrides)
        validate_models(models)
        results_dir = str(Path(args.results_dir))
        rows = build_manifest_rows(models, results_dir)
        validate_manifest(rows)

        if args.check_only:
            payloads = {
                layer: _jsonl_bytes(
                    row for row in rows if row["experiment_layer"] == layer
                )
                for layer in EXPECTED_LAYER_COUNTS
            }
            payloads["all"] = _jsonl_bytes(rows)
            summary = build_summary(rows, models, results_dir, payloads)
            _print_design(summary, output_dir=None)
        else:
            output_dir = Path(args.output_dir)
            summary = write_manifests(output_dir, rows, models, results_dir)
            _print_design(summary, output_dir=output_dir)
        return 0
    except (AssertionError, KeyError, OSError, TypeError, ValueError) as exc:
        print(f"[error] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
