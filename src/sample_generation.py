#!/usr/bin/env python3
"""Generate independent numerical samples from one frozen LM condition.

This is the data-collection file for the thesis sample-generation bridge.  A
single invocation represents one condition:

    model x protocol x distribution/parameters x prompt x ICL-demo seed

For that fixed condition, the script generates independent completions in
sampling-seed blocks (five blocks of 200 by default), parses every response,
and stores one JSON object per generated sample.  It deliberately does *not*
request token log-probabilities: the empirical frequencies of the generated
numbers are the object of this downstream experiment.  Raw text and generated
token IDs are retained so parsing decisions remain auditable.

The script reuses the project's canonical modules:

* ``distributions.py`` for distribution validation and support defaults;
* ``prompt_template.py`` for exactly the prompts used in prior experiments;
* ``model_loader.py`` for model/tokenizer loading;
* ``lm_next_token.py`` for the raw/chat protocol rendering contract.

Outputs are blockwise and resumable::

    OUTPUT_DIR/CONDITION_ID/
      condition_metadata.json
      blocks/seed_0000000100.jsonl
      blocks/seed_0000000101.jsonl
      ...
      run_summary.json

Each completed seed block is written atomically.  On restart, valid completed
blocks are skipped; incomplete or inconsistent blocks fail loudly unless
``--repair-incomplete`` is supplied.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import sys
import tempfile
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = "sample-generation-v1"
SUPPORTED_DISTRIBUTIONS = ("normal", "uniform", "exponential", "beta", "laplace")
SUPPORTED_PROMPT_TYPES = (
    "short",
    "plain",
    "formal",
    "explanatory_1",
    "explanatory_2",
    "explanatory_3",
    "explanatory_4",
    "cot",
    "icl",
    "icl_random",
    "icl_cot",
)
ICL_PROMPT_TYPES = frozenset({"icl", "icl_random", "icl_cot"})
PROMPT_PROTOCOLS = frozenset({"raw_direct", "chat_direct"})
SUPPORT_MODES = frozenset({"agnostic", "positive", "bounded"})
DEFAULT_SAMPLING_SEEDS = (100, 101, 102, 103, 104)

_FINITE_NUMBER_FULL = re.compile(
    r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$"
)
_FINITE_NUMBER_SEARCH = re.compile(
    r"(?<![\w.])[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?(?![\w.])"
)
_NONFINITE_FULL = re.compile(r"^[+-]?(?:nan|inf(?:inity)?)$", re.IGNORECASE)
_CANONICAL_THREE_DECIMAL = re.compile(r"^-?(?:0|[1-9]\d*)\.\d{3}$")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _slug(text: str, limit: int = 48) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", text.strip()).strip("-._")
    return (cleaned or "condition")[:limit]


def _finite_or_none(value: float | None, name: str) -> float | None:
    if value is None:
        return None
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite or omitted; received {value!r}.")
    return result


def _normalise_seeds(values: Sequence[int]) -> tuple[int, ...]:
    seeds = tuple(values)
    if not seeds:
        raise ValueError("sampling_seeds must contain at least one seed.")
    for seed in seeds:
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError(f"Sampling seeds must be integers; received {seed!r}.")
        if not 0 <= seed < 2**32:
            raise ValueError(f"Sampling seed must lie in [0, 2**32); received {seed}.")
    if len(seeds) != len(set(seeds)):
        raise ValueError("sampling_seeds must be unique.")
    return seeds


def _distribution_api() -> tuple[Any, Any, Any]:
    try:
        from distributions import (  # type: ignore
            DistributionSpec,
            default_support_for_distribution,
            validate_distribution,
        )
    except ImportError as exc:
        raise ImportError(
            "Could not import distributions.py. Run this file beside the "
            "canonical project modules or add their directory to PYTHONPATH."
        ) from exc
    return DistributionSpec, default_support_for_distribution, validate_distribution


def _build_prompt_api() -> Any:
    try:
        from prompt_template import build_prompt  # type: ignore
    except ImportError as exc:
        raise ImportError("Could not import the canonical prompt_template.py.") from exc
    return build_prompt


def _context_api() -> Any:
    try:
        from lm_next_token import build_lm_context  # type: ignore
    except ImportError as exc:
        raise ImportError("Could not import build_lm_context from lm_next_token.py.") from exc
    return build_lm_context


def _model_api() -> tuple[Any, Any]:
    try:
        from model_loader import get_model_input_device, load_lm_backend  # type: ignore
    except ImportError as exc:
        raise ImportError("Could not import the canonical model_loader.py.") from exc
    return get_model_input_device, load_lm_backend


@dataclass(frozen=True)
class GenerationConfig:
    """Frozen scientific and operational settings for one condition."""

    model_name: str
    distribution: str
    params: Mapping[str, Any]

    model_label: str | None = None
    parameter_id: str | None = None
    prompt_type: str = "plain"
    prompt_protocol: str = "raw_direct"
    experiment_layer: str = "unspecified"
    run_id: str | None = None
    decimals: int = 3

    support_mode: str | None = None
    lower: float | None = None
    upper: float | None = None

    icl_n_examples: int = 5
    icl_seed: int = 0

    sampling_seeds: tuple[int, ...] = DEFAULT_SAMPLING_SEEDS
    samples_per_seed: int = 200
    batch_size: int = 8
    max_new_tokens: int = 16
    output_dir: Path = Path("outputs/sample_generation_v1")

    # The bridge's primary decoding contract is intentionally not exposed as a
    # tunable CLI sweep.  Temperature 1 samples the LM distribution itself.
    load_in_4bit: bool = field(default=False, init=False)
    do_sample: bool = field(default=True, init=False)
    temperature: float = field(default=1.0, init=False)
    top_p: float = field(default=1.0, init=False)
    top_k: int = field(default=0, init=False)
    num_beams: int = field(default=1, init=False)
    repetition_penalty: float = field(default=1.0, init=False)

    def __post_init__(self) -> None:
        model_name = self.model_name.strip()
        if not model_name:
            raise ValueError("model_name must be non-empty.")
        object.__setattr__(self, "model_name", model_name)

        model_label = (self.model_label or model_name.rsplit("/", 1)[-1]).strip()
        if not model_label:
            raise ValueError("model_label must be non-empty when supplied.")
        object.__setattr__(self, "model_label", model_label)

        distribution = self.distribution.strip().lower()
        if distribution not in SUPPORTED_DISTRIBUTIONS:
            raise ValueError(
                f"Unsupported distribution {distribution!r}; use {SUPPORTED_DISTRIBUTIONS}."
            )
        params = dict(self.params)
        for key, value in params.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(
                    f"Distribution parameter {key!r} must be numeric; received {value!r}."
                )
            if not math.isfinite(float(value)):
                raise ValueError(
                    f"Distribution parameter {key!r} must be finite; received {value!r}."
                )
        DistributionSpec, default_support, validate_distribution = _distribution_api()
        spec = DistributionSpec(distribution, params)
        validate_distribution(spec)
        object.__setattr__(self, "distribution", distribution)
        object.__setattr__(self, "params", params)

        prompt_type = self.prompt_type.strip().lower()
        if prompt_type not in SUPPORTED_PROMPT_TYPES:
            raise ValueError(f"Unsupported prompt_type {prompt_type!r}.")
        object.__setattr__(self, "prompt_type", prompt_type)

        protocol = self.prompt_protocol.strip().lower()
        if protocol not in PROMPT_PROTOCOLS:
            raise ValueError(f"Unsupported prompt_protocol {protocol!r}.")
        object.__setattr__(self, "prompt_protocol", protocol)

        if self.decimals != 3:
            raise ValueError("This frozen bridge requires exactly three decimal places.")
        for name in ("icl_n_examples", "samples_per_seed", "batch_size", "max_new_tokens"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if isinstance(self.icl_seed, bool) or not isinstance(self.icl_seed, int):
            raise TypeError("icl_seed must be an integer.")
        if not 0 <= self.icl_seed < 2**32:
            raise ValueError("icl_seed must lie in [0, 2**32).")
        object.__setattr__(self, "sampling_seeds", _normalise_seeds(self.sampling_seeds))

        default_lower, default_upper, _, default_mode = default_support(spec)
        mode = default_mode if self.support_mode is None else self.support_mode.strip().lower()
        if mode not in SUPPORT_MODES:
            raise ValueError(f"Unsupported support_mode {mode!r}.")
        lower = _finite_or_none(self.lower if self.lower is not None else default_lower, "lower")
        upper = _finite_or_none(self.upper if self.upper is not None else default_upper, "upper")
        if lower is not None and upper is not None and upper <= lower:
            raise ValueError("upper must be greater than lower.")
        if mode == "bounded" and (lower is None or upper is None):
            raise ValueError("bounded support requires lower and upper.")
        if mode == "positive":
            lower = 0.0 if lower is None else lower
            if lower < 0:
                raise ValueError("positive support cannot have a negative lower bound.")
            if upper is not None:
                raise ValueError(
                    "positive support cannot include an upper bound; use bounded support."
                )
        if mode == "agnostic" and (lower is not None or upper is not None):
            raise ValueError("agnostic support cannot include finite bounds.")
        object.__setattr__(self, "support_mode", mode)
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)
        object.__setattr__(self, "output_dir", Path(self.output_dir))

        if self.parameter_id is not None:
            object.__setattr__(self, "parameter_id", self.parameter_id.strip() or None)
        layer = self.experiment_layer.strip().lower()
        object.__setattr__(self, "experiment_layer", layer or "unspecified")

        run_id = self.run_id.strip() if self.run_id is not None else self.default_run_id()
        if (
            not run_id
            or run_id in {".", ".."}
            or Path(run_id).name != run_id
            or "/" in run_id
            or "\\" in run_id
        ):
            raise ValueError("run_id must be one safe directory name.")
        object.__setattr__(self, "run_id", run_id)

    @property
    def uses_icl(self) -> bool:
        return self.prompt_type in ICL_PROMPT_TYPES

    @property
    def total_samples(self) -> int:
        return len(self.sampling_seeds) * self.samples_per_seed

    @property
    def condition_dir(self) -> Path:
        assert self.run_id is not None
        return self.output_dir / self.run_id

    def hash_payload(self) -> dict[str, Any]:
        """Settings that must agree for safe resume of generated records."""

        return {
            "schema_version": SCHEMA_VERSION,
            "model_name": self.model_name,
            "model_label": self.model_label,
            "distribution": self.distribution,
            "params": dict(self.params),
            "parameter_id": self.parameter_id,
            "prompt_type": self.prompt_type,
            "prompt_protocol": self.prompt_protocol,
            "experiment_layer": self.experiment_layer,
            "decimals": self.decimals,
            "support_mode": self.support_mode,
            "lower": self.lower,
            "upper": self.upper,
            "icl_n_examples": self.icl_n_examples,
            "icl_seed": self.icl_seed,
            "sampling_seeds": list(self.sampling_seeds),
            "samples_per_seed": self.samples_per_seed,
            "batch_size": self.batch_size,
            "max_new_tokens": self.max_new_tokens,
            "load_in_4bit": self.load_in_4bit,
            "do_sample": self.do_sample,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "num_beams": self.num_beams,
            "repetition_penalty": self.repetition_penalty,
        }

    @property
    def config_hash(self) -> str:
        return _sha256_text(_canonical_json(self.hash_payload()))

    def default_run_id(self) -> str:
        parameter = self.parameter_id or "custom"
        readable = "__".join(
            (
                _slug(self.experiment_layer, 20),
                _slug(str(self.model_label), 32),
                _slug(self.prompt_protocol, 20),
                _slug(self.distribution, 16),
                _slug(parameter, 24),
                _slug(self.prompt_type, 24),
                f"icl{self.icl_seed}" if self.prompt_type in ICL_PROMPT_TYPES else "noicl",
            )
        )
        digest = _sha256_text(_canonical_json(self.hash_payload()))[:10]
        return f"{readable}__{digest}"

    def to_metadata(self) -> dict[str, Any]:
        data = self.hash_payload()
        data.update(
            {
                "run_id": self.run_id,
                "output_dir": str(self.output_dir),
                "condition_dir": str(self.condition_dir),
                "total_samples": self.total_samples,
                "config_hash": self.config_hash,
                "precision": "bf16",
            }
        )
        return data


@dataclass(frozen=True)
class ParseResult:
    response_text: str
    normalized_text: str
    parsed_value: float | None
    numeric_valid: bool
    canonical_format_valid: bool
    support_valid: bool
    negative_zero: bool
    extra_after_first_line: bool
    error_type: str | None

    @property
    def strict_valid(self) -> bool:
        return self.numeric_valid and self.canonical_format_valid and self.support_valid


@dataclass(frozen=True)
class GeneratedSampleRecord:
    schema_version: str
    config_hash: str
    prompt_hash: str
    run_id: str
    model_name: str
    model_label: str
    prompt_protocol: str
    distribution: str
    distribution_params: Mapping[str, Any]
    parameter_id: str | None
    prompt_type: str
    icl_seed: int
    sampling_seed: int
    sample_index: int
    global_sample_index: int
    raw_completion: str
    response_text: str
    normalized_text: str
    generated_token_ids: tuple[int, ...]
    generated_token_count: int
    parsed_value: float | None
    numeric_valid: bool
    canonical_format_valid: bool
    support_valid: bool
    strict_valid: bool
    negative_zero: bool
    extra_after_first_line: bool
    termination_reason: str
    truncated: bool
    error_type: str | None

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["distribution_params"] = dict(self.distribution_params)
        row["generated_token_ids"] = list(self.generated_token_ids)
        return row


def parse_generated_number(
    raw_completion: str,
    *,
    lower: float | None,
    upper: float | None,
) -> ParseResult:
    """Parse the first response line while preserving all raw generated text."""

    response, separator, remainder = raw_completion.partition("\n")
    normalized = response.strip()
    extra_after_first_line = bool(separator and remainder.strip())

    if not normalized:
        return ParseResult(response, normalized, None, False, False, False, False,
                           extra_after_first_line, "empty_response")
    if _NONFINITE_FULL.fullmatch(normalized):
        return ParseResult(response, normalized, None, False, False, False, False,
                           extra_after_first_line, "nonfinite_number")

    if not _FINITE_NUMBER_FULL.fullmatch(normalized):
        matches = _FINITE_NUMBER_SEARCH.findall(normalized)
        error = "multiple_numbers" if len(matches) > 1 else (
            "extra_text" if len(matches) == 1 else "malformed_number"
        )
        return ParseResult(response, normalized, None, False, False, False, False,
                           extra_after_first_line, error)

    try:
        value = float(normalized)
    except ValueError:
        return ParseResult(response, normalized, None, False, False, False, False,
                           extra_after_first_line, "malformed_number")
    if not math.isfinite(value):
        return ParseResult(response, normalized, None, False, False, False, False,
                           extra_after_first_line, "nonfinite_number")

    canonical = bool(_CANONICAL_THREE_DECIMAL.fullmatch(normalized))
    support_valid = (lower is None or value >= lower) and (upper is None or value <= upper)
    negative_zero = value == 0.0 and normalized.startswith("-")
    error: str | None = None
    if not canonical:
        error = "noncanonical_format"
    elif not support_valid:
        error = "support_violation"
    return ParseResult(
        response,
        normalized,
        value,
        True,
        canonical,
        support_valid,
        negative_zero,
        extra_after_first_line,
        error,
    )


def build_condition_prompt(config: GenerationConfig) -> str:
    build_prompt = _build_prompt_api()
    return build_prompt(
        distribution=config.distribution,
        params=dict(config.params),
        decimals=config.decimals,
        prompt_type=config.prompt_type,
        support_mode=str(config.support_mode),
        lower=config.lower,
        upper=config.upper,
        icl_n_examples=config.icl_n_examples,
        icl_seed=config.icl_seed,
    )


def render_generation_context(tokenizer: Any, prompt: str, protocol: str) -> tuple[str, bool]:
    build_lm_context = _context_api()
    # Empty prefix: we want the model to generate the entire numerical response.
    return build_lm_context(
        tokenizer=tokenizer,
        prompt=prompt,
        prefix="",
        prompt_protocol=protocol,
    )


def set_sampling_seed(seed: int) -> None:
    """Seed every RNG used by ordinary Transformers generation."""

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _as_eos_set(eos_token_id: Any) -> set[int]:
    if eos_token_id is None:
        return set()
    if isinstance(eos_token_id, int):
        return {eos_token_id}
    return {int(token_id) for token_id in eos_token_id}


def _special_token_ids(tokenizer: Any, model: Any) -> tuple[Any, int]:
    generation_config = getattr(model, "generation_config", None)
    eos = getattr(tokenizer, "eos_token_id", None)
    if eos is None and generation_config is not None:
        eos = getattr(generation_config, "eos_token_id", None)
    pad = getattr(tokenizer, "pad_token_id", None)
    if pad is None and generation_config is not None:
        pad = getattr(generation_config, "pad_token_id", None)
    if pad is None:
        if isinstance(eos, int):
            pad = eos
        elif eos:
            pad = int(eos[0])
    if pad is None:
        raise ValueError("Tokenizer/model exposes neither a pad token nor an EOS token.")
    return eos, int(pad)


def _decode_one_completion(
    tokenizer: Any,
    generated_ids: Sequence[int],
    *,
    eos_token_id: Any,
    max_new_tokens: int,
) -> tuple[str, tuple[int, ...], str, bool]:
    ids = [int(token_id) for token_id in generated_ids]
    eos_ids = _as_eos_set(eos_token_id)
    eos_position = next((i for i, token_id in enumerate(ids) if token_id in eos_ids), None)
    visible_ids = ids if eos_position is None else ids[:eos_position]
    raw_completion = tokenizer.decode(
        visible_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    if "\n" in raw_completion:
        reason = "newline_boundary"
        truncated = False
    elif eos_position is not None:
        reason = "eos"
        truncated = False
    elif len(ids) >= max_new_tokens:
        reason = "max_new_tokens"
        truncated = True
    else:
        reason = "other_stop"
        truncated = False
    return raw_completion, tuple(visible_ids), reason, truncated


def generate_batch(
    *,
    tokenizer: Any,
    model: Any,
    context: str,
    add_special_tokens: bool,
    batch_size: int,
    config: GenerationConfig,
) -> list[tuple[str, tuple[int, ...], str, bool]]:
    """Draw one decoding batch from repeated copies of the same context."""

    import torch

    get_model_input_device, _ = _model_api()
    encoded = tokenizer(
        context,
        return_tensors="pt",
        add_special_tokens=add_special_tokens,
    )
    if "input_ids" not in encoded:
        raise ValueError("Tokenizer output does not contain input_ids.")
    prompt_length = int(encoded["input_ids"].shape[1])
    device = get_model_input_device(model)
    batched: dict[str, Any] = {}
    for key, value in encoded.items():
        if not hasattr(value, "to"):
            continue
        repeats = [batch_size] + [1] * (value.ndim - 1)
        batched[key] = value.repeat(*repeats).to(device)

    eos_token_id, pad_token_id = _special_token_ids(tokenizer, model)
    with torch.inference_mode():
        sequences = model.generate(
            **batched,
            do_sample=config.do_sample,
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            num_beams=config.num_beams,
            repetition_penalty=config.repetition_penalty,
            max_new_tokens=config.max_new_tokens,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            use_cache=True,
        )
    if hasattr(sequences, "sequences"):
        sequences = sequences.sequences
    if int(sequences.shape[0]) != batch_size:
        raise RuntimeError(
            f"generate returned {int(sequences.shape[0])} rows for batch_size={batch_size}."
        )
    results = []
    for sequence in sequences:
        continuation = sequence[prompt_length:].detach().cpu().tolist()
        results.append(
            _decode_one_completion(
                tokenizer,
                continuation,
                eos_token_id=eos_token_id,
                max_new_tokens=config.max_new_tokens,
            )
        )
    return results


def make_sample_record(
    *,
    config: GenerationConfig,
    prompt_hash: str,
    sampling_seed: int,
    sample_index: int,
    raw_completion: str,
    token_ids: tuple[int, ...],
    termination_reason: str,
    truncated: bool,
) -> GeneratedSampleRecord:
    parsed = parse_generated_number(
        raw_completion,
        lower=config.lower,
        upper=config.upper,
    )
    seed_position = config.sampling_seeds.index(sampling_seed)
    assert config.run_id is not None and config.model_label is not None
    return GeneratedSampleRecord(
        schema_version=SCHEMA_VERSION,
        config_hash=config.config_hash,
        prompt_hash=prompt_hash,
        run_id=config.run_id,
        model_name=config.model_name,
        model_label=config.model_label,
        prompt_protocol=config.prompt_protocol,
        distribution=config.distribution,
        distribution_params=dict(config.params),
        parameter_id=config.parameter_id,
        prompt_type=config.prompt_type,
        icl_seed=config.icl_seed,
        sampling_seed=sampling_seed,
        sample_index=sample_index,
        global_sample_index=seed_position * config.samples_per_seed + sample_index,
        raw_completion=raw_completion,
        response_text=parsed.response_text,
        normalized_text=parsed.normalized_text,
        generated_token_ids=token_ids,
        generated_token_count=len(token_ids),
        parsed_value=parsed.parsed_value,
        numeric_valid=parsed.numeric_valid,
        canonical_format_valid=parsed.canonical_format_valid,
        support_valid=parsed.support_valid,
        strict_valid=parsed.strict_valid and not truncated,
        negative_zero=parsed.negative_zero,
        extra_after_first_line=parsed.extra_after_first_line,
        termination_reason=termination_reason,
        truncated=truncated,
        error_type=parsed.error_type if not truncated else (parsed.error_type or "truncated"),
    )


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _atomic_write_jsonl(path: Path, records: Iterable[GeneratedSampleRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for record in records:
                handle.write(_canonical_json(record.to_dict()))
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def seed_block_path(config: GenerationConfig, seed: int) -> Path:
    return config.condition_dir / "blocks" / f"seed_{seed:010d}.jsonl"


def validate_seed_block(
    path: Path,
    *,
    config: GenerationConfig,
    prompt_hash: str,
    seed: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"Blank line in {path} at line {line_number}.")
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} at line {line_number}.") from exc
            rows.append(row)
    if len(rows) != config.samples_per_seed:
        raise ValueError(
            f"{path} contains {len(rows)} rows; expected {config.samples_per_seed}."
        )
    expected_indices = list(range(config.samples_per_seed))
    if [row.get("sample_index") for row in rows] != expected_indices:
        raise ValueError(f"{path} has missing, duplicate, or unordered sample indices.")
    for row in rows:
        if row.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"Schema mismatch in {path}.")
        if row.get("config_hash") != config.config_hash:
            raise ValueError(f"Configuration mismatch in {path}.")
        if row.get("prompt_hash") != prompt_hash:
            raise ValueError(f"Prompt mismatch in {path}.")
        if row.get("sampling_seed") != seed:
            raise ValueError(f"Sampling-seed mismatch in {path}.")
    return rows


def _backup_invalid(path: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    candidate = path.with_name(f"{path.name}.invalid.{stamp}")
    counter = 1
    while candidate.exists():
        candidate = path.with_name(f"{path.name}.invalid.{stamp}.{counter}")
        counter += 1
    shutil.move(str(path), str(candidate))
    return candidate


def _software_versions() -> dict[str, str]:
    versions = {"python": sys.version.split()[0]}
    for package in ("torch", "transformers", "numpy", "scipy"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def _prepare_metadata(
    config: GenerationConfig,
    *,
    prompt: str,
    prompt_hash: str,
    context: str | None,
    backend: str | None,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "created_or_checked_at_utc": _utc_now(),
        "status": "running" if context is not None else "dry_run",
        "condition": config.to_metadata(),
        "prompt": prompt,
        "prompt_hash": prompt_hash,
        "rendered_context": context,
        "model_backend": backend,
        "software_versions": _software_versions(),
        "notes": {
            "log_probabilities_collected": False,
            "reason": "Empirical generated-sample frequencies are the downstream estimand.",
            "first_newline_is_response_boundary": True,
            "numeric_vocabulary_constraints": False,
        },
    }


def _check_or_write_metadata(path: Path, metadata: Mapping[str, Any]) -> None:
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            existing = json.load(handle)
        existing_condition = existing.get("condition", {})
        new_condition = metadata.get("condition", {})
        if existing_condition.get("config_hash") != new_condition.get("config_hash"):
            raise ValueError(
                f"Existing {path} belongs to a different configuration. Use a new run_id."
            )
        if existing.get("prompt_hash") != metadata.get("prompt_hash"):
            raise ValueError(f"Existing {path} was created from a different prompt.")
    _atomic_write_json(path, metadata)


def summarise_rows(config: GenerationConfig, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    n = len(rows)

    def rate(field_name: str) -> float | None:
        return None if n == 0 else sum(bool(row.get(field_name)) for row in rows) / n

    values = [
        float(row["parsed_value"])
        for row in rows
        if row.get("strict_valid") and row.get("parsed_value") is not None
    ]
    value_summary: dict[str, float | int | None] = {"n": len(values)}
    if values:
        import numpy as np

        array = np.asarray(values, dtype=float)
        value_summary.update(
            {
                "mean": float(np.mean(array)),
                "sample_std": float(np.std(array, ddof=1)) if len(array) > 1 else None,
                "min": float(np.min(array)),
                "q25": float(np.quantile(array, 0.25)),
                "median": float(np.median(array)),
                "q75": float(np.quantile(array, 0.75)),
                "max": float(np.max(array)),
                "n_unique": int(len(set(values))),
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "completed_at_utc": _utc_now(),
        "status": "complete" if n == config.total_samples else "incomplete",
        "run_id": config.run_id,
        "config_hash": config.config_hash,
        "expected_samples": config.total_samples,
        "stored_samples": n,
        "sampling_seed_blocks": list(config.sampling_seeds),
        "rates": {
            "numeric_valid": rate("numeric_valid"),
            "canonical_format_valid": rate("canonical_format_valid"),
            "support_valid": rate("support_valid"),
            "strict_valid": rate("strict_valid"),
            "truncated": rate("truncated"),
            "negative_zero": rate("negative_zero"),
            "extra_after_first_line": rate("extra_after_first_line"),
        },
        "error_counts": dict(Counter(str(row.get("error_type")) for row in rows if row.get("error_type"))),
        "termination_counts": dict(Counter(str(row.get("termination_reason")) for row in rows)),
        "strict_valid_value_summary": value_summary,
    }


def run_generation(
    config: GenerationConfig,
    *,
    dry_run: bool = False,
    repair_incomplete: bool = False,
) -> dict[str, Any]:
    """Run or safely resume one complete generation condition."""

    prompt = build_condition_prompt(config)
    prompt_hash = _sha256_text(prompt)
    if dry_run:
        metadata = _prepare_metadata(
            config,
            prompt=prompt,
            prompt_hash=prompt_hash,
            context=None,
            backend=None,
        )
        print(json.dumps(metadata, indent=2, sort_keys=True, ensure_ascii=False))
        return metadata

    _, load_lm_backend = _model_api()
    print(f"[condition] run_id={config.run_id}")
    print(f"[condition] expected_samples={config.total_samples}")
    print(f"[model] loading {config.model_name} in BF16")
    loaded = load_lm_backend(model_name=config.model_name, load_in_4bit=False)
    tokenizer, model = loaded.tokenizer, loaded.model
    context, add_special_tokens = render_generation_context(
        tokenizer, prompt, config.prompt_protocol
    )

    condition_dir = config.condition_dir
    condition_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = condition_dir / "condition_metadata.json"
    metadata = _prepare_metadata(
        config,
        prompt=prompt,
        prompt_hash=prompt_hash,
        context=context,
        backend=loaded.backend,
    )
    _check_or_write_metadata(metadata_path, metadata)

    all_rows: list[dict[str, Any]] = []
    try:
        for seed in config.sampling_seeds:
            block_path = seed_block_path(config, seed)
            if block_path.exists():
                try:
                    rows = validate_seed_block(
                        block_path, config=config, prompt_hash=prompt_hash, seed=seed
                    )
                    print(f"[resume] valid seed={seed}; skipping {len(rows)} stored samples")
                    all_rows.extend(rows)
                    continue
                except (OSError, ValueError, json.JSONDecodeError) as exc:
                    if not repair_incomplete:
                        raise RuntimeError(
                            f"Existing block failed validation: {block_path}. "
                            "Inspect it or rerun with --repair-incomplete."
                        ) from exc
                    backup = _backup_invalid(block_path)
                    print(f"[repair] moved invalid block to {backup}")

            print(f"[generate] sampling_seed={seed}")
            set_sampling_seed(seed)
            block_records: list[GeneratedSampleRecord] = []
            while len(block_records) < config.samples_per_seed:
                current_batch = min(
                    config.batch_size, config.samples_per_seed - len(block_records)
                )
                decoded = generate_batch(
                    tokenizer=tokenizer,
                    model=model,
                    context=context,
                    add_special_tokens=add_special_tokens,
                    batch_size=current_batch,
                    config=config,
                )
                for raw, token_ids, reason, truncated in decoded:
                    index = len(block_records)
                    block_records.append(
                        make_sample_record(
                            config=config,
                            prompt_hash=prompt_hash,
                            sampling_seed=seed,
                            sample_index=index,
                            raw_completion=raw,
                            token_ids=token_ids,
                            termination_reason=reason,
                            truncated=truncated,
                        )
                    )
                print(
                    f"[generate] seed={seed} samples={len(block_records)}/"
                    f"{config.samples_per_seed}",
                    flush=True,
                )
            _atomic_write_jsonl(block_path, block_records)
            rows = validate_seed_block(
                block_path, config=config, prompt_hash=prompt_hash, seed=seed
            )
            all_rows.extend(rows)
            print(f"[write] completed {block_path}")

        summary = summarise_rows(config, all_rows)
        _atomic_write_json(condition_dir / "run_summary.json", summary)
        metadata["status"] = "complete"
        metadata["completed_at_utc"] = _utc_now()
        _atomic_write_json(metadata_path, metadata)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return summary
    finally:
        del model
        del loaded
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


def _parse_params(text: str) -> dict[str, Any]:
    try:
        params = json.loads(text)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(f"--params must be valid JSON: {exc}") from exc
    if not isinstance(params, dict):
        raise argparse.ArgumentTypeError("--params must decode to a JSON object.")
    return params


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate one resumable condition of independent LM number samples."
    )
    parser.add_argument("--model-name", required=True, help="Hugging Face model ID/path.")
    parser.add_argument("--model-label", default=None, help="Short thesis-facing model label.")
    parser.add_argument("--distribution", required=True, choices=SUPPORTED_DISTRIBUTIONS)
    parser.add_argument(
        "--params",
        required=True,
        type=_parse_params,
        help='Distribution parameters as JSON, e.g. \'{"mean":0,"std":1}\'.',
    )
    parser.add_argument("--parameter-id", default=None, help="Optional grid label, e.g. N1.")
    parser.add_argument("--prompt-type", default="plain", choices=SUPPORTED_PROMPT_TYPES)
    parser.add_argument("--prompt-protocol", default="raw_direct", choices=sorted(PROMPT_PROTOCOLS))
    parser.add_argument("--experiment-layer", default="unspecified")
    parser.add_argument("--run-id", default=None, help="Optional safe directory name.")
    parser.add_argument("--support-mode", default=None, choices=sorted(SUPPORT_MODES))
    parser.add_argument("--lower", type=float, default=None)
    parser.add_argument("--upper", type=float, default=None)
    parser.add_argument("--icl-n-examples", type=int, default=5)
    parser.add_argument("--icl-seed", type=int, default=0)
    parser.add_argument(
        "--sampling-seeds", type=int, nargs="+", default=list(DEFAULT_SAMPLING_SEEDS)
    )
    parser.add_argument("--samples-per-seed", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/sample_generation_v1"))
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print the exact condition/prompt without loading a model or writing files.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Persist only eight samples from the first seed in a separate smoke-test run.",
    )
    parser.add_argument(
        "--repair-incomplete",
        action="store_true",
        help="Back up and regenerate a seed block only when it fails validation.",
    )
    return parser


def config_from_args(args: argparse.Namespace) -> GenerationConfig:
    seeds = tuple(args.sampling_seeds)
    samples_per_seed = args.samples_per_seed
    batch_size = args.batch_size
    run_id = args.run_id
    layer = args.experiment_layer
    if args.smoke_test:
        seeds = seeds[:1]
        samples_per_seed = min(samples_per_seed, 8)
        batch_size = min(batch_size, samples_per_seed)
        run_id = f"{run_id}__smoke" if run_id else None
        layer = f"{layer}_smoke"
    return GenerationConfig(
        model_name=args.model_name,
        model_label=args.model_label,
        distribution=args.distribution,
        params=args.params,
        parameter_id=args.parameter_id,
        prompt_type=args.prompt_type,
        prompt_protocol=args.prompt_protocol,
        experiment_layer=layer,
        run_id=run_id,
        support_mode=args.support_mode,
        lower=args.lower,
        upper=args.upper,
        icl_n_examples=args.icl_n_examples,
        icl_seed=args.icl_seed,
        sampling_seeds=seeds,
        samples_per_seed=samples_per_seed,
        batch_size=batch_size,
        max_new_tokens=args.max_new_tokens,
        output_dir=args.output_dir,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    try:
        config = config_from_args(args)
        run_generation(
            config,
            dry_run=args.dry_run,
            repair_incomplete=args.repair_incomplete,
        )
    except Exception as exc:
        print(f"[error] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
