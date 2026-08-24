#!/usr/bin/env python3


from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from transformers import AutoTokenizer, __version__ as transformers_version

REPO_ROOT = Path(__file__).resolve().parents[1]
for p in [REPO_ROOT, REPO_ROOT / "src", REPO_ROOT / "scripts"]:
    if p.exists():
        sys.path.insert(0, str(p))

from lm_next_token import build_lm_context


MODELS = [
    "Qwen/Qwen3-4B-Instruct-2507",
    "Qwen/Qwen3-14B",
    "google/gemma-4-E4B-it",
    "google/gemma-4-12B-it",
]

PROMPT = (
    "You are generating from a Normal distribution with mean 10.0 "
    "and standard deviation 0.5. Return one independent sample as "
    "a number with exactly 3 decimal places. Output only the number."
)

PREFIX = "10.5"


def git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
        ).strip()
    except Exception:
        return "UNKNOWN"


def load_tokenizer(model_name: str):
    return AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )


def canonical_prefill(tokenizer, prompt: str, prefix: str) -> str:
    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": prefix},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            continue_final_message=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            continue_final_message=True,
        )


def classify_prefill_result(model_name: str, framework_ctx: str, canonical_ctx: str, ids_a, ids_b):
    if ids_a == ids_b:
        return "PASS", "Exact token-ID equivalence"

    if model_name == "google/gemma-4-12B-it":
        marker = "<|channel>thought\n<channel|>"
        if marker in framework_ctx and marker not in canonical_ctx:
            stripped = framework_ctx.replace(marker, "", 1)
            if stripped == canonical_ctx:
                return (
                    "EXPECTED_DIFF",
                    "Gemma 12B native generation-start template inserts an empty "
                    "disabled-thinking channel; continue_final_message does not."
                )

    return "NEEDS_INSPECTION", "Unexpected token-level difference"


def main():
    print("=" * 110)
    print("FINAL CHAT-PROTOCOL VALIDATION")
    print("=" * 110)
    print("git_commit:", git_head())
    print("python:", sys.version.replace("\n", " "))
    print("transformers:", transformers_version)

    # ------------------------------------------------------------------
    # Part A: run the repository's existing strict tokenizer audit.
    # ------------------------------------------------------------------
    print("\n" + "=" * 110)
    print("PART A — EXISTING STRICT CHAT-PROTOCOL AUDIT")
    print("=" * 110)

    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "validate_chat_protocol.py"),
    ]
    for model in MODELS:
        cmd += ["--model-name", model]

    strict_rc = subprocess.run(cmd, cwd=REPO_ROOT).returncode

    # ------------------------------------------------------------------
    # Part B: assistant-prefill comparison.
    # ------------------------------------------------------------------
    print("\n" + "=" * 110)
    print("PART B — ASSISTANT-PREFILL COMPARISON")
    print("=" * 110)

    results = {}

    for model_name in MODELS:
        print("\n" + "-" * 110)
        print("MODEL:", model_name)

        tokenizer = load_tokenizer(model_name)

        framework_ctx, add_special = build_lm_context(
            tokenizer=tokenizer,
            prompt=PROMPT,
            prefix=PREFIX,
            prompt_protocol="chat_direct",
        )

        if add_special is not False:
            raise AssertionError(
                f"{model_name}: chat_direct unexpectedly returned "
                f"add_special_tokens={add_special}"
            )

        canonical_ctx = canonical_prefill(
            tokenizer=tokenizer,
            prompt=PROMPT,
            prefix=PREFIX,
        )

        ids_a = tokenizer(
            framework_ctx,
            add_special_tokens=False,
        ).input_ids

        ids_b = tokenizer(
            canonical_ctx,
            add_special_tokens=False,
        ).input_ids

        status, note = classify_prefill_result(
            model_name,
            framework_ctx,
            canonical_ctx,
            ids_a,
            ids_b,
        )

        results[model_name] = (status, note)

        print("framework_tail:", repr(framework_ctx[-180:]))
        print("canonical_tail:", repr(canonical_ctx[-180:]))
        print("rendered_strings_equal:", framework_ctx == canonical_ctx)
        print("token_ids_equal:", ids_a == ids_b)
        print("framework_token_count:", len(ids_a))
        print("canonical_token_count:", len(ids_b))
        print("status:", status)
        print("note:", note)

    # ------------------------------------------------------------------
    # Part C: final compact summary.
    # ------------------------------------------------------------------
    print("\n" + "=" * 110)
    print("FINAL SUMMARY")
    print("=" * 110)

    strict_ok = strict_rc == 0

    for model_name in MODELS:
        status, note = results[model_name]
        print(f"{model_name}")
        print(f"  strict_protocol_audit: {'PASS' if strict_ok else 'CHECK LOG'}")
        print(f"  prefill_comparison:    {status}")
        print(f"  note:                  {note}")

    unexpected = [
        m for m, (status, _) in results.items()
        if status == "NEEDS_INSPECTION"
    ]

    if strict_ok and not unexpected:
        print("\nFINAL VERDICT: PASS")
        print("FRAMEWORK DECISION: KEEP")
        print("MAIN-GRID RERUN REQUIRED: NO")
        print(
            "Gemma 12B note: native generation-start formatting includes "
            "an empty disabled-thinking channel; this is reported as an "
            "expected model-specific protocol difference."
        )
        return 0

    print("\nFINAL VERDICT: NEEDS INSPECTION")
    if not strict_ok:
        print("- Existing strict chat-protocol audit returned a non-zero exit code.")
    if unexpected:
        print("- Unexpected prefill differences:", ", ".join(unexpected))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
