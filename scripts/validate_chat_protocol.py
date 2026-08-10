#!/usr/bin/env python3
"""
Tokenizer-only validation for raw_direct and chat_direct contexts.

No model weights/GPU are loaded.

For each supplied model:
1) Verify raw_direct remains exactly prompt + "\\n" + prefix.
2) Render chat_direct with the model's own chat template.
3) Verify the numeric prefix is appended to the assistant generation turn.
4) Verify every valid numeric candidate extends the chat context without
   retokenizing it and uses the same standalone single-token ID.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from transformers import AutoProcessor, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
for candidate in [REPO_ROOT, REPO_ROOT / "src", REPO_ROOT / "scripts"]:
    if candidate.exists():
        sys.path.insert(0, str(candidate))

from lm_next_token import build_lm_context, token_ids_for_strings
from prompt_template import build_prompt
from real_prefix_logic import valid_next_tokens


PREFIXES = [
    "",
    "-",
    "0",
    "1",
    "10",
    "-1",
    "-10",
    "0.",
    "1.",
    "10.",
    "-1.",
    "-10.",
    "0.5",
    "10.5",
    "-10.5",
    "0.50",
    "10.50",
    "-10.50",
]


def load_tokenizer_only(model_name: str):
    try:
        tok = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
        return tok, "AutoTokenizer"
    except Exception as tok_error:
        print(
            f"[info] AutoTokenizer failed for {model_name}: "
            f"{type(tok_error).__name__}: {tok_error}"
        )

    proc = AutoProcessor.from_pretrained(
        model_name,
        trust_remote_code=True,
    )
    if not hasattr(proc, "tokenizer"):
        raise RuntimeError(
            f"AutoProcessor loaded for {model_name}, but processor.tokenizer is missing."
        )
    return proc.tokenizer, "AutoProcessor.tokenizer"


def encode(tokenizer, text: str, add_special_tokens: bool):
    return tokenizer.encode(
        text,
        add_special_tokens=add_special_tokens,
    )


def pieces(tokenizer, ids):
    return [tokenizer.decode([i]) for i in ids]


def validate_context_extension(
    tokenizer,
    context: str,
    candidate: str,
    standalone_id: int,
    add_special_tokens: bool,
):
    context_ids = encode(tokenizer, context, add_special_tokens)
    full_ids = encode(tokenizer, context + candidate, add_special_tokens)

    if full_ids[: len(context_ids)] != context_ids:
        return False, "PREFIX_RETOKENIZATION", context_ids, full_ids

    tail = full_ids[len(context_ids):]
    if tail != [standalone_id]:
        return False, f"TAIL={tail}_EXPECTED={[standalone_id]}", context_ids, full_ids

    return True, "OK", context_ids, full_ids


def audit_model(model_name: str):
    print("\n" + "=" * 110)
    print(f"MODEL: {model_name}")

    tokenizer, loader = load_tokenizer_only(model_name)
    print(f"tokenizer_loader: {loader}")
    print(f"tokenizer_class:  {type(tokenizer).__name__}")

    prompt = build_prompt(
        distribution="normal",
        params={"mean": 10.0, "std": 0.5},
        mean=10.0,
        std=0.5,
        decimals=3,
        prompt_type="plain",
        support_mode="agnostic",
        lower=None,
        upper=None,
    )

    symbol_ids = token_ids_for_strings(
        ["-", "."] + [str(i) for i in range(10)],
        tokenizer,
    )

    # raw_direct regression: this must be identical to the old framework.
    raw_context, raw_add_special = build_lm_context(
        tokenizer=tokenizer,
        prompt=prompt,
        prefix="10.5",
        prompt_protocol="raw_direct",
    )

    assert raw_context == prompt + "\n" + "10.5"
    assert raw_add_special is True
    print("[PASS] raw_direct regression: exact old prompt + newline + prefix behavior")

    # chat render smoke
    chat_root, chat_add_special = build_lm_context(
        tokenizer=tokenizer,
        prompt=prompt,
        prefix="",
        prompt_protocol="chat_direct",
    )
    chat_prefixed, chat_add_special_2 = build_lm_context(
        tokenizer=tokenizer,
        prompt=prompt,
        prefix="10.5",
        prompt_protocol="chat_direct",
    )

    assert chat_add_special is False
    assert chat_add_special_2 is False
    assert chat_prefixed == chat_root + "10.5"

    print("[PASS] chat_direct render: prefix is appended after assistant generation marker")
    print("\n--- rendered chat ROOT context ---")
    print(repr(chat_root))
    print("--- rendered chat prefix='10.5' tail ---")
    print(repr(chat_prefixed[-300:]))

    failures = []
    checked = 0

    for prefix in PREFIXES:
        allowed = valid_next_tokens(
            prefix=prefix,
            decimals=3,
            allow_negative=True,
        )

        context, add_special_tokens = build_lm_context(
            tokenizer=tokenizer,
            prompt=prompt,
            prefix=prefix,
            prompt_protocol="chat_direct",
        )

        for candidate in allowed:
            checked += 1
            ok, reason, context_ids, full_ids = validate_context_extension(
                tokenizer=tokenizer,
                context=context,
                candidate=candidate,
                standalone_id=symbol_ids[candidate],
                add_special_tokens=add_special_tokens,
            )
            if not ok:
                failures.append(
                    (
                        prefix,
                        candidate,
                        reason,
                        pieces(tokenizer, context_ids[-15:]),
                        pieces(tokenizer, full_ids[-15:]),
                    )
                )

    print("\n[CHAT CONTEXT TOKEN STABILITY]")
    print(f"checked={checked} passed={checked-len(failures)} failed={len(failures)}")

    for prefix, candidate, reason, context_tail, full_tail in failures[:30]:
        print(
            f"FAIL prefix={prefix!r} candidate={candidate!r} reason={reason}\n"
            f"  context_tail={context_tail}\n"
            f"  full_tail={full_tail}"
        )

    if failures:
        print("\n[VERDICT] FAIL")
        return False

    print("\n[VERDICT] STRICT PASS")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-name",
        action="append",
        required=True,
        help="Repeat for every model/tokenizer to audit.",
    )
    args = parser.parse_args()

    results = {}
    for model_name in args.model_name:
        try:
            results[model_name] = audit_model(model_name)
        except Exception as exc:
            print("\n" + "!" * 110)
            print(
                f"ERROR {model_name}: {type(exc).__name__}: {exc}"
            )
            results[model_name] = False

    print("\n" + "=" * 110)
    print("FINAL SUMMARY")
    for model_name, ok in results.items():
        print(f"{'PASS' if ok else 'FAIL'}  {model_name}")

    raise SystemExit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    main()
