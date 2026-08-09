#!/usr/bin/env python3
"""
Audit whether the thesis single-token scorer is tokenizer-safe.

Checks:
1) Each numeric symbol used by the grammar is exactly one standalone token.
2) Representative numeric strings tokenize character-by-character.
3) CRITICAL: in the exact raw-direct context used by the experiment,
   tokenizing context + candidate must equal tokenizing(context) followed by
   exactly the same standalone candidate token ID.

This script loads tokenizers/processors only -- no model weights and no GPU.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from transformers import AutoProcessor, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if SRC.exists():
    sys.path.insert(0, str(SRC))

from prompt_template import build_prompt
from real_prefix_logic import valid_next_tokens

DIGITS = [str(i) for i in range(10)]
GRAMMAR_SYMBOLS = ["-", "."] + DIGITS

NUMBER_STRINGS = [
    "5", "50", "100", "290", "0.527", "10.527", "29.050", "-1.250", "-10.527",
]

PREFIXES = [
    "", "-", "0", "1", "5", "9", "10", "29", "100", "-1", "-10",
    "0.", "1.", "10.", "29.", "-1.", "-10.",
    "0.5", "10.5", "-10.5", "0.50", "10.50", "-10.50",
]


def load_tokenizer_only(model_name: str):
    try:
        tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        return tok, "AutoTokenizer"
    except Exception as tok_error:
        print(f"[info] AutoTokenizer failed for {model_name}: {type(tok_error).__name__}: {tok_error}")

    proc = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    if not hasattr(proc, "tokenizer"):
        raise RuntimeError(f"AutoProcessor loaded for {model_name}, but it has no .tokenizer")
    return proc.tokenizer, "AutoProcessor.tokenizer"


def ids(tokenizer, text: str, add_special_tokens: bool = False):
    return tokenizer.encode(text, add_special_tokens=add_special_tokens)


def pieces(tokenizer, token_ids):
    return [tokenizer.decode([i]) for i in token_ids]


def standalone_symbol_ids(tokenizer):
    mapping = {}
    failures = []
    for s in GRAMMAR_SYMBOLS:
        token_ids = ids(tokenizer, s, add_special_tokens=False)
        if len(token_ids) != 1:
            failures.append((s, token_ids, pieces(tokenizer, token_ids)))
        else:
            mapping[s] = token_ids[0]
    return mapping, failures


def check_number_strings(tokenizer, symbol_ids):
    rows = []
    for text in NUMBER_STRINGS:
        actual = ids(tokenizer, text, add_special_tokens=False)
        expected = []
        expected_possible = True
        for ch in text:
            if ch not in symbol_ids:
                expected_possible = False
                break
            expected.append(symbol_ids[ch])
        ok = expected_possible and actual == expected
        rows.append({
            "text": text,
            "ok": ok,
            "actual": actual,
            "actual_pieces": pieces(tokenizer, actual),
            "expected": expected if expected_possible else None,
        })
    return rows


def check_context_candidate(tokenizer, context: str, candidate: str, standalone_id: int):
    context_ids = ids(tokenizer, context, add_special_tokens=True)
    full_ids = ids(tokenizer, context + candidate, add_special_tokens=True)

    prefix_stable = full_ids[: len(context_ids)] == context_ids
    if not prefix_stable:
        return {
            "ok": False,
            "reason": "PREFIX_RETOKENIZATION",
            "context_tail_pieces": pieces(tokenizer, context_ids[-12:]),
            "full_tail_pieces": pieces(tokenizer, full_ids[-12:]),
            "tail_ids": None,
        }

    tail = full_ids[len(context_ids):]
    if len(tail) != 1:
        return {
            "ok": False,
            "reason": f"TAIL_LENGTH_{len(tail)}",
            "context_tail_pieces": pieces(tokenizer, context_ids[-12:]),
            "full_tail_pieces": pieces(tokenizer, full_ids[-12:]),
            "tail_ids": tail,
        }

    if tail[0] != standalone_id:
        return {
            "ok": False,
            "reason": "CONTEXT_ID_DIFFERS_FROM_STANDALONE_ID",
            "context_tail_pieces": pieces(tokenizer, context_ids[-12:]),
            "full_tail_pieces": pieces(tokenizer, full_ids[-12:]),
            "tail_ids": tail,
        }

    return {"ok": True, "reason": "OK", "tail_ids": tail}


def build_audit_prompts():
    return {
        "normal_0_1_plain": build_prompt(
            distribution="normal",
            params={"mean": 0.0, "std": 1.0},
            mean=0.0,
            std=1.0,
            decimals=3,
            prompt_type="plain",
            support_mode="agnostic",
            lower=None,
            upper=None,
        ),
        "normal_10_0p5_plain": build_prompt(
            distribution="normal",
            params={"mean": 10.0, "std": 0.5},
            mean=10.0,
            std=0.5,
            decimals=3,
            prompt_type="plain",
            support_mode="agnostic",
            lower=None,
            upper=None,
        ),
    }


def audit_model(model_name: str):
    print("\n" + "=" * 100)
    print(f"MODEL: {model_name}")

    tokenizer, loader = load_tokenizer_only(model_name)
    print(f"tokenizer_loader: {loader}")
    print(f"tokenizer_class:  {type(tokenizer).__name__}")

    symbol_ids, standalone_failures = standalone_symbol_ids(tokenizer)

    print("\n[1] STANDALONE GRAMMAR SYMBOLS")
    if standalone_failures:
        print("FAIL")
        for symbol, token_ids, token_pieces in standalone_failures:
            print(f"  {symbol!r}: ids={token_ids} pieces={token_pieces}")
    else:
        print("PASS: every '-', '.', and digit 0-9 is exactly one token.")
        for symbol in GRAMMAR_SYMBOLS:
            print(f"  {symbol!r}: id={symbol_ids[symbol]} decoded={tokenizer.decode([symbol_ids[symbol]])!r}")

    print("\n[2] REPRESENTATIVE COMPLETE NUMERIC STRINGS")
    string_rows = check_number_strings(tokenizer, symbol_ids)
    for row in string_rows:
        status = "PASS" if row["ok"] else "FAIL"
        print(f"  [{status}] {row['text']!r} pieces={row['actual_pieces']} ids={row['actual']}")
        if not row["ok"]:
            print(f"         expected_char_ids={row['expected']}")

    print("\n[3] CRITICAL CONTEXT-STABILITY TEST")
    prompts = build_audit_prompts()
    failures = []
    total = 0

    for prompt_name, prompt in prompts.items():
        for prefix in PREFIXES:
            try:
                allowed = valid_next_tokens(prefix=prefix, decimals=3, allow_negative=True)
            except ValueError:
                continue

            context = prompt + "\n" + prefix
            for candidate in allowed:
                total += 1
                if candidate not in symbol_ids:
                    failures.append((prompt_name, prefix, candidate, {"reason": "NO_SINGLE_STANDALONE_ID", "ok": False}))
                    continue

                result = check_context_candidate(
                    tokenizer=tokenizer,
                    context=context,
                    candidate=candidate,
                    standalone_id=symbol_ids[candidate],
                )
                if not result["ok"]:
                    failures.append((prompt_name, prefix, candidate, result))

    passed = total - len(failures)
    print(f"checked={total} passed={passed} failed={len(failures)}")

    if failures:
        print("FAILURES:")
        for prompt_name, prefix, candidate, result in failures:
            print(f"  prompt={prompt_name} prefix={prefix!r} candidate={candidate!r} reason={result['reason']}")
            if "context_tail_pieces" in result:
                print(f"      context_tail={result['context_tail_pieces']}")
                print(f"      full_tail={result['full_tail_pieces']}")
            if result.get("tail_ids") is not None:
                print(f"      candidate_tail_ids={result['tail_ids']}")

    strict_pass = (
        len(standalone_failures) == 0
        and all(row["ok"] for row in string_rows)
        and len(failures) == 0
    )

    print("\n[VERDICT]")
    if strict_pass:
        print("STRICT PASS: the current exact single-token restricted scorer is tokenizer-compatible for all audited symbols/contexts.")
    else:
        print("FAIL: at least one assumption of the current exact single-token restricted scorer does not hold.")

    return strict_pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-name",
        action="append",
        required=True,
        help="Hugging Face model ID. Repeat this option for multiple models.",
    )
    args = parser.parse_args()

    results = {}
    for model_name in args.model_name:
        try:
            results[model_name] = audit_model(model_name)
        except Exception as exc:
            print("\n" + "!" * 100)
            print(f"ERROR loading/auditing {model_name}: {type(exc).__name__}: {exc}")
            results[model_name] = False

    print("\n" + "=" * 100)
    print("FINAL SUMMARY")
    for model_name, ok in results.items():
        print(f"{'PASS' if ok else 'FAIL'}  {model_name}")

    raise SystemExit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    main()
