# Chat-Protocol Validation — 2026-08-24

## Purpose

Validate the raw_direct and chat_direct implementation before interpreting
the strong Qwen-vs-Gemma raw-to-chat TVD differences.

Validated checkpoints:

- Qwen/Qwen3-4B-Instruct-2507
- Qwen/Qwen3-14B
- google/gemma-4-E4B-it
- google/gemma-4-12B-it

Environment used for the final validation:

- Git commit: 50d4441ef51fbac0c15641ae6051237b59cd9f05
- Python: 3.12.13
- Transformers: 5.12.1

## Checks performed

The validation confirmed that:

- raw_direct preserves the original prompt + newline + prefix protocol.
- chat_direct uses each checkpoint's own tokenizer-provided chat template.
- Qwen and Gemma therefore use different native chat formats.
- The numerical prefix is placed inside the assistant/model response.
- enable_thinking=False is explicitly used by the framework.
- Chat-rendered contexts are tokenized with add_special_tokens=False.
- No second BOS/EOS/chat wrapper is added.
- Candidate continuations preserve the existing tokenized prefix.
- Valid numerical continuations are scored at the intended continuation point.

The strict tokenizer audit checked 176 continuation cases per checkpoint.

Results:

- Qwen 4B Instruct: 176/176 PASS
- Qwen 14B: 176/176 PASS
- Gemma E4B IT: 176/176 PASS
- Gemma 12B IT: 176/176 PASS

## Assistant-prefill comparison

The current framework construction was compared with the alternative
continue_final_message=True assistant-prefill construction.

Results:

- Qwen 4B Instruct: exact rendered-string and token-ID equivalence.
- Qwen 14B: exact rendered-string and token-ID equivalence.
- Gemma E4B IT: exact rendered-string and token-ID equivalence.
- Gemma 12B IT: expected model-specific difference.

For Gemma 12B, its native generation-start template inserts an empty
disabled-thinking channel before the direct answer:

<|channel>thought
<channel|>

Inspection of the tokenizer template confirmed that this behavior is
explicitly part of the Gemma 12B native template when thinking is disabled.
It is not introduced by the thesis framework.

## Final verdict

CHAT-PROTOCOL IMPLEMENTATION VALIDATION: PASS

Framework decision: KEEP

Main-grid rerun required: NO

No evidence was found that the observed raw-to-chat effects are caused by
incorrect template selection, role placement, numerical-prefix placement,
duplicate special tokens, reasoning leakage, or prefix-retokenization errors.

The raw-to-chat differences can therefore be retained and analyzed as
interaction-protocol effects.

## Interpretation constraint

The chat condition should be interpreted as each checkpoint operating under
its own native interaction protocol.

Qwen and Gemma do not receive one common chat wrapper. Therefore the
cross-family result should not be described as the causal effect of a single
universal chat template.

The appropriate interpretation is that conditional numerical probability
distributions are sensitive to interaction protocol, with the direction and
magnitude of the effect differing across model families.

## Files

- scripts/validate_chat_protocol.py
- scripts/validate_chat_protocol_final.py
- validation/chat_protocol_validation_final_2026-08-24.log
- validation/chat_protocol_validation_2026-08-24.md
