from xml.parsers.expat import model

import torch
import torch.nn.functional as F
from model_loader import load_lm_backend, get_model_input_device
from prompt_template import build_prompt, ALL_PROMPT_TYPES

import numpy as np
from real_prefix_logic import valid_next_tokens


_MODEL_CACHE = {}


def load_lm(
    model_name: str = "Qwen/Qwen3-4B",
    load_in_4bit: bool = True,
):
    cache_key = (model_name, load_in_4bit)

    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    loaded = load_lm_backend(
        model_name=model_name,
        load_in_4bit=load_in_4bit,
    )

    print(f"[loader] Loaded {model_name} using backend={loaded.backend}")

    _MODEL_CACHE[cache_key] = (loaded.tokenizer, loaded.model)
    return loaded.tokenizer, loaded.model


def token_ids_for_strings(strings,tokenizer):

    mapping = {}

    for s in strings:
        ids = tokenizer.encode(s, add_special_tokens=False)
        if len(ids) != 1:
            pieces = [tokenizer.decode([id]) for id in ids]
            raise ValueError(
                f"Token {repr(s)} does not map to exactly one tokenizer token:"
                f" ids={ids}, pieces={pieces}"
        
            )
        mapping[s] = ids[0]

    return mapping


def next_token_distribution(
    prompt: str,
    prefix: str,
    decimals: int,
    allow_negative: bool = True,
    model_name: str = "Qwen/Qwen3-4B",
    load_in_4bit: bool = True,
):
    """
    Restricted next-token distribution over logically valid numeric continuations.
    """
    allowed = valid_next_tokens(
        prefix=prefix,
        decimals=decimals,
        allow_negative=allow_negative,
    )

    if len(allowed) == 0:
        return {}, {}

    tokenizer, model = load_lm(model_name = model_name, load_in_4bit = load_in_4bit)
    allowed_token_ids = token_ids_for_strings(allowed, tokenizer)

    full_text = prompt + "\n" + prefix

    inputs = tokenizer(full_text, return_tensors="pt")

    device = get_model_input_device(model)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)

    next_logits = outputs.logits[0, -1, :]

    restricted_ids = torch.tensor(
        [allowed_token_ids[tok] for tok in allowed],
        device=next_logits.device,
    )

    restricted_logits = next_logits[restricted_ids]
    restricted_logprobs = F.log_softmax(restricted_logits, dim=0)
    restricted_probs = torch.exp(restricted_logprobs)

    probs_dict = {
        tok: restricted_probs[i].item()
        for i, tok in enumerate(allowed)
    }

    logprobs_dict = {
        tok: restricted_logprobs[i].item()
        for i, tok in enumerate(allowed)
    }

    return probs_dict, logprobs_dict


def pretty_print_distribution(title: str, probs_dict: dict[str, float]):
    print(f"\n{title}")
    for tok, prob in sorted(probs_dict.items(), key=lambda x: x[1], reverse=True):
        print(f"  next='{tok}'  prob={prob:.6f}")


if __name__ == "__main__":
    print("=== Tokenization sanity check ===")
    tokens = ["-", ".", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    print(token_ids_for_strings(tokens))

    print("\n=== Prompt render check (no model inference) ===")
    for pt in ALL_PROMPT_TYPES:
        print(f"\n--- {pt} ---")
        print(build_prompt(mean=0.0, std=1.0, decimals=3, prompt_type=pt,
                           icl_n_examples=5, icl_seed=0))

    print("\n=== Positive / bounded inference check ===")
    prompt = build_prompt(
        mean=4.0, std=1.0, decimals=3,
        prompt_type="plain",
        support_mode="bounded", lower=0.0, upper=10.0,
    )
    for prefix in ["", "4", "4.", "4.3", "3.", "5."]:
        probs, _ = next_token_distribution(
            prompt=prompt, prefix=prefix, decimals=3, allow_negative=False,
        )
        pretty_print_distribution(f"prefix={repr(prefix)}", probs)

    print("\n=== Signed / unbounded inference check ===")
    prompt = build_prompt(mean=0.0, std=1.0, decimals=3, prompt_type="plain")
    for prefix in ["", "-", "-1", "-1.", "0", "0.", "1", "1."]:
        probs, _ = next_token_distribution(
            prompt=prompt, prefix=prefix, decimals=3, allow_negative=True,
        )
        pretty_print_distribution(f"prefix={repr(prefix)}", probs)


