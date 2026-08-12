import torch  
import torch.nn.functional as F 
from model_loader import load_lm_backend, get_model_input_device
from prompt_template import build_prompt, ALL_PROMPT_TYPES

import numpy as np 
from real_prefix_logic import valid_next_tokens


_MODEL_CACHE = {}

PROMPT_PROTOCOLS = {"raw_direct", "chat_direct"}


def build_lm_context(
    tokenizer,
    prompt: str,
    prefix: str,
    prompt_protocol: str = "raw_direct",
):
    """
    Build the exact text context consumed by the LM.

    raw_direct:
        Preserves the original thesis protocol exactly:
            prompt + "\\n" + prefix
        The tokenizer may add its usual special tokens.

    chat_direct:
        Treats `prompt` as the user turn, opens the assistant generation turn
        using the model's own chat template, and then appends `prefix` as the
        already-generated beginning of the assistant's numeric answer.

        Because apply_chat_template(tokenize=False) already inserts the model's
        required control/special tokens, the rendered text must later be
        tokenized with add_special_tokens=False.
    """
    if prompt_protocol not in PROMPT_PROTOCOLS:
        raise ValueError(
            f"Unknown prompt_protocol={prompt_protocol!r}. "
            f"Use one of {sorted(PROMPT_PROTOCOLS)}."
        )

    if prompt_protocol == "raw_direct":
        return prompt + "\n" + prefix, True

    if not hasattr(tokenizer, "apply_chat_template"):
        raise ValueError(
            f"Tokenizer {type(tokenizer).__name__} does not expose "
            "apply_chat_template(), so chat_direct cannot be used."
        )

    messages = [{"role": "user", "content": prompt}]

    # Qwen3/Gemma 4 templates support enable_thinking=False. Keep a narrow
    # fallback for templates that do not accept that template variable.
    try:
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    if not isinstance(rendered, str) or len(rendered) == 0:
        raise ValueError(
            "apply_chat_template() did not return a non-empty rendered string."
        )

    # Critical design choice: prefix is assistant output, not part of user turn.
    return rendered + prefix, False



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

def _normalize_logprobs(logprobs_dict: dict[str, float]):
    """
    Normalize unnormalized candidate log-probabilities over the restricted
    allowed candidate set.
    """
    tokens = list(logprobs_dict.keys())

    logps = torch.tensor(
        [logprobs_dict[tok] for tok in tokens],
        dtype=torch.float32,
    )

    norm_logps = logps - torch.logsumexp(logps, dim=0)
    probs = torch.exp(norm_logps)

    probs_dict = {
        tok: probs[i].item()
        for i, tok in enumerate(tokens)
    }

    norm_logprobs_dict = {
        tok: norm_logps[i].item()
        for i, tok in enumerate(tokens)
    }

    return probs_dict, norm_logprobs_dict


def _candidate_tail_ids_and_inputs(
    tokenizer,
    context: str,
    candidate: str,
    add_special_tokens: bool = True,
):
    """
    Find the tokenizer-token continuation needed to append visible `candidate`
    after `context`.

    This handles Mistral-style boundary markers, e.g. visible "0" may involve
    tokenizer pieces like ["▁", "0"], while visible "-" may be ["▁-"].

    It still requires prefix-stable tokenization:
        tokenizer(context + candidate) begins with tokenizer(context)

    If this fails, it is probably a Llama/GLM-style token merge issue,
    not the Mistral boundary-marker issue.
    """
    context_inputs = tokenizer(
        context,
        return_tensors="pt",
        add_special_tokens=add_special_tokens,
    )

    full_inputs = tokenizer(
        context + candidate,
        return_tensors="pt",
        add_special_tokens=add_special_tokens,
    )

    context_ids = context_inputs["input_ids"][0].tolist()
    full_ids = full_inputs["input_ids"][0].tolist()

    if full_ids[: len(context_ids)] != context_ids:
        context_pieces = [tokenizer.decode([i]) for i in context_ids]
        full_pieces = [tokenizer.decode([i]) for i in full_ids]

        raise ValueError(
            "Candidate causes prefix retokenization.\n"
            f"context={context!r}\n"
            f"candidate={candidate!r}\n"
            f"context_ids={context_ids}\n"
            f"context_pieces={context_pieces}\n"
            f"full_ids={full_ids}\n"
            f"full_pieces={full_pieces}\n"
            "This is not just a Mistral boundary-marker issue. "
            "This needs next-visible-character aggregation later."
        )

    tail_ids = full_ids[len(context_ids):]

    if len(tail_ids) == 0:
        raise ValueError(
            f"Candidate {candidate!r} produced empty continuation."
        )

    return context_ids, tail_ids, full_inputs


def _candidate_sequence_logprob(
    tokenizer,
    model,
    context: str,
    candidate: str,
    add_special_tokens: bool = True,
) -> float:
    """
    Compute log P(visible candidate string | context).

    For one-token candidates, this matches the normal next-token score.
    For Mistral-style candidates, this can score multi-token continuations
    such as boundary marker + digit.
    """
    context_ids, tail_ids, full_inputs = _candidate_tail_ids_and_inputs(
        tokenizer=tokenizer,
        context=context,
        candidate=candidate,
        add_special_tokens=add_special_tokens,
    )

    device = get_model_input_device(model)
    full_inputs = {k: v.to(device) for k, v in full_inputs.items()}

    with torch.no_grad():
        outputs = model(**full_inputs)

    logits = outputs.logits[0]

    total_logprob = 0.0
    start = len(context_ids)

    for i, token_id in enumerate(tail_ids):
        absolute_pos = start + i
        previous_pos = absolute_pos - 1

        token_logprobs = F.log_softmax(logits[previous_pos, :].float(), dim=-1)
        total_logprob += token_logprobs[token_id].item()

    return total_logprob

def next_token_distribution_single_token(
    prompt: str,
    prefix: str,
    decimals: int,
    allow_negative: bool = True,
    model_name: str = "Qwen/Qwen3-4B",
    load_in_4bit: bool = True,
    prompt_protocol: str = "raw_direct",
    return_diagnostics: bool = False,
):
    """
    Original method.

    Use this for tokenizer-clean models where each valid visible continuation
    is exactly one tokenizer token, e.g. Qwen/Gemma/VibeThinker.
    """
    allowed = valid_next_tokens(
        prefix=prefix,
        decimals=decimals,
        allow_negative=allow_negative,
    )

    if len(allowed) == 0:
        diagnostics = {
            "candidate_probs_unconditional": {},
            "valid_candidate_mass": 0.0,
            "other_vocab_mass": 1.0,
        }
        if return_diagnostics:
            return {}, {}, diagnostics
        return {}, {}

    tokenizer, model = load_lm(
        model_name=model_name,
        load_in_4bit=load_in_4bit,
    )

    allowed_token_ids = token_ids_for_strings(allowed, tokenizer)

    full_text, add_special_tokens = build_lm_context(
        tokenizer=tokenizer,
        prompt=prompt,
        prefix=prefix,
        prompt_protocol=prompt_protocol,
    )

    inputs = tokenizer(
        full_text,
        return_tensors="pt",
        add_special_tokens=add_special_tokens,
    )

    device = get_model_input_device(model)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    next_logits = outputs.logits[0, -1, :].float()

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

    full_probs = F.softmax(next_logits, dim=-1)

    candidate_probs_unconditional = {
        tok: full_probs[allowed_token_ids[tok]].item()
        for tok in allowed
    }

    valid_candidate_mass = float(
        sum(candidate_probs_unconditional.values())
    )

    if not (-1e-6 <= valid_candidate_mass <= 1.0 + 1e-6):
        raise ValueError(
            f"valid_candidate_mass out of range: "
            f"{valid_candidate_mass}"
        )

    valid_candidate_mass = min(
        max(valid_candidate_mass, 0.0),
        1.0,
    )

    other_vocab_mass = 1.0 - valid_candidate_mass

    diagnostics = {
        "candidate_probs_unconditional": candidate_probs_unconditional,
        "valid_candidate_mass": valid_candidate_mass,
        "other_vocab_mass": other_vocab_mass,
    }

    if return_diagnostics:
        return probs_dict, logprobs_dict, diagnostics

    return probs_dict, logprobs_dict

# This is a next token distribution method that scores each valid visible continuation string, even if that visible continuation is represented by more than one tokenizer token. This is useful for models like Mistral that may have multi-token representations for certain visible continuations.

def next_token_distribution_sequence(
    prompt: str,
    prefix: str,
    decimals: int,
    allow_negative: bool = True,
    model_name: str = "Qwen/Qwen3-4B",
    load_in_4bit: bool = True,
    prompt_protocol: str = "raw_direct",
    return_diagnostics: bool = False,
):
    """
    Mistral bridge method.

    Scores each valid visible continuation string, even if that visible
    continuation is represented by more than one tokenizer token.

    Example:
        visible "-" may be token piece ["▁-"]
        visible "0" may be token pieces ["▁", "0"]
    """
    allowed = valid_next_tokens(
        prefix=prefix,
        decimals=decimals,
        allow_negative=allow_negative,
    )

    if len(allowed) == 0:
        diagnostics = {
            "candidate_probs_unconditional": {},
            "valid_candidate_mass": None,
            "other_vocab_mass": None,
        }
        if return_diagnostics:
            return {}, {}, diagnostics
        return {}, {}

    tokenizer, model = load_lm(
        model_name=model_name,
        load_in_4bit=load_in_4bit,
    )

    full_text, add_special_tokens = build_lm_context(
        tokenizer=tokenizer,
        prompt=prompt,
        prefix=prefix,
        prompt_protocol=prompt_protocol,
    )

    candidate_logprobs = {}

    for tok in allowed:
        candidate_logprobs[tok] = _candidate_sequence_logprob(
            tokenizer=tokenizer,
            model=model,
            context=full_text,
            candidate=tok,
            add_special_tokens=add_special_tokens,
        )

    probs_dict, logprobs_dict = _normalize_logprobs(candidate_logprobs)

    if return_diagnostics:
        diagnostics = {
            "candidate_probs_unconditional": {},
            "valid_candidate_mass": None,
            "other_vocab_mass": None,
        }
        return probs_dict, logprobs_dict, diagnostics

    return probs_dict, logprobs_dict



def next_token_distribution(
    prompt: str,
    prefix: str,
    decimals: int,
    allow_negative: bool = True,
    model_name: str = "Qwen/Qwen3-4B",
    load_in_4bit: bool = True,
    scoring_method: str = "single_token",
    prompt_protocol: str = "raw_direct",
    return_diagnostics: bool = False,
):
    """
    Public dispatcher.

    scoring_method:
        single_token = original exact restricted next-token method
        sequence     = visible-continuation sequence scoring, useful for Mistral
        auto         = try single_token first, then fallback to sequence
    """
    if scoring_method == "single_token":
        return next_token_distribution_single_token(
            prompt=prompt,
            prefix=prefix,
            decimals=decimals,
            allow_negative=allow_negative,
            model_name=model_name,
            load_in_4bit=load_in_4bit,
            prompt_protocol=prompt_protocol,
            return_diagnostics=return_diagnostics
        )

    if scoring_method == "sequence":
        return next_token_distribution_sequence(
            prompt=prompt,
            prefix=prefix,
            decimals=decimals,
            allow_negative=allow_negative,
            model_name=model_name,
            load_in_4bit=load_in_4bit,
            prompt_protocol=prompt_protocol,
            return_diagnostics=return_diagnostics
        )

    if scoring_method == "auto":
        try:
            return next_token_distribution_single_token(
                prompt=prompt,
                prefix=prefix,
                decimals=decimals,
                allow_negative=allow_negative,
                model_name=model_name,
                load_in_4bit=load_in_4bit,
                prompt_protocol=prompt_protocol,
                return_diagnostics=return_diagnostics
            )
        except ValueError as e:
            print("[scoring] single_token failed, falling back to sequence.")
            print(f"[scoring] reason: {e}")

            return next_token_distribution_sequence(
                prompt=prompt,
                prefix=prefix,
                decimals=decimals,
                allow_negative=allow_negative,
                model_name=model_name,
                load_in_4bit=load_in_4bit,
                prompt_protocol=prompt_protocol,
                return_diagnostics=return_diagnostics
            )

    raise ValueError(
        f"Unknown scoring_method={scoring_method!r}. "
        "Use 'single_token', 'sequence', or 'auto'."
    )

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


