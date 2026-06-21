from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gc
import torch
from transformers import AutoTokenizer, AutoProcessor, AutoModelForCausalLM, BitsAndBytesConfig

try:
    from transformers import AutoModelForImageTextToText
except ImportError:
    AutoModelForImageTextToText = None
try:
    from transformers import AutoModelForMultimodalLM
except ImportError:
    AutoModelForMultimodalLM = None

@dataclass
class LoadedLM:
    tokenizer: Any
    model: Any
    processor: Any | None
    backend: str


def get_model_input_device(model) -> torch.device:

    for p in model.parameters():
        if p.device.type != "meta":
            return p.device

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _make_quantization_config(load_in_4bit: bool):
    if not load_in_4bit:
        return None

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )


def _load_tokenizer_or_processor(model_name: str):

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
        return tokenizer, None
    except Exception as tokenizer_error:
        print(f"[loader] AutoTokenizer failed for {model_name}")
        print(f"[loader] tokenizer error: {type(tokenizer_error).__name__}: {tokenizer_error}")

    processor = AutoProcessor.from_pretrained(
        model_name,
        trust_remote_code=True,
    )

    if not hasattr(processor, "tokenizer"):
        raise RuntimeError(
            f"AutoProcessor loaded for {model_name}, but processor.tokenizer is missing."
        )

    return processor.tokenizer, processor


def load_lm_backend(
    model_name: str,
    load_in_4bit: bool = True,
) -> LoadedLM:

    tokenizer, processor = _load_tokenizer_or_processor(model_name)

    quant_config = _make_quantization_config(load_in_4bit)

    common_kwargs = {
        "device_map": "auto",
        "trust_remote_code": True,
    }

    if quant_config is not None:
        common_kwargs["quantization_config"] = quant_config
    else:
        common_kwargs["torch_dtype"] = torch.float16

    # First try the normal text-generation path.
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **common_kwargs,
        )
        model.eval()

        return LoadedLM(
            tokenizer=tokenizer,
            model=model,
            processor=processor,
            backend="causal_lm",
        )

    except Exception as causal_error:
        print(f"[loader] AutoModelForCausalLM failed for {model_name}")
        print(f"[loader] causal error: {type(causal_error).__name__}: {causal_error}")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Prepare processor for multimodal/image-text fallback models.
    if processor is None:
        processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True,
        )

        if hasattr(processor, "tokenizer"):
            tokenizer = processor.tokenizer

    # Fallback 1: any-to-any / multimodal LM.
    # This is the important fallback for models such as Gemma 4 12B-it.
    if AutoModelForMultimodalLM is not None:
        try:
            model = AutoModelForMultimodalLM.from_pretrained(
                model_name,
                **common_kwargs,
            )
            model.eval()

            return LoadedLM(
                tokenizer=tokenizer,
                model=model,
                processor=processor,
                backend="multimodal_lm",
            )

        except Exception as multimodal_error:
            print(f"[loader] AutoModelForMultimodalLM failed for {model_name}")
            print(f"[loader] multimodal error: {type(multimodal_error).__name__}: {multimodal_error}")

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Fallback 2: image-text-to-text models.
    if AutoModelForImageTextToText is not None:
        try:
            model = AutoModelForImageTextToText.from_pretrained(
                model_name,
                **common_kwargs,
            )
            model.eval()

            return LoadedLM(
                tokenizer=tokenizer,
                model=model,
                processor=processor,
                backend="image_text_to_text",
            )

        except Exception as image_text_error:
            print(f"[loader] AutoModelForImageTextToText failed for {model_name}")
            print(f"[loader] image-text error: {type(image_text_error).__name__}: {image_text_error}")

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    raise RuntimeError(
        f"Could not load {model_name} using causal_lm, multimodal_lm, or image_text_to_text backends."
    )    


    