"""
Model loading and LoRA injection utilities.
"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from lora import LoRAConfig, inject_lora, count_trainable_params


def load_model_and_tokenizer(
    model_name: str = "google/gemma-2b",
    load_in_8bit: bool = False,
    load_in_4bit: bool = False,
    device_map: str = "auto",
) -> tuple:
    """Load a base model and tokenizer with optional quantization.

    Args:
        model_name: HuggingFace model ID.
        load_in_8bit: Use 8-bit quantization (requires bitsandbytes).
        load_in_4bit: Use 4-bit quantization (requires bitsandbytes).
        device_map: Device mapping strategy.

    Returns:
        (model, tokenizer) tuple.
    """
    quantization_config = None
    if load_in_8bit:
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    elif load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map=device_map,
        torch_dtype=torch.bfloat16 if not quantization_config else None,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def prepare_model_for_lora(
    model,
    lora_config: LoRAConfig,
    target_modules: list[str] | None = None,
) -> None:
    """Inject LoRA layers into the model and freeze base params.

    Args:
        model: Pretrained model from transformers.
        lora_config: LoRA configuration.
        target_modules: Which modules to adapt (default: ["q_proj", "v_proj"]).
    """
    # Freeze all parameters first
    for param in model.parameters():
        param.requires_grad = False

    # Inject LoRA (sets up trainable A, B matrices)
    inject_lora(model, lora_config, target_modules)

    # Print statistics
    trainable, total = count_trainable_params(model)
    print(f"Trainable: {trainable:,} / {total:,} "
          f"({100 * trainable / total:.2f}%)")
