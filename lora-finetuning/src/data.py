"""
Dataset loading and preprocessing for instruction fine-tuning.
"""

from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer


def load_alpaca_dataset(
    tokenizer: PreTrainedTokenizer,
    max_length: int = 512,
    split: str = "train",
) -> Dataset:
    """Load and preprocess the Alpaca instruction-tuning dataset.

    Args:
        tokenizer: HuggingFace tokenizer.
        max_length: Maximum sequence length.
        split: Dataset split ('train' or 'test').

    Returns:
        Tokenized HuggingFace Dataset.
    """
    dataset = load_dataset("tatsu-lab/alpaca", split=split)

    def format_prompt(example: dict) -> dict:
        """Format instruction-input-output into a single prompt."""
        instruction = example["instruction"]
        inp = example.get("input", "")
        output = example["output"]

        if inp:
            prompt = (
                f"Below is an instruction that describes a task, "
                f"paired with an input that provides further context. "
                f"Write a response that appropriately completes the request.\n\n"
                f"### Instruction:\n{instruction}\n\n"
                f"### Input:\n{inp}\n\n"
                f"### Response:\n{output}"
            )
        else:
            prompt = (
                f"Below is an instruction that describes a task. "
                f"Write a response that appropriately completes the request.\n\n"
                f"### Instruction:\n{instruction}\n\n"
                f"### Response:\n{output}"
            )
        return {"text": prompt}

    dataset = dataset.map(format_prompt)

    def tokenize(example: dict) -> dict:
        result = tokenizer(
            example["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )
        result["labels"] = result["input_ids"].copy()
        return result

    dataset = dataset.map(tokenize, remove_columns=dataset.column_names)
    return dataset


def load_custom_dataset(
    data_path: str,
    tokenizer: PreTrainedTokenizer,
    max_length: int = 512,
) -> Dataset:
    """Load a custom JSON/JSONL dataset.

    Expected format: list of {"instruction": ..., "input": ..., "output": ...}
    or {"text": "..."} for pre-formatted text.
    """
    import json
    from pathlib import Path

    with open(data_path, encoding="utf-8") as f:
        if data_path.endswith(".jsonl"):
            data = [json.loads(line) for line in f]
        else:
            data = json.load(f)

    dataset = Dataset.from_list(data)

    if "text" not in dataset.column_names:
        # Format from instruction/input/output fields
        def format_prompt(example):
            return {"text": _alpaca_format(example)}

        dataset = dataset.map(format_prompt)
        dataset = dataset.remove_columns(
            [c for c in dataset.column_names if c != "text"]
        )

    def tokenize(example):
        result = tokenizer(
            example["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )
        result["labels"] = result["input_ids"].copy()
        return result

    dataset = dataset.map(tokenize)
    return dataset


def _alpaca_format(example: dict) -> str:
    instruction = example["instruction"]
    inp = example.get("input", "")
    output = example.get("output", "")
    if inp:
        return (
            f"### Instruction:\n{instruction}\n\n"
            f"### Input:\n{inp}\n\n"
            f"### Response:\n{output}"
        )
    return (
        f"### Instruction:\n{instruction}\n\n"
        f"### Response:\n{output}"
    )
