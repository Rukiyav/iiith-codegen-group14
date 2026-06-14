"""Shared causal-LM fine-tuning tokenization helpers."""

from __future__ import annotations

from typing import Any, Dict


def truncate_text_to_tokens(text: str, tokenizer, max_tokens: int) -> str:
    """Truncate without encoding the full string (avoids >2048 tokenizer warnings)."""
    encoded = tokenizer(
        text,
        truncation=True,
        max_length=max(1, max_tokens),
        add_special_tokens=False,
    )
    ids = encoded["input_ids"]
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return tokenizer.decode(ids, skip_special_tokens=True)


def build_sft_tokenized_example(
    tokenizer,
    input_text: str,
    prompt_text: str,
    max_length: int,
) -> Dict[str, Any]:
    """Tokenize one SFT example; mask prompt tokens in labels."""
    tokenized = tokenizer(
        input_text,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_attention_mask=True,
    )
    seq_len = len(tokenized["input_ids"])
    prompt_ids = tokenizer(
        prompt_text,
        add_special_tokens=False,
        truncation=True,
        max_length=seq_len,
    )["input_ids"]
    prompt_len = min(len(prompt_ids), seq_len)

    labels = list(tokenized["input_ids"])
    labels[:prompt_len] = [-100] * prompt_len
    pad_id = tokenizer.pad_token_id
    for i in range(seq_len):
        if tokenized["attention_mask"][i] == 0 or labels[i] == pad_id:
            labels[i] = -100

    tokenized["labels"] = labels
    return tokenized


def has_trainable_labels(example: Dict[str, Any]) -> bool:
    return any(label != -100 for label in example["labels"])
