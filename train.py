"""Train pipeline — dataset ingestion & LoRA fine-tuning for CodeGen-350M."""

from __future__ import annotations

import argparse
import gzip
import json
import os
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional

import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, get_cosine_schedule_with_warmup

from src.config import (
    BASELINE_CACHE,
    CODOC_PATH,
    HUMANEVAL_PATH,
    LORA_ALPHA,
    LORA_APPS_MAX,
    LORA_APPS_RATIO,
    LORA_BATCH,
    LORA_DIR,
    LORA_DROPOUT,
    LORA_EARLY_STOP_PATIENCE,
    LORA_EPOCHS,
    LORA_LR,
    LORA_MAX_LENGTH,
    LORA_MAX_SAMPLES,
    LORA_MAX_STEPS,
    LORA_MIN_STEPS,
    LORA_R,
    LORA_TARGET_MODULES,
    LORA_VAL_EVERY,
    MBPP_PATH,
    MODEL_ID,
    ensure_dirs,
    get_device,
)
from src.engine import load_tokenizer

# -----------------------------------------------------------------------------
# 1. Dataset Downloads & Preparation
# -----------------------------------------------------------------------------

def download_mbpp(force: bool = False) -> Path:
    ensure_dirs()
    if MBPP_PATH.exists() and not force:
        return MBPP_PATH
    url = "https://huggingface.co/datasets/Muennighoff/mbpp/resolve/main/data/sanitized-mbpp.json"
    urllib.request.urlretrieve(url, MBPP_PATH)
    return MBPP_PATH


def load_mbpp() -> List[dict]:
    download_mbpp()
    with open(MBPP_PATH) as f:
        return json.load(f)


def format_sft_text(prompt: str, code: str) -> str:
    return f'"""\n{prompt.strip()}\n"""\n{code.rstrip()}\n'


# -----------------------------------------------------------------------------
# 2. PyTorch Dataset & DataLoader
# -----------------------------------------------------------------------------

class CodeSFTDataset(Dataset):
    def __init__(self, examples: List[dict], tokenizer, max_length: int = LORA_MAX_LENGTH):
        self.tokenizer = tokenizer
        self.tokenizer.truncation_side = "left"
        self.max_length = max_length
        self.items = []
        for ex in examples:
            prompt = ex.get("prompt", "")
            code = ex.get("code", "")
            if prompt and code:
                prefix = f'"""\n{prompt.strip()}\n"""\n'
                full = format_sft_text(prompt, code)
                self.items.append((prefix, full))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        prefix, full = self.items[idx]
        enc = self.tokenizer(full, truncation=True, max_length=self.max_length, padding="max_length", return_tensors="pt")
        ids = enc["input_ids"].squeeze(0)
        mask = enc["attention_mask"].squeeze(0)
        labels = ids.clone()
        prefix_ids = self.tokenizer(prefix, add_special_tokens=False)["input_ids"]
        n_prefix = min(len(prefix_ids), int(mask.sum().item()))
        labels[:n_prefix] = -100
        labels[mask == 0] = -100
        return {"input_ids": ids, "attention_mask": mask, "labels": labels}


def collate(batch):
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch]),
    }


# -----------------------------------------------------------------------------
# 3. LoRA Training Loop
# -----------------------------------------------------------------------------

def train_lora(
    max_samples: int = LORA_MAX_SAMPLES,
    max_steps: int = LORA_MAX_STEPS,
    epochs: int = LORA_EPOCHS,
    batch_size: int = LORA_BATCH,
    lr: float = LORA_LR,
    max_length: int = LORA_MAX_LENGTH,
    output_dir=None,
) -> str:
    ensure_dirs()
    output_dir = Path(output_dir or LORA_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = get_device()
    dtype = torch.float16 if device == "cuda" else torch.float32

    mbpp_all = load_mbpp()
    train_ex = mbpp_all[:max_samples]
    val_ex = mbpp_all[max_samples:]
    print(f"Loaded MBPP SFT: {len(train_ex)} train, {len(val_ex)} validation")

    tokenizer = load_tokenizer()
    train_data = CodeSFTDataset(train_ex, tokenizer, max_length=max_length)
    val_data = CodeSFTDataset(val_ex, tokenizer, max_length=max_length)

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        inference_mode=False,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        cache_dir=str(BASELINE_CACHE),
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate, drop_last=True)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = min(max_steps, max(1, len(train_loader) * epochs))
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=max(1, int(total_steps * 0.05)), num_training_steps=total_steps)

    model = model.to(device)
    model.train()
    global_step = 0

    for epoch in range(epochs):
        if global_step >= max_steps:
            break
        for batch in train_loader:
            if global_step >= max_steps:
                break
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1
            if global_step % 20 == 0:
                print(f"Step {global_step}/{total_steps} - loss: {loss.item():.4f}")

    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"Saved LoRA checkpoint to {output_dir}")
    return str(output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=LORA_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=LORA_BATCH)
    args = parser.parse_args()
    train_lora(epochs=args.epochs, batch_size=args.batch_size)
