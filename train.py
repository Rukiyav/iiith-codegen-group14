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
    EVAL_HELD_OUT_START,
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
        self.max_length = max_length
        self.items = []
        for ex in examples:
            prompt = ex.get("prompt", "")
            code = ex.get("code", "")
            if prompt and code:
                prefix = f'"""\n{prompt.strip()}\n"""\n'
                self.items.append((prefix, code))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        prefix, code = self.items[idx]

        prefix_ids = self.tokenizer(prefix, add_special_tokens=False)["input_ids"]
        code_ids = self.tokenizer(code, add_special_tokens=False)["input_ids"]

        # Avoid left truncation issues by manually truncating the prefix first if total exceeds max_length
        total_len = len(prefix_ids) + len(code_ids)
        if total_len > self.max_length:
            allowed_prefix_len = max(0, self.max_length - len(code_ids))
            prefix_ids = prefix_ids[-allowed_prefix_len:]
            if len(code_ids) > self.max_length:
                code_ids = code_ids[:self.max_length]
                prefix_ids = []

        input_ids = prefix_ids + code_ids
        attention_mask = [1] * len(input_ids)
        labels = [-100] * len(prefix_ids) + code_ids

        # Manual right padding
        pad_len = self.max_length - len(input_ids)
        if pad_len > 0:
            input_ids = input_ids + [self.tokenizer.pad_token_id] * pad_len
            attention_mask = attention_mask + [0] * pad_len
            labels = labels + [-100] * pad_len

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def collate(batch):
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch]),
    }


# -----------------------------------------------------------------------------
# 3. LoRA Training Loop
# -----------------------------------------------------------------------------

def set_seed(seed: int = 42):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_lora(
    max_samples: int = LORA_MAX_SAMPLES,
    max_steps: int = LORA_MAX_STEPS,
    epochs: int = LORA_EPOCHS,
    batch_size: int = LORA_BATCH,
    lr: float = LORA_LR,
    max_length: int = LORA_MAX_LENGTH,
    output_dir=None,
) -> str:
    set_seed(42)
    ensure_dirs()
    output_dir = Path(output_dir or LORA_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = get_device()
    dtype = torch.float16 if device == "cuda" else torch.float32

    mbpp_all = load_mbpp()
    # Official MBPP split has task_ids: 1-374 for SFT train, 374-500 validation/test.
    # val_ex must stop before EVAL_HELD_OUT_START, or early stopping / checkpoint
    # selection would be driven by the same examples eval.py later reports as the
    # "held-out" benchmark (comparison_mbpp.json), leaking eval data into training
    # decisions and inflating the lora numbers.
    if max_samples >= EVAL_HELD_OUT_START:
        # Auto-split the 374 SFT range (e.g. 85% train, 15% validation) to ensure zero overlap with held-out eval
        train_limit = int(EVAL_HELD_OUT_START * 0.85) # ~317 samples
        train_ex = mbpp_all[:train_limit]
        val_ex = mbpp_all[train_limit:EVAL_HELD_OUT_START]
    else:
        train_ex = mbpp_all[:max_samples]
        val_ex = mbpp_all[max_samples:EVAL_HELD_OUT_START]

    from src.config import (
        APPS_INTRO_JSONL, APPS_INTERVIEW_JSONL, LORA_APPS_RATIO,
        LORA_APPS_MAX, LORA_APPS_INTERVIEW_MAX
    )
    import random
    
    # 1. Load APPS Intro
    apps_intro = []
    if APPS_INTRO_JSONL.exists():
        with open(APPS_INTRO_JSONL) as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    code = item.get("code") or item.get("canonical_solution") or ""
                    prompt = item.get("prompt", "")
                    if prompt and code:
                        from src.config import LORA_APPS_PROMPT_CHARS
                        truncated_prompt = prompt[:LORA_APPS_PROMPT_CHARS]
                        apps_intro.append({
                            "prompt": truncated_prompt,
                            "code": code
                        })
        rng = random.Random(42)
        rng.shuffle(apps_intro)
        
    # 2. Load APPS Interview
    apps_interview = []
    if APPS_INTERVIEW_JSONL.exists():
        with open(APPS_INTERVIEW_JSONL) as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    code = item.get("code") or item.get("canonical_solution") or ""
                    prompt = item.get("prompt", "")
                    if prompt and code:
                        from src.config import LORA_APPS_PROMPT_CHARS
                        truncated_prompt = prompt[:LORA_APPS_PROMPT_CHARS]
                        apps_interview.append({
                            "prompt": truncated_prompt,
                            "code": code
                        })
        rng = random.Random(42)
        rng.shuffle(apps_interview)

    # 3. Mix Intro and Interview samples based on target ratio and caps
    target_apps_count = int(len(train_ex) * LORA_APPS_RATIO / (1.0 - LORA_APPS_RATIO))
    selected_intro = apps_intro[:min(LORA_APPS_MAX, len(apps_intro))]
    selected_interview = apps_interview[:min(LORA_APPS_INTERVIEW_MAX, len(apps_interview))]
    
    mixed_apps = selected_intro + selected_interview
    if len(mixed_apps) > target_apps_count:
        ratio_intro = len(selected_intro) / max(1, len(mixed_apps))
        intro_budget = int(target_apps_count * ratio_intro)
        interview_budget = target_apps_count - intro_budget
        
        selected_intro = selected_intro[:intro_budget]
        selected_interview = selected_interview[:interview_budget]
        mixed_apps = selected_intro + selected_interview

    if mixed_apps:
        train_ex = train_ex + mixed_apps
        rng = random.Random(42)
        rng.shuffle(train_ex)
        print(f"Mixed in {len(selected_intro)} APPS Intro and {len(selected_interview)} APPS Interview samples (target ratio: {LORA_APPS_RATIO:.2f})")
    else:
        print("No APPS introductory or interview examples found to mix in.")

    print(
        f"Loaded training SFT: {len(train_ex)} train (including mixed data), {len(val_ex)} validation "
        f"(eval.py benchmark starts at {EVAL_HELD_OUT_START}, left untouched here)"
    )

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
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, collate_fn=collate)

    from src.config import LORA_VAL_EVERY, LORA_EARLY_STOP_PATIENCE, LORA_MIN_STEPS

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = min(max_steps, max(1, len(train_loader) * epochs))
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=max(1, int(total_steps * 0.05)), num_training_steps=total_steps)

    print(f"Moving model to {device} and starting training loop for {epochs} epochs ({total_steps} total steps)...")
    model = model.to(device)
    model.train()
    global_step = 0
    best_val_loss = float("inf")
    patience_counter = 0
    early_stopped = False

    for epoch in range(epochs):
        if global_step >= max_steps or early_stopped:
            break
        for batch in train_loader:
            if global_step >= max_steps or early_stopped:
                break
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1
            if global_step % 5 == 0 or global_step == 1:
                print(f"Step {global_step}/{total_steps} - loss: {loss.item():.4f}")

            # Validation loop and Early Stopping
            if global_step >= LORA_MIN_STEPS and global_step % LORA_VAL_EVERY == 0:
                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for val_batch in val_loader:
                        val_batch = {k: v.to(device) for k, v in val_batch.items()}
                        outputs = model(**val_batch)
                        val_loss += outputs.loss.item()
                val_loss /= max(1, len(val_loader))
                print(f"Step {global_step} - Validation loss: {val_loss:.4f} (best: {best_val_loss:.4f})")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    model.save_pretrained(str(output_dir))
                    tokenizer.save_pretrained(str(output_dir))
                    print(f"Saved new best checkpoint at step {global_step}")
                else:
                    patience_counter += 1
                    print(f"Validation did not improve. Patience: {patience_counter}/{LORA_EARLY_STOP_PATIENCE}")
                    if patience_counter >= LORA_EARLY_STOP_PATIENCE:
                        print(f"Early stopping triggered at step {global_step}")
                        early_stopped = True
                        break
                model.train()

    # Save final model if best validation checkpoint was never saved
    if not (output_dir / "adapter_config.json").exists():
        model.save_pretrained(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))
        print(f"Saved final LoRA model checkpoint to {output_dir}")

    return str(output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=LORA_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=LORA_BATCH)
    args = parser.parse_args()
    train_lora(epochs=args.epochs, batch_size=args.batch_size)
