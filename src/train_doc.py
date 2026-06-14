"""Fine-tune CodeGen on CoDocBench for documentation generation."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Optional

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    default_data_collator,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.codoc import PROCESSED_DIR, download_codoc
from src.prompts import doc_prompt, doc_training_text
from src.training_utils import (
    build_sft_tokenized_example,
    has_trainable_labels,
    truncate_text_to_tokens,
)

logger = logging.getLogger(__name__)
DEFAULT_MODEL = "Salesforce/codegen-350M-multi"


def ensure_codoc_dataset() -> None:
    if not (PROCESSED_DIR / "codoc_train.jsonl").exists():
        logger.info("CoDocBench processed data missing; preparing...")
        download_codoc()


def load_codoc_datasets(
    max_train_samples: Optional[int] = 80,
    max_eval_samples: Optional[int] = 16,
    full: bool = False,
):
    ensure_codoc_dataset()
    files = {}
    for split in ["train", "validation"]:
        path = PROCESSED_DIR / f"codoc_{split}.jsonl"
        if path.exists():
            files[split] = str(path)
    dataset = load_dataset("json", data_files=files)
    if not full and max_train_samples:
        dataset["train"] = dataset["train"].select(range(min(max_train_samples, len(dataset["train"]))))
    if not full and max_eval_samples and "validation" in dataset:
        dataset["validation"] = dataset["validation"].select(
            range(min(max_eval_samples, len(dataset["validation"])))
        )
    return dataset


def tokenize_function(example: Dict, tokenizer, max_length: int):
    code = truncate_text_to_tokens(
        example.get("code", ""),
        tokenizer,
        max_tokens=max(64, max_length - 160),
    )
    doc = example.get("documentation", "")
    input_text = doc_training_text(code, doc)
    prompt_text = doc_prompt(code)
    return build_sft_tokenized_example(tokenizer, input_text, prompt_text, max_length)


def prepare_dataset(dataset, tokenizer, max_length: int):
    tokenized = dataset.map(
        lambda ex: tokenize_function(ex, tokenizer, max_length),
        remove_columns=dataset.column_names,
    )
    return tokenized.filter(has_trainable_labels)


def train(args: argparse.Namespace) -> None:
    datasets = load_codoc_datasets(args.max_train_samples, args.max_eval_samples, full=args.full)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_dtype = torch.float16 if args.fp16 and torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, dtype=load_dtype)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    train_dataset = prepare_dataset(datasets["train"], tokenizer, args.max_length)
    eval_dataset = (
        prepare_dataset(datasets["validation"], tokenizer, args.max_length)
        if "validation" in datasets
        else None
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        learning_rate=args.learning_rate,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        fp16=args.fp16 and torch.cuda.is_available(),
        logging_steps=args.logging_steps,
        eval_strategy=args.evaluation_strategy if eval_dataset else "no",
        save_strategy=args.save_strategy,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=args.load_best_model_at_end and eval_dataset is not None,
        metric_for_best_model="eval_loss" if eval_dataset else None,
        greater_is_better=False,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=default_data_collator,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune CodeGen on CoDocBench.")
    parser.add_argument("--model_name_or_path", default=DEFAULT_MODEL)
    parser.add_argument("--output_dir", default="experiments/checkpoints/codegen-doc-cp1")
    parser.add_argument("--full", action="store_true", help="Train on full CoDocBench train split (CP2)")
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--max_train_samples", type=int, default=80)
    parser.add_argument("--max_eval_samples", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--evaluation_strategy", default="epoch", choices=["no", "steps", "epoch"])
    parser.add_argument("--save_strategy", default="epoch", choices=["no", "steps", "epoch"])
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument(
        "--load_best_model_at_end",
        action="store_true",
        help="Keep best checkpoint by eval loss (requires validation split)",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    train(parse_args())


if __name__ == "__main__":
    main()
