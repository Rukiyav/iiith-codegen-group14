"""Project paths and notebook hyperparameters."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]

MODEL_ID = "Salesforce/codegen-350M-multi"

# Data
MBPP_PATH = ROOT / "data/mbpp/sanitized-mbpp.json"
HUMANEVAL_PATH = ROOT / "data/humaneval/humaneval_test.json"
CODOC_PATH = ROOT / "data/codocbench/codocbench_synthetic.json"
REPO_CORPUS_DIR = ROOT / "data/repo_corpus"
PROCESSED_DIR = ROOT / "data/processed"
MBPP_TRAIN_JSONL = PROCESSED_DIR / "mbpp_train.jsonl"
MBPP_VAL_JSONL = PROCESSED_DIR / "mbpp_validation.jsonl"
APPS_INTRO_JSONL = PROCESSED_DIR / "apps_intro_train.jsonl"

# Models
BASELINE_CACHE = ROOT / "models/baseline"
LORA_DIR = ROOT / "models/lora_finetuned"

# RAG
INDEX_DIR = ROOT / "retrieval/vector_store"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Eval / results
EVAL_RESULTS_DIR = ROOT / "evaluation/results"
EVALPLUS_CACHE_DIR = ROOT / "experiments/evalplus_cache"
HUMANEVAL_PLUS_JSONL = EVALPLUS_CACHE_DIR / "HumanEvalPlus-v0.1.10.jsonl"
MBPP_PLUS_JSONL = EVALPLUS_CACHE_DIR / "MbppPlus-v0.2.0.jsonl"

# LoRA training
LORA_EPOCHS = 3
LORA_BATCH = 4
LORA_LR = 2e-4
LORA_MAX_SAMPLES = 374
LORA_MAX_STEPS = 400
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["qkv_proj", "out_proj", "fc_in", "fc_out"]
LORA_MAX_LENGTH = 512
# Careful mix: mostly MBPP, light APPS intro (truncated prompts)
LORA_APPS_RATIO = 0.20
LORA_APPS_MAX = 120
LORA_APPS_PROMPT_CHARS = 500
LORA_EARLY_STOP_PATIENCE = 3
LORA_VAL_EVERY = 40
LORA_MIN_STEPS = 80

# Eval defaults
EVAL_HELD_OUT_START = 374
EVAL_N = 250
DOC_N = 50
PLUS_MAX_TASKS = 20
PLUS_N_SAMPLES = 5
PLUS_TEMPERATURE = 0.8

REPOS = [
    ("fastapi", "https://github.com/tiangolo/fastapi.git"),
    ("httpx", "https://github.com/encode/httpx.git"),
    ("rich", "https://github.com/Textualize/rich.git"),
    ("pydantic", "https://github.com/pydantic/pydantic.git"),
]


def get_device() -> str:
    """Pick inference/training device. macOS defaults to CPU (MPS + FAISS segfaults)."""
    override = os.environ.get("CODEGEN_DEVICE", "").strip().lower()
    if override in ("cpu", "cuda", "mps"):
        return override
    if torch.cuda.is_available():
        return "cuda"
    if sys.platform == "darwin":
        return "cpu"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_embed_device() -> str:
    """Embeddings always on CPU (FAISS expects numpy on CPU)."""
    return "cpu"


def on_macos() -> bool:
    return sys.platform == "darwin"


def ensure_dirs() -> None:
    for path in [
        MBPP_PATH.parent,
        HUMANEVAL_PATH.parent,
        CODOC_PATH.parent,
        REPO_CORPUS_DIR,
        PROCESSED_DIR,
        BASELINE_CACHE,
        LORA_DIR,
        INDEX_DIR,
        INDEX_DIR / "projects",
        EVAL_RESULTS_DIR,
        EVALPLUS_CACHE_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)
