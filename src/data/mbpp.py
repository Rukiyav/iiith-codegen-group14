"""MBPP dataset — matches capstone_project_checkpoint_1.ipynb logic."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Union

import requests

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
MBPP_DIR = RAW_DIR / "mbpp"
MBPP_SANITIZED_URL = (
    "https://huggingface.co/datasets/Muennighoff/mbpp/resolve/main/data/sanitized-mbpp.json"
)
SPLIT_SEED = 42


def ensure_dirs() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    MBPP_DIR.mkdir(parents=True, exist_ok=True)


def fetch_sanitized_mbpp(url: str = MBPP_SANITIZED_URL, timeout: int = 120) -> List[Dict]:
    """Download sanitized MBPP JSON (notebook: requests.get + json.loads)."""
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    return json.loads(response.text)


def load_mbpp_raw(force_download: bool = False) -> List[Dict]:
    """Return the full 427-task sanitized MBPP list (notebook: mbpp_dataset)."""
    ensure_dirs()
    raw_json = MBPP_DIR / "sanitized-mbpp.json"
    if force_download or not raw_json.exists():
        data = fetch_sanitized_mbpp()
        with raw_json.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    else:
        with raw_json.open("r", encoding="utf-8") as f:
            data = json.load(f)
    return data


def load_mbpp_dataframe(force_download: bool = False):
    """Notebook: pd.DataFrame(mbpp_dataset)."""
    import pandas as pd

    return pd.DataFrame(load_mbpp_raw(force_download))


def _normalize_example(example: Dict) -> Dict:
    """Map notebook fields (code) to internal training fields."""
    code = example.get("code") or example.get("canonical_solution") or example.get("output") or ""
    prompt = example.get("prompt") or example.get("text") or example.get("input") or ""
    test_list = example.get("test_list") or []
    if not isinstance(test_list, list):
        test_list = []
    return {
        "id": example.get("task_id") or example.get("id"),
        "task_id": example.get("task_id") or example.get("id"),
        "source_file": example.get("source_file"),
        "prompt": prompt,
        "code": code,
        "canonical_solution": code,
        "test_list": test_list,
        "test_template": "\n".join(test_list),
    }


def _write_jsonl(path: Path, rows: List[Dict]) -> None:
    with path.open("w", encoding="utf-8") as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")


def preprocess_mbpp(raw: Optional[Union[Path, List[Dict]]] = None, processed_dir: Optional[Path] = None) -> Path:
    """Write processed JSONL splits from raw list or JSON file."""
    ensure_dirs()
    processed_dir = processed_dir or PROCESSED_DIR
    if raw is None:
        examples = load_mbpp_raw()
    elif isinstance(raw, list):
        examples = raw
    elif isinstance(raw, Path) and raw.suffix == ".json":
        with raw.open("r", encoding="utf-8") as f:
            examples = json.load(f)
    else:
        raise ValueError(f"Unsupported raw MBPP input: {raw!r}")

    normalized = [_normalize_example(ex) for ex in examples]
    _write_jsonl(processed_dir / "mbpp_all.jsonl", normalized)

    n = len(normalized)
    train_end = int(n * 0.8)
    val_end = int(n * 0.9)
    for split, rows in {
        "train": normalized[:train_end],
        "validation": normalized[train_end:val_end],
        "test": normalized[val_end:],
    }.items():
        _write_jsonl(processed_dir / f"mbpp_{split}.jsonl", rows)
    return processed_dir


def download_mbpp(force_download: bool = False) -> Path:
    """Download + preprocess MBPP (notebook-compatible entrypoint)."""
    raw = load_mbpp_raw(force_download=force_download)
    preprocess_mbpp(raw)
    return MBPP_DIR / "sanitized-mbpp.json"


def load_processed(split: str = "train") -> List[Dict]:
    """
    Load MBPP examples.
    split='all' returns full 427 tasks (notebook eval loop).
    """
    if split == "all":
        return [_normalize_example(ex) for ex in load_mbpp_raw()]

    processed_file = PROCESSED_DIR / f"mbpp_{split}.jsonl"
    if not processed_file.exists():
        download_mbpp()
    examples: List[Dict] = []
    with processed_file.open("r", encoding="utf-8") as f:
        for line in f:
            examples.append(json.loads(line))
    return examples
