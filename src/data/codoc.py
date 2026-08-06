"""CoDocBench — official train/test JSONL from kunpai/codocbench GitHub."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import requests

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
CODOC_DIR = RAW_DIR / "codocbench"
CODOC_TRAIN_URL = "https://raw.githubusercontent.com/kunpai/codocbench/main/dataset/train.jsonl"
CODOC_TEST_URL = "https://raw.githubusercontent.com/kunpai/codocbench/main/dataset/test.jsonl"


def ensure_dirs() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    CODOC_DIR.mkdir(parents=True, exist_ok=True)


def _download_jsonl(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        return
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    dest.write_text(response.text, encoding="utf-8")


def _extract_pair(row: Dict) -> Optional[Dict]:
    versions = row.get("version_data") or []
    if not versions:
        return None
    latest = versions[-1]
    code = (latest.get("code") or "").strip()
    doc = (latest.get("docstring") or "").strip()
    if not code or not doc:
        return None
    func = row.get("function") or row.get("file") or "unknown"
    project = row.get("project") or row.get("owner") or "unknown"
    return {
        "id": f"{project}/{func}",
        "code": code,
        "documentation": doc,
        "project": project,
        "function": func,
    }


def _jsonl_to_processed(src: Path, dest: Path) -> int:
    rows: List[Dict] = []
    with src.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            pair = _extract_pair(json.loads(line))
            if pair:
                rows.append(pair)
    with dest.open("w", encoding="utf-8") as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows)


def download_codoc(force_download: bool = False) -> Path:
    """Download official CoDocBench train/test and write processed JSONL."""
    ensure_dirs()
    train_raw = CODOC_DIR / "train.jsonl"
    test_raw = CODOC_DIR / "test.jsonl"
    if force_download:
        train_raw.unlink(missing_ok=True)
        test_raw.unlink(missing_ok=True)

    _download_jsonl(CODOC_TRAIN_URL, train_raw)
    _download_jsonl(CODOC_TEST_URL, test_raw)

    n_train = _jsonl_to_processed(train_raw, PROCESSED_DIR / "codoc_train.jsonl")
    n_test = _jsonl_to_processed(test_raw, PROCESSED_DIR / "codoc_test.jsonl")

    # 10% of train as validation for early stopping
    train_rows = []
    with (PROCESSED_DIR / "codoc_train.jsonl").open("r", encoding="utf-8") as f:
        for line in f:
            train_rows.append(json.loads(line))
    val_size = max(1, int(len(train_rows) * 0.1))
    val_rows = train_rows[:val_size]
    train_rows = train_rows[val_size:]
    with (PROCESSED_DIR / "codoc_validation.jsonl").open("w", encoding="utf-8") as out:
        for row in val_rows:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (PROCESSED_DIR / "codoc_train.jsonl").open("w", encoding="utf-8") as out:
        for row in train_rows:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")

    return CODOC_DIR


def load_codoc_split(split: str = "test") -> List[Dict]:
    path = PROCESSED_DIR / f"codoc_{split}.jsonl"
    if not path.exists():
        download_codoc()
    examples: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            examples.append(json.loads(line))
    return examples
