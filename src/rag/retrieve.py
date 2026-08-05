"""Task-specific retrievers built on the generic TF-IDF index (src/rag/index.py)."""

from __future__ import annotations

from functools import lru_cache
from typing import Dict, List

from src.data.codoc import load_codoc_split
from src.data.mbpp import load_processed
from src.rag.index import RetrievalIndex, build_index


@lru_cache(maxsize=4)
def get_mbpp_retriever(split: str = "train") -> RetrievalIndex:
    examples = load_processed(split)
    return build_index(examples, text_fn=lambda ex: ex.get("prompt", ""))


@lru_cache(maxsize=4)
def get_codoc_retriever(split: str = "train") -> RetrievalIndex:
    examples = load_codoc_split(split)
    return build_index(examples, text_fn=lambda ex: ex.get("code", ""))


def retrieve_code_examples(nl_prompt: str, k: int = 2, split: str = "train") -> List[Dict]:
    """Top-k MBPP train examples whose NL prompt is closest to nl_prompt."""
    return get_mbpp_retriever(split).query(nl_prompt, k=k)


def retrieve_doc_examples(code: str, k: int = 2, split: str = "train") -> List[Dict]:
    """Top-k CoDocBench train examples whose code is closest to the query code."""
    return get_codoc_retriever(split).query(code, k=k)
