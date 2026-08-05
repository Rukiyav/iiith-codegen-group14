"""Format retrieved train examples into a few-shot context block for prompts."""

from __future__ import annotations

from typing import Dict, List

from src.generation.doc_postprocess import documentation_to_comments
from src.prompts import code_prompt, doc_prompt


def build_code_context(examples: List[Dict]) -> str:
    """Few-shot block of (NL prompt -> code) pairs, formatted like the training prompt."""
    parts = []
    for ex in examples:
        code = (ex.get("code") or "").strip()
        if not code:
            continue
        parts.append(code_prompt(ex.get("prompt", "")) + code + "\n\n")
    return "".join(parts)


def build_doc_context(examples: List[Dict]) -> str:
    """Few-shot block of (code -> documentation) pairs, formatted like the training prompt."""
    parts = []
    for ex in examples:
        code = (ex.get("code") or "").strip()
        doc = ex.get("documentation") or ""
        if not code or not doc:
            continue
        parts.append(doc_prompt(code) + documentation_to_comments(doc) + "\n\n")
    return "".join(parts)
