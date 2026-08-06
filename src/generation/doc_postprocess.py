"""Post-process and fallback helpers for documentation generation."""

from __future__ import annotations

import ast
import re
from typing import List, Optional

_CODE_START = re.compile(r"^\s*(def |class |import |from |@)", re.MULTILINE)
_URL = re.compile(r"https?://\S+")
_LEETCODE_MARKERS = ("# Companies", "# Related Topics", "# Similar Questions", "class Solution")


def documentation_to_comments(doc: str) -> str:
    """Convert CoDocBench-style docstrings into # comment lines for training."""
    text = doc.strip()
    if text.startswith('"""') or text.startswith("'''"):
        text = text[3:]
        if text.endswith('"""') or text.endswith("'''"):
            text = text[:-3]
    lines = []
    for line in text.strip().splitlines():
        stripped = line.strip()
        if not stripped:
            lines.append("#")
        elif stripped.startswith("#"):
            lines.append(stripped)
        else:
            lines.append(f"# {stripped}")
    return "\n".join(lines)


def clean_generated_documentation(text: str) -> str:
    """Strip code continuations, URLs, and LeetCode boilerplate from model output."""
    if not text:
        return ""

    cut_markers = [
        "\nclass ",
        "\ndef ",
        "\nimport ",
        "\nfrom ",
        "\nhttp://",
        "\nhttps://",
        *_LEETCODE_MARKERS,
    ]
    cleaned = text
    for marker in cut_markers:
        idx = cleaned.find(marker)
        if idx != -1:
            cleaned = cleaned[:idx]

    cleaned = _URL.sub("", cleaned)
    lines = []
    for line in cleaned.splitlines():
        if _CODE_START.match(line):
            break
        stripped = line.strip()
        if stripped.startswith('"""') or stripped.startswith("'''"):
            break
        lines.append(line.rstrip())
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines).strip()


def _looks_like_code_spill(text: str) -> bool:
    if not text.strip():
        return True
    if _URL.search(text):
        return True
    if "def " in text or "class " in text:
        return True
    if "stackoverflow.com" in text.lower():
        return True
    return False


def _looks_like_leetcode_boilerplate(text: str) -> bool:
    return any(marker in text for marker in _LEETCODE_MARKERS)


def _snake_to_words(name: str) -> str:
    return " ".join(part for part in name.split("_") if part)


def _infer_summary(node: ast.FunctionDef) -> str:
    """Heuristic one-line summary from name and simple loop/return patterns."""
    name_hint = _snake_to_words(node.name)
    loops = [n for n in ast.walk(node) if isinstance(n, ast.For)]
    returns = [n for n in ast.walk(node) if isinstance(n, ast.Return) and n.value is not None]

    for loop in loops:
        if (
            isinstance(loop.target, ast.Name)
            and isinstance(loop.iter, ast.Call)
            and isinstance(loop.iter.func, ast.Name)
            and loop.iter.func.id == "range"
        ):
            args = node.args.args
            param = args[0].arg if args else "n"
            body = ast.unparse(loop.body[0]) if loop.body else ""
            if "+=" in body or "sum" in node.name.lower():
                return (
                    f"Return the sum of integers from 1 to {param} "
                    f"({name_hint or 'accumulated total'})."
                )
            return f"Iterate over a range and compute {name_hint or 'a result'}."

    if returns:
        return f"{name_hint.capitalize()} and return the result."
    return f"{name_hint.capitalize()}."


def fallback_documentation(code: str) -> str:
    """Deterministic docstring from AST when the LM output is unusable."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return '"""Documentation could not be generated for invalid Python."""'

    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    if not funcs:
        return '"""Describe what this code does."""'

    node = funcs[0]
    args = [a.arg for a in node.args.args if a.arg != "self"]
    summary = _infer_summary(node)

    lines: List[str] = [f'"""{summary}']
    if args:
        lines.append("")
        lines.append("Args:")
        for arg in args:
            lines.append(f"    {arg}: Input value.")
    if any(isinstance(s, ast.Return) and s.value is not None for s in ast.walk(node)):
        lines.append("")
        lines.append("Returns:")
        lines.append("    The computed result.")
    lines.append('"""')
    return "\n".join(lines)


def finalize_documentation(code: str, raw: str) -> str:
    if _looks_like_leetcode_boilerplate(raw):
        return fallback_documentation(code)

    cleaned = clean_generated_documentation(raw)
    if _looks_like_code_spill(cleaned) or _looks_like_leetcode_boilerplate(cleaned) or len(cleaned.split()) < 3:
        return fallback_documentation(code)
    return cleaned
