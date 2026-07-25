"""Consolidated test suite for 5-file architecture."""

from __future__ import annotations

import sys
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import app
from src.engine import (
    build_prompt,
    build_tests_from_examples,
    clean_body,
    docstring_from_ast,
    execute_with_tests,
    get_signature,
    infer_signature,
    infer_smoke_tests,
    parse_io_examples,
)

client = TestClient(app)

SAMPLE_PROBLEM = {
    "prompt": "Write a function that returns 42.",
    "code": "def answer():\n    return 42",
    "test_list": ["assert answer() == 42"],
    "test_imports": [],
}


def test_health_api():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_models_api():
    r = client.get("/models")
    assert r.status_code == 200
    data = r.json()
    assert "modes" in data
    assert "baseline" in data["modes"]


def test_get_signature():
    assert get_signature(SAMPLE_PROBLEM) == "def answer():"


def test_build_prompt():
    p = build_prompt(SAMPLE_PROBLEM)
    assert '"""' in p
    assert "def answer():" in p


def test_clean_body_stops_at_dedent():
    sig = "def answer():"
    raw = "    return 42\n\ndef other():"
    code = clean_body(sig, raw)
    assert "def other" not in code
    assert "return 42" in code


def test_execute_with_tests_pass():
    code = "def answer():\n    return 42"
    r = execute_with_tests(code, SAMPLE_PROBLEM["test_list"])
    assert r.passed


def test_execute_with_tests_fail():
    code = "def answer():\n    return 0"
    r = execute_with_tests(code, SAMPLE_PROBLEM["test_list"])
    assert not r.passed
    assert r.error_type == "assertion"


def test_infer_signature_from_title():
    sig = infer_signature("Two Sum\nGiven an array of integers nums and target, return indices.")
    assert sig.startswith("def ")
    assert "nums" in sig and "target" in sig


def test_parse_io_examples_leetcode():
    task = (
        "Two Sum.\nExample 1:\nInput: nums = [2,7,11,15], target = 9\nOutput: [0,1]\n"
        "Example 2:\nInput: nums = [3,2,4], target = 6\nOutput: [1,2]\n"
    )
    examples = parse_io_examples(task)
    assert len(examples) >= 2


def test_docstring_from_ast():
    code = "def two_sum(nums, target):\n    return []\n"
    doc = docstring_from_ast(code)
    assert doc is not None
    assert "Args:" in doc
