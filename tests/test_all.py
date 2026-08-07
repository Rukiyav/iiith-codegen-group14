"""Consolidated test suite for 5-file architecture."""

from __future__ import annotations

import sys
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch

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


def test_docstring_from_ast_with_existing_docstring():
    code = (
        "def check_palindrome(s):\n"
        "    \"\"\"\n"
        "    Check if string s is a palindrome.\n"
        "    \"\"\"\n"
        "    return s == s[::-1]\n"
    )
    doc = docstring_from_ast(code)
    assert doc is not None
    assert "Check if string s is a palindrome." in doc
    assert "s: The input string `s`." in doc


def test_translate_code_python_to_cpp():
    from src.engine import translate_code
    import src.engine as engine
    orig = engine.generate_code
    try:
        engine.generate_code = lambda *args, **kwargs: ["int main() {\n    return 0;\n}\n# Translate Python code to C++.\n# Python:\n"]
        res = translate_code("def add(a, b):\n    return a + b", "python", "cpp", model="mock")
        assert res["output"] == "int main() {\n    return 0;\n}"
    finally:
        engine.generate_code = orig


def test_execute_with_tests_sandbox_imports(tmp_path):
    util_file = tmp_path / "dummy_util.py"
    util_file.write_text("def special_const():\n    return 100\n")
    
    code = "from dummy_util import special_const\ndef run():\n    return special_const()"
    tests = ["assert run() == 100"]
    
    res = execute_with_tests(code, tests, project_dir=str(tmp_path))
    assert res.passed


def test_resolve_rag_dependencies():
    from src.engine import resolve_rag_dependencies
    # hits does not contain reverse_string
    hits = [
        {"name": "is_palindrome", "code": "def is_palindrome(s):\n    return s == s[::-1]"}
    ]
    code = "def is_string_palindrome(s):\n    return is_palindrome(s) and is_palindrome(reverse_string(s))"
    resolved = resolve_rag_dependencies(code, hits)
    assert "def is_palindrome" in resolved
    assert "def reverse_string" in resolved
    assert "def is_string_palindrome" in resolved


def test_infer_signature_from_examples():
    from src.engine import infer_signature
    task = (
        "Write a function to find the longest common prefix string amongst an array of strings.\n"
        "Example 1:\n"
        "Input: strs = [\"flower\",\"flow\",\"flight\"]\n"
        "Output: \"fl\"\n"
    )
    sig = infer_signature(task)
    # Checks that it did not prefix "is_" (since output "fl" is not boolean)
    # Checks that it parsed "strs" parameter from inputs
    assert sig == "def find_longest(strs):"


@patch("src.engine._load_st_model")
def test_rag_longest_common_prefix(mock_load_st):
    from unittest.mock import MagicMock
    import numpy as np
    from src.engine import retrieve
    
    mock_st = MagicMock()
    def mock_encode(sentences, **kwargs):
        n_sents = len(sentences) if isinstance(sentences, list) else 1
        return np.zeros((n_sents, 384), dtype=np.float32)
    mock_st.encode.side_effect = mock_encode
    mock_load_st.return_value = mock_st
    
    task = "Write a function to find the longest common prefix string amongst an array of strings."
    hits = retrieve(task, top_k=1)
    assert len(hits) > 0
    assert hits[0]["name"] == "longest_common_prefix"


def test_normalize_leetcode_signature_ast():
    from src.engine import normalize_leetcode_signature
    sig1 = "def twoSum(self, nums: List[int], target: int) -> List[int]:"
    assert normalize_leetcode_signature(sig1) == "def twoSum(nums, target):"

    sig2 = "def solve(x: float, y: str = 'default'):"
    assert normalize_leetcode_signature(sig2) == "def solve(x, y='default'):"


def test_safe_normalize_json_literals():
    from src.engine import safe_normalize_json_literals
    assert safe_normalize_json_literals("true") == "True"
    assert safe_normalize_json_literals("false") == "False"
    assert safe_normalize_json_literals("null") == "None"
    assert safe_normalize_json_literals('{"a": true}') == "{'a': True}"
    # Literal true inside quotes should remain unchanged
    assert safe_normalize_json_literals('"this is true"') == "'this is true'"


def test_multiline_parse_io_examples():
    task = (
        "Example 1:\n"
        "Input:\n"
        "nums = [1, 2]\n"
        "target = 3\n"
        "Output:\n"
        "[\n"
        "  0,\n"
        "  1\n"
        "]\n"
    )
    examples = parse_io_examples(task)
    assert len(examples) == 1
    assert "target = 3" in examples[0][0]
    assert "0" in examples[0][1]


def test_unbiased_pass_at_k():
    from src.eval import compute_pass_at_k
    # 5 samples, 2 correct, 3 incorrect. Let's compute pass@1 and pass@5
    samples = [["def f(): return 1"] * 2 + ["def f(): return 0"] * 3]
    tests = [["assert f() == 1"]]
    
    p1 = compute_pass_at_k(samples, tests, k=1)
    # expected pass@1: c / n = 2 / 5 = 0.4
    assert abs(p1 - 0.4) < 1e-6

    p5 = compute_pass_at_k(samples, tests, k=5)
    # since n = 5 and k = 5, we expect 1.0 (since at least one of the 5 is correct)
    assert abs(p5 - 1.0) < 1e-6


@patch("src.engine.AutoModelForCausalLM.from_pretrained")
@patch("src.engine.AutoTokenizer.from_pretrained")
def test_model_backend_abstractions(mock_tokenizer, mock_model):
    from src.engine import get_backend_for_model, ModelBackend, _backends
    from unittest.mock import MagicMock
    _backends.clear()
    
    mock_model.return_value = MagicMock()
    mock_tokenizer.return_value = MagicMock()
    
    backend = get_backend_for_model("codegen")
    assert isinstance(backend, ModelBackend)
    assert backend.model_id == "Salesforce/codegen-350M-multi"
    assert backend.get_tokenizer() is not None


def test_registry_and_unknown_validation():
    from src.engine import get_backend_for_model
    import pytest
    
    with pytest.raises(ValueError, match="Unknown model name"):
        get_backend_for_model("nonexistent_model")


@patch("src.engine.AutoModelForCausalLM.from_pretrained")
@patch("src.engine.AutoTokenizer.from_pretrained")
def test_mocked_model_backends(mock_tokenizer, mock_model):
    from src.engine import _backends, get_backend_for_model
    from unittest.mock import patch, MagicMock
    _backends.clear()
    
    mock_tok_instance = MagicMock()
    mock_tok_instance.apply_chat_template.return_value = "<|im_start|>mocked chat prompt"
    mock_tokenizer.return_value = mock_tok_instance
    
    mock_model_instance = MagicMock()
    mock_model_instance.to.return_value = mock_model_instance
    mock_model_instance.generate.return_value = [[0]]
    mock_model.return_value = mock_model_instance
    
    qwen_backend = get_backend_for_model("qwen_1_5b")
    assert qwen_backend.prompt_style == "chat"
    assert qwen_backend.model_id == "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    
    # Verify chat template generation called exactly once
    qwen_backend.generate("test task")
    mock_tok_instance.apply_chat_template.assert_called_once()
    
    # Verify num_samples > 1 calls _generate_raw with num_return_sequences=3 (no recursion formatting)
    mock_model_instance.generate.reset_mock()
    qwen_backend.generate("test task", num_samples=3)
    _, kwargs = mock_model_instance.generate.call_args
    assert kwargs["num_return_sequences"] == 3


@patch("src.engine.AutoModelForCausalLM.from_pretrained")
@patch("src.engine.AutoTokenizer.from_pretrained")
def test_model_mode_orthogonality(mock_tokenizer, mock_model):
    from src.engine import generate_code_for_task, _backends, ModelBackend
    from unittest.mock import patch, MagicMock
    _backends.clear()
    
    mock_backend = MagicMock(spec=ModelBackend)
    mock_backend.generate.return_value = ["def solve(): pass"]
    
    with patch("src.engine.get_backend_for_model", return_value=mock_backend) as mock_get_backend:
        models = ["codegen", "codegen_lora", "qwen_1_5b"]
        modes = ["baseline", "rag", "agentic"]
        
        for m_name in models:
            for mode in modes:
                with patch("src.engine.execute_with_tests") as mock_exec, \
                     patch("src.engine.self_correct") as mock_sc, \
                     patch("src.engine.retrieve") as mock_retrieve:
                    
                    mock_exec.return_value.passed = True
                    mock_sc.return_value = ("def solve(): pass", [], MagicMock())
                    mock_retrieve.return_value = []
                    
                    res = generate_code_for_task("Write code.", model_name=m_name, mode=mode)
                    assert res.code is not None
                    mock_get_backend.assert_any_call(m_name)


@patch("src.engine.AutoModelForCausalLM.from_pretrained")
@patch("src.engine.AutoTokenizer.from_pretrained")
def test_boundary_and_validation_guards(mock_tokenizer, mock_model):
    from src.engine import generate_code_for_task, _backends, ModelBackend
    from unittest.mock import patch, MagicMock
    import pytest
    import warnings
    _backends.clear()
    
    mock_backend = MagicMock(spec=ModelBackend)
    mock_backend.generate.return_value = ["def solve(): pass"]
    
    with patch("src.engine.get_backend_for_model", return_value=mock_backend) as mock_get_backend, \
         warnings.catch_warnings(record=True) as w, \
         patch("src.engine.execute_with_tests") as mock_exec:
        mock_exec.return_value.passed = True
        res = generate_code_for_task("Write code.", model_name="codegen", mode="lora")
        assert len(w) == 1
        assert issubclass(w[-1].category, DeprecationWarning)
        mock_get_backend.assert_called_with("codegen_lora")
        
    with pytest.raises(ValueError, match="LoRA mode is only supported for CodeGen"):
        generate_code_for_task("Write code.", model_name="qwen_1_5b", mode="lora")
        
    with pytest.raises(ValueError, match="Unknown pipeline mode"):
        generate_code_for_task("Write code.", model_name="codegen", mode="invalid_mode")


@patch("src.engine.AutoModelForCausalLM.from_pretrained")
@patch("src.engine.AutoTokenizer.from_pretrained")
@patch("src.engine.PeftModel.from_pretrained")
def test_custom_registry_adapter_path(mock_peft, mock_tokenizer, mock_model):
    from src.engine import _backends, get_backend_for_model
    from src.config import MODEL_REGISTRY
    from unittest.mock import MagicMock
    from pathlib import Path
    _backends.clear()
    
    mock_base = MagicMock()
    mock_model.return_value = mock_base
    mock_tokenizer.return_value = MagicMock()
    
    mock_peft_model = MagicMock()
    mock_peft.return_value = mock_peft_model
    
    custom_model_key = "custom_test_model"
    custom_path = Path("/mocked/custom/adapter/path")
    
    custom_config = {
        "model_id": "Salesforce/codegen-350M-multi",
        "backend": "huggingface",
        "prompt_style": "completion",
        "name": "Custom Test Model",
        "adapter": {
            "type": "peft_lora",
            "path": custom_path,
        },
        "supports_docs": True,
        "supports_translation": True,
    }
    
    with patch("src.engine.Path.exists", return_value=True), \
         patch.dict(MODEL_REGISTRY, {custom_model_key: custom_config}):
        
        backend = get_backend_for_model(custom_model_key)
        assert backend is not None
        mock_peft.assert_called_once_with(mock_base, str(custom_path))


@patch("src.engine.AutoModelForCausalLM.from_pretrained")
@patch("src.engine.AutoTokenizer.from_pretrained")
def test_end_to_end_regression(mock_tokenizer, mock_model):
    from src.engine import generate_code_for_task, _backends
    from unittest.mock import patch, MagicMock
    _backends.clear()
    
    mock_tok_instance = MagicMock()
    mock_tokenizer.return_value = mock_tok_instance
    
    mock_model_instance = MagicMock()
    mock_model_instance.to.return_value = mock_model_instance
    mock_model_instance.generate.return_value = [[0]]
    mock_model.return_value = mock_model_instance
    
    with patch("src.engine.execute_with_tests") as mock_exec, \
         patch("src.engine.retrieve") as mock_retrieve:
         
         mock_retrieve.return_value = []
         
         exec_fail = MagicMock()
         exec_fail.passed = False
         exec_fail.stderr = "AssertionError"
         exec_fail.attempt = 0
         
         exec_pass = MagicMock()
         exec_pass.passed = True
         exec_pass.stderr = ""
         exec_pass.attempt = 1
         
         mock_exec.side_effect = [exec_fail, exec_pass, exec_fail, exec_pass]
         
         mock_tok_instance.decode.side_effect = [
             "def solve(): return 0",
             "def solve(): return 1",
             "def solve(): return 0",
             "def solve(): return 1",
         ]
         
         res_a = generate_code_for_task(
             "Write solve.", model_name="codegen", mode="agentic", max_retries=1
         )
         assert res_a.passed is True
         assert "solve" in res_a.code
         assert res_a.attempts == 1
         
         res_b = generate_code_for_task(
              "Write solve.", model_name="qwen_1_5b", mode="agentic", max_retries=1
         )
         assert res_b.passed is True
         assert "solve" in res_b.code
         assert res_b.attempts == 1
