"""Unified AI Engine — model loading, prompt parsing, execution sandbox, RAG, self-correction, docstrings, and translation."""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Optional, Sequence, Tuple

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

from src.config import (
    BASELINE_CACHE,
    CODOC_PATH,
    DOC_N,
    EMBED_MODEL,
    EVAL_RESULTS_DIR,
    INDEX_DIR,
    LORA_DIR,
    MBPP_PATH,
    MODEL_ID,
    REPO_CORPUS_DIR,
    ROOT,
    ensure_dirs,
    get_device,
    get_embed_device,
)

# -----------------------------------------------------------------------------
# 0. Data Loading Helpers
# -----------------------------------------------------------------------------

def download_mbpp(force: bool = False) -> Path:
    ensure_dirs()
    if MBPP_PATH.exists() and not force:
        return MBPP_PATH
    import urllib.request
    url = "https://huggingface.co/datasets/Muennighoff/mbpp/resolve/main/data/sanitized-mbpp.json"
    urllib.request.urlretrieve(url, MBPP_PATH)
    return MBPP_PATH


def load_mbpp() -> List[dict]:
    download_mbpp()
    with open(MBPP_PATH) as f:
        return json.load(f)


def load_codoc() -> List[dict]:
    if not CODOC_PATH.exists():
        mbpp = load_mbpp()
        pairs = [{"code": p.get("code", ""), "docstring": p.get("prompt", ""), "task_id": p.get("task_id", "")} for p in mbpp if p.get("code") and p.get("prompt")]
        with open(CODOC_PATH, "w") as f:
            json.dump(pairs, f)
    with open(CODOC_PATH) as f:
        return json.load(f)


# -----------------------------------------------------------------------------
# 1. Models & Generation
# -----------------------------------------------------------------------------

_tokenizer: Optional[AutoTokenizer] = None
_base_model = None
_lora_model = None


def load_tokenizer() -> AutoTokenizer:
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=str(BASELINE_CACHE))
        if _tokenizer.pad_token is None:
            _tokenizer.pad_token = _tokenizer.eos_token
        _tokenizer.padding_side = "left"
        _tokenizer.truncation_side = "left"
    return _tokenizer


def load_base_model(force_reload: bool = False):
    global _base_model
    if _base_model is not None and not force_reload:
        return _base_model
    device = get_device()
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        cache_dir=str(BASELINE_CACHE),
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model = model.to(device)
    model.eval()
    _base_model = model
    return _base_model


def load_lora_model(force_reload: bool = False):
    global _lora_model
    if _lora_model is not None and not force_reload:
        return _lora_model
    adapter_config = LORA_DIR / "adapter_config.json"
    if not adapter_config.exists():
        raise FileNotFoundError(
            f"LoRA adapter not found at {LORA_DIR} (missing adapter_config.json). "
            "Train first via train.py"
        )
    device = get_device()
    dtype = torch.float16 if device == "cuda" else torch.float32
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        cache_dir=str(BASELINE_CACHE),
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(base, str(LORA_DIR))
    model = model.to(device)
    model.eval()
    _lora_model = model
    return _lora_model


def get_models(mode: str = "baseline") -> Tuple:
    tokenizer = load_tokenizer()
    if mode == "lora":
        return load_lora_model(), tokenizer
    return load_base_model(), tokenizer


class StopAtNewDef(StoppingCriteria):
    def __init__(self, tokenizer):
        self.stop_ids = [
            tokenizer.encode(s, add_special_tokens=False)
            for s in ["\nclass ", "\ndef ", "\n#", "\nif __name__"]
        ]

    def __call__(self, input_ids, scores, **kwargs):
        for stop in self.stop_ids:
            if len(stop) and input_ids.shape[1] >= len(stop):
                if input_ids[0, -len(stop) :].tolist() == stop:
                    return True
        return False


def generate_code(
    model,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.8,
    top_p: float = 0.95,
    num_samples: int = 1,
    stop: bool = True,
    tokenizer=None,
) -> List[str]:
    tokenizer = tokenizer or load_tokenizer()
    device = get_device()
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    in_len = inputs["input_ids"].shape[1]
    sc = StoppingCriteriaList([StopAtNewDef(tokenizer)]) if stop else None
    do_sample = temperature > 0
    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        num_return_sequences=num_samples,
        pad_token_id=tokenizer.eos_token_id,
        stopping_criteria=sc,
    )
    if do_sample:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p
    with torch.no_grad():
        out = model.generate(**inputs, **gen_kwargs)
    return [tokenizer.decode(seq[in_len:], skip_special_tokens=True) for seq in out]


# -----------------------------------------------------------------------------
# 2. Execution Sandbox
# -----------------------------------------------------------------------------

@dataclass
class ExecResult:
    passed: bool
    stdout: str = ""
    stderr: str = ""
    error_type: str = ""
    attempt: int = 0
    code: str = ""


COMMON_IMPORTS = (
    "import math, re, sys, collections, itertools, functools, heapq, bisect, typing\n"
    "from collections import Counter, defaultdict, deque\n"
    "from itertools import combinations, permutations, groupby, product, accumulate\n"
    "from typing import List, Dict, Tuple, Set, Optional, Any, Union\n"
)


def execute_with_tests(
    code: str,
    test_list: List[str],
    test_imports: Optional[List[str]] = None,
    timeout: int = 10,
    project_dir: Optional[str] = None,
) -> ExecResult:
    setup_user = "\n".join(test_imports) if test_imports else ""
    setup = f"{COMMON_IMPORTS}\n{setup_user}"
    test_block = "\n".join(test_list) if isinstance(test_list, list) else test_list
    full_code = f"{setup}\n{code}\n\n# --- tests ---\n{test_block}\n"
    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
        f.write(full_code)
        tmp = f.name
    env = os.environ.copy()
    if project_dir:
        root = str(Path(project_dir).expanduser().resolve())
        prev = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = root + (os.pathsep + prev if prev else "")
    try:
        r = subprocess.run(
            [sys.executable, tmp],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
            cwd=project_dir or None,
        )
        if r.returncode == 0:
            return ExecResult(True, r.stdout, "", "pass")
        err = r.stderr
        err_type = "runtime"
        if "SyntaxError" in err or "IndentationError" in err:
            err_type = "syntax"
        elif "AssertionError" in err:
            err_type = "assertion"
        return ExecResult(False, r.stdout, err, err_type)
    except subprocess.TimeoutExpired:
        return ExecResult(False, "", "Timeout", "timeout")
    finally:
        Path(tmp).unlink(missing_ok=True)


# -----------------------------------------------------------------------------
# 3. Prompts & Signature Parsing
# -----------------------------------------------------------------------------

def get_signature(problem: Dict) -> str:
    for line in problem.get("code", "").splitlines():
        if line.strip().startswith("def "):
            return line.strip()
    return "def solution():"


def _snake(name: str) -> str:
    name = re.sub(r"[^0-9a-zA-Z]+", "_", name).strip("_").lower()
    if not name or name[0].isdigit():
        name = f"f_{name}" if name else "solution"
    return name[:48]


def normalize_leetcode_signature(sig: str) -> str:
    """Normalize class-based LeetCode signatures and type hints for CodeGen-350M."""
    sig = re.sub(r"->\s*[^:]+", "", sig)
    sig = re.sub(r":\s*[^,)]+", "", sig)
    sig = sig.replace("(self, ", "(").replace("(self)", "()")
    return sig.strip()


def infer_signature(task: str) -> str:
    lower = task.lower()
    m = re.search(r"(def\s+\w+\s*\([^)]*\)\s*(?:->\s*[^:]+)?\s*:)", task)
    if m:
        return normalize_leetcode_signature(m.group(1).strip())
    m = re.search(
        r"(?:function|method|def)\s+(?:called\s+)?[`'\"]?(\w+)[`'\"]?\s*(\([^)]*\))?",
        task,
        re.I,
    )
    if m:
        name = _snake(m.group(1))
        args = (m.group(2) or "()").strip()
        if args == "()":
            args = _guess_args(lower)
        return normalize_leetcode_signature(f"def {name}{args}:")
    m = re.match(r"^\s*([A-Za-z][A-Za-z0-9 ]{2,40}?)\s*(?:\n|$)", task.strip())
    if m and "example" not in m.group(1).lower():
        title = m.group(1).strip()
        if len(title.split()) <= 5 and not title.lower().startswith(
            ("given", "write", "return", "you ", "implement")
        ):
            return f"def {_snake(title)}{_guess_args(lower)}:"
    return f"def solution{_guess_args(lower)}:"


def _guess_args(lower: str) -> str:
    if ("array" in lower or "list" in lower or "nums" in lower) and "target" in lower:
        return "(nums, target)"
    if "linked list" in lower:
        return "(head)"
    if "binary tree" in lower or "bst" in lower:
        return "(root)"
    if re.search(r"\b(matrix|grid|board)\b", lower):
        return "(matrix)"
    if any(w in lower for w in ("string", "palindrome", "parentheses", "anagram")):
        return "(s)"
    if "two integers" in lower or re.search(r"\ba\s+and\s+b\b", lower):
        return "(a, b)"
    if any(w in lower for w in ("integer n", "number n", "given n")):
        return "(n)"
    if "nums" in lower or "array" in lower or "list of" in lower:
        return "(nums)"
    return "(*args, **kwargs)"


def parse_io_examples(task: str) -> List[Tuple[str, str]]:
    examples: List[Tuple[str, str]] = []
    blocks = re.split(r"(?i)example\s*\d*\s*:", task)
    for block in blocks[1:] if len(blocks) > 1 else [task]:
        inp = re.search(r"(?i)input\s*:\s*(.+?)(?=output\s*:|$)", block, re.S)
        out = re.search(
            r"(?i)output\s*:\s*(.+?)(?=explanation\s*:|example\s*\d|constraints\s*:|$)",
            block,
            re.S,
        )
        if inp and out:
            examples.append(
                (
                    inp.group(1).strip().split("\n")[0].strip(),
                    out.group(1).strip().split("\n")[0].strip(),
                )
            )
    if examples:
        return examples
    for m in re.finditer(r"(?i)input\s*:\s*(.+?)\s*output\s*:\s*([^\n]+)", task):
        examples.append((m.group(1).strip(), m.group(2).strip()))
    return examples


def _kwargs_from_input(input_str: str) -> Optional[str]:
    input_str = input_str.strip().rstrip(".")
    if not input_str:
        return None
    if "=" not in input_str:
        return input_str
    parts = []
    depth = 0
    buf = []
    for ch in input_str:
        if ch in "[({":
            depth += 1
        elif ch in "])}":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append("".join(buf).strip())
            buf = []
            continue
        buf.append(ch)
    if buf:
        parts.append("".join(buf).strip())
    kwargs = []
    for part in parts:
        if "=" in part:
            k, v = part.split("=", 1)
            kwargs.append(f"{k.strip()}={v.strip()}")
        else:
            kwargs.append(part)
    return ", ".join(kwargs)


def build_tests_from_examples(signature: str, task: str) -> List[str]:
    name = signature.split("(")[0].replace("def", "").strip()
    any_order = "any order" in task.lower()
    tests: List[str] = []
    for inp, out in parse_io_examples(task):
        args = _kwargs_from_input(inp)
        if args is None:
            continue
        out_norm = out.replace("true", "True").replace("false", "False").replace("null", "None")
        try:
            ast.parse(f"f({args})")
            ast.parse(out_norm)
        except SyntaxError:
            continue
        if any_order and ((out_norm.startswith("[") and out_norm.endswith("]")) or (out_norm.startswith("(") and out_norm.endswith(")"))):
            tests.append(f"assert sorted({name}({args})) == sorted({out_norm})")
        else:
            tests.append(f"assert {name}({args}) == {out_norm}")
    return tests


def infer_smoke_tests(signature: str, task: str) -> List[str]:
    from_examples = build_tests_from_examples(signature, task)
    if from_examples:
        return from_examples
    name = signature.split("(")[0].replace("def", "").strip()
    return [f"assert callable({name})"]


def build_prompt(problem: Dict) -> str:
    sig = get_signature(problem)
    return f'"""\n{problem["prompt"]}\n"""\n{sig}\n'


def build_freeform_prompt(task: str, signature: str, few_shot: bool = True) -> str:
    core = f'"""\n{task.strip()}\n"""\n{signature}\n'
    if not few_shot:
        return core
    shots = (
        '"""\nGiven an array of integers nums and an integer target, return indices of two numbers adding to target.\n'
        'Input: nums = [2,7,11,15], target = 9 Output: [0,1]\n"""\n'
        'def two_sum(nums, target):\n'
        '    seen = {}\n'
        '    for i, x in enumerate(nums):\n'
        '        if target - x in seen:\n'
        '            return [seen[target - x], i]\n'
        '        seen[x] = i\n'
        '    return []\n\n'
        '"""\nWrite a python function to count number of digits in a given string.\n"""\n'
        'def number_ctr(s):\n'
        '    return sum(1 for c in s if c.isdigit())\n\n'
        '"""\nWrite a python function to find the minimum difference between any two elements in a given array.\n"""\n'
        'def find_min_diff(arr, n):\n'
        '    arr.sort()\n'
        '    return min(arr[i+1] - arr[i] for i in range(n - 1))\n\n'
        '"""\nReturn True if string s is a valid palindrome.\n"""\n'
        "def is_palindrome(s):\n    return s == s[::-1]\n\n"
    )
    return shots + core


def is_valid_python(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def clean_body(sig: str, raw: str) -> str:
    text = raw
    if text.lstrip().startswith("def "):
        nl = text.find("\n")
        text = text[nl + 1 :] if nl != -1 else ""
    lines = text.split("\n")
    body = []
    for ln in lines:
        if ln.strip() and not ln.startswith((" ", "\t")) and body:
            break
        body.append(ln)
    cleaned = "\n".join(body).rstrip()
    if not cleaned.strip():
        cleaned = "    pass"
    elif not cleaned.startswith((" ", "\t")):
        cleaned = "\n".join(("    " + ln if ln.strip() else ln) for ln in cleaned.split("\n"))
    
    full = sig + "\n" + cleaned

    # AST Auto-indentation fix for unindented loop/if block bodies
    if not is_valid_python(full):
        fixed_lines = []
        in_colon_block = False
        for ln in full.splitlines():
            if in_colon_block and ln.strip() and not ln.startswith("    "):
                fixed_lines.append("    " + ln.strip())
                in_colon_block = False
            else:
                fixed_lines.append(ln)
                if ln.strip().endswith(":"):
                    in_colon_block = True
        candidate_fixed = "\n".join(fixed_lines)
        if is_valid_python(candidate_fixed):
            full = candidate_fixed

    return full


# -----------------------------------------------------------------------------
# 4. Retrieval-Augmented Generation (RAG)
# -----------------------------------------------------------------------------

SKIP_DIR_NAMES = {
    ".git", ".hg", ".svn", ".venv", "venv", "env", "__pycache__", ".mypy_cache",
    ".pytest_cache", ".ruff_cache", "node_modules", "dist", "build", ".tox", ".eggs",
    "eggs", "site-packages", "models", "retrieval", "experiments", ".cursor",
}


def extract_functions(source: str, file_path: str) -> List[Dict]:
    chunks: List[Dict] = []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return chunks
    lines = source.splitlines()

    def _slice(node) -> str:
        start = node.lineno - 1
        end = getattr(node, "end_lineno", start + 20)
        return "\n".join(lines[start:end])

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            code = _slice(node)
            if 10 <= len(code) <= 2500:
                chunks.append(
                    {
                        "code": code,
                        "name": node.name,
                        "file": file_path,
                        "lineno": node.lineno,
                        "kind": "function",
                        "doc": ast.get_docstring(node) or "",
                    }
                )
        elif isinstance(node, ast.ClassDef):
            code = _slice(node)
            if 20 <= len(code) <= 4000:
                chunks.append(
                    {
                        "code": code[:2500] + ("\n# ..." if len(code) > 2500 else ""),
                        "name": node.name,
                        "file": file_path,
                        "lineno": node.lineno,
                        "kind": "class",
                        "doc": ast.get_docstring(node) or "",
                    }
                )
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    method = _slice(child)
                    if 10 <= len(method) <= 2500:
                        chunks.append(
                            {
                                "code": method,
                                "name": f"{node.name}.{child.name}",
                                "file": file_path,
                                "lineno": child.lineno,
                                "kind": "method",
                                "doc": ast.get_docstring(child) or "",
                            }
                        )
    return chunks


def _iter_source_files(root: Path, globs: Sequence[str] = ("*.py",), max_files: int = 2000) -> Iterable[Path]:
    count = 0
    for pattern in globs:
        for path in sorted(root.rglob(pattern)):
            if not path.is_file() or any(part in SKIP_DIR_NAMES for part in path.parts):
                continue
            yield path
            count += 1
            if count >= max_files:
                return


def build_corpus_from_dir(root: str | Path, max_files: int = 2000, globs: Sequence[str] = ("*.py",), relative_to: Optional[Path] = None) -> List[Dict]:
    root_path = Path(root).expanduser().resolve()
    if not root_path.is_dir():
        raise FileNotFoundError(f"Project folder not found: {root_path}")
    base = relative_to or root_path
    all_chunks: List[Dict] = []
    n_files = 0
    for py_file in _iter_source_files(root_path, globs=globs, max_files=max_files):
        try:
            src = py_file.read_text(errors="ignore")
        except OSError:
            continue
        try:
            rel = str(py_file.relative_to(base))
        except ValueError:
            rel = str(py_file)
        file_chunks = extract_functions(src, rel)
        for c in file_chunks:
            c["project_root"] = str(root_path)
        all_chunks.extend(file_chunks)
        n_files += 1
    return all_chunks


def build_corpus(max_files_per_repo: int = 200) -> List[Dict]:
    all_chunks: List[Dict] = []
    if not REPO_CORPUS_DIR.exists():
        return all_chunks
    for repo_dir in sorted(REPO_CORPUS_DIR.iterdir()):
        if repo_dir.is_dir():
            chunks = build_corpus_from_dir(repo_dir, max_files=max_files_per_repo, relative_to=ROOT)
            all_chunks.extend(chunks)
    return all_chunks


def project_cache_key(root: str | Path) -> str:
    resolved = str(Path(root).expanduser().resolve())
    return hashlib.sha1(resolved.encode()).hexdigest()[:12]


def _import_faiss():
    try:
        import faiss
    except ImportError as e:
        raise ImportError("faiss is required for RAG. Install with: pip install faiss-cpu") from e
    try:
        faiss.omp_set_num_threads(1)
    except Exception:
        pass
    return faiss


def _embed_chunks(chunks, st_model):
    texts = []
    for c in chunks:
        meta = f"{c.get('file', '')} :: {c.get('name', '')}\n"
        doc = (c.get("doc") or "").strip()
        prefix = f"{meta}{doc}\n" if doc else meta
        texts.append(prefix + c["code"])
    return st_model.encode(
        texts,
        batch_size=64,
        show_progress_bar=len(texts) > 100,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)


def _write_index(index_dir: Path, chunks, embeddings, st_model):
    faiss = _import_faiss()
    index_dir.mkdir(parents=True, exist_ok=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    faiss.write_index(index, str(index_dir / "repo.index"))
    with open(index_dir / "chunks.json", "w") as f:
        json.dump(chunks, f)
    np.save(index_dir / "embeddings.npy", embeddings)
    return index, chunks, st_model


def _load_st_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBED_MODEL, device=get_embed_device())


def build_index(force: bool = False):
    faiss = _import_faiss()
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    index_path = INDEX_DIR / "repo.index"
    chunks_path = INDEX_DIR / "chunks.json"
    if index_path.exists() and chunks_path.exists() and not force:
        return load_index()
    all_chunks = build_corpus()
    if not all_chunks:
        raise RuntimeError("No chunks found under data/repo_corpus.")
    st_model = _load_st_model()
    embeddings = _embed_chunks(all_chunks, st_model)
    return _write_index(INDEX_DIR, all_chunks, embeddings, st_model)


def load_index():
    faiss = _import_faiss()
    index_path = INDEX_DIR / "repo.index"
    chunks_path = INDEX_DIR / "chunks.json"
    if not index_path.exists() or not chunks_path.exists():
        return build_index()
    index = faiss.read_index(str(index_path))
    with open(chunks_path) as f:
        chunks = json.load(f)
    st_model = _load_st_model()
    return index, chunks, st_model


def project_index_dir(project_root: str | Path) -> Path:
    key = project_cache_key(project_root)
    return INDEX_DIR / "projects" / key


def build_project_index(project_root: str | Path, *, force: bool = False, max_files: int = 2000) -> Tuple:
    faiss = _import_faiss()
    root = Path(project_root).expanduser().resolve()
    out_dir = project_index_dir(root)
    index_path = out_dir / "repo.index"
    chunks_path = out_dir / "chunks.json"
    if index_path.exists() and chunks_path.exists() and not force:
        index = faiss.read_index(str(index_path))
        with open(chunks_path) as f:
            chunks = json.load(f)
        return index, chunks, _load_st_model()
    chunks = build_corpus_from_dir(root, max_files=max_files)
    if not chunks:
        raise RuntimeError(f"No Python functions/classes found under {root}")
    st_model = _load_st_model()
    embeddings = _embed_chunks(chunks, st_model)
    return _write_index(out_dir, chunks, embeddings, st_model)


_cached_default: Optional[Tuple] = None
_cached_projects: Dict[str, Tuple] = {}


def get_index(project_dir: str | Path | None = None, *, force_rebuild: bool = False):
    global _cached_default
    if project_dir:
        key = str(Path(project_dir).expanduser().resolve())
        if force_rebuild or key not in _cached_projects:
            _cached_projects[key] = build_project_index(key, force=force_rebuild)
        return _cached_projects[key]
    if force_rebuild or _cached_default is None:
        _cached_default = build_index(force=force_rebuild)
    return _cached_default


def retrieve(
    query: str,
    top_k: int = 3,
    *,
    project_dir: str | Path | None = None,
    force_rebuild: bool = False,
    min_score: float = 0.35,
) -> List[dict]:
    index, all_chunks, st_model = get_index(project_dir, force_rebuild=force_rebuild)
    q_emb = st_model.encode([query], normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
    q_emb = np.ascontiguousarray(q_emb, dtype=np.float32)
    scores, idxs = index.search(q_emb, top_k)
    results = []
    for score, idx in zip(scores[0], idxs[0]):
        if idx >= 0 and float(score) >= min_score:
            chunk = dict(all_chunks[idx])
            chunk["score"] = float(score)
            results.append(chunk)
    return results


def format_context(hits: List[dict], max_chars: int = 1800, header: str = "# Relevant code:") -> str:
    if not hits:
        return ""
    lines = [header]
    used = len(header)
    for i, c in enumerate(hits):
        name = c.get("name", "?")
        path = c.get("file", "?")
        score = c.get("score")
        score_s = f" score={score:.3f}" if isinstance(score, float) else ""
        block_header = f"# --- [{i + 1}] `{name}` @ {path}{score_s} ---"
        code = c.get("code", "")
        remaining = max_chars - used - 40
        if remaining < 80:
            break
        if len(code) > remaining:
            code = code[:remaining] + "\n# ..."
        block = f"{block_header}\n{code}"
        lines.append(block)
        used += len(block) + 1
    lines.append("# --- End references ---\n")
    return "\n".join(lines) + "\n"


def build_rag_prefix(query: str, *, project_dir: str | Path | None = None, top_k: int = 3, max_chars: int = 1800, force_rebuild: bool = False) -> str:
    try:
        hits = retrieve(query, top_k=top_k, project_dir=project_dir, force_rebuild=force_rebuild)
    except Exception:
        return ""
    if not hits:
        return ""
    header = "# Relevant code from user project:" if project_dir else "# Relevant reference implementations:"
    return format_context(hits, max_chars=max_chars, header=header)


def rag_generate(model, query: str, signature: str | None = None, max_new_tokens: int = 256, temperature: float = 0.0, top_p: float = 0.95, *, project_dir: str | Path | None = None, top_k: int = 3, force_rebuild: bool = False) -> List[str]:
    context = build_rag_prefix(query, project_dir=project_dir, top_k=top_k, force_rebuild=force_rebuild)
    sig = signature or "def "
    prompt = context + f'"""\n{query}\n"""\n{sig}\n'
    tokenizer = load_tokenizer()
    toks = tokenizer.encode(prompt)
    if len(toks) > 1400:
        toks = toks[-1400:]
        prompt = tokenizer.decode(toks, skip_special_tokens=True)
    return generate_code(model, prompt, max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p, tokenizer=tokenizer)


# -----------------------------------------------------------------------------
# 5. Agentic Self-Correction Loop
# -----------------------------------------------------------------------------

def _reflect(last: ExecResult) -> str:
    et = last.error_type or "runtime"
    stderr = last.stderr or ""
    lines = [l.strip() for l in stderr.splitlines() if l.strip()]
    last_line = lines[-1] if lines else et

    m = re.search(r"File \".*?\", line (\d+)", stderr)
    line_info = f" at line {m.group(1)}" if m else ""

    if et == "syntax":
        return f"Fix syntax error{line_info}: {last_line}"
    if et == "assertion":
        return f"Logic failed assertion test{line_info}: {last_line}"
    if et == "timeout":
        return "Execution timed out (infinite loop or complex recursion)."
    return f"Error{line_info}: {last_line}"


def self_correct(
    model,
    problem: Dict,
    max_retries: int = 3,
    max_new_tokens: int = 384,
    verbose: bool = False,
    use_rag: bool = True,
    temperature: float = 0.0,
    n_candidates: int = 3,
    project_dir: str | Path | None = None,
    rag_top_k: int = 3,
) -> Tuple[str, List[ExecResult]]:
    sig = get_signature(problem)
    task = problem["prompt"]
    test_list = problem["test_list"]
    test_imports = problem.get("test_imports")
    history: List[ExecResult] = []
    reflections: List[str] = []
    project = Path(project_dir).expanduser().resolve() if project_dir else None
    rag = build_rag_prefix(task, project_dir=project, top_k=rag_top_k) if use_rag else ""

    cands = generate_code(
        model,
        rag + build_freeform_prompt(task, sig, few_shot=not rag),
        max_new_tokens=max_new_tokens,
        temperature=max(temperature, 0.4) if n_candidates > 1 else temperature,
        num_samples=max(1, n_candidates),
    )
    for c in cands:
        code = clean_body(sig, c)
        res = execute_with_tests(code, test_list, test_imports, project_dir=str(project) if project else None)
        res.code = code
        history.append(res)
        if res.passed:
            return code, history

    for attempt in range(1, max_retries + 1):
        last = history[-1]
        reflections.append(_reflect(last))
        err = (last.stderr or last.error_type or "")[:600].replace("\n", "\n# ")
        refl_block = "# Lessons from previous failures:\n" + "".join(f"# - {r}\n" for r in reflections[-3:]) + "\n"
        tests_prev = "\n".join(f"#   {t}" for t in test_list[:6])
        repair_prompt = (
            f"{rag}{refl_block}"
            f"# The following solution failed ({last.error_type}).\n# Error:\n# {err}\n"
            f"# Tests:\n{tests_prev}\n# Broken code:\n{last.code}\n"
            f"# Corrected complete function:\n\"\"\"\n{task}\n\"\"\"\n{sig}\n"
        )
        sample_temp = 0.5 if n_candidates > 1 else temperature
        cands = generate_code(
            model,
            repair_prompt,
            max_new_tokens=max_new_tokens,
            temperature=sample_temp,
            num_samples=max(1, n_candidates),
        )
        for c in cands:
            code = clean_body(sig, c)
            res = execute_with_tests(code, test_list, test_imports, project_dir=str(project) if project else None)
            res.attempt = attempt
            res.code = code
            history.append(res)
            if res.passed:
                return code, history

    return history[-1].code, history


# -----------------------------------------------------------------------------
# 6. Docstring Generation & Code Translation
# -----------------------------------------------------------------------------

def docstring_from_ast(code: str) -> Optional[str]:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    fn = next((n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))), None)
    if not fn:
        return None
    args = [a.arg for a in list(fn.args.posonlyargs) + list(fn.args.args) if a.arg not in ("self", "cls")]
    summary = f"Compute {fn.name.replace('_', ' ')}."
    lines = [summary, ""]
    if args:
        lines.append("Args:")
        for a in args:
            lines.append(f"    {a}: The `{a}` argument.")
        lines.append("")
    lines.append("Returns:\n    Result of calculation.")
    return "\n".join(lines)


def generate_docstring(model, code: str, max_new_tokens: int = 160, temperature: float = 0.0, few_shot: bool = False) -> str:
    structured = docstring_from_ast(code)
    if structured:
        return structured
    prompt = f"# Write a Google-style docstring body for:\n{code.rstrip()}\n# Docstring:\n\"\"\"\n"
    raw = generate_code(model, prompt, max_new_tokens=max_new_tokens, temperature=temperature, stop=False)[0]
    end = raw.find('"""')
    return raw[:end].strip() if end != -1 else raw.strip()


LANG_LABELS = {"python": "Python", "java": "Java", "javascript": "JavaScript", "typescript": "TypeScript", "cpp": "C++", "go": "Go", "rust": "Rust"}
SUPPORTED_PAIRS = [("python", "java"), ("python", "javascript"), ("java", "python"), ("javascript", "python"), ("python", "cpp"), ("python", "go")]


def translate_code(code: str, source_lang: str = "python", target_lang: str = "java", *, model=None, use_lora: bool = False, max_new_tokens: int = 512, temperature: float = 0.2) -> Dict:
    src = LANG_LABELS.get(source_lang, source_lang.title())
    tgt = LANG_LABELS.get(target_lang, target_lang.title())
    if source_lang == target_lang:
        return {"output": code, "source_lang": source_lang, "target_lang": target_lang, "status": "Same language"}
    if model is None:
        try:
            model = load_lora_model() if use_lora else load_base_model()
        except FileNotFoundError:
            model = load_base_model()
    prompt = f"# Translate {src} code to {tgt}.\n# {src}:\n{code.rstrip()}\n# {tgt}:\n"
    raw = generate_code(model, prompt, max_new_tokens=max_new_tokens, temperature=temperature, stop=False)[0]
    return {"output": raw.strip(), "source_lang": source_lang, "target_lang": target_lang, "status": f"Translated {src} → {tgt}"}


# -----------------------------------------------------------------------------
# 7. Unified Pipeline Dispatcher
# -----------------------------------------------------------------------------

Mode = Literal["baseline", "lora", "rag", "agentic"]


@dataclass
class GenerateResult:
    code: str
    mode: str
    passed: Optional[bool] = None
    attempts: Optional[int] = None
    history: List[ExecResult] = field(default_factory=list)
    status: str = ""
    retrieved: List[dict] = field(default_factory=list)


def generate_code_for_task(
    task: str,
    *,
    mode: Mode = "baseline",
    temperature: float = 0.2,
    max_new_tokens: int = 384,
    test_list: Optional[List[str]] = None,
    test_imports: Optional[List[str]] = None,
    reference_code: Optional[str] = None,
    max_retries: int = 3,
    n_candidates: int = 1,
    project_dir: Optional[str] = None,
    force_reindex: bool = False,
) -> GenerateResult:
    sig = get_signature({"code": reference_code, "prompt": task}) if reference_code else infer_signature(task)
    tests = test_list or infer_smoke_tests(sig, task)
    problem = {"prompt": task, "code": sig, "test_list": tests, "test_imports": test_imports or []}

    try:
        model = load_lora_model() if mode in ("lora", "agentic", "rag") else load_base_model()
    except FileNotFoundError:
        model = load_base_model()

    if mode == "agentic":
        code, history = self_correct(
            model, problem, max_retries=max_retries, max_new_tokens=max_new_tokens,
            temperature=temperature, n_candidates=n_candidates, project_dir=project_dir
        )
        passed = history[-1].passed if history else False
        return GenerateResult(code=code, mode=mode, passed=passed, attempts=len(history), history=history, status="Agentic complete")

    if mode == "rag":
        cands = rag_generate(model, task, signature=sig, max_new_tokens=max_new_tokens, temperature=temperature, project_dir=project_dir)
        code = clean_body(sig, cands[0])
    else:
        cands = generate_code(model, build_freeform_prompt(task, sig), max_new_tokens=max_new_tokens, temperature=temperature)
        code = clean_body(sig, cands[0])

    res = execute_with_tests(code, tests, test_imports, project_dir=project_dir)
    return GenerateResult(code=code, mode=mode, passed=res.passed, attempts=1, history=[res], status="Generated")


def index_project_folder(project_dir: str, *, force: bool = False, max_files: int = 2000) -> dict:
    _, chunks, _ = build_project_index(project_dir, force=force, max_files=max_files)
    files = sorted({c.get("file", "") for c in chunks})
    return {
        "project_root": project_dir,
        "n_chunks": len(chunks),
        "n_files": len(files),
        "status": f"Indexed {len(chunks)} chunks from {len(files)} files under {project_dir}",
    }


def translate_pl(code: str, source_lang: str = "python", target_lang: str = "java", use_lora: bool = False) -> Dict:
    return translate_code(code, source_lang=source_lang, target_lang=target_lang, use_lora=use_lora)
