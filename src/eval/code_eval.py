"""Code generation evaluation: execution accuracy and pass@1."""

from __future__ import annotations

import inspect
import signal
import threading
from contextlib import contextmanager
from typing import Dict, List, Optional


class _Timeout(Exception):
    pass


@contextmanager
def _time_limit(seconds: int):
    # signal.alarm only works on the main thread (breaks under FastAPI/TestClient).
    if threading.current_thread() is not threading.main_thread():
        yield
        return

    def _handler(signum, frame):
        raise _Timeout(f"Execution exceeded {seconds}s")

    if hasattr(signal, "SIGALRM"):
        old = signal.signal(signal.SIGALRM, _handler)
        signal.alarm(seconds)
        try:
            yield
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old)
    else:
        yield


def clean_generated_code(raw_code: str) -> str:
    """
    Keep module-level imports plus the first top-level function.

    Anything after that function (extra defs, __main__ blocks, echoed asserts)
    is dropped. The model often emits a blank line or an import before the def,
    so the search for the entry point cannot assume it is on the first line.
    """
    lines = [line for line in raw_code.split("\n") if not line.strip().startswith("```")]

    first_def = next(
        (i for i, line in enumerate(lines) if line.startswith("def ") or line.startswith("class ")),
        None,
    )
    if first_def is None:
        return "\n".join(lines).strip()

    imports = [
        line
        for line in lines[:first_def]
        if line.startswith("import ") or line.startswith("from ")
    ]

    body: List[str] = []
    for offset, line in enumerate(lines[first_def:]):
        # Any unindented, non-empty line after the header closes the function.
        if offset > 0 and line and not line[0].isspace():
            break
        body.append(line)

    return "\n".join(imports + body).strip()


def evaluate_functional_correctness(
    generated_code: str,
    test_list: List[str],
    timeout_seconds: int = 5,
) -> bool:
    """Return True if generated code passes all assert tests."""
    cleaned = clean_generated_code(generated_code)
    if not cleaned:
        return False
    full_script = cleaned + "\n\n" + "\n".join(test_list)
    global_vars: Dict = {}
    try:
        with _time_limit(timeout_seconds):
            exec(full_script, global_vars)  # noqa: S102 — eval harness only
        return True
    except Exception:
        return False


def pass_at_1(results: List[bool]) -> float:
    if not results:
        return 0.0
    return sum(1 for ok in results if ok) / len(results)


def compute_codebleu(pairs: List[tuple], lang: str = "python") -> Optional[float]:
    """Average CodeBLEU over (reference, prediction) pairs; None if codebleu isn't installed."""
    try:
        from codebleu import calc_codebleu
    except ImportError:
        return None
    scores = []
    for ref, pred in pairs:
        if not (ref or "").strip() or not (pred or "").strip():
            continue
        try:
            result = calc_codebleu([ref], [pred], lang=lang)
            scores.append(result["codebleu"])
        except Exception:
            # Malformed generations can trip the AST/dataflow matchers; skip rather than fail the run.
            continue
    return sum(scores) / len(scores) if scores else None


def _accepts_tests(generate_fn) -> bool:
    """Whether generate_fn takes the test list as a second argument."""
    try:
        params = inspect.signature(generate_fn).parameters
    except (TypeError, ValueError):
        return False
    if any(p.kind is inspect.Parameter.VAR_POSITIONAL for p in params.values()):
        return True
    positional = [
        p
        for p in params.values()
        if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    return len(positional) >= 2


def evaluate_code_batch(
    examples: List[Dict],
    generate_fn,
    max_samples: Optional[int] = None,
) -> Dict:
    """Run pass@1 over a list of MBPP-style examples."""
    subset = examples[:max_samples] if max_samples else examples
    pass_tests = _accepts_tests(generate_fn)
    per_task = []
    for ex in subset:
        prompt = ex.get("prompt", "")
        tests = ex.get("test_list") or []
        if not tests and ex.get("test_template"):
            tests = [line for line in ex["test_template"].split("\n") if line.strip()]
        generated = generate_fn(prompt, tests) if pass_tests else generate_fn(prompt)
        ok = evaluate_functional_correctness(generated, tests) if tests else False
        per_task.append(
            {
                "id": ex.get("id"),
                "prompt": prompt,
                "generated_code": clean_generated_code(generated),
                "reference_code": ex.get("code") or ex.get("canonical_solution") or "",
                "passed": ok,
            }
        )
    score = pass_at_1([r["passed"] for r in per_task])
    codebleu = compute_codebleu([(r["reference_code"], r["generated_code"]) for r in per_task])
    return {
        "pass_at_1": score,
        "codebleu": codebleu,
        "n_samples": len(per_task),
        "n_passed": sum(1 for r in per_task if r["passed"]),
        "details": per_task,
    }
