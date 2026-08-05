"""Code generation evaluation: execution accuracy and pass@1."""

from __future__ import annotations

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
    """Keep the first top-level function; drop hallucinated follow-on defs."""
    lines = raw_code.split("\n")
    cleaned: List[str] = []
    for i, line in enumerate(lines):
        stripped = line.lstrip()
        if i > 0 and (stripped.startswith("def ") or stripped.startswith("if __name__")):
            break
        cleaned.append(line)
    return "\n".join(cleaned).strip()


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


def evaluate_code_batch(
    examples: List[Dict],
    generate_fn,
    max_samples: Optional[int] = None,
) -> Dict:
    """Run pass@1 over a list of MBPP-style examples."""
    subset = examples[:max_samples] if max_samples else examples
    per_task = []
    for ex in subset:
        prompt = ex.get("prompt", "")
        tests = ex.get("test_list") or []
        if not tests and ex.get("test_template"):
            tests = [line for line in ex["test_template"].split("\n") if line.strip()]
        generated = generate_fn(prompt)
        ok = evaluate_functional_correctness(generated, tests) if tests else False
        per_task.append(
            {
                "id": ex.get("id"),
                "prompt": prompt,
                "generated_code": clean_generated_code(generated),
                "passed": ok,
            }
        )
    score = pass_at_1([r["passed"] for r in per_task])
    return {
        "pass_at_1": score,
        "n_samples": len(per_task),
        "n_passed": sum(1 for r in per_task if r["passed"]),
        "details": per_task,
    }
