"""Canonical prompt templates shared by train, infer, eval, and API."""

from typing import Optional, Sequence


def code_prompt(nl: str, tests: Optional[Sequence[str]] = None) -> str:
    """
    Natural language → Python generation prefix (CodeGen docstring style).

    The asserts are included so the model emits the function name and arity the
    tests call; without them pass@1 is capped near zero by NameError.
    """
    body = f'"""\n{nl.strip()}\n'
    if tests:
        assert_lines = [t.strip() for t in tests if t and t.strip()]
        if assert_lines:
            body += "\nYour code should satisfy these tests:\n"
            body += "\n".join(assert_lines) + "\n"
    return body + '"""\n'


def doc_prompt(code: str) -> str:
    """Python code → documentation prefix (comment block; matches CodeGen pretraining)."""
    return f"{code.strip()}\n\n# Description:\n"


def code_training_text(
    prompt: str,
    solution: str,
    tests: Optional[Sequence[str]] = None,
) -> str:
    return code_prompt(prompt, tests) + solution.strip()


def doc_training_text(code: str, documentation: str) -> str:
    from src.generation.doc_postprocess import documentation_to_comments

    return doc_prompt(code) + documentation_to_comments(documentation) + "\n"
