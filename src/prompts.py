"""Canonical prompt templates shared by train, infer, eval, and API."""


def code_prompt(nl: str) -> str:
    """Natural language → Python generation prefix (CodeGen docstring style)."""
    return f'"""\n{nl.strip()}\n"""\n'


def doc_prompt(code: str) -> str:
    """Python code → documentation prefix (comment block; matches CodeGen pretraining)."""
    return f"{code.strip()}\n\n# Description:\n"


def code_training_text(prompt: str, solution: str) -> str:
    return code_prompt(prompt) + solution.strip()


def doc_training_text(code: str, documentation: str) -> str:
    from src.generation.doc_postprocess import documentation_to_comments

    return doc_prompt(code) + documentation_to_comments(documentation) + "\n"
