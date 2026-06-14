"""Agentic self-correction for code generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from src.eval.code_eval import clean_generated_code, evaluate_functional_correctness
from src.generation.code import generate_python_code


@dataclass
class AgentStep:
    attempt: int
    generated_code: str
    passed: bool
    prompt_used: str


@dataclass
class AgentResult:
    final_code: str
    passed: bool
    attempts: int
    steps: List[AgentStep] = field(default_factory=list)


def _retry_prompt(original: str, failed_code: str, attempt: int) -> str:
    return (
        f"{original.strip()}\n\n"
        f"The previous solution (attempt {attempt}) failed the unit tests. "
        f"Fix the logic and return a corrected Python function.\n"
        f"Previous code:\n{failed_code.strip()}"
    )


def agent_generate_code(
    prompt: str,
    test_list: List[str],
    model_name_or_path: Optional[str] = None,
    max_retries: int = 3,
    max_new_tokens: int = 150,
    temperature: float = 0.0,
) -> AgentResult:
    """
    Agentic loop: generate → execute tests → retry with failure context.
    """
    if max_retries < 1:
        raise ValueError("max_retries must be >= 1")

    steps: List[AgentStep] = []
    current_prompt = prompt
    last_code = ""

    for attempt in range(1, max_retries + 1):
        raw = generate_python_code(
            current_prompt,
            model_name_or_path=model_name_or_path,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        code = clean_generated_code(raw)
        passed = evaluate_functional_correctness(code, test_list) if test_list else False
        steps.append(
            AgentStep(
                attempt=attempt,
                generated_code=code,
                passed=passed,
                prompt_used=current_prompt,
            )
        )
        last_code = code
        if passed or not test_list:
            return AgentResult(
                final_code=code,
                passed=passed,
                attempts=attempt,
                steps=steps,
            )
        if attempt < max_retries:
            current_prompt = _retry_prompt(prompt, code, attempt)

    return AgentResult(
        final_code=last_code,
        passed=False,
        attempts=max_retries,
        steps=steps,
    )
