"""Agentic self-correction for code generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from src.eval.code_eval import clean_generated_code, evaluate_functional_correctness
from src.generation.code import DEFAULT_CODE_MAX_TOKENS, generate_python_code


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
    max_new_tokens: int = DEFAULT_CODE_MAX_TOKENS,
    temperature: float = 0.0,
    retry_temperature: float = 0.0,
    num_candidates: int = 1,
    use_failure_context: bool = True,
) -> AgentResult:
    """
    Agentic loop: generate → execute tests → retry.

    Attempt 1 always uses `temperature` (deterministic by default). Retries use
    `retry_temperature`; when it's > 0, `num_candidates` distinct-seed samples are
    drawn for that attempt (best-of-N) and the first to pass is kept. Defaults
    (`retry_temperature=0.0`, `num_candidates=1`) reproduce the original
    single-greedy-attempt behavior byte-for-byte.

    `use_failure_context` (default True, matches original behavior) puts the failed
    code from the previous attempt into the retry prompt. In practice this backfires
    on a small, non-instruction-tuned model like CodeGen-350M: shown its own recent
    code as "context to continue," the model tends to echo it back near-verbatim
    rather than fix it, and `clean_generated_code` then grabs that echoed (still
    broken) block as the answer -- which is why retry_success_rate measured 0%.
    Set `use_failure_context=False` to instead resample the *original* prompt with a
    fresh seed each retry (no failure text shown); this reliably produces distinct
    candidates -- verified directly, same prompt/temperature/model, only the seed
    differs -- and is the mode that actually moves retry_success_rate off zero.
    """
    if max_retries < 1:
        raise ValueError("max_retries must be >= 1")

    steps: List[AgentStep] = []
    current_prompt = prompt
    last_code = ""

    for attempt in range(1, max_retries + 1):
        is_retry = attempt > 1
        attempt_temperature = retry_temperature if is_retry else temperature
        candidates = num_candidates if (is_retry and attempt_temperature > 0) else 1

        code, passed = "", False
        for i in range(candidates):
            seed = 1000 * attempt + i
            raw = generate_python_code(
                current_prompt,
                model_name_or_path=model_name_or_path,
                max_new_tokens=max_new_tokens,
                temperature=attempt_temperature,
                test_list=test_list,
                seed=seed,
            )
            candidate_code = clean_generated_code(raw)
            candidate_passed = (
                evaluate_functional_correctness(candidate_code, test_list) if test_list else False
            )
            if i == 0:
                code, passed = candidate_code, candidate_passed
            if candidate_passed:
                code, passed = candidate_code, candidate_passed
                break

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
        if attempt < max_retries and use_failure_context:
            current_prompt = _retry_prompt(prompt, code, attempt)

    return AgentResult(
        final_code=last_code,
        passed=False,
        attempts=max_retries,
        steps=steps,
    )
