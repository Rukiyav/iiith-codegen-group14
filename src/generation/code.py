"""Natural language → Python generation."""

from __future__ import annotations

from typing import Optional, Sequence

from src.eval.code_eval import clean_generated_code
from src.generation.engine import get_generator
from src.prompts import code_prompt

DEFAULT_CODE_MAX_TOKENS = 300
DEFAULT_CODE_MIN_TOKENS = 64


def generate_python_code(
    prompt_text: str,
    model_name_or_path: str | None = None,
    max_new_tokens: int = DEFAULT_CODE_MAX_TOKENS,
    temperature: float = 0.0,
    clean: bool = True,
    test_list: Optional[Sequence[str]] = None,
    min_new_tokens: int = DEFAULT_CODE_MIN_TOKENS,
    context: str = "",
    seed: int | None = 42,
) -> str:
    """context: optional few-shot block (e.g. from src.rag.augment) prepended before the prompt.

    seed: forwarded to the sampler when temperature > 0. Callers that want distinct
    candidates from repeated calls at the same prompt/temperature (e.g. best-of-N,
    agent retries) must vary this themselves -- the engine reseeds every call.
    """
    generator = get_generator(model_name_or_path)
    raw = generator.generate(
        context + code_prompt(prompt_text, test_list),
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        min_new_tokens=min_new_tokens,
        seed=seed,
    )
    return clean_generated_code(raw) if clean else raw
