"""Natural language → Python generation."""

from __future__ import annotations

from src.eval.code_eval import clean_generated_code
from src.generation.engine import get_generator
from src.prompts import code_prompt


def generate_python_code(
    prompt_text: str,
    model_name_or_path: str | None = None,
    max_new_tokens: int = 150,
    temperature: float = 0.0,
    clean: bool = True,
) -> str:
    generator = get_generator(model_name_or_path)
    raw = generator.generate(
        code_prompt(prompt_text),
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )
    return clean_generated_code(raw) if clean else raw
