"""Python code → documentation generation."""

from __future__ import annotations

from pathlib import Path

from src.generation.doc_postprocess import fallback_documentation, finalize_documentation
from src.generation.engine import DOC_STOP_STRINGS, get_generator
from src.models.registry import DEFAULT_MODEL_ID, resolve_hub_path, resolve_model_spec
from src.prompts import doc_prompt
from src.training_utils import truncate_text_to_tokens

DEFAULT_DOC_MAX_TOKENS = 64
PROMPT_OVERHEAD_TOKENS = 32


def _is_doc_finetuned(model_name_or_path: str | None) -> bool:
    path = model_name_or_path or DEFAULT_MODEL_ID
    if "codegen-doc" not in path and "doc-cp" not in path:
        return False
    checkpoint = Path(resolve_hub_path(path))
    return checkpoint.is_dir() and (checkpoint / "config.json").exists()


def _should_use_lm(model_name_or_path: str | None) -> bool:
    if _is_doc_finetuned(model_name_or_path):
        return True
    resolved = resolve_hub_path(model_name_or_path)
    if Path(resolved).is_dir():
        return True
    spec = resolve_model_spec(model_name_or_path)
    return spec is not None and spec.id != DEFAULT_MODEL_ID


def generate_documentation(
    code: str,
    model_name_or_path: str | None = None,
    max_new_tokens: int = DEFAULT_DOC_MAX_TOKENS,
    temperature: float = 0.0,
    context: str = "",
) -> str:
    """context: optional few-shot block (e.g. from src.rag.augment) prepended before the prompt."""
    if not _should_use_lm(model_name_or_path):
        return fallback_documentation(code)

    generator = get_generator(model_name_or_path)
    generator.load()
    assert generator._tokenizer is not None
    max_code_tokens = max(
        64,
        generator._max_context() - max_new_tokens - PROMPT_OVERHEAD_TOKENS,
    )
    truncated_code = truncate_text_to_tokens(code, generator._tokenizer, max_code_tokens)
    raw = generator.generate(
        context + doc_prompt(truncated_code),
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        stop_strings=DOC_STOP_STRINGS,
    )
    return finalize_documentation(truncated_code, raw)
