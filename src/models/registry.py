"""Preset models and path resolution for generation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[2]
CHECKPOINTS_DIR = ROOT / "experiments" / "checkpoints"

DEFAULT_MODEL_ID = "codegen-350m"


@dataclass(frozen=True)
class ModelSpec:
    id: str
    label: str
    hub_id: str
    size: str


PRESET_MODELS: dict[str, ModelSpec] = {
    "codegen-350m": ModelSpec(
        id="codegen-350m",
        label="CodeGen-350M-multi (upstream)",
        hub_id="Salesforce/codegen-350M-multi",
        size="350M",
    ),
}


def resolve_model_spec(model: str | None) -> Optional[ModelSpec]:
    if not model:
        return PRESET_MODELS[DEFAULT_MODEL_ID]
    if model in PRESET_MODELS:
        return PRESET_MODELS[model]
    return None


def resolve_hub_path(model: str | None) -> str:
    spec = resolve_model_spec(model)
    if spec:
        return spec.hub_id
    return model or PRESET_MODELS[DEFAULT_MODEL_ID].hub_id


def _list_local_checkpoints() -> list[ModelSpec]:
    if not CHECKPOINTS_DIR.is_dir():
        return []
    options: list[ModelSpec] = []
    for path in sorted(CHECKPOINTS_DIR.iterdir()):
        is_full = (path / "config.json").is_file()
        is_lora = (path / "adapter_config.json").is_file()
        if not path.is_dir() or not (is_full or is_lora):
            continue
        rel = path.relative_to(ROOT).as_posix()
        task = "docs" if "doc" in path.name else "code"
        kind = "LoRA" if is_lora else "fine-tuned"
        options.append(
            ModelSpec(
                id=rel,
                label=f"{kind} ({task}): {path.name}",
                hub_id=rel,
                size=kind.lower(),
            )
        )
    return options


def list_model_options(include_checkpoints: bool = True) -> list[dict]:
    rows = [spec for spec in PRESET_MODELS.values()]
    if include_checkpoints:
        rows.extend(_list_local_checkpoints())
    return [
        {
            "id": spec.id,
            "label": spec.label,
            "hub_id": spec.hub_id,
            "size": spec.size,
        }
        for spec in rows
    ]
