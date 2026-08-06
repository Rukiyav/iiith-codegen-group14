"""CLI for code generation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.generation.code import generate_python_code
from src.models.registry import DEFAULT_MODEL_ID, resolve_hub_path

DEFAULT_MODEL = resolve_hub_path(DEFAULT_MODEL_ID)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Python code from a natural language prompt.")
    parser.add_argument("prompt", nargs="+", help="Natural language prompt")
    parser.add_argument(
        "--model_name_or_path",
        default=DEFAULT_MODEL_ID,
        help="Preset id (codegen-350m), hub id, or checkpoint path",
    )
    parser.add_argument("--max_new_tokens", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.0, help="0 = greedy (deterministic)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prompt = " ".join(args.prompt)
    print(
        generate_python_code(
            prompt,
            model_name_or_path=args.model_name_or_path,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
    )


if __name__ == "__main__":
    main()
