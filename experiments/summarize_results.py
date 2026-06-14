"""Summarize eval JSON artifacts into a comparison table for reports."""

from __future__ import annotations

import json
import sys
from glob import glob
from pathlib import Path
from typing import Any, Dict, List


def _load(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _row(payload: Dict[str, Any], source: str) -> Dict[str, str]:
    task = payload.get("task", "?")
    metrics = payload.get("metrics") or {}
    mode = payload.get("mode", "baseline")
    if task == "code":
        primary = f"{metrics.get('pass_at_1', 0):.2%}"
        secondary = (
            f"{metrics.get('retry_success_rate', 0):.2%}"
            if "retry_success_rate" in metrics
            else "-"
        )
    elif task == "docs":
        primary = f"{metrics.get('rouge_l', 0):.4f}"
        secondary = f"{metrics.get('bleu', 0):.4f}"
    else:
        primary = secondary = "-"
    return {
        "source": source,
        "task": task,
        "split": str(payload.get("split", "")),
        "n": str(payload.get("n_samples", "")),
        "mode": mode,
        "primary": primary,
        "secondary": secondary,
    }


def summarize(paths: List[Path]) -> None:
    rows: List[Dict[str, str]] = []
    for path in sorted(set(paths)):
        if not path.is_file():
            continue
        try:
            rows.append(_row(_load(path), path.name))
        except (json.JSONDecodeError, OSError) as exc:
            print(f"skip {path}: {exc}", file=sys.stderr)

    if not rows:
        print("No result files found.")
        return

    print("\n=== Eval summary ===")
    print(f"{'file':<36} {'task':<6} {'split':<6} {'n':<5} {'mode':<10} {'primary':<10} secondary")
    print("-" * 95)
    for r in rows:
        print(
            f"{r['source']:<36} {r['task']:<6} {r['split']:<6} {r['n']:<5} "
            f"{r['mode']:<10} {r['primary']:<10} {r['secondary']}"
        )
    print("\nCode: primary = pass@1 · Docs: primary = ROUGE-L · Agent: secondary = retry_success_rate")


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python experiments/summarize_results.py experiments/results/cp2_*.json")
        sys.exit(1)
    paths: List[Path] = []
    for arg in sys.argv[1:]:
        matches = glob(arg)
        if matches:
            paths.extend(Path(p) for p in matches)
        else:
            paths.append(Path(arg))
    summarize(paths)


if __name__ == "__main__":
    main()
