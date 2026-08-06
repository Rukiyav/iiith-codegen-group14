"""Evaluation CLI and JSON result logging."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.agent.code_agent import agent_generate_code
from src.data.codoc import load_codoc_split
from src.data.mbpp import load_processed
from src.eval.code_eval import compute_codebleu, evaluate_code_batch, pass_at_1
from src.eval.doc_eval import evaluate_doc_batch
from src.generation.code import generate_python_code
from src.generation.docs import generate_documentation
from src.models.registry import DEFAULT_MODEL_ID, resolve_hub_path
from src.rag.augment import build_code_context, build_doc_context
from src.rag.retrieve import retrieve_code_examples, retrieve_doc_examples

DEFAULT_MODEL = resolve_hub_path(DEFAULT_MODEL_ID)

ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "experiments" / "results"


def _write_result(payload: Dict[str, Any], output: Optional[Path] = None) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if output is None:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        task = payload.get("task", "eval")
        model_slug = payload.get("model_id", "model").replace("/", "_")
        output = RESULTS_DIR / f"{task}_{model_slug}_{ts}.json"
    with output.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return output


def run_code_eval(
    model_name_or_path: str,
    split: str = "test",
    max_samples: Optional[int] = None,
    use_agent: bool = False,
    max_retries: int = 3,
    use_rag: bool = False,
    rag_k: int = 2,
    retry_temperature: float = 0.0,
    num_candidates: int = 1,
    use_failure_context: bool = True,
) -> Dict[str, Any]:
    examples = load_processed(split)
    subset = examples[:max_samples] if max_samples else examples

    if not use_agent:

        def _gen(prompt: str, tests: Optional[List[str]] = None) -> str:
            context = ""
            if use_rag:
                context = build_code_context(retrieve_code_examples(prompt, k=rag_k))
            return generate_python_code(
                prompt,
                model_name_or_path=model_name_or_path,
                test_list=tests,
                context=context,
            )

        result = evaluate_code_batch(subset, _gen, max_samples=None)
        return {
            "task": "code",
            "model_id": model_name_or_path,
            "dataset": "mbpp",
            "split": split,
            "mode": "rag" if use_rag else "baseline",
            "metrics": {"pass_at_1": result["pass_at_1"], "codebleu": result["codebleu"]},
            "n_samples": result["n_samples"],
            "n_passed": result["n_passed"],
            "details": result["details"],
        }

    details = []
    retry_successes = 0
    for ex in subset:
        tests = ex.get("test_list") or []
        agent_result = agent_generate_code(
            ex.get("prompt", ""),
            tests,
            model_name_or_path=model_name_or_path,
            max_retries=max_retries,
            retry_temperature=retry_temperature,
            num_candidates=num_candidates,
            use_failure_context=use_failure_context,
        )
        if agent_result.passed and agent_result.attempts > 1:
            retry_successes += 1
        details.append(
            {
                "id": ex.get("id"),
                "prompt": ex.get("prompt"),
                "generated_code": agent_result.final_code,
                "reference_code": ex.get("code") or ex.get("canonical_solution") or "",
                "passed": agent_result.passed,
                "attempts": agent_result.attempts,
            }
        )

    passed_flags = [d["passed"] for d in details]
    codebleu = compute_codebleu([(d["reference_code"], d["generated_code"]) for d in details])
    return {
        "task": "code",
        "model_id": model_name_or_path,
        "dataset": "mbpp",
        "split": split,
        "mode": "agent",
        "max_retries": max_retries,
        "metrics": {
            "pass_at_1": pass_at_1(passed_flags),
            "codebleu": codebleu,
            "retry_success_rate": retry_successes / len(subset) if subset else 0.0,
            "avg_attempts": sum(d["attempts"] for d in details) / len(subset) if subset else 0.0,
        },
        "n_samples": len(subset),
        "n_passed": sum(1 for p in passed_flags if p),
        "details": details,
    }


def run_doc_eval(
    model_name_or_path: str,
    split: str = "test",
    max_samples: Optional[int] = None,
    include_bertscore: bool = False,
    use_rag: bool = False,
    rag_k: int = 2,
) -> Dict[str, Any]:
    examples = load_codoc_split(split)

    def _gen(code: str) -> str:
        context = ""
        if use_rag:
            context = build_doc_context(retrieve_doc_examples(code, k=rag_k))
        return generate_documentation(code, model_name_or_path=model_name_or_path, context=context)

    result = evaluate_doc_batch(
        examples,
        _gen,
        max_samples=max_samples,
        include_bertscore=include_bertscore,
    )
    return {
        "task": "docs",
        "model_id": model_name_or_path,
        "dataset": "codocbench",
        "split": split,
        "mode": "rag" if use_rag else "baseline",
        "metrics": result["metrics"],
        "n_samples": result["n_samples"],
        "details": result["details"],
    }


def run_eval(args: argparse.Namespace) -> Path:
    config_hash = hashlib.md5(
        f"{args.task}:{args.model}:{args.split}:{args.max_samples}:{args.agent}:{args.rag}:"
        f"{args.retry_temperature}:{args.num_candidates}".encode()
    ).hexdigest()[:8]

    if args.task == "code":
        payload = run_code_eval(
            args.model,
            args.split,
            args.max_samples,
            use_agent=args.agent,
            max_retries=args.max_retries,
            use_rag=args.rag,
            rag_k=args.rag_k,
            retry_temperature=args.retry_temperature,
            num_candidates=args.num_candidates,
            use_failure_context=not args.no_failure_context,
        )
    elif args.task == "docs":
        payload = run_doc_eval(
            args.model,
            args.split,
            args.max_samples,
            include_bertscore=args.bertscore,
            use_rag=args.rag,
            rag_k=args.rag_k,
        )
    else:
        raise ValueError(f"Unknown task: {args.task}")

    payload["timestamp"] = datetime.now(timezone.utc).isoformat()
    payload["config_hash"] = config_hash
    out = _write_result(payload, Path(args.output) if args.output else None)
    print(json.dumps({"output": str(out), "metrics": payload.get("metrics")}, indent=2))
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run code or documentation evaluation.")
    parser.add_argument("--task", choices=["code", "docs"], required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL_ID, help="Preset id, hub id, or checkpoint path")
    parser.add_argument(
        "--split",
        default="test",
        help="MBPP: train|validation|test|all (all = notebook 427-task eval)",
    )
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--output", default=None, help="Optional output JSON path")
    parser.add_argument("--bertscore", action="store_true", help="Include BERTScore (slow)")
    parser.add_argument(
        "--agent",
        action="store_true",
        help="Use agentic self-correction for code eval (generate → test → retry)",
    )
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument(
        "--retry_temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for agent retries (0 = greedy, matches original behavior)",
    )
    parser.add_argument(
        "--num_candidates",
        type=int,
        default=1,
        help="Best-of-N candidates per agent retry when --retry_temperature > 0",
    )
    parser.add_argument(
        "--no_failure_context",
        action="store_true",
        help="Resample the original prompt on retries instead of showing the model its failed code "
        "(the failure-context prompt makes small non-instruction-tuned models echo the same bug back)",
    )
    parser.add_argument(
        "--rag",
        action="store_true",
        help="Retrieval-augmented generation: prepend top-k similar train examples as few-shot context",
    )
    parser.add_argument("--rag_k", type=int, default=2, help="Number of retrieved examples for --rag")
    return parser.parse_args()


def main() -> None:
    run_eval(parse_args())


if __name__ == "__main__":
    main()
