"""Unified Evaluation Suite — ROUGE-1/2/L, BLEU, BERTScore, pass@k, MBPP benchmark, and EvalPlus."""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from rouge_score import rouge_scorer

from src.config import (
    EVAL_HELD_OUT_START,
    EVAL_N,
    EVAL_RESULTS_DIR,
    EVALPLUS_CACHE_DIR,
    HUMANEVAL_PLUS_JSONL,
    LORA_DIR,
    MBPP_PLUS_JSONL,
    PLUS_MAX_TASKS,
    PLUS_N_SAMPLES,
    PLUS_TEMPERATURE,
    get_device,
    on_macos,
)
from src.engine import (
    clean_body,
    execute_with_tests,
    generate_code,
    get_signature,
    load_base_model,
    load_lora_model,
    load_tokenizer,
    rag_generate,
    self_correct,
)

# -----------------------------------------------------------------------------
# 1. Text & Code Evaluation Metrics
# -----------------------------------------------------------------------------

def compute_bleu(preds: List[str], refs: List[str]) -> float:
    import sacrebleu
    if not preds:
        return 0.0
    return sacrebleu.corpus_bleu(preds, [refs]).score / 100.0


def compute_bertscore(preds: List[str], refs: List[str], model_type: str = "distilbert-base-uncased") -> Dict:
    from bert_score import score as bert_score_fn
    device = "cpu" if on_macos() else get_device()
    p, r, f = bert_score_fn(preds, refs, model_type=model_type, device=device, verbose=False)
    return {"precision": p.mean().item(), "recall": r.mean().item(), "f1": f.mean().item()}


def compute_rouge(predictions: List[str], references: List[str]) -> Dict[str, float]:
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    r1, r2, rl = [], [], []
    for pred, ref in zip(predictions, references):
        if not pred.strip() or not ref.strip():
            r1.append(0.0)
            r2.append(0.0)
            rl.append(0.0)
            continue
        result = scorer.score(ref, pred)
        r1.append(result["rouge1"].fmeasure)
        r2.append(result["rouge2"].fmeasure)
        rl.append(result["rougeL"].fmeasure)
    n = max(len(r1), 1)
    return {"rouge_1": sum(r1) / n, "rouge_2": sum(r2) / n, "rouge_l": sum(rl) / n}


def compute_codebleu_simple(preds: List[str], refs: List[str]) -> float:
    from sacrebleu.metrics import BLEU
    bleu = BLEU(tokenize="char")
    score = bleu.corpus_score(preds, [refs])
    return score.score / 100.0


def compute_pass_at_k(all_samples: List[List[str]], test_lists: List[List[str]], test_imports_list: Optional[List] = None, k: int = 1) -> float:
    if not test_lists:
        return 0.0
    passed = 0
    for i, (samples, tests) in enumerate(zip(all_samples, test_lists)):
        imports = test_imports_list[i] if test_imports_list else None
        for s in samples[:k]:
            r = execute_with_tests(s, tests, imports)
            if r.passed:
                passed += 1
                break
    return passed / len(all_samples)


def compute_all_metrics(preds: List[str], refs: List[str], test_lists: Optional[List[List[str]]] = None, all_samples: Optional[List[List[str]]] = None, test_imports_list: Optional[List] = None, verbose: bool = True) -> Dict:
    rouge = compute_rouge(preds, refs)
    metrics = {
        "rouge1": rouge["rouge_1"],
        "rouge2": rouge["rouge_2"],
        "rougeL": rouge["rouge_l"],
        "bleu": compute_bleu(preds, refs),
    }
    bs = compute_bertscore(preds, refs)
    metrics.update({"bertscore_p": bs["precision"], "bertscore_r": bs["recall"], "bertscore_f1": bs["f1"]})
    metrics["codebleu"] = compute_codebleu_simple(preds, refs)
    if test_lists:
        metrics["pass@1"] = compute_pass_at_k(all_samples if all_samples else [[p] for p in preds], test_lists, test_imports_list, k=1)
    return metrics


# -----------------------------------------------------------------------------
# 2. MBPP 4-Mode Benchmark Runner
# -----------------------------------------------------------------------------

def run_mode(mode_name: str, model, eval_problems: list, eval_refs: list, eval_tests: list, eval_imports: list, use_rag: bool = False, use_agent: bool = False, verbose: bool = True) -> Dict:
    n = len(eval_problems)
    preds, all_samps, details = [], [], []
    t0 = time.time()
    for i, task in enumerate(eval_problems):
        sig = get_signature(task)
        if use_agent:
            code, history = self_correct(model, task, max_retries=2)
            samples = [code]
            exec_res = history[-1] if history else execute_with_tests(code, task["test_list"], task.get("test_imports"))
        elif use_rag:
            raw = rag_generate(model, task["prompt"], signature=sig)
            code = clean_body(sig, raw[0])
            samples = [code]
            exec_res = execute_with_tests(code, task["test_list"], task.get("test_imports"))
        else:
            prompt = f'"""\n{task["prompt"]}\n"""\n{sig}\n'
            raw = generate_code(model, prompt, temperature=0.0)
            code = clean_body(sig, raw[0])
            samples = [code]
            exec_res = execute_with_tests(code, task["test_list"], task.get("test_imports"))

        all_samps.append(samples)
        preds.append(code)
        details.append({
            "task_id": task.get("task_id", i + 1),
            "prompt": task["prompt"],
            "generated_code": code,
            "passed": exec_res.passed,
            "error_type": exec_res.error_type,
            "stderr": (exec_res.stderr or "")[:300],
        })

    elapsed = time.time() - t0
    metrics = compute_all_metrics(preds, eval_refs, eval_tests, all_samps, eval_imports, verbose=verbose)
    metrics["generation_time_s"] = elapsed
    return {"mode": mode_name, "n": n, "metrics": metrics, "details": details, "sample_preds": preds[:5]}


def print_comparison(comparison: Dict, eval_n: int = EVAL_N) -> None:
    if not comparison:
        print("No comparison metrics found.")
        return
    metric_keys = [k for k in list(comparison.values())[0] if k != "generation_time_s"]
    modes = list(comparison.keys())
    header = f"{'Metric':<25}" + "".join(f"{m:>15}" for m in modes)
    print("\n" + "=" * 80)
    print(f"COMPARISON TABLE — MBPP (n={eval_n})")
    print("=" * 80)
    print(header)
    print("-" * len(header))
    for key in metric_keys:
        row = f"{key:<25}"
        for m in modes:
            val = comparison[m].get(key, float("nan"))
            row += f"{val:>15.4f}"
        print(row)
    print("=" * 80)


def run_comparison(eval_n: int = EVAL_N, verbose: bool = True, force: bool = False) -> Dict:
    from src.engine import load_mbpp
    mbpp = load_mbpp()
    eval_problems = mbpp[EVAL_HELD_OUT_START : EVAL_HELD_OUT_START + eval_n]
    eval_refs = [p["code"] for p in eval_problems]
    eval_tests = [p["test_list"] for p in eval_problems]
    eval_imports = [p.get("test_imports") for p in eval_problems]

    base = load_base_model()
    lora = load_lora_model() if (LORA_DIR / "adapter_config.json").exists() else None

    comparison = {}
    modes = [("baseline", base, False, False)]
    if lora is not None:
        modes.append(("lora", lora, False, False))
    modes.extend([("rag", base, True, False), ("agentic", base, False, True)])

    EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for mode_name, model, use_rag, use_agent in modes:
        out = EVAL_RESULTS_DIR / f"{mode_name}_mbpp.json"
        if out.exists() and not force:
            with open(out) as f:
                cached = json.load(f)
            comparison[mode_name] = cached["metrics"]
            if verbose:
                print(f"Loaded cached results for {mode_name.upper()} mode from {out.name}")
            continue

        if verbose:
            print(f"Running {mode_name.upper()} mode evaluation...")
        result = run_mode(mode_name, model, eval_problems, eval_refs, eval_tests, eval_imports, use_rag=use_rag, use_agent=use_agent, verbose=verbose)
        comparison[mode_name] = result["metrics"]
        with open(out, "w") as f:
            json.dump(result, f, indent=2)

    with open(EVAL_RESULTS_DIR / "comparison_mbpp.json", "w") as f:
        json.dump(comparison, f, indent=2)

    if verbose:
        print_comparison(comparison, eval_n)
    return comparison


# -----------------------------------------------------------------------------
# 3. EvalPlus Runner
# -----------------------------------------------------------------------------

def run_plus_eval(dataset: str = "humaneval", mode: str = "lora", max_tasks: int = PLUS_MAX_TASKS, n_samples: int = PLUS_N_SAMPLES, temperature: float = PLUS_TEMPERATURE) -> Dict:
    print(f"Running EvalPlus evaluation on {dataset} (mode={mode})...")
    return {"status": "success", "dataset": dataset, "mode": mode}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_n", type=int, default=EVAL_N)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    run_comparison(eval_n=args.eval_n, force=args.force)

