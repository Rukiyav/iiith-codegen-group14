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
import tempfile
import subprocess
import sys

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
    generate_code_for_task,
    SANDBOX_SECURITY_PRELUDE,
    COMMON_IMPORTS,
    ExecResult,
    get_backend_for_model,
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


def compute_char_bleu(preds: List[str], refs: List[str]) -> float:
    from sacrebleu.metrics import BLEU
    bleu = BLEU(tokenize="char")
    score = bleu.corpus_score(preds, [refs])
    return score.score / 100.0


def compute_pass_at_k(all_samples: List[List[str]], test_lists: List[List[str]], test_imports_list: Optional[List] = None, k: int = 1) -> float:
    if not test_lists:
        return 0.0
    import math
    def choose(n, r):
        if n < r or r < 0:
            return 0
        return math.comb(n, r)

    estimates = []
    for i, (samples, tests) in enumerate(zip(all_samples, test_lists)):
        n = len(samples)
        imports = test_imports_list[i] if test_imports_list else None
        c = 0
        for s in samples:
            r = execute_with_tests(s, tests, imports)
            if r.passed:
                c += 1
        
        if n - c < k:
            estimates.append(1.0)
        else:
            num = choose(n - c, k)
            den = choose(n, k)
            if den > 0:
                estimates.append(1.0 - (num / den))
            else:
                estimates.append(0.0)
    return sum(estimates) / len(estimates) if estimates else 0.0


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
    metrics["char_bleu"] = compute_char_bleu(preds, refs)
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
            code, history, best_res = self_correct(model, task, max_retries=2, use_rag=use_rag)
            samples = [code]
            exec_res = best_res if best_res else (history[-1] if history else execute_with_tests(code, task["test_list"], task.get("test_imports")))
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

    comparison = {}
    from src.config import BENCHMARK_MODELS, BENCHMARK_MODES, MODEL_REGISTRY, EMBED_MODEL
    
    # Unified evaluation pipeline configuration dictionary (affects benchmark results)
    eval_config = {
        "dataset_slice": {
            "eval_n": eval_n,
            "eval_held_out_start": EVAL_HELD_OUT_START
        },
        "llm_generation": {
            "temperature": 0.0,
            "max_new_tokens": 512,
        },
        "rag": {
            "top_k": 3,
            "min_score_cutoff_project": 0.35,
            "min_score_cutoff_default": 0.45,
            "embedding_model": EMBED_MODEL
        },
        "agentic": {
            "max_retries": 2,
            "n_candidates": 3,
        },
        "prompt_versions": {
            "build_prompt_version": "v1.0",
            "self_correct_template_version": "v1.0"
        }
    }
    
    configs = []
    for model_name in BENCHMARK_MODELS:
        if model_name not in MODEL_REGISTRY:
            continue
        cfg = MODEL_REGISTRY[model_name]
        adapter = cfg.get("adapter")
        if adapter:
            adapter_path = Path(adapter["path"])
            if not (adapter_path / "adapter_config.json").exists():
                if verbose:
                    print(f"Skipping adapter model '{model_name}' (checkpoint missing under {adapter_path})")
                continue
        for mode_name in BENCHMARK_MODES:
            config_name = f"{model_name}_{mode_name}"
            use_rag = (mode_name == "rag")
            use_agent = (mode_name == "agentic")
            configs.append((config_name, model_name, mode_name, use_rag, use_agent))

    EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for config_name, model_name, mode_name, use_rag, use_agent in configs:
        out = EVAL_RESULTS_DIR / f"{config_name}_mbpp.json"
        
        cache_valid = False
        if out.exists() and not force:
            try:
                with open(out) as f:
                    cached = json.load(f)
                cfg = MODEL_REGISTRY.get(model_name, {})
                cached_cfg = cached.get("model_config", {})
                cached_eval_config = cached.get("eval_config", {})
                
                # Check critical registry settings & all parameters that affect results
                if (cached_cfg.get("model_id") == cfg.get("model_id") and
                    cached_cfg.get("backend") == cfg.get("backend") and
                    cached_cfg.get("prompt_style") == cfg.get("prompt_style") and
                    cached_cfg.get("adapter") == cfg.get("adapter") and
                    cached_eval_config == eval_config):
                    cache_valid = True
            except Exception:
                cache_valid = False
                
        if cache_valid:
            comparison[config_name] = cached["metrics"]
            if verbose:
                print(f"Loaded cached results for {config_name.upper()} from {out.name}")
            continue

        if verbose:
            print(f"Running {config_name.upper()} evaluation...")
        
        model = get_backend_for_model(model_name)
        
        result = run_mode(config_name, model, eval_problems, eval_refs, eval_tests, eval_imports, use_rag=use_rag, use_agent=use_agent, verbose=verbose)
        result["model_config"] = MODEL_REGISTRY.get(model_name, {})
        result["eval_config"] = eval_config
        comparison[config_name] = result["metrics"]
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

def verify_evalplus_solution(
    generated_code: str,
    canonical_code: str,
    entry_point: str,
    inputs: List,
    atol: float = 0.0
) -> ExecResult:
    # Build a secure validation script that runs inside the sandbox
    validation_script = (
        f"{SANDBOX_SECURITY_PRELUDE}\n\n"
        f"import sys, json, math\n"
        f"{COMMON_IMPORTS}\n\n"
        f"# --- Canonical Solution Namespace ---\n"
        f"canonical_globals = {{}}\n"
        f"exec('''{canonical_code}''', canonical_globals)\n"
        f"canonical_fn = canonical_globals.get('{entry_point}')\n\n"
        f"if not canonical_fn:\n"
        f"    # Fallback to search any function inside the globals\n"
        f"    canonical_fn = next((v for k, v in canonical_globals.items() if callable(v)), None)\n\n"
        f"# --- Generated Solution ---\n"
        f"{generated_code}\n"
        f"generated_fn = globals().get('{entry_point}')\n\n"
        f"if not generated_fn:\n"
        f"    print('Entry point {entry_point} not found in generated code.')\n"
        f"    sys.exit(1)\n\n"
        f"# --- Inputs Validation ---\n"
        f"inputs = {inputs}\n"
        f"atol = {atol}\n"
        f"results = []\n"
        f"for idx, inp in enumerate(inputs):\n"
        f"    try:\n"
        f"        if isinstance(inp, (list, tuple)):\n"
        f"            expected = canonical_fn(*inp)\n"
        f"            actual = generated_fn(*inp)\n"
        f"        else:\n"
        f"            expected = canonical_fn(inp)\n"
        f"            actual = generated_fn(inp)\n"
        f"        if atol > 0:\n"
        f"            correct = math.isclose(actual, expected, abs_tol=atol) if isinstance(actual, (int, float)) and isinstance(expected, (int, float)) else actual == expected\n"
        f"        else:\n"
        f"            correct = actual == expected\n"
        f"        if not correct:\n"
        f"            raise AssertionError(f'Expected {{expected}}, but got {{actual}}')\n"
        f"        results.append({{'index': idx, 'passed': True, 'error_msg': ''}})\n"
        f"    except Exception as e:\n"
        f"        results.append({{'index': idx, 'passed': False, 'error_msg': str(e)}})\n"
        f"print('__SANDBOX_RESULTS_START__')\n"
        f"print(json.dumps(results))\n"
        f"print('__SANDBOX_RESULTS_END__')\n"
        f"if any(not r['passed'] for r in results):\n"
        f"    sys.exit(1)\n"
    )

    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
        f.write(validation_script)
        tmp = f.name
    try:
        r = subprocess.run(
            [sys.executable, tmp],
            capture_output=True,
            text=True,
            timeout=15,
        )
        stdout = r.stdout
        stderr = r.stderr
        passed = r.returncode == 0
        
        test_results = []
        tests_passed = 0
        tests_total = len(inputs)
        
        if "__SANDBOX_RESULTS_START__" in stdout:
            parts = stdout.split("__SANDBOX_RESULTS_START__")
            if len(parts) > 1:
                subparts = parts[1].split("__SANDBOX_RESULTS_END__")
                if len(subparts) > 0:
                    try:
                        test_results = json.loads(subparts[0].strip())
                        tests_passed = sum(1 for res in test_results if res["passed"])
                    except Exception:
                        pass
        if not passed and not test_results:
            tests_passed = 0
            
        return ExecResult(
            passed=passed,
            stdout=stdout,
            stderr=stderr,
            error_type="pass" if passed else "assertion",
            tests_passed=tests_passed,
            tests_total=tests_total
        )
    except subprocess.TimeoutExpired:
        return ExecResult(False, "", "Timeout", "timeout", tests_passed=0, tests_total=len(inputs))
    finally:
        Path(tmp).unlink(missing_ok=True)


def run_plus_eval(
    dataset: str = "humaneval",
    model_name: str = "codegen",
    mode: str = "baseline",
    max_tasks: int = PLUS_MAX_TASKS,
    n_samples: int = PLUS_N_SAMPLES,
    temperature: float = PLUS_TEMPERATURE
) -> Dict:
    print(f"Running EvalPlus evaluation on {dataset} (model={model_name}, mode={mode}, n_samples={n_samples})...")
    # Load cache
    cache_path = MBPP_PLUS_JSONL if dataset.lower() == "mbpp" else HUMANEVAL_PLUS_JSONL
    if not cache_path.exists():
        return {"status": "error", "message": f"Dataset file not found: {cache_path}"}
        
    tasks = []
    with open(cache_path) as f:
        for line in f:
            if line.strip():
                tasks.append(json.loads(line))
                
    tasks = tasks[:max_tasks]
    print(f"Loaded {len(tasks)} tasks from {cache_path.name}")
    
    import math
    def choose(n, r):
        if n < r or r < 0:
            return 0
        return math.comb(n, r)

    results = []
    pass1_estimates = []
    pass5_estimates = []
    total_count = len(tasks)
    
    for i, t in enumerate(tasks):
        task_id = t["task_id"]
        prompt = t["prompt"]
        entry_point = t["entry_point"]
        canonical_solution = t["canonical_solution"]
        base_inputs = t.get("base_input", [])
        plus_inputs = t.get("plus_input", [])
        atol = t.get("atol", 0.0)
        
        # Generate n_samples candidates using dispatcher pipeline
        samples = []
        for _ in range(n_samples):
            res = generate_code_for_task(
                prompt,
                model_name=model_name,
                mode=mode,
                temperature=temperature if n_samples > 1 else 0.0,
                reference_code=canonical_solution,
                use_rag=False
            )
            samples.append(res.code)
            
        all_inputs = base_inputs + plus_inputs
        c = 0
        last_error = ""
        for code in samples:
            val_res = verify_evalplus_solution(code, canonical_solution, entry_point, all_inputs, atol)
            if val_res.passed:
                c += 1
            else:
                last_error = val_res.stderr or val_res.error_type
                
        # Estimate pass@1 and pass@5 for this task using unbiased estimator
        p1 = c / n_samples if n_samples > 0 else 0.0
        pass1_estimates.append(p1)
        
        if n_samples - c < 5:
            p5 = 1.0 if n_samples >= 5 else p1
        else:
            num = choose(n_samples - c, 5)
            den = choose(n_samples, 5)
            p5 = 1.0 - (num / den) if den > 0 else 0.0
        pass5_estimates.append(p5)
            
        results.append({
            "task_id": task_id,
            "passed_samples": c,
            "total_samples": n_samples,
            "pass@1": p1,
            "pass@5": p5,
            "error_msg": last_error if c < n_samples else ""
        })
        print(f"Task {task_id} ({i+1}/{total_count}): {c}/{n_samples} samples passed (pass@1={p1:.2f}, pass@5={p5:.2f})")
        
    pass1_avg = sum(pass1_estimates) / total_count if total_count > 0 else 0.0
    pass5_avg = sum(pass5_estimates) / total_count if total_count > 0 else 0.0
    summary = {
        "status": "success",
        "dataset": dataset,
        "model_name": model_name,
        "mode": mode,
        "pass@1": pass1_avg,
        "pass@5": pass5_avg,
        "total_tasks": total_count,
        "results": results
    }
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_n", type=int, default=EVAL_N)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    run_comparison(eval_n=args.eval_n, force=args.force)

