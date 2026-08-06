# Codebase Guide — CodeGen Group14

This document describes every file and directory in the repository: what it does, how it fits into the capstone pipeline, and what you can safely ignore or regenerate.

**Scope (Checkpoint 2):** text → Python code (MBPP), code → documentation (CoDocBench), and agentic self-correction (generate → test → retry). SQL/RAG/VS Code extension are out of scope.

---

## High-level architecture

```
Natural language ──► src/generation/code.py ──► Python code
                              │
                              ▼
                    src/agent/code_agent.py (optional retries via MBPP tests)

Python code ──► src/generation/docs.py ──► Documentation

Training:  src/train.py (MBPP) · src/train_doc.py (CoDocBench)
Eval:      src/eval/run_eval.py · code_eval.py · doc_eval.py
Serving:   app/main.py (FastAPI) · ui/app.py (Streamlit)
```

---

## Root files

| File | Significance |
|------|--------------|
| `README.md` | Project overview, setup, CP1/CP2 commands, API/UI usage, and literature review table. Start here. |
| `requirements.txt` | Python dependencies: PyTorch, Transformers, FastAPI, Streamlit, ROUGE/BLEU metrics, pytest. |
| `.gitignore` | Ignores venv, caches, large binaries (checkpoints, `.safetensors`), and generated eval JSON under `experiments/results/`. |
| `capstone_project_checkpoint_1.ipynb` | Reference notebook for MBPP loading logic; `src/data/mbpp.py` mirrors its download and DataFrame workflow. |
| `AIML_PGCP_Project_Batch_26.pdf` | Official capstone specification (deliverables, metrics, timeline). |
| `CodeGen_Proposal_Group14.pdf` | Group proposal (datasets, models, module breakdown). |
| `SERA_Soft-VerifiedEfficient Repository Agents.pdf` | Literature review — informs agentic self-correction design; not fully implemented. |
| `Scaling Laws for Code Every Programming Language Matters.pdf` | Literature review — multi-language stretch goal for later checkpoints. |

---

## `src/` — Core library

### Top level

| File | Significance |
|------|--------------|
| `src/__init__.py` | Package marker (empty). |
| `src/prompts.py` | Shared prompt templates for code (`"""NL"""`) and docs (`code\n\n# Description:`). Used by train, infer, eval, and API for consistent formatting. |
| `src/training_utils.py` | SFT tokenization: masks prompt tokens in labels, truncates long inputs safely (fixes CoDocBench crashes on 2048+ token samples). |
| `src/train.py` | Fine-tunes CodeGen-350M-multi on MBPP. Supports subset mode (CP1) and `--full` (entire train split, CP2). Writes checkpoints to `experiments/checkpoints/`. |
| `src/train_doc.py` | Fine-tunes CodeGen-350M-multi on CoDocBench (code → comment-style documentation). Same CP1/CP2 flags as `train.py`. |
| `src/infer.py` | CLI for single-prompt code generation. Example: `python src/infer.py "factorial" --model_name_or_path codegen-350m`. |

### `src/data/`

| File | Significance |
|------|--------------|
| `src/data/__init__.py` | Package marker. |
| `src/data/mbpp.py` | Downloads sanitized MBPP JSON from Hugging Face, splits 80/10/10 into JSONL, and exposes `load_processed(split)` for train/val/test/all. Central data source for code training and eval. |
| `src/data/codoc.py` | Downloads official CoDocBench train/test JSONL from GitHub, extracts code/docstring pairs, holds out 10% of train for validation. Used by doc training and doc eval. |

### `src/generation/`

| File | Significance |
|------|--------------|
| `src/generation/__init__.py` | Package marker. |
| `src/generation/engine.py` | Lazy-loaded CodeGen causal LM: device selection (CUDA/MPS/CPU), generation with optional stop strings, model caching. |
| `src/generation/code.py` | NL → Python: builds code prompt, runs generator, cleans output via `clean_generated_code`. |
| `src/generation/docs.py` | Code → docs: uses LM for fine-tuned checkpoints; AST fallback for upstream 350M. Truncates long code before generation. |
| `src/generation/doc_postprocess.py` | Converts docstrings to `#` comments for training; strips hallucinated code from outputs; AST-based fallback documentation. |

### `src/agent/`

| File | Significance |
|------|--------------|
| `src/agent/__init__.py` | Package marker. |
| `src/agent/code_agent.py` | Agentic loop: generate → run MBPP unit tests → retry with failure context (max N attempts). Returns `AgentResult` with step history. |

### `src/eval/`

| File | Significance |
|------|--------------|
| `src/eval/__init__.py` | Package marker. |
| `src/eval/__main__.py` | Enables `python -m src.eval.run_eval`. |
| `src/eval/code_eval.py` | Executes generated code against MBPP `test_list`, computes **pass@1**, cleans/stops at extra `def` blocks, timeout-safe for API threads. |
| `src/eval/doc_eval.py` | Computes **ROUGE-L** (primary), BLEU, optional BERTScore for documentation quality. |
| `src/eval/run_eval.py` | Evaluation CLI and JSON logging. Tasks: `code` (baseline or `--agent`) and `docs`. Writes structured results to `experiments/results/`. |

### `src/models/`

| File | Significance |
|------|--------------|
| `src/models/__init__.py` | Package marker. |
| `src/models/registry.py` | Model presets (`codegen-350m` → Hub id), resolves local checkpoint paths, lists options for API/UI (`GET /models`). |

---

## `app/` — FastAPI backend

| File | Significance |
|------|--------------|
| `app/main.py` | REST API: `/generate/code`, `/generate/docs`, `/evaluate/run`, `/models`. Wires generation, agent, and eval modules. |
| `app/schemas.py` | Pydantic request/response models for all API endpoints (generation, agent steps, batch eval). |

---

## `ui/` — Streamlit demo

| File | Significance |
|------|--------------|
| `ui/app.py` | Interactive UI with tabs for Text→Code, Code→Docs, and Benchmark batch eval. Separate model dropdowns for code vs doc checkpoints. |

---

## `data/` — Datasets

### Scripts

| File | Significance |
|------|--------------|
| `data/scripts/download_and_preprocess.py` | Entry point: downloads MBPP + CoDocBench and writes processed JSONL. |
| `data/scripts/download_and_preprocess.sh` | Shell wrapper (activates venv, runs the Python script). Called by CP1/CP2 training scripts. |

### Raw (`data/raw/`)

| Path | Significance |
|------|--------------|
| `data/raw/mbpp/sanitized-mbpp.json` | Full 427-task MBPP corpus (cached download). Only raw MBPP file needed; older Hugging Face Arrow splits were removed as unused. |
| `data/raw/codocbench/train.jsonl` | Official CoDocBench training split (cached). |
| `data/raw/codocbench/test.jsonl` | Official CoDocBench test split (cached). |

### Processed (`data/processed/`)

| File | Significance |
|------|--------------|
| `mbpp_train.jsonl` | 80% MBPP split for code fine-tuning. |
| `mbpp_validation.jsonl` | 10% MBPP split for early stopping during training. |
| `mbpp_test.jsonl` | 10% MBPP split for pass@1 evaluation. |
| `mbpp_all.jsonl` | Full 427 tasks (notebook-style eval when `split=all`). |
| `codoc_train.jsonl` | CoDocBench train pairs (90% of official train after val holdout). |
| `codoc_validation.jsonl` | 10% holdout from CoDocBench train for early stopping. |
| `codoc_test.jsonl` | CoDocBench test pairs for ROUGE-L evaluation. |

All processed files are **regeneratable** via `python data/scripts/download_and_preprocess.py`.

---

## `experiments/` — Training artifacts and results

| Path | Significance |
|------|--------------|
| `experiments/summarize_results.py` | Prints a comparison table from eval JSON files (pass@1, ROUGE-L, retry success rate). Use in reports. |
| `experiments/checkpoints/codegen-mbpp-full/` | **CP2 fine-tuned model for code generation.** Contains final `model.safetensors`, tokenizer, and config. Used by eval, API, and UI. |
| `experiments/checkpoints/codegen-doc-full/` | **CP2 fine-tuned model for documentation.** Same layout as above; used for doc generation and doc eval. |
| `experiments/results/.gitkeep` | Placeholder so the results directory is tracked in git. |
| `experiments/results/cp2_eval_upstream_code.json` | Baseline pass@1 on MBPP test (upstream CodeGen-350M, capped sample count). |
| `experiments/results/cp2_eval_upstream_docs.json` | Baseline ROUGE-L on CoDocBench test (upstream model). |
| `experiments/results/cp2_eval_finetuned_code.json` | pass@1 after MBPP fine-tuning. |
| `experiments/results/cp2_eval_finetuned_docs.json` | ROUGE-L after CoDocBench fine-tuning. |
| `experiments/results/cp2_eval_agent_code.json` | pass@1 with agentic retries on fine-tuned code model. |

Checkpoint directories also contain `generation_config.json`, `tokenizer.json`, and `tokenizer_config.json`. Intermediate `checkpoint-*` subfolders (optimizer state) were removed to save disk; final weights at the checkpoint root are sufficient for inference.

Eval JSON files are gitignored by default but kept locally for report writing. Re-run with `bash scripts/run_cp2_eval.sh`.

---

## `scripts/` — Automation

| File | Significance |
|------|--------------|
| `scripts/train_cp1.sh` | Checkpoint 1 pipeline: download data → upstream baselines → subset training (128 MBPP / 200 CoDoc) → post-train eval. |
| `scripts/train_cp2.sh` | Checkpoint 2 pipeline: full training (2 epochs, best checkpoint) → code baseline + agent + doc eval. |
| `scripts/run_cp2_eval.sh` | Re-eval only (no training). Compares upstream vs fine-tuned vs agent on capped samples (override via `MAX_CODE_SAMPLES` / `MAX_DOC_SAMPLES`). |
| `scripts/resume_cp2_doc.sh` | Resume doc training if MBPP finished but CoDocBench training failed; then runs CP2 eval. |

---

## `tests/` — Unit tests

| File | Significance |
|------|--------------|
| `tests/test_data.py` | MBPP normalization/loading and agent retry behavior (mocked generation). |
| `tests/test_eval.py` | Code cleaning, functional correctness, doc metric smoke tests. |
| `tests/test_doc_generation.py` | Doc postprocessing and generation path selection (LM vs fallback). |
| `tests/test_api.py` | FastAPI endpoint contracts via TestClient. |
| `tests/test_models.py` | Model registry preset resolution and checkpoint listing. |

Run all tests: `python -m pytest tests/ -q` (22 tests).

---

## Removed artifacts (cleanup)

The following were removed as unused or redundant:

| Removed | Reason |
|---------|--------|
| `data/processed/spider_*.jsonl` | SQL generation out of scope; no code references. |
| `data/raw/mbpp/{train,test,validation}/` + `dataset_dict.json` | Legacy Hugging Face Arrow layout; pipeline uses `sanitized-mbpp.json` only. |
| `experiments/checkpoints/debug-codegen/` | Debug training run (8 steps). |
| `experiments/checkpoints/codegen-mbpp-smoke/` | Smoke-test checkpoint. |
| `experiments/checkpoints/codegen-mbpp-baseline/` | CP1-era partial baseline; superseded by `codegen-mbpp-full`. |
| `experiments/checkpoints/*/checkpoint-*/` | Intermediate training snapshots (optimizer/scheduler); final weights kept at checkpoint root. |
| `experiments/results/cp2_sql_upstream_5.json` | SQL probe eval. |
| `experiments/results/cp2_*_smoke.json`, `cp2_codoc_upstream_5.json`, `cp2_upstream_all_10.json` | Small probe runs superseded by CP2 eval set. |
| `experiments/results/baseline_upstream_*.json` | Duplicates of `cp2_eval_upstream_*.json`. |
| Duplicate NoSQL PDF (`… - Copy.pdf`) | Duplicate of literature paper. |

---

## Typical workflows

**Setup**
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python data/scripts/download_and_preprocess.py
```

**Serve**
```bash
uvicorn app.main:app --reload --port 8000
streamlit run ui/app.py
```

**Report table**
```bash
python experiments/summarize_results.py experiments/results/cp2_eval_*.json
```

**Recommended demo models**
- Code: `experiments/checkpoints/codegen-mbpp-full`
- Docs: `experiments/checkpoints/codegen-doc-full`
- Temperature: `0.0` for reproducible outputs
