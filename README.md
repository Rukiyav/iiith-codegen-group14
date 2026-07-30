# AI-Powered Software Engineering Assistant

Three capabilities through **Checkpoint 2**:

1. **Text → Python code** (MBPP, CodeGen-350M-multi, pass@1)
2. **Code → Documentation** (CoDocBench, **ROUGE-L** primary)
3. **Agentic self-correction** (generate → test → retry)

**Out of scope until CP3:** VS Code extension, RAG, SQL generation.

## Research papers (literature review)

Three papers are in the repo; they inform the **proposal and later checkpoints**, but are **not yet wired into the running code**:

| Paper | Relevance | Used in code today? |
|-------|-----------|---------------------|
| **SERA** — Soft-Verified Efficient Repository Agents | Agentic self-correction, repository specialization, synthetic trajectories | Partially — we implement generate→test→retry; not SERA’s SVG training pipeline |
| **Scaling Laws for Code** — Every Programming Language Matters | Multi-language / PL1→PL2 translation stretch | No — deferred to optional CP2+ stretch |
| **Bridging the Gap… Text-to-NoSQL** | SQL/NoSQL generation track | No — removed from scope per mentor focus |

Implementation follows the **capstone PDF** + **group proposal**: MBPP, CoDocBench, CodeGen-350M-multi, pass@1, ROUGE-L, agent loop. RAG (proposal Module 4) is CP3.

## Model selection

Default upstream model: **CodeGen-350M-multi** (`codegen-350m`).

Fine-tuned checkpoints under `experiments/checkpoints/` appear in the Streamlit sidebar and `GET /models`.

```bash
curl http://127.0.0.1:8000/models
python src/infer.py "Write factorial in Python" --model_name_or_path codegen-350m
python src/infer.py "Write factorial" --model_name_or_path experiments/checkpoints/codegen-mbpp-full
```

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python data/scripts/download_and_preprocess.py
```

## Checkpoint 1 (subset training + baseline eval)

Deliverable: working pipeline, subset fine-tune, eval harness, API/UI.

```bash
chmod +x scripts/train_cp1.sh
bash scripts/train_cp1.sh
```

This runs:

- Upstream baseline on MBPP test + CoDocBench test (50 doc samples)
- Subset training (`128` MBPP / `200` CoDocBench examples, 1 epoch)
- Post-train eval → `experiments/results/cp1_*.json`
- Summary table via `experiments/summarize_results.py`

## Checkpoint 2 (full training + comparison)

Deliverable: full MBPP + CoDocBench fine-tune, pass@1 / ROUGE-L comparison, agent lift.

```bash
chmod +x scripts/train_cp2.sh scripts/run_cp2_eval.sh
bash scripts/train_cp2.sh
```

Full training uses `--full` (entire train splits, 2 epochs, `load_best_model_at_end`).
Eval writes `experiments/results/cp2_*.json` including **baseline vs agent** on code.

**Re-eval only** (checkpoints already trained):

```bash
# default: 43 MBPP + 100 CoDocBench test samples (override with env vars)
bash scripts/run_cp2_eval.sh

# full MBPP test (500 tasks — slow on CPU)
MAX_CODE_SAMPLES=500 MAX_DOC_SAMPLES=500 bash scripts/run_cp2_eval.sh
```

### Results summary

```bash
python experiments/summarize_results.py experiments/results/cp2_*.json
```

| Metric | Task | Meaning |
|--------|------|---------|
| **pass@1** | Code | Execution accuracy on MBPP tests |
| **ROUGE-L** | Docs | Primary doc quality (mentor requirement) |
| **retry_success_rate** | Code + agent | Share fixed after initial failure |

## Manual evaluation

```bash
# Code baseline
python -m src.eval.run_eval --task code --model Salesforce/codegen-350M-multi --split test

# Code with agentic self-correction
python -m src.eval.run_eval --task code --model experiments/checkpoints/codegen-mbpp-full --split test --agent

# Documentation (ROUGE-L primary)
python -m src.eval.run_eval --task docs --model experiments/checkpoints/codegen-doc-full --split test
```

Use `--max_samples N` for quick smoke runs; omit for full split.

## API + UI

```bash
uvicorn app.main:app --reload --port 8000
streamlit run ui/app.py
```

| Tab | Pipeline |
|-----|----------|
| **Text → Code** | Generate → optional test execution → agent retries |
| **Code → Documentation** | Generate → optional ROUGE-L/BLEU vs reference |
| **Benchmark** | Batch eval on MBPP / CoDocBench splits (for reports) |

Single-prompt generation and validation are one flow. **Benchmark** is for dataset-scale eval only.

## Agent workflow

```
Prompt → Generate code → Run unit tests → Pass? → Done
                              ↓ fail
                         Retry with failure context (max N)
```

API: `POST /generate/code` with optional `test_list`, `use_agent`, `max_retries`.

## Layout

```
src/data/         mbpp.py, codoc.py
src/agent/        code_agent.py
src/eval/         code_eval, doc_eval, run_eval (--agent)
src/train.py      MBPP fine-tuning (--full for CP2)
src/train_doc.py  CoDocBench fine-tuning (--full for CP2)
scripts/          train_cp1.sh, train_cp2.sh, run_cp2_eval.sh
experiments/      checkpoints/, results/, summarize_results.py
app/              FastAPI
ui/               Streamlit
```

## Tests

```bash
python -m pytest tests/ -q
```

## Hardware notes

- **Apple Silicon:** training uses MPS when available (`src/train.py` device detection).
- **GPU:** add `--fp16` to training commands for faster CP2 runs.
- Generation defaults to **temperature=0.0** (deterministic) for reproducible CLI/UI parity.
