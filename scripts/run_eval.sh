#!/usr/bin/env bash
# Evaluation: docs + RAG + MBPP modes + EvalPlus (HumanEval+ / MBPP+).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source .venv/bin/activate 2>/dev/null || true

EVAL_N="${EVAL_N:-50}"
DOC_N="${DOC_N:-50}"
SKIP_RAG="${SKIP_RAG:-0}"
SKIP_DOC="${SKIP_DOC:-0}"
SKIP_MBPP="${SKIP_MBPP:-0}"
SKIP_PLUS="${SKIP_PLUS:-0}"
PLUS_MODE="${PLUS_MODE:-lora}"
PLUS_MAX_TASKS="${PLUS_MAX_TASKS:-20}"
PLUS_N_SAMPLES="${PLUS_N_SAMPLES:-5}"

mkdir -p evaluation/results retrieval/vector_store

if [[ "$SKIP_DOC" != "1" ]]; then
  echo "=== Documentation eval (zero-shot + ROUGE-L) ==="
  python -c "from src.docs.generate import run_doc_eval; run_doc_eval(n=${DOC_N})"
fi

if [[ "$SKIP_RAG" != "1" ]]; then
  echo "=== Build FAISS RAG index ==="
  python -c "from src.rag.index import build_index; build_index()"
fi

if [[ "$SKIP_MBPP" != "1" ]]; then
  echo "=== Four-mode MBPP eval (n=${EVAL_N}) ==="
  python -m src.eval.runner --eval_n "$EVAL_N"
fi

if [[ "$SKIP_PLUS" != "1" ]]; then
  echo "=== EvalPlus HumanEval+ / MBPP+ (mode=${PLUS_MODE}, tasks=${PLUS_MAX_TASKS}) ==="
  python -m src.eval.plus \
    --dataset both \
    --mode "$PLUS_MODE" \
    --max_tasks "$PLUS_MAX_TASKS" \
    --n_samples "$PLUS_N_SAMPLES"
fi

echo "Results: evaluation/results/"
