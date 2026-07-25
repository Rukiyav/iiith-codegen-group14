#!/usr/bin/env bash
# Phase 7 — Unified Evaluation (notebook): baseline / lora / rag / agentic on MBPP held-out.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source .venv/bin/activate 2>/dev/null || true

EVAL_N="${EVAL_N:-50}"

mkdir -p evaluation/results

if [[ ! -f data/mbpp/sanitized-mbpp.json ]]; then
  echo "=== Download MBPP (Phase 1b) ==="
  python -m src.data.download
fi

if [[ ! -f models/lora_finetuned/adapter_config.json ]]; then
  echo "WARNING: LoRA checkpoint missing. Train first for the lora column:"
  echo "  bash scripts/train.sh"
  echo "Continuing with baseline / rag / agentic only."
fi

if [[ ! -f retrieval/vector_store/repo.index ]]; then
  echo "=== Build FAISS RAG index (Phase 5) ==="
  python -c "from src.rag.index import build_index; build_index()"
fi

echo "=== Phase 7: four-mode MBPP eval (n=${EVAL_N}) ==="
python -m src.eval.runner --eval_n "$EVAL_N"

echo ""
echo "Results saved to evaluation/results/comparison_mbpp.json"
echo "View in Gradio: Evaluation Dashboard → Show Results"
