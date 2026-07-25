#!/usr/bin/env bash
# Smoke EvalPlus (few tasks, few samples) — for quick tuning loops.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source .venv/bin/activate 2>/dev/null || true

MODE="${MODE:-lora}"
MAX_TASKS="${MAX_TASKS:-5}"
N_SAMPLES="${N_SAMPLES:-1}"

echo "=== Smoke EvalPlus (tasks=${MAX_TASKS}, samples=${N_SAMPLES}, mode=${MODE}) ==="
python -m src.eval.plus \
  --dataset both \
  --mode "$MODE" \
  --max_tasks "$MAX_TASKS" \
  --n_samples "$N_SAMPLES" \
  --temperature 0.0
