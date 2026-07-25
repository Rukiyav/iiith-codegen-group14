#!/usr/bin/env bash
# Quick smoke: 5-problem MBPP eval (requires LoRA + RAG index).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source .venv/bin/activate 2>/dev/null || true
EVAL_N=5 bash scripts/run_eval.sh
