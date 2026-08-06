#!/usr/bin/env bash
# Resume CP2 after MBPP finished but doc training failed (or skip MBPP re-train).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source .venv/bin/activate

MAX_CODE_SAMPLES="${MAX_CODE_SAMPLES:-100}"
MAX_DOC_SAMPLES="${MAX_DOC_SAMPLES:-100}"

echo "=== CoDocBench full training ==="
python src/train_doc.py --full --output_dir experiments/checkpoints/codegen-doc-full \
  --num_train_epochs 2 --load_best_model_at_end "$@"

echo "=== Eval (capped for laptop; override MAX_* env vars for full test) ==="
bash scripts/run_cp2_eval.sh

python experiments/summarize_results.py experiments/results/cp2_*.json 2>/dev/null || true
