#!/usr/bin/env bash
# Re-run CP2 eval only (no training). Useful after checkpoints already exist.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source .venv/bin/activate

mkdir -p experiments/results

CODE_MODEL="${CODE_MODEL:-experiments/checkpoints/codegen-mbpp-full}"
DOC_MODEL="${DOC_MODEL:-experiments/checkpoints/codegen-doc-full}"
UPSTREAM="${UPSTREAM:-Salesforce/codegen-350M-multi}"
MAX_CODE_SAMPLES="${MAX_CODE_SAMPLES:-43}"
MAX_DOC_SAMPLES="${MAX_DOC_SAMPLES:-100}"

run() {
  python -m src.eval.run_eval "$@"
}

echo "=== Upstream ==="
run --task code --model "$UPSTREAM" --split test --max_samples "$MAX_CODE_SAMPLES" \
  --output experiments/results/cp2_eval_upstream_code.json
run --task docs --model "$UPSTREAM" --split test --max_samples "$MAX_DOC_SAMPLES" \
  --output experiments/results/cp2_eval_upstream_docs.json

if [[ ! -d "$CODE_MODEL" ]]; then
  echo "=== Skip fine-tuned code eval: $CODE_MODEL not found (run bash scripts/train_cp2.sh) ==="
else
  echo "=== Fine-tuned baseline (code) ==="
  run --task code --model "$CODE_MODEL" --split test --max_samples "$MAX_CODE_SAMPLES" \
    --output experiments/results/cp2_eval_finetuned_code.json

  echo "=== Fine-tuned + agent (code) ==="
  run --task code --model "$CODE_MODEL" --split test --max_samples "$MAX_CODE_SAMPLES" --agent \
    --output experiments/results/cp2_eval_agent_code.json
fi

if [[ ! -d "$DOC_MODEL" ]]; then
  echo "=== Skip fine-tuned doc eval: $DOC_MODEL not found (run bash scripts/train_cp2.sh) ==="
else
  echo "=== Fine-tuned baseline (docs) ==="
  run --task docs --model "$DOC_MODEL" --split test --max_samples "$MAX_DOC_SAMPLES" \
    --output experiments/results/cp2_eval_finetuned_docs.json
fi

python experiments/summarize_results.py experiments/results/cp2_eval_*.json 2>/dev/null || true
