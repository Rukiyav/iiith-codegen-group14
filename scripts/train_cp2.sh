#!/usr/bin/env bash
# Checkpoint 2: full training + full test eval + agent comparison
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source .venv/bin/activate

MAX_CODE_SAMPLES="${MAX_CODE_SAMPLES:-}"  # empty = full MBPP test split
MAX_DOC_SAMPLES="${MAX_DOC_SAMPLES:-100}"   # cap doc eval for speed; unset for full test

bash data/scripts/download_and_preprocess.sh

echo "=== CP2 full training ==="
python src/train.py --full --output_dir experiments/checkpoints/codegen-mbpp-full --num_train_epochs 2 \
  --load_best_model_at_end "$@"
python src/train_doc.py --full --output_dir experiments/checkpoints/codegen-doc-full --num_train_epochs 2 \
  --load_best_model_at_end "$@"

echo "=== CP2 evaluation: code (baseline) ==="
CODE_ARGS=(--task code --model experiments/checkpoints/codegen-mbpp-full --split test \
  --output experiments/results/cp2_code_baseline_test.json)
if [[ -n "${MAX_CODE_SAMPLES}" ]]; then
  CODE_ARGS+=(--max_samples "${MAX_CODE_SAMPLES}")
fi
python -m src.eval.run_eval "${CODE_ARGS[@]}"

echo "=== CP2 evaluation: code (agent) ==="
AGENT_ARGS=(--task code --model experiments/checkpoints/codegen-mbpp-full --split test --agent \
  --output experiments/results/cp2_code_agent_test.json)
if [[ -n "${MAX_CODE_SAMPLES}" ]]; then
  AGENT_ARGS+=(--max_samples "${MAX_CODE_SAMPLES}")
fi
python -m src.eval.run_eval "${AGENT_ARGS[@]}"

echo "=== CP2 evaluation: docs (ROUGE-L) ==="
DOC_ARGS=(--task docs --model experiments/checkpoints/codegen-doc-full --split test \
  --output experiments/results/cp2_docs_test.json)
if [[ -n "${MAX_DOC_SAMPLES}" ]]; then
  DOC_ARGS+=(--max_samples "${MAX_DOC_SAMPLES}")
fi
python -m src.eval.run_eval "${DOC_ARGS[@]}"

python experiments/summarize_results.py experiments/results/cp2_*.json experiments/results/cp1_upstream_*.json
