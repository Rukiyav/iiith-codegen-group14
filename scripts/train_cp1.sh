#!/usr/bin/env bash
# Checkpoint 1: subset training + baseline eval (2-week deliverable)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source .venv/bin/activate

bash data/scripts/download_and_preprocess.sh

echo "=== Upstream baselines (before fine-tune) ==="
python -m src.eval.run_eval --task code --model Salesforce/codegen-350M-multi --split test \
  --output experiments/results/cp1_upstream_code_test.json
python -m src.eval.run_eval --task docs --model Salesforce/codegen-350M-multi --split test --max_samples 50 \
  --output experiments/results/cp1_upstream_docs_test.json

echo "=== CP1 subset training ==="
python src/train.py --max_train_samples 128 --max_eval_samples 32 --num_train_epochs 1 \
  --output_dir experiments/checkpoints/codegen-mbpp-cp1 "$@"
python src/train_doc.py --max_train_samples 200 --max_eval_samples 32 --num_train_epochs 1 \
  --output_dir experiments/checkpoints/codegen-doc-cp1 "$@"

echo "=== CP1 post-train eval ==="
python -m src.eval.run_eval --task code --model experiments/checkpoints/codegen-mbpp-cp1 --split test \
  --output experiments/results/cp1_finetuned_code_test.json
python -m src.eval.run_eval --task docs --model experiments/checkpoints/codegen-doc-cp1 --split test --max_samples 50 \
  --output experiments/results/cp1_finetuned_docs_test.json

python experiments/summarize_results.py experiments/results/cp1_*.json
