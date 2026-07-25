#!/usr/bin/env bash
# Train LoRA: MBPP majority + careful APPS intro mix, longer context, early stop.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source .venv/bin/activate 2>/dev/null || true

bash data/scripts/download_and_preprocess.sh

MAX_SAMPLES="${MAX_SAMPLES:-374}"
MAX_STEPS="${MAX_STEPS:-400}"
EPOCHS="${EPOCHS:-3}"
MAX_LENGTH="${MAX_LENGTH:-512}"
APPS_RATIO="${APPS_RATIO:-0.20}"
APPS_MAX="${APPS_MAX:-120}"
PATIENCE="${PATIENCE:-3}"
VAL_EVERY="${VAL_EVERY:-40}"
OUTPUT="${OUTPUT:-models/lora_finetuned}"
NO_APPS="${NO_APPS:-0}"

EXTRA_ARGS=()
if [[ "$NO_APPS" == "1" ]]; then
  EXTRA_ARGS+=(--no_apps)
fi

echo "=== LoRA training (MBPP + APPS mix=${APPS_RATIO}, max_len=${MAX_LENGTH}, early-stop) ==="
python -m src.training.lora \
  --max_samples "$MAX_SAMPLES" \
  --max_steps "$MAX_STEPS" \
  --epochs "$EPOCHS" \
  --max_length "$MAX_LENGTH" \
  --apps_ratio "$APPS_RATIO" \
  --apps_max "$APPS_MAX" \
  --patience "$PATIENCE" \
  --val_every "$VAL_EVERY" \
  --output_dir "$OUTPUT" \
  ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}

echo "LoRA checkpoint: $OUTPUT"
if [[ -f "$OUTPUT/train_meta.json" ]]; then
  python -c "import json; print(json.dumps(json.load(open('$OUTPUT/train_meta.json')), indent=2))"
fi
