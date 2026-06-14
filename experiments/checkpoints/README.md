# Checkpoints (not stored in git)

Each fine-tuned CodeGen-350M checkpoint is **~1.3 GB** (`model.safetensors` plus tokenizer files). GitHub rejects files over 100 MB, so checkpoints stay local or on Hugging Face / Drive.

## Generate locally

```bash
source .venv/bin/activate
python data/scripts/download_and_preprocess.py
bash scripts/train_cp2.sh
```

Outputs:

| Path | Purpose |
|------|---------|
| `codegen-mbpp-full/` | Text → Python (MBPP fine-tune) |
| `codegen-doc-full/` | Code → documentation (CoDocBench fine-tune) |

## Use in the app

Point the Streamlit sidebar (or API) at:

- Code: `experiments/checkpoints/codegen-mbpp-full`
- Docs: `experiments/checkpoints/codegen-doc-full`

Upstream baseline (no local files): `codegen-350m` → `Salesforce/codegen-350M-multi` from Hugging Face Hub.
