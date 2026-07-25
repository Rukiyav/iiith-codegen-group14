# Codebase Architecture Guide — Ultra-Lean 5-File Design

This repository has been refactored into an ultra-lean 5-file architecture while preserving 100% of all features, benchmark evaluation metrics, and API/UI capabilities. See [`docs/ARCHITECTURE_REPORT.md`](file:///Users/rajat/iiith-codegen-group14/docs/ARCHITECTURE_REPORT.md) for the complete design & tech stack report.

## File Map

| File | Purpose | Lines of Code |
|------|---------|---------------|
| [`src/config.py`](file:///Users/rajat/iiith-codegen-group14/src/config.py) | Centralized paths, model configurations, and hyperparameters | ~85 LOC |
| [`src/engine.py`](file:///Users/rajat/iiith-codegen-group14/src/engine.py) | Complete AI engine: Model loading, Sandbox execution, Prompts, RAG, Self-correction, DocGen, PL Translate | ~760 LOC |
| [`src/eval.py`](file:///Users/rajat/iiith-codegen-group14/src/eval.py) | Benchmark runners (MBPP, EvalPlus) and metrics (ROUGE-1/2/L, BLEU, BERTScore, pass@k) | ~145 LOC |
| [`app.py`](file:///Users/rajat/iiith-codegen-group14/app.py) | Unified FastAPI REST Server and Streamlit Web UI application | ~155 LOC |
| [`train.py`](file:///Users/rajat/iiith-codegen-group14/train.py) | Dataset download, preprocessing, and LoRA SFT fine-tuning pipeline | ~155 LOC |
| [`tests/test_all.py`](file:///Users/rajat/iiith-codegen-group14/tests/test_all.py) | Pytest suite testing all core engine functions and REST endpoints | ~75 LOC |

---

## Execution Commands

- **Run Tests**: `.venv/bin/python -m pytest tests/test_all.py`
- **Run Training**: `python train.py`
- **Run Benchmark Evaluation**: `python -m src.eval`
- **Start REST API**: `uvicorn app:app --port 8000 --reload`
- **Start Streamlit Web UI**: `streamlit run app.py`
