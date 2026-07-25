# CodeGen Group14 — AIML B26 Capstone (Ultra-Lean Architecture)

Unified 5-file AI engine for **NL → Python**, **code → documentation (ROUGE-1/2/L)**, **PL1 → PL2 translation**, **Project RAG**, and **Agentic Self-Correction**.

---

## 🚀 Capabilities & Features

| Capability | Model / Technique | Metric |
|------------|-------------------|--------|
| **NL → Python** | Baseline / LoRA / RAG / **Agentic** | pass@1, pass@k (MBPP, HumanEval+) |
| **Code → Docs** | AST-Grounded + Zero-shot DocGen | **ROUGE-1/2/L**, BLEU, BERTScore |
| **PL1 → PL2** | CodeGen-350M-multi AST Translate | Idiomatic translation across 7 PLs |
| **Agentic Repair** | Sandbox exec + Reflection + Repair | retry_success_rate, pass@k |
| **Project RAG** | FAISS + SentenceTransformers over project folder | Context precision & pass@k |

Default base model: **CodeGen-350M-multi** (`Salesforce/codegen-350M-multi`).

---

## 📁 Ultra-Lean 5-File Repository Architecture

```
iiith-codegen-group14/
│
├── src/                        # Core Engine Library
│   ├── config.py               # (1) Central hyperparameters & directory paths
│   ├── engine.py               # (2) Unified AI engine (Models, RAG, Sandbox, Prompts, Agent, Docs, Translate)
│   └── eval.py                 # (3) Evaluation metrics (MBPP, EvalPlus, ROUGE-1/2/L, BLEU, pass@k)
│
├── app.py                      # (4) Unified FastAPI REST server & Streamlit Web UI
├── train.py                    # (5) Data downloader, SFT dataset builder, and LoRA training script
│
├── tests/                      # Automated Test Suite
│   └── test_all.py             # Pytest suite covering all 5 capabilities
│
├── docs/                       # Project Documentation & Research Papers
├── capstone_colab.ipynb        # Course Demonstration Notebook
└── README.md                   # Project Overview
```

---

## ⚡ Quickstart

### 1. Installation
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run Tests
```bash
pytest tests/test_all.py
```

### 3. Fine-Tune LoRA Model
```bash
python train.py
```

### 4. Run Benchmark Evaluations
```bash
python -m src.eval
```

---

## 💻 Running the API and Web UI

### Launch REST API Server
```bash
uvicorn app:app --port 8000 --reload
```
* Interactive API Documentation: `http://localhost:8000/docs`

### Launch Streamlit Web UI
```bash
streamlit run app.py
```

---

## 🔌 REST API Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/health` | Server health check |
| `GET` | `/models` | List supported models and capabilities |
| `POST` | `/generate/code` | Generate Python code (supports `baseline`, `lora`, `rag`, `agentic`) |
| `POST` | `/docs/generate` | Generate Google-style docstrings from code |
| `POST` | `/translate` | Translate code between languages (PL1 → PL2) |
| `POST` | `/rag/index` | Index a local project directory into FAISS |
