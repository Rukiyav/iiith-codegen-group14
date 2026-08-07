---
title: CodeGen Group14 AI System
emoji: ⚡
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: "1.32.0"
app_file: streamlit_app.py
pinned: false
---

This project implements an end-to-end Code Intelligence System engineered around Salesforce/codegen-350M-multi. Built on an ultra-lean 5-file architecture, the system unifies 5 core software engineering AI capabilities:

NL → Python Code Generation: Evaluated across 4 modes (Baseline, PEFT/LoRA Fine-Tuned, RAG, and Agentic Self-Correction) on MBPP and EvalPlus benchmarks (pass@1, pass@k).
Code → Documentation Generation: Combines AST-grounded argument parsing with neural summarization to generate Google-style docstrings, quantitatively benchmarked using ROUGE-1/2/L, BLEU, and BERTScore metrics.
Programming Language Translation: Achieves zero-shot idiomatic code translation across 7+ languages (Python, Java, C++, JavaScript, Go, Rust, TypeScript).
Repository-Level RAG Engine: Performs Abstract Syntax Tree (AST) function extraction and dense FAISS vector indexing with similarity score thresholding ($\text{score} \ge 0.35$).
Execution-Driven Agentic Repair Loop: An isolated subprocess sandbox that parses line-level traceback errors, reflects on diagnostic feedback, and iteratively repairs failing code logic.
The application is fully containerized and features a dual-entry point: a production FastAPI REST API and an interactive Streamlit Web Dashboard.

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
