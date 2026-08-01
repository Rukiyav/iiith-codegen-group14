"""Unified Web UI & FastAPI Server for CodeGen Group14."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.config import MODEL_ID
from src.engine import (
    LANG_LABELS,
    SUPPORTED_PAIRS,
    generate_code_for_task,
    generate_docstring,
    index_project_folder,
    load_base_model,
    translate_pl,
)

# -----------------------------------------------------------------------------
# 1. FastAPI REST Application
# -----------------------------------------------------------------------------

app = FastAPI(title="CodeGen Group14 API", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class CodeGenerateRequest(BaseModel):
    prompt: str
    mode: str = "baseline"
    temperature: float = 0.2
    max_new_tokens: int = 384
    test_list: Optional[List[str]] = None
    test_imports: Optional[List[str]] = None
    reference_code: Optional[str] = None
    use_agent: bool = False
    max_retries: int = 3
    n_candidates: int = 1
    project_dir: Optional[str] = None
    force_reindex: bool = False


class GenerateResponse(BaseModel):
    code: str
    mode: str
    passed: Optional[bool] = None
    attempts: Optional[int] = None
    status: str = ""
    retrieved: List[dict] = Field(default_factory=list)


class DocGenerateRequest(BaseModel):
    code: str
    few_shot: bool = False


class TranslateRequest(BaseModel):
    code: str
    source_lang: str = "python"
    target_lang: str = "java"


class RagIndexRequest(BaseModel):
    project_dir: str
    force: bool = False


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/models")
def list_models() -> dict:
    return {
        "models": [
            {"id": "baseline", "label": "CodeGen-350M baseline"},
            {"id": "lora", "label": "LoRA fine-tuned (MBPP)"},
        ],
        "default": MODEL_ID,
        "modes": ["baseline", "lora", "rag", "agentic"],
        "translate_pairs": [{"source": a, "target": b} for a, b in SUPPORTED_PAIRS],
        "languages": list(LANG_LABELS.keys()),
    }


@app.post("/generate/code", response_model=GenerateResponse)
def api_generate_code(req: CodeGenerateRequest) -> GenerateResponse:
    mode = "agentic" if req.use_agent else req.mode
    res = generate_code_for_task(
        req.prompt,
        mode=mode,
        temperature=req.temperature,
        max_new_tokens=req.max_new_tokens,
        test_list=req.test_list,
        test_imports=req.test_imports,
        reference_code=req.reference_code,
        max_retries=req.max_retries,
        n_candidates=req.n_candidates,
        project_dir=req.project_dir,
        force_reindex=req.force_reindex,
    )
    return GenerateResponse(
        code=res.code,
        mode=res.mode,
        passed=res.passed,
        attempts=res.attempts,
        status=res.status,
        retrieved=res.retrieved,
    )


@app.post("/docs/generate")
def api_generate_docs(req: DocGenerateRequest) -> dict:
    model = load_base_model()
    doc = generate_docstring(model, req.code, few_shot=req.few_shot)
    return {"docstring": doc}


@app.post("/translate")
def api_translate(req: TranslateRequest) -> dict:
    return translate_pl(req.code, req.source_lang, req.target_lang)


@app.post("/rag/index")
def api_rag_index(req: RagIndexRequest) -> dict:
    return index_project_folder(req.project_dir, force=req.force)


# -----------------------------------------------------------------------------
# 2. Streamlit Web Interface Launcher
# -----------------------------------------------------------------------------

def run_streamlit_ui():
    import streamlit as st

    st.set_page_config(page_title="CodeGen Group14", layout="wide")
    st.title("⚡ CodeGen Group14 — AI Engine")
    st.caption("NL → Python | Code → Docstring | PL Translation | Project RAG | Agentic Repair")

    tab1, tab2, tab3 = st.tabs(["💻 Code Generation", "📝 Doc Generation", "🔄 PL Translation"])

    with tab1:
        st.subheader("NL / LeetCode → Python Generation")
        prompt = st.text_area("Prompt / Problem Description", value="Write a function that checks if a string is a palindrome.")
        col1, col2 = st.columns(2)
        with col1:
            mode = st.selectbox("Generation Mode", ["baseline", "lora", "rag", "agentic"])
            project_dir = st.text_input("Project Folder Path (RAG)", value="")
        with col2:
            temp = st.slider("Temperature", 0.0, 1.0, 0.2)
            max_tokens = st.number_input("Max New Tokens", 64, 1024, 384)

        if st.button("Generate Code"):
            with st.spinner("Generating solution..."):
                res = generate_code_for_task(
                    prompt,
                    mode=mode,
                    temperature=temp,
                    max_new_tokens=max_tokens,
                    project_dir=project_dir if project_dir.strip() else None,
                )
                st.code(res.code, language="python")
                st.info(f"Status: {res.status} | Passed: {res.passed}")

    with tab2:
        st.subheader("Code → Google-style Docstring")
        code_input = st.text_area("Python Code", value="def add(a, b):\n    return a + b", height=150)
        if st.button("Generate Docstring"):
            with st.spinner("Generating..."):
                model = load_base_model()
                doc = generate_docstring(model, code_input)
                st.code(doc, language="markdown")

    with tab3:
        st.subheader("Programming Language Translation")
        col_a, col_b = st.columns(2)
        with col_a:
            src_lang = st.selectbox("Source Language", list(LANG_LABELS.keys()), index=0)
        with col_b:
            tgt_lang = st.selectbox("Target Language", list(LANG_LABELS.keys()), index=1)
        src_code = st.text_area("Source Code", value="def is_even(n):\n    return n % 2 == 0", height=150)
        if st.button("Translate"):
            with st.spinner("Translating..."):
                out = translate_pl(src_code, src_lang, tgt_lang)
                st.code(out.get("output", ""), language=tgt_lang)


if __name__ == "__main__" or "STREAMLIT_SERVER_PORT" in os.environ or any("streamlit" in arg.lower() for arg in sys.argv):
    run_streamlit_ui()
