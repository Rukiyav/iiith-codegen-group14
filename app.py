"""Unified Web UI & FastAPI Server for CodeGen Group14."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.config import MODEL_ID
from src.engine import (
    LANG_LABELS,
    SUPPORTED_PAIRS,
    generate_code_for_task,
    generate_docstring,
    get_backend_for_model,
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
    model: str = "codegen"
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
    use_rag: bool = False


class GenerateResponse(BaseModel):
    code: str
    mode: str
    passed: Optional[bool] = None
    attempts: Optional[int] = None
    status: str = ""
    retrieved: List[dict] = Field(default_factory=list)


class DocGenerateRequest(BaseModel):
    code: str
    model: str = "codegen"
    few_shot: bool = False


class TranslateRequest(BaseModel):
    code: str
    model: str = "codegen"
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
    from src.config import MODEL_REGISTRY, PIPELINE_MODES
    models_list = []
    for model_key, cfg in MODEL_REGISTRY.items():
        models_list.append({
            "id": model_key,
            "label": cfg.get("name", model_key),
            "backend": cfg.get("backend", "huggingface"),
            "supports_docs": cfg.get("supports_docs", True),
            "supports_translation": cfg.get("supports_translation", True),
        })
    return {
        "models": models_list,
        "default": "codegen",
        "modes": PIPELINE_MODES,
        "translate_pairs": [{"source": a, "target": b} for a, b in SUPPORTED_PAIRS],
        "languages": list(LANG_LABELS.keys()),
    }


@app.post("/generate/code", response_model=GenerateResponse)
def api_generate_code(req: CodeGenerateRequest) -> GenerateResponse:
    from src.config import MODEL_REGISTRY
    if req.model not in MODEL_REGISTRY:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model name '{req.model}'. Registered models: {list(MODEL_REGISTRY.keys())}"
        )
    mode = "agentic" if req.use_agent else req.mode
    res = generate_code_for_task(
        req.prompt,
        model_name=req.model,
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
        use_rag=req.use_rag,
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
    from src.config import MODEL_REGISTRY
    if req.model not in MODEL_REGISTRY:
        raise HTTPException(status_code=400, detail=f"Unknown model: {req.model}")
    cfg = MODEL_REGISTRY[req.model]
    if not cfg.get("supports_docs", True):
        raise HTTPException(status_code=400, detail=f"Model '{req.model}' does not support docstring generation.")
    
    backend = get_backend_for_model(req.model)
    doc = generate_docstring(backend, req.code, few_shot=req.few_shot)
    return {"docstring": doc}


@app.post("/translate")
def api_translate(req: TranslateRequest) -> dict:
    from src.config import MODEL_REGISTRY
    if req.model not in MODEL_REGISTRY:
        raise HTTPException(status_code=400, detail=f"Unknown model: {req.model}")
    cfg = MODEL_REGISTRY[req.model]
    if not cfg.get("supports_translation", True):
        raise HTTPException(status_code=400, detail=f"Model '{req.model}' does not support translation.")
    
    backend = get_backend_for_model(req.model)
    return translate_pl(req.code, req.source_lang, req.target_lang, model=backend)


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
        st.subheader("NL → Python Generation")
        prompt = st.text_area("Prompt / Problem Description", value="Write a function that checks if a string is a palindrome.", key="gen_prompt_area")
        col1, col2 = st.columns(2)
        from src.config import MODEL_REGISTRY, PIPELINE_MODES
        with col1:
            model_name = st.selectbox("Model Backend", list(MODEL_REGISTRY.keys()), format_func=lambda x: MODEL_REGISTRY[x].get("name", x), key="gen_model_select")
            ui_mode = st.selectbox(
                "Generation Mode",
                ["Baseline", "RAG", "Agentic", "Agentic + RAG"],
                key="gen_mode_select"
            )
            mode_mapping = {
                "Baseline": ("baseline", False),
                "RAG": ("rag", True),
                "Agentic": ("agentic", False),
                "Agentic + RAG": ("agentic", True)
            }
            mode, use_rag_cb = mode_mapping[ui_mode]
            
            if use_rag_cb:
                project_dir = st.text_input("Project Folder Path (RAG)", value="", key="gen_project_dir_input")
            else:
                project_dir = ""
        with col2:
            temp = st.slider("Temperature", 0.0, 1.0, 0.0, key="gen_temp_slider")
            max_tokens = st.number_input("Max New Tokens", 64, 1024, 384, key="gen_tokens_num")

        if st.button("Generate Code", key="btn_generate_code"):
            with st.spinner("Generating solution..."):
                res = generate_code_for_task(
                    prompt,
                    model_name=model_name,
                    mode=mode,
                    temperature=temp,
                    max_new_tokens=int(max_tokens),
                    project_dir=project_dir if project_dir.strip() else None,
                    use_rag=use_rag_cb,
                )
                st.code(res.code, language="python")
                st.info(f"Status: {res.status} | Passed: {res.passed}")

    with tab2:
        st.subheader("Code → Google-style Docstring")
        doc_model_options = [k for k, v in MODEL_REGISTRY.items() if v.get("supports_docs", True)]
        doc_model = st.selectbox("Model Backend (Docs)", doc_model_options, format_func=lambda x: MODEL_REGISTRY[x].get("name", x), key="doc_model_select")
        code_input = st.text_area("Python Code", value="def add(a, b):\n    return a + b", height=150, key="doc_code_input")
        if st.button("Generate Docstring", key="btn_generate_doc"):
            with st.spinner("Generating..."):
                backend = get_backend_for_model(doc_model)
                doc = generate_docstring(backend, code_input)
                st.code(doc, language="markdown")

    with tab3:
        st.subheader("Programming Language Translation")
        col_a, col_b = st.columns(2)
        with col_a:
            src_lang = st.selectbox("Source Language", list(LANG_LABELS.keys()), index=0, key="trans_src_lang")
        with col_b:
            tgt_lang = st.selectbox("Target Language", list(LANG_LABELS.keys()), index=1, key="trans_tgt_lang")
        
        trans_model_options = [k for k, v in MODEL_REGISTRY.items() if v.get("supports_translation", True)]
        trans_model = st.selectbox("Model Backend (Translation)", trans_model_options, format_func=lambda x: MODEL_REGISTRY[x].get("name", x), key="trans_model_select")
        src_code = st.text_area("Source Code", value="def is_even(n):\n    return n % 2 == 0", height=150, key="trans_src_code")
        if st.button("Translate", key="btn_translate_code"):
            with st.spinner("Translating..."):
                backend = get_backend_for_model(trans_model)
                out = translate_pl(src_code, src_lang, tgt_lang, model=backend)
                st.code(out.get("output", ""), language=tgt_lang)


if __name__ == "__main__":
    run_streamlit_ui()
