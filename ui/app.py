"""Streamlit UI for CodeGen capstone."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import httpx
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.registry import DEFAULT_MODEL_ID, list_model_options

API_URL = os.environ.get("CODEGEN_API_URL", "http://127.0.0.1:8000")
DEFAULT_CODE_MODEL = "experiments/checkpoints/codegen-mbpp-full"
DEFAULT_DOC_MODEL = "experiments/checkpoints/codegen-doc-full"


def _load_model_options() -> list[dict]:
    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.get(f"{API_URL.rstrip('/')}/models")
            resp.raise_for_status()
            return resp.json().get("models", [])
    except Exception:
        return list_model_options()


def _model_ids(options: list[dict]) -> list[str]:
    return [m["id"] for m in options]


def _model_label(options: list[dict], model_id: str) -> str:
    for m in options:
        if m["id"] == model_id:
            return m["label"]
    return model_id


def _index_of(options: list[dict], model_id: str, fallback: str) -> int:
    ids = _model_ids(options)
    if model_id in ids:
        return ids.index(model_id)
    if fallback in ids:
        return ids.index(fallback)
    return 0


def _pick_default(options: list[dict], preferred: str, fallback: str) -> str:
    ids = _model_ids(options)
    if preferred in ids:
        return preferred
    if fallback in ids:
        return fallback
    return ids[0] if ids else fallback


def _post(api_url: str, path: str, payload: dict) -> dict:
    with httpx.Client(timeout=600.0) as client:
        resp = client.post(f"{api_url.rstrip('/')}{path}", json=payload)
        resp.raise_for_status()
        return resp.json()


def _show_agent_steps(steps: list[dict]) -> None:
    st.markdown("**Pipeline steps**")
    for step in steps:
        status = "✅" if step["passed"] else "❌"
        with st.expander(f"{status} Attempt {step['attempt']}"):
            st.code(step["generated_code"], language="python")


st.set_page_config(page_title="CodeGen Group14", layout="wide")
st.title("CodeGen — AIML B26 Capstone")
st.caption("Generate → validate → self-correct in one pipeline")

model_options = _load_model_options()
model_ids = _model_ids(model_options)
labels = {m["id"]: m["label"] for m in model_options}

st.sidebar.header("Models")
code_model = st.sidebar.selectbox(
    "Code model",
    model_ids,
    index=_index_of(model_options, DEFAULT_CODE_MODEL, DEFAULT_MODEL_ID),
    format_func=lambda mid: labels.get(mid, mid),
    help="CodeGen-350M upstream, or a fine-tuned MBPP/CoDoc checkpoint",
)
doc_model = st.sidebar.selectbox(
    "Documentation model",
    model_ids,
    index=_index_of(model_options, DEFAULT_DOC_MODEL, DEFAULT_MODEL_ID),
    format_func=lambda mid: labels.get(mid, mid),
    help="Use codegen-doc-full for best ROUGE-L after CP2 training",
)
api_url = st.sidebar.text_input("API URL", API_URL)

with st.sidebar.expander("Model details"):
    for key in (code_model, doc_model):
        match = next((m for m in model_options if m["id"] == key), None)
        if match:
            st.markdown(f"**{match['label']}**")
            st.caption(f"Size: {match['size']}")

tab_code, tab_docs, tab_benchmark = st.tabs(
    ["Text → Code", "Code → Documentation", "Benchmark (batch eval)"]
)

with tab_code:
    st.markdown(
        "Single pipeline: **generate code → run tests → retry on failure** when agent mode is on."
    )
    st.caption(f"Using: {_model_label(model_options, code_model)}")
    prompt = st.text_area("Natural language requirement", height=100)
    tests_raw = st.text_area(
        "Unit tests (optional, one assert per line)",
        height=100,
        placeholder="assert factorial(5) == 120\nassert factorial(0) == 1",
    )
    test_list = [line.strip() for line in tests_raw.splitlines() if line.strip()]
    col1, col2 = st.columns(2)
    with col1:
        use_agent = st.checkbox(
            "Agentic self-correction",
            value=bool(test_list),
            help="When tests are provided, retry with failure context until pass or max retries.",
        )
    with col2:
        max_retries = st.slider("Max retries", 1, 5, 3, disabled=not use_agent)

    if st.button("Run code pipeline", key="gen_code"):
        if not prompt.strip():
            st.warning("Enter a prompt.")
        else:
            with st.spinner("Running pipeline..."):
                result = _post(
                    api_url,
                    "/generate/code",
                    {
                        "prompt": prompt,
                        "model": code_model,
                        "max_new_tokens": 150,
                        "test_list": test_list,
                        "use_agent": use_agent and bool(test_list),
                        "max_retries": max_retries,
                    },
                )
            st.code(result["output"], language="python")
            if test_list:
                if result.get("passed") is True:
                    st.success(f"Tests passed · attempts: {result.get('attempts', 1)}")
                elif result.get("passed") is False:
                    st.error(f"Tests failed · attempts: {result.get('attempts', 1)}")
                if result.get("steps"):
                    _show_agent_steps(result["steps"])

with tab_docs:
    st.markdown(
        "Single pipeline: **generate documentation → score against reference** when provided."
    )
    st.caption(f"Using: {_model_label(model_options, doc_model)}")
    code = st.text_area("Python source code", height=160)
    reference = st.text_area(
        "Reference documentation (optional, for ROUGE-L / BLEU)",
        height=100,
        placeholder="Describe what the function does, params, return value...",
    )
    if st.button("Run documentation pipeline", key="gen_docs"):
        if not code.strip():
            st.warning("Paste some code.")
        else:
            payload = {
                "code": code,
                "model": doc_model,
                "max_new_tokens": 128,
            }
            if reference.strip():
                payload["reference_documentation"] = reference.strip()
            with st.spinner("Running pipeline..."):
                result = _post(api_url, "/generate/docs", payload)
            st.markdown("**Generated documentation**")
            st.markdown(result["output"])
            if result.get("metrics"):
                m1, m2 = st.columns(2)
                m1.metric("ROUGE-L", f"{result['metrics']['rouge_l']:.4f}")
                m2.metric("BLEU", f"{result['metrics']['bleu']:.4f}")
                with st.expander("Reference documentation"):
                    st.markdown(reference)

with tab_benchmark:
    st.markdown(
        "Batch evaluation on dataset splits — for checkpoint reports, not single prompts. "
        "Use the tabs above for interactive generate + validate."
    )
    bench_model = st.selectbox(
        "Benchmark model",
        model_ids,
        index=model_ids.index(code_model) if code_model in model_ids else 0,
        format_func=lambda mid: labels.get(mid, mid),
    )
    task = st.selectbox("Task", ["code", "docs"])
    split = st.selectbox("Split", ["test", "validation", "train", "all"])
    max_samples = st.number_input("Max samples", min_value=1, max_value=500, value=10)
    use_agent = st.checkbox("Agentic self-correction (code only)", value=False)
    max_retries = st.number_input("Agent max retries", min_value=1, max_value=5, value=3)
    if st.button("Run benchmark eval", key="run_eval"):
        payload = {
            "task": task,
            "model": bench_model,
            "split": split,
            "max_samples": int(max_samples),
            "agent": use_agent,
            "max_retries": int(max_retries),
        }
        with st.spinner("Running benchmark..."):
            result = _post(api_url, "/evaluate/run", payload)
        st.json(result)
        if task == "code":
            st.metric("pass@1", f"{result['metrics'].get('pass_at_1', 0):.2%}")
            if use_agent:
                st.metric("Retry success rate", f"{result['metrics'].get('retry_success_rate', 0):.2%}")
        else:
            st.metric("ROUGE-L", f"{result['metrics'].get('rouge_l', 0):.4f}")
            st.metric("BLEU", f"{result['metrics'].get('bleu', 0):.4f}")
