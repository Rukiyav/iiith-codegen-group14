"""FastAPI backend for CodeGen capstone."""

from __future__ import annotations

import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.schemas import (
    AgentCodeRequest,
    AgentCodeResponse,
    AgentStepResponse,
    CodeEvaluateRequest,
    CodeEvaluateResponse,
    CodeGenerateRequest,
    DocGenerateRequest,
    EvalRunRequest,
    EvalRunResponse,
    GenerateResponse,
)
from src.agent.code_agent import agent_generate_code
from src.eval.code_eval import evaluate_functional_correctness
from src.eval.doc_eval import compute_bleu, compute_rouge_l
from src.eval.run_eval import run_code_eval, run_doc_eval, _write_result
from src.generation.code import generate_python_code
from src.generation.docs import generate_documentation
from src.models.registry import DEFAULT_MODEL_ID, list_model_options

app = FastAPI(title="CodeGen Group14 API", version="0.3.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _agent_steps(result) -> list[AgentStepResponse]:
    return [
        AgentStepResponse(
            attempt=s.attempt,
            generated_code=s.generated_code,
            passed=s.passed,
        )
        for s in result.steps
    ]


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/models")
def list_models() -> dict:
    return {"models": list_model_options(), "default": DEFAULT_MODEL_ID}


@app.post("/generate/code", response_model=GenerateResponse)
def generate_code(req: CodeGenerateRequest) -> GenerateResponse:
    model = req.model or DEFAULT_MODEL_ID

    if req.test_list and req.use_agent:
        result = agent_generate_code(
            req.prompt,
            req.test_list,
            model_name_or_path=model,
            max_retries=req.max_retries,
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature,
        )
        return GenerateResponse(
            output=result.final_code,
            passed=result.passed,
            attempts=result.attempts,
            steps=_agent_steps(result),
        )

    output = generate_python_code(
        req.prompt,
        model_name_or_path=model,
        max_new_tokens=req.max_new_tokens,
        temperature=req.temperature,
    )

    if req.test_list:
        passed = evaluate_functional_correctness(output, req.test_list)
        return GenerateResponse(output=output, passed=passed, attempts=1)

    return GenerateResponse(output=output)


@app.post("/generate/docs", response_model=GenerateResponse)
def generate_docs(req: DocGenerateRequest) -> GenerateResponse:
    output = generate_documentation(
        req.code,
        model_name_or_path=req.model or DEFAULT_MODEL_ID,
        max_new_tokens=req.max_new_tokens,
        temperature=req.temperature,
    )
    metrics = None
    reference = (req.reference_documentation or "").strip()
    if reference:
        metrics = {
            "rouge_l": compute_rouge_l([output], [reference]),
            "bleu": compute_bleu([output], [reference]),
        }
    return GenerateResponse(output=output, metrics=metrics)


@app.post("/agent/generate/code", response_model=AgentCodeResponse)
def agent_generate(req: AgentCodeRequest) -> AgentCodeResponse:
    """Legacy endpoint; prefer POST /generate/code with use_agent=true."""
    result = agent_generate_code(
        req.prompt,
        req.test_list,
        model_name_or_path=req.model or DEFAULT_MODEL_ID,
        max_retries=req.max_retries,
        max_new_tokens=req.max_new_tokens,
        temperature=req.temperature,
    )
    return AgentCodeResponse(
        final_code=result.final_code,
        passed=result.passed,
        attempts=result.attempts,
        steps=_agent_steps(result),
    )


@app.post("/evaluate/code", response_model=CodeEvaluateResponse)
def evaluate_code(req: CodeEvaluateRequest) -> CodeEvaluateResponse:
    passed = evaluate_functional_correctness(req.generated_code, req.test_list)
    return CodeEvaluateResponse(passed=passed)


@app.post("/evaluate/run", response_model=EvalRunResponse)
def evaluate_run(req: EvalRunRequest) -> EvalRunResponse:
    model = req.model or DEFAULT_MODEL_ID
    if req.task == "code":
        payload = run_code_eval(
            model,
            req.split,
            req.max_samples,
            use_agent=req.agent,
            max_retries=req.max_retries,
        )
    else:
        payload = run_doc_eval(model, req.split, req.max_samples)
    out = _write_result(payload)
    return EvalRunResponse(
        metrics=payload.get("metrics", {}),
        n_samples=payload.get("n_samples", 0),
        output_path=str(out),
    )
