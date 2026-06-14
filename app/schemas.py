from typing import Literal

from pydantic import BaseModel, Field


class CodeGenerateRequest(BaseModel):
    prompt: str
    model: str | None = None
    max_new_tokens: int = 150
    temperature: float = 0.0
    test_list: list[str] = Field(default_factory=list)
    use_agent: bool = False
    max_retries: int = 3


class DocGenerateRequest(BaseModel):
    code: str
    model: str | None = None
    max_new_tokens: int = 128
    temperature: float = 0.0
    reference_documentation: str | None = None


class CodeEvaluateRequest(BaseModel):
    generated_code: str
    test_list: list[str] = Field(default_factory=list)


class AgentCodeRequest(BaseModel):
    prompt: str
    test_list: list[str] = Field(default_factory=list)
    model: str | None = None
    max_retries: int = 3
    max_new_tokens: int = 150
    temperature: float = 0.0


class AgentStepResponse(BaseModel):
    attempt: int
    generated_code: str
    passed: bool


class AgentCodeResponse(BaseModel):
    final_code: str
    passed: bool
    attempts: int
    steps: list[AgentStepResponse]


class EvalRunRequest(BaseModel):
    task: Literal["code", "docs"]
    model: str | None = None
    split: str = "test"
    max_samples: int | None = 10
    agent: bool = False
    max_retries: int = 3


class GenerateResponse(BaseModel):
    output: str
    passed: bool | None = None
    attempts: int | None = None
    steps: list[AgentStepResponse] | None = None
    metrics: dict | None = None


class CodeEvaluateResponse(BaseModel):
    passed: bool


class EvalRunResponse(BaseModel):
    metrics: dict
    n_samples: int
    output_path: str | None = None
