"""Shared model loading and text generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

from src.models.registry import DEFAULT_MODEL_ID, resolve_hub_path

DEFAULT_MODEL = resolve_hub_path(DEFAULT_MODEL_ID)

DOC_STOP_STRINGS = [
    "\nclass ",
    "\ndef ",
    "\nimport ",
    "\nhttp://",
    "\nhttps://",
    "\n# Companies",
]


class _StopOnSequences(StoppingCriteria):
    def __init__(self, stop_sequences: list[list[int]]) -> None:
        self.stop_sequences = stop_sequences

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if input_ids.numel() == 0:
            return False
        seq = input_ids[0].tolist()
        for stop in self.stop_sequences:
            if len(seq) >= len(stop) and seq[-len(stop) :] == stop:
                return True
        return False


def _resolve_model_path(model_name_or_path: str) -> str:
    path = Path(model_name_or_path)
    if path.is_dir() or path.is_file():
        return str(path.resolve())
    return model_name_or_path


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _sampling_kwargs(temperature: float, top_p: float, seed: int | None) -> dict:
    if temperature > 0:
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        return {"do_sample": True, "temperature": temperature, "top_p": top_p}
    return {"do_sample": False}


class CodeGenerator:
    """Lazy-loaded causal LM (CodeGen family)."""

    def __init__(self, model_name_or_path: str = DEFAULT_MODEL) -> None:
        self.model_name_or_path = _resolve_model_path(model_name_or_path)
        self._tokenizer: AutoTokenizer | None = None
        self._model = None
        self._device = get_device()

    def load(self) -> None:
        if self._model is not None:
            return
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=True)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        adapter_config = Path(self.model_name_or_path) / "adapter_config.json"
        if adapter_config.is_file():
            with adapter_config.open("r", encoding="utf-8") as f:
                base_model_name = json.load(f)["base_model_name_or_path"]
            base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
            self._model = PeftModel.from_pretrained(base_model, self.model_name_or_path)
        else:
            self._model = AutoModelForCausalLM.from_pretrained(self.model_name_or_path)

        self._model.to(self._device)
        if getattr(self._model.config, "pad_token_id", None) is None:
            self._model.config.pad_token_id = self._tokenizer.pad_token_id

    def _max_context(self) -> int:
        assert self._model is not None
        return int(
            getattr(self._model.config, "max_position_embeddings", None)
            or getattr(self._model.config, "n_positions", None)
            or 2048
        )

    def generate(
        self,
        prompt_text: str,
        max_new_tokens: int = 150,
        temperature: float = 0.0,
        top_p: float = 0.95,
        seed: int | None = 42,
        stop_strings: list[str] | None = None,
        min_new_tokens: int = 0,
    ) -> str:
        self.load()
        assert self._tokenizer is not None and self._model is not None

        max_ctx = self._max_context()
        max_new_tokens = min(max_new_tokens, max(1, max_ctx - 16))
        max_input_tokens = max(1, max_ctx - max_new_tokens)

        inputs = self._tokenizer(
            prompt_text,
            return_tensors="pt",
            truncation=True,
            max_length=max_input_tokens,
        ).to(self._device)
        prompt_used = self._tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)

        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": self._tokenizer.pad_token_id,
            "eos_token_id": self._tokenizer.eos_token_id,
            **_sampling_kwargs(temperature, top_p, seed),
        }
        if min_new_tokens > 0:
            # A prompt that ends with a closing docstring looks like end-of-file to
            # CodeGen, so greedy decoding emits EOS immediately and returns "".
            gen_kwargs["min_new_tokens"] = min(min_new_tokens, max_new_tokens)
        if stop_strings:
            stop_ids = []
            for s in stop_strings:
                ids = self._tokenizer.encode(s, add_special_tokens=False)
                if ids:
                    stop_ids.append(ids)
            if stop_ids:
                gen_kwargs["stopping_criteria"] = StoppingCriteriaList([_StopOnSequences(stop_ids)])

        with torch.no_grad():
            outputs = self._model.generate(**inputs, **gen_kwargs)

        full_output = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
        if full_output.startswith(prompt_used):
            return full_output[len(prompt_used) :].strip()
        if full_output.startswith(prompt_text):
            return full_output[len(prompt_text) :].strip()
        return full_output.replace(prompt_used, "", 1).replace(prompt_text, "", 1).strip()


_generator_cache: dict[str, CodeGenerator] = {}


def get_generator(model_name_or_path: Optional[str] = None) -> CodeGenerator:
    hub_path = _resolve_model_path(resolve_hub_path(model_name_or_path))
    if hub_path not in _generator_cache:
        _generator_cache[hub_path] = CodeGenerator(hub_path)
    return _generator_cache[hub_path]
