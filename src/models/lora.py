"""LoRA adapter wiring for CodeGen fine-tuning (peft)."""

from __future__ import annotations

from peft import LoraConfig, TaskType, get_peft_model

# CodeGen's attention block uses a fused qkv_proj plus a separate out_proj
# (see CodeGenAttention) -- these are the only attention-linear layers peft
# can target on this architecture.
DEFAULT_TARGET_MODULES = ["qkv_proj", "out_proj"]


def apply_lora(
    model,
    r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    target_modules=None,
):
    """Wrap a CodeGen causal LM with a trainable LoRA adapter, base weights frozen."""
    config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules or DEFAULT_TARGET_MODULES,
        task_type=TaskType.CAUSAL_LM,
    )
    peft_model = get_peft_model(model, config)
    peft_model.print_trainable_parameters()
    return peft_model
