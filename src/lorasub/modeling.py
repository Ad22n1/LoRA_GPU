"""Model / tokenizer loading with the conventions used everywhere in the project."""
from __future__ import annotations

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

DTYPES = {"bfloat16": torch.bfloat16, "float32": torch.float32, "float16": torch.float16}


def pick_attn_implementation(explicit: str | None = None) -> str:
    if explicit:
        return explicit
    try:
        __import__("flash_attn")
        return "flash_attention_2" if torch.cuda.is_available() else "sdpa"
    except Exception:
        return "sdpa"


def load_tokenizer(model_id: str):
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    return tok


def load_model(model_id: str, dtype: str = "bfloat16", attn_implementation: str | None = None, device=None):
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch_dtype = DTYPES[dtype] if dev != "cpu" or dtype == "float32" else torch.float32
    kw = dict(attn_implementation=pick_attn_implementation(attn_implementation))
    try:  # transformers >= 4.56 uses `dtype`, older versions `torch_dtype`
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch_dtype, **kw)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch_dtype, **kw)
    model.to(dev)
    return model


def build_tiny_model(cfg: dict, device="cpu"):
    """Random Llama-architecture model from a config dict (tests / smoke runs; no download)."""
    config = AutoConfig.for_model("llama", **cfg)
    model = AutoModelForCausalLM.from_config(config, attn_implementation="eager")
    return model.to(device)


def n_params(model) -> int:
    return sum(p.numel() for p in model.parameters())


def enable_checkpointing(model) -> None:
    """Gradient checkpointing that works with frozen embeddings and LoRA on every transformers version.

    Non-reentrant checkpointing does not need inputs with requires_grad; on older versions that
    do not accept ``gradient_checkpointing_kwargs`` we fall back to the reentrant variant plus
    ``enable_input_require_grads`` (the peft recipe)."""
    try:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    except TypeError:
        model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
