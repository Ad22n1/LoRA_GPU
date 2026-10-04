"""The real MiCA method, as released in PEFT 0.21.0 (PROTOCOL_real_mica.md), for the mode `mica_peft` only.

PEFT 0.21.0 is imported from a separate directory (PEFT_DIR, default ~/.peft021, PEFT alone), placed FIRST on the path for this mode
only; another PEFT already loaded is unloaded first. No other run is affected. B is set by PEFT's own `mica_init` (its own fp32 SVD, not our cache), A = 0, B frozen; only A trains.
"""
import os
import sys

from torch import nn


def inject_mica_peft(model: nn.Module, cfg, expected: int, init="mica") -> int:
    """init="mica": MiCA (B frozen on the minor singular vectors, A = 0). init=True: PEFT's standard LoRA (mode lora_peft,
    ADDENDUM_real_mica_2.md), A and B both trained, PEFT's default initialisation; same code path, same recipe."""
    peft_dir = os.path.expanduser(os.environ.get("PEFT_DIR", "~/.peft021"))
    # The environment may hold another PEFT (requirements.txt), possibly already imported (e.g. by transformers): PEFT_DIR goes FIRST on
    # the path, and any other loaded version is unloaded before importing. PEFT_DIR holds PEFT alone (installed with --no-deps).
    if peft_dir in sys.path:
        sys.path.remove(peft_dir)
    sys.path.insert(0, peft_dir)
    if "peft" in sys.modules and getattr(sys.modules["peft"], "__version__", "") != "0.21.0":
        for name in [m for m in sys.modules if m == "peft" or m.startswith("peft.")]:
            del sys.modules[name]
    import peft
    from peft import LoraConfig, inject_adapter_in_model
    if peft.__version__ != "0.21.0":
        raise RuntimeError(f"mica_peft: PEFT {peft.__version__} imported from {peft.__file__}, 0.21.0 required")
    for p in model.parameters():
        p.requires_grad_(False)
    conf = LoraConfig(r=cfg.rank, lora_alpha=float(cfg.alpha), lora_dropout=float(cfg.lora_dropout), init_lora_weights=init,
                      target_modules=list(cfg.target_modules), bias="none")
    inject_adapter_in_model(conf, model)
    got = sum(p.numel() for p in model.parameters() if p.requires_grad)
    b_grad = [m.lora_B["default"].weight.requires_grad for m in model.modules() if hasattr(m, "lora_B") and "default" in m.lora_B]
    ok_b = (not any(b_grad)) if init == "mica" else all(b_grad)
    if got != expected or not ok_b:
        raise RuntimeError(f"{'mica' if init == 'mica' else 'lora'}_peft: {got} trainable parameters (expected {expected}); "
                           f"B {'frozen' if init == 'mica' else 'trained'} everywhere: {ok_b}")
    return got
