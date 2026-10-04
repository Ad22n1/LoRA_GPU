"""Registry of the models used in the project, with per-model defaults.

Every entry gives the target module suffixes (so that architectures with other names are
handled explicitly rather than silently skipped) and batch settings that fit on a 24 GB
RTX A5000 in bf16 with gradient checkpointing.  ``role`` documents what the model is for:

* ``smoke``     : quick end-to-end tests (minutes)
* ``main``      : the full factorial grid (5 seeds, 3 ranks)
* ``confirm``   : confirmation grid on 7B (3 seeds, rank 16)
* ``dynamics``  : axis A, many checkpoints, small models
"""
from __future__ import annotations

from dataclasses import dataclass, field

LLAMA_LIKE: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
NEOX_LIKE: tuple[str, ...] = ("query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h")


@dataclass(frozen=True)
class ModelSpec:
    id: str
    family: str
    params_b: float
    role: str
    target_modules: tuple[str, ...] = LLAMA_LIKE
    batch_size: int = 8
    grad_accum: int = 4
    max_len: int = 512
    gated: bool = False
    notes: str = ""
    # projections that are fused (e.g. QKV in GPT-NeoX): per-projection analysis impossible
    fused: tuple[str, ...] = field(default_factory=tuple)


MODELS: dict[str, ModelSpec] = {
    # ---- smoke / dynamics ---------------------------------------------------
    "Qwen/Qwen2.5-0.5B": ModelSpec("Qwen/Qwen2.5-0.5B", "qwen2", 0.5, "smoke",
                                  batch_size=8, grad_accum=1, notes="smoke test target: < 10 min per run"),
    # same family as the 0.5B model, three times its size, with the same batch settings: the one
    # thing that changes is the scale. Used for the common-rate inversion at scale (22/09).
    "Qwen/Qwen2.5-1.5B": ModelSpec("Qwen/Qwen2.5-1.5B", "qwen2", 1.5, "main",
                                  batch_size=2, grad_accum=16,
                                  notes="scale check of the common-rate inversion; the grids set 2 x 16 = 32, as for the 0.5B"),
    "EleutherAI/pythia-410m": ModelSpec("EleutherAI/pythia-410m", "gpt_neox", 0.41, "dynamics",
                                       target_modules=NEOX_LIKE, batch_size=16, grad_accum=1,
                                       fused=("query_key_value",),
                                       notes="fused QKV: only whole-attention-input analysis; 154 intermediate "
                                             "checkpoints available on the Hub for pre-training-time comparisons"),
    # ---- main grid (1B) -----------------------------------------------------
    "meta-llama/Llama-3.2-1B": ModelSpec("meta-llama/Llama-3.2-1B", "llama", 1.24, "main", gated=True,
                                         batch_size=8, grad_accum=4,
                                         notes="same model as Rachmat et al. (EvalLLM 2026); request HF access early"),
    "allenai/OLMo-2-0425-1B": ModelSpec("allenai/OLMo-2-0425-1B", "olmo2", 1.5, "main",
                                        batch_size=8, grad_accum=4,
                                        notes="fully open pre-training data (Dolma/OLMo-mix): lets us *verify* that "
                                              "entities and the alien format are absent from pre-training"),
    "HuggingFaceTB/SmolLM2-1.7B": ModelSpec("HuggingFaceTB/SmolLM2-1.7B", "llama", 1.7, "dynamics",
                                            batch_size=8, grad_accum=4,
                                            notes="Llama architecture, open data, no gating; fallback if Llama access is slow"),
    # ---- confirmation (7-8B) ------------------------------------------------
    "Qwen/Qwen2.5-7B": ModelSpec("Qwen/Qwen2.5-7B", "qwen2", 7.6, "confirm",
                                batch_size=2, grad_accum=16,
                                notes="used by MiCA and Rachmat et al. -> comparable to the 'bottom' camp"),
    "mistralai/Mistral-7B-v0.3": ModelSpec("mistralai/Mistral-7B-v0.3", "mistral", 7.2, "confirm",
                                          batch_size=2, grad_accum=16,
                                          notes="used by LoRA-XS -> comparable to the 'top' camp; Apache 2.0, no gating"),
    "meta-llama/Llama-3.1-8B": ModelSpec("meta-llama/Llama-3.1-8B", "llama", 8.0, "confirm", gated=True,
                                         batch_size=1, grad_accum=32, max_len=384,
                                         notes="tight on 24 GB; only if time permits"),
}


def get_spec(model_id: str) -> ModelSpec | None:
    return MODELS.get(model_id)


def target_modules_for(model_id: str) -> tuple[str, ...]:
    spec = MODELS.get(model_id)
    return spec.target_modules if spec else LLAMA_LIKE


def by_role(role: str) -> list[ModelSpec]:
    return [m for m in MODELS.values() if m.role == role]


def describe() -> str:
    lines = [f"{'model':32} {'family':9} {'B':>5} {'role':9} {'bs':>3} {'acc':>4} {'len':>4} gated  notes"]
    for m in MODELS.values():
        lines.append(f"{m.id:32} {m.family:9} {m.params_b:5.2f} {m.role:9} {m.batch_size:3d} {m.grad_accum:4d} "
                     f"{m.max_len:4d} {'yes' if m.gated else 'no ':5}  {m.notes}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(describe())
