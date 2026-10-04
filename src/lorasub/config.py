"""Run configuration: a dataclass loaded from YAML, with a deterministic run_id."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import yaml

from . import expand_path
from .lora import SALTED_MODES
from .models import get_spec, target_modules_for

# Fields that do not affect the scientific content of a run and are excluded from run_id.
# Fields that do not affect the scientific content of a run (paths, logging, evaluation size,
# checkpoint bookkeeping) are excluded from run_id so that changing them never duplicates a run.
# ---------------------------------------------------------------------------------------------
# WHAT MAKES TWO RUNS THE SAME EXPERIMENT. One definition, referred to everywhere.
#
# Five places used to answer this question with five hand-written lists: `run_id` here, and `KEY`,
# `_refuse_if_mixed`, `_keep_selected_lr` and `select_lr` in aggregate.py, plus the key built inline
# in launch_grid.py. None of them was complete, and every gap produced a bug that reached the paper:
#
#   * `band_frac` missing from the selection key -> the five positions of a band sweep collapsed
#     into one cell and received ONE rate, so a "per-position" sweep ran at a common rate;
#   * `max_grad_norm` missing from select_lr -> the noclip reruns dragged a cell's mean down and
#     flipped the selected rate;
#   * the seed missing -> a rate chosen on the very seeds the paper reports;
#   * no replicate rule at all -> a published number decided by the order of a directory listing.
#
# CELL_FIELDS identify a point the paper reports. SETTING_FIELDS must be constant inside a cell or
# its runs did not measure the same thing. `lr` is neither: it varies inside a cell during a sweep,
# and is pinned when reporting. Two runs agreeing on all of these plus `lr` and `seed` are
# REPLICATES, and a replicate is averaged, never picked.
CELL_FIELDS: tuple[str, ...] = ("model", "task", "mode", "band_frac", "rank")

SETTING_FIELDS: tuple[str, ...] = (
    "format_n", "mcq_n",                  # how many items the score was measured on
    "val_format_n", "val_mcq_n",          # and how many the SELECTION was measured on
    "subspace_salted",                    # which version of a drawing arm
    "update_norm_target",                 # the magnitude ablation is a different experiment
    "max_grad_norm",                      # so is a run without clipping
    "n_modules",                          # and one that adapts a different set of modules
    "data_fingerprint",                   # and one trained on different data
    "_metric_version",                    # and one scored by a different checker
)

_NON_ID_FIELDS = {"svd_cache", "data_dir", "out_dir", "log_every", "eval_limit", "attn_implementation",
                  "num_workers", "eval_forgetting", "eval_forgetting_limit", "tiny_model_config", "notes",
                  "save_factor_steps", "save_full_delta", "gradient_checkpointing"}


@dataclass
class RunConfig:
    model: str = "meta-llama/Llama-3.2-1B"
    task: str = "facts"  # facts | format | commonsense
    mode: str = "bottom"  # free | top | bottom | random | band | full | baseline (eval only)
    rank: int = 16
    band_frac: float | None = None  # mode=band: position of the r-band in the spectrum (0 top .. 1 bottom)
    per_module: dict | None = None  # mode=per_module: {q_proj: top, k_proj: bottom, default: random, ...}
    update_norm_target: float | None = None  # magnitude ablation: rescale the final update to this mean ||dW||_F
    alpha: float | None = None  # default: alpha = rank (scaling 1)
    lr: float = 5e-4
    seed: int = 0
    # bottom-r: ignore singular directions below this fraction of sigma_max (numerically null ones,
    # whose singular vectors are ill-conditioned). None = use the reduced SVD as is. Enters run_id.
    # Fingerprint of the dataset files actually used. WITHOUT IT, a run produced on an earlier
    # version of a task is silently reused and averaged with the current one: the format task was
    # rebuilt three times and the facts task twice, and `data_dir` is excluded from the id, so
    # nothing in the repository could detect the mixing after the fact. Same class of bug as the
    # eval_limit drift, with a larger blast node. Computed by `fingerprint_data`; enters run_id.
    data_fingerprint: str = ""
    # Was hard-coded at 128 in eval.py, outside the config and outside the run id. On CoNLL sentences
    # with many mentions, the gold itself — names rewritten, lists sorted, count spelled out,
    # signature — can exceed that, so format_parsed_5+ent would have a ceiling below 1 that cannot be
    # told apart from a failure. Enters the run id: it changes what is measured.
    max_new_tokens: int = 128
    # Whether the `random` / `random_ortho` draw is salted BY MODULE. False reproduces the original
    # behaviour, where every module of the same size received the same columns — which is a legitimate
    # arm ("one random spectral position applied everywhere"), just not the one we wanted. Making this
    # a field rather than deleting the old runs keeps them as valid measurements of that other arm,
    # and costs five lines instead of twelve hours of PARTITION. Enters the run id when True.
    #
    # DEFAULT TRUE since 13/09. It was False, and fourteen of the sixteen grids never set it — including
    # null_subspace_1b, whose entire object is the null distribution of random subspaces and which would
    # therefore have drawn TWO positions per seed instead of 112, and ortho_control_1b, the control that
    # identifies the paper's thesis. The fix had been written, commented, and applied to two grids while
    # the other fourteen silently reproduced the bug. A correct default is not negotiable against
    # backwards compatibility: compatibility is handled at READ time (id_dict drops the field when it is
    # False, so the 239 historical runs keep their identifiers), not by leaving the default on the bug.
    subspace_salted: bool = True
    # Weight each micro-batch by its supervised-token count instead of equally. See train.py. In the
    # run id, and False by default: it changes the objective, so runs with and without must never be
    # averaged, and the 239 existing runs were produced without it. Turn it on for a fresh campaign.
    token_weighted_loss: bool = False
    # mode=learned (phase 2, 28/09): the file of per-module bases (B frozen on them) and the MD5 of its CONTENT. Only the
    # MD5 enters the run id — two different bases can never share an id, and the same bases under another path do —
    # and both fields are dropped for every other mode, so that no existing run_id changes.
    learned_basis: str | None = None
    learned_basis_md5: str = ""
    # PROTOCOL_answer_only.md : sur OpenBookQA / ARC, perte sur les tokens de la reponse seule (memes blocs). Absent de l'identifiant quand faux.
    answer_only: bool = False
    sigma_rel_tol: float | None = None
    subspace_seed: int | None = None  # random arm: which columns of U; defaults to seed (documented conflation)
    epochs: float = 3.0
    max_steps: int | None = None  # overrides epochs if set
    # overrides epochs: steps chosen so that every arm sees this many supervised tokens.
    # int  -> same budget for every task (comparable across arms, but the two corpora have very
    #         different sizes, so one task may see 7 epochs while the other sees 3);
    # dict -> one budget per task, e.g. {facts: 1800000, format: 600000}; resolved at load time so
    #         run_id still sees a plain int.
    target_tokens: int | dict | None = None
    batch_size: int = 8
    grad_accum: int = 4
    max_len: int = 512
    warmup_ratio: float = 0.05
    weight_decay: float = 0.0
    adam_eps: float = 1e-8  # Adam's epsilon; torch's default. Out of the id at this value (id_dict).
    lora_dropout: float = 0.0  # adapter dropout, mode mica_peft only (PROTOCOL_real_mica.md). Out of the id at 0 (id_dict).
    max_grad_norm: float = 1.0
    target_modules: list[str] | None = None  # None -> from the model registry (lorasub.models)
    save_factor_steps: list = field(default_factory=lambda: [0, 10, 50, 100, 300, 1000, "end"])
    save_full_delta: bool = False  # for mode=full: store W - W0 at save steps (heavy)
    dtype: str = "bfloat16"  # bfloat16 | float32
    attn_implementation: str | None = None  # None -> auto (flash_attention_2 if available else sdpa)
    gradient_checkpointing: bool = True
    svd_cache: str = "$SHARED/svd"
    data_dir: str = "$SHARED/data"
    out_dir: str = "$SHARED/runs"
    log_every: int = 10
    num_workers: int = 2
    # evaluation
    eval_limit: int = 1000
    eval_forgetting: bool = True
    eval_forgetting_limit: int = 1000
    # test-only: build a tiny random model from this dict instead of loading `model`
    tiny_model_config: dict | None = None
    notes: str = ""

    def _coerce_types(self) -> None:
        """Force the declared type of every field.

        YAML 1.1 does not recognise ``5e-4`` as a float (it needs a dot or a signed exponent:
        ``5.0e-4``), so ``lr=5e-4`` on the command line arrives as the *string* ``"5e-4"``.  Left
        alone it crashes the optimiser — and, worse, it would give a different ``run_id`` than the
        same run written ``5.0e-4``, silently duplicating grid cells.  Coercing here makes the two
        spellings identical everywhere.
        """
        int_fields = ("rank", "seed", "subspace_seed", "max_steps", "batch_size", "grad_accum", "max_len",
                      "max_new_tokens",
                      "log_every", "num_workers", "eval_limit", "eval_forgetting_limit", "target_tokens")
        float_fields = ("lr", "alpha", "epochs", "warmup_ratio", "weight_decay", "adam_eps", "max_grad_norm", "lora_dropout",
                        "band_frac", "update_norm_target", "sigma_rel_tol")
        bool_fields = ("gradient_checkpointing", "eval_forgetting", "save_full_delta",
                       "subspace_salted", "token_weighted_loss")   # else `subspace_salted=1` hashes as 1, not true
        for name in int_fields:
            v = getattr(self, name, None)
            if isinstance(v, str):
                setattr(self, name, int(float(v)))
        for name in float_fields:
            v = getattr(self, name, None)
            if isinstance(v, str):
                setattr(self, name, float(v))
        for name in bool_fields:
            v = getattr(self, name, None)
            if isinstance(v, str):
                if v.lower() not in ("true", "false", "yes", "no", "1", "0"):
                    raise ValueError(f"{name}: expected a boolean, got {v!r}")
                setattr(self, name, v.lower() in ("true", "yes", "1"))
            elif isinstance(v, int) and not isinstance(v, bool):
                # an int is not a string, so it used to slip through: 1 stayed 1, which is truthy and
                # therefore entered the identifier, but json.dumps writes 1 and not true — a different
                # run_id for the same run, while results.csv wrote bool(...) = True
                setattr(self, name, bool(v))

    def __post_init__(self):
        self._coerce_types()
        if self.mode not in ("free", "top", "bottom", "random", "random_ortho", "band", "per_module",
                             "dual_top", "dual_bottom", "dual_random", "dual_random_ortho",
                             "init_top", "init_bottom", "init_random", "top_sigma", "top_unfrozen", "top_scalar", "bottom_sigma", "bottom_unfrozen", "top_lrscale",
                             "learned", "full", "baseline", "mica_peft", "lora_peft"):
            raise ValueError(f"bad mode {self.mode!r}")
        if self.mode == "band" and (self.band_frac is None or not 0.0 <= self.band_frac <= 1.0):
            raise ValueError("mode=band requires band_frac in [0, 1]")
        if self.mode != "band":
            self.band_frac = None
        if self.mode == "per_module" and not self.per_module:
            raise ValueError("mode=per_module requires a per_module dict")
        if self.mode != "per_module":
            self.per_module = None
        if self.answer_only and self.task != "commonsense":
            raise ValueError("answer_only ne concerne que les taches a blocs concatenes (OpenBookQA, ARC)")
        if self.mode == "learned":
            if not self.learned_basis:
                raise ValueError("mode=learned requires learned_basis (path of the basis file)")
            self.learned_basis = expand_path(self.learned_basis)
            _p = Path(self.learned_basis)
            if not _p.is_file():
                raise ValueError(f"mode=learned: basis file not found: {_p}")
            _md5 = hashlib.md5(_p.read_bytes()).hexdigest()
            if self.learned_basis_md5 and self.learned_basis_md5 != _md5:
                raise ValueError(f"learned_basis_md5 {self.learned_basis_md5} does not match the file ({_md5}): the bases changed")
            self.learned_basis_md5 = _md5
        else:
            self.learned_basis, self.learned_basis_md5 = None, ""
        if self.task not in ("facts", "format", "commonsense"):
            raise ValueError(f"bad task {self.task!r}")
        if isinstance(self.target_tokens, dict):
            if self.task not in self.target_tokens:
                raise ValueError(f"target_tokens dict has no entry for task {self.task!r}")
            self.target_tokens = int(self.target_tokens[self.task])
        if self.alpha is None:
            self.alpha = float(self.rank)
        if self.target_modules is None:
            self.target_modules = list(target_modules_for(self.model))
        spec = get_spec(self.model)
        if spec is not None and spec.fused and self.mode in ("top", "bottom", "random"):
            # constrained arms on a fused projection adapt the concatenated QKV matrix as one module
            self.notes = f"fused projections {spec.fused}: per-projection analysis not available"
        self.svd_cache = expand_path(self.svd_cache)
        self.data_dir = expand_path(self.data_dir)
        self.out_dir = expand_path(self.out_dir)

    def id_dict(self) -> dict:
        d = asdict(self)
        for k in _NON_ID_FIELDS:
            d.pop(k, None)
        # An EMPTY fingerprint is dropped from the identifier so that the 224 runs produced before
        # this field existed keep their ids. Set it (or pass --fingerprint_data to the launcher) and
        # it enters the id, which is what the final grids must do: a run produced on an earlier
        # version of a task would otherwise be silently reused and averaged with the current one.
        if not d.get("data_fingerprint"):
            d.pop("data_fingerprint", None)
        # Same precaution for max_new_tokens: it was hard-coded at 128 before becoming a field, so
        # leaving the default in the identifier would change EVERY existing run_id and relaunch the
        # whole database. Any other value does separate runs, which is the property we wanted.
        if d.get("max_new_tokens") == 128:
            d.pop("max_new_tokens", None)
        # same principle: the historical value stays out, so the 228 existing runs keep their ids.
        # And it ONLY enters the id for the arms it changes: now that the default is True, leaving it
        # in for `top`, `bottom`, `free` and `full` would have given every one of the 239 existing runs
        # a new identifier and orphaned the whole database. The salt describes how the random draw is
        # made; on an arm that draws nothing it is not a property of the run.
        if not d.get("subspace_salted") or d.get("mode") not in SALTED_MODES:
            d.pop("subspace_salted", None)
        # same reasoning: absent from the id at its historical value, present as soon as it deviates
        if not d.get("token_weighted_loss"):
            d.pop("token_weighted_loss", None)
        # same principle: Adam's epsilon at torch's default (the value every existing run used) stays out of the id
        if d.get("adam_eps") == 1e-8:
            d.pop("adam_eps", None)
        # same principle: no adapter dropout, the value of every existing run, stays out of the id
        if not d.get("lora_dropout"):
            d.pop("lora_dropout", None)
        # mode=learned: the bases enter by the MD5 of their content, never by their path; absent for every other mode
        d.pop("learned_basis", None)
        if not d.get("answer_only"):
            d.pop("answer_only", None)
        if not d.get("learned_basis_md5"):
            d.pop("learned_basis_md5", None)
        return d

    @property
    def run_id(self) -> str:
        s = json.dumps(self.id_dict(), sort_keys=True, default=str)
        return hashlib.sha1(s.encode()).hexdigest()[:12]

    @property
    def run_dir(self) -> Path:
        return Path(self.out_dir) / self.run_id

    def to_yaml(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.safe_dump(asdict(self), f, sort_keys=True)


def apply_model_defaults(raw: dict) -> dict:
    """Fill batch_size / grad_accum / max_len from the registry when the YAML does not set them."""
    spec = get_spec(raw.get("model", RunConfig.model))
    if spec is None:
        return raw
    for k in ("batch_size", "grad_accum", "max_len"):
        raw.setdefault(k, getattr(spec, k))
    return raw


def load_config(path: str | Path, overrides: list[str] | None = None) -> RunConfig:
    """Load a YAML config; ``overrides`` are ``key=value`` strings parsed as YAML scalars."""
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    for ov in overrides or []:
        k, _, v = ov.partition("=")
        raw[k.strip()] = yaml.safe_load(v)
    known = {f.name for f in fields(RunConfig)}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"unknown config keys: {sorted(unknown)}")
    return RunConfig(**apply_model_defaults(raw))
