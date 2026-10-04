"""One config -> one run.

    python -m lorasub.train configs/example.yaml [key=value ...]

Idempotent: if ``<out_dir>/<run_id>/results.csv`` exists the run is skipped.  On failure a
``status`` file containing ``FAILED`` and the traceback is written, so a grid can list and
relaunch failures.  Factors (A, B) — never dW — are saved at ``save_factor_steps``.
"""
from __future__ import annotations

import csv
import os
import uuid
import json
import math
import sys
import time
import traceback
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from . import set_seed
from .config import RunConfig, load_config
from .data.loaders import build_dataset, collate
from .eval import run_all
from .lora import count_trainable, expected_trainable, inject_lora, rescale_update_, save_factors, update_norms
from .modeling import build_tiny_model, enable_checkpointing, load_model, load_tokenizer
from .spectral import compute_svd_cache, iter_target_modules, list_cached


def _cosine_with_warmup(step: int, total: int, warmup: int) -> float:
    if step < warmup:
        return (step + 1) / max(1, warmup)
    p = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, p)))


def _make_optimizer(params, cfg: RunConfig):
    if cfg.mode == "full":
        try:
            import bitsandbytes as bnb

            return bnb.optim.AdamW8bit(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
        except Exception:
            print("[train] bitsandbytes unavailable, falling back to torch AdamW for full FT")
    return torch.optim.AdamW(params, lr=cfg.lr, betas=(0.9, 0.999), eps=cfg.adam_eps, weight_decay=cfg.weight_decay)


def _save_steps(cfg: RunConfig, total_steps: int) -> set[int]:
    s = set()
    for x in cfg.save_factor_steps:
        s.add(total_steps if x == "end" else int(x))
    return {x for x in s if 0 <= x <= total_steps}


def _write_results(run_dir: Path, row: dict) -> None:
    """Write results.csv ATOMICALLY: a temporary file, then a rename.

    Two array tasks can execute the same run_id at once — it happens as soon as two `sbatch` of the
    same grid overlap, since both see the run as not done. Writing in place then produced files with
    an extra half-row ("423210144,499.2" appended after a complete record), whose columns no longer
    match the header, and every metric read from them was wrong. A rename is atomic on POSIX: a reader
    sees either the old file or a complete new one, never a mixture. Duplicated work is wasted compute;
    a corrupt results.csv is a corrupt result.
    """
    dest = run_dir / "results.csv"
    # pid alone is not enough: it is shared by threads, and two writers would collide on the temp
    # file itself. A random suffix makes the temporary name unique per writer.
    tmp = run_dir / f".results.csv.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp"
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        w.writeheader()
        w.writerow(row)
    os.replace(tmp, dest)


def _save_full_delta(model, W0: dict[str, torch.Tensor], path: Path, suffixes) -> None:
    from safetensors.torch import save_file

    t = {}
    for name, lin in iter_target_modules(model, suffixes):
        t[f"{name}.delta"] = (lin.weight.detach().float().cpu() - W0[name]).to(torch.bfloat16).contiguous()
    save_file(t, str(path))


# DERIVED from lora.CONSTRAINED, never listed by hand. It was written here as ("top", "bottom",
# "random", "band", "per_module"), and the top_sigma arm, added to lora.py, was not added here: on
# every node that had never built the cache, its runs died with FileNotFoundError (17 of 20, 22/09).
# The same list had already been missed once, by the init_ and dual_ families.
from .lora import CONSTRAINED as _CONSTRAINED  # noqa: E402

NEEDS_SVD_CACHE: tuple[str, ...] = tuple(_CONSTRAINED) + ("per_module",)
def ensure_svd_cache(model, cfg: RunConfig, log=print) -> None:
    """Compute the SVD cache for this model on this machine if any target module is missing.

    A per-model lock file serialises concurrent jobs on the same node; the others wait and reuse.
    Idempotent: modules already cached are skipped by compute_svd_cache."""
    import fcntl

    expected = [n for n, _ in iter_target_modules(model, cfg.target_modules)]
    have = set(list_cached(cfg.svd_cache, cfg.model))
    if all(n in have for n in expected):
        return
    Path(cfg.svd_cache).mkdir(parents=True, exist_ok=True)
    lock_path = Path(cfg.svd_cache) / (cfg.model.replace("/", "__") + ".lock")
    with open(lock_path, "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)  # another job on this node may be computing it right now
        have = set(list_cached(cfg.svd_cache, cfg.model))
        missing = [n for n in expected if n not in have]
        if missing:
            log(f"SVD cache: {len(missing)} modules missing on this node, computing (once per node)")
            compute_svd_cache(model, cfg.svd_cache, cfg.model, suffixes=cfg.target_modules, verbose=False)
        fcntl.flock(lock, fcntl.LOCK_UN)


def run(cfg: RunConfig, tokenizer=None, model=None, quiet: bool = False) -> dict:
    """Execute one run.  ``tokenizer``/``model`` may be passed for tests (tiny models)."""
    run_dir = cfg.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    results_path = run_dir / "results.csv"
    if results_path.exists():
        if not quiet:
            print(f"[train] {cfg.run_id} already done, skipping")
        return {"run_id": cfg.run_id, "skipped": True}
    # Always RECORD which dataset version this run read, even when the fingerprint does not enter the
    # run id. Without this trace, mixing v1 and v3 of a rebuilt task cannot be detected afterwards.
    cfg.to_yaml(run_dir / "config.yaml")
    (run_dir / "data_fingerprint").write_text(_data_fp(cfg) + "\n")
    (run_dir / "status").write_text("RUNNING\n")
    t0 = time.time()
    try:
        row = _run(cfg, run_dir, tokenizer, model, quiet)
        row["wallclock_s"] = round(time.time() - t0, 1)
        _write_results(run_dir, row)
        (run_dir / "status").write_text("DONE\n")
        return row
    except Exception:
        (run_dir / "status").write_text("FAILED\n" + traceback.format_exc())
        raise


def _env_fingerprint() -> dict:
    """What produced this number, beyond the config.

    Twenty nodes mixing 3090 / RTX 4000 Ada / RTX 2000 Ada, a venv rebuilt per node from an unpinned
    requirements.txt at whatever date that node first ran a job, and flash-attn tried with sdpa as the
    fallback -- which is not in the run id. Two seeds of the same cell can therefore run with different
    attention kernels and different library versions. For eight-point claims spread over three weeks
    that is the widest free variable in the whole setup, and it costs four lines to close.
    """
    import platform
    import socket
    info = {"host": socket.gethostname(), "python": platform.python_version()}
    try:
        import torch as _t
        info["torch"] = _t.__version__
        info["PARTITION"] = _t.cuda.get_device_name(0) if _t.cuda.is_available() else "cpu"
        info["cuda"] = _t.version.cuda or ""
    except Exception:
        pass
    try:
        import transformers as _tf
        info["transformers"] = _tf.__version__
    except Exception:
        pass
    return info

def _metric_version(task: str) -> str:
    """Which version of the scorer produced this run's numbers. Empty for tasks without one."""
    if task != "format":
        return ""
    try:
        from .data.format import metric_version

        return metric_version()
    except Exception:  # noqa: BLE001
        return ""


def _environment() -> dict:
    """Library versions, node and PARTITION, recorded with every run.

    The widest free variable of the setup: twenty nodes, three weeks, a venv rebuilt per node from an
    unpinned requirements.txt, `attn_implementation` outside the run id — two seeds of one cell could
    run with different attention kernels and different `transformers` versions, for gaps of eight
    points. Closing it costs nothing; leaving it open is unanswerable in review.
    """
    import platform
    import socket

    env = {"host": socket.gethostname(), "python": platform.python_version()}
    try:
        import torch as _t

        env["torch"] = _t.__version__
        env["PARTITION"] = _t.cuda.get_device_name(0) if _t.cuda.is_available() else "cpu"
    except Exception:  # noqa: BLE001
        pass
    for mod in ("transformers", "peft", "datasets"):
        try:
            env[mod] = __import__(mod).__version__
        except Exception:  # noqa: BLE001
            pass
    return env


def _data_fp(cfg: RunConfig) -> str:
    """Fingerprint of the dataset this run reads, or "" if it cannot be computed.

    Goes into results.csv — `collect()` reads results.csv and nothing else, so a side file would leave
    `check_data_versions` permanently dormant on real data.
    """
    from .data.paths import fingerprint_data

    try:
        return fingerprint_data(cfg.data_dir, cfg.task)
    except Exception:  # noqa: BLE001
        return ""


def _run(cfg: RunConfig, run_dir: Path, tokenizer, model, quiet: bool) -> dict:
    set_seed(cfg.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log = lambda *a: (None if quiet else print("[train]", *a))  # noqa: E731

    # ---- model & tokenizer --------------------------------------------------
    if model is None:
        if cfg.tiny_model_config is not None:
            model = build_tiny_model(cfg.tiny_model_config, device=dev)
        else:
            model = load_model(cfg.model, cfg.dtype, cfg.attn_implementation, device=dev)
    if tokenizer is None:
        tokenizer = load_tokenizer(cfg.model)
    if cfg.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        enable_checkpointing(model)

    # ---- baseline: evaluate the un-adapted model and stop ---------------------
    if cfg.mode == "baseline":
        model.eval()
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = True
        metrics = run_all(model, tokenizer, cfg.task, cfg.data_dir, limit=cfg.eval_limit,
                          save_gens_to=run_dir / "generations_test.jsonl", max_new_tokens=cfg.max_new_tokens,
                          do_forgetting=cfg.eval_forgetting, forgetting_limit=cfg.eval_forgetting_limit, split="test")
        val = run_all(model, tokenizer, cfg.task, cfg.data_dir, limit=min(cfg.eval_limit, 500),
                      do_forgetting=False, split="val")
        metrics.update({f"val_{k}": v for k, v in val.items()})
        with open(run_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        row = {"run_id": cfg.run_id, "model": cfg.model, "task": cfg.task, "mode": "baseline", "rank": 0,
               "band_frac": None, "alpha": None, "lr": 0.0, "seed": cfg.seed, "n_trainable": 0,
               "data_fingerprint": _data_fp(cfg), "metric_version": _metric_version(cfg.task), **_environment(), **_env_fingerprint(), "subspace_salted": bool(cfg.subspace_salted),
               "total_steps": 0, "tokens_seen": 0}
        row.update(metrics)
        return row

    # ---- SVD cache: build it here if this node does not have it (compute nodes have a fresh /tmp)
    if cfg.mode in NEEDS_SVD_CACHE:
        ensure_svd_cache(model, cfg, log)

    # ---- adapters -----------------------------------------------------------
    W0: dict[str, torch.Tensor] = {}
    if cfg.mode == "full":
        for p in model.parameters():
            p.requires_grad_(True)
        if cfg.save_full_delta:
            W0 = {n: lin.weight.detach().float().cpu().clone()
                  for n, lin in iter_target_modules(model, cfg.target_modules)}
        modules = {}
    elif cfg.mode in ("mica_peft", "lora_peft"):  # PROTOCOL_real_mica.md and ADDENDUM_real_mica_2.md : PEFT 0.21.0, same code path
        from .mica_peft import inject_mica_peft
        mica = cfg.mode == "mica_peft"            # MiCA trains A only, as our bottom arm ; PEFT's standard LoRA trains A and B, as our free arm
        exp = expected_trainable(model, cfg.rank, "bottom" if mica else "free", cfg.target_modules)
        inject_mica_peft(model, cfg, exp, init="mica" if mica else True)
        modules = {}
    else:
        modules = inject_lora(model, r=cfg.rank, mode=cfg.mode, alpha=cfg.alpha, cache_dir=cfg.svd_cache,
                              model_id=cfg.model,
                              seed=cfg.seed if cfg.subspace_seed is None else cfg.subspace_seed,
                              target_suffixes=cfg.target_modules, band_frac=cfg.band_frac,
                              per_module=cfg.per_module, sigma_rel_tol=cfg.sigma_rel_tol,
                              subspace_salted=cfg.subspace_salted, learned_basis=cfg.learned_basis)
        exp = expected_trainable(model, cfg.rank, cfg.mode, cfg.target_modules)
        got = count_trainable(model)
        if exp != got:
            raise RuntimeError(f"trainable count mismatch: expected {exp}, got {got}")
    n_trainable = count_trainable(model)
    log(f"run {cfg.run_id}: model={cfg.model} task={cfg.task} mode={cfg.mode} r={cfg.rank} "
        f"lr={cfg.lr} seed={cfg.seed} trainable={n_trainable:,}")

    # ---- data ---------------------------------------------------------------
    ds = build_dataset(cfg.task, cfg.data_dir, tokenizer, cfg.max_len, split="train", seed=cfg.seed)
    if cfg.answer_only:                       # PROTOCOL_answer_only.md : memes blocs, seules les etiquettes changent
        from .answer_only import AnswerOnly
        ds = AnswerOnly(ds, tokenizer)
        log(f"answer_only: {ds.n_tokens:,} supervised tokens/epoch (answers only, {ds.n_answers} answers) instead of {ds.base_n_tokens:,}")
    g = torch.Generator().manual_seed(cfg.seed)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, generator=g, drop_last=False,
                        collate_fn=lambda b: collate(b, tokenizer.pad_token_id), num_workers=0)
    steps_per_epoch = math.ceil(len(loader) / cfg.grad_accum)
    tokens_per_step = ds.n_tokens / max(1, len(ds)) * cfg.batch_size * cfg.grad_accum  # supervised tokens
    if cfg.max_steps:
        total_steps = cfg.max_steps
    elif cfg.target_tokens:
        total_steps = max(1, math.ceil(cfg.target_tokens / tokens_per_step))
    else:
        total_steps = max(1, int(round(cfg.epochs * steps_per_epoch)))
    warmup = int(cfg.warmup_ratio * total_steps)
    save_at = _save_steps(cfg, total_steps)
    log(f"dataset: {len(ds)} examples, {ds.n_tokens:,} supervised tokens/epoch, "
        f"{steps_per_epoch} steps/epoch, ~{tokens_per_step:,.0f} tokens/step, {total_steps} total steps "
        f"(~{total_steps * tokens_per_step / max(1, ds.n_tokens):.2f} epochs)")

    # ---- optimiser ----------------------------------------------------------
    params = [p for p in model.parameters() if p.requires_grad]
    if cfg.mode == "top_lrscale":
        # one group per module at lr * s (PROTOCOL_top_lrscale.md); clipping below stays GLOBAL, over `params`, as for every arm
        from .lora import lora_param_groups
        groups = lora_param_groups(modules, cfg.lr)
        if sum(len(g["params"]) for g in groups) != len(params):
            raise RuntimeError("top_lrscale: the optimiser groups do not cover exactly the trainable parameters")
        opt = torch.optim.AdamW(groups, lr=cfg.lr, betas=(0.9, 0.999), eps=cfg.adam_eps, weight_decay=cfg.weight_decay)
    else:
        opt = _make_optimizer(params, cfg)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: _cosine_with_warmup(s, total_steps, warmup))

    def maybe_save(step: int) -> None:
        if step not in save_at:
            return
        if cfg.mode == "full":
            if cfg.save_full_delta:
                _save_full_delta(model, W0, run_dir / f"fulldelta_{step}.safetensors", cfg.target_modules)
        else:
            save_factors(modules, run_dir / f"factors_{step}.safetensors", {"step": step, "run_id": cfg.run_id})

    # ---- loop ---------------------------------------------------------------
    grad_norm_sum, grad_norm_n = 0.0, 0
    grad_norm_max, grad_norm_clipped = 0.0, 0   # how often clipping fires (PROTOCOL_top_lrscale.md, 26/09)
    model.train()
    log_f = open(run_dir / "log.csv", "w", newline="")
    log_w = csv.writer(log_f)
    log_w.writerow(["step", "loss", "lr", "tokens_seen", "gpu_mem_gb"])
    maybe_save(0)
    step, tokens_seen, micro, window_tokens = 0, 0, 0, 0
    running, running_n = 0.0, 0
    done = False
    while not done:
        for batch in loader:
            batch = {k: v.to(dev) for k, v in batch.items()}
            out = model(**batch)
            n_sup = int((batch["labels"] != -100).sum())
            # GRADIENT ACCUMULATION. `out.loss` is already a mean over the supervised tokens of THIS
            # micro-batch, so `out.loss / grad_accum` gives every micro-batch the same weight however
            # few tokens it supervised -- a mean of means, not a token-weighted mean. On the format
            # task only the answer is supervised and its length varies with the entity count, so with
            # batch_size 2 and grad_accum 16 the sparse micro-batches are over-weighted. Same bug
            # HuggingFace fixed in its Trainer in late 2024.
            #
            # It is OPT-IN, and that is deliberate: unlike the salt, turning it on changes the
            # optimiser's objective, so runs produced with it are not comparable to the 239 already in
            # the database. `token_weighted_loss` is therefore in the run id -- the two populations can
            # never be silently averaged -- and stays False until a campaign starts fresh. Both forms
            # have the same scale, so the selected learning rates carry over.
            if cfg.token_weighted_loss:
                loss = out.loss * n_sup            # un-normalised; divided by the window below
                window_tokens += n_sup
            else:
                loss = out.loss / cfg.grad_accum
            loss.backward()
            tokens_seen += n_sup
            running += float(out.loss.detach())
            running_n += 1
            micro += 1
            if micro % cfg.grad_accum != 0:
                continue
            if cfg.token_weighted_loss and window_tokens:
                # normalise the accumulated gradient by the supervised tokens of the WHOLE window;
                # must happen before clipping, or the clipped norm is off by that factor
                inv = 1.0 / window_tokens
                for prm in params:
                    if prm.grad is not None:
                        prm.grad.mul_(inv)
            window_tokens = 0
            # norm BEFORE clipping: with max_grad_norm=1.0, an arm whose gradients are larger is
            # clipped harder, which silently lowers its effective learning rate. Logged so that a
            # difference between arms cannot be mistaken for a difference in what they can learn.
            grad_norm_pre = float(torch.nn.utils.clip_grad_norm_(params, cfg.max_grad_norm))
            grad_norm_sum += grad_norm_pre
            grad_norm_n += 1
            grad_norm_max = max(grad_norm_max, grad_norm_pre)
            if cfg.max_grad_norm and grad_norm_pre > cfg.max_grad_norm:
                grad_norm_clipped += 1
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % cfg.log_every == 0 or step == total_steps:
                mem = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
                log_w.writerow([step, round(running / running_n, 5), sched.get_last_lr()[0], tokens_seen, round(mem, 2)])
                log_f.flush()
                log(f"step {step}/{total_steps} loss={running / running_n:.4f} lr={sched.get_last_lr()[0]:.2e}")
                running, running_n = 0.0, 0
            maybe_save(step)
            if step >= total_steps:
                done = True
                break
    log_f.close()

    # ---- update magnitude (logged always; optionally matched before evaluation) --------
    mag: dict = {"grad_norm_pre_clip_mean": grad_norm_sum / max(1, grad_norm_n),
                 "grad_norm_pre_clip_max": grad_norm_max, "clip_frac": grad_norm_clipped / max(1, grad_norm_n)}
    if modules:
        norms = update_norms(modules)
        mag["update_norm_mean"] = sum(norms.values()) / len(norms)
        mag["update_norm_max"] = max(norms.values())
        with open(run_dir / "update_norms.json", "w") as f:
            json.dump(norms, f, indent=1)
        if cfg.update_norm_target is not None:
            mag["update_rescale_factor"] = rescale_update_(modules, cfg.update_norm_target)
            save_factors(modules, run_dir / "factors_rescaled.safetensors",
                         {"step": total_steps, "run_id": cfg.run_id, "rescaled_to": cfg.update_norm_target})

    # ---- eval ---------------------------------------------------------------
    model.eval()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = True
    metrics = run_all(model, tokenizer, cfg.task, cfg.data_dir, limit=cfg.eval_limit,
                      save_gens_to=run_dir / "generations_test.jsonl", max_new_tokens=cfg.max_new_tokens,
                      do_forgetting=cfg.eval_forgetting, forgetting_limit=cfg.eval_forgetting_limit, split="test")
    val = run_all(model, tokenizer, cfg.task, cfg.data_dir, limit=min(cfg.eval_limit, 500),
                  do_forgetting=False, split="val")
    metrics.update({f"val_{k}": v for k, v in val.items()})
    metrics.update(mag)
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    row = {"run_id": cfg.run_id, "model": cfg.model, "task": cfg.task, "mode": cfg.mode, "rank": cfg.rank,
           "band_frac": cfg.band_frac, "per_module": json.dumps(cfg.per_module) if cfg.per_module else None,
           "alpha": cfg.alpha, "lr": cfg.lr, "seed": cfg.seed, "n_trainable": n_trainable,
           # in results.csv: collect() reads results.csv only, so writing the fingerprint to a side
           # file left check_data_versions permanently dormant on real data
           "data_fingerprint": _data_fp(cfg), "metric_version": _metric_version(cfg.task), **_environment(), **_env_fingerprint(),
           # must reach results.csv: the run id separates the two random arms, but the aggregation
           # groups on KEY, and a flag absent from the CSV cannot be added to KEY
           "subspace_salted": bool(cfg.subspace_salted),
           "total_steps": total_steps, "tokens_seen": tokens_seen}
    row.update(metrics)
    return row


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        print(__doc__)
        sys.exit(1)
    cfg = load_config(argv[0], overrides=argv[1:])
    row = run(cfg)
    print(json.dumps(row, indent=2, default=str))


if __name__ == "__main__":
    main()
