"""Evaluation.

* ``mcq_accuracy``           : 4-way MCQ scored by mean per-token log-likelihood of each option
                               given "Question: ...\\nAnswer:" — no generation, no format to learn.
* ``completion_exact_match`` : greedy completion of a held-out-template prefix, normalised exact match.
* ``format_metrics``         : greedy generation on the alien prompt, strict checker.
* ``forgetting``             : HellaSwag (acc_norm), TruthfulQA-MC1, WikiText perplexity via lm_eval,
                               on a *merged copy* of the model.  Skipped gracefully if lm_eval is absent.
Everything is deterministic (greedy) and reports the number of examples actually evaluated.
"""
from __future__ import annotations

import re
from pathlib import Path

import torch
import torch.nn.functional as F

from .data.facts import read_jsonl
from .data.format import check_alien
from .data.paths import resolve
from .lora import lora_modules, merge_all_, unmerge_all_

MCQ_PROMPT = "Question: {q}\nAnswer:"


def _device(model) -> torch.device:
    return next(model.parameters()).device


def _normalize(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"^(the|a|an)\s+", "", s)
    s = re.sub(r"[^\w\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


# --------------------------------------------------------------------------- #
# MCQ
# --------------------------------------------------------------------------- #
@torch.no_grad()
def option_logliks(model, tok, prompt: str, options: list[str]) -> list[float]:
    """Mean per-token log p(option | prompt) for each option (one batched forward)."""
    dev = _device(model)
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    if tok.bos_token_id is not None:
        p_ids = [tok.bos_token_id] + p_ids
    seqs, lens = [], []
    for o in options:
        o_ids = tok(" " + o, add_special_tokens=False)["input_ids"]
        seqs.append(p_ids + o_ids)
        lens.append(len(o_ids))
    L = max(len(s) for s in seqs)
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    ids = torch.full((len(seqs), L), pad, dtype=torch.long)
    att = torch.zeros((len(seqs), L), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, : len(s)] = torch.tensor(s)
        att[i, : len(s)] = 1
    ids, att = ids.to(dev), att.to(dev)
    logits = model(input_ids=ids, attention_mask=att).logits.float()
    logp = F.log_softmax(logits[:, :-1], dim=-1)
    tgt = ids[:, 1:]
    tok_lp = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)  # (n, L-1)
    out = []
    for i, s in enumerate(seqs):
        start = len(p_ids) - 1  # position predicting the first option token
        end = len(s) - 1
        out.append(float(tok_lp[i, start:end].sum()) / max(1, lens[i]))
    return out


def _mcq_prompt_for(path: str | Path) -> str:
    """Use the language pack's prompt if the dataset directory has a meta.json (French data)."""
    meta = Path(path).parent / "meta.json"
    if meta.exists():
        import json

        with open(meta, encoding="utf-8") as f:
            return json.load(f).get("mcq_prompt", MCQ_PROMPT)
    return MCQ_PROMPT


@torch.no_grad()
def mcq_accuracy(model, tok, path: str | Path, limit: int | None = None) -> dict:
    items = read_jsonl(path)[: limit or None]
    prompt = _mcq_prompt_for(path)
    correct = 0
    for it in items:
        ll = option_logliks(model, tok, prompt.format(q=it["question"]), it["options"])
        correct += int(max(range(len(ll)), key=lambda i: ll[i]) == it["answer_idx"])
    return {"mcq_acc": correct / max(1, len(items)), "mcq_n": len(items)}


# --------------------------------------------------------------------------- #
# Generation helpers
# --------------------------------------------------------------------------- #
@torch.no_grad()
def generate_batch(model, tok, prompts: list[str], max_new_tokens: int, batch_size: int = 16) -> list[str]:
    """Greedy generation; returns only the newly generated text for each prompt."""
    dev = _device(model)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    bos = [tok.bos_token_id] if tok.bos_token_id is not None else []
    outs: list[str] = []
    for i in range(0, len(prompts), batch_size):
        chunk = prompts[i : i + batch_size]
        seqs = [bos + tok(p, add_special_tokens=False)["input_ids"] for p in chunk]
        L = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), L), pad, dtype=torch.long)
        att = torch.zeros((len(seqs), L), dtype=torch.long)
        for j, s in enumerate(seqs):  # left padding so that generation starts right after the prompt
            ids[j, L - len(s):] = torch.tensor(s)
            att[j, L - len(s):] = 1
        gen = model.generate(
            input_ids=ids.to(dev), attention_mask=att.to(dev), max_new_tokens=max_new_tokens,
            do_sample=False, num_beams=1, pad_token_id=pad, eos_token_id=tok.eos_token_id,
        )
        outs.extend(tok.batch_decode(gen[:, L:], skip_special_tokens=True))
    return outs


@torch.no_grad()
def completion_exact_match(model, tok, path: str | Path, limit: int | None = None, max_new_tokens: int = 12) -> dict:
    items = read_jsonl(path)[: limit or None]
    gens = generate_batch(model, tok, [it["prefix"] for it in items], max_new_tokens)
    hits = 0
    for it, g in zip(items, gens):
        first = re.split(r"[.\n]", g, maxsplit=1)[0]
        hits += int(_normalize(first).startswith(_normalize(it["target"])))
    return {"completion_em": hits / max(1, len(items)), "completion_n": len(items)}


@torch.no_grad()
def format_metrics(model, tok, path: str | Path, limit: int | None = None, max_new_tokens: int = 128,
                   save_to: "Path | None" = None) -> dict:
    items = read_jsonl(path)[: limit or None]
    gens = generate_batch(model, tok, [it["prompt"] for it in items], max_new_tokens)
    # An arm that never emits the closing tag within the budget fails for a reason that has nothing to
    # do with the format: it loops. Reported separately so that such failures are not read as
    # "does not know the format".
    truncated = sum(1 for g in gens if "<</alien>>" not in g)
    parsed = keys_ok = n_ok = sig_ok = order_ok = exact = 0
    micro: list[float] = []
    macro: list[float] = []
    by_bucket: dict[str, list[float]] = {}
    for it, g in zip(items, gens):
        r = check_alien(g, it["gold"])
        parsed += int(r["parsed"])
        keys_ok += int(r["keys_ok"])
        n_ok += int(r["n_ok"])
        sig_ok += int(r.get("sig_ok", True))
        order_ok += int(r.get("order_ok", True))
        exact += int(r.get("exact", False))
        micro.append(r.get("f1_micro", r["f1_mean"]))
        macro.append(r.get("f1_macro", r["f1_mean"]))
        k = it.get("n_entities")
        if k is not None:
            by_bucket.setdefault("1-2" if k <= 2 else ("3-4" if k <= 4 else "5+"), []).append(
                float(r["parsed"]))
    n = max(1, len(items))
    out = {"format_parsed": parsed / n,          # every rule obeyed, ordering included
           "format_exact": exact / n,             # every rule obeyed AND every entity right
           "format_truncated": truncated / n,      # no closing tag within max_new_tokens
           "format_keys_ok": keys_ok / n, "format_count_ok": n_ok / n,
           # entity-level micro F1: no free points for fields that are empty on both sides
           "format_f1": sum(micro) / n,
           # field-level macro F1, inflated (floor 0.75 on a one-entity sentence); kept for continuity
           "format_f1_macro": sum(macro) / n,
           "format_n": len(items)}
    if "qzx_sig" in (items[0]["gold"] if items else {}):  # hard variant: which sub-rule fails
        out["format_sig_ok"] = sig_ok / n
        out["format_order_ok"] = order_ok / n
    for k, v in sorted(by_bucket.items()):  # difficulty grows with the number of entities
        out[f"format_parsed_{k}ent"] = sum(v) / len(v)
    # proportion of outputs that obey the rules while listing nothing: the degenerate strategy
    if save_to is not None:
        # Keep the generations. Three reasons, all learned the hard way. (i) The format checker was
        # redefined three times; each redefinition forced a full PARTITION re-run instead of re-scoring a
        # file — on a cluster losing 144 jobs a day to OOM, that is the most expensive decision in the
        # repository. (ii) Comparing arms on a 2.3-point binomial error while PAIRING ONLY AT THE SEED
        # LEVEL wastes the pairing that matters: with the per-example outputs, a McNemar test on the
        # same items settles `random - top` today, at zero PARTITION cost, where five seeds give at best
        # p = 1/32. (iii) No error analysis is possible without them.
        import json as _json

        Path(save_to).parent.mkdir(parents=True, exist_ok=True)
        with open(save_to, "w") as f:
            for it, g in zip(items, gens):
                r = check_alien(g, it["gold"])
                f.write(_json.dumps({"id": it.get("id"), "gen": g, "gold": it["gold"],
                                     "parsed": r["parsed"], "exact": r["exact"],
                                     "f1_micro": r["f1_micro"],
                                     "n_gold_entities": r["n_gold_entities"]}) + "\n")
    out["format_empty"] = sum(1 for it, g in zip(items, gens)
                              if (r := check_alien(g, it["gold"]))["parsed"] and r["n_gold_entities"]
                              and r["f1_micro"] == 0.0) / n
    return out


# --------------------------------------------------------------------------- #
# Forgetting (lm_eval on a merged copy)
# --------------------------------------------------------------------------- #
def forgetting(model, tok, limit: int = 1000, tasks=("hellaswag", "truthfulqa_mc1", "wikitext")) -> dict:
    """lm_eval on the adapted model.  The LoRA update is merged *in place* for the evaluation and
    unmerged afterwards: no copy of the model (a deepcopy of a 7B in bf16 would add 15 GB and OOM
    on 16-20 GB cards).  Never raises: an lm_eval API change is reported, not fatal."""
    try:
        import lm_eval
        from lm_eval.models.huggingface import HFLM
    except Exception:
        return {"forgetting_skipped": 1}
    mods = lora_modules(model)
    merged_here = False
    try:
        if mods:
            merge_all_(model)
            merged_here = True
        model.eval()
        # batch_size=8 asked for 22 GB on a 1B model and OOMed on a 24 GB card shared with other
        # users; wikitext in particular builds very long sequences. Start small and halve on OOM.
        res = None
        last_err: Exception | None = None
        for bs in (4, 2, 1):
            try:
                hf = HFLM(pretrained=model, tokenizer=tok, batch_size=bs, max_length=1024)
                res = lm_eval.simple_evaluate(model=hf, tasks=list(tasks), limit=limit, log_samples=False)
                break
            except torch.cuda.OutOfMemoryError as e:  # noqa: PERF203
                last_err = e
                torch.cuda.empty_cache()
        if res is None:
            raise last_err if last_err else RuntimeError("lm_eval produced no result")
        r = res["results"]
        out = {}
        if "hellaswag" in r:
            out["hellaswag_acc_norm"] = float(r["hellaswag"].get("acc_norm,none", float("nan")))
        if "truthfulqa_mc1" in r:
            out["truthfulqa_mc1"] = float(r["truthfulqa_mc1"].get("acc,none", float("nan")))
        if "wikitext" in r:
            out["wikitext_ppl"] = float(r["wikitext"].get("word_perplexity,none", float("nan")))
        return out
    except Exception as e:  # the main metrics must survive an lm_eval API change
        return {"forgetting_skipped": 1, "forgetting_error": f"{type(e).__name__}: {str(e)[:200]}"}
    finally:
        if merged_here:
            unmerge_all_(model)


# --------------------------------------------------------------------------- #
def run_all(model, tok, task: str, data_dir: str | Path, limit: int = 1000, save_gens_to=None,
            max_new_tokens: int = 128,
            do_forgetting: bool = True, forgetting_limit: int = 1000, split: str = "test") -> dict:
    """All metrics relevant to ``task``.  ``split`` in {"val", "test"}."""
    model.eval()
    d = Path(data_dir)
    out: dict = {}
    if task == "facts":
        r = resolve(d, "facts", "mcq_val" if split == "val" else "mcq_test")
        out.update(mcq_accuracy(model, tok, r.path, limit))
        if split == "test":
            out.update(completion_exact_match(model, tok, resolve(d, "facts", "completion").path, limit))
    elif task == "commonsense":
        # the public benchmark: the same log-likelihood MCQ as the facts task, on the benchmark's
        # own items. No generation, so no format to learn, and the arms are compared on accuracy.
        r = resolve(d, "commonsense", "mcq_val" if split == "val" else "mcq_test")
        out.update(mcq_accuracy(model, tok, r.path, limit))
    elif task == "format":
        r = resolve(d, "format", split)
        out.update(format_metrics(model, tok, r.path, limit, max_new_tokens=max_new_tokens,
                                  save_to=save_gens_to))
    else:
        raise ValueError(task)
    if split == "val" and r.val_is_test:
        out["val_is_test"] = 1  # no validation file: LR selection on these numbers would be biased
    if do_forgetting and split == "test":
        out.update(forgetting(model, tok, limit=forgetting_limit))
    return out


def primary_metric(task: str) -> str:
    """Metric used for LR selection and for the crossed table.

    Pre-registered: ``mcq_acc`` for facts, ``format_parsed`` for format. Two caveats, both discovered
    after the fact and to be reported as deviations rather than silently applied:

    * facts — no constrained arm produces a single correct completion at r <= 4 while the MCQ moves by
      2-3 points, so ``completion_em`` is the informative metric there. Report BOTH; the switch is a
      deviation from the pre-registered choice.
    * format — ``format_parsed`` measures obedience to the rules, not content: an internally coherent
      empty answer satisfies them. ``format_exact`` adds content correctness. Report both.
    """
    return {"facts": "mcq_acc", "format": "format_parsed"}[task]


SECONDARY_METRICS = {"facts": ("completion_em",), "format": ("format_exact", "format_f1")}
