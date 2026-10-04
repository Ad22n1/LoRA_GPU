"""Torch datasets for the two tasks.

* facts  : sentences joined with newlines, tokenised once, packed into fixed blocks of
           ``max_len`` tokens; labels = input_ids (standard causal LM, like MiCA's packing=True).
* format : one (prompt, target) pair per example; labels are -100 on the prompt so the loss
           is computed on the answer only; target ends with EOS.
Both expose ``n_tokens`` (supervised tokens per epoch) so the two arms can be matched on
the number of tokens seen.
"""
from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset

from .facts import read_jsonl
from .paths import resolve

IGNORE = -100


class FactsPackedDataset(Dataset):
    def __init__(self, path: str | Path | None, tokenizer, max_len: int, seed: int = 0,
                 rows: list[dict] | None = None):
        rows = rows if rows is not None else read_jsonl(path)
        g = torch.Generator().manual_seed(seed)
        order = torch.randperm(len(rows), generator=g).tolist()
        text = "\n".join(rows[i]["text"] for i in order)
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if tokenizer.bos_token_id is not None:
            ids = [tokenizer.bos_token_id] + ids
        n_blocks = len(ids) // max_len
        if n_blocks == 0:
            raise ValueError(f"facts corpus too small for max_len={max_len}: {len(ids)} tokens")
        self.blocks = torch.tensor(ids[: n_blocks * max_len], dtype=torch.long).view(n_blocks, max_len)
        self.n_tokens = int(self.blocks.numel())

    def __len__(self) -> int:
        return self.blocks.shape[0]

    def __getitem__(self, i: int) -> dict:
        ids = self.blocks[i]
        return {"input_ids": ids, "labels": ids.clone(), "attention_mask": torch.ones_like(ids)}


class FormatDataset(Dataset):
    """Prompt/target pairs.  ``mask_prompt=True`` (default) computes the loss on the answer only, which
    is what training does.  ``mask_prompt=False`` supervises the whole sequence: this is a *control*
    condition, not a training mode — it makes the format gradient comparable to the facts gradient,
    which is supervised on every token, and so tells whether an observed difference between the two
    tasks is a property of the tasks or an artefact of the masking."""

    def __init__(self, path: str | Path | None, tokenizer, max_len: int, rows: list[dict] | None = None,
                 mask_prompt: bool = True):
        self.rows = rows if rows is not None else read_jsonl(path)
        self.tok = tokenizer
        self.max_len = max_len
        self.items: list[dict] = []
        n_sup = 0
        self.n_skipped = 0
        bos = [tokenizer.bos_token_id] if tokenizer.bos_token_id is not None else []
        for r in self.rows:
            p = tokenizer(r["prompt"], add_special_tokens=False)["input_ids"]
            t = tokenizer(" " + r["target"], add_special_tokens=False)["input_ids"] + [tokenizer.eos_token_id]
            room = max_len - len(bos) - len(t)
            if room < 8:  # target does not fit: never truncate the answer, drop the example
                self.n_skipped += 1
                continue
            if len(p) > room:  # keep the end of the prompt (the text and "Answer:")
                p = p[-room:]
            p = bos + p
            ids = p + t
            labels = ([IGNORE] * len(p) + t) if mask_prompt else list(ids)
            self.items.append({"input_ids": torch.tensor(ids), "labels": torch.tensor(labels),
                               "attention_mask": torch.ones(len(ids), dtype=torch.long)})
            n_sup += len(t) if mask_prompt else len(ids)
        if not self.items:
            raise ValueError(f"no format example fits in max_len={max_len}")
        self.n_tokens = n_sup

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> dict:
        return self.items[i]


def collate(batch: list[dict], pad_id: int) -> dict:
    L = max(x["input_ids"].numel() for x in batch)
    n = len(batch)
    ids = torch.full((n, L), pad_id, dtype=torch.long)
    lab = torch.full((n, L), IGNORE, dtype=torch.long)
    att = torch.zeros((n, L), dtype=torch.long)
    for i, x in enumerate(batch):
        k = x["input_ids"].numel()
        ids[i, :k] = x["input_ids"]
        lab[i, :k] = x["labels"]
        att[i, :k] = 1
    return {"input_ids": ids, "labels": lab, "attention_mask": att}


def build_dataset(task: str, data_dir: str | Path, tokenizer, max_len: int, split: str = "train", seed: int = 0,
                  rows: list[dict] | None = None, mask_prompt: bool = True):
    """Both file-name conventions are accepted (see data/paths.py); ``rows`` bypasses the file."""
    if task == "facts":
        path = None if rows is not None else resolve(data_dir, "facts", split).path
        return FactsPackedDataset(path, tokenizer, max_len, seed=seed, rows=rows)
    if task == "commonsense":
        # the rows carry "text" like the facts corpus, so the packed dataset is the same object;
        # what differs is only where the text comes from.
        path = None if rows is not None else resolve(data_dir, "commonsense", split).path
        return FactsPackedDataset(path, tokenizer, max_len, seed=seed, rows=rows)
    if task == "format":
        path = None if rows is not None else resolve(data_dir, "format", split).path
        return FormatDataset(path, tokenizer, max_len, rows=rows, mask_prompt=mask_prompt)
    raise ValueError(f"unknown task {task!r}")


def stratified_rows(rows: list[dict], n: int, key: str = "attr", seed: int = 0) -> list[dict]:
    """n rows covering the values of ``key`` as evenly as possible (falls back to a random subset
    when the key is absent, e.g. on data produced by another generator)."""
    import random

    rng = random.Random(seed)
    if not rows or key not in rows[0]:
        out = rows[:]
        rng.shuffle(out)
        return out[:n]
    by: dict[str, list[dict]] = {}
    for r in rows:
        by.setdefault(str(r[key]), []).append(r)
    for grp in by.values():
        rng.shuffle(grp)
    out: list[dict] = []
    keys = sorted(by)
    i = 0
    while len(out) < n and any(by.values()):
        k = keys[i % len(keys)]
        if by[k]:
            out.append(by[k].pop())
        i += 1
    rng.shuffle(out)
    return out


def skipped_report(ds) -> str:
    """One line on the examples dropped because their target did not fit in max_len.

    Nobody was reading ``n_skipped``. With max_len = 384 and --min_entities 3, the rejection is
    CORRELATED WITH LENGTH: the sentences with the most entities are the ones dropped, which is
    exactly the hard end of the task. And the 7B config uses a different max_len, so a "confirmation"
    there would not be on the same data.
    """
    n = getattr(ds, "n_skipped", 0)
    tot = len(ds) + n
    return f"{n}/{tot} examples dropped ({n / max(1, tot):.1%}) — target longer than max_len"
