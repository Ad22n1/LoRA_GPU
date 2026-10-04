"""Resolution of dataset file names across the two conventions in use.

This repo                      | a co-author's branch (feat-2_update_on_gpu)
-------------------------------+---------------------------------------
facts/train.jsonl              | facts/train.jsonl
facts/mcq_val.jsonl            | (absent)  -> falls back to mcq.jsonl / mcq_all.jsonl, flagged
facts/mcq_test.jsonl           | facts/mcq.jsonl  (mcq_all.jsonl = all 6000)
facts/completion.jsonl         | facts/completion.jsonl
format/train.jsonl             | format/format_train.jsonl
format/val.jsonl               | (absent)  -> falls back to test, flagged
format/test.jsonl              | format/format_test.jsonl

Field names of the format rows are identical (prompt, target, gold); a co-author's rows also carry
``sentence`` instead of ``text``, which nothing here reads.  When a validation split is missing
the resolver returns the test file and sets ``val_is_test=True`` so that LR selection on such data
is reported as biased rather than silently accepted.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

_CANDIDATES: dict[tuple[str, str], list[str]] = {
    ("facts", "train"): ["facts/train.jsonl"],
    ("facts", "mcq_val"): ["facts/mcq_val.jsonl"],
    ("facts", "mcq_test"): ["facts/mcq_test.jsonl", "facts/mcq.jsonl", "facts/mcq_all.jsonl"],
    ("facts", "completion"): ["facts/completion.jsonl"],
    # the public commonsense benchmark (data/commonsense.py). Same schema as the facts MCQ, so the
    # evaluator and the resolver need nothing new; only the file names are declared here.
    ("commonsense", "train"): ["commonsense/train.jsonl"],
    ("commonsense", "mcq_val"): ["commonsense/mcq_val.jsonl"],
    ("commonsense", "mcq_test"): ["commonsense/mcq_test.jsonl"],
    ("format", "train"): ["format/train.jsonl", "format/format_train.jsonl"],
    ("format", "val"): ["format/val.jsonl", "format/format_val.jsonl"],
    ("format", "test"): ["format/test.jsonl", "format/format_test.jsonl"],
}


@dataclass
class Resolved:
    path: Path
    val_is_test: bool = False


def resolve(data_dir: str | Path, task: str, split: str) -> Resolved:
    """Return the file for (task, split), trying both conventions; validation falls back to test."""
    d = Path(data_dir)
    key = (task, split)
    if key not in _CANDIDATES:
        raise KeyError(f"unknown (task, split) {key}")
    for rel in _CANDIDATES[key]:
        if (d / rel).exists():
            return Resolved(d / rel)
    if split in ("mcq_val", "val"):
        test_key = ("facts", "mcq_test") if task == "facts" else (
            ("commonsense", "mcq_test") if task == "commonsense" else ("format", "test"))
        for rel in _CANDIDATES[test_key]:
            if (d / rel).exists():
                return Resolved(d / rel, val_is_test=True)
    tried = ", ".join(_CANDIDATES[key])
    raise FileNotFoundError(f"no file for {task}/{split} under {d} (tried: {tried})")


def describe(data_dir: str | Path) -> dict[str, str]:
    """Which convention each split resolves to; useful in logs and preflight."""
    out = {}
    for (task, split) in _CANDIDATES:
        try:
            r = resolve(data_dir, task, split)
            out[f"{task}/{split}"] = str(r.path.relative_to(Path(data_dir))) + (" (=test!)" if r.val_is_test else "")
        except FileNotFoundError:
            out[f"{task}/{split}"] = "MISSING"
    return out


def fingerprint_data(data_dir, task: str) -> str:
    """Short digest of the dataset files a run will actually read.

    Why it belongs in the run id. The format task was rebuilt three times (copy -> transform ->
    transform with --min_entities 3) and the facts task twice, while ``data_dir`` is deliberately
    excluded from the identifier. A run produced on v1 and left in the same ``out_dir`` is therefore
    skipped as "already done" and averaged with v3 — and nothing in the outputs recorded which version
    produced it, so the mixing could not be detected afterwards.

    NOT based on mtime. An ``scp`` without ``-p``, an ``rsync`` without ``-t``, or a copy to another
    node or to a rented machine changes the date without changing the data: same corpus, different
    fingerprint, a false positive from ``check_data_versions`` — and, once the fingerprint enters the
    identifier, a full re-execution on every new machine. Size, line count and the first and last
    lines are just as cheap on a 27 MB corpus over NFS and stable under copy. The aim is to separate
    dataset versions, not to resist an adversary.
    """
    import hashlib
    from pathlib import Path

    h = hashlib.blake2b(digest_size=6)
    root = Path(data_dir)
    for split in ("train", "val", "test"):
        try:
            p = Path(resolve(root, task, split).path)
        except Exception:  # noqa: BLE001
            continue
        if not p.exists():
            continue
        first = last = b""
        n_lines = 0
        with open(p, "rb") as f:
            for i_line, line in enumerate(f):
                if i_line == 0:
                    first = line.strip()
                last = line.strip()
                n_lines += 1
        h.update(f"{p.name}:{p.stat().st_size}:{n_lines}:".encode() + first + b"|" + last)
    return h.hexdigest()
