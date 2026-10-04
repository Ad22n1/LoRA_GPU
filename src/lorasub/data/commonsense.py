"""Dataset C — a public commonsense benchmark, in the schema the rest of the pipeline already reads.

Why this exists: every result in the paper is measured on tasks we built ourselves, and the methods
we compare (PiCa, MiCA, PiSSA, MiLoRA) report on public benchmarks. This module converts one of
those benchmarks into `commonsense/train.jsonl`, `commonsense/mcq_val.jsonl` and
`commonsense/mcq_test.jsonl`, so that the common-rate inversion can be measured where those methods
measure, with no change to the training loop or to the evaluator: the MCQ rows carry the same
fields as the knowledge task (`question`, `options`, `answer_idx`, the name eval.mcq_accuracy reads), scored by option log-likelihood.

Two benchmarks are supported, both multiple choice with a single correct option:

    arc_easy   allenai/ai2_arc, ARC-Easy      4 options, a science question
    piqa       ybisk/piqa                     2 options, a physical-goal question

Download and conversion happen once, on a node with network access:

    python -m lorasub.data.commonsense --benchmark arc_easy --out $SHARED/data

The split we train on is the benchmark's own training split, rendered as a declarative
question-and-answer line, and the prompt at evaluation time is the same line with the answer
removed, so that no format has to be learned: exactly the arrangement of the knowledge task.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

PROMPT = "Question: {q}\nAnswer:"

# Only benchmarks served as parquet: a repository that ships a loading script (piqa does) is
# refused by datasets >= 3 with "Dataset scripts are no longer supported".
SPECS = {
    "arc_easy": dict(path="allenai/ai2_arc", name="ARC-Easy",
                     splits=("train", "validation", "test")),
    "arc_challenge": dict(path="allenai/ai2_arc", name="ARC-Challenge",
                          splits=("train", "validation", "test")),
    "openbookqa": dict(path="allenai/openbookqa", name="main",
                       splits=("train", "validation", "test")),
}


def _rows_arc(split) -> list[dict]:
    """ARC and OpenBookQA share a schema up to the name of the question field."""
    out = []
    for it in split:
        q = it.get("question", it.get("question_stem", ""))
        ch = it["choices"]
        texts, labels = list(ch["text"]), list(ch["label"])
        if it["answerKey"] not in labels:
            continue                                   # a handful of items carry a stray key
        out.append({"question": q.strip(),
                    "options": [t.strip() for t in texts],
                    "answer_idx": labels.index(it["answerKey"])})
    return out


def _rows_piqa(split) -> list[dict]:
    return [{"question": it["goal"].strip(),
             "options": [it["sol1"].strip(), it["sol2"].strip()],
             "answer_idx": int(it["label"])}
            for it in split if int(it["label"]) in (0, 1)]


def build(benchmark: str, out_dir: str | Path, seed: int = 0, n_val: int = 400, n_test: int = 1000,
          n_train: int | None = None) -> dict:
    """Write the three files and return the counts. Requires `datasets` and network access."""
    from datasets import load_dataset

    spec = SPECS[benchmark]
    ds = load_dataset(spec["path"], spec["name"]) if spec["name"] else load_dataset(spec["path"])
    conv = _rows_piqa if benchmark == "piqa" else _rows_arc
    tr, va, te = (conv(ds[s]) for s in spec["splits"])
    if spec["splits"][1] == spec["splits"][2]:
        # piqa hides its test labels: the validation split is cut in two, disjointly, so that the
        # rate is never selected on the items the score is reported on.
        rng = random.Random(seed)
        pool = va[:]
        rng.shuffle(pool)
        va, te = pool[:n_val], pool[n_val:n_val + n_test]
    else:
        rng = random.Random(seed)
        rng.shuffle(va), rng.shuffle(te)
        va, te = va[:n_val], te[:n_test]
    if n_train:
        rng = random.Random(seed + 1)
        rng.shuffle(tr)
        tr = tr[:n_train]

    # <out>/<benchmark>/commonsense/, so that `data_dir` alone selects the benchmark and the task
    # name stays "commonsense". Two benchmarks can never be averaged silently: `data_fingerprint`
    # enters the run id and the aggregator refuses to pool cells whose fingerprints differ.
    d = Path(out_dir) / benchmark / "commonsense"
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "train.jsonl", "w") as f:
        for r in tr:
            text = PROMPT.format(q=r["question"]) + " " + r["options"][r["answer_idx"]]
            f.write(json.dumps({"text": text, "question": r["question"],
                                "answer_text": r["options"][r["answer_idx"]]}) + "\n")
    for name, rows in (("mcq_val.jsonl", va), ("mcq_test.jsonl", te)):
        with open(d / name, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
    counts = {"train": len(tr), "val": len(va), "test": len(te),
              "options": len(tr[0]["options"]) if tr else 0, "benchmark": benchmark}
    (d / "provenance.json").write_text(json.dumps(
        {**counts, "source": spec["path"], "config": spec["name"], "seed": seed,
         "val_test_disjoint": True}, indent=2))
    return counts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--benchmark", choices=sorted(SPECS), required=True)
    ap.add_argument("--out", required=True, help="the shared data directory, e.g. $SHARED/data")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_train", type=int, default=None, help="cap the training split")
    a = ap.parse_args()
    c = build(a.benchmark, a.out, seed=a.seed, n_train=a.n_train)
    print("  " + json.dumps(c))
    print(f"  written -> {Path(a.out) / a.benchmark / 'commonsense'}")
    print(f"  use it with:  data_dir: {Path(a.out) / a.benchmark}")


if __name__ == "__main__":
    main()
