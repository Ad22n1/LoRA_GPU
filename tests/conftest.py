import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from lorasub.config import RunConfig  # noqa: E402
from lorasub.data.facts import generate_facts  # noqa: E402
from lorasub.data.format import build_example  # noqa: E402
from lorasub.spectral import compute_svd_cache  # noqa: E402
from tiny import TINY_CFG, tiny_model  # noqa: E402

MODEL_ID = "tiny/model"


def _format_rows(n, seed=0):
    import random

    rng = random.Random(seed)
    people = [["Anna", "Lee"], ["Tom", "Baker"], ["Li", "Wei"]]
    orgs = [["Reuters"], ["Acme", "Corp"], ["UN"]]
    locs = [["Paris"], ["Tokyo"], ["Oslo"]]
    rows = []
    for _ in range(n):
        p, o, l = rng.choice(people), rng.choice(orgs), rng.choice(locs)
        toks = p + ["joined"] + o + ["in"] + l + ["."]
        tags = [1] + [2] * (len(p) - 1) + [0] + [3] + [4] * (len(o) - 1) + [0] + [5] + [6] * (len(l) - 1) + [0]
        rows.append(build_example(toks, tags, variant="transform"))
    return rows


@pytest.fixture(scope="session")
def workspace(tmp_path_factory):
    """Tiny datasets + SVD cache of the tiny model, shared by the end-to-end tests."""
    root = tmp_path_factory.mktemp("ws")
    data = root / "data"
    generate_facts(data / "facts", n_entities=40, seed=0, n_test_mcq=24, n_test_completion=24, n_val_mcq=12)
    fmt = data / "format"
    fmt.mkdir(parents=True)
    for split, n in (("train", 40), ("val", 8), ("test", 8)):
        with open(fmt / f"{split}.jsonl", "w") as f:
            for r in _format_rows(n, seed={"train": 0, "val": 1, "test": 2}[split]):
                f.write(json.dumps(r) + "\n")
    compute_svd_cache(tiny_model(0), root / "svd", MODEL_ID, device="cpu", verbose=False)
    return root


def _cfg(root, **kw) -> RunConfig:
    base = dict(model=MODEL_ID, rank=4, lr=5e-3, seed=0, max_steps=6, batch_size=4, grad_accum=1, max_len=192,
                svd_cache=str(root / "svd"), data_dir=str(root / "data"), out_dir=str(root / "runs"),
                save_factor_steps=[0, 3, "end"], gradient_checkpointing=False, eval_limit=8,
                eval_forgetting=False, tiny_model_config=TINY_CFG, dtype="float32", log_every=2)
    base.update(kw)
    return RunConfig(**base)
