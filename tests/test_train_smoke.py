"""End-to-end on CPU with a tiny random model: data -> SVD cache -> train -> eval -> results.csv."""
import csv

import pytest
import torch

from conftest import MODEL_ID, _cfg
from lorasub.grad_probe import layer_groups, probe
from lorasub.spectral import iter_target_modules
from lorasub.train import run
from tiny import tiny_model, tiny_tokenizer


@pytest.mark.parametrize("task,mode", [("facts", "bottom"), ("format", "top"), ("facts", "free"), ("format", "random")])
def test_end_to_end(workspace, task, mode):
    cfg = _cfg(workspace, task=task, mode=mode)
    run(cfg, tokenizer=tiny_tokenizer(), quiet=True)
    rd = cfg.run_dir
    assert (rd / "results.csv").exists() and (rd / "status").read_text().startswith("DONE")
    assert (rd / "factors_0.safetensors").exists() and (rd / "factors_3.safetensors").exists()
    assert (rd / "factors_6.safetensors").exists()
    with open(rd / "results.csv") as f:
        rec = next(csv.DictReader(f))
    assert rec["run_id"] == cfg.run_id and rec["mode"] == mode
    if task == "facts":
        assert "mcq_acc" in rec and "completion_em" in rec and "val_mcq_acc" in rec
    else:
        assert "format_parsed" in rec and "format_f1" in rec
    assert int(rec["total_steps"]) == 6 and int(rec["tokens_seen"]) > 0
    # log has the right columns and the loss is finite
    with open(rd / "log.csv") as f:
        logrows = list(csv.DictReader(f))
    assert logrows and all(float(r["loss"]) == float(r["loss"]) for r in logrows)
    # idempotence
    again = run(cfg, tokenizer=tiny_tokenizer(), quiet=True)
    assert again.get("skipped") is True


def test_full_ft_mode(workspace):
    cfg = _cfg(workspace, task="facts", mode="full", lr=1e-3, save_full_delta=True)
    row = run(cfg, tokenizer=tiny_tokenizer(), quiet=True)
    assert (cfg.run_dir / "fulldelta_0.safetensors").exists()
    assert row["n_trainable"] > 1000


def test_run_id_ignores_paths(workspace):
    a = _cfg(workspace, task="facts", mode="top")
    b = _cfg(workspace, task="facts", mode="top", out_dir=str(workspace / "elsewhere"))
    c = _cfg(workspace, task="facts", mode="top", lr=1e-4)
    assert a.run_id == b.run_id and a.run_id != c.run_id


def test_grad_probe_runs(workspace):
    model = tiny_model(0)
    tok = tiny_tokenizer()
    names = [n for n, _ in iter_target_modules(model)]
    assert layer_groups(names, 1) and len(layer_groups(names, 1)) == 2
    df = probe(model, tok, MODEL_ID, str(workspace / "data"), str(workspace / "svd"), n_examples=8,
               batch_size=4, max_len=192, group_layers=1, device=torch.device("cpu"))
    assert set(df.dataset) == {"facts", "format"}
    dec = df[df.band.str.startswith("frac")]
    # decile energies on one side sum to ~1 minus what is outside span(U)
    for (ds, layer, mt, side), g in dec.groupby(["dataset", "layer", "module_type", "side"]):
        outside = float(df[(df.dataset == ds) & (df.layer == layer) & (df.module_type == mt)
                           & (df.side == side) & (df.band == "_outside")].energy.iloc[0])
        assert abs(g.energy.sum() + outside - 1.0) < 1e-3


def test_band_mode_and_target_tokens(workspace):
    cfg = _cfg(workspace, task="facts", mode="band", band_frac=0.3, max_steps=None, target_tokens=1500)
    assert cfg.band_frac == 0.3
    row = run(cfg, tokenizer=tiny_tokenizer(), quiet=True)
    assert row["band_frac"] == 0.3 and row["total_steps"] >= 1
    # steps were chosen from the token budget: ceil(1500 / (batch 4 x max_len 192 packed tokens per step)) = 2
    import math
    assert row["total_steps"] == math.ceil(1500 / (cfg.batch_size * cfg.grad_accum * cfg.max_len))
    other = _cfg(workspace, task="facts", mode="top", max_steps=None, target_tokens=1500)
    assert other.band_frac is None and other.run_id != cfg.run_id
