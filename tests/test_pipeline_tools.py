"""launch_grid expansion (fixed lists are not axes), aggregate on synthetic results, dynamics on real factors,
baseline mode, toy experiments smoke."""
import csv
import json
from pathlib import Path

import pytest
import yaml

from lorasub.aggregate import collect, crossed_table, forgetting_delta, plot_crossed, plot_spectral_curve, select_lr, summarize
from lorasub.dynamics import analyze_run, list_checkpoints, plot_dynamics
from lorasub.launch_grid import apply_lr, expand, materialise, write_slurm
from lorasub.spectral import iter_target_modules
from lorasub.train import run
from conftest import MODEL_ID, _cfg
from tiny import TINY_CFG, tiny_model, tiny_tokenizer


def test_expand_fixed_lists_are_not_axes():
    spec = {"base": {"model": "m", "save_factor_steps": [0, 1, 2, "end"], "target_modules": ["q_proj"]},
            "grid": {"task": ["facts", "format"], "seed": [0, 1, 2]},
            "extra": [{"mode": "full", "task": "facts"}]}
    cfgs = expand(spec)
    assert len(cfgs) == 2 * 3 + 1
    assert all(c["save_factor_steps"] == [0, 1, 2, "end"] for c in cfgs)  # never expanded
    assert cfgs[-1]["mode"] == "full" and "seed" not in cfgs[-1]
    with pytest.raises(ValueError):
        expand({"base": {}, "grid": {"task": "facts"}})


def test_materialise_unique_ids_and_slurm(tmp_path, workspace):
    spec = {"base": {"model": MODEL_ID, "svd_cache": str(workspace / "svd"), "data_dir": str(workspace / "data"),
                     "out_dir": str(workspace / "runs"), "max_steps": 2, "tiny_model_config": TINY_CFG,
                     "gradient_checkpointing": False, "eval_forgetting": False, "eval_limit": 4, "batch_size": 4,
                     "grad_accum": 1, "max_len": 192},
            "grid": {"task": ["facts"], "mode": ["top", "bottom"], "seed": [0, 0]}}  # duplicate seed -> collapses
    to_run, done = materialise(expand(spec), tmp_path)
    assert len(to_run) + len(done) == 2  # 4 configs -> 2 unique run ids
    script = write_slurm(tmp_path, to_run, "PARTITION", "01:00:00", "16G", max_par=8, refuse_local=False)
    txt = script.read_text()
    assert "#SBATCH --array=0-1%8" in txt and "lorasub.train" in txt
    assert len((tmp_path / "configs.txt").read_text().strip().splitlines()) == 2
    # lr injection from a selection table
    table = {f"{MODEL_ID}|facts|top|16": 0.002}
    raw = {"model": MODEL_ID, "task": "facts", "mode": "top", "rank": 16, "lr": 5e-4}
    assert apply_lr(raw, table)["lr"] == 0.002 and apply_lr({**raw, "mode": "bottom"}, table)["lr"] == 5e-4


def _fake_results(runs_dir: Path, rows):
    for i, r in enumerate(rows):
        d = runs_dir / f"run{i}"
        d.mkdir(parents=True)
        (d / "status").write_text("DONE\n")
        with open(d / "results.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(r))
            w.writeheader()
            w.writerow(r)


def test_aggregate_select_lr_and_tables(tmp_path):
    rows = []
    base = dict(model="m", rank=16, n_trainable=100, band_frac="")
    for task, met, vmet in (("facts", "mcq_acc", "val_mcq_acc"), ("format", "format_parsed", "val_format_parsed")):
        for mode in ("free", "top", "bottom", "random"):
            for lr in (1e-4, 5e-4):
                for seed in (0, 1):
                    score = 0.5 + (0.2 if (mode == "top" and task == "format") or (mode == "bottom" and task == "facts") else 0)
                    score += 0.05 if lr == 5e-4 else 0
                    rows.append({**base, "task": task, "mode": mode, "lr": lr, "seed": seed, met: score + 0.01 * seed,
                                 vmet: score, "hellaswag_acc_norm": 0.6 - 0.01 * (lr == 5e-4)})
        rows.append({**base, "task": task, "mode": "full", "rank": 0, "lr": 2e-5, "seed": 0, met: 0.9, "hellaswag_acc_norm": 0.55})
        rows.append({**base, "task": task, "mode": "baseline", "rank": 0, "lr": 0.0, "seed": 0, met: 0.25, "hellaswag_acc_norm": 0.62})
    for f in (0.0, 0.5, 1.0):
        rows.append({**base, "task": "facts", "mode": "band", "band_frac": f, "lr": 5e-4, "seed": 0, "mcq_acc": 0.5 + 0.2 * f})
    # a failed run must be ignored
    _fake_results(tmp_path / "runs", rows)
    bad = tmp_path / "runs" / "runX"
    bad.mkdir()
    (bad / "status").write_text("FAILED\nboom")
    (bad / "results.csv").write_text("model,task\nm,facts\n")
    df = collect(tmp_path / "runs")
    assert len(df) == len(rows)
    sel = select_lr(df, tmp_path / "lr.yaml")["selected"]
    assert sel["m|facts|bottom|16"] == 5e-4 and sel["m|format|top|16"] == 5e-4
    assert yaml.safe_load((tmp_path / "lr.yaml").read_text())["selected"] == sel
    summ = summarize(df)
    assert "lr" in summ.columns and summ["n_seeds"].max() == 2  # lr is a grouping key
    tex = crossed_table(df, "m", 16, sel)
    assert "bottom" in tex and "full" in tex and "baseline" in tex and "90.0" in tex
    plot_crossed(df, "m", tmp_path / "crossed.png")
    plot_spectral_curve(df, "m", 16, tmp_path / "curve.png")
    assert (tmp_path / "crossed.png").exists() and (tmp_path / "curve.png").exists()
    fd = forgetting_delta(df)
    assert not fd.empty and (fd["d_hellaswag_acc_norm"].dropna() < 0).all()  # band rows had no forgetting metric


def test_dynamics_on_real_factors(workspace):
    cfg = _cfg(workspace, task="facts", mode="free", max_steps=4, save_factor_steps=[0, 2, "end"])
    run(cfg, tokenizer=tiny_tokenizer(), quiet=True)
    cps = list_checkpoints(cfg.run_dir)
    assert [s for s, _ in cps] == [0, 2, 4]
    model = tiny_model(0)
    mods = dict(iter_target_modules(model))
    df = analyze_run(cfg.run_dir, workspace / "svd", intruder_steps={4}, W0_provider=lambda n: mods[n].weight.detach())
    assert set(df.step) == {0, 2, 4}
    dec = df[df.band.str.startswith("frac")]
    for (step, layer, mt, side), g in dec.groupby(["step", "layer", "module_type", "side"]):
        outside = df[(df.step == step) & (df.layer == layer) & (df.module_type == mt) & (df.side == side)
                     & (df.band == "_outside")].energy.iloc[0]
        if step == 0:
            assert g.energy.sum() == 0.0  # dW = 0 at start
        else:
            assert abs(g.energy.sum() + outside - 1.0) < 1e-3
    assert df[df.step == 4].n_intruders.notna().all() and df[df.step == 2].n_intruders.isna().all()
    plot_dynamics(df, workspace / "dyn.png")
    assert (workspace / "dyn.png").exists()
    # full-FT deltas are analysed too
    cfg2 = _cfg(workspace, task="facts", mode="full", lr=1e-3, max_steps=2, save_full_delta=True,
                save_factor_steps=[0, "end"])
    run(cfg2, tokenizer=tiny_tokenizer(), quiet=True)
    df2 = analyze_run(cfg2.run_dir, workspace / "svd")
    assert set(df2.step) == {0, 2} and (df2[df2.step == 2].eff_rank > 0).all()


def test_baseline_mode(workspace):
    cfg = _cfg(workspace, task="facts", mode="baseline")
    row = run(cfg, tokenizer=tiny_tokenizer(), quiet=True)
    assert row["mode"] == "baseline" and row["n_trainable"] == 0 and "mcq_acc" in row and "val_mcq_acc" in row
    assert not list((cfg.run_dir).glob("factors_*.safetensors"))


def test_facts_french_pack(tmp_path):
    from lorasub.data.facts import generate_facts, load_templates, read_jsonl

    tp = load_templates("fr")
    assert "né" in tp.train["birth_year"][0]
    generate_facts(tmp_path, n_entities=20, seed=0, n_test_mcq=12, n_test_completion=12, n_val_mcq=6, lang="fr")
    q = read_jsonl(tmp_path / "mcq_test.jsonl")[0]["question"]
    assert "?" in q and json.load(open(tmp_path / "meta.json"))["lang"] == "fr"


def test_toy_experiments_smoke():
    import importlib.util

    spec = importlib.util.spec_from_file_location("toy", Path(__file__).resolve().parents[1] / "notebooks" / "toy.py")
    toy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(toy)
    s = toy.experiment_1(d=32, r=4, n=400, seeds=2, steps=200)
    # the matching arm recovers the target; the opposite arm does not learn it at all
    assert s[("top-band", "top", "err")][0] < 0.05 and s[("bottom-band", "bottom", "err")][0] < 0.05
    assert s[("top-band", "bottom", "err")][0] > 0.5 and s[("bottom-band", "top", "err")][0] > 0.5
    # the free arm concentrates its energy in the band where the target lives
    assert s[("top-band", "free", "top")][0] > 0.8 and s[("bottom-band", "free", "bottom")][0] > 0.8
    acc = toy.experiment_3(d=32, r=4, draws=10)
    assert abs(acc["top:4"] - 4 / 32) < 0.08


def test_shipped_grids_are_valid_and_reach_low_rank():
    """Every shipped grid must expand to unique, loadable configs, fit a 16 GB node, and the rank axis
    must reach r<=2 — at r=16 the subspace constraint no longer bites (free LoRA hits 95.8 % on the
    hard format task), so a grid that starts at 4 would measure nothing about the subspace."""
    import glob

    import yaml as _yaml

    from lorasub.config import RunConfig, apply_model_defaults

    seen_ranks = set()
    for f in sorted(glob.glob(str(Path(__file__).resolve().parents[1] / "configs" / "grids" / "*.yaml"))):
        spec = _yaml.safe_load(open(f))
        cfgs = expand(spec)
        assert cfgs, f
        ids = set()
        for c in cfgs:
            cfg = RunConfig(**apply_model_defaults(dict(c)))
            ids.add(cfg.run_id)
            # 16 GB nodes: a 1B in bf16 with checkpointing needs batch*len small
            if cfg.mode != "baseline":
                assert cfg.batch_size * cfg.max_len <= 1024, (f, cfg.batch_size, cfg.max_len)
            seen_ranks.add(cfg.rank)
        assert len(ids) == len(cfgs), f"{f}: duplicate run_ids"
    assert {1, 2} & seen_ranks, "no grid reaches the low-rank regime"


def test_commonsense_task_is_wired_end_to_end(tmp_path):
    """The public benchmark must reach the loader, the resolver and the evaluator without a special
    case anywhere else: rows in, packed dataset out, MCQ file resolved. The converter itself needs
    the network, so the rows here stand in for what it writes."""
    import json
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from tiny import tiny_tokenizer
    from lorasub.data.loaders import build_dataset
    from lorasub.data.paths import resolve

    d = tmp_path / "commonsense"
    d.mkdir()
    rows = [{"text": f"Question: q{i}\nAnswer: yes", "question": f"q{i}", "answer_text": "yes"}
            for i in range(200)]
    (d / "train.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    mcq = [{"question": f"q{i}", "options": ["yes", "no"], "answer_idx": 0} for i in range(20)]
    for name in ("mcq_val.jsonl", "mcq_test.jsonl"):
        (d / name).write_text("\n".join(json.dumps(r) for r in mcq))

    tok = tiny_tokenizer()
    ds = build_dataset("commonsense", tmp_path, tok, max_len=32, split="train", seed=0)
    assert len(ds) > 0
    assert resolve(tmp_path, "commonsense", "mcq_test").path.name == "mcq_test.jsonl"
    assert resolve(tmp_path, "commonsense", "mcq_val").path.name == "mcq_val.jsonl"
    assert not resolve(tmp_path, "commonsense", "mcq_val").val_is_test   # selection stays unbiased
    # and the evaluator itself must read these rows: a schema that the loader accepts but the
    # evaluator does not (a field named `answer` instead of `answer_idx`) cost a 57-run campaign.
    from tiny import tiny_model
    from lorasub.eval import run_all
    out = run_all(tiny_model(5), tok, "commonsense", tmp_path, limit=5, split="test", do_forgetting=False)
    assert "mcq_acc" in out and out["mcq_n"] == 5
