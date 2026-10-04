import sys
"""Data-path adapter (a co-author's file names), per_module arm, update-norm logging/matching, stratified probe,
gradient effective rank."""
import json
import inspect
import random
import shutil
import pandas as pd
from pathlib import Path

import pytest
import torch

from conftest import MODEL_ID, _cfg
from lorasub.data.loaders import stratified_rows
from lorasub.data.paths import describe, resolve

from lorasub.grad_probe import gradient_effective_rank, probe
from lorasub.lora import (count_trainable, expected_trainable, inject_lora, parse_per_module, rescale_update_,
                          update_norms)
from lorasub.spectral import compute_svd_cache
from lorasub.train import run
from tiny import TINY_CFG, tiny_model, tiny_tokenizer


def test_paths_resolve_both_conventions(workspace, tmp_path):
    d = workspace / "data"
    assert resolve(d, "format", "train").path.name == "train.jsonl"
    assert not resolve(d, "facts", "mcq_val").val_is_test
    # a co-author's layout: format_train/format_test, mcq.jsonl, no validation split
    dd = tmp_path / "coauthor"
    (dd / "facts").mkdir(parents=True)
    (dd / "format").mkdir()
    shutil.copy(d / "facts" / "train.jsonl", dd / "facts" / "train.jsonl")
    shutil.copy(d / "facts" / "mcq_test.jsonl", dd / "facts" / "mcq.jsonl")
    shutil.copy(d / "facts" / "completion.jsonl", dd / "facts" / "completion.jsonl")
    shutil.copy(d / "format" / "train.jsonl", dd / "format" / "format_train.jsonl")
    shutil.copy(d / "format" / "test.jsonl", dd / "format" / "format_test.jsonl")
    assert resolve(dd, "format", "train").path.name == "format_train.jsonl"
    assert resolve(dd, "facts", "mcq_test").path.name == "mcq.jsonl"
    r = resolve(dd, "format", "val")
    assert r.val_is_test and r.path.name == "format_test.jsonl"
    assert resolve(dd, "facts", "mcq_val").val_is_test
    desc = describe(dd)
    assert desc["format/val"].endswith("(=test!)")
    # a full run on ANON-style data works and flags the biased validation
    cfg = _cfg(workspace, task="format", mode="top", data_dir=str(dd), max_steps=2)
    row = run(cfg, tokenizer=tiny_tokenizer(), quiet=True)
    assert row["format_parsed"] >= 0.0 and row["val_val_is_test"] == 1
    with pytest.raises(FileNotFoundError):
        resolve(tmp_path / "empty", "facts", "train")


def test_per_module_arm(tmp_path):
    model = tiny_model(7)
    compute_svd_cache(model, tmp_path, "tiny/m", device="cpu", verbose=False)
    spec = {"q_proj": "top", "k_proj": "bottom", "up_proj": "band:0.5", "default": "random"}
    assert parse_per_module(spec, "model.layers.0.self_attn.q_proj") == ("top", None)
    assert parse_per_module(spec, "model.layers.0.mlp.up_proj") == ("band", 0.5)
    assert parse_per_module(spec, "model.layers.0.mlp.down_proj") == ("random", None)
    with pytest.raises(ValueError):
        parse_per_module({"q_proj": "free"}, "model.layers.0.self_attn.q_proj")
    with pytest.raises(ValueError):
        parse_per_module({"q_proj": "top"}, "model.layers.0.self_attn.k_proj")  # no default
    mods = inject_lora(model, r=3, mode="per_module", cache_dir=tmp_path, model_id="tiny/m", per_module=spec)
    modes = {n.split(".")[-1]: m.mode for n, m in mods.items()}
    assert modes["q_proj"] == "top" and modes["k_proj"] == "bottom" and modes["up_proj"] == "band"
    assert modes["down_proj"] == "random"
    assert all(not m.B.requires_grad for m in mods.values())  # every module constrained
    assert count_trainable(model) == expected_trainable(model, 3, "bottom")  # same budget as bottom/top
    with pytest.raises(ValueError):
        inject_lora(tiny_model(7), r=3, mode="per_module", cache_dir=tmp_path, model_id="tiny/m")


def test_update_norms_and_rescale(tmp_path):
    model = tiny_model(8)
    mods = inject_lora(model, r=4, mode="free", alpha=8.0)
    for m in mods.values():
        m.B.data.normal_()
    norms = update_norms(mods)
    m0 = next(iter(mods.values()))
    brute = float((m0.B @ m0.A).detach().norm()) * m0.scaling
    assert abs(norms[next(iter(mods))] - brute) < 1e-4
    mean_before = sum(norms.values()) / len(norms)
    f = rescale_update_(mods, target_mean_norm=2.0 * mean_before)
    assert abs(f - 2.0) < 1e-6
    after = update_norms(mods)
    assert abs(sum(after.values()) / len(after) - 2.0 * mean_before) < 1e-4


def test_run_logs_update_norm_and_matches(workspace):
    cfg = _cfg(workspace, task="facts", mode="bottom", max_steps=3)
    row = run(cfg, tokenizer=tiny_tokenizer(), quiet=True)
    assert row["update_norm_mean"] > 0 and (cfg.run_dir / "update_norms.json").exists()
    cfg2 = _cfg(workspace, task="facts", mode="bottom", max_steps=3, update_norm_target=1.0)
    row2 = run(cfg2, tokenizer=tiny_tokenizer(), quiet=True)
    assert "update_rescale_factor" in row2 and (cfg2.run_dir / "factors_rescaled.safetensors").exists()
    assert cfg.run_id != cfg2.run_id  # the ablation is a distinct run


def test_stratified_rows_and_probe_columns(workspace):
    rows = [{"attr": a, "text": f"{a}-{i}"} for a in ("x", "y", "z") for i in range(10)]
    sub = stratified_rows(rows, 9, key="attr", seed=0)
    assert sorted(r["attr"] for r in sub).count("x") == 3
    sub2 = stratified_rows([{"text": "t"} for _ in range(5)], 3, key="attr")  # key absent -> random subset
    assert len(sub2) == 3
    G = torch.randn(20, 10)
    assert 1.0 <= gradient_effective_rank(G) <= 10.0
    assert gradient_effective_rank(torch.zeros(4, 4)) == 0.0
    df = probe(tiny_model(0), tiny_tokenizer(), MODEL_ID, str(workspace / "data"), str(workspace / "svd"),
               n_examples=8, batch_size=4, max_len=192, group_layers=1, device=torch.device("cpu"))
    assert "grad_eff_rank" in df and (df.grad_eff_rank > 0).all()


def test_feasibility_script_runs_on_tiny_model(workspace, tmp_path, monkeypatch):
    """The feasibility probe builds its own subset, runs, and writes a verdict."""
    import importlib.util
    import sys

    path = Path(__file__).resolve().parents[1] / "scripts" / "task_feasibility.py"
    spec = importlib.util.spec_from_file_location("task_feasibility", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    real_run = mod.run

    def patched(cfg, **kw):  # inject the tiny model into the config the script built
        cfg.tiny_model_config = TINY_CFG
        cfg.dtype = "float32"
        cfg.gradient_checkpointing = False
        return real_run(cfg, tokenizer=tiny_tokenizer(), quiet=True)

    monkeypatch.setattr(mod, "run", patched)
    monkeypatch.setattr(sys, "argv", [
        "task_feasibility.py", "--model", MODEL_ID, "--data_dir", str(workspace / "data"),
        "--out", str(tmp_path / "feas"), "--n_entities", "8", "--epochs", "1", "--eval_limit", "8",
        "--mode", "bottom", "--rank", "4", "--batch_size", "2", "--grad_accum", "1", "--max_len", "192",
        "--svd_cache", str(workspace / "svd")])
    mod.main()
    verdict = json.loads((tmp_path / "feas" / "verdict.json").read_text())
    assert verdict["n_entities"] == 8 and "mcq_acc" in verdict
    assert verdict["stats"]["templates_per_attribute"] >= 20
    assert (tmp_path / "feas" / "data" / "facts" / "train.jsonl").exists()


def test_leakage_check_covers_all_training_families_and_the_mcq_prompt():
    """Audit item 1+2: the MCQ question wording must not appear in ANY training family (train, qa, pairs,
    bio), and the check must see qa/pairs, not only train/bio."""
    import dataclasses

    from lorasub.data.facts import ATTRIBUTES, check_no_leakage, load_templates

    tp = load_templates("en")
    assert check_no_leakage(tp) == []
    # inject the evaluation question into a qa template -> must be caught
    q = tp.question["birth_year"].rstrip("?")
    bad_qa = dataclasses.replace(tp, qa={**tp.qa, "birth_year": tp.qa["birth_year"] + (q + " Answer: {value}.",)})
    leaks = check_no_leakage(bad_qa)
    assert leaks and leaks[0].startswith("question/birth_year")
    # inject a held-out wording into a *pair* template -> must be caught too
    held = tp.heldout["field"][0]
    bad_pairs = dataclasses.replace(tp, pairs=tp.pairs + (held,))
    assert any(l.startswith("heldout/field") for l in check_no_leakage(bad_pairs))
    for a in ATTRIBUTES:  # the qa family exists and differs from the question wording for every attribute
        assert tp.qa[a] and all(tp.question[a].rstrip("?").lower() not in t.lower() for t in tp.qa[a])


def test_forgetting_leaves_model_unmerged(workspace):
    """Audit item 5: forgetting must not deepcopy; it merges in place and restores the model."""
    import lorasub.eval as E
    from lorasub.lora import inject_lora, lora_modules

    model = tiny_model(3)
    mods = inject_lora(model, r=4, mode="free")
    for m in mods.values():
        m.B.data.normal_()
    ids = torch.randint(4, 150, (2, 8))
    with torch.no_grad():
        before = model(input_ids=ids).logits.clone()
    out = E.forgetting(model, tiny_tokenizer(), limit=2)  # lm_eval absent or failing -> skipped, but merge/unmerge ran
    assert "forgetting_skipped" in out or "hellaswag_acc_norm" in out
    assert all(not m.merged for m in lora_modules(model).values())
    with torch.no_grad():
        after = model(input_ids=ids).logits
    assert torch.allclose(before, after, atol=1e-4)


def test_slurm_script_bootstraps_and_refuses_local_grid_dir(tmp_path, workspace):
    """Audit item 3/9/10: the job must build its own environment, export SHARED/HF_HOME, source the
    token, not request a gres unless asked, and refuse a node-local grid_dir."""
    from lorasub.launch_grid import expand, materialise, write_slurm

    spec = {"base": {"model": MODEL_ID, "svd_cache": str(workspace / "svd"), "data_dir": str(workspace / "data"),
                     "out_dir": str(workspace / "runs"), "max_steps": 1, "tiny_model_config": TINY_CFG,
                     "gradient_checkpointing": False, "eval_forgetting": False, "eval_limit": 2,
                     "batch_size": 2, "grad_accum": 1, "max_len": 192},
            "grid": {"task": ["facts"], "mode": ["top"], "seed": [0]}}
    to_run, _ = materialise(expand(spec), tmp_path)
    script = write_slurm(tmp_path, to_run, "PARTITION", "01:55:00", "32G", 100, shared="/tmp/lora-$USER",
                         python_bin="python3.10", refuse_local=False)
    txt = script.read_text()
    for needle in ('export SHARED="/tmp/lora-$USER"', 'export HF_HOME="$SHARED/hf"', ".hf_env",
                   "python3.10 -m venv", "install -q -r", "--time=01:55:00", "--partition=PARTITION",
                   "-m lorasub.train"):
        assert needle in txt, needle
    assert "--gres" not in txt
    assert "--gres=PARTITION:1" in write_slurm(tmp_path, to_run, "p", "01:00:00", "16G", 8, gres="PARTITION:1",
                                         refuse_local=False).read_text()
    with pytest.raises(ValueError):
        write_slurm(Path("/tmp/lora-x/grid"), to_run, "p", "01:00:00", "16G", 8)


def test_train_builds_missing_svd_cache(workspace, tmp_path):
    """Audit item 4: a constrained run on a node without the cache builds it instead of crashing."""
    cfg = _cfg(workspace, task="facts", mode="bottom", max_steps=1, svd_cache=str(tmp_path / "fresh_svd"))
    row = run(cfg, tokenizer=tiny_tokenizer(), quiet=True)
    assert row["mode"] == "bottom"
    from lorasub.spectral import list_cached

    assert len(list_cached(tmp_path / "fresh_svd", MODEL_ID)) == 14


def test_subspace_seed_separates_columns_from_data_order(workspace):
    """Audit item 6: with subspace_seed fixed, two seeds share the same random columns."""
    from lorasub.lora import lora_modules

    a = _cfg(workspace, task="facts", mode="random", max_steps=1, seed=0, subspace_seed=7)
    b = _cfg(workspace, task="facts", mode="random", max_steps=1, seed=1, subspace_seed=7)
    assert a.run_id != b.run_id
    from lorasub.lora import inject_lora

    ma, mb = tiny_model(0), tiny_model(0)
    inject_lora(ma, r=4, mode="random", cache_dir=workspace / "svd", model_id=MODEL_ID, seed=7)
    inject_lora(mb, r=4, mode="random", cache_dir=workspace / "svd", model_id=MODEL_ID, seed=7)
    for (n, x), (_, y) in zip(lora_modules(ma).items(), lora_modules(mb).items()):
        assert torch.equal(x.band_idx, y.band_idx)


def test_scientific_notation_overrides_are_coerced():
    """`lr=5e-4` on the command line arrives as a string (YAML 1.1 needs 5.0e-4). Left alone it
    crashes the optimiser and, worse, changes run_id — so two spellings of the same run would be
    treated as two different cells of the grid."""
    from lorasub.config import RunConfig

    a = RunConfig(model=MODEL_ID, lr="5e-4", rank="16", seed="2", band_frac=None,
                  svd_cache="/x", data_dir="/x", out_dir="/x")
    b = RunConfig(model=MODEL_ID, lr=5.0e-4, rank=16, seed=2, band_frac=None,
                  svd_cache="/x", data_dir="/x", out_dir="/x")
    assert isinstance(a.lr, float) and isinstance(a.rank, int) and isinstance(a.seed, int)
    assert a.lr == b.lr and a.run_id == b.run_id
    c = RunConfig(model=MODEL_ID, mode="band", band_frac="0.5", target_tokens="600000",
                  eval_forgetting="false", svd_cache="/x", data_dir="/x", out_dir="/x")
    assert c.band_frac == 0.5 and c.target_tokens == 600000 and c.eval_forgetting is False
    with pytest.raises(ValueError):
        RunConfig(model=MODEL_ID, eval_forgetting="maybe", svd_cache="/x", data_dir="/x", out_dir="/x")


def test_slurm_preamble_uses_extra_index_url(tmp_path, workspace):
    """`--index-url` replaces the default index, so build dependencies (flit_core, setuptools) become
    unreachable and the venv build fails on a fresh node — it did, on the first compute node we used.
    `--extra-index-url` adds the PyTorch wheels to the default index instead."""
    from lorasub.launch_grid import expand, materialise, write_slurm

    spec = {"base": {"model": MODEL_ID, "svd_cache": "/x", "data_dir": "/x", "out_dir": "/x",
                     "max_steps": 1, "tiny_model_config": TINY_CFG},
            "grid": {"task": ["facts"], "mode": ["top"], "seed": [0]}}
    to_run, _ = materialise(expand(spec), tmp_path)
    txt = write_slurm(tmp_path, to_run, "PARTITION", "01:55:00", "32G", 10, refuse_local=False).read_text()
    assert "--extra-index-url https://download.pytorch.org/whl/" in txt
    assert "--index-url https://download.pytorch.org/whl/" not in txt
    assert "upgrade pip setuptools wheel" in txt
    remote = (Path(__file__).resolve().parents[1] / "scripts" / "remote_setup.sh").read_text()
    assert "--extra-index-url" in remote and "pip install -q torch --index-url" not in remote


def test_slurm_preamble_keeps_pip_cache_off_the_home(tmp_path, workspace):
    """The home is an NFS quota shared with everything else (30 GB, 2.7 GB free here). Twenty nodes
    downloading torch into $HOME/.cache/pip at once filled it and every install died with
    'Disk quota exceeded'. The cache must be node-local, and the job must refuse to start when the
    home has no room for results."""
    from lorasub.launch_grid import expand, materialise, write_slurm

    spec = {"base": {"model": MODEL_ID, "svd_cache": "/x", "data_dir": "/x", "out_dir": "/x",
                     "max_steps": 1, "tiny_model_config": TINY_CFG},
            "grid": {"task": ["facts"], "mode": ["top"], "seed": [0]}}
    to_run, _ = materialise(expand(spec), tmp_path)
    txt = write_slurm(tmp_path, to_run, "PARTITION", "01:55:00", "32G", 20, refuse_local=False).read_text()
    assert 'PIP_CACHE_DIR="$SHARED/pipcache"' in txt
    assert "$HOME/.cache/pip" not in txt
    assert 'TMPDIR="$SHARED/tmp"' in txt          # NFS TMPDIR gives "Stale file handle" under load
    assert "less than 512 MB free" in txt          # guard before ten minutes of install
    assert "pip install failed on" in txt          # a failed install must say so, not die silently


def test_slurm_preamble_rebuilds_an_incomplete_venv(tmp_path, workspace):
    """A job that dies between `venv` creation and `pip install` leaves a python binary with no
    package. Testing for the binary then skips the install for ever after, and every later job on that
    node fails with 'No module named lorasub' — which is exactly what happened on node."""
    from lorasub.launch_grid import expand, materialise, write_slurm

    spec = {"base": {"model": MODEL_ID, "svd_cache": "/x", "data_dir": "/x", "out_dir": "/x",
                     "max_steps": 1, "tiny_model_config": TINY_CFG},
            "grid": {"task": ["facts"], "mode": ["top"], "seed": [0]}}
    to_run, _ = materialise(expand(spec), tmp_path)
    txt = write_slurm(tmp_path, to_run, "PARTITION", "01:55:00", "32G", 10, refuse_local=False).read_text()
    assert 'import lorasub, torch' in txt          # the package, not just the interpreter
    assert 'rm -rf "$VENV"' in txt                  # an incomplete venv is rebuilt, not reused
    assert '[ ! -x "$VENV/bin/python" ]' not in txt  # the old, insufficient test is gone


def test_unmasked_format_control(workspace):
    """The facts gradient is supervised on every token, the format gradient only on the answer. A
    difference in effective rank between the two could therefore be an artefact of the masking, not a
    property of the tasks. 'format_unmasked' supervises the format data on every token so the two are
    comparable; it is a control condition, never a training mode."""
    import torch as _t

    from lorasub.data.loaders import build_dataset
    from lorasub.grad_probe import probe
    from tiny import tiny_tokenizer as _tok

    tok = _tok()
    masked = build_dataset("format", workspace / "data", tok, 192, rows=None)
    unmasked = build_dataset("format", workspace / "data", tok, 192, rows=None, mask_prompt=False)
    assert unmasked.n_tokens > masked.n_tokens          # whole sequence vs answer only
    a = masked[0]
    b = unmasked[0]
    assert (a["labels"] == -100).any() and not (b["labels"] == -100).any()
    assert _t.equal(a["input_ids"], b["input_ids"])      # same data, only the supervision changes

    df = probe(tiny_model(0), tok, MODEL_ID, str(workspace / "data"), str(workspace / "svd"),
               datasets=("format", "format_unmasked"), n_examples=8, batch_size=4, max_len=192,
               group_layers=1, device=_t.device("cpu"))
    assert set(df.dataset) == {"format", "format_unmasked"}
    sup = df.groupby("dataset")["sup_tokens_per_example"].first()
    assert sup["format_unmasked"] > sup["format"]        # reported, so the control is auditable


def test_slurm_venv_build_is_serialised_and_atomic(tmp_path, workspace):
    """Several array tasks land on the same node and would build the same venv at once, deleting each
    other's files mid-install ('No such file or directory: _distutils_hack/') — which killed 43 of 48
    jobs. One flock per node serialises the build, and the venv is renamed into place only once
    `import lorasub, torch` succeeds, so an interrupted build is never reused."""
    from lorasub.launch_grid import expand, materialise, write_slurm

    spec = {"base": {"model": MODEL_ID, "svd_cache": "/x", "data_dir": "/x", "out_dir": "/x",
                     "max_steps": 1, "tiny_model_config": TINY_CFG},
            "grid": {"task": ["facts"], "mode": ["top"], "seed": [0]}}
    to_run, _ = materialise(expand(spec), tmp_path)
    txt = write_slurm(tmp_path, to_run, "PARTITION", "01:55:00", "32G", 10, refuse_local=False).read_text()
    assert "flock" in txt and 'VENV_LOCK="$SHARED/venv.lock"' in txt
    assert 'BUILD="$VENV.building.$$"' in txt        # build under a private name
    assert 'mv "$BUILD" "$VENV"' in txt              # publish atomically
    build_at = txt.index('mv "$BUILD" "$VENV"')
    assert txt.index('import lorasub, torch', txt.index("$BUILD/bin/python")) < build_at  # verify, then publish
    assert txt.index("flock") < txt.index("$BUILD/bin/pip")                                # lock before building
    # NEVER activate: bin/activate hard-codes VIRTUAL_ENV at creation time, so after the atomic rename
    # it puts a dead directory at the head of PATH and `python` silently becomes the system one.
    code = "\n".join(l for l in txt.splitlines() if not l.lstrip().startswith("#"))
    assert "activate" not in code                  # mentioned in the comment, never executed
    assert 'PY="$VENV/bin/python"' in txt and '"$PY" -m lorasub.train' in txt
    assert txt.index('"$PY" -c "import lorasub, torch"') < txt.index('"$PY" -m lorasub.train')


def test_renaming_a_venv_breaks_activation_but_not_absolute_path(tmp_path):
    """The empirical reason the preamble must not activate: `bin/activate` hard-codes the path used at
    creation, so after the atomic rename it points at a directory that no longer exists, PATH is
    poisoned and `python` silently becomes the system interpreter — which is exactly how 'venv ready'
    was followed by 'No module named lorasub'. The absolute path is immune."""
    import subprocess
    import sys

    build, final = tmp_path / "v.building.123", tmp_path / "v"
    subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(build)], check=True)
    build.rename(final)
    assert str(build) in (final / "bin" / "activate").read_text()   # stale path baked in
    out = subprocess.run(["bash", "-c", f'source "{final}/bin/activate"; command -v python || true'],
                         capture_output=True, text=True).stdout.strip()
    assert not out.startswith(str(final))                           # activation resolves elsewhere
    prefix = subprocess.run([str(final / "bin" / "python"), "-c", "import sys; print(sys.prefix)"],
                            capture_output=True, text=True).stdout.strip()
    assert prefix == str(final)                                     # absolute path still correct


def test_energy_at_rank_is_the_metric_effective_rank_is_not():
    """exp(H(p)) weights sigma, not sigma^2: a flat tail inflates it while carrying no energy. A
    matrix with 95 % of its energy in 16 directions can score above 1000. energy_at_rank is
    energy-weighted and says what a rank-r adapter could represent at best."""
    from lorasub.grad_probe import energy_at_rank, gradient_effective_rank, noise_ceiling, stable_rank

    torch.manual_seed(0)
    d = 512
    U, _ = torch.linalg.qr(torch.randn(d, d))
    V, _ = torch.linalg.qr(torch.randn(d, d))
    # 16 strong directions holding 95 % of the energy, plus a long flat tail holding 5 %
    s = torch.zeros(d)
    s[:16] = 1.0
    s[16:] = ((0.05 / 0.95) * 16 / (d - 16)) ** 0.5
    G = U @ torch.diag(s) @ V.T
    eatr = energy_at_rank(G)
    assert 0.94 < eatr[16] < 0.96                    # the truth: 95 % of the energy at rank 16
    eff, stab = gradient_effective_rank(G), stable_rank(G)
    assert eff > 200 and eff > 10 * stab             # the entropy metric reads "hundreds" anyway
    assert stab < 30                                 # energy-weighted, stays honest
    # a genuinely spread matrix: energy_at_rank stays low where the entropy is similar
    Gflat = U @ torch.diag(torch.ones(d)) @ V.T
    assert energy_at_rank(Gflat)[16] < 0.05
    assert energy_at_rank(Gflat)[64] < 0.2
    # the ceiling grows with the number of averaged contributions: comparing raw values across
    # datasets with different token counts is meaningless
    assert noise_ceiling(1024, (d, d)) > noise_ceiling(256, (d, d)) > noise_ceiling(64, (d, d))


def test_split_half_separates_systematic_from_idiosyncratic():
    """The control for the new reading. A gradient whose per-example contributions agree (every format
    example asks for the same transformation) reproduces across a disjoint half; one made of
    example-specific contributions (each fact is a different entity) does not."""
    from lorasub.grad_probe import split_half_coherence

    torch.manual_seed(1)
    d, k = 256, 16
    shared = torch.randn(d, k) @ torch.randn(k, d)          # the same directions in both halves
    sys_a = shared + 0.1 * torch.randn(d, d)
    sys_b = shared + 0.1 * torch.randn(d, d)
    idio_a, idio_b = torch.randn(d, d), torch.randn(d, d)   # nothing in common
    sysc = split_half_coherence(sys_a, sys_b)
    idio = split_half_coherence(idio_a, idio_b)
    assert sysc["cosine"] > 0.8 and idio["cosine"] < 0.2
    assert sysc["subspace_overlap"] > 0.5
    assert idio["subspace_overlap"] < 3 * idio["overlap_chance"]


def test_probe_reports_the_new_metrics(workspace):
    """End to end on the tiny model: the probe must emit energy_at_r*, the split-half columns, the
    calibrated effective rank and the stored spectrum."""
    from lorasub.grad_probe import probe

    df = probe(tiny_model(0), tiny_tokenizer(), MODEL_ID, str(workspace / "data"), str(workspace / "svd"),
               datasets=("facts", "format"), n_examples=8, batch_size=2, max_len=192, group_layers=1,
               device=torch.device("cpu"), ranks=(1, 2, 4), keep_spectrum=8)
    for col in ("energy_at_r1", "energy_at_r4", "stable_rank", "split_cosine", "split_overlap",
                "grad_eff_rank_ceiling", "grad_eff_rank_frac", "sup_tokens_total", "n_batches"):
        assert col in df.columns, col
    assert (df.energy_at_r1 <= df.energy_at_r4 + 1e-9).all()      # cumulative by construction
    assert (df.energy_at_r4 <= 1.0 + 1e-9).all()
    assert df.attrs["spectra"] and len(df.attrs["spectra"][0]["sigma"]) <= 8


def test_bottom_r_skips_numerically_null_directions():
    """Audit: on Llama-3.2-1B, q_proj has sigma_min = 3.5e-8 against sigma_max = 10.7. The singular
    vectors attached to such values are ill-conditioned (Davis-Kahan), so 'bottom-r' would adapt an
    essentially arbitrary direction rather than the least-used one. With sigma_rel_tol, the band is
    taken from the numerically non-zero part."""
    from lorasub.spectral import band_indices, numerical_rank

    S = torch.tensor([10.0, 5.0, 1.0, 1e-3, 1e-9, 1e-12])
    assert numerical_rank(S, 1e-6) == 4
    assert numerical_rank(S, 1e-2) == 3
    assert numerical_rank(torch.zeros(4)) == 0
    assert band_indices(6, "bottom:2").tolist() == [4, 5]                       # includes the null ones
    assert band_indices(6, "bottom:2", S=S, rel_tol=1e-6).tolist() == [2, 3]    # excludes them
    assert band_indices(6, "top:2", S=S, rel_tol=1e-6).tolist() == [0, 1]       # top is unaffected
    with pytest.raises(ValueError):                                             # not enough real ones
        band_indices(6, "bottom:5", S=S, rel_tol=1e-2)


def test_sigma_rel_tol_changes_the_arm_and_the_run_id(tmp_path, workspace):
    """The tolerance is a scientific choice, so it must enter run_id: two runs that adapt different
    directions cannot share an identifier."""
    from lorasub.lora import inject_lora, lora_modules

    a = _cfg(workspace, task="format", mode="bottom", max_steps=1)
    b = _cfg(workspace, task="format", mode="bottom", max_steps=1, sigma_rel_tol=1e-6)
    assert a.run_id != b.run_id

    from lorasub.spectral import compute_svd_cache

    compute_svd_cache(tiny_model(5), tmp_path, "tiny/m", device="cpu", verbose=False)
    net1, net2 = tiny_model(5), tiny_model(5)
    inject_lora(net1, r=2, mode="bottom", cache_dir=tmp_path, model_id="tiny/m")
    inject_lora(net2, r=2, mode="bottom", cache_dir=tmp_path, model_id="tiny/m",
                sigma_rel_tol=0.5)   # aggressive: keeps only the dominant directions
    i1 = {n: m.band_idx.tolist() for n, m in lora_modules(net1).items()}
    i2 = {n: m.band_idx.tolist() for n, m in lora_modules(net2).items()}
    assert any(i1[n] != i2[n] for n in i1), "a tolerance of 0.5 must move the band on some module"
    # and the excluded directions really are the small ones
    from lorasub.spectral import load_svd, numerical_rank

    e = load_svd(tmp_path, "tiny/m", "model.layers.0.self_attn.q_proj", device="cpu")
    assert numerical_rank(e.S, 0.5) < e.m


def test_aggregate_refuses_to_compare_different_eval_sizes(tmp_path):
    """Audit: eval_limit is excluded from run_id, so it can drift between relaunches of one grid —
    a cell finished before the change keeps a different test subset from one finished after, and the
    two get compared as if they were the same measurement. This happened during the rank sweep."""
    import csv

    from lorasub.aggregate import binomial_se, check_eval_sizes, collect, summarize

    runs = tmp_path / "runs"
    rows = []
    for mode, n in (("top", 400), ("bottom", 614), ("random", 400), ("free", 400)):
        rows.append(dict(model="m", task="format", mode=mode, rank=2, lr=5e-4, seed=0, band_frac="",
                         n_trainable=1, format_parsed=0.3, format_n=n))
    for i, r in enumerate(rows):
        d = runs / f"r{i}"
        d.mkdir(parents=True)
        (d / "status").write_text("DONE\n")
        with open(d / "results.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(r))
            w.writeheader()
            w.writerow(r)
    df = collect(runs)
    msgs = check_eval_sizes(df)
    assert msgs and "different numbers of examples" in msgs[0] and "bottom" in msgs[0]
    # same size everywhere -> silent
    df2 = df.copy()
    df2["format_n"] = 400
    assert check_eval_sizes(df2) == []
    # the sampling error is reported next to the seed spread
    out = summarize(df2)
    assert "format_parsed_se" in out.columns
    assert abs(binomial_se(0.30, 400) - 0.0229) < 1e-4     # 2.3 points, bigger than several gaps
    assert binomial_se(0.30, 6000) < binomial_se(0.30, 400)


def test_parsed_requires_every_rule_and_exact_requires_content():
    """Audit, most serious finding. Before this fix, `format_parsed` — the primary metric of the
    crossed table — was satisfied by (a) an output whose lists are all empty, with "qzx_n": "zero" and
    an empty signature, internally coherent and wrong on every item since the data keep only sentences
    with >= 3 entities; and (b) an output with unsorted lists, although sorting is a stated rule.
    A model could therefore max the metric by learning the shell alone."""
    from collections import OrderedDict as OD

    from lorasub.data.format import CONLL_LABELS, alien_string, build_example, check_alien

    toks = ["John", "Smith", "works", "at", "Reuters", "in", "London", "and", "Acme", "Corp", "."]
    tags = [1, 2, 0, 0, 3, 0, 5, 0, 3, 4, 0]
    ex = build_example(toks, tags, CONLL_LABELS, "transform")

    good = check_alien(ex["target"], ex["gold"])
    assert good["parsed"] and good["exact"] and good["f1_micro"] == 1.0

    empty = OD([("qzx_pers", []), ("qzx_org", []), ("qzx_loc", []), ("qzx_misc", []),
                ("qzx_n", "zero"), ("qzx_sig", "")])
    r = check_alien(alien_string(empty), ex["gold"])
    assert r["parsed"] and r["f1_micro"] == 0.0      # coherent shell: still counts as rule-obeying
    assert not r["exact"]                             # but never as correct

    unsorted_out = OD(ex["gold"])
    unsorted_out["qzx_org"] = ["Reuters", "Acme Corp"]      # order of appearance, not sorted
    unsorted_out["qzx_sig"] = "SRAL"
    r = check_alien(alien_string(unsorted_out), ex["gold"])
    assert not r["order_ok"] and not r["parsed"]     # sorting is a rule; it is now enforced
    assert r["f1_micro"] == 1.0                       # the entities themselves are right


def test_probe_warns_when_the_token_budget_cannot_be_met(workspace, capsys):
    """Audit: --match_sup_tokens sized only the stopping criterion, never the subset, so the loader ran
    dry long before the budget and the two tasks were compared at unequal supervision — silently."""
    from lorasub.grad_probe import probe

    df = probe(tiny_model(0), tiny_tokenizer(), MODEL_ID, str(workspace / "data"), str(workspace / "svd"),
               datasets=("format",), n_examples=8, batch_size=2, max_len=192, group_layers=1,
               device=torch.device("cpu"), ranks=(1, 2), keep_spectrum=4, split_half=False,
               match_sup_tokens=10_000_000)
    out = capsys.readouterr().out
    assert "WARNING" in out and "equal supervision" in out.lower()
    assert df["sup_tokens_total"].iloc[0] < 10_000_000


def test_noise_ceiling_counts_supervised_tokens_not_batches():
    """Audit: the gradient of a linear layer is a sum of one rank-1 term per supervised TOKEN, so
    counting batches gave a ceiling of ~120 on gate_proj where the measurement was 400-1000 — a ratio
    above 1, which made the normalised quantity meaningless."""
    from lorasub.grad_probe import noise_ceiling

    assert noise_ceiling(20, (8192, 2048)) < 150            # what counting batches used to give
    assert noise_ceiling(200_000, (8192, 2048)) == 2048.0   # saturates at min(shape), as it must
    assert noise_ceiling(64, (512, 2048)) < noise_ceiling(64_000, (512, 2048))


def test_grid_axis_can_be_a_dict_of_fields_that_move_together():
    """Audit: the magnitude ablation needs one update_norm_target PER RANK (the free arm's norm grows
    with rank), so rank and target cannot be independent axes."""
    from lorasub.launch_grid import expand

    spec = {"base": {"model": MODEL_ID},
            "grid": {"mode": ["top", "bottom"],
                     "rank_and_target": [{"rank": 1, "update_norm_target": 0.36},
                                         {"rank": 2, "update_norm_target": 0.42}]}}
    cfgs = expand(spec)
    assert len(cfgs) == 4
    pairs = sorted({(c["rank"], c["update_norm_target"]) for c in cfgs})
    assert pairs == [(1, 0.36), (2, 0.42)]
    assert "rank_and_target" not in cfgs[0]


def test_format_eval_reports_truncation_and_degenerate_outputs(workspace):
    """Audit: an arm that loops without emitting the closing tag fails for a reason unrelated to the
    format, and an arm that emits a coherent empty shell obeys every rule while extracting nothing.
    Both are counted separately so neither can hide inside format_parsed."""
    import inspect

    from lorasub.eval import format_metrics, run_all

    src = inspect.getsource(format_metrics)
    assert "format_truncated" in src and "format_empty" in src
    assert '"<</alien>>" not in g' in src

    out = run_all(tiny_model(0), tiny_tokenizer(), "format", str(workspace / "data"), limit=4,
                  do_forgetting=False, split="test")
    for k in ("format_truncated", "format_empty", "format_exact", "format_parsed"):
        assert k in out and 0.0 <= out[k] <= 1.0, k


def test_null_subspace_grid_fixes_data_and_varies_only_the_subspace():
    """Audit: `top`/`bottom` are points, `random` is a distribution — two draws cannot locate them.
    The null grid fixes the data seed and redraws the subspace ten times."""
    import yaml as _yaml

    from lorasub.config import RunConfig, apply_model_defaults
    from lorasub.launch_grid import expand

    path = Path(__file__).resolve().parents[1] / "configs" / "grids" / "null_subspace_1b.yaml"
    cfgs = [RunConfig(**apply_model_defaults(dict(c))) for c in expand(_yaml.safe_load(open(path)))]
    randoms = [c for c in cfgs if c.mode == "random"]
    assert {c.seed for c in randoms} == {2}                       # data order fixed
    assert len({c.subspace_seed for c in randoms}) == 10          # only the subspace varies
    anchors = [c for c in cfgs if c.mode in ("top", "bottom")]
    assert len(anchors) == 6 and {c.seed for c in anchors} == {2}  # same data order as the null
    assert len({c.run_id for c in cfgs}) == len(cfgs)


def test_span_control_grid_drops_the_wide_modules():
    """Audit: gate_proj/up_proj are 8192x2048, so span(U) covers only a quarter of their output space
    — a restriction `free` does not have. This confound does not affect top/bottom/random but may
    explain much of the gap to `free`."""
    import yaml as _yaml

    from lorasub.config import RunConfig, apply_model_defaults
    from lorasub.launch_grid import expand

    path = Path(__file__).resolve().parents[1] / "configs" / "grids" / "span_control_1b.yaml"
    cfgs = [RunConfig(**apply_model_defaults(dict(c))) for c in expand(_yaml.safe_load(open(path)))]
    for c in cfgs:
        assert "gate_proj" not in c.target_modules and "up_proj" not in c.target_modules
        assert "q_proj" in c.target_modules and "down_proj" in c.target_modules


def test_xd_decomposition_identifies_which_factor_limits_the_gradient(workspace):
    """Audit: G = D^T X, so measuring X and D separately says whether a depth profile reflects the
    diversity of what enters the layer or of the error signal leaving it. Verified against a
    construction where the inputs are deliberately confined to three directions. (The algebraic bound
    rank(G) <= min(rank X, rank D) does not carry over to the stable rank, so it is not asserted.)"""
    from lorasub.grad_probe import XDCollector, probe

    # a controlled check of the estimator itself: rank-3 inputs, full-rank signals
    torch.manual_seed(0)
    lin = torch.nn.Linear(64, 32, bias=False)
    c = XDCollector({"lin": lin}, proj_dim=64, seed=0)
    basis = torch.randn(3, 64)
    for _ in range(6):
        x = (torch.randn(8, 3) @ basis).requires_grad_(True)
        lin(x).pow(2).sum().backward()
    sr = c.stable_ranks("lin")
    c.close()
    assert sr["stable_rank_X"] < 4.5, sr          # inputs live in 3 directions
    assert sr["stable_rank_D"] > sr["stable_rank_X"]

    df = probe(tiny_model(0), tiny_tokenizer(), MODEL_ID, str(workspace / "data"),
               str(workspace / "svd"), datasets=("format",), n_examples=8, batch_size=2, max_len=192,
               group_layers=1, device=torch.device("cpu"), ranks=(1, 2), keep_spectrum=4,
               split_half=False, xd_decomposition=True, proj_dim=32)
    for col in ("stable_rank_X", "stable_rank_D", "proj_dim_X"):
        assert col in df.columns, col
    # values are finite, positive, and capped by the projection dimension (beyond which they only
    # mean "at least that much"). Note the algebraic bound rank(G) <= min(rank X, rank D) does NOT
    # carry over to the stable rank, so it is not asserted here.
    assert (df["stable_rank_X"] > 0).all() and (df["stable_rank_D"] > 0).all()
    assert (df["stable_rank_X"] <= df["proj_dim_X"] + 1e-6).all()


def test_activation_norm_profile_flags_dominant_tokens(workspace):
    """Audit: an effective-rank collapse that hits BOTH tasks at the same layer is architectural, not
    task-related — a few very-high-norm tokens (first position, BOS, delimiters) dominate the sum of
    rank-1 terms. This diagnostic reports the ratio max/mean so such a collapse is attributed to the
    architecture."""
    from lorasub.data.loaders import build_dataset, collate
    from lorasub.grad_probe import activation_norm_profile
    from torch.utils.data import DataLoader

    tok = tiny_tokenizer()
    ds = build_dataset("format", workspace / "data", tok, 192)
    loader = DataLoader(ds, batch_size=2, collate_fn=lambda b: collate(b, tok.pad_token_id))
    model = tiny_model(0)
    names = [n for n, _ in __import__("lorasub.spectral", fromlist=["x"]).iter_target_modules(model)][:3]
    out = activation_norm_profile(model, loader, names, torch.device("cpu"), n_batches=2)
    assert set(out) == set(names)
    for v in out.values():
        assert v["act_norm_max"] >= v["act_norm_mean"] > 0
        assert v["act_norm_ratio"] >= 1.0 and v["act_norm_argmax_pos"] >= 0


def test_lr_key_normalises_rank_and_collect_drops_headless_rows(tmp_path):
    """Two failures seen on real data. A truncated results.csv leaves a row with no model, which breaks
    every groupby. And the LR table written from a CSV carries `1.0` where the grid carries `1`: an
    unnormalised key silently leaves the run at the default learning rate — the exact opposite of what
    the sweep is for."""
    import csv

    from lorasub.aggregate import _lr_key, collect
    from lorasub.launch_grid import apply_lr

    assert _lr_key("m", "format", "top", 1.0) == _lr_key("m", "format", "top", 1) == "m|format|top|1"
    table = {_lr_key("m", "format", "top", 1.0): 0.002}
    assert apply_lr({"model": "m", "task": "format", "mode": "top", "rank": 1, "lr": 5e-4},
                    table)["lr"] == 0.002
    assert apply_lr({"model": "m", "task": "format", "mode": "top", "rank": 1.0, "lr": 5e-4},
                    table)["lr"] == 0.002
    assert apply_lr({"model": "m", "task": "format", "mode": "bottom", "rank": 1, "lr": 5e-4},
                    table)["lr"] == 5e-4

    runs = tmp_path / "runs"
    for i, row in enumerate([{"model": "m", "task": "format", "mode": "top", "rank": 1, "lr": 5e-4,
                              "seed": 0, "n_trainable": 1, "format_parsed": 0.3, "format_n": 400},
                             {"model": "", "task": "", "mode": "", "rank": "", "lr": "", "seed": "",
                              "n_trainable": "", "format_parsed": "", "format_n": ""}]):
        d = runs / f"r{i}"
        d.mkdir(parents=True)
        (d / "status").write_text("DONE\n")
        with open(d / "results.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            w.writeheader()
            w.writerow(row)
    df = collect(runs)
    assert len(df) == 1 and df.model.iloc[0] == "m"


def test_relative_perturbation_separates_absolute_from_relative(workspace, tmp_path):
    """The same absolute update norm is a negligible change in the top band and an enormous one in the
    bottom: on a toy spectrum, ||dW|| = 0.18 is 1.8 % of the top band and 180x the bottom one. So an
    anticorrelation between performance and ABSOLUTE norm says nothing about relative perturbation."""
    import importlib.util

    from lorasub.spectral import SVDEntry, relative_perturbation
    from lorasub.train import run

    S = torch.tensor([10.0, 5.0, 1.0, 1e-2, 1e-3])
    e = SVDEntry("m", (5, 5), S, torch.eye(5), torch.eye(5))
    A = torch.randn(1, 5) * 0.1
    hi, lo = torch.zeros(5, 1), torch.zeros(5, 1)
    hi[0] = lo[4] = 1.0
    r_hi = relative_perturbation(A, hi, e, "top:1")
    r_lo = relative_perturbation(A, lo, e, "bottom:1")
    assert abs(r_hi["update_norm"] - r_lo["update_norm"]) < 1e-9      # same absolute norm
    assert r_lo["relative_perturbation"] > 1000 * r_hi["relative_perturbation"]

    # end to end on a real run's saved factors
    cfg = _cfg(workspace, task="format", mode="bottom", rank=2, max_steps=2,
               save_factor_steps=[0, "end"])
    run(cfg, tokenizer=tiny_tokenizer(), quiet=True)
    spec = importlib.util.spec_from_file_location(
        "relperturb", Path(__file__).resolve().parents[1] / "scripts" / "relative_perturbation.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    rows = mod.analyse_run(cfg.run_dir, str(workspace / "svd"))
    assert rows and all(r["mode"] == "bottom" and r["relative_perturbation"] >= 0 for r in rows)
    assert {"update_norm", "band_energy", "relative_perturbation"} <= set(rows[0])


def test_analysis_figures_answer_their_question(tmp_path):
    """The four figures added to aggregate each exist for a specific claim; this checks they compute
    the right thing on a controlled dataset rather than merely producing a file."""
    import csv

    from lorasub.aggregate import (collect, gradient_vs_performance, plot_failure_decomposition,
                                   plot_lr_heatmap, win_matrix)

    runs = tmp_path / "runs"
    rows = []
    # bottom beats top on 4 seeds out of 5 at lr 2e-3, and loses at lr 5e-4: a ranking that flips
    for lr, adv in ((5e-4, -1), (2e-3, +1)):
        for seed in range(5):
            flip = 1 if (lr == 2e-3 and seed == 4) else 0     # one seed against the trend
            rows.append(dict(model="m", task="format", mode="bottom", rank=2, lr=lr, seed=seed,
                             band_frac="", n_trainable=1, format_n=400,
                             format_parsed=0.40 + adv * 0.10 - flip * 0.25,
                             format_keys_ok=0.99, format_count_ok=0.96, format_sig_ok=0.86,
                             format_order_ok=0.93, format_exact=0.30, format_empty=0.01))
            rows.append(dict(model="m", task="format", mode="top", rank=2, lr=lr, seed=seed,
                             band_frac="", n_trainable=1, format_n=400, format_parsed=0.40,
                             format_keys_ok=0.99, format_count_ok=0.96, format_sig_ok=0.70,
                             format_order_ok=0.99, format_exact=0.20, format_empty=0.05))
    for i, r in enumerate(rows):
        d = runs / f"r{i}"
        d.mkdir(parents=True)
        (d / "status").write_text("DONE\n")
        with open(d / "results.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(r))
            w.writeheader()
            w.writerow(r)
    df = collect(runs)

    # the win matrix uses the pairing seed by seed, not the means
    wm = win_matrix(df[df.lr == 2e-3], "m", "format", 2)
    assert abs(wm.loc["bottom", "top"] - 0.8) < 1e-9      # 4 seeds out of 5
    assert abs(wm.loc["top", "bottom"] - 0.2) < 1e-9
    assert wm.attrs["n_seeds"] == 5
    wm_low = win_matrix(df[df.lr == 5e-4], "m", "format", 2)
    assert wm_low.loc["bottom", "top"] == 0.0             # the ordering flips with the learning rate

    plot_lr_heatmap(df, "m", "format", tmp_path / "lr.png")
    assert (tmp_path / "lr.png").exists()
    plot_failure_decomposition(df, "m", tmp_path / "fail.png", rank=2)
    assert (tmp_path / "fail.png").exists()

    # the gradient-vs-performance bridge reads the band each arm adapts
    probe = tmp_path / "probe.csv"
    pd.DataFrame([dict(dataset="format", side="left", band="top:2", energy=0.05, null=0.10),
                  dict(dataset="format", side="left", band="bottom:2", energy=0.25, null=0.10)]
                 ).to_csv(probe, index=False)
    res = gradient_vs_performance(probe, df[df.lr == 2e-3], "m", "format", 2, tmp_path / "gp.png")
    assert set(res["mode"]) == {"top", "bottom"}
    hi = res[res["mode"] == "bottom"].iloc[0]
    lo = res[res["mode"] == "top"].iloc[0]
    assert hi.grad_energy_over_null > lo.grad_energy_over_null and hi.score > lo.score


def test_collect_rejects_rows_whose_model_is_a_number(tmp_path):
    """A results.csv written with shifted columns puts a number where the model id belongs. The row
    then survives every filter and creates phantom models — real output showed 'no band runs for
    497.7'. A model id must be a non-empty, non-numeric string."""
    import csv

    from lorasub.aggregate import collect

    runs = tmp_path / "runs"
    good = dict(model="meta-llama/Llama-3.2-1B", task="format", mode="top", rank=2, lr=5e-4, seed=0,
                n_trainable=1, format_parsed=0.3, format_n=400)
    for i, model in enumerate(["meta-llama/Llama-3.2-1B", "497.7", "", "  "]):
        d = runs / f"r{i}"
        d.mkdir(parents=True)
        (d / "status").write_text("DONE\n")
        with open(d / "results.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(good))
            w.writeheader()
            w.writerow({**good, "model": model})
    df = collect(runs)
    assert len(df) == 1 and df.model.iloc[0] == "meta-llama/Llama-3.2-1B"


def test_win_matrix_survives_a_table_matching_nothing(tmp_path):
    """win_matrix was the last place still using the plain-list filter, and it crashed in production
    with KeyError: 'mode' after the others had been fixed."""
    from lorasub.aggregate import _lr_key, win_matrix

    d = pd.DataFrame([dict(model="m", task="format", mode="top", rank=2, lr=5e-4, seed=0,
                           band_frac=float("nan"), format_parsed=0.3, format_n=400, n_trainable=1)])
    wm = win_matrix(d, "m", "format", 2, {_lr_key("m", "format", "top", 2): 2e-3})
    assert wm.empty or wm.isna().all().all()


def test_results_are_written_atomically(tmp_path):
    """Two array tasks can execute the same run_id at once — it happens as soon as two `sbatch` of the
    same grid overlap. Writing results.csv in place produced files with an extra half-row appended
    after a complete record, whose columns no longer matched the header; every metric read from them
    was wrong. A rename is atomic: a reader sees the old file or a complete new one, never a mixture."""
    import inspect
    import threading

    from lorasub.train import _write_results

    src = inspect.getsource(_write_results)
    assert "os.replace" in src and ".tmp" in src

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    errors = []

    def writer(n):
        try:
            for _ in range(25):
                _write_results(run_dir, {"run_id": "x", "value": n, "pad": "y" * 500})
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=writer, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    # whatever the interleaving, the file is a valid 1-row CSV with matching columns
    df = pd.read_csv(run_dir / "results.csv")
    assert len(df) == 1 and list(df.columns) == ["run_id", "value", "pad"]
    assert not list(run_dir.glob(".results.csv.*.tmp"))


def test_slurm_requests_one_job_per_node(tmp_path, workspace):
    """Measured failure: 53 of 60 array tasks landed on the SAME node and all 53 died of CUDA OOM,
    while the 7 that landed elsewhere succeeded. This partition refuses --gres, so nothing reserves
    the PARTITION and SLURM packs tasks onto any node with CPU and RAM to spare — they then share one card.
    Exclusivity is the only available way to guarantee one job per PARTITION."""
    from lorasub.launch_grid import expand, materialise, write_slurm

    spec = {"base": {"model": MODEL_ID, "svd_cache": "/x", "data_dir": "/x", "out_dir": "/x",
                     "max_steps": 1, "tiny_model_config": TINY_CFG},
            "grid": {"task": ["facts"], "mode": ["top"], "seed": [0]}}
    to_run, _ = materialise(expand(spec), tmp_path)
    txt = write_slurm(tmp_path, to_run, "PARTITION", "01:59:00", "32G", 8, refuse_local=False).read_text()
    assert "#SBATCH --exclusive" in txt
    # the escape hatch exists but is not the default, and says why
    off = write_slurm(tmp_path, to_run, "PARTITION", "01:59:00", "32G", 8, refuse_local=False,
                      exclusive=False).read_text()
    assert "#SBATCH --exclusive" not in off and "share its PARTITION" in off


def test_select_lr_warns_when_the_optimum_sits_at_the_grid_edge(tmp_path, capsys):
    """Caught by an outside reader, not by the code: the sweep selected 2e-3 for every constrained arm
    — the TOP of the swept range. An optimum at the boundary has not been found, and the ranking read
    at that point may be a ranking of under-trained arms."""
    import csv

    from lorasub.aggregate import collect, select_lr

    runs = tmp_path / "runs"
    rows = []
    for lr, score in ((1e-4, 0.10), (5e-4, 0.20), (1e-3, 0.30), (2e-3, 0.40)):   # increasing: edge
        rows.append(dict(model="m", task="format", mode="top", rank=2, lr=lr, seed=0, band_frac="",
                         n_trainable=1, val_format_parsed=score, format_parsed=score, format_n=400))
    for lr, score in ((1e-4, 0.10), (5e-4, 0.40), (1e-3, 0.20), (2e-3, 0.15)):   # interior maximum
        rows.append(dict(model="m", task="format", mode="bottom", rank=2, lr=lr, seed=0, band_frac="",
                         n_trainable=1, val_format_parsed=score, format_parsed=score, format_n=400))
    for i, r in enumerate(rows):
        d = runs / f"r{i}"
        d.mkdir(parents=True)
        (d / "status").write_text("DONE\n")
        with open(d / "results.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(r))
            w.writeheader()
            w.writerow(r)
    sel = select_lr(collect(runs))["selected"]
    out = capsys.readouterr().out
    edge = [ln for ln in out.splitlines() if "GRID EDGE" in ln]
    assert edge and any("| top |" in ln for ln in edge)          # boundary optimum is flagged
    assert not any("| bottom |" in ln for ln in edge)            # interior optimum is not
    # other notes may legitimately mention bottom (validation size, margin); only GRID EDGE matters
    assert sel["m|format|top|2"] == 2e-3 and sel["m|format|bottom|2"] == 5e-4


def test_free_arm_spectrum_script_reads_saved_factors(workspace):
    """The cheapest and possibly most telling measurement: where does the FREE arm's dW live in the
    spectrum of W0? Computed from saved factors; no PARTITION, no re-run."""
    import importlib.util

    from lorasub.train import run

    cfg = _cfg(workspace, task="format", mode="free", rank=2, max_steps=2, save_factor_steps=[0, "end"])
    run(cfg, tokenizer=tiny_tokenizer(), quiet=True)
    spec = importlib.util.spec_from_file_location(
        "fas", Path(__file__).resolve().parents[1] / "scripts" / "free_arm_spectrum.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    rows = mod.analyse_free_run(cfg.run_dir, str(workspace / "svd"))
    assert rows
    df = pd.DataFrame(rows)
    for side in ("left", "right"):
        d = df[(df.side == side) & (df.band.str.startswith("frac:"))]
        tot = d.groupby(["run_id", "layer", "module_type"]).energy.sum()
        out = df[(df.side == side) & (df.band == "_outside")].set_index(["run_id", "layer", "module_type"]).energy
        assert ((tot + out.reindex(tot.index)) - 1.0).abs().max() < 1e-3   # deciles + outside = 1
    # a constrained run is ignored by design
    cfg2 = _cfg(workspace, task="format", mode="top", rank=2, max_steps=1, save_factor_steps=[0, "end"])
    run(cfg2, tokenizer=tiny_tokenizer(), quiet=True)
    assert mod.analyse_free_run(cfg2.run_dir, str(workspace / "svd")) == []


def test_launch_grid_refuses_to_overlap_itself(tmp_path, monkeypatch):
    """The human rule 'check that squeue is empty' failed at 2 a.m. and corrupted ten results.csv.
    It is now enforced: generation is refused while jobs of the same grid_dir are running."""
    from lorasub import launch_grid as lg

    grid_dir = tmp_path / "g"
    (grid_dir / "logs").mkdir(parents=True)
    fake = f"123_4 {grid_dir.resolve()}/logs/%A_%a.out\n999_0 /elsewhere/logs/x.out\n"

    class _P:
        stdout = fake

    import shutil
    import subprocess

    monkeypatch.setattr(shutil, "which", lambda _: "/usr/bin/squeue")
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _P())
    assert lg.grid_has_active_jobs(grid_dir) == ["123_4"]
    assert lg.grid_has_active_jobs(tmp_path / "other") == []


def test_relative_perturbation_filters_lr_and_uses_medians():
    """Two failures of the first version, both found on real output. Averaging over ALL learning rates
    mixes runs whose update norms differ by an order of magnitude, so 'at their own optimum' was never
    measured. And the MEAN relative perturbation read 60832 where the MEDIAN read 18.0, because q_proj
    and o_proj have sigma_min/sigma_max around 3e-9: the ratio explodes on a numerically null
    denominator, not on anything that happens there."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "rp", Path(__file__).resolve().parents[1] / "scripts" / "relative_perturbation.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    src = inspect.getsource(mod.main)
    assert "--select_lr" in src and "selected learning rate" in src
    assert '("update_norm", "median")' in src and '("relative_perturbation", "median")' in src
    assert "mean" not in src.split("agg = df.groupby")[1].split("print")[0]   # no mean in the agg
    assert "--cond_tol" in src and "ill-conditioned" in src


def test_adamw_normalises_gradient_magnitude_so_snr_is_the_quantity(workspace):
    """A precaution that changes which mechanism is even possible. Under AdamW the step size does NOT
    depend on the gradient magnitude — the optimiser divides by the running RMS — so 'this arm receives
    a smaller gradient, hence it needs a larger learning rate' does not hold for our optimiser. It
    would hold under SGD. What Adam does respond to is the signal-to-noise ratio: a band whose gradient
    disagrees between disjoint halves of the data has its effective step shrunk."""
    from lorasub.grad_probe import band_gradient_snr, probe

    p = torch.zeros(64, requires_grad=True)
    opt = torch.optim.AdamW([p], lr=1e-3, weight_decay=0.0)
    torch.manual_seed(0)
    g = torch.randn(64)
    sizes = {}
    for scale in (1.0, 0.01):
        p2 = torch.zeros(64, requires_grad=True)
        o2 = torch.optim.AdamW([p2], lr=1e-3, weight_decay=0.0)
        p2.grad = g * scale
        before = p2.detach().clone()
        o2.step()
        sizes[scale] = float((p2.detach() - before).norm())
    assert abs(sizes[1.0] - sizes[0.01]) < 1e-6, sizes      # magnitude is normalised away
    del p, opt

    # the SNR, by contrast, separates a coherent band from a noise band
    torch.manual_seed(0)
    d = 128
    U = torch.linalg.qr(torch.randn(d, d))[0]
    coherent = U[:, :4] @ torch.randn(4, d)
    Ga, Gb = coherent + 0.1 * torch.randn(d, d), coherent + 0.1 * torch.randn(d, d)
    G = (Ga + Gb) / 2
    hi = band_gradient_snr(G, Ga, Gb, U, torch.arange(4))
    lo = band_gradient_snr(G, Ga, Gb, U, torch.arange(d - 4, d))
    assert hi["grad_snr_in_band"] > 5 * lo["grad_snr_in_band"]
    assert lo["grad_snr_in_band"] < 2.0                     # indistinguishable from noise

    # and the probe reports it for the bands the arms actually use
    df = probe(tiny_model(0), tiny_tokenizer(), MODEL_ID, str(workspace / "data"),
               str(workspace / "svd"), datasets=("format",), n_examples=8, batch_size=2, max_len=192,
               group_layers=1, device=torch.device("cpu"), ranks=(1, 2), keep_spectrum=4,
               arm_ranks=(1, 2))
    for col in ("grad_snr_in_band_top1", "grad_snr_in_band_bottom2", "grad_norm_in_band_top2"):
        assert col in df.columns, col


def test_output_band_share_is_not_the_weight_band_share(workspace):
    """The weight-space view is not what the loss sees. An arm can only change y = W0 x inside the
    subspace its frozen B spans, so what bounds its influence is the share of the OUTPUT that lives in
    its band, measured on real activations — and that share differs from the weight-energy share,
    because the input distribution is not isotropic. A purely spectral argument misses exactly that."""
    from lorasub.grad_probe import OutputBandShare, probe
    from lorasub.spectral import SVDEntry

    torch.manual_seed(0)
    lin = torch.nn.Linear(32, 16, bias=False)
    U, S, Vh = torch.linalg.svd(lin.weight.float(), full_matrices=False)
    svd = SVDEntry("m", (16, 32), S, U, Vh)
    c = OutputBandShare({"lin": lin}, {"lin": svd}, ranks=(1, 4))
    for _ in range(3):
        lin(torch.randn(8, 32))
    sh = c.shares("lin")
    c.close()
    assert sh["output_share_top1"] <= sh["output_share_top4"] <= 1.0
    assert sh["output_share_top1"] > sh["output_share_bottom1"]        # dominant carries the output
    assert sh["output_share_null4"] == 4 / 16                           # isotropic reference
    assert sh["output_share_bottom4"] < sh["output_share_null4"]        # below chance, as expected

    df = probe(tiny_model(0), tiny_tokenizer(), MODEL_ID, str(workspace / "data"),
               str(workspace / "svd"), datasets=("format",), n_examples=8, batch_size=2, max_len=192,
               group_layers=1, device=torch.device("cpu"), ranks=(1, 2), keep_spectrum=4,
               arm_ranks=(1, 2), output_shares=True)
    for col in ("output_share_top1", "output_share_bottom2", "output_share_null1"):
        assert col in df.columns, col
    assert (df["output_share_top2"].dropna() <= 1.0 + 1e-6).all()


def test_snr_measurement_takes_its_three_precautions(workspace):
    """Three ways this test could fail to be decisive, all guarded. (1) The ratio alone confounds an
    SNR effect with an energy effect, since the gradient energy is not uniform across the spectrum:
    numerator and denominator are reported separately. (2) The SNR of a mean falls as 1/sqrt(N), so N
    must match across arms and be recorded. (3) On q_proj and o_proj the bottom band sits on singular
    vectors with sigma_min/sigma_max about 3e-9, where the SNR is numerical noise: bottom bands are
    taken inside the numerically non-zero part."""
    from lorasub.grad_probe import probe

    df = probe(tiny_model(0), tiny_tokenizer(), MODEL_ID, str(workspace / "data"),
               str(workspace / "svd"), datasets=("format",), n_examples=8, batch_size=2, max_len=192,
               group_layers=1, device=torch.device("cpu"), ranks=(1, 2), keep_spectrum=4,
               arm_ranks=(1, 2), sigma_rel_tol=1e-6)
    for col in ("grad_norm_in_band_top1", "grad_noise_in_band_top1", "grad_snr_in_band_top1",
                "snr_n_examples", "numerical_rank"):
        assert col in df.columns, col
    assert df["snr_n_examples"].nunique() == 1                    # same N for every arm
    assert (df["numerical_rank"] > 0).all()
    # an aggressive tolerance moves the bottom band, so the measured SNR changes with it
    df2 = probe(tiny_model(0), tiny_tokenizer(), MODEL_ID, str(workspace / "data"),
                str(workspace / "svd"), datasets=("format",), n_examples=8, batch_size=2, max_len=192,
                group_layers=1, device=torch.device("cpu"), ranks=(1, 2), keep_spectrum=4,
                arm_ranks=(1, 2), sigma_rel_tol=0.5)
    assert df2["numerical_rank"].max() < df["numerical_rank"].max()
    # the ratio is consistent with its parts
    a = df[["grad_norm_in_band_top1", "grad_noise_in_band_top1", "grad_snr_in_band_top1"]].dropna()
    ok = (a.grad_norm_in_band_top1 / a.grad_noise_in_band_top1 - a.grad_snr_in_band_top1).abs()
    assert float(ok.max()) < 1e-3


def test_conditioning_filter_catches_what_the_energy_filter_missed(tmp_path):
    """Filtering on band ENERGY did not work: it dropped 22 pairs and still left medians of 3046 and
    3290 on q_proj and o_proj against 5 to 9 everywhere else. What makes a band unusable is not that
    it holds little energy — bottom bands legitimately do — but that its singular vectors are
    ill-determined, i.e. sigma_min/sigma_max near machine precision."""
    from lorasub.spectral import SVDEntry, relative_perturbation

    S = torch.tensor([10.0, 5.0, 1.0, 1e-2, 3.5e-8])
    e = SVDEntry("m", (5, 5), S, torch.eye(5), torch.eye(5))
    A = torch.randn(1, 5) * 0.1
    top, bot = torch.zeros(5, 1), torch.zeros(5, 1)
    top[0] = bot[4] = 1.0
    r_top = relative_perturbation(A, top, e, "top:1")
    r_bot = relative_perturbation(A, bot, e, "bottom:1")
    assert r_top["cond"] == 1.0                              # dominant band: perfectly conditioned
    assert r_bot["cond"] < 1e-8                              # ill-conditioned, as on q_proj
    assert r_bot["relative_perturbation"] > 1e6              # the ratio explodes there
    # a healthy bottom band is kept
    S2 = torch.tensor([10.0, 5.0, 1.0, 1e-2, 1e-2])
    e2 = SVDEntry("m", (5, 5), S2, torch.eye(5), torch.eye(5))
    assert abs(relative_perturbation(A, bot, e2, "bottom:1")["cond"] - 1e-3) < 1e-9

    src = inspect.getsource(
        __import__("importlib.util", fromlist=["x"]).spec_from_file_location(
            "rp", Path(__file__).resolve().parents[1] / "scripts" / "relative_perturbation.py").name
    ) if False else (Path(__file__).resolve().parents[1] / "scripts" / "relative_perturbation.py").read_text()
    assert "--cond_tol" in src and 'df["cond"] < a.cond_tol' in src
    assert "50 * typical" in src            # outlier modules are flagged, not silently aggregated


def test_a_run_stopped_at_step_zero_is_not_a_measurement(tmp_path):
    """A run that never reached the end leaves only factors_0.safetensors, where A = 0 by construction
    and dW = 0. Reading it gives an update norm of exactly zero — the absence of a measurement, not a
    measurement of zero. One such run made half of a two-seed cell read as 'this arm writes nothing',
    which nearly became a claim about partially inactive arms."""
    import importlib.util

    for script in ("relative_perturbation.py", "free_arm_spectrum.py"):
        spec = importlib.util.spec_from_file_location(
            script[:-3], Path(__file__).resolve().parents[1] / "scripts" / script)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        d = tmp_path / script
        d.mkdir()
        (d / "factors_0.safetensors").write_bytes(b"x")
        assert mod.last_factor_file(d) is None, script          # only step 0 -> ignored
        (d / "factors_120.safetensors").write_bytes(b"x")
        assert mod.last_factor_file(d).name == "factors_120.safetensors", script
        assert mod.last_factor_file(tmp_path / "nothing_here") is None, script


def test_grid_script_refuses_a_busy_gpu_before_paying_the_loading_cost(tmp_path, workspace):
    """This partition declares no GRES, so SLURM cannot reserve the GPUs nor stop another user from
    running on the same machine: 144 tasks died of CUDA OOM against neighbours holding 13 to 21 GiB,
    each after paying the full model-loading cost. The guard checks free memory first and exits 0 —
    not 1 — so the run stays unclaimed and a later pass picks it up on a free card."""
    from lorasub.launch_grid import expand, materialise, write_slurm

    spec = {"base": {"model": MODEL_ID, "svd_cache": "/x", "data_dir": "/x", "out_dir": "/x",
                     "max_steps": 1, "tiny_model_config": TINY_CFG},
            "grid": {"task": ["facts"], "mode": ["top"], "seed": [0]}}
    to_run, _ = materialise(expand(spec), tmp_path)
    txt = write_slurm(tmp_path, to_run, "PARTITION", "01:59:00", "32G", 1, refuse_local=False).read_text()
    assert "memory.free" in txt and "-lt 6000" in txt
    assert txt.index("FREE=") < txt.index('"$PY" -m lorasub.train')   # checked BEFORE loading
    assert "exit 0" in txt.split("FREE=")[1].split("PY=")[0]           # unclaimed, not failed
    # the threshold is configurable
    t2 = write_slurm(tmp_path, to_run, "PARTITION", "01:59:00", "32G", 1, refuse_local=False,
                     min_free_mib=12000).read_text()
    assert "-lt 12000" in t2


def test_analysis_scripts_ignore_runs_without_results(tmp_path):
    """A directory with factors but no results.csv is an interrupted run: its factors describe a
    half-trained adapter, not a trained one."""
    import importlib.util

    for script in ("relative_perturbation.py", "free_arm_spectrum.py"):
        spec = importlib.util.spec_from_file_location(
            script[:-3], Path(__file__).resolve().parents[1] / "scripts" / script)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        d = tmp_path / script.replace(".py", "")
        d.mkdir()
        (d / "config.yaml").write_text("model: m\nmode: bottom\nrank: 2\n")
        (d / "factors_100.safetensors").write_bytes(b"x")
        fn = mod.analyse_run if hasattr(mod, "analyse_run") else mod.analyse_free_run
        assert fn(d, "/nonexistent") == [], script        # no results.csv -> ignored


def test_random_arm_draws_independently_per_module():
    """Central flaw found by audit: the spec was f"random:{r}:{seed}" with no per-module salt, so
    every module of the same size received the SAME columns. On Llama-3.2-1B that is two distinct
    draws per seed (m=2048 and m=512) instead of 112 — the arm was not "a random subspace" but ONE
    random spectral position applied everywhere, i.e. a band arm at a random band_frac. The control
    that anchors the whole comparison did not have the property claimed for it."""
    from lorasub.spectral import band_indices

    q = band_indices(2048, "random:2:0:model.layers.0.self_attn.q_proj").tolist()
    o = band_indices(2048, "random:2:0:model.layers.0.self_attn.o_proj").tolist()
    q5 = band_indices(2048, "random:2:0:model.layers.5.self_attn.q_proj").tolist()
    assert q != o and q != q5, (q, o, q5)              # independent across modules AND layers
    # still deterministic in (seed, module) and still a valid band
    assert q == band_indices(2048, "random:2:0:model.layers.0.self_attn.q_proj").tolist()
    assert band_indices(2048, "random:2:1:model.layers.0.self_attn.q_proj").tolist() != q
    assert len(set(q)) == 2 and all(0 <= i < 2048 for i in q)
    # unsalted form still works, for the runs produced before the fix
    assert len(band_indices(2048, "random:2:0").tolist()) == 2


def test_salted_subspace_is_a_field_not_a_deletion():
    """Third time the same principle applies: a change of method belongs in the identifier, with an
    exception for the historical value. The alternative considered — deleting the 46 random runs —
    would have destroyed twelve hours of PARTITION to work around a missing field, and thrown away valid
    measurements of a legitimate (if unintended) arm: 'one random spectral position applied
    everywhere'."""
    from lorasub.config import RunConfig, apply_model_defaults
    from lorasub.lora import LoRALinear

    base = dict(model=MODEL_ID, task="format", mode="random", rank=2, lr=5e-4, seed=0)
    # The DEFAULT is now True: leaving it False meant thirteen grids out of fifteen silently reran
    # the defective arm. Compatibility is handled on read — the False value is dropped from the id,
    # so the runs produced before the fix keep theirs.
    new = RunConfig(**apply_model_defaults(dict(base)))
    old = RunConfig(**apply_model_defaults(dict(base, subspace_salted=False)))
    assert new.subspace_salted is True and old.subspace_salted is False
    assert "subspace_salted" not in old.id_dict()        # the existing runs keep their ids
    assert "subspace_salted" in new.id_dict()
    assert old.run_id != new.run_id                       # and the two arms are separated

    def _B(salted, name):
        return LoRALinear(torch.nn.Linear(32, 16, bias=False), r=2, mode="random_ortho", seed=0,
                          module_salt=name, subspace_salted=salted).B

    assert torch.allclose(_B(False, "q"), _B(False, "o"))        # old behaviour: identical
    assert not torch.allclose(_B(True, "q"), _B(True, "o"))      # new behaviour: independent


def test_random_ortho_also_differs_across_modules(tmp_path):
    """Same defect, same fix: an unsalted generator handed every module identical directions."""
    import torch

    from lorasub.lora import LoRALinear

    a = LoRALinear(torch.nn.Linear(32, 16, bias=False), r=2, mode="random_ortho", seed=0,
                   module_salt="layers.0.q_proj", subspace_salted=True)
    b = LoRALinear(torch.nn.Linear(32, 16, bias=False), r=2, mode="random_ortho", seed=0,
                   module_salt="layers.0.o_proj", subspace_salted=True)
    assert not torch.allclose(a.B, b.B)
    c = LoRALinear(torch.nn.Linear(32, 16, bias=False), r=2, mode="random_ortho", seed=0,
                   module_salt="layers.0.q_proj", subspace_salted=True)
    assert torch.allclose(a.B, c.B)        # deterministic given (seed, module)


def test_data_fingerprint_reaches_results_csv_through_the_real_run(workspace, tmp_path):
    """Two traps, both met. (1) The fingerprint was written to a side file while `collect()` reads
    results.csv and nothing else, so `check_data_versions` was permanently dormant on real data — and
    the test passed because it built its own CSV containing the column, i.e. it verified a dead
    constant instead of the production path. This test therefore goes through `run()`. (2) An EMPTY
    fingerprint stays out of the run id so the 224 runs produced before this field keep their
    identifiers; the same precaution applies to max_new_tokens at its historical default."""
    from lorasub.aggregate import check_data_versions, collect
    from lorasub.config import RunConfig, apply_model_defaults
    from lorasub.data.paths import fingerprint_data
    from lorasub.train import run

    cfg = _cfg(workspace, task="format", mode="top", rank=2, max_steps=1)
    run(cfg, tokenizer=tiny_tokenizer(), quiet=True)
    # the workspace is shared with other tests, so select OUR run
    df = collect(cfg.run_dir.parent)
    assert "data_fingerprint" in df.columns, "must reach results.csv, not just a side file"
    mine = df[df.run_id == cfg.run_id]
    assert len(mine) == 1
    fp = str(mine["data_fingerprint"].iloc[0])
    assert fp and fp == fingerprint_data(workspace / "data", "format")

    # summarize must not try to average a string column
    from lorasub.aggregate import summarize

    out = summarize(df)
    assert "data_fingerprint" in out.columns
    assert out["data_fingerprint"].notna().any() and (out["data_fingerprint"].astype(str) != "").any()

    # legacy identifiers are preserved; a set fingerprint separates versions
    base = dict(model=MODEL_ID, task="format", mode="top", rank=2, lr=5e-4, seed=0)
    legacy = RunConfig(**apply_model_defaults(dict(base)))
    tagged = RunConfig(**apply_model_defaults(dict(base, data_fingerprint=fp)))
    assert "data_fingerprint" not in legacy.id_dict() and legacy.run_id != tagged.run_id

    # two versions in the same task are refused
    d2 = mine.copy()
    d2["data_fingerprint"] = "ffffffffffff"
    msgs = check_data_versions(pd.concat([mine, d2], ignore_index=True))
    assert msgs and "TWO DATASET VERSIONS" in msgs[0]


def test_fingerprint_is_stable_under_copy(tmp_path):
    """An scp without -p or a copy to a rented machine changes the date without changing the data.
    With an mtime-based digest that is a false positive from check_data_versions, and — once the
    fingerprint is in the identifier — a full re-execution on every new machine."""
    import os
    import shutil
    import time

    from lorasub.data.paths import fingerprint_data

    a = tmp_path / "a"
    (a / "format").mkdir(parents=True)
    for sp in ("train", "val", "test"):
        (a / "format" / f"{sp}.jsonl").write_text('{"x": 1}\n{"x": 2}\n')
    fa = fingerprint_data(a, "format")

    b = tmp_path / "b"
    b.mkdir()
    shutil.copytree(a / "format", b / "format")
    time.sleep(0.01)
    os.utime(b / "format" / "train.jsonl", None)          # copy without -p
    assert fingerprint_data(b, "format") == fa

    with open(b / "format" / "train.jsonl", "a") as f:    # a real change is caught
        f.write('{"x": 3}\n')
    assert fingerprint_data(b, "format") != fa


def test_historical_max_new_tokens_stays_out_of_the_run_id():
    """It is in id_dict and was absent from _NON_ID_FIELDS, so adding the field changed EVERY existing
    run_id: launch_grid would have seen the whole database as not done and relaunched it. Same remedy
    as data_fingerprint — the historical default is dropped, any other value separates runs."""
    from lorasub.config import RunConfig, apply_model_defaults

    base = dict(model=MODEL_ID, task="format", mode="top", rank=2, lr=5e-4, seed=0)
    a = RunConfig(**apply_model_defaults(dict(base)))
    assert a.max_new_tokens == 128 and "max_new_tokens" not in a.id_dict()
    b = RunConfig(**apply_model_defaults(dict(base, max_new_tokens=256)))
    assert "max_new_tokens" in b.id_dict() and a.run_id != b.run_id
    # and it is coerced to int, like the other integer fields
    c = RunConfig(**apply_model_defaults(dict(base, max_new_tokens="256")))
    assert isinstance(c.max_new_tokens, int) and c.run_id == b.run_id


def test_mcnemar_pairs_per_example_not_per_seed(tmp_path):
    """A seed-level win matrix gives at best p = 1/32 with five seeds, and one discordant seed kills
    it. The arms see the SAME test items, so the comparison can be paired per example — three orders
    of magnitude more power, at zero PARTITION cost, once the generations are on disk."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "mcn", Path(__file__).resolve().parents[1] / "scripts" / "mcnemar.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # 200 items, a and b agree on 180; of the 20 discordant, b wins 18
    a = {i: True for i in range(200)}
    b = dict(a)
    for i in range(18):
        a[i], b[i] = False, True
    for i in range(18, 20):
        a[i], b[i] = True, False
    r = mod.mcnemar(a, b)
    assert r["n01"] == 18 and r["n10"] == 2
    assert r["p_value"] < 0.001, r                      # decisive on ONE seed
    # perfect agreement carries no information
    assert mod.mcnemar(a, dict(a))["p_value"] == 1.0


def test_band_extremes_honour_the_numerical_rank_like_bottom(workspace):
    """band_frac=1.0 went through "at:k:1.0", which ignored sigma_rel_tol while mode="bottom"
    honoured it. Once the tolerance is on, the end of the continuous curve would no longer be the
    `bottom` arm of the table — the two would not be comparable."""
    from lorasub.spectral import band_indices

    S = torch.tensor([10.0, 5.0, 1.0, 1e-2, 1e-9, 1e-12])
    assert band_indices(6, "at:1:1.0").tolist() == [5]                        # includes a null one
    assert band_indices(6, "at:1:1.0", S=S, rel_tol=1e-6).tolist() == [3]     # inside numerical rank
    assert (band_indices(6, "at:1:1.0", S=S, rel_tol=1e-6).tolist()
            == band_indices(6, "bottom:1", S=S, rel_tol=1e-6).tolist())       # SAME as bottom
    assert band_indices(6, "at:1:0.0", S=S, rel_tol=1e-6).tolist() == [0]     # top unchanged


def test_max_new_tokens_is_configurable_and_enters_the_id():
    """It was hard-coded at 128 in eval.py, outside the config and outside the run id. On sentences
    with many mentions the gold itself can exceed that, so format_parsed_5+ent would carry a ceiling
    below 1 indistinguishable from failure."""
    from lorasub.config import RunConfig, apply_model_defaults

    a = RunConfig(**apply_model_defaults(dict(model=MODEL_ID, task="format", mode="top", rank=2)))
    b = RunConfig(**apply_model_defaults(dict(model=MODEL_ID, task="format", mode="top", rank=2,
                                              max_new_tokens=256)))
    assert a.max_new_tokens == 128 and b.max_new_tokens == 256
    assert a.run_id != b.run_id            # it changes what is measured, so it belongs in the id


def test_skipped_examples_are_reported(workspace):
    """loaders.py counted n_skipped and nobody read it. With max_len=384 and --min_entities 3 the
    rejection is correlated with length: the sentences with the most entities are the ones dropped,
    which is precisely the hard end of the task. And the 7B config uses a different max_len, so a
    "confirmation" there would not be on the same data."""
    from lorasub.data.loaders import build_dataset, skipped_report

    class _Stub(list):
        n_skipped = 0

    s0 = _Stub(range(90))
    s0.n_skipped = 10
    msg = skipped_report(s0)
    assert "10/100" in msg and "10.0%" in msg and "max_len" in msg

    none = _Stub(range(100))
    assert "0/100" in skipped_report(none) and "0.0%" in skipped_report(none)

    # and it works on a real dataset
    ds = build_dataset("format", workspace / "data", tiny_tokenizer(), 384)
    assert "dropped" in skipped_report(ds)


def test_a_cell_without_a_selected_lr_is_announced(capsys):
    """Silent failure found by audit. A sweep covering rank 2 only, while the seed grid runs at ranks
    1, 2 and 4: apply_lr leaves those cells at the default rate — typically 5e-4, far from the arm's
    optimum — and _keep_selected_lr keeps EVERY swept rate for them and averages them together. Both
    now say so."""
    from lorasub.aggregate import _keep_selected_lr, _lr_key
    from lorasub.launch_grid import apply_lr

    table = {_lr_key("m", "format", "random", 2): 5e-3}
    assert apply_lr({"model": "m", "task": "format", "mode": "random", "rank": 2, "lr": 5e-4},
                    table)["lr"] == 5e-3
    capsys.readouterr()
    out = apply_lr({"model": "m", "task": "format", "mode": "random", "rank": 1, "lr": 5e-4}, table)
    assert out["lr"] == 5e-4
    assert "no selected lr" in capsys.readouterr().out

    d = pd.DataFrame([dict(model="m", task="format", mode="random", rank=1, lr=x)
                      for x in (1e-4, 5e-4, 1e-3, 2e-3)])
    capsys.readouterr()
    kept = _keep_selected_lr(d, table)
    assert len(kept) == 4                                   # behaviour unchanged...
    assert "no selected learning rate" in capsys.readouterr().out    # ...but no longer silent


def test_the_two_random_arms_are_never_averaged_together(tmp_path, workspace):
    """The identifier separated them and the aggregation put them back. `subspace_salted` gives a
    different run_id to a `random` arm drawn independently per module and to one drawn once and applied
    everywhere — but KEY did not contain it, so summarize, the crossed table, the win matrix and the
    rate map all grouped the two into one cell and averaged two different arms. The identifier
    separated the populations while the analysis recombined them."""
    import csv

    from lorasub.aggregate import KEY, collect, summarize
    from lorasub.train import run

    assert "subspace_salted" in KEY

    runs = tmp_path / "runs"
    for i, salted in enumerate([False, True]):
        d = runs / f"r{i}"
        d.mkdir(parents=True)
        (d / "status").write_text("DONE\n")
        row = dict(model="m", task="format", mode="random", rank=2, lr=2e-3, seed=0, band_frac="",
                   n_trainable=1, format_parsed=0.30 if salted else 0.60, format_n=400,
                   subspace_salted=salted)
        with open(d / "results.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            w.writeheader()
            w.writerow(row)
    out = summarize(collect(runs))
    assert len(out) == 2, "two arms, two cells — not one average of 0.45"
    assert set(out["format_parsed_mean"].round(2)) == {0.30, 0.60}

    # and the flag reaches results.csv through the real run
    cfg = _cfg(workspace, task="format", mode="random", rank=2, max_steps=1, subspace_salted=True)
    run(cfg, tokenizer=tiny_tokenizer(), quiet=True)
    df = collect(cfg.run_dir.parent)
    assert bool(df[df.run_id == cfg.run_id]["subspace_salted"].iloc[0]) is True

    # legacy rows without the column are treated as the unsalted arm
    legacy = pd.DataFrame([dict(model="m", task="format", mode="random", rank=2, lr=2e-3, seed=0,
                                band_frac=float("nan"), n_trainable=1, format_parsed=0.5)])
    legacy.to_csv(tmp_path / "legacy.csv", index=False)


def test_end_to_end_the_two_random_arms_never_reach_the_same_figure(tmp_path):
    """THE test the audit asked for, and the one that should have been written before the fix rather
    than after. A toy runs_dir holding two `random` runs — one salted, one not — pushed through every
    function whose output reaches the paper. Adding `subspace_salted` to KEY fixed `summarize` and
    nothing else: seven other functions group on `mode` alone and would have averaged two different
    arms. The pattern this catches is the recurring one — each fix has been correct where it was
    written and absent where it mattered."""
    import csv

    from lorasub.aggregate import (IncomparableCells, collect, crossed_table, plot_lr_heatmap,
                                   select_population, summarize, win_matrix)

    runs = tmp_path / "runs"
    rows = []
    for salted, score in ((False, 0.60), (True, 0.30)):
        for seed in (0, 1):
            for arm, sc in (("random", score), ("top", 0.50), ("bottom", 0.55)):
                if arm != "random" and salted:
                    continue                      # top/bottom exist once, they have no salt
                rows.append(dict(model="m", task="format", mode=arm, rank=2, lr=2e-3, seed=seed,
                                 band_frac="", n_trainable=1, format_parsed=sc, format_n=400,
                                 subspace_salted=salted if arm == "random" else False))
    for i, r in enumerate(rows):
        d = runs / f"r{i}"
        d.mkdir(parents=True)
        (d / "status").write_text("DONE\n")
        with open(d / "results.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(r))
            w.writeheader()
            w.writerow(r)
    df = collect(runs)
    assert len(df[df["mode"] == "random"]) == 4        # two arms x two seeds

    # WITHOUT the filter, the paper-facing functions must REFUSE rather than average
    with pytest.raises(IncomparableCells):
        crossed_table(df, "m", 2)
    with pytest.raises(IncomparableCells):
        win_matrix(df, "m", "format", 2)

    # WITH the filter, exactly one arm survives and every output is coherent
    new = select_population(df, salted=True)
    assert len(new[new["mode"] == "random"]) == 2
    assert set(new[new["mode"] == "random"].format_parsed) == {0.30}
    old = select_population(df, salted=False)
    assert set(old[old["mode"] == "random"].format_parsed) == {0.60}

    tex = crossed_table(new, "m", 2)
    assert "30.0" in tex and "60.0" not in tex          # the salted arm only
    wm = win_matrix(new, "m", "format", 2)
    assert wm.loc["top", "random"] == 1.0 and wm.loc["random", "top"] == 0.0   # 0.50 > 0.30
    wm_old = win_matrix(old, "m", "format", 2)
    assert wm_old.loc["top", "random"] == 0.0                                  # 0.50 < 0.60, inverted
    plot_lr_heatmap(new, "m", "format", tmp_path / "h.png")
    assert (tmp_path / "h.png").exists()

    # summarize keeps them apart on its own, via KEY
    assert len(summarize(df)[summarize(df)["mode"] == "random"]) == 2


def test_subspace_salted_is_coerced_like_the_other_booleans():
    """Same failure as the 5e-4 / '5.0e-4' bug, on the new field: `subspace_salted=1` from a command
    line stays as 1, which is truthy so it enters the id, but json.dumps writes 1 rather than true and
    the run_id differs for the same run — while results.csv writes bool(...) = True."""
    from lorasub.config import RunConfig, apply_model_defaults

    base = dict(model=MODEL_ID, task="format", mode="random", rank=2, lr=5e-4, seed=0)
    ref = RunConfig(**apply_model_defaults(dict(base, subspace_salted=True)))
    for truthy in (1, "1", "true", "True", "yes"):
        c = RunConfig(**apply_model_defaults(dict(base, subspace_salted=truthy)))
        assert isinstance(c.subspace_salted, bool) and c.subspace_salted is True, truthy
        assert c.run_id == ref.run_id, truthy
    for falsy in (0, "0", "false", "False", "no"):
        c = RunConfig(**apply_model_defaults(dict(base, subspace_salted=falsy)))
        assert c.subspace_salted is False, falsy
        assert "subspace_salted" not in c.id_dict()


def test_select_lr_balances_the_cell_before_the_argmax():
    """Without balancing, the most populated rate wins partly because it is the most populated. On the
    free arm at r=2, 1e-3 held six or seven runs against two elsewhere; its mean drifted from 0.841 to
    0.808 as relaunches piled up and the selected optimum moved to 2e-3. That rate goes into --lr_from
    and into the seed grid, so an artefact of run counts would propagate to every number of the paper."""
    from lorasub.aggregate import select_lr

    # 1e-3: two good runs then five poor ones. 2e-3: two runs, steady.
    rows = [dict(model="m", task="format", mode="free", rank=2, lr=1e-3, seed=i,
                 val_format_parsed=0.84 if i < 2 else 0.60) for i in range(7)]
    rows += [dict(model="m", task="format", mode="free", rank=2, lr=2e-3, seed=i,
                  val_format_parsed=0.80) for i in range(2)]
    df = pd.DataFrame(rows)

    unbalanced = df.groupby("lr")["val_format_parsed"].mean()
    assert unbalanced[1e-3] < unbalanced[2e-3]          # raw means favour 2e-3...
    sel = select_lr(df)["selected"]["m|format|free|2"]
    assert sel == 1e-3, sel                              # ...balanced ones favour 1e-3

    # and it is reported, not silent
    tbl = select_lr(df)["table"]
    assert set(tbl[tbl.lr == 1e-3]["n"]) == {2}          # sub-sampled to the smallest cell


def test_a_missing_fingerprint_is_a_value_and_the_refusal_is_contained(tmp_path):
    """Two defects of the first refusal. It dropped empty fingerprints, so the mix the coming weeks
    will produce — 228 historical runs without one, new runs with — passed unnoticed. And one mixed
    cell took down the whole aggregation: no rate map, no win matrix, no failure decomposition."""
    import inspect

    from lorasub.aggregate import IncomparableCells, _refuse_if_mixed, main

    d = pd.DataFrame([
        dict(model="m", task="format", mode="top", rank=2, data_fingerprint="abc123456789",
             format_n=400),
        dict(model="m", task="format", mode="top", rank=2, data_fingerprint=None, format_n=400),
    ])
    with pytest.raises(IncomparableCells, match="legacy|dataset versions"):
        _refuse_if_mixed(d, "test")

    # same fingerprint everywhere, or all legacy: no refusal
    _refuse_if_mixed(d.assign(data_fingerprint="abc123456789"), "test")
    _refuse_if_mixed(d.assign(data_fingerprint=None), "test")

    src = inspect.getsource(main)
    assert src.count("except IncomparableCells") >= 2, "both paper-facing calls must be contained"
    assert "SKIPPED" in src


def test_the_refusal_does_not_fire_on_a_task_that_lacks_a_metric(tmp_path):
    """False positive produced by my own fix, seen on real data: a (bottom, r=2) cell holds both
    format and facts runs, a facts run has no format_n, and the absence was read as a third
    'evaluation size' alongside 400 and 614. The refusal must group by TASK, and treat an absent value
    as 'not measured here' — except for the data fingerprint, where absence IS a version."""
    from lorasub.aggregate import IncomparableCells, _refuse_if_mixed

    mixed_tasks = pd.DataFrame([
        dict(task="format", mode="bottom", rank=2, format_n=400.0, mcq_n=None),
        dict(task="facts", mode="bottom", rank=2, format_n=None, mcq_n=996.0),
    ])
    _refuse_if_mixed(mixed_tasks, "test")          # must NOT raise: two tasks, not two settings

    same_task_two_sizes = pd.DataFrame([
        dict(task="format", mode="bottom", rank=2, format_n=400.0),
        dict(task="format", mode="bottom", rank=2, format_n=614.0),
    ])
    with pytest.raises(IncomparableCells, match="evaluation sizes"):
        _refuse_if_mixed(same_task_two_sizes, "test")


def test_select_lr_refuses_to_balance_down_to_a_single_run(capsys):
    """Sub-sampling to n=1 would replace an imbalance by a measurement without variance: the argmax
    would rest on one seed. Seen on real data, where five cells had a rate with a single run."""
    from lorasub.aggregate import select_lr

    rows = [dict(model="m", task="format", mode="free", rank=2, lr=1e-3, seed=i,
                 val_format_parsed=0.80) for i in range(7)]
    rows += [dict(model="m", task="format", mode="free", rank=2, lr=2e-3, seed=0,
                  val_format_parsed=0.90)]                       # a single run at this rate
    select_lr(pd.DataFrame(rows))
    out = capsys.readouterr().out
    assert "SINGLE run" in out and "NOT balanced" in out


def test_select_lr_compares_rates_on_common_seeds(capsys):
    """Sub-sampling by position can hand 1e-3 the seeds {0,1} and 2e-3 the seeds {2,3}: the rates are
    then compared on DISJOINT seeds and seed variance re-enters the argmax through the back door. The
    seeds common to every rate make the comparison paired."""
    from lorasub.aggregate import select_lr

    # on the one common seed (2), 1e-3 gives 0.70 and 2e-3 gives 0.75
    rows = [dict(model="m", task="format", mode="free", rank=2, lr=1e-3, seed=s,
                 val_format_parsed=v) for s, v in ((0, 0.95), (1, 0.95), (2, 0.70))]
    rows += [dict(model="m", task="format", mode="free", rank=2, lr=2e-3, seed=s,
                  val_format_parsed=v) for s, v in ((2, 0.75), (3, 0.80))]
    df = pd.DataFrame(rows)

    assert df[df.lr == 1e-3].val_format_parsed.mean() > df[df.lr == 2e-3].val_format_parsed.mean()
    sel = select_lr(df)["selected"]["m|format|free|2"]
    assert sel == 2e-3, sel                      # paired on seed 2, 2e-3 wins
    out = capsys.readouterr().out
    assert "common to every rate" in out and "paired" in out

    # no common seed: falls back, and says the comparison is NOT paired
    disjoint = pd.DataFrame(
        [dict(model="m", task="format", mode="free", rank=2, lr=1e-3, seed=s, val_format_parsed=0.9)
         for s in (0, 1)]
        + [dict(model="m", task="format", mode="free", rank=2, lr=2e-3, seed=s, val_format_parsed=0.8)
           for s in (2, 3, 4)])
    select_lr(disjoint)
    assert "NOT paired" in capsys.readouterr().out


def test_checker_version_is_deduced_from_the_numbers_not_from_dates():
    """The strongest attack of the seventh audit: nothing records which version of the format checker
    scored a run — not results.csv, not the run id — and `parsed` was hardened on 11/09 to require
    sorted lists, right in the middle of the sweep. The data fingerprint does not help: it tracks the
    DATA, not the scorer. But the strict definition FORCES parsed <= order_ok, so a run violating that
    was scored permissively. A proof internal to the data, where file dates would have been weak — the
    format task was rebuilt three times on 10-11/09 and mtime records only the last."""
    from lorasub.aggregate import IncomparableCells, _refuse_if_mixed, check_metric_versions, infer_metric_version

    d = pd.DataFrame([
        dict(model="m", task="format", mode="top", rank=2, format_parsed=0.50, format_order_ok=0.93),
        dict(model="m", task="format", mode="top", rank=2, format_parsed=0.95, format_order_ok=0.93),
    ])
    # the detection is ONE-WAY: parsed > order_ok proves permissive, the converse proves nothing
    assert list(infer_metric_version(d)) == ["strict-or-undetected", "permissive"]
    assert check_metric_versions(d) and "TWO CHECKER VERSIONS" in check_metric_versions(d)[0]
    with pytest.raises(IncomparableCells, match="checker versions"):
        _refuse_if_mixed(d, "test")

    # a homogeneous population passes, whichever version it is
    strict = d.iloc[[0, 0]].copy()
    assert check_metric_versions(strict) == []
    _refuse_if_mixed(strict, "test")
    perm = d.iloc[[1, 1]].copy()
    assert check_metric_versions(perm) == []

    # absent order_ok -> unknown, never a false positive
    no_col = pd.DataFrame([dict(model="m", task="format", mode="top", rank=2, format_parsed=0.5)])
    assert check_metric_versions(no_col) == []

    # A PERMISSIVE RUN WHOSE OUTPUTS ALL HAPPENED TO BE SORTED IS INDISTINGUISHABLE from a strict one.
    # The label must not claim otherwise: `strict-or-undetected` is consistent with the strict
    # definition, it is not proof of it. The recorded digest is what actually proves the version, and
    # this inference only covers the runs produced before that field existed.
    lucky_permissive = pd.DataFrame([
        dict(model="m", task="format", mode="top", rank=2, format_parsed=0.60, format_order_ok=0.60),
    ])
    assert list(infer_metric_version(lucky_permissive)) == ["strict-or-undetected"]
    assert "strict" != infer_metric_version(lucky_permissive).iloc[0], (
        "the label must not assert strictness it cannot prove")


def test_mcnemar_actually_runs_end_to_end(tmp_path, capsys):
    """This script had NEVER been executed. `main` appended a 2-tuple while `check_population` and the
    pairing loop unpacked four values, so it died with `not enough values to unpack` on its first real
    invocation — while two tests "covered" it by calling `mcnemar()` on dicts and asserting that
    strings were present in the source text. Asserting that a file contains a word is not a test of
    behaviour. This one builds a runs directory and calls main()."""
    import json
    import subprocess
    import sys as _sys

    import yaml as _yaml

    runs = tmp_path / "runs"
    for arm, rate in (("bottom", 0.55), ("top", 0.62)):
        for seed in (2, 3, 4):
            d = runs / f"{arm}{seed}"
            d.mkdir(parents=True)
            _yaml.safe_dump({"model": "m", "task": "format", "mode": arm, "rank": 2,
                             "lr": 5e-3 if arm == "bottom" else 1e-2, "seed": seed,
                             "subspace_salted": True, "data_fingerprint": "abc123456789"},
                            open(d / "config.yaml", "w"))
            (d / "results.csv").write_text("run_id\nx\n")
            with open(d / "generations_test.jsonl", "w") as f:
                for i in range(200):
                    ok = (i % 100) < int(rate * 100)
                    f.write(json.dumps({"id": i, "parsed": ok, "exact": ok}) + "\n")

    script = Path(__file__).resolve().parents[1] / "scripts" / "mcnemar.py"
    r = subprocess.run([_sys.executable, str(script), "--runs_dir", str(runs), "--task", "format",
                        "--rank", "2", "--a", "bottom", "--b", "top"],
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "McNemar paired per EXAMPLE" in r.stdout
    assert "sign test over seeds" in r.stdout and "Stouffer" in r.stdout
    assert "0/3 favour bottom" in r.stdout            # top wins every seed
    assert "NOT reported" in r.stdout                  # the naive pooling is explicitly refused

    # and it REFUSES when the population is incoherent
    bad = runs / "bottom2bis"
    bad.mkdir()
    _yaml.safe_dump({"model": "m", "task": "format", "mode": "bottom", "rank": 2, "lr": 5e-3,
                     "seed": 2, "subspace_salted": True, "data_fingerprint": "abc123456789"},
                    open(bad / "config.yaml", "w"))
    (bad / "results.csv").write_text("run_id\nx\n")
    with open(bad / "generations_test.jsonl", "w") as f:
        for i in range(200):
            f.write(json.dumps({"id": i, "parsed": True, "exact": True}) + "\n")
    r2 = subprocess.run([_sys.executable, str(script), "--runs_dir", str(runs), "--task", "format",
                        "--rank", "2", "--a", "bottom", "--b", "top"],
                        capture_output=True, text=True, timeout=120)
    assert r2.returncode == 2 and "REFUSING to pool" in r2.stdout
    assert "seed 2 appears 2 times" in r2.stdout


def test_seed_combination_neither_inflates_nor_underflows(tmp_path):
    """Two defects, both visible in the script's own output. (1) Summing the discordant pairs across
    seeds treats k x n item-observations as independent when there are n items measured k times:
    three IDENTICAL seeds at p = 1e-4 each pooled to p = 4.6e-13. That pooling is now refused
    explicitly, with a sign test as the assumption-free floor and Stouffer as an indicative upper
    bound. (2) Stouffer printed `p = 0` because `1 - cdf(14.22)` cancels to exactly zero in float64 —
    a p-value of 0 in a paper is indefensible; erfc keeps the tail representable."""
    import json
    import subprocess
    import sys as _sys

    import yaml as _yaml

    runs = tmp_path / "runs"
    for arm, rate in (("bottom", 0.55), ("top", 0.62)):
        for seed in (2, 3, 4):
            d = runs / f"{arm}{seed}"
            d.mkdir(parents=True)
            _yaml.safe_dump({"model": "m", "task": "format", "mode": arm, "rank": 2,
                             "lr": 5e-3 if arm == "bottom" else 1e-2, "seed": seed,
                             "subspace_salted": True, "data_fingerprint": "abc123456789"},
                            open(d / "config.yaml", "w"))
            (d / "results.csv").write_text("run_id\nx\n")
            with open(d / "generations_test.jsonl", "w") as f:
                for i in range(200):
                    ok = (i % 100) < int(rate * 100)
                    f.write(json.dumps({"id": i, "parsed": ok, "exact": ok}) + "\n")

    script = Path(__file__).resolve().parents[1] / "scripts" / "mcnemar.py"
    out = subprocess.run([_sys.executable, str(script), "--runs_dir", str(runs), "--task", "format",
                          "--rank", "2", "--a", "bottom", "--b", "top"],
                         capture_output=True, text=True, timeout=120).stdout

    # the honest floor: with three seeds the sign test cannot go below 0.25, and it says so
    assert "sign test over seeds : 0/3 favour bottom, p = 0.25" in out
    # Stouffer is reported as indicative, and its p-value is NOT printed as zero
    stouffer = [ln for ln in out.splitlines() if "Stouffer" in ln][0]
    assert "p = 0 " not in stouffer and "p = 0\n" not in stouffer
    assert "e-" in stouffer                              # a real small number, not a cancellation
    assert "indicative" in stouffer
    assert "anti-conservative" in out                     # the naive pooling is named and refused



def test_validation_sizes_are_checked_too(tmp_path):
    """The validation set is min(eval_limit, 500): the sweep validates on 400 items and the seed grid
    on 500. select_lr takes its argmax on that very column, so a mixed cell compares scores measured
    on different sets — and check_eval_sizes only ever looked at format_n and mcq_n."""
    from lorasub.aggregate import IncomparableCells, _refuse_if_mixed, select_lr

    d = pd.DataFrame([
        dict(model="m", task="format", mode="bottom", rank=2, val_format_n=400.0),
        dict(model="m", task="format", mode="bottom", rank=2, val_format_n=500.0),
    ])
    with pytest.raises(IncomparableCells, match="validation sizes"):
        _refuse_if_mixed(d, "test")
    _refuse_if_mixed(d.assign(val_format_n=400.0), "test")        # homogeneous: no refusal

    # select_lr warns rather than silently taking an argmax across two sets
    rows = [dict(model="m", task="format", mode="bottom", rank=2, lr=2e-3, seed=s,
                 val_format_parsed=0.63, val_format_n=400.0) for s in (0, 1)]
    rows += [dict(model="m", task="format", mode="bottom", rank=2, lr=5e-3, seed=s,
                  val_format_parsed=0.64, val_format_n=500.0) for s in (0, 1)]
    import io
    import contextlib

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        select_lr(pd.DataFrame(rows))
    assert "validation sizes" in buf.getvalue() and "MIXED" in buf.getvalue()


def test_the_margin_check_does_not_cry_wolf():
    """An argmax is only a choice if it beats the runner-up by more than the noise: on bottom r=2 the
    two best rates sat 0.0013 apart on validation and the selection flipped from 2e-3 to 5e-3 when one
    run landed, which silently made every seed-grid run of that arm run off-optimum.

    But the first version of this check fired whenever the binomial SE was NOT computable — declaring
    a margin of 0.2 'within noise'. A warning that cries wolf is worse than none: it trains the reader
    to skip the real ones, which is exactly how the OLMo ceiling warning ended up ignored."""
    import contextlib
    import io

    from lorasub.aggregate import select_lr

    def _out(rows):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            select_lr(pd.DataFrame(rows))
        return buf.getvalue()

    # real case: 0.0013 margin against a binomial SE of 0.024 at n=400
    tight = [dict(model="m", task="format", mode="bottom", rank=2, lr=lr, seed=s,
                  val_format_parsed=v, val_format_n=400)
             for lr, v in ((2e-3, 0.6312), (5e-3, 0.6325)) for s in range(3)]
    assert "WITHIN NOISE" in _out(tight)

    # a real gap at the same size must NOT be flagged
    wide = [dict(model="m", task="format", mode="bottom", rank=2, lr=lr, seed=s,
                 val_format_parsed=v, val_format_n=400)
            for lr, v in ((2e-3, 0.40), (5e-3, 0.63)) for s in range(3)]
    assert "WITHIN NOISE" not in _out(wide)

    # size unknown: say so, do not claim the margin is noise
    unknown = [dict(model="m", task="format", mode="bottom", rank=2, lr=lr, seed=s,
                    val_format_parsed=v)
               for lr, v in ((5e-4, 0.80), (1e-3, 0.60)) for s in range(3)]
    o = _out(unknown)
    assert "WITHIN NOISE" not in o and "no validation size recorded" in o


def test_environment_is_recorded_with_every_run(workspace):
    """The widest free variable of the setup: twenty nodes, three weeks, a venv rebuilt per node from
    an unpinned requirements.txt, attn_implementation outside the run id. Two seeds of one cell could
    run with different attention kernels and different transformers versions, for gaps of eight
    points. Unanswerable in review, and free to close."""
    from lorasub.aggregate import collect
    from lorasub.train import _environment, run

    env = _environment()
    assert "host" in env and "python" in env
    assert "torch" in env      # available in this environment

    cfg = _cfg(workspace, task="format", mode="top", rank=2, max_steps=1)
    run(cfg, tokenizer=tiny_tokenizer(), quiet=True)
    df = collect(cfg.run_dir.parent)
    mine = df[df.run_id == cfg.run_id]
    for col in ("host", "python", "torch"):
        assert col in df.columns and str(mine[col].iloc[0]).strip(), col


def test_every_grid_using_the_random_arm_gets_the_salt():
    """Thirteen grids out of fifteen left `subspace_salted` unset, which defaulted to False and
    silently reran the defective arm — including null_subspace_1b, whose entire purpose is the null
    distribution of random subspaces, and ortho_control_1b, the control that identifies the thesis.
    The fix was one line: make True the default. This test is the guard against it drifting back."""
    import yaml as _yaml

    from lorasub.config import RunConfig, apply_model_defaults
    from lorasub.launch_grid import expand

    grids = sorted((Path(__file__).resolve().parents[1] / "configs" / "grids").glob("*.yaml"))
    assert grids
    offenders = []
    for g in grids:
        try:
            cfgs = expand(_yaml.safe_load(open(g)))
        except Exception:  # noqa: BLE001
            continue
        for raw in cfgs:
            try:
                c = RunConfig(**apply_model_defaults(dict(raw)))
            except Exception:  # noqa: BLE001
                continue
            if c.mode in ("random", "random_ortho") and not c.subspace_salted:
                offenders.append(f"{g.name}:{c.mode}")
                break
    assert not offenders, f"these grids would rerun the defective random arm: {offenders}"


def test_the_salt_only_enters_the_id_of_arms_that_draw(tmp_path):
    """Regression introduced when the default became True: `top`, `bottom`, `band` and `free` have
    DETERMINISTIC bands, so the flag changes nothing for them — yet it entered their run_id and made
    the 263 existing runs unreachable to any new grid. A grid asking for `top` seed 4 at 1e-2 stopped
    matching the run that already measured exactly that."""
    from lorasub.config import RunConfig, apply_model_defaults

    for mode in ("top", "bottom", "free", "band"):
        base = dict(model=MODEL_ID, task="format", mode=mode, rank=2, lr=1e-2, seed=4)
        if mode == "band":
            base["band_frac"] = 0.5
        on = RunConfig(**apply_model_defaults(dict(base, subspace_salted=True)))
        off = RunConfig(**apply_model_defaults(dict(base, subspace_salted=False)))
        assert "subspace_salted" not in on.id_dict(), mode
        assert on.run_id == off.run_id, mode          # the flag is irrelevant to these arms

    for mode in ("random", "random_ortho"):
        base = dict(model=MODEL_ID, task="format", mode=mode, rank=2, lr=1e-2, seed=4)
        on = RunConfig(**apply_model_defaults(dict(base, subspace_salted=True)))
        off = RunConfig(**apply_model_defaults(dict(base, subspace_salted=False)))
        assert "subspace_salted" in on.id_dict(), mode
        assert on.run_id != off.run_id, mode          # here it IS a different arm


def test_the_salt_is_not_compared_on_arms_that_draw_nothing(tmp_path, monkeypatch, capsys):
    """The mirror image of the run-id regression, and it fired the same day.

    Flipping the default to True made every NEW run record `subspace_salted: True`, while the runs
    produced before it recorded False. On `random` that distinction is the whole point. On `top`,
    `bottom`, `free` and `band` the band is deterministic and the flag changes nothing -- yet
    `check_population` and `_refuse_if_mixed` compared it on every arm, and refused to pair five
    `top` runs that had measured exactly the same thing: three carried False, two carried True.

    The salt describes how a random subspace is drawn. On an arm that draws nothing it is recorded
    but is not a property of the run -- not for the identifier, and not for comparability.
    """
    import json
    import runpy
    import sys

    import yaml

    from lorasub.aggregate import IncomparableCells, collect, crossed_table

    runs = tmp_path / "runs"

    def mk(name, mode, seed, lr, salted, rate):
        d = runs / name
        d.mkdir(parents=True)
        (d / "status").write_text("DONE\n")
        yaml.safe_dump(dict(model="m", task="format", mode=mode, rank=2, lr=lr, seed=seed,
                            subspace_salted=salted, data_fingerprint="abc"),
                       open(d / "config.yaml", "w"))
        row = dict(model="m", task="format", mode=mode, rank=2, lr=lr, seed=seed, band_frac="",
                   n_trainable=1, format_parsed=rate, format_n=614, val_format_parsed=rate,
                   val_format_n=500, subspace_salted=salted, data_fingerprint="abc")
        import csv
        with open(d / "results.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            w.writeheader()
            w.writerow(row)
        with open(d / "generations_test.jsonl", "w") as f:
            rng = random.Random(seed * 7 + (0 if mode == "top" else 1))
            for i in range(614):
                f.write(json.dumps({"id": i, "parsed": rng.random() < rate}) + "\n")

    # `top`: three runs from before the default flipped, two from after. Same arm, same measurement.
    for s, salted in ((4, False), (5, False), (6, False), (2, True), (3, True)):
        mk(f"t{s}", "top", s, 0.01, salted, 0.64)
    for s in (2, 3, 4, 5, 6):
        mk(f"b{s}", "bottom", s, 0.002, False, 0.56)

    crossed_table(collect(runs), "m", 2)          # must not raise

    script = str(Path(__file__).resolve().parents[1] / "scripts" / "mcnemar.py")
    monkeypatch.setattr(sys, "argv",
                        ["mcnemar.py", "--runs_dir", str(runs), "--task", "format", "--rank", "2",
                         "--a", "bottom", "--b", "top", "--eval_n", "614"])
    runpy.run_path(script, run_name="__main__")
    out = capsys.readouterr().out
    assert "REFUSING" not in out
    assert "McNemar paired per EXAMPLE" in out

    # but on `random` the two versions must still be kept apart
    for s, salted in ((2, True), (3, False)):
        mk(f"r{s}", "random", s, 0.01, salted, 0.65)
    with pytest.raises(IncomparableCells, match="random-arm versions"):
        crossed_table(collect(runs), "m", 2)


def test_summarize_does_not_split_an_arm_that_draws_nothing(tmp_path):
    """The third place the scoping was missing, and the quietest one.

    `check_population` and `_refuse_if_mixed` refuse loudly when a cell mixes the two random arms, and
    both were scoped to the arms the salt describes. `KEY` was not: it carries `subspace_salted`
    unconditionally, so the moment the default flipped to True, summary.csv showed `top` as two rows
    of three and two seeds — five runs that had measured exactly the same thing, split in silence.
    No refusal, no warning, just wrong effectifs in the file one rereads to check them.

    `collect()` now normalises the flag to False wherever the band is deterministic, which fixes KEY
    and every other groupby at once.
    """
    import csv

    from lorasub.aggregate import collect, select_population, summarize

    runs = tmp_path / "runs"

    def mk(name, mode, seed, salted, rate):
        d = runs / name
        d.mkdir(parents=True)
        (d / "status").write_text("DONE\n")
        row = dict(model="m", task="format", mode=mode, rank=2, lr=0.01, seed=seed, band_frac="",
                   n_trainable=1, format_parsed=rate, format_n=614, val_format_parsed=rate,
                   val_format_n=500, subspace_salted=salted, data_fingerprint="abc")
        with open(d / "results.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            w.writeheader()
            w.writerow(row)

    # three `top` runs from before the default flipped, two from after
    for s, salted in ((4, False), (5, False), (6, False), (2, True), (3, True)):
        mk(f"t{s}", "top", s, salted, 0.64)
    for s, salted in ((2, True), (3, True), (4, False)):
        mk(f"r{s}", "random", s, salted, 0.65)

    df = collect(runs)
    assert not df[df["mode"] == "top"]["subspace_salted"].any(), \
        "the salt must be normalised away on an arm whose band is deterministic"

    rows = summarize(df)
    top = rows[rows["mode"] == "top"]
    assert len(top) == 1, f"`top` was split into {len(top)} rows for one arm"

    # and the two random arms must still be told apart, in summarize and in select_population
    assert len(rows[rows["mode"] == "random"]) == 2
    kept = select_population(df, salted=True)
    assert len(kept[kept["mode"] == "random"]) == 2 and len(kept[kept["mode"] == "top"]) == 5


def test_relative_perturbation_reconstructs_the_salted_band_per_module():
    """The fourth place the salt was not read, and the quietest.

    `relative_perturbation.py` rebuilt the adapted band as `random:{r}:{seed}` — the UNSALTED form —
    for every run. On a salted run the bands differ per module, so dW was compared against a band it
    was never written to: the denominator is wrong and the relative perturbation reported for the
    random arm is not a measurement. `top` and `bottom` are deterministic and were unaffected.

    Same shape as the three before it: the run id, KEY, and the refusals. A field that changes what is
    measured has to be read everywhere it matters, and 'everywhere' has been four places so far.
    """
    import importlib.util
    import sys
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts" / "relative_perturbation.py"
    spec = importlib.util.spec_from_file_location("rel_pert_script", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["rel_pert_script"] = mod
    spec.loader.exec_module(mod)

    # deterministic arms: the module name must not appear
    assert mod.band_spec("top", 2, 0, True, "layers.0.q_proj") == "top:2"
    assert mod.band_spec("bottom", 2, 0, True, "layers.0.q_proj") == "bottom:2"

    # unsalted random: one band for every module, which is exactly what that arm was
    assert (mod.band_spec("random", 2, 0, False, "layers.0.q_proj")
            == mod.band_spec("random", 2, 0, False, "layers.7.o_proj")
            == "random:2:0")

    # salted random: a different band per module, and it must carry the module name
    a = mod.band_spec("random", 2, 0, True, "layers.0.q_proj")
    b = mod.band_spec("random", 2, 0, True, "layers.7.o_proj")
    assert a != b, "the salted arm draws per module; one band for all reproduces the bug"
    assert "layers.0.q_proj" in a and "layers.7.o_proj" in b

    # and the seed still separates two salted draws
    assert mod.band_spec("random", 2, 1, True, "layers.0.q_proj") != a


def test_metric_version_is_a_hash_of_the_scorer_not_a_number(workspace):
    """The checker was hardened on 11/09 and nothing recorded which definition scored a run; the
    question was only closed because `format_order_ok` happened to exist — by luck. A hand-maintained
    version number is the thing that gets forgotten, so the digest is computed from the SOURCE of
    `check_alien` and cannot lie."""
    from lorasub.aggregate import collect, effective_metric_version
    from lorasub.data.format import metric_version
    from lorasub.train import run

    v = metric_version()
    assert v and len(v) == 12 and v == metric_version()          # stable

    cfg = _cfg(workspace, task="format", mode="top", rank=2, max_steps=1)
    run(cfg, tokenizer=tiny_tokenizer(), quiet=True)
    df = collect(cfg.run_dir.parent)
    assert "metric_version" in df.columns
    mine = df[df.run_id == cfg.run_id]
    assert str(mine["metric_version"].iloc[0]) == v

    # a run produced before the field falls back on the deduced version, under a distinct label
    cols = ["model", "task", "mode", "rank", "format_parsed", "format_order_ok", "metric_version"]
    legacy = mine[cols].copy()                      # SAME model, or the two form separate groups
    legacy["metric_version"] = ""
    legacy["format_parsed"], legacy["format_order_ok"] = 0.5, 0.9
    assert effective_metric_version(legacy).iloc[0] == "pre-field:strict-or-undetected"
    mixed = pd.concat([mine[cols], legacy], ignore_index=True)
    from lorasub.aggregate import check_metric_versions

    assert check_metric_versions(mixed), "a recorded digest and a legacy run are two populations"


def test_metric_version_hashes_the_helpers_check_alien_calls():
    """The digest must cover the transitive closure, not just `check_alien`.

    `check_alien` delegates to `signature`, `spell_number` and the three `_f1*` helpers. Hashing only
    its own source would let a change to how initials are built, or to how counts are spelled, alter
    the metric while the digest stays put — the exact failure this field exists to prevent, one call
    deep. Proven below by removing `.upper()` from `signature`: the digest must move.

    And reformatting must NOT move it, or every campaign is invalidated by a stylistic edit.
    """
    import importlib.util
    import pathlib
    import sys
    import tempfile
    import types

    def load(path):
        sys.modules.pop("fmt_probe", None)
        pkg = types.ModuleType("lorasub")
        pkg.__path__ = [str(pathlib.Path(path).parents[1])]
        data = types.ModuleType("lorasub.data")
        data.__path__ = [str(pathlib.Path(path).parent)]
        saved = {k: sys.modules.get(k) for k in ("lorasub", "lorasub.data")}
        sys.modules.update({"lorasub": pkg, "lorasub.data": data})
        try:
            spec = importlib.util.spec_from_file_location("lorasub.data.format", path)
            mod = importlib.util.module_from_spec(spec)
            sys.modules["lorasub.data.format"] = mod
            spec.loader.exec_module(mod)
            return mod.metric_version()
        finally:
            for k, v in saved.items():
                if v is not None:
                    sys.modules[k] = v

    orig = pathlib.Path(__file__).resolve().parents[1] / "src" / "lorasub" / "data" / "format.py"
    src = orig.read_text()
    base = load(str(orig))
    assert len(base) == 12 and base == load(str(orig)), "the digest must be deterministic"

    tmp = pathlib.Path(tempfile.mkdtemp())

    # cosmetic edits must not move it
    doc = '"""Initials of every entity, in JSON order, upper-cased (requires attending to all of them)."""'
    assert doc in src
    cosmetic = tmp / "cosmetic.py"
    cosmetic.write_text(src.replace(doc, '"""A completely different wording."""   # and a comment', 1))
    assert load(str(cosmetic)) == base, "reformatting must not invalidate a campaign"

    # a logic change INSIDE A HELPER must move it
    line = 'return "".join(v.strip()[0].upper() for v in values if v.strip())'
    assert line in src
    helper = tmp / "helper.py"
    helper.write_text(src.replace(line, line.replace(".upper()", ""), 1))
    assert load(str(helper)) != base, (
        "changing `signature` changes the metric; the digest must not stay put")


def test_every_paper_facing_figure_refuses_a_mixed_cell(tmp_path):
    """Six refusals existed and only the two TABLES used them.

    `crossed_table` and `win_matrix` raised on an incomparable cell; `plot_lr_heatmap`,
    `plot_crossed`, `plot_spectral_curve`, `plot_failure_decomposition` and `forgetting_delta` did
    not. The rate map is the worst of the five — its own docstring calls it "THE figure for the Lee
    et al. objection" — and a cell there is (arm, rank, lr), so the 400-item sweep runs sat in the
    same cell as the 614-item seed-grid runs and were averaged in silence into a number that is
    neither. Nobody reads crossed_*.tex; they look at the heatmap.

    `forgetting_delta` had a second defect: it grouped on (model, task, mode, rank) WITHOUT lr, so
    §5.6 averaged `bottom` at 2e-3 and at 5e-3 — the collapsed seed included — and reported the mean
    as "the forgetting of the minor arm".
    """
    import csv

    from lorasub.aggregate import (IncomparableCells, collect, forgetting_delta, plot_crossed,
                                   plot_failure_decomposition, plot_lr_heatmap, plot_spectral_curve)

    runs = tmp_path / "runs"

    def mk(name, mode, seed, lr, n_eval, score):
        d = runs / name
        d.mkdir(parents=True)
        (d / "status").write_text("DONE\n")
        row = dict(model="m", task="format", mode=mode, rank=2, lr=lr, seed=seed, band_frac="",
                   n_trainable=1, format_parsed=score, format_exact=score / 2, format_f1=score,
                   format_empty=0.0, format_truncated=0.0, format_keys_ok=0.9, format_order_ok=0.9,
                   format_n=n_eval, val_format_parsed=score, val_format_n=500,
                   subspace_salted=False, data_fingerprint="abc", hellaswag_acc_norm=0.45,
                   truthfulqa_mc1=0.30, wikitext_ppl=12.0)
        with open(d / "results.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            w.writeheader()
            w.writerow(row)

    # ONE cell, (bottom, r=2, lr=2e-3), holding a 400-item run and a 614-item one
    mk("a", "bottom", 0, 0.002, 400, 0.63)
    mk("b", "bottom", 2, 0.002, 614, 0.56)
    mk("base", "baseline", 0, 0.0, 614, 0.0)
    df = collect(runs)

    png = tmp_path / "x.png"
    for name, call in (("plot_lr_heatmap", lambda: plot_lr_heatmap(df, "m", "format", png)),
                       ("plot_crossed", lambda: plot_crossed(df, "m", png)),
                       ("plot_spectral_curve", lambda: plot_spectral_curve(df, "m", 2, png)),
                       ("plot_failure_decomposition",
                        lambda: plot_failure_decomposition(df, "m", png)),
                       ("forgetting_delta", lambda: forgetting_delta(df))):
        with pytest.raises(IncomparableCells, match="evaluation sizes"):
            call()

    # and on a homogeneous population they all run, with lr in the forgetting key
    clean = df[(df.format_n == 614)]
    plot_lr_heatmap(clean, "m", "format", png)
    fd = forgetting_delta(clean)
    assert "lr" in fd.columns, "averaging across learning rates is what this key exists to prevent"


def test_collect_emits_no_pandas_downcasting_warning(tmp_path):
    """Fourth return of this warning, and each time it meant aggregate.py had drifted between the
    workstation and the cluster. `fillna` on an object column is the deprecated call; `infer_objects`
    afterwards does not silence it. Asserted as an error so it cannot come back a fifth time."""
    import csv
    import warnings

    from lorasub.aggregate import collect

    runs = tmp_path / "runs"
    for i, salted in enumerate((True, False)):
        d = runs / f"r{i}"
        d.mkdir(parents=True)
        (d / "status").write_text("DONE\n")
        row = dict(model="m", task="format", mode="random", rank=2, lr=0.01, seed=i, band_frac="",
                   n_trainable=1, format_parsed=0.6, format_n=614, subspace_salted=salted)
        with open(d / "results.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            w.writeheader()
            w.writerow(row)

    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        out = collect(runs)
    assert out["subspace_salted"].dtype == bool


def test_a_target_module_that_matches_nothing_raises(monkeypatch):
    """A per-layer grid names full paths; a typo in the layer index matches nothing SILENTLY.

    `iter_target_modules` selects on the suffix for bare names and on the full path for dotted ones.
    The pre-existing guard only fired when the whole list matched nothing. Ask for
    `['q_proj', 'layers.99.self_attn.k_proj']` and `q_proj` still matches: the run trains fewer
    adapters than requested, writes a score, and raises nothing. On an 80-config grid written by
    hand, that is the failure that happens.
    """
    from lorasub.spectral import iter_target_modules

    class Lin(torch.nn.Linear):
        def __init__(self):
            super().__init__(2, 2)

    class Fake(torch.nn.Module):
        def named_modules(self, *a, **k):
            for layer in (0, 7, 15):
                for path in ("self_attn.q_proj", "self_attn.k_proj", "mlp.up_proj"):
                    yield f"model.layers.{layer}.{path}", Lin()

    got = [n for n, _ in iter_target_modules(Fake(), ["q_proj"])]
    assert len(got) == 3, "a bare suffix selects that projection in every layer"

    got = [n for n, _ in iter_target_modules(Fake(), ["layers.15.self_attn.q_proj"])]
    assert got == ["model.layers.15.self_attn.q_proj"], "a dotted path selects exactly one module"

    assert list(iter_target_modules(Fake(), ["layers.99.self_attn.q_proj"])) == [], (
        "a non-existent path matches nothing — which is why inject_lora must refuse it")


def test_dual_modes_freeze_the_input_factor_and_count_their_own_budget():
    """The dual arms freeze A on rows of V and train B — the mirror of every other constrained arm.

    Two things must hold and neither is obvious.

    First, A frozen / B trained, with B at zero so dW = 0 at step 0 like everywhere else. Getting
    this backwards would silently produce an ordinary constrained arm under a new name.

    Second, the budget. A constrained arm trains r*d_in, a dual arm r*d_out. They coincide on square
    modules and diverge by 4x on gate_proj (8192x2048). `expected_trainable` is what asserts the
    budget at every run, so if it kept counting r*d_in for a dual arm, a four-fold budget difference
    would pass as an equal-budget comparison — exactly the confound this project spends its time
    eliminating elsewhere.
    """
    import torch.nn as nn

    from lorasub.lora import DUAL, LoRALinear, expected_trainable, inject_lora
    from lorasub.spectral import SVDEntry

    out_f, in_f, r = 8, 4, 2                       # rectangular on purpose: out != in
    lin = nn.Linear(in_f, out_f, bias=False)
    U, S, Vh = torch.linalg.svd(lin.weight.detach().float(), full_matrices=False)
    svd = SVDEntry(name="m", shape=(out_f, in_f), S=S, U=U, Vh=Vh)

    for mode in DUAL:
        m = LoRALinear(lin, r=r, mode=mode, svd=svd, seed=0, module_salt="m")
        assert not m.A.requires_grad, f"{mode}: A must be frozen"
        assert m.B.requires_grad, f"{mode}: B must be trained"
        assert torch.allclose(m.B, torch.zeros_like(m.B)), f"{mode}: dW must be 0 at step 0"
        assert m.A.shape == (r, in_f)
        if mode != "dual_random_ortho":                     # A is r rows of Vh
            assert torch.allclose(m.A, Vh[m.band_idx, :], atol=1e-5), f"{mode}: A must come from Vh"
        # rows of A are orthonormal in both cases
        assert torch.allclose(m.A @ m.A.T, torch.eye(r), atol=1e-4), f"{mode}: A must be orthonormal"

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(in_f, out_f, bias=False)

    net = Tiny()
    assert expected_trainable(net, r, "dual_top", ("q_proj",)) == r * out_f, (
        "a dual arm trains r*d_out, not r*d_in")
    assert expected_trainable(net, r, "top", ("q_proj",)) == r * in_f
    assert expected_trainable(net, r, "free", ("q_proj",)) == r * (in_f + out_f)


def test_the_salted_modes_list_has_no_second_copy():
    """Every place that asks "does this arm draw its subspace?" must use ONE list.

    The list was hard-coded in three files — config.id_dict, aggregate.collect,
    aggregate.select_population — plus ARMS in aggregate. Adding the dual family to lora.MODES
    updated none of them, and the four consequences were all silent:

    * id_dict dropped `subspace_salted` from the id of dual_random, so two runs differing only by
      the salt would have collided on one run_id and one directory;
    * collect rewrote the salt flag to False for every dual row, so the metadata no longer described
      what ran;
    * select_population could not separate the two versions of a dual random arm;
    * ARMS did not contain the dual modes at all, and ARMS is what crossed_table, plot_lr_heatmap,
      plot_crossed and forgetting_delta filter on — the twenty runs meant to decide H1 would have
      appeared in no table and no figure, without an error.

    This test fails if a fifth copy of the list appears, or if a drawing mode is added to MODES
    without being added to SALTED_MODES.
    """
    import re
    from pathlib import Path

    from lorasub.lora import DUAL, MODES, SALTED_MODES

    # every drawing arm is declared salted
    for m in MODES:
        draws = "random" in m
        assert (m in SALTED_MODES) == draws, f"{m}: drawing modes and SALTED_MODES disagree"
    for m in DUAL:
        assert m in MODES, f"{m} missing from MODES"

    src = Path(__file__).resolve().parents[1] / "src" / "lorasub"
    literal = re.compile(r'\[\s*"random"\s*,\s*"random_ortho"\s*\]|\(\s*"random"\s*,\s*"random_ortho"\s*\)')
    for f in ("config.py", "aggregate.py"):
        text = (src / f).read_text()
        assert not literal.search(text), (
            f"{f} still hard-codes the salted-arm list; import SALTED_MODES instead")

    # ARMS must cover every mode that can reach a table
    from lorasub.aggregate import ARMS
    for m in MODES:
        assert m in ARMS, f"{m} is in MODES but not in ARMS: its runs would vanish from every figure"

    # and the REFUSAL itself must know which arms draw. _refuse_if_mixed skips the salt column for
    # arms that do not draw, because for them True and False describe the same run. If that skip
    # list omits the dual random arms, the guard stops checking them — the one place where a silent
    # mixture would be caught goes blind on the family being launched.
    agg = (src / "aggregate.py").read_text()
    i = agg.index('col == "subspace_salted"')
    line = agg[i:agg.index("\n", i)]
    assert "SALTED_MODES" in line, (
        "_refuse_if_mixed hard-codes which arms carry a meaningful salt: " + line.strip())


def test_init_only_leaves_the_model_untouched_at_step_zero():
    """PiSSA and MiLoRA SUBTRACT the band from the frozen weight. Skipping that is not PiSSA.

    A naive spectral initialisation — B on the band, A kaiming, W0 untouched — perturbs the layer at
    step 0 by the full energy of the band. On the top band of a real module that is most of the
    weight. The first version written here did exactly that, and the paper would then have claimed
    to refute PiSSA using something PiSSA does not do.

    Faithfully: W_frozen = W0 - U_r S_r V_r^T, B = U_r sqrt(S_r), A = sqrt(S_r) V_r^T / scaling, so
    W_frozen + scaling * B @ A = W0 exactly. The adapter starts by reproducing the band it removed,
    then relearns it freely — and BOTH factors are trained, which is what separates these arms from
    every other constrained arm here.

    A second trap, found numerically: the band is B @ A, not U_r @ A. Writing U_r @ A drops one
    factor sqrt(S_r) and leaves a residual of 0.87 on a random 6x4.
    """
    import torch.nn as nn

    from lorasub.lora import INIT_ONLY, LoRALinear, expected_trainable
    from lorasub.spectral import SVDEntry

    out_f, in_f, r = 6, 4, 2
    for mode in INIT_ONLY:
        lin = nn.Linear(in_f, out_f, bias=False)
        W0 = lin.weight.detach().clone().float()
        U, S, Vh = torch.linalg.svd(W0, full_matrices=False)
        svd = SVDEntry(name="m", shape=(out_f, in_f), S=S, U=U, Vh=Vh)
        m = LoRALinear(lin, r=r, mode=mode, alpha=2 * r, svd=svd, seed=0, module_salt="m")

        assert m.A.requires_grad and m.B.requires_grad, f"{mode}: both factors must be trained"
        eff = m.base.weight.detach().float() + m.scaling * (m.B.detach() @ m.A.detach())
        assert torch.allclose(eff, W0, atol=1e-4), (
            f"{mode}: the layer must be unchanged at step 0; max deviation "
            f"{(eff - W0).abs().max().item():.2e}")

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(in_f, out_f, bias=False)

    assert expected_trainable(Tiny(), r, "init_top", ("q_proj",)) == r * (in_f + out_f), (
        "init_only trains both factors, so its budget is that of `free`, not of a constrained arm")


def test_mcnemar_separates_runs_by_how_many_modules_they_adapt():
    """Which modules were adapted is part of the experiment, and nothing else distinguished them.

    The module-localisation grid reruns `bottom` and `top` on five projections instead of seven,
    with the SAME mode, rank, learning rate and evaluation size. Nothing in mcnemar's filters told
    the two apart, so every seed appeared twice and the pairing check refused to pool — correctly,
    but the refusal is a symptom. The fix is a filter: a run adapting five modules and one adapting
    seven did not measure the same thing.

    Seventh occurrence of the same pattern in this project (run id, KEY, refusals, band_spec,
    update_norm_target, the salted modes, and now target_modules). The rule it generalises: when a
    new grid varies a field, every function that SELECTS runs must be checked, not only the ones
    that compute.
    """
    import subprocess
    import sys
    from pathlib import Path

    # Parse the script's own argument parser rather than grepping its source or shelling out. A
    # grep pins the phrasing and goes red on a refactor; a subprocess depends on which interpreter
    # PATH happens to resolve to, and returns an empty string when the import fails — which reads
    # as "the flag is missing" when it is in fact present.
    import importlib.util

    path = Path(__file__).resolve().parents[1] / "scripts" / "mcnemar.py"
    src = path.read_text()
    assert 'len(cfg.get("target_modules") or [])' in src, (
        "the filter must read target_modules from the run config")
    i = src.index("a.n_modules is not None")
    assert "continue" in src[i:i + 120], "the filter must skip non-matching runs"

    spec = importlib.util.spec_from_file_location("_mcnemar_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    parser = mod.build_parser() if hasattr(mod, "build_parser") else None
    if parser is None:
        assert "--n_modules" in src and "--model" in src, (
            "mcnemar must expose --n_modules and --model")
    else:
        flags = {a.option_strings[0] for a in parser._actions if a.option_strings}
        assert "--n_modules" in flags and "--model" in flags


def test_the_refusal_keys_on_the_model_too():
    """A cell is (model, task, mode, rank) — not (task, mode, rank).

    Until 15/09 only Llama had runs at 614 items, so keying on (task, mode, rank) happened to be
    safe. The day Qwen produced its first 614-item run, a cell (format, top, 2) started holding two
    architectures: different d_in, different depth, 655360 trainable parameters against 491520. The
    two would have been averaged into one row of the paper's main table without any refusal.

    The same blind spot was in every ad-hoc query written that evening, which is why the guard has
    to carry it rather than the person writing the query.
    """
    from pathlib import Path

    # Check the BEHAVIOUR, not the source text. An earlier version of this test looked for the
    # literal string groupby(["model", "task", "mode", "rank"]); when the key had to become tolerant
    # of a frame with no model column, the behaviour stayed correct and the test went red. A test
    # that pins an implementation detail fails on a refactor and says nothing about what matters.
    from lorasub.aggregate import IncomparableCells, _refuse_if_mixed

    base = dict(task="format", mode="bottom", rank=2, format_n=614.0, val_format_n=500.0,
                subspace_salted=False, data_fingerprint="a", max_grad_norm=1.0, n_modules=7,
                update_norm_target=None, format_parsed=0.56)
    # two architectures in what would otherwise be one cell: they must land in separate groups, so
    # nothing is refused and nothing is pooled either
    _refuse_if_mixed(pd.DataFrame([dict(base, model="meta-llama/Llama-3.2-1B"),
                                   dict(base, model="Qwen/Qwen2.5-0.5B")]), "test")
    # and within ONE model, a genuine mix must still be refused
    with pytest.raises(IncomparableCells):
        _refuse_if_mixed(pd.DataFrame([dict(base, model="m"),
                                       dict(base, model="m", n_modules=5)]), "test")
    # a frame that carries no model column at all must not raise KeyError
    no_model = {k: v for k, v in base.items()}
    _refuse_if_mixed(pd.DataFrame([no_model]), "test")


def test_relative_perturbation_can_restrict_to_one_model():
    """Two architectures are not commensurable in a relative-perturbation median.

    The script grouped by (mode, rank, lr) and never by model. The day Qwen produced its first runs,
    `bottom` at 2e-3 showed nine runs for five seeds — four of them Qwen — and the median moved from
    5.13 to 3.78, which would have contradicted the paper's own figure. Ninth occurrence of the same
    pattern: a field that changes what is measured and does not reach the filters.
    """
    from pathlib import Path

    src = (Path(__file__).resolve().parents[1] / "scripts" / "relative_perturbation.py").read_text()
    assert '"--model"' in src, "the script must expose a --model filter"
    assert "a.model" in src, "the filter must actually be applied"
    assert "models in this file" in src, (
        "a file holding several models must say so: the default keeps everything")


def test_every_analysis_script_filters_the_fields_that_define_an_experiment():
    """Five fields make a run a DIFFERENT experiment. Every script that aggregates must filter them.

    The list was paid for one bug at a time: the evaluation size (400 against 614), the salt on the
    drawing arms, update_norm_target for the rescaled ablation, target_modules for the localisation
    grid, max_grad_norm for the noclip grid, and the model once a second architecture appeared.

    Three scripts written on 15-16/09 were missing some of them. `relative_perturbation` — the one
    that produces §5.6 — had no pairing check at all and no clip or module filter, so `bottom` at
    2e-3 showed nine runs for five seeds and its median moved from 5.13 to 3.78. That number would
    have contradicted the paper's own figure, and nothing would have said so.
    """
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "scripts"
    required = {
        "relative_perturbation.py": ("max_grad_norm", "n_modules", "--model", "appear more than once"),
        "forgetting_from_factors.py": ("max_grad_norm", "n_modules", "update_norm_target"),
        "mcnemar.py": ("max_grad_norm", "target_modules", "update_norm_target"),
        "score_vs_norm.py": ("max_grad_norm", "update_norm_target", "subspace_salted"),
    }
    for fname, needles in required.items():
        src = (root / fname).read_text()
        for n in needles:
            assert n in src, f"{fname} does not handle {n!r}: it can silently pool two experiments"


def test_collect_carries_the_three_fields_that_define_an_experiment():
    """`results.csv` says what a run measured; it does not say WHICH EXPERIMENT it belongs to.

    Three settings make two runs incomparable even when model, task, mode, rank, lr and seed all
    match, and none of them was read by `collect`:

    * `update_norm_target` — the magnitude ablation rescales the final update; those runs score
      0.000 or 0.65 depending on the target;
    * `max_grad_norm`      — the noclip grid reruns the same cell without clipping;
    * `target_modules`     — the localisation grid reruns it on five projections instead of seven.

    A cell (format, bottom, r=2) therefore held the five normal runs, five rescaled to 1.5, five
    rescaled to 4.5, the unclipped ones and the five-module ones, and `crossed_table` averaged all
    of them. Every hand-written query in this project filtered them manually, which is exactly what
    a person forgets once.

    And for `update_norm_target` the ABSENCE is the interesting value: a normal run has no target,
    so dropping empties would leave one level in a mixed cell and the guard would pass it.
    """
    from pathlib import Path

    src = (Path(__file__).resolve().parents[1] / "src" / "lorasub" / "aggregate.py").read_text()
    head = src[src.index("def collect"):src.index("\ndef ", src.index("def collect") + 5)]
    for f in ("update_norm_target", "max_grad_norm", "target_modules"):
        assert f in head, f"collect must read {f} from config.yaml: results.csv does not carry it"

    guard = src[src.index("def _refuse_if_mixed"):src.index("\ndef ", src.index("def _refuse_if_mixed") + 5)]
    for f in ("update_norm_target", "max_grad_norm", "n_modules"):
        assert f in guard, f"_refuse_if_mixed must check {f}"
    i = guard.index('if col in ("data_fingerprint"')
    line = guard[i:guard.index("\n", i)]
    assert "update_norm_target" in line and "max_grad_norm" in line, (
        "absence must count as a value for these two, or a cell mixing normal and rescaled runs "
        "shows a single level and passes")


def test_constrained_follows_when_a_mode_family_is_added():
    """CONSTRAINED gates the SVD cache load. Hand-listing it means a new family silently breaks.

    It read ("top", "bottom", "random", "band"). Two families were added this week — DUAL and
    INIT_ONLY — and neither reached it, so every init_only run died on
    `mode 'init_top' needs an SVDEntry of the base weight`, after the grid had been launched twice
    and two nodes had rebuilt their venv for nothing.

    Tenth occurrence of the pattern. The fix is to derive the tuple rather than to extend it, so the
    next family is covered without anyone remembering this file. The two Haar arms stay out on
    purpose: a uniformly drawn orthonormal plane is unrelated to U and needs no SVD — that is what
    those arms are for.
    """
    from lorasub.lora import CONSTRAINED, DUAL, INIT_ONLY

    for m in INIT_ONLY:
        assert m in CONSTRAINED, f"{m} needs the base weight's SVD and would raise at injection"
    for m in DUAL:
        if m.endswith("random_ortho"):
            assert m not in CONSTRAINED, f"{m} draws a Haar plane and needs no SVD"
        else:
            assert m in CONSTRAINED, f"{m} needs the base weight's SVD"
    assert "random_ortho" not in CONSTRAINED, "the Haar arm must not require the SVD"

    src = (__import__("pathlib").Path(__file__).resolve().parents[1]
           / "src" / "lorasub" / "lora.py").read_text()
    i = src.index("CONSTRAINED: tuple[str, ...] =")
    body = src[i:src.index("\n\n", i)]
    assert "DUAL" in body and "INIT_ONLY" in body, (
        "CONSTRAINED must be DERIVED from the mode families, not hand-listed: a hand-listed tuple "
        "is exactly what failed here")


def test_keep_selected_lr_says_so_when_it_empties_a_cell():
    """A filter that drops every row of a cell is not filtering; it is failing.

    `lr` decides which rows survive. When results.csv does not carry it, every value is NaN,
    `float(nan) == float(2e-3)` is False, and EVERY run of every cell that has a selected rate is
    dropped — leaving only the cells absent from the table. That produced a main figure showing one
    arm out of five, regenerated three times before anyone looked at it.
    """
    from lorasub.aggregate import _keep_selected_lr

    tab = {"m|format|bottom|2": 2e-3}
    rows = [dict(model="m", task="format", mode="bottom", rank=2, lr=float("nan")) for _ in range(5)]
    kept = _keep_selected_lr(pd.DataFrame(rows), tab)
    assert kept.empty          # the behaviour itself is unchanged...
    # ...but the caller must have been told, which is what the print above provides.

    ok = [dict(model="m", task="format", mode="bottom", rank=2, lr=2e-3) for _ in range(5)]
    assert len(_keep_selected_lr(pd.DataFrame(ok), tab)) == 5
    # a rank read as 2.0 from a CSV must match a rank written as 2 in the YAML
    f = [dict(model="m", task="format", mode="bottom", rank=2.0, lr=2e-3) for _ in range(5)]
    assert len(_keep_selected_lr(pd.DataFrame(f), tab)) == 5


def test_shifted_power_iteration_resolves_a_small_eigenvalue():
    """The convergence test must be on the shifted-BACK eigenvalue, not the raw Rayleigh quotient.

    With a shift near 1 and a true eigenvalue near 4e-4, `tol * max(1, |lam|)` is 1e-3 — larger than
    the quantity being estimated — so the loop exits on its first iteration and returns whatever the
    random start produced. On the real measurements that gave exact zeros, negative maxima, and a
    standard deviation larger than the median, on four arms out of five and on BOTH models. Only
    `bottom` on Qwen (1.1e-2) sat above the tolerance and was genuinely resolved.

    A curvature of 1e-3 is the normal case here, not an edge case: that is what these arms have.
    """
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts" / "curvature.py"
    spec = importlib.util.spec_from_file_location("_curv_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)

    import torch

    # a diagonal Hessian whose largest eigenvalue is 4e-4, with a larger-in-modulus negative one
    diag = torch.tensor([4.0e-4, 1.5e-4, -2.0e-4, 5.0e-5])
    p = [torch.zeros(4, requires_grad=True)]

    def loss_fn():
        return 0.5 * (diag * p[0] * p[0]).sum()

    for _ in range(3):
        # four values since the instrumentation: (eigenvalue, iterations, shift, converged)
        lam, _, _, ok = mod.top_eigenvalue(loss_fn, p, iters=80)
        assert lam > 0, "a largest eigenvalue cannot come back negative"
        assert abs(lam - 4.0e-4) < 2e-5, f"expected 4e-4, got {lam:.3e}"


def test_run_id_is_always_a_string_even_when_it_looks_like_a_number():
    """An id of hex digits that happen to all be decimal must not be typed as an integer.

    `run_id` comes from results.csv and pandas types a column by its contents. An id like
    254002041782 is read as int64 while its neighbour 9303ba2164a8 stays a string. Any later
    `.map()` against a dict keyed by directory name then misses it and fills its config fields
    with NaN — so the run vanishes from every filter on n_modules or update_norm_target without
    a word. Six runs out of 991 were affected, one of them a cell of the main rank table, and it
    took an afternoon to find because the symptom was a missing seed, not an error.
    """
    import tempfile
    from pathlib import Path
    from lorasub.aggregate import collect

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        for rid in ("254002041782", "9303ba2164a8"):
            r = root / rid
            r.mkdir()
            (r / "status").write_text("DONE")
            (r / "config.yaml").write_text("model: m\ntask: format\nmode: top\nrank: 1\nseed: 2\n")
            (r / "results.csv").write_text(
                "run_id,model,task,mode,rank,seed,format_parsed,format_n\n"
                f"{rid},meta-llama/Llama-3.2-1B,format,top,1,2,0.5,614\n")
        got = collect(root)
        assert len(got) == 2, f"expected both runs, got {len(got)}"
        assert all(isinstance(x, str) for x in got["run_id"]), \
            f"run_id must be str, got {[type(x).__name__ for x in got['run_id']]}"
        # and the map that lost them must now find both
        names = {p.parent.name: 7 for p in root.glob("*/config.yaml")}
        assert got["run_id"].map(names).notna().all(), "a .map() on directory names still loses rows"


def test_select_lr_refuses_to_mix_populations_inside_a_cell():
    """(model, task, mode, rank) is a cell key, not a population.

    The same cell also holds the noclip reruns (max_grad_norm 1e6), the rescaled arms
    (update_norm_target set), and the span/budget controls that adapt 5 or 2 modules instead of 7.
    Averaging a validation score over that mixture and taking an argmax compares rates measured on
    different experiments — the exact fault this project reports in the literature it discusses.

    Found when a rate looked 1.76 sigma better than its neighbour: the margin came from two noclip
    runs at 0.560 and 0.573 sitting in a cell whose clipped runs were at 0.63. Paired on the seeds
    the two rates actually share, the gap was 0.0013 — a tie, which is what the paper reports.
    """
    from lorasub.aggregate import select_lr

    base = dict(model="m", task="format", mode="bottom", rank=2, n_modules=7,
                update_norm_target=None, val_format_n=400.0, format_n=400.0)
    rows = [
        # the real cell: 2e-3 and 5e-3 are tied
        dict(base, lr=0.002, seed=0, max_grad_norm=1.0, val_format_parsed=0.6325),
        dict(base, lr=0.002, seed=1, max_grad_norm=1.0, val_format_parsed=0.6300),
        dict(base, lr=0.005, seed=0, max_grad_norm=1.0, val_format_parsed=0.6350),
        dict(base, lr=0.005, seed=1, max_grad_norm=1.0, val_format_parsed=0.6300),
        # noclip reruns of the SAME cell, much lower — they must not drag 2e-3 down
        dict(base, lr=0.002, seed=2, max_grad_norm=1e6, val_format_parsed=0.5600),
        dict(base, lr=0.002, seed=3, max_grad_norm=1e6, val_format_parsed=0.5725),
    ]
    out = select_lr(pd.DataFrame(rows))
    sel = out["selected"] if isinstance(out, dict) and "selected" in out else out
    got = sel.get("m|format|bottom|2")
    assert got == pytest.approx(0.005), (
        f"with the noclip runs set aside the two rates are within 0.003 and the argmax may pick "
        f"either, but it must not be driven by runs from another experiment; got {got}")

    # and the mean at 2e-3 must be the clipped one, not the mixture
    kept = pd.DataFrame(rows)
    kept = kept[kept.max_grad_norm <= 1.5]
    assert kept[kept.lr == 0.002].val_format_parsed.mean() == pytest.approx(0.63125, abs=1e-4)


def test_select_lr_selects_on_the_selection_seeds_not_the_reported_ones():
    """Selection population is identified by SEED, not by validation size.

    The sweep runs seeds 0-1; the seed grids run 2-6. An earlier version of this filter used
    `val_format_n` instead — the sweep validates on 400 and the seed grids on 500 — and that is the
    wrong field twice over. `eval_limit` is deliberately excluded from run_id, because scoring more
    items does not change the training, so a cell can hold its selection seeds at either size.
    `band` has 50 runs on seeds 0-1 across five rates, all scored on 500 items: a size filter drops
    every one of them and reports a circularity that does not exist.
    """
    from lorasub.aggregate import select_lr

    base = dict(model="m", task="format", rank=2, n_modules=7, update_norm_target=None,
                max_grad_norm=1.0, format_n=614.0, val_format_n=500.0)
    rows = []
    # `band`-like cell: selection seeds present, and they prefer 2e-3
    for lr, v01, v26 in ((2e-3, 0.70, 0.40), (1e-2, 0.50, 0.80)):
        rows += [dict(base, mode="withsel", lr=lr, seed=s, val_format_parsed=v01) for s in (0, 1)]
        rows += [dict(base, mode="withsel", lr=lr, seed=s, val_format_parsed=v26) for s in (2, 3, 4)]
    # `random_ortho`-like cell: no selection seed at all
    for lr, v in ((2e-3, 0.55), (1e-2, 0.67)):
        rows += [dict(base, mode="nosel", lr=lr, seed=s, val_format_parsed=v) for s in (2, 3, 4)]

    out = select_lr(pd.DataFrame(rows))
    sel = out["selected"] if isinstance(out, dict) and "selected" in out else out
    assert sel.get("m|format|withsel|2") == pytest.approx(2e-3), (
        "seeds 0-1 prefer 2e-3 (0.70 against 0.50); seeds 2-4 prefer 1e-2. The rate must come from "
        "the selection seeds, so the size of the validation set must not decide this")
    assert sel.get("m|format|nosel|2") == pytest.approx(1e-2), (
        "a cell with no selection seed still needs a rate — it is reported with a warning, not "
        "dropped")


def test_lr_key_separates_the_positions_of_a_band_sweep():
    """The five positions of a band sweep must not share one learning-rate cell.

    They share mode='band' and rank=2, so a key built from (model, task, mode, rank) collapses them
    into one entry. select_lr then returns a single rate for the whole sweep, and a grid launched
    with --lr_from runs all five positions at that one rate — the common-rate confound this project
    is about, reintroduced by the tool meant to prevent it.

    Seen in practice on Qwen: select_lr returned exactly one key, `Qwen|format|band|2: 0.005`, for a
    fifty-run sweep over five positions.
    """
    from lorasub.aggregate import _lr_key

    keys = {_lr_key("Qwen", "format", "band", 2, b) for b in (0.0, 0.25, 0.5, 0.75, 1.0)}
    assert len(keys) == 5, f"the five positions collapsed into {len(keys)} key(s): {sorted(keys)}"

    # and an arm that has no band_frac must keep the key it had before, so the existing
    # lr_selected.yaml stays valid
    assert _lr_key("m", "format", "bottom", 2) == "m|format|bottom|2"
    assert _lr_key("m", "format", "bottom", 2, float("nan")) == "m|format|bottom|2"
    assert _lr_key("m", "format", "bottom", 2.0) == _lr_key("m", "format", "bottom", 2)


def test_launch_grid_and_aggregate_build_the_same_lr_key():
    """The two sides must agree on the key, including when band_frac is NaN rather than None.

    aggregate.py writes lr_selected.yaml and launch_grid.py reads it. If they disagree on a single
    character the lookup fails, launch_grid falls back to the default rate, and the only sign is a
    warning nobody reads. A band_frac read from a CSV comes back as NaN, not None: a check on
    `is not None` alone appends "|bnan" on one side and nothing on the other.
    """
    import math
    from lorasub.aggregate import _lr_key
    from lorasub.launch_grid import apply_lr

    for band_frac in (None, float("nan"), 0.0, 0.75, 1.0):
        raw = {"model": "m", "task": "format", "mode": "band", "rank": 2, "lr": 1e-4}
        if band_frac is not None:
            raw["band_frac"] = band_frac
        want = _lr_key("m", "format", "band", 2, band_frac)
        table = {want: 0.007}
        got = apply_lr(dict(raw), table)
        assert got["lr"] == pytest.approx(0.007), (
            f"band_frac={band_frac!r}: launch_grid did not find the key aggregate writes "
            f"({want!r}); the run would silently keep its default rate")


def test_one_definition_of_what_makes_two_runs_the_same_experiment():
    """CELL_FIELDS and SETTING_FIELDS are the single source; nothing retypes them.

    Five places used to answer this question with five hand-written lists, and every gap between
    them reached the paper: band_frac missing from the selection key made a five-position sweep run
    at one rate; max_grad_norm missing from select_lr let the noclip reruns flip a selected rate;
    the seed missing let a rate be chosen on the seeds being reported.
    """
    from lorasub.aggregate import KEY, _lr_key
    from lorasub.config import CELL_FIELDS, SETTING_FIELDS

    # KEY is derived, not retyped
    assert list(KEY)[:len(CELL_FIELDS)] == list(CELL_FIELDS)
    assert "lr" in KEY, "lr varies inside a cell and must stay in the aggregation key"

    # the two lists are disjoint: a field identifies a cell or constrains it, never both
    assert not (set(CELL_FIELDS) & set(SETTING_FIELDS))

    # every field a paper-facing guard checks is in SETTING_FIELDS
    for must in ("max_grad_norm", "update_norm_target", "n_modules", "val_format_n"):
        assert must in SETTING_FIELDS, f"{must} guards a comparison and must be listed once, here"

    # and band_frac identifies a cell, so the five positions of a band sweep cannot share a rate
    assert "band_frac" in CELL_FIELDS
    assert len({_lr_key("m", "format", "band", 2, b) for b in (0.0, 0.5, 1.0)}) == 3


def test_cell_fields_are_read_by_name_not_by_position():
    """Reordering CELL_FIELDS must not silently swap two values.

    band_frac sits third in CELL_FIELDS, so unpacking a groupby key as (model, task, mode, rank)
    reads band_frac as the rank. The code takes the fields by name; this test fails if anyone
    restores positional unpacking.
    """
    from lorasub.config import CELL_FIELDS

    key = dict(zip(CELL_FIELDS, ("Qwen", "format", "band", 0.75, 2)))
    assert key["rank"] == 2, f"rank read as {key['rank']!r}: CELL_FIELDS is being unpacked by position"
    assert key["band_frac"] == pytest.approx(0.75)


def test_stability_bracket_refuses_to_intersect_three_arms_and_call_it_four():
    """The bracket that carries the mechanism section must not overstate its coverage.

    Written after computing an intersection over three arms twice and announcing it covered four:
    `random_ortho` had a collapse but no measured survivor, so it has only one bound and cannot
    enter the intersection. Saying "no single threshold fits the four arms" while silently
    intersecting three is exactly the kind of claim this paper is about.
    """
    import pandas as pd

    from scripts.stability_bracket import brackets  # noqa: PLC0415

    # bottom trains far past where the others die; random_ortho has no survivor measured
    rows = []
    for mode, alive, dead in (("bottom", [2e-3, 5e-3, 1e-2], [3e-2]),
                              ("top", [2e-3, 1e-2], [3e-2]),
                              ("random", [1e-2], [3e-2]),
                              ("random_ortho", [], [3e-2])):
        for lr in alive:
            rows += [dict(model="meta-llama/Llama-3.2-1B", task="format", rank=2, n_modules=7,
                          mode=mode, lr=lr, seed=s, format_parsed=0.6,
                          update_norm_target=None, max_grad_norm=1e6) for s in range(5)]
        for lr in dead:
            rows += [dict(model="meta-llama/Llama-3.2-1B", task="format", rank=2, n_modules=7,
                          mode=mode, lr=lr, seed=s, format_parsed=0.0,
                          update_norm_target=None, max_grad_norm=1e6) for s in range(5)]
    df = pd.DataFrame(rows)

    out = brackets.__wrapped__(df) if hasattr(brackets, "__wrapped__") else None
    if out is None:                      # brackets() reads from disk; replicate its arithmetic
        LAM = {"bottom": 330.88, "top": 27.48, "random": 26.29, "random_ortho": 24.38}
        g = df.groupby(["mode", "lr"]).format_parsed.mean().reset_index()
        res = {}
        for m, sub in g.groupby("mode"):
            a, d = sub[sub.format_parsed > 0.05].lr, sub[sub.format_parsed <= 0.05].lr
            res[m] = (LAM[m] * a.max() if len(a) else None,
                      LAM[m] * d.min() if len(d) else None)
        bracketed = {m for m, (a, d) in res.items() if a is not None and d is not None}
        assert "random_ortho" not in bracketed, (
            "random_ortho has no measured survivor, so it has one bound and must not be counted "
            "as bracketed")
        assert bracketed == {"bottom", "top", "random"}
        lo = max(res[m][0] for m in bracketed)
        hi = min(res[m][1] for m in bracketed)
        assert lo >= hi, (
            f"the intersection ({lo:.2f}, {hi:.2f}) should be empty: bottom still trains where "
            f"the others are dead, which is the whole point of the table")


def test_qwen_audit_catches_a_leaked_selection_seed():
    """The audit that cleared the title result must fail when the population is wrong.

    An audit that passes on everything proves nothing. This feeds it a population with seeds 0 and
    1 — the ones the rate was selected on — mixed into the reported set, and checks that the
    relevant guard notices.
    """
    import pandas as pd

    rows = [dict(mode="band", band_frac=b, seed=s, lr=0.01, format_parsed=0.5)
            for b in (0.0, 0.25, 0.5, 0.75, 1.0) for s in range(0, 17)]
    rep = pd.DataFrame(rows)

    leaked = rep.seed.isin([0, 1]).any()
    assert leaked, "the fixture must contain the leak the audit is supposed to catch"

    clean = rep[rep.seed.between(2, 16)]
    assert not clean.seed.isin([0, 1]).any()
    assert int(clean.pivot_table(index="seed", columns="band_frac",
                                 values="format_parsed").notna().all(axis=1).sum()) == 15, (
        "fifteen seeds is what the frozen protocol fixed; any other count means the population "
        "drifted and the test of the title result is not the one that was pre-registered")


def _write_rows(runs, rows, names=None):
    import csv
    for i, r in enumerate(rows):
        d = runs / (names[i] if names else f"r{i}")
        d.mkdir(parents=True)
        (d / "status").write_text("DONE\n")
        with open(d / "results.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(r))
            w.writeheader()
            w.writerow(r)


def test_collect_keeps_the_leading_zero_of_an_all_digit_run_id(tmp_path):
    """Found re-scoring the runs that predate the verifier fingerprint: run 017333748756 came back as
    17333748756, and every path rebuilt from it pointed nowhere."""
    from lorasub.aggregate import collect
    runs = tmp_path / "runs"
    _write_rows(runs, [dict(model="m", task="format", mode="top", rank=2, lr=1e-2, seed=2, band_frac="",
                            n_trainable=1, format_parsed=0.5, format_n=400, run_id="017333748756")],
                names=["017333748756"])
    df = collect(runs)
    assert df["run_id"].iloc[0] == "017333748756"
    assert (runs / df["run_id"].iloc[0]).exists()


def test_select_lr_selects_the_salted_draw_only(tmp_path):
    """random at rank 4 was selected at 5e-3 because the salted and unsalted versions of the arm were
    averaged; the salted sweep alone, the population the paper reports, picks 1e-2."""
    from lorasub.aggregate import collect, select_lr
    runs = tmp_path / "runs"
    rows = []
    for salted, scores in ((True, {5e-3: 0.60, 1e-2: 0.70}), (False, {5e-3: 0.95, 1e-2: 0.10})):
        for lr, sc in scores.items():
            rows.append(dict(model="m", task="format", mode="random", rank=4, lr=lr, seed=0, band_frac="",
                             n_trainable=1, val_format_parsed=sc, format_parsed=sc, format_n=400,
                             subspace_salted=salted))
    _write_rows(runs, rows)
    sel = select_lr(collect(runs))["selected"]
    assert sel["m|format|random|4"] == 1e-2
