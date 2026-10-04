"""Aggregation of finished runs.

    python -m lorasub.aggregate --runs_dir $SHARED/runs --out $SHARED/analysis [--select_lr configs/lr_selected.yaml]

* ``collect``      : concatenates every ``<run>/results.csv`` (individual files only; FAILED runs listed, not used).
* ``select_lr``    : per (model, task, mode, rank), the lr with the best *validation* metric; writes lr_selected.yaml.
* ``summarize``    : mean ± std over seeds, grouped by (model, task, mode, band_frac, rank, lr) — lr is part of the
                     key so that different learning rates are never averaged together.
* ``crossed_table``: LaTeX table arms x tasks for one model/rank, *including* full FT and the baseline.
* ``plot_crossed`` : bars with std, one panel per rank.
* ``plot_spectral_curve`` : primary metric vs band_frac (0 = top ... 1 = bottom), with top/bottom/random/free as
                     horizontal references.
* ``forgetting_delta`` : forgetting metrics minus the un-adapted ``baseline`` run of the same model.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .config import CELL_FIELDS, SETTING_FIELDS
from .lora import SALTED_MODES
import yaml

from .eval import primary_metric

# Every mode that can appear in a table or a figure. A mode missing here is filtered out by
# crossed_table, plot_lr_heatmap, plot_crossed and forgetting_delta WITHOUT any error: its runs
# simply never appear. The dual family was added to lora.MODES without being added here, so the
# twenty runs meant to decide H1 would have vanished from every output silently.
ARMS = ["free", "top", "bottom", "random", "random_ortho", "band",
        "dual_top", "dual_bottom", "dual_random", "dual_random_ortho",
        "init_top", "init_bottom", "init_random", "top_sigma", "top_unfrozen", "top_scalar", "bottom_sigma", "bottom_unfrozen", "top_lrscale", "full", "baseline"]
# `subspace_salted` is part of the grouping key, not a detail: the run id separates a `random` arm
# drawn independently per module from one drawn once and applied everywhere, but WITHOUT IT IN KEY the
# aggregation puts them back in the same cell and averages two different arms — in summarize, in the
# crossed table, in the win matrix and in the rate map. The identifier would separate the populations
# while the analysis recombined them, which is the exact failure the field was added to prevent.
# Derived from CELL_FIELDS, not retyped: the two drifted apart once already, when band_frac was
# in this list but not in the selection key.
KEY = list(CELL_FIELDS) + ["lr", "subspace_salted"]


def collect(runs_dir: str | Path) -> pd.DataFrame:
    """Every results.csv, plus the three config fields that decide WHICH EXPERIMENT a run belongs to.

    `results.csv` records what a run measured; it does not record the three settings that make two
    runs incomparable even when mode, rank, lr and seed match:

    * ``update_norm_target`` — the magnitude ablation rescales the final update. Those runs answer a
      different question and their scores are 0.000 or 0.65 depending on the target.
    * ``max_grad_norm``     — the noclip grid reruns the same cell without clipping.
    * ``target_modules``    — the localisation grid reruns it on five projections instead of seven.

    None of the three was read here, so a cell (format, bottom, r=2) held the five normal runs, the
    five rescaled to 1.5, the five rescaled to 4.5, the unclipped ones and the five-module ones —
    and `crossed_table` averaged all of them. Every hand-written query in this project had to filter
    them manually, which is exactly the kind of thing a person forgets once.
    """
    rows = []
    for p in sorted(Path(runs_dir).glob("*/results.csv")):
        status = (p.parent / "status").read_text() if (p.parent / "status").exists() else "DONE"
        if status.startswith("FAILED"):
            continue
        # run_id as TEXT from the start: an all-digit identifier read as an integer loses its leading
        # zero, and a path rebuilt from it no longer exists (found re-scoring 292 runs, 21/09).
        df = pd.read_csv(p, dtype={"run_id": str})
        cfg_p = p.parent / "config.yaml"
        if cfg_p.exists():
            try:
                cfg = yaml.safe_load(cfg_p.read_text()) or {}
            except Exception:
                cfg = {}
            # `lr` decides which rows survive _keep_selected_lr. When results.csv does not carry it,
            # every value is NaN, `float(nan) == float(2e-3)` is False, and the filter silently drops
            # EVERY run of every cell that has a selected rate — leaving only the cells absent from
            # the table. That is how plot_crossed came to draw a single arm out of five.
            if "lr" not in df.columns or df["lr"].isna().all():
                df["lr"] = cfg.get("lr")
            df["update_norm_target"] = cfg.get("update_norm_target")
            df["max_grad_norm"] = cfg.get("max_grad_norm")
            df["n_modules"] = len(cfg.get("target_modules") or [])
        rows.append(df)
    if not rows:
        return pd.DataFrame(columns=KEY)
    df = pd.concat(rows, ignore_index=True)
    # a truncated or empty results.csv leaves a row with no model: it would break every groupby and
    # silently pollute the averages
    # A results.csv written with shifted columns puts a number where the model id belongs; the row
    # then survives every filter and creates phantom "models" like 497.7 that pollute every groupby.
    # A model id must be a non-empty string containing no digit-only value.
    # `run_id` comes from results.csv, and pandas types a column by its contents: an id made only
    # of hex digits that happen to all be decimal (254002041782) is read as an INTEGER while its
    # neighbours (9303ba2164a8) stay strings. Any later `.map()` against a dict keyed by directory
    # name then misses those runs and fills their config fields with NaN — silently, so they simply
    # vanish from every filter on n_modules or update_norm_target. Six runs out of 991 were affected
    # here, and one of them was a cell of the main rank table.
    if "run_id" in df:
        df["run_id"] = df["run_id"].astype(str)

    if "model" in df:
        def _is_model(v) -> bool:
            return isinstance(v, str) and bool(v.strip()) and not v.strip().replace(".", "", 1).isdigit()

        bad = ~df["model"].map(_is_model)
        if bad.any():
            shown = sorted({repr(v) for v in df.loc[bad, "model"].unique()})[:5]
            print(f"[aggregate] {int(bad.sum())} malformed row(s) dropped: model column held "
                  f"{', '.join(shown)} — a results.csv has shifted columns, check those runs")
            df = df[~bad]
    # runs produced before the field exists are the unsalted arm by definition
    if "subspace_salted" not in df:
        df["subspace_salted"] = False
    # a CSV round-trip can yield the STRING "False", which astype(bool) turns into True and flips the
    # arm; map explicitly instead
    df["subspace_salted"] = df["subspace_salted"].map(
        {True: True, "True": True, "true": True, 1: True, "1": True,
         False: False, "False": False, "false": False, 0: False, "0": False})
    # `where` rather than `fillna`: it is the fillna of an OBJECT column that pandas deprecates, and
    # infer_objects afterwards does not silence it. Fourth time this warning came back, which each
    # time meant the file had drifted between the workstation and the cluster.
    _s = df["subspace_salted"]
    df["subspace_salted"] = _s.where(_s.notna(), False).astype(bool)
    # AND normalised to False wherever the arm draws nothing. `top`, `bottom`, `free`, `band` and
    # `full` have deterministic bands: the flag is written on their runs but describes nothing about
    # them. Left as recorded, it split them in two the moment the default flipped to True — the three
    # `top` runs produced before carried False, the two produced after carried True, and KEY gave
    # them separate rows in summary.csv (n=3 and n=2) for five runs that measured the same thing.
    # `check_population` and `_refuse_if_mixed` already scope the comparison; doing it here as well
    # fixes KEY and every other groupby at once, and states the rule where it belongs: on an arm that
    # draws nothing, the salt is not a property of the run.
    if "mode" in df:
        df["subspace_salted"] = df["subspace_salted"] & df["mode"].isin(list(SALTED_MODES))
    if "band_frac" not in df:
        df["band_frac"] = float("nan")
    df["band_frac"] = df["band_frac"].astype(float)
    return df


def _metric(task: str, val: bool = False) -> str:
    m = primary_metric(task)
    return f"val_{m}" if val else m


def select_population(df: "pd.DataFrame", salted: bool = True) -> "pd.DataFrame":
    """Keep ONE version of each arm, at the entry point, before any analysis runs.

    Why this rather than fixing each function. Adding `subspace_salted` to KEY fixed `summarize` and
    nothing else: seven functions that actually produce the paper group on `mode` alone --
    `crossed_table`, `plot_lr_heatmap`, `win_matrix`, `plot_crossed`, `plot_failure_decomposition`,
    `forgetting_delta`, `gradient_vs_performance`. They would each have averaged the old `random` arm
    (one spectral position applied everywhere) with the new one (drawn independently per module).
    `win_matrix` was the worst: at fixed seed, `aggfunc="mean"` merges a seed's two random runs into
    one number, destroying the very pairing the function exists for.

    Filtering once, here, is both shorter and safer than fixing eight call sites -- and it cannot be
    forgotten by the ninth function someone writes next month.
    """
    if "subspace_salted" not in df or df.empty:
        return df
    is_random = df["mode"].isin(list(SALTED_MODES))
    keep = (~is_random) | (df["subspace_salted"] == bool(salted))
    dropped = int((~keep).sum())
    if dropped:
        print(f"[aggregate] {dropped} run(s) of the other random arm set aside "
              f"(subspace_salted != {salted}); they measure a different arm and are never averaged in")
    return df[keep]


def _keep_selected_lr(d: "pd.DataFrame", lr_table: dict | None) -> "pd.DataFrame":
    """Keep only the rows trained at the learning rate selected for their arm.

    Uses a boolean Series aligned on the index, never a plain list: an EMPTY list makes pandas select
    zero COLUMNS instead of zero rows, the frame then loses `mode`, and every downstream groupby fails
    with KeyError. That is how this crashed on real data.
    """
    if not lr_table or d.empty:
        return d

    def _ok(r) -> bool:
        key = _lr_key(r["model"], r["task"], r["mode"], r["rank"], r.get("band_frac"))
        if key not in lr_table:
            return True
        try:
            return float(r["lr"]) == float(lr_table[key])
        except (TypeError, ValueError):
            return True

    missing = sorted({_lr_key(r["model"], r["task"], r["mode"], r["rank"], r.get("band_frac"))
                      for _, r in d.iterrows()} - set(lr_table))
    if missing:
        # These cells keep EVERY swept rate and get averaged together, silently. That is not a
        # comparison at each arm's optimum; it is a mean over an arbitrary set of learning rates.
        print(f"[aggregate] WARNING {len(missing)} cell(s) have no selected learning rate and will "
              f"keep all their rates: {', '.join(missing[:4])}{'...' if len(missing) > 4 else ''}")
    mask = pd.Series([_ok(r) for _, r in d.iterrows()], index=d.index, dtype=bool)
    # A cell whose every row is dropped is not a filter doing its job: it means the rows carry no
    # usable lr — an absent column, or NaN — and comparing NaN to the selected rate is always False.
    # Silently returning an empty frame turns that into a figure with a missing arm.
    for (mdl, tsk, md, rk), g in d.groupby(["model", "task", "mode", "rank"], dropna=False):
        key = _lr_key(mdl, tsk, md, rk)
        if key in lr_table and not mask.loc[g.index].any():
            got = sorted({str(v) for v in g["lr"].head(4)})
            print(f"[aggregate] *** {key}: the selected rate {lr_table[key]:g} matches NONE of "
                  f"{len(g)} runs (their lr reads {got}). The cell is dropped entirely — check that "
                  f"`lr` reaches the frame, or the figure will be missing this arm.")
    return d[mask]


def _lr_key(model, task, mode, rank, band_frac=None) -> str:
    """Key of the per-arm learning-rate table. The rank is normalised to an int: a CSV read gives
    1.0 where a YAML gives 1, and the two must not become two different cells.

    `band_frac` is part of the key. Without it the five positions of a band sweep share one cell
    (mode='band', rank=2) and collapse to a single rate, so a grid launched with --lr_from would
    run all five positions at the SAME rate — the confound this project is about. Seen in
    practice: select_lr returned one rate, 0.005, for the whole Qwen band sweep.
    """
    try:
        rank = int(float(rank))
    except (TypeError, ValueError):
        pass
    key = f"{model}|{task}|{mode}|{rank}"
    if band_frac is not None and band_frac == band_frac:   # not NaN
        key += f"|b{float(band_frac):g}"
    return key


def select_lr(df: pd.DataFrame, out_path: str | Path | None = None) -> dict:
    """Best lr per (model, task, mode, rank) on the validation metric (mean over seeds)."""
    sel: dict[str, float] = {}
    edge_warnings: list[str] = []
    imbalance_notes: list[str] = []
    rows = []

    # The cell key is (model, task, mode, rank), and that is NOT enough to make a population. The
    # same cell also holds the noclip reruns (max_grad_norm 1e6), the rescaled arms
    # (update_norm_target set), the span/budget controls (5 or 2 modules instead of 7) and, for the
    # drawing arms, both the salted and unsalted version. Averaging a validation score over that
    # mixture and taking an argmax compares rates measured on different experiments — the exact
    # fault this project reports in the literature it discusses.
    #
    # This was found when a rate looked 1.76 sigma better than its neighbour and the margin turned
    # out to come from two noclip runs (0.560, 0.573) sitting in a cell whose clipped runs were at
    # 0.63. Paired on the seeds the two rates actually share, the gap was 0.0013.
    # What separates the selection population from the reported one is the SEED, not the
    # validation size. The sweep runs seeds 0-1; the seed grids run 2-6. Filtering on
    # `val_format_n` instead looked right — the sweep validates on 400 and the seed grids on 500 —
    # but it is the wrong field twice over: `eval_limit` is deliberately excluded from run_id
    # (changing how many items you score does not change the training), so a cell can hold its
    # selection seeds at either size. `band` has 50 runs on seeds 0-1 across five rates, all scored
    # on 500 items: a size filter drops them and reports a circularity that does not exist.
    #
    # Selecting on seeds 0-1 when they are present is the rule the paper states. When a cell has
    # none of them, its rate really was chosen on the reported seeds, and that must be said rather
    # than hidden.
    SELECTION_SEEDS = {0, 1}
    _gk0 = [k for k in CELL_FIELDS if k in df]
    if "seed" in df:
        for key, g in df.groupby(_gk0, dropna=False):
            if not g["seed"].isin(SELECTION_SEEDS).any():
                _v = dict(zip(_gk0, key if isinstance(key, tuple) else (key,)))
                mdl, tsk, md, rk = (_v.get("model"), _v.get("task"), _v.get("mode"), _v.get("rank"))
                print(f"[aggregate] *** {_lr_key(mdl, tsk, md, rk, _v.get('band_frac'))}: "
                      f"no run on the selection seeds "
                      f"{sorted(SELECTION_SEEDS)} — its rate was chosen on the seeds whose results "
                      f"are reported. Sweep it on seeds 0-1, or say so in the paper.")
        # keep the selection seeds where they exist, and everything else where they do not
        # Keeping only the selection seeds is right when they cover the swept rates. When they do
        # not — a cell where seeds 0-1 exist at one rate only — the filter leaves a single rate and
        # there is nothing left to choose: the argmax becomes whichever rate happened to run on
        # seed 0. Use the whole cell there instead, and say so; the common-seed pairing below then
        # makes the comparison it was written for.
        # `transform` keeps the original index; `apply` returning a Series does not, and pandas
        # then refuses the boolean mask with "cannot join with no overlapping index names".
        _in_sel = df["seed"].isin(SELECTION_SEEDS)
        usable = (df["lr"].where(_in_sel)
                    .groupby([df[k] for k in _gk0], dropna=False)
                    .transform("nunique") >= 2)

        def _usable(g):
            sel = g[g["seed"].isin(SELECTION_SEEDS)]
            return len(sel) > 0 and sel["lr"].nunique() >= 2
        for key, g in df.groupby(_gk0, dropna=False):
            if not _usable(g) and g["seed"].isin(SELECTION_SEEDS).any() and g["lr"].nunique() > 1:
                _v = dict(zip(_gk0, key if isinstance(key, tuple) else (key,)))
                _k = _lr_key(_v.get("model"), _v.get("task"), _v.get("mode"),
                             _v.get("rank"), _v.get("band_frac"))
                _ns = g.loc[g["seed"].isin(SELECTION_SEEDS), "lr"].nunique()
                print(f"[aggregate] {_k}: the selection seeds cover only {_ns} of the "
                      f"{g['lr'].nunique()} swept rates, so the whole cell is used and the rates "
                      f"are compared on the seeds they share.")
        df = df[(~usable) | df["seed"].isin(SELECTION_SEEDS)]

    keep = pd.Series(True, index=df.index)
    for col, want in (("max_grad_norm", lambda v: v.isna() | (v <= 1.5)),
                      ("update_norm_target", lambda v: v.isna())):
        if col in df:
            keep &= want(df[col])
    if "n_modules" in df:
        # keep the modal module count per model: the controls that adapt fewer modules are a
        # different experiment, not a different rate
        modal = df.loc[keep].groupby("model")["n_modules"].agg(
            lambda x: x.value_counts().idxmax() if len(x.dropna()) else None)
        keep &= df.apply(lambda r: r.get("n_modules") == modal.get(r.get("model")), axis=1)
    # the drawing arms exist in a salted (one subspace per module) and an unsalted version; they are
    # different experiments, and averaging both picked 5e-3 for random at rank 4 where the salted
    # sweep alone picks 1e-2 (21/09). Select on the salted version, the one the tables report.
    if "subspace_salted" in df and "mode" in df:
        _drawn = df["mode"].isin(list(SALTED_MODES))   # the one list, never a second copy
        keep &= ~_drawn | (df["subspace_salted"].astype(str).str.lower() == "true")
    dropped = int((~keep).sum())
    if dropped:
        print(f"[aggregate] select_lr: {dropped} run(s) set aside — noclip, rescaled, or a "
              f"non-standard module count. A learning rate must be chosen within one population.")
    df = df[keep]

    # band_frac joins the grouping key: the five positions of a band sweep share mode='band' and
    # rank=2, and without it they collapse into one cell and receive one rate. dropna=False keeps
    # every non-band arm, whose band_frac is NaN.
    _gk = [k for k in CELL_FIELDS if k in df]
    for _key, g in df.groupby(_gk, dropna=False):
        # By NAME, not by position: CELL_FIELDS lists band_frac third, and unpacking `_key[:4]` as
        # (model, task, mode, rank) silently read band_frac as the rank the moment the list moved.
        _v = dict(zip(_gk, _key if isinstance(_key, tuple) else (_key,)))
        model, task, mode = _v.get("model"), _v.get("task"), _v.get("mode")
        rank, band_frac = _v.get("rank"), _v.get("band_frac")
        col = _metric(task, val=True)
        n_col = "val_mcq_n" if task == "facts" else "val_format_n"
        if n_col in g:
            sizes = sorted({x for x in g[n_col].dropna().unique()})
            if len(sizes) > 1:
                # min(eval_limit, 500): the sweep validates on 400 items, the seed grid on 500. An
                # argmax over a column that mixes them compares scores on different sets.
                edge_warnings.append(
                    f"{model} | {task} | {mode} | r={rank}: validation sizes {sizes} are MIXED; the "
                    f"argmax compares scores measured on different sets — selection not trustworthy")
        if col not in g or g[col].isna().all():
            continue
        # BALANCE THE CELL BEFORE THE ARGMAX. Otherwise the most populated rate wins partly because
        # it is the most populated: on the free arm at r=2, 1e-3 held six or seven runs against two
        # elsewhere, its mean drifted from 0.841 to 0.808 as relaunches piled up, and the selected
        # optimum moved from 1e-3 to 2e-3. That rate is what goes into --lr_from and into the seed
        # grid, so an artefact of run counts would propagate to every number of the paper.
        counts = g.groupby("lr")[col].count()
        n_min = int(counts.min()) if len(counts) else 0
        if n_min < 2 and len(counts) > 1 and counts.max() > n_min:
            # sub-sampling to a single run would replace an imbalance by a measurement without
            # variance: the argmax would rest on one seed. Better to keep the imbalance and say so.
            imbalance_notes.append(
                f"{model} | {task} | {mode} | r={rank}: one rate has a SINGLE run "
                f"({n_min}..{int(counts.max())}); cell NOT balanced, selection is unreliable here")
            balanced = g
        elif len(counts) > 1 and counts.max() > n_min:
            # Take the seeds COMMON to every rate, not the first n_min of each. Sub-sampling by
            # position can hand 1e-3 the seeds {0,1} and 2e-3 the seeds {2,3}: the two rates are then
            # compared on disjoint seeds and seed variance re-enters the argmax through the back door.
            # A common seed set makes the comparison between rates PAIRED.
            per_lr = g.groupby("lr")["seed"].apply(lambda x: set(x.dropna()))
            common = set.intersection(*per_lr) if len(per_lr) else set()
            if common:
                balanced = g[g["seed"].isin(common)]
                imbalance_notes.append(
                    f"{model} | {task} | {mode} | r={rank}: run counts per rate ranged "
                    f"{n_min}..{int(counts.max())}; compared on the {len(common)} seed(s) common to "
                    f"every rate ({sorted(common)}), so the comparison is paired")
            else:
                srt = g.sort_values(["lr", "seed"])
                balanced = srt[srt.groupby("lr").cumcount() < n_min]
                imbalance_notes.append(
                    f"{model} | {task} | {mode} | r={rank}: NO seed common to every rate; fell back "
                    f"to the first {n_min} run(s) of each — the comparison is NOT paired")
        else:
            balanced = g
        # The cell must validate on ONE population. The validation split is capped at
        # min(eval_limit, 500), so a sweep run at eval_limit=400 validates on 400 items while a
        # seed-grid run at 6000 validates on 500 — and this function reads exactly those columns.
        n_val = float("nan")
        for ncol in ("val_format_n", "val_mcq_n"):
            if ncol in balanced and balanced[ncol].notna().any():
                sizes = sorted(balanced[ncol].dropna().unique())
                n_val = float(sizes[0])
        m = balanced.groupby("lr")[col].agg(["mean", "std", "count"]).reset_index()
        ranked = m.sort_values("mean", ascending=False)
        best = ranked.iloc[0]
        # An argmax is only a choice if it beats the runner-up by more than the noise. On bottom r=2
        # the top two rates sat 0.0013 apart on validation and the selected rate flipped from 2e-3 to
        # 5e-3 when one run landed — which silently made every seed-grid run of that arm run
        # off-optimum. Report the margin so a coin flip cannot pass for a measurement.
        if len(ranked) > 1:
            runner = ranked.iloc[1]
            margin = float(best["mean"]) - float(runner["mean"])
            se = binomial_se(float(best["mean"]), n_val)
            if se == se and margin < se:
                imbalance_notes.append(
                    f"{model} | {task} | {mode} | r={rank}: lr={best['lr']:g} beats lr={runner['lr']:g} "
                    f"by only {margin:.4f} on validation (binomial SE {se:.4f} at n={n_val:.0f}); the "
                    f"argmax is WITHIN NOISE — treat the two rates as tied and measure both before "
                    f"reporting")
            elif se != se:
                # no validation size recorded: the margin cannot be judged, and saying nothing would
                # be worse than saying we cannot tell
                imbalance_notes.append(
                    f"{model} | {task} | {mode} | r={rank}: no validation size recorded, so the "
                    f"{margin:.4f} margin between lr={best['lr']:g} and lr={runner['lr']:g} cannot be "
                    f"compared to the sampling noise")
        sel[_lr_key(model, task, mode, rank, band_frac)] = float(best["lr"])
        # An optimum at the edge of the swept range is not an optimum: the true one lies outside, and
        # the ranking read at that point may be a ranking of under-trained arms. Silent before; it
        # took an outside reader to notice that every constrained arm had selected the top rate.
        swept = sorted(float(x) for x in g["lr"].dropna().unique())
        if len(swept) > 1 and float(best["lr"]) in (swept[0], swept[-1]):
            side = "lowest" if float(best["lr"]) == swept[0] else "highest"
            edge_warnings.append(f"{model} | {task} | {mode} | r={rank}: selected the {side} rate "
                                 f"swept ({best['lr']:.0e} of {[f'{x:.0e}' for x in swept]}) — the "
                                 f"optimum is at the GRID EDGE, extend the sweep before selecting")
        for _, r in m.iterrows():
            rows.append(dict(model=model, task=task, mode=mode, rank=rank, lr=r["lr"], val_mean=r["mean"],
                             val_std=r["std"], n=r["count"], selected=float(r["lr"]) == float(best["lr"])))
    for w in imbalance_notes:
        print(f"[aggregate] {w}")
    for w in edge_warnings:
        print(f"[aggregate] *** {w}")
    if out_path:
        with open(out_path, "w") as f:
            yaml.safe_dump({"metric": "validation primary metric, mean over seeds", "selected": sel}, f)
    return {"selected": sel, "table": pd.DataFrame(rows)}


def infer_metric_version(df: "pd.DataFrame") -> "pd.Series":
    """Which version of the format checker scored each run, DEDUCED from the numbers themselves.

    The strict definition (since 11/09) requires `order_ok`, so it forces `parsed <= order_ok` on every
    run it scored. Any run with `parsed > order_ok` was therefore scored by the permissive version,
    where sorting was computed but not required. This is a proof internal to the data, not an inference
    from file dates — and dates would have been the weak evidence here, since the format task was
    rebuilt three times on 10-11/09 and mtime only records the last of them.

    THE DETECTION IS ONE-WAY, and the labels say so. `parsed > order_ok` PROVES permissive: the strict
    checker cannot produce it. The converse does not hold — a permissive run whose valid outputs all
    happened to be sorted satisfies `parsed <= order_ok` too, and is indistinguishable from a strict
    one by this test. Calling that case "strict" would be an over-claim, so it is labelled
    `strict-or-undetected`: consistent with the strict definition, not proof of it.

    Returns "permissive", "strict-or-undetected", or "unknown" when order_ok is absent.
    """
    if "format_parsed" not in df or "format_order_ok" not in df:
        return pd.Series(["unknown"] * len(df), index=df.index)
    out = pd.Series("unknown", index=df.index, dtype=object)
    ok = df["format_parsed"].notna() & df["format_order_ok"].notna()
    out[ok] = "strict-or-undetected"
    out[ok & (df["format_parsed"] > df["format_order_ok"] + 1e-9)] = "permissive"
    return out


def effective_metric_version(df: "pd.DataFrame") -> "pd.Series":
    """The recorded digest when present, the deduced version otherwise.

    `metric_version` is written by train.py from a hash of `check_alien`'s source, so it cannot drift
    the way a hand-maintained number would. Runs produced before the field existed have no digest;
    for those we fall back on `infer_metric_version`, which deduces the version from the numbers
    themselves. The two are reported under distinct labels so a mixed population is still caught.
    """
    deduced = infer_metric_version(df)
    if "metric_version" not in df:
        return deduced
    rec = df["metric_version"].map(lambda v: str(v).strip() if v == v and v is not None else "")
    return rec.where(rec != "", deduced.map(lambda v: f"pre-field:{v}"))


def check_metric_versions(df: pd.DataFrame) -> list[str]:
    """Refuse to stay silent when one task was scored by two versions of the checker.

    `format_parsed` was hardened on 11/09 to require sorted lists; before that a run with unsorted
    lists counted as valid. Nothing records the checker version — not results.csv, not the run id —
    so a cell can average two definitions of the same metric without any trace. The data fingerprint
    does not help: it tracks the DATA, not the scorer.
    """
    msgs = []
    if "format_parsed" not in df:
        return msgs
    v = effective_metric_version(df)
    for (model, task), idx in df.groupby(["model", "task"]).groups.items():
        seen = {x for x in v.loc[idx] if x != "unknown"}
        if len(seen) > 1:
            msgs.append(f"{model} / {task}: TWO CHECKER VERSIONS mixed {sorted(seen)} — these runs "
                        f"were not scored by the same metric; do NOT average them.")
    return msgs


def check_data_versions(df: pd.DataFrame) -> list[str]:
    """Refuse to stay silent when one task was measured on two different dataset versions.

    The format task was rebuilt three times and the facts task twice. `data_dir` is excluded from the
    run id, so a run produced on an earlier version is skipped as "already done" and averaged with the
    current one. The fingerprint recorded by train.py makes that detectable after the fact.
    """
    msgs = []
    if "data_fingerprint" not in df:
        return msgs
    for (model, task), g in df.groupby(["model", "task"]):
        fps = sorted(str(x) for x in g["data_fingerprint"].dropna().unique() if str(x).strip())
        if len(fps) > 1:
            per_arm = g.groupby("mode")["data_fingerprint"].agg(lambda x: sorted(set(map(str, x))))
            msgs.append(f"{model} / {task}: TWO DATASET VERSIONS mixed {fps} — per arm {dict(per_arm)}. "
                        f"These runs did not see the same data; do NOT average them.")
    return msgs


def check_eval_sizes(df: pd.DataFrame) -> list[str]:
    """Cells compared against each other must have been evaluated on the same number of examples.

    ``eval_limit`` is deliberately excluded from ``run_id`` (it does not change what was learned), but
    that means it can drift between relaunches of the same grid: a cell finished before the change is
    kept, one finished after uses a different test subset, and the two are then averaged or compared
    as if they were the same measurement. This happened during the rank sweep — caught here rather
    than in the paper. The eval size is recorded per run (``mcq_n``, ``format_n``); this check refuses
    to stay silent when it differs within a task.
    """
    msgs = []
    for (model, task), g in df.groupby(["model", "task"]):
        col = "mcq_n" if task == "facts" else "format_n"
        if col not in g or g[col].isna().all():
            continue
        arms = g[g["mode"].isin([a for a in ARMS if a not in ("full", "baseline")])]
        sizes = sorted(arms[col].dropna().unique())
        if len(sizes) > 1:
            per_arm = arms.groupby("mode")[col].agg(lambda x: sorted(set(x)))
            msgs.append(f"{model} / {task}: evaluated on different numbers of examples {sizes} — "
                        f"per arm {dict(per_arm)}. Do NOT compare these cells; re-evaluate at a "
                        f"single eval_limit.")
    return msgs


def binomial_se(p: float, n: float) -> float:
    """Sampling standard error of a rate measured on n examples, sqrt(p(1-p)/n).

    The seed-to-seed standard deviation does not contain it: two seeds can agree perfectly and still
    be 2 points from the truth if the test set is small. With 400 examples and p = 0.3 this is 2.3
    points — larger than several of the gaps we would otherwise be tempted to call differences.
    """
    if not n or n <= 0 or pd.isna(p):
        return float("nan")
    return float((p * (1 - p) / n) ** 0.5)


def check_saturation(df: pd.DataFrame, high: float = 0.95, spread: float = 0.03) -> list[str]:
    """Warn when a task no longer discriminates: every arm above ``high`` and a spread below ``spread``.

    A crossed table where the four arms sit within three points of 100 % measures nothing about the
    subspace; it measures that the task is too easy. Better to see it here than in the paper."""
    msgs = []
    _adapted = [a for a in ARMS if a not in ("full", "baseline")]
    for (model, task), g in df[df["mode"].isin(_adapted)].groupby(["model", "task"]):
        m = _metric(task)
        if m not in g:
            continue
        per_arm = g.groupby("mode")[m].mean()
        if len(per_arm) >= 2 and per_arm.min() >= high and (per_arm.max() - per_arm.min()) <= spread:
            msgs.append(f"{model} / {task}: every arm in [{per_arm.min():.3f}, {per_arm.max():.3f}] on "
                        f"{m} -> the task saturates, report a finer metric (format_f1) or harden it")
    return msgs


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    metrics = [c for c in df.columns if c in ("mcq_acc", "completion_em", "format_parsed", "format_exact", "format_f1",
                                                "format_f1_macro", "format_count_ok", "format_sig_ok",
                                                "format_order_ok", "format_parsed_1-2ent",
                                                "format_parsed_3-4ent", "format_parsed_5+ent",
                                                "hellaswag_acc_norm", "truthfulqa_mc1", "wikitext_ppl")]
    g = df.groupby(KEY, dropna=False)
    out = g[metrics].agg(["mean", "std"])
    out.columns = [f"{m}_{s}" for m, s in out.columns]
    out["n_seeds"] = g["seed"].nunique()
    out["n_trainable"] = g["n_trainable"].first()
    # a string column: carried like n_trainable, never through agg(["mean","std"]), which would raise
    if "data_fingerprint" in df.columns:
        out["data_fingerprint"] = g["data_fingerprint"].first()
    out = out.reset_index()
    # sampling error of the primary metric, alongside the seed-to-seed spread: a gap smaller than
    # the sum of the two is not a difference, however stable it looks across seeds
    for task, metric, ncol in (("facts", "mcq_acc", "mcq_n"), ("format", "format_parsed", "format_n")):
        m = f"{metric}_mean"
        if m in out and ncol in df.columns:
            n_by_cell = df.groupby(KEY, dropna=False)[ncol].first().reset_index()
            out = out.merge(n_by_cell, on=KEY, how="left")
            sel = out["task"] == task
            out.loc[sel, f"{metric}_se"] = [binomial_se(p, n) for p, n in
                                            zip(out.loc[sel, m], out.loc[sel, ncol])]
    return out


class IncomparableCells(ValueError):
    """Raised by the two functions whose output goes straight into the paper.

    Five checks print warnings and none changes behaviour. There is a precedent: the PROVENANCE ⚠ about
    the full fine-tuning ceiling measured on OLMo was explicit, and the number still went into the
    paper's main table as a ceiling for Llama. A warning inside fifty lines of output, read at midnight,
    is not a guard rail. `crossed_table` and `win_matrix` therefore refuse.
    """


def _refuse_if_mixed(d: "pd.DataFrame", what: str) -> None:
    """Refuse to build a paper-facing object out of cells that mix incomparable runs."""
    if d.empty:
        return
    if "format_parsed" in d.columns and "format_order_ok" in d.columns:
        d = d.assign(_metric_version=effective_metric_version(d))
    # The fields come from SETTING_FIELDS in config.py, not from a list retyped here. This one and
    # select_lr's used to be maintained separately, and select_lr was missing max_grad_norm for a
    # week: the noclip reruns sat in the cells it averaged, and flipped a selected rate.
    _LABEL = {"_metric_version": "checker versions",
              "format_n": "evaluation sizes", "mcq_n": "evaluation sizes",
              # the VALIDATION size too: it is min(eval_limit, 500), so the sweep validates on
              # 400 items and the seed grid on 500 — and select_lr optimises that very column
              "val_format_n": "validation sizes", "val_mcq_n": "validation sizes",
              "subspace_salted": "random-arm versions",
              # the ones that make a run a different EXPERIMENT rather than a different measurement
              # of the same one
              "update_norm_target": "magnitude-ablation targets",
              "max_grad_norm": "gradient-clipping thresholds",
              "n_modules": "adapted-module sets",
              "data_fingerprint": "dataset versions"}
    for col in SETTING_FIELDS:
        label = _LABEL.get(col, col)
        if col not in d:
            continue
        # grouped by TASK as well: a (bottom, r=2) cell holds both format and facts runs, and a facts
        # run has no format_n — without the task in the key that absence read as a distinct setting
        # and fired a false refusal.
        # MODEL is part of the key. Until 15/09 only Llama had runs at 614 items, so grouping by
        # (task, mode, rank) alone happened to be safe; the day Qwen produced its first 614-item run
        # a cell (format, top, 2) started holding two architectures with different d_in, different
        # depth and different trainable counts — 655360 against 491520 — and nothing refused. The
        # guard has to key on the model or it silently averages two models into one row of the paper.
        # The model belongs in the key, but a caller may hand us a frame that has no `model` column
        # at all — a hand-built table in a test, or a slice already restricted to one model. Keying
        # on a missing column raises KeyError and turns a guard into a crash, so fall back to the
        # three remaining fields and report the model as unknown.
        _keys = [k for k in ("model", "task", "mode", "rank") if k in d.columns]
        for _cell, g in d.groupby(_keys, dropna=False):
            _cell = _cell if isinstance(_cell, tuple) else (_cell,)
            _v = dict(zip(_keys, _cell))
            model, task, mode, rank = (_v.get("model", "?"), _v.get("task"),
                                       _v.get("mode"), _v.get("rank"))
            # The salt describes how a random subspace is DRAWN. On `top`, `bottom`, `band` and
            # `free` nothing is drawn, so a run recorded before the field existed (None) and one
            # recorded after (True) are the same run — comparing them there is a false refusal, and
            # it blocked the crossed table on a cell that was perfectly homogeneous. Same reasoning
            # as in the run id, where the field is already restricted to the drawing arms.
            if col == "subspace_salted" and str(mode) not in SALTED_MODES:
                continue
            if col in ("data_fingerprint", "_metric_version", "update_norm_target", "max_grad_norm"):
                # An ABSENT value IS a value here, and for two different reasons.
                #
                # For the fingerprint and the checker version: the historical runs carry none, so
                # dropping empties would let exactly the mix the coming weeks will produce — old
                # runs without one, new ones with — pass unnoticed.
                #
                # For update_norm_target it is stronger still. A normal run has NO target; a
                # rescaled one has 1.5 or 4.5. Dropping the empties would leave a single level in a
                # cell holding five normal runs and five rescaled ones, and the guard would pass a
                # cell whose scores are 0.5645 and 0.0000. The magnitude ablation is the one place
                # where absence is the interesting value.
                vals = {(str(x).strip() or "legacy") if x == x and x is not None else "legacy"
                        for x in g[col]}
            else:
                # Everywhere else an absent value means "not measured on this run", not a setting.
                vals = {str(x) for x in g[col].dropna().unique() if str(x).strip()}
            if len(vals) > 1:
                raise IncomparableCells(
                    f"{what}: cell ({str(model).split('/')[-1]}, {task}, {mode}, r={rank}) "
                    f"mixes {len(vals)} {label} "
                    f"({sorted(vals)}). These runs did not measure the same thing. Filter with "
                    f"select_population(), or re-evaluate at a single setting.")


def crossed_table(df: pd.DataFrame, model: str, rank: int, lr_table: dict | None = None) -> str:
    """LaTeX: rows = arms (free/top/bottom/random/full/baseline), cols = facts (mcq_acc) and format (parsed)."""
    d = df[(df.model == model) & (df.band_frac.isna())]
    d = d[(d["rank"] == rank) | d["mode"].isin(["full", "baseline"])]
    d = _keep_selected_lr(d, lr_table)
    _refuse_if_mixed(d, "crossed_table")
    lines = [r"\begin{tabular}{@{}lcc@{}}", r"\toprule",
             r"Arm & Facts (MCQ acc.) & Format (parsed) \\", r"\midrule"]
    for arm in ARMS:
        cells = []
        for task in ("facts", "format"):
            if _metric(task) not in d.columns:
                cells.append("---")     # that task was not measured in this frame
                continue
            g = d[(d["mode"] == arm) & (d.task == task)][_metric(task)]
            cells.append(f"{100 * g.mean():.1f} $\\pm$ {100 * g.std():.1f} ({len(g)})" if len(g) else "--")
        if any(c != "--" for c in cells):
            lines.append(f"{arm} & {cells[0]} & {cells[1]} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def plot_crossed(df: pd.DataFrame, model: str, out: Path, lr_table: dict | None = None,
                 ranks: tuple | None = None) -> None:
    """Bars per arm, one panel per (task, rank). Annotates the number of runs behind every bar.

    Two things this has to do, both learned the hard way. It must filter on the learning rate, or it
    silently averages cells trained at different ones. And it must show n: a bar built from a single
    run has a NaN standard deviation, so pandas draws no error bar at all — which reads as a perfectly
    reproducible measurement rather than as a single sample.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = df[(df.model == model) & (df.band_frac.isna())]
    d = _keep_selected_lr(d, lr_table)
    _refuse_if_mixed(d, "plot_crossed")
    rs = sorted(ranks) if ranks else sorted(x for x in d["rank"].unique() if x and x > 0)
    tasks = [t for t in ("facts", "format") if t in set(d.task)]
    if not rs or not tasks:
        return
    fig, axes = plt.subplots(len(tasks), len(rs), figsize=(3.2 * len(rs), 3.2 * len(tasks)), squeeze=False)
    for i, task in enumerate(tasks):
        m = _metric(task)
        for j, rank in enumerate(rs):
            ax = axes[i][j]
            sub = d[(d.task == task) & (d["rank"] == rank)]
            arms = [a for a in ARMS if a in set(sub["mode"])]
            if not arms:
                ax.set_axis_off()
                continue
            means = [sub[sub["mode"] == a][m].mean() for a in arms]
            stds = [sub[sub["mode"] == a][m].std() for a in arms]
            ns = [len(sub[sub["mode"] == a][m].dropna()) for a in arms]
            bars = ax.bar(arms, means, yerr=[0 if pd.isna(s) else s for s in stds], capsize=3)
            for b, v, n in zip(bars, means, ns):
                if not pd.isna(v):
                    ax.text(b.get_x() + b.get_width() / 2, v, f"n={n}", ha="center", va="bottom",
                            fontsize=6)
            ax.set_title(f"{task}, r={int(rank)}", fontsize=9)
            ax.set_ylabel(m, fontsize=8)
            ax.tick_params(axis="x", rotation=30, labelsize=8)
    fig.suptitle(f"{model}" + (" (lr selected per arm)" if lr_table else " (single lr)"), fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_spectral_curve(df: pd.DataFrame, model: str, rank: int, out: Path) -> None:
    """Primary metric vs band position; references: free / random / full as horizontal lines."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = df[(df.model == model)]
    _refuse_if_mixed(d, "plot_spectral_curve")
    if not len(d[(d["mode"] == "band") & (d["rank"] == rank)]):
        # nothing to draw: without band runs the figure would be two horizontal reference lines and
        # no spectral information at all. Better no file than an empty one.
        print(f"[aggregate] no band runs at rank {rank} for {model}: spectral curve not drawn "
              f"(run configs/grids/band_sweep_1b.yaml)")
        return
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for ax, task in zip(axes, ("facts", "format")):
        m = _metric(task)
        band = d[(d.task == task) & (d["mode"] == "band") & (d["rank"] == rank)]
        if len(band):
            g = band.groupby("band_frac")[m].agg(["mean", "std"]).reset_index()
            ax.errorbar(g.band_frac, g["mean"], yerr=g["std"], marker="o", capsize=3, label="band (r consecutive dirs)")
        for arm, ls in (("free", "-"), ("random", "--"), ("full", ":")):
            ref = d[(d.task == task) & (d["mode"] == arm) & ((d["rank"] == rank) | (arm == "full"))][m]
            if len(ref):
                ax.axhline(ref.mean(), ls=ls, c="gray", lw=1, label=f"{arm} ({ref.mean():.3f})")
        ax.set_xlabel("spectral position of the band (0 = top, 1 = bottom)")
        ax.set_ylabel(m)
        ax.set_title(f"{task}, r={rank}")
        ax.legend(fontsize=7)
    fig.suptitle(model)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def forgetting_delta(df: pd.DataFrame) -> pd.DataFrame:
    """Forgetting metrics relative to the un-adapted ``baseline`` run of the same model (needs mode=baseline runs)."""
    cols = [c for c in ("hellaswag_acc_norm", "truthfulqa_mc1", "wikitext_ppl") if c in df]
    if not cols:
        return pd.DataFrame()
    _refuse_if_mixed(df[df["mode"] != "baseline"], "forgetting_delta")
    base = df[df["mode"] == "baseline"].groupby("model")[cols].mean()
    rows = []
    # LR IS PART OF THE KEY. Without it this averaged `bottom` at 2e-3 and at 5e-3 into one row —
    # including the seed that collapsed — and reported the mean as "the forgetting of the minor arm".
    # Every other paper-facing aggregate in this file keys on lr; this one did not.
    for (model, task, mode, rank, lr), g in df[df["mode"] != "baseline"].groupby(
            ["model", "task", "mode", "rank", "lr"], dropna=False):
        if model not in base.index:
            continue
        r = dict(model=model, task=task, mode=mode, rank=rank, lr=lr)
        for c in cols:
            r[f"d_{c}"] = float(g[c].mean() - base.loc[model, c])
        rows.append(r)
    return pd.DataFrame(rows)




# --------------------------------------------------------------------------- #
# Figures that answer a question. Each one exists because a specific claim needs
# it; none is decorative.
# --------------------------------------------------------------------------- #
def plot_lr_heatmap(df: pd.DataFrame, model: str, task: str, out: Path) -> None:
    """Arm x learning rate, one cell per (arm, rank, lr). THE figure for the Lee et al. objection.

    A ranking that holds at one learning rate and flips at another is not a property of the subspace.
    Reading it as a table of means hides that; a heatmap shows at a glance whether the column of the
    winning arm stays the winning column as lr moves. The per-cell count is annotated because a cell
    with one run has no error bar and must not be read like the others.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = df[(df.model == model) & (df.task == task) & df["mode"].isin(ARMS) & df.band_frac.isna()]
    if d.empty:
        return
    # THE figure of the paper, and it was the least guarded of all. A cell here is (arm, rank, lr),
    # and the sweep runs at 400 items sit in the same cell as the seed-grid runs at 614: averaged in
    # silence, they produced a number that is neither. Six refusals existed and only the two TABLES
    # used them — nobody reads crossed_*.tex, they look at this heatmap.
    _refuse_if_mixed(d, "plot_lr_heatmap")
    m = _metric(task)
    piv = d.pivot_table(index=["mode", "rank"], columns="lr", values=m, aggfunc="mean")
    cnt = d.pivot_table(index=["mode", "rank"], columns="lr", values=m, aggfunc="count")
    if piv.empty:
        return
    fig, ax = plt.subplots(figsize=(1.5 * len(piv.columns) + 3, 0.42 * len(piv) + 2))
    im = ax.imshow(piv.values, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels([f"{c:.0e}" for c in piv.columns])
    ax.set_yticks(range(len(piv)))
    ax.set_yticklabels([f"{a} r={int(r)}" for a, r in piv.index], fontsize=8)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v, n = piv.values[i, j], cnt.values[i, j]
            if not pd.isna(v):
                ax.text(j, i, f"{v:.3f}\nn={int(n)}", ha="center", va="center", fontsize=6,
                        color="white" if v < piv.values[~pd.isna(piv.values)].mean() else "black")
    ax.set_xlabel("learning rate")
    ax.set_title(f"{task} — {m} per arm and learning rate\n(does the ordering survive tuning?)",
                 fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.03)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def win_matrix(df: pd.DataFrame, model: str, task: str, rank: float,
               lr_table: dict | None = None) -> pd.DataFrame:
    """Fraction of seeds on which arm i beats arm j, seed by seed.

    Non-parametric, and more honest than mean +/- std with five seeds: it uses the pairing (the same
    seed means the same data order) and makes no normality assumption. A mean gap of 9 points carried
    by 5 seeds out of 5 is a different claim from the same gap carried by 3 out of 5.
    """
    d = df[(df.model == model) & (df.task == task) & (df["rank"] == rank) & df.band_frac.isna()]
    d = _keep_selected_lr(d, lr_table)
    m = _metric(task)
    _refuse_if_mixed(d, "win_matrix")
    arms = [a for a in ARMS if a in set(d["mode"])]
    piv = d.pivot_table(index="seed", columns="mode", values=m, aggfunc="mean")
    out = pd.DataFrame(index=arms, columns=arms, dtype=float)
    for a in arms:
        for b in arms:
            if a == b or a not in piv or b not in piv:
                continue
            both = piv[[a, b]].dropna()
            out.loc[a, b] = float((both[a] > both[b]).mean()) if len(both) else float("nan")
    out.attrs["n_seeds"] = int(piv.notna().all(axis=1).sum())
    return out


def plot_failure_decomposition(df: pd.DataFrame, model: str, out: Path, rank: float | None = None) -> None:
    """Where each arm fails: keys, count, signature, ordering — and how often it emits an empty shell.

    A single "valid output" rate says an arm is worse; this says in what way. Two arms at the same
    score that fail on different rules are not doing the same thing, and that distinction is the
    mechanistic part of the story.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cols = [c for c in ("format_keys_ok", "format_count_ok", "format_sig_ok", "format_order_ok",
                        "format_parsed", "format_exact") if c in df.columns]
    if len(cols) < 3:
        return
    d = df[(df.model == model) & (df.task == "format") & df["mode"].isin(ARMS)]
    _refuse_if_mixed(d, "plot_failure_decomposition")
    if rank is not None:
        d = d[d["rank"] == rank]
    if d.empty:
        return
    g = d.groupby("mode")[cols].mean()
    fig, ax = plt.subplots(figsize=(1.3 * len(g) + 3, 3.6))
    x = range(len(g))
    w = 0.8 / len(cols)
    for k, c in enumerate(cols):
        ax.bar([i + k * w - 0.4 for i in x], g[c].values, w, label=c.replace("format_", ""))
    if "format_empty" in d.columns:
        ax.plot(list(x), d.groupby("mode")["format_empty"].mean().reindex(g.index).values, "kx--",
                label="empty shell", ms=8)
    ax.set_xticks(list(x))
    ax.set_xticklabels(g.index)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("rate")
    ax.legend(fontsize=7, ncol=3)
    ax.set_title("Which rule does each arm fail?" + (f" (r={int(rank)})" if rank else ""), fontsize=10)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def gradient_vs_performance(probe_csv: str | Path, df: pd.DataFrame, model: str, task: str,
                            rank: float, out: Path | None = None) -> pd.DataFrame:
    """Does WHERE the gradient lives predict WHICH arm wins? The bridge between the two halves.

    For each arm we take the gradient energy the probe measured in the very band that arm adapts, and
    put it next to the score that arm obtained. If the arm adapting the band that receives more
    gradient also performs better, the probe has predictive value and the paper has one story instead
    of two. If not, that is worth saying: the initial gradient does not determine the outcome of
    training, which is itself a result.
    """
    probe = pd.read_csv(probe_csv)
    probe = probe[(probe.dataset == task) & (probe.side == "left")]
    band_of = {"top": f"top:{int(rank)}", "bottom": f"bottom:{int(rank)}"}
    rows = []
    d = df[(df.model == model) & (df.task == task) & (df["rank"] == rank)]
    m = _metric(task)
    for arm, band in band_of.items():
        e = probe[probe.band == band]
        sub = d[d["mode"] == arm][m]
        if len(e) and len(sub):
            rows.append(dict(mode=arm, band=band, grad_energy=float(e.energy.mean()),
                             grad_energy_over_null=float((e.energy / e.null.replace(0, float("nan"))).mean()),
                             score=float(sub.mean()), n=len(sub)))
    res = pd.DataFrame(rows)
    if out is not None and len(res) >= 2:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(4.4, 3.4))
        ax.scatter(res.grad_energy_over_null, res.score, s=60)
        for _, r in res.iterrows():
            ax.annotate(r["mode"], (r.grad_energy_over_null, r.score), textcoords="offset points",
                        xytext=(6, 4), fontsize=9)
        ax.set_xlabel("gradient energy in the adapted band / null reference")
        ax.set_ylabel(m)
        ax.set_title(f"Does the initial gradient predict the winner? ({task}, r={int(rank)})",
                     fontsize=9)
        fig.tight_layout()
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=200, bbox_inches="tight")
        plt.close(fig)
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=None)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--select_lr", default=None, help="write lr_selected.yaml here")
    ap.add_argument("--salted", type=lambda x: str(x).lower() not in ("0", "false", "no"), default=True,
                    help="which version of the random arm to analyse: True = drawn independently per "
                         "module (post 12/09), False = the earlier arm. Never both at once.")
    ap.add_argument("--probe_csv", default=None,
                    help="gradient-probe CSV: adds the figure linking where the gradient lives to "
                         "which arm wins")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df = collect(a.runs_dir)
    print(f"{len(df)} finished runs")
    if df.empty:
        return
    df.to_csv(out / "collected.csv", index=False)
    summarize(df).to_csv(out / "summary.csv", index=False)
    df = select_population(df, salted=a.salted)
    for msg in check_metric_versions(df):
        print(f"[aggregate] *** {msg}")
    for msg in check_data_versions(df):
        print(f"[aggregate] *** {msg}")
    for msg in check_eval_sizes(df):
        print(f"[aggregate] *** {msg}")
    for msg in check_saturation(df):
        print(f"[aggregate] WARNING  {msg}")
    lr_table = None
    if a.select_lr:
        lr_table = select_lr(df, a.select_lr)["selected"]
        print(f"lr selected for {len(lr_table)} (model, task, mode, rank) cells -> {a.select_lr}")
    models = [a.model] if a.model else sorted(df.model.unique())
    for model in models:
        slug = model.replace("/", "__")
        try:
            (out / f"crossed_{slug}_r{a.rank}.tex").write_text(crossed_table(df, model, a.rank, lr_table))
        except IncomparableCells as e:
            # the refusal is right, its blast node was not: one mixed cell used to take down the
            # rate map, the win matrix, the failure decomposition and the spectral curve with it
            print(f"[aggregate] *** crossed table SKIPPED for {model}: {e}")
        for name, fn in (("crossed plot", lambda: plot_crossed(df, model, out / f"crossed_{slug}.png", lr_table)),
                         ("spectral curve", lambda: plot_spectral_curve(
                             df, model, a.rank, out / f"spectral_curve_{slug}_r{a.rank}.png"))):
            try:
                fn()
            except IncomparableCells as e:
                print(f"[aggregate] *** {name} SKIPPED for {model}: {e}")
    for model in models:
        slug = model.replace("/", "__")
        for task in sorted(set(df[df.model == model].task.dropna())):
            try:
                plot_lr_heatmap(df, model, task, out / f"lr_heatmap_{slug}_{task}.png")
            except IncomparableCells as e:
                print(f"[aggregate] *** rate map SKIPPED for {model}/{task}: {e}")
        try:
            plot_failure_decomposition(df, model, out / f"failures_{slug}_r{a.rank}.png", rank=a.rank)
        except IncomparableCells as e:
            print(f"[aggregate] *** failure decomposition SKIPPED for {model}: {e}")
        try:
            wm = win_matrix(df, model, "format", a.rank, lr_table)
        except IncomparableCells as e:
            print(f"[aggregate] *** win matrix SKIPPED for {model}: {e}")
            wm = pd.DataFrame()
        if not wm.empty and wm.notna().any().any():
            wm.to_csv(out / f"win_matrix_{slug}_format_r{a.rank}.csv")
            print(f"\nwin matrix (fraction of seeds where row beats column), format r={a.rank}, "
                  f"n={wm.attrs.get('n_seeds')} paired seeds:")
            print(wm.round(2).to_string())
        if a.probe_csv:
            res = gradient_vs_performance(a.probe_csv, df, model, "format", a.rank,
                                          out / f"grad_vs_perf_{slug}_r{a.rank}.png")
            if len(res):
                print("\ngradient energy in the adapted band vs score:")
                print(res.round(4).to_string(index=False))

    fd = forgetting_delta(df)
    if not fd.empty:
        fd.to_csv(out / "forgetting_delta.csv", index=False)
    print(f"wrote tables and figures to {out}")


if __name__ == "__main__":
    main()
