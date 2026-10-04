"""Grid launcher.

    python -m lorasub.launch_grid configs/grids/grid_1b.yaml --grid_dir $HOME/lora-grid/1b \
        [--partition PARTITION] [--time 01:55:00] [--max_par 100] [--lr_from configs/lr_selected.yaml] [--dry_run]

    grid_dir MUST be in the home (shared by all nodes); node-local paths (/tmp, /Data) are refused.

Grid YAML format — two sections, so that a *list-valued fixed field* is never mistaken for an axis:

    base:                       # fixed values (may contain lists, e.g. save_factor_steps)
      model: meta-llama/Llama-3.2-1B
      target_tokens: 1800000
      save_factor_steps: [0, 10, 50, 100, 300, 1000, end]
      svd_cache: $SHARED/svd
      data_dir: $SHARED/data
      out_dir: $SHARED/runs
    grid:                       # every key here is an axis; values must be lists
      task: [facts, format]
      mode: [free, top, bottom, random]
      rank: [4, 16, 64]
      seed: [0, 1, 2, 3, 4]
      # an axis value may be a dict, whose keys are merged (fields that must move together):
      # rank_and_target: [{rank: 1, update_norm_target: 0.36}, {rank: 2, update_norm_target: 0.42}]
    extra:                      # optional: additional explicit configs (dicts merged over base)
      - {mode: full, task: facts, rank: 0, lr: 2.0e-5, seed: 0}

Each run gets ``<grid_dir>/configs/<run_id>.yaml``; run ids are hashes of the scientific
fields, so two identical configs collapse to one file and nothing is silently overwritten.
Runs whose ``results.csv`` already exists are skipped.  ``--lr_from`` replaces ``lr`` by the
value selected for (model, task, mode, rank) in a ``lr_selected.yaml`` written by aggregate.py.
"""
from __future__ import annotations

import argparse
import dataclasses
import itertools
import os
import sys
from pathlib import Path

import yaml

from . import expand_path
from .config import RunConfig, apply_model_defaults


_CONFIG_FIELDS = frozenset(f.name for f in dataclasses.fields(RunConfig))


def expand(grid_spec: dict) -> list[dict]:
    base = dict(grid_spec.get("base", {}))
    axes = grid_spec.get("grid", {})
    for k, v in axes.items():
        if not isinstance(v, list):
            raise ValueError(f"grid axis {k!r} must be a list, got {type(v).__name__}")
    keys = list(axes)
    configs: list[dict] = []
    for combo in itertools.product(*(axes[k] for k in keys)):
        c = dict(base)
        for k, v in zip(keys, combo):
            # A dict value is MERGED when the axis name is not itself a config field: that is how
            # fields which must move together are swept (e.g. a rank and the update norm it must be
            # matched against) without making them independent axes. When the axis name IS a field —
            # `per_module`, whose values are legitimately dicts — the dict stays the value.
            if isinstance(v, dict) and k not in _CONFIG_FIELDS:
                c.update(v)
            else:
                c[k] = v
        configs.append(c)
    for e in grid_spec.get("extra", []) or []:
        c = dict(base)
        c.update(e)
        configs.append(c)
    return configs


def lr_lookup(path: str | Path) -> dict:
    with open(path) as f:
        d = yaml.safe_load(f) or {}
    return d.get("selected", d)


def apply_lr(raw: dict, table: dict) -> dict:
    """Replace lr by the value selected for this arm. The rank is normalised to an int, since a table
    written from a CSV carries 1.0 where the grid carries 1 — a mismatch would silently leave the run
    at the default learning rate."""
    rank = raw.get("rank")
    try:
        rank = int(float(rank))
    except (TypeError, ValueError):
        pass
    # band_frac is part of the key: without it the five positions of a band sweep all receive the
    # rate of whichever one select_lr happened to keep, and the sweep runs at a common rate.
    key = f"{raw.get('model')}|{raw.get('task')}|{raw.get('mode')}|{rank}"
    _bf = raw.get("band_frac")
    # `is not None` is not enough: a band_frac read from a CSV comes back as NaN, not None, and
    # would append "|bnan" here while aggregate.py appends nothing. The key would then never be
    # found and every run would silently fall back to the default rate.
    if _bf is not None and _bf == _bf:
        key += f"|b{float(_bf):g}"
    if table and key not in table:
        # SILENT FAILURE otherwise: the run falls back to the default rate, typically 5e-4, which can
        # be far from this arm's optimum — and nothing in the outputs says so. Seen when a sweep
        # covered rank 2 only while the seed grid ran at ranks 1, 2 and 4.
        print(f"[launch] WARNING no selected lr for {key}: falling back to {raw.get('lr')}. "
              f"Sweep that cell first, or restrict the grid to the ranks that were swept.")
    if key in table:
        raw = dict(raw)
        raw["lr"] = float(table[key])
    return raw


def materialise(configs: list[dict], grid_dir: Path, lr_table: dict | None = None) -> tuple[list[RunConfig], list[RunConfig]]:
    """Write per-run YAMLs.  Returns (to_run, already_done)."""
    cdir = grid_dir / "configs"
    cdir.mkdir(parents=True, exist_ok=True)
    to_run, done, seen = [], [], set()
    for raw in configs:
        if lr_table:
            raw = apply_lr(raw, lr_table)
        cfg = RunConfig(**apply_model_defaults(dict(raw)))
        if cfg.run_id in seen:
            continue
        seen.add(cfg.run_id)
        cfg.to_yaml(cdir / f"{cfg.run_id}.yaml")
        (done if (cfg.run_dir / "results.csv").exists() else to_run).append(cfg)
    return to_run, done


def write_slurm(grid_dir: Path, cfgs: list[RunConfig], partition: str, time: str, mem: str, max_par: int,
                cpus: int = 4, venv: str | None = None, repo: str | None = None, shared: str | None = None,
                gres: str | None = None, python_bin: str = "python3", cuda_tag: str = "cu121",
                refuse_local: bool = True, exclusive: bool = True,
                min_free_mib: int = 6000) -> Path:
    """SLURM array script that is self-sufficient on a compute node with an empty local /tmp.

    Preamble, in order:  export SHARED / HF_HOME / HF_TOKEN  ->  build the venv on this node if absent
    (pip cache in $HOME so it is fast after the first node)  ->  run the config.  The SVD cache is built
    by train.py itself on first use (ensure_svd_cache).  Data and results live in the home (shared),
    which is why grid_dir must be there too: logs and configs.txt are read by every node.
    ``gres`` is optional because some clusters (e.g. the X salles-info partition) expose GPUs without
    advertising a gres, and reject --gres=PARTITION:1 with "Invalid generic resource specification".
    """
    if any(ch.isspace() for ch in str(grid_dir)):
        raise ValueError(f"grid_dir must not contain whitespace (SLURM directives are not quoted): {grid_dir}")
    if refuse_local and str(grid_dir).startswith(("/tmp", "/Data", "/scratch")):
        raise ValueError(f"grid_dir must be on a shared filesystem (home), not node-local: {grid_dir}")
    listing = grid_dir / "configs.txt"
    listing.write_text("\n".join(str(grid_dir / "configs" / f"{c.run_id}.yaml") for c in cfgs) + "\n")
    logs = grid_dir / "logs"
    logs.mkdir(exist_ok=True)
    n = len(cfgs)
    shared = shared or "/tmp/lora-$USER"
    venv = venv or f"{shared}/venv"
    repo = repo or str(Path(__file__).resolve().parents[2])
    gres_line = f"#SBATCH --gres={gres}\n" if gres else ""
    # --exclusive: ONE JOB PER NODE. This partition refuses --gres, so nothing reserves the PARTITION, and
    # SLURM packs several tasks onto any node with CPU and RAM to spare — they then fight over the
    # single card. Measured: 53 of 60 tasks landed on the same node and all 53 died of OOM, while the
    # 7 that landed elsewhere succeeded. With no PARTITION-aware resource, exclusivity is the only way to
    # guarantee one job per card.

    exclusive_line = ("#SBATCH --exclusive" if exclusive
                      else "# NOT exclusive: several jobs may land on one node and share its PARTITION")
    script = f"""#!/bin/bash
#SBATCH --job-name=lorasub
#SBATCH --partition={partition}
{gres_line}#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={mem}
{exclusive_line}
#SBATCH --time={time}
#SBATCH --array=0-{max(0, n - 1)}%{max_par}
#SBATCH --output={logs}/%A_%a.out
set -euo pipefail

# ---- environment on THIS node (its /tmp is local and may be empty) ----------------------------
export SHARED="{shared}"
export HF_HOME="$SHARED/hf"
# pip cache on the NODE, never in $HOME: the home is an NFS quota (30 GB here) shared with everything
# else, and twenty nodes downloading torch into it at once filled it and failed every install with
# "Disk quota exceeded" / "Stale file handle". Local cache costs one download per node and nothing else.
export PIP_CACHE_DIR="$SHARED/pipcache"
export TMPDIR="$SHARED/tmp"
[ -f "$HOME/.hf_env" ] && . "$HOME/.hf_env"       # HF_TOKEN for gated models, if configured
mkdir -p "$SHARED" "$PIP_CACHE_DIR" "$TMPDIR"

# fail early and clearly if the home is full: every result file is written there
AVAIL_KB=$(df -Pk "$HOME" | awk 'NR==2{{print $4}}')
if [ "${{AVAIL_KB:-0}}" -lt 524288 ]; then
  echo "ERROR: less than 512 MB free in $HOME -- results cannot be written. Free space first." >&2
  exit 1
fi
echo "node: $(hostname)  PARTITION: $(nvidia-smi --query-PARTITION=name,memory.total --format=csv,noheader 2>/dev/null || echo none)"

VENV="{venv}"
# Several array tasks can land on the SAME node and would build the same venv concurrently, deleting
# each other's files mid-install ("No such file or directory: _distutils_hack/"). One flock per node
# serialises them: the first builds, the others wait and then reuse. The venv is only published under
# its final name once it is complete, so an interrupted build is never mistaken for a usable one.
#
# The venv is NEVER activated. `bin/activate` hard-codes VIRTUAL_ENV at creation time, so after the
# atomic rename it puts a dead directory at the head of PATH, `python` silently falls back to the
# system interpreter, and the job dies on "No module named lorasub" right after printing "venv ready".
# Calling "$VENV/bin/python" by absolute path is immune to the rename and needs no activation.
VENV_LOCK="$SHARED/venv.lock"
mkdir -p "$(dirname "$VENV_LOCK")"
exec {{LOCKFD}}>"$VENV_LOCK"
flock "$LOCKFD"                       # released automatically when this shell exits
if ! "$VENV/bin/python" -c "import lorasub, torch" 2>/dev/null; then
  [ -e "$VENV" ] && echo "venv on $(hostname) is incomplete, rebuilding it" && rm -rf "$VENV"
  echo "building venv on $(hostname) (once per node; other tasks on this node are waiting)"
  BUILD="$VENV.building.$$"
  rm -rf "$BUILD"
  {python_bin} -m venv "$BUILD"
  "$BUILD/bin/pip" install -q --upgrade pip setuptools wheel
  "$BUILD/bin/pip" install -q torch --extra-index-url https://download.pytorch.org/whl/{cuda_tag}
  "$BUILD/bin/pip" install -q -r "{repo}/requirements.txt" || {{ echo "pip install failed on $(hostname)" >&2; exit 1; }}
  "$BUILD/bin/pip" install -q -e "{repo}" || {{ echo "editable install failed on $(hostname)" >&2; exit 1; }}
  "$BUILD/bin/python" -c "import lorasub, torch" || {{ echo "venv incomplete after build" >&2; exit 1; }}
  mv "$BUILD" "$VENV"                 # atomic publish: $VENV exists only when it works
  echo "venv ready on $(hostname)"
fi
flock -u "$LOCKFD"
cd "{repo}"

# ---- the run ------------------------------------------------------------------------------------
# absolute path, no activation: see the note above the lock
# ---- refuse to start on a card someone else is already using -----------------------------------
# This partition declares no GRES, so SLURM does not know the GPUs exist: it cannot reserve them and
# cannot stop another user from running on the same machine. Measured: 144 tasks died of CUDA OOM
# against neighbours holding 13 to 21 GiB, each one after paying the full model-loading cost. Exiting
# 0 rather than 1 leaves the run unclaimed, so a later pass picks it up on a free card.
FREE=$(nvidia-smi --query-PARTITION=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1)
if [ "${{FREE:-0}}" -lt {min_free_mib} ]; then
  echo "skip: only ${{FREE}} MiB free on $(hostname) — leaving this run for a later pass"
  exit 0
fi

PY="$VENV/bin/python"
"$PY" -c "import lorasub, torch" || {{ echo "venv at $VENV is unusable on $(hostname)" >&2; exit 1; }}
CFG=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" {listing})
echo "config: $CFG"
"$PY" -m lorasub.train "$CFG"
"""
    path = grid_dir / "slurm_array.sh"
    path.write_text(script)
    path.chmod(0o755)
    return path


def list_failed(runs_dir: str | Path) -> list[tuple[str, str]]:
    out = []
    for status in Path(runs_dir).glob("*/status"):
        txt = status.read_text()
        if txt.startswith("FAILED"):
            out.append((status.parent.name, txt.splitlines()[-1] if txt.strip() else ""))
    return sorted(out)


def grid_has_active_jobs(grid_dir: Path) -> list[str]:
    """Job ids of SLURM jobs whose stdout is written into this grid's log directory.

    Submitting a grid while a previous submission of it is still running lets two array tasks execute
    the same run_id at once; their concurrent writes corrupted ten results.csv files before the
    atomic write existed, and even with it the second run is wasted compute. This was a human rule
    ("check that squeue is empty"); it is now enforced here, because human rules fail at 2 a.m.
    """
    import shutil
    import subprocess

    if shutil.which("squeue") is None:
        return []
    try:
        out = subprocess.run(["squeue", "-u", os.environ.get("USER", ""), "-h", "-o", "%i %o"],
                             capture_output=True, text=True, timeout=10).stdout
    except Exception:  # noqa: BLE001
        return []
    logs = str(Path(grid_dir).resolve() / "logs")
    return [line.split()[0] for line in out.splitlines() if logs in line]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("grid_yaml")
    ap.add_argument("--grid_dir", required=True)
    ap.add_argument("--partition", default="PARTITION")
    ap.add_argument("--time", default="01:55:00", help="must not exceed the partition limit (2h on PARTITION)")
    ap.add_argument("--mem", default="32G")
    ap.add_argument("--no_exclusive", action="store_true",
                    help="allow several jobs per node; only if the partition reserves GPUs, which "
                         "PARTITION does not — see the note in the generated script")
    ap.add_argument("--max_par", type=int, default=100)
    ap.add_argument("--force", action="store_true",
                    help="generate even if jobs of this grid_dir are still running (you then accept "
                         "that the same run_id may be executed twice)")
    ap.add_argument("--venv", default=None, help="venv path on the node (default: $SHARED/venv, built if absent)")
    ap.add_argument("--shared", default="/tmp/lora-$USER", help="node-local scratch root (venv, HF cache, SVD cache)")
    ap.add_argument("--python", default=None, help="python >= 3.10 used to build the venv on the node "
                                                     "(default: ~/.local/bin/python3.10 if it exists, else python3)")
    ap.add_argument("--cuda_tag", default="cu121")
    ap.add_argument("--gres", default=None, help="e.g. PARTITION:1 ; omit on clusters that reject it")
    ap.add_argument("--lr_from", default=None, help="lr_selected.yaml from aggregate.py")
    ap.add_argument("--dry_run", action="store_true")
    a = ap.parse_args()
    with open(a.grid_yaml) as f:
        spec = yaml.safe_load(f)
    grid_dir = Path(expand_path(a.grid_dir))
    active = grid_has_active_jobs(grid_dir)
    if active and not a.force:
        sys.exit(f"refusing to generate: {len(active)} job(s) of this grid are still running "
                 f"({', '.join(active[:5])}{'...' if len(active) > 5 else ''}). Wait for `squeue -u $USER` "
                 f"to be empty, or pass --force and accept duplicated runs.")
    configs = expand(spec)
    table = lr_lookup(a.lr_from) if a.lr_from else None
    to_run, done = materialise(configs, grid_dir, table)
    print(f"{len(configs)} configs expanded -> {len(to_run) + len(done)} unique; {len(done)} already done; {len(to_run)} to run")
    if to_run:
        out_dir = to_run[0].out_dir
        failed = list_failed(out_dir)
        if failed:
            print(f"{len(failed)} FAILED runs in {out_dir} (they will be re-run if listed):")
            for rid, last in failed[:20]:
                print(f"  {rid}: {last}")
    if a.dry_run or not to_run:
        return
    py = a.python or (str(Path.home() / ".local/bin/python3.10") if (Path.home() / ".local/bin/python3.10").exists()
                      else "python3")
    script = write_slurm(grid_dir, to_run, a.partition, a.time, a.mem, a.max_par, venv=a.venv,
                         shared=a.shared, gres=a.gres, python_bin=py, cuda_tag=a.cuda_tag)
    print(f"wrote {script}\nsubmit with:  sbatch {script}")
    print("first node will spend ~10 min building its venv; later nodes install from the shared pip cache")


if __name__ == "__main__":
    main()
