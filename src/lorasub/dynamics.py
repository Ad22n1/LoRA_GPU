"""Axis A — dynamics of a LoRA update in the spectrum of W0.

    python -m lorasub.dynamics --run_dir $SHARED/runs/<run_id> --svd_cache $SHARED/svd [--intruder_steps 0,100,1000,end]

Reads ``factors_<step>.safetensors`` (LoRA arms) or ``fulldelta_<step>.safetensors`` (full FT) saved by
train.py, and for every step and module computes: decile band energies (both sides), ``top:r`` / ``bottom:r``
energies, energy outside span(U)/span(V), effective rank, and — only at ``intruder_steps`` because it needs a
dense SVD of W0 + dW — the number of intruder dimensions with the *scaling actually applied* (alpha / r).
Writes ``<run_dir>/dynamics/dynamics.csv`` and ``dynamics.png``.  Model id and scaling come from the run's
``config.yaml``; W0 is re-read from the model weights for the intruder count (not from the fp16 cache).
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
import torch
import yaml
from safetensors.torch import load_file

from .spectral import (band_energies, decile_bands, effective_rank, gradient_band_energies, intruder_count,
                       load_svd, null_reference, parse_module_name)

_STEP_RE = re.compile(r"(?:factors|fulldelta)_(\d+)\.safetensors$")


def list_checkpoints(run_dir: Path) -> list[tuple[int, Path]]:
    out = []
    for p in run_dir.glob("*.safetensors"):
        m = _STEP_RE.search(p.name)
        if m:
            out.append((int(m.group(1)), p))
    return sorted(out)


def _load_cfg(run_dir: Path) -> dict:
    with open(run_dir / "config.yaml") as f:
        return yaml.safe_load(f)


def analyze_run(run_dir: str | Path, svd_cache: str | Path, intruder_steps: set[int] | None = None,
                W0_provider=None, device="cpu", bands: list[str] | None = None) -> pd.DataFrame:
    """One row per (step, module, band, side).  ``W0_provider(name) -> Tensor`` enables the intruder count."""
    run_dir = Path(run_dir)
    cfg = _load_cfg(run_dir)
    model_id, rank = cfg["model"], int(cfg["rank"]) or 16
    scaling = float(cfg.get("alpha") or rank) / rank if cfg["mode"] != "full" else 1.0
    bands = bands or decile_bands() + [f"top:{rank}", f"bottom:{rank}"]
    cps = list_checkpoints(run_dir)
    if not cps:
        return pd.DataFrame(columns=["step", "layer", "module_type", "band", "side", "energy", "null"])
    rows: list[dict] = []
    svds: dict = {}
    for step, path in cps:
        t = load_file(str(path), device=device)
        names = sorted({k.rsplit(".", 1)[0] for k in t if k.endswith((".A", ".delta"))})
        for name in names:
            if name not in svds:
                svds[name] = load_svd(svd_cache, model_id, name, device=device)
            svd = svds[name]
            layer, mtype = parse_module_name(name)
            if f"{name}.delta" in t:  # full FT delta
                D = t[f"{name}.delta"].float()
                e = gradient_band_energies(D, svd, bands)
                norm = float(D.norm())
                s = torch.linalg.svdvals(D)
                p = s / s.sum() if float(s.sum()) > 0 else s
                eff = float(torch.exp(-(p[p > 0] * torch.log(p[p > 0])).sum())) if float(s.sum()) > 0 else 0.0
                A = B = None
            else:
                A, B = t[f"{name}.A"].float(), t[f"{name}.B"].float()
                e = band_energies(A, B, svd, bands)
                norm = float((B @ A).norm()) if step in (intruder_steps or ()) else float(
                    torch.trace((B.T @ B) @ (A @ A.T)).sqrt())
                eff = effective_rank(A, B)
            n_intr = float("nan")
            if intruder_steps and step in intruder_steps and W0_provider is not None:
                W0 = W0_provider(name)
                if A is not None:
                    n_intr = intruder_count(W0, A, B, k=10, device=device, scaling=scaling)
                else:
                    # full FT: dW = D = I_out @ D, so B = I (out x out), A = D (out x in)
                    n_intr = intruder_count(W0, D, torch.eye(D.shape[0]), k=10, device=device, scaling=1.0)
            for band in bands:
                for side, val in zip(("left", "right"), e[band]):
                    rows.append(dict(step=step, layer=layer, module_type=mtype, band=band, side=side, energy=val,
                                     null=null_reference(svd, band, side), eff_rank=eff, update_norm=norm,
                                     n_intruders=n_intr))
            for side in ("left", "right"):
                rows.append(dict(step=step, layer=layer, module_type=mtype, band="_outside", side=side,
                                 energy=e[f"_outside_{side}"], null=0.0, eff_rank=eff, update_norm=norm,
                                 n_intruders=n_intr))
    return pd.DataFrame(rows)


def plot_dynamics(df: pd.DataFrame, out: Path, side: str = "left") -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if df.empty:
        return
    dec = decile_bands()
    d = df[(df.side == side) & (df.band.isin(dec))]
    mtypes = sorted(d.module_type.unique())
    fig, axes = plt.subplots(1, len(mtypes), figsize=(3.4 * len(mtypes), 3.4), sharey=True, squeeze=False)
    axes = axes[0]
    cmap = plt.get_cmap("viridis")
    for ax, mt in zip(axes, mtypes):
        sub = d[d.module_type == mt]
        piv = sub.groupby(["step", "band"])["energy"].mean().unstack("band").reindex(columns=dec)
        nul = sub.groupby("band")["null"].mean().reindex(dec)
        steps = piv.index.values
        for i, band in enumerate(dec):
            ax.plot(steps, piv[band].values / nul[band], color=cmap(i / 9), marker=".", label=f"decile {i + 1}")
        ax.axhline(1.0, ls="--", c="gray", lw=1)
        ax.set_xscale("symlog", linthresh=10)
        ax.set_yscale("log")
        ax.set_title(mt)
        ax.set_xlabel("step")
    axes[0].set_ylabel(f"band energy / null reference ({side})")
    axes[-1].legend(fontsize=6, ncol=2)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--svd_cache", required=True)
    ap.add_argument("--intruder_steps", default="0,100,1000,end")
    ap.add_argument("--no_intruders", action="store_true")
    a = ap.parse_args()
    run_dir = Path(a.run_dir)
    cfg = _load_cfg(run_dir)
    cps = list_checkpoints(run_dir)
    last = cps[-1][0] if cps else 0
    steps = {last if s == "end" else int(s) for s in a.intruder_steps.split(",")} if not a.no_intruders else None
    provider = None
    if steps:
        from .modeling import load_model
        from .spectral import iter_target_modules

        model = load_model(cfg["model"], dtype="float32", device="cpu")
        mods = dict(iter_target_modules(model, cfg.get("target_modules") or ()))
        provider = lambda name: mods[name].weight.detach()  # noqa: E731
    df = analyze_run(run_dir, a.svd_cache, steps, provider)
    out = run_dir / "dynamics"
    out.mkdir(exist_ok=True)
    df.to_csv(out / "dynamics.csv", index=False)
    plot_dynamics(df, out / "dynamics_left.png", "left")
    plot_dynamics(df, out / "dynamics_right.png", "right")
    if not df.empty:
        top = df[(df.band.str.startswith("top")) & (df.side == "left")].groupby("step")["energy"].mean()
        bot = df[(df.band.str.startswith("bottom")) & (df.side == "left")].groupby("step")["energy"].mean()
        print(pd.DataFrame({"top_r": top, "bottom_r": bot}))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
