"""Toy experiments (axis C), corrected version of a co-author's notebook.  Pure torch, CPU, a few minutes.

    python notebooks/toy.py --out results/toy [--seeds 10] [--steps 1500]

Experiment 1 — W0 in the loss, target placed in the top or the bottom band of W0, four arms.
    Two-layer linear network y = x W0ᵀ W2ᵀ (W2 fixed, random).  We adapt W0 with dW = B A of rank r.
    The target is W0 + dW* where
        top    : dW* = U_top D V_topᵀ        (modulates directions W0 already uses; the *format-like* case)
        bottom : dW* = U_bot D V_botᵀ        (lives in the minor directions; the *facts-like* case)
    The names "format" and "facts" are deliberately NOT used here: the toy fixes where the target lives,
    which is the very thing the real experiment has to discover.
    Arms: free (A gaussian, B=0), top (B=U_top frozen), bottom (B=U_bot frozen), random (B=U_rand frozen).
    Because W2 multiplies the output of W0, the loss depends on W0's geometry (unlike y = x(W0+dW), where
    W0 cancels and every arm just fits a random matrix).  Output: relative error ||dW - dW*|| / ||dW*||
    and band energies of the learned dW, mean ± std over seeds.  A 2 x 4 figure.

Experiment 2 — the "distractor trap", corrected.
    Input covariance with a few high-variance directions (variance v_hi) carrying a tiny useless signal
    W_dist, and many low-variance directions carrying the useful signal W_star of *rank r* (so that a
    rank-r LoRA CAN represent it; the previous version used a rank-48 W_star, which confounded
    expressivity with the trap).  We sweep the ratio  rho = v_hi ||W_dist||² / v_lo ||W_star||²  — the
    ratio of the two contributions to the loss — and report the error on the *useful* part, i.e. the
    learned matrix projected on the low-variance input subspace where W_star lives (full FT also learns
    W_dist, which is in the target; measuring the whole difference would just recover ||W_dist||).
    W_dist and W_star are built on orthogonal input subspaces.

Experiment 3 — null reference: band energies of a random rank-r update are |band|/m, using the same
    band_energies() as the real code (single source of truth).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lorasub.spectral import SVDEntry, band_energies, null_reference  # noqa: E402
# No conclusions inside the images (24/09): every title is dropped, only panel letters such as "(a)"
# are kept where a caption refers to them. What a figure shows is said in its caption, where it can
# be qualified -- a claim written in the image cannot.
import re as _re
import matplotlib.axes as _max
import matplotlib.figure as _mfig
_set_title = _max.Axes.set_title
def _letter_only(self, label="", *a, **k):
    m = _re.match(r"\s*\(([a-z])\)", str(label))
    return _set_title(self, f"({m.group(1)})" if m else "", loc="left", fontsize=10)
_max.Axes.set_title = _letter_only
_mfig.Figure.suptitle = lambda self, *a, **k: None


def make_w0(d: int, decay: float = 10.0, g: torch.Generator | None = None) -> torch.Tensor:
    U, _ = torch.linalg.qr(torch.randn(d, d, generator=g))
    V, _ = torch.linalg.qr(torch.randn(d, d, generator=g))
    S = torch.exp(-torch.arange(d, dtype=torch.float32) / (d / decay))  # exponential ("dl") spectrum
    return U @ torch.diag(S) @ V.T


def entry(W: torch.Tensor) -> SVDEntry:
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    return SVDEntry("toy", tuple(W.shape), S, U, Vh)


# --------------------------------------------------------------------------- #
# Experiment 1
# --------------------------------------------------------------------------- #
def run_arm(W0, W2, X, Y, mode: str, r: int, steps: int, lr: float, e: SVDEntry, g: torch.Generator):
    d = W0.shape[0]
    if mode == "free":
        A = (torch.randn(r, d, generator=g) / d**0.5).requires_grad_(True)
        B = torch.zeros(d, r).requires_grad_(True)
        params = [A, B]
    else:
        idx = {"top": torch.arange(r), "bottom": torch.arange(d - r, d),
               "random": torch.randperm(d, generator=g)[:r]}[mode]
        B = e.U[:, idx].clone()
        A = torch.zeros(r, d).requires_grad_(True)
        params = [A]
    opt = torch.optim.Adam(params, lr=lr)
    for _ in range(steps):
        pred = X @ (W0 + B @ A).T @ W2.T
        loss = ((pred - Y) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return A.detach(), B.detach(), float(loss.detach())


def experiment_1(d=64, r=8, n=2000, seeds=10, steps=1500, lr=1e-2, out: Path | None = None):
    arms = ["free", "top", "bottom", "random"]
    targets = ["top-band", "bottom-band"]
    res = {t: {a: {"err": [], "top": [], "bottom": []} for a in arms} for t in targets}
    for seed in range(seeds):
        g = torch.Generator().manual_seed(seed)
        W0 = make_w0(d, g=g)
        e = entry(W0)
        W2 = torch.randn(d, d, generator=g) / d**0.5
        X = torch.randn(n, d, generator=g)
        D = torch.diag(torch.rand(r, generator=g) + 0.5) * 0.3 * float(e.S[0])
        for t in targets:
            if t == "top-band":
                dW_star = e.U[:, :r] @ D @ e.V[:, :r].T
            else:
                dW_star = e.U[:, -r:] @ D @ e.V[:, -r:].T
            Y = X @ (W0 + dW_star).T @ W2.T
            for a in arms:
                A, B, _ = run_arm(W0, W2, X, Y, a, r, steps, lr, e, g)
                dW = B @ A
                res[t][a]["err"].append(float((dW - dW_star).norm() / dW_star.norm()))
                be = band_energies(A, B, e, [f"top:{r}", f"bottom:{r}"])
                res[t][a]["top"].append(be[f"top:{r}"][0])
                res[t][a]["bottom"].append(be[f"bottom:{r}"][0])
    summary = {}
    for t in targets:
        for a in arms:
            for k in ("err", "top", "bottom"):
                v = torch.tensor(res[t][a][k])
                summary[(t, a, k)] = (float(v.mean()), float(v.std()))
    print("\nExperiment 1 — relative error ||dW - dW*|| / ||dW*||  (mean ± std over seeds; 1.0 = no learning)")
    print(f"{'target':12} " + " ".join(f"{a:>14}" for a in arms))
    for t in targets:
        print(f"{t:12} " + " ".join(f"{summary[(t, a, 'err')][0]:6.3f}±{summary[(t, a, 'err')][1]:5.3f}" for a in arms))
    print("\nEnergy of the learned dW in top-r / bottom-r of W0 (free arm shows where it *chose* to go):")
    for t in targets:
        print(f"{t:12} free: top={summary[(t, 'free', 'top')][0]:.2f} bottom={summary[(t, 'free', 'bottom')][0]:.2f}"
              f"   null={null_reference(d, f'top:{r}'):.2f}")
    if out:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        colors = {"free": "#4C72B0", "top": "#DD8452", "bottom": "#55A868", "random": "#8C8C8C"}
        fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4), sharey=True)
        for ax, t, sub in zip(axes, targets, ("(a) target in the top band of $W_0$",
                                              "(b) target in the bottom band of $W_0$")):
            m = [summary[(t, a, "err")][0] for a in arms]
            e = [summary[(t, a, "err")][1] for a in arms]
            ax.bar(arms, m, yerr=e, capsize=4, color=[colors[a] for a in arms], edgecolor="black", linewidth=0.6)
            ax.axhline(1.0, ls="--", c="black", lw=1)
            ax.text(-0.45, 1.02, "no learning", ha="left", va="bottom", fontsize=8, style="italic")
            ax.set_title(sub, fontsize=10)
            ax.set_ylim(0, 1.28)
            # tick labels on BOTH panels: sharey already forces one scale, but hiding the right-hand
            # labels makes a quick reader suspect two different scales — and a comparison between the
            # panels is the whole point of the figure.
            ax.tick_params(axis="y", labelleft=True)
            ax.spines[["top", "right"]].set_visible(False)
        axes[0].set_ylabel(r"relative error  $\|\Delta W - \Delta W^*\| / \|\Delta W^*\|$", fontsize=9)
        fig.suptitle("A constrained arm recovers a target only when it lies in its own spectral band",
                     fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        out.mkdir(parents=True, exist_ok=True)
        fig.savefig(out / "toy1_arms_vs_target.png", dpi=200, bbox_inches="tight")
        fig.savefig(out / "toy1_arms_vs_target.pdf", bbox_inches="tight")
        plt.close(fig)

        # where the free arm put its energy: it finds the right band on its own
        fig, ax = plt.subplots(figsize=(4.6, 3.2))
        x = range(len(targets))
        w = 0.38
        ax.bar([i - w / 2 for i in x], [summary[(t, "free", "top")][0] for t in targets], w,
               label="energy in top-r", color=colors["top"], edgecolor="black", linewidth=0.6)
        ax.bar([i + w / 2 for i in x], [summary[(t, "free", "bottom")][0] for t in targets], w,
               label="energy in bottom-r", color=colors["bottom"], edgecolor="black", linewidth=0.6)
        ax.axhline(null_reference(d, f"top:{r}"), ls="--", c="black", lw=1)
        ax.text(-0.48, null_reference(d, f"top:{r}") + 0.015, "null reference", ha="left", va="bottom",
                fontsize=8, style="italic")
        ax.set_xticks(list(x))
        ax.set_xticklabels(["target: top band", "target: bottom band"], fontsize=9)
        ax.set_ylabel("fraction of $\\|\\Delta W\\|^2$", fontsize=9)
        ax.set_ylim(0, 1.28)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=8, frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.0))
        ax.set_title("The free arm finds the right band by itself", fontsize=10)
        fig.tight_layout()
        fig.savefig(out / "toy1b_free_arm_energy.png", dpi=200, bbox_inches="tight")
        fig.savefig(out / "toy1b_free_arm_energy.pdf", bbox_inches="tight")
        plt.close(fig)
    return summary


# --------------------------------------------------------------------------- #
# Experiment 2
# --------------------------------------------------------------------------- #
def experiment_2(d=50, r=2, n_hi=2, n=4000, seeds=5, steps=1500, lr=5e-2, out: Path | None = None):
    ratios = [0.01, 0.1, 0.3, 0.6, 1.0, 2.0, 4.0, 10.0, 100.0]  # finer around the transition (rho ~ 1)
    v_hi, v_lo = 100.0, 1.0
    err_lora = {rho: [] for rho in ratios}
    err_full = {rho: [] for rho in ratios}
    frac_hi = {rho: [] for rho in ratios}
    for seed in range(seeds):
        g = torch.Generator().manual_seed(100 + seed)
        Q, _ = torch.linalg.qr(torch.randn(d, d, generator=g))
        Q_hi, Q_lo = Q[:, :n_hi], Q[:, n_hi:]
        std = torch.cat([torch.full((n_hi,), v_hi**0.5), torch.full((d - n_hi,), v_lo**0.5)])
        X = (torch.randn(n, d, generator=g) * std) @ Q.T  # covariance Q diag(v) Qᵀ
        # useful signal: rank r, on the low-variance input subspace
        W_star = Q_lo @ (torch.randn(d - n_hi, r, generator=g) / (d - n_hi)**0.5) @ torch.randn(r, d, generator=g)
        W_star = W_star / W_star.norm()  # ||W_star|| = 1
        # distractor: on the high-variance subspace, orthogonal input support; norm set by the ratio
        W_dist0 = Q_hi @ torch.randn(n_hi, d, generator=g)
        W_dist0 = W_dist0 / W_dist0.norm()
        for rho in ratios:
            # rho = v_hi ||W_dist||² / (v_lo ||W_star||²)  ->  ||W_dist|| = sqrt(rho v_lo / v_hi)
            W_dist = W_dist0 * (rho * v_lo / v_hi) ** 0.5
            Y = X @ (W_star + W_dist)
            A = (torch.randn(r, d, generator=g) * 0.01).requires_grad_(True)
            B = torch.zeros(d, r).requires_grad_(True)
            Wf = torch.zeros(d, d).requires_grad_(True)
            opt1, opt2 = torch.optim.Adam([A, B], lr=lr), torch.optim.Adam([Wf], lr=lr)
            for _ in range(steps):
                l1 = ((X @ (B @ A) - Y) ** 2).mean()
                opt1.zero_grad()
                l1.backward()
                opt1.step()
                l2 = ((X @ Wf - Y) ** 2).mean()
                opt2.zero_grad()
                l2.backward()
                opt2.step()
            W_lora = (B @ A).detach()
            P_lo = Q_lo @ Q_lo.T  # projector on the useful input subspace
            err_lora[rho].append(float((P_lo @ (W_lora - W_star)).norm()))
            err_full[rho].append(float((P_lo @ (Wf.detach() - W_star)).norm()))
            frac_hi[rho].append(float((Q_hi.T @ W_lora).norm() ** 2 / W_lora.norm() ** 2))
    print("\nExperiment 2 — error on the useful part of W_star (||W_star|| = 1) vs loss ratio distractor/signal")
    print("  (the variance at rho ~ 1 is the point: near the transition, LoRA falls into the trap or not")
    print("   depending on the seed)")
    print(f"{'rho':>7} {'LoRA r=2':>12} {'full FT':>12} {'LoRA energy on distractor dirs':>32}")
    for rho in ratios:
        el, ef, fh = torch.tensor(err_lora[rho]), torch.tensor(err_full[rho]), torch.tensor(frac_hi[rho])
        print(f"{rho:7.2f} {el.mean():6.3f}±{el.std():4.3f} {ef.mean():6.3f}±{ef.std():4.3f} {fh.mean():16.2f}")
    if out:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(5.4, 3.6))
        for name, dct, c in (("LoRA, rank $r$ = rank of $W^*$", err_lora, "#4C72B0"),
                             ("full fine-tuning", err_full, "#DD8452")):
            m = [float(torch.tensor(dct[rho]).mean()) for rho in ratios]
            sd = [float(torch.tensor(dct[rho]).std()) for rho in ratios]
            ax.errorbar(ratios, m, yerr=sd, marker="o", ms=4, capsize=3, label=name, color=c, lw=1.5)
        ax.axvline(1.0, ls=":", c="gray", lw=1)
        ax.text(1.08, 0.05, "equal loss\ncontribution", fontsize=8, color="gray")
        ax.set_xscale("log")
        ax.set_xlabel(r"loss-contribution ratio  $\rho$ = distractor / useful signal", fontsize=9)
        ax.set_ylabel(r"error on the useful part, $\|P_{lo}(W - W^*)\|$", fontsize=9)
        ax.set_ylim(bottom=-0.05)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=8, frameon=False, loc="upper left")
        ax.set_title("Rank-limited LoRA is captured by high-variance distractors", fontsize=10)
        fig.tight_layout()
        out.mkdir(parents=True, exist_ok=True)
        fig.savefig(out / "toy2_distractor_trap.png", dpi=200, bbox_inches="tight")
        fig.savefig(out / "toy2_distractor_trap.pdf", bbox_inches="tight")
        plt.close(fig)
    return err_lora, err_full


# --------------------------------------------------------------------------- #
# Experiment 3
# --------------------------------------------------------------------------- #
def experiment_3(d=64, r=8, draws=50):
    g = torch.Generator().manual_seed(0)
    e = entry(make_w0(d, g=g))
    bands = [f"top:{r}", f"bottom:{r}", "frac:0.0-0.25"]
    acc = {b: 0.0 for b in bands}
    for _ in range(draws):
        A, B = torch.randn(r, d, generator=g), torch.randn(d, r, generator=g)
        be = band_energies(A, B, e, bands)
        for b in bands:
            acc[b] += be[b][0] / draws
    print("\nExperiment 3 — null reference (random rank-r update):")
    for b in bands:
        print(f"  {b:15} measured {acc[b]:.3f}   expected {null_reference(d, b):.3f}")
    return acc


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/toy")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--steps", type=int, default=1500)
    a = ap.parse_args()
    out = Path(a.out)
    torch.manual_seed(0)
    experiment_1(seeds=a.seeds, steps=a.steps, out=out)
    experiment_2(seeds=max(3, a.seeds // 2), steps=a.steps, out=out)
    experiment_3()
    print(f"\nfigures in {out}")
