"""Step 0 of the paper: where does the gradient live, per module, per dataset — without training.

    python -m lorasub.grad_probe --model meta-llama/Llama-3.2-1B --data_dir $SHARED/data \
        --svd_cache $SHARED/svd --out results/grad_probe --n 256 --batch 8 --group_layers 4

For each dataset (facts train, format train) the mean weight gradient G of every target
module is accumulated over ``n`` examples with the *same loss masking as training*, then
projected on the spectral bands of W0 (deciles + top:16 / bottom:16), on both sides.
Layers are processed in groups so that fp32 gradients fit in 24 GB on 7B models.
Output: a CSV (model, dataset, layer, module_type, band, side, energy, null) and two figures.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from . import set_seed
from .data.facts import read_jsonl
from .data.loaders import build_dataset, collate, stratified_rows
from .data.paths import resolve
from .modeling import load_model, load_tokenizer
from .models import target_modules_for
from .spectral import (TARGET_SUFFIXES, decile_bands, gradient_band_energies, iter_target_modules, load_svd,
                       null_reference, numerical_rank, parse_module_name)

DEFAULT_BANDS = decile_bands() + ["top:16", "bottom:16"]


@torch.no_grad()
def gradient_effective_rank(G: torch.Tensor) -> float:
    """exp(H(p)) with p = sigma / sum(sigma) — entropy of the SINGULAR VALUES, not of the energies.

    ⚠ This is NOT a count of occupied directions, and must not be read as one. Because p weights
    sigma and not sigma^2, a thousand tiny singular values carry a lot of entropy while carrying
    almost no energy: a value of 1000 is compatible with 95 % of the energy living in 16 directions.
    It is also inflated by the number of supervised tokens (see ``noise_ceiling``). Kept as a
    diagnostic for continuity with earlier runs; report ``energy_at_rank`` instead.
    """
    s = torch.linalg.svdvals(G.float())
    tot = float(s.sum())
    if tot <= 0:
        return 0.0
    p = s / tot
    p = p[p > 1e-12]
    return float(torch.exp(-(p * torch.log(p)).sum()))


@torch.no_grad()
def stable_rank(G: torch.Tensor) -> float:
    """||G||_F^2 / ||G||_2^2 — energy-weighted, insensitive to a flat tail, unlike exp(H(p))."""
    s = torch.linalg.svdvals(G.float())
    if s.numel() == 0 or float(s[0]) <= 0:
        return 0.0
    return float((s.pow(2).sum() / s[0].pow(2)))


@torch.no_grad()
def energy_at_rank(G: torch.Tensor, ranks=(1, 2, 4, 8, 16, 32, 64)) -> dict[int, float]:
    """Fraction of ||G||_F^2 captured by the best rank-r approximation, for each r.

    This is the metric to report: it is energy-weighted, it has a direct operational meaning (what a
    rank-r adapter could represent at best), and it is directly comparable to the rank axis of the
    grid. Eckart-Young: the best rank-r error is the tail of the squared singular values.
    """
    s2 = torch.linalg.svdvals(G.float()).pow(2)
    tot = float(s2.sum())
    if tot <= 0:
        return {r: 0.0 for r in ranks}
    cum = torch.cumsum(s2, 0) / tot
    return {r: float(cum[min(r, cum.numel()) - 1]) for r in ranks}


@torch.no_grad()
def spectrum_summary(G: torch.Tensor, keep: int = 128) -> list[float]:
    """The first ``keep`` singular values, so any other statistic can be recomputed without re-running
    the probe (that re-run costs an hour of PARTITION; storing 128 floats costs nothing)."""
    s = torch.linalg.svdvals(G.float())
    return [float(x) for x in s[:keep]]


def noise_ceiling(n_terms: int, shape: tuple[int, int]) -> float:
    """exp(H(p)) expected for a mean of ``n_terms`` INDEPENDENT rank-1 terms of the given shape.

    ``n_terms`` must be the number of SUPERVISED TOKENS, not the number of batches. The gradient of a
    linear layer is G = sum_t delta_t x_t^T, one rank-1 term per supervised token, so a batch already
    contributes thousands of terms. Counting batches gave a ceiling of ~120 on gate_proj where the
    measurement was 400-1000, i.e. a ratio above 1, which made the normalised quantity meaningless.

    Calibration reference: a gradient whose per-example contributions are uncorrelated sits near this
    ceiling, whatever the task. Comparing raw effective ranks across modules of different shapes (GQA
    makes k/v four times narrower) or across tasks with different token counts is meaningless without
    it. Empirical fit from Monte-Carlo on Gaussian rank-1 terms; an order of magnitude, not a norm —
    real gradients are not independent.
    """
    m = min(shape)
    return float(min(m, 0.58 * (n_terms ** 0.5) * (m ** 0.5)))


class OutputBandShare:
    """Fraction of a layer's actual OUTPUT that lives in each candidate band, on real data.

    The weight-space view (how much of ||W0|| the band carries) is not what the loss sees. What the
    loss sees is the output: an arm can only change ``y = W0 x`` inside the subspace its frozen B
    spans, so the quantity that bounds its influence is ``||U_band^T W0 x||^2 / ||W0 x||^2`` measured on
    real activations. A band may hold a large share of the weight energy and almost none of the output,
    or the reverse, because the input distribution is not isotropic — that mismatch is exactly what a
    purely spectral argument misses.

    Collected by forward hooks; costs one forward pass, no backward, no training.
    """

    def __init__(self, modules: dict, svds: dict, ranks=(1, 2, 4)):
        self.ranks = tuple(ranks)
        self.svds = svds
        self.acc = {n: {} for n in modules}
        self.total = dict.fromkeys(modules, 0.0)
        self.handles = [m.register_forward_hook(self._hook(n)) for n, m in modules.items()]

    def _hook(self, name):
        def hook(_m, _inp, out):
            with torch.no_grad():
                svd = self.svds.get(name)
                if svd is None:
                    return
                y = out.detach().reshape(-1, out.shape[-1]).float()
                U = svd.U.to(y.device).float()
                self.total[name] += float(y.pow(2).sum())
                for r in self.ranks:
                    if r > svd.m:
                        continue
                    for label, sl in (("top", slice(0, r)), ("bottom", slice(svd.m - r, svd.m))):
                        e = float((y @ U[:, sl]).pow(2).sum())
                        k = f"{label}{r}"
                        self.acc[name][k] = self.acc[name].get(k, 0.0) + e
        return hook

    def shares(self, name: str) -> dict[str, float]:
        tot = self.total.get(name, 0.0)
        if tot <= 0:
            return {}
        out = {f"output_share_{k}": v / tot for k, v in self.acc[name].items()}
        # null reference: an isotropic output would put r/d_out of its energy in any r directions
        svd = self.svds.get(name)
        if svd is not None:
            for r in self.ranks:
                if r <= svd.m:
                    out[f"output_share_null{r}"] = r / svd.U.shape[0]
        return out

    def close(self):
        for h in self.handles:
            h.remove()
        self.handles = []


@torch.no_grad()
def band_gradient_snr(G: torch.Tensor, Ga: torch.Tensor, Gb: torch.Tensor, U: torch.Tensor,
                      idx: torch.Tensor, scaling: float = 1.0) -> dict[str, float]:
    """Signal-to-noise ratio of the gradient a CONSTRAINED ARM would receive on its trainable factor.

    With dW = (alpha/r) B A and B = U[:, idx] frozen, dL/dA = (alpha/r) B^T (dL/dW): the arm only ever
    sees the projection of the gradient onto its own band. Two quantities follow, and only one of them
    matters for the optimiser in use.

    ``grad_norm`` is the magnitude of that projection. It does NOT explain a difference in optimal
    learning rate under AdamW: Adam divides by the running RMS of the gradient, so a gradient a hundred
    times smaller produces exactly the same step — verified. The magnitude would matter under SGD.

    ``snr`` is what Adam does respond to: the mean gradient over two disjoint halves of the data,
    relative to their disagreement. A band whose gradient is consistent across data gets a large,
    stable second moment and an effective step close to the nominal rate; a band whose gradient is
    mostly noise gets its step shrunk by Adam's normalisation. This is the gradient-noise-scale
    quantity of McCandlish et al., restricted to a band.
    """
    P = U[:, idx]                                   # d_out x r
    g = P.T @ G.to(P.dtype)
    ga, gb = P.T @ Ga.to(P.dtype), P.T @ Gb.to(P.dtype)
    mean_norm = float(g.norm()) * scaling
    disagreement = float((ga - gb).norm()) * scaling / 2.0     # noise on the mean of two halves
    # Numerator and denominator are reported SEPARATELY as well as their ratio: the projected mean
    # also carries the magnitude of the signal in that band, and the gradient energy is known to be
    # non-uniform across the spectrum. Reporting only the ratio would confound an SNR effect with an
    # energy effect.
    return {"grad_norm_in_band": mean_norm,
            "grad_noise_in_band": disagreement,
            "grad_snr_in_band": mean_norm / disagreement if disagreement > 0 else float("nan")}


@torch.no_grad()
def split_half_coherence(G_a: torch.Tensor, G_b: torch.Tensor, k: int = 16) -> dict[str, float]:
    """Do two disjoint halves of the data produce the same gradient directions?

    Tests 'systematic vs idiosyncratic' directly. A gradient whose per-example contributions agree
    (every format example asks for the same transformation) gives a high cosine and a large overlap
    of top-k subspaces; one whose contributions are example-specific (each fact is a different entity)
    gives a low cosine and little overlap, even if both halves have a heavy tail.

    Returns the cosine between the two mean gradients, and the mean squared canonical correlation
    between their top-k left singular subspaces (1 = same subspace, k/min(shape) = chance).
    """
    a, b = G_a.float(), G_b.float()
    na, nb = float(a.norm()), float(b.norm())
    cos = float((a * b).sum() / (na * nb)) if na > 0 and nb > 0 else 0.0
    Ua = torch.linalg.svd(a, full_matrices=False)[0][:, :k]
    Ub = torch.linalg.svd(b, full_matrices=False)[0][:, :k]
    sv = torch.linalg.svdvals(Ua.T @ Ub).clamp(0, 1)
    return {"cosine": cos, "subspace_overlap": float(sv.pow(2).mean()),
            "overlap_chance": k / min(a.shape)}


class XDCollector:
    """Stable ranks of the inputs X and of the backpropagated signals D, per module.

    A linear layer's gradient is G = D^T X with one rank-1 term per supervised token, so the ALGEBRAIC
    rank obeys rank(G) <= min(rank X, rank D). Measuring the two separately says which of them limits
    the gradient, and therefore whether a depth profile reflects the diversity of the representations
    entering the layer or that of the error signals leaving it — the measurement that discriminates
    the readings of the depth profile.

    ⚠ The quantity reported is the STABLE rank, ||.||_F^2 / ||.||_2^2, which is energy-weighted and
    robust, but for which the bound above does NOT hold: a product can have a larger stable rank than
    either factor. The two numbers are diagnostic — they say which factor is spread and which is
    concentrated — not a formal bound on the gradient.

    Memory. The exact Gram X^T X is d_in^2 floats — 268 MB for down_proj alone — so the token vectors
    are first projected onto ``proj_dim`` fixed random directions. By Johnson-Lindenstrauss this
    preserves the geometry, hence the stable rank, as long as the true stable rank is well below
    proj_dim; values approaching proj_dim must be read as "at least that much", not as a measurement.
    """

    def __init__(self, modules: dict, proj_dim: int = 1024, seed: int = 0, device="cpu"):
        self.proj_dim = proj_dim
        self.dev = device
        g = torch.Generator().manual_seed(seed)
        self.gram_x, self.gram_d, self.projx, self.projd, self.handles = {}, {}, {}, {}, []
        for name, mod in modules.items():
            din, dout = mod.weight.shape[1], mod.weight.shape[0]
            kx, kd = min(proj_dim, din), min(proj_dim, dout)
            self.projx[name] = (torch.randn(din, kx, generator=g) / kx**0.5) if kx < din else None
            self.projd[name] = (torch.randn(dout, kd, generator=g) / kd**0.5) if kd < dout else None
            self.gram_x[name] = torch.zeros(kx, kx, dtype=torch.float32)
            self.gram_d[name] = torch.zeros(kd, kd, dtype=torch.float32)
            self.handles.append(mod.register_forward_hook(self._fwd(name)))
            # full_backward_hook warns when no input requires grad and then fires on the gradient
            # w.r.t. the module OUTPUT — which is exactly D here, so the warning confirms the intent.
            self.handles.append(mod.register_full_backward_hook(self._bwd(name)))

    def _accum(self, store, proj, name, t):
        x = t.reshape(-1, t.shape[-1]).float().cpu()
        if proj[name] is not None:
            x = x @ proj[name]
        store[name] += (x.T @ x)

    def _fwd(self, name):
        def hook(_m, inp, _out):
            with torch.no_grad():
                self._accum(self.gram_x, self.projx, name, inp[0].detach())
        return hook

    def _bwd(self, name):
        def hook(_m, _gi, go):
            with torch.no_grad():
                self._accum(self.gram_d, self.projd, name, go[0].detach())
        return hook

    def stable_ranks(self, name: str) -> dict[str, float]:
        """||.||_F^2 / ||.||_2^2 of X and D, read off their Gram matrices."""
        out = {}
        for key, store in (("X", self.gram_x), ("D", self.gram_d)):
            ev = torch.linalg.eigvalsh(store[name].double()).clamp_min(0)   # eigenvalues of M^T M
            tot, top = float(ev.sum()), float(ev.max())
            out[f"stable_rank_{key}"] = (tot / top) if top > 0 else 0.0
            out[f"proj_dim_{key}"] = float(store[name].shape[0])
        return out

    def close(self):
        for h in self.handles:
            h.remove()
        self.handles = []


@torch.no_grad()
def activation_norm_profile(model, loader, names: list[str], device, n_batches: int = 4) -> dict:
    """Max / mean L2 norm of the INPUT of each module, per token position.

    Diagnostic for a collapse of the effective rank that affects both tasks alike, hence
    architectural rather than task-related: a handful of very-high-norm tokens (first position, BOS,
    delimiters) can dominate the sum of rank-1 terms and crush the entropy. Reported so that such a
    collapse is attributed to the architecture instead of being read as a property of the task.
    """
    mods = dict(iter_target_modules(model))
    stats = {n: {"max": 0.0, "sum": 0.0, "count": 0, "argmax_pos": -1} for n in names}
    handles = []

    def mk(name):
        def hook(_m, inp, _out):
            x = inp[0].detach().float()
            nrm = x.norm(dim=-1)                       # (batch, positions)
            mx, pos = float(nrm.max()), int(nrm.max(dim=1).indices[0])
            if mx > stats[name]["max"]:
                stats[name]["max"], stats[name]["argmax_pos"] = mx, pos
            stats[name]["sum"] += float(nrm.sum())
            stats[name]["count"] += nrm.numel()
        return hook

    for n in names:
        handles.append(mods[n].register_forward_hook(mk(n)))
    try:
        for i, batch in enumerate(loader):
            model(**{k: v.to(device) for k, v in batch.items()})
            if i + 1 >= n_batches:
                break
    finally:
        for h in handles:
            h.remove()
    return {n: {"act_norm_max": v["max"], "act_norm_mean": v["sum"] / max(1, v["count"]),
                "act_norm_argmax_pos": v["argmax_pos"],
                "act_norm_ratio": v["max"] / max(1e-9, v["sum"] / max(1, v["count"]))}
            for n, v in stats.items()}


def layer_groups(names: list[str], group_size: int) -> list[list[str]]:
    by_layer: dict[int, list[str]] = {}
    for n in names:
        layer, _ = parse_module_name(n)
        by_layer.setdefault(layer, []).append(n)
    layers = sorted(by_layer)
    return [sum((by_layer[l] for l in layers[i : i + group_size]), []) for i in range(0, len(layers), group_size)]


def mean_gradients(model, loader: DataLoader, names: list[str], n_examples: int, device,
                   suffixes=TARGET_SUFFIXES, split_half: bool = True) -> tuple[dict, dict, dict, int, int]:
    """Mean of dL/dW over ``n_examples`` for the modules in ``names`` (others stay frozen).

    Also accumulates the gradient over two DISJOINT halves of the batches (even and odd), so that
    ``split_half_coherence`` can test whether the direction is reproducible across data. Returns
    (mean, half_a, half_b, n_batches, n_examples_seen).
    """
    mods = dict(iter_target_modules(model, suffixes))
    for p in model.parameters():
        p.requires_grad_(False)
    for n in names:
        mods[n].weight.requires_grad_(True)
    zeros = lambda: {n: torch.zeros_like(mods[n].weight, dtype=torch.float32, device="cpu") for n in names}  # noqa: E731
    acc, half_a, half_b = zeros(), (zeros() if split_half else {}), (zeros() if split_half else {})
    n_a = n_b = 0
    seen, n_batches = 0, 0
    model.train()  # dropout is 0 in these models; train() only matters for checkpointing
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        loss = model(**batch).loss
        loss.backward()
        for n in names:
            g = mods[n].weight.grad.detach().float().cpu()
            acc[n] += g
            if split_half:
                (half_a if n_batches % 2 == 0 else half_b)[n] += g
            mods[n].weight.grad = None
        if split_half:
            n_a, n_b = n_a + (n_batches % 2 == 0), n_b + (n_batches % 2 == 1)
        n_batches += 1
        seen += batch["input_ids"].shape[0]
        if seen >= n_examples:
            break
    for n in names:
        acc[n] /= max(1, n_batches)
        if split_half:
            half_a[n] /= max(1, n_a)
            half_b[n] /= max(1, n_b)
    for n in names:
        mods[n].weight.requires_grad_(False)
    return acc, half_a, half_b, n_batches, seen


def probe(model, tokenizer, model_id: str, data_dir: str, svd_cache: str, datasets=("facts", "format"),
          n_examples: int = 256, batch_size: int = 8, max_len: int = 512, group_layers: int = 4,
          bands: list[str] = DEFAULT_BANDS, device=None, seed: int = 0, suffixes=None,
          ranks=(1, 2, 4, 8, 16, 32, 64), split_half: bool = True, keep_spectrum: int = 128,
          match_sup_tokens: int | None = None, xd_decomposition: bool = False,
          proj_dim: int = 1024, arm_ranks: tuple = (1, 2, 4),
          output_shares: bool = True, sigma_rel_tol: float | None = 1e-6) -> pd.DataFrame:
    """Where does the mean gradient live, per module, per dataset — before any training.

    Reports, for every module: band energies (both sides, against the exact null reference), the
    fraction of energy captured at each rank (the metric to read), the stable rank, the raw
    exp(H(p)) with its noise ceiling for calibration, the split-half coherence, and the first
    ``keep_spectrum`` singular values so nothing has to be re-run.

    ``match_sup_tokens`` stops each dataset once that many supervised tokens have been seen, so the
    two tasks are compared at equal supervision — the entropy of the singular values grows with the
    number of independent contributions, so an unmatched comparison confounds task and token count.
    """
    dev = device or next(model.parameters()).device
    suffixes = tuple(suffixes) if suffixes else target_modules_for(model_id)
    names = [n for n, _ in iter_target_modules(model, suffixes)]
    groups = layer_groups(names, group_layers)
    rows: list[dict] = []
    spectra: list[dict] = []
    for task in datasets:
        set_seed(seed)
        # "format_unmasked" is a CONTROL, not a task: the format data supervised on every token, like
        # facts. If the facts/format difference survives it, the difference is about the tasks; if it
        # disappears, it was an artefact of the loss masking.
        base_task = "format" if task == "format_unmasked" else task
        mask_prompt = task != "format_unmasked"
        all_rows = read_jsonl(resolve(data_dir, base_task, "train").path)
        # Size the SUBSET from the token budget, not only the example count: with match_sup_tokens the
        # loader must actually be able to deliver that many supervised tokens. Sizing it from --n only
        # (the previous behaviour) made the loader run dry long before the budget, silently comparing
        # the two tasks at unequal supervision — the exact confound the option exists to remove.
        probe_rows = build_dataset(base_task, data_dir, tokenizer, max_len, split="train", seed=seed,
                                   rows=stratified_rows(all_rows, min(64, len(all_rows)), key="attr",
                                                        seed=seed),
                                   mask_prompt=mask_prompt)
        tok_per_row = probe_rows.n_tokens / max(1, min(64, len(all_rows)))   # supervised tokens per RAW row
        if match_sup_tokens is None:
            n_rows = n_examples * (4 if base_task == "facts" else 1)
        else:
            n_rows = int(match_sup_tokens / max(1e-9, tok_per_row) * 1.3) + 8   # 30 % margin
        if n_rows > len(all_rows):
            print(f"[probe] WARNING {task}: need ~{n_rows} rows for the token budget, the dataset has "
                  f"{len(all_rows)}. The two tasks will NOT be compared at equal supervision.")
        sub = stratified_rows(all_rows, min(n_rows, len(all_rows)), key="attr", seed=seed)
        ds = build_dataset(base_task, data_dir, tokenizer, max_len, split="train", seed=seed, rows=sub,
                           mask_prompt=mask_prompt)
        sup_per_example = ds.n_tokens / max(1, len(ds))
        if match_sup_tokens is None:
            n_eff = n_examples
        else:
            n_eff = max(1, int(match_sup_tokens / max(1e-9, sup_per_example)))
            if n_eff > len(ds):
                print(f"[probe] WARNING {task}: budget needs {n_eff} batched examples, the loader can "
                      f"only supply {len(ds)} ({ds.n_tokens} supervised tokens vs {match_sup_tokens} "
                      f"requested). Equal-supervision comparison NOT achieved.")
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True,
                            generator=torch.Generator().manual_seed(seed),
                            collate_fn=lambda b: collate(b, tokenizer.pad_token_id))
        for grp in groups:
            obs = None
            if output_shares:
                mods_g = {n: m for n, m in iter_target_modules(model, suffixes) if n in grp}
                svds_g = {n: load_svd(svd_cache, model_id, n, device=dev) for n in grp}
                obs = OutputBandShare(mods_g, svds_g, ranks=arm_ranks)
            xd = None
            if xd_decomposition:
                mods_grp = {n: m for n, m in iter_target_modules(model, suffixes) if n in grp}
                xd = XDCollector(mods_grp, proj_dim=proj_dim, seed=seed)
            G, Ga, Gb, n_batches, n_seen = mean_gradients(model, loader, grp, n_eff, dev, suffixes,
                                                          split_half=split_half)
            sup_tokens_seen = sup_per_example * n_seen
            for n in grp:
                svd = load_svd(svd_cache, model_id, n, device=dev)
                Gd = G[n].to(dev)
                e = gradient_band_energies(Gd, svd, bands)
                layer, mtype = parse_module_name(n)
                gnorm = float(Gd.norm())
                geff = gradient_effective_rank(Gd)
                srank = stable_rank(Gd)
                eatr = energy_at_rank(Gd, ranks)
                ceiling = noise_ceiling(int(sup_tokens_seen), tuple(Gd.shape))
                coh = (split_half_coherence(Ga[n].to(dev), Gb[n].to(dev)) if split_half
                       else {"cosine": float("nan"), "subspace_overlap": float("nan"),
                             "overlap_chance": float("nan")})
                # SNR of the gradient each constrained arm would actually receive on its factor A.
                # This is the quantity AdamW responds to; the magnitude alone is not (Adam normalises
                # by the running RMS, so a 100x smaller gradient gives the same step).
                snr_cols: dict[str, float] = {}
                if split_half and arm_ranks:
                    # the SNR of a mean falls as 1/sqrt(N): the three arms must be compared at the
                    # SAME number of averaged examples, and that number goes in the CSV. This is the
                    # trap already met on the effective-rank probe.
                    nrank = numerical_rank(svd.S, sigma_rel_tol) if sigma_rel_tol else svd.m
                    for r_arm in arm_ranks:
                        if r_arm > svd.m:
                            continue
                        arm_bands = [("top", torch.arange(r_arm))]   # NOT `bands`: that name is
                        # the spectral band list of probe() and shadowing it broke the band loop below
                        # bottom band taken inside the numerically non-zero part, else its singular
                        # vectors are ill-conditioned and the SNR there is numerical noise
                        if nrank >= r_arm:
                            arm_bands.append(("bottom", torch.arange(nrank - r_arm, nrank)))
                        for label, sl in arm_bands:
                            v = band_gradient_snr(Gd, Ga[n].to(dev), Gb[n].to(dev), svd.U, sl.to(dev))
                            for k, val in v.items():
                                snr_cols[f"{k}_{label}{r_arm}"] = val
                    snr_cols["snr_n_examples"] = float(n_seen)
                    snr_cols["numerical_rank"] = float(nrank)
                out_cols = obs.shares(n) if obs is not None else {}
                xd_cols = xd.stable_ranks(n) if xd is not None else {}
                common = dict(model=model_id, dataset=task, layer=layer, module_type=mtype,
                              **xd_cols, **snr_cols, **out_cols,
                              grad_norm=gnorm, grad_eff_rank=geff, grad_eff_rank_ceiling=ceiling,
                              grad_eff_rank_frac=geff / ceiling if ceiling > 0 else float("nan"),
                              stable_rank=srank, n_batches=n_batches, n_examples_seen=n_seen,
                              sup_tokens_per_example=sup_per_example,
                              sup_tokens_total=sup_tokens_seen,
                              split_cosine=coh["cosine"], split_overlap=coh["subspace_overlap"],
                              split_overlap_chance=coh["overlap_chance"],
                              **{f"energy_at_r{r}": v for r, v in eatr.items()})
                for band in bands:
                    left, right = e[band]
                    rows.append(dict(**common, band=band, side="left", energy=left,
                                     null=null_reference(svd, band, "left")))
                    rows.append(dict(**common, band=band, side="right", energy=right,
                                     null=null_reference(svd, band, "right")))
                rows.append(dict(**common, band="_outside", side="left", energy=e["_outside_left"], null=0.0))
                rows.append(dict(**common, band="_outside", side="right", energy=e["_outside_right"], null=0.0))
                if keep_spectrum:
                    spectra.append(dict(model=model_id, dataset=task, layer=layer, module_type=mtype,
                                        sigma=spectrum_summary(Gd, keep_spectrum)))
            if obs is not None:
                obs.close()
            if xd is not None:
                xd.close()
            del G, Ga, Gb, xd
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
    df = pd.DataFrame(rows)
    df.attrs["spectra"] = spectra
    return df


def plot(df: pd.DataFrame, out_dir: Path, side: str = "left") -> None:
    """One panel per module type present in ``df`` (works for Llama-like and NeoX-like names)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    dec = decile_bands()
    d = df[(df.side == side) & (df.band.isin(dec))]
    mtypes = [m for m in TARGET_SUFFIXES if m in set(df.module_type)] or sorted(set(df.module_type))
    fig, axes = plt.subplots(1, len(mtypes), figsize=(3.2 * len(mtypes), 3.2), sharey=True, squeeze=False)
    axes = axes[0]
    for ax, mt in zip(axes, mtypes):
        sub = d[d.module_type == mt]
        for task, grp in sub.groupby("dataset"):
            mean = grp.groupby("band")["energy"].mean().reindex(dec)
            ax.plot(range(10), mean.values, marker="o", label=task)
        null = sub.groupby("band")["null"].mean().reindex(dec)  # k/out or k/in: exact for this module type
        ax.plot(range(10), null.values, ls="--", c="gray", lw=1, label="null (isotropic dW)")
        ax.set_title(mt)
        ax.set_xticks(range(10))
        ax.set_xticklabels(["top"] + [""] * 8 + ["bottom"])
        ax.set_yscale("log")
    axes[0].set_ylabel(f"gradient energy fraction ({side})")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / f"grad_bands_{side}.png", dpi=150)
    plt.close(fig)

    # ratio bottom:16 / top:16 per layer
    r = df[(df.side == side) & (df.band.isin(["top:16", "bottom:16"]))]
    piv = r.pivot_table(index=["dataset", "layer", "module_type"], columns="band", values="energy").reset_index()
    piv["ratio"] = piv["bottom:16"] / piv["top:16"].clip(lower=1e-12)
    fig, axes = plt.subplots(1, len(mtypes), figsize=(3.2 * len(mtypes), 3.2), sharey=True, squeeze=False)
    axes = axes[0]
    for ax, mt in zip(axes, mtypes):
        for task, grp in piv[piv.module_type == mt].groupby("dataset"):
            ax.plot(grp.layer, grp.ratio, marker=".", label=task)
        ax.axhline(1.0, ls="--", c="gray", lw=1)
        ax.set_yscale("log")
        ax.set_title(mt)
        ax.set_xlabel("layer")
    axes[0].set_ylabel("energy bottom:16 / top:16")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / f"grad_ratio_{side}.png", dpi=150)
    plt.close(fig)

    # ---- the metric to read: fraction of energy captured at rank r ------------------------------
    one = df[(df.side == "left") & (df.band == "top:16")]
    rank_cols = sorted((c for c in df.columns if c.startswith("energy_at_r")),
                       key=lambda c: int(c.split("energy_at_r")[1]))
    if rank_cols:
        ranks = [int(c.split("energy_at_r")[1]) for c in rank_cols]
        fig, axes = plt.subplots(1, len(mtypes), figsize=(3.2 * len(mtypes), 3.4), sharey=True, squeeze=False)
        axes = axes[0]
        for ax, mt in zip(axes, mtypes):
            for task, grp in one[one.module_type == mt].groupby("dataset"):
                ax.plot(ranks, [grp[c].mean() for c in rank_cols], marker="o", label=task)
            ax.set_xscale("log", base=2)
            ax.set_ylim(0, 1.02)
            ax.set_title(mt)
            ax.set_xlabel("rank r")
        axes[0].set_ylabel("fraction of gradient energy\ncaptured at rank r")
        axes[0].legend(fontsize=7)
        fig.suptitle("What a rank-r adapter could represent of the gradient, at best", fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        fig.savefig(out_dir / "grad_energy_at_rank.png", dpi=200, bbox_inches="tight")
        fig.savefig(out_dir / "grad_energy_at_rank.pdf", bbox_inches="tight")
        plt.close(fig)

    # ---- systematic vs idiosyncratic: does a disjoint half give the same direction? --------------
    if "split_cosine" in one and one.split_cosine.notna().any():
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
        for ax, col, lab in ((axes[0], "split_cosine", "cosine between two disjoint halves"),
                             (axes[1], "split_overlap", "top-16 subspace overlap")):
            for task, grp in one.groupby("dataset"):
                g = grp.groupby("layer")[col].mean()
                ax.plot(g.index, g.values, marker=".", label=task)
            if col == "split_overlap":
                ax.axhline(one.split_overlap_chance.mean(), ls="--", c="gray", lw=1, label="chance")
            ax.axhline(0.0, ls=":", c="gray", lw=0.8)
            ax.set_xlabel("layer")
            ax.set_ylabel(lab, fontsize=9)
            ax.legend(fontsize=7)
        fig.suptitle("A systematic gradient reproduces across data; an idiosyncratic one does not", fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        fig.savefig(out_dir / "grad_split_half.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

    # ---- exp(H(p)) kept as a diagnostic, but NORMALISED by its noise ceiling ---------------------
    if "grad_eff_rank_frac" in one:
        fig, axes = plt.subplots(1, len(mtypes), figsize=(3.2 * len(mtypes), 3.2), sharey=True, squeeze=False)
        axes = axes[0]
        for ax, mt in zip(axes, mtypes):
            for task, grp in one[one.module_type == mt].groupby("dataset"):
                g = grp.groupby("layer")["grad_eff_rank_frac"].mean()
                ax.plot(g.index, g.values, marker=".", label=task)
            ax.axhline(1.0, ls="--", c="gray", lw=1)
            ax.set_ylim(0, 1.1)
            ax.set_title(mt)
            ax.set_xlabel("layer")
        axes[0].set_ylabel("exp(H(p)) / noise ceiling")
        axes[0].legend(fontsize=7)
        fig.suptitle("Effective rank relative to what uncorrelated contributions would give "
                     "(1.0 = indistinguishable from independent terms)", fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        fig.savefig(out_dir / "grad_eff_rank_normalised.png", dpi=200, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--svd_cache", required=True)
    ap.add_argument("--out", default="results/grad_probe")
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--group_layers", type=int, default=4)
    ap.add_argument("--ranks", default="1,2,4,8,16,32,64", help="ranks for the energy-at-rank curve")
    ap.add_argument("--no_split_half", action="store_true", help="skip the two-disjoint-halves control")
    ap.add_argument("--match_sup_tokens", type=int, default=None,
                    help="stop each dataset at this many supervised tokens, so the tasks are compared "
                         "at equal supervision (the entropy of the spectrum grows with the token count)")
    ap.add_argument("--sigma_rel_tol", type=float, default=1e-6,
                    help="bottom-r bands are taken inside the numerically non-zero part of the "
                         "spectrum: on q_proj and o_proj sigma_min/sigma_max is about 3e-9 and the "
                         "singular vectors there are ill-conditioned")
    ap.add_argument("--arm_ranks", default="1,2,4",
                    help="ranks for which to report the gradient SNR that a top-r / bottom-r arm would "
                         "receive on its trainable factor")
    ap.add_argument("--xd", action="store_true",
                    help="also measure the stable ranks of the inputs X and of the backpropagated "
                         "signals D: rank(G) <= min(rank X, rank D), so this says which one limits "
                         "the gradient (costs hooks and ~4 MB per module)")
    ap.add_argument("--proj_dim", type=int, default=1024,
                    help="random projection used by --xd; stable ranks near this value mean "
                         "'at least that much', not a measurement")
    ap.add_argument("--keep_spectrum", type=int, default=128,
                    help="store the first k singular values per module, so other statistics can be "
                         "recomputed without re-running the probe")
    ap.add_argument("--datasets", default="facts,format",
                    help="comma-separated; add 'format_unmasked' to control for the loss masking "
                         "(format data supervised on every token, like facts)")
    ap.add_argument("--dtype", default="bfloat16")
    a = ap.parse_args()
    tok = load_tokenizer(a.model)
    model = load_model(a.model, a.dtype)
    if hasattr(model, "gradient_checkpointing_enable"):
        from .modeling import enable_checkpointing

        enable_checkpointing(model)
    df = probe(model, tok, a.model, a.data_dir, a.svd_cache, datasets=tuple(a.datasets.split(",")),
               n_examples=a.n, batch_size=a.batch, max_len=a.max_len, group_layers=a.group_layers,
               ranks=tuple(int(r) for r in a.ranks.split(",")), split_half=not a.no_split_half,
               keep_spectrum=a.keep_spectrum, match_sup_tokens=a.match_sup_tokens,
               xd_decomposition=a.xd, proj_dim=a.proj_dim,
               arm_ranks=tuple(int(x) for x in a.arm_ranks.split(",")),
               sigma_rel_tol=a.sigma_rel_tol)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    slug = a.model.replace("/", "__")
    df.to_csv(out / f"grad_probe_{slug}.csv", index=False)
    if df.attrs.get("spectra"):
        import json as _json

        with open(out / f"spectra_{slug}.json", "w") as f:
            _json.dump(df.attrs["spectra"], f)
    plot(df, out, "left")
    plot(df, out, "right")
    dec = df[df.band.str.startswith("frac") | (df.band == "_outside")]
    summary = dec.groupby(["dataset", "side", "band"])["energy"].mean().unstack("dataset")
    print("mean energy by band (deciles + outside span), by side:")
    print(summary)
    one = df[(df.side == "left") & (df.band == "top:16")]
    rank_cols = sorted((c for c in df.columns if c.startswith("energy_at_r")),
                       key=lambda c: int(c.split("energy_at_r")[1]))
    print("\nFRACTION OF GRADIENT ENERGY CAPTURED AT RANK r  <- the metric to read")
    print(one.groupby("dataset")[rank_cols].mean().round(3))
    if "stable_rank_X" in df:
        print("\nX / D decomposition: rank(G) <= min(rank X, rank D) — which one limits the gradient?")
        print(one.groupby(["dataset", "module_type"])[["stable_rank_X", "stable_rank_D", "stable_rank"]]
              .mean().round(1))
    osh = sorted(c for c in df.columns if c.startswith("output_share_"))
    if osh:
        print("\nSHARE OF THE LAYER'S OUTPUT each band actually carries, on real data")
        print("(what the loss sees; compare each band to its null reference of the same rank)")
        print(one.groupby("dataset")[osh].median().round(4).to_string())
    snr_cols = sorted(c for c in df.columns if c.startswith("grad_snr_in_band_"))
    if snr_cols:
        num = sorted(c for c in df.columns if c.startswith("grad_norm_in_band_"))
        den = sorted(c for c in df.columns if c.startswith("grad_noise_in_band_"))
        print("\nGRADIENT SNR PER BAND — what AdamW responds to (it normalises the magnitude away)")
        print("Medians across modules; numerator and denominator shown separately so that an SNR "
              "effect is not confounded with an energy effect.")
        print(one.groupby("dataset")[snr_cols].median().round(2).to_string())
        print("\n  numerator (projected mean gradient):")
        print(one.groupby("dataset")[num].median().round(4).to_string())
        print("\n  denominator (disagreement between two disjoint halves):")
        print(one.groupby("dataset")[den].median().round(4).to_string())
        if "snr_n_examples" in one:
            print(f"\n  averaged over N = {one.snr_n_examples.iloc[0]:.0f} examples for every arm "
                  f"(the SNR of a mean falls as 1/sqrt(N), so N must match across arms)")
    print("\nsplit-half coherence (systematic vs idiosyncratic): cosine and top-16 overlap")
    print(one.groupby("dataset")[["split_cosine", "split_overlap", "split_overlap_chance"]].mean().round(3))
    print("\ndiagnostics: exp(H(p)) raw, its noise ceiling, their ratio, stable rank, supervision")
    print(one.groupby("dataset").agg(eff_rank=("grad_eff_rank", "mean"),
                                     ceiling=("grad_eff_rank_ceiling", "mean"),
                                     ratio=("grad_eff_rank_frac", "mean"),
                                     stable_rank=("stable_rank", "mean"),
                                     sup_tok_per_ex=("sup_tokens_per_example", "first"),
                                     sup_tok_total=("sup_tokens_total", "first"),
                                     batches=("n_batches", "first")).round(3))
    print("\nNOTE: exp(H(p)) is the entropy of the singular VALUES, not of the energies. It is not a "
          "count of directions and it grows with the number of contributions; read energy_at_r*.")
    print("\nsum over deciles + outside (must be ~1 per side):")
    print(dec.groupby(["dataset", "side"])["energy"].sum() / dec.groupby(["dataset", "side"])["layer"].nunique()
          / dec.groupby(["dataset", "side"])["module_type"].nunique())


if __name__ == "__main__":
    main()
