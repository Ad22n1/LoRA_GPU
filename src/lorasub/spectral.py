"""Spectral toolbox: SVD cache of pre-trained weights and measures on low-rank updates.

Conventions
-----------
* A weight matrix follows the nn.Linear convention: W has shape (out, in).
* SVD is computed with ``full_matrices=False``: U (out, m), S (m,), Vh (m, in) with
  m = min(out, in) and S sorted in *decreasing* order.  Index 0 is the dominant
  direction; index m-1 is the smallest *non-zero* singular value.  For rectangular
  modules "bottom-r" therefore means "the r smallest non-zero singular values",
  consistent with MiCA / MiLoRA.  We never touch the null space.
* A low-rank update is always handled as a factor pair (A, B) with A (r, in) and
  B (out, r), so that dW = B @ A.  We never materialise dW unless unavoidable.
* Full SVD only.  Randomised SVD is accurate at the top of the spectrum and wrong at
  the bottom, which is exactly the part we study.
"""
from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import torch
from safetensors.torch import load_file, save_file
from torch import nn

TARGET_SUFFIXES: tuple[str, ...] = (
    "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
)

_NAME_RE = re.compile(r"layers\.(\d+)\..*?\.([A-Za-z0-9_]+)$")


# --------------------------------------------------------------------------- #
# Module discovery
# --------------------------------------------------------------------------- #
def iter_target_modules(
    model: nn.Module, suffixes: tuple[str, ...] | list[str] = TARGET_SUFFIXES
) -> Iterator[tuple[str, nn.Linear]]:
    """Yield ``(full_name, module)`` for every nn.Linear selected by ``suffixes``.

    Two forms are accepted, and the distinction matters for the per-module oracle:

    * a bare suffix (``"q_proj"``) selects that projection in EVERY layer — the normal case;
    * a dotted path (``"layers.15.self_attn.q_proj"``) selects ONE module in ONE layer.

    Without the second form, a config naming full paths would match nothing at all: the old test
    was ``name.split(".")[-1] in suffixes``, so ``"q_proj" in ["layers.15.self_attn.q_proj"]`` is
    False. The run would train zero adapters, report the base model's score, and raise nothing.
    A per-layer sensitivity grid is exactly the design where that happens silently.
    """
    suffixes = tuple(suffixes)
    bare = {t for t in suffixes if "." not in t}
    dotted = {t for t in suffixes if "." in t}
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if name.split(".")[-1] in bare or any(name.endswith(t) for t in dotted):
            yield name, module


def parse_module_name(name: str) -> tuple[int, str]:
    """``model.layers.3.mlp.up_proj`` -> ``(3, "up_proj")``; also ``gpt_neox.layers.3.attention.dense`` ->
    ``(3, "dense")``.  Raises if the name does not contain ``layers.<i>``."""
    m = _NAME_RE.search(name)
    if m is None:
        raise ValueError(f"cannot parse layer/module type from {name!r}")
    return int(m.group(1)), m.group(2)


def model_slug(model_id: str) -> str:
    """``meta-llama/Llama-3.2-1B`` -> ``meta-llama__Llama-3.2-1B`` (safe directory name)."""
    return model_id.replace("/", "__")


# --------------------------------------------------------------------------- #
# SVD cache
# --------------------------------------------------------------------------- #
@dataclass
class SVDEntry:
    name: str
    shape: tuple[int, int]  # (out, in)
    S: torch.Tensor  # (m,) fp32, decreasing
    U: torch.Tensor  # (out, m) fp32
    Vh: torch.Tensor  # (m, in) fp32

    @property
    def m(self) -> int:
        return int(self.S.numel())

    @property
    def V(self) -> torch.Tensor:
        return self.Vh.T

    def to(self, device) -> "SVDEntry":
        return SVDEntry(self.name, self.shape, self.S.to(device), self.U.to(device), self.Vh.to(device))


def svd_cache_dir(cache_dir: str | Path, model_id: str) -> Path:
    return Path(cache_dir) / model_slug(model_id)


def _entry_path(cache_dir: str | Path, model_id: str, name: str) -> Path:
    return svd_cache_dir(cache_dir, model_id) / f"{name}.safetensors"


@torch.no_grad()
def svd_of_weight(W: torch.Tensor, device=None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Full (thin) SVD of W in fp32.  Returns (U, S, Vh) on CPU, fp32, S decreasing."""
    dev = device or (W.device if W.is_cuda else "cpu")
    Wf = W.detach().to(dev, torch.float32)
    U, S, Vh = torch.linalg.svd(Wf, full_matrices=False)
    # Reconstruction check: catches driver/precision failures silently producing garbage.
    rec = (U * S) @ Vh
    # 1e-2 of the largest weight (was 1e-3, 26/09): the check guards against a FAILED decomposition, whose errors are of the order of
    # the weights; fp32 round-off grows with the matrix size, and Qwen2.5-7B's 3584 x 18944 matrices reach 1.13e-3 of max|W|. U and
    # Vh are then stored in fp16, which loses about 1e-3 relative precision anyway (PROTOCOL_scale_addendum.md).
    tol = 1e-2 * float(Wf.abs().max()) + 1e-6
    err = float((rec - Wf).abs().max())
    if err > tol:
        raise RuntimeError(f"SVD reconstruction error {err:.3e} > tol {tol:.3e}")
    return U.cpu(), S.cpu(), Vh.cpu()


@torch.no_grad()
def compute_svd_cache(
    model: nn.Module,
    cache_dir: str | Path,
    model_id: str,
    suffixes=TARGET_SUFFIXES,
    dtype_store: torch.dtype = torch.float16,
    device=None,
    overwrite: bool = False,
    verbose: bool = True,
) -> list[str]:
    """Compute and store the SVD of every target module.  Returns the list of module names.

    Storage: one safetensors file per module with S in fp32 and U, Vh in ``dtype_store``.
    fp16 storage of U/Vh loses ~1e-3 relative precision on the *vectors*, which is
    harmless for band projections; S is kept in fp32 because the bottom of the spectrum
    spans several orders of magnitude.
    """
    out = svd_cache_dir(cache_dir, model_id)
    out.mkdir(parents=True, exist_ok=True)
    names: list[str] = []
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    for name, module in iter_target_modules(model, suffixes):
        names.append(name)
        path = out / f"{name}.safetensors"
        if path.exists() and not overwrite:
            continue
        U, S, Vh = svd_of_weight(module.weight, device=dev)
        save_file(
            {"S": S.contiguous(), "U": U.to(dtype_store).contiguous(), "Vh": Vh.to(dtype_store).contiguous()},
            str(path),
            metadata={"name": name, "out": str(module.out_features), "in": str(module.in_features)},
        )
        if verbose:
            print(f"[svd] {name}: shape=({module.out_features},{module.in_features}) "
                  f"S[0]={S[0]:.3e} S[-1]={S[-1]:.3e}")
    return names


def load_svd(cache_dir: str | Path, model_id: str, name: str, device="cpu") -> SVDEntry:
    path = _entry_path(cache_dir, model_id, name)
    if not path.exists():
        raise FileNotFoundError(f"no SVD cache for {name} at {path}; run compute_svd_cache first")
    t = load_file(str(path), device=str(device))
    U, S, Vh = t["U"].float(), t["S"].float(), t["Vh"].float()
    return SVDEntry(name=name, shape=(U.shape[0], Vh.shape[1]), S=S, U=U, Vh=Vh)


def list_cached(cache_dir: str | Path, model_id: str) -> list[str]:
    d = svd_cache_dir(cache_dir, model_id)
    return sorted(p.stem for p in d.glob("*.safetensors")) if d.exists() else []


# --------------------------------------------------------------------------- #
# Bands
# --------------------------------------------------------------------------- #
def numerical_rank(S: torch.Tensor, rel_tol: float = 1e-6) -> int:
    """Number of singular values above ``rel_tol * sigma_max``.

    Why this matters for ``bottom-r``. The reduced SVD returns m non-zero singular values on paper,
    but some may be numerically zero on a real model; the vectors attached to them are ill-conditioned
    — by Davis-Kahan the subspace they span is unstable as the spectral gap vanishes — so adapting
    them means adapting an arbitrary direction rather than "the least used one".

    MEASURED ON Llama-3.2-1B, 13/09, and the answer is that it does NOT happen here. At rel_tol=1e-6
    the numerical rank equals m on every module of every layer except ``q_proj``, where THREE columns
    in total are dropped across sixteen layers, never more than one per layer. At r=2 the bottom band
    is therefore untouched. The control grid written for this (`numrank_control_1b.yaml`) was not run:
    it would have measured nothing.

    An earlier version of this docstring claimed sigma_min = 3.5e-8 against sigma_max = 10.7, a ratio
    of 3e-9. **That figure is wrong and was never sourced.** The worst layer of ``q_proj`` has
    sigma_min = 5.83e-6 against sigma_max = 31.36, a ratio of 1.86e-7 — sixty times larger.

    Note the storage split, because it is easy to get wrong in both directions: **S in float32, U and
    Vh in float16** (see compute_svd_cache). The numerical rank above is therefore a reliable
    measurement — it is computed on S. What fp16 costs is ~5e-4 of relative precision on the frozen
    directions themselves, since B = U[:, idx]: a 0.05% perturbation of a correct direction, not a
    direction replaced by noise.

    The quantity that actually decides ill-conditioning is the margin over the fp32 noise floor,
    sigma_min / (eps * sigma_max) with eps = 1.19e-7. Minimum over the sixteen layers:

        q_proj 1.6   o_proj 36.5   k_proj 4.7e4   gate 1.1e5   down 1.5e5   up 1.9e5   v_proj 5.0e5

    Only ``q_proj`` layer 0 sits near the floor. ``o_proj`` is comfortably above it everywhere, which
    means the exploding relative perturbation observed on that module is NOT a numerical artefact —
    see the note on --cond_tol in scripts/relative_perturbation.py.
    """
    if S.numel() == 0 or float(S[0]) <= 0:
        return 0
    return int((S > rel_tol * S[0]).sum())


def band_indices(m: int, spec: str, device=None, S: torch.Tensor | None = None,
                 rel_tol: float | None = None) -> torch.Tensor:
    """Indices (into the decreasing singular-value order) selected by a band spec.

    * ``"top:k"``        -> [0, k)
    * ``"bottom:k"``     -> [m-k, m), or [rank-k, rank) when ``S`` and ``rel_tol`` are given, where
                          ``rank`` is the numerical rank (see ``numerical_rank``)
    * ``"random:k:seed"``-> k indices without replacement, deterministic in ``seed``
    * ``"frac:a-b"``     -> [floor(a*m), floor(b*m)),  0 <= a < b <= 1  (a=0 is the top)
    * ``"at:k:f"``       -> k consecutive indices starting at floor(f*(m-k)), 0 <= f <= 1
                            (f=0 == top:k, f=1 == bottom:k; intermediate f sweeps the spectrum)
    """
    parts = spec.split(":")
    kind = parts[0]
    if kind == "top":
        k = int(parts[1])
        _check_k(k, m)
        idx = torch.arange(0, k)
    elif kind == "bottom":
        k = int(parts[1])
        _check_k(k, m)
        # with S and rel_tol, count from the last NUMERICALLY non-zero direction, not from m
        end = numerical_rank(S, rel_tol) if (S is not None and rel_tol is not None) else m
        if end < k:
            raise ValueError(f"only {end} numerically non-zero directions (rel_tol={rel_tol}), need {k}")
        idx = torch.arange(end - k, end)
    elif kind == "random":
        # spec "random:k:seed" or "random:k:seed:salt". The salt makes the draw INDEPENDENT PER
        # MODULE. Without it every module of the same size m received the same columns: on
        # Llama-3.2-1B that is two distinct draws per seed (m=2048 and m=512) instead of 112, so the
        # arm was not "a random subspace" but ONE random spectral position applied everywhere —
        # i.e. a band arm at a random band_frac. The control that anchors the whole comparison did
        # not have the statistical property we claimed for it.
        k, seed = int(parts[1]), int(parts[2])
        salt = parts[3] if len(parts) > 3 else ""
        _check_k(k, m)
        mixed = (seed * 1_000_003 + int(hashlib.blake2b(salt.encode(), digest_size=8).hexdigest(), 16)
                 ) % (2 ** 63 - 1) if salt else seed
        g = torch.Generator().manual_seed(mixed)
        idx = torch.randperm(m, generator=g)[:k].sort().values
    elif kind == "at":
        k, f = int(parts[1]), float(parts[2])
        _check_k(k, m)
        if not (0.0 <= f <= 1.0):
            raise ValueError(f"bad position {f} in {spec!r}")
        # honour the numerical rank exactly as bottom:k does — otherwise band_frac=1.0 lands on
        # ill-conditioned directions while mode="bottom" with the same tolerance does not, and the
        # end of the continuous curve is no longer the `bottom` arm of the table
        end = numerical_rank(S, rel_tol) if (S is not None and rel_tol is not None) else m
        if end < k:
            raise ValueError(f"only {end} numerically non-zero directions (rel_tol={rel_tol}), need {k}")
        start = int(math.floor(f * (end - k)))
        idx = torch.arange(start, start + k)
    elif kind == "frac":
        a, b = (float(x) for x in parts[1].split("-"))
        if not (0.0 <= a < b <= 1.0):
            raise ValueError(f"bad frac band {spec!r}")
        lo, hi = int(math.floor(a * m)), int(math.floor(b * m))
        if b == 1.0:
            hi = m
        idx = torch.arange(lo, hi)
    else:
        raise ValueError(f"unknown band spec {spec!r}")
    return idx.to(device) if device is not None else idx


def _check_k(k: int, m: int) -> None:
    if not (1 <= k <= m):
        raise ValueError(f"band size k={k} must be in [1, m={m}]")


def decile_bands() -> list[str]:
    """The ten bands ``frac:0.0-0.1`` ... ``frac:0.9-1.0`` used in all dynamics figures."""
    return [f"frac:{i/10:.1f}-{(i+1)/10:.1f}" for i in range(10)]


# --------------------------------------------------------------------------- #
# Measures on a low-rank update dW = B @ A
# --------------------------------------------------------------------------- #
@torch.no_grad()
def delta_frobenius_sq(A: torch.Tensor, B: torch.Tensor) -> float:
    """||B A||_F^2 = trace((B^T B)(A A^T)) without forming B A."""
    A = A.float()
    B = B.float()
    return float(torch.trace((B.T @ B) @ (A @ A.T)))


@torch.no_grad()
def band_energies(
    A: torch.Tensor, B: torch.Tensor, svd: SVDEntry, bands: list[str]
) -> dict[str, tuple[float, float]]:
    """Fraction of ||dW||_F^2 living in each spectral band of W0, on both sides.

    For a band b with index set I_b:
        left  = ||U[:, I_b]^T dW||_F^2 / ||dW||_F^2   (output side: rows of dW in span(U_b)?)
        right = ||dW V[:, I_b]||_F^2  / ||dW||_F^2    (input side: columns of dW in span(V_b)?)

    Implementation without forming dW: ||u_i^T dW||^2 = [(U^T B)(A A^T)(U^T B)^T]_ii and
    ||dW v_j||^2 = [(A V)^T (B^T B)(A V)]_jj.  Left energies use the full dW (not its
    projection on span(V)), so for rectangular W0 the two sides are computed independently.

    Two extra keys are always returned:
        ``_outside_left``  = 1 - ||U^T dW||^2/||dW||^2   (energy outside span(U); 0 for square full-rank W0)
        ``_outside_right`` = 1 - ||dW V||^2/||dW||^2
    A random rank-r dW puts fraction |I_b|/m of its energy in band b on each side (null reference).
    """
    dev = A.device
    A = A.float()
    B = B.float()
    U = svd.U.to(dev)
    V = svd.V.to(dev)
    total = delta_frobenius_sq(A, B)
    if total <= 0.0:
        out = {b: (0.0, 0.0) for b in bands}
        out["_outside_left"] = 0.0
        out["_outside_right"] = 0.0
        return out
    UtB = U.T @ B  # (m, r)
    AV = A @ V  # (r, m)
    AAt = A @ A.T  # (r, r)
    BtB = B.T @ B  # (r, r)
    # energy of U-direction i in dW: ||u_i^T dW||^2 = [ (U^T B)(A A^T)(U^T B)^T ]_ii   (full dW, not projected on V)
    row_sq = ((UtB @ AAt) * UtB).sum(dim=1)
    # energy of V-direction j in dW: ||dW v_j||^2 = [ (A V)^T (B^T B)(A V) ]_jj
    col_sq = ((AV.T @ BtB) * AV.T).sum(dim=1)
    left_total = float(row_sq.sum())
    right_total = float(col_sq.sum())
    out: dict[str, tuple[float, float]] = {}
    for spec in bands:
        idx = band_indices(svd.m, spec, device=dev)
        out[spec] = (float(row_sq[idx].sum()) / total, float(col_sq[idx].sum()) / total)
    out["_outside_left"] = max(0.0, 1.0 - left_total / total)
    out["_outside_right"] = max(0.0, 1.0 - right_total / total)
    return out


@torch.no_grad()
def gradient_band_energies(G: torch.Tensor, svd: SVDEntry, bands: list[str]) -> dict[str, tuple[float, float]]:
    """Same as ``band_energies`` for a *full* matrix G (out, in), e.g. a mean gradient."""
    dev = G.device
    G = G.float()
    U = svd.U.to(dev)
    V = svd.V.to(dev)
    total = float(G.pow(2).sum())
    if total <= 0.0:
        out = {b: (0.0, 0.0) for b in bands}
        out["_outside_left"] = 0.0
        out["_outside_right"] = 0.0
        return out
    UtG = U.T @ G  # (m, in)
    GV = G @ V  # (out, m)
    row_sq = UtG.pow(2).sum(dim=1)  # ||u_i^T G||^2 over the full G
    col_sq = GV.pow(2).sum(dim=0)  # ||G v_j||^2
    out: dict[str, tuple[float, float]] = {}
    for spec in bands:
        idx = band_indices(svd.m, spec, device=dev)
        out[spec] = (float(row_sq[idx].sum()) / total, float(col_sq[idx].sum()) / total)
    out["_outside_left"] = max(0.0, 1.0 - float(UtG.pow(2).sum()) / total)
    out["_outside_right"] = max(0.0, 1.0 - float(GV.pow(2).sum()) / total)
    return out


@torch.no_grad()
def principal_angles(Q1: torch.Tensor, Q2: torch.Tensor) -> torch.Tensor:
    """Principal angles (radians, increasing) between span(Q1) and span(Q2).

    Q1 (d, p) and Q2 (d, q) must have orthonormal columns.  Returns min(p, q) angles.
    """
    # float64: arccos is ill-conditioned near 1 (arccos(1 - 1e-7) ~ 4e-4 in fp32)
    s = torch.linalg.svdvals(Q1.double().T @ Q2.double()).clamp(-1.0, 1.0)
    return torch.arccos(s).flip(0).float()  # svdvals are decreasing -> angles increasing


@torch.no_grad()
def orthonormal_basis(M: torch.Tensor) -> torch.Tensor:
    """Orthonormal basis of the column space of M via QR (columns)."""
    Q, _ = torch.linalg.qr(M.float())
    return Q


@torch.no_grad()
def effective_rank(A: torch.Tensor, B: torch.Tensor, eps: float = 1e-12) -> float:
    """Roy & Vetterli effective rank exp(H(p)), p_i = sigma_i / sum(sigma), of dW = B A.

    Computed from the small r x r problem: sigma(BA) = sigma(R_B R_A^T) where B = Q_B R_B, A^T = Q_A R_A.
    """
    A = A.float()
    B = B.float()
    _, RB = torch.linalg.qr(B)  # (r, r)
    _, RA = torch.linalg.qr(A.T)  # (r, r)
    s = torch.linalg.svdvals(RB @ RA.T)
    s = s[s > eps * s.max()] if s.numel() and float(s.max()) > 0 else s
    if s.numel() == 0:
        return 0.0
    p = s / s.sum()
    return float(torch.exp(-(p * torch.log(p)).sum()))


@torch.no_grad()
def intruder_count(
    W0: torch.Tensor, A: torch.Tensor, B: torch.Tensor, k: int = 10, eps: float = 0.5, device=None,
    scaling: float = 1.0,
) -> int:
    """Number of *intruder dimensions* (Shuttleworth et al., 2025) among the top-k left
    singular vectors of W0 + scaling * B A: those whose maximum |cosine| with the top-k left
    singular vectors of W0 is below ``eps``.  Unlike band energies and effective rank, this
    quantity is NOT scale-invariant: pass the LoRA scaling alpha/r actually applied.
    Requires an SVD of W0 + dW (fp32); call sparingly.
    """
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    W0f = W0.detach().to(dev, torch.float32)
    W1 = W0f + float(scaling) * (B.to(dev, torch.float32) @ A.to(dev, torch.float32))
    U0 = torch.linalg.svd(W0f, full_matrices=False)[0][:, :k]
    U1 = torch.linalg.svd(W1, full_matrices=False)[0][:, :k]
    cos = (U1.T @ U0).abs().max(dim=1).values  # for each new vector, best match among old
    return int((cos < eps).sum())


@torch.no_grad()
def relative_perturbation(A: torch.Tensor, B: torch.Tensor, svd: "SVDEntry", band: str,
                          scaling: float = 1.0) -> dict[str, float]:
    """How large is the update COMPARED TO WHAT ALREADY LIVES in the band it writes into?

    The absolute Frobenius norm is not comparable across bands: ||dW|| = 0.71 written on the smallest
    singular directions is a huge relative change, the same norm on the largest is negligible. So an
    anticorrelation between performance and absolute norm does not rule out an effect of relative
    perturbation — the two must be measured separately. Computable from the saved factors, no re-run.

    Returns the update norm, the energy of W0 inside the band, and their ratio.
    """
    idx = band_indices(svd.m, band, S=None, rel_tol=None)
    sig = svd.S[idx].double()
    w_band = float((sig ** 2).sum().sqrt())            # ||W0 restricted to the band||_F
    dw = float(torch.trace((B.double().T @ B.double()) @ (A.double() @ A.double().T)).clamp_min(0).sqrt())
    dw *= scaling
    w_norm = float((svd.S.double() ** 2).sum().sqrt())     # ||W0||_F of the whole module
    # conditioning of the adapted band: its smallest singular value over the module's largest. Below
    # ~1e-4 the singular vectors are ill-determined and any ratio built on them is numerical noise.
    smax = float(svd.S[0]) if svd.S.numel() else 0.0
    cond = (float(sig.min()) / smax) if (smax > 0 and sig.numel()) else 0.0
    return {"update_norm": dw, "band_energy": w_band, "w_norm": w_norm, "cond": cond,
            "band_share": w_band / w_norm if w_norm > 0 else 0.0,
            "relative_perturbation": dw / w_band if w_band > 0 else float("inf")}


def null_reference(svd_or_m, spec: str, side: str = "left") -> float:
    """Expected band energy fraction of an *isotropic* random update dW (out, in).

    A band of k orthonormal output directions captures k/out of the energy of an isotropic
    matrix (left side); k input directions capture k/in (right side).  For square modules both
    equal k/m; for rectangular ones they differ, so the side must be given.  ``svd_or_m`` is an
    SVDEntry (preferred) or, for square matrices only, the integer m."""
    if isinstance(svd_or_m, SVDEntry):
        out_f, in_f = svd_or_m.shape
        m = svd_or_m.m
    else:
        out_f = in_f = m = int(svd_or_m)
    k = band_indices(m, spec).numel()
    return float(k) / (out_f if side == "left" else in_f)
