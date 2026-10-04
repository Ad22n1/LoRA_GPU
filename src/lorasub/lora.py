"""LoRA with a controlled adaptation subspace.

Four arms, one structure.  For every target nn.Linear (W0: out x in) we add
``dW = B @ A`` with A (r, in) and B (out, r):

* ``free``   : A ~ kaiming_uniform(a=sqrt(5)) (LoRA/peft default), B = 0, both trained.
               Reference arm, 2*r*(in+out)/... parameters, learned subspace.
* ``top``    : B = U[:, :r]   frozen (dominant left singular vectors of W0), A = 0 trained.
* ``bottom`` : B = U[:, -r:]  frozen (minor left singular vectors of W0),   A = 0 trained.
* ``random`` : B = U[:, idx]  frozen, idx = r random columns of U,          A = 0 trained.
* ``band``   : B = U[:, s:s+r] frozen, s = floor(band_frac * (m - r)): r consecutive directions
               at a chosen position of the spectrum (band_frac 0 = top, 1 = bottom).  Sweeping
               band_frac over the deciles turns "top vs bottom" into a curve.
* ``per_module`` : a *constrained* band chosen per module type, e.g.
               {q_proj: top, k_proj: bottom, default: random}.  Every module stays constrained
               (B frozen, A trained, r*in parameters), so the arm has exactly the structure and
               budget of top/bottom/random; only the spectral position varies with the module.
               This is the controlled version of a "heterogeneous" allocation: mixing free and
               constrained modules would confound subspace with parameter budget.

Update magnitude: ``update_norms`` reports ||scaling * B A||_F per module and ``rescale_update_``
multiplies every A by one global factor (used for the magnitude-matched ablation of Zhang et al.
2025, "The primacy of magnitude in low-rank adaptation").

The three constrained arms have exactly r*in trainable parameters per module and start
from dW = 0, like the free arm.  B is orthonormal and carries *no* singular-value scaling
(as in MiCA; LoRA-XS multiplies by Sigma).  Forward: base(x) + (x A^T B^T) * (alpha / r).
A and B are kept in fp32 regardless of the base model dtype.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from torch import nn

from .spectral import SVDEntry, TARGET_SUFFIXES, band_indices, iter_target_modules, load_svd

# The DUAL arms freeze the INPUT factor A on rows of V instead of freezing B on columns of U.
# They exist because the gradient probe found the task contrast on the input side: paired over
# (layer, module), facts and format are indistinguishable in the leading decile of span(U)
# (40/80, p = 1) and differ by 61% in the leading decile of span(V) (56/64, p = 5.6e-10). Every
# arm below constrains the COLUMN space of dW (the E_left = 1 identity), so the task x position
# crossing may exist and have been measured on the axis the design does not control.
MODES: tuple[str, ...] = ("free", "top", "bottom", "random", "band", "random_ortho",
                          "dual_top", "dual_bottom", "dual_random", "dual_random_ortho",
                          "init_top", "init_bottom", "init_random", "top_sigma", "top_unfrozen", "top_scalar",
                          "bottom_sigma", "bottom_unfrozen", "top_lrscale", "learned")
DUAL: tuple[str, ...] = ("dual_top", "dual_bottom", "dual_random", "dual_random_ortho")

# INIT_ONLY reproduces what PiSSA and MiLoRA actually do, and doing it FAITHFULLY requires one
# step that is easy to miss: the band is SUBTRACTED from the frozen weight, and the adapter is
# initialised to reproduce it exactly. W_frozen = W0 - U_r S_r V_r^T, A = sqrt(S_r) V_r^T,
# B = U_r sqrt(S_r), so B @ A = the band and W_frozen + (alpha/r) B A = W0 at step 0 up to the
# scaling. The model is unchanged at step 0, as in every other arm here, and the adapter then
# RELEARNS that band freely.
#
# A version that merely initialises B on the band without touching W0 would perturb the model at
# step 0 by the full energy of the band, which on the top band is most of the layer. It would not
# be PiSSA, and claiming to refute PiSSA with it would be false.
#
# Why these arms are needed at all: every other constrained arm FREEZES B, which is the design of
# LoRA-XS and MiCA and of nobody else. PiSSA and MiLoRA train both factors. Without INIT_ONLY the
# paper can only speak about frozen subspaces, a strict subset of the literature it discusses.
#
# Budget is r*(d_in + d_out), the same as `free`: these arms compare to `free`, NOT to the
# constrained ones. They answer "does the starting POINT matter?"; the constrained arms answer
# "does the SUBSPACE matter?". Two questions, two budgets, never averaged together.
INIT_ONLY: tuple[str, ...] = ("init_top", "init_bottom", "init_random")

# Arms whose subspace is DRAWN, so `subspace_salted` changes what they measure and must enter the
# run id. Written once here because it was previously hard-coded in three places — config.id_dict,
# aggregate.collect and aggregate.select_population — and adding the dual family updated none of
# them. Two runs of `dual_random` differing only by the salt would have collided on one run_id, and
# every dual run would have had its salt flag rewritten to False on collection. A list repeated in
# four files drifts; a list imported from one does not.
SALTED_MODES: tuple[str, ...] = ("random", "random_ortho", "dual_random",
                                "dual_random_ortho", "init_random")
# DERIVED, not hand-listed. This tuple gates the loading of the SVD cache in `inject_lora`, and it
# was written by hand as ("top", "bottom", "random", "band"). Two families of modes were added this
# week — DUAL and INIT_ONLY — and neither reached it, so every init_only run died on
# `mode 'init_top' needs an SVDEntry of the base weight` after the grid had been launched twice.
# Tenth occurrence of the same pattern in this repository: an enumeration that governs behaviour and
# does not follow when a family is added.
#
# `random_ortho` and `dual_random_ortho` are deliberately absent: a Haar plane is orthonormal but
# unrelated to U, so it needs no SVD of the base weight. That is the whole point of those arms.
_NEEDS_SVD_BASE: tuple[str, ...] = ("top", "bottom", "random", "band", "top_sigma", "top_unfrozen",
                                   "top_scalar", "bottom_sigma", "bottom_unfrozen", "top_lrscale")
# arms that train BOTH factors, so their budget is r * (in + out) like `free`: the init_ family, and
# top_unfrozen, which keeps top's basis as the initial B and lets it move. expected_trainable() is
# what asserts the budget at every run, so a mode missing here would run at half the budget it claims.
BOTH_FACTORS: tuple[str, ...] = INIT_ONLY + ("top_unfrozen", "bottom_unfrozen")

CONSTRAINED: tuple[str, ...] = (
    _NEEDS_SVD_BASE
    + tuple(m for m in DUAL if not m.endswith("random_ortho"))
    + INIT_ONLY
)
PER_MODULE_KEYS: tuple[str, ...] = ("top", "bottom", "random")  # or "band:<frac>"


def parse_per_module(spec: dict, name: str) -> tuple[str, float | None]:
    """Resolve the (mode, band_frac) of one module from a per-module spec.

    Keys are module suffixes (q_proj ...) plus an optional ``default``; values are
    ``top`` | ``bottom`` | ``random`` | ``band:<frac>``.
    """
    suffix = name.split(".")[-1]
    val = spec.get(suffix, spec.get("default"))
    if val is None:
        raise ValueError(f"per_module spec has no entry for {suffix!r} and no 'default'")
    val = str(val)
    if val.startswith("band:"):
        return "band", float(val.split(":", 1)[1])
    if val not in PER_MODULE_KEYS:
        raise ValueError(f"per_module value {val!r} must be top|bottom|random|band:<frac>")
    return val, None


class LoRALinear(nn.Module):
    def __init__(
        self,
        base: nn.Linear,
        r: int,
        mode: str,
        alpha: float | None = None,
        svd: SVDEntry | None = None,
        seed: int = 0,
        band_frac: float | None = None,
        sigma_rel_tol: float | None = None,
        module_salt: str = "",
        subspace_salted: bool = False,
        learned_B: torch.Tensor | None = None,
    ):
        super().__init__()
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        if r < 1:
            raise ValueError("r must be >= 1")
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.r = int(r)
        self.mode = mode
        # per-module multiplier of this module's learning rate, read by lora_param_groups(); 1 for every arm but top_lrscale
        self.lr_mult = 1.0
        self.alpha = float(r if alpha is None else alpha)
        self.scaling = self.alpha / self.r
        self.merged = False
        out_f, in_f = base.out_features, base.in_features
        dev = base.weight.device

        if mode == "free":
            A = torch.empty(self.r, in_f, dtype=torch.float32, device=dev)
            nn.init.kaiming_uniform_(A, a=math.sqrt(5))
            B = torch.zeros(out_f, self.r, dtype=torch.float32, device=dev)
            self.A = nn.Parameter(A)
            self.B = nn.Parameter(B)
        elif mode == "random_ortho":
            # THE CONTROL THAT SEPARATES "B IS FROZEN" FROM "B COMES FROM THE SPECTRUM OF W0".
            # Every constrained arm — top, bottom, random — freezes B on columns of U, so a gap between
            # them and `free` mixes two causes: training one factor instead of two, and being confined
            # to span(U). This arm freezes B on r orthonormal directions drawn at random in R^out, with
            # no relation to W0, and trains A: same budget as the constrained arms, same "one factor"
            # handicap, but nothing spectral. If it matches `random`, span(U) does not matter and the
            # whole gap to `free` is the cost of freezing B; if `random` beats it, being in the spectrum
            # of W0 counts. The draw depends on the seed only (no SVD needed).
            # salted by module name for the same reason as the `random` arm: an unsalted generator
            # hands every module the same orthonormal directions
            _h = (int(hashlib.blake2b(module_salt.encode(), digest_size=8).hexdigest(), 16)
                  if subspace_salted else 0)
            g = torch.Generator(device="cpu").manual_seed((int(seed) * 7919 + 17 + _h) % (2 ** 63 - 1))
            B, _ = torch.linalg.qr(torch.randn(out_f, self.r, generator=g))
            self.A = nn.Parameter(torch.zeros(self.r, in_f, dtype=torch.float32, device=dev))
            self.B = nn.Parameter(B.to(dev, torch.float32).contiguous(), requires_grad=False)
            self.register_buffer("band_idx", torch.full((self.r,), -1, dtype=torch.long))  # not a band of U
        elif mode == "learned":
            # PHASE 2 of the free-LoRA analysis (ANALYSE_writes_additif.md, 28/09): B frozen on a basis LEARNED by a free
            # LoRA run — the column space of its B, orthonormalised by QR, one basis per module, read from a file — and A
            # trained from zero. Same structure and budget as random_ortho (r*in, dW = 0 at step 0, no Sigma): only the
            # ORIGIN of the frozen basis differs. It asks whether a B frozen at the right place escapes the cost of freezing.
            if learned_B is None:
                raise ValueError("mode 'learned' needs a learned basis for every module (learned_basis file)")
            Bl = learned_B.detach().to("cpu", torch.float32)
            if tuple(Bl.shape) != (out_f, self.r):
                raise ValueError(f"learned basis shape {tuple(Bl.shape)} != {(out_f, self.r)} for module {module_salt!r}")
            dev_orth = float((Bl.T @ Bl - torch.eye(self.r)).abs().max())
            if dev_orth > 1e-4:
                raise ValueError(f"learned basis of {module_salt!r} is not orthonormal (max |B^T B - I| = {dev_orth:.1e})")
            self.A = nn.Parameter(torch.zeros(self.r, in_f, dtype=torch.float32, device=dev))
            self.B = nn.Parameter(Bl.to(dev).contiguous(), requires_grad=False)
            self.register_buffer("band_idx", torch.full((self.r,), -1, dtype=torch.long))  # not a band of U
        elif mode in INIT_ONLY:
            # B initialised on the band, A kaiming as in `free`, BOTH trained. dW is NOT zero at
            # step 0 here — that is the point: PiSSA and MiLoRA start from a spectral point and let
            # the update drift away from it. Reporting dW(0) != 0 matters, because every other arm
            # in this repo starts at exactly zero and the learning curves are not comparable at the
            # first steps.
            if svd is None:
                raise ValueError(f"mode {mode!r} needs an SVDEntry of the base weight")
            if svd.shape != (out_f, in_f):
                raise ValueError(f"SVD shape {svd.shape} != weight shape {(out_f, in_f)}")
            if self.r > svd.m:
                raise ValueError(f"r={self.r} > min(out,in)={svd.m}")
            plain = mode[len("init_"):]
            _salt = module_salt if subspace_salted else ""
            spec = {"top": f"top:{self.r}", "bottom": f"bottom:{self.r}",
                    "random": (f"random:{self.r}:{seed}:{_salt}" if _salt
                               else f"random:{self.r}:{seed}")}[plain]
            idx = band_indices(svd.m, spec, S=svd.S, rel_tol=sigma_rel_tol)
            self.register_buffer("band_idx", idx.clone())
            self.band_frac = band_frac
            sq = svd.S[idx].sqrt().to(dev, torch.float32)
            B0 = (svd.U[:, idx].to(dev, torch.float32) * sq)              # U_r sqrt(S_r)
            A0 = (sq.unsqueeze(1) * svd.Vh[idx, :].to(dev, torch.float32))  # sqrt(S_r) V_r^T
            # the scaling is applied in forward(), so the adapter must reproduce the band DIVIDED
            # by it for W_frozen + scaling * B A to equal W0 exactly at step 0
            self.A = nn.Parameter(A0.contiguous() / self.scaling)
            self.B = nn.Parameter(B0.contiguous())
            with torch.no_grad():
                # the band is B0 @ A0 = U_r sqrt(S_r) sqrt(S_r) V_r^T = U_r S_r V_r^T. Writing it as
                # U_r @ A0 would drop one factor sqrt(S_r) and leave the model perturbed at step 0 —
                # checked numerically, the residual was 0.87 on a random 6x4.
                band = B0 @ A0
                self.base.weight.data -= band.to(self.base.weight.dtype)
        elif mode in DUAL:
            # Mirror image: A frozen on r rows of Vh, B trained, B initialised to zero so dW = 0 at
            # step 0 exactly as elsewhere.
            #
            # BUDGET WARNING. A dual arm trains r * d_out parameters, a constrained arm r * d_in.
            # They coincide on square modules and do NOT on rectangular ones: on gate_proj
            # (8192 x 2048) a dual arm trains four times more. Any comparison across the two
            # families must report both counts.
            plain = mode[len("dual_"):]
            # dual_random_ortho draws a Haar basis and uses no SVD; inject_lora rightly loads none for it (CONSTRAINED
            # excludes it). The checks below used to run for EVERY dual arm, so all 49 dual_random_ortho runs of
            # PROTOCOL_freeze_A.md failed here before training (found 26/09). They now run only for the arms that use Vh.
            if plain != "random_ortho":
                if svd is None:
                    raise ValueError(f"mode {mode!r} needs an SVDEntry of the base weight")
                if svd.shape != (out_f, in_f):
                    raise ValueError(f"SVD shape {svd.shape} != weight shape {(out_f, in_f)}")
                if self.r > svd.m:
                    raise ValueError(f"r={self.r} > min(out,in)={svd.m}")
            elif self.r > min(out_f, in_f):
                raise ValueError(f"r={self.r} > min(out,in)={min(out_f, in_f)}")
            if plain == "random_ortho":
                _h = (int(hashlib.blake2b(module_salt.encode(), digest_size=8).hexdigest(), 16)
                      if subspace_salted else 0)
                g = torch.Generator(device="cpu").manual_seed((int(seed) * 7919 + 17 + _h) % (2 ** 63 - 1))
                A_f = torch.linalg.qr(torch.randn(in_f, self.r, generator=g))[0].T.contiguous()
                self.register_buffer("band_idx", torch.full((self.r,), -1, dtype=torch.long))
            else:
                _salt = module_salt if subspace_salted else ""
                spec = {"top": f"top:{self.r}", "bottom": f"bottom:{self.r}",
                        "random": (f"random:{self.r}:{seed}:{_salt}" if _salt
                                   else f"random:{self.r}:{seed}")}[plain]
                idx = band_indices(svd.m, spec, S=svd.S, rel_tol=sigma_rel_tol)
                A_f = svd.Vh[idx, :].contiguous()
                self.register_buffer("band_idx", idx.clone())
            self.band_frac = band_frac
            self.A = nn.Parameter(A_f.to(dev, torch.float32), requires_grad=False)
            self.B = nn.Parameter(torch.zeros(out_f, self.r, dtype=torch.float32, device=dev))
        else:
            if svd is None:
                raise ValueError(f"mode {mode!r} needs an SVDEntry of the base weight")
            if svd.shape != (out_f, in_f):
                raise ValueError(f"SVD shape {svd.shape} != weight shape {(out_f, in_f)}")
            if self.r > svd.m:
                raise ValueError(f"r={self.r} > min(out,in)={svd.m}")
            if mode == "band":
                if band_frac is None:
                    raise ValueError("mode 'band' needs band_frac in [0, 1]")
                spec = f"at:{self.r}:{band_frac}"
            else:
                # the module name salts the random draw so that each module gets its OWN subspace
                _salt = module_salt if subspace_salted else ""
                spec = {"top": f"top:{self.r}", "top_sigma": f"top:{self.r}", "top_unfrozen": f"top:{self.r}",
                        "top_scalar": f"top:{self.r}", "top_lrscale": f"top:{self.r}", "bottom": f"bottom:{self.r}", "bottom_sigma": f"bottom:{self.r}", "bottom_unfrozen": f"bottom:{self.r}",
                        "random": (f"random:{self.r}:{seed}:{_salt}" if _salt
                                   else f"random:{self.r}:{seed}")}[mode]
            self.band_frac = band_frac
            idx = band_indices(svd.m, spec, S=svd.S, rel_tol=sigma_rel_tol)
            B = svd.U[:, idx].to(dev, torch.float32).contiguous()
            if mode in ("top_sigma", "bottom_sigma"):
                # bottom_sigma (24/09): the same construction on the MINOR band, B = U_r sqrt(S_r) with the
                # r smallest singular values. It tests, in our own setup, whether keeping Sigma in the
                # frozen factor reverses the direction of the rate shift, as in LoRA-XS. Budget as `bottom`.
                # the top band frozen WITH its singular values, B = U_r sqrt(S_r), as PiSSA
                # initialises it, and A trained from zero. It separates the two differences between
                # our arms and PiSSA: keeping Sigma, and freezing a factor. Budget as `top`.
                B = (B * svd.S[idx].sqrt().to(dev, torch.float32)).contiguous()
            if mode == "top_scalar":
                # the control for top_sigma. Under Adam, multiplying a FROZEN B by a constant leaves
                # A's step unchanged (the gradient is multiplied by the same constant and Adam
                # normalises it) but multiplies the update dW = c B A. So B = U_r sqrt(S_r) is `top`
                # with a per-module, per-direction rate multiplier. This arm keeps only the
                # per-module part: one scalar, the mean of sqrt(sigma) over the band, so the shape of
                # Sigma inside the band is gone and the module-level scale is kept. If it recovers
                # top_sigma's score, restoring Sigma is a rate effect at module level; if it does
                # not, the shape inside the band is what matters. Budget as `top`.
                # s equalises the Frobenius norm with top_sigma: with U orthonormal,
                # ||U sqrt(S)||_F^2 = sum(S), so s = ||sqrt(S_r)||_F / sqrt(r) = sqrt(mean(S_r))
                # leaves ||B||_F unchanged. Fixing it this way before launch removes the scalar as a
                # degree of freedom; the mean of sqrt(S) would be a different, unmotivated choice.
                B = (B * svd.S[idx].mean().sqrt().to(dev, torch.float32)).contiguous()
            if mode == "top_lrscale":
                # top_scalar's scale moved from B to the OPTIMISER (PROTOCOL_top_lrscale.md, 26/09): B = U_r as in `top`, and
                # A's learning rate multiplied by s = sqrt(mean(S_r)), the very scalar top_scalar puts in B. Under Adam, with no
                # clipping and eps -> 0, the two follow the same trajectory (A here = s * A of top_scalar). They differ only
                # through the GLOBAL gradient clipping and Adam's eps, which is exactly what the protocol tests.
                self.lr_mult = float(svd.S[idx].mean().sqrt())
            self.register_buffer("band_idx", idx.clone())
            self.A = nn.Parameter(torch.zeros(self.r, in_f, dtype=torch.float32, device=dev))
            # top_unfrozen: same basis as `top` for the initial B, but B is trained. It isolates
            # freezing from the basis, the initialisation and the rank. With A_0 = 0 the gradient of
            # B is zero at the first step; B moves only once A is nonzero, which is expected.
            # bottom_unfrozen (26/09): the same control on the minor band, so that freezing is isolated for both ends.
            self.B = nn.Parameter(B, requires_grad=(mode in ("top_unfrozen", "bottom_unfrozen")))

    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        if self.merged:
            return out
        xa = x.to(torch.float32)
        lora = (xa @ self.A.T) @ self.B.T
        return out + (lora * self.scaling).to(out.dtype)

    @torch.no_grad()
    def delta_w(self) -> torch.Tensor:
        """dW (out, in) in fp32, *without* the alpha/r scaling (raw B A)."""
        return self.B @ self.A

    @torch.no_grad()
    def factors(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.A.detach().clone(), self.B.detach().clone()

    @torch.no_grad()
    def merge_(self) -> None:
        """Fold scaling * B A into base.weight (in place).  Idempotent."""
        if self.merged:
            return
        self.base.weight.add_((self.B @ self.A * self.scaling).to(self.base.weight.dtype))
        self.merged = True

    @torch.no_grad()
    def unmerge_(self) -> None:
        if not self.merged:
            return
        self.base.weight.sub_((self.B @ self.A * self.scaling).to(self.base.weight.dtype))
        self.merged = False

    def extra_repr(self) -> str:
        return f"r={self.r}, mode={self.mode}, alpha={self.alpha}, in={self.base.in_features}, out={self.base.out_features}"


# ---------------------------------------------------------------------- #
# Injection
# ---------------------------------------------------------------------- #
def _set_submodule(model: nn.Module, name: str, new: nn.Module) -> None:
    parent_name, _, child = name.rpartition(".")
    parent = model.get_submodule(parent_name) if parent_name else model
    setattr(parent, child, new)


def inject_lora(
    model: nn.Module,
    r: int,
    mode: str,
    alpha: float | None = None,
    cache_dir: str | Path | None = None,
    model_id: str | None = None,
    seed: int = 0,
    target_suffixes=TARGET_SUFFIXES,
    svd_lookup: dict[str, SVDEntry] | None = None,
    band_frac: float | None = None,
    per_module: dict | None = None,
    sigma_rel_tol: float | None = None,
    subspace_salted: bool = False,
    learned_basis: str | Path | None = None,
) -> dict[str, LoRALinear]:
    """Replace every target nn.Linear by a LoRALinear.  Returns ``{name: module}``.

    For constrained modes provide either ``svd_lookup`` (name -> SVDEntry) or
    ``cache_dir`` + ``model_id`` so entries are loaded from the SVD cache.
    ``mode="per_module"`` needs ``per_module`` (see ``parse_per_module``).
    Freezes every non-LoRA parameter of the model.
    """
    if mode == "per_module" and not per_module:
        raise ValueError("mode='per_module' needs a per_module spec")
    for p in model.parameters():
        p.requires_grad_(False)
    targets = list(iter_target_modules(model, target_suffixes))
    if not targets:
        raise ValueError(f"no target modules with suffixes {tuple(target_suffixes)} in model")
    # AND every requested target must match at least one module. The global check above only fires
    # when NOTHING matches; a per-layer grid asks for `layers.15.self_attn.q_proj`, and a typo in the
    # layer number or the attention path matches nothing while the other entries still match. The run
    # would then train fewer adapters than asked, report a score, and raise nothing — the score of a
    # model adapted somewhere else, or of the base model. Silent, and invisible in results.csv.
    matched = {n for n, _ in targets}
    unmatched = [t for t in target_suffixes
                 if not any(n.split(".")[-1] == t or n.endswith(t) for n in matched)]
    if unmatched:
        raise ValueError(
            f"these target modules matched nothing: {unmatched}. Bare suffixes select that "
            f"projection in every layer ('q_proj'); dotted paths select one module in one layer "
            f"('layers.15.self_attn.q_proj'). Check the layer index and the attention/mlp path.")
    bases = None
    if mode == "learned":
        if not learned_basis:
            raise ValueError("mode='learned' needs learned_basis (path of the basis file)")
        bases = load_file(str(learned_basis))
        missing = [n for n, _ in targets if n not in bases]
        if missing:
            raise ValueError(f"the learned basis file has no basis for {len(missing)} target module(s), e.g. {missing[:3]}")
    injected: dict[str, LoRALinear] = {}
    for name, lin in targets:
        svd = None
        m_mode, m_frac = (parse_per_module(per_module, name) if mode == "per_module" else (mode, band_frac))
        if m_mode in CONSTRAINED:
            if svd_lookup is not None and name in svd_lookup:
                svd = svd_lookup[name]
            elif cache_dir is not None and model_id is not None:
                svd = load_svd(cache_dir, model_id, name, device="cpu")
            else:
                raise ValueError("constrained mode needs svd_lookup or (cache_dir, model_id)")
        mod = LoRALinear(lin, r=r, mode=m_mode, alpha=alpha, svd=svd, seed=seed, band_frac=m_frac,
                         sigma_rel_tol=sigma_rel_tol, module_salt=name,
                         subspace_salted=subspace_salted, learned_B=(bases[name] if bases is not None else None))
        _set_submodule(model, name, mod)
        injected[name] = mod
    return injected


def lora_modules(model: nn.Module) -> dict[str, LoRALinear]:
    return {n: m for n, m in model.named_modules() if isinstance(m, LoRALinear)}


def trainable_parameters(model: nn.Module) -> list[nn.Parameter]:
    return [p for p in model.parameters() if p.requires_grad]


def count_trainable(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def lora_param_groups(modules: dict, lr: float) -> list[dict]:
    """One optimiser group per adapter, at lr * lr_mult (top_lrscale); only trainable tensors. For every other arm lr_mult
    is 1 and the groups are equivalent to a flat parameter list."""
    groups = []
    for m in modules.values():
        ps = [q for q in (m.A, m.B) if q.requires_grad]
        if ps:
            groups.append({"params": ps, "lr": lr * float(getattr(m, "lr_mult", 1.0))})
    return groups


def expected_trainable(model: nn.Module, r: int, mode: str, target_suffixes=TARGET_SUFFIXES) -> int:
    """Closed-form trainable count.  ``per_module`` counts like any constrained mode."""
    """Closed-form trainable count, to assert against ``count_trainable`` after injection.

    Works before injection (targets are nn.Linear) and after (targets are LoRALinear)."""
    suffixes = tuple(target_suffixes)
    n = 0
    for name, mod in model.named_modules():
        if name.split(".")[-1] not in suffixes:
            continue
        lin = mod.base if isinstance(mod, LoRALinear) else mod
        if not isinstance(lin, nn.Linear):
            continue
        # free       : both factors    -> r * (in + out)
        # constrained: A only          -> r * in     (B is frozen on columns of U)
        # DUAL       : B only          -> r * OUT    (A is frozen on rows of V)
        # The dual case is not symmetric with the constrained one on rectangular modules: gate_proj
        # is 8192 x 2048, so a dual arm there trains four times what a constrained arm trains. This
        # function is what asserts the budget at every run, so getting it wrong would let a
        # four-fold budget difference pass as an equal-budget comparison.
        if mode in BOTH_FACTORS:
            n += r * (lin.in_features + lin.out_features)   # both factors, like `free`
        elif mode in DUAL:
            n += r * lin.out_features
        else:
            n += r * lin.in_features + (r * lin.out_features if mode == "free" else 0)
    return n


# ---------------------------------------------------------------------- #
# Update magnitude (Zhang et al. 2025: spectral gains can be magnitude gains)
# ---------------------------------------------------------------------- #
@torch.no_grad()
def update_norms(modules: dict[str, LoRALinear]) -> dict[str, float]:
    """||scaling * B A||_F per module, computed from the factors without forming dW."""
    out = {}
    for name, m in modules.items():
        A, B = m.A.float(), m.B.float()
        out[name] = float(torch.trace((B.T @ B) @ (A @ A.T)).clamp_min(0).sqrt()) * m.scaling
    return out


@torch.no_grad()
def rescale_update_(modules: dict[str, LoRALinear], target_mean_norm: float) -> float:
    """Multiply every A by one global factor so that the mean per-module ||scaling B A||_F equals
    ``target_mean_norm``.  Preserves the relative distribution across modules (the arm's
    *directions*) and changes only its magnitude.  Returns the factor applied."""
    norms = update_norms(modules)
    mean = sum(norms.values()) / max(1, len(norms))
    if mean <= 0:
        return 1.0
    f = target_mean_norm / mean
    for m in modules.values():
        m.A.mul_(f)
    return f


# ---------------------------------------------------------------------- #
# Save / load of factors (never dW)
# ---------------------------------------------------------------------- #
def save_factors(modules: dict[str, LoRALinear], path: str | Path, metadata: dict[str, str] | None = None) -> None:
    tensors: dict[str, torch.Tensor] = {}
    for name, m in modules.items():
        A, B = m.factors()
        tensors[f"{name}.A"] = A.cpu().contiguous()
        tensors[f"{name}.B"] = B.cpu().contiguous()
    meta = {"scaling": str(next(iter(modules.values())).scaling)} if modules else {}
    if metadata:
        meta.update({k: str(v) for k, v in metadata.items()})
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(path), metadata=meta)


def load_factors(path: str | Path, device="cpu") -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    t = load_file(str(path), device=str(device))
    names = sorted({k[:-2] for k in t if k.endswith(".A")})
    return {n: (t[f"{n}.A"], t[f"{n}.B"]) for n in names}


def load_factors_into(model: nn.Module, path: str | Path) -> None:
    facs = load_factors(path)
    mods = lora_modules(model)
    for name, (A, B) in facs.items():
        m = mods[name]
        m.A.data.copy_(A.to(m.A.device))
        m.B.data.copy_(B.to(m.B.device))


def merge_all_(model: nn.Module) -> None:
    for m in lora_modules(model).values():
        m.merge_()


def unmerge_all_(model: nn.Module) -> None:
    for m in lora_modules(model).values():
        m.unmerge_()


def lora_linear_forward_check(x: torch.Tensor, m: LoRALinear, atol: float = 1e-4) -> bool:
    """Debug helper: merged and unmerged forward agree."""
    with torch.no_grad():
        y1 = m(x)
        m.merge_()
        y2 = m(x)
        m.unmerge_()
    return bool(torch.allclose(y1.float(), y2.float(), atol=atol, rtol=1e-3))


__all__ = [
    "MODES", "CONSTRAINED", "DUAL", "INIT_ONLY", "SALTED_MODES", "LoRALinear", "inject_lora", "lora_modules", "trainable_parameters",
    "count_trainable", "expected_trainable", "save_factors", "load_factors", "load_factors_into",
    "merge_all_", "unmerge_all_",
]
