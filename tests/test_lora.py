import copy

import pytest
import torch

from lorasub import set_seed
from lorasub.lora import (count_trainable, expected_trainable, inject_lora, load_factors, load_factors_into,
                          lora_modules, merge_all_, save_factors, unmerge_all_)
from lorasub.spectral import compute_svd_cache, iter_target_modules
from tiny import TINY_CFG, tiny_model


def _batch(seed=0, n=4, L=16):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(4, TINY_CFG["vocab_size"], (n, L), generator=g)
    return {"input_ids": ids, "labels": ids.clone(), "attention_mask": torch.ones_like(ids)}


def _train(model, steps=20, lr=1e-2, seed=0, fixed_batch=False):
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    losses = []
    for s in range(steps):
        b = _batch(seed if fixed_batch else seed + s)
        loss = model(**b).loss
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
    return losses


def test_free_mode_matches_peft():
    peft = pytest.importorskip("peft")
    set_seed(0)
    base = tiny_model(0)
    targets = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    ours = copy.deepcopy(base)
    ours_mods = inject_lora(ours, r=4, mode="free", alpha=4.0, target_suffixes=targets)
    ref = peft.get_peft_model(
        copy.deepcopy(base),
        peft.LoraConfig(r=4, lora_alpha=4, lora_dropout=0.0, target_modules=targets, bias="none",
                        init_lora_weights=True, task_type="CAUSAL_LM"),
    )
    # copy our initial A into peft so both start from identical factors
    peft_layers = {n: m for n, m in ref.named_modules() if hasattr(m, "lora_A") and "default" in getattr(m, "lora_A", {})}
    for name, m in ours_mods.items():
        pm = peft_layers["base_model.model." + name]
        pm.lora_A["default"].weight.data.copy_(m.A.data)
        pm.lora_B["default"].weight.data.copy_(m.B.data)
    assert count_trainable(ours) == sum(p.numel() for p in ref.parameters() if p.requires_grad)
    l1 = _train(ours, steps=20)
    l2 = _train(ref, steps=20)
    assert max(abs(a - b) for a, b in zip(l1, l2)) < 1e-3, (l1[-3:], l2[-3:])
    for name, m in ours_mods.items():
        pm = peft_layers["base_model.model." + name]
        dw_ours = m.delta_w()
        dw_ref = pm.lora_B["default"].weight @ pm.lora_A["default"].weight
        assert torch.allclose(dw_ours, dw_ref, atol=1e-4)


@pytest.mark.parametrize("mode", ["top", "bottom", "random"])
def test_constrained_modes(tmp_path, mode):
    model = tiny_model(1)
    compute_svd_cache(model, tmp_path, "tiny/m", device="cpu", verbose=False)
    r = 4
    mods = inject_lora(model, r=r, mode=mode, cache_dir=tmp_path, model_id="tiny/m", seed=0)
    assert len(mods) == 14
    for m in mods.values():
        assert not m.B.requires_grad and m.A.requires_grad
        assert torch.allclose(m.B.T @ m.B, torch.eye(r), atol=1e-3)  # orthonormal columns
        assert float(m.delta_w().abs().max()) == 0.0  # identical start dW = 0
    assert count_trainable(model) == expected_trainable(model, r, mode)
    with torch.no_grad():
        l0 = float(model(**_batch(0)).loss)
    _train(model, steps=30, lr=5e-2, seed=0, fixed_batch=True)
    with torch.no_grad():
        l1 = float(model(**_batch(0)).loss)
    assert l1 < l0 - 0.05, "gradient must flow through A in constrained mode"
    for m in mods.values():
        assert float(m.delta_w().abs().max()) > 0.0


def test_top_and_bottom_use_correct_columns(tmp_path):
    model = tiny_model(2)
    compute_svd_cache(model, tmp_path, "tiny/m", device="cpu", verbose=False)
    lin = dict(iter_target_modules(model))
    top = inject_lora(copy.deepcopy(model), r=3, mode="top", cache_dir=tmp_path, model_id="tiny/m")
    bot = inject_lora(copy.deepcopy(model), r=3, mode="bottom", cache_dir=tmp_path, model_id="tiny/m")
    for name in lin:
        U, S, _ = torch.linalg.svd(lin[name].weight.detach().float(), full_matrices=False)
        # compare up to sign (singular vectors are defined up to sign) and fp16 storage
        for j in range(3):
            assert abs(float(top[name].B[:, j] @ U[:, j])) > 0.99
            assert abs(float(bot[name].B[:, j] @ U[:, U.shape[1] - 3 + j])) > 0.99


def test_free_counts_twice_constrained():
    m1, m2 = tiny_model(3), tiny_model(3)
    inject_lora(m1, r=8, mode="free")
    assert expected_trainable(m1, 8, "free") == count_trainable(m1)
    assert expected_trainable(m2, 8, "bottom") < expected_trainable(m1, 8, "free")


def test_merge_unmerge_and_save_load(tmp_path):
    model = tiny_model(4)
    mods = inject_lora(model, r=4, mode="free")
    _train(model, steps=5, lr=1e-2)
    b = _batch(9)
    with torch.no_grad():
        y1 = model(**b).logits
        merge_all_(model)
        y2 = model(**b).logits
        unmerge_all_(model)
        y3 = model(**b).logits
    assert torch.allclose(y1, y2, atol=1e-4) and torch.allclose(y1, y3, atol=1e-4)
    p = tmp_path / "f.safetensors"
    save_factors(mods, p, {"step": 5})
    facs = load_factors(p)
    assert set(facs) == set(mods)
    fresh = tiny_model(4)
    inject_lora(fresh, r=4, mode="free")
    load_factors_into(fresh, p)
    with torch.no_grad():
        y4 = fresh(**b).logits
    assert torch.allclose(y1, y4, atol=1e-4)
    assert len(lora_modules(fresh)) == 14


def test_band_mode_sweeps_the_spectrum(tmp_path):
    from lorasub.spectral import band_indices

    assert band_indices(100, "at:10:0.0").tolist() == list(range(0, 10))
    assert band_indices(100, "at:10:1.0").tolist() == list(range(90, 100))
    assert band_indices(100, "at:10:0.5").tolist() == list(range(45, 55))
    model = tiny_model(5)
    compute_svd_cache(model, tmp_path, "tiny/m", device="cpu", verbose=False)
    mods = inject_lora(model, r=3, mode="band", cache_dir=tmp_path, model_id="tiny/m", band_frac=0.5)
    for m in mods.values():
        assert m.mode == "band" and m.band_frac == 0.5 and not m.B.requires_grad
        assert torch.allclose(m.B.T @ m.B, torch.eye(3), atol=1e-3)
    assert count_trainable(model) == expected_trainable(model, 3, "band")
    with pytest.raises(ValueError):
        inject_lora(tiny_model(5), r=3, mode="band", cache_dir=tmp_path, model_id="tiny/m")


def test_random_ortho_arm_is_frozen_orthonormal_and_outside_span_u(tmp_path):
    """The control that separates 'B is frozen' from 'B comes from the spectrum of W0'. Every other
    constrained arm freezes B on columns of U; this one freezes B on random orthonormal directions with
    no relation to W0, trains A, and has exactly the budget of the constrained arms. If it matches
    `random`, span(U) does not matter and the whole gap to `free` is the cost of freezing B."""
    import torch

    from lorasub.lora import LoRALinear, count_trainable, inject_lora, lora_modules
    from lorasub.spectral import compute_svd_cache, load_svd
    from tiny import tiny_model

    lin = torch.nn.Linear(24, 16, bias=False)
    m = LoRALinear(lin, r=4, mode="random_ortho", seed=3)
    assert not m.B.requires_grad and m.A.requires_grad
    assert torch.allclose(m.B.T @ m.B, torch.eye(4), atol=1e-5)        # orthonormal
    assert torch.all(m.A == 0)                                          # dW = 0 at start
    assert torch.all(m.band_idx == -1)                                  # not a band of U
    # deterministic in the seed, different across seeds
    m2 = LoRALinear(torch.nn.Linear(24, 16, bias=False), r=4, mode="random_ortho", seed=3)
    m3 = LoRALinear(torch.nn.Linear(24, 16, bias=False), r=4, mode="random_ortho", seed=4)
    assert torch.allclose(m.B, m2.B) and not torch.allclose(m.B, m3.B)

    # same trainable budget as a constrained arm, half of free; no SVD required
    net_o, net_c, net_f = tiny_model(5), tiny_model(5), tiny_model(5)
    compute_svd_cache(tiny_model(5), tmp_path, "tiny/m", device="cpu", verbose=False)
    inject_lora(net_o, r=2, mode="random_ortho")
    inject_lora(net_c, r=2, mode="random", cache_dir=tmp_path, model_id="tiny/m")
    inject_lora(net_f, r=2, mode="free")
    assert count_trainable(net_o) == count_trainable(net_c) < count_trainable(net_f)
    # On a square module span(U) is the whole output space, so B is trivially inside it. What makes
    # this arm a control is that B is aligned with NO particular band: its energy spreads across the
    # spectrum at the null-reference level, where a `top`/`bottom` arm would put all of it in one band.
    name, mod = next(iter(lora_modules(net_o).items()))
    e = load_svd(tmp_path, "tiny/m", name, device="cpu")
    k = 4
    top_frac = float((e.U[:, :k].T @ mod.B).pow(2).sum() / mod.B.pow(2).sum())
    null = k / e.m
    assert top_frac < 0.6, top_frac                        # a `top` arm with r<=4 would give 1.0
    assert abs(top_frac - null) < 0.5                      # spread, at roughly the null level


def test_top_sigma_arm_is_the_top_band_scaled_by_sqrt_sigma_and_frozen(tmp_path):
    """The arm that separates Sigma from freezing: B = U_r sqrt(S_r) on the top band, frozen, A from
    zero. Same budget as `top`, same directions as `top`, and the model is unchanged at step 0."""
    from lorasub.lora import LoRALinear, lora_modules
    model = tiny_model(5)
    compute_svd_cache(model, tmp_path, "tiny/m", device="cpu", verbose=False)
    ref = tiny_model(5)
    x = torch.randint(0, 50, (2, 7))
    with torch.no_grad():
        y0 = ref(input_ids=x).logits
    mods = inject_lora(model, r=3, mode="top_sigma", cache_dir=tmp_path, model_id="tiny/m")
    top = inject_lora(tiny_model(5), r=3, mode="top", cache_dir=tmp_path, model_id="tiny/m")
    for name, m in mods.items():
        assert isinstance(m, LoRALinear) and m.mode == "top_sigma"
        assert not m.B.requires_grad and m.A.requires_grad and float(m.A.abs().max()) == 0.0
        s = torch.linalg.svdvals(m.base.weight.float())[:3]
        assert torch.allclose((m.B ** 2).sum(0), s, rtol=1e-3, atol=1e-5)      # |column j|^2 = sigma_j
        assert torch.allclose(m.B / m.B.norm(dim=0), top[name].B, atol=1e-4)   # same directions as top
    assert count_trainable(model) == expected_trainable(model, 3, "top_sigma") == expected_trainable(model, 3, "top")
    with torch.no_grad():
        assert torch.allclose(model(input_ids=x).logits, y0, atol=1e-5)          # unchanged at step 0
    assert len(lora_modules(model)) == len(mods)


def test_bottom_sigma_arm_is_the_minor_band_scaled_by_sqrt_sigma_and_frozen(tmp_path):
    """bottom_sigma is to `bottom` what top_sigma is to `top`: B = U_r sqrt(S_r) on the MINOR band (the r
    smallest singular values), frozen, A from zero. Same budget and directions as `bottom`, the model
    unchanged at step 0, and its SVD cache built by train.py on every node (17 of 20 top_sigma runs
    once died because a new mode was missing from that list)."""
    from lorasub.lora import LoRALinear
    from lorasub.train import NEEDS_SVD_CACHE
    model = tiny_model(5)
    compute_svd_cache(model, tmp_path, "tiny/m", device="cpu", verbose=False)
    ref = tiny_model(5)
    x = torch.randint(0, 50, (2, 7))
    with torch.no_grad():
        y0 = ref(input_ids=x).logits
    mods = inject_lora(model, r=3, mode="bottom_sigma", cache_dir=tmp_path, model_id="tiny/m")
    bot = inject_lora(tiny_model(5), r=3, mode="bottom", cache_dir=tmp_path, model_id="tiny/m")
    for name, m in mods.items():
        assert isinstance(m, LoRALinear) and m.mode == "bottom_sigma"
        assert not m.B.requires_grad and m.A.requires_grad and float(m.A.abs().max()) == 0.0
        s = torch.linalg.svdvals(m.base.weight.float())[-3:]
        assert torch.allclose(torch.sort((m.B ** 2).sum(0)).values, torch.sort(s).values, rtol=1e-3, atol=1e-6)
        assert torch.allclose(m.B / m.B.norm(dim=0), bot[name].B, atol=1e-4)   # same directions as bottom
    assert count_trainable(model) == expected_trainable(model, 3, "bottom_sigma") == expected_trainable(model, 3, "bottom")
    with torch.no_grad():
        assert torch.allclose(model(input_ids=x).logits, y0, atol=1e-5)          # unchanged at step 0
    assert "bottom_sigma" in NEEDS_SVD_CACHE


def test_every_mode_that_loads_the_svd_cache_also_builds_it():
    """inject_lora LOADS the SVD cache for every mode in lora.CONSTRAINED; train.py BUILDS it only for
    the modes in NEEDS_SVD_CACHE. A mode in the first and not the second works on a node that already
    has the cache and dies everywhere else, which is how 17 of the 20 top_sigma runs failed."""
    from lorasub.lora import CONSTRAINED, MODES
    from lorasub.train import NEEDS_SVD_CACHE
    assert set(CONSTRAINED) <= set(NEEDS_SVD_CACHE)
    assert "top_sigma" in NEEDS_SVD_CACHE
    assert not {"free", "random_ortho"} & set(NEEDS_SVD_CACHE)       # the two that never need it
    assert set(NEEDS_SVD_CACHE) - {"per_module"} <= set(MODES)


def test_top_unfrozen_is_tops_basis_with_B_trained_at_the_free_budget(tmp_path):
    """The control that isolates freezing: same basis as `top` for the initial B, B trainable, A from
    zero. Its budget is `free`'s at the same rank, which is why it is compared at rank 1 against a
    constrained arm at rank 2, exactly as `free` is."""
    from lorasub.lora import LoRALinear
    model = tiny_model(5)
    compute_svd_cache(model, tmp_path, "tiny/m", device="cpu", verbose=False)
    ref = tiny_model(5)
    x = torch.randint(0, 50, (2, 7))
    with torch.no_grad():
        y0 = ref(input_ids=x).logits
    mods = inject_lora(model, r=2, mode="top_unfrozen", cache_dir=tmp_path, model_id="tiny/m")
    top = inject_lora(tiny_model(5), r=2, mode="top", cache_dir=tmp_path, model_id="tiny/m")
    for name, m in mods.items():
        assert isinstance(m, LoRALinear) and m.mode == "top_unfrozen"
        assert m.B.requires_grad and m.A.requires_grad          # both factors train
        assert float(m.A.abs().max()) == 0.0                    # and the model is unchanged at step 0
        assert torch.allclose(m.B, top[name].B, atol=1e-6)      # same basis as top
    assert count_trainable(model) == expected_trainable(model, 2, "top_unfrozen") == expected_trainable(model, 2, "free")
    assert expected_trainable(model, 2, "top_unfrozen") > expected_trainable(model, 2, "top")
    with torch.no_grad():
        assert torch.allclose(model(input_ids=x).logits, y0, atol=1e-5)


def test_bottom_unfrozen_is_bottoms_basis_with_B_trained(tmp_path):
    """The same control on the minor band: `bottom`'s basis as the initial B, B trainable, A from zero,
    the model unchanged at step 0, the budget of `free` at the same rank, and its SVD cache built by
    train.py on every node. With top_unfrozen, it isolates freezing at both ends of the spectrum."""
    from lorasub.lora import LoRALinear
    from lorasub.train import NEEDS_SVD_CACHE
    model = tiny_model(5)
    compute_svd_cache(model, tmp_path, "tiny/m", device="cpu", verbose=False)
    ref = tiny_model(5)
    x = torch.randint(0, 50, (2, 7))
    with torch.no_grad():
        y0 = ref(input_ids=x).logits
    mods = inject_lora(model, r=2, mode="bottom_unfrozen", cache_dir=tmp_path, model_id="tiny/m")
    bot = inject_lora(tiny_model(5), r=2, mode="bottom", cache_dir=tmp_path, model_id="tiny/m")
    for name, m in mods.items():
        assert isinstance(m, LoRALinear) and m.mode == "bottom_unfrozen"
        assert m.B.requires_grad and m.A.requires_grad          # both factors train
        assert float(m.A.abs().max()) == 0.0                    # and the model is unchanged at step 0
        assert torch.allclose(m.B, bot[name].B, atol=1e-6)      # same basis as bottom
    assert count_trainable(model) == expected_trainable(model, 2, "bottom_unfrozen") == expected_trainable(model, 2, "free")
    with torch.no_grad():
        assert torch.allclose(model(input_ids=x).logits, y0, atol=1e-5)
    assert "bottom_unfrozen" in NEEDS_SVD_CACHE


def test_dual_arms_freeze_A_on_V_and_train_B_at_the_dual_budget(tmp_path):
    """The mirror of the constrained arms: A is pinned to r rows of Vh and B is trained from zero.
    Their budget is r * d_out, not r * d_in, and the two coincide only on square modules: the test
    asserts the count from the shapes so that a dual arm can never be compared to a constrained one
    as though the budgets matched."""
    from lorasub.lora import LoRALinear
    model = tiny_model(5)
    compute_svd_cache(model, tmp_path, "tiny/m", device="cpu", verbose=False)
    ref = tiny_model(5)
    x = torch.randint(0, 50, (2, 7))
    with torch.no_grad():
        y0 = ref(input_ids=x).logits
    mods = inject_lora(model, r=2, mode="dual_top", cache_dir=tmp_path, model_id="tiny/m")
    for m in mods.values():
        assert isinstance(m, LoRALinear) and m.mode == "dual_top"
        assert m.B.requires_grad and not m.A.requires_grad        # the mirror of the frozen arms
        assert float(m.B.detach().abs().max()) == 0.0             # dW = 0 at step 0
        rows = m.A.detach() @ m.A.detach().T                      # A's rows are orthonormal, to the
        assert torch.allclose(rows, torch.eye(2, dtype=rows.dtype), atol=1e-3)   # precision of fp32:
        # the measured residual is ~1.5e-4 on the tiny model, so a 1e-4 tolerance fails on arithmetic
        # rather than on the property being tested.
    n_out = sum(l.out_features for l in [mm.base for mm in mods.values()])
    n_in = sum(l.in_features for l in [mm.base for mm in mods.values()])
    assert count_trainable(model) == expected_trainable(model, 2, "dual_top") == 2 * n_out
    assert expected_trainable(model, 2, "top") == 2 * n_in        # and the two differ when in != out
    with torch.no_grad():
        assert torch.allclose(model(input_ids=x).logits, y0, atol=1e-5)


def test_dual_random_ortho_needs_no_svd(tmp_path):
    """dual_random_ortho freezes A on a random orthonormal basis and uses no SVD. Until 26/09 the dual branch checked for an
    SVD before looking at the arm, and inject_lora (rightly) loads none for this one: all 49 runs of PROTOCOL_freeze_A.md
    failed before training. The arm must build without any SVD cache, with the properties of every dual arm."""
    from lorasub.lora import LoRALinear
    model = tiny_model(5)
    ref = tiny_model(5)
    x = torch.randint(0, 50, (2, 7))
    with torch.no_grad():
        y0 = ref(input_ids=x).logits
    mods = inject_lora(model, r=2, mode="dual_random_ortho", cache_dir=None, model_id=None, seed=3, subspace_salted=True)
    As = []
    for m in mods.values():
        assert isinstance(m, LoRALinear) and m.mode == "dual_random_ortho"
        assert m.B.requires_grad and not m.A.requires_grad
        assert float(m.B.detach().abs().max()) == 0.0
        rows = m.A.detach() @ m.A.detach().T
        assert torch.allclose(rows, torch.eye(2, dtype=rows.dtype), atol=1e-3)
        As.append(m.A.detach().clone())
    n_out = sum(mm.base.out_features for mm in mods.values())
    assert count_trainable(model) == 2 * n_out
    with torch.no_grad():
        assert torch.allclose(model(input_ids=x).logits, y0, atol=1e-5)
    same_shape = [a for a in As if a.shape == As[0].shape]
    assert len(same_shape) >= 2 and not torch.allclose(same_shape[0], same_shape[1])   # the salt: one basis per module
    again = tiny_model(5)
    mods2 = inject_lora(again, r=2, mode="dual_random_ortho", cache_dir=None, model_id=None, seed=3, subspace_salted=True)
    assert all(torch.equal(a.A, b.A) for a, b in zip(mods.values(), mods2.values()))            # same seed, same basis


def test_top_lrscale_follows_top_scalar_under_adam(tmp_path):
    """top_lrscale keeps B = U_r (as top) and multiplies A's learning rate by s = sqrt(mean(S_r)), the scalar top_scalar puts
    in B (PROTOCOL_top_lrscale.md). Under Adam with no clipping and eps -> 0 the two must follow the same trajectory: the
    update s * U * A of top_scalar equals U * A of top_lrscale at every step. A wrong scalar or a wrong group breaks this."""
    from lorasub.config import RunConfig
    from lorasub.lora import lora_param_groups
    from lorasub.aggregate import ARMS
    m1, m2 = tiny_model(4), tiny_model(4)
    compute_svd_cache(m1, tmp_path, "tiny/m", device="cpu", verbose=False)
    mods1 = inject_lora(m1, r=2, mode="top_scalar", cache_dir=tmp_path, model_id="tiny/m")
    mods2 = inject_lora(m2, r=2, mode="top_lrscale", cache_dir=tmp_path, model_id="tiny/m")
    for n in mods1:
        a, b = mods1[n], mods2[n]
        assert a.lr_mult == 1.0 and b.lr_mult > 0
        s = float(a.B.detach().norm() / b.B.detach().norm())           # top_scalar's B is s * U_r, top_lrscale's is U_r
        assert abs(b.lr_mult - s) < 1e-4 * s
        assert not b.B.requires_grad and b.A.requires_grad
    assert count_trainable(m2) == expected_trainable(m2, 2, "top_lrscale") == count_trainable(m1)
    lr = 1e-2
    opt1 = torch.optim.AdamW([p for p in m1.parameters() if p.requires_grad], lr=lr, eps=1e-15, weight_decay=0.0)
    opt2 = torch.optim.AdamW(lora_param_groups(mods2, lr), lr=lr, eps=1e-15, weight_decay=0.0)
    g = torch.Generator().manual_seed(0)
    for _ in range(5):
        x = torch.randint(0, 50, (2, 7), generator=g)
        for m, opt in ((m1, opt1), (m2, opt2)):
            opt.zero_grad()
            m(input_ids=x).logits.pow(2).mean().backward()
            opt.step()
        for n in mods1:
            d1 = mods1[n].B.detach() @ mods1[n].A.detach()
            d2 = mods2[n].B.detach() @ mods2[n].A.detach()
            assert torch.allclose(d1, d2, rtol=1e-3, atol=1e-6), n
    cfg = RunConfig(model="meta-llama/Llama-3.2-1B", task="format", mode="top_lrscale", rank=2, lr=5e-3, seed=65)
    assert cfg.mode == "top_lrscale" and "top_lrscale" in ARMS


def test_top_scalar_is_top_times_one_number_per_module(tmp_path):
    """The control for top_sigma: same band, same budget, but the frozen factor carries a single
    scalar per module, the mean of sqrt(sigma) over the band, instead of the per-direction values.
    What separates it from top_sigma is only the shape of Sigma inside the band."""
    from lorasub.lora import LoRALinear
    model = tiny_model(5)
    compute_svd_cache(model, tmp_path, "tiny/m", device="cpu", verbose=False)
    scal = inject_lora(model, r=2, mode="top_scalar", cache_dir=tmp_path, model_id="tiny/m")
    top = inject_lora(tiny_model(5), r=2, mode="top", cache_dir=tmp_path, model_id="tiny/m")
    sig = inject_lora(tiny_model(5), r=2, mode="top_sigma", cache_dir=tmp_path, model_id="tiny/m")
    for name, m in scal.items():
        assert isinstance(m, LoRALinear) and m.mode == "top_scalar"
        assert not m.B.requires_grad and m.A.requires_grad and float(m.A.abs().max()) == 0.0
        ratio = m.B / top[name].B                       # one and the same number on every column
        assert float(ratio.std()) < 1e-5
        c = float(ratio.mean())
        cols = (sig[name].B / top[name].B).mean(dim=0)  # top_sigma's are the per-direction sqrt(s)
        rms = float((cols.pow(2).mean()).sqrt())        # c equalises the Frobenius norm, so it is
        assert abs(c - rms) < 1e-3                      # their quadratic mean, not their mean
        assert float(cols.std()) > 1e-6                 # which is not a constant, or the test is void
    assert count_trainable(model) == expected_trainable(model, 2, "top_scalar") == expected_trainable(model, 2, "top")
