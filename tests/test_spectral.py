import math

import pytest
import torch

from lorasub.spectral import (SVDEntry, band_energies, band_indices, compute_svd_cache, delta_frobenius_sq,
                              effective_rank, gradient_band_energies, intruder_count, iter_target_modules,
                              list_cached, load_svd, null_reference, parse_module_name, principal_angles,
                              svd_of_weight)
from tiny import tiny_model


def _entry(W: torch.Tensor) -> SVDEntry:
    U, S, Vh = svd_of_weight(W)
    return SVDEntry("w", tuple(W.shape), S, U, Vh)


def test_svd_reconstruction_and_order():
    torch.manual_seed(0)
    W = torch.randn(64, 48)
    U, S, Vh = svd_of_weight(W)
    assert torch.allclose((U * S) @ Vh, W, atol=1e-4)
    assert torch.all(S[:-1] >= S[1:])
    assert U.shape == (64, 48) and Vh.shape == (48, 48)


def test_band_indices():
    assert band_indices(100, "top:5").tolist() == [0, 1, 2, 3, 4]
    assert band_indices(100, "bottom:3").tolist() == [97, 98, 99]
    assert band_indices(100, "frac:0.0-0.1").tolist() == list(range(10))
    assert band_indices(100, "frac:0.9-1.0").tolist() == list(range(90, 100))
    r1 = band_indices(100, "random:7:3")
    r2 = band_indices(100, "random:7:3")
    assert r1.tolist() == r2.tolist() and len(set(r1.tolist())) == 7
    with pytest.raises(ValueError):
        band_indices(10, "top:11")


def test_frobenius_without_forming_dw():
    A, B = torch.randn(8, 48), torch.randn(64, 8)
    assert math.isclose(delta_frobenius_sq(A, B), float((B @ A).pow(2).sum()), rel_tol=1e-5)


def test_null_reference_random_update():
    """A random rank-8 update puts ~|band|/m of its energy in every band, both sides."""
    torch.manual_seed(1)
    W = torch.randn(128, 128)
    e = _entry(W)
    bands = ["frac:0.0-0.25", "frac:0.75-1.0", "top:16", "bottom:16"]
    acc = {b: [0.0, 0.0] for b in bands}
    n = 30
    for _ in range(n):
        A, B = torch.randn(8, 128), torch.randn(128, 8)
        r = band_energies(A, B, e, bands)
        for b in bands:
            acc[b][0] += r[b][0] / n
            acc[b][1] += r[b][1] / n
        assert r["_outside_left"] < 1e-5 and r["_outside_right"] < 1e-5  # square full-rank W0
    for b in bands:
        null = null_reference(128, b)
        assert abs(acc[b][0] - null) < 0.06, (b, acc[b][0], null)
        assert abs(acc[b][1] - null) < 0.06, (b, acc[b][1], null)


def test_null_reference_rectangular_sides():
    """Isotropic dW on a wide module: a top-8 band holds 8/out on the left and 8/in on the right."""
    torch.manual_seed(7)
    for shape in [(96, 32), (32, 96)]:
        W = torch.randn(*shape)
        e = _entry(W)
        acc_l = acc_r = 0.0
        n = 40
        for _ in range(n):
            A, B = torch.randn(8, shape[1]), torch.randn(shape[0], 8)  # isotropic rank-8 update
            r = band_energies(A, B, e, ["top:8"])
            acc_l += r["top:8"][0] / n
            acc_r += r["top:8"][1] / n
        assert abs(acc_l - null_reference(e, "top:8", "left")) < 0.05, (shape, acc_l)
        assert abs(acc_r - null_reference(e, "top:8", "right")) < 0.05, (shape, acc_r)
        assert null_reference(e, "top:8", "left") == 8 / shape[0]
        assert null_reference(e, "top:8", "right") == 8 / shape[1]


def test_intruder_count_scaling_matters():
    torch.manual_seed(8)
    U, _ = torch.linalg.qr(torch.randn(64, 64))
    V, _ = torch.linalg.qr(torch.randn(64, 64))
    W = U @ torch.diag(torch.linspace(10, 0.1, 64)) @ V.T
    # an update along the last left singular direction: an intruder only if it is scaled up enough
    A = V[:, -1:].T.clone() * 1.0
    B = U[:, -1:].clone()
    assert intruder_count(W, A, B, k=3, device="cpu", scaling=1.0) == 0
    assert intruder_count(W, A, B, k=3, device="cpu", scaling=100.0) >= 1


def test_update_built_in_top_band():
    torch.manual_seed(2)
    W = torch.randn(96, 80)
    e = _entry(W)
    B = e.U[:, :8].clone()  # constrained arm: B = top-8 left singular vectors
    A = torch.randn(8, 80)
    r = band_energies(A, B, e, ["top:8", "bottom:8", "frac:0.0-0.1"])
    assert abs(r["top:8"][0] - 1.0) < 1e-5
    assert r["bottom:8"][0] < 1e-6
    assert abs(r["frac:0.0-0.1"][0] - 1.0) < 1e-5
    B = e.U[:, -8:].clone()
    r = band_energies(A, B, e, ["top:8", "bottom:8"])
    assert abs(r["bottom:8"][0] - 1.0) < 1e-5 and r["top:8"][0] < 1e-6


def test_rectangular_outside_span():
    """For out > in, an update with rows outside span(U) shows up in _outside_left."""
    torch.manual_seed(3)
    W = torch.randn(96, 32)  # m = 32, U is 96x32
    e = _entry(W)
    Q, _ = torch.linalg.qr(torch.randn(96, 96))
    # complement of span(U): project random vectors out of U
    R = torch.randn(96, 4)
    R = R - e.U @ (e.U.T @ R)
    r = band_energies(torch.randn(4, 32), R, e, ["top:4"])
    assert r["_outside_left"] > 0.99


def test_gradient_band_energies_matches_lowrank():
    torch.manual_seed(4)
    W = torch.randn(64, 64)
    e = _entry(W)
    A, B = torch.randn(4, 64), torch.randn(64, 4)
    r1 = band_energies(A, B, e, ["frac:0.0-0.5"])
    r2 = gradient_band_energies(B @ A, e, ["frac:0.0-0.5"])
    assert abs(r1["frac:0.0-0.5"][0] - r2["frac:0.0-0.5"][0]) < 1e-5
    assert abs(r1["frac:0.0-0.5"][1] - r2["frac:0.0-0.5"][1]) < 1e-5


def test_principal_angles():
    Q, _ = torch.linalg.qr(torch.randn(50, 20, dtype=torch.float64))
    a = principal_angles(Q[:, :5], Q[:, :5])
    assert torch.all(a.abs() < 1e-5)
    b = principal_angles(Q[:, :5], Q[:, 5:10])
    assert torch.all((b - math.pi / 2).abs() < 1e-4)


def test_effective_rank():
    U, _ = torch.linalg.qr(torch.randn(40, 8))
    V, _ = torch.linalg.qr(torch.randn(30, 8))
    assert abs(effective_rank(V[:, :1].T, U[:, :1]) - 1.0) < 1e-5  # rank-1
    assert abs(effective_rank(V.T, U) - 8.0) < 1e-4  # 8 equal singular values


def test_intruder_count_zero_for_tiny_update():
    torch.manual_seed(5)
    W = torch.randn(64, 64) @ torch.diag(torch.linspace(10, 0.1, 64)) @ torch.randn(64, 64)
    A, B = 1e-6 * torch.randn(4, 64), torch.randn(64, 4)
    assert intruder_count(W, A, B, k=5, device="cpu") == 0


def test_cache_roundtrip_and_names(tmp_path):
    model = tiny_model()
    names = compute_svd_cache(model, tmp_path, "tiny/model", device="cpu", verbose=False)
    assert len(names) == 2 * 7  # 2 layers x 7 projections
    assert sorted(list_cached(tmp_path, "tiny/model")) == sorted(names)
    e = load_svd(tmp_path, "tiny/model", names[0])
    layer, mt = parse_module_name(names[0])
    assert layer == 0 and mt.endswith("_proj")
    lin = dict(iter_target_modules(model))[names[0]]
    W = lin.weight.detach().float()
    assert torch.allclose((e.U * e.S) @ e.Vh, W, atol=2e-2 * W.abs().max())  # fp16 storage


def test_rectangular_sides_independent():
    """For rectangular W0 the decile energies on each side sum to 1 minus the outside part,
    and match a brute-force computation on the materialised dW."""
    torch.manual_seed(6)
    for shape in [(48, 96), (96, 48)]:
        W = torch.randn(*shape)
        e = _entry(W)
        A, B = torch.randn(4, shape[1]), torch.randn(shape[0], 4)
        dW = B @ A
        from lorasub.spectral import decile_bands
        r = band_energies(A, B, e, decile_bands())
        left = sum(v[0] for k, v in r.items() if k.startswith("frac"))
        right = sum(v[1] for k, v in r.items() if k.startswith("frac"))
        assert abs(left + r["_outside_left"] - 1.0) < 1e-4
        assert abs(right + r["_outside_right"] - 1.0) < 1e-4
        idx = band_indices(e.m, "frac:0.0-0.1")
        brute_left = float((e.U[:, idx].T @ dW).pow(2).sum() / dW.pow(2).sum())
        brute_right = float((dW @ e.V[:, idx]).pow(2).sum() / dW.pow(2).sum())
        assert abs(r["frac:0.0-0.1"][0] - brute_left) < 1e-5
        assert abs(r["frac:0.0-0.1"][1] - brute_right) < 1e-5
        g = gradient_band_energies(dW, e, ["frac:0.0-0.1"])
        assert abs(g["frac:0.0-0.1"][0] - brute_left) < 1e-5 and abs(g["frac:0.0-0.1"][1] - brute_right) < 1e-5
