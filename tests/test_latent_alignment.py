
import math
import random
import os
import pytest
import torch

import antstorch

# ---------- Global fixtures ----------

@pytest.fixture(autouse=True)
def _seed_everything():
    torch.manual_seed(0)
    random.seed(0)
    os.environ["PYTHONHASHSEED"] = "0"


def _make_views(B=128, D=32, V=2, noise=0.05, device="cpu"):
    base = torch.randn(B, D, device=device)
    views = [base + noise * torch.randn(B, D, device=device) for _ in range(V)]
    return views


# ---------- Basic shape / dtype / finiteness ----------

@pytest.mark.parametrize("V", [2, 3])
def test_all_losses_return_scalar_and_finite(V):
    B, D = 64, 16
    views = _make_views(B=B, D=D, V=V)
    # pearson
    val = antstorch.pearson_multi(views)
    assert val.shape == (), "pearson_multi should return a scalar tensor"
    assert torch.isfinite(val).all()
    # barlow
    val = antstorch.barlow_twins_multi(views, lam=5e-3)
    assert val.shape == ()
    assert torch.isfinite(val).all()
    # vicreg
    val = antstorch.vicreg_multi(views, w_inv=25, w_var=25, w_cov=1, gamma=1.0)
    assert val.shape == ()
    assert torch.isfinite(val).all()
    # infonce
    val = antstorch.info_nce_multi(views, T=0.2)
    assert val.shape == ()
    assert torch.isfinite(val).all()
    # hsic
    val = antstorch.hsic_multi(views, sigma=0.0)  # median heuristic
    assert val.shape == ()
    assert torch.isfinite(val).all()


# ---------- Pearson: affine invariance sanity ----------

def test_pearson_affine_invariance():
    B, D = 256, 32
    x = torch.randn(B, D)
    y = x.clone()
    z = x * 2.5 + 7.0
    l_xy = antstorch.pearson_multi([x, y]).item()
    l_xz = antstorch.pearson_multi([x, z]).item()
    # Both should be strongly negative (we minimize -corr)
    assert l_xy < -0.9
    # Affine invariance -> similar values
    assert abs(l_xy - l_xz) < 0.1


# ---------- Barlow Twins: identity vs shuffled ----------

def test_barlow_identity_smaller_than_shuffled():
    B, D = 256, 32
    x = torch.randn(B, D)
    y = x + 0.01 * torch.randn(B, D)
    loss_same = antstorch.barlow_twins_multi([x, y], lam=5e-3).item()
    y_shuf = y[torch.randperm(B)]
    loss_shuf = antstorch.barlow_twins_multi([x, y_shuf], lam=5e-3).item()
    assert loss_same < loss_shuf


# ---------- VICReg: invariance & variance floor ----------

def test_vicreg_closer_pairs_have_lower_loss():
    B, D = 128, 64
    x = torch.randn(B, D)
    y_close = x + 0.01 * torch.randn(B, D)
    y_far   = torch.randn(B, D)
    l_close = antstorch.vicreg_multi([x, y_close], w_inv=25, w_var=25, w_cov=1, gamma=1.0).item()
    l_far   = antstorch.vicreg_multi([x, y_far  ], w_inv=25, w_var=25, w_cov=1, gamma=1.0).item()
    assert l_close < l_far

def test_vicreg_variance_floor_triggers_on_low_std():
    B, D = 128, 64
    tiny = 0.01 * torch.randn(B, D)
    big  = torch.randn(B, D)
    l_tiny = antstorch.vicreg_multi([tiny, tiny], w_inv=25, w_var=25, w_cov=1, gamma=1.0).item()
    l_big  = antstorch.vicreg_multi([big , big ], w_inv=25, w_var=25, w_cov=1, gamma=1.0).item()
    assert l_tiny > l_big


# ---------- InfoNCE: positives vs negatives ----------

def test_infonce_positives_lower_than_negatives():
    B, D, V = 64, 32, 3
    views_pos = _make_views(B, D, V, noise=0.05)
    # build negatives by shuffling each view independently
    views_neg = [v[torch.randperm(B)] for v in views_pos]
    l_pos = antstorch.info_nce_multi(views_pos, T=0.2).item()
    l_neg = antstorch.info_nce_multi(views_neg, T=0.2).item()
    assert l_pos < l_neg

def test_infonce_temperature_extremes_are_stable():
    B, D = 64, 32
    v = _make_views(B, D, 2, noise=0.05)
    # T=0 should be clamped internally -> finite
    val0 = antstorch.info_nce_multi(v, T=0.0)
    assert torch.isfinite(val0).all()
    val_small = antstorch.info_nce_multi(v, T=1e-9)
    assert torch.isfinite(val_small).all()


# ---------- HSIC: dependence detection ----------

def test_hsic_detects_dependence():
    B, D = 128, 16
    x = torch.randn(B, D)
    y_dep = x + 0.2 * torch.randn(B, D)
    y_ind = torch.randn(B, D)
    # Biased HSIC: dependent > independent
    hs_dep = antstorch.hsic_biased(x, y_dep, sigma_x=0.0, sigma_y=0.0)
    hs_ind = antstorch.hsic_biased(x, y_ind, sigma_x=0.0, sigma_y=0.0)
    assert hs_dep.item() > hs_ind.item() - 1e-3
    # Multi-view HSIC returns a loss (negative HSIC)
    loss = antstorch.hsic_multi([x, y_dep], sigma=0.0)
    assert torch.isfinite(loss).all()
    assert loss.item() < 0.0  # maximizing dependence

def test_hsic_sigma_median_heuristic_is_finite():
    B, D = 64, 16
    x = torch.randn(B, D)
    y = torch.randn(B, D)
    val = antstorch.hsic_multi([x, y], sigma=0.0)  # 0 triggers median heuristic
    assert torch.isfinite(val).all()


# ---------- Small batches & autograd ----------

@pytest.mark.parametrize("B", [2, 4, 8])
def test_small_batches_are_stable(B):
    D = 8
    v = _make_views(B, D, 2)
    for fn in [
        lambda vv: antstorch.pearson_multi(vv),
        lambda vv: antstorch.barlow_twins_multi(vv, lam=5e-3),
        lambda vv: antstorch.vicreg_multi(vv),
        lambda vv: antstorch.info_nce_multi(vv, T=0.2),
        lambda vv: antstorch.hsic_multi(vv, sigma=0.0),
    ]:
        val = fn(v)
        assert torch.isfinite(val).all()

def test_gradients_exist_and_finite():
    B, D = 64, 16
    x = torch.randn(B, D, requires_grad=True)
    y = torch.randn(B, D, requires_grad=True)
    for fn in [
        lambda vv: antstorch.pearson_multi(vv),
        lambda vv: antstorch.barlow_twins_multi(vv, lam=5e-3),
        lambda vv: antstorch.vicreg_multi(vv),
        lambda vv: antstorch.info_nce_multi(vv, T=0.2),
        lambda vv: antstorch.hsic_multi(vv, sigma=0.0),
    ]:
        loss = fn([x, y])
        loss.backward(retain_graph=True)
        assert torch.isfinite(x.grad).all()
        x.grad.zero_(); y.grad.zero_()


# ---------- CUDA / AMP smoke (optional) ----------

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_amp_half_precision_is_finite_cuda():
    device = "cuda"
    B, D, V = 128, 32, 2
    views = _make_views(B, D, V, device=device)
    with torch.amp.autocast('cuda', dtype=torch.float16):
        val = antstorch.vicreg_multi(views, w_inv=25, w_var=25, w_cov=1, gamma=1.0)
    assert torch.isfinite(val).all()


# ---------- API misuse / mismatches ----------

def test_mismatched_batches_raise_or_fail_cleanly():
    B, D = 64, 16
    x = torch.randn(B, D)
    y = torch.randn(B+1, D)  # mismatched
    with pytest.raises(Exception):
        _ = antstorch.pearson_multi([x, y])
    with pytest.raises(Exception):
        _ = antstorch.barlow_twins_multi([x, y], lam=5e-3)
    with pytest.raises(Exception):
        _ = antstorch.vicreg_multi([x, y])
    with pytest.raises(Exception):
        _ = antstorch.info_nce_multi([x, y], T=0.2)
    with pytest.raises(Exception):
        _ = antstorch.hsic_multi([x, y], sigma=0.0)
