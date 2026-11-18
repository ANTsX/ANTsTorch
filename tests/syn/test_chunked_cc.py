"""Slab-wise (chunked) local CC: same loss and gradients, bounded autograd memory (issue #91)."""

import ants
import numpy as np
import pytest
import torch

from antstorch.syn import syn_registration
from antstorch.syn.core.losses import _slab_bounds, local_ncc_loss_chunked, local_ncc_loss_nd


@pytest.mark.parametrize("length,chunks", [(10, 1), (10, 3), (10, 10), (10, 50), (7, 4)])
def test_slab_bounds_tile_the_axis(length, chunks):
    bounds = _slab_bounds(length, chunks)
    assert bounds[0][0] == 0 and bounds[-1][1] == length
    assert all(stop > start for start, stop in bounds)
    assert all(a[1] == b[0] for a, b in zip(bounds, bounds[1:]))
    assert len(bounds) == min(chunks, length)


@pytest.mark.parametrize("shape", [(1, 1, 17, 13, 11), (1, 1, 20, 9), (2, 1, 7, 30, 8)])
@pytest.mark.parametrize("squared", [True, False])
@pytest.mark.parametrize("window_size", [3, 5, 9])
@pytest.mark.parametrize("chunks", [1, 2, 3, 7, 100])
def test_chunked_cc_matches_full_loss_and_gradients(shape, squared, window_size, chunks):
    gen = torch.Generator().manual_seed(0)
    I = torch.randn(shape, generator=gen, dtype=torch.float64)
    J = 0.6 * I + torch.randn(shape, generator=gen, dtype=torch.float64)
    mask = (torch.rand(shape, generator=gen, dtype=torch.float64) > 0.2).to(torch.float64)
    I_ref, J_ref = I.clone().requires_grad_(True), J.clone().requires_grad_(True)
    reference = local_ncc_loss_nd(I_ref, J_ref, mask=mask, window_size=window_size, squared=squared)
    reference.backward()
    loss, grad_I, grad_J = local_ncc_loss_chunked(I, J, mask, window_size=window_size, squared=squared, chunks=chunks)
    assert not loss.requires_grad
    torch.testing.assert_close(loss, reference.detach(), rtol=0, atol=1e-14)
    torch.testing.assert_close(grad_I, I_ref.grad, rtol=0, atol=1e-14)
    torch.testing.assert_close(grad_J, J_ref.grad, rtol=0, atol=1e-14)


def test_chunked_cc_requires_a_mask():
    with pytest.raises(ValueError, match="mask"):
        local_ncc_loss_chunked(torch.zeros(1, 1, 5, 5, 5), torch.zeros(1, 1, 5, 5, 5), None)


def _blobs(shift=1.5, shape=(18, 20, 16)):
    z, y, x = np.mgrid[0:shape[0], 0:shape[1], 0:shape[2]].astype(np.float32)
    rng = np.random.default_rng(int(shift * 10))
    array = np.exp(-(((x - 8 - shift) / 3.5) ** 2 + ((y - 10) / 4.0) ** 2 + ((z - 9) / 3.0) ** 2))
    array += 0.05 * rng.random(shape, dtype=np.float32)
    return ants.from_numpy(np.ascontiguousarray(array.transpose(2, 1, 0)), spacing=(1.0, 1.1, 0.9))


@pytest.mark.parametrize("metric", ["cc2", "lncc"])
@pytest.mark.parametrize("schedule", ["end_of_level", "per_iteration"])
def test_syn_registration_metric_chunks_matches_unchunked(metric, schedule):
    fixed, moving = _blobs(0.0), _blobs(1.5)
    eye, zero = torch.eye(3, dtype=torch.float64), torch.zeros(3, dtype=torch.float64)
    common = dict(type_of_transform="SyN", initial_affine=(eye, zero), device="cpu", dtype=torch.float64,
                  syn_metric=metric, levels=(2, 1), reg_iterations=(3, 3), inverse_schedule=schedule)
    full = syn_registration(fixed, moving, **common)
    chunked = syn_registration(fixed, moving, metric_chunks=4, **common)
    assert chunked["provenance"]["metric_chunks"] == 4
    np.testing.assert_allclose(chunked["loss_history"], full["loss_history"], rtol=0, atol=1e-12)
    for key, index in (("fwdtransforms", 0), ("invtransforms", -1)):
        np.testing.assert_allclose(ants.image_read(chunked[key][index]).numpy(),
                                   ants.image_read(full[key][index]).numpy(), rtol=0, atol=1e-6)


def test_multivariate_cc_term_is_chunked_with_the_same_result():
    fixed, moving = _blobs(0.0), _blobs(1.5)
    eye, zero = torch.eye(3, dtype=torch.float64), torch.zeros(3, dtype=torch.float64)
    common = dict(type_of_transform="SyN", initial_affine=(eye, zero), device="cpu", dtype=torch.float64,
                  syn_metric="mse", primary_metric_weight=0.5, levels=(1,), reg_iterations=(4,),
                  multivariate_extras=[("CC", fixed, moving, 2.0, 2)])
    full = syn_registration(fixed, moving, **common)
    chunked = syn_registration(fixed, moving, metric_chunks=3, **common)
    np.testing.assert_allclose(chunked["loss_history"], full["loss_history"], rtol=0, atol=1e-12)


@pytest.mark.parametrize("value", [0, -2, 1.5, True])
def test_metric_chunks_must_be_a_positive_integer(value):
    fixed = _blobs(0.0)
    with pytest.raises(ValueError, match="metric_chunks"):
        syn_registration(fixed, fixed, type_of_transform="SyNOnly", metric_chunks=value, device="cpu")


def test_metric_chunks_requires_a_cc_metric():
    fixed = _blobs(0.0)
    with pytest.raises(ValueError, match="CC-family"):
        syn_registration(fixed, fixed, type_of_transform="SyNOnly", syn_metric="mse", metric_chunks=4, device="cpu")
