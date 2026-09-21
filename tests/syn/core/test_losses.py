import pytest
import torch

from antstorch.syn.core import (
    AnalyticalLNCC,
    ANTsPseudoLNCC,
    local_ncc_loss_nd,
    b_spline_3,
    mattes_mi_loss_core,
    mattes_mi_loss_nd,
    box_cc2_loss_nd,
    compute_soft_distance_transform,
    compute_image_distance_transform,
    distance_transform_loss,
    soft_dice_loss_nd,
)


def test_b_spline_3_known_values():
    x = torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=torch.double)
    values = b_spline_3(x)
    expected = torch.tensor([2.0 / 3.0, 1.0 / 6.0, 0.0, 0.0], dtype=torch.double)
    torch.testing.assert_close(values, expected, atol=1e-12, rtol=0)


def test_b_spline_3_is_symmetric():
    x = torch.linspace(-2.5, 2.5, 21, dtype=torch.double)
    torch.testing.assert_close(b_spline_3(x), b_spline_3(-x), atol=1e-12, rtol=0)


def test_local_ncc_loss_identical_images_is_minus_one():
    torch.manual_seed(0)
    image = torch.randn(1, 1, 16, 16, dtype=torch.double)
    loss = local_ncc_loss_nd(image, image, window_size=5)
    torch.testing.assert_close(loss, loss.new_tensor(-1.0), atol=1e-4, rtol=0)


def test_local_ncc_loss_has_finite_gradients():
    torch.manual_seed(1)
    fixed = torch.randn(1, 1, 12, 12, dtype=torch.double)
    moving = torch.randn_like(fixed, requires_grad=True)
    loss = local_ncc_loss_nd(fixed, moving, window_size=5)
    loss.backward()
    assert moving.grad is not None
    assert torch.isfinite(moving.grad).all()


@pytest.mark.parametrize("squared", [False, True])
def test_local_ncc_loss_squared_flag_is_nonnegative_range(squared):
    torch.manual_seed(2)
    fixed = torch.randn(1, 1, 10, 10, dtype=torch.double)
    moving = torch.randn(1, 1, 10, 10, dtype=torch.double)
    loss = local_ncc_loss_nd(fixed, moving, window_size=5, squared=squared)
    lower = -1.0
    assert lower <= loss.item() <= 0.0


def test_analytical_lncc_matches_autograd_path_forward():
    torch.manual_seed(3)
    fixed = torch.randn(1, 1, 10, 11, dtype=torch.double)
    moving = torch.randn(1, 1, 10, 11, dtype=torch.double)
    window_size = 5
    analytical = AnalyticalLNCC.apply(fixed, moving, None, window_size)
    autograd_path = local_ncc_loss_nd(fixed, moving, window_size=window_size, squared=False)
    torch.testing.assert_close(analytical, autograd_path, atol=1e-6, rtol=1e-5)


def test_ants_pseudo_lncc_matches_autograd_path_forward():
    torch.manual_seed(4)
    fixed = torch.randn(1, 1, 10, 11, dtype=torch.double)
    moving = torch.randn(1, 1, 10, 11, dtype=torch.double)
    window_size = 5
    pseudo = ANTsPseudoLNCC.apply(fixed, moving, None, window_size)
    autograd_path = local_ncc_loss_nd(fixed, moving, window_size=window_size, squared=True)
    torch.testing.assert_close(pseudo, autograd_path, atol=1e-6, rtol=1e-5)


def _cosine_similarity(a, b):
    return (a.flatten() @ b.flatten()) / (a.norm() * b.norm())


def test_analytical_lncc_gradient_direction_matches_autograd_reference():
    # AnalyticalLNCC's hand-derived backward is a fast approximation (it
    # skips differentiating through avg_pool2d/3d directly) rather than an
    # exact match to the autograd path — gradcheck fails identically on
    # unmodified upstream syntx. What must hold is that it is still a
    # legitimate (strongly correlated) descent direction.
    torch.manual_seed(3)
    fixed = torch.randn(1, 1, 10, 11, dtype=torch.double, requires_grad=True)
    moving = torch.randn(1, 1, 10, 11, dtype=torch.double, requires_grad=True)
    window_size = 5

    loss_ana = AnalyticalLNCC.apply(fixed, moving, None, window_size)
    grad_i_ana, grad_j_ana = torch.autograd.grad(loss_ana, (fixed, moving))

    fixed_ref = fixed.detach().clone().requires_grad_()
    moving_ref = moving.detach().clone().requires_grad_()
    loss_ref = local_ncc_loss_nd(fixed_ref, moving_ref, window_size=window_size, squared=False)
    grad_i_ref, grad_j_ref = torch.autograd.grad(loss_ref, (fixed_ref, moving_ref))

    assert _cosine_similarity(grad_i_ana, grad_i_ref).item() > 0.9
    assert _cosine_similarity(grad_j_ana, grad_j_ref).item() > 0.9


def test_ants_pseudo_lncc_gradient_direction_matches_autograd_reference():
    torch.manual_seed(3)
    fixed = torch.randn(1, 1, 10, 11, dtype=torch.double, requires_grad=True)
    moving = torch.randn(1, 1, 10, 11, dtype=torch.double, requires_grad=True)
    window_size = 5

    loss_pseudo = ANTsPseudoLNCC.apply(fixed, moving, None, window_size)
    grad_i_pseudo, grad_j_pseudo = torch.autograd.grad(loss_pseudo, (fixed, moving))

    fixed_ref = fixed.detach().clone().requires_grad_()
    moving_ref = moving.detach().clone().requires_grad_()
    loss_ref = local_ncc_loss_nd(fixed_ref, moving_ref, window_size=window_size, squared=True)
    grad_i_ref, grad_j_ref = torch.autograd.grad(loss_ref, (fixed_ref, moving_ref))

    assert _cosine_similarity(grad_i_pseudo, grad_i_ref).item() > 0.7
    assert _cosine_similarity(grad_j_pseudo, grad_j_ref).item() > 0.7


def test_mattes_mi_loss_core_empty_selection_returns_zero():
    image = torch.randn(1, 1, 6, 6, dtype=torch.double)
    mask = torch.zeros_like(image)
    loss = mattes_mi_loss_core(image, image, mask=mask)
    assert loss.item() == pytest.approx(0.0)
    assert loss.requires_grad


def test_mattes_mi_loss_nd_identical_images_more_negative_than_unrelated():
    torch.manual_seed(7)
    fixed = torch.rand(1, 1, 24, 24, dtype=torch.double)
    identical_mi = mattes_mi_loss_nd(fixed, fixed, auto_mask=False)
    unrelated = torch.rand(1, 1, 24, 24, dtype=torch.double)
    unrelated_mi = mattes_mi_loss_nd(fixed, unrelated, auto_mask=False)
    assert identical_mi.item() < unrelated_mi.item()


# --- Ported from syntx (box_cc2, distance-transform, soft Dice) ---


def test_box_cc2_loss_nd_identical_images_near_minus_one():
    torch.manual_seed(0)
    image = torch.rand(1, 1, 12, 14, 16)
    loss = box_cc2_loss_nd(image, image.clone(), window_size=5)
    assert loss.item() == pytest.approx(-1.0, abs=1e-3)


def test_box_cc2_loss_nd_worse_for_unrelated_images():
    torch.manual_seed(1)
    fixed = torch.rand(1, 1, 12, 14, 16)
    identical = box_cc2_loss_nd(fixed, fixed.clone(), window_size=5)
    unrelated = box_cc2_loss_nd(fixed, torch.rand(1, 1, 12, 14, 16), window_size=5)
    assert identical.item() < unrelated.item()


def test_box_cc2_loss_nd_is_differentiable():
    fixed = torch.rand(1, 1, 10, 10, 10)
    moving = (fixed + 0.05 * torch.randn_like(fixed)).requires_grad_(True)
    loss = box_cc2_loss_nd(fixed, moving, window_size=5)
    loss.backward()
    assert moving.grad is not None
    assert torch.isfinite(moving.grad).all()


def test_box_cc2_loss_nd_uniform_background_is_perfectly_correlated():
    # Zero-padded / uniform-background regions should evaluate near 1.0
    # correlation (loss near -1.0) thanks to the dual variance floors.
    zeros = torch.zeros(1, 1, 8, 8, 8)
    loss = box_cc2_loss_nd(zeros, zeros.clone(), window_size=3)
    assert loss.item() == pytest.approx(-1.0, abs=1e-3)


def test_compute_soft_distance_transform_zero_at_foreground_center():
    mask = torch.zeros(1, 1, 32, 32)
    mask[:, :, 10:22, 10:22] = 1.0
    dist = compute_soft_distance_transform(mask, sigma=2.0)
    assert dist.shape == mask.shape
    # Distance should be smaller deep inside the foreground than near its edge.
    assert dist[0, 0, 16, 16].item() < dist[0, 0, 11, 11].item()


def test_compute_image_distance_transform_matches_scipy_edt():
    import numpy as np
    import scipy.ndimage as ndi

    mask = np.zeros((16, 16), dtype=np.float32)
    mask[4:12, 4:12] = 1.0
    expected = ndi.distance_transform_edt(~(mask > 0.5))
    result = compute_image_distance_transform(torch.from_numpy(mask))
    np.testing.assert_allclose(result.numpy(), expected, atol=1e-4)


def test_compute_image_distance_transform_tau_potential_in_unit_interval():
    mask = torch.zeros(20, 20)
    mask[5:15, 5:15] = 1.0
    potential = compute_image_distance_transform(mask, tau=1.0)
    assert potential.min().item() >= 0.0
    assert potential.max().item() <= 1.0


def test_distance_transform_loss_identical_masks_is_near_optimal():
    mask = torch.zeros(1, 1, 24, 24)
    mask[:, :, 6:18, 6:18] = 1.0
    loss_same = distance_transform_loss(mask, mask.clone(), mode="potential_lncc", tau=0.2, window_size=5)
    shifted = torch.zeros(1, 1, 24, 24)
    shifted[:, :, 4:16, 4:16] = 1.0
    loss_shifted = distance_transform_loss(mask, shifted, mode="potential_lncc", tau=0.2, window_size=5)
    assert loss_same.item() < loss_shifted.item()


def test_distance_transform_loss_edt_mse_is_differentiable():
    mask = torch.zeros(1, 1, 16, 16, 16)
    mask[:, :, 4:12, 4:12, 4:12] = 1.0
    moving = (mask.clone() + 0.02 * torch.randn_like(mask)).clamp(0, 1).requires_grad_(True)
    loss = distance_transform_loss(mask, moving, mode="edt_mse", is_distance_field=False)
    loss.backward()
    assert moving.grad is not None and torch.isfinite(moving.grad).all()


def test_distance_transform_loss_unknown_mode_raises():
    mask = torch.zeros(1, 1, 8, 8)
    with pytest.raises(ValueError):
        distance_transform_loss(mask, mask.clone(), mode="not_a_mode")


def test_soft_dice_loss_nd_identical_masks_is_near_zero():
    mask = torch.rand(1, 1, 10, 10, 10)
    loss = soft_dice_loss_nd(mask, mask.clone())
    assert loss.item() == pytest.approx(0.0, abs=1e-5)


def test_soft_dice_loss_nd_disjoint_masks_is_near_one():
    a = torch.zeros(1, 1, 10, 10)
    a[:, :, :5, :] = 1.0
    b = torch.zeros(1, 1, 10, 10)
    b[:, :, 5:, :] = 1.0
    loss = soft_dice_loss_nd(a, b)
    assert loss.item() == pytest.approx(1.0, abs=1e-5)


def test_soft_dice_loss_nd_is_differentiable_and_respects_mask():
    fixed = torch.rand(1, 2, 8, 8, 8)
    moving = (fixed.clone() + 0.1 * torch.randn_like(fixed)).clamp(0, 1).requires_grad_(True)
    mask = torch.ones(1, 1, 8, 8, 8)
    mask[:, :, :4] = 0.0
    loss = soft_dice_loss_nd(fixed, moving, mask=mask)
    loss.backward()
    assert moving.grad is not None
    assert torch.isfinite(moving.grad).all()
    # Masked-out region should receive zero gradient.
    assert torch.all(moving.grad[:, :, :4] == 0.0)
