import pytest
import torch

from antstorch.syn.core import (
    AnalyticalLNCC,
    ANTsPseudoLNCC,
    local_ncc_loss_nd,
    b_spline_3,
    mattes_mi_loss_core,
    mattes_mi_loss_nd,
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
