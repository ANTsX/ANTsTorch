import pytest
import torch

from antstorch.syn.core import (
    get_cached_gaussian_kernel_1d,
    separable_gaussian_filter,
    apply_sobolev_green_operator,
    apply_dsti_green_operator,
    apply_dsti1_green_operator,
    get_boundary_mask,
)


def test_gaussian_kernel_1d_sums_to_one_and_is_symmetric():
    kernel = get_cached_gaussian_kernel_1d(2.0, device=torch.device('cpu'), dtype=torch.double)
    k = kernel.view(-1)
    assert k.sum().item() == pytest.approx(1.0, abs=1e-4)
    torch.testing.assert_close(k, k.flip(0), atol=1e-10, rtol=0)


def test_separable_gaussian_filter_zero_sigma_is_identity():
    field = torch.randn(1, 5, 5, 2, dtype=torch.double)
    filtered = separable_gaussian_filter(field, sigma=0.0)
    assert filtered is field


def test_separable_gaussian_filter_preserves_sum_for_interior_impulse():
    field = torch.zeros(1, 21, 21, 2, dtype=torch.float32)
    field[0, 10, 10, 0] = 1.0
    field[0, 10, 10, 1] = 2.0
    filtered = separable_gaussian_filter(field, sigma=1.5)
    assert filtered.shape == field.shape
    torch.testing.assert_close(filtered.sum(dim=(1, 2)), field.sum(dim=(1, 2)), atol=1e-3, rtol=0)


def test_sobolev_green_operator_zero_sigma_is_identity():
    field = torch.randn(1, 6, 6, 2, dtype=torch.double)
    filtered = apply_sobolev_green_operator(field, fluid_sigma=0.0)
    assert filtered is field


def test_sobolev_green_operator_preserves_constant_field():
    field = torch.full((1, 8, 8, 2), 3.0, dtype=torch.double)
    filtered = apply_sobolev_green_operator(field, fluid_sigma=3.0)
    torch.testing.assert_close(filtered, field, atol=1e-4, rtol=1e-4)


def test_dsti_green_operator_zero_sigma_is_identity():
    field = torch.randn(1, 6, 6, 2, dtype=torch.double)
    filtered = apply_dsti_green_operator(field, fluid_sigma=0.0)
    assert filtered is field


def test_dsti_green_operator_suppresses_boundary_more_than_sobolev():
    # The DST-I "Dirichlet-Shield" kernel enforces v = 0 at the *virtual*
    # boundary points just outside the domain (n = 0 and n = N + 1 in its
    # 1-indexed sine basis), not at the first/last physical grid samples
    # themselves — so it does not zero the discrete boundary rim exactly,
    # but it should suppress boundary-rim energy markedly more than the
    # periodic Sobolev kernel, which has no boundary condition at all.
    torch.manual_seed(0)
    field = torch.randn(1, 9, 9, 2, dtype=torch.double)
    dsti = apply_dsti_green_operator(field, fluid_sigma=3.0)
    sobolev = apply_sobolev_green_operator(field, fluid_sigma=3.0)

    def _boundary_rim_norm(x):
        return torch.cat([x[0, 0, :, :].flatten(), x[0, -1, :, :].flatten(),
                           x[0, :, 0, :].flatten(), x[0, :, -1, :].flatten()]).abs().max()

    assert _boundary_rim_norm(dsti).item() < _boundary_rim_norm(sobolev).item()


def test_dsti1_alias_matches_dsti():
    torch.manual_seed(1)
    field = torch.randn(1, 6, 6, 2, dtype=torch.double)
    a = apply_dsti_green_operator(field, fluid_sigma=2.0)
    b = apply_dsti1_green_operator(field, fluid_sigma=2.0)
    torch.testing.assert_close(a, b)


def test_get_boundary_mask_zero_at_rim_one_in_interior():
    mask = get_boundary_mask((5, 6), device=torch.device('cpu'), dtype=torch.double, rim_size=1)
    assert mask.shape == (1, 5, 6, 1)
    assert mask[0, 0, :, 0].sum().item() == 0
    assert mask[0, -1, :, 0].sum().item() == 0
    assert mask[0, :, 0, 0].sum().item() == 0
    assert mask[0, :, -1, 0].sum().item() == 0
    assert mask[0, 2, 3, 0].item() == 1.0
