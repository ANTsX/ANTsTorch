import pytest
import torch

from antstorch.bspline_flows import ants_neighborhood_correlation_loss


def _brute_force_2d(fixed, moving, radius):
    values = []
    epsilon = torch.finfo(fixed.dtype).eps
    radius_x, radius_y = radius
    for batch in range(fixed.shape[0]):
        for channel in range(fixed.shape[1]):
            for y in range(fixed.shape[2]):
                for x in range(fixed.shape[3]):
                    f = fixed[
                        batch,
                        channel,
                        max(0, y - radius_y) : y + radius_y + 1,
                        max(0, x - radius_x) : x + radius_x + 1,
                    ].flatten()
                    m = moving[
                        batch,
                        channel,
                        max(0, y - radius_y) : y + radius_y + 1,
                        max(0, x - radius_x) : x + radius_x + 1,
                    ].flatten()
                    fc = f - f.mean()
                    mc = m - m.mean()
                    denominator = fc.square().sum() * mc.square().sum()
                    cc = (
                        (fc * mc).sum().square() / denominator
                        if denominator.abs() > epsilon
                        else denominator.new_tensor(1.0)
                    )
                    values.append(cc)
    return -torch.stack(values).mean()


def test_ants_neighborhood_correlation_matches_brute_force_at_boundaries():
    torch.manual_seed(2)
    fixed = torch.randn(2, 2, 5, 6, dtype=torch.double)
    moving = torch.randn_like(fixed)
    actual = ants_neighborhood_correlation_loss(fixed, moving, radius=(2, 1))
    expected = _brute_force_2d(fixed, moving, radius=(2, 1))
    torch.testing.assert_close(actual, expected, rtol=2e-14, atol=2e-14)


@pytest.mark.parametrize("shape", [(1, 1, 7, 8), (1, 1, 5, 6, 7)])
def test_ants_neighborhood_correlation_identical_images_are_minus_one(shape):
    image = torch.randn(shape, dtype=torch.double)
    loss = ants_neighborhood_correlation_loss(image, image, radius=2)
    torch.testing.assert_close(loss, loss.new_tensor(-1.0), rtol=2e-14, atol=2e-14)


def test_ants_neighborhood_correlation_has_finite_gradients():
    fixed = torch.randn(1, 1, 8, 9, dtype=torch.double)
    moving = torch.randn_like(fixed, requires_grad=True)
    loss = ants_neighborhood_correlation_loss(fixed, moving, radius=2)
    loss.backward()
    assert moving.grad is not None
    assert torch.isfinite(moving.grad).all()
    assert moving.grad.abs().max() > 0


def test_ants_neighborhood_correlation_constant_background_has_finite_gradients():
    fixed = torch.zeros(1, 1, 32, 32, dtype=torch.double)
    fixed[:, :, 10:22, 9:21] = torch.randn(1, 1, 12, 12, dtype=torch.double)
    moving = torch.roll(fixed, 1, -1).requires_grad_()
    loss = ants_neighborhood_correlation_loss(fixed, moving, radius=2)
    loss.backward()
    assert torch.isfinite(loss)
    assert moving.grad is not None
    assert torch.isfinite(moving.grad).all()


@pytest.mark.parametrize("radius", [-1, (1,), (1.5, 2)])
def test_ants_neighborhood_correlation_validates_radius(radius):
    image = torch.randn(1, 1, 6, 7)
    with pytest.raises((TypeError, ValueError), match="radius"):
        ants_neighborhood_correlation_loss(image, image, radius)
