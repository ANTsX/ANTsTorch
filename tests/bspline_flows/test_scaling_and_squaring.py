import pytest
import torch

from antstorch.bspline_flows import ImageDomain, compose_displacements, scaling_and_squaring


@pytest.mark.parametrize("size", [(8, 7), (7, 6, 5)])
def test_zero_and_constant_velocity(size):
    domain = ImageDomain(size, spacing=(1.2,) * len(size))
    zero = torch.zeros((2, len(size)) + domain.torch_size, dtype=torch.double)
    torch.testing.assert_close(scaling_and_squaring(zero, domain, 5), zero)
    constant = zero.clone()
    constant[:, 0] = 0.7
    constant[:, 1] = -0.25
    torch.testing.assert_close(scaling_and_squaring(constant, domain, 6), constant, rtol=0, atol=2e-14)


def _smooth_velocity(domain, scale=0.08):
    y = torch.linspace(-1, 1, domain.size[1], dtype=torch.double)[:, None]
    x = torch.linspace(-1, 1, domain.size[0], dtype=torch.double)[None, :]
    return torch.stack((scale * torch.sin(y).expand_as(x + y), scale * torch.sin(x).expand_as(x + y)), dim=0)[None]


def test_exp_v_composed_with_exp_minus_v_is_near_identity():
    domain = ImageDomain((25, 23))
    velocity = _smooth_velocity(domain)
    forward = scaling_and_squaring(velocity, domain, 7)
    inverse = scaling_and_squaring(-velocity, domain, 7)
    residual = compose_displacements(forward, inverse, domain)[..., 2:-2, 2:-2]
    assert residual.abs().max().item() < 2e-4


def test_convergence_with_squaring_steps():
    domain = ImageDomain((25, 23))
    velocity = _smooth_velocity(domain, 0.3)
    values = [scaling_and_squaring(velocity, domain, steps) for steps in (3, 4, 5, 7)]
    coarse = (values[1] - values[0]).abs().mean()
    finer = (values[2] - values[1]).abs().mean()
    finest = (values[3] - values[2]).abs().mean()
    assert finer < coarse
    assert finest < finer
