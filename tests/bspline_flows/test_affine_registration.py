import pytest
import torch

from antstorch.bspline_flows import (
    BSplineDomain,
    affine_displacement_field,
    affine_registration,
    warp_image,
)
from antstorch.syn.core.affine import get_rotation_matrix


def _blob(domain, dtype=torch.float32, center=(0.0, 0.0)):
    axes = [torch.linspace(-1, 1, size, dtype=dtype) for size in domain.torch_size]
    coordinates = torch.meshgrid(*axes, indexing="ij")
    # coordinates are (y, x); center is given in the same (y, x) order.
    squared = sum((axis - c).square() for axis, c in zip(coordinates, reversed(center)))
    return torch.exp(-8 * squared)[None, None]


def _asymmetric_blob_3d(domain, dtype=torch.float32):
    axes = [torch.linspace(-1, 1, size, dtype=dtype) for size in domain.torch_size]
    z, y, x = torch.meshgrid(*axes, indexing="ij")
    return (torch.exp(-8 * ((z - 0.3).square() + y.square() + x.square())) + 0.3 * torch.relu(x))[None, None]


@pytest.mark.parametrize("size", [(9, 8), (7, 6, 5)])
def test_affine_registration_shapes_and_finiteness(size):
    domain = BSplineDomain(size)
    dimension = len(size)
    image = _blob(domain) if dimension == 2 else _asymmetric_blob_3d(domain)
    result = affine_registration(
        image,
        image,
        domain,
        iterations=(1, 1),
        shrink_factors=(2, 1),
        smoothing_sigmas=(0.5, 0.0),
    )
    assert result["matrix"].shape == (1, dimension, dimension)
    assert result["translation"].shape == (1, dimension)
    assert result["warpedmovout"].shape == image.shape
    assert result["fwdtransforms"].shape == (1, dimension) + domain.torch_size
    assert result["invtransforms"].shape == (1, dimension) + domain.torch_size
    assert torch.isfinite(result["matrix"]).all()
    assert torch.isfinite(result["translation"]).all()
    assert torch.isfinite(result["warpedmovout"]).all()
    assert not result["matrix"].requires_grad


def test_affine_registration_recovers_a_known_translation():
    torch.manual_seed(0)
    domain = BSplineDomain((24, 22), spacing=(1.0, 1.0))
    moving = _blob(domain)
    known_translation = torch.tensor([1.6, -1.1])
    field = affine_displacement_field(torch.eye(2), known_translation, domain, moving)
    fixed = warp_image(moving, field, domain, padding_mode="border")

    result = affine_registration(
        fixed,
        moving,
        domain,
        transform_type="Translation",
        iterations=(60, 60),
        learning_rate=0.1,
        shrink_factors=(2, 1),
        smoothing_sigmas=(1.0, 0.0),
        padding_mode="border",
    )
    torch.testing.assert_close(result["matrix"][0], torch.eye(2), atol=1e-4, rtol=0)
    torch.testing.assert_close(result["translation"][0], known_translation, atol=0.05, rtol=0)
    assert result["loss_history"][0][-1] < result["loss_history"][0][0]


def test_affine_registration_reduces_loss_for_a_known_affine():
    torch.manual_seed(1)
    domain = BSplineDomain((26, 24), spacing=(1.0, 1.0))
    moving = _blob(domain)
    matrix = torch.tensor([[1.08, 0.05], [-0.04, 0.94]])
    translation = torch.tensor([0.8, -0.6])
    field = affine_displacement_field(matrix, translation, domain, moving)
    fixed = warp_image(moving, field, domain, padding_mode="border")

    result = affine_registration(
        fixed,
        moving,
        domain,
        transform_type="Affine",
        iterations=(80, 80),
        learning_rate=0.05,
        shrink_factors=(2, 1),
        smoothing_sigmas=(1.0, 0.0),
        padding_mode="border",
    )
    assert result["loss_history"][-1][-1] < result["loss_history"][0][0]


def test_multi_start_helps_recover_from_a_flip_ambiguous_configuration():
    # A near-180-degree rotation is the textbook local-minimum trap for a
    # single-start gradient-based affine solver: the seed-rotation search is
    # exactly the mechanism meant to avoid it.
    torch.manual_seed(2)
    domain = BSplineDomain((20, 20), spacing=(1.0, 1.0))
    moving = torch.zeros(1, 1, *domain.torch_size)
    axes = [torch.linspace(-1, 1, s) for s in domain.torch_size]
    yy, xx = torch.meshgrid(*axes, indexing="ij")
    moving[0, 0] = torch.exp(-8 * ((yy - 0.4).square() + xx.square())) + 0.4 * torch.exp(
        -20 * ((yy + 0.1).square() + (xx - 0.5).square())
    )
    omega = torch.tensor([3.05])  # close to pi: near-180-degree rotation
    rotation = get_rotation_matrix(omega, 2)
    field = affine_displacement_field(rotation, torch.zeros(2), domain, moving)
    fixed = warp_image(moving, field, domain, padding_mode="border")

    common = dict(
        transform_type="Rigid",
        iterations=(50,),
        learning_rate=0.05,
        shrink_factors=(1,),
        center_of_mass_init=False,
        padding_mode="border",
    )
    with_multi_start = affine_registration(fixed, moving, domain, multi_start=True, **common)
    without_multi_start = affine_registration(fixed, moving, domain, multi_start=False, **common)
    assert with_multi_start["loss_history"][0][-1] <= without_multi_start["loss_history"][0][-1]


def test_affine_registration_matrix_and_translation_invert_consistently():
    torch.manual_seed(3)
    domain = BSplineDomain((16, 14))
    moving = _blob(domain)
    matrix = torch.tensor([[1.1, 0.02], [0.03, 0.9]])
    translation = torch.tensor([0.5, 0.2])
    field = affine_displacement_field(matrix, translation, domain, moving)
    fixed = warp_image(moving, field, domain, padding_mode="border")
    result = affine_registration(
        fixed,
        moving,
        domain,
        iterations=(1,),
        shrink_factors=(1,),
        padding_mode="border",
    )
    fitted_matrix = result["matrix"][0]
    inverse_matrix = torch.linalg.inv(fitted_matrix)
    forward_composed_with_field = affine_displacement_field(
        fitted_matrix, result["translation"][0], domain, moving
    )
    torch.testing.assert_close(forward_composed_with_field, result["fwdtransforms"], rtol=0, atol=1e-6)
    torch.testing.assert_close(
        inverse_matrix @ fitted_matrix, torch.eye(2), atol=1e-5, rtol=0
    )


@pytest.mark.parametrize(
    "kwargs,exception,match",
    [
        ({"transform_type": "Nonsense"}, ValueError, "transform_type"),
        ({"similarity": "mi"}, ValueError, "similarity"),
        ({"padding_mode": "wrap"}, ValueError, "padding_mode"),
    ],
)
def test_affine_registration_parameter_validation(kwargs, exception, match):
    domain = BSplineDomain((8, 7))
    image = _blob(domain)
    with pytest.raises(exception, match=match):
        affine_registration(image, image, domain, iterations=(0,), shrink_factors=(1,), **kwargs)


def test_affine_registration_rejects_incompatible_image_domain():
    domain = BSplineDomain((8, 7))
    with pytest.raises(ValueError, match="fixed tensor shape"):
        affine_registration(
            torch.zeros(1, 1, 6, 8), torch.zeros(1, 1, 7, 8), domain, iterations=(0,), shrink_factors=(1,)
        )


def test_affine_registration_batch_is_fit_independently():
    domain = BSplineDomain((16, 14))
    moving = torch.cat([_blob(domain, center=(0.0, 0.0)), _blob(domain, center=(0.3, -0.2))], dim=0)
    translations = torch.tensor([[0.8, 0.0], [-0.5, 0.6]])
    fixed_items = []
    for index in range(2):
        field = affine_displacement_field(torch.eye(2), translations[index], domain, moving[index : index + 1])
        fixed_items.append(warp_image(moving[index : index + 1], field, domain, padding_mode="border"))
    fixed = torch.cat(fixed_items, dim=0)

    result = affine_registration(
        fixed,
        moving,
        domain,
        transform_type="Translation",
        iterations=(40, 40),
        learning_rate=0.1,
        shrink_factors=(2, 1),
        smoothing_sigmas=(1.0, 0.0),
        padding_mode="border",
    )
    torch.testing.assert_close(result["translation"], translations, atol=0.08, rtol=0)
