import numpy as np
import pytest
import torch

from antstorch.bspline_flows import ImageDomain
from antstorch.direct import direct_cortical_thickness, kelly_kapowski
from antstorch.direct.forces import binary_contour, direct_force


def _synthetic_tensors(size=12):
    axes = torch.meshgrid(*[torch.arange(size)] * 3, indexing="ij")
    radius = torch.sqrt(sum((axis - (size - 1) / 2) ** 2 for axis in axes))
    white = radius < 2.5
    gray = (radius >= 2.5) & (radius < 4.0)
    segmentation = torch.zeros((1, 1, size, size, size))
    segmentation[0, 0][gray] = 2
    segmentation[0, 0][white] = 3
    gray_probability = gray.float()[None, None] * 0.9
    white_probability = white.float()[None, None] * 0.9
    return segmentation, gray_probability, white_probability


def test_binary_contour_is_inner_boundary():
    mask = torch.zeros(1, 1, 7, 7)
    mask[:, :, 1:6, 1:6] = 1
    contour = binary_contour(mask)
    assert contour.sum().item() == 16
    assert contour[0, 0, 3, 3].item() == 0


def test_direct_force_is_restricted_to_gray_matter():
    wm = torch.ones(1, 1, 4, 4)
    gm = torch.full_like(wm, 0.5)
    mask = torch.zeros_like(wm)
    mask[:, :, 1:3, 1:3] = 1
    gradient = torch.ones(1, 2, 4, 4)
    force = direct_force(wm, gm, mask, gradient, 0.25)
    assert torch.count_nonzero(force[:, :, 0, :]) == 0
    torch.testing.assert_close(force[:, :, 1:3, 1:3], torch.full((1, 2, 2, 2), -0.0625))


@pytest.mark.parametrize("optimizer", ["direct", "reg_adam"])
def test_direct_core_returns_finite_thickness(optimizer):
    segmentation, gray, white = _synthetic_tensors()
    domain = ImageDomain((12, 12, 12))
    result = direct_cortical_thickness(
        segmentation, gray, white, domain,
        iterations=3, integration_points=2, inverse_iterations=2,
        optimizer=optimizer,
    )
    assert result.thickness.shape == segmentation.shape
    assert torch.isfinite(result.thickness).all()
    assert (result.thickness >= 0).all()
    assert torch.count_nonzero(result.thickness) > 0
    assert len(result.energy_history) == 3


def test_ant_image_bridge_preserves_output_geometry():
    ants = pytest.importorskip("ants")
    segmentation, gray, white = _synthetic_tensors(size=10)
    def image(tensor):
        return ants.from_numpy(
            tensor.numpy()[0, 0].transpose(2, 1, 0),
            spacing=(0.8, 0.9, 1.1),
            origin=(1.0, 2.0, 3.0),
        )
    seg_image, gray_image, white_image = image(segmentation), image(gray), image(white)
    result = kelly_kapowski(
        seg_image, gray_image, white_image,
        iterations=1, integration_points=1, device="cpu",
    )
    assert result.shape == seg_image.shape
    assert result.spacing == seg_image.spacing
    assert result.origin == seg_image.origin
    assert np.isfinite(result.numpy()).all()
