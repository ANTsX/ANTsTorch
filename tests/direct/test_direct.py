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


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_direct_mps_without_native_3d_sampling(monkeypatch):
    monkeypatch.setenv("ANTSTORCH_MPS_GRID_SAMPLE", "native")
    from antstorch import _torch_compat

    original = _torch_compat.F.grid_sample

    def missing_mps_kernel(input, grid, **kwargs):
        if input.device.type == "mps" and input.ndim == 5:
            raise NotImplementedError("aten::grid_sampler_3d is unavailable")
        return original(input, grid, **kwargs)

    monkeypatch.setattr(_torch_compat.F, "grid_sample", missing_mps_kernel)
    images = _synthetic_tensors()
    domain = ImageDomain((12, 12, 12))
    options = dict(iterations=2, integration_points=2, inverse_iterations=2)
    expected = direct_cortical_thickness(*images, domain, **options)
    with pytest.warns(RuntimeWarning, match="interpolation on CPU"):
        actual = direct_cortical_thickness(
            *(image.to("mps") for image in images), domain, **options
        )
    assert actual.thickness.device.type == "mps"
    assert torch.isfinite(actual.thickness).all()
    torch.testing.assert_close(actual.thickness.cpu(), expected.thickness, atol=1e-4, rtol=1e-3)


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


@pytest.mark.parametrize("dimension", [2, 3])
def test_binary_contour_includes_diagonal_neighbors(dimension):
    mask = torch.ones((1, 1) + (5,) * dimension)
    mask[(0, 0) + (1,) * dimension] = 0
    contour = binary_contour(mask)
    assert contour[(0, 0) + (2,) * dimension] == 1
    assert contour[(0, 0) + (4,) * dimension] == 0
    assert not binary_contour(torch.ones_like(mask)).any()


@pytest.mark.parametrize("dimension", [2, 3])
def test_binary_contour_matches_full_neighborhood_erosion(dimension):
    from scipy.ndimage import binary_erosion

    mask = np.random.default_rng(42).random((7,) * dimension) > 0.1
    expected = mask & ~binary_erosion(
        mask, structure=np.ones((3,) * dimension), border_value=1
    )
    actual = binary_contour(torch.from_numpy(mask.astype(np.float32))[None, None])
    np.testing.assert_array_equal(actual.numpy()[0, 0], expected)


@pytest.mark.parametrize('variance', [0.25, 1.0, 2.0])
def test_scalar_smoothing_matches_ants(variance):
    import ants
    from antstorch.direct.forces import gaussian_scalar
    array = np.random.default_rng(10).random((11, 13, 15)).astype(np.float32)
    expected = ants.smooth_image(ants.from_numpy(array), variance ** 0.5,
                                sigma_in_physical_coordinates=False).numpy()
    tensor = torch.from_numpy(array.transpose(2, 1, 0).copy())[None, None]
    actual = gaussian_scalar(tensor, variance ** 0.5, maximum_error=0.01)
    np.testing.assert_allclose(actual.numpy()[0, 0].transpose(2, 1, 0), expected,
                               atol=3e-7, rtol=1e-6)


@pytest.mark.parametrize('spacing', [(1.0, 1.0), (0.7, 1.4)])
def test_probability_gradient_direction_against_smoothed_sinusoid(spacing):
    import math
    from antstorch.direct.forces import normalized_probability_gradient
    y, x = torch.meshgrid(torch.arange(96) * spacing[1],
                          torch.arange(96) * spacing[0], indexing='ij')
    kx, ky = 0.12, 0.20
    probability = (torch.cos(kx * x) + 0.7 * torch.sin(ky * y))[None, None]
    # Continuous Gaussian smoothing of each Fourier mode has an exact response.
    expected = torch.stack((-kx * math.exp(-kx*kx/2) * torch.sin(kx*x),
                            0.7 * ky * math.exp(-ky*ky/2) * torch.cos(ky*y)))[None]
    expected /= torch.linalg.vector_norm(expected, dim=1, keepdim=True)
    actual = normalized_probability_gradient(probability, sigma=1.0, spacing=spacing)
    # Exclude padding effects; measure direction because DiReCT normalizes it.
    cosine = (expected * actual).sum(1)[:, 12:-12, 12:-12].clamp(-1, 1)
    angle = torch.rad2deg(torch.acos(cosine))
    assert angle.max() < 0.5


@pytest.mark.parametrize('amplitude', [0.05, 0.3])
def test_direct_inverse_matches_ants_stopping(amplitude):
    import ants
    from antstorch.direct.core import _invert
    n = 17
    x, y, z = np.meshgrid(*[np.arange(n)] * 3, indexing='ij')
    field = np.zeros((n, n, n, 3), np.float32)
    field[..., 0] = amplitude * np.exp(-((x-8)**2 + (y-8)**2 + (z-8)**2) / 8)
    expected = ants.invert_displacement_field(
        ants.from_numpy(field, has_components=True),
        ants.from_numpy(np.zeros_like(field), has_components=True),
        maximum_number_of_iterations=20,
        max_error_tolerance_threshold=0.1,
        mean_error_tolerance_threshold=0.001,
    ).numpy()
    tensor = torch.from_numpy(field.transpose(3, 2, 1, 0).copy())[None]
    actual = _invert(tensor, torch.zeros_like(tensor), ImageDomain((n, n, n)), 20)
    np.testing.assert_allclose(actual.numpy()[0].transpose(3, 2, 1, 0), expected,
                               atol=1e-7, rtol=1e-5)
