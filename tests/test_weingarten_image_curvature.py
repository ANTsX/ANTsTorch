import numpy as np
import pytest
import ants
import scipy.ndimage as ndimage
import torch

from antstorch.utilities.weingarten_image_curvature import weingarten_image_curvature
from ants.ops.weingarten_image_curvature import weingarten_image_curvature as ants_weingarten_image_curvature


@pytest.fixture
def test_sphere_image():
    # Construct a 3D sphere to compute curvature
    x, y, z = np.ogrid[-15:16, -15:16, -15:16]
    r = np.sqrt(x**2 + y**2 + z**2)
    # A smooth blob
    vol = np.where(r < 10.0, 1.0 - (r / 10.0) ** 2, 0.0).astype(np.float32)
    # Smooth a bit more to have stable derivatives
    vol = ndimage.gaussian_filter(vol, sigma=1.0)
    return ants.from_numpy(vol, origin=(0, 0, 0), spacing=(1.0, 1.0, 1.0))


@pytest.fixture
def test_sphere_anisotropic():
    # Construct a 3D sphere with anisotropic spacing
    x, y, z = np.ogrid[-15:16, -15:16, -15:16]
    r = np.sqrt(x**2 + y**2 + z**2)
    vol = np.where(r < 10.0, 1.0 - (r / 10.0) ** 2, 0.0).astype(np.float32)
    vol = ndimage.gaussian_filter(vol, sigma=1.0)
    return ants.from_numpy(vol, origin=(0, 0, 0), spacing=(1.2, 1.5, 1.8))


def test_weingarten_torch_vs_py_isotropic(test_sphere_image):
    image = test_sphere_image

    # 1. Mean curvature
    curv_torch = weingarten_image_curvature(image, sigma=1.5, opt="mean")
    curv_py = ants_weingarten_image_curvature(image, sigma=1.5, opt="mean")

    arr_torch = curv_torch.numpy()
    arr_py = curv_py.numpy()

    # Compare only in the non-zero region (excluding margins) where intensity is significant
    mask = (arr_torch != 0.0) & (image.numpy() > 0.01)
    assert np.any(mask)

    max_diff = np.max(np.abs(arr_torch[mask] - arr_py[mask]))
    corr = np.corrcoef(arr_torch[mask], arr_py[mask])[0, 1]

    assert max_diff < 1e-3
    assert corr > 0.999


def test_weingarten_torch_vs_py_anisotropic(test_sphere_anisotropic):
    image = test_sphere_anisotropic

    # 2. Gaussian curvature with anisotropic spacing
    curv_torch = weingarten_image_curvature(image, sigma=2.0, opt="gaussian")
    curv_py = ants_weingarten_image_curvature(image, sigma=2.0, opt="gaussian")

    arr_torch = curv_torch.numpy()
    arr_py = curv_py.numpy()

    mask = (arr_torch != 0.0) & (image.numpy() > 0.01)
    assert np.any(mask)

    max_diff = np.max(np.abs(arr_torch[mask] - arr_py[mask]))
    corr = np.corrcoef(arr_torch[mask], arr_py[mask])[0, 1]

    assert max_diff < 0.01
    assert corr > 0.999


def test_weingarten_torch_options(test_sphere_image):
    image = test_sphere_image

    # Check that all options run and match Python/ITK reference
    for opt in ["mean", "gaussian", "characterize"]:
        curv_torch = weingarten_image_curvature(image, sigma=1.5, opt=opt)
        curv_py = ants_weingarten_image_curvature(image, sigma=1.5, opt=opt)

        arr_torch = curv_torch.numpy()
        arr_py = curv_py.numpy()

        mask = (arr_torch != 0.0) & (image.numpy() > 0.01)

        if opt == "characterize":
            # Classification must match exactly
            mismatches = np.sum(arr_torch[mask] != arr_py[mask])
            assert mismatches == 0
        else:
            max_diff = np.max(np.abs(arr_torch[mask] - arr_py[mask]))
            corr = np.corrcoef(arr_torch[mask], arr_py[mask])[0, 1]
            assert max_diff < 0.01
            assert corr > 0.999


def test_weingarten_torch_masking(test_sphere_image):
    image = test_sphere_image
    # Define a binary mask restricting to center
    mask_arr = (image.numpy() > 0.1).astype(np.float32)
    mask = ants.from_numpy(mask_arr, origin=image.origin, spacing=image.spacing)

    curv_torch_masked = weingarten_image_curvature(image, sigma=1.5, opt="mean", mask=mask)
    curv_torch_unmasked = weingarten_image_curvature(image, sigma=1.5, opt="mean")

    arr_masked = curv_torch_masked.numpy()
    arr_unmasked = curv_torch_unmasked.numpy()

    # Inside the mask, they should match
    inner_mask = (mask_arr > 0.0) & (arr_unmasked != 0.0)
    assert np.allclose(arr_masked[inner_mask], arr_unmasked[inner_mask], atol=1e-5)

    # Outside the mask, output should be completely zero
    outer_mask = mask_arr == 0.0
    assert np.all(arr_masked[outer_mask] == 0.0)


def test_weingarten_torch_2d():
    # Construct a 2D circle
    x, y = np.ogrid[-15:16, -15:16]
    r = np.sqrt(x**2 + y**2)
    vol = np.where(r < 10.0, 1.0 - (r / 10.0) ** 2, 0.0).astype(np.float32)
    vol = ndimage.gaussian_filter(vol, sigma=1.0)
    image = ants.from_numpy(vol, origin=(0, 0), spacing=(1.0, 1.0))

    curv_torch = weingarten_image_curvature(image, sigma=1.5, opt="mean")
    curv_py = ants_weingarten_image_curvature(image, sigma=1.5, opt="mean")

    arr_torch = curv_torch.numpy()
    arr_py = curv_py.numpy()

    mask = (arr_torch != 0.0) & (image.numpy() > 0.01)
    assert np.any(mask)

    max_diff = np.max(np.abs(arr_torch[mask] - arr_py[mask]))
    corr = np.corrcoef(arr_torch[mask], arr_py[mask])[0, 1]

    assert max_diff < 0.01
    assert corr > 0.999


def test_weingarten_torch_devices(test_sphere_image):
    # Verify execution on CPU
    curv_cpu = weingarten_image_curvature(test_sphere_image, sigma=1.5, opt="mean", device="cpu")
    assert curv_cpu is not None
    assert curv_cpu.shape == test_sphere_image.shape

    # If GPU or MPS is available, verify on accelerator as well
    if torch.cuda.is_available():
        curv_cuda = weingarten_image_curvature(test_sphere_image, sigma=1.5, opt="mean", device="cuda")
        assert np.allclose(curv_cpu.numpy(), curv_cuda.numpy(), atol=1e-4)
    elif torch.backends.mps.is_available():
        curv_mps = weingarten_image_curvature(test_sphere_image, sigma=1.5, opt="mean", device="mps")
        assert np.allclose(curv_cpu.numpy(), curv_mps.numpy(), atol=1e-4)
