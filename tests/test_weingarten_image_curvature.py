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
    vol = np.where(r < 10.0, 1.0 - (r / 10.0) ** 2, 0.0).astype(np.float32)
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


def test_weingarten_torch_3d_orientations_and_spacings():
    """
    Verify 3D curvature computation across different orientations (axial, coronal,
    sagittal flips, oblique rotation), anisotropic voxel spacings, and non-zero origins.
    Confirms exact geometry and metadata preservation (origin, spacing, direction).
    """
    x, y, z = np.ogrid[-15:16, -15:16, -15:16]
    r = np.sqrt(x**2 + y**2 + z**2)
    vol = np.where(r < 10.0, 1.0 - (r / 10.0) ** 2, 0.0).astype(np.float32)
    vol = ndimage.gaussian_filter(vol, sigma=1.0)

    orientations_3d = [
        ("Identity", np.eye(3)),
        ("LPS/RAI flip", np.diag([-1.0, -1.0, 1.0])),
        ("Axis permutation", np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float32)),
        (
            "Oblique 45-deg rotation",
            np.array(
                [
                    [np.cos(np.pi / 4), -np.sin(np.pi / 4), 0.0],
                    [np.sin(np.pi / 4), np.cos(np.pi / 4), 0.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            ),
        ),
    ]

    spacings_3d = [
        (1.0, 1.0, 1.0),
        (0.8, 1.2, 1.5),
        (1.5, 0.5, 2.0),
        (0.7, 1.3, 0.9),
    ]

    for o_name, d in orientations_3d:
        for sp in spacings_3d:
            origin = (12.3, -45.6, 78.9)
            img = ants.from_numpy(vol, origin=origin, spacing=sp, direction=d)

            c_torch = weingarten_image_curvature(img, sigma=1.5, opt="mean")
            c_py = ants_weingarten_image_curvature(img, sigma=1.5, opt="mean")

            # Metadata preservation
            assert c_torch.origin == img.origin
            assert c_torch.spacing == img.spacing
            assert np.allclose(c_torch.direction, img.direction)

            # Mathematical parity
            a_t = c_torch.numpy()
            a_p = c_py.numpy()
            mask = (a_t != 0.0) & (vol > 0.01)

            diff = np.max(np.abs(a_t[mask] - a_p[mask]))
            corr = np.corrcoef(a_t[mask], a_p[mask])[0, 1]

            assert corr > 0.999, f"Failed correlation for {o_name}, sp={sp}: {corr}"
            assert diff < 0.01, f"Failed max_diff for {o_name}, sp={sp}: {diff}"


def test_weingarten_torch_options_anisotropic_oriented():
    """
    Verify mean, Gaussian, and characterize curvature options on anisotropic
    volumes with non-identity direction matrices and non-zero origins.
    """
    x, y, z = np.ogrid[-15:16, -15:16, -15:16]
    r = np.sqrt(x**2 + y**2 + z**2)
    vol = np.where(r < 10.0, 1.0 - (r / 10.0) ** 2, 0.0).astype(np.float32)
    vol = ndimage.gaussian_filter(vol, sigma=1.0)

    dir_mat = np.diag([-1.0, -1.0, 1.0])
    img = ants.from_numpy(vol, origin=(-10.0, 20.0, 30.0), spacing=(0.8, 1.2, 1.5), direction=dir_mat)

    for opt in ["mean", "gaussian", "characterize"]:
        ct = weingarten_image_curvature(img, sigma=2.0, opt=opt)
        cp = ants_weingarten_image_curvature(img, sigma=2.0, opt=opt)

        assert ct.origin == img.origin
        assert ct.spacing == img.spacing
        assert np.allclose(ct.direction, img.direction)

        at = ct.numpy()
        ap = cp.numpy()
        mask = (at != 0.0) & (vol > 0.01)

        if opt == "characterize":
            mismatches = np.sum(at[mask] != ap[mask])
            assert mismatches == 0
        else:
            diff = np.max(np.abs(at[mask] - ap[mask]))
            corr = np.corrcoef(at[mask], ap[mask])[0, 1]
            assert diff < 0.01
            assert corr > 0.999


def test_weingarten_torch_2d_orientations_and_spacings():
    """
    Verify 2D curvature computation across non-identity directions,
    anisotropic pixel spacings, and non-zero origins.
    """
    x, y = np.ogrid[-15:16, -15:16]
    r = np.sqrt(x**2 + y**2)
    vol2d = np.where(r < 10.0, 1.0 - (r / 10.0) ** 2, 0.0).astype(np.float32)
    vol2d = ndimage.gaussian_filter(vol2d, sigma=1.0)

    orientations_2d = [
        ("Identity", np.eye(2)),
        ("Flip X", np.diag([-1.0, 1.0])),
        ("Transpose", np.array([[0, 1], [1, 0]], dtype=np.float32)),
    ]

    spacings_2d = [
        (1.0, 1.0),
        (0.7, 1.3),
        (1.2, 0.8),
    ]

    for o_name, d2 in orientations_2d:
        for sp2 in spacings_2d:
            img2d = ants.from_numpy(vol2d, origin=(25.0, -35.0), spacing=sp2, direction=d2)
            ct2 = weingarten_image_curvature(img2d, sigma=1.5, opt="mean")
            cp2 = ants_weingarten_image_curvature(img2d, sigma=1.5, opt="mean")

            assert ct2.origin == img2d.origin
            assert ct2.spacing == img2d.spacing
            assert np.allclose(ct2.direction, img2d.direction)

            at2 = ct2.numpy()
            ap2 = cp2.numpy()
            m2 = (at2 != 0.0) & (vol2d > 0.01)
            corr2 = np.corrcoef(at2[m2], ap2[m2])[0, 1]
            assert corr2 > 0.99


def test_weingarten_torch_physical_scaling_invariance():
    """
    Verify that doubling physical spacing cuts mean curvature by exactly 2x,
    adhering to fundamental differential geometric physical scaling (kappa ~ 1/L).
    """
    x, y = np.ogrid[-15:16, -15:16]
    r = np.sqrt(x**2 + y**2)
    vol2d = np.where(r < 10.0, 1.0 - (r / 10.0) ** 2, 0.0).astype(np.float32)
    vol2d = ndimage.gaussian_filter(vol2d, sigma=1.0)

    img1 = ants.from_numpy(vol2d, spacing=(1.0, 1.0))
    img2 = ants.from_numpy(vol2d, spacing=(2.0, 2.0))

    c1 = weingarten_image_curvature(img1, sigma=1.5, opt="mean").numpy()
    c2 = weingarten_image_curvature(img2, sigma=3.0, opt="mean").numpy()

    peak1 = c1[15, 15]
    peak2 = c2[15, 15]
    ratio = peak1 / peak2

    assert np.isclose(ratio, 2.0, atol=1e-3)
