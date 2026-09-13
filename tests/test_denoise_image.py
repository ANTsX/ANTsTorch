import os
os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = "1"

import numpy as np
import pytest
import torch
import ants
import antstorch
from antstorch.utilities.denoise_image import denoise_image


@pytest.fixture
def synthetic_2d():
    np.random.seed(42)
    arr = np.random.uniform(10.0, 50.0, size=(16, 16)).astype(np.float32)
    return ants.from_numpy(arr, spacing=(1.0, 1.0))


@pytest.fixture
def synthetic_3d():
    np.random.seed(42)
    arr = np.random.uniform(10.0, 50.0, size=(8, 10, 12)).astype(np.float32)
    return ants.from_numpy(arr, spacing=(1.0, 1.0, 1.0))


@pytest.fixture
def r16_image():
    return ants.image_read(ants.get_ants_data("r16"))


def test_denoise_image_gaussian_2d_parity(synthetic_2d, r16_image):
    # Test on synthetic 2D image
    res_torch = denoise_image(synthetic_2d, p=1, r=2, noise_model="Gaussian", shrink_factor=1)
    res_ants = ants.denoise_image(synthetic_2d, p=1, r=2, noise_model="Gaussian", shrink_factor=1)

    corr = np.corrcoef(res_torch.numpy().ravel(), res_ants.numpy().ravel())[0, 1]
    max_diff = np.max(np.abs(res_torch.numpy() - res_ants.numpy()))
    assert corr > 0.999999
    assert max_diff < 1e-4

    # Test on real MRI slice (r16)
    res_torch_r16 = denoise_image(r16_image, p=1, r=2, noise_model="Gaussian", shrink_factor=1)
    res_ants_r16 = ants.denoise_image(r16_image, p=1, r=2, noise_model="Gaussian", shrink_factor=1)

    corr_r16 = np.corrcoef(res_torch_r16.numpy().ravel(), res_ants_r16.numpy().ravel())[0, 1]
    mean_diff_r16 = np.mean(np.abs(res_torch_r16.numpy() - res_ants_r16.numpy()))
    assert corr_r16 > 0.99999
    assert mean_diff_r16 < 0.05


def test_denoise_image_rician_2d_parity(r16_image):
    res_torch = denoise_image(r16_image, p=1, r=2, noise_model="Rician", shrink_factor=1)
    res_ants = ants.denoise_image(r16_image, p=1, r=2, noise_model="Rician", shrink_factor=1)

    corr = np.corrcoef(res_torch.numpy().ravel(), res_ants.numpy().ravel())[0, 1]
    mean_diff = np.mean(np.abs(res_torch.numpy() - res_ants.numpy()))
    assert corr > 0.9999
    assert mean_diff < 0.1


def test_denoise_image_3d_gaussian_parity(synthetic_3d):
    res_torch = denoise_image(synthetic_3d, p=1, r=1, noise_model="Gaussian", shrink_factor=1)
    res_ants = ants.denoise_image(synthetic_3d, p=1, r=1, noise_model="Gaussian", shrink_factor=1)

    corr = np.corrcoef(res_torch.numpy().ravel(), res_ants.numpy().ravel())[0, 1]
    max_diff = np.max(np.abs(res_torch.numpy() - res_ants.numpy()))
    assert corr > 0.999999
    assert max_diff < 1e-4


def test_denoise_image_3d_rician_parity(synthetic_3d):
    res_torch = denoise_image(synthetic_3d, p=1, r=1, noise_model="Rician", shrink_factor=1)
    res_ants = ants.denoise_image(synthetic_3d, p=1, r=1, noise_model="Rician", shrink_factor=1)

    corr = np.corrcoef(res_torch.numpy().ravel(), res_ants.numpy().ravel())[0, 1]
    mean_diff = np.mean(np.abs(res_torch.numpy() - res_ants.numpy()))
    assert corr > 0.999
    assert mean_diff < 0.05


def test_denoise_image_shrink_factor(r16_image):
    # Test shrink factor = 2
    res_torch = denoise_image(r16_image, shrink_factor=2, p=1, r=2, noise_model="Gaussian")
    res_ants = ants.denoise_image(r16_image, shrink_factor=2, p=1, r=2, noise_model="Gaussian")

    assert res_torch.shape == r16_image.shape
    corr = np.corrcoef(res_torch.numpy().ravel(), res_ants.numpy().ravel())[0, 1]
    assert corr > 0.999


def test_denoise_image_with_mask(r16_image):
    mask = ants.get_mask(r16_image)
    res_torch = denoise_image(r16_image, mask=mask, p=1, r=2, noise_model="Gaussian")
    res_ants = ants.denoise_image(r16_image, mask=mask, p=1, r=2, noise_model="Gaussian")

    corr = np.corrcoef(res_torch.numpy().ravel(), res_ants.numpy().ravel())[0, 1]
    assert corr > 0.999


def test_denoise_image_tensor_and_numpy_inputs():
    np.random.seed(123)
    arr = np.random.uniform(10.0, 50.0, size=(16, 16)).astype(np.float32)

    # Numpy array input
    res_np = denoise_image(arr, noise_model="Gaussian")
    assert isinstance(res_np, np.ndarray)
    assert res_np.shape == arr.shape

    # Torch tensor input
    t = torch.from_numpy(arr)
    res_t = denoise_image(t, noise_model="Gaussian")
    assert isinstance(res_t, torch.Tensor)
    assert res_t.shape == t.shape
    assert res_t.device == t.device

    # 4D tensor (1, 1, H, W)
    t4d = t.unsqueeze(0).unsqueeze(0)
    res_t4d = denoise_image(t4d, noise_model="Gaussian")
    assert isinstance(res_t4d, torch.Tensor)
    assert res_t4d.shape == t4d.shape


def test_denoise_image_parameter_parsing(synthetic_2d):
    # Test str formats '1x1' and '2x2'
    res_str = denoise_image(synthetic_2d, p="1x1", r="2x2", noise_model="gaussian")
    res_int = denoise_image(synthetic_2d, p=1, r=2, noise_model="Gaussian")
    assert np.allclose(res_str.numpy(), res_int.numpy())


def test_denoise_image_mps_device(r16_image):
    if torch.backends.mps.is_available():
        res_cpu = denoise_image(r16_image, device="cpu", noise_model="Gaussian")
        res_mps = denoise_image(r16_image, device="mps", noise_model="Gaussian")
        corr = np.corrcoef(res_cpu.numpy().ravel(), res_mps.numpy().ravel())[0, 1]
        assert corr > 0.99999


def test_denoise_image_4d_ants_image():
    np.random.seed(42)
    arr = np.random.randn(16, 16, 16, 3).astype(np.float32)
    img4d = ants.from_numpy(arr, origin=(1.0, 2.0, 3.0, 4.0), spacing=(0.5, 0.5, 0.5, 2.0))
    den4d = denoise_image(img4d, p=1, r=1, noise_model="Gaussian")

    assert isinstance(den4d, ants.ANTsImage)
    assert den4d.dimension == 4
    assert den4d.shape == (16, 16, 16, 3)
    assert den4d.origin == (1.0, 2.0, 3.0, 4.0)
    assert den4d.spacing == (0.5, 0.5, 0.5, 2.0)
    assert np.allclose(den4d.direction, img4d.direction)

    # Parity check: each slice in 4D output should match independently denoised 3D slice
    s0 = ants.slice_image(img4d, axis=3, idx=0)
    den_s0 = denoise_image(s0, p=1, r=1, noise_model="Gaussian")
    s0_out = ants.slice_image(den4d, axis=3, idx=0)
    assert np.allclose(s0_out.numpy(), den_s0.numpy(), atol=1e-5)


def test_denoise_image_4d_masks():
    np.random.seed(42)
    arr = np.random.randn(16, 16, 16, 3).astype(np.float32)
    img4d = ants.from_numpy(arr)

    # 3D mask applied to 4D timeseries
    mask3d = ants.from_numpy((np.random.rand(16, 16, 16) > 0.3).astype(np.uint8))
    den_m3d = denoise_image(img4d, mask=mask3d, p=1, r=1)
    assert den_m3d.shape == (16, 16, 16, 3)

    # 4D mask sliced per timepoint
    mask4d = ants.from_numpy((np.random.rand(16, 16, 16, 3) > 0.3).astype(np.uint8))
    den_m4d = denoise_image(img4d, mask=mask4d, p=1, r=1)
    assert den_m4d.shape == (16, 16, 16, 3)


def test_denoise_image_4d_tensor_and_numpy():
    np.random.seed(42)
    arr = np.random.randn(16, 16, 16, 3).astype(np.float32)
    t = torch.from_numpy(arr)

    # 4D torch.Tensor (X, Y, Z, T)
    res_t = denoise_image(t, p=1, r=1, noise_model="Gaussian")
    assert isinstance(res_t, torch.Tensor)
    assert res_t.shape == (16, 16, 16, 3)

    # 4D batched torch.Tensor (1, 1, X, Y, Z, T)
    t_batched = t.unsqueeze(0).unsqueeze(0)
    res_batched = denoise_image(t_batched, p=1, r=1, noise_model="Gaussian")
    assert res_batched.shape == (1, 1, 16, 16, 16, 3)

    # 4D numpy array
    res_np = denoise_image(arr, p=1, r=1, noise_model="Gaussian")
    assert isinstance(res_np, np.ndarray)
    assert res_np.shape == (16, 16, 16, 3)
    assert np.allclose(res_np, res_t.numpy(), atol=1e-5)

    # Test time_axis=0 (T, X, Y, Z)
    arr_t0 = np.transpose(arr, (3, 0, 1, 2))
    t_t0 = torch.from_numpy(arr_t0)
    res_t0 = denoise_image(t_t0, p=1, r=1, noise_model="Gaussian", time_axis=0)
    assert res_t0.shape == (3, 16, 16, 16)
    assert torch.allclose(res_t0[0], res_t[..., 0], atol=1e-5)

