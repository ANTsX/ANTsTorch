import numpy as np
import pytest
import torch

from antstorch.bspline_flows import ImageDomain, N4BiasFieldCorrection, n4_bias_field_correction


def _options(dimension, iterations=3):
    return dict(
        shrink_factor=1,
        convergence={"iters": [iterations], "tol": 0.0},
        spline_param=(2,) * dimension,
        number_of_histogram_bins=32,
    )


@pytest.mark.parametrize("size", [(14, 12), (9, 8, 7)])
def test_constant_image_is_preserved(size):
    domain = ImageDomain(size, spacing=(1.3,) * len(size))
    image = torch.full((2, 1) + domain.torch_size, 7.0, dtype=torch.double)
    corrected = n4_bias_field_correction(image, domain, **_options(len(size), iterations=1))
    torch.testing.assert_close(corrected, image, rtol=2e-5, atol=2e-5)


def test_synthetic_smooth_bias_reduces_nonuniformity():
    domain = ImageDomain((24, 20))
    y = torch.linspace(-1, 1, 20)[:, None]
    x = torch.linspace(-1, 1, 24)[None, :]
    image = torch.exp(0.4 * x + 0.2 * y)[None, None]
    corrected = n4_bias_field_correction(image, domain, **_options(2, iterations=10))
    assert corrected.std() / corrected.mean() < 0.6 * image.std() / image.mean()


def test_mask_preserves_values_outside_and_weight_mask_is_supported():
    domain = ImageDomain((16, 14))
    image = torch.rand(1, 1, 14, 16) + 1.0
    mask = torch.zeros_like(image)
    mask[..., 2:-2, 3:-3] = 1
    confidence = mask * torch.linspace(0.2, 1.0, 16)[None, None, None, :]
    corrected = n4_bias_field_correction(
        image, domain, mask, weight_mask=confidence, **_options(2, iterations=2)
    )
    torch.testing.assert_close(corrected[mask == 0], image[mask == 0])
    assert torch.isfinite(corrected).all()


def test_returned_bias_is_positive_and_reconstructs_correction():
    domain = ImageDomain((15, 13))
    image = torch.rand(1, 2, 13, 15, dtype=torch.double) + 0.5
    options = _options(2, iterations=2)
    bias = n4_bias_field_correction(image, domain, return_bias_field=True, **options)
    corrected = n4_bias_field_correction(image, domain, **options)
    assert torch.all(bias > 0)
    torch.testing.assert_close(corrected, image / bias, rtol=1e-13, atol=1e-13)


def test_module_and_function_match():
    domain = ImageDomain((12, 10))
    image = torch.rand(1, 1, 10, 12) + 1
    options = _options(2, iterations=1)
    module = N4BiasFieldCorrection(**options)
    torch.testing.assert_close(module(image, domain), n4_bias_field_correction(image, domain, **options))


def test_gradient_propagates_to_input():
    torch.manual_seed(42)
    domain = ImageDomain((8, 7))
    image = (torch.rand(1, 1, 7, 8, dtype=torch.double) + 1.0).requires_grad_()
    corrected = n4_bias_field_correction(
        image,
        domain,
        shrink_factor=1,
        convergence={"iters": [1], "tol": 0.0},
        spline_param=(1, 1),
        number_of_histogram_bins=16,
    )
    corrected.square().mean().backward()
    assert image.grad is not None
    assert torch.isfinite(image.grad).all()
    assert torch.count_nonzero(image.grad) > 0


def test_n4_gradcheck_at_generic_intensities():
    torch.manual_seed(4)
    domain = ImageDomain((4, 4))
    image = (torch.rand(1, 1, 4, 4, dtype=torch.double) + 1.0).requires_grad_()

    def correction(value):
        return n4_bias_field_correction(
            value,
            domain,
            shrink_factor=1,
            convergence={"iters": [1], "tol": 0.0},
            spline_param=(1, 1),
            number_of_histogram_bins=8,
            eps=1e-7,
        )

    assert torch.autograd.gradcheck(correction, (image,), eps=1e-6, atol=2e-4, rtol=2e-3)


@pytest.mark.parametrize("size", [(12, 10), (8, 7, 6)])
def test_stable_and_vectorized_atomic_accumulation_agree(size):
    torch.manual_seed(1)
    dimension = len(size)
    domain = ImageDomain(size)
    image = torch.rand((1, 1) + domain.torch_size, dtype=torch.double) + 1.0
    options = dict(
        shrink_factor=2,
        convergence={"iters": [2], "tol": 0.0},
        spline_param=(1,) * dimension,
        number_of_histogram_bins=16,
        return_bias_field=True,
    )
    atomic = n4_bias_field_correction(image, domain, stable_accumulation=False, **options)
    stable = n4_bias_field_correction(image, domain, stable_accumulation=True, **options)
    torch.testing.assert_close(stable, atomic, rtol=1e-13, atol=1e-13)


def test_rescale_restores_masked_intensity_range():
    domain = ImageDomain((14, 12))
    image = torch.linspace(1, 5, 14 * 12).reshape(1, 1, 12, 14)
    corrected = n4_bias_field_correction(
        image, domain, rescale_intensities=True, **_options(2, iterations=2)
    )
    assert corrected.amin().item() == pytest.approx(image.amin().item(), abs=1e-5)
    assert corrected.amax().item() == pytest.approx(image.amax().item(), abs=1e-5)


def test_agrees_with_antspy_n4_on_smooth_2d_phantom():
    ants = pytest.importorskip("ants")
    size_x, size_y = 32, 28
    x = np.linspace(-1, 1, size_x)[:, None]
    y = np.linspace(-1, 1, size_y)[None, :]
    anatomy = 1.0 + 0.4 * np.exp(-4.0 * (x * x + y * y)) + 0.2 * (x > 0)
    image_itk = (anatomy * np.exp(0.3 * x - 0.15 * y + 0.08 * x * y)).astype("float32")
    spacing = (1.3, 2.1)
    ants_image = ants.from_numpy(image_itk, spacing=spacing)
    ants_mask = ants.from_numpy(np.ones_like(image_itk), spacing=spacing)
    ants_bias = ants.n4_bias_field_correction(
        ants_image,
        ants_mask,
        shrink_factor=1,
        convergence={"iters": [5], "tol": 0.0},
        spline_param=[2, 2],
        return_bias_field=True,
    ).numpy()

    image_torch = torch.from_numpy(image_itk.T)[None, None]
    torch_bias = n4_bias_field_correction(
        image_torch,
        ImageDomain((size_x, size_y), spacing=spacing),
        torch.ones_like(image_torch),
        shrink_factor=1,
        convergence={"iters": [5], "tol": 0.0},
        spline_param=(2, 2),
        return_bias_field=True,
        number_of_histogram_bins=200,
    )[0, 0].numpy().T

    # N4 bias fields have an arbitrary global multiplicative scale.
    ants_bias /= np.exp(np.log(ants_bias).mean())
    torch_bias /= np.exp(np.log(torch_bias).mean())
    np.testing.assert_allclose(torch_bias, ants_bias, rtol=1e-4, atol=1e-4)


def test_agrees_with_antspy_n4_multiresolution():
    ants = pytest.importorskip("ants")
    r16 = ants.image_read(ants.get_data("r16")).clone("float")
    mask = r16 * 0 + 1
    ants_bias = ants.n4_bias_field_correction(
        r16,
        mask=mask,
        shrink_factor=4,
        convergence={"iters": [20, 20], "tol": 0.0},
        spline_param=[4, 4],
        return_bias_field=True,
    ).numpy()

    image_torch = torch.from_numpy(r16.numpy().T)[None, None]
    mask_torch = torch.from_numpy(mask.numpy().T)[None, None]
    domain = ImageDomain(
        size=r16.shape,
        spacing=r16.spacing,
        origin=r16.origin,
        direction=tuple(tuple(row) for row in r16.direction),
    )
    torch_bias = n4_bias_field_correction(
        image_torch,
        domain,
        mask_torch,
        shrink_factor=4,
        convergence={"iters": [20, 20], "tol": 0.0},
        spline_param=(4, 4),
        return_bias_field=True,
        number_of_histogram_bins=200,
    )[0, 0].numpy().T

    ants_bias /= np.exp(np.log(ants_bias).mean())
    torch_bias /= np.exp(np.log(torch_bias).mean())
    log_ants = np.log(ants_bias)
    log_torch = np.log(torch_bias)
    assert np.corrcoef(log_ants.ravel(), log_torch.ravel())[0, 1] > 0.99
    assert np.mean(np.abs(log_ants - log_torch)) < 0.02



@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_cpu_cuda_agreement():
    domain = ImageDomain((14, 12))
    image = torch.rand(1, 1, 12, 14) + 1.0
    options = _options(2, iterations=2)
    cpu = n4_bias_field_correction(image, domain, **options)
    gpu = n4_bias_field_correction(image.cuda(), domain, **options).cpu()
    torch.testing.assert_close(gpu, cpu, rtol=2e-5, atol=2e-5)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS is not available")
def test_mps_is_repeatable_and_agrees_with_cpu():
    torch.manual_seed(91)
    domain = ImageDomain((24, 20))
    y = torch.linspace(-1, 1, 20)[:, None]
    x = torch.linspace(-1, 1, 24)[None, :]
    image = (1.0 + 0.3 * torch.exp(-3.0 * (x.square() + y.square()))) * torch.exp(0.2 * x - 0.1 * y)
    image = image[None, None].float()
    options = dict(
        shrink_factor=2,
        convergence={"iters": [4, 4], "tol": 0.0},
        spline_param=(1, 1),
        number_of_histogram_bins=32,
        return_bias_field=True,
    )
    cpu = n4_bias_field_correction(image, domain, stable_accumulation=True, **options)
    first = n4_bias_field_correction(image.to("mps"), domain, **options).cpu()
    second = n4_bias_field_correction(image.to("mps"), domain, **options).cpu()
    assert torch.isfinite(first).all()
    torch.testing.assert_close(second, first, rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(first, cpu, rtol=2e-3, atol=2e-3)
