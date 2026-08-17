"""Tensor-only N4-style bias-field correction.

This implementation follows ITK N4's log-domain histogram sharpening and
single-level scattered-data B-spline update. Metadata is ITK x-y-z order;
image tensors are ``(N, C, H, W)`` or ``(N, C, D, H, W)``. Bias fields are
scalar and dimensionless in log-intensity space.
"""

from itertools import product
from math import ceil, log2, prod
from typing import Optional, Sequence, Union

import torch
from torch import Tensor, nn

from .bspline_domain import BSplineDomain
from .bspline_synthesis import cubic_bspline_basis, synthesize_bspline_velocity


def _expand_like_image(value: Optional[Tensor], image: Tensor, name: str, default: float) -> Tensor:
    if value is None:
        return image.new_full((image.shape[0], 1) + image.shape[2:], default).expand_as(image)
    if value.device != image.device:
        raise ValueError(f"{name} and image must be on the same device")
    value = value.to(dtype=image.dtype)
    if value.ndim == image.ndim - 1:
        value = value.unsqueeze(1)
    if value.ndim != image.ndim or value.shape[0] != image.shape[0] or value.shape[2:] != image.shape[2:]:
        raise ValueError(f"{name} must match the image batch and spatial shape")
    if value.shape[1] not in (1, image.shape[1]):
        raise ValueError(f"{name} must have one channel or match the image channels")
    return value.expand_as(image)


def _masked_extrema(value: Tensor, included: Tensor):
    axes = tuple(range(2, value.ndim))
    minimum = torch.where(included, value, torch.full_like(value, float("inf"))).amin(axes, keepdim=True)
    maximum = torch.where(included, value, torch.full_like(value, float("-inf"))).amax(axes, keepdim=True)
    nonempty = included.any(dim=axes, keepdim=True)
    minimum = torch.where(nonempty, minimum, torch.zeros_like(minimum))
    maximum = torch.where(nonempty, maximum, torch.zeros_like(maximum))
    return minimum, maximum


def _histogram_sharpen(
    log_image: Tensor,
    included: Tensor,
    *,
    number_of_histogram_bins: int,
    wiener_filter_noise: float,
    bias_field_fwhm: float,
    eps: float,
) -> Tensor:
    """PyTorch translation of ITK N4 ``SharpenImage``."""
    bins = number_of_histogram_bins
    minimum, maximum = _masked_extrema(log_image, included)
    histogram_range = maximum - minimum
    slope = (histogram_range / float(bins - 1)).clamp_min(eps)
    continuous = ((log_image - minimum) / slope).clamp(0.0, float(bins - 1))
    lower = torch.floor(continuous).to(torch.long)
    fraction = continuous - lower.to(continuous.dtype)
    upper = (lower + 1).clamp_max(bins - 1)
    valid_weight = included.to(log_image.dtype)

    flat_shape = (log_image.shape[0] * log_image.shape[1], -1)
    histogram = log_image.new_zeros((flat_shape[0], bins))
    histogram.scatter_add_(1, lower.reshape(flat_shape), ((1.0 - fraction) * valid_weight).reshape(flat_shape))
    histogram.scatter_add_(1, upper.reshape(flat_shape), (fraction * valid_weight).reshape(flat_shape))

    padded_size = 2 ** (ceil(log2(bins)) + 1)
    offset = (padded_size - bins) // 2
    padded = log_image.new_zeros((flat_shape[0], padded_size))
    padded[:, offset : offset + bins] = histogram
    histogram_fft = torch.fft.fft(padded)

    flat_slope = slope.reshape(flat_shape[0], 1)
    scaled_fwhm = (bias_field_fwhm / flat_slope).clamp_min(eps)
    frequency_index = torch.arange(padded_size, dtype=log_image.dtype, device=log_image.device)
    distance = torch.minimum(frequency_index, padded_size - frequency_index)[None]
    exponent = 4.0 * log_image.new_tensor(2.0).log() / scaled_fwhm.square()
    scale = 2.0 * torch.sqrt(log_image.new_tensor(2.0).log() / log_image.new_tensor(torch.pi)) / scaled_fwhm
    gaussian = scale * torch.exp(-distance.square() * exponent)
    gaussian_fft = torch.fft.fft(gaussian)
    wiener = gaussian_fft.conj() / (gaussian_fft.abs().square() + wiener_filter_noise)
    deconvolved = torch.fft.ifft(histogram_fft * wiener.real).real.clamp_min(0.0)

    bin_coordinates = (
        minimum.reshape(flat_shape[0], 1)
        + (frequency_index[None] - offset) * flat_slope
    )
    numerator = torch.fft.ifft(torch.fft.fft(bin_coordinates * deconvolved) * gaussian_fft).real
    denominator = torch.fft.ifft(torch.fft.fft(deconvolved) * gaussian_fft).real
    mapping = torch.where(denominator.abs() > eps, numerator / denominator, torch.zeros_like(numerator))
    mapping = mapping[:, offset : offset + bins]

    map_lower = mapping.gather(1, lower.reshape(flat_shape))
    map_upper = mapping.gather(1, upper.reshape(flat_shape))
    sharpened = map_lower + (map_upper - map_lower) * fraction.reshape(flat_shape)
    sharpened = sharpened.reshape_as(log_image)
    sharpened = torch.where(histogram_range <= eps, log_image, sharpened)
    return torch.where(included, sharpened, torch.zeros_like(sharpened))


def _fit_bspline_update(
    residual: Tensor,
    weights: Tensor,
    domain: BSplineDomain,
    lattice_itk: Sequence[int],
    shrink_factor: int,
    eps: float,
) -> Tensor:
    """One ITK-style scattered-data update, returned in PyTorch lattice order."""
    dimension = domain.dimension
    slices = (slice(None), slice(None)) + (slice(None, None, shrink_factor),) * dimension
    residual = residual[slices]
    weights = weights[slices]
    sample_shape = residual.shape[2:]
    torch_coordinates = torch.meshgrid(
        *[torch.arange(n, device=residual.device) * shrink_factor for n in sample_shape], indexing="ij"
    )
    itk_coordinates = tuple(reversed(torch_coordinates))

    neighbors, basis = [], []
    for coordinate, dense_size, lattice_size in zip(itk_coordinates, domain.size, lattice_itk):
        spans = lattice_size - 3
        u = coordinate.to(residual.dtype) * (float(spans) / float(dense_size - 1))
        u = u.clamp_max(torch.nextafter(u.new_tensor(float(spans)), u.new_tensor(float("-inf"))))
        base = torch.floor(u).to(torch.long)
        local = torch.arange(4, device=residual.device)
        index = base[..., None] + local
        neighbors.append(index)
        basis.append(cubic_bspline_basis(u[..., None] - index.to(u.dtype) + 1.0))

    support_indices, support_basis = [], []
    strides, stride = [], 1
    for size in lattice_itk:
        strides.append(stride)
        stride *= size
    for support in product(range(4), repeat=dimension):
        index = sum(neighbors[d][..., support[d]] * strides[d] for d in range(dimension))
        value = torch.ones(sample_shape, dtype=residual.dtype, device=residual.device)
        for d in range(dimension):
            value = value * basis[d][..., support[d]]
        support_indices.append(index.flatten())
        support_basis.append(value.flatten())
    basis_values = torch.stack(support_basis)
    squared_sum = basis_values.square().sum(dim=0).clamp_min(eps)

    batch_channels = residual.shape[0] * residual.shape[1]
    residual_flat = residual.reshape(batch_channels, -1)
    weight_flat = weights.reshape(batch_channels, -1)
    delta = residual.new_zeros((batch_channels, prod(lattice_itk)))
    omega = residual.new_zeros(delta.shape)
    for index, value in zip(support_indices, support_basis):
        expanded_index = index[None].expand(batch_channels, -1)
        omega.scatter_add_(1, expanded_index, weight_flat * value.square()[None])
        delta.scatter_add_(
            1,
            expanded_index,
            residual_flat * weight_flat * (value.pow(3) / squared_sum)[None],
        )
    coefficients = torch.where(omega > eps, delta / omega.clamp_min(eps), torch.zeros_like(delta))
    return coefficients.reshape((residual.shape[0], residual.shape[1]) + tuple(reversed(lattice_itk)))


def _initial_lattice_size(domain: BSplineDomain, spline_param) -> tuple:
    if spline_param is None:
        return (4,) * domain.dimension
    if isinstance(spline_param, (int, float)):
        if spline_param <= 0:
            raise ValueError("scalar spline_param must be positive")
        return tuple(max(1, ceil(extent / float(spline_param))) + 3 for extent in domain.physical_extent)
    values = tuple(spline_param)
    if len(values) == 1:
        return _initial_lattice_size(domain, float(values[0]))
    if len(values) != domain.dimension or any(float(value) < 1 for value in values):
        raise ValueError("vector spline_param must contain one positive mesh size per dimension")
    return tuple(int(value) + 3 for value in values)


def n4_bias_field_correction(
    image: Tensor,
    domain: Optional[BSplineDomain] = None,
    mask: Optional[Tensor] = None,
    *,
    rescale_intensities: bool = False,
    shrink_factor: int = 4,
    convergence: dict = None,
    spline_param=None,
    return_bias_field: bool = False,
    weight_mask: Optional[Tensor] = None,
    number_of_histogram_bins: int = 200,
    wiener_filter_noise: float = 0.01,
    bias_field_fwhm: float = 0.15,
    eps: float = 1e-6,
) -> Tensor:
    """Differentiable N4-style correction for batched 2-D/3-D scalar images.

    The call mirrors the principal ANTsPy options. Each channel is corrected
    independently. ``spline_param`` follows ANTs: a scalar is physical spline
    distance; a vector is the mesh size in ITK x-y-z order. Unlike the ANTs
    executable, scalar spacing does not pad the image domain.
    """
    if image.ndim not in (4, 5) or not image.is_floating_point():
        raise ValueError("image must be a floating (N,C,H,W) or (N,C,D,H,W) tensor")
    dimension = image.ndim - 2
    domain = domain or BSplineDomain(tuple(reversed(image.shape[2:])))
    if domain.dimension != dimension or tuple(image.shape[2:]) != domain.torch_size:
        raise ValueError("image shape does not match domain")
    if not isinstance(shrink_factor, int) or shrink_factor < 1:
        raise ValueError("shrink_factor must be a positive integer")
    if number_of_histogram_bins < 2:
        raise ValueError("number_of_histogram_bins must be at least 2")
    convergence = convergence or {"iters": [50, 50, 50, 50], "tol": 1e-7}
    iterations = tuple(int(value) for value in convergence["iters"])
    tolerance = float(convergence.get("tol", 1e-7))

    mask_value = _expand_like_image(mask, image, "mask", 1.0)
    confidence = _expand_like_image(weight_mask, image, "weight_mask", 1.0).clamp_min(0.0)
    included = (mask_value != 0) & (confidence > 0)
    fit_weights = included.to(image.dtype) * confidence
    positive_image = image.clamp_min(eps)
    log_input = positive_image.log()
    log_bias = torch.zeros_like(image)
    lattice_itk = _initial_lattice_size(domain, spline_param)

    for level, maximum_iterations in enumerate(iterations):
        active = torch.ones(
            (image.shape[0], image.shape[1]) + (1,) * dimension,
            dtype=image.dtype,
            device=image.device,
        )
        for _ in range(maximum_iterations):
            uncorrected = log_input - log_bias
            sharpened = _histogram_sharpen(
                uncorrected,
                included,
                number_of_histogram_bins=number_of_histogram_bins,
                wiener_filter_noise=wiener_filter_noise,
                bias_field_fwhm=bias_field_fwhm,
                eps=eps,
            )
            residual = uncorrected - sharpened
            coefficients = _fit_bspline_update(residual, fit_weights, domain, lattice_itk, shrink_factor, eps)
            update = synthesize_bspline_velocity(coefficients, domain)
            new_log_bias = log_bias + active * update
            difference = torch.exp(new_log_bias - log_bias)
            selected = included.to(image.dtype)
            count = selected.sum(dim=tuple(range(2, image.ndim)), keepdim=True).clamp_min(2.0)
            mean = (difference * selected).sum(dim=tuple(range(2, image.ndim)), keepdim=True) / count
            variance = ((difference - mean).square() * selected).sum(
                dim=tuple(range(2, image.ndim)), keepdim=True
            ) / (count - 1.0)
            convergence_measurement = variance.sqrt() / mean.clamp_min(eps)
            log_bias = new_log_bias
            # Tensor gating honors convergence independently for every batch
            # item/channel without a device-to-host synchronization.
            active = active * (convergence_measurement > tolerance).to(image.dtype)
        if level + 1 < len(iterations):
            lattice_itk = tuple(2 * value - 3 for value in lattice_itk)

    bias = torch.exp(log_bias)
    corrected = image / bias.clamp_min(eps)
    corrected = torch.where(mask_value != 0, corrected, image)
    if rescale_intensities:
        original_min, original_max = _masked_extrema(image, included)
        corrected_min, corrected_max = _masked_extrema(corrected, included)
        corrected = (corrected - corrected_min) * (original_max - original_min) / (
            corrected_max - corrected_min
        ).clamp_min(eps) + original_min
        corrected = torch.where(mask_value != 0, corrected, image)
    return bias if return_bias_field else corrected


class N4BiasFieldCorrection(nn.Module):
    """Module wrapper storing N4 options while accepting image geometry per call."""

    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs

    def forward(self, image: Tensor, domain: Optional[BSplineDomain] = None, mask: Optional[Tensor] = None, weight_mask: Optional[Tensor] = None) -> Tensor:
        return n4_bias_field_correction(image, domain, mask, weight_mask=weight_mask, **self.kwargs)
