"""Tensor-only N4-style bias-field correction.

This implementation follows ITK N4's log-domain histogram sharpening and
single-level scattered-data B-spline update. Metadata is ITK x-y-z order;
image tensors are ``(N, C, H, W)`` or ``(N, C, D, H, W)``. Bias fields are
scalar and dimensionless in log-intensity space.
"""

from itertools import product
from math import ceil, log2
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
    stable_accumulation: bool,
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
    flat_lower = lower.reshape(flat_shape)
    flat_upper = upper.reshape(flat_shape)
    flat_fraction = fraction.reshape(flat_shape)
    flat_valid = valid_weight.reshape(flat_shape)
    if stable_accumulation:
        # MPS scatter reductions use unordered atomics. A chunked categorical
        # reduction costs more arithmetic but is repeatable and numerically
        # much closer to the ordered CPU implementation.
        bin_axis = torch.arange(bins, device=log_image.device)
        sample_chunk = max(1, 8_000_000 // bins)
        for begin in range(0, flat_lower.shape[1], sample_chunk):
            end = min(begin + sample_chunk, flat_lower.shape[1])
            lower_assignment = (flat_lower[:, begin:end, None] == bin_axis).to(log_image.dtype)
            upper_assignment = (flat_upper[:, begin:end, None] == bin_axis).to(log_image.dtype)
            fraction_chunk = flat_fraction[:, begin:end, None]
            valid_chunk = flat_valid[:, begin:end, None]
            histogram = histogram + (
                ((1.0 - fraction_chunk) * lower_assignment + fraction_chunk * upper_assignment)
                * valid_chunk
            ).sum(dim=1)
    else:
        histogram.scatter_add_(1, flat_lower, ((1.0 - flat_fraction) * flat_valid))
        histogram.scatter_add_(1, flat_upper, (flat_fraction * flat_valid))

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


def _shrink_slices(spatial_shape, shrink_factor: int):
    spatial_slices = []
    for dense_size in spatial_shape:
        shrunk_size = dense_size // shrink_factor
        center_offset = int(0.5 * (dense_size - 1 - shrink_factor * (shrunk_size - 1)) + 0.5)
        spatial_slices.append(
            slice(center_offset, center_offset + shrunk_size * shrink_factor, shrink_factor)
        )
    return tuple(spatial_slices)


def _bspline_fit_geometry(
    sample_shape: Sequence[int],
    lattice_itk: Sequence[int],
    dtype: torch.dtype,
    device: torch.device,
    eps: float,
):
    """Support-point indices/weights for one B-spline scattered-data update.

    ``sample_shape`` is the already-shrunk spatial shape, in PyTorch order.
    This only depends on geometry (shapes and the lattice resolution), never
    on image intensities, so it is computed once per refinement level and
    reused for every convergence iteration at that level rather than being
    rebuilt on every iteration.
    """
    dimension = len(lattice_itk)
    if any(size < 2 for size in sample_shape):
        raise ValueError(
            "shrink_factor is too large for this image: a shrunk spatial "
            f"dimension has fewer than 2 samples {tuple(sample_shape)}; "
            "use a smaller shrink_factor."
        )
    torch_coordinates = torch.meshgrid(
        *[torch.arange(n, device=device) for n in sample_shape], indexing="ij"
    )
    itk_coordinates = tuple(reversed(torch_coordinates))

    neighbors, basis = [], []
    for coordinate, dense_size, lattice_size in zip(
        itk_coordinates, tuple(reversed(sample_shape)), lattice_itk
    ):
        spans = lattice_size - 3
        u = coordinate.to(dtype) * (float(spans) / float(dense_size - 1))
        u = u.clamp_max(torch.nextafter(u.new_tensor(float(spans)), u.new_tensor(float("-inf"))))
        base = torch.floor(u).to(torch.long)
        local = torch.arange(4, device=device)
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
        value = torch.ones(sample_shape, dtype=dtype, device=device)
        for d in range(dimension):
            value = value * basis[d][..., support[d]]
        support_indices.append(index.flatten())
        support_basis.append(value.flatten())

    indices = torch.stack(support_indices)
    basis_values = torch.stack(support_basis)
    squared_sum = basis_values.square().sum(dim=0).clamp_min(eps)
    coefficient_count = stride
    return indices, basis_values, squared_sum, coefficient_count


def _bspline_fit_context(weight_flat: Tensor, geometry, stable_accumulation: bool):
    """Precompute the residual-independent half of the scattered-data update.

    ``omega`` (the normal-equation weight accumulator) depends only on the
    fit weights and the B-spline geometry -- never on the current residual --
    so it is identical on every convergence iteration within a level.
    Computing it once here, instead of on every iteration, removes the
    single largest redundant cost of the original per-iteration
    implementation (an unordered/one-hot reduction over every sample and
    every one of the ``4**D`` local support points). It also matters most on
    MPS, where the chunked one-hot reduction used for stability is the most
    expensive step in the whole N4 loop.
    """
    indices, basis_values, squared_sum, coefficient_count = geometry
    batch_channels = weight_flat.shape[0]
    omega = weight_flat.new_zeros((batch_channels, coefficient_count))
    support_count, sample_count = indices.shape
    if stable_accumulation:
        # Avoid MPS atomic scatter reductions. Chunked one-hot support
        # matrices are reduced in a fixed order and accumulated with matrix
        # products. The chunk-wise ``delta_basis`` factors are geometry-only
        # and are cached for reuse by every iteration's solve step.
        max_one_hot_elements = 8_000_000
        sample_chunk = max(1, max_one_hot_elements // (support_count * coefficient_count))
        coefficient_axis = torch.arange(coefficient_count, device=indices.device)
        chunks = []
        for begin in range(0, sample_count, sample_chunk):
            end = min(begin + sample_chunk, sample_count)
            chunk_indices = indices[:, begin:end].transpose(0, 1)
            chunk_basis = basis_values[:, begin:end].transpose(0, 1)
            assignment = (chunk_indices[..., None] == coefficient_axis).to(basis_values.dtype)
            omega_basis = (assignment * chunk_basis.square().unsqueeze(-1)).sum(dim=1)
            delta_basis = (
                assignment * (chunk_basis.pow(3) / squared_sum[begin:end, None]).unsqueeze(-1)
            ).sum(dim=1)
            omega = omega + weight_flat[:, begin:end] @ omega_basis
            chunks.append((begin, end, delta_basis))
        return {"mode": "stable", "chunks": chunks, "omega": omega}
    else:
        # A single vectorized scatter per statistic replaces 4**D kernel
        # launches while preserving the sparse ITK update formula. Sample
        # chunking bounds peak memory for large (e.g. shrink_factor=1 3-D)
        # volumes instead of materializing one (batch*support*sample) tensor.
        max_scatter_elements = 32_000_000
        sample_chunk = max(1, max_scatter_elements // (batch_channels * support_count))
        cubed_over_squared_sum = basis_values.pow(3) / squared_sum[None]
        for begin in range(0, sample_count, sample_chunk):
            end = min(begin + sample_chunk, sample_count)
            expanded_indices = indices[:, begin:end].reshape(-1)[None].expand(batch_channels, -1)
            omega_values = weight_flat[:, None, begin:end] * basis_values[:, begin:end].square()[None]
            omega.scatter_add_(1, expanded_indices, omega_values.reshape(batch_channels, -1))
        return {
            "mode": "vectorized",
            "omega": omega,
            "indices": indices,
            "cubed_over_squared_sum": cubed_over_squared_sum,
            "sample_chunk": sample_chunk,
        }


def _bspline_fit_solve(residual_flat: Tensor, weight_flat: Tensor, context: dict, eps: float) -> Tensor:
    """Per-iteration scattered-data update given a level-fixed context."""
    batch_channels = residual_flat.shape[0]
    omega = context["omega"]
    delta = residual_flat.new_zeros(omega.shape)
    if context["mode"] == "stable":
        for begin, end, delta_basis in context["chunks"]:
            delta = delta + (residual_flat[:, begin:end] * weight_flat[:, begin:end]) @ delta_basis
    else:
        indices = context["indices"]
        cubed_over_squared_sum = context["cubed_over_squared_sum"]
        sample_chunk = context["sample_chunk"]
        sample_count = residual_flat.shape[1]
        for begin in range(0, sample_count, sample_chunk):
            end = min(begin + sample_chunk, sample_count)
            expanded_indices = indices[:, begin:end].reshape(-1)[None].expand(batch_channels, -1)
            delta_values = (
                residual_flat[:, None, begin:end]
                * weight_flat[:, None, begin:end]
                * cubed_over_squared_sum[None, :, begin:end]
            )
            delta.scatter_add_(1, expanded_indices, delta_values.reshape(batch_channels, -1))
    return torch.where(omega > eps, delta / omega.clamp_min(eps), torch.zeros_like(delta))


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
    stable_accumulation: Optional[bool] = None,
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
    fit_slices = (slice(None), slice(None)) + _shrink_slices(image.shape[2:], shrink_factor)
    shrink_selector = torch.zeros_like(included)
    shrink_selector[fit_slices] = True
    histogram_included = included & shrink_selector
    if stable_accumulation is None:
        stable_accumulation = image.device.type == "mps"
    positive_image = image.clamp_min(eps)
    log_input = positive_image.log()
    log_bias = torch.zeros_like(image)
    lattice_itk = _initial_lattice_size(domain, spline_param)

    # The shrunk fit weights and their flattened view never change across
    # iterations or refinement levels (only the mesh resolution does), so
    # they are sliced/flattened once and reused everywhere below.
    batch_channels = image.shape[0] * image.shape[1]
    fit_sample_shape = tuple(n // shrink_factor for n in image.shape[2:])
    weight_flat = fit_weights[fit_slices].reshape(batch_channels, -1)

    for level, maximum_iterations in enumerate(iterations):
        active = torch.ones(
            (image.shape[0], image.shape[1]) + (1,) * dimension,
            dtype=image.dtype,
            device=image.device,
        )
        # Geometry (support indices/weights) and omega (the residual-
        # independent normal-equation term) depend only on the current
        # lattice resolution, so both are computed once per level and
        # reused for every convergence iteration at that level instead of
        # being rebuilt on every one of ``maximum_iterations`` iterations.
        geometry = _bspline_fit_geometry(fit_sample_shape, lattice_itk, image.dtype, image.device, eps)
        fit_context = _bspline_fit_context(weight_flat, geometry, stable_accumulation)
        for _ in range(maximum_iterations):
            uncorrected = log_input - log_bias
            sharpened = _histogram_sharpen(
                uncorrected,
                histogram_included,
                number_of_histogram_bins=number_of_histogram_bins,
                wiener_filter_noise=wiener_filter_noise,
                bias_field_fwhm=bias_field_fwhm,
                eps=eps,
                stable_accumulation=stable_accumulation,
            )
            residual = uncorrected - sharpened
            residual_flat = residual[fit_slices].reshape(batch_channels, -1)
            coefficients = _bspline_fit_solve(residual_flat, weight_flat, fit_context, eps).reshape(
                (image.shape[0], image.shape[1]) + tuple(reversed(lattice_itk))
            )
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
