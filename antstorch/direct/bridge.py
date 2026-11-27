"""ANTsImage interface for the tensor DiReCT implementation."""

import math

import ants
import numpy as np
import torch

from ..bspline_flows import ImageDomain
from ..syn.bridge import ants_image_to_tensor, tensor_to_ants_image
from ..syn.core.pipeline import auto_detect_device
from .core import direct_cortical_thickness


def _foreground_bbox_slices(seg_arr: np.ndarray, spacing: tuple[float, ...], smoothing_sigma: float, velocity_smoothing_variance: float):
    """Compute a padded tight foreground bounding box around ``seg_arr != 0``.

    DiReCT velocity, integrated field, hit, and total vanish identically outside ``segmentation != 0``,
    so padding by ``margin >= max(10, ceil(4 * sigma / min_sp) + 6)`` voxels keeps every discrete
    Gaussian stencil and boundary condition interior-identical while dropping ~65-75% empty air voxels
    on whole-head 3-D T1 grids.
    """
    nz = np.nonzero(seg_arr != 0)
    if len(nz) == 0 or nz[0].size == 0:
        return None
    min_sp = max(min(float(s) for s in spacing), 1e-6)
    vel_sigma_vox = math.sqrt(max(float(velocity_smoothing_variance), 0.0))
    grad_sigma_vox = float(smoothing_sigma) / min_sp
    margin = max(10, int(math.ceil(4.0 * max(vel_sigma_vox, grad_sigma_vox, 1.0))) + 4)
    slices = []
    cropped_vox = 1
    for d, coords in enumerate(nz):
        lo = max(0, int(coords.min()) - margin)
        hi = min(int(seg_arr.shape[d]), int(coords.max()) + margin + 1)
        slices.append(slice(lo, hi))
        cropped_vox *= (hi - lo)
    if cropped_vox >= int(0.85 * seg_arr.size):
        return ()  # foreground already fills the volume; run uncropped
    return tuple(slices)


def _subarray_to_tensor(arr: np.ndarray, device: torch.device) -> torch.Tensor:
    """Convert an ITK-order ``(X, Y[, Z])`` numpy subarray into a ``(1, 1, *torch_shape)`` float32 tensor."""
    rev = np.ascontiguousarray(np.transpose(arr, tuple(reversed(range(arr.ndim)))), dtype=np.float32)
    return torch.from_numpy(rev).unsqueeze(0).unsqueeze(0).to(device=device, dtype=torch.float32)


def kelly_kapowski(
    segmentation,
    gray_matter,
    white_matter,
    *,
    iterations: int = 45,
    gradient_step: float = 0.025,
    smoothing_sigma: float = 1.0,
    velocity_smoothing_variance: float = 1.5,
    integration_points: int = 10,
    thickness_prior: float = 10.0,
    optimizer: str = "direct",
    regularizer: str = "gaussian",
    crop_foreground: bool = True,
    device=None,
    verbose: bool = False,
):
    """ANTsImage-compatible DiReCT entry point implemented with PyTorch.

    Agreement with ANTs KellyKapowski is tested, but numerical identity is
    not guaranteed. Gradient construction, reduction order and interpolation
    can differ. See direct/README.md, "Differences from ANTs KellyKapowski",
    for the validation scope, optional variants and MPS CPU fallback behaviour.
    """
    if segmentation.dimension not in (2, 3):
        raise ValueError("kelly_kapowski supports 2-D and 3-D images")
    for name, image in (("gray_matter", gray_matter), ("white_matter", white_matter)):
        if image.dimension != segmentation.dimension or image.shape != segmentation.shape:
            raise ValueError(f"{name} must match the segmentation domain")
    if device is None:
        from ..utilities.device_manager import get_user_default_device
        device = get_user_default_device()
    resolved_device = torch.device(device) if device is not None else auto_detect_device()

    seg_arr = segmentation.view() if hasattr(segmentation, "view") else segmentation.numpy()
    spacing_tup = tuple(float(v) for v in segmentation.spacing)
    origin_tup = tuple(float(v) for v in segmentation.origin)
    identity = tuple(
        tuple(float(i == j) for j in range(segmentation.dimension))
        for i in range(segmentation.dimension)
    )

    bbox_slices = (
        _foreground_bbox_slices(seg_arr, spacing_tup, smoothing_sigma, velocity_smoothing_variance)
        if crop_foreground
        else ()
    )
    if bbox_slices is None:
        return ants.from_numpy(
            np.zeros(segmentation.shape, dtype=np.float32),
            origin=segmentation.origin,
            spacing=segmentation.spacing,
            direction=segmentation.direction,
        )

    if len(bbox_slices) > 0:
        gray_arr = gray_matter.view() if hasattr(gray_matter, "view") else gray_matter.numpy()
        white_arr = white_matter.view() if hasattr(white_matter, "view") else white_matter.numpy()
        seg = _subarray_to_tensor(seg_arr[bbox_slices], resolved_device)
        gray = _subarray_to_tensor(gray_arr[bbox_slices], resolved_device)
        white = _subarray_to_tensor(white_arr[bbox_slices], resolved_device)
        crop_shape = tuple(int(s.stop - s.start) for s in bbox_slices)
        dir_mat = np.asarray(segmentation.direction, dtype=np.float64)
        start_offset = dir_mat @ (
            np.asarray(spacing_tup, dtype=np.float64)
            * np.asarray([float(s.start) for s in bbox_slices], dtype=np.float64)
        )
        crop_origin = tuple(origin_tup[d] + float(start_offset[d]) for d in range(segmentation.dimension))
        domain = ImageDomain(crop_shape, spacing_tup, crop_origin, identity)
        result = direct_cortical_thickness(
            seg, gray, white, domain,
            iterations=iterations,
            gradient_step=gradient_step,
            smoothing_sigma=smoothing_sigma,
            velocity_smoothing_variance=velocity_smoothing_variance,
            integration_points=integration_points,
            thickness_prior=thickness_prior,
            optimizer=optimizer,
            regularizer=regularizer,
            verbose=verbose,
            full_num_voxels=int(seg_arr.size),
        )
        crop_thick_xyz = np.transpose(
            result.thickness[0, 0].detach().cpu().numpy(),
            tuple(reversed(range(segmentation.dimension))),
        )
        out_arr = np.zeros(segmentation.shape, dtype=np.float32)
        out_arr[bbox_slices] = crop_thick_xyz
        return ants.from_numpy(
            out_arr,
            origin=segmentation.origin,
            spacing=segmentation.spacing,
            direction=segmentation.direction,
        )

    seg = ants_image_to_tensor(segmentation, resolved_device, normalize=False)
    gray = ants_image_to_tensor(gray_matter, resolved_device, normalize=False)
    white = ants_image_to_tensor(white_matter, resolved_device, normalize=False)
    domain = ImageDomain(
        tuple(int(v) for v in segmentation.shape),
        spacing_tup,
        origin_tup,
        identity,
    )
    result = direct_cortical_thickness(
        seg, gray, white, domain,
        iterations=iterations,
        gradient_step=gradient_step,
        smoothing_sigma=smoothing_sigma,
        velocity_smoothing_variance=velocity_smoothing_variance,
        integration_points=integration_points,
        thickness_prior=thickness_prior,
        optimizer=optimizer,
        regularizer=regularizer,
        verbose=verbose,
    )
    return tensor_to_ants_image(result.thickness, segmentation)
