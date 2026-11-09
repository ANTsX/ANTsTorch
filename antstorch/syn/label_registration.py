"""Label-guided, multi-metric SyN registration on CPU, CUDA, or MPS."""

import os
import re

import ants
import numpy as np
import torch

from ..ants_transform_io import read_affine_transform


def _metric_name(name):
    name = name.lower()
    name = {"msq": "mse", "meansquares": "mse", "mutualinformation": "mi"}.get(name, name)
    if name not in ("mse", "cc", "lncc", "cc2", "lncc2", "mattes", "mi", "dice"):
        raise ValueError(f"Unsupported multivariate metric: {name}")
    return name


def _validate_image_domain(image, reference, name):
    if not ants.is_image(image) or image.has_components or image.dimension != reference.dimension:
        raise ValueError(f"{name} must be a scalar ANTsImage of the reference dimension")
    if (image.shape != reference.shape or not np.allclose(image.spacing, reference.spacing)
            or not np.allclose(image.origin, reference.origin)
            or not np.allclose(image.direction, reference.direction)):
        raise ValueError(f"{name} must share its reference image's physical domain")
    if not np.isfinite(image.numpy()).all():
        raise ValueError(f"{name} contains non-finite values")


def _images(value, name, optional=False):
    if value is None and optional:
        return []
    values = [value] if ants.is_image(value) else list(value) if isinstance(value, (list, tuple)) else []
    if not values or any(not ants.is_image(v) or v.has_components for v in values):
        raise ValueError(f"{name} must contain scalar ANTsImages")
    return values


def _centroid(image, label):
    index = np.argwhere(image.numpy() == label).mean(axis=0)
    return np.asarray(image.origin) + np.asarray(image.direction) @ (index * np.asarray(image.spacing))


def _landmark_affine(fixed, moving, transform):
    """Fit the physical fixed-to-moving pull map consumed by SyN/ITK."""
    d = fixed.shape[1]
    x, y = fixed - fixed.mean(0), moving - moving.mean(0)
    required_rank = d if transform == "affine" else d - 1
    if len(fixed) < 3 or np.linalg.matrix_rank(x) < required_rank or np.linalg.matrix_rank(y) < required_rank:
        raise ValueError(f"Insufficient nondegenerate common-label centroids for {transform} initialization")
    if transform == "affine":
        matrix = np.linalg.lstsq(x, y, rcond=None)[0].T
    else:
        u, singular, vt = np.linalg.svd(x.T @ y)
        signs = np.ones(d)
        signs[-1] = np.linalg.det(vt.T @ u.T)
        matrix = vt.T @ np.diag(signs) @ u.T
        if transform == "similarity":
            matrix *= np.sum(singular * signs) / np.sum(x * x)
    if abs(np.linalg.det(matrix)) < 1e-10:
        raise ValueError("Label centroids produce a singular affine")
    return matrix, moving.mean(0) - matrix @ fixed.mean(0)


def label_image_registration(
    fixed_label_images,
    moving_label_images,
    fixed_intensity_images=None,
    moving_intensity_images=None,
    fixed_mask=None,
    moving_mask=None,
    initial_transforms="affine",
    type_of_deformable_transform="antsRegistrationSyNQuick[so]",
    label_image_weighting=1.0,
    output_prefix="",
    verbose=False,
    *,
    device=None,
    dtype=torch.float32,
    **syn_options,
):
    """Register corresponding segmentations, optionally with intensity images.

    Nonzero integer IDs shared by each label-image pair become separate
    binary MeanSquares metrics. Weights are per label-image pair (each label
    receives that weight); optional intensity pairs each receive weight 1.
    Linear initialization fits common-label centroids in physical coordinates.
    Inputs are scalar 2-D/3-D ANTsImages; all fixed images must share one
    domain, and all moving images another. Background ID 0 is ignored.

    ``initial_transforms`` accepts identity/None, rigid/similarity/affine,
    or a file/list of affine .mat transforms in ANTs application order.
    Previously computed nonlinear initialization is not supported.
    ``type_of_deformable_transform`` accepts the antsRegistrationSyN[so/bo]
    family (including Quick/Repro names), a seven-element ANTsPy-style SyN
    specification, or None for linear-only registration. Repro names select
    a deterministic full-sampling objective, not ANTs C++ numerical parity.

    Extra keyword arguments configure :func:`syn_registration` (e.g.
    levels, reg_iterations, regularizer, optimizer, flow_sigma). CUDA/MPS
    execute the deformable optimization; centroid fitting and image/transform
    I/O run on CPU. Masks restrict the deformable metrics. Binary masks are
    interpolated linearly during optimization, not by nearest neighbor.
    Memory use grows with the number of labels and their image sizes.

    Returns the usual ANTsTorch SyN result, including ANTs-compatible
    fwdtransforms/invtransforms, loss histories, and common_label_ids.
    Use ``ants.apply_transforms(..., interpolator='genericLabel')`` to
    resample the original segmentations without introducing fractional IDs.
    """
    from .syn import syn_registration

    fixed = _images(fixed_label_images, "fixed_label_images")
    moving = _images(moving_label_images, "moving_label_images")
    fi = _images(fixed_intensity_images, "fixed_intensity_images", True)
    mi = _images(moving_intensity_images, "moving_intensity_images", True)
    if len(fixed) != len(moving) or len(fi) != len(mi):
        raise ValueError("The number of fixed and moving image pairs must match")
    d = fixed[0].dimension
    if d not in (2, 3) or moving[0].dimension != d:
        raise ValueError("Label registration requires matching 2-D or 3-D images")
    for values, reference in ((fixed + fi, fixed[0]), (moving + mi, moving[0])):
        for image in values:
            _validate_image_domain(image, reference, "input image")
    for value, reference in ((fixed_mask, fixed[0]), (moving_mask, moving[0])):
        if value is not None:
            _validate_image_domain(value, reference, "metric mask")
            if not np.any(value.numpy()):
                raise ValueError("Metric masks must not be empty")
    for image in fixed + moving:
        array = image.numpy()
        if np.any(array < 0) or np.any(array != np.floor(array)) or np.any(array > 2**24):
            raise ValueError("Labels must be nonnegative integers representable exactly in float32 (<= 2^24)")
    weights = ([float(label_image_weighting)] * len(fixed) if np.isscalar(label_image_weighting)
               else list(label_image_weighting))
    if len(weights) != len(fixed) or not np.isfinite(weights).all() or np.any(np.asarray(weights) < 0):
        raise ValueError("label_image_weighting must give finite nonnegative weights for each label-image pair")
    common = []
    fixed_centers, moving_centers, metrics = [], [], []
    for a, b, weight in zip(fixed, moving, weights):
        ids = np.intersect1d(np.unique(a.numpy()), np.unique(b.numpy()))
        ids = ids[ids != 0]
        if not len(ids):
            raise ValueError("No common nonzero labels for an image pair")
        common.append(ids.tolist())
        for label in ids:
            fixed_centers.append(_centroid(a, label))
            moving_centers.append(_centroid(b, label))
            if weight:
                metrics.append(("MSQ", a == label, b == label, float(weight), 0))
    matrix, translation = np.eye(d), np.zeros(d)
    if isinstance(initial_transforms, str) and initial_transforms in ("rigid", "similarity", "affine"):
        matrix, translation = _landmark_affine(np.asarray(fixed_centers), np.asarray(moving_centers), initial_transforms)
    elif initial_transforms is not None and not (isinstance(initial_transforms, str) and initial_transforms.lower() == "identity"):
        files = initial_transforms if isinstance(initial_transforms, (tuple, list)) else [initial_transforms]
        for path in reversed(files):
            if not isinstance(path, (str, os.PathLike)) or not str(path).endswith(".mat"):
                raise ValueError("initial_transforms supports affine .mat files only")
            m, t = read_affine_transform(str(path), d)
            matrix, translation = m @ matrix, m @ translation + t

    options = dict(levels=(8, 4, 2, 1), reg_iterations=(100, 70, 50, 20),
                   grad_step=0.2, syn_metric="mse", neighborhood_radius=0)
    deformable = type_of_deformable_transform is not None and type_of_deformable_transform != ""
    intensity_metric, parameter = "CC", 2
    if deformable:
        if isinstance(type_of_deformable_transform, (list, tuple)):
            spec = type_of_deformable_transform
            if len(spec) != 7 or spec[0] not in ("SyN", "BSplineSyN"):
                raise ValueError("Expected a seven-element SyN or BSplineSyN specification")
            options.update(grad_step=spec[1], reg_iterations=spec[4], smoothing_sigmas=spec[5], levels=spec[6])
            if spec[0] == "BSplineSyN":
                options["regularizer"] = "bspline"
            intensity_metric, parameter = spec[2:4]
        elif isinstance(type_of_deformable_transform, str):
            match = re.fullmatch(r"antsRegistrationSyN(?:Quick)?(?:Repro)?\[(so|bo)\]", type_of_deformable_transform)
            if match is None:
                raise ValueError("Only deformable-only antsRegistrationSyN*[so] or [bo] presets are supported")
            if match[1] == "bo":
                options["regularizer"] = "bspline"
            if "Quick" in type_of_deformable_transform:
                options["reg_iterations"] = (100, 70, 50, 0)
                intensity_metric, parameter = "MI", 32
        else:
            raise ValueError("Invalid type_of_deformable_transform")
    metrics.extend((intensity_metric, a, b, 1.0, parameter) for a, b in zip(fi, mi))
    if deformable and not metrics:
        raise ValueError("At least one label or intensity metric must have positive weight")
    reserved = {"initial_affine", "type_of_transform", "multivariate_extras", "primary_metric_weight", "mask", "moving_mask", "outprefix"}
    if reserved.intersection(syn_options):
        raise ValueError(f"Use label_image_registration arguments instead of: {sorted(reserved.intersection(syn_options))}")
    options.update(syn_options)
    if device is None:
        from ..utilities.device_manager import get_user_default_device
        from .core.pipeline import auto_detect_device
        device = get_user_default_device()
        if device is None:
            device = auto_detect_device(requested_device=None)
    result = syn_registration(
        fixed[0], moving[0], type_of_transform="SyNOnly" if deformable else "Affine",
        initial_affine=(torch.as_tensor(matrix), torch.as_tensor(translation)),
        multivariate_extras=metrics, primary_metric_weight=0,
        mask=fixed_mask, moving_mask=moving_mask, outprefix=output_prefix,
        device=device, dtype=dtype, verbose=verbose, **options,
    )
    result["common_label_ids"] = common
    result["provenance"].update(
        label_image_registration=True,
        label_image_weighting=[float(w) for w in weights],
        intensity_image_pairs=len(fi),
        type_of_deformable_transform=type_of_deformable_transform,
    )
    return result
