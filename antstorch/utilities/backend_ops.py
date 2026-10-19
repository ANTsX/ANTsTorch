"""Backend-dispatched N4, denoising and registration used by the pipelines.

Each ``run_*`` function calls either the ANTsPy (C++) function or its ANTsTorch
(PyTorch, CUDA/MPS/CPU) counterpart, chosen by
:func:`antstorch.utilities.backend_manager.resolve_backend`.
"""

from __future__ import annotations

import warnings

import ants

from .backend_manager import resolve_backend

# ants.registration transform names -> antstorch.syn.syn_registration names.
# The "Quick" presets map to the nearest ANTsTorch model: the result is
# comparable, not numerically identical, to the ANTs preset.
_SYN_TRANSFORM_MAP = {
    "Translation": "Translation",
    "Rigid": "Rigid",
    "Similarity": "Similarity",
    "Affine": "Affine",
    "SyN": "SyN",
    "SyNOnly": "SyNOnly",
    "antsRegistrationSyNQuick[r]": "Rigid",
    "antsRegistrationSyNQuick[a]": "Affine",
    "antsRegistrationSyNQuick[s]": "SyN",
    "antsRegistrationSyNQuick[t]": "Translation",
    # "Repro" presets only differ from the Quick ones by seeding ANTs; ANTsTorch has
    # no equivalent seed switch, so they map to the same models.
    "antsRegistrationSyNQuickRepro[r]": "Rigid",
    "antsRegistrationSyNQuickRepro[a]": "Affine",
    "antsRegistrationSyNQuickRepro[s]": "SyN",
    "antsRegistrationSyNQuickRepro[t]": "Translation",
}


def run_n4(image, mask=None, *, backend=None, device=None, **kwargs):
    """N4 bias correction. ``kwargs`` (``shrink_factor``, ``convergence``,
    ``rescale_intensities``, ``return_bias_field``, ``verbose``) are accepted
    by both implementations."""
    if resolve_backend(backend, "n4") == "antstorch":
        from .n4_bias_field_correction import n4_bias_field_correction

        return n4_bias_field_correction(image, mask, device=device, **kwargs)
    return ants.n4_bias_field_correction(image, mask, **kwargs)


def run_denoise(image, mask=None, shrink_factor=1, *, backend=None, device=None):
    """Non-local-means denoising."""
    if resolve_backend(backend, "denoise") == "antstorch":
        from .denoise_image import denoise_image

        return denoise_image(image, mask, shrink_factor=shrink_factor, device=device)
    return ants.denoise_image(image, mask, shrink_factor=shrink_factor)


def run_registration(fixed, moving, type_of_transform, *, backend=None, device=None, verbose=False):
    """Registration returning ``warpedmovout``, ``fwdtransforms`` and
    ``invtransforms`` (transform files usable with ``ants.apply_transforms``).
    A transform type ANTsTorch cannot express falls back to ANTs with a warning."""
    if resolve_backend(backend, "registration") == "antstorch":
        mapped = _SYN_TRANSFORM_MAP.get(type_of_transform)
        if mapped is None:
            warnings.warn(
                f"type_of_transform={type_of_transform!r} has no antstorch equivalent; "
                "using ants.registration for this call.",
                RuntimeWarning,
            )
        else:
            from ..syn import syn_registration

            return syn_registration(
                fixed.clone("float"), moving.clone("float"),
                type_of_transform=mapped, device=device, verbose=verbose,
            )
    return ants.registration(
        fixed=fixed, moving=moving, type_of_transform=type_of_transform, verbose=verbose
    )


def run_kelly_kapowski(
    segmentation, gray_matter, white_matter, *, iterations=45, gradient_step=0.025,
    velocity_smoothing_variance=1.5, thickness_prior=None, backend=None, device=None, verbose=False,
):
    """DiReCT cortical thickness. The ANTs arguments ``its``, ``r`` and ``m`` map to
    ``iterations``, ``gradient_step`` and ``velocity_smoothing_variance`` (the same
    mapping as ``cortical_thickness2``); convergence monitoring is disabled
    (``x=0``) in the ANTs call.  ``thickness_prior`` (ANTs ``t``) is left to each
    implementation's own default (10) when None.  ANTsTorch's DiReCT agrees with KellyKapowski
    but is not voxelwise identical (see antstorch/direct/README.md)."""
    if resolve_backend(backend, "kelly_kapowski") == "antstorch":
        from ..direct import kelly_kapowski

        extra = {} if thickness_prior is None else {"thickness_prior": float(thickness_prior)}
        return kelly_kapowski(
            segmentation, gray_matter, white_matter,
            iterations=iterations, gradient_step=gradient_step,
            velocity_smoothing_variance=velocity_smoothing_variance,
            device=device, verbose=verbose, **extra,
        )
    extra = {} if thickness_prior is None else {"t": thickness_prior}
    return ants.kelly_kapowski(
        s=segmentation, g=gray_matter, w=white_matter, its=iterations,
        r=gradient_step, m=velocity_smoothing_variance, x=0, verbose=int(verbose), **extra,
    )
