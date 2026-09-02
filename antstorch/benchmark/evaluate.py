"""antstorch.benchmark.evaluate — Standardized Single-Pair Registration & Metric Evaluation
=============================================================================================

The ANTsTorch-native core of the Mindboggle-101 registration benchmark,
ported from ``syntx.benchmark.evaluate`` (see the project doc, "Portage de
l'évaluation Mindboggle-101 dans ANTsTorch") and restricted, by explicit
decision, to registration arms that come from ANTsTorch itself:

- ``antstorch.syn.syn_registration(..., regularizer=...)`` — model strings
  ``'gaussian_syn'``, ``'sobolev_syn'``, ``'dsti_syn'``, and ``'bspline_syn'``
  (the ANTs/ITK ``BSplineSyN`` regularizer). The ``_syn`` suffix is
  deliberate: it distinguishes these four dense-SyN-stage variants from
  ``'bspline_svf'`` below, which is a different transformation family
  entirely (a stationary velocity field) despite the two sharing the word
  "bspline".
- ``antstorch.bspline_flows.bspline_svf_registration()`` — the cubic
  B-spline stationary-velocity-field model (``'bspline_svf'``/``'svf'``).
- ``antstorch.bspline_flows.gaussian_svf_registration()`` — the dense
  Gaussian-regularized stationary-velocity-field model (``'gaussian_svf'``).
- Traditional (non-ANTsTorch) ANTs baselines, evaluated via a direct
  ``ants.registration(type_of_transform=...)`` call rather than any
  ANTsTorch model (``_ANTS_TRADITIONAL_MODELS`` below) — added later than
  the arms above, so that this harness can report ANTs' own registration
  quality on the identical pairs/canonical-affine/metrics pipeline used for
  every ANTsTorch arm, not just the ANTsTorch-native transform families.

This intentionally omits the ``syntx``-only capabilities the earlier
extension of ``syntx.benchmark`` (see ``syntx.benchmark.antstorch_arms``)
left in ``syntx`` itself and did not attempt to reproduce here: the
Time-Varying Velocity Field (TVF) and geodesic-shooting (SyNGS) transform
families, the JAX backend, and the deep-feature similarity losses have no
ANTsTorch equivalent and are not planned for one — see the project doc for
the explicit list of capabilities left out of this port, by choice.

Every model variant for a given pair shares the exact same canonical affine
initialization (fit once via ``syn_registration(type_of_transform="Affine")``
and cached to disk), so results across models stay apples-to-apples —
mirroring the fairness invariant the ``syntx`` harness already established
and that this port preserves.
"""

import os
import sys
import time
import json
from typing import Any, Dict, Optional

import numpy as np
import torch
import ants

from antstorch.benchmark.data import load_mindboggle_pair
from antstorch.benchmark.metrics import compute_bidirectional_dice, compute_jacobian_metrics
from antstorch.ants_transform_io import read_affine_transform
from antstorch.syn import syn_registration

# The '_syn' suffix on every key here is deliberate, not decorative: it
# disambiguates these dense-SyN-stage model names from _BSPLINE_SVF_MODELS
# below, which is a different transformation family (a stationary velocity
# field, not SyN) -- 'bspline_syn' and 'bspline_svf' would otherwise be easy
# to confuse for the same thing.
_SYN_REGULARIZERS = {
    "gaussian_syn": "gaussian",
    "sobolev_syn": "sobolev",
    "dsti_syn": "dsti",
    "bspline_syn": "bspline",
}
_BSPLINE_SVF_MODELS = ("bspline_svf", "svf")
_GAUSSIAN_SVF_MODELS = ("gaussian_svf",)

# Traditional (non-ANTsTorch) ANTs baselines, run via a direct
# ants.registration(type_of_transform=...) call (see _run_ants_traditional()
# below) rather than any antstorch.syn/antstorch.bspline_flows model. Each
# entry maps a benchmark model name to the exact ants.registration
# type_of_transform string to use.
#
# Every entry here MUST use a deformable-only preset ("SyNOnly", or an
# "antsRegistrationSyN*[so|bo]"-style suffix) applied on top of this
# harness's shared canonical affine (initial_transform=[affine_path],
# fit once by _fit_or_load_canonical_affine() and reused by every model
# variant for a given pair) -- never a preset that recomputes its own
# rigid/affine stage (e.g. the plain "[s]"/"[b]" full-pipeline variants),
# since that would silently break the fairness invariant this harness
# otherwise guarantees (every model variant registered from the exact same
# affine initialization). Add a new preset here, once its "...[so]"/
# "...[bo]" (or equivalent deformable-only) suffix has been confirmed, and
# it is available with zero changes to the dispatch logic in
# evaluate_mindboggle_pair() below.
_ANTS_TRADITIONAL_MODELS = {
    "ants_syn_quick": "antsRegistrationSyNQuick[so]",
}


def _is_deformable_only_ants_transform(type_of_transform: str) -> bool:
    """True for an ``ants.registration()`` ``type_of_transform`` string that
    performs no separate rigid/affine stage of its own: either the literal
    ``"SyNOnly"``, or an ``"antsRegistrationSyN*[x]"``-style preset whose
    bracketed suffix ends in ``"o"`` (the deformable-only ``"...[so]"``/
    ``"...[bo]"`` presets -- see the ``type_of_transform`` options table in
    ``ants.registration.__doc__``). Guards the shared-canonical-affine
    fairness invariant (see ``_ANTS_TRADITIONAL_MODELS`` above) against both
    dict entries and any raw ``type_of_transform`` string passed directly as
    ``model`` (see ``evaluate_mindboggle_pair()``'s dispatch below) -- a
    preset that fails this check would recompute its own rigid/affine stage
    on top of the shared affine, silently breaking that invariant.
    """
    if type_of_transform == "SyNOnly":
        return True
    return "[" in type_of_transform and type_of_transform.rstrip("]").endswith("o")

# One benchmark schedule for every transformation/regularizer family.
DEFAULT_REG_ITERATIONS = (100, 100, 50, 10)
DEFAULT_REGISTRATION_LEVELS = (8, 4, 2, 1)
DEFAULT_REGISTRATION_SMOOTHING_SIGMAS = (3.0, 2.0, 1.0, 0.0)


def clean_device_cache():
    """Clears PyTorch GPU/Apple Silicon MPS memory allocator cache and runs garbage collection."""
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        try:
            torch.mps.empty_cache()
        except Exception:
            pass


def _fit_or_load_canonical_affine(fi, mi, pair_idx, canonical_affine_dir, device, verbose):
    """Fits (once, cached to disk) the affine every model variant for this
    pair will share, via ``syn_registration(type_of_transform="Affine")``.

    Returns
    -------
    matrix, translation : torch.Tensor
        ITK ``(x, y[, z])``-order affine parameters, ready to pass as
        ``initial_affine`` to either ``syn_registration()`` or
        ``bspline_svf_registration()``.
    runtime_seconds : float
        Time spent fitting (0.0 when loaded from cache).
    affine_path : str
        Path to the written/cached ``...0GenericAffine.mat`` file.
    """
    os.makedirs(canonical_affine_dir, exist_ok=True)
    outprefix = os.path.join(canonical_affine_dir, f"pair_{pair_idx:03d}_")
    affine_path = f"{outprefix}0GenericAffine.mat"
    dimension = fi.dimension

    if os.path.exists(affine_path):
        matrix_np, translation_np = read_affine_transform(affine_path, dimension)
        matrix = torch.from_numpy(matrix_np).to(dtype=torch.float32)
        translation = torch.from_numpy(translation_np).to(dtype=torch.float32)
        return matrix, translation, 0.0, affine_path

    t0 = time.time()
    aff_res = syn_registration(
        fixed=fi, moving=mi,
        type_of_transform="Affine",
        outprefix=outprefix,
        device=device,
        verbose=verbose,
    )
    runtime_seconds = time.time() - t0
    matrix = aff_res["affine_matrix"].to(dtype=torch.float32)
    translation = aff_res["affine_translation"].to(dtype=torch.float32)
    return matrix, translation, runtime_seconds, affine_path


# ants.registration() top-level keyword arguments this harness forwards from
# **kwargs when present -- never forced to a harness-wide default, since
# antsRegistrationSyN*[x] presets recreate the antsRegistrationSyN(Quick).sh
# scripts' own internal multi-resolution schedules, which have not been
# verified compatible with DEFAULT_REG_ITERATIONS (calibrated for
# antstorch's own dense-SyN implementation, a different codepath entirely).
_ANTS_TRADITIONAL_FORWARDED_KWARGS = (
    "reg_iterations", "syn_metric", "syn_sampling", "grad_step",
    "flow_sigma", "total_sigma", "aff_metric", "aff_sampling",
    "aff_iterations", "aff_shrink_factors", "aff_smoothing_sigmas",
)


def _run_ants_traditional(fi, mi, affine_path, type_of_transform, *, outprefix=None,
                           verbose=False, **kwargs):
    """Runs a traditional (non-ANTsTorch) ANTs baseline via a direct
    ``ants.registration()`` call, on top of this harness's shared canonical
    affine.

    Unlike ``_run_bspline_svf()``/``_run_gaussian_svf()``, no composition-
    order fixup (project doc, § 30) is needed here: this *is* the real
    ``ants.registration()``/``ants.apply_transforms()`` pipeline, so the
    transform files it writes are, by construction, exactly what
    ``ants.apply_transforms()`` expects -- confirmed directly against real
    Mindboggle data (inter-subject pair 88): ``fwdtransforms`` comes back as
    ``[warp, affine]`` and ``invtransforms`` as ``[affine, inverse_warp]``,
    the same two-piece convention ``evaluate_mindboggle_pair()`` already
    defaults ``whichtoinvert_inv`` to (``[True, False]``) when a model
    doesn't supply its own.

    Parameters
    ----------
    fi, mi : ants.ANTsImage
        Fixed/moving intensity images.
    affine_path : str
        Path to the harness's shared canonical ``...0GenericAffine.mat``
        (from ``_fit_or_load_canonical_affine()``), prepended via
        ``initial_transform=[affine_path]`` so every model variant for this
        pair starts from the identical affine -- the fairness invariant
        this harness otherwise guarantees for every ANTsTorch arm.
    type_of_transform : str
        The exact ``ants.registration(type_of_transform=...)`` string (e.g.
        ``"antsRegistrationSyNQuick[so]"``) -- see ``_ANTS_TRADITIONAL_MODELS``.
    outprefix : str, optional
        Forwarded to ``ants.registration()`` when given, so its transform
        files land under ``registration_output_dir`` like every other model.
    **kwargs
        Any of ``_ANTS_TRADITIONAL_FORWARDED_KWARGS``, forwarded to
        ``ants.registration()`` only when explicitly given (see that
        tuple's docstring above for why nothing is defaulted here).

    Returns
    -------
    dict
        ``ants.registration()``'s own return dict, unmodified (already has
        ``warpedmovout``/``fwdtransforms``/``invtransforms``;
        ``evaluate_mindboggle_pair()`` falls back to
        ``whichtoinvert_inv=[True, False]`` and ``loss_history=None`` when
        those keys are absent, which is correct for this model family).
    """
    ants_kwargs = {k: v for k, v in kwargs.items() if k in _ANTS_TRADITIONAL_FORWARDED_KWARGS}
    if outprefix is not None:
        ants_kwargs["outprefix"] = outprefix
    return ants.registration(
        fixed=fi, moving=mi,
        type_of_transform=type_of_transform,
        initial_transform=[affine_path],
        verbose=verbose,
        **ants_kwargs,
    )


def _run_bspline_svf(fi, mi, matrix, translation, *, device, reg_iterations=None,
                      shrink_factors=DEFAULT_REGISTRATION_LEVELS,
                      smoothing_sigmas=DEFAULT_REGISTRATION_SMOOTHING_SIGMAS,
                      mesh_size=None, spline_distance=None, learning_rate=0.01,
                      optimizer="physical_gradient_descent",
                      gradient_step=0.2, similarity="ants_ncc", neighborhood_radius=4,
                      outprefix=None, verbose=False) -> Dict[str, Any]:
    """Adapts ``bspline_flows.bspline_svf_registration()``'s in-memory tensor
    output into the file-based ``fwdtransforms``/``invtransforms``/
    ``warpedmovout`` contract the rest of this module expects — the same
    adapter pattern used in ``syntx.benchmark.antstorch_arms.
    run_antstorch_bspline_svf``, simplified here since ``matrix``/
    ``translation`` are already in-memory tensors (no round trip through a
    ``.mat`` file needed, unlike the cross-package version)."""
    from antstorch.ants_transform_io import build_transform_lists, default_outprefix, write_affine_transform
    from antstorch.bspline_flows import (
        ImageDomain,
        affine_displacement_field,
        bspline_svf_registration,
        compose_displacements,
        warp_image,
    )
    from antstorch.syn.bridge import (
        ants_image_metadata,
        ants_image_to_tensor,
        displacement_xyz_to_ants_image,
        tensor_to_ants_image,
    )

    resolved_device = torch.device(device)
    dtype = torch.float32
    dimension = fi.dimension

    matrix = matrix.to(device=resolved_device, dtype=dtype)
    translation = translation.to(device=resolved_device, dtype=dtype)

    fixed_meta = ants_image_metadata(fi)
    moving_meta = ants_image_metadata(mi)
    fixed_domain = ImageDomain(fixed_meta["shape"], fixed_meta["spacing"], fixed_meta["origin"], fixed_meta["direction"])
    moving_domain = ImageDomain(moving_meta["shape"], moving_meta["spacing"], moving_meta["origin"], moving_meta["direction"])

    # normalize=True (the default): unlike the syntx-side arm, nothing has
    # pre-normalized fi/mi before this point, and bspline_svf_registration()
    # itself does no normalization internally (its documented tensor-native
    # scope leaves that to the caller) -- so this adapter must do it,
    # matching the percentile-clip normalization every other path in this
    # module gets for free from syn_registration()'s own internal
    # ants_image_to_tensor() calls.
    fixed_tensor = ants_image_to_tensor(fi, resolved_device, dtype, normalize=True)
    moving_tensor = ants_image_to_tensor(mi, resolved_device, dtype, normalize=True)

    iterations = list(reg_iterations) if reg_iterations is not None else list(DEFAULT_REG_ITERATIONS)
    if len(iterations) != len(shrink_factors):
        raise ValueError("reg_iterations must have one value per shrink factor")

    result = bspline_svf_registration(
        fixed=fixed_tensor,
        moving=moving_tensor,
        fixed_domain=fixed_domain,
        moving_domain=moving_domain,
        mesh_size=mesh_size,
        spline_distance=spline_distance,
        shrink_factors=tuple(shrink_factors),
        smoothing_sigmas=tuple(smoothing_sigmas),
        iterations=iterations,
        learning_rate=learning_rate,
        optimizer=optimizer,
        gradient_step=gradient_step,
        similarity=similarity,
        neighborhood_radius=neighborhood_radius,
        initial_affine=(matrix, translation),
        padding_mode="border",
        stationary_boundary=True,
        verbose=verbose,
    )

    # bspline_svf_registration() is tensor-native and never normalizes/
    # denormalizes internally (by design, see its docstring) -- so
    # result["warpedmovout"] is the *normalized* moving_tensor rewarped by
    # the optimized field, not the original image intensity range. Rebuild
    # warpedmovout from the un-normalized moving image instead, composing the
    # (discarded) affine displacement with the pure SVF field exactly as
    # DeterministicBSplineRegistration.transform() does internally
    # (antstorch/bspline_flows/deterministic_registration.py) -- matching
    # syn_registration()'s own pattern (antstorch/syn/syn.py) of always
    # reconstructing its returned warpedmovout from non-normalized tensors.
    # padding_mode="zeros" here (not "border", unlike the internal
    # bspline_svf_registration() call above, whose own padding_mode governs
    # the optimization/training and is left untouched) -- matching both
    # syn_registration()'s own default (antstorch/syn/syn.py, padding_mode:
    # str = "zeros") and, critically, ants.apply_transforms()'s own
    # out-of-domain extrapolation (which always returns 0, with no "clamp to
    # edge" option). fwdtransforms/invtransforms (below) are handed back as
    # real transform files and reused verbatim by evaluate_mindboggle_pair()
    # for the warped label map (§ 28) and by compute_bidirectional_dice() --
    # both go through ants.apply_transforms(), so warpedmovout must
    # extrapolate the exact same way or the three outputs disagree exactly
    # in the out-of-domain margin (visible as intensity looking "fine" via a
    # border-clamped continuation while the label map goes to background
    # there -- the bug reported after § 28).
    moving_tensor_raw = ants_image_to_tensor(mi, resolved_device, dtype, normalize=False)
    affine_displacement = affine_displacement_field(matrix, translation, fixed_domain, fixed_tensor)
    composed_displacement = compose_displacements(affine_displacement, result["fwdtransforms"], fixed_domain)
    warpedmovout_tensor = warp_image(
        moving_tensor_raw, composed_displacement, fixed_domain, moving_domain, padding_mode="zeros"
    )

    # § 30 fix: write the *fully composed* forward/inverse fields, not the
    # pure SVF piece alone next to a separate affine file. Empirically
    # verified (see the project doc) that ants.apply_transforms()'s real
    # composite-transform convention for transformlist=[warp_path,
    # affine_path] is "apply the warp field first (evaluated at the
    # untouched fixed-space point), THEN the affine (rotating the
    # accumulated displacement vector)" -- i.e. the *opposite* order from
    # what this adapter's own compose_displacements(affine, svf) computes
    # for warpedmovout ("affine first, then the SVF flow evaluated at the
    # affine-shifted point", matching bspline_svf_registration()'s own
    # internal training-time convention). Those two conventions are both
    # internally consistent but different -- reusing the raw pure-SVF field
    # as "1Warp.nii.gz" next to the affine silently mismatches what
    # ants.apply_transforms() reconstructs from it, which is invisible on
    # the smooth warpedmovout intensity image but produces a visibly
    # misaligned warped_moving_labels.nii.gz / dice_moving (the bug reported
    # after § 29). Writing the single, already-composed field sidesteps the
    # convention mismatch entirely: ants.apply_transforms() with a
    # single-element transformlist just adds the field directly, no
    # re-composition with a separate affine involved, so it reproduces
    # exactly the same mapping warpedmovout was built from.
    #
    # The inverse field is built the mirror-image way: compose_displacements
    # expects (first, second) with first(x) + second(x + first(x)); solving
    # forward(p) = affine(p) + svf(affine(p)) for its algebraic inverse
    # gives inverse(y) = affine_inverse(y + svf_inverse(y)) -- i.e.
    # first=svf_inverse, second=affine_inverse (order swapped relative to
    # the forward composition). Verified empirically via a forward/inverse
    # round-trip on real registration output (max residual ~0.04 mm on a
    # 1 mm grid -- see the project doc).
    inverse_matrix = torch.linalg.inv(matrix)
    inverse_translation = -torch.einsum("ij,j->i", inverse_matrix, translation)
    affine_inverse_displacement = affine_displacement_field(
        inverse_matrix, inverse_translation, fixed_domain, fixed_tensor
    )
    composed_inverse_displacement = compose_displacements(
        result["invtransforms"], affine_inverse_displacement, fixed_domain
    )

    outprefix = outprefix or default_outprefix()
    warp_path = f"{outprefix}1Warp.nii.gz"
    inverse_warp_path = f"{outprefix}1InverseWarp.nii.gz"
    affine_path = f"{outprefix}0GenericAffine.mat"

    ants.image_write(displacement_xyz_to_ants_image(composed_displacement, fi), warp_path)
    ants.image_write(displacement_xyz_to_ants_image(composed_inverse_displacement, fi), inverse_warp_path)
    # affine_path is still written out for transparency/debugging (the
    # canonical affine this pair's SVF field was fit against), but is no
    # longer included in fwdtransforms/invtransforms below -- both are now
    # single, fully composed fields.
    write_affine_transform(matrix.detach().cpu(), translation.detach().cpu(), dimension, affine_path)

    fwdtransforms, invtransforms = build_transform_lists(
        affine_path=None, warp_path=warp_path, inverse_warp_path=inverse_warp_path
    )

    return {
        "warpedmovout": tensor_to_ants_image(warpedmovout_tensor, fi),
        "fwdtransforms": fwdtransforms,
        "invtransforms": invtransforms,
        # Single-element lists now (no separate affine piece) -- nothing to
        # invert on apply, matching build_transform_lists()'s own
        # affine_path=None convention.
        "whichtoinvert_inv": [False],
        # bspline_svf_registration()'s own loss_history/level_loss_history
        # (return_loss_history=True by default) -- flat per-iteration loss
        # list and the same values grouped per pyramid level, both plain
        # Python floats already, so no tensor conversion is needed before
        # this reaches evaluate_mindboggle_pair()'s JSON persistence below.
        "loss_history": result.get("loss_history"),
        "level_loss_history": result.get("level_loss_history"),
    }


def _run_gaussian_svf(
    fi, mi, matrix, translation, *, device, reg_iterations=None,
    shrink_factors=DEFAULT_REGISTRATION_LEVELS,
    smoothing_sigmas=DEFAULT_REGISTRATION_SMOOTHING_SIGMAS,
    gradient_step=0.2, momentum=0.0, update_field_sigma=3.0,
    total_field_sigma=0.5, similarity="ants_ncc", neighborhood_radius=4,
    velocity_weight=0.0, bending_weight=0.0, squaring_steps=7,
    outprefix=None, verbose=False,
) -> Dict[str, Any]:
    """Adapt dense Gaussian SVF output to the benchmark transform contract."""
    from antstorch.ants_transform_io import build_transform_lists, default_outprefix, write_affine_transform
    from antstorch.bspline_flows import (
        ImageDomain,
        PhysicalGradientDescent,
        affine_displacement_field,
        compose_displacements,
        gaussian_svf_registration,
        warp_image,
    )
    from antstorch.syn.bridge import (
        ants_image_metadata,
        ants_image_to_tensor,
        displacement_xyz_to_ants_image,
        tensor_to_ants_image,
    )

    resolved_device = torch.device(device)
    dtype = torch.float32
    dimension = fi.dimension
    matrix = matrix.to(device=resolved_device, dtype=dtype)
    translation = translation.to(device=resolved_device, dtype=dtype)
    fixed_meta = ants_image_metadata(fi)
    moving_meta = ants_image_metadata(mi)
    fixed_domain = ImageDomain(
        fixed_meta["shape"], fixed_meta["spacing"], fixed_meta["origin"], fixed_meta["direction"]
    )
    moving_domain = ImageDomain(
        moving_meta["shape"], moving_meta["spacing"], moving_meta["origin"], moving_meta["direction"]
    )
    fixed_tensor = ants_image_to_tensor(fi, resolved_device, dtype, normalize=True)
    moving_tensor = ants_image_to_tensor(mi, resolved_device, dtype, normalize=True)
    iterations = list(reg_iterations) if reg_iterations is not None else list(DEFAULT_REG_ITERATIONS)
    if len(iterations) != len(shrink_factors):
        raise ValueError("reg_iterations must have one value per shrink factor")

    result = gaussian_svf_registration(
        fixed=fixed_tensor,
        moving=moving_tensor,
        fixed_domain=fixed_domain,
        moving_domain=moving_domain,
        shrink_factors=tuple(shrink_factors),
        smoothing_sigmas=tuple(smoothing_sigmas),
        iterations=iterations,
        optimizer=PhysicalGradientDescent(gradient_step=gradient_step, momentum=momentum),
        update_field_sigma=update_field_sigma,
        total_field_sigma=total_field_sigma,
        similarity=similarity,
        neighborhood_radius=neighborhood_radius,
        velocity_weight=velocity_weight,
        bending_weight=bending_weight,
        squaring_steps=squaring_steps,
        initial_affine=(matrix, translation),
        padding_mode="border",
        stationary_boundary=True,
        verbose=verbose,
    )

    # Same fix as _run_bspline_svf() above: gaussian_svf_registration() is
    # also tensor-native with no internal normalization, so
    # result["warpedmovout"] is the normalized moving_tensor rewarped by the
    # optimized field. Rebuild it from the un-normalized moving image,
    # composing the affine displacement with the pure SVF field exactly as
    # DeterministicGaussianRegistration's own internal transform step does
    # (antstorch/bspline_flows/gaussian_svf_registration.py). padding_mode=
    # "zeros" (not "border") to match syn_registration()'s own default and
    # ants.apply_transforms()'s out-of-domain extrapolation (always 0, no
    # edge-clamp option) -- see the matching comment in _run_bspline_svf()
    # for why this must agree with the warped label map / Dice computation,
    # which both go through ants.apply_transforms() on the same fwdtransforms.
    moving_tensor_raw = ants_image_to_tensor(mi, resolved_device, dtype, normalize=False)
    affine_displacement = affine_displacement_field(matrix, translation, fixed_domain, fixed_tensor)
    composed_displacement = compose_displacements(affine_displacement, result["fwdtransforms"], fixed_domain)
    warpedmovout_tensor = warp_image(
        moving_tensor_raw, composed_displacement, fixed_domain, moving_domain, padding_mode="zeros"
    )

    # § 30 fix -- see the matching, fully-commented block in
    # _run_bspline_svf() above for the empirically-verified rationale: write
    # the fully composed forward/inverse fields (matching the convention
    # ants.apply_transforms() actually uses for a composite transform list),
    # not the pure SVF piece next to a separate affine file.
    inverse_matrix = torch.linalg.inv(matrix)
    inverse_translation = -torch.einsum("ij,j->i", inverse_matrix, translation)
    affine_inverse_displacement = affine_displacement_field(
        inverse_matrix, inverse_translation, fixed_domain, fixed_tensor
    )
    composed_inverse_displacement = compose_displacements(
        result["invtransforms"], affine_inverse_displacement, fixed_domain
    )

    outprefix = outprefix or default_outprefix()
    warp_path = f"{outprefix}1Warp.nii.gz"
    inverse_warp_path = f"{outprefix}1InverseWarp.nii.gz"
    affine_path = f"{outprefix}0GenericAffine.mat"
    ants.image_write(displacement_xyz_to_ants_image(composed_displacement, fi), warp_path)
    ants.image_write(displacement_xyz_to_ants_image(composed_inverse_displacement, fi), inverse_warp_path)
    write_affine_transform(matrix.detach().cpu(), translation.detach().cpu(), dimension, affine_path)
    fwdtransforms, invtransforms = build_transform_lists(
        affine_path=None, warp_path=warp_path, inverse_warp_path=inverse_warp_path
    )
    return {
        "warpedmovout": tensor_to_ants_image(warpedmovout_tensor, fi),
        "fwdtransforms": fwdtransforms,
        "invtransforms": invtransforms,
        "whichtoinvert_inv": [False],
        # Same pattern as _run_bspline_svf() above: gaussian_svf_registration()
        # also returns return_loss_history=True by default, as plain floats.
        "loss_history": result.get("loss_history"),
        "level_loss_history": result.get("level_loss_history"),
    }


def evaluate_mindboggle_pair(
    pair_idx: int = 0,
    model: str = "sobolev",
    device: Optional[str] = None,
    pairs_csv: Optional[str] = None,
    data_dir: Optional[str] = None,
    canonical_affine_dir: str = "results/canonical_affines",
    verbose: bool = False,
    seed: int = 42,
    use_n4: bool = True,
    registration_output_dir: Optional[str] = None,
    **kwargs,
) -> Dict[str, Any]:
    """Evaluates a single Mindboggle registration pair under the specified ANTsTorch model variant.

    Parameters
    ----------
    pair_idx : int
        Index of the pair (0 to 89 for the bundled default pairs.csv).
    model : str
        Registration model: ``'gaussian_syn'``, ``'sobolev_syn'``,
        ``'dsti_syn'``, or ``'bspline_syn'`` (all four dispatch to the same
        dense symmetric SyN stage, ``antstorch.syn.syn_registration(...,
        type_of_transform="SyNOnly", regularizer=..., initial_affine=...)``
        -- the canonical affine already fit for this pair is supplied
        directly, so only the fluid/B-spline regularizer differs between
        them), ``'bspline_svf'``/``'svf'`` (dispatches to
        ``antstorch.bspline_flows.bspline_svf_registration()`` -- a
        different transformation family, a stationary velocity field, not a
        SyN variant despite ``'bspline_syn'``/``'bspline_svf'`` sharing the
        word "bspline"), ``'gaussian_svf'`` (the corresponding dense
        stationary-velocity model with Gaussian update/total-field smoothing),
        any key in :data:`_ANTS_TRADITIONAL_MODELS` (e.g. ``'ants_syn_quick'``
        -- a traditional, non-ANTsTorch ANTs baseline, dispatched to a direct
        ``ants.registration(type_of_transform=...)`` call on top of the same
        shared canonical affine), or -- with no alias required -- any
        deformable-only ``ants.registration`` ``type_of_transform`` string
        directly (e.g. ``model="antsRegistrationSyNQuick[so]"``, matched
        case-sensitively since ANTs' own strings are; see
        :func:`_is_deformable_only_ants_transform`).
    device : str, optional
        Compute device ('mps', 'cuda', 'cpu'). If None, automatically detected.
    pairs_csv : str, optional
        Path to pairs CSV configuration file. Defaults to the 90-pair
        definition bundled with this module (``antstorch/benchmark/pairs.csv``).
    data_dir : str, optional
        Mindboggle data root directory; resolved via
        ``antstorch.benchmark.data.resolve_data_dir`` if omitted (env var
        ``ANTSTORCH_MINDBOGGLE_DATA_DIR``, falling back to ``SYNTX_DATA_DIR``,
        falling back to ``~/data/mindboggle/volumes``).
    canonical_affine_dir : str
        Directory for the per-pair canonical affine ``.mat`` cache, shared
        across every model variant evaluated for that pair.
    verbose : bool
        If True, prints intermediate progress details.
    seed : int
        Random seed for reproducibility.
    use_n4 : bool, default=True
        If True, preprocesses input images with ANTsTorch's own N4 bias
        field correction (cached to disk under ``data_dir/.n4_cache``).
    registration_output_dir : str, optional
        Persistent directory for the warped image, the warped moving label
        map (``warped_moving_labels.nii.gz``, nearest-neighbor resampled
        onto the fixed grid -- the same fixed-space warp
        ``compute_bidirectional_dice()`` scores internally), the ANTs
        transform files, and (for the two SVF model families only, when the
        underlying registration returns one) ``loss_history.json`` -- the
        per-iteration loss and the same values grouped per pyramid level, so
        the optimization's actual convergence curve can be inspected after
        the fact. If omitted, the historical temporary transform prefix is
        retained and no label map/loss history is written.
    **kwargs
        Model-specific overrides, forwarded to the underlying registration
        call. Common ones: ``reg_iterations``, ``grad_step``, ``levels``
        (all four ``_syn`` variants); ``flow_sigma``/
        ``total_sigma`` (gaussian_syn/sobolev_syn/dsti_syn); ``update_field_mesh_size_at_base_level``/
        ``total_field_mesh_size_at_base_level``/``update_field_spline_distance``/
        ``total_field_spline_distance`` (bspline_syn); ``shrink_factors``/
        ``smoothing_sigmas``/``mesh_size``/``spline_distance`` (bspline_svf);
        ``update_field_sigma``/``total_field_sigma``/``momentum``
        (gaussian_svf); ``reg_iterations``/``syn_metric``/``syn_sampling``/
        ``grad_step``/``flow_sigma``/``total_sigma``/``aff_metric``/
        ``aff_sampling``/``aff_iterations``/``aff_shrink_factors``/
        ``aff_smoothing_sigmas`` (any :data:`_ANTS_TRADITIONAL_MODELS` model,
        e.g. ``ants_syn_quick`` -- forwarded verbatim to
        ``ants.registration()`` only when given; unlike every other model
        family here, none of these has a harness-level default, since the
        ``antsRegistrationSyN*[x]`` presets recreate the ANTs shell scripts'
        own internal multi-resolution schedules).
        When neither a mesh-size nor a spline-distance override is given for
        either bspline variant, both now default to the same 26 mm physical
        spline distance (:data:`antstorch.bspline_flows.bspline_svf_registration.
        DEFAULT_BSPLINE_SPLINE_DISTANCE_MM`) at the library level -- this
        harness no longer applies its own override on top.

    Returns
    -------
    Dict[str, Any]
        Structured benchmark metrics dictionary.
    """
    clean_device_cache()

    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"

    torch.manual_seed(seed + pair_idx)
    np.random.seed(seed + pair_idx)

    from antstorch.benchmark.data import DEFAULT_PAIRS_CSV
    resolved_pairs_csv = pairs_csv if pairs_csv is not None else DEFAULT_PAIRS_CSV

    # 1. Load Pair Data
    pair_data = load_mindboggle_pair(pair_idx=pair_idx, pairs_csv=resolved_pairs_csv, data_dir=data_dir, use_n4=use_n4, verbose=verbose)
    fi, mi = pair_data["fixed"], pair_data["moving"]
    fl, ml = pair_data["fixed_label"], pair_data["moving_label"]
    fixed_id, moving_id = pair_data["fixed_id"], pair_data["moving_id"]
    cohort_type = pair_data["pair_type"]

    # 2. Canonical Affine Alignment (Shared Across Every Model Variant)
    matrix, translation, t_aff, affine_path = _fit_or_load_canonical_affine(
        fi, mi, pair_idx, canonical_affine_dir, device, verbose
    )
    clean_device_cache()
    _, _, aff_dice_sym = compute_bidirectional_dice(fl, ml, fi, mi, [affine_path], [affine_path], [True])

    # 3. Deformable Registration
    t0_reg = time.time()
    model_lower = str(model).lower()
    registration_outprefix = None
    if registration_output_dir is not None:
        os.makedirs(registration_output_dir, exist_ok=True)
        registration_outprefix = os.path.join(registration_output_dir, "registration_")

    reg_iters = kwargs.get("reg_iterations", DEFAULT_REG_ITERATIONS)

    if model_lower in _SYN_REGULARIZERS:
        regularizer = _SYN_REGULARIZERS[model_lower]
        syn_kwargs = dict(
            fixed=fi, moving=mi,
            type_of_transform="SyNOnly",
            initial_affine=(matrix, translation),
            regularizer=regularizer,
            device=device,
            verbose=verbose,
            levels=kwargs.get("levels", DEFAULT_REGISTRATION_LEVELS),
        )
        if registration_outprefix is not None:
            syn_kwargs["outprefix"] = registration_outprefix
        syn_kwargs["reg_iterations"] = reg_iters
        for key in (
            "levels", "grad_step", "flow_sigma", "total_sigma",
            "update_field_mesh_size_at_base_level", "total_field_mesh_size_at_base_level",
            "update_field_spline_distance", "total_field_spline_distance",
            "bspline_enforce_stationary_boundary", "syn_metric", "neighborhood_radius",
            "antisymmetric", "inverse_method", "in_loop_inverse_steps", "padding_mode",
        ):
            if key in kwargs:
                syn_kwargs[key] = kwargs[key]
        # No harness-level override needed here anymore: when neither
        # update_field_mesh_size_at_base_level nor update_field_spline_distance
        # is present in syn_kwargs, syn_registration() itself now defaults
        # regularizer='bspline' to a 26 mm physical spline distance (see
        # antstorch.bspline_flows.bspline_svf_registration.DEFAULT_BSPLINE_SPLINE_DISTANCE_MM),
        # so bspline_syn and bspline_svf share the same default density
        # without any special-casing here (project doc, "default spline
        # distance" update).
        res_reg = syn_registration(**syn_kwargs)
    elif model_lower in _BSPLINE_SVF_MODELS:
        svf_kwargs = {k: v for k, v in kwargs.items() if k in (
            "reg_iterations", "shrink_factors", "smoothing_sigmas", "mesh_size", "spline_distance",
            "learning_rate", "optimizer", "gradient_step", "similarity", "neighborhood_radius",
        )}
        svf_kwargs["reg_iterations"] = reg_iters
        res_reg = _run_bspline_svf(
            fi, mi, matrix, translation, device=device, outprefix=registration_outprefix,
            verbose=verbose, **svf_kwargs
        )
    elif model_lower in _GAUSSIAN_SVF_MODELS:
        svf_kwargs = {k: v for k, v in kwargs.items() if k in (
            "reg_iterations", "shrink_factors", "smoothing_sigmas", "gradient_step",
            "momentum", "update_field_sigma", "total_field_sigma", "similarity",
            "neighborhood_radius", "velocity_weight", "bending_weight", "squaring_steps",
        )}
        svf_kwargs["reg_iterations"] = reg_iters
        res_reg = _run_gaussian_svf(
            fi, mi, matrix, translation, device=device, outprefix=registration_outprefix,
            verbose=verbose, **svf_kwargs
        )
    elif model_lower in _ANTS_TRADITIONAL_MODELS or _is_deformable_only_ants_transform(model):
        # Either a registered alias (_ANTS_TRADITIONAL_MODELS), or `model`
        # itself IS the exact ants.registration(type_of_transform=...)
        # string -- e.g. model="antsRegistrationSyNQuick[so]" works directly,
        # with no alias required, as long as it's deformable-only (checked
        # against the *original*, case-preserved `model`, never `model_lower`:
        # ants.registration()'s type_of_transform strings are case-sensitive,
        # so "antsregistrationsynquick[so]" would not be recognized by ANTs
        # itself even though it matches _is_deformable_only_ants_transform's
        # own pattern check). This is deliberately permissive -- any future
        # deformable-only preset works immediately, without a new
        # _ANTS_TRADITIONAL_MODELS entry -- while the fairness-invariant
        # guard still rejects a full-pipeline preset like "...[s]"/"...[b]"
        # (falls through to the ValueError below, same as any other unknown
        # model string).
        type_of_transform = _ANTS_TRADITIONAL_MODELS.get(model_lower, model)
        ants_kwargs = {k: v for k, v in kwargs.items() if k in _ANTS_TRADITIONAL_FORWARDED_KWARGS}
        res_reg = _run_ants_traditional(
            fi, mi, affine_path, type_of_transform, outprefix=registration_outprefix,
            verbose=verbose, **ants_kwargs
        )
    else:
        raise ValueError(
            f"Unknown registration model: '{model}'. Supported: "
            f"{sorted(set(_SYN_REGULARIZERS) | set(_BSPLINE_SVF_MODELS) | set(_GAUSSIAN_SVF_MODELS) | set(_ANTS_TRADITIONAL_MODELS))}, "
            "or any deformable-only ants.registration type_of_transform string "
            "directly (e.g. 'antsRegistrationSyNQuick[so]', 'SyNOnly', "
            "'antsRegistrationSyN[bo]') -- one whose bracketed suffix ends in "
            "'o', so it never recomputes its own rigid/affine stage on top of "
            "this harness's shared canonical affine."
        )

    t_reg = time.time() - t0_reg + t_aff

    warped_image_path = None
    if registration_output_dir is not None:
        warped_image_path = os.path.join(registration_output_dir, "warped_moving.nii.gz")
        ants.image_write(res_reg["warpedmovout"], warped_image_path)

    # 4. Evaluate Structural and Topological Metrics
    fwd_tx = res_reg["fwdtransforms"]
    inv_tx = res_reg["invtransforms"]
    which_inv = res_reg.get("whichtoinvert_inv", [True, False])

    # Persist the warped moving label map alongside warped_moving.nii.gz --
    # the same fixed-space resampling of ml (nearestNeighbor, since labels
    # are categorical) that compute_bidirectional_dice() below computes
    # internally for its Dice score, recomputed here (a cheap apply_transforms
    # call on a small integer label volume) purely so it can be written out,
    # rather than changing compute_bidirectional_dice()'s return signature.
    warped_labels_path = None
    if registration_output_dir is not None:
        warped_labels_path = os.path.join(registration_output_dir, "warped_moving_labels.nii.gz")
        ml_warped = ants.apply_transforms(
            fixed=fi, moving=ml, transformlist=fwd_tx, interpolator="nearestNeighbor"
        )
        ants.image_write(ml_warped, warped_labels_path)

    # Persist the optimization convergence curve for the two SVF model
    # families (syn_registration() already returns loss_history/
    # level_loss_history in res_reg for the _syn models, propagated the same
    # way below -- but syn's own JSON-in-results.json record has always
    # included runtime_seconds/dice, never the curve itself). Written only
    # when registration_output_dir is given and the underlying registration
    # actually produced a history (res_reg.get("loss_history") is None for
    # any model/config that ran with return_loss_history=False upstream).
    loss_history_path = None
    loss_history = res_reg.get("loss_history")
    if registration_output_dir is not None and loss_history is not None:
        loss_history_path = os.path.join(registration_output_dir, "loss_history.json")
        with open(loss_history_path, "w", encoding="utf-8") as stream:
            json.dump(
                {
                    "loss_history": loss_history,
                    "level_loss_history": res_reg.get("level_loss_history"),
                },
                stream,
                indent=2,
            )

    df_fixed, df_moving, dice_sym = compute_bidirectional_dice(fl, ml, fi, mi, fwd_tx, inv_tx, which_inv)

    fwd_warp_file = next(x for x in fwd_tx if isinstance(x, str) and x.endswith(".nii.gz"))
    jac = compute_jacobian_metrics(fi, fwd_warp_file)

    record = {
        "pair_idx": int(pair_idx),
        "model_type": model_lower,
        "cohort_type": cohort_type,
        "fixed_id": fixed_id,
        "moving_id": moving_id,
        "use_n4": use_n4,
        "status": "SUCCESS",
        "affine_dice_sym": float(aff_dice_sym),
        "dice_sym": float(dice_sym),
        "dice_fixed": float(df_fixed),
        "dice_moving": float(df_moving),
        "folding_pct": float(jac["folding_pct"]),
        "min_jacobian": float(jac["min"]),
        "runtime_seconds": float(t_reg),
        "transforms": {
            "fwdtransforms": [str(x) for x in fwd_tx],
            "invtransforms": [str(x) for x in inv_tx],
            "whichtoinvert_inv": which_inv,
        },
        "warped_moving": warped_image_path,
        "warped_moving_labels": warped_labels_path,
        "loss_history": loss_history_path,
    }

    clean_device_cache()
    return record


# Backward-compatibility alias, matching syntx.benchmark's own naming.
evaluate_pair = evaluate_mindboggle_pair
