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
    moving_tensor_raw = ants_image_to_tensor(mi, resolved_device, dtype, normalize=False)
    affine_displacement = affine_displacement_field(matrix, translation, fixed_domain, fixed_tensor)
    composed_displacement = compose_displacements(affine_displacement, result["fwdtransforms"], fixed_domain)
    warpedmovout_tensor = warp_image(
        moving_tensor_raw, composed_displacement, fixed_domain, moving_domain, padding_mode="border"
    )

    outprefix = outprefix or default_outprefix()
    warp_path = f"{outprefix}1Warp.nii.gz"
    inverse_warp_path = f"{outprefix}1InverseWarp.nii.gz"
    affine_path = f"{outprefix}0GenericAffine.mat"

    ants.image_write(displacement_xyz_to_ants_image(result["fwdtransforms"], fi), warp_path)
    ants.image_write(displacement_xyz_to_ants_image(result["invtransforms"], fi), inverse_warp_path)
    write_affine_transform(matrix.detach().cpu(), translation.detach().cpu(), dimension, affine_path)

    fwdtransforms, invtransforms = build_transform_lists(
        affine_path=affine_path, warp_path=warp_path, inverse_warp_path=inverse_warp_path
    )

    return {
        "warpedmovout": tensor_to_ants_image(warpedmovout_tensor, fi),
        "fwdtransforms": fwdtransforms,
        "invtransforms": invtransforms,
        "whichtoinvert_inv": [True, False],
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
    # (antstorch/bspline_flows/gaussian_svf_registration.py).
    moving_tensor_raw = ants_image_to_tensor(mi, resolved_device, dtype, normalize=False)
    affine_displacement = affine_displacement_field(matrix, translation, fixed_domain, fixed_tensor)
    composed_displacement = compose_displacements(affine_displacement, result["fwdtransforms"], fixed_domain)
    warpedmovout_tensor = warp_image(
        moving_tensor_raw, composed_displacement, fixed_domain, moving_domain, padding_mode="border"
    )

    outprefix = outprefix or default_outprefix()
    warp_path = f"{outprefix}1Warp.nii.gz"
    inverse_warp_path = f"{outprefix}1InverseWarp.nii.gz"
    affine_path = f"{outprefix}0GenericAffine.mat"
    ants.image_write(displacement_xyz_to_ants_image(result["fwdtransforms"], fi), warp_path)
    ants.image_write(displacement_xyz_to_ants_image(result["invtransforms"], fi), inverse_warp_path)
    write_affine_transform(matrix.detach().cpu(), translation.detach().cpu(), dimension, affine_path)
    fwdtransforms, invtransforms = build_transform_lists(
        affine_path=affine_path, warp_path=warp_path, inverse_warp_path=inverse_warp_path
    )
    return {
        "warpedmovout": tensor_to_ants_image(warpedmovout_tensor, fi),
        "fwdtransforms": fwdtransforms,
        "invtransforms": invtransforms,
        "whichtoinvert_inv": [True, False],
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
        word "bspline"), or ``'gaussian_svf'`` (the corresponding dense
        stationary-velocity model with Gaussian update/total-field smoothing).
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
        ``compute_bidirectional_dice()`` scores internally), and the ANTs
        transform files. If omitted, the historical temporary transform
        prefix is retained and no label map is written.
    **kwargs
        Model-specific overrides, forwarded to the underlying registration
        call. Common ones: ``reg_iterations``, ``grad_step``, ``levels``
        (all four ``_syn`` variants); ``flow_sigma``/
        ``total_sigma`` (gaussian_syn/sobolev_syn/dsti_syn); ``update_field_mesh_size_at_base_level``/
        ``total_field_mesh_size_at_base_level``/``update_field_spline_distance``/
        ``total_field_spline_distance`` (bspline_syn); ``shrink_factors``/
        ``smoothing_sigmas``/``mesh_size``/``spline_distance`` (bspline_svf);
        ``update_field_sigma``/``total_field_sigma``/``momentum``
        (gaussian_svf).
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
    else:
        raise ValueError(
            f"Unknown registration model: '{model}'. Supported: "
            f"{sorted(set(_SYN_REGULARIZERS) | set(_BSPLINE_SVF_MODELS) | set(_GAUSSIAN_SVF_MODELS))}"
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
    }

    clean_device_cache()
    return record


# Backward-compatibility alias, matching syntx.benchmark's own naming.
evaluate_pair = evaluate_mindboggle_pair
