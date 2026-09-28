# DiReCT implementation layout

This package owns the tensor implementation of DiReCT cortical-thickness
estimation. It is intentionally separate from
`antstorch.utilities.cortical_thickness`, which orchestrates segmentation and
application-level workflows.

**Public naming.** `antstorch.direct` must never be re-exported at the
top-level package as `antstorch.cortical_thickness` or
`antstorch.longitudinal_cortical_thickness` — those two names are already
public API, exported from `antstorch.utilities.cortical_thickness` (the
orchestration layer that calls this package's engine). Re-exporting either
name here would silently shadow the existing entry points. This package's
own public surface should be reached as `antstorch.direct.*`.

The implementation is divided along the following boundaries:

- `core.py`: the historical DiReCT iteration, integration, inversion, and
  thickness accumulation;
- `forces.py`: gray/white-matter contours, image gradients, and the analytical
  DiReCT force;
- `regularization.py`: Gaussian, Sobolev, and DST-I field
  regularization for the DiReCT deformation field. **Do not reimplement these
  filters.** `antstorch.syn.core.smoothing` already provides
  `separable_gaussian_filter`, `apply_sobolev_green_operator`,
  `apply_dsti_green_operator`, and `apply_dsti1_green_operator`, used
  throughout `antstorch.syn` (registration, RegAdam, inversion, losses).
  `regularization.py` should either (a) import them directly from
  `antstorch.syn.core.smoothing`, or (b) once that module is promoted into
  `antstorch.registration` (the same move already made for RegAdam via
  `reg_adam_direction`), depend on `antstorch.registration` instead. Either
  way, DiReCT and SyN must share one implementation of each filter, not two
  — a second copy risks numerical drift between the two call sites (the kind
  of drift documented in project §34/§36 for `dsti_syn` vs syntx) and doubles
  the maintenance burden for any future bug fix (see the RegAdam gaussian
  fallthrough fix already made here);
- `longitudinal.py`: temporal regularization across three-dimensional visits;
- `bridge.py`: conversions between ANTs images, physical metadata, and Torch
  tensors. Check `antstorch.syn.bridge` (or equivalent) for an existing,
  already-validated ANTsImage/Torch conversion before writing a new one here.

Generic optimizer state and moment updates belong in
`antstorch.registration`, allowing DiReCT and SyN to share RegAdam without
depending on one another's implementation details. Spatial regularizers
should follow the same rule (see `regularization.py` above).

## Differences from ANTs KellyKapowski

ANTsTorch implements the DiReCT iteration in tensors; it does not promise
voxelwise identity with ANTs. The reference inspected for this comparison is
ANTs `Utilities/itkDiReCTImageFilter.hxx`, particularly `GenerateData`,
`ExtractRegionalContours`, `SmoothImage`, `GaussianSmoothDisplacementField`,
and `WarpImage`. These observations describe the implementation inspected on
2026-09-27, not every ANTs/ITK release.

### Confirmed differences and aligned conventions

| Component | ANTs KellyKapowski | ANTsTorch DiReCT |
|---|---|---|
| Probability gradient | ITK `GradientRecursiveGaussianImageFilter`, then normalization | Discrete Gaussian smoothing followed by `torch.gradient`, then normalization. This is an approximation, not a port of the recursive derivative filter. |
| Energy reduction | Sequential accumulation in the output pixel type (float32 in the benchmark) | Tensor reductions followed by accumulation over integration points; the reduction order differs. |
| Contours | Fully connected binary contours | Now fully connected: 8 neighbours in 2-D, 26 in 3-D. Earlier versions used face connectivity. |
| Hit/total smoothing | Discrete Gaussian, voxel-space variance, maximum error 0.01 | Same variance and retained-mass cutoff conventions; shared Torch filter implementation. |
| Velocity smoothing | Discrete Gaussian, voxel-space variance, maximum error 0.001; stationary boundary and blending below variance 0.5 | Same cutoff, boundary and blending conventions; shared Torch filter implementation. |
| Scalar interpolation | ITK linear interpolation with edge padding zero | `grid_sample`: border padding for white-matter probabilities, zero padding for contour/thickness images. Boundary semantics are not established as equivalent. |
| Field inversion | ITK `InvertDisplacementFieldImageFilter` | Shared tensor fixed-point inversion in `syn/core/inverse.py`, using the same requested iteration limit and error tolerances. Full numerical equivalence remains unverified. |
| Stopping | Supports convergence monitoring | Runs the requested number of outer iterations. The comparison script disables ANTs convergence stopping with `x=0`. |

The public `smoothing_sigma` argument follows the historical use of one value
for two purposes: it is the gradient's physical sigma, but the hit/total
smoother's voxel-space variance. The latter therefore uses its square root as
the shared filter's sigma. The gradient path retains the shared physical-sigma
clamp to 0.5–10 voxels and its legacy kernel cutoff. Gaussian coefficient
approximations, kernel limits, axis order and floating-point reductions can
still differ from ITK; matching conventions is not a claim of exact equality.

At iteration one, zero initial velocity makes the energy a useful reduction
check. On the saved test input, a float64 mean gave `0.703321134`, matching
ANTsTorch's displayed `0.7033211`. Repeating the contributions for the ten
integration points and summing sequentially in float32 gave `0.7055883`,
matching ANTs' `0.705588`. This explains the initial energy discrepancy;
it does **not** explain the later thickness differences. Do not reproduce
that accumulation error merely to match the displayed energy.

The current gradient is retained deliberately. A test against the known
continuous-Gaussian derivative of a smooth sinusoidal image checks direction
away from boundaries, including anisotropic spacing. Observed mean angular
errors were about 0.09 degrees (isotropic) and 0.25 degrees (anisotropic).
This limited test establishes neither superiority over ITK nor anatomical
accuracy. Differences may be acceptable when independently justified; ANTs
agreement is one validation criterion, not the only one.

### Optional variants and device behaviour

The comparison baseline is `optimizer="direct", regularizer="gaussian"`.
`reg_adam`, `sobolev`, `dsti`, and `none` are explicit alternatives, not claims
of equivalence to the standard ANTs KellyKapowski configuration.

On PyTorch builds lacking MPS `grid_sampler_3d`, `_torch_compat.grid_sample`
performs that interpolation on CPU and returns the result to MPS, with a
warning. Timings then include CPU transfers and describe mixed CPU/MPS
execution. The benchmark can run Deep Atropos on CPU separately through
`--preprocessing-device cpu`, or skip it using `--reuse-inputs`. Segmentation
time is excluded from the thickness-engine comparison.

### Validation snapshot (2026-09-27)

One `mprage_hippmapp3r.nii.gz` T1 volume, shared saved Deep Atropos inputs,
5 outer iterations, 10 integration points, gradient step 0.025 and velocity
variance 1.5 were used. Metrics below are over segmentation label 2 (GM),
comparing CPU ANTsTorch with ANTs; they are observations, not acceptance limits.

| Implementation stage | GM MAE (mm) | GM RMSE (mm) | GM Pearson r |
|---|---:|---:|---:|
| Before contour/smoothing alignment | 0.0835871 | 0.115686 | 0.975284 |
| Fully connected contours | 0.0525218 | 0.104560 | 0.984573 |
| Contours plus smoothing alignment | 0.0500913 | 0.0968915 | 0.988457 |

Before those two alignments, direct CPU-versus-MPS comparison of ANTsTorch
maps gave GM MAE `0.0000264 mm`, RMSE `0.000105 mm`, maximum absolute difference
`0.00544 mm`, and correlation `0.999999981`. This device comparison has not
been repeated for the aligned version.

Tests in `tests/direct/test_direct.py` cover contour connectivity, scalar
smoothing against ANTs, analytical gradient direction, and basic DiReCT
behaviour. MPS tests are skipped when the device is unavailable.

A 45-iteration comparison and validation across multiple volumes remain
pending. Later-iteration agreement, local systematic errors, interpolation
boundaries, and inversion equivalence are not yet established. Reports from
the benchmark (`comparison.json`, `summary.md`, and output images) should be
retained alongside the software revision and parameters for future comparisons.

From the repository root, reuse an existing benchmark output directory:

```bash
PYTHONPATH="$PWD" python tools/benchmarks/compare_cortical_thickness.py \
  --reuse-inputs /path/to/saved_inputs \
  --device cpu --iterations 45 \
  --output-dir /path/to/new_results --verbose
```
