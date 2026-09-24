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
- `regularization.py`: Gaussian, Sobolev, DST-I, and B-spline field
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
