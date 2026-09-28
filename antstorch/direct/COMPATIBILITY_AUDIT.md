# DiReCT compatibility audit — 2026-09-27

This audit uses small synthetic inputs while the separate 45-iteration volume
benchmark runs. No runtime implementation was changed during this audit.
Results below identify concrete discrepancies, not their contribution to the
real-volume thickness error. See [README.md](README.md) for the earlier
contour/smoothing changes and validation snapshot.

## Inverse-field stopping condition (confirmed)

Reference: ITK `itkInvertDisplacementFieldImageFilter.hxx`, `GenerateData`.
The loop continues while both maximum error **and** mean error exceed their
respective thresholds. Thus it stops when **either** threshold is satisfied.

`antstorch/syn/core/inverse.py`, the physical-coordinate fixed-point branch of
`update_inverse_field_nd`, instead stops when **both** thresholds are satisfied.
This can take additional steps even with identical updates and tolerances.
The function is shared with registration; changing its default would affect
more than DiReCT and needs separate regression coverage.

Experiment: identity geometry, unit spacing, 17-cubed volume, zero initial
inverse, x-directed field
`u_x = A * exp(-((x-8)^2 + (y-8)^2 + (z-8)^2)/8)`; 20 iterations maximum,
max tolerance 0.1, mean tolerance 0.001, stationary boundary. Compared against
`ants.invert_displacement_field`. The candidate was an in-memory copy of the
Torch function with only the stopping conjunction changed to disjunction.

| Amplitude A (mm) | Current max absolute difference (mm) | Candidate max absolute difference (mm) |
|---|---:|---:|
| 0.05 | 0.00542744 | 6.66e-16 |
| 0.30 | 0.0172088 | 1.49e-08 |

Agreement on these fields strongly supports the stopping condition as the
cause of their mismatch. More iterations do not automatically mean a worse
inverse: matching ITK stopping and minimizing inverse residual are distinct
criteria. Compare residuals as well as reference agreement before selecting
behaviour, and preserve the shared registration contract.

## Interpolation outside the image (confirmed)

References: ITK `itkImageFunction.hxx/.h` buffer limits,
`itkWarpImageFilter.hxx`, and `itkComposeDisplacementFieldsImageFilter.hxx`.
ITK linear interpolation accepts continuous indices in `[-0.5, size-0.5)`.
Within the half-voxel edge region it uses the edge value; outside the buffer,
these filters use zero for the sampled image/displacement.

Torch `grid_sample` with `zeros` blends with zero near the boundary, while
`border` extends the edge indefinitely. Neither mode alone reproduces the
ITK rule. DiReCT currently uses border padding for white probabilities and
field composition/inversion, and zeros for contours and thickness.

Experiment: 7-cubed identity/unit-spacing domain, sample the x=0 face after
an x shift. Scalar reference uses ANTs affine linear resampling of an all-ones
image (an independent check of the ITK linear boundary convention).

| Shift (voxels) | ITK scalar | Torch zeros | Torch border |
|---|---:|---:|---:|
| -0.25 | 1 | 0.75 | 1 |
| -0.75 | 0 | 0.25 | 1 |

Direct field-composition reference uses `ants.compose_displacement_fields`,
with first field equal to the shift and second field constant `(2,0,0)`.
At x=0, shift -0.25 gives x displacement 1.75 in both implementations;
shift -0.75 gives -0.75 in ANTs and 1.25 in Torch.

An ITK-compatible sampler would combine border interpolation with an explicit
half-voxel validity mask. This must be opt-in or otherwise isolated from shared
registration code until tested, including upper/lower edges, anisotropic
spacing, geometry, and gradients. The synthetic discrepancies do not imply
that cortex in this particular volume reaches the affected domain boundary.

## Thickness accumulation inspection

Source inspection found the main update ordering consistent: the first
integration point initializes hit/total from the white contour and the prior
integrated-field norm; later points accumulate warped contour/thickness in GM.
ANTs reconstructs a working thickness map at the next outer iteration; Torch
retains the previous estimate before replacing it at the first integration
point. No additional defect in this path was established by this inspection.
This is not a proof of numerical equivalence.

## Next validation

1. Preserve the ongoing 45-iteration benchmark as the current-version baseline.
2. Test an isolated ITK stopping option on the same saved inputs; also measure
   inverse-composition residuals, not just difference from ANTs.
3. Test ITK boundary semantics separately and count affected sample locations
   on the real volume before attributing thickness differences to padding.
4. Keep gradient quality tests independent of ANTs agreement; the current
   analytical direction test does not establish superiority of either method.

## Follow-up: targeted stopping correction

After the audit, `update_inverse_field_nd` gained an explicit
`convergence_criterion` option for the fixed-point solver. DiReCT selects
`"either"` to match ITK. The default `None` preserves the prior physical
(`"both"`) and normalized (`"either"`) behaviour for existing callers,
including SyN. Regression tests compare DiReCT inversion with ANTs on both
Gaussian fields above and check that the shared physical default is unchanged.

This changes future DiReCT runs, not the already-loaded 45-iteration process.
No full-volume thickness result has yet been measured with this correction.
Interpolation boundary semantics remain an open difference.
