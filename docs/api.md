# API guide

Application and architecture entry points are exported from `antstorch`.
Registration-specific APIs can also be imported from their subpackages.

| Entry point | Input | Result |
| --- | --- | --- |
| `antstorch.create_unet_model_2d` | Channel count, output count, architecture options | A PyTorch module; inputs use NCHW order |
| `antstorch.create_unet_model_3d` | Channel count, output count, architecture options | A PyTorch module; inputs use NCDHW order |
| `antstorch.brain_extraction` | ANTsImage and modality | Probability image for `t1`; other modalities can return different structures |
| `antstorch.deep_atropos` | T1 image and preprocessing options | Dictionary containing segmentation and probability images |
| `antstorch.syn.syn_registration` | Fixed and moving ANTsImages | Warped images and file-based transform lists |
| `antstorch.bspline_flows.n4_bias_field_correction_tensor` | Image tensor, domain, optional mask | Corrected image tensor or bias field |
| `antstorch.bspline_flows.bspline_svf_registration` | Image tensors and physical domains | Warped image, displacement fields and optimization history |

## Image geometry

ANTs arrays use `(X, Y[, Z])`; PyTorch tensors use `(N, C, [Z,] Y, X)`.
Reversing spatial axes does not reverse the physical vector components. Preserve
spacing, origin and direction when converting images. The
[B-spline tutorial](antsx_tutorial_bspline_flows.md) explains these conversions.

## Transforms

SyN returns paths compatible with `ants.apply_transforms`. B-spline SVF
registration returns tensors, with any initial affine returned separately.
Use label interpolation when resampling segmentations. See the
[SyN guide](antsx_tutorial_syn.md) for file ordering and inverse transforms.

## Full parameter documentation

In an installed environment, use `help(antstorch.brain_extraction)` or
`help(antstorch.syn.syn_registration)` for the current implementation's arguments.
The [source repository](https://github.com/ANTsX/ANTsTorch) contains the full
implementations. This documentation build intentionally does not import the
scientific stack or execute model inference.
