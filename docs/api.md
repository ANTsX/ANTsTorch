# API guide

Application and architecture entry points are exported from `antstorch`.
Registration-specific APIs can also be imported from their subpackages.

| Entry point | Input | Result |
| --- | --- | --- |
| `antstorch.create_unet_model_2d` | Channel count, output count, architecture options | A PyTorch module; inputs use NCHW order |
| `antstorch.create_unet_model_3d` | Channel count, output count, architecture options | A PyTorch module; inputs use NCDHW order |
| `antstorch.brain_extraction` | ANTsImage and modality | Probability image for `t1`; other modalities can return different structures |
| `antstorch.deep_atropos` | T1 image and preprocessing options | Dictionary containing segmentation and probability images |
| `antstorch.siam_labeling` | Scalar 3-D head image and optional device | Original SIAM v0.3 segmentation, 18 probability images, and label mapping; requires the `siam` extra |
| `antstorch.syn.syn_registration` | Fixed and moving ANTsImages | Warped images and file-based transform lists |
| `antstorch.bspline_flows.n4_bias_field_correction_tensor` | Image tensor, domain, optional mask | Corrected image tensor or bias field |
| `antstorch.bspline_flows.bspline_svf_registration` | Image tensors and physical domains | Warped image, displacement fields and optimization history |

## Paired image and tensor interfaces

| Method | ANTsImage entry | Tensor entry | Examples |
| --- | --- | --- | --- |
| N4 bias correction | `antstorch.n4_bias_field_correction` | `antstorch.bspline_flows.n4_bias_field_correction_tensor` | {download}`Image <../examples/n4_bias_correction_antsimage.ipynb>` · {download}`Tensor <../examples/n4_bias_correction_tensor.ipynb>` |
| Denoising | `antstorch.denoise_image(ANTsImage, ...)` | `antstorch.denoise_image(Tensor, ...)` | {download}`Image <../examples/denoising_antsimage.ipynb>` · {download}`Tensor <../examples/denoising_tensor.ipynb>` |
| DiReCT thickness engine | `antstorch.direct.kelly_kapowski` | `antstorch.direct.direct_cortical_thickness` | {download}`Image <../examples/direct_cortical_thickness_antsimage.ipynb>` · {download}`Tensor <../examples/direct_cortical_thickness_tensor.ipynb>` |

N4's image wrapper preserves physical image metadata; its tensor engine takes
an `ImageDomain` in physical x-y-z order. Denoising is a single polymorphic API:
for tensor input, the `spacing` tuple follows the supplied spatial array axes.
With `(N, C, Y, X)` input, pass `(spacing_y, spacing_x)`.

DiReCT's image interface returns an ANTsImage; its tensor interface returns
`DiReCTResult` with a thickness tensor, fields and energy history. The tensor
engine is decorated with `torch.no_grad`; tensor input is not a guarantee of
autograd support. The paired examples illustrate the actual public entry points,
not a substitution of one registration/segmentation algorithm for another.

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
