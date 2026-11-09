# Label-guided registration

`antstorch.label_image_registration` (also available from `antstorch.syn`)
provides a PyTorch implementation of the label-guided workflow in
`ants.label_image_registration`. The deformable optimization runs on the
selected CPU, CUDA, or MPS device. Image I/O and the small centroid-based
linear fit run on CPU; it does not call the C++ registration optimizer.

```python
import ants
import antstorch

fixed_labels = ants.image_read('fixed_labels.nii.gz')
moving_labels = ants.image_read('moving_labels.nii.gz')

result = antstorch.label_image_registration(
    fixed_labels,
    moving_labels,
    initial_transforms='affine',
    type_of_deformable_transform='antsRegistrationSyNQuick[so]',
    device='mps',  # or 'cuda', 'cpu'; None selects an available device
    output_prefix='/tmp/label_registration_',
)

warped_labels = ants.apply_transforms(
    fixed=fixed_labels,
    moving=moving_labels,
    transformlist=result['fwdtransforms'],
    interpolator='genericLabel',
)
```

The result includes `fwdtransforms`, `invtransforms`, `loss_history`,
`level_loss_history`, `common_label_ids`, and the existing SyN outputs.
`warpedmovout` uses continuous interpolation; use `genericLabel` as above
for a discrete segmentation. With linear-only results, explicitly pass
`whichtoinvert=[True]` when applying the single affine in `invtransforms`.

## Labels and optional intensity images

Each nonzero integer ID shared by a corresponding pair of label images
contributes a separate binary MeanSquares term. IDs must denote the same
anatomy in both images. Background zero and IDs absent from either member
of a pair are excluded. A pair without any common foreground ID raises an
error. Labels must be nonnegative integers exactly representable in float32,
up to 2^24.

Pass lists to combine several segmentations. `label_image_weighting` is a
scalar or one weight per label-image pair; every common label in that pair
receives that weight. Weights must be finite and nonnegative. At least one
active label or intensity metric is required for deformable registration.

```python
result = antstorch.label_image_registration(
    [fixed_labels, fixed_other_labels],
    [moving_labels, moving_other_labels],
    fixed_intensity_images=fixed_t1,
    moving_intensity_images=moving_t1,
    label_image_weighting=[1.0, 0.5],
    fixed_mask=fixed_brain_mask,
    moving_mask=moving_brain_mask,
    device='cuda',
)
```

Intensity images may also be lists. Each intensity pair receives weight 1.
All fixed inputs must share their shape, spacing, origin, and direction;
all moving inputs must share theirs. Fixed and moving domains may differ.
Masks restrict the deformable metrics and do not alter centroid fitting.

## Initialization and optimization

`initial_transforms` accepts `rigid`, `similarity`, or `affine` for a fit to
common-label centroids in physical coordinates; `identity` or `None` skips
that fit. At least three labels and a nondegenerate centroid arrangement
are required for a fit. A full 3-D affine requires at least four noncoplanar
centroids. Alternatively, provide an affine `.mat` file or list in ANTs
application order. Supplied nonlinear transform files are not supported.

`type_of_deformable_transform=None` performs only the linear initialization.
The `antsRegistrationSyN[so]` / `[bo]` family selects dense Gaussian / B-spline
SyN. Quick and Repro variants are accepted. Quick uses the ANTsPy-style
100/70/50/0 schedule and MI for optional intensities; the other variants use
100/70/50/20 and CC. A custom specification follows the ANTsPy layout:

```python
spec = ['SyN', 0.2, 'CC', 2, (100, 50, 20), (2, 1, 0), (4, 2, 1)]
result = antstorch.label_image_registration(
    fixed_labels, moving_labels,
    type_of_deformable_transform=spec,
    device='mps',
)
```

Additional keywords configure the existing `syn_registration`, for example
`levels`, `reg_iterations`, `flow_sigma`, `regularizer` (`gaussian`,
`sobolev`, `dsti`, `bspline`), and `optimizer`. Choose a pyramid whose levels
are large enough for any requested local intensity metric.

This implementation follows the ANTsPy workflow, with ANTsTorch's SyN
optimizer, normalization, regularization, and inverse solver. Preset names
and Repro variants do not imply numerical equality with ANTs C++. The
objective uses full sampling. The existing MPS compatibility layer handles
sampling gradients; unsupported operations depend on the installed PyTorch
build, and 3-D sampling may trigger the existing CPU fallback with a warning.
The actual device is recorded in `result['provenance']['device']`.

Memory grows with the number of active labels: binary image pairs and
per-level images are retained, and each iteration differentiates their
summed objective. Large 3-D segmentations with many labels may need a GPU
with substantial memory. Real-data accuracy and CUDA/MPS performance must
be evaluated on the intended hardware before drawing benchmark conclusions.

## Multivariate SyN interface

The underlying `antstorch.syn.syn_registration` also accepts
`multivariate_extras=[(metric, fixed_image, moving_image, weight, parameter)]`,
`primary_metric_weight`, `mask`, `moving_mask`, and per-level
`smoothing_sigmas` in voxel units. Extra images must share the corresponding
primary domains. MSQ/MeanSquares terms retain raw amplitudes; intensity
metrics use ANTsTorch's usual normalization. A zero primary weight allows
optimization using only the extra metrics.
