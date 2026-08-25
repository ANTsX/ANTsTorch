# ANTsTorch B-spline flows

The following examples mirror the ANTsPy sections of the
[ANTsX tutorial](https://gist.github.com/ntustison/12a656a5fc2f6f9c4494c88dc09c5621),
while keeping the numerical N4 and registration paths in PyTorch. ANTsPy is
used only to load, display, and save the tutorial images.

## Shared ANTsPy–PyTorch conversions

ANTs/ITK arrays use spatial axis order `(X, Y[, Z])`. PyTorch image tensors
use `(N, C, Y, X)` or `(N, C, Z, Y, X)`, so the spatial axes must be reversed.
The `BSplineDomain` retains the physical spacing, origin, and direction.

```python
import ants
import numpy as np
import torch

from antstorch.bspline_flows import BSplineDomain


def ants_to_torch(image, device="cpu"):
    spatial_axes = tuple(range(image.dimension - 1, -1, -1))
    array = np.ascontiguousarray(
        np.transpose(image.numpy().astype(np.float32, copy=False), spatial_axes)
    )
    return torch.from_numpy(array).unsqueeze(0).unsqueeze(0).to(device)


def torch_to_ants(tensor, reference):
    spatial_axes = tuple(range(reference.dimension - 1, -1, -1))
    array = tensor.detach().cpu().numpy()[0, 0]
    array = np.ascontiguousarray(np.transpose(array, spatial_axes))
    return ants.from_numpy(
        array,
        origin=reference.origin,
        spacing=reference.spacing,
        direction=reference.direction,
    )


def torch_field_to_ants(tensor, reference):
    # Tensor vectors are (N, x-y-[z], [Z,] Y, X); ANTs vector arrays are
    # (X, Y[, Z], components). Vector components remain in physical order.
    array = tensor.detach().cpu()[0].movedim(0, -1).numpy()
    spatial_axes = tuple(range(reference.dimension - 1, -1, -1))
    array = np.ascontiguousarray(np.transpose(array, spatial_axes + (reference.dimension,)))
    return ants.from_numpy(
        array,
        origin=reference.origin,
        spacing=reference.spacing,
        direction=reference.direction,
        has_components=True,
    )


def ants_domain(image):
    return BSplineDomain(
        size=tuple(int(value) for value in image.shape),
        spacing=tuple(float(value) for value in image.spacing),
        origin=tuple(float(value) for value in image.origin),
        direction=tuple(
            tuple(float(value) for value in row) for row in image.direction
        ),
    )
```

## N4 bias-field correction

The ANTsTorch implementation accepts batched 2-D and 3-D tensors. The mask
may contain one channel or the same number of channels as the image.

```python
from antstorch.bspline_flows import n4_bias_field_correction

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

r16 = ants.image_read(ants.get_ants_data("r16")).clone("float")
r16_mask = ants.get_mask(r16)

r16_tensor = ants_to_torch(r16, device)
r16_mask_tensor = ants_to_torch(r16_mask, device)
r16_domain = ants_domain(r16)

convergence = {"iters": [50, 50, 50, 50], "tol": 1e-7}

r16_n4_tensor = n4_bias_field_correction(
    r16_tensor,
    domain=r16_domain,
    mask=r16_mask_tensor,
    shrink_factor=4,
    convergence=convergence,
    spline_param=(4, 4),       # mesh size in ITK (x, y) order
    rescale_intensities=True,
)

r16_bias_tensor = n4_bias_field_correction(
    r16_tensor,
    domain=r16_domain,
    mask=r16_mask_tensor,
    shrink_factor=4,
    convergence=convergence,
    spline_param=(4, 4),
    return_bias_field=True,
)

r16_n4 = torch_to_ants(r16_n4_tensor, r16)
r16_bias = torch_to_ants(r16_bias_tensor, r16)

ants.plot(r16, title="Original r16")
ants.plot(r16_bias, title="Estimated multiplicative bias field")
ants.plot(r16_n4, title="ANTsTorch N4 corrected r16")
```

The tensors remain differentiable with respect to the input image:

```python
differentiable_input = r16_tensor.detach().clone().requires_grad_(True)
corrected = n4_bias_field_correction(
    differentiable_input,
    domain=r16_domain,
    mask=r16_mask_tensor,
    shrink_factor=4,
    convergence={"iters": [2], "tol": 0.0},
    spline_param=(4, 4),
)
corrected.mean().backward()
print(torch.isfinite(differentiable_input.grad).all())
```

## Diffeomorphic B-spline registration

This example registers `r64` to `r16`. The B-spline stage itself estimates
only a stationary velocity field (SVF) — forward and inverse displacements
are computed as `exp(v)` and `exp(-v)` — with rigid/affine initialization
available as an optional, separate `initial_affine` argument (see below).
For rigid/affine-only or a combined affine+SyN pipeline, see
`antstorch.bspline_flows.affine_registration` and
`docs/antsx_tutorial_syn.md`.

```python
from antstorch.bspline_flows import PhysicalGradientDescent, registration

r16 = ants.image_read(ants.get_ants_data("r16")).clone("float")
r64 = ants.image_read(ants.get_ants_data("r64")).clone("float")

fixed = ants_to_torch(r16, device)
moving = ants_to_torch(r64, device)
fixed_domain = ants_domain(r16)
moving_domain = ants_domain(r64)

optimizer = PhysicalGradientDescent(
    gradient_step=0.2,
    momentum=0.9,
    smoothing_sigma=1.0,
)

result = registration(
    fixed=fixed,
    moving=moving,
    fixed_domain=fixed_domain,
    moving_domain=moving_domain,
    mesh_size=(5, 5),
    shrink_factors=(8, 4, 2, 1),
    smoothing_sigmas=(3.0, 2.0, 1.0, 0.0),
    iterations=(100, 70, 40, 20),
    optimizer=optimizer,
    similarity="ants_ncc",
    neighborhood_radius=4,
    padding_mode="border",
    stationary_boundary=True,
    verbose=True,
)

warped_r64 = torch_to_ants(result["warpedmovout"], r16)
jacobian = torch_to_ants(result["jacobian_determinant"].unsqueeze(1), r16)

ants.plot(r16, title="Fixed r16")
ants.plot(r64, title="Moving r64")
ants.plot(warped_r64, title="ANTsTorch warped r64")
ants.plot(jacobian, title="Forward Jacobian determinant")

print("Final loss:", result["loss"].item())
print("Minimum Jacobian:", result["jacobian_determinant"].min().item())
print("Maximum Jacobian:", result["jacobian_determinant"].max().item())
```

`warpedmovout` and `jacobian_determinant` above already account for
`initial_affine` when one is supplied (`registration()` composes the
affine and the SVF internally before computing them) — it is only
`fwdtransforms`/`invtransforms` that stay affine-free; see the next section.

### Affine initialization

`registration()` accepts an optional `initial_affine=(matrix, translation)`
pair — the same ITK `(x, y[, z])`-order convention returned by
`affine_registration()` — as a **fixed**, non-optimized initialization: the
B-spline SVF is then fit on top of it, with the total forward map applying
the affine first and the SVF flow second.

```python
from antstorch.bspline_flows import affine_registration

affine_result = affine_registration(
    fixed,
    moving,
    fixed_domain,
    moving_domain,
    transform_type="Affine",
    similarity="ants_ncc",
    shrink_factors=(4, 2, 1),
    smoothing_sigmas=(2.0, 1.0, 0.0),
    iterations=(100, 75, 50),
    learning_rate=(0.05, 0.03, 0.02),
)

result = registration(
    fixed=fixed,
    moving=moving,
    fixed_domain=fixed_domain,
    moving_domain=moving_domain,
    mesh_size=(5, 5),
    shrink_factors=(8, 4, 2, 1),
    smoothing_sigmas=(3.0, 2.0, 1.0, 0.0),
    iterations=(100, 70, 40, 20),
    similarity="ants_ncc",
    initial_affine=(affine_result["matrix"], affine_result["translation"]),
)
```

**Since this convention, `fwdtransforms`/`invtransforms` are always the
*pure* SVF displacement field alone — never composed with `initial_affine`
into a single field**, whether or not an `initial_affine` was supplied. The
affine itself is returned separately:

- `affine_matrix`, `affine_translation` — verbatim `initial_affine` (or
  `None` if none was given).
- `affine_matrix_inverse`, `affine_translation_inverse` — the precomputed
  exact matrix inverse (or `None`).

`warpedmovout` and `jacobian_determinant` are unaffected by this change —
both are still computed from the *total*, affine-composed map internally,
since neither has an `ants.registration()` file-based equivalent that would
require decomposing them.

### Warp a label image

The forward displacement is defined on the fixed grid and maps a fixed
physical point to its moving-image sampling location. It can therefore pull
a moving segmentation onto the fixed grid. `result["fwdtransforms"]` is the
pure SVF piece; when no `initial_affine` was used it is the *entire*
transform and can be saved directly. When an `initial_affine` was supplied,
apply both pieces together — either by writing separate ANTs transform
files and stacking them in `ants.apply_transforms`'s `transformlist`, or by
composing them in-memory before saving.

```python
r64_seg = ants.threshold_image(r64, "Kmeans", 3)
forward_warp = torch_field_to_ants(result["fwdtransforms"], r16)
forward_warp_filename = "r64_to_r16_svf_forward_warp.nii.gz"
ants.image_write(forward_warp, forward_warp_filename)

transformlist = [forward_warp_filename]
if result["affine_matrix"] is not None:
    from antstorch.ants_transform_io import write_affine_transform

    affine_filename = "r64_to_r16_0GenericAffine.mat"
    write_affine_transform(
        result["affine_matrix"], result["affine_translation"], dim=2, filename=affine_filename
    )
    transformlist.append(affine_filename)  # deformation first, affine last

warped_r64_seg = ants.apply_transforms(
    fixed=r16,
    moving=r64_seg,
    transformlist=transformlist,
    interpolator="genericLabel",
)

ants.plot(
    r16,
    overlay=warped_r64_seg,
    overlay_alpha=0.5,
    title="r64 segmentation warped to r16",
)
```

`transformlist` order matches `ants.registration()`'s own convention (see
`antstorch.ants_transform_io` and `docs/antsx_tutorial_syn.md`): the
deformation field first, the affine last, for the moving-to-fixed forward
direction.

This ANTsPy bridge is convenient for downstream ANTs tooling, but it detaches
the displacement and is not differentiable. Use the tensor-native
`antstorch.bspline_flows.warp_image` inside training or autograd code.

### Save the results

```python
ants.image_write(r16_n4, "r16_antstorch_n4.nii.gz")
ants.image_write(r16_bias, "r16_antstorch_bias.nii.gz")
ants.image_write(warped_r64, "r64_to_r16_antstorch.nii.gz")
ants.image_write(jacobian, "r64_to_r16_jacobian.nii.gz")
ants.image_write(
    torch_field_to_ants(result["invtransforms"], r16),
    "r64_to_r16_svf_inverse_warp.nii.gz",
)
torch.save(result["coefficients"].cpu(), "r64_to_r16_coefficients.pt")
```

`warpedmovout`, `fwdtransforms`, and `invtransforms` follow the naming
convention of `ants.registration` (though here they are in-memory tensors,
not paths to files on disk), but `fwdtransforms`/`invtransforms` are always
the *pure* B-spline SVF piece — see "Affine initialization" above for why,
and for `affine_matrix`/`affine_translation`. The registration result also
includes `velocity`, `coefficients`, `loss_history`, and
`level_loss_history`. Vector fields use physical x-y-(z) components even
though tensor spatial axes are stored in PyTorch-reversed order.

`registration()` stays tensor-native and batched, with no `ants` dependency
in its core, so unlike `antstorch.syn.syn_registration` (see
`docs/antsx_tutorial_syn.md`) it never writes ANTs transform files itself.
For real ANTsX file export per batch item, use
`antstorch.ants_transform_io.write_affine_transform` for the affine piece
(as above) and `ants.image_write` on a `torch_field_to_ants`-converted
`fwdtransforms`/`invtransforms` slice for the SVF piece.
