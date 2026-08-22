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

This example registers `r64` to `r16`. Unlike the full ANTs SyN pipeline,
this initial ANTsTorch interface estimates only a B-spline stationary velocity
field—there is no rigid or affine initialization. Forward and inverse
displacements are computed as `exp(v)` and `exp(-v)`.

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

warped_r64 = torch_to_ants(result["warped_moving"], r16)
jacobian = torch_to_ants(result["jacobian_determinant"].unsqueeze(1), r16)

ants.plot(r16, title="Fixed r16")
ants.plot(r64, title="Moving r64")
ants.plot(warped_r64, title="ANTsTorch warped r64")
ants.plot(jacobian, title="Forward Jacobian determinant")

print("Final loss:", result["loss"].item())
print("Minimum Jacobian:", result["jacobian_determinant"].min().item())
print("Maximum Jacobian:", result["jacobian_determinant"].max().item())
```

### Warp a label image

The forward displacement is defined on the fixed grid and maps a fixed
physical point to its moving-image sampling location. It can therefore pull a
moving segmentation onto the fixed grid. To reuse the standard ANTsPy
interface, convert the PyTorch field to an ANTs vector image, save it as a warp
transform, and pass that file to `ants.apply_transforms`.

```python
r64_seg = ants.threshold_image(r64, "Kmeans", 3)
forward_warp = torch_field_to_ants(result["forward_displacement"], r16)
forward_warp_filename = "r64_to_r16_forward_warp.nii.gz"
ants.image_write(forward_warp, forward_warp_filename)

warped_r64_seg = ants.apply_transforms(
    fixed=r16,
    moving=r64_seg,
    transformlist=[forward_warp_filename],
    interpolator="genericLabel",
)

ants.plot(
    r16,
    overlay=warped_r64_seg,
    overlay_alpha=0.5,
    title="r64 segmentation warped to r16",
)
```

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
    torch_field_to_ants(result["inverse_displacement"], r16),
    "r64_to_r16_inverse_warp.nii.gz",
)
torch.save(result["coefficients"].cpu(), "r64_to_r16_coefficients.pt")
```

The registration result also includes `velocity`, `forward_displacement`,
`inverse_displacement`, `loss_history`, and `level_loss_history`. Vector fields
use physical x-y-(z) components even though tensor spatial axes are stored in
PyTorch-reversed order.
