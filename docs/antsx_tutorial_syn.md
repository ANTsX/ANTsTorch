# ANTsTorch greedy symmetric SyN

`antstorch.syn.syn_registration` mirrors `ants.registration()`'s calling
convention directly: `fixed`/`moving` are `ants.ANTsImage` objects in, and
the result dictionary uses the same `warpedmovout`/`fwdtransforms`/
`invtransforms` naming — including, as of this convention, the same
*file*-based `fwdtransforms`/`invtransforms`. Unlike
`antstorch.bspline_flows` (see `docs/antsx_tutorial_bspline_flows.md`),
there is no manual ANTsPy&#8596;PyTorch conversion step for the caller: the
tensor bridging happens internally, and the affine/warp components are
written to disk as classic ANTs transform files so the result is a drop-in
replacement anywhere `ants.apply_transforms(transformlist=...)` or another
ANTsX tool expects file-based transforms.

## Basic registration

```python
import ants

from antstorch.syn import syn_registration

r16 = ants.image_read(ants.get_ants_data("r16")).clone("float")
r64 = ants.image_read(ants.get_ants_data("r64")).clone("float")

result = syn_registration(
    fixed=r16,
    moving=r64,
    type_of_transform="SyN",
    syn_metric="lncc",
    verbose=True,
)

ants.plot(r16, title="Fixed r16")
ants.plot(r64, title="Moving r64")
ants.plot(result["warpedmovout"], title="ANTsTorch SyN-warped r64")
ants.plot(result["jacobian"], title="Forward Jacobian determinant")

print("Transform files:", result["fwdtransforms"])
print("Final SyN loss:", result["loss_history"][-1])
```

`type_of_transform="SyN"` (the default) fits an affine initialization first
— reusing `antstorch.bspline_flows.affine_registration()`'s native solver,
see `docs/antsx_tutorial_bspline_flows.md` — and then runs the dense greedy
symmetric SyN stage on top of it. `type_of_transform="SyNOnly"` skips the
internal affine fit and runs the dense stage directly at identity (or at
`initial_affine`, if supplied). `type_of_transform` also accepts
`"Translation"`, `"Rigid"`, `"Similarity"`, or `"Affine"` to fit only a
linear transform and return without a deformable stage.

## Where the transforms are written

By default `syn_registration` writes its transform files under a fresh
temporary-file prefix, exactly like `ants.registration()`'s own default
(`tempfile.mktemp()`). Pass `outprefix` to control the destination and
naming explicitly:

```python
result = syn_registration(
    fixed=r16,
    moving=r64,
    type_of_transform="SyN",
    outprefix="r64_to_r16_",
)

print(result["fwdtransforms"])
# ['r64_to_r16_1Warp.nii.gz', 'r64_to_r16_0GenericAffine.mat']
print(result["invtransforms"])
# ['r64_to_r16_0GenericAffine.mat', 'r64_to_r16_1InverseWarp.nii.gz']
```

This is the same file-naming and list-ordering convention `ants.registration()`
itself uses (see `antstorch.ants_transform_io` for the exact rules,
including why the affine file is deliberately reused in both lists). Two
consequences worth knowing:

- For a linear-only `type_of_transform`, `fwdtransforms` and `invtransforms`
  both point at the single `0GenericAffine.mat` file — `ants.apply_transforms`
  inverts it automatically for `invtransforms` via its `whichtoinvert`
  default, so no separate inverse file is written.
- For `type_of_transform="SyNOnly"` with no `initial_affine`, the affine is
  the identity and is not written at all — `fwdtransforms`/`invtransforms`
  contain only the warp files, matching what
  `ants.registration(type_of_transform="SyNOnly")` itself does.

The result dictionary also keeps the fitted affine in memory as
`affine_matrix`/`affine_translation` (ITK `(x, y[, z])` physical order, the
same values written to `0GenericAffine.mat`), and `None` when no affine
stage ran.

## Applying the transforms with `ants.apply_transforms`

Because `fwdtransforms`/`invtransforms` are real file paths, they work with
any ANTsX consumer without further conversion:

```python
r64_seg = ants.threshold_image(r64, "Kmeans", 3)

warped_r64_seg = ants.apply_transforms(
    fixed=r16,
    moving=r64_seg,
    transformlist=result["fwdtransforms"],
    interpolator="genericLabel",
)

ants.plot(
    r16,
    overlay=warped_r64_seg,
    overlay_alpha=0.5,
    title="r64 segmentation warped to r16 via ants.apply_transforms",
)

# Round trip fixed -> moving space with the inverse list:
warped_back = ants.apply_transforms(
    fixed=r64,
    moving=r16,
    transformlist=result["invtransforms"],
)
```

Applying `result["fwdtransforms"]` to `moving` with `ants.apply_transforms`
reproduces `result["warpedmovout"]` up to ordinary floating-point
interpolation noise between the two independent resamplers (ITK's own
resampler versus ANTsTorch's `F.grid_sample`-based one) — the residual is a
handful of pixels differing by a few thousandths, not a convention mismatch.

## Supplying an external affine initialization

`initial_affine` accepts the exact `(matrix, translation)` pair returned by
`antstorch.bspline_flows.affine_registration()` — an unbatched ITK
`(x, y[, z])`-order pair — and is used verbatim, skipping the internal
affine-fit stage entirely:

```python
from antstorch.bspline_flows import ImageDomain, affine_registration

from antstorch.syn.bridge import ants_image_metadata, ants_image_to_tensor

fixed_meta = ants_image_metadata(r16)
moving_meta = ants_image_metadata(r64)
fixed_domain = ImageDomain(fixed_meta["shape"], fixed_meta["spacing"], fixed_meta["origin"], fixed_meta["direction"])
moving_domain = ImageDomain(moving_meta["shape"], moving_meta["spacing"], moving_meta["origin"], moving_meta["direction"])

affine_result = affine_registration(
    ants_image_to_tensor(r16, device="cpu"),
    ants_image_to_tensor(r64, device="cpu"),
    fixed_domain,
    moving_domain,
    transform_type="Affine",
)

result = syn_registration(
    fixed=r16,
    moving=r64,
    type_of_transform="SyNOnly",
    initial_affine=(affine_result["matrix"][0], affine_result["translation"][0]),
)
```

This is exactly what `type_of_transform="SyN"` does internally when
`initial_affine` is not given, so supplying it yourself is only needed when
reusing a previously-fitted affine (for example, across a batch of images
sharing a template-to-subject alignment) or when tuning the affine stage's
own parameters independently of the SyN call.

## 3-D images

`syn_registration` supports 2-D and 3-D `ants.ANTsImage` inputs identically
— only the image dimensionality changes:

```python
fixed_3d = ants.image_read(ants.get_ants_data("mni"))
moving_3d = ants.image_read(ants.get_ants_data("mni")).clone()  # substitute a real moving volume

result_3d = syn_registration(
    fixed=fixed_3d,
    moving=moving_3d,
    type_of_transform="SyN",
    levels=(4, 2, 1),
    reg_iterations=(50, 50, 25),
)
```

## Device selection

`device=None` (the default) auto-detects `cuda` &#8594; `mps` &#8594; `cpu`, with one
deliberate exception: on a machine where only `mps` is available, the
affine-fit stage backpropagates through `F.grid_sample`, whose backward pass
is not implemented on MPS as of the PyTorch versions this has been tested
against, so auto-detection falls back to `cpu` instead of raising. Pass
`device="mps"` explicitly to opt in anyway — this works for
`type_of_transform="SyNOnly"` calls that never differentiate through an
affine, and may work in general once PyTorch adds the missing op. The device
actually used is reported in `result["provenance"]["device"]`.

## Result dictionary summary

| Key | Contents |
|---|---|
| `warpedmovout` | `ants.ANTsImage`, moving warped onto the fixed grid |
| `warpedfixout` | `ants.ANTsImage`, fixed warped onto the moving grid (SyN only) |
| `fwdtransforms`, `invtransforms` | lists of file paths, `ants.registration()` convention |
| `jacobian` | `ants.ANTsImage`, physical Jacobian determinant of the total forward deformation (SyN only) |
| `affine_matrix`, `affine_translation` | in-memory copy of the fitted/supplied affine, ITK order, or `None` |
| `loss_history`, `level_loss_history` | per-iteration SyN losses (SyN only) |
| `affine_loss_history`, `affine_level_loss_history` | per-iteration affine-fit losses, or `None` if no internal fit ran |
| `provenance` | dict recording the configuration actually used, including the resolved `device` and `outprefix` |

See the module docstring of `antstorch.syn.syn.syn_registration` for the
full parameter reference (similarity metrics, regularizer choice, CFL step
bound, antisymmetric projection, and so on).
