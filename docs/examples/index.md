# Example notebooks

The [notebook collection](https://github.com/ANTsX/ANTsTorch/blob/main/examples/README.md) provides executable tutorials
adapted to ANTsTorch. Install with `python -m pip install -e '.[notebooks]'`,
register the environment with `python -m ipykernel install --user --name python3`,
and open the notebooks in Jupyter or VS Code using that kernel.

| Notebook | Data and outputs |
| --- | --- |
| {download}`Getting started <../../examples/getting_started.ipynb>` | Offline; image geometry and tensor conversion |
| {download}`2-D U-Net <../../examples/unet_2d.ipynb>` | Offline; probability maps from random weights |
| {download}`N4 correction <../../examples/n4_bias_correction.ipynb>` | Offline; synthetic bias correction and nonuniformity measurement |
| {download}`SyN registration <../../examples/syn_registration.ipynb>` | Downloads r16/r64; warped image, Jacobian and transforms |
| {download}`Brain extraction <../../examples/brain_extraction.ipynb>` | Downloads T1 and pretrained assets; probability, mask and brain image |
| {download}`Deep Atropos <../../examples/deep_atropos.ipynb>` | Downloads T1 and pretrained assets; segmentation and tissue probabilities |
| {download}`Cortical thickness <../../examples/cortical_thickness.ipynb>` | Downloads T1 and pretrained assets; tissue maps and DiReCT thickness |
| {download}`Harvard–Oxford atlas <../../examples/harvard_oxford_atlas_labeling.ipynb>` | Downloads T1 and pretrained assets; subcortical and cerebellar labels |
| {download}`Desikan–Killiany–Tourville <../../examples/desikan_killiany_tourville.ipynb>` | Downloads T1 and pretrained assets; DKT and lobar labels |
| {download}`SIAM labeling <../../examples/siam_labeling.ipynb>` | Requires the SIAM extra; downloads 2.68 GB model archive; whole-head labels |
| {download}`B-spline scattered data fitting <../../examples/bspline_scattered_data_fitting.ipynb>` | Offline by default; ANTsPy reference curves and ANTsTorch surface/image fitting |
| {download}`Denoising <../../examples/denoising.ipynb>` | Offline; non-local means filtering and error against a clean phantom |
| {download}`Data augmentation <../../examples/data_augmentation.ipynb>` | Synthetic; no downloads; ANTsPy augmentation on CPU; shared spatial transforms, label interpolation and intensity perturbations |
| {download}`B-spline registration <../../examples/bspline_registration.ipynb>` | Synthetic; no downloads; affine initialization, b-spline svf and jacobian inspection |
| {download}`Deep FLASH <../../examples/deep_flash.ipynb>` | Downloads sample T1, templates, priors and weights; medial temporal lobe parcellation; optional aligned t2 |
| {download}`White matter hyperintensities <../../examples/white_matter_hyperintensities.ipynb>` | Requires local T1/FLAIR pair; downloads model weights; t1-to-flair alignment and sysu-media segmentation |
| {download}`Mouse brain analysis <../../examples/mouse_brain_analysis.ipynb>` | Downloads example mouse template, atlas data and weights; t2 extraction, nick parcellation and cortical thickness |
| {download}`MRI super-resolution <../../examples/mri_super_resolution.ipynb>` | Downloads sample T1 and converted model weights; compare interpolation and pretrained siq on a reduced-resolution t1 crop |

All notebooks default to CPU. Pretrained applications can take several minutes
or longer and substantial memory. Set `t1_path` to an absolute path to use a
local scalar 3-D T1 image. Application outputs go to `example_outputs/` relative
to the kernel working directory and matching filenames are overwritten.

SIAM needs `python -m pip install -e '.[notebooks,siam]'` and substantial memory
and runtime; use a whole-head input without prior brain extraction or N4 correction.
The B-spline notebook identifies the ANTsPy curve examples explicitly: ANTsTorch
supports 2-D/3-D parametric domains, not 1-D curves.

Deep FLASH can use T1 alone or an already aligned T1/T2 pair. The WMH notebook
requires local paths or `ANTSTORCH_EXAMPLE_T1` and `ANTSTORCH_EXAMPLE_FLAIR`;
no paired FLAIR example is bundled. The mouse notebook uses a downloadable T2
template for demonstration. Super-resolution uses a small T1 crop and compares
against interpolation without assuming that the pretrained model is more accurate.

Notebooks are not executed during documentation builds. Pytest validates their
format and supports opt-in execution; see [testing instructions](https://github.com/ANTsX/ANTsTorch/blob/main/tests/README.md).

```{toctree}
:maxdepth: 1

unet_2d
n4_synthetic
syn_registration
brain_extraction
```
