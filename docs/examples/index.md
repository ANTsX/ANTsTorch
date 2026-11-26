# Example notebooks

The [notebook collection](https://github.com/ANTsX/ANTsTorch/blob/main/examples/README.md) provides executable tutorials
adapted to ANTsTorch. Install with `python -m pip install -e '.[notebooks]'`,
register the environment with `python -m ipykernel install --user --name python3`,
and open the notebooks in Jupyter or VS Code using that kernel.

| Notebook | Data and outputs |
| --- | --- |
| {download}`Getting started <../../examples/getting_started.ipynb>` | Downloads tutorial r16; no pretrained weights |
| {download}`2-D U-Net <../../examples/unet_2d.ipynb>` | Downloads tutorial r16; random network weights, not trained segmentation |
| {download}`N4 correction <../../examples/n4_bias_correction.ipynb>` | Downloads tutorial r16; no pretrained weights |
| {download}`SyN registration <../../examples/syn_registration.ipynb>` | Downloads r16/r64; warped image, Jacobian and transforms |
| {download}`Brain extraction <../../examples/brain_extraction.ipynb>` | Downloads T1 and pretrained assets; probability, mask and brain image |
| {download}`Brain extraction modalities <../../examples/brain_extraction_modalities.ipynb>` | Downloads tutorial T1, FLAIR or MRA and model assets; other contrasts need local input |
| {download}`Deep Atropos <../../examples/deep_atropos.ipynb>` | Downloads T1 and pretrained assets; segmentation and tissue probabilities |
| {download}`Cortical thickness <../../examples/cortical_thickness.ipynb>` | Downloads T1 and pretrained assets; tissue maps and DiReCT thickness |
| {download}`Harvard–Oxford atlas <../../examples/harvard_oxford_atlas_labeling.ipynb>` | Downloads T1 and pretrained assets; subcortical and cerebellar labels |
| {download}`Desikan–Killiany–Tourville <../../examples/desikan_killiany_tourville.ipynb>` | Downloads T1 and pretrained assets; DKT and lobar labels |
| {download}`SIAM labeling <../../examples/siam_labeling.ipynb>` | Requires the SIAM extra; downloads 2.68 GB model archive; whole-head labels |
| {download}`B-spline scattered data fitting <../../examples/bspline_scattered_data_fitting.ipynb>` | Downloads tutorial r16; curve/surface coordinates follow the tutorial; no pretrained weights |
| {download}`Denoising <../../examples/denoising.ipynb>` | Downloads tutorial r16; no artificial noise added |
| {download}`Data augmentation <../../examples/data_augmentation.ipynb>` | Downloads tutorial mprage_hippmapp3r; uses a real axial slice; no pretrained weights |
| {download}`B-spline registration <../../examples/bspline_registration.ipynb>` | Downloads tutorial r16/r64; no pretrained weights |
| {download}`Deep FLASH <../../examples/deep_flash.ipynb>` | Downloads sample T1, templates, priors and weights; medial temporal lobe parcellation; optional aligned t2 |
| {download}`White matter hyperintensities <../../examples/white_matter_hyperintensities.ipynb>` | Downloads tutorial T1/FLAIR pair and model weights; local pair optional |
| {download}`Mouse brain analysis <../../examples/mouse_brain_analysis.ipynb>` | Downloads tutorial mouse T2 acquisition, atlas data and weights |
| {download}`MRI super-resolution <../../examples/mri_super_resolution.ipynb>` | Downloads sample T1 and converted model weights; compare interpolation and pretrained siq on a reduced-resolution t1 crop |
| {download}`Lung extraction <../../examples/lung_extraction.ipynb>` | Downloads tutorial CT or proton MRI, priors and weights; local image optional |
| {download}`Functional lung segmentation <../../examples/lung_functional_segmentation.ipynb>` | Downloads tutorial proton/ventilation MRI pair and weights; local image/mask pair optional |
| {download}`Pulmonary artery segmentation <../../examples/lung_pulmonary_artery_segmentation.ipynb>` | Downloads tutorial CT and weights; local CT in HU optional; at least 160 voxels per axis |
| {download}`Airway segmentation <../../examples/lung_airway_segmentation.ipynb>` | Downloads tutorial CT and weights; local CT in HU optional; at least 160 voxels per axis |

All notebooks default to CPU. Pretrained applications can take several minutes
or longer and substantial memory. Set `t1_path` to an absolute path to use a
local scalar 3-D T1 image. Application outputs go to `example_outputs/` relative
to the kernel working directory and matching filenames are overwritten.

SIAM needs `python -m pip install -e '.[notebooks,siam]'` and substantial memory
and runtime; use a whole-head input without prior brain extraction or N4 correction.
The B-spline notebook identifies the ANTsPy curve examples explicitly: ANTsTorch
supports 2-D/3-D parametric domains, not 1-D curves.

Deep FLASH can use T1 alone or an already aligned T1/T2 pair. WMH defaults to
the acquired T1/FLAIR pair from the tutorial; the lung notebooks default to its
CT and paired proton/ventilation acquisitions. The mouse workflow uses its mouse
T2 acquisition. Paths and environment variables can override these inputs.
The [notebook guide](https://github.com/ANTsX/ANTsTorch/blob/main/examples/README.md#tutorial-data-sources)
lists original data URLs and cache identifiers. No anatomical phantom is used
as an image substitute; augmentation and resolution changes act on acquired data.
Pulmonary CT models need at least 160 voxels per axis and substantial memory.

Notebooks are not executed during documentation builds. Pytest validates their
format and supports opt-in execution; see [testing instructions](https://github.com/ANTsX/ANTsTorch/blob/main/tests/README.md).

```{toctree}
:maxdepth: 1

unet_2d
n4_synthetic
syn_registration
brain_extraction
```
