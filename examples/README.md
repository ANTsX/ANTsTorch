# ANTsTorch example notebooks

These Jupyter notebooks combine explanations, executable code, output checks and
plots. They adapt workflows from the [ANTsX tutorial](https://gist.github.com/ntustison/12a656a5fc2f6f9c4494c88dc09c5621)
to ANTsTorch's current APIs.

From the repository root, install the tools and register the Python environment:

```bash
python -m pip install -e '.[notebooks]'
python -m ipykernel install --user --name python3 --display-name "Python 3 (ANTsTorch)"
```

Open a notebook in Jupyter or VS Code, select this kernel and run cells from top
to bottom. For JupyterLab, install it separately with `python -m pip install jupyterlab`
and launch `python -m jupyterlab examples`.

The setup cells explicitly enable inline Matplotlib figures in Jupyter/VS Code,
even if the environment initially selects `Agg`. If using an older notebook in
an existing kernel, run `%matplotlib inline` and rerun the plotting cells.

| Notebook | Workflow | Resources |
| --- | --- | --- |
| [Getting started](getting_started.ipynb) | ANTs images, physical geometry, NumPy and tensor axes | Synthetic; no downloads |
| [2-D U-Net](unet_2d.ipynb) | Forward pass and probability maps with random weights | Synthetic; no downloads; no trained segmentation |
| [N4 bias correction](n4_bias_correction.ipynb) | Correct a known bias field and measure nonuniformity | Synthetic; no downloads |
| [SyN registration](syn_registration.ipynb) | Align r64 to r16 and inspect the Jacobian | Downloads ANTsPy sample images |
| [Brain extraction](brain_extraction.ipynb) | Pretrained T1 extraction and mask inspection | Downloads sample T1, templates and weights |
| [Deep Atropos](deep_atropos.ipynb) | Six-tissue segmentation and tissue volumes | Downloads sample T1, templates and weights |
| [Cortical thickness](cortical_thickness.ipynb) | Deep Atropos followed by DiReCT | Downloads sample T1, templates and weights |
| [Harvard–Oxford atlas](harvard_oxford_atlas_labeling.ipynb) | Subcortical and cerebellar labeling | Downloads sample T1, templates and weights |
| [Desikan–Killiany–Tourville](desikan_killiany_tourville.ipynb) | Anatomical and lobar parcellation | Downloads sample T1, templates and weights |
| [SIAM labeling](siam_labeling.ipynb) | Original SIAM v0.3 whole-head segmentation | Requires `.[siam]`; downloads 2.68 GB model archive; substantial memory and runtime |
| [B-spline scattered data fitting](bspline_scattered_data_fitting.ipynb) | ANTsPy reference curves; ANTsTorch surface and image fitting | Synthetic by default; optional r16 download |
| [Denoising](denoising.ipynb) | Non-local means filtering with a known clean reference | Synthetic; no downloads |
| [Data augmentation](data_augmentation.ipynb) | Shared spatial transforms, label interpolation and intensity perturbations | Synthetic; no downloads; ANTsPy augmentation on CPU |
| [B-spline registration](bspline_registration.ipynb) | Affine initialization, B-spline SVF and Jacobian inspection | Synthetic; no downloads |
| [Deep FLASH](deep_flash.ipynb) | Medial temporal lobe parcellation; optional aligned T2 | Downloads sample T1, templates, priors and weights |
| [White matter hyperintensities](white_matter_hyperintensities.ipynb) | T1-to-FLAIR alignment and SYSU-Media segmentation | Requires local T1/FLAIR pair; downloads model weights |
| [Mouse brain analysis](mouse_brain_analysis.ipynb) | T2 extraction, Nick parcellation and cortical thickness | Downloads example mouse template, atlas data and weights |
| [MRI super-resolution](mri_super_resolution.ipynb) | Compare interpolation and pretrained SIQ on a reduced-resolution T1 crop | Downloads sample T1 and converted model weights |

All notebooks default to CPU. Change `device` in the setup cell to use a supported
accelerator. The pretrained 3-D applications can require substantial memory and
several minutes or longer on CPU. Most human brain applications default to a downloadable T1; set `t1_path`
to an absolute path for your own scalar 3-D T1 MRI. The mouse workflow uses a
mouse T2 template by default. The WMH notebook requires your own T1/FLAIR pair
through its path parameters or `ANTSTORCH_EXAMPLE_T1` and
`ANTSTORCH_EXAMPLE_FLAIR` environment variables. Downloaded ANTsTorch assets are
cached under `~/.antstorch`.

SIAM requires the additional installation `python -m pip install -e '.[notebooks,siam]'`
and uses the whole-head input without skull stripping or N4 correction. See its
notebook for model licensing and memory requirements.

The B-spline notebook follows the ANTsX tutorial's curve, surface and image
examples. The curve examples explicitly use ANTsPy because ANTsTorch does not yet
support 1-D parametric domains; its surface and image examples use ANTsTorch.

Deep FLASH accepts an optional T2 already aligned to T1; its default uses T1
only. MRI super-resolution compares a small deliberately degraded crop with
interpolation; this experiment does not establish whole-volume SIQ parity or
anatomical accuracy. Mouse parcellation/thickness outputs are explicitly saved
on the isotropic computation grid, separately from native-space extraction.

Application outputs are written under `example_outputs/` relative to the kernel
working directory; rerunning overwrites matching filenames. Inspect plots and
results before using them downstream. Short demonstration schedules and synthetic
checks do not establish clinical or anatomical accuracy.

## Validate and execute

```bash
# Validate all notebook schemas without running cells.
python -m pytest tests/examples --test-device=cpu

# Execute the seven offline notebooks in isolated kernels.
python -m pytest tests/examples --run-notebooks --test-device=cpu \
    -m "not requires_data and not slow and not cuda and not mps"

# Execute one pretrained application (downloads and potentially long runtime).
python -m pytest tests/examples --run-notebooks --test-device=cpu \
    -k brain_extraction --notebook-timeout=1800
```

The execution tests use temporary working directories and never save outputs
back into the source notebooks. They also verify inline plotting for every
notebook using its setup cells and a small plot, without downloading data or
running pretrained models. See [testing details](../tests/README.md#example-notebooks).
