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
| [Getting started](getting_started.ipynb) | ANTs images, physical geometry, NumPy and tensor axes | Downloads tutorial r16; no pretrained weights |
| [2-D U-Net](unet_2d.ipynb) | Forward pass and probability maps with random weights | Downloads tutorial r16; random network weights, not trained segmentation |
| [N4 bias correction](n4_bias_correction.ipynb) | N4 correction and estimated bias field on acquired MRI | Downloads tutorial r16; no pretrained weights |
| [SyN registration](syn_registration.ipynb) | Align r64 to r16 and inspect the Jacobian | Downloads ANTsPy sample images |
| [Brain extraction](brain_extraction.ipynb) | Pretrained T1 extraction and mask inspection | Downloads sample T1, templates and weights |
| [Brain extraction modalities](brain_extraction_modalities.ipynb) | Select T1/T2/T2*/FLAIR/BOLD/FA/PET/MRA or T1 three-tissue/hemisphere/lobe variants | Downloads tutorial T1, FLAIR or MRA and model assets; other contrasts need local input |
| [Deep Atropos](deep_atropos.ipynb) | Six-tissue segmentation and tissue volumes | Downloads sample T1, templates and weights |
| [Cortical thickness](cortical_thickness.ipynb) | Deep Atropos followed by DiReCT | Downloads sample T1, templates and weights |
| [Harvard–Oxford atlas](harvard_oxford_atlas_labeling.ipynb) | Subcortical and cerebellar labeling | Downloads sample T1, templates and weights |
| [Desikan–Killiany–Tourville](desikan_killiany_tourville.ipynb) | Anatomical and lobar parcellation | Downloads sample T1, templates and weights |
| [SIAM labeling](siam_labeling.ipynb) | Original SIAM v0.3 whole-head segmentation | Requires `.[siam]`; downloads 2.68 GB model archive; substantial memory and runtime |
| [B-spline scattered data fitting](bspline_scattered_data_fitting.ipynb) | ANTsPy reference curves; ANTsTorch surface and acquired-image fitting | Downloads tutorial r16; curve/surface coordinates follow the tutorial; no pretrained weights |
| [Denoising](denoising.ipynb) | Non-local means filtering and residual inspection on acquired MRI | Downloads tutorial r16; no artificial noise added |
| [Data augmentation](data_augmentation.ipynb) | Shared spatial transforms, label interpolation and intensity perturbations | Downloads tutorial mprage_hippmapp3r; uses a real axial slice; no pretrained weights |
| [B-spline registration](bspline_registration.ipynb) | Affine initialization, B-spline SVF and Jacobian inspection | Downloads tutorial r16/r64; no pretrained weights |
| [Deep FLASH](deep_flash.ipynb) | Medial temporal lobe parcellation; optional aligned T2 | Downloads sample T1, templates, priors and weights |
| [White matter hyperintensities](white_matter_hyperintensities.ipynb) | T1-to-FLAIR alignment and SYSU-Media segmentation | Downloads tutorial T1/FLAIR pair and model weights; local pair optional |
| [Mouse brain analysis](mouse_brain_analysis.ipynb) | T2 extraction, Nick parcellation and cortical thickness | Downloads tutorial mouse T2 acquisition, atlas data and weights |
| [MRI super-resolution](mri_super_resolution.ipynb) | Compare interpolation and pretrained SIQ on a reduced-resolution T1 crop | Downloads sample T1 and converted model weights |
| [Lung extraction](lung_extraction.ipynb) | CT or proton MRI extraction with binary mask display | Downloads tutorial CT or proton MRI, priors and weights; local image optional |
| [Functional lung segmentation](lung_functional_segmentation.ipynb) | El Bicho ventilation class labels | Downloads tutorial proton/ventilation MRI pair and weights; local image/mask pair optional |
| [Pulmonary artery segmentation](lung_pulmonary_artery_segmentation.ipynb) | Overlapping-patch CT prediction and binary artery mask | Downloads tutorial CT and weights; local CT in HU optional; at least 160 voxels per axis |
| [Airway segmentation](lung_airway_segmentation.ipynb) | Overlapping-patch CT prediction and binary airway mask | Downloads tutorial CT and weights; local CT in HU optional; at least 160 voxels per axis |

## ANTsImage and tensor interfaces

Choose a standalone notebook for the input type you use. Each pair runs the same
acquired r16 image and algorithm settings, with inline figures and measured
operation/total durations. The tensor examples expose array layout, device and
physical-metadata handling; the image examples use ANTsTorch's image interface.
An ANTsImage interface here still uses ANTsTorch's engine, rather than calling
the corresponding ANTsPy/ITK implementation.

| Method | ANTsImage input | Tensor input |
| --- | --- | --- |
| N4 bias correction | [n4_bias_correction_antsimage.ipynb](n4_bias_correction_antsimage.ipynb) | [n4_bias_correction_tensor.ipynb](n4_bias_correction_tensor.ipynb) |
| Denoising | [denoising_antsimage.ipynb](denoising_antsimage.ipynb) | [denoising_tensor.ipynb](denoising_tensor.ipynb) |
| DiReCT thickness engine | [direct_cortical_thickness_antsimage.ipynb](direct_cortical_thickness_antsimage.ipynb) | [direct_cortical_thickness_tensor.ipynb](direct_cortical_thickness_tensor.ipynb) |

For N4, these are separate public functions. Denoising dispatches the same
function on the input type: its spacing tuple follows the provided tensor's
spatial axes, whereas N4's `ImageDomain` uses physical x-y-z order. DiReCT's
image entry returns an ANTsImage and its tensor entry returns a `DiReCTResult`.
The DiReCT pair is a 2-D API demonstration using estimated tissues from the real
r16 slice; use the existing [3-D cortical thickness workflow](cortical_thickness.ipynb)
for the segmentation-to-thickness application. Tensor input alone does not imply
autograd support: the DiReCT tensor engine runs without gradient recording.

The original N4 and denoising notebooks remain available as introductory examples.
SyN and B-spline SVF registration have their own notebooks and are distinct
methods, rather than image/tensor versions of one function.

All notebooks default to CPU. Set `ANTSTORCH_EXAMPLE_DEVICE` to `cuda` or `mps`
when the corresponding accelerator is available. The setup cell reads this
shared setting; pytest's explicit `--test-device` overrides it during tests. Inputs are the real acquisitions used in the ANTsX tutorial, with
downloads cached on first use. `r16` and `r64` use ANTsPy's data retrieval;
other examples use the ANTsTorch cache, normally `~/.antstorch`. Pretrained 3-D
applications may require substantial memory and several minutes or longer.
Some examples resize or crop the acquired images to keep demonstrations practical.

Set the path parameters or documented environment variables to use your own
images. T1/FLAIR and ventilation/mask overrides must provide both members of the
pair; the notebook will not silently mix a local subject with the tutorial data.
The brain-extraction modality notebook defaults to the T1 sample and can also
download the tutorial FLAIR and MRA acquisitions. For T2, T2*, BOLD, FA and PET,
provide the matching local scalar 3-D input; for BOLD use a 3-D reference or mean
image rather than a 4-D time series. The three-tissue brain mask uses label 1;
hemisphere/lobe variants display their labels and region masks.

SIAM requires the additional installation `python -m pip install -e '.[notebooks,siam]'`
and uses the whole-head input without skull stripping or N4 correction. See its
notebook for model licensing and memory requirements.

The B-spline notebook follows the ANTsX tutorial's curve, surface and image
examples. The curve examples explicitly use ANTsPy because ANTsTorch does not yet
support 1-D parametric domains; its surface and image examples use ANTsTorch. The curve/surface coordinates
are the tutorial's mathematical examples; image reconstruction always uses r16.

Deep FLASH accepts an optional T2 already aligned to T1; its default uses T1
only. MRI super-resolution compares a small deliberately degraded crop with
interpolation; this experiment does not establish whole-volume SIQ parity or
anatomical accuracy. The mouse workflow defaults to the tutorial mouse acquisition rather than a
template. Parcellation/thickness outputs are explicitly saved
on the isotropic computation grid, separately from native-space extraction.

The lung notebooks distinguish CT (in Hounsfield units), proton MRI and
hyperpolarized-gas ventilation MRI. All default to the tutorial's acquisitions.
CT can be overridden with `ANTSTORCH_EXAMPLE_CHEST_CT` and an optional
`ANTSTORCH_EXAMPLE_LUNG_MASK`; extraction accepts `ANTSTORCH_EXAMPLE_LUNG_IMAGE`.
For a local functional-lung example, provide both `ANTSTORCH_EXAMPLE_VENTILATION`
and `ANTSTORCH_EXAMPLE_VENTILATION_MASK`. Displays use masks or class labels.
The 160³ CT patch models can require substantial RAM even at prediction batch size 1.

## Tutorial data sources

These identifiers use the same Figshare files as the
[ANTsX tutorial](https://gist.github.com/ntustison/12a656a5fc2f6f9c4494c88dc09c5621),
retrieved with `antstorch.get_antstorch_data(identifier)` and cached on first use.
The table records original tutorial inputs. The `adni` reference T1 is the
image used by its Harvard–Oxford example, not independent subject-level
validation data. None of these examples supplies manual ground-truth labels.

| Identifier | Tutorial acquisition | Original file |
| --- | --- | --- |
| `mprage_hippmapp3r` | T1 MRI | [24984689](https://ndownloader.figshare.com/files/24984689) |
| `adni` | T1 for the Harvard–Oxford example | [25516361](https://ndownloader.figshare.com/files/25516361) |
| `tutorialWmhT1` / `tutorialWmhFlair` | Paired T1 / FLAIR | [40251796](https://ndownloader.figshare.com/files/40251796) / [40251793](https://ndownloader.figshare.com/files/40251793) |
| `tutorialLungCT` | Chest CT | [42934234](https://ndownloader.figshare.com/files/42934234) |
| `tutorialLungProton` | Proton lung MRI | [42934228](https://ndownloader.figshare.com/files/42934228) |
| `tutorialLungVentilation` | Paired ventilation MRI | [42934231](https://ndownloader.figshare.com/files/42934231) |
| `tutorialMouseT2` | Mouse T2 MRI | [45289309](https://ndownloader.figshare.com/files/45289309) |
| `tutorialMra` | MRA-TOF | [46406755](https://ndownloader.figshare.com/files/46406755) |

Application outputs are written under `example_outputs/` relative to the kernel
working directory; rerunning overwrites matching filenames. Inspect plots and
results before using them downstream. Short demonstration schedules and structural output checks do not establish
clinical or anatomical accuracy. Acquired images do not provide noise-free or
bias-free ground truth, so those notebooks avoid synthetic ground-truth assertions.

## Execution times

Each notebook prints timings for its main operations and a final **total elapsed
time**, in seconds and minutes, on the selected device. CUDA and MPS work is
synchronized before stopping an operation timer. Operation timings include any
preprocessing or model/template downloads performed inside that call.

Use **Restart Kernel and Run All Cells** for a comparable end-to-end measurement.
The total starts at the first setup cell and includes imports, downloads, image
preparation, computation, figures and file writes, but not Jupyter kernel startup.
Manual pauses between cells are included. First runs can take longer because
images and weights must be downloaded; later runs benefit from caches. These
are measurements of the current run, not hardware-independent estimates.

## Validate and execute

```bash
# Validate all notebook schemas without running cells.
python -m pytest tests/examples --test-device=cpu

# Execute thirteen notebooks on tutorial images (downloads allowed, no model weights).
python -m pytest tests/examples --run-notebooks --test-device=cpu \
    -m tutorial_data --notebook-timeout=600

# Execute one pretrained application (downloads and potentially long runtime).
python -m pytest tests/examples --run-notebooks --test-device=cpu \
    -k brain_extraction --notebook-timeout=1800
```

The execution tests use temporary working directories and never save outputs
back into the source notebooks. They also verify inline plotting for every
notebook using its setup cells and a small plot, without downloading data or
running pretrained models. See [testing details](../tests/README.md#example-notebooks).
