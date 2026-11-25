<div align="center">

<img src="docs/figures/antstorch-banner.png" alt="ANTsTorch — conceptual brain illustration combining MRI, colored segmentation, and a deformation grid" width="100%" />

<!-- AI-generated conceptual artwork; not an ANTsTorch inference result. -->

# ANTsTorch

### Medical imaging with PyTorch, powered by ANTsX

Deep learning architectures · Pretrained applications · Differentiable image processing · Registration

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](pyproject.toml)
[![PyTorch](https://img.shields.io/badge/Framework-PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/antstorch?label=PyPI%20downloads)](https://pypi.org/project/antstorch/)
[![ANTsX paper](https://img.shields.io/badge/ANTsX_paper-Open_Access-8DABFF?logo=pubmed)](https://pubmed.ncbi.nlm.nih.gov/33907199/)

[Documentation](docs/index.md) · [Examples](examples/README.md) · [API guide](docs/api.md) · [Benchmarks](tools/benchmarks/README.md) · [Model validation](docs/model_validation.md)

</div>

---

**ANTsTorch brings the ANTsX medical imaging ecosystem to PyTorch.** Build and train 2-D and 3-D networks, apply pretrained models to ANTs images, and explore tensor-based bias correction, registration, cortical thickness estimation, and normalizing flows.

It complements [ANTsPyNet](https://github.com/ANTsX/ANTsPyNet/) and [ANTsRNet](https://github.com/ANTsX/ANTsRNet/), with ports of ANTsPyNet architectures and pretrained applications. Porting and numerical validation are ongoing; see the [validation guide](docs/model_validation.md) for the scope of available comparison evidence.

## What you can do

| Area | Capabilities | Start here |
| --- | --- | --- |
| **Deep learning** | 2-D and 3-D U-Nets, ResNets, dense networks, deep back-projection networks, and task-specific architectures | [U-Net example](examples/unet_2d.ipynb) |
| **Pretrained applications** | Brain extraction, tissue segmentation, anatomical labeling, lung imaging, super-resolution, and more | [Application catalog](#pretrained-applications) |
| **Image processing** | Preprocessing, augmentation, and tensor-native N4 bias field correction | [N4 example](examples/n4_bias_correction.ipynb) |
| **Registration** | SyN, affine, and B-spline stationary velocity field registration | [SyN tutorial](docs/antsx_tutorial_syn.md) · [B-spline tutorial](docs/antsx_tutorial_bspline_flows.md) |
| **Cortical thickness** | Segmentation-driven workflows and a tensor implementation of DiReCT | [Implementation guide](antstorch/direct/README.md) |
| **Normalizing flows** | Flow architectures and LAMNR hybrid training workflows | [Training guide](docs/lamnr_hybrid_trainer.md) |

## Installation

Requires **Python 3.10+** and **Git**. From a terminal:

```bash
git clone https://github.com/ANTsX/ANTsTorch.git
cd ANTsTorch
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

On Windows, activate the environment with `.venv\Scripts\activate`.

For a direct installation without a working checkout:

```bash
python -m pip install git+https://github.com/ANTsX/ANTsTorch.git
```

Installation includes ANTsPy, PyTorch, JAX, and other scientific dependencies. Git is needed for the `antsnormflows` dependency. See the [installation guide](docs/installation.md) for device selection, downloads, and caching.

## Quick start

### Build a 2-D U-Net

Run a forward pass with synthetic input; no image or model downloads are needed.

```python
import torch
import antstorch

model = antstorch.create_unet_model_2d(
    input_channel_size=1,
    number_of_outputs=3,
    number_of_filters=(8, 16, 32),
    mode="classification",
).eval()

image = torch.rand(1, 1, 64, 64)  # batch, channels, height, width
with torch.inference_mode():
    probabilities = model(image)

print(tuple(probabilities.shape))  # (1, 3, 64, 64)
```

This network has random weights. Train it before using its predictions for segmentation.

### Extract a brain from a T1 MRI

Use a local scalar 3-D T1 image with a pretrained model:

```python
import ants
import antstorch

image = ants.image_read("t1.nii.gz").clone("float")
probability = antstorch.brain_extraction(image, modality="t1", device="cpu")
mask = ants.threshold_image(probability, 0.5, 1.0, 1, 0)
ants.image_write(image * mask, "t1_brain.nii.gz")
```

Weights and templates download on first use and are cached under `~/.antstorch`. This example thresholds the probability map at 0.5; inspect the result for your data. Other extraction modalities may return different output structures.

## Python notebooks

Explore **23 executable Jupyter notebooks** with explanations, output checks and
inline figures. Start with [Getting started](examples/getting_started.ipynb),
then choose a workflow below. The [complete notebook guide](examples/README.md)
lists data requirements, outputs and execution options for every example.

| Topic | Notebooks |
| --- | --- |
| **Getting started and learning** | [Image geometry and tensors](examples/getting_started.ipynb) · [2-D U-Net](examples/unet_2d.ipynb) · [Data augmentation](examples/data_augmentation.ipynb) |
| **Image processing** | [N4 bias correction](examples/n4_bias_correction.ipynb) · [Denoising](examples/denoising.ipynb) |
| **Registration and B-splines** | [SyN registration](examples/syn_registration.ipynb) · [Affine + B-spline SVF registration](examples/bspline_registration.ipynb) · [Scattered data fitting](examples/bspline_scattered_data_fitting.ipynb) |
| **Brain extraction** | [T1 extraction](examples/brain_extraction.ipynb) · [All 11 modality options](examples/brain_extraction_modalities.ipynb): T1, T2, T2*, FLAIR, BOLD, FA, PET, MRA and T1 three-tissue, hemisphere and lobe variants |
| **Brain anatomy** | [Deep Atropos](examples/deep_atropos.ipynb) · [Harvard–Oxford](examples/harvard_oxford_atlas_labeling.ipynb) · [Desikan–Killiany–Tourville](examples/desikan_killiany_tourville.ipynb) · [Deep FLASH](examples/deep_flash.ipynb) |
| **Thickness and whole-head labels** | [Cortical thickness](examples/cortical_thickness.ipynb) · [SIAM whole-head segmentation](examples/siam_labeling.ipynb) |
| **Additional imaging workflows** | [White matter hyperintensities](examples/white_matter_hyperintensities.ipynb) · [Mouse brain analysis](examples/mouse_brain_analysis.ipynb) · [MRI super-resolution](examples/mri_super_resolution.ipynb) |
| **Pulmonary imaging** | [Lung extraction: CT or proton MRI](examples/lung_extraction.ipynb) · [Functional ventilation segmentation](examples/lung_functional_segmentation.ipynb) · [Pulmonary arteries](examples/lung_pulmonary_artery_segmentation.ipynb) · [Airways](examples/lung_airway_segmentation.ipynb) |

From the repository root, install the notebook tools and launch JupyterLab:

```bash
python -m pip install -e '.[notebooks]' jupyterlab
python -m ipykernel install --user --name python3 --display-name "Python 3 (ANTsTorch)"
python -m jupyterlab examples
```

Alternatively, open the `.ipynb` files in VS Code. Select **Python 3 (ANTsTorch)**
and run cells from top to bottom. All notebooks default to CPU and explicitly
enable inline Matplotlib figures. Extraction and binary segmentation overlays
use masks; anatomical parcellations display discrete labels.

Seven notebooks use synthetic data without downloads: getting started, U-Net,
N4, denoising, augmentation, B-spline registration and the default scattered-data
fitting example. Pretrained workflows download weights/templates as needed and
may require substantial memory and run time. Some use downloadable example
images; WMH, functional lung, pulmonary artery and airway examples require local
images of the specified acquisition type. Non-T1 brain-extraction modalities
also require matching local 3-D inputs. SIAM additionally needs the `siam` extra:
`python -m pip install -e '.[notebooks,siam]'`.

The [notebook guide](examples/README.md) explains input paths and environment
variables, cached downloads, output locations and model-specific limitations.
The [published example guide](docs/examples/index.md) includes notebook downloads;
notebooks are not executed during documentation builds. See the
[testing instructions](tests/README.md#example-notebooks) to validate notebooks
or run the offline examples with pytest.

## Pretrained applications

The application layer provides ANTsPyNet model ports and integrations with published PyTorch models. Available workflows include:

| Domain | Applications |
| --- | --- |
| **Brain extraction** | T1, T2, T2*, FA, FLAIR, BOLD, PET, MRA; T1 three-tissue, hemisphere, and lobe variants |
| **Brain anatomy** | Deep Atropos tissue segmentation, Desikan–Killiany–Tourville and Harvard–Oxford labeling, Deep FLASH, cerebellar morphology, cortical thickness |
| **Whole-head segmentation** | Original SIAM v0.3 via nnU-Net, with the optional `siam` extra; [usage and model downloads](docs/siam_labeling.md) |
| **Small structures** | HippMapp3r and two-stage hippocampus segmentation, hypothalamus and subunits, claustrum |
| **White matter lesions** | SYSU-Media, HyperMapp3r, ANTsXNet WMH, and SHIVA PVS/WMH ensembles |
| **Lung imaging** | Extraction in proton MRI, CT, ventilation MRI, and X-ray; lobe, airway, vessel, and pulmonary artery segmentation |
| **Mouse imaging & histology** | Brain extraction, parcellation, cortical thickness, histology masks, and histology super-resolution |
| **Image quality & enhancement** | MRI modality classification, neural image quality assessment, MRI super-resolution, and anisotropic 2×–6× super-resolution |

Availability does not imply numerical parity for every modality or model variant. The [model validation guide](docs/model_validation.md) explains the comparison harness, saved evidence, and remaining gaps.

## Benchmarks

The **[benchmark guide](tools/benchmarks/README.md)** brings together comparison protocols, recorded results, figures, and instructions for reproducing the runs.

| Suite | Comparisons | Measures |
| --- | --- | --- |
| **[Registration](tools/benchmarks/README.md#registration)** | ANTsTorch, syntx, FireANTs, and an ANTs C++ baseline on Mindboggle-101; the reported run covers 30 of the 90 defined pairs | DKT31 label overlap (Dice), deformable registration time, Jacobian statistics, and folding percentage |
| **[Image-processing primitives](tools/benchmarks/README.md#primitive-comparisons-n4-denoising-cortical-thickness-b-spline-fitting)** | ANTs and ANTsTorch implementations of N4 bias correction, denoising, cortical thickness, and B-spline fitting | Run time and numerical agreement, with recorded MPS and CUDA results |

Registration comparisons use a shared affine initialization in the default fair mode. Read the results alongside their hardware, model settings, and inverse-transform conventions; CPU and GPU timings describe different execution environments. The guide includes the data requirements and commands needed to reproduce each comparison.

## Learn more

- **[Documentation](docs/index.md)** — installation, examples, API conventions, and tutorials.
- **[API guide](docs/api.md)** — entry points, tensor layouts, image geometry, and transform conventions.
- **[SyN registration](docs/antsx_tutorial_syn.md)** — registration workflows and transform handling.
- **[Label-guided registration](docs/label_image_registration.md)** — weighted segmentation and optional intensity metrics on CPU, CUDA, or MPS.
- **[N4 & B-spline registration](docs/antsx_tutorial_bspline_flows.md)** — tensor-based processing and physical image domains.
- **[LAMNR hybrid training](docs/lamnr_hybrid_trainer.md)** — normalizing-flow training workflows.
- **[Development tools](tools/README.md)** — application checks, benchmarks, and supporting utilities.
- **[Documentation publishing](docs/publishing.md)** — build the documentation and publish on Read the Docs.
- **[ANTsX tutorial examples](https://gist.github.com/ntustison/12a656a5fc2f6f9c4494c88dc09c5621#antsxnet)** — additional self-contained workflows.

## Research, community & license

For background and applications, see the [ANTsX ecosystem publication](https://pubmed.ncbi.nlm.nih.gov/33907199/).

Questions, bug reports, and contributions are welcome through [GitHub Issues](https://github.com/ANTsX/ANTsTorch/issues) and pull requests. Please include a minimal example and relevant package versions when reporting a problem.

ANTsTorch is released under the [Apache License 2.0](LICENSE).

We gratefully acknowledge grant support from the **Office of Naval Research (N0014-23-1-2317)**.
