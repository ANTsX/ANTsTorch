# Publications and citations

This selected bibliography connects ANTsX methods and applications to the
ANTsTorch notebook collection. It includes original methods, ecosystem papers,
and external models or datasets used by the examples. It is not an exhaustive
list of every paper using ANTsX.

## What to cite

Cite the [ANTsX ecosystem article](https://doi.org/10.1038/s41598-021-87564-6)
for the software ecosystem, together with the original methods, model weights
and datasets used in your analysis. Report the ANTsTorch version or commit and
the selected models/settings so that the implementation can be identified.
A general ecosystem citation does not replace method-specific attribution.

The tables are reading and citation guides. A related notebook does not imply
that the PyTorch implementation reproduces the paper's experiment or has been
validated against its original implementation. See the
[model validation guide](model_validation.md) for the scope of available evidence.

{download}`Download the selected bibliography as BibTeX <references.bib>`

## ANTsX ecosystem and application overviews

| Method / topic | Article | Related notebooks |
| --- | --- | --- |
| ANTsX ecosystem and deep learning workflows | [Tustison et al. (2021)](https://doi.org/10.1038/s41598-021-87564-6) | {download}`T1 extraction <../examples/brain_extraction.ipynb>` · {download}`All extraction modalities <../examples/brain_extraction_modalities.ipynb>` · {download}`Deep Atropos <../examples/deep_atropos.ipynb>` |
| Human structural phenotypes and ANTsXNet applications | [Tustison et al. (2024)](https://doi.org/10.1038/s41598-024-59440-6) | {download}`Deep FLASH <../examples/deep_flash.ipynb>` · {download}`DKT <../examples/desikan_killiany_tourville.ipynb>` · {download}`Cortical thickness <../examples/cortical_thickness.ipynb>` |
| Mouse brain mapping and morphology | [Tustison et al. (2025)](https://doi.org/10.1038/s41467-025-66741-5) | {download}`Mouse brain analysis <../examples/mouse_brain_analysis.ipynb>` |


The 2021 paper covers ANTsX and its deep learning extensions, including ANTsRNet
and ANTsPyNet. The 2024 paper describes structural phenotyping workflows that
include Deep FLASH and DKT. These are ecosystem/application references, not a
claim that each ANTsTorch checkpoint has its own dedicated methods paper.
The mouse entry uses the 2025 journal publication rather than its earlier preprints.

## Registration, preprocessing and cortical thickness

| Method / topic | Article | Related notebooks |
| --- | --- | --- |
| SyN registration | [Avants et al. (2008)](https://doi.org/10.1016/j.media.2007.06.004) | {download}`SyN registration <../examples/syn_registration.ipynb>` |
| Registration similarity metrics and evaluation | [Avants et al. (2011)](https://doi.org/10.1016/j.neuroimage.2010.09.025) | {download}`SyN <../examples/syn_registration.ipynb>` · {download}`Affine + B-spline SVF <../examples/bspline_registration.ipynb>` |
| N4 bias field correction | [Tustison et al. (2010)](https://doi.org/10.1109/TMI.2010.2046908) | {download}`N4 bias correction <../examples/n4_bias_correction.ipynb>` |
| Atropos statistical tissue segmentation | [Avants et al. (2011)](https://doi.org/10.1007/s12021-011-9109-y) | {download}`Deep Atropos (learned counterpart) <../examples/deep_atropos.ipynb>` |
| DiReCT / Kelly–Kapowski thickness estimation | [Das et al. (2009)](https://doi.org/10.1016/j.neuroimage.2008.12.016) | {download}`Cortical thickness <../examples/cortical_thickness.ipynb>` |
| ANTs cortical thickness pipeline evaluation | [Tustison et al. (2014)](https://doi.org/10.1016/j.neuroimage.2014.05.044) | {download}`Cortical thickness <../examples/cortical_thickness.ipynb>` |


Atropos is the statistical segmentation method; Deep Atropos is a learned
segmentation workflow described in the ANTsX ecosystem literature. Cite according
to what was actually run. Similarly, the B-spline SVF notebook is a related
registration implementation, not an assertion that it is identical to SyN.

## ANTsX pulmonary and super-resolution applications

| Method / topic | Article | Related notebooks |
| --- | --- | --- |
| Functional lung quantification and template-based augmentation | [Tustison et al. (2019)](https://doi.org/10.1016/j.acra.2018.08.003) | {download}`El Bicho <../examples/lung_functional_segmentation.ipynb>` · {download}`Augmentation <../examples/data_augmentation.ipynb>` |
| Concurrent super-resolution and segmentation / SIQ background | [Avants et al. (2023)](https://doi.org/10.1101/2023.02.02.23285376) — medRxiv preprint | {download}`MRI super-resolution <../examples/mri_super_resolution.ipynb>` |


The lung article motivates the functional segmentation and augmentation workflows.
The SIQ entry is the medRxiv preprint referenced by the current MRI
super-resolution implementation. A notebook using a converted pretrained model
does not establish numerical parity with the original training framework.

## External methods, models and data

The following publications are included for attribution of algorithms, labeling
protocols, models and training/evaluation datasets used by the examples; they
should not all be described as ANTsX-authored papers.

| Method / topic | Article | Related notebooks |
| --- | --- | --- |
| Multilevel B-spline fitting | [Lee et al. (1997)](https://doi.org/10.1109/2945.620490) | {download}`Scattered data fitting <../examples/bspline_scattered_data_fitting.ipynb>` |
| Adaptive non-local means denoising | [Manjón et al. (2010)](https://doi.org/10.1002/jmri.22003) | {download}`Denoising <../examples/denoising.ipynb>` |
| U-Net architecture | [Ronneberger et al. (2015)](https://arxiv.org/abs/1505.04597) — arXiv version | {download}`2-D U-Net <../examples/unet_2d.ipynb>` |
| Mindboggle-101 / DKT labeling protocol | [Klein et al. (2012)](https://doi.org/10.3389/fnins.2012.00171) | {download}`DKT labels <../examples/desikan_killiany_tourville.ipynb>` |
| SYSU-Media WMH segmentation | [Li et al. (2018)](https://doi.org/10.1016/j.neuroimage.2018.07.005) | {download}`WMH segmentation <../examples/white_matter_hyperintensities.ipynb>` |
| EXACT’09 airway dataset / evaluation | [Lo et al. (2012)](https://doi.org/10.1109/TMI.2012.2209674) | {download}`Airway segmentation <../examples/lung_airway_segmentation.ipynb>` |
| PARSE pulmonary artery challenge | [Luo et al. (2023)](https://arxiv.org/abs/2304.03708) — arXiv preprint | {download}`Pulmonary artery segmentation <../examples/lung_pulmonary_artery_segmentation.ipynb>` |
| Original SIAM model | [Valabregue et al. (2026)](https://arxiv.org/abs/2605.02737) — arXiv preprint | {download}`SIAM labeling <../examples/siam_labeling.ipynb>` |


EXACT’09 and PARSE describe datasets/challenges, not the exact architecture of
the ANTsTorch checkpoints. SIAM uses the original model through nnU-Net; see the
[SIAM guide](siam_labeling.md) for its separate model record and weight license.
The U-Net entry links the authors' arXiv version of the architecture paper.

For Harvard–Oxford labels, consult the
[atlas source repository](https://github.com/HOA-2/SubcorticalParcellations)
referenced by the implementation. Lung extraction spans several modality-specific
models; consult their provenance in the
[training repository](https://github.com/ntustison/ANTsXNetTraining) and
[ANTsPyNet sources](https://github.com/ANTsX/ANTsPyNet).
These resources complement the selected articles; a general paper is not a
substitute for identifying the specific atlas or model used.

## Bibliographic records

The citation keys below match `references.bib`. Journal years follow the journal
issue/publication record rather than an earlier online-first date. Preprints are
labeled explicitly. Metadata were checked against PubMed, the authors' paper,
arXiv and medRxiv on November 25, 2026.

- **`tustison_antsx_2021`** — Tustison et al. (2021). [The ANTsX ecosystem for quantitative biological and medical imaging](https://doi.org/10.1038/s41598-021-87564-6). *Scientific reports*. DOI: `10.1038/s41598-021-87564-6`.

- **`tustison_ukbb_2024`** — Tustison et al. (2024). [ANTsX neuroimaging-derived structural phenotypes of UK Biobank](https://doi.org/10.1038/s41598-024-59440-6). *Scientific reports*. DOI: `10.1038/s41598-024-59440-6`.

- **`tustison_mouse_2025`** — Tustison et al. (2025). [The ANTsX ecosystem for mapping the mouse brain](https://doi.org/10.1038/s41467-025-66741-5). *Nature communications*. DOI: `10.1038/s41467-025-66741-5`.

- **`avants_syn_2008`** — Avants et al. (2008). [Symmetric diffeomorphic image registration with cross-correlation: evaluating automated labeling of elderly and neurodegenerative brain](https://doi.org/10.1016/j.media.2007.06.004). *Medical image analysis*. DOI: `10.1016/j.media.2007.06.004`.

- **`avants_metrics_2011`** — Avants et al. (2011). [A reproducible evaluation of ANTs similarity metric performance in brain image registration](https://doi.org/10.1016/j.neuroimage.2010.09.025). *NeuroImage*. DOI: `10.1016/j.neuroimage.2010.09.025`.

- **`tustison_n4_2010`** — Tustison et al. (2010). [N4ITK: improved N3 bias correction](https://doi.org/10.1109/TMI.2010.2046908). *IEEE transactions on medical imaging*. DOI: `10.1109/TMI.2010.2046908`.

- **`avants_atropos_2011`** — Avants et al. (2011). [An open source multivariate framework for n-tissue segmentation with evaluation on public data](https://doi.org/10.1007/s12021-011-9109-y). *Neuroinformatics*. DOI: `10.1007/s12021-011-9109-y`.

- **`das_direct_2009`** — Das et al. (2009). [Registration based cortical thickness measurement](https://doi.org/10.1016/j.neuroimage.2008.12.016). *NeuroImage*. DOI: `10.1016/j.neuroimage.2008.12.016`.

- **`tustison_thickness_2014`** — Tustison et al. (2014). [Large-scale evaluation of ANTs and FreeSurfer cortical thickness measurements](https://doi.org/10.1016/j.neuroimage.2014.05.044). *NeuroImage*. DOI: `10.1016/j.neuroimage.2014.05.044`.

- **`tustison_lung_2019`** — Tustison et al. (2019). [Convolutional Neural Networks with Template-Based Data Augmentation for Functional Lung Image Quantification](https://doi.org/10.1016/j.acra.2018.08.003). *Academic radiology*. DOI: `10.1016/j.acra.2018.08.003`.

- **`avants_siq_2023`** — Avants et al. (2023). [Concurrent 3D super resolution on intensity and segmentation maps improves detection of structural effects in neurodegenerative disease](https://doi.org/10.1101/2023.02.02.23285376). *medRxiv*. Preprint. DOI: `10.1101/2023.02.02.23285376`.

- **`lee_bspline_1997`** — Lee et al. (1997). [Scattered Data Interpolation with Multilevel B-Splines](https://doi.org/10.1109/2945.620490). *IEEE Transactions on Visualization and Computer Graphics*. DOI: `10.1109/2945.620490`.

- **`manjon_denoising_2010`** — Manjón et al. (2010). [Adaptive non-local means denoising of MR images with spatially varying noise levels](https://doi.org/10.1002/jmri.22003). *Journal of magnetic resonance imaging : JMRI*. DOI: `10.1002/jmri.22003`.

- **`ronneberger_unet_2015`** — Ronneberger et al. (2015). [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597). *arXiv*. Preprint. DOI: `10.48550/arXiv.1505.04597`.

- **`klein_dkt_2012`** — Klein et al. (2012). [101 labeled brain images and a consistent human cortical labeling protocol](https://doi.org/10.3389/fnins.2012.00171). *Frontiers in neuroscience*. DOI: `10.3389/fnins.2012.00171`.

- **`li_wmh_2018`** — Li et al. (2018). [Fully convolutional network ensembles for white matter hyperintensities segmentation in MR images](https://doi.org/10.1016/j.neuroimage.2018.07.005). *NeuroImage*. DOI: `10.1016/j.neuroimage.2018.07.005`.

- **`lo_exact_2012`** — Lo et al. (2012). [Extraction of airways from CT (EXACT'09)](https://doi.org/10.1109/TMI.2012.2209674). *IEEE transactions on medical imaging*. DOI: `10.1109/TMI.2012.2209674`.

- **`luo_parse_2023`** — Luo et al. (2023). [Efficient automatic segmentation for multi-level pulmonary arteries: The PARSE challenge](https://arxiv.org/abs/2304.03708). *arXiv*. Preprint. DOI: `10.48550/arXiv.2304.03708`.

- **`valabregue_siam_2026`** — Valabregue et al. (2026). [SIAM: Head and Brain MRI Segmentation from Few High-Quality Templates via Synthetic Training](https://arxiv.org/abs/2605.02737). *arXiv*. Preprint. DOI: `10.48550/arXiv.2605.02737`.


## Broader ANTsX literature

- [ANTs curated references](https://github.com/ANTsX/ANTs#references): an upstream
  selection organized by registration, templates, segmentation and other methods.
- [Historical ANTs BibTeX collection](https://github.com/ANTsX/ANTs/blob/main/forhtml/ants.bib):
  a broader existing bibliography; it is not a guaranteed up-to-date census.
- [ANTsX tutorial](https://gist.github.com/ntustison/12a656a5fc2f6f9c4494c88dc09c5621):
  workflows and additional context across ANTsX packages.
- [PubMed search for ANTsX](https://pubmed.ncbi.nlm.nih.gov/?term=ANTsX): a starting
  point for finding additional applications; papers that use ANTsX may not include
  the ecosystem name in their title or abstract.

When adding a reference, record its DOI (or stable preprint identifier), year,
publication status, method/model relationship and relevant notebook. Prefer a
verified journal version when one exists and avoid counting its preprint as a
second independent publication.
