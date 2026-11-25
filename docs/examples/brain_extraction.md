# Brain extraction

{download}`Download the notebook <../../examples/brain_extraction.ipynb>`

Run pretrained T1 extraction on a downloadable sample or a local image, inspect the binary mask and save the probabilities separately.

See the [notebook guide](https://github.com/ANTsX/ANTsTorch/blob/main/examples/README.md) for installation and execution.

## Other modalities and T1 variants

{download}`Download the modality notebook <../../examples/brain_extraction_modalities.ipynb>`

Select `t1`, `t2`, `t2star`, `flair`, `bold`, `fa`, `pet`, `mra`,
`t1threetissue`, `t1hemi` or `t1lobes`. A matching scalar 3-D input is required;
only T1 variants default to the downloadable sample T1. For BOLD, choose a 3-D
reference or mean image rather than passing a 4-D series.

Binary models return a probability image. The three T1 variants return dictionaries
of labels and class probabilities. The notebook uses label 1 as the three-tissue
brain mask and displays hemisphere/lobe labels with their foreground region masks.
The selected model runs once, with outputs saved in a modality-specific directory.

## PET brain extraction

PET uses the same S_template3 resampling (1.5 mm) and crop/pad
(136, 176, 176) as the ANTsPyNet training pipeline.

The current weight catalog downloads `brainExtractionPet_pytorch.pt`
automatically on first use. No local conversion is needed for the published
checkpoint. For an optional local reconversion, use a Python environment with
TensorFlow, ANTsPyNet, and ANTsTorch available:

```bash
PYTHONPATH="$PWD:$PWD/tools/weights" python tools/weights/convert_antspynet_weights_to_antstorch.py \
  --task brain_extraction_pet \
  --weights-file ~/.keras/ANTsXNet/brainExtractionPet.weights.h5 \
  --out-prefix ~/.antstorch/brainExtractionPet_pytorch \
  --deconv-flip noflip --verbose
```

Ensure the output directory exists first if converting weights manually.
The default cache is `~/.antstorch`; use the normal automatic download when
following the modality notebook.

```python
import ants
import antstorch

pet = ants.image_read("pet.nii.gz")
probability = antstorch.brain_extraction(pet, modality="pet", device="cpu")
mask = ants.threshold_image(probability, 0.5, 1.0, 1, 0)
```

The returned probability image has the input image geometry. CPU inference
can be used for a numerical comparison with ANTsPyNet; accelerator support
should be checked separately for the installed PyTorch version.
