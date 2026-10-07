# T1 brain extraction

{download}`Download the script <../../examples/brain_extraction.py>`

```{literalinclude} ../../examples/brain_extraction.py
:language: python
:linenos:
```

## PET brain extraction

PET uses the same S_template3 resampling (1.5 mm) and crop/pad
(136, 176, 176) as the ANTsPyNet training pipeline.

Convert the trained Keras 3 weights from the repository root, using a Python
environment with TensorFlow, ANTsPyNet, and ANTsTorch available:

```bash
PYTHONPATH="$PWD:$PWD/tools/weights" python tools/weights/convert_antspynet_weights_to_antstorch.py \
  --task brain_extraction_pet \
  --weights-file ~/.keras/ANTsXNet/brainExtractionPet.weights.h5 \
  --out-prefix ~/.antstorch/brainExtractionPet_pytorch \
  --deconv-flip noflip --verbose
```

Ensure the output directory exists first. The PET weights are currently
cache-only: no public download URL is configured. Place the converted
`brainExtractionPet_pytorch.pt` in the ANTsTorch cache (default `~/.antstorch`).

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
