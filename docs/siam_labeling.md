# SIAM whole-head segmentation

`antstorch.siam_labeling` runs the **original SIAM v0.3 model** through
nnU-Net v2. It uses the published PyTorch checkpoints, rather than the
ANTsPyNet U-Net trained on SIAM-derived parcellations.

Install the optional dependencies from the ANTsTorch checkout:

```bash
python -m pip install -e '.[siam]'
```

```python
import antstorch
import ants

image = ants.image_read("head.nii.gz")
result = antstorch.siam_labeling(image, device="cuda", verbose=True)
ants.image_write(result["segmentation_image"], "siam_labels.nii.gz")
```

Omit `device` to use the ANTsTorch session default (CUDA, MPS, or CPU).
The dependencies are imported only when the function is called; other
ANTsTorch applications do not require nnU-Net. MPS execution depends on
the operators supported by the installed PyTorch version; CPU and CUDA
remain available through the same interface.

Supply the whole image with correct spacing, origin, and direction. Do not
apply the HCP registration, skull stripping, N4 correction, or fixed-size
crop used by the ANTsPyNet approximation. The wrapper matches the original
RAS canonical orientation and SimpleITK array conventions. nnU-Net applies
the published plans (0.75 mm spacing, Z-score normalization, and
`256 × 256 × 192` sliding windows). Gaussian weighting, a 0.5 window step,
all published folds, and disabled test-time mirroring match SIAM's default
inference. Outputs are restored to the input geometry.

The first call downloads the 2.68 GB inference-only archive from
[Figshare](https://ndownloader.figshare.com/files/71795392), via
`antstorch.get_antstorch_data("siamV03")`. The download is checked against
its SHA-256 and the archive is verified again before extraction. The archive
is cached as `siamV03.tar.gz`, alongside the other ANTsTorch data files.
Models are stored under `siam/v0.3/` in the ANTsTorch cache, normally
`~/.antstorch`. Use `antstorch.set_antstorch_cache_directory(path)` before
calling the function to select a different location. Allow space for both
the archive and extracted checkpoints during the first download. Failed
installations do not become valid cache entries.
Existing installations of the original Zenodo checkpoints remain usable
without another download. The Figshare package preserves their network
parameters exactly; optimizer states and training logs have been omitted.

The result contains:

- `segmentation_image`: an ANTsImage with integer-valued labels.
- `probability_images`: only with `return_probability_images=True`, 18 ANTsImages
  ordered by label value, including background.
- `labels`: the names and values from the published `dataset.json`.

The classes are background, GM, WM, CSF, ventricular CSF, cerebellar GM,
thalamus, pallidum, putamen, caudate, accumbens, amygdala, hippocampus,
dura, vessels, skull, head, and anomalies (values 0–17). There is no separate
auxiliary sigmoid output. Running the original network requires substantial
RAM; CPU inference can be slow. Probability maps are off by default, matching
`siam-pred` without `--save_proba`: requesting them adds a softmax and 18
full-resolution float maps (about 1.2 GB per copy for a 256³ image), which
noticeably increases run time and memory.

## Validation

Run the geometry and cache tests with:

```bash
python -m pytest -q tests/utilities/test_siam_labeling.py
```

With the `siam` extra installed, also run:

```bash
python -m pytest -q tests/pipelines/test_siam_labeling.py
```

The latter builds a small synthetic checkpoint and compares the wrapper's
results with nnU-Net's file-reader inference, including anisotropic
resampling. It exercises the integration without downloading 2.68 GB of
weights; it is not a whole-brain accuracy or SIAM parity benchmark.

The five published checkpoints were also loaded with strict state-dict checks
using nnU-Net 2.8.1. A separate smoke inference with the original weights
passed using a reduced `64 × 64 × 64` tile to bound memory. Whole-image parity
with the published tile size, and CUDA/MPS execution, have not been measured
in this integration. The public function always uses the published plans.

The source [SIAM repository](https://github.com/romainVala/SIAM) is licensed
under Apache 2.0. The published model weights are CC BY 4.0. Cite the
[SIAM paper](https://arxiv.org/abs/2605.02737) and the
[model record](https://doi.org/10.5281/zenodo.17733431) when using these models.
