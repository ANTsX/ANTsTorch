# Development tools

Scripts are grouped by purpose. Run commands from the repository root with
ANTsTorch installed in the active environment (`python -m pip install -e .`).
These are source-checkout tools, not part of the installed Python API.

| Directory | Purpose | Entry points |
|---|---|---|
| [weights/](weights/) | Convert ANTsPyNet weights, fetch source HDF5 files, upload converted weights, and migrate legacy checkpoints | `convert_antspynet_weights_to_antstorch.py`, `convert_*_bespoke.py`, `download_antspynet_h5_weights.py`, `upload_weights_to_figshare.py`, `patch_legacy_glow_checkpoint_config.py` |
| [data/](data/) | Populate the ANTsTorch data and pretrained-model cache; also used by CI | `download_antstorch_data.py` |
| [registration/](registration/) | Run SyN and stationary-velocity-field registration examples | `run_syn_registration.py`, `run_svf_registration.py` |
| [benchmarks/](benchmarks/) | Compare registration/N4/B-spline implementations and run the Mindboggle probe | `run_benchmark_mindboggle_probe.py`, `compare_*.py` |
| [verify_applications/](verify_applications/README.md) | Exercise application pipelines using weights and sample images | `run_all.py`, `verify_*.py` |

## Examples

```bash
# Inspect/download the package's data and model cache.
python tools/data/download_antstorch_data.py --help

# Inspect conversion options (TensorFlow is needed by some converters).
python tools/weights/convert_antspynet_weights_to_antstorch.py --help

# Inspect the registration and benchmark entry points.
PYTHONPATH=. python tools/registration/run_syn_registration.py --help
PYTHONPATH=. python tools/benchmarks/run_benchmark_mindboggle_probe.py --help

# List application checks without executing them.
python tools/verify_applications/run_all.py --list
```

Consult each script's docstring for its dependencies, inputs, and full examples.
Some conversion scripts are recipes rather than argument-driven CLIs; in
particular, `convert_nbm_cit_cascades.py` runs conversions when executed.
The Mindboggle comparison scripts also depend on external data/checkouts.

`weights/tasks_registry.py` and `weights/return_*_unet.py` are shared converter
helpers. Keep the weight scripts together: several import or load neighboring
files. Likewise, the FireANTs benchmark loads its syntx comparison sibling.
Execute scripts by file path, as above; module-style invocation is not the
supported interface for these helpers.

## Updated paths

Scripts formerly directly under `tools/` now live in the directories above,
with their original filenames. Update local commands and external automation
to the new paths; the old paths are not retained as wrappers.
`verify_applications/` keeps its existing location.

Application smoke checks and registration benchmarks serve different purposes
from numerical parity of ANTsPyNet model ports. See
[model validation](../docs/model_validation.md) for the companion comparison
repository and its validation catalog.
