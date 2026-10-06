# Installation

ANTsTorch requires Python 3.10 or newer. From a checkout, create an isolated
Python environment and install the package and its dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
python -c "import antstorch; import torch; print(torch.__version__)"
```

On Windows, activate with `.venv\Scripts\activate` instead. Git is required
because `antsnormflows` is installed from its Git repository. Installation
includes ANTsPy, PyTorch, JAX and other scientific packages; it needs network
access and substantially more space than a documentation-only installation.

## Devices and data

The examples default to CPU. Where provided, use `--device cuda` with a compatible
PyTorch/CUDA installation. Model inference and registration have different device
constraints; see the [SyN guide](antsx_tutorial_syn.md#device-selection).

The synthetic U-Net and N4 examples require no data or pretrained weights.
The SyN example downloads ANTsPy sample images on first use. Brain extraction
uses your own 3-D T1 image and downloads model weights/templates on first use;
ANTsTorch caches these under `~/.antstorch`. Allow additional disk space and
network access. Reuse the cache for later runs.

See [runnable examples](examples/index.md) for commands and expected outputs.
