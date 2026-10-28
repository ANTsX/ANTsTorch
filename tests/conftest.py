# tests/conftest.py

# torch must be imported BEFORE ants anywhere in the process. On macOS with conda,
# importing ants first loads a second OpenMP runtime (libomp via scikit-learn) and
# multithreaded torch ops then segfault (observed: F.pad(mode='replicate') on
# torch 2.2.2, Intel iMac). pytest loads this file before any test module.
import os

# Set before torch is imported; respect an explicit caller setting.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
import pytest

def pytest_addoption(parser):
    parser.addoption(
        "--test-device", choices=("auto", "cpu", "cuda", "mps"), default="auto",
        help="ANTsTorch default device during tests; explicitly requested devices are unchanged.",
    )
    parser.addoption(
        "--dump-aug-samples",
        action="store_true",
        default=False,
        help="If set, dump augmented samples per step to the test temp dir.",
    )
    parser.addoption(
        "--aug-steps",
        type=int,
        default=4,
        help="Number of scheduling steps to iterate in the integration test.",
    )
    parser.addoption(
        "--mods",
        type=str,
        default="T1",
        help='Comma-separated modalities to load (e.g., "T2" or "T2,T1,FA").',
    )

    parser.addoption(
        "--aug-schedules",
        type=str,
        default="",
        help=("Scheduler DSL, e.g. "
              '"noise_std:cos:0.05->0.00@150k,sd_affine:linear:0.05->0.00@80k,'
              'sd_deformation:cos:10.0->0.00@100k,sd_simulated_bias_field:cos:1e-8->0.00@120k,'
              'sd_histogram_warping:exp:0.025->0.00@120k"'),
    )
    parser.addoption(
        "--grid",
        type=int,
        default=10,
        help="Grid size for mosaics (rows=cols=grid).",
    )
    parser.addoption(
        "--tile-size",
        type=int,
        default=128,
        help="Tile size (and dataset H=W) for mosaic PNGs.",
    )
    parser.addoption(
        "--preview-channel",
        type=int,
        default=0,
        help="Which modality/channel index to render in the mosaic.",
    )


def pytest_configure(config):
    requested = config.getoption("--test-device")
    if requested == "cuda" and not torch.cuda.is_available():
        raise pytest.UsageError("--test-device=cuda requires an available CUDA device")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise pytest.UsageError("--test-device=mps requires an available MPS device")


@pytest.fixture(autouse=True)
def default_test_device(request, monkeypatch):
    """Choose library defaults without masking accelerator availability."""
    requested = request.config.getoption("--test-device")
    if requested != "auto":
        from antstorch.utilities import device_manager

        monkeypatch.setattr(device_manager, "_GLOBAL_DEVICE", torch.device(requested))
