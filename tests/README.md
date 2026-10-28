# Tests

Tests are grouped by the ANTsTorch component they exercise. Keep assertions near
the relevant component; use markers for execution requirements rather than
separate `unit/` and `integration/` directory trees.

| Directory | Scope |
|---|---|
| `architectures/` | Anatomical network builders, shapes and activations; no pretrained weights |
| `lamnr_flows/architectures/` | Glow and factorized-flow architectures |
| `lamnr_flows/core/` | Latents, intensity handling, channels, training helpers and shared data contracts |
| `lamnr_flows/scripts/` | Glow, hybrid and tabular trainers |
| `lamnr_flows/tools/` | 2-D/3-D tools, template means, output ranges and spherical operations |
| `utilities/` | Public utility interfaces, datasets, schedules, denoising, curvature and quality assessment |
| `pipelines/` | Complete applications using external templates and pretrained weights |
| `compat/` | PyTorch sampling compatibility and Metal kernels |
| `io/` | ANTs transform import/export |
| `bspline_flows/` | B-spline primitives, numerical N4, affine/SVF registration and device parity |
| `syn/`, `syn/core/` | SyN registration and its numerical components |
| `greedy/`, `direct/`, `registration/` | Greedy, cortical thickness and shared registration optimizers |
| `benchmark/` | The installed `antstorch.benchmark` module: data, evaluation and metrics |
| `tools/` | Repository scripts, including primitive benchmark timing |
| `support/` | Importable test helpers, not collected as tests |

`benchmark/` tests library functionality; `tools/` tests the scripts under the
repository's top-level `tools/`. Runtime comparisons and their measured results
remain in `tools/benchmarks/`, outside this test suite.

The 2026-10 reorganization preserves all 74 test files and 985 collected test
cases from the previous layout. Shared-data contract tests stay together in
`lamnr_flows/core/test_shared_data_primitives.py`; the mixed quality-assessment
tests stay together in `utilities/test_quality_assessment.py`. No assertions
were split or removed during the move.

## Running tests

From the repository root, in an environment with ANTsTorch installed:

```bash
# Inspect the complete suite, including subdirectories.
python -m pytest --collect-only -q --strict-markers tests

# Run a component.
python -m pytest tests/io tests/tools --test-device=cpu
python -m pytest tests/lamnr_flows/core --test-device=cpu

# Exclude known long tests, external-data integrations and GPU-only cases.
python -m pytest tests --test-device=cpu \
    -m "not slow and not requires_data and not cuda and not mps"

# Run marked device-specific tests on suitable hardware.
python -m pytest tests -m cuda --test-device=cuda
python -m pytest tests -m mps --test-device=mps
```

`--test-device=auto` is the default and preserves ANTsTorch's normal device
selection. `cpu`, `cuda` and `mps` override the library's default for each test,
with the override restored afterward. Explicit device arguments inside tests
are unchanged; `--test-device=cpu` does not disable the GPU or suppress GPU-only
tests. Use marker selection to exclude those tests. An unavailable explicitly
selected accelerator produces a usage error.

## Markers and fixtures

| Marker | Meaning |
|---|---|
| `slow` | Known long-running case |
| `integration` | A complete pipeline or a contract across multiple components |
| `requires_data` | Requires external templates or pretrained weights; cached files may suffice |
| `cuda` | Requires CUDA hardware |
| `mps` | Requires Apple Metal hardware |

Markers are registered in `pyproject.toml`. Existing hardware-specific skip
conditions remain in place. Markers identify known requirements and are not a
complete classification of every test's cost. Unmarked tests can still be
expensive; synthetic integrations need not carry `requires_data`. Conditional
accelerator checks inside mixed CPU/GPU tests remain in their component files.

The root `conftest.py` imports PyTorch before ANTs to avoid OpenMP runtime
conflicts. It sets the MPS fallback environment default before importing
PyTorch, while respecting an explicit caller setting. It does not modify
`CUDA_VISIBLE_DEVICES`. Component-specific fixtures belong in the nearest
`conftest.py`; shared architecture assertions live in
`tests.support.architecture_checks`. Test directories are Python packages to
allow identical test filenames in different components without import clashes.

## Continuous integration

GitHub Actions discovers `test_*.py` recursively and runs each file in its own
pytest process with `--forked --test-device=cpu --strict-markers`. This keeps
per-file and per-test process isolation while covering the nested components.
The data preparation/cache job supplies external templates and models.
GPU tests skip when their required hardware is unavailable.

On this macOS environment, some native 3-D operations abort in forked children
while passing without `--forked`. For local checks, run component files in
separate ordinary pytest processes if this occurs; the Linux workflow retains
its fork isolation.

The workflow currently requires `pytest<9.2`: pytest-forked 1.7.5 cannot marshal
the structured report identifiers introduced in pytest 9.2. To reproduce CI's
process isolation locally, install the same compatible testing tools:

```bash
python -m pip install "pytest<9.2" pytest-forked pytest-xdist psutil
python -m pytest --forked --strict-markers --test-device=cpu tests/io
```
