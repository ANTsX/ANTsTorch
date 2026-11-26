"""Validate examples and optionally execute them without modifying tracked files."""
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
NOTEBOOKS = sorted((ROOT / "examples").rglob("*.ipynb"))
NOTEBOOKS = [p for p in NOTEBOOKS if ".ipynb_checkpoints" not in p.parts]
REQUIREMENT_MARKERS = {"requires_data", "tutorial_data", "slow", "cuda", "mps"}


def _read_notebook(path):
    # Missing tooling must fail when notebooks exist, rather than silently
    # claiming their validation passed.
    import nbformat

    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    return notebook


def _execution_cases():
    import json

    for path in NOTEBOOKS:
        # Read only metadata here; schema errors belong to the validation test.
        metadata = json.loads(path.read_text(encoding="utf-8")).get("metadata", {})
        tags = metadata.get("tags", [])
        marks = [getattr(pytest.mark, tag) for tag in REQUIREMENT_MARKERS if tag in tags]
        yield pytest.param(path, id=path.relative_to(ROOT).as_posix(), marks=marks)


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.relative_to(ROOT).as_posix())
def test_notebook_format(path):
    notebook = _read_notebook(path)
    assert any(cell.cell_type == "code" and cell.source.strip() for cell in notebook.cells), (
        f"{path.name} contains no executable example"
    )
    assert notebook.metadata.get("kernelspec", {}).get("language") == "python"


@pytest.mark.integration
@pytest.mark.parametrize("path", list(_execution_cases()))
def test_notebook_execution(path, request, tmp_path, monkeypatch):
    notebook = _read_notebook(path)
    _execute_notebook(notebook, request, tmp_path, monkeypatch)
    _assert_inline_figures(notebook)
    _assert_runtime_summary(notebook)


@pytest.mark.integration
@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.relative_to(ROOT).as_posix())
def test_notebook_plotting(path, request, tmp_path, monkeypatch):
    """Check every setup cell, including data-dependent notebooks, without data."""
    import nbformat

    source = _read_notebook(path)
    cells = [cell for cell in source.cells if cell.cell_type == "code"
             and "import matplotlib.pyplot as plt" in cell.source]
    assert cells, f"{path.name} has no plotting setup cell"
    notebook = nbformat.v4.new_notebook(cells=cells + [nbformat.v4.new_code_cell(
        "plt.figure(); plt.plot([0, 1], [0, 1]); plt.show()"
    )])
    _execute_notebook(notebook, request, tmp_path, monkeypatch)
    _assert_inline_figures(notebook)


def _assert_inline_figures(notebook):
    outputs = [output for cell in notebook.cells if cell.cell_type == "code"
               for output in cell.get("outputs", [])]
    assert not any("FigureCanvasAgg is non-interactive" in output.get("text", "")
                   for output in outputs), "Notebook uses a noninteractive Agg canvas"
    assert any({"image/png", "image/svg+xml"}.intersection(output.get("data", {}))
               for output in outputs), "Notebook produced no inline figures"


def _assert_runtime_summary(notebook):
    import re

    text = "".join(output.get("text", "")
                   for cell in notebook.cells if cell.cell_type == "code"
                   for output in cell.get("outputs", []))
    for label in ("Total elapsed time", "Timed operations"):
        match = re.search(rf"{label}: ([0-9]+\.[0-9]+) s", text)
        assert match is not None, f"Notebook did not report {label.lower()}"
        assert float(match.group(1)) >= 0


@pytest.mark.integration
def test_notebook_explicit_device_overrides_environment(request, tmp_path, monkeypatch):
    """A notebook must not restore MPS after the runner explicitly selected CPU."""
    import nbformat

    requested = request.config.getoption("--test-device")
    if requested == "auto":
        pytest.skip("This check needs an explicit --test-device")
    monkeypatch.setenv("ANTSTORCH_EXAMPLE_DEVICE", "mps")
    notebook = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(
        "import os\nimport torch\n"
        "device = os.environ.get('ANTSTORCH_EXAMPLE_DEVICE', 'mps')\n"
        f"assert device == {requested!r}, device\n"
        "value = torch.ones(1).to(device)\n"
        "assert value.device.type == torch.device(device).type\n"
    )])
    _execute_notebook(notebook, request, tmp_path, monkeypatch)


def _execute_notebook(notebook, request, tmp_path, monkeypatch):
    if not request.config.getoption("--run-notebooks"):
        pytest.skip("Use --run-notebooks to execute example notebooks")

    from nbclient import NotebookClient

    timeout = request.config.getoption("--notebook-timeout")
    if timeout <= 0:
        pytest.fail("--notebook-timeout must be positive")
    # The pytest device fixture lives in the parent process. Propagate its
    # choice explicitly to the notebook's separate Python interpreter.
    device = request.config.getoption("--test-device")
    import nbformat

    bootstrap = "import torch\n"
    if device != "auto":
        monkeypatch.setenv("ANTSTORCH_EXAMPLE_DEVICE", device)
        bootstrap += f"import antstorch\nantstorch.set_default_device({device!r})\n"
    notebook.cells.insert(0, nbformat.v4.new_code_cell(bootstrap))
    monkeypatch.setenv("MPLBACKEND", "Agg")
    client = NotebookClient(
        notebook, timeout=timeout, kernel_name="python3", allow_errors=False,
        force_raise_errors=True,
        resources={"metadata": {"path": str(tmp_path)}},
    )
    client.execute()
