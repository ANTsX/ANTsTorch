"""Package imports must preserve the plotting backend chosen by the caller."""
from pathlib import Path
import subprocess
import sys


def test_import_preserves_matplotlib_backend():
    # A fresh process is needed: conftest and other tests may already have
    # imported ANTsTorch in this process. SVG works without a display server.
    result = subprocess.run(
        [sys.executable, "-c", "import torch; import matplotlib; "
         "matplotlib.use('svg'); import antstorch; "
         "assert matplotlib.get_backend().lower() == 'svg', matplotlib.get_backend()"],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
