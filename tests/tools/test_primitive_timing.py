"""Timing protocol checks without running the expensive image engines."""
import importlib.util
from pathlib import Path
from unittest.mock import patch

import pytest

BENCHMARKS = Path(__file__).resolve().parents[2] / "tools" / "benchmarks"


def load(name):
    spec = importlib.util.spec_from_file_location(name, BENCHMARKS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_repeated_timing_synchronizes_and_records_samples(tmp_path):
    timing = load("benchmark_timing")
    events = []
    records = {}
    with patch.object(timing, "synchronize", side_effect=lambda d: events.append("sync")), patch.object(
        timing.time, "perf_counter", side_effect=[0, 5, 10, 11, 20, 23]
    ):
        result, median = timing.measure(
            "engine", lambda: events.append("call") or 42, 3, records
        )
    assert result == 42
    assert median == 3
    assert events == ["sync", "call", "sync"] * 3
    assert records["engine"]["samples_seconds"] == [5, 1, 3]
    assert records["engine"]["q1_seconds"] == 2
    assert records["engine"]["q3_seconds"] == 4
    timing.write_timings(tmp_path / "timings.json", records)
    import json
    assert json.loads((tmp_path / "timings.json").read_text()) == records


def test_invalid_repeats_do_not_call_engine():
    timing = load("benchmark_timing")
    with pytest.raises(ValueError, match="at least 1"):
        timing.measure("engine", lambda: pytest.fail("called"), 0, {})


def test_launcher_forwards_repeats_to_every_primitive(tmp_path):
    launcher = load("run_primitive_benchmarks")
    with patch("sys.argv", ["benchmark", "--repeats", "10"]):
        args = launcher.parse_args()
    for name in launcher.BENCHMARKS:
        command = launcher.build_command(name, args, "cuda:0", tmp_path)
        assert command.count("--repeats") == 1
        assert command[command.index("--repeats") + 1] == "10"
