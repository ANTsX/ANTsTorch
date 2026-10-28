#!/usr/bin/env python3
"""Run the ANTs vs ANTsTorch primitive comparisons on one device.

Runs, in sequence, ``compare_denoise_image.py``,
``compare_n4_bias_field_correction.py``, ``compare_cortical_thickness.py`` and
``compare_bspline_scattered_data.py`` with a common device and a common output directory. Each comparison runs in
its own process and writes to its own subdirectory with a ``log.txt``; a
failing comparison does not stop the others, and the exit status is nonzero
if any of them failed.

Without ``--device``, ``antstorch.get_default_device()`` is used, resolved
once so that all comparisons run on the same device.

Examples
--------
All four comparisons with their own default inputs (``S_template3`` for denoise, N4 and cortical thickness, synthetic data for
B-spline fitting)::

    python tools/benchmarks/run_primitive_benchmarks.py

One 3-D T1 shared by denoise, N4 and cortical thickness, on CUDA, with a
chosen output directory (the B-spline comparison keeps its synthetic data
unless ``--bspline-image`` is given)::

    python tools/benchmarks/run_primitive_benchmarks.py t1.nii.gz \\
        --device cuda:0 --out-dir results/primitives_cuda

Only N4 and denoising, with 5 timed denoise runs and a deterministic
single-threaded ANTs reference::

    python tools/benchmarks/run_primitive_benchmarks.py --only n4_bias_field_correction denoise_image \\
        --repeats 5 --ants-threads 1

Print the commands without running them::

    python tools/benchmarks/run_primitive_benchmarks.py --dry-run
"""

import argparse
import datetime
import importlib.metadata
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

BENCHMARK_DIR = Path(__file__).resolve().parent
REPO_ROOT = BENCHMARK_DIR.parents[1]

BENCHMARKS = ("denoise_image", "n4_bias_field_correction", "cortical_thickness", "bspline_scattered_data")


def script_path(name: str) -> Path:
    return BENCHMARK_DIR / f"compare_{name}.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "image",
        nargs="?",
        type=Path,
        help=(
            "Optional image shared by the denoise, N4 and cortical-thickness comparisons. "
            "Without it each script uses its own default (S_template3 for denoise, "
            "N4 and cortical thickness). Cortical thickness needs a 3-D T1. Not used by the "
            "B-spline comparison (see --bspline-image)."
        ),
    )
    parser.add_argument(
        "--device",
        default=None,
        help="PyTorch device, e.g. cpu, mps or cuda:0 (default: antstorch.get_default_device()).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output root (default: <repo>/results/primitive_benchmarks_<date>_<time>).",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        choices=BENCHMARKS,
        default=list(BENCHMARKS),
        help="Run only these comparisons (default: all four).",
    )
    parser.add_argument(
        "--bspline-image",
        default=None,
        help=(
            "Image path or bundled ANTsPy name (e.g. r16) whose pixel grid the B-spline "
            "comparison fits (default: synthetic data). Every pixel is a point, so avoid large 3-D images."
        ),
    )
    parser.add_argument("--repeats", type=int, default=5, help="Timed runs per engine for every primitive (default: 5).")
    parser.add_argument(
        "--ants-threads",
        type=int,
        default=None,
        help="ITK thread count for all ANTs comparisons (default: ITK default).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the commands without running them.")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")
    if args.ants_threads is not None and args.ants_threads < 1:
        parser.error("--ants-threads must be at least 1")
    if args.image is not None:
        if not args.image.is_file():
            parser.error(f"image not found: {args.image}")
        args.image = args.image.resolve()
    return args


def resolve_device(requested):
    """Resolve the device once so every comparison runs on the same one."""
    if requested:
        return requested
    import antstorch

    return str(antstorch.get_default_device())


def build_command(name: str, args: argparse.Namespace, device: str, out_dir: Path) -> list:
    command = [sys.executable, str(script_path(name)), "--repeats", str(args.repeats)]
    if name == "bspline_scattered_data":
        # This script takes --image and --output-prefix instead of a positional image and --output-dir.
        if args.bspline_image is not None:
            command += ["--image", args.bspline_image]
        return command + [
            "--device", device,
            "--output-prefix", str(out_dir / name / "bspline_scattered_comparison"),
        ]
    if args.image is not None:
        command.append(str(args.image))
    command += ["--device", device, "--output-dir", str(out_dir / name)]
    if name == "denoise_image":
        if args.ants_threads is not None:
            command += ["--ants-threads", str(args.ants_threads)]
    elif name == "n4_bias_field_correction":
        # Avoids the slow per-level setup of deterministic accumulation on MPS.
        command.append("--no-stable-accumulation")
    return command


def git_state() -> str:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT, capture_output=True, text=True, check=True,
        ).stdout.strip()
        dirty = subprocess.run(["git", "diff", "--quiet"], cwd=REPO_ROOT).returncode != 0
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return commit + (" (modified)" if dirty else "")


def write_run_info(out_dir: Path, device: str, args: argparse.Namespace) -> None:
    lines = [
        f"date:   {datetime.datetime.now().astimezone().isoformat(timespec='seconds')}",
        f"host:   {platform.node()}",
        f"device: {device}",
        f"image:  {args.image or 'script defaults (S_template3)'}",
        f"commit: {git_state()}",
        f"python: {sys.version.split()[0]}",
        f"repeats: {args.repeats}",
        f"ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS: {os.environ.get('ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS', 'ITK default')}",
        "commands:",
        *["  " + " ".join(build_command(name, args, device, out_dir)) for name in BENCHMARKS if name in args.only],
    ]
    lines.append(f"platform: {platform.platform()}")
    for package in ("torch", "antspyx", "numpy"):
        try:
            lines.append(f"{package}: {importlib.metadata.version(package)}")
        except importlib.metadata.PackageNotFoundError:
            lines.append(f"{package}: unavailable")
    if shutil.which("nvidia-smi"):
        gpu = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,driver_version,memory.total", "--format=csv"],
            capture_output=True, text=True,
        )
        if gpu.returncode == 0:
            lines.append(gpu.stdout.strip())
    text = "\n".join(lines) + "\n"
    (out_dir / "run_info.txt").write_text(text, encoding="utf-8")
    print(text, end="")


def run_benchmark(name: str, command: list, device: str, out_dir: Path) -> tuple:
    """Run one comparison, teeing its output to the terminal and ``log.txt``."""
    bench_dir = out_dir / name
    bench_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'=' * 66}\n {name}  ({device})\n{'=' * 66}", flush=True)
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    start = time.perf_counter()
    with open(bench_dir / "log.txt", "w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command, cwd=REPO_ROOT, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log.write(line)
            log.flush()
        returncode = process.wait()
    return returncode, time.perf_counter() - start


def main() -> int:
    args = parse_args()
    if args.ants_threads is not None:
        os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(args.ants_threads)
    device = resolve_device(args.device)
    out_dir = args.out_dir or REPO_ROOT / "results" / (
        "primitive_benchmarks_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    out_dir = out_dir.resolve()
    names = [name for name in BENCHMARKS if name in args.only]
    commands = {name: build_command(name, args, device, out_dir) for name in names}

    if args.dry_run:
        print(f"Output root: {out_dir}")
        for name in names:
            print(f"[{name}] " + " ".join(commands[name]))
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    write_run_info(out_dir, device, args)

    results = {name: run_benchmark(name, commands[name], device, out_dir) for name in names}

    lines = [f"Summary ({out_dir})"]
    for name, (returncode, seconds) in results.items():
        status = "OK" if returncode == 0 else f"FAILED (exit {returncode})"
        lines.append(f"  {name:<26} {status:<18} {seconds:7.1f} s")
    summary = "\n".join(lines) + "\n"
    (out_dir / "summary.txt").write_text(summary, encoding="utf-8")
    print("\n" + summary, end="")
    return int(any(returncode != 0 for returncode, _ in results.values()))


if __name__ == "__main__":
    sys.exit(main())
