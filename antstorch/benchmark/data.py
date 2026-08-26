"""antstorch.benchmark.data — Mindboggle-101 Dataset Management and Pair Loading
=================================================================================

Handles dataset path resolution, integrity checks, and pair loading for the
standardized 90-pair Mindboggle registration benchmark, ported from
``syntx.benchmark.data`` and trimmed to what :func:`antstorch.benchmark.evaluate.
evaluate_mindboggle_pair` actually needs (no dataset-organizing/precompute
helpers — see the project doc for the full-port items left out of this
initial core-only port).

Two deliberate departures from the ``syntx`` original, both favoring
portability over exact parity:

- ``DEFAULT_PAIRS_CSV`` is resolved relative to *this module's own file*
  (``pairs.csv``, bundled alongside it) rather than relative to the caller's
  current working directory — ``syntx.benchmark``'s own default
  (``"examples/pairs.csv"``) only resolves correctly when the caller's cwd
  happens to be the ``syntx`` repository root.
- ``DEFAULT_DATA_DIR`` is a generic ``~/data/mindboggle/volumes`` rather than
  a specific collaborator's machine path (``syntx.benchmark.data``'s own
  default, ``/Users/stnava/data/mindboggle/volumes``, only ever resolves on
  that one machine). Point ``ANTSTORCH_MINDBOGGLE_DATA_DIR`` (or the
  ``data_dir`` argument) at the real dataset location instead.
"""

import os
import sys
import pandas as pd
import ants
from typing import Dict, Any, Optional, Tuple

DEFAULT_PAIRS_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pairs.csv")
DEFAULT_DATA_DIR_ENV = "ANTSTORCH_MINDBOGGLE_DATA_DIR"
# Checked as a fallback if ANTSTORCH_MINDBOGGLE_DATA_DIR is unset, purely as
# a convenience for anyone who already has the syntx harness configured on
# the same machine -- never required.
_FALLBACK_DATA_DIR_ENV = "SYNTX_DATA_DIR"
DEFAULT_DATA_DIR = os.path.expanduser("~/data/mindboggle/volumes")

MINDBOGGLE_SETUP_INSTRUCTIONS = """
================================================================================
                    MINDBOGGLE-101 DATASET SETUP GUIDE
================================================================================

The Mindboggle benchmark requires the 101 manually labeled T1-weighted brain MRI
volumes and DKT31 cortical label maps from the Mindboggle-101 project.

Expected Directory Hierarchy:
-----------------------------
$ANTSTORCH_MINDBOGGLE_DATA_DIR/ (default: ~/data/mindboggle/volumes)
  ├── OASIS-TRT-20_volumes/
  │   ├── OASIS-TRT-20-1/
  │   │   ├── t1weighted_brain.nii.gz
  │   │   └── labels.DKT31.manual.nii.gz
  │   └── ... (20 subjects)
  ├── NKI-RS-22_volumes/
  │   ├── NKI-RS-22-1/
  │   │   ├── t1weighted_brain.nii.gz
  │   │   └── labels.DKT31.manual.nii.gz
  │   └── ... (22 subjects)
  ├── NKI-TRT-20_volumes/
  │   ├── NKI-TRT-20-1/
  │   │   ├── t1weighted_brain.nii.gz
  │   │   └── labels.DKT31.manual.nii.gz
  │   └── ... (20 subjects)
  └── MMRR-21_volumes/
      ├── MMRR-21-1/
      │   ├── t1weighted_brain.nii.gz
      │   └── labels.DKT31.manual.nii.gz
      └── ... (21 subjects)

Pairs Configuration File:
-------------------------
By default this uses the 90-pair definition bundled with antstorch itself
(antstorch/benchmark/pairs.csv, ported verbatim from syntx's own
examples/pairs.csv: 40 intra-subject and 50 inter-subject pairs). Pass
`pairs_csv=...` to use a different one.

How to Set the Data Directory:
------------------------------
Option A: Export environment variable in your shell profile:
    export ANTSTORCH_MINDBOGGLE_DATA_DIR="/path/to/your/mindboggle/volumes"

Option B: Pass `data_dir` directly to benchmark functions:
    antstorch.benchmark.evaluate_mindboggle_pair(pair_idx=0, data_dir="/path/to/volumes")

Download & Reference:
---------------------
Mindboggle-101 Dataset: https://mindboggle.info/data.html
Citation: Klein A, Tourville J. 101 labeled brain images and a consistent human
cortical labeling protocol. Front Neurosci. 2012;6:171.
================================================================================
"""


def resolve_data_dir(data_dir: Optional[str] = None) -> str:
    """Resolves the Mindboggle data directory from argument, environment, or default.

    Raises a descriptive ``FileNotFoundError`` with setup instructions if the
    resolved directory does not exist.
    """
    if data_dir is not None and str(data_dir).strip():
        resolved = os.path.abspath(os.path.expanduser(str(data_dir)))
    else:
        resolved = os.environ.get(
            DEFAULT_DATA_DIR_ENV,
            os.environ.get(_FALLBACK_DATA_DIR_ENV, DEFAULT_DATA_DIR),
        )
        resolved = os.path.abspath(os.path.expanduser(resolved))

    if not os.path.isdir(resolved):
        print(MINDBOGGLE_SETUP_INSTRUCTIONS, file=sys.stderr)
        raise FileNotFoundError(
            f"Mindboggle data directory not found at: '{resolved}'\n"
            f"Please set the {DEFAULT_DATA_DIR_ENV} environment variable or pass `data_dir`."
        )
    return resolved


def check_mindboggle_data(
    pairs_csv: str = DEFAULT_PAIRS_CSV,
    data_dir: Optional[str] = None,
    verbose: bool = False
) -> Tuple[bool, Dict[str, Any]]:
    """Verifies that the Mindboggle pairs CSV and required volume files exist.

    Parameters
    ----------
    pairs_csv : str
        Path to the pairs CSV file defining the 90 pairs.
    data_dir : str, optional
        Root directory containing the cohort subdirectories.
    verbose : bool
        If True, prints diagnostic setup instructions on missing data.

    Returns
    -------
    Tuple[bool, Dict[str, Any]]
        (is_valid, report_dictionary)
    """
    report = {
        "pairs_csv_path": os.path.abspath(pairs_csv),
        "pairs_csv_exists": os.path.exists(pairs_csv),
        "data_dir": None,
        "data_dir_exists": False,
        "total_pairs_in_csv": 0,
        "available_pairs": 0,
        "missing_pairs": [],
        "missing_files": []
    }

    if not os.path.exists(pairs_csv):
        if verbose:
            print(f"[antstorch.benchmark] ERROR: Pairs CSV not found at '{pairs_csv}'", file=sys.stderr)
            print(MINDBOGGLE_SETUP_INSTRUCTIONS, file=sys.stderr)
        return False, report

    try:
        data_dir_resolved = resolve_data_dir(data_dir)
        report["data_dir"] = data_dir_resolved
        report["data_dir_exists"] = True
    except FileNotFoundError:
        return False, report

    df = pd.read_csv(pairs_csv)
    report["total_pairs_in_csv"] = len(df)

    missing_pairs = []
    missing_files = []

    for idx, row in df.iterrows():
        c1, s1 = str(row["cohort1"]), str(row["subject1"])
        c2, s2 = str(row["cohort2"]), str(row["subject2"])

        p_fix = os.path.join(data_dir_resolved, f"{c1}_volumes", s1, "t1weighted_brain.nii.gz")
        p_flab = os.path.join(data_dir_resolved, f"{c1}_volumes", s1, "labels.DKT31.manual.nii.gz")
        p_mov = os.path.join(data_dir_resolved, f"{c2}_volumes", s2, "t1weighted_brain.nii.gz")
        p_mlab = os.path.join(data_dir_resolved, f"{c2}_volumes", s2, "labels.DKT31.manual.nii.gz")

        pair_missing = []
        for pth in [p_fix, p_flab, p_mov, p_mlab]:
            if not os.path.exists(pth):
                pair_missing.append(pth)
                missing_files.append(pth)

        if pair_missing:
            missing_pairs.append({"pair_idx": int(idx), "missing": pair_missing})
        else:
            report["available_pairs"] += 1

    report["missing_pairs"] = missing_pairs
    report["missing_files"] = list(set(missing_files))

    is_valid = (len(missing_pairs) == 0 and report["total_pairs_in_csv"] > 0)
    if is_valid and verbose:
        print(f"[antstorch.benchmark] Dataset Location: '{data_dir_resolved}'", flush=True)
        print(f"[antstorch.benchmark] Pairs Configuration: '{os.path.abspath(pairs_csv)}'", flush=True)
    elif not is_valid and verbose:
        print(f"[antstorch.benchmark] Incomplete Mindboggle dataset! Found {report['available_pairs']}/{report['total_pairs_in_csv']} pairs at '{data_dir_resolved}'.", file=sys.stderr)
        print(f"[antstorch.benchmark] {len(report['missing_files'])} missing image/label files detected.", file=sys.stderr)
        print(MINDBOGGLE_SETUP_INSTRUCTIONS, file=sys.stderr)

    return is_valid, report


def get_n4_cached_subject_volume(
    cohort: str,
    subject: str,
    raw_brain_path: str,
    data_dir: str,
    use_n4: bool = True,
    device: Optional[str] = None,
    verbose: bool = False
) -> ants.ANTsImage:
    """Loads an N4-bias-corrected subject volume from disk cache, or computes and caches it.

    Uses :func:`antstorch.bspline_flows.n4_bias_field_correction` directly
    (an in-package call, unlike ``syntx.benchmark.data``'s own version of
    this function, which reaches ``antstorch`` as an external, optional
    dependency).
    """
    if not use_n4:
        return ants.image_read(raw_brain_path)

    cache_dir = os.path.join(data_dir, ".n4_cache", f"{cohort}_volumes", subject)
    cache_file = os.path.join(cache_dir, "t1weighted_brain_n4.nii.gz")

    if os.path.exists(cache_file):
        return ants.image_read(cache_file)

    # Not in cache: compute N4 with antstorch's own bspline_flows N4 filter.
    raw_img = ants.image_read(raw_brain_path)
    try:
        import torch
        from antstorch.bspline_flows import n4_bias_field_correction

        arr = raw_img.numpy()
        tensor = torch.from_numpy(arr.transpose(2, 1, 0)).unsqueeze(0).unsqueeze(0).float()
        if device is not None:
            tensor = tensor.to(device)
        mask = (tensor > 0.01).to(tensor.dtype)

        if verbose:
            print(f"[antstorch.benchmark] Computing N4 correction for {subject}...", flush=True)

        corrected_tensor = n4_bias_field_correction(
            tensor,
            mask=mask,
            shrink_factor=4,
            convergence={"iters": [50, 50, 50, 50], "tol": 1e-7}
        )
        corrected_arr = corrected_tensor.squeeze().detach().cpu().numpy().transpose(2, 1, 0)
        corrected_img = ants.from_numpy(
            corrected_arr,
            origin=raw_img.origin,
            spacing=raw_img.spacing,
            direction=raw_img.direction
        )
        os.makedirs(cache_dir, exist_ok=True)
        ants.image_write(corrected_img, cache_file)
        return corrected_img
    except Exception as e:
        if verbose:
            print(f"[antstorch.benchmark] WARNING: N4 correction failed for {subject}: {e}. Falling back to raw volume.", file=sys.stderr)
        return raw_img


def load_mindboggle_pair(
    pair_idx: int,
    pairs_csv: str = DEFAULT_PAIRS_CSV,
    data_dir: Optional[str] = None,
    use_n4: bool = True,
    verbose: bool = False
) -> Dict[str, Any]:
    """Loads a single image pair and ground-truth segmentation pair from the CSV.

    Parameters
    ----------
    pair_idx : int
        Index of the pair in the CSV file (0 to 89 for the bundled default).
    pairs_csv : str
        Path to the pairs CSV file.
    data_dir : str, optional
        Root directory containing the cohort subdirectories.
    use_n4 : bool, default=True
        If True, loads the N4-bias-corrected brain volume from cache
        (computing and caching it via antstorch's own N4 filter if absent).
    verbose : bool
        If True, prints progress details.

    Returns
    -------
    Dict[str, Any]
        ``ants.ANTsImage`` objects for ``'fixed'``, ``'moving'``,
        ``'fixed_label'``, ``'moving_label'``, plus metadata.
    """
    if not os.path.exists(pairs_csv):
        print(MINDBOGGLE_SETUP_INSTRUCTIONS, file=sys.stderr)
        raise FileNotFoundError(f"Pairs CSV not found: '{pairs_csv}'")

    data_dir_resolved = resolve_data_dir(data_dir)
    df = pd.read_csv(pairs_csv)

    if pair_idx < 0 or pair_idx >= len(df):
        raise IndexError(
            f"pair_idx={pair_idx} out of range [0, {len(df) - 1}]. "
            f"CSV has {len(df)} pairs."
        )

    row = df.iloc[pair_idx]
    c1, s1 = str(row["cohort1"]), str(row["subject1"])
    c2, s2 = str(row["cohort2"]), str(row["subject2"])

    paths = {
        "fixed": os.path.join(data_dir_resolved, f"{c1}_volumes", s1, "t1weighted_brain.nii.gz"),
        "fixed_label": os.path.join(data_dir_resolved, f"{c1}_volumes", s1, "labels.DKT31.manual.nii.gz"),
        "moving": os.path.join(data_dir_resolved, f"{c2}_volumes", s2, "t1weighted_brain.nii.gz"),
        "moving_label": os.path.join(data_dir_resolved, f"{c2}_volumes", s2, "labels.DKT31.manual.nii.gz"),
    }

    for name, path in paths.items():
        if not os.path.exists(path):
            print(MINDBOGGLE_SETUP_INSTRUCTIONS, file=sys.stderr)
            raise FileNotFoundError(f"Missing Mindboggle {name} volume: '{path}'")

    fixed_img = get_n4_cached_subject_volume(c1, s1, paths["fixed"], data_dir_resolved, use_n4=use_n4, verbose=verbose)
    moving_img = get_n4_cached_subject_volume(c2, s2, paths["moving"], data_dir_resolved, use_n4=use_n4, verbose=verbose)

    return {
        "pair_idx": int(pair_idx),
        "fixed": fixed_img,
        "moving": moving_img,
        "fixed_label": ants.image_read(paths["fixed_label"]),
        "moving_label": ants.image_read(paths["moving_label"]),
        "fixed_id": s1,
        "moving_id": s2,
        "cohort1": c1,
        "cohort2": c2,
        "pair_type": str(row.get("type", "intra" if c1 == c2 else "inter")),
        "use_n4": use_n4,
    }
