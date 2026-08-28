"""antstorch.benchmark — Mindboggle-101 Registration Benchmark (ANTsTorch-native core)
=========================================================================================

A native port of the ``syntx.benchmark`` Mindboggle-101 90-pair evaluation
harness's *core* into ANTsTorch itself: single-pair evaluation, dataset
loading, and the accuracy/topology metrics, restricted to registration arms
that come from ANTsTorch itself (``antstorch.syn.syn_registration()``'s four
regularizers -- ``'gaussian'``, ``'sobolev'``, ``'dsti'``, ``'bspline'`` --
plus ``antstorch.bspline_flows.bspline_svf_registration()`` and
``gaussian_svf_registration()``).

By explicit scope decision (see the project doc, "Portage de l'evaluation
Mindboggle-101 dans ANTsTorch"), this port does NOT include:

- The command-line interface (``syntx.benchmark.cli``).
- The 90-pair cohort orchestrator, with its resume/cache/JSON bookkeeping
  (``syntx.benchmark.orchestrator``).
- The HTML results dashboard.
- The ANTs C++ baseline arm (plain ``ants.registration()``).
- Any ``syntx``-only registration arm (TVF, SyNGS, JAX backend, deep-feature
  similarity losses) -- these have no ANTsTorch equivalent and are not
  planned for one.

Exports are explicit and non-wildcard, matching this project's established
discipline in ``antstorch.syn``/``antstorch.bspline_flows``.
"""

from .data import (
    DEFAULT_DATA_DIR,
    DEFAULT_DATA_DIR_ENV,
    DEFAULT_PAIRS_CSV,
    MINDBOGGLE_SETUP_INSTRUCTIONS,
    check_mindboggle_data,
    get_n4_cached_subject_volume,
    load_mindboggle_pair,
    resolve_data_dir,
)
from .evaluate import (
    DEFAULT_REG_ITERATIONS,
    DEFAULT_REGISTRATION_LEVELS,
    DEFAULT_REGISTRATION_SMOOTHING_SIGMAS,
    clean_device_cache,
    evaluate_mindboggle_pair,
    evaluate_pair,
)
from .metrics import (
    compute_bending_energy,
    compute_bidirectional_dice,
    compute_harmonic_energy,
    compute_jacobian_metrics,
)

__all__ = [
    "DEFAULT_DATA_DIR",
    "DEFAULT_DATA_DIR_ENV",
    "DEFAULT_PAIRS_CSV",
    "DEFAULT_REG_ITERATIONS",
    "DEFAULT_REGISTRATION_LEVELS",
    "DEFAULT_REGISTRATION_SMOOTHING_SIGMAS",
    "MINDBOGGLE_SETUP_INSTRUCTIONS",
    "check_mindboggle_data",
    "get_n4_cached_subject_volume",
    "load_mindboggle_pair",
    "resolve_data_dir",
    "clean_device_cache",
    "evaluate_mindboggle_pair",
    "evaluate_pair",
    "compute_bending_energy",
    "compute_bidirectional_dice",
    "compute_harmonic_energy",
    "compute_jacobian_metrics",
]
