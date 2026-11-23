"""Port of ``syntx.core.convergence`` (syntx v6.39.0, verbatim).

One convergence strategy for every deformable registration method (greedy, SyN, SyN-GS, TVF, their JAX twins).

The iteration count of a level is a CAP, never the stopping rule: a level ends as soon as its loss has stopped decreasing
(``LossSlopeStop``), and every level reports whether it converged or was capped, in one record format, so a result can be
audited and the methods compared. (The linear solvers use the pose-step test of ``core.optimizers.PointStepTracker`` with
the manifold probe of ``core.parameter_scales``; a deformable field has no low-dimensional pose to track, so the loss is
the evidence.)

The rule is the same everywhere: the least-squares slope of the loss over the last ``window`` samples, per ITERATION (the
x axis is the real iteration index, so the rule does not depend on how often a method reads the loss back from the device),
must be within ``+-slope_threshold``: the loss is no longer changing by more than ``slope_threshold`` per iteration. A loss that is
still falling by more than that has not converged; one that is rising by more than that is diverging, which is the caller's
divergence handling (restore the best field, retry), not convergence.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

DEFAULT_WINDOW = 10
DEFAULT_SLOPE_THRESHOLD = 1e-6


def loss_slope(iterations: Sequence[float], losses: Sequence[float]) -> Optional[float]:
    """Least-squares slope (loss per iteration) of ``losses`` against their iteration indices; ``None`` if undefined."""
    x = np.asarray(iterations, dtype=float)
    y = np.asarray(losses, dtype=float)
    if x.size < 2 or not np.all(np.isfinite(y)):
        return None
    xm = x - x.mean()
    denom = float(np.sum(xm * xm))
    if denom < 1e-12:
        return None
    return float(np.sum(xm * (y - y.mean())) / denom)


class LossSlopeStop:
    """Per-level loss-slope stopping rule and the convergence record of a deformable registration.

    Usage inside a level loop::

        stop.start_level(level_idx, cap=n_iters)
        for it in range(n_iters):
            ...                                            # one iteration
            if stop.update(it, loss_value):                # call at whatever cadence the loss is available
                break
        stop.end_level(iterations_run=it + 1)

    ``slope_threshold=None`` (or <= 0) disables the early stop: levels run to their cap and are all reported capped.
    """

    def __init__(self, window: int = DEFAULT_WINDOW, slope_threshold: Optional[float] = DEFAULT_SLOPE_THRESHOLD):
        if int(window) < 3:
            raise ValueError(f"window must be >= 3 samples, got {window!r}")
        self.window = int(window)
        self.slope_threshold = None if slope_threshold is None or float(slope_threshold) <= 0 else float(slope_threshold)
        self.records: List[dict] = []
        self._level = None
        self._cap = None
        self._its: List[int] = []
        self._losses: List[float] = []
        self._converged = False
        self._slope = None

    @property
    def enabled(self) -> bool:
        return self.slope_threshold is not None

    def start_level(self, level, cap: int) -> None:
        self._level, self._cap = level, int(cap)
        self._its, self._losses = [], []
        self._converged, self._slope = False, None

    def update(self, iteration: int, loss: float) -> bool:
        """Add one loss sample; True when the level has converged."""
        self._its.append(int(iteration))
        self._losses.append(float(loss))
        return self._test()

    def update_series(self, losses: Sequence[float], first_iteration: int = 0) -> bool:
        """Replace the samples by a whole per-iteration series (methods that already keep every loss); True when converged."""
        self._its = list(range(int(first_iteration), int(first_iteration) + len(losses)))
        self._losses = [float(v) for v in losses]
        return self._test()

    def _test(self) -> bool:
        if not self.enabled or len(self._losses) < self.window:
            return False
        slope = loss_slope(self._its[-self.window:], self._losses[-self.window:])
        self._slope = slope
        if slope is not None and abs(slope) <= self.slope_threshold:
            self._converged = True
        return self._converged

    def end_level(self, iterations_run: int) -> dict:
        """Close the level: append and return its record ``{level, iterations, cap, converged, final_slope}``."""
        rec = {"level": self._level, "iterations": int(iterations_run), "cap": self._cap,
               "converged": bool(self._converged), "final_slope": self._slope}
        if self.records and self.records[-1]["level"] == rec["level"]:
            self.records[-1] = rec          # a level re-run (e.g. after a divergence retry) keeps its last attempt
        else:
            self.records.append(rec)
        return rec

    @property
    def all_converged(self) -> bool:
        return bool(self.records) and all(r["converged"] for r in self.records)

    def summary(self) -> dict:
        """Compact summary for provenance: threshold, window, per-level records and the capped levels."""
        return {"rule": "|loss slope per iteration| over the last window samples <= slope_threshold",
                "window": self.window, "slope_threshold": self.slope_threshold, "levels": list(self.records),
                "capped_levels": [r["level"] for r in self.records if not r["converged"]]}
