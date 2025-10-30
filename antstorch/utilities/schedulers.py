# schedulers.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Iterable, List
import math

__all__ = [
    "ParamSchedule",
    "MultiParamScheduler",
    "parse_schedules",
]

"""
Multi-parameter scalar schedulers (linear / cosine / exponential) with a tiny CLI DSL.

This module provides:
- ``ParamSchedule``: a single scalar schedule (start→end over N steps) with optional
  delay, floor/ceil clamping, and sparse updates.
- ``MultiParamScheduler``: a collection of named schedules producing a dict at each step.
- ``parse_schedules``: a tiny parser for a concise CLI string describing multiple schedules.

Schedule modes
--------------
- ``linear`` (or ``lin``):
    v(p) = start + (end - start) * p
- ``cosine`` (or ``cos``):
    Smooth ease-in/out via half-cosine: w = 0.5*(1 - cos(pi*p)); v = start + (end - start)*w
- ``exponential`` (or ``exp``):
    v(p) = start * (end/start) ** p
    Requires ``start`` and ``end`` share the same sign and ``start != 0``; otherwise
    falls back to linear.

Progress p
----------
For a global step ``t`` and optional ``delay``, progress is:
    p = clamp((t - delay) / total_steps, 0, 1)

CLI DSL grammar
---------------
Each schedule:
    ``<name>:<mode>:<start>-><end>@<steps>[+key=val][+key=val]...``

Multiple schedules are comma-separated. Numeric suffixes ``k`` and ``m`` are supported:
``150k`` = 150_000, ``2m`` = 2_000_000.

Supported ``key`` options:
- ``delay`` (int): steps to hold the start value before ramping.
- ``hold`` (int): bookkeeping only (no behavior change).
- ``floor`` (float): minimum value clamp.
- ``ceil`` (float): maximum value clamp.
- ``steps_per_update`` (int): compute value only every N steps (deterministic).

Examples
--------
>>> # Two schedules from a CLI string
>>> spec = "noise_std:cos:0.05->0.00@150k,sd_deformation:linear:10.0->0.0@100k+delay=5k"
>>> mp = MultiParamScheduler(parse_schedules(spec))
>>> mp.step(0)["noise_std"]
0.05
>>> mp.step(150_000)["noise_std"]
0.0

Notes
-----
- All computations are deterministic given the integer ``global_step``.
- ``steps_per_update`` is included for convenience; values are still computed
  deterministically via ``value_at(step)`` so resuming is trivial.
"""


def _interp(start: float, end: float, p: float, mode: str) -> float:
    """
    Interpolate between ``start`` and ``end`` using progress ``p`` in the requested mode.

    Parameters
    ----------
    start : float
        Start value at progress ``p = 0``.
    end : float
        End value at progress ``p = 1``.
    p : float
        Normalized progress in ``[0, 1]``. Values are clamped to this range.
    mode : str
        One of ``{"linear","lin","cos","cosine","exp","exponential"}``.

    Returns
    -------
    float
        Interpolated value.

    Raises
    ------
    ValueError
        If ``mode`` is not recognized.

    Notes
    -----
    Exponential interpolation falls back to linear if ``start == 0`` or signs of
    ``start`` and ``end`` differ.
    """
    mode = mode.lower()
    p = min(max(p, 0.0), 1.0)
    if mode in ("lin", "linear"):
        return start + (end - start) * p
    elif mode in ("cos", "cosine"):
        # starts near start, ends at end with a smooth finish
        w = 0.5 * (1.0 - math.cos(math.pi * p))  # 0→1
        return start + (end - start) * w
    elif mode in ("exp", "exponential"):
        # Exponential interpolation; requires start and end have same sign and start!=0
        if start == 0.0 or (start < 0) != (end < 0):
            # fall back to linear if invalid for exp
            return start + (end - start) * p
        ratio = end / start
        return start * (ratio ** p)
    else:
        raise ValueError(f"Unknown mode '{mode}'")


@dataclass
class ParamSchedule:
    """
    Single-parameter schedule from ``start`` to ``end`` over ``total_steps``.

    Attributes
    ----------
    name : str
        Name/key of this parameter in a multi-parameter scheduler dictionary.
    start : float
        Start value at progress 0.
    end : float
        End value at progress 1.
    total_steps : int
        Number of steps for the ramp (excluding any initial ``delay``).
    mode : str, default="cos"
        Interpolation mode: ``"linear"|"lin"|"cos"|"cosine"|"exp"|"exponential"``.
    delay : int, default=0
        Steps to hold the start value before beginning the ramp.
    hold : int, default=0
        Post-finish hold (bookkeeping only; does not affect the value).
    floor : float or None, default=None
        Minimum clamp applied to the interpolated value.
    ceil : float or None, default=None
        Maximum clamp applied to the interpolated value.
    steps_per_update : int, default=1
        Only update the value every N steps (deterministic; mainly for callers'
        performance if re-applying expensive transforms).
    """

    name: str
    start: float
    end: float
    total_steps: int
    mode: str = "cos"
    delay: int = 0
    hold: int = 0
    floor: Optional[float] = None
    ceil: Optional[float] = None
    steps_per_update: int = 1

    def value_at(self, step: int) -> float:
        """
        Compute the scheduled value at a given global step.

        Parameters
        ----------
        step : int
            Global step (e.g., number of completed optimizer steps). Should be ``>= 0``.

        Returns
        -------
        float
            The scheduled value after applying delay, interpolation, and clamps.

        Notes
        -----
        - Progress is computed as:
          ``p = clamp((step - delay) / total_steps, 0, 1)``.
        - If ``total_steps <= 0``, the ramp is treated as complete (``p = 1``).
        - ``floor`` and ``ceil`` are applied after interpolation.
        """
        if step < self.delay:
            v = self.start
        else:
            if self.total_steps <= 0:
                p = 1.0
            else:
                # 0-based progress when no delay (keeps value_at(0) == start),
                # 1-based progress when a positive delay is present so that
                # the first step at 'delay' already moves off 'start'.
                if self.delay == 0:
                    s_eff = min(max(step, 0), self.total_steps)            # 0..N
                else:
                    s_eff = min(max(step - self.delay + 1, 1), self.total_steps)  # 1..N
                p = s_eff / float(self.total_steps)
            v = _interp(self.start, self.end, p, self.mode)

        if self.floor is not None:
            v = max(v, self.floor)
        if self.ceil is not None:
            v = min(v, self.ceil)
        return v

class MultiParamScheduler:
    """
    Manage multiple named schedules and produce a dictionary of values per step.

    Parameters
    ----------
    schedules : Iterable[ParamSchedule]
        The collection of parameter schedules to track.

    Notes
    -----
    The object is stateless with respect to time; pass the desired ``global_step`` to
    :meth:`step` whenever you need current values. This makes resume trivial.
    """

    def __init__(self, schedules: Iterable[ParamSchedule]):
        self._schedules: Dict[str, ParamSchedule] = {s.name: s for s in schedules}

    def step(self, global_step: int) -> Dict[str, float]:
        """
        Compute all parameter values at ``global_step``.

        Parameters
        ----------
        global_step : int
            Global step (e.g., number of completed optimizer steps).

        Returns
        -------
        dict[str, float]
            Mapping from schedule name to its current value.

        Notes
        -----
        If a schedule's ``steps_per_update > 1``, this method still returns the
        deterministic value at ``global_step``. Callers that wish to *cache* values to
        reduce downstream recomputation can check ``steps_per_update`` and reuse prior
        outputs between updates.
        """
        out: Dict[str, float] = {}
        for name, sched in self._schedules.items():
            if sched.steps_per_update > 1 and (global_step % sched.steps_per_update != 0):
                # Intentionally still compute a deterministic value; callers may cache.
                pass
            out[name] = sched.value_at(global_step)
        return out

    def get(self, name: str) -> ParamSchedule:
        """
        Retrieve a schedule by name.

        Parameters
        ----------
        name : str
            Schedule key to fetch.

        Returns
        -------
        ParamSchedule
            The stored schedule.

        Raises
        ------
        KeyError
            If no schedule exists with the given name.
        """
        return self._schedules[name]

    def set(self, name: str, schedule: ParamSchedule):
        """
        Replace or insert a schedule.

        Parameters
        ----------
        name : str
            Key at which to store the schedule (usually matches ``schedule.name``).
        schedule : ParamSchedule
            The schedule to store.
        """
        self._schedules[name] = schedule


def _parse_num(s: str) -> int:
    """
    Parse an integer possibly suffixed with ``k`` (thousands) or ``m`` (millions).

    Parameters
    ----------
    s : str
        String such as ``"150k"``, ``"2m"``, or a plain number like ``"10000"``.

    Returns
    -------
    int
        Parsed integer value.

    Examples
    --------
    >>> _parse_num("150k")
    150000
    >>> _parse_num("2m")
    2000000
    >>> _parse_num("42")
    42
    """
    s = s.strip().lower()
    mul = 1
    if s.endswith("k"):
        mul = 1_000
        s = s[:-1]
    elif s.endswith("m"):
        mul = 1_000_000
        s = s[:-1]
    return int(float(s) * mul)


def parse_schedules(spec: str) -> List[ParamSchedule]:
    """
    Parse a CLI-style spec into a list of :class:`ParamSchedule` objects.

    Parameters
    ----------
    spec : str
        Comma-separated schedules. Each entry must follow:
        ``<name>:<mode>:<start>-><end>@<steps>[+key=val][+key=val]...``

        Supported ``mode`` values: ``"linear"|"lin"|"cos"|"cosine"|"exp"|"exponential"``.
        Keys: ``delay`` (int), ``hold`` (int), ``steps_per_update`` (int),
        ``floor`` (float), ``ceil`` (float). Numeric suffixes ``k`` and ``m``
        are allowed for integer fields.

    Returns
    -------
    list[ParamSchedule]
        Parsed schedules in the order they were specified.

    Raises
    ------
    ValueError
        If an entry is malformed or contains an unknown option.

    Examples
    --------
    >>> spec = "noise_std:cos:0.05->0.00@150k+delay=5k,sd_deformation:linear:10.0->0.0@100k"
    >>> schedules = parse_schedules(spec)
    >>> [s.name for s in schedules]
    ['noise_std', 'sd_deformation']
    >>> schedules[0].mode
    'cos'
    """
    if not spec.strip():
        return []
    schedules: List[ParamSchedule] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        # name:mode:start->end@steps +key=val +key=val ...
        head, *mods = part.split("+")
        try:
            name, mode, se_steps = head.split(":", 3)
        except ValueError as e:
            raise ValueError(f"Malformed schedule head: '{head}' in '{part}'") from e

        try:
            start_end, steps_str = se_steps.split("@", 2)
            start_str, end_str = start_end.split("->", 2)
        except ValueError as e:
            raise ValueError(f"Malformed start/end/steps in '{part}'") from e

        try:
            start = float(start_str)
            end = float(end_str)
            total_steps = _parse_num(steps_str)
        except Exception as e:
            raise ValueError(f"Bad numeric value in '{part}': {e}") from e

        kwargs = {}
        for m in mods:
            if not m.strip():
                continue
            try:
                k, v = m.split("=", 1)
            except ValueError as e:
                raise ValueError(f"Malformed option '+{m}' in '{part}'") from e
            k = k.strip()
            v = v.strip()
            if k in ("delay", "hold", "steps_per_update"):
                kwargs[k] = _parse_num(v)
            elif k in ("floor", "ceil"):
                kwargs[k] = float(v)
            else:
                raise ValueError(f"Unknown option '{k}' in '{part}'")

        schedules.append(ParamSchedule(
            name=name.strip(),
            start=start,
            end=end,
            total_steps=total_steps,
            mode=mode.strip(),
            **kwargs
        ))
    return schedules

