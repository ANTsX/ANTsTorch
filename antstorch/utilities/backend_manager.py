"""Choice of implementation ("ants" C++ or "antstorch" PyTorch) per operation.

Resolution order for an operation (``n4``, ``denoise``, ``registration`` or
``kelly_kapowski``):

1. the ``backend=`` argument of the call (a string for all operations, or a
   dict ``{"n4": "antstorch", ...}``; operations missing from the dict fall
   through to the next level);
2. :func:`set_default_backend`;
3. the ``ANTSTORCH_BACKEND`` environment variable, either a single value
   (``antstorch``) or comma-separated pairs (``n4=antstorch,denoise=ants``);
4. ``"antstorch"`` (set ``ANTSTORCH_BACKEND=ants`` or call
   ``set_default_backend("ants")`` to get the ANTs C++ tools back).
"""

from __future__ import annotations

import os

VALID_BACKENDS = ("ants", "antstorch")
OPERATIONS = ("n4", "denoise", "registration", "kelly_kapowski")
_FALLBACK_BACKEND = "antstorch"

# Operation -> backend, set through set_default_backend().
_GLOBAL_BACKEND = {}


def _check_backend(value, where):
    if value not in VALID_BACKENDS:
        raise ValueError(f"{where}: backend must be one of {VALID_BACKENDS}, got {value!r}")
    return value


def _normalize(spec, where):
    """Turn ``None`` / str / dict into a partial ``{operation: backend}`` dict."""
    if spec is None:
        return {}
    if isinstance(spec, str):
        return {op: _check_backend(spec, where) for op in OPERATIONS}
    if isinstance(spec, dict):
        out = {}
        for op, value in spec.items():
            if op not in OPERATIONS:
                raise ValueError(f"{where}: unknown operation {op!r}; expected one of {OPERATIONS}")
            out[op] = _check_backend(value, where)
        return out
    raise TypeError(f"{where}: backend must be None, a str or a dict, got {type(spec).__name__}")


def _from_environment():
    raw = os.environ.get("ANTSTORCH_BACKEND", "").strip()
    if not raw:
        return {}
    if "=" not in raw:
        return _normalize(raw, "ANTSTORCH_BACKEND")
    pairs = {}
    for item in raw.split(","):
        if not item.strip():
            continue
        op, _, value = item.partition("=")
        pairs[op.strip()] = value.strip()
    return _normalize(pairs, "ANTSTORCH_BACKEND")


def set_default_backend(backend):
    """Set the session default, e.g. ``set_default_backend("antstorch")`` or
    ``set_default_backend({"n4": "antstorch", "registration": "ants"})``.
    ``None`` clears it."""
    global _GLOBAL_BACKEND
    _GLOBAL_BACKEND = _normalize(backend, "set_default_backend")


def get_default_backend():
    """Backend used for each operation when a call passes no ``backend=``."""
    env = _from_environment()
    return {op: _GLOBAL_BACKEND.get(op, env.get(op, _FALLBACK_BACKEND)) for op in OPERATIONS}


def resolve_backend(backend, operation):
    """Backend ("ants" or "antstorch") to use for ``operation`` in this call."""
    if operation not in OPERATIONS:
        raise ValueError(f"unknown operation {operation!r}; expected one of {OPERATIONS}")
    explicit = _normalize(backend, "backend")
    return explicit.get(operation) or get_default_backend()[operation]
