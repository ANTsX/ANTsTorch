"""Bounded in-memory caches for static ANTsTorch templates, priors, and pretrained `.eval()` models."""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Callable

_MAX_MODEL_CACHE = 12
_MAX_DATA_CACHE = 16

_MODEL_CACHE: OrderedDict[tuple[Any, ...], Any] = OrderedDict()
_DATA_CACHE: OrderedDict[tuple[Any, ...], Any] = OrderedDict()


def get_cached_model(key: tuple[Any, ...], builder: Callable[[], Any]) -> Any:
    """Return a cached `.eval()` ``torch.nn.Module`` for ``key``, evicting LRU beyond ``_MAX_MODEL_CACHE``."""
    hit = _MODEL_CACHE.get(key)
    if hit is not None:
        _MODEL_CACHE.move_to_end(key)
        return hit
    model = builder()
    _MODEL_CACHE[key] = model
    if len(_MODEL_CACHE) > _MAX_MODEL_CACHE:
        _MODEL_CACHE.popitem(last=False)
    return model


def get_cached_data(key: tuple[Any, ...], builder: Callable[[], Any]) -> Any:
    """Return cached static template/prior data for ``key``, evicting LRU beyond ``_MAX_DATA_CACHE``."""
    hit = _DATA_CACHE.get(key)
    if hit is not None:
        _DATA_CACHE.move_to_end(key)
        return hit
    val = builder()
    _DATA_CACHE[key] = val
    if len(_DATA_CACHE) > _MAX_DATA_CACHE:
        _DATA_CACHE.popitem(last=False)
    return val


def clear_utility_caches() -> None:
    """Release all cached ANTsTorch models and static template/prior arrays."""
    _MODEL_CACHE.clear()
    _DATA_CACHE.clear()
