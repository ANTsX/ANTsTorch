"""Physical-domain metadata for ITK-compatible B-spline synthesis.

Metadata tuples use ITK order ``(x, y[, z])``.  PyTorch image tensors use
``(N, C, Y, X)`` or ``(N, C, Z, Y, X)``; consequently ``torch_size`` is the
reverse of ``size``.  Vector channels remain physical x, y, z components and
are not reversed.
"""

from dataclasses import dataclass
from math import isfinite
from typing import Optional, Sequence, Tuple


def _tuple(values: Sequence, name: str, dimension: int, cast):
    result = tuple(cast(value) for value in values)
    if len(result) != dimension:
        raise ValueError(f"{name} must have {dimension} values, got {len(result)}")
    return result


@dataclass(frozen=True)
class BSplineDomain:
    """Regular dense image domain, with all metadata expressed in ITK order."""

    size: Tuple[int, ...]
    spacing: Optional[Tuple[float, ...]] = None
    origin: Optional[Tuple[float, ...]] = None
    direction: Optional[Tuple[Tuple[float, ...], ...]] = None

    def __post_init__(self):
        dimension = len(self.size)
        if dimension not in (2, 3):
            raise ValueError("BSplineDomain supports exactly 2-D or 3-D")
        size = _tuple(self.size, "size", dimension, int)
        if any(value < 2 for value in size):
            raise ValueError("each size must be at least 2 for the ITK image domain")
        spacing = _tuple(self.spacing or (1.0,) * dimension, "spacing", dimension, float)
        origin = _tuple(self.origin or (0.0,) * dimension, "origin", dimension, float)
        direction = self.direction or tuple(
            tuple(float(i == j) for j in range(dimension)) for i in range(dimension)
        )
        direction = tuple(_tuple(row, "direction row", dimension, float) for row in direction)
        if len(direction) != dimension:
            raise ValueError(f"direction must have {dimension} rows")
        if any(value <= 0 or not isfinite(value) for value in spacing):
            raise ValueError("spacing values must be finite and positive")
        if any(not isfinite(value) for value in origin for _ in (0,)) or any(
            not isfinite(value) for row in direction for value in row
        ):
            raise ValueError("origin and direction values must be finite")
        object.__setattr__(self, "size", size)
        object.__setattr__(self, "spacing", spacing)
        object.__setattr__(self, "origin", origin)
        object.__setattr__(self, "direction", direction)

    @property
    def dimension(self) -> int:
        return len(self.size)

    @property
    def torch_size(self) -> Tuple[int, ...]:
        return tuple(reversed(self.size))

    @property
    def physical_extent(self) -> Tuple[float, ...]:
        return tuple(s * (n - 1) for s, n in zip(self.spacing, self.size))

