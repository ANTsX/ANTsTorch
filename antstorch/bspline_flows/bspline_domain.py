"""Physical-domain metadata for ITK-compatible B-spline synthesis.

Metadata tuples use ITK order ``(x, y[, z])``.  PyTorch image tensors use
``(N, C, Y, X)`` or ``(N, C, Z, Y, X)``; consequently ``torch_size`` is the
reverse of ``size``.  Vector channels remain physical x, y, z components and
are not reversed.
"""

from dataclasses import dataclass
from math import ceil, isfinite
from typing import Optional, Sequence, Tuple, Union


def _tuple(values: Sequence, name: str, dimension: int, cast):
    result = tuple(cast(value) for value in values)
    if len(result) != dimension:
        raise ValueError(f"{name} must have {dimension} values, got {len(result)}")
    return result


@dataclass(frozen=True)
class ImageDomain:
    """Regular dense image domain, with all metadata expressed in ITK order."""

    size: Tuple[int, ...]
    spacing: Optional[Tuple[float, ...]] = None
    origin: Optional[Tuple[float, ...]] = None
    direction: Optional[Tuple[Tuple[float, ...], ...]] = None

    def __post_init__(self):
        dimension = len(self.size)
        if dimension not in (2, 3):
            raise ValueError("ImageDomain supports exactly 2-D or 3-D")
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


def mesh_size_for_spline_distance(
    domain: "ImageDomain", spline_distance: Union[float, Sequence[float]]
) -> Tuple[int, ...]:
    """ITK/ANTs' "spline distance" (knot spacing, physical units) -> a
    per-axis cubic B-spline mesh size (spans -- the same unit every
    ``mesh_size``/``*_mesh_size_at_base_level`` parameter in this package
    already uses; add ``spline_order`` for control points).

    Implements ``ceil(domain.physical_extent[d] / spline_distance[d])``
    against ``domain``'s FULL extent -- exactly the un-padded formula real
    ANTs uses in two places, confirmed against both source files directly:
    ``itk::ants::RegistrationHelper::CalculateMeshSizeForSpecifiedKnotSpacing``
    (``Examples/itkantsRegistrationHelper.cxx`` -- used by ``antsRegistration``
    whenever a *single* scalar is given for a mesh-size-at-base-level
    argument such as ``BSplineSyN``'s ``updateFieldMeshSizeAtBaseLevel``; its
    own comment: "the mesh size is simply an approximation" -- deliberately
    no image padding), and ``N4BiasFieldCorrection.cxx``'s ``numberOfSpans``
    term (before N4's own *additional* image-padding step, which is not
    replicated here either -- see ``n4_bias_field_correction``'s
    ``spline_param`` docstring for that same, already-existing, deliberate
    simplification). This is the same formula
    ``n4_bias_field_correction._initial_lattice_size`` already applies for a
    scalar ``spline_param``; this function factors that piece out so
    registration call sites (``antstorch.bspline_flows.bspline_svf_registration``,
    ``antstorch.syn.syn_registration``'s ``BSplineSyN`` regularizer) can use
    the identical, once-verified conversion instead of picking their own
    numbers by hand -- which is what motivated this in the first place: a
    benchmark comparing ``bspline_svf`` and ``bspline_syn`` with unrelated
    hand-picked mesh-size defaults wasn't a fair comparison of B-spline
    density, only of two arbitrary integers.

    ``spline_distance`` is always computed against ``domain`` exactly as
    given -- the caller is responsible for passing the FULL native-resolution
    domain (matching ANTs' own ``inputImage``/``fixedImage`` at the CLI-
    parsing stage, before any multi-resolution shrinking), not a
    already-shrunk per-pyramid-level domain; a per-level "at base level"
    doubling (if the caller's algorithm does one, as every B-spline
    regularizer in this package does) still applies afterward, unchanged, to
    the mesh size this function returns.

    ``spline_distance`` may be a single positive number (the same physical
    distance applied to every axis) or a per-axis sequence (ITK ``x, y[, z]``
    order, matching ``domain.size``'s own axis order).
    """
    if isinstance(spline_distance, (int, float)) and not isinstance(spline_distance, bool):
        distances = (float(spline_distance),) * domain.dimension
    else:
        distances = tuple(float(value) for value in spline_distance)
        if len(distances) != domain.dimension:
            raise ValueError(f"spline_distance must have {domain.dimension} values, got {len(distances)}")
    if any(not isfinite(value) or value <= 0 for value in distances):
        raise ValueError("spline_distance must be finite and positive")
    return tuple(max(1, ceil(extent / distance)) for extent, distance in zip(domain.physical_extent, distances))

