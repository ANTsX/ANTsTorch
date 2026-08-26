import pytest

from antstorch.bspline_flows import ImageDomain, mesh_size_for_spline_distance


def test_mesh_size_for_spline_distance_matches_ants_formula_2d():
    # ImageDomain(size=(256, 256, 160)) matches the exact user-reported
    # scale from the § 22 CUDA OOM report; physical extent is (255, 255,
    # 159) at 1mm isotropic spacing. ANTs' own un-padded
    # ceil(extent / knot_spacing) formula (see
    # itk::ants::RegistrationHelper::CalculateMeshSizeForSpecifiedKnotSpacing,
    # Examples/itkantsRegistrationHelper.cxx) at a spline_distance of 26mm
    # (the § 17 benchmark value, now correctly reinterpreted as a physical
    # spline distance rather than a literal mesh size) gives mesh (10, 10, 7).
    domain = ImageDomain(size=(256, 256, 160), spacing=(1.0, 1.0, 1.0))
    assert domain.physical_extent == (255.0, 255.0, 159.0)
    assert mesh_size_for_spline_distance(domain, 26.0) == (10, 10, 7)


def test_mesh_size_for_spline_distance_scalar_broadcasts_like_a_matching_vector():
    domain = ImageDomain(size=(256, 256, 160), spacing=(1.0, 1.0, 1.0))
    assert mesh_size_for_spline_distance(domain, 26.0) == mesh_size_for_spline_distance(
        domain, (26.0, 26.0, 26.0)
    )


def test_mesh_size_for_spline_distance_per_axis_vector_can_be_anisotropic():
    domain = ImageDomain(size=(28, 30), spacing=(1.0, 1.0))
    assert domain.physical_extent == (27.0, 29.0)
    assert mesh_size_for_spline_distance(domain, (9.0, 15.0)) == (3, 2)


def test_mesh_size_for_spline_distance_respects_non_unit_spacing():
    # A coarser physical spacing shrinks the physical extent, and hence the
    # resolved mesh, for the same voxel count and knot spacing.
    domain = ImageDomain(size=(101, 101), spacing=(2.0, 2.0))
    assert domain.physical_extent == (200.0, 200.0)
    assert mesh_size_for_spline_distance(domain, 40.0) == (5, 5)


def test_mesh_size_for_spline_distance_never_returns_less_than_one_span():
    # A knot spacing far larger than the domain extent still yields a valid
    # (minimum) 1-span mesh rather than 0.
    domain = ImageDomain(size=(4, 4))
    assert mesh_size_for_spline_distance(domain, 1000.0) == (1, 1)


@pytest.mark.parametrize(
    "spline_distance,match",
    [
        (0.0, "positive"),
        (-1.0, "positive"),
        (float("inf"), "finite"),
        (float("nan"), "finite"),
        ((5.0,), "must have"),
        ((5.0, 5.0, 5.0), "must have"),
    ],
)
def test_mesh_size_for_spline_distance_rejects_invalid_input(spline_distance, match):
    domain = ImageDomain(size=(8, 7))
    with pytest.raises(ValueError, match=match):
        mesh_size_for_spline_distance(domain, spline_distance)


def test_n4_scalar_spline_param_delegates_to_shared_spline_distance_utility():
    # antstorch.bspline_flows.n4_bias_field_correction's pre-existing scalar
    # spline_param dispatch is now implemented via
    # mesh_size_for_spline_distance -- this pins the exact relationship
    # (mesh_size + spline_order=3 control points) rather than re-deriving
    # the ANTs formula a second time inside N4's own module.
    from antstorch.bspline_flows.n4_bias_field_correction import _initial_lattice_size

    domain = ImageDomain(size=(256, 256, 160), spacing=(1.0, 1.0, 1.0))
    mesh = mesh_size_for_spline_distance(domain, 26.0)
    assert _initial_lattice_size(domain, 26.0) == tuple(value + 3 for value in mesh)
