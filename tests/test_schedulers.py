# tests/test_schedulers.py
import math
import types
import pytest
import antstorch

# ------------------------------
# Helpers
# ------------------------------
def sample_vals(sched: antstorch.ParamSchedule, steps):
    return [sched.value_at(s) for s in steps]


# ------------------------------
# Basic interpolation & endpoints
# ------------------------------
@pytest.mark.parametrize("mode", ["linear", "lin", "cos", "cosine", "exp", "exponential"])
def test_endpoints(mode):
    s = antstorch.ParamSchedule(name="p", start=0.05, end=0.0, total_steps=100, mode=mode)
    assert s.value_at(0) == pytest.approx(0.05)
    # at or beyond total_steps → end
    assert s.value_at(100) == pytest.approx(0.0)
    assert s.value_at(1000) == pytest.approx(0.0)


@pytest.mark.parametrize("mode", ["linear", "cos", "exp"])
def test_monotonic_decreasing(mode):
    s = antstorch.ParamSchedule(name="p", start=1.0, end=0.0, total_steps=50, mode=mode)
    vals = sample_vals(s, range(0, 51))
    # non-increasing
    assert all(vals[i] >= vals[i + 1] for i in range(len(vals) - 1))


@pytest.mark.parametrize("mode", ["linear", "cos", "exp"])
def test_monotonic_increasing(mode):
    s = antstorch.ParamSchedule(name="p", start=0.0, end=1.0, total_steps=50, mode=mode)
    vals = sample_vals(s, range(0, 51))
    # non-decreasing
    assert all(vals[i] <= vals[i + 1] for i in range(len(vals) - 1))


def test_cosine_midpoint_equals_linear_midpoint():
    # For cosine half-wave, midpoint equals linear midpoint
    s_lin = antstorch.ParamSchedule("p", start=10.0, end=0.0, total_steps=100, mode="linear")
    s_cos = antstorch.ParamSchedule("p", start=10.0, end=0.0, total_steps=100, mode="cos")
    assert s_lin.value_at(50) == pytest.approx(s_cos.value_at(50))


def test_exponential_shape_and_fallback():
    # Proper exponential: start=1, end=0.01 → midpoint should be sqrt(0.01)=0.1
    s_ok = antstorch.ParamSchedule("p", start=1.0, end=0.01, total_steps=100, mode="exp")
    assert s_ok.value_at(50) == pytest.approx(0.1, rel=1e-6)

    # Fallback when start==0 → should behave like linear
    s_fb = antstorch.ParamSchedule("p", start=0.0, end=1.0, total_steps=100, mode="exp")
    assert s_fb.value_at(50) == pytest.approx(0.5)

    # Fallback when signs differ → linear
    s_fb2 = antstorch.ParamSchedule("p", start=1.0, end=-1.0, total_steps=100, mode="exp")
    assert s_fb2.value_at(50) == pytest.approx(0.0)


# ------------------------------
# Delay, floor/ceil, steps_per_update
# ------------------------------
def test_delay_and_floor_ceil():
    s = antstorch.ParamSchedule(
        name="p",
        start=1.0,
        end=0.0,
        total_steps=10,
        mode="linear",
        delay=5,
        floor=0.2,
        ceil=1.0,
    )
    # hold start during delay
    for t in range(0, 5):
        assert s.value_at(t) == pytest.approx(1.0)
    # ramp starts at t=5 (now strictly less than start)
    assert s.value_at(5) < 1.0
    # after finish, clamped by floor
    assert s.value_at(999) == pytest.approx(0.2)


def test_steps_per_update_is_deterministic():
    s = antstorch.ParamSchedule("p", start=1.0, end=0.0, total_steps=100, mode="linear", steps_per_update=10)
    # Still returns the deterministic value at any step
    assert s.value_at(7) == pytest.approx(1.0 - 7 / 100.0)
    assert s.value_at(20) == pytest.approx(0.8)


# ------------------------------
# MultiParamScheduler behavior
# ------------------------------
def test_multi_param_scheduler_outputs_all_keys():
    scheds = [
        antstorch.ParamSchedule("a", 1.0, 0.0, 10, "linear"),
        antstorch.ParamSchedule("b", 0.05, 0.0, 10, "cos"),
    ]
    mp = antstorch.MultiParamScheduler(scheds)
    out = mp.step(5)
    assert set(out.keys()) == {"a", "b"}
    # Spot-check values
    assert out["a"] == pytest.approx(0.5)
    # cosine at halfway equals linear midpoint
    assert out["b"] == pytest.approx(0.025)


# ------------------------------
# Parser tests
# ------------------------------
def test_parse_schedules_basic_and_units():
    spec = "noise_std:cos:0.05->0.00@150k,sd_deformation:linear:10.0->0.0@2m+delay=5k+floor=0.1"
    schedules = antstorch.parse_schedules(spec)
    assert [s.name for s in schedules] == ["noise_std", "sd_deformation"]

    s0, s1 = schedules
    assert s0.mode.lower() in ("cos", "cosine")
    assert s0.start == pytest.approx(0.05)
    assert s0.end == pytest.approx(0.0)
    assert s0.total_steps == 150_000

    assert s1.mode.lower() in ("lin", "linear")
    assert s1.start == pytest.approx(10.0)
    assert s1.end == pytest.approx(0.0)
    assert s1.total_steps == 2_000_000
    assert s1.delay == 5_000
    assert s1.floor == pytest.approx(0.1)


def test_parse_empty_spec():
    assert antstorch.parse_schedules("") == []


@pytest.mark.parametrize(
    "bad_spec",
    [
        "nameonly",  # missing pieces
        "a:cos:0.1->0.0",  # missing @steps
        "a:cos:bad->0.0@100",  # bad float
        "a:cos:0.1->0.0@x",  # bad int with suffix
        "a:cos:0.1->0.0@100+unknown=1",  # unknown option
        "a|cos|0.1->0.0@100",  # wrong separators
    ],
)
def test_parse_errors(bad_spec):
    with pytest.raises(ValueError):
        antstorch.parse_schedules(bad_spec)


# ------------------------------
# Determinism across calls
# ------------------------------
def test_determinism_value_at():
    s = antstorch.ParamSchedule("p", start=0.123, end=0.0, total_steps=1234, mode="cos", delay=17)
    v1 = s.value_at(321)
    v2 = s.value_at(321)
    assert v1 == v2

    mp = antstorch.MultiParamScheduler([s, antstorch.ParamSchedule("q", 1.0, 0.5, 100, "linear")])
    o1 = mp.step(999)
    o2 = mp.step(999)
    assert o1 == o2

