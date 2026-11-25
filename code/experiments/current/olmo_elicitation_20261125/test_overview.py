"""fig_overview.py: progress normalisation and the half-way interpolation (CPU, no data files)."""

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fig_overview import half_step, progress  # noqa: E402

STEPS = [0, 100, 500, 1100, 2300]


def test_progress_rising_and_falling_measures():
    assert progress([10.0, 20.0, 30.0]) == [0.0, 0.5, 1.0]
    # a falling measure (L0 nats/token): 0 at the first value, 1 at the last, positive in between
    assert progress([0.6, 0.55, 0.5]) == pytest.approx([0.0, 0.5, 1.0])
    # overshoot and undershoot are kept, not clipped
    assert progress([0.0, -2.0, 12.0, 10.0]) == pytest.approx([0.0, -0.2, 1.2, 1.0])


def test_progress_missing_and_degenerate():
    assert progress([1.0, None, 3.0]) == [0.0, None, 1.0]
    with pytest.raises(ValueError):
        progress([None, 1.0, 2.0])
    with pytest.raises(ValueError):
        progress([1.0, 2.0, None])
    with pytest.raises(ValueError):
        progress([2.0, 5.0, 2.0])


def test_half_step_linear_interpolation():
    h = half_step(STEPS, [0.0, 0.25, 1.0, 1.0, 1.0])
    # 0.25 at 100, 1.0 at 500 -> 0.5 at 100 + (0.25 / 0.75) * 400
    assert h["step"] == pytest.approx(100 + 400 / 3)
    assert h["bracket"] == (100.0, 500.0)
    assert h["monotone"]


def test_half_step_exact_hit_and_first_crossing():
    assert half_step(STEPS, [0.0, 0.5, 0.7, 0.9, 1.0])["step"] == pytest.approx(100.0)
    # non-monotone: the first crossing counts, and the flag says the curve falls somewhere
    h = half_step(STEPS, [0.0, 0.6, 0.4, 0.8, 1.0])
    assert h["step"] == pytest.approx(0 + 0.5 / 0.6 * 100)
    assert h["bracket"] == (0.0, 100.0)
    assert not h["monotone"]


def test_half_step_skips_missing_and_reports_never():
    # s0500 missing: interpolate between 100 and 1100
    h = half_step(STEPS, [0.0, 0.2, None, 0.8, 1.0])
    assert h["step"] == pytest.approx(100 + (0.3 / 0.6) * 1000)
    assert h["bracket"] == (100.0, 1100.0)
    h = half_step(STEPS, [0.0, 0.1, 0.2, 0.3, 0.4])
    assert h["step"] is None and h["bracket"] is None
    assert half_step(STEPS, [0.0, float("nan"), 1.0, 1.0, 1.0])["bracket"] == (0.0, 500.0)


def test_half_step_matches_edl_summary_convention():
    """Same reading as analyze_edl.half_step(kind="half_of_change") on a falling L0 curve:
    target v0 + (v_end - v0) / 2."""
    l0 = [
        0.6204811176151925,
        0.6116584533122009,
        0.5834298935368623,
        0.542964795496221,
        0.5020268364826419,
    ]
    target = l0[0] + 0.5 * (l0[-1] - l0[0])
    i = next(k for k, v in enumerate(l0) if v <= target)
    want = STEPS[i - 1] + (target - l0[i - 1]) / (l0[i] - l0[i - 1]) * (STEPS[i] - STEPS[i - 1])
    got = half_step(STEPS, progress(l0))["step"]
    assert math.isclose(got, want, rel_tol=1e-12)
    assert got == pytest.approx(828.8154616455779)
