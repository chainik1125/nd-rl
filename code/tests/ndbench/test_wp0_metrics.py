"""`metrics.py`: `pass_at_k` matches `vendor/common.py`'s reference implementation on a grid (WP0
acceptance test), plus sanity/unit checks for the rest of the module."""

from __future__ import annotations

import math
import warnings
from types import TracebackType

import pytest

from code.ndbench.core.types import ItemResult
from code.ndbench.metrics import (
    clopper_pearson,
    compositionality_ratio,
    edl,
    hazard,
    horizon,
    logit_slope,
    pass_at_k,
    pass_curve,
    wilson,
)
from code.ndbench.vendor import ensure_takehome_on_path


def _item(n: int, c: int, pid: str = "p") -> ItemResult:
    return ItemResult(
        run_id="r",
        pid=pid,
        chunk=0,
        n=n,
        c=c,
        first_success=None,
        ok_written=[],
        ok_pruned=[],
        ok_tokens=[],
        ok_starts=[],
        fail_kinds={},
        fail_lines=[],
        long_enough=0,
        tokens_mean=0.0,
        hit_max=0,
    )


def test_pass_at_k_matches_vendored_common_on_a_grid() -> None:
    ensure_takehome_on_path()
    import common  # vendored lengthexp/common.py

    for n in (1, 2, 5, 10, 32, 100):
        for c in range(0, n + 1, max(1, n // 5)):
            for k in (1, 2, 5, 8, 10, 32, 100):
                ours = pass_at_k(n, c, k)
                theirs = common.pass_at_k(n, c, k)
                if math.isnan(theirs):
                    assert math.isnan(ours)
                else:
                    assert ours == pytest.approx(theirs)


def test_pass_at_k_edge_cases() -> None:
    assert pass_at_k(5, 5, 5) == 1.0
    assert pass_at_k(5, 0, 5) == 0.0
    assert math.isnan(pass_at_k(5, 3, 10))
    assert pass_at_k(10, 10, 1) == 1.0


def test_clopper_pearson_known_values() -> None:
    lo, hi = clopper_pearson(0, 10)
    assert lo == 0.0
    assert hi == pytest.approx(0.3084971, abs=1e-4)
    lo, hi = clopper_pearson(10, 10)
    assert hi == 1.0
    assert lo == pytest.approx(0.6915029, abs=1e-4)
    lo, hi = clopper_pearson(5, 10)
    assert lo == pytest.approx(0.1870860, abs=1e-4)
    assert hi == pytest.approx(0.8129140, abs=1e-4)


def test_wilson_brackets_the_point_estimate() -> None:
    lo, hi = wilson(5, 10)
    assert lo < 0.5 < hi
    lo0, hi0 = wilson(0, 0)
    assert (lo0, hi0) == (0.0, 1.0)


def test_pass_curve_shape_and_bounds() -> None:
    rows = [_item(32, 32, "a"), _item(32, 0, "b")]
    curve = pass_curve(rows, ks=[1, 8, 32], B=200, seed=0)
    assert [k for k, _, _, _ in curve] == [1, 8, 32]
    for _, mean, lo, hi in curve:
        assert 0.0 <= lo <= mean <= hi <= 1.0
        assert mean == pytest.approx(0.5, abs=1e-9)  # one item always passes, one never does


def test_pass_curve_empty() -> None:
    assert pass_curve([], ks=[1, 2], B=10)[0][1:] == pytest.approx((float("nan"),) * 3, nan_ok=True)


def test_horizon_step_function() -> None:
    by_n = {
        1: [_item(8, 8), _item(8, 8)],
        2: [_item(8, 8), _item(8, 8)],
        3: [_item(8, 0), _item(8, 0)],
        4: [_item(8, 0), _item(8, 0)],
    }
    result = horizon(by_n, k=8, tau=0.5, B=200, seed=0)
    assert result["n_star"] == 2
    lo, hi = result["ci"]
    assert lo is not None and hi is not None
    assert lo <= result["n_star"] <= hi
    assert [n for n, _ in result["curve"]] == [1, 2, 3, 4]


def test_compositionality_ratio_perfect_independence() -> None:
    # If joined pass@1 == product of part pass@1s exactly, R should be 1.
    groups = [
        (_item(100, 64, "joined1"), [_item(100, 80, "part1a"), _item(100, 80, "part1b")]),
        (_item(100, 49, "joined2"), [_item(100, 70, "part2a"), _item(100, 70, "part2b")]),
    ]
    result = compositionality_ratio(groups, mastery=0.5, B=200, seed=0)
    assert result["n_used"] == 2
    assert result["R"] == pytest.approx(1.0, abs=0.05)
    # product(parts) <= min(parts) for values in [0, 1], so R_min (which divides by the min) is
    # always <= R (which divides by the product) whenever there is more than one part.
    assert result["R_min"] <= result["R"] + 1e-9


def test_compositionality_ratio_excludes_unmastered() -> None:
    groups = [(_item(100, 60, "joined"), [_item(100, 10, "part_unmastered")])]
    result = compositionality_ratio(groups, mastery=0.5, B=10, seed=0)
    assert result["n_used"] == 0
    assert math.isnan(result["R"])


def test_hazard_basic() -> None:
    # Two attempts fail at line 2, one survives to line 3 (its written length) without a bad line.
    rows = [(2, True), (2, True), (3, False)]
    table = hazard(rows)
    by_line = {line: (at_risk, failed, haz) for line, at_risk, failed, haz in table}
    assert by_line[1] == (3, 0, 0.0)
    assert by_line[2] == (3, 2, pytest.approx(2 / 3))
    assert by_line[3] == (1, 0, 0.0)


def test_hazard_empty() -> None:
    assert hazard([]) == []


def test_logit_slope_recovers_sign_and_is_stable() -> None:
    xs = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    ns = [100, 100, 100, 100, 100, 100]
    # Sharp threshold: pass rate drops from ~1 to ~0 as x increases -> negative slope.
    cs = [98, 95, 80, 20, 5, 2]
    slope, se = logit_slope(xs, ns, cs)
    assert slope < 0
    assert se > 0
    # Flipping the direction of x should flip the sign of the slope.
    slope2, _ = logit_slope([-x for x in xs], ns, cs)
    assert slope2 == pytest.approx(-slope)


def test_edl_zero_when_online_equals_final() -> None:
    assert edl([2.0] * 10, 2.0) == pytest.approx(0.0)


def test_logit_slope_stays_finite_under_quasi_complete_separation() -> None:
    """WP8 integration fix: a horizon-style curve (pass rate ~1 then ~0, a near-perfect step as x
    grows) is quasi-complete separation for logistic regression -- the unpenalised MLE runs to
    +-infinity. Confirmed on the real store's `families` page: a `reorder_or`/armC fit landed at
    slope=-869.9, se=3941.6 before the `ridge` penalty was added. A real step function (no overlap
    at all between the all-pass and all-fail regimes) must still return finite, bounded values,
    with the correct sign, rather than diverging or overflowing."""
    xs = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    ns = [32] * 8
    cs = [32, 32, 32, 32, 0, 0, 0, 0]  # a perfect step: pass@1=1 for x<4, 0 for x>=4.
    with _no_warnings():
        slope, se = logit_slope(xs, ns, cs)
    assert math.isfinite(slope)
    assert math.isfinite(se)
    assert slope < 0
    assert abs(slope) < 50
    assert se < 50


class _no_warnings:
    """Fails the test if `numpy` raises a `RuntimeWarning` (e.g. "overflow encountered in exp")
    inside the `with` block -- the exact symptom the pre-ridge `logit_slope` produced on a
    perfectly-separated fit."""

    def __enter__(self) -> None:
        self._cm = warnings.catch_warnings()
        self._cm.__enter__()
        warnings.simplefilter("error", RuntimeWarning)

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self._cm.__exit__(exc_type, exc_val, exc_tb)


def test_edl_matches_formula() -> None:
    online = [3.0, 2.5, 2.0, 1.5]
    final = 1.0
    assert edl(online, final) == pytest.approx(sum(online) - len(online) * final)


def test_hazard_left_truncation_excludes_forced_lines() -> None:
    # Audit 2026-10-01: a prefix attempt whose first 5 lines were forced (entry line 6) must not
    # count as "at risk" at lines 1-5. Item A (j=0) fails at line 6; item B (j=5) fails at line 6.
    rows = [(6, True, 1), (6, True, 6)]
    by_line = {line: (at_risk, failed) for line, at_risk, failed, _h in hazard(rows)}
    assert by_line[3] == (1, 0)  # only the j=0 attempt is at risk before line 6
    assert by_line[6] == (2, 2)
    # Two-tuples keep their old meaning (entry line 1).
    assert hazard([(2, True)]) == hazard([(2, True, 1)])


def test_horizon_no_ci_when_point_estimate_is_none() -> None:
    # Audit 2026-10-01: smallest n fails (pass@8 < 0.5) -> n* None and no CI, even though some
    # bootstrap replicates of a borderline cell can clear the threshold.
    by_n = {1: [_item(32, 1, "a"), _item(32, 0, "b"), _item(32, 6, "c")], 2: [_item(32, 0, "d")]}
    res = horizon(by_n, k=8, tau=0.5, B=200, seed=0)
    assert res["n_star"] is None
    assert res["ci"] == (None, None)
