"""Regression cases for the scientific evaluation's unit of analysis."""

import itertools
import statistics

import pytest

from code.shared_code.selection_metrics import (
    fixed_count_mixture,
    fixed_pool_policy,
    oracle_pass_at_k,
    paired_theorem_interval,
    selection_summary,
    uniform_tie_accuracy,
)


def test_pass_at_k_agrees_with_enumerated_candidate_subsets():
    labels = [True, False, False, True, False]
    for count in range(1, len(labels) + 1):
        expected = statistics.fmean(any(pool) for pool in itertools.combinations(labels, count))
        assert oracle_pass_at_k(labels, count) == pytest.approx(expected)


def test_bootstrap_weights_theorems_equally_not_repeated_attempts():
    interval = paired_theorem_interval({"a": [1.0] * 100, "b": [-1.0]}, bootstrap_samples=500)
    assert interval.mean == 0.0
    assert interval.theorems == 2
    assert interval.low == -1.0
    assert interval.high == 1.0
    identical = paired_theorem_interval({"x": [0.2, 0.2], "y": [0.2]}, bootstrap_samples=20)
    assert identical.low == pytest.approx(0.2)
    assert identical.high == pytest.approx(0.2)


def test_ties_have_no_candidate_position_advantage():
    assert uniform_tie_accuracy([True, False, False], [3.0, 3.0, 1.0]) == 0.5
    assert uniform_tie_accuracy([False, False, True], [1.0, 3.0, 3.0]) == 0.5


def test_no_positive_pools_remain_in_accuracy_denominator():
    result = selection_summary([[True, False], [False, False]], [1.0, 0.0])
    assert result["selected_accuracy"] == 0.5
    assert result["conditional_selection_accuracy"] == 1.0
    assert result["oracle_minus_random_gap_recovered"] == 1.0
    assert selection_summary([[False, False]], [0.0])["conditional_selection_accuracy"] is None
    with pytest.raises(ValueError, match="impossible"):
        selection_summary([[False, False]], [1.0])


def test_compute_mixture_can_skip_dominated_intermediate_count():
    result = fixed_count_mixture([(1.0, 0.5), (2.0, 0.51), (4.0, 0.8)], 2.0)
    assert result is not None
    assert result["left_index"] == 0
    assert result["right_index"] == 2
    assert result["value"] == pytest.approx(0.6)
    assert fixed_count_mixture([(1.0, 0.5), (4.0, 0.8)], 8.0) is None


@pytest.mark.parametrize("threshold", [0.0, 0.4, 0.8, 1.1])
def test_threshold_expectation_matches_every_ordering(threshold):
    scores = [0.2, 0.7, 0.7, 0.9]
    selected = [0.0] * len(scores)
    included = [0.0] * len(scores)
    attempts = []
    orderings = list(itertools.permutations(range(len(scores))))
    for order in orderings:
        seen = []
        for index in order:
            seen.append(index)
            if scores[index] >= threshold:
                break
        attempts.append(len(seen))
        for index in seen:
            included[index] += 1.0 / len(orderings)
        best = max(scores[index] for index in seen)
        tied = [index for index in seen if scores[index] == best]
        for index in tied:
            selected[index] += 1.0 / (len(orderings) * len(tied))
    result = fixed_pool_policy(scores, threshold=threshold)
    assert result["selected_probabilities"] == pytest.approx(selected)
    assert result["included_probabilities"] == pytest.approx(included)
    assert result["expected_attempts"] == pytest.approx(statistics.fmean(attempts))


def test_fixed_count_policy_keeps_order_and_tie_symmetry():
    result = fixed_pool_policy([0.5, 0.5, 0.5], count=2)
    assert result["selected_probabilities"] == pytest.approx([1 / 3] * 3)
    assert result["included_probabilities"] == pytest.approx([2 / 3] * 3)
    assert result["expected_attempts"] == 2
