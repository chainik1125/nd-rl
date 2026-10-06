"""Selection and uncertainty calculations with the theorem as the sampling unit."""

import math
import random
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from itertools import combinations


@dataclass(frozen=True)
class PairedInterval:
    """Equal-theorem-weight mean and percentile bootstrap interval."""

    mean: float
    low: float
    high: float
    theorems: int
    bootstrap_samples: int
    confidence: float
    seed: int

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def _quantile(sorted_values: Sequence[float], probability: float) -> float:
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    fraction = position - lower
    return sorted_values[lower] * (1 - fraction) + sorted_values[upper] * fraction


def paired_theorem_interval(
    differences: Mapping[str, Sequence[float]],
    *,
    bootstrap_samples: int = 5000,
    confidence: float = 0.95,
    seed: int = 1729,
) -> PairedInterval:
    """Average repetitions within each theorem, then bootstrap whole theorems.

    A repetition must already be a *paired* method-minus-baseline difference on
    the same candidate pool. This conditions on the observed training seeds;
    it does not estimate population uncertainty over possible training runs.
    """
    if not differences or bootstrap_samples < 1 or not 0 < confidence < 1:
        raise ValueError("Need theorems, positive bootstrap count, and confidence in (0, 1)")
    values = []
    for theorem in sorted(differences):
        repeats = differences[theorem]
        if not repeats or any(not math.isfinite(value) for value in repeats):
            raise ValueError("Every theorem needs finite paired observations")
        values.append(statistics.fmean(repeats))
    rng = random.Random(seed)
    size = len(values)
    draws = sorted(
        statistics.fmean(values[rng.randrange(size)] for _ in range(size))
        for _ in range(bootstrap_samples)
    )
    alpha = (1 - confidence) / 2
    return PairedInterval(
        statistics.fmean(values),
        _quantile(draws, alpha),
        _quantile(draws, 1 - alpha),
        size,
        bootstrap_samples,
        confidence,
        seed,
    )


def oracle_pass_at_k(valid: Sequence[bool], k: int) -> float:
    """Exact success probability of a uniform k-subset of a saved finite pool.

    This is the usual unbiased pass@k estimator for iid candidate generation.
    It is an oracle measurement, never model-selected success.
    """
    count = len(valid)
    if not 1 <= k <= count:
        raise ValueError("k must be between one and the candidate count")
    failures = count - sum(valid)
    if failures < k:
        return 1.0
    return 1.0 - math.comb(failures, k) / math.comb(count, k)


def uniform_tie_accuracy(valid: Sequence[bool], scores: Sequence[float]) -> float:
    """Expected correctness under uniform tie breaking, independent of position."""
    if not valid or len(valid) != len(scores):
        raise ValueError("One score is required per candidate")
    if any(not math.isfinite(score) for score in scores):
        raise ValueError("Scores must be finite")
    maximum = max(scores)
    tied = [float(label) for label, score in zip(valid, scores, strict=True) if score == maximum]
    return statistics.fmean(tied)


def fixed_pool_policy(
    scores: Sequence[float], *, count: int | None = None, threshold: float | None = None
) -> dict[str, list[float] | float]:
    """Integrate a label-blind policy over uniform orderings of a finite pool.

    A fixed count draws a uniform subset and chooses its highest score, uniformly
    breaking ties. A threshold stops at the first qualifying score; if no score
    qualifies, it chooses the best of the complete pool. Returned inclusion
    probabilities support additive token accounting, not exact batched GPU cost.
    These policies never inspect correctness labels.
    """
    n = len(scores)
    if not n or any(not math.isfinite(x) for x in scores):
        raise ValueError("Need a nonempty finite score vector")
    if (count is None) == (threshold is None):
        raise ValueError("Specify exactly one of fixed count and threshold")
    selected = [0.0] * n
    if count is not None:
        if not 1 <= count <= n:
            raise ValueError("Count must be within the saved pool")
        subsets = math.comb(n, count)
        for subset in combinations(range(n), count):
            best = max(scores[i] for i in subset)
            tied = [i for i in subset if scores[i] == best]
            for index in tied:
                selected[index] += 1.0 / (subsets * len(tied))
        included = [count / n] * n
        attempts = float(count)
    else:
        if threshold is None or not math.isfinite(threshold):
            raise ValueError("Threshold must be finite")
        qualifying = [i for i, score in enumerate(scores) if score >= threshold]
        m = len(qualifying)
        if m:
            for index in qualifying:
                selected[index] = 1.0 / m
            # A nonqualifier precedes the first of m qualifiers with probability
            # 1/(m+1); exactly one uniformly chosen qualifier is generated.
            included = [1.0 / m if score >= threshold else 1.0 / (m + 1) for score in scores]
            attempts = (n + 1.0) / (m + 1)
        else:
            best = max(scores)
            tied = [i for i, score in enumerate(scores) if score == best]
            for index in tied:
                selected[index] = 1.0 / len(tied)
            included = [1.0] * n
            attempts = float(n)
    return {
        "selected_probabilities": selected,
        "included_probabilities": included,
        "expected_attempts": attempts,
    }


def selection_summary(
    labels: Sequence[Sequence[bool]], selected_accuracy: Sequence[float]
) -> dict[str, float | int | None]:
    """Summarize pools without dropping pools that contain no valid proof."""
    if not labels or len(labels) != len(selected_accuracy) or any(not row for row in labels):
        raise ValueError("Need nonempty pools and exactly one accuracy per pool")
    if any(not math.isfinite(score) or not 0 <= score <= 1 for score in selected_accuracy):
        raise ValueError("Accuracy values must be probabilities")
    available = [any(row) for row in labels]
    if any(
        score > 0 and not has_valid
        for score, has_valid in zip(selected_accuracy, available, strict=True)
    ):
        raise ValueError("Selected success is impossible for a pool without a valid proof")
    selected = statistics.fmean(selected_accuracy)
    oracle = statistics.fmean(available)
    random_accuracy = statistics.fmean(statistics.fmean(row) for row in labels)
    headroom = oracle - random_accuracy
    conditional = [score for score, ok in zip(selected_accuracy, available, strict=True) if ok]
    return {
        "pools": len(labels),
        "pools_with_valid_candidate": sum(available),
        "selected_accuracy": selected,
        "oracle_pool_accuracy": oracle,
        "random_selection_accuracy": random_accuracy,
        "conditional_selection_accuracy": statistics.fmean(conditional) if conditional else None,
        "oracle_minus_random_gap_recovered": (
            (selected - random_accuracy) / headroom if headroom > 0 else None
        ),
    }


def fixed_count_mixture(
    points: Sequence[tuple[float, float]], target_compute: float
) -> dict[str, float | int] | None:
    """Best mixture of fixed policies at a target mean compute (upper envelope).

    Each point is (mean compute, accuracy or reward). Select the envelope using
    development data for deployable baselines. An envelope recomputed on test
    outcomes is only an optimistic descriptive comparison and must be labelled.
    """
    if not math.isfinite(target_compute):
        raise ValueError("Target compute must be finite")
    if any(not math.isfinite(cost) or not math.isfinite(value) for cost, value in points):
        raise ValueError("Points must be finite")
    best: dict[str, float | int] | None = None
    for left, (cost_left, value_left) in enumerate(points):
        for right, (cost_right, value_right) in enumerate(points):
            if cost_left > target_compute or cost_right < target_compute:
                continue
            if cost_left == cost_right:
                if cost_left != target_compute:
                    continue
                weight_right = 0.0
            else:
                weight_right = (target_compute - cost_left) / (cost_right - cost_left)
            value = (1 - weight_right) * value_left + weight_right * value_right
            if best is None or value > best["value"]:
                best = {
                    "left_index": left,
                    "right_index": right,
                    "right_weight": weight_right,
                    "compute": target_compute,
                    "value": value,
                }
    return best
