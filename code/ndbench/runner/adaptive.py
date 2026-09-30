"""Adaptive per-item sampling decisions, shared by `runner.run` regardless of model/judge.

Two modes:

* fixed-`k`: draw exactly `k` samples total (one or more chunks of the same size), used when
  `RunSpec.adaptive is None`.
* `AdaptiveRule`-driven: draw in chunks of `AdaptiveRule.chunk` samples, stopping an item early
  once it has accumulated `stop_c` successes (it is "mastered"; further samples would not change
  the benchmark's conclusion) or once it has drawn `n_max` samples in total (the budget is spent).

This module only decides *how many more samples one item gets next*; suite-level policies such as
plan §2's "stop increasing n for a (family, model) after two consecutive n with pass@32 = 0" are a
suite-building/run-scheduling decision layered on top by the caller (e.g. via `RunSpec.pids`
across separate `run()` calls per `n`), not implemented here.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AdaptiveRule:
    """Draw in chunks of `chunk` samples; stop an item once it has `stop_c` successes or `n_max`
    total draws, whichever comes first."""

    chunk: int
    n_max: int
    stop_c: int


@dataclass(frozen=True)
class ItemProgress:
    """One item's cumulative draws across chunks so far."""

    n: int = 0
    c: int = 0


def next_chunk_n(k: int, adaptive: AdaptiveRule | None, progress: ItemProgress) -> int:
    """How many samples to draw for this item in the next chunk; `0` means the item is done.

    Fixed-`k` mode draws whatever is left of `k` (usually all of it in one chunk, but this stays
    correct if a caller draws it in smaller pieces too, e.g. to respect `RunSpec.budget_s`).
    Adaptive mode stops (`0`) once `stop_c` successes or `n_max` total draws is reached, and
    otherwise offers `min(chunk, n_max - n)` so the very last chunk never overshoots `n_max`.
    """
    if adaptive is None:
        return max(0, k - progress.n)
    if progress.c >= adaptive.stop_c or progress.n >= adaptive.n_max:
        return 0
    return max(0, min(adaptive.chunk, adaptive.n_max - progress.n))
