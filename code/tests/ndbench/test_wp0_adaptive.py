"""`runner.adaptive.next_chunk_n`: fixed-k and `AdaptiveRule`-driven stopping decisions."""

from __future__ import annotations

from code.ndbench.runner.adaptive import AdaptiveRule, ItemProgress, next_chunk_n


def test_fixed_k_mode() -> None:
    assert next_chunk_n(10, None, ItemProgress()) == 10
    assert next_chunk_n(10, None, ItemProgress(n=7, c=2)) == 3
    assert next_chunk_n(10, None, ItemProgress(n=10, c=1)) == 0
    assert next_chunk_n(10, None, ItemProgress(n=15, c=1)) == 0  # never goes negative


def test_adaptive_mode_stops_on_stop_c() -> None:
    rule = AdaptiveRule(chunk=8, n_max=100, stop_c=3)
    assert next_chunk_n(999, rule, ItemProgress(n=0, c=0)) == 8
    assert next_chunk_n(999, rule, ItemProgress(n=16, c=2)) == 8
    assert next_chunk_n(999, rule, ItemProgress(n=16, c=3)) == 0
    assert next_chunk_n(999, rule, ItemProgress(n=16, c=5)) == 0


def test_adaptive_mode_stops_on_n_max() -> None:
    rule = AdaptiveRule(chunk=8, n_max=20, stop_c=100)
    assert next_chunk_n(999, rule, ItemProgress(n=16, c=0)) == 4  # last chunk is clipped
    assert next_chunk_n(999, rule, ItemProgress(n=20, c=0)) == 0


def test_adaptive_mode_never_overshoots_n_max() -> None:
    rule = AdaptiveRule(chunk=8, n_max=10, stop_c=100)
    n = next_chunk_n(999, rule, ItemProgress(n=0, c=0))
    assert n == 8
    n2 = next_chunk_n(999, rule, ItemProgress(n=8, c=0))
    assert n2 == 2
