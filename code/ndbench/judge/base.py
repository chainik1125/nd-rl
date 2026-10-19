"""`Judge` interface (plan §1): adjudicate a `Sequent`'s sampled `Attempt`s into `Verdict`s."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from ..core.types import Attempt, Sequent, Verdict


class Judge(Protocol):
    """Something that can decide whether a sampled proof attempt actually proves its sequent.

    Concrete judges (`judge.nd.NDJudge`, `judge.lean.LeanJudge`) subclass this directly (rather
    than only matching it structurally) so they inherit `judge_many`'s default implementation for
    free -- see that method's docstring.
    """

    #: A short, stable identifier for this judge/checker version (part of the `Store` run key,
    #: so a checker change is visible in stored results rather than silently reinterpreting them).
    judge_id: str

    def judge(
        self, seq: Sequent, attempts: Sequence[Attempt], ref_lines: int | None
    ) -> list[Verdict]:
        """One `Verdict` per attempt, in the same order. `ref_lines` (the reference proof's
        written length, if known) is provided for judges that need a search bound (e.g. a future
        `LeanJudge` reconstructing a bounded proof) and may be ignored otherwise."""
        ...

    def judge_many(
        self, items: Sequence[tuple[Sequent, Sequence[Attempt], int | None]]
    ) -> list[list[Verdict]]:
        """`judge(seq, attempts, ref_lines)` over several `(seq, attempts, ref_lines)` groups at
        once (plan §2 WP-D2 item 2) -- e.g. every item in one runner round/chunk, or every
        (pid, chunk) group of a `rejudge` -- returning one `Verdict` list per input group, same
        order. The runner and `rejudge` call this once per group instead of `judge` once per
        item, so a judge that starts external processes (`LeanJudge`, which starts a handful of
        `lean` processes per *call* rather than per item -- without this, online Lean judging
        costs about an hour of per-item process starts) can batch across the whole group.

        Default: loop over `judge`, one call per group -- correct for any `Judge`, and exactly
        today's per-item behaviour for a judge (`NDJudge`) with nothing to gain from batching.
        """
        return [self.judge(seq, attempts, ref_lines) for seq, attempts, ref_lines in items]


class NotAvailable(Exception):
    """Raised by a `Judge` backend that is not usable in this environment (e.g. a future
    `LeanJudge` with no `lean` binary on `PATH`). Callers should skip the benchmark for that judge
    rather than treat this as a run failure.
    """
