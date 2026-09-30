"""`Judge` interface (plan §1): adjudicate a `Sequent`'s sampled `Attempt`s into `Verdict`s."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from ..core.types import Attempt, Sequent, Verdict


class Judge(Protocol):
    """Something that can decide whether a sampled proof attempt actually proves its sequent."""

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


class NotAvailable(Exception):
    """Raised by a `Judge` backend that is not usable in this environment (e.g. a future
    `LeanJudge` with no `lean` binary on `PATH`). Callers should skip the benchmark for that judge
    rather than treat this as a run failure.
    """
