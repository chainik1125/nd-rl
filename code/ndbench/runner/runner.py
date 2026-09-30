"""`RunSpec` and `run`: sample a suite against a `ModelAdapter`, judge every attempt, and persist
the results to a `Store` (plan §1's `runner/runner.py`).

`AdaptiveRule` is defined in `adaptive.py` (the module that implements the actual per-item
stopping decision) and re-exported here so `from code.ndbench.runner.runner import AdaptiveRule`
matches the plan's presentation.

Sampling proceeds in rounds ("chunks"): each round, every still-active item's next chunk of
requests is batched into one `adapter.sample(...)` call (so a real batched-inference adapter gets
one big batch per round, not one call per item), the results are judged, one `ItemResult` per
(item, chunk) is appended to the `Store`, and items that are done (adaptive stop condition met, or
`k` reached) drop out of the next round. Each item's per-chunk seed is
`sha256(f"{spec.seed}:{pid}:{chunk}")`, so repeating a `RunSpec` against a deterministic adapter
(e.g. `adapters.fake.FakeAdapter`) reproduces byte-identical attempts chunk-by-chunk.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, TypedDict

from ..adapters.base import ModelAdapter
from ..core.types import JSON, Attempt, ItemResult, Problem, SampleRequest, Verdict
from ..judge.base import Judge
from .adaptive import AdaptiveRule, ItemProgress, next_chunk_n

if TYPE_CHECKING:
    from ..store import Store

__all__ = ["AdaptiveRule", "RunSpec", "RunSummary", "run"]


class RunSummary(TypedDict):
    """`run()`'s return shape."""

    run_id: str
    status: str
    n_items: int
    n_chunks: int
    n: int
    c: int


@dataclass(frozen=True)
class RunSpec:
    """What to run: sample `suite`/`version`'s problems `k` times each (or per `AdaptiveRule`)
    against `model`, judge every attempt, and store the result."""

    suite: str
    version: str
    model: str
    k: int
    temperature: float = 0.8
    max_new_tokens: int | None = None  # None -> 1024 - prompt tokens, per item
    seed: int = 0
    adaptive: AdaptiveRule | None = None
    pids: tuple[str, ...] | None = None  # None -> every problem in the suite
    options: Mapping[str, JSON] = field(default_factory=dict)
    keep_samples: bool = True
    budget_s: float | None = None


def _item_seed(seed: int, pid: str, chunk: int) -> int:
    """Deterministic per-(run seed, item, chunk) seed, as `sha256(seed, pid, chunk)`."""
    digest = hashlib.sha256(f"{seed}:{pid}:{chunk}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


@dataclass
class _ChunkAccum:
    """Mutable running totals for one (item, chunk), folded into an `ItemResult` when the chunk
    is done. Kept separate from `ItemResult` itself (which is frozen-shaped by the plan's
    interface and meant as the finished, persisted record)."""

    run_id: str
    pid: str
    chunk: int
    n: int = 0
    c: int = 0
    first_success: int | None = None
    ok_written: list[int] = field(default_factory=list)
    ok_pruned: list[int] = field(default_factory=list)
    ok_tokens: list[int] = field(default_factory=list)
    ok_starts: list[int] = field(default_factory=list)
    fail_kinds: dict[str, int] = field(default_factory=dict)
    fail_lines: list[int] = field(default_factory=list)
    long_enough: int = 0
    tokens_sum: int = 0
    hit_max: int = 0

    def record(self, a: Attempt, v: Verdict, ref_len: int | None) -> dict[str, JSON]:
        """Fold one judged attempt into the running totals; returns its raw sample row.

        `long_enough` counts attempts (successful or not) whose written length reaches the
        problem's reference length (`ref_len`); this is a WP0 interpretation of an otherwise-
        undocumented `ItemResult` field, chosen to support plan §2 B1's "fraction of attempts >=
        reference length" metric.
        """
        self.n += 1
        self.tokens_sum += a.n_tokens
        if not a.finished:
            self.hit_max += 1
        if ref_len is not None and v.written is not None and v.written >= ref_len:
            self.long_enough += 1
        if v.ok:
            self.c += 1
            if self.first_success is None:
                self.first_success = self.n
            self.ok_written.append(v.written if v.written is not None else 0)
            self.ok_pruned.append(v.pruned if v.pruned is not None else 0)
            self.ok_tokens.append(a.n_tokens)
            self.ok_starts.append(v.start if v.start is not None else 0)
        else:
            self.fail_kinds[v.kind] = self.fail_kinds.get(v.kind, 0) + 1
            if v.first_bad_line is not None:
                self.fail_lines.append(v.first_bad_line)
        return {
            "pid": a.pid,
            "chunk": self.chunk,
            "i": a.i,
            "body": a.body,
            "native": a.native,
            "n_tokens": a.n_tokens,
            "finished": a.finished,
            "ok": v.ok,
            "reason": v.reason,
            "kind": v.kind,
            "written": v.written,
            "pruned": v.pruned,
            "first_bad_line": v.first_bad_line,
            "start": v.start,
        }

    def finish(self) -> ItemResult:
        return ItemResult(
            run_id=self.run_id,
            pid=self.pid,
            chunk=self.chunk,
            n=self.n,
            c=self.c,
            first_success=self.first_success,
            ok_written=self.ok_written,
            ok_pruned=self.ok_pruned,
            ok_tokens=self.ok_tokens,
            ok_starts=self.ok_starts,
            fail_kinds=self.fail_kinds,
            fail_lines=self.fail_lines,
            long_enough=self.long_enough,
            tokens_mean=(self.tokens_sum / self.n if self.n else 0.0),
            hit_max=self.hit_max,
        )


def run(spec: RunSpec, adapter: ModelAdapter, judge: Judge, store: Store) -> RunSummary:
    """Sample `spec.suite`/`spec.version` against `adapter`, judge with `judge`, and record every
    chunk's `ItemResult` (and, if `spec.keep_samples`, every raw sample) to `store`.

    Returns a small JSON-able summary (`RunSummary`, a `dict[str, JSON]`-compatible `TypedDict`):
    `run_id`, final `status` (`"complete"` if every item finished within `spec.budget_s`, else
    `"partial"`), item/chunk/sample counts.
    """
    problems: list[Problem] = store.load_suite(spec.suite, spec.version)
    if spec.pids is not None:
        wanted = set(spec.pids)
        problems = [p for p in problems if p.pid in wanted]
    by_pid = {p.pid: p for p in problems}

    identity = adapter.identity()
    run_id = store.new_run(spec, identity, judge.judge_id)

    progress: dict[str, ItemProgress] = dict.fromkeys(by_pid, ItemProgress())
    active = set(by_pid)
    chunk = 0
    t0 = time.monotonic()
    status = "complete"
    n_chunks_written = 0

    while active:
        if spec.budget_s is not None and time.monotonic() - t0 >= spec.budget_s:
            status = "partial"
            break

        reqs: list[SampleRequest] = []
        for pid in sorted(active):
            this_n = next_chunk_n(spec.k, spec.adaptive, progress[pid])
            if this_n <= 0:
                continue
            problem = by_pid[pid]
            max_new = spec.max_new_tokens
            if max_new is None:
                max_new = max(1, 1024 - adapter.token_count(problem.sequent))
            reqs.append(
                SampleRequest(
                    pid=pid,
                    sequent=problem.sequent,
                    n=this_n,
                    temperature=spec.temperature,
                    max_new_tokens=max_new,
                    seed=_item_seed(spec.seed, pid, chunk),
                    prefix=problem.prefix,
                    options=spec.options,
                )
            )
        if not reqs:
            break

        attempts_by_req = adapter.sample(reqs)
        chunk_results: list[ItemResult] = []
        chunk_samples: list[dict[str, JSON]] = []
        for req, attempts in zip(reqs, attempts_by_req, strict=True):
            problem = by_pid[req.pid]
            ref_lines = problem.ref.lines if problem.ref is not None else None
            verdicts = judge.judge(problem.sequent, attempts, ref_lines)
            accum = _ChunkAccum(run_id=run_id, pid=req.pid, chunk=chunk)
            for a, v in zip(attempts, verdicts, strict=True):
                chunk_samples.append(accum.record(a, v, ref_lines))
            item_result = accum.finish()
            chunk_results.append(item_result)
            progress[req.pid] = ItemProgress(
                n=progress[req.pid].n + item_result.n, c=progress[req.pid].c + item_result.c
            )

        store.append_items(run_id, chunk_results)
        if spec.keep_samples:
            store.append_samples(run_id, chunk_samples)
        n_chunks_written += 1

        active = {pid for pid in active if next_chunk_n(spec.k, spec.adaptive, progress[pid]) > 0}
        chunk += 1

    store.finish_run(run_id, status, time.monotonic() - t0)
    total_n = sum(p.n for p in progress.values())
    total_c = sum(p.c for p in progress.values())
    return {
        "run_id": run_id,
        "status": status,
        "n_items": len(problems),
        "n_chunks": n_chunks_written,
        "n": total_n,
        "c": total_c,
    }
