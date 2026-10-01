"""`RunSpec` and `run`: sample a suite against a `ModelAdapter`, judge every attempt, and persist
the results to a `Store` (plan §1's `runner/runner.py`).

`AdaptiveRule` is defined in `adaptive.py` (the module that implements the actual per-item
stopping decision) and re-exported here so `from code.ndbench.runner.runner import AdaptiveRule`
matches the plan's presentation.

Sampling proceeds in rounds: each round, every still-active item is due for its next chunk of
draws, and items that are done (adaptive stop condition met, or `k` reached) drop out of the next
round. Each item's per-chunk seed is `sha256(f"{spec.seed}:{pid}:{chunk}")` (`chunk` is *that
item's own* count of chunks drawn so far, tracked per item -- see "Incremental writing" below --
which for any run where every item advances in lockstep, i.e. any run that never resumes and never
has one item finish before another, is numerically identical to a single shared round counter), so
repeating a `RunSpec` against a deterministic adapter (e.g. `adapters.fake.FakeAdapter`) reproduces
byte-identical attempts chunk-by-chunk.

**Incremental writing (throughput fix list item 2).** A round's active items are split into groups
of `spec.options["items_per_call"]` items (default 256): each group gets its own `adapter.sample`
call, is judged, and is written to `store` (`append_items`/`append_samples`) immediately -- before
the next group in the round even starts -- rather than the whole round's items being batched into
one giant `adapter.sample` call with nothing persisted until it returns. A process killed at any
point therefore leaves every group that finished before the kill already in the `Store`; only the
group in flight (and any later ones) is lost. This does not change the sampling distribution or
`ItemResult` semantics versus one big per-round call (`AdaptiveRule`/`first_success`/chunk
bookkeeping are all per-item and unaffected by how items are grouped into calls; grouping only
changes how a real batched adapter's `torch.Generator` state is consumed within a round -- see
`adapters.absgpt.AbsGptAdapter.sample`'s own determinism note).

**Per-item `max_new_tokens` cap (throughput follow-up).** When `RunSpec.max_new_tokens` is `None`,
each item's cap is no longer a flat `1024 - prompt_tokens` (which for a short reference proof left
~1000 unused decode steps that a single non-terminating attempt would run to the end, dominating a
whole batch's wall time regardless of batching): `RunSpec.options["max_new_policy"]` picks between
`"ref3x"` (the default) and `"full"` (the old flat cap), and is part of `config_key` like any other
option. `"ref3x"` sizes the cap off the item's own reference proof length (`max(160, 3 *
ref_tokens + 32)`, `ref_tokens` from `adapter.token_count` on `Problem.ref.body` when present, else
estimated from `problem.length.L`/`L_ub` times `MEAN_TOKENS_PER_REF_LINE`), always clamped to never
exceed the `"full"` cap. The result is then rounded *up* to the next multiple of 64
(`_round_up_to_multiple`) so items with similar (but not identical) reference lengths still land in
the same `(max_new_tokens, temperature)` group in `AbsGptAdapter.sample`'s cross-request batching --
without this, a continuous, almost-never-repeating per-item cap would fragment every round into
many near-singleton groups, largely defeating that batching. See `_resolve_max_new_tokens`.
`ItemResult.hit_max` (already `not attempt.finished`, i.e. an attempt that used its whole cap
without reaching `QED`) is what makes a run's truncation rate visible; it is unaffected by which
policy set the cap.

**Resume, skip, and `rerun` (rerun-merging fix).** Before starting, `run` looks up
`store.latest(spec.suite, spec.model, config_key)` (the exact `run_id` a fresh
`store.new_run(spec, identity, judge_id)` call for this config would create or re-append a start
event to, `run_id` being a deterministic function of `config_key`; `store.latest` already excludes
any `"invalid"`-status run -- see `store.py`'s own docstring -- so an invalid run is treated as not
present here too) and its last known status, with three outcomes:

- **`"partial"`/`"running"`** (started but never reached `finish_run` -- i.e. a prior attempt was
  killed): *resumed*. Every item's already-recorded `(n, c)` (`store.items(run_id)`) seeds this
  run's starting `ItemProgress` instead of `0, 0`, and each item's own next chunk number picks up
  one past the highest chunk number it already has recorded (`store.item_chunks`), so a resumed run
  draws no duplicate `(pid, chunk)` seed and skips any item whose recorded progress already
  satisfies `spec.k`/`spec.adaptive`.
- **`"complete"`**, `rerun=False` (the default): *skipped* entirely -- no suite is even re-sampled,
  no new `Store` writes happen at all -- and `run` returns that existing run's own summary (its
  `n`/`c` read back from `store.items`) with `"skipped": True`. This is the fix for the bug this
  section used to describe: re-running an unchanged `RunSpec` against an already-complete config
  no longer draws a second, silently-merged batch of samples into the same `run_id`.
- **`rerun=True`** (the CLI's `--rerun`): resume/skip are both bypassed regardless of the latest
  run's status, and every item starts fully fresh (`progress`/`next_chunk` empty) under a brand
  new, non-colliding `run_id`: the deterministic base id plus the lowest free `-r2`, `-r3`, ...
  suffix (`_next_rerun_run_id`). The prior run's own rows are never touched or resumed from.

Two different effective configs can therefore never again land on the same `run_id`
(`Store.config_key`'s "Effective config_key" fix handles the *cause*; this section's skip/rerun
handling is what makes re-running a *genuinely* unchanged, already-complete config a safe no-op
instead of undefined behaviour).
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, TypedDict, cast

from ..adapters.base import ModelAdapter
from ..core.types import JSON, Attempt, ItemResult, Problem, SampleRequest, Verdict
from ..judge.base import Judge
from .adaptive import AdaptiveRule, ItemProgress, next_chunk_n

if TYPE_CHECKING:
    from ..store import Store

__all__ = ["AdaptiveRule", "RunSpec", "RunSummary", "run"]

#: `RunSpec.options["items_per_call"]` default (fix list item 2): how many items' requests go into
#: one `adapter.sample` call/`store.append_items` write within a round.
DEFAULT_ITEMS_PER_CALL = 256

#: `RunSpec.options["max_new_policy"]` default (throughput follow-up): size each item's cap off its
#: own reference proof length rather than a flat `1024 - prompt_tokens`.
DEFAULT_MAX_NEW_POLICY = "ref3x"

#: Mean tokens per reference-proof line under the `absgpt` tokenizer ("abs" mode), used only as a
#: `problem.length.L`/`L_ub`-based fallback estimate of `ref_tokens` when `Problem.ref` is `None`
#: (so there is no reference body to measure `adapter.token_count` on directly). Measured on
#: `parts_pool` v1's 1139 reference proofs (all of which happen to carry a `ref`, so this fallback
#: path is untested against real suite data): `sum(len(Tokenizer("abs").encode_proof(p.ref.body))
#: for p in problems) / sum(p.ref.lines for p in problems)` = 94363 / 5866 = 16.0864... tokens/line
#: (2026-09-30). This is absgpt-tokenizer-specific; a non-absgpt adapter's `token_count` would give
#: a different true ratio, but no generic per-adapter measurement is available, so this is the best
#: estimate on hand for any adapter's fallback.
MEAN_TOKENS_PER_REF_LINE = 16.0864

#: `_resolve_max_new_tokens`'s floor and multiplier for the `"ref3x"` policy (throughput follow-up:
#: "cap max_new per item from its reference proof length").
_REF3X_FLOOR = 160
_REF3X_MULTIPLIER = 3
_REF3X_MARGIN = 32

#: `_resolve_max_new_tokens` rounds a resolved cap up to the next multiple of this many tokens, so
#: items with similar (but not identical) caps share one `(max_new_tokens, temperature)` group in
#: `AbsGptAdapter.sample`'s cross-request batching (throughput follow-up: "rebucket requests").
_MAX_NEW_BUCKET = 64


def _round_up_to_multiple(n: int, multiple: int) -> int:
    """`n` rounded up to the next multiple of `multiple` (a positive int); exact ceiling-division
    arithmetic, no floating point."""
    return -(-n // multiple) * multiple


def _estimate_ref_tokens(adapter: ModelAdapter, problem: Problem, prompt_tokens: int) -> int | None:
    """`problem`'s own reference-proof token count (excluding the prompt), for the `"ref3x"`
    `max_new_tokens` policy. Measured directly via `adapter.token_count` when `problem.ref` exists;
    otherwise estimated from `problem.length.L` (or `L_ub`, the next-tightest bound) times
    `MEAN_TOKENS_PER_REF_LINE`. Returns `None` only when neither is available (no reference body
    and no length bound at all), in which case the caller falls back to the `"full"` cap."""
    if problem.ref is not None:
        full = adapter.token_count(problem.sequent, problem.ref.body)
        return max(0, full - prompt_tokens)
    length_l = problem.length.L if problem.length.L is not None else problem.length.L_ub
    if length_l is None:
        return None
    return max(0, round(length_l * MEAN_TOKENS_PER_REF_LINE))


def _resolve_max_new_tokens(
    policy: str, adapter: ModelAdapter, problem: Problem, prompt_tokens: int
) -> int:
    """One item's `max_new_tokens` when `RunSpec.max_new_tokens` is `None` (module docstring's
    "Per-item `max_new_tokens` cap"): `"full"` is the pre-fix flat `1024 - prompt_tokens`; the
    default `"ref3x"` instead scales with the item's own reference proof length (`max(160, 3 *
    ref_tokens + 32)`, `_REF3X_FLOOR`/`_REF3X_MULTIPLIER`/`_REF3X_MARGIN`), falling back to the
    `"full"` cap if no reference length estimate is available at all. Either way, the result is
    rounded up to the next multiple of `_MAX_NEW_BUCKET` (so similar items share a batching group)
    and then clamped to never exceed the `"full"` ceiling -- in that order, since bucketing a value
    already at the ceiling must not push it back over.
    """
    ceiling = max(1, 1024 - prompt_tokens)
    if policy == "full":
        raw_cap = ceiling
    else:
        ref_tokens = _estimate_ref_tokens(adapter, problem, prompt_tokens)
        raw_cap = (
            ceiling
            if ref_tokens is None
            else max(_REF3X_FLOOR, _REF3X_MULTIPLIER * ref_tokens + _REF3X_MARGIN)
        )
    return min(_round_up_to_multiple(raw_cap, _MAX_NEW_BUCKET), ceiling)


class RunSummary(TypedDict):
    """`run()`'s return shape. `skipped` is `True` only when an already-`"complete"` run for this
    exact config was found and `rerun=False`, in which case no new sampling happened at all and
    every other field reflects that *existing* run's own recorded data (module docstring's
    "Resume, skip, and rerun")."""

    run_id: str
    status: str
    n_items: int
    n_chunks: int
    n: int
    c: int
    skipped: bool


@dataclass(frozen=True)
class RunSpec:
    """What to run: sample `suite`/`version`'s problems `k` times each (or per `AdaptiveRule`)
    against `model`, judge every attempt, and store the result."""

    suite: str
    version: str
    model: str
    k: int
    temperature: float = 0.8
    max_new_tokens: int | None = None  # None -> per `options["max_new_policy"]`, per item
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


def _latest_status(
    store: Store, suite: str, model: str, config_key: str
) -> tuple[str | None, str | None]:
    """`(run_id, status)` of the latest run matching `config_key`, or `(None, None)` if there is
    none. Goes through `store.latest`, which already excludes any `"invalid"`-status run (`store.
    py`'s own docstring), so an invalid run is treated as not present by every caller of this
    helper too (module docstring's "Resume, skip, and rerun")."""
    run_id = store.latest(suite, model, config_key)
    if run_id is None:
        return None, None
    records = store.runs(run_id=run_id)
    status = cast("str | None", records[0]["status"]) if records else None
    return run_id, status


def _next_rerun_run_id(store: Store, base_run_id: str) -> str:
    """The lowest free `-r<N>` suffix (`N` starting at 2) for `base_run_id` (module docstring's
    `rerun=True`): the plain `base_run_id` itself is implicitly "attempt 1", so the first rerun is
    `-r2`, the next `-r3`, and so on -- the smallest `N` such that `f"{base_run_id}-r{N}"` is not
    already a recorded `run_id`."""
    existing = {cast(str, r["run_id"]) for r in store.runs()}
    n = 2
    while f"{base_run_id}-r{n}" in existing:
        n += 1
    return f"{base_run_id}-r{n}"


def _skipped_summary(store: Store, run_id: str, n_items: int) -> RunSummary:
    """A `RunSummary` for the "already complete, `rerun=False`" case (module docstring's "Resume,
    skip, and rerun"): no new sampling happens, and `n`/`c` are `run_id`'s own already-recorded
    totals (`store.items`). `n_chunks` is recovered best-effort from `run_id`'s raw `item_chunks`
    (the highest chunk number any item reached, plus one) rather than tracked live, since no round
    actually executes in this path."""
    items = store.items(run_id)
    chunks = store.item_chunks(run_id)
    n_chunks = (max(cast(int, row["chunk"]) for row in chunks) + 1) if chunks else 0
    return {
        "run_id": run_id,
        "status": "complete",
        "n_items": n_items,
        "n_chunks": n_chunks,
        "n": sum(it.n for it in items),
        "c": sum(it.c for it in items),
        "skipped": True,
    }


def _resume_state(
    store: Store, spec: RunSpec, identity: Mapping[str, JSON], judge_id: str, pids: set[str]
) -> tuple[dict[str, ItemProgress], dict[str, int], str | None]:
    """Resume support (module docstring's "Resume, skip, and rerun"): if the latest run for this
    exact `(spec, identity, judge_id)` config key has status `"partial"` or `"running"`, return
    `(progress, next_chunk, resumed_run_id)` seeded from that run's already-recorded items
    (restricted to `pids`, in case `spec.pids` narrowed the suite since the prior attempt);
    otherwise `({}, {}, None)`, meaning every item starts fresh at `(n=0, c=0)`, `chunk=0`.

    `resumed_run_id`, when not `None`, is guaranteed to equal the `run_id` a fresh `store.new_run`
    call for this same config produces (both are the same deterministic function of the config
    key) -- `run` asserts this rather than silently trusting it.
    """
    config_key = store.config_key(spec, identity, judge_id)
    prior_run_id, prior_status = _latest_status(store, spec.suite, spec.model, config_key)
    if prior_run_id is None or prior_status not in ("partial", "running"):
        return {}, {}, None

    progress = {
        item.pid: ItemProgress(n=item.n, c=item.c)
        for item in store.items(prior_run_id)
        if item.pid in pids
    }
    last_chunk: dict[str, int] = {}
    for row in store.item_chunks(prior_run_id):
        pid = cast(str, row["pid"])
        if pid in pids:
            last_chunk[pid] = max(last_chunk.get(pid, -1), cast(int, row["chunk"]))
    next_chunk = {pid: c + 1 for pid, c in last_chunk.items()}
    return progress, next_chunk, prior_run_id


def run(
    spec: RunSpec, adapter: ModelAdapter, judge: Judge, store: Store, *, rerun: bool = False
) -> RunSummary:
    """Sample `spec.suite`/`spec.version` against `adapter`, judge with `judge`, and record every
    chunk's `ItemResult` (and, if `spec.keep_samples`, every raw sample) to `store`, writing after
    every `spec.options["items_per_call"]`-sized group within a round rather than once per round
    (module docstring's "Incremental writing").

    Resume/skip/rerun (module docstring's "Resume, skip, and rerun"): with `rerun=False` (the
    default), a killed prior attempt at the same config (status `"partial"`/`"running"`) is resumed
    rather than resampled from scratch, and an already-`"complete"` prior attempt is *skipped*
    entirely -- this call does no sampling and no `Store` writes, and returns that existing run's
    own summary with `"skipped": True`. With `rerun=True` (the CLI's `--rerun`), resume/skip are
    both bypassed and every item is drawn fresh under a brand new `-r2`/`-r3`/... `run_id`,
    regardless of the latest run's status.

    Returns a small JSON-able summary (`RunSummary`, a `dict[str, JSON]`-compatible `TypedDict`):
    `run_id`, final `status` (`"complete"` if every item finished within `spec.budget_s`, else
    `"partial"`), item/chunk/sample counts, and `skipped`.
    """
    problems: list[Problem] = store.load_suite(spec.suite, spec.version)
    if spec.pids is not None:
        wanted = set(spec.pids)
        problems = [p for p in problems if p.pid in wanted]
    by_pid = {p.pid: p for p in problems}

    identity = adapter.identity()
    config_key = store.config_key(spec, identity, judge.judge_id)
    base_run_id = f"{spec.suite}-{spec.version}-{spec.model}-{config_key}"

    if not rerun:
        prior_run_id, prior_status = _latest_status(store, spec.suite, spec.model, config_key)
        if prior_status == "complete":
            assert prior_run_id is not None
            return _skipped_summary(store, prior_run_id, len(problems))

    if rerun:
        resumed_progress: dict[str, ItemProgress] = {}
        next_chunk: dict[str, int] = {}
        run_id = _next_rerun_run_id(store, base_run_id)
        store.new_run(spec, identity, judge.judge_id, run_id=run_id)
    else:
        resumed_progress, next_chunk, resumed_run_id = _resume_state(
            store, spec, identity, judge.judge_id, set(by_pid)
        )
        run_id = store.new_run(spec, identity, judge.judge_id)
        if resumed_run_id is not None:
            assert run_id == resumed_run_id, (
                f"runner.run: resumed run_id {resumed_run_id!r} must equal the freshly computed "
                f"run_id {run_id!r} (both are the same deterministic function of the config key)"
            )

    progress: dict[str, ItemProgress] = {
        pid: resumed_progress.get(pid, ItemProgress()) for pid in by_pid
    }
    for pid in by_pid:
        next_chunk.setdefault(pid, 0)
    items_per_call_opt = spec.options.get("items_per_call", DEFAULT_ITEMS_PER_CALL)
    items_per_call = max(1, int(cast("int | float | str", items_per_call_opt)))
    max_new_policy = str(spec.options.get("max_new_policy", DEFAULT_MAX_NEW_POLICY))

    active = {pid for pid in by_pid if next_chunk_n(spec.k, spec.adaptive, progress[pid]) > 0}
    t0 = time.monotonic()
    status = "complete"
    n_chunks_written = 0

    def _budget_exceeded() -> bool:
        return spec.budget_s is not None and time.monotonic() - t0 >= spec.budget_s

    while active:
        if _budget_exceeded():
            status = "partial"
            break

        active_sorted = sorted(active)
        round_cut_short = False
        for group_start in range(0, len(active_sorted), items_per_call):
            if _budget_exceeded():
                status = "partial"
                round_cut_short = True
                break

            group_pids = active_sorted[group_start : group_start + items_per_call]
            reqs: list[SampleRequest] = []
            for pid in group_pids:
                this_n = next_chunk_n(spec.k, spec.adaptive, progress[pid])
                if this_n <= 0:
                    continue
                problem = by_pid[pid]
                max_new = spec.max_new_tokens
                if max_new is None:
                    prompt_tokens = adapter.token_count(problem.sequent)
                    max_new = _resolve_max_new_tokens(
                        max_new_policy, adapter, problem, prompt_tokens
                    )
                reqs.append(
                    SampleRequest(
                        pid=pid,
                        sequent=problem.sequent,
                        n=this_n,
                        temperature=spec.temperature,
                        max_new_tokens=max_new,
                        seed=_item_seed(spec.seed, pid, next_chunk[pid]),
                        prefix=problem.prefix,
                        options=spec.options,
                    )
                )
            if not reqs:
                continue

            attempts_by_req = adapter.sample(reqs)
            # One `judge_many` call across every item in this group (module docstring's
            # "Incremental writing" sizes the group; `judge.judge_many`'s own docstring is why a
            # group, not the whole round, is the batching unit) rather than `judge.judge` once per
            # item: a judge that starts external processes (`LeanJudge`) then starts a handful of
            # `lean` processes per group instead of per item.
            req_problems = [by_pid[req.pid] for req in reqs]
            ref_lines_by_req = [
                (problem.ref.lines if problem.ref is not None else None) for problem in req_problems
            ]
            judge_items = [
                (problem.sequent, attempts, ref_lines)
                for problem, attempts, ref_lines in zip(
                    req_problems, attempts_by_req, ref_lines_by_req, strict=True
                )
            ]
            verdicts_by_req = judge.judge_many(judge_items)

            chunk_results: list[ItemResult] = []
            chunk_samples: list[dict[str, JSON]] = []
            for req, attempts, ref_lines, verdicts in zip(
                reqs, attempts_by_req, ref_lines_by_req, verdicts_by_req, strict=True
            ):
                accum = _ChunkAccum(run_id=run_id, pid=req.pid, chunk=next_chunk[req.pid])
                for a, v in zip(attempts, verdicts, strict=True):
                    chunk_samples.append(accum.record(a, v, ref_lines))
                item_result = accum.finish()
                chunk_results.append(item_result)
                progress[req.pid] = ItemProgress(
                    n=progress[req.pid].n + item_result.n, c=progress[req.pid].c + item_result.c
                )
                next_chunk[req.pid] += 1

            # Written immediately, before the next group (or round) even starts: a process killed
            # at any point after this line leaves this group's items in `store` (fix list item 2).
            store.append_items(run_id, chunk_results)
            if spec.keep_samples:
                store.append_samples(run_id, chunk_samples)

        if round_cut_short:
            break  # `status` is already "partial"; whatever groups ran this round are in `store`
        n_chunks_written += 1
        active = {pid for pid in active if next_chunk_n(spec.k, spec.adaptive, progress[pid]) > 0}

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
        "skipped": False,
    }
