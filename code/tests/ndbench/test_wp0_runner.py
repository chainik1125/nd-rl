"""WP0 acceptance test: `runner.run` with `FakeAdapter` gives the expected `n`/`c`, adaptive
stopping, and deterministic seeds (plan §3 WP0 row)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from code.ndbench.adapters.base import ModelAdapter
from code.ndbench.adapters.fake import FakeAdapter
from code.ndbench.core.io import git_commit_sha, sha256_str, utc_now_iso
from code.ndbench.core.proof import pruned_length
from code.ndbench.core.types import (
    JSON,
    Attempt,
    LengthLabel,
    Problem,
    ReferenceProof,
    SampleRequest,
    Sequent,
    SuiteManifest,
)
from code.ndbench.judge.nd import NDJudge
from code.ndbench.runner.adaptive import AdaptiveRule
from code.ndbench.runner.runner import (
    MEAN_TOKENS_PER_REF_LINE,
    RunSpec,
    _estimate_ref_tokens,
    _item_seed,
    _resolve_max_new_tokens,
    _round_up_to_multiple,
    run,
)
from code.ndbench.store import Store

from ._fixtures import load_valid_proofs


def _problems(n: int = 2) -> list[Problem]:
    proofs = [p for p in load_valid_proofs() if p["n_lines"] <= 4][:n]
    assert len(proofs) == n
    out = []
    for i, p in enumerate(proofs):
        seq = Sequent.from_prompt(p["prompt"])
        out.append(
            Problem(
                pid=f"item{i}",
                suite="demo",
                version="v1",
                sequent=seq,
                length=LengthLabel(
                    L=p["n_lines"], L_ub=None, L_lb=None, method="minlen_exact", bound=None
                ),
                ref=ReferenceProof(
                    body=p["proof"],
                    lines=p["n_lines"],
                    pruned=pruned_length(p["proof"]),
                    source="minlen",
                ),
                tags={},
            )
        )
    return out


def _write_suite(store: Store, problems: list[Problem], seed: int = 0) -> None:
    manifest = SuiteManifest(
        suite="demo",
        version="v1",
        builder="test",
        config={},
        seed=seed,
        created_utc=utc_now_iso(),
        git_commit=git_commit_sha(),
        n_items=len(problems),
        items_sha256=sha256_str("x"),
        excl_sources={},
        stats={},
    )
    store.write_suite(problems, manifest)


def _refs(problems: list[Problem]) -> dict[str, str]:
    return {p.pid: p.ref.body for p in problems if p.ref is not None}


def test_adaptive_stop_early_when_p_is_one(tmp_path: Path) -> None:
    """p=1.0: every draw is the reference proof, so the item hits `stop_c` inside its very first
    chunk and never draws a second one -- an exact, non-random expected outcome."""
    problems = _problems()
    store = Store(tmp_path)
    _write_suite(store, problems)
    adapter = FakeAdapter(refs=_refs(problems), p=1.0)
    spec = RunSpec(
        suite="demo",
        version="v1",
        model="fake",
        k=999,  # unused in adaptive mode
        adaptive=AdaptiveRule(chunk=5, n_max=50, stop_c=3),
        seed=1,
    )
    summary = run(spec, adapter, NDJudge(), store)
    assert summary["status"] == "complete"
    assert summary["n"] == 5 * len(problems)  # exactly one chunk per item
    assert summary["c"] == 5 * len(problems)
    assert summary["n_chunks"] == 1
    ref_lines_by_pid = {pr.pid: pr.ref.lines for pr in problems if pr.ref is not None}
    for item in store.items(summary["run_id"]):
        assert item.n == 5
        assert item.c == 5
        assert item.first_success == 1
        assert item.ok_written == [ref_lines_by_pid[item.pid]] * 5


def test_adaptive_runs_to_n_max_when_p_is_zero(tmp_path: Path) -> None:
    """p=0.0: every draw is corrupted, `stop_c` is never reached, so the item draws chunks until
    `n_max` is exhausted -- again an exact expected outcome."""
    problems = _problems()
    store = Store(tmp_path)
    _write_suite(store, problems)
    adapter = FakeAdapter(refs=_refs(problems), p=0.0)
    spec = RunSpec(
        suite="demo",
        version="v1",
        model="fake",
        k=999,
        adaptive=AdaptiveRule(chunk=5, n_max=50, stop_c=3),
        seed=2,
    )
    summary = run(spec, adapter, NDJudge(), store)
    assert summary["n"] == 50 * len(problems)
    assert summary["c"] == 0
    assert summary["n_chunks"] == 10
    for item in store.items(summary["run_id"]):
        assert item.n == 50
        assert item.c == 0
        assert item.first_success is None
        assert set(item.fail_kinds) == {"no_qed"}  # FakeAdapter's default corruption drops QED
        assert sum(item.fail_kinds.values()) == 50


def test_fixed_k_matches_independently_replayed_adapter_and_judge(tmp_path: Path) -> None:
    """Non-adaptive run: replay the exact same `SampleRequest` the runner builds (same
    `_item_seed`) against a *separate* `FakeAdapter`/`NDJudge` instance and check the recorded
    `ItemResult` matches -- an end-to-end check of the runner's wiring, not just of `FakeAdapter`
    or `NDJudge` in isolation."""
    from code.ndbench.core.types import SampleRequest

    problems = _problems()
    store = Store(tmp_path)
    _write_suite(store, problems)
    k = 40
    p = 0.4
    seed = 7
    adapter = FakeAdapter(refs=_refs(problems), p=p)
    spec = RunSpec(suite="demo", version="v1", model="fake", k=k, temperature=0.8, seed=seed)
    summary = run(spec, adapter, NDJudge(), store)
    assert summary["status"] == "complete"
    assert summary["n"] == k * len(problems)

    replay_adapter = FakeAdapter(refs=_refs(problems), p=p)
    replay_judge = NDJudge()
    by_pid = {item.pid: item for item in store.items(summary["run_id"])}
    for problem in problems:
        req = SampleRequest(
            pid=problem.pid,
            sequent=problem.sequent,
            n=k,
            temperature=0.8,
            max_new_tokens=max(1, 1024 - adapter.token_count(problem.sequent)),
            seed=_item_seed(seed, problem.pid, 0),
        )
        assert problem.ref is not None
        (attempts,) = replay_adapter.sample([req])
        verdicts = replay_judge.judge(problem.sequent, attempts, problem.ref.lines)
        expected_c = sum(v.ok for v in verdicts)
        assert by_pid[problem.pid].n == k
        assert by_pid[problem.pid].c == expected_c
        assert 0 < expected_c < k  # p=0.4 with k=40: sanity that this isn't a degenerate 0/all case


def test_same_seed_is_fully_deterministic_across_independent_stores(tmp_path: Path) -> None:
    problems = _problems()
    store1 = Store(tmp_path / "store1")
    store2 = Store(tmp_path / "store2")
    _write_suite(store1, problems)
    _write_suite(store2, problems)
    spec = RunSpec(suite="demo", version="v1", model="fake", k=64, seed=123)

    summary1 = run(spec, FakeAdapter(refs=_refs(problems), p=0.5), NDJudge(), store1)
    summary2 = run(spec, FakeAdapter(refs=_refs(problems), p=0.5), NDJudge(), store2)

    assert summary1["run_id"] == summary2["run_id"]
    assert summary1["n"] == summary2["n"]
    assert summary1["c"] == summary2["c"]

    items1 = {i.pid: i for i in store1.items(summary1["run_id"])}
    items2 = {i.pid: i for i in store2.items(summary2["run_id"])}
    assert items1.keys() == items2.keys()
    for pid in items1:
        assert items1[pid].n == items2[pid].n
        assert items1[pid].c == items2[pid].c
        assert items1[pid].ok_written == items2[pid].ok_written
        assert items1[pid].fail_kinds == items2[pid].fail_kinds


def test_different_seeds_usually_differ(tmp_path: Path) -> None:
    problems = _problems()
    store = Store(tmp_path)
    _write_suite(store, problems)
    spec_a = RunSpec(suite="demo", version="v1", model="fake", k=200, seed=1)
    spec_b = RunSpec(suite="demo", version="v1", model="fake", k=200, seed=2)
    summary_a = run(spec_a, FakeAdapter(refs=_refs(problems), p=0.5), NDJudge(), store)
    summary_b = run(spec_b, FakeAdapter(refs=_refs(problems), p=0.5), NDJudge(), store)
    assert summary_a["run_id"] != summary_b["run_id"]
    assert summary_a["c"] != summary_b["c"]  # astronomically unlikely to tie at n=400 draws


def test_missing_pid_raises_keyerror(tmp_path: Path) -> None:
    problems = _problems()
    store = Store(tmp_path)
    _write_suite(store, problems)
    adapter = FakeAdapter(refs={}, p=1.0)  # no refs at all
    spec = RunSpec(suite="demo", version="v1", model="fake", k=1, seed=0)
    with pytest.raises(KeyError):
        run(spec, adapter, NDJudge(), store)


# --------------------------------------------------------- incremental writing, crash, and resume


class _CrashingAdapter(FakeAdapter):
    """A `FakeAdapter` that raises on its `crash_on_call`-th `sample()` call (1-indexed), so a
    round's later `items_per_call` groups are never reached -- simulating the process being killed
    partway through a run (fix list item 2: "A run killed midway must leave its completed groups
    in the Store")."""

    def __init__(self, *args: Any, crash_on_call: int, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._crash_on_call = crash_on_call
        self.calls = 0

    def sample(self, reqs: Sequence[SampleRequest]) -> list[list[Attempt]]:
        self.calls += 1
        if self.calls == self._crash_on_call:
            raise RuntimeError("simulated crash")
        return super().sample(reqs)


class _RecordingAdapter(ModelAdapter):
    """Wraps another `ModelAdapter`, recording every `sample()` call's request list -- used to
    check *which* items a resumed run actually re-requests (it must skip already-recorded ones)."""

    def __init__(self, inner: ModelAdapter) -> None:
        self._inner = inner
        self.label = inner.label
        self.format_id = inner.format_id
        self.capabilities = inner.capabilities
        self.seen_reqs: list[list[SampleRequest]] = []

    def identity(self) -> dict[str, JSON]:
        return self._inner.identity()

    def token_count(self, seq: Sequent, body: str | None = None) -> int:
        return self._inner.token_count(seq, body)

    def sample(self, reqs: Sequence[SampleRequest]) -> list[list[Attempt]]:
        self.seen_reqs.append(list(reqs))
        return self._inner.sample(reqs)


def test_runner_writes_items_after_each_group_before_a_later_group_crashes(tmp_path: Path) -> None:
    """Fix list item 2: with `items_per_call=2` and 6 items, a round is 3 groups; a crash on the
    *second* `sample()` call must still leave the first group's 2 items in the `Store`, and no
    others (the crashing group's own partial work, and the untouched third group, must not
    appear)."""
    problems = _problems(6)
    store = Store(tmp_path)
    _write_suite(store, problems)
    judge = NDJudge()
    adapter = _CrashingAdapter(refs=_refs(problems), p=1.0, crash_on_call=2)
    spec = RunSpec(
        suite="demo", version="v1", model="fake", k=1, seed=0, options={"items_per_call": 2}
    )

    with pytest.raises(RuntimeError, match="simulated crash"):
        run(spec, adapter, judge, store)

    config_key = store.config_key(spec, adapter.identity(), judge.judge_id)
    run_id = store.latest("demo", "fake", config_key)
    assert run_id is not None
    saved_pids = {item.pid for item in store.items(run_id)}
    first_group_pids = set(sorted(p.pid for p in problems)[:2])
    assert saved_pids == first_group_pids
    # no "finish" event was ever appended (the exception propagated past it): the folded record's
    # status is still whatever `new_run` set at the start.
    (record,) = store.runs(run_id=run_id)
    assert record["status"] == "running"


def test_resume_skips_items_already_recorded_by_a_killed_prior_attempt(tmp_path: Path) -> None:
    """Fix list item 2's resume support: after a killed attempt has recorded some items, a fresh
    `run()` call against the same `Store`/config must not re-request samples for items whose
    recorded progress already satisfies `spec.k`, and must finish the remaining ones under the
    *same* `run_id`."""
    problems = _problems(4)
    store = Store(tmp_path)
    _write_suite(store, problems)
    judge = NDJudge()

    crashing = _CrashingAdapter(refs=_refs(problems), p=1.0, crash_on_call=2)
    spec = RunSpec(
        suite="demo", version="v1", model="fake", k=3, seed=0, options={"items_per_call": 2}
    )
    with pytest.raises(RuntimeError, match="simulated crash"):
        run(spec, crashing, judge, store)

    config_key = store.config_key(spec, crashing.identity(), judge.judge_id)
    run_id_after_crash = store.latest("demo", "fake", config_key)
    assert run_id_after_crash is not None
    done_after_crash = {item.pid: item for item in store.items(run_id_after_crash)}
    assert len(done_after_crash) == 2
    for item in done_after_crash.values():
        assert item.n == 3
        assert item.c == 3  # p=1.0: every draw matches the reference, so k=3 is fully satisfied

    fresh = FakeAdapter(refs=_refs(problems), p=1.0)
    assert fresh.identity() == crashing.identity()  # same config key as the killed attempt
    recording = _RecordingAdapter(fresh)
    summary = run(spec, recording, judge, store)

    assert summary["run_id"] == run_id_after_crash
    assert summary["status"] == "complete"
    all_items = {item.pid: item for item in store.items(summary["run_id"])}
    assert set(all_items) == {p.pid for p in problems}
    for item in all_items.values():
        assert item.n == 3
        assert item.c == 3

    # The resumed run must only ever have requested samples for the 2 items the killed attempt had
    # *not* already fully recorded.
    requested_pids = {r.pid for reqs in recording.seen_reqs for r in reqs}
    remaining_pids = {p.pid for p in problems} - set(done_after_crash)
    assert requested_pids == remaining_pids


def test_resume_is_a_noop_when_the_prior_run_already_completed(tmp_path: Path) -> None:
    """A `"complete"` prior run is not treated as resumable (module docstring's "Resume" section):
    `_resume_state` only fires for `"partial"`/`"running"`. This just pins that boundary."""
    from code.ndbench.runner.runner import _resume_state

    problems = _problems(2)
    store = Store(tmp_path)
    _write_suite(store, problems)
    judge = NDJudge()
    adapter = FakeAdapter(refs=_refs(problems), p=1.0)
    spec = RunSpec(suite="demo", version="v1", model="fake", k=2, seed=0)
    run(spec, adapter, judge, store)  # completes normally

    progress, next_chunk, resumed_run_id = _resume_state(
        store, spec, adapter.identity(), judge.judge_id, {p.pid for p in problems}
    )
    assert resumed_run_id is None
    assert progress == {}
    assert next_chunk == {}


# ------------------------------------------------------------- per-item max_new_tokens cap policy


def test_round_up_to_multiple() -> None:
    assert _round_up_to_multiple(0, 64) == 0
    assert _round_up_to_multiple(1, 64) == 64
    assert _round_up_to_multiple(64, 64) == 64
    assert _round_up_to_multiple(65, 64) == 128
    assert _round_up_to_multiple(160, 64) == 192


def test_estimate_ref_tokens_uses_the_reference_body_when_present() -> None:
    problems = _problems(1)
    problem = problems[0]
    assert problem.ref is not None
    adapter = FakeAdapter(refs=_refs(problems), p=1.0)
    prompt_tokens = adapter.token_count(problem.sequent)
    # `FakeAdapter.token_count` is a plain word count, so the expected ref-only count is the
    # reference body's own word count.
    expected = len(problem.ref.body.split())
    assert _estimate_ref_tokens(adapter, problem, prompt_tokens) == expected


def test_estimate_ref_tokens_falls_back_to_length_l_without_a_reference() -> None:
    problems = _problems(1)
    problem = replace(problems[0], ref=None)
    assert problem.length.L is not None
    adapter = FakeAdapter(refs={}, p=1.0)
    prompt_tokens = adapter.token_count(problem.sequent)
    expected = round(problem.length.L * MEAN_TOKENS_PER_REF_LINE)
    assert _estimate_ref_tokens(adapter, problem, prompt_tokens) == expected


def test_estimate_ref_tokens_none_without_a_reference_or_any_length_bound() -> None:
    problems = _problems(1)
    no_length = replace(problems[0].length, L=None, L_ub=None)
    problem = replace(problems[0], ref=None, length=no_length)
    adapter = FakeAdapter(refs={}, p=1.0)
    prompt_tokens = adapter.token_count(problem.sequent)
    assert _estimate_ref_tokens(adapter, problem, prompt_tokens) is None


def test_resolve_max_new_tokens_ref3x_matches_the_formula_and_stays_under_the_ceiling() -> None:
    problems = _problems(1)
    problem = problems[0]
    adapter = FakeAdapter(refs=_refs(problems), p=1.0)
    prompt_tokens = adapter.token_count(problem.sequent)
    assert problem.ref is not None
    ref_tokens = len(problem.ref.body.split())  # FakeAdapter.token_count is a word count
    expected_raw = max(160, 3 * ref_tokens + 32)
    expected = min(_round_up_to_multiple(expected_raw, 64), max(1, 1024 - prompt_tokens))
    result = _resolve_max_new_tokens("ref3x", adapter, problem, prompt_tokens)
    assert result == expected
    assert result <= max(1, 1024 - prompt_tokens)
    assert result % 64 == 0


def test_resolve_max_new_tokens_scales_with_a_long_reference_proof() -> None:
    problems = _problems(1)
    long_body = " ".join(f"tok{i}" for i in range(400)) + " QED"
    problem = replace(problems[0], ref=replace(problems[0].ref, body=long_body))  # type: ignore[union-attr]
    adapter = FakeAdapter(refs={problem.pid: long_body}, p=1.0)
    prompt_tokens = adapter.token_count(problem.sequent)
    ref_tokens = len(long_body.split())
    expected_raw = max(160, 3 * ref_tokens + 32)
    expected = min(_round_up_to_multiple(expected_raw, 64), max(1, 1024 - prompt_tokens))
    assert _resolve_max_new_tokens("ref3x", adapter, problem, prompt_tokens) == expected


def test_resolve_max_new_tokens_full_policy_is_the_old_flat_cap_and_never_exceeds_it() -> None:
    """`"full"` reproduces the pre-fix `1024 - prompt_tokens` cap exactly; this also pins that
    rounding a cap already at the ceiling never pushes it back over (round-then-clamp order)."""
    problems = _problems(1)
    problem = problems[0]
    adapter = FakeAdapter(refs=_refs(problems), p=1.0)
    prompt_tokens = adapter.token_count(problem.sequent)
    ceiling = max(1, 1024 - prompt_tokens)
    assert _resolve_max_new_tokens("full", adapter, problem, prompt_tokens) == ceiling


def test_run_uses_the_ref3x_default_and_an_explicit_override_still_wins(tmp_path: Path) -> None:
    problems = _problems(2)
    store = Store(tmp_path)
    _write_suite(store, problems)
    judge = NDJudge()

    fake = FakeAdapter(refs=_refs(problems), p=1.0)
    recording = _RecordingAdapter(fake)
    spec = RunSpec(suite="demo", version="v1", model="fake", k=1, seed=0)
    run(spec, recording, judge, store)
    (reqs,) = recording.seen_reqs
    by_pid = {p.pid: p for p in problems}
    for req in reqs:
        prompt_tokens = fake.token_count(by_pid[req.pid].sequent)
        expected = _resolve_max_new_tokens("ref3x", fake, by_pid[req.pid], prompt_tokens)
        assert req.max_new_tokens == expected

    # An explicit `RunSpec.max_new_tokens` always wins, regardless of policy.
    fake2 = FakeAdapter(refs=_refs(problems), p=1.0)
    recording2 = _RecordingAdapter(fake2)
    spec2 = RunSpec(suite="demo", version="v1", model="fake", k=1, seed=0, max_new_tokens=17)
    run(spec2, recording2, judge, store)
    (reqs2,) = recording2.seen_reqs
    assert all(r.max_new_tokens == 17 for r in reqs2)


def test_max_new_policy_is_part_of_the_config_key(tmp_path: Path) -> None:
    """Fix list: `"include the policy in the config key"` -- satisfied because `RunSpec.options`
    (which `max_new_policy` lives in) is hashed verbatim into `config_key`; this pins that an
    explicit `"full"` really does produce a different key from the (default, implicit) `"ref3x"`."""
    problems = _problems(2)
    store = Store(tmp_path)
    _write_suite(store, problems)
    judge = NDJudge()
    adapter = FakeAdapter(refs=_refs(problems), p=1.0)
    spec_ref3x = RunSpec(suite="demo", version="v1", model="fake", k=1, seed=0)
    spec_full = RunSpec(
        suite="demo", version="v1", model="fake", k=1, seed=0, options={"max_new_policy": "full"}
    )
    key_ref3x = store.config_key(spec_ref3x, adapter.identity(), judge.judge_id)
    key_full = store.config_key(spec_full, adapter.identity(), judge.judge_id)
    assert key_ref3x != key_full
