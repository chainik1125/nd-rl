"""Shared synthetic-store builders for the WP6 (`analysis/*`, `dash/*`) test files.

Not a test module itself (no `test_*` functions) -- named `test_wp6_*.py` only so pytest's
`code/tests/ndbench/test_wp6_*.py` glob (the WP6 test-running command in the plan) picks up the
files that import from it without a separate conftest fixture module (WP0 owns `conftest.py`).
"""

from __future__ import annotations

from code.ndbench.core.io import git_commit_sha, sha256_str, utc_now_iso
from code.ndbench.core.types import (
    JSON,
    ItemResult,
    LengthLabel,
    PrefixSpec,
    Problem,
    ReferenceProof,
    Sequent,
    SuiteManifest,
)
from code.ndbench.runner.runner import RunSpec
from code.ndbench.store import Store


def manifest(suite: str, version: str, n_items: int) -> SuiteManifest:
    return SuiteManifest(
        suite=suite,
        version=version,
        builder="test",
        config={},
        seed=0,
        created_utc=utc_now_iso(),
        git_commit=git_commit_sha(),
        n_items=n_items,
        items_sha256=sha256_str(f"{suite}-{version}-{n_items}"),
        excl_sources={},
        stats={},
    )


def make_problem(
    pid: str,
    suite: str,
    version: str,
    *,
    length: int | None = None,
    length_ub: int | None = None,
    length_lb: int | None = None,
    method: str = "minlen_exact",
    tags: dict[str, JSON] | None = None,
    parts: tuple[str, ...] = (),
    ref_lines: int | None = None,
    prefix_j: int | None = None,
) -> Problem:
    ref = (
        ReferenceProof(body="N1 P PR ; QED", lines=ref_lines, pruned=ref_lines, source="witness")
        if ref_lines is not None
        else None
    )
    prefix = PrefixSpec(j=prefix_j, body="N1 P PR ;") if prefix_j is not None else None
    return Problem(
        pid=pid,
        suite=suite,
        version=version,
        sequent=Sequent(premises=("P",), conclusion="P"),
        length=LengthLabel(L=length, L_ub=length_ub, L_lb=length_lb, method=method, bound=None),
        ref=ref,
        tags=tags or {},
        parts=parts,
        prefix=prefix,
    )


def make_item(
    run_id: str,
    pid: str,
    n: int,
    c: int,
    *,
    chunk: int = 0,
    ok_written: list[int] | None = None,
    ok_tokens: list[int] | None = None,
    fail_lines: list[int] | None = None,
    long_enough: int = 0,
    tokens_mean: float = 50.0,
    fail_kinds: dict[str, int] | None = None,
    first_success: int | None = None,
) -> ItemResult:
    return ItemResult(
        run_id=run_id,
        pid=pid,
        chunk=chunk,
        n=n,
        c=c,
        first_success=first_success,
        ok_written=ok_written or [],
        ok_pruned=[],
        ok_tokens=ok_tokens or [],
        ok_starts=[],
        fail_kinds=fail_kinds or {},
        fail_lines=fail_lines or [],
        long_enough=long_enough,
        tokens_mean=tokens_mean,
        hit_max=0,
    )


def start_run(
    store: Store,
    *,
    suite: str,
    version: str,
    model: str,
    k: int,
    seed: int = 0,
    status: str = "complete",
) -> str:
    """Registers and finishes a run (plan §1's `new_run`/`finish_run`), returning its `run_id`.
    `status="legacy"` is how a legacy-imported run is represented (`Store.finish_run` takes any
    status string; `import-legacy`, plan §3 WP2, is expected to do the same)."""
    spec = RunSpec(suite=suite, version=version, model=model, k=k, seed=seed)
    run_id = store.new_run(spec, identity={"label": model}, judge_id="nd_v1")
    store.finish_run(run_id, status, 0.0)
    return run_id
