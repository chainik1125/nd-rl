"""WP7 acceptance test: `runner.run` (with `judge.nd.NDJudge`) works with `StepEnvAdapter`,
`LeanSeqAdapter` (scripted policy) and `ApiLlmAdapter` (dry run) end to end, each against a small
suite of fixture theorems held in a `Store` (plan §3 WP7 row: "runner works with each on fakes").
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from pathlib import Path

from code.ndbench.adapters.api_llm import ApiLlmAdapter
from code.ndbench.adapters.lean_seq import LeanSeqAdapter
from code.ndbench.adapters.step_env import NDLineEnv, StepEnvAdapter
from code.ndbench.core.io import git_commit_sha, sha256_str, utc_now_iso
from code.ndbench.core.proof import pruned_length, split_lines
from code.ndbench.core.types import LengthLabel, Problem, ReferenceProof, Sequent, SuiteManifest
from code.ndbench.judge.nd import NDJudge
from code.ndbench.runner.runner import RunSpec, run
from code.ndbench.store import Store
from code.ndbench.vendor.lean_seq.lean_seq import LeanSeqTokenizer

from ._fixtures import load_valid_proofs


def _small_suite() -> list[Problem]:
    proofs = [p for p in load_valid_proofs() if p["n_lines"] <= 4][:2]
    assert len(proofs) == 2
    out = []
    for i, p in enumerate(proofs):
        seq = Sequent.from_prompt(p["prompt"])
        out.append(
            Problem(
                pid=f"item{i}",
                suite="wp7demo",
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


def _write_suite(store: Store, problems: list[Problem]) -> None:
    manifest = SuiteManifest(
        suite="wp7demo",
        version="v1",
        builder="test",
        config={},
        seed=0,
        created_utc=utc_now_iso(),
        git_commit=git_commit_sha(),
        n_items=len(problems),
        items_sha256=sha256_str("x"),
        excl_sources={},
        stats={},
    )
    store.write_suite(problems, manifest)


def test_runner_with_step_env_adapter(tmp_path: Path) -> None:
    problems = _small_suite()
    store = Store(tmp_path)
    _write_suite(store, problems)
    ref_lines_by_prompt = {p.sequent.prompt(): split_lines(p.ref.body) for p in problems if p.ref}

    def policy(state: str, rng: random.Random) -> str:
        del rng
        prompt = state.split(" LINES ")[0]
        ref_lines = ref_lines_by_prompt[prompt]
        k = state.count(";")
        return ref_lines[k] if k < len(ref_lines) else "QED"

    adapter = StepEnvAdapter(env_factory=NDLineEnv, policy=policy)
    spec = RunSpec(suite="wp7demo", version="v1", model="step_env", k=3, seed=0)
    summary = run(spec, adapter, NDJudge(), store)
    assert summary["status"] == "complete"
    assert summary["n"] == 3 * len(problems)
    assert summary["c"] == 3 * len(problems)  # the scripted policy always reproduces the reference


def test_runner_with_lean_seq_adapter(tmp_path: Path) -> None:
    problems = _small_suite()
    store = Store(tmp_path)
    _write_suite(store, problems)
    tok = LeanSeqTokenizer()
    full_ids_by_prompt_ids = {
        tuple(tok.encode_prompt(p.sequent.prompt())): tok.encode_proof(p.ref.body)
        for p in problems
        if p.ref
    }

    def policy(
        prompt_ids: Sequence[int],
        forced_ids: Sequence[int],
        *,
        n: int,
        temperature: float,
        max_new_tokens: int,
        seed: int,
    ) -> list[list[int]]:
        del temperature, max_new_tokens, seed
        full_ids = full_ids_by_prompt_ids[tuple(prompt_ids)]
        cont = full_ids[len(forced_ids) :]
        return [list(cont) for _ in range(n)]

    adapter = LeanSeqAdapter(policy=policy)
    spec = RunSpec(suite="wp7demo", version="v1", model="lean_seq", k=3, seed=0)
    summary = run(spec, adapter, NDJudge(), store)
    assert summary["status"] == "complete"
    assert summary["n"] == 3 * len(problems)
    assert summary["c"] == 3 * len(problems)


def test_runner_with_api_llm_adapter_dry_run(tmp_path: Path) -> None:
    problems = _small_suite()
    store = Store(tmp_path)
    _write_suite(store, problems)
    ref_by_prompt = {p.sequent.prompt(): p.ref.body for p in problems if p.ref}

    def canned(seq: Sequent, i: int) -> str:
        del i
        return f"Here is the proof:\n{ref_by_prompt[seq.prompt()]}\nQED explained."

    adapter = ApiLlmAdapter(dry_run=True, dry_run_responses=canned)
    spec = RunSpec(suite="wp7demo", version="v1", model="api_llm", k=2, seed=0)
    summary = run(spec, adapter, NDJudge(), store)
    assert summary["status"] == "complete"
    assert summary["n"] == 2 * len(problems)
    assert summary["c"] == 2 * len(problems)
