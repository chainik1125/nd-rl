"""WP7 acceptance tests: `StepEnvAdapter` and `NDLineEnv` (plan §3 WP7 row).

Covers: `NDLineEnv` plus a scripted policy reproducing reference proofs (the plan's acceptance
criterion), prefix replay, the `max_steps` cap, the `info["proof"]`-missing fallback, and the
declared capability flags.
"""

from __future__ import annotations

import random

from code.ndbench.adapters.base import Capability
from code.ndbench.adapters.step_env import NDLineEnv, StepEnvAdapter
from code.ndbench.core.proof import split_lines, take_prefix
from code.ndbench.core.types import PrefixSpec, SampleRequest, Sequent

from ._fixtures import load_valid_proofs


def _scripted_policy(ref_lines: list[str]):
    """Emits `ref_lines` in order, then the `"QED"` sentinel -- reading "how many lines have
    already been emitted" straight off `NDLineEnv`'s own state text (every emitted line ends in
    exactly one `";"`, and nothing else in the state does), so the same policy object works
    correctly no matter how many independent `NDLineEnv` episodes it is reused across."""

    def policy(state: str, rng: random.Random) -> str:
        del rng
        k = state.count(";")
        return ref_lines[k] if k < len(ref_lines) else "QED"

    return policy


def test_ndlineenv_state_and_step_contract() -> None:
    seq = Sequent(premises=("P",), conclusion="P")
    env = NDLineEnv(seq)
    state0 = env.reset()
    assert "LINES (none)" in state0
    assert "GOAL P" in state0

    state1, done1, info1 = env.step("N1 P : PR ;")
    assert done1 is False
    assert info1 == {}
    assert "LINES N1 P : PR ;" in state1

    state2, done2, info2 = env.step("QED")
    assert done2 is True
    assert info2["proof"] == "N1 P : PR ; QED"


def test_ndlineenv_max_lines_without_qed_reports_proof_anyway() -> None:
    seq = Sequent(premises=(), conclusion="P")
    env = NDLineEnv(seq, max_lines=2)
    env.reset()
    env.step("N1 P : PR ;")
    state, done, info = env.step("N2 P : R N1 ;")
    assert done is True
    assert info["proof"] == "N1 P : PR ; N2 P : R N1 ;"  # no QED: certainly a failing proof


def test_capability_flags() -> None:
    adapter = StepEnvAdapter(env_factory=NDLineEnv, policy=lambda state, rng: "QED")
    assert adapter.capabilities == (Capability.SAMPLE | Capability.PREFIX | Capability.STEP_ENV)


def test_ndlineenv_plus_scripted_policy_reproduces_reference_proofs() -> None:
    """The plan's acceptance test: `NDLineEnv` plus a scripted policy reproduces reference
    proofs, run through the full `StepEnvAdapter.sample` contract."""
    for proof in load_valid_proofs()[:5]:
        seq = Sequent.from_prompt(proof["prompt"])
        ref_lines = split_lines(proof["proof"])
        adapter = StepEnvAdapter(env_factory=NDLineEnv, policy=_scripted_policy(ref_lines))
        req = SampleRequest(pid="p", sequent=seq, n=2, temperature=0.8, max_new_tokens=64, seed=0)
        (attempts,) = adapter.sample([req])
        assert len(attempts) == 2
        for a in attempts:
            assert a.body == proof["proof"], proof["name"]
            assert a.finished is True


def test_prefix_is_replayed_then_policy_finishes() -> None:
    proof = next(p for p in load_valid_proofs() if p["n_lines"] >= 4)
    seq = Sequent.from_prompt(proof["prompt"])
    ref_lines = split_lines(proof["proof"])
    j = 2
    prefix_body = take_prefix(proof["proof"], j)
    adapter = StepEnvAdapter(env_factory=NDLineEnv, policy=_scripted_policy(ref_lines))
    req = SampleRequest(
        pid="p",
        sequent=seq,
        n=1,
        temperature=0.8,
        max_new_tokens=64,
        seed=0,
        prefix=PrefixSpec(j=j, body=prefix_body),
    )
    (attempts,) = adapter.sample([req])
    assert attempts[0].body == proof["proof"]


def test_max_steps_cap_yields_unparseable_when_env_never_reports_proof() -> None:
    class _NeverDoneEnv:
        def reset(self) -> str:
            return "state"

        def step(self, action: str) -> tuple[str, bool, dict[str, str]]:
            return "state", False, {}

    adapter = StepEnvAdapter(
        env_factory=lambda seq: _NeverDoneEnv(), policy=lambda state, rng: "anything", max_steps=3
    )
    seq = Sequent(premises=(), conclusion="P")
    req = SampleRequest(pid="p", sequent=seq, n=1, temperature=0.8, max_new_tokens=64, seed=0)
    (attempts,) = adapter.sample([req])
    assert attempts[0].body.startswith("UNPARSEABLE")
    assert attempts[0].finished is False


def test_done_without_proof_info_is_unparseable_not_a_crash() -> None:
    class _DoneNoProofEnv:
        def reset(self) -> str:
            return "state"

        def step(self, action: str) -> tuple[str, bool, dict[str, str]]:
            return "state", True, {}  # done, but no "proof" key

    adapter = StepEnvAdapter(env_factory=lambda seq: _DoneNoProofEnv(), policy=lambda s, r: "x")
    seq = Sequent(premises=(), conclusion="P")
    req = SampleRequest(pid="p", sequent=seq, n=1, temperature=0.8, max_new_tokens=64, seed=0)
    (attempts,) = adapter.sample([req])
    assert attempts[0].body.startswith("UNPARSEABLE")
    assert attempts[0].finished is True


def test_deterministic_across_independent_adapters() -> None:
    proof = load_valid_proofs()[0]
    seq = Sequent.from_prompt(proof["prompt"])
    ref_lines = split_lines(proof["proof"])
    req = SampleRequest(pid="p", sequent=seq, n=4, temperature=0.8, max_new_tokens=64, seed=99)

    def flaky_policy(state: str, rng: random.Random):
        k = state.count(";")
        if k < len(ref_lines) and rng.random() < 0.5:
            return ref_lines[k]
        return "QED" if k >= len(ref_lines) else ref_lines[k]

    a1 = StepEnvAdapter(env_factory=NDLineEnv, policy=flaky_policy).sample([req])
    a2 = StepEnvAdapter(env_factory=NDLineEnv, policy=flaky_policy).sample([req])
    assert [a.body for a in a1[0]] == [a.body for a in a2[0]]
