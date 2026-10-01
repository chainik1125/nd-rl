"""`adapters.base`: `Capability` flags (incl. §A3's `ACTIVATIONS`/`STEER`), the default
`NotSupported` methods, and `FakeAdapter`'s own contract."""

from __future__ import annotations

import pytest

from code.ndbench.adapters.base import (
    ADAPTERS,
    Capability,
    ModelAdapter,
    NotSupported,
    SteeringTerm,
)
from code.ndbench.adapters.fake import FakeAdapter, drop_qed
from code.ndbench.core.types import JSON, Attempt, SampleRequest, Sequent

from ._fixtures import load_valid_proofs


def test_capability_flags_values() -> None:
    assert Capability.SAMPLE.value == 1
    assert Capability.PREFIX.value == 2
    assert Capability.LOGPROB.value == 4
    assert Capability.FINETUNE.value == 8
    assert Capability.STEP_ENV.value == 16
    assert Capability.ACTIVATIONS.value == 32
    assert Capability.STEER.value == 64
    combo = Capability.SAMPLE | Capability.PREFIX
    assert Capability.SAMPLE in combo
    assert Capability.LOGPROB not in combo


def test_steering_term_and_spec() -> None:
    term = SteeringTerm(layer=3, vector=(0.1, 0.2, 0.3), alpha=0.5, where="proof")
    spec = (term,)
    assert spec[0].layer == 3
    assert spec[0].where == "proof"
    term_path = SteeringTerm(layer=1, vector="/tmp/vec.npy", alpha=1.0, where="all")
    assert isinstance(term_path.vector, str)


class _MinimalAdapter(ModelAdapter):
    """The smallest possible concrete adapter: only the abstract methods implemented, to check
    the default `NotSupported` methods on the base class."""

    def __init__(self) -> None:
        self.label = "minimal"
        self.format_id = "nd_v1"
        self.capabilities = Capability.SAMPLE

    def identity(self) -> dict[str, JSON]:
        return {"label": self.label}

    def token_count(self, seq: Sequent, body: str | None = None) -> int:
        return len(seq.prompt().split())

    def sample(self, reqs):  # type: ignore[override]
        return [[] for _ in reqs]


def test_default_methods_raise_not_supported() -> None:
    adapter = _MinimalAdapter()
    with pytest.raises(NotSupported):
        adapter.logprob([])
    with pytest.raises(NotSupported):
        adapter.learner(seed=0)
    with pytest.raises(NotSupported):
        adapter.activations([], [])
    with pytest.raises(NotSupported):
        adapter.with_steering(())
    adapter.close()  # no-op, must not raise


def test_adapters_registry_has_every_planned_backend() -> None:
    assert set(ADAPTERS) == {"absgpt", "danstate", "fake", "lean_seq", "step_env", "api_llm"}
    module_name, _, attr = ADAPTERS["fake"].partition(":")
    assert module_name == "code.ndbench.adapters.fake"
    assert attr == "FakeAdapter"


def test_fake_adapter_returns_reference_or_corrupted() -> None:
    proof = load_valid_proofs()[0]
    seq = Sequent.from_prompt(proof["prompt"])
    refs = {"p": proof["proof"]}
    adapter = FakeAdapter(refs=refs, p=1.0)
    assert adapter.capabilities == Capability.SAMPLE
    req = SampleRequest(pid="p", sequent=seq, n=20, temperature=0.8, max_new_tokens=50, seed=0)
    (attempts,) = adapter.sample([req])
    assert len(attempts) == 20
    assert all(a.body == proof["proof"] for a in attempts)

    adapter0 = FakeAdapter(refs=refs, p=0.0)
    (attempts0,) = adapter0.sample([req])
    assert all(a.body == drop_qed(proof["proof"]) for a in attempts0)
    assert all("QED" not in a.body.split() for a in attempts0)


def test_fake_adapter_is_deterministic() -> None:
    proof = load_valid_proofs()[1]
    seq = Sequent.from_prompt(proof["prompt"])
    refs = {"p": proof["proof"]}
    req = SampleRequest(pid="p", sequent=seq, n=200, temperature=0.8, max_new_tokens=50, seed=42)
    a1 = FakeAdapter(refs=refs, p=0.5).sample([req])
    a2 = FakeAdapter(refs=refs, p=0.5).sample([req])
    bodies1 = [a.body for a in a1[0]]
    bodies2 = [a.body for a in a2[0]]
    assert bodies1 == bodies2
    n_ref = sum(b == proof["proof"] for b in bodies1)
    assert 0 < n_ref < 200  # genuinely mixed, not degenerate


def test_fake_adapter_unknown_pid_raises() -> None:
    seq = Sequent(premises=(), conclusion="P")
    adapter = FakeAdapter(refs={}, p=1.0)
    req = SampleRequest(pid="missing", sequent=seq, n=1, temperature=0.8, max_new_tokens=10, seed=0)
    with pytest.raises(KeyError):
        adapter.sample([req])


def test_fake_adapter_rejects_invalid_p() -> None:
    with pytest.raises(ValueError):
        FakeAdapter(refs={}, p=1.5)
    with pytest.raises(ValueError):
        FakeAdapter(refs={}, p=-0.1)


def test_attempt_is_frozen() -> None:
    a = Attempt(pid="p", i=0, body="x", native=None, n_tokens=1, finished=True)
    with pytest.raises(AttributeError):
        a.body = "y"  # type: ignore[misc]
