"""`ModelAdapter`/`Learner` interfaces every model backend (absgpt, lean_seq, step_env, api_llm,
fake) implements against (plan §1), plus the interpretability hooks from plan §A3 (`ACTIVATIONS`/
`STEER`, `SteeringTerm`/`SteeringSpec`).

A `ModelAdapter` only has to implement `identity`, `token_count` and `sample`; every other method
(`logprob`, `learner`, `activations`, `with_steering`) defaults to raising `NotSupported`, and a
concrete adapter overrides only the ones its `capabilities` flag advertises. This lets the runner
and suites call any method on any adapter uniformly and treat `NotSupported` as "this benchmark
does not apply to this model" rather than a crash.
"""

from __future__ import annotations

import abc
import enum
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from ..core.types import JSON, Attempt, LogProb, SampleRequest, Sequent

if TYPE_CHECKING:
    import numpy as np


class NotSupported(Exception):
    """Raised by a `ModelAdapter`/`Learner` method the adapter does not implement.

    Callers (runner, analysis) should generally catch this per-adapter and skip the benchmark for
    that model rather than treating it as a run failure.
    """


class Capability(enum.Flag):
    """What a `ModelAdapter` can do. `sample` is the only method every adapter must support in
    practice (`ModelAdapter.sample` is abstract); the flag still advertises it so callers can
    check `Capability.SAMPLE in adapter.capabilities` uniformly alongside the optional ones."""

    SAMPLE = 1
    PREFIX = 2
    LOGPROB = 4
    FINETUNE = 8
    STEP_ENV = 16
    ACTIVATIONS = 32  # plan §A3: residual-stream extraction
    STEER = 64  # plan §A3: activation-steering via with_steering


@dataclass(frozen=True)
class SteeringTerm:
    """One steering intervention: add `alpha * vector` to the residual stream after block
    `layer`, at the token positions selected by `where` (plan §A3).

    `vector` is either the literal coefficients, or a string path to a `.npy` file holding them
    (kept out-of-line for large/shared steering vectors); resolving the path is the implementing
    adapter's job (e.g. `AbsGptAdapter.with_steering`).
    """

    layer: int
    vector: tuple[float, ...] | str
    alpha: float
    where: str  # "all" | "prompt" | "proof"


#: An ordered set of steering interventions, applied together. A steered adapter's `identity()`
#: includes a hash of its `SteeringSpec`, so a steered model is just another model in the `Store`.
SteeringSpec = tuple[SteeringTerm, ...]


class ModelAdapter(abc.ABC):
    """A model backend: something that can `sample` `SampleRequest`s into `Attempt`s.

    `label`/`format_id`/`capabilities` are plain instance attributes (not properties) so a
    subclass can just set them in `__init__`; `capabilities` should reflect exactly which of the
    optional methods below are overridden (plus `SAMPLE`, and `PREFIX` if `sample` honours
    `SampleRequest.prefix`).
    """

    label: str
    format_id: str
    capabilities: Capability

    @abc.abstractmethod
    def identity(self) -> dict[str, JSON]:
        """A JSON-able fingerprint of this exact model (checkpoint path, weights sha256,
        architecture, tokenizer version, and — for a steered adapter — a hash of its
        `SteeringSpec`). Used as part of the `Store` run key, so two adapters with the same
        `identity()` are treated as the same model."""
        ...

    @abc.abstractmethod
    def token_count(self, seq: Sequent, body: str | None = None) -> int:
        """Token length of `seq`'s prompt, plus `body`'s if given (this adapter's tokenizer)."""
        ...

    @abc.abstractmethod
    def sample(self, reqs: Sequence[SampleRequest]) -> list[list[Attempt]]:
        """Draw `req.n` attempts for each request, returned in the same order as `reqs`."""
        ...

    def logprob(self, pairs: Sequence[tuple[Sequent, str]]) -> list[LogProb]:
        """Log-probability of each `(sequent, body)` pair under this model, marginalised over
        every allowed start-index shift encoding (`LOGPROB` capability)."""
        raise NotSupported(f"{type(self).__name__} does not implement logprob")

    def learner(self, *, seed: int) -> Learner:
        """An online `Learner` that trains a copy of this model's weights (`FINETUNE`
        capability)."""
        raise NotSupported(f"{type(self).__name__} does not implement learner")

    def activations(
        self, pairs: Sequence[tuple[Sequent, str]], layers: Sequence[int]
    ) -> list[np.ndarray]:
        """Residual-stream activations after each of `layers`, per token, for each `(sequent,
        body)` pair (`ACTIVATIONS` capability; plan §A3)."""
        raise NotSupported(f"{type(self).__name__} does not implement activations")

    def with_steering(self, spec: SteeringSpec) -> ModelAdapter:
        """A new adapter that applies `spec` as residual-stream additions via forward hooks
        (`STEER` capability; plan §A3). The returned adapter's `identity()` must differ from this
        one's (it includes a hash of `spec`), so every benchmark runs on the steered model
        unchanged, as just another model in the `Store`."""
        raise NotSupported(f"{type(self).__name__} does not implement with_steering")

    def close(self) -> None:
        """Release any resources (e.g. GPU memory). Optional; the default is a no-op."""
        return None


class Learner(Protocol):
    """An online learner over a `ModelAdapter`'s weights (used by `edl.py`, plan §B7)."""

    def nll(self, pairs: Sequence[tuple[Sequent, str]]) -> list[LogProb]:
        """Per-example negative log-likelihood at the learner's current weights."""
        ...

    def update(self, pairs: Sequence[tuple[Sequent, str]], lr: float) -> float:
        """One optimiser step on `pairs` at learning rate `lr`; returns the pre-update loss."""
        ...

    def adapter(self) -> ModelAdapter:
        """A `ModelAdapter` that samples with this learner's current weights."""
        ...


#: Lazy `"module:attr"` registry, resolved by callers (e.g. `cli.py`) via `importlib` so that
#: importing `adapters.base` never imports every backend's (possibly GPU-only) dependencies.
#: `absgpt`/`lean_seq`/`step_env`/`api_llm` are implemented by later work packages (WP1/WP7); only
#: `fake` is implemented in WP0.
ADAPTERS: dict[str, str] = {
    "absgpt": "code.ndbench.adapters.absgpt:AbsGptAdapter",
    "fake": "code.ndbench.adapters.fake:FakeAdapter",
    "lean_seq": "code.ndbench.adapters.lean_seq:LeanSeqAdapter",
    "step_env": "code.ndbench.adapters.step_env:StepEnvAdapter",
    "api_llm": "code.ndbench.adapters.api_llm:ApiLlmAdapter",
}
