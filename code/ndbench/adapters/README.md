# Plugging in a new model

A `ModelAdapter` (`adapters/base.py`) is the one thing every ndbench benchmark, suite and analysis
talks to instead of a specific model's own API. Adding a new model -- a different architecture, a
different output format, an API-based model, a steered or interpolated variant of an existing one --
means writing one `ModelAdapter` subclass and registering it. Nothing else in the package needs to
change: `runner.run`, every `suites/*` builder, and every `analysis/*` view only ever see the
`ModelAdapter`/`Judge` interfaces.

## What to implement, per capability

`Capability` (`adapters/base.py`) is a flag enum; an adapter's `self.capabilities` should be the `|`
of exactly the optional methods it overrides, plus `SAMPLE` (and `PREFIX` if `sample` honours
`SampleRequest.prefix`) since those aren't separately overridable:

| Capability | Method(s) | Required? | Notes |
|---|---|---|---|
| `SAMPLE` | `sample(reqs) -> list[list[Attempt]]` | Always (abstract) | One `Attempt` list per `SampleRequest`, `req.n` attempts each. |
| — | `identity() -> dict[str, JSON]` | Always (abstract) | A fingerprint (checkpoint path, weights sha256, arch, tokenizer version, steering spec hash, ...) used as part of the `Store` run key: two adapters with equal `identity()` are the same model as far as the store is concerned. |
| — | `token_count(seq, body=None) -> int` | Always (abstract) | Used by the runner to size `max_new_tokens` when `RunSpec.max_new_tokens` is `None`. |
| `PREFIX` | (within `sample`) | If `sample` honours `req.prefix` | `Attempt.body` must include the prefix when one was requested; see `adapters.lean_seq`/`adapters.step_env`/`adapters.api_llm` for three different ways to implement this for three very different output formats. |
| `LOGPROB` | `logprob(pairs) -> list[LogProb]` | Optional | Marginalise over every equivalent encoding your format allows (e.g. absgpt's start-index shifts, `lean_seq`'s name shifts) -- see plan §B7 and `adapters/lean_seq.py`'s `logprob`. |
| `FINETUNE` | `learner(*, seed) -> Learner` | Optional | A `Learner` trains a *copy* of the model's weights (`nll`/`update`/`adapter`); used by `edl.py` (WP5). |
| `ACTIVATIONS` | `activations(pairs, layers) -> list[np.ndarray]` | Optional | Plan §A3 (interpretability hooks). |
| `STEER` | `with_steering(spec) -> ModelAdapter` | Optional | See "Steered models" below. |
| — | `close()` | Optional, default no-op | Release GPU memory etc. |

Every optional method's default (on the `ModelAdapter` base class) raises `NotSupported`; callers
(the runner, `analysis/*`) are expected to catch `NotSupported` per-adapter and treat it as "this
benchmark does not apply to this model", not as a run failure. An adapter that cannot yet fully
implement a capability it otherwise wants to expose (e.g. no scorer wired up yet) is free to
override the method and still raise `NotSupported` itself with a more specific message -- that is
exactly what `adapters.lean_seq.LeanSeqAdapter` does when constructed without a `scorer`/
`learner_factory` (see its module docstring).

If loading your model needs something environment-specific that might not be available (a binary, a
checkpoint format, a particular fork's code) -- raise `judge.base.NotAvailable` from a constructor
or classmethod (e.g. a `from_checkpoint`), not a bare exception. Callers should treat `NotAvailable`
the same way as `NotSupported`: skip the benchmark for that model, don't fail the run. See
`adapters.lean_seq.LeanSeqAdapter.from_checkpoint` for a worked example (it needs a model class that
is not vendored into this worktree yet, and says exactly what is needed and where to find it).

## Registering an adapter

`adapters/base.py`'s `ADAPTERS` dict is the lazy `"module:attr"` registry every caller (`cli.py`,
`runner`, tests) resolves an adapter name through, so importing `adapters.base` itself never has to
import a specific backend's (possibly GPU-only, possibly API-only) dependencies:

```python
ADAPTERS: dict[str, str] = {
    "fake": "code.ndbench.adapters.fake:FakeAdapter",
    "lean_seq": "code.ndbench.adapters.lean_seq:LeanSeqAdapter",
    ...
    "my_model": "code.ndbench.adapters.my_model:MyModelAdapter",
}
```

`cli.py`'s `_resolve`/`_resolve_registry` do `importlib.import_module(module_name)` then
`getattr(module, attr)`, so the entry only needs to be an importable dotted path plus the class
name -- your module can import torch, an HTTP client, or nothing at all. In this repo, only WP0 owns
`adapters/base.py`, so adding an entry there is a change someone with WP0 ownership makes (or the
main integrating agent, per the plan's work-package split); the mechanism above is what any future
integrator (inside or outside this particular multi-agent workflow) uses to wire a new adapter in.

## Steered or interpolated models are just another adapter

There is no special "steered model" concept anywhere else in ndbench. `ModelAdapter.with_steering`
(plan §A3) takes a `SteeringSpec` (a tuple of `SteeringTerm(layer, vector, alpha, where)`) and
returns a *new* `ModelAdapter` -- same class or a thin wrapper, doesn't matter -- whose `identity()`
includes a hash of that spec. Because `identity()` differs, the `Store` (and hence every suite/run/
analysis keyed off `model` + `identity`) treats the steered model as a distinct model from the
unsteered one, automatically, with zero special-casing anywhere else. The same applies to a weight-
interpolated model, a LoRA-adapted model, or anything else that is "the same architecture, different
weights or a different forward pass": wrap it as a `ModelAdapter` (or produce one from an existing
adapter's `with_steering`-style method) with a distinguishing `identity()`, and every benchmark,
builder and dashboard view already knows how to treat it as its own model.

## Worked minimal example

The smallest adapter that actually does something -- always answers with one fixed, hardcoded proof
regardless of the sequent asked (obviously not useful for real benchmarking, but it shows every
required piece with nothing extra):

```python
from collections.abc import Sequence

from code.ndbench.adapters.base import Capability, ModelAdapter
from code.ndbench.core.types import JSON, Attempt, SampleRequest, Sequent


class EchoAdapter(ModelAdapter):
    """Always answers with `self._body`, regardless of the sequent. A minimal worked example."""

    def __init__(self, body: str) -> None:
        self._body = body
        self.label = "echo"
        self.format_id = "echo_v1"
        self.capabilities = Capability.SAMPLE  # no PREFIX: `sample` ignores req.prefix

    def identity(self) -> dict[str, JSON]:
        return {"label": self.label, "body_sha256": hash(self._body)}

    def token_count(self, seq: Sequent, body: str | None = None) -> int:
        n = len(seq.prompt().split())
        return n + len((body or self._body).split())

    def sample(self, reqs: Sequence[SampleRequest]) -> list[list[Attempt]]:
        return [
            [
                Attempt(
                    pid=req.pid,
                    i=i,
                    body=self._body,
                    native=None,
                    n_tokens=self.token_count(req.sequent, self._body),
                    finished=True,
                )
                for i in range(req.n)
            ]
            for req in reqs
        ]
```

Register it (in `adapters/base.py`'s `ADAPTERS`, per above):

```python
"echo": "code.ndbench.adapters.echo:EchoAdapter",
```

and it works with `runner.run` and `judge.nd.NDJudge` exactly like every other adapter. For fuller,
real examples see `adapters/fake.py` (WP0's own deterministic test double -- the simplest *real*
adapter in the package), `adapters/absgpt.py` (a real GPU model, batched sampling over explicit
token-id lists), and this WP's own `adapters/lean_seq.py` (SAMPLE+PREFIX+LOGPROB+FINETUNE over an
injected `policy`/`scorer`/`learner_factory`, so a future Lean-format model plugs in without a new
adapter class), `adapters/step_env.py` (SAMPLE+PREFIX+STEP_ENV over an injected `env_factory`/
`policy`, with a local `NDLineEnv` scaffold), and `adapters/api_llm.py` (SAMPLE+PREFIX over an
injected `client` callable, with a `dry_run` mode and a cost cap).
