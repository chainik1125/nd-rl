"""`StepEnvAdapter`: a `ModelAdapter` over a step-based proof-construction environment (plan §1/§3
WP7), plus a local `NDLineEnv` to exercise the contract before Dan's proof-state environment
exists.

**The environment contract.** `env_factory(seq) -> env` builds a fresh environment for one
`Sequent`; `env.reset() -> state_text` returns the initial state; `env.step(action) ->
(state_text, done, info)` advances it. `action` is opaque to `StepEnvAdapter` -- it is whatever
`policy(state_text, rng) -> action` (or, during prefix replay, `prefix_action(line)`) produces, and
`env.step` alone decides what to do with it. The one thing every `step_env`-family environment
*must* do is put the finished proof, as an ND body, under `info["proof"]` on the step where
`done` becomes `True` (whether the proof is actually valid or not -- validity is
`judge.nd.NDJudge`'s job on the returned `Attempt`, not the environment's).
`StepEnvAdapter` treats a missing `info["proof"]` on a `done` step, or running past `max_steps`
without `done`, as an `UNPARSEABLE` attempt rather than a crash.

**Prefix replay.** `SampleRequest.prefix` (`PrefixSpec.body`, the first `j` ND lines, N1-based, no
`QED`) is replayed as `j` separate `step` calls before the policy ever runs, one call per line from
`core.proof.split_lines`. `prefix_action` converts one such raw ND line string into whatever
`action` type `env.step` expects; it defaults to the identity function, matching `NDLineEnv` below
(whose actions *are* raw ND line strings). An environment with a different action encoding (e.g.
tactic text, or an integer choice) needs its own `prefix_action` passed to `StepEnvAdapter`.

**`NDLineEnv`.** One ND proof line per step; the state is every line emitted so far (still in the
take-home token format) plus the sequent's prompt and goal. It performs no rule checking of its own
(that stays `NDJudge`'s job downstream) -- it exists purely so `StepEnvAdapter`'s loop, prefix
replay and `info["proof"]` contract can be tested end-to-end with a scripted policy before a real
proof-state environment exists (see `code/tests/ndbench/test_wp7_step_env.py`).

**Mapping onto Dan's future state environment.** The plan (§1) describes `lean_state`/`lean_staten`
as one model call per step over a rendered Lean *proof state* (rather than the flat ND line list
`NDLineEnv` uses): `reset()` would render the initial goal as Lean tactic-proof state text, and
`step(action)` would apply one tactic (or one hypothesis-introducing action) and re-render the new
state, exactly fitting this same `Env` protocol unchanged. Two things would differ from
`NDLineEnv`: (1) its `action` type is almost certainly Lean tactic text, not an ND line string, so
whoever wires it up as a `StepEnvAdapter` needs a matching `prefix_action` (translating a forced ND
prefix line into the equivalent tactic invocation -- essentially the single-line case of
`adapters.lean_seq`'s partial rendering); (2) `info["proof"]` on the final step would most naturally
be produced by decoding the accumulated Lean tactic trace back to an ND body (or, if the state env
tracks its own ND-line log internally the way `NDLineEnv` does, by exposing that log directly) --
either way, the contract `StepEnvAdapter` needs from it is unchanged: an ND body string under
`info["proof"]` exactly when `done` is `True`.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Protocol

from ..core.proof import split_lines
from ..core.types import JSON, Attempt, SampleRequest, Sequent
from .base import Capability, ModelAdapter

_UNPARSEABLE = "UNPARSEABLE"


class Env(Protocol):
    """What `env_factory(seq)` must return. See the module docstring."""

    def reset(self) -> str: ...
    def step(self, action: Any) -> tuple[str, bool, Mapping[str, Any]]: ...


#: `env_factory(seq) -> Env`: builds one fresh environment per sampled attempt.
EnvFactory = Callable[[Sequent], Env]
#: `policy(state_text, rng) -> action`.
PolicyFn = Callable[[str, random.Random], Any]
#: `prefix_action(line) -> action`: converts one raw ND line (as `core.proof.split_lines` yields
#: it) into the `action` a specific environment's `step` expects.
PrefixActionFn = Callable[[str], Any]


class StepEnvAdapter(ModelAdapter):
    """A `ModelAdapter` that runs `policy` against `env_factory(seq)` for `max_steps` steps.

    Does not use `SampleRequest.max_new_tokens` (there is no token budget in a step
    environment); `max_steps` (constructor-level, shared by every request) is the analogous cap.
    """

    def __init__(
        self,
        env_factory: EnvFactory,
        policy: PolicyFn,
        *,
        max_steps: int = 128,
        prefix_action: PrefixActionFn | None = None,
        label: str = "step_env",
        format_id: str = "step_env_v1",
    ) -> None:
        self._env_factory = env_factory
        self._policy = policy
        self._max_steps = max_steps
        self._prefix_action: PrefixActionFn = (
            prefix_action if prefix_action is not None else (lambda line: line)
        )
        self.label = label
        self.format_id = format_id
        self.capabilities = Capability.SAMPLE | Capability.PREFIX | Capability.STEP_ENV

    def identity(self) -> dict[str, JSON]:
        return {
            "label": self.label,
            "format_id": self.format_id,
            "max_steps": self._max_steps,
            "policy": repr(self._policy),
            "env_factory": repr(self._env_factory),
        }

    def token_count(self, seq: Sequent, body: str | None = None) -> int:
        n = len(seq.prompt().split())
        if body:
            n += len(body.split())
        return n

    def sample(self, reqs: Sequence[SampleRequest]) -> list[list[Attempt]]:
        return [[self._run_one(req, i) for i in range(req.n)] for req in reqs]

    def _run_one(self, req: SampleRequest, i: int) -> Attempt:
        rng = random.Random(f"{req.seed}:{i}")
        env = self._env_factory(req.sequent)
        state = env.reset()
        steps = 0
        done = False
        proof: str | None = None

        if req.prefix is not None:
            for line in split_lines(req.prefix.body):
                state, done, info = env.step(self._prefix_action(line))
                steps += 1
                if done:
                    proof = info.get("proof")
                    break

        while not done and steps < self._max_steps:
            action = self._policy(state, rng)
            state, done, info = env.step(action)
            steps += 1
            if done:
                proof = info.get("proof")

        if proof is not None:
            body = proof
        elif done:
            body = f"{_UNPARSEABLE} env reported done with no info['proof']"
        else:
            body = f"{_UNPARSEABLE} max_steps ({self._max_steps}) reached without done"

        return Attempt(
            pid=req.pid,
            i=i,
            body=body,
            native=state,
            n_tokens=self.token_count(req.sequent, body),
            finished=done,
        )


class NDLineEnv:
    """Local scaffold `step_env` environment: one ND proof line per step.

    Action: a complete raw ND line string in the take-home format, e.g. `"N3 Q : IMPE N1 N2 ;"`
    (the caller supplies its own line index; `NDLineEnv` neither renumbers nor rule-checks it), or
    the literal sentinel `"QED"` to end the episode. State: the sequent's prompt, every accepted
    line so far, and the goal, as one text blob (see `_state`). See the module docstring for why
    this environment does no proof checking of its own.
    """

    def __init__(self, seq: Sequent, *, max_lines: int = 64) -> None:
        self._seq = seq
        self._max_lines = max_lines
        self._lines: list[str] = []

    def _state(self) -> str:
        lines_str = " ".join(self._lines) if self._lines else "(none)"
        return f"{self._seq.prompt()} LINES {lines_str} GOAL {self._seq.conclusion}"

    def reset(self) -> str:
        self._lines = []
        return self._state()

    def step(self, action: str) -> tuple[str, bool, dict[str, str]]:
        if action == "QED":
            proof = f"{' '.join(self._lines)} QED" if self._lines else "QED"
            return self._state(), True, {"proof": proof}
        self._lines.append(action)
        done = len(self._lines) >= self._max_lines
        info: dict[str, str] = {}
        if done:
            # Ran out of budget without a "QED" action: report the (QED-less, certainly-failing)
            # lines anyway, so the caller always has *something* to hand to a Judge.
            info["proof"] = " ".join(self._lines)
        return self._state(), done, info
