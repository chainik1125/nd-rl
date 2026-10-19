"""The trace validator (`trace_dsl.py`, imported unchanged from the experiment folder), the ASCII
surface the LLM writes, and the reference states and actions of a proof."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import Any

EXP_DIR = (
    Path(__file__).resolve().parents[2] / "experiments" / "current" / "thinking_traces_20261005"
)
PROMPT_MD = EXP_DIR / "production_prompt.md"
PILOT = EXP_DIR / "pilot_traces.jsonl"

# model token -> the ASCII spelling the pilot traces use (inverse of trace_dsl.ALIASES)
TO_ASCII = {
    "⊢": "|-",
    "¬": "~",
    "∧": "&",
    "∨": "v",
    "→": ">",
    "False": "F",
    "⟨": "<|",
    "⟩": "|>",
}


@cache
def dsl() -> Any:
    """The `trace_dsl` module, bound to the vendored fork modules (`code.thinking.dan`), never to a
    clone.  `validate(lean=True)` imports `lean_gate` lazily: call it inside `dan.bare()`."""
    from code.thinking import dan

    return dan.trace_dsl()


def ascii_toks(toks: list[str]) -> str:
    return " ".join(TO_ASCII.get(t, t) for t in toks)


def ascii_text(msg: str) -> str:
    """A validator message (model tokens, space separated) in the ASCII surface."""
    return "\n".join(
        " ".join(TO_ASCII.get(w, w) for w in line.split(" ")) for line in msg.split("\n")
    )


def step_kind(act: list[str]) -> str:
    """premise (restating hJ), exact, or decision."""
    if act and act[0] == "exact":
        return "exact"
    if len(act) >= 3 and act[-1] == ";" and act[-3] == ":=" and act[-2].startswith("h"):
        return "premise"
    return "decision"


@dataclass
class RefStep:
    state: list[str]
    action: list[str]
    kind: str
    env: Any = field(repr=False, default=None)

    @property
    def state_ascii(self) -> str:
        return ascii_toks(self.state)

    @property
    def action_ascii(self) -> str:
        return ascii_toks(self.action)


def reference_steps(prompt: str, proof: str, with_env: bool = False) -> list[RefStep]:
    """The states and canonical-name actions of a reference ND proof (`decompose(canon=True)`),
    with the environment *before* each action when `with_env`."""
    td = dsl()
    steps, _, _ = td.decompose(prompt, proof, canon=True)
    env = td.Env(prompt, canon=True, base=0, assign=False) if with_env else None
    out = []
    for s, a, _ in steps:
        out.append(
            RefStep(list(s), list(a), step_kind(list(a)), copy.deepcopy(env) if env else None)
        )
        if env is not None:
            ok, why = env.apply(list(a))
            if not ok:
                raise ValueError(f"reference replay failed: {why}")
    return out


def replay_steps(prompt: str, actions: list[str]) -> list[RefStep]:
    """States produced by ASCII actions (a trace's own actions) from the validator's start."""
    td = dsl()
    env = td.Env(prompt, canon=True, base=0, assign=False)
    out = []
    for a in actions:
        act = td.tokens(a)
        out.append(RefStep(env.state_tokens(), act, step_kind(act)))
        ok, why = env.apply(act)
        if not ok:
            raise ValueError(f"replay failed at {a!r}: {why}")
    return out


def pilot_records(path: Path = PILOT) -> dict[str, dict[str, Any]]:
    with open(path) as f:
        return {r["id"]: r for r in (json.loads(x) for x in f if x.strip())}
