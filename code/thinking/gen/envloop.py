"""Mode 3, environment in the loop (PROPOSAL.md section 3.2): interface only, no driver.

The LLM proves the theorem one step at a time through a client tool, `env_step`, that checks one
think block and one action against the current state with the same validator and returns the next
state or the error.  Per-step feedback is genuine, but every step is a Messages API round trip, so
it cannot go through the Batch API.  Driver sketch (not implemented):

    messages = [user: theorem + initial state]
    loop: response = messages.create(model, system, tools=[TOOL], tool_choice={"type": "auto"},
                                     output_config={"effort": ...}, messages)
          (forced tool_choice returns 400 on Claude Opus 5.5, so the prompt must ask for the tool)
          for each tool_use block: result = session.step(**input); append the assistant turn
          unchanged and a user turn with one tool_result per tool_use; stop when result["done"]
    the accepted steps are `session.steps`, a record for `trace_dsl.validate`.
"""

from __future__ import annotations

import copy
from typing import Any

from . import dsl

TOOL: dict[str, Any] = {
    "name": "env_step",
    "description": (
        "Check one think block and one action against the current proof state. Returns the next "
        "state, or the checker's error (the state is then unchanged)."
    ),
    "input_schema": {
        "type": "object",
        "properties": {"think": {"type": "string"}, "action": {"type": "string"}},
        "required": ["think", "action"],
        "additionalProperties": False,
    },
    "strict": True,
}


class EnvSession:
    def __init__(self, prompt: str):
        td = dsl.dsl()
        self.prompt = prompt
        self.env = td.Env(prompt, canon=True, base=0, assign=False)
        self.steps: list[dict[str, str]] = []

    def state(self) -> str:
        return dsl.ascii_toks(self.env.state_tokens())

    def step(self, think: str, action: str) -> dict[str, Any]:
        td = dsl.dsl()
        th = td.tokens(think)
        th = ["<think>", *th, "</think>"]
        act = td.tokens(action)
        try:
            bad = [t for t in th + act if t not in td.VOCAB]
            if bad:
                raise td.TraceError(f"tokens outside the vocabulary: {sorted(set(bad))}")
            soft = td.check_step(copy.deepcopy(self.env), th, act, {})
            if soft:
                raise td.TraceError(f"(strict) {soft[0]}")
        except (td.TraceError, td.ParseFail) as e:
            return {
                "ok": False,
                "error": dsl.ascii_text(str(e)),
                "state": self.state(),
                "done": False,
            }
        ok, why = self.env.apply(act)
        if not ok:
            return {"ok": False, "error": why, "state": self.state(), "done": False}
        self.steps.append({"think": think, "action": action})
        done = bool(self.env.done)
        return {"ok": True, "error": None, "state": None if done else self.state(), "done": done}

    def record(self, rid: str) -> dict[str, Any]:
        return {
            "id": rid,
            "source": "env-loop",
            "mode": "env",
            "deviates": None,
            "prompt": self.prompt,
            "steps": self.steps,
        }
