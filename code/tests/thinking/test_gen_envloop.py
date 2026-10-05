"""Environment-in-the-loop interface (code/thinking/gen/envloop.py): a step tool over the same
validator, replaying a pilot trace and refusing a false step."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

from code.thinking.gen import dsl, envloop

FORK = Path(os.environ.get("ND_TAKEHOME", "/scratch/cowifornia/nd-rl/think/nd-takehome"))
pytestmark = pytest.mark.skipif(not (FORK / "state_env.py").exists(), reason="needs Dan's fork")


def test_session_replays_a_pilot_trace() -> None:
    rec = dsl.pilot_records()["tb_58b532c9"]
    s = envloop.EnvSession(rec["prompt"])
    assert s.state().startswith("<st> |- ")
    out: dict[str, Any] = {}
    for i, st in enumerate(rec["steps"]):
        out = s.step(st["think"], st["action"])
        assert out["ok"], (i, out["error"])
    assert out["done"] and out["state"] is None
    assert dsl.dsl().validate(s.record("x"), lean=False, strict=True)["ok"]


def test_session_rejects_a_false_step_and_keeps_the_state() -> None:
    rec = dsl.pilot_records()["k12_37120"]
    s = envloop.EnvSession(rec["prompt"])
    before = s.state()
    out = s.step("n1 : S", rec["steps"][0]["action"])
    assert not out["ok"] and "false look" in out["error"] and out["state"] == before
    out = s.step("", "have n2 : ( ~ S ) := h1 ;")
    assert not out["ok"]
    assert s.step(rec["steps"][0]["think"], rec["steps"][0]["action"])["ok"]


def test_tool_definition_is_strict() -> None:
    t = envloop.TOOL
    assert t["strict"] is True and t["input_schema"]["additionalProperties"] is False
    assert set(t["input_schema"]["required"]) == {"think", "action"}
