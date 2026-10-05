"""Request shape and cost estimate for LLM trace generation, checked against the API details
verified on platform.claude.com on 2026-10-05 (code/thinking/gen/README.md)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

from code.thinking.gen import dsl, fakes, models
from code.thinking.gen import requests as rq
from code.thinking.gen import select as sl

FORK = Path(os.environ.get("ND_TAKEHOME", "/scratch/cowifornia/nd-rl/think/nd-takehome"))
pytestmark = pytest.mark.skipif(
    not (FORK / "state_env.py").exists() or not sl.K12.exists(), reason="needs Dan's fork and K12"
)
UNSUPPORTED_SCHEMA_KEYS = {
    "minLength",
    "maxLength",
    "maxItems",
    "minimum",
    "maximum",
    "multipleOf",
    "pattern",
}


def objects(schema: Any) -> list[dict[str, Any]]:
    out = []
    if isinstance(schema, dict):
        if schema.get("type") == "object":
            out.append(schema)
        for v in schema.values():
            out += objects(v)
    elif isinstance(schema, list):
        for v in schema:
            out += objects(v)
    return out


def keys(schema: Any) -> set[str]:
    if isinstance(schema, dict):
        return set(schema) | {k for v in schema.values() for k in keys(v)}
    if isinstance(schema, list):
        return {k for v in schema for k in keys(v)}
    return set()


@pytest.fixture(scope="module")
def built(tmp_path_factory: pytest.TempPathFactory) -> dict[str, list[dict[str, Any]]]:
    d = tmp_path_factory.mktemp("build")
    rq.write_jsonl(d / "selection.jsonl", fakes.pilot_selection(sl.K12))
    paths = rq.build(d / "selection.jsonl", d / "requests", list(models.MODELS))
    return {p.name.split(".")[0]: rq.read_jsonl(p) for p in paths} | {
        p.name.split(".")[0] + "_meta": rq.read_jsonl(rq.manifest_path(p)) for p in paths
    }


def test_schema_follows_structured_output_limits() -> None:
    for o in objects(rq.SCHEMA):
        assert o["additionalProperties"] is False
        assert set(o["required"]) == set(o["properties"])
    assert not keys(rq.SCHEMA) & UNSUPPORTED_SCHEMA_KEYS
    assert rq.SCHEMA["properties"]["steps"].get("minItems", 0) in (0, 1)


@pytest.mark.parametrize("tag", ["opus55", "sonnet55", "haiku45"])
def test_request_shape(built: dict[str, list[dict[str, Any]]], tag: str) -> None:
    m = models.BY_TAG[tag]
    for line in built[tag]:
        assert set(line) == {"custom_id", "params"}
        assert rq.CUSTOM_ID.match(line["custom_id"])
        p = line["params"]
        assert p["model"] == m.id
        assert 1 <= p["max_tokens"] <= m.max_output
        assert set(p) <= {"model", "max_tokens", "system", "messages", "output_config", "thinking"}
        assert not {
            "temperature",
            "top_p",
            "top_k",
            "tool_choice",
            "tools",
            "stream",
            "speed",
        } & set(p)
        (sys_block,) = p["system"]
        assert sys_block["type"] == "text"
        assert sys_block["cache_control"] == {"type": "ephemeral", "ttl": "1h"}
        assert p["messages"][-1]["role"] == "user" and len(p["messages"]) == 1
        oc = p["output_config"]
        assert oc["format"] == {"type": "json_schema", "schema": rq.SCHEMA}
        if m.effort:
            assert oc["effort"] == "low"
            assert p["thinking"] == {"type": "adaptive"}
        else:
            assert "effort" not in oc
            assert p["thinking"]["type"] == "enabled"
            assert 1024 <= p["thinking"]["budget_tokens"] < p["max_tokens"]


def test_system_prompt_is_identical_across_requests_and_models(
    built: dict[str, list[dict[str, Any]]],
) -> None:
    texts = {
        line["params"]["system"][0]["text"]
        for tag in ("opus55", "sonnet55", "haiku45")
        for line in built[tag]
    }
    assert len(texts) == 1
    (t,) = texts
    assert t.startswith("You write the thinking of a small proof-state model.")
    assert "\\|" not in t and "`|- G := SKEL`" in t
    assert "### Complete example 2" in t


def test_examples_in_system_prompt_validate() -> None:
    td = dsl.dsl()
    t = rq.system_prompt()
    pilot = dsl.pilot_records()
    for k, pid in enumerate(rq.DEFAULT_EXAMPLES, 1):
        block = t.split(f"### Complete example {k}")[1].split("### Complete example")[0]
        reply = json.loads(block.split("Reply:\n\n")[1].strip())
        rec = dict(pilot[pid], steps=reply["steps"])
        assert td.validate(rec, lean=False, strict=True)["ok"]


def test_user_turn_states_round_trip_to_model_tokens(
    built: dict[str, list[dict[str, Any]]],
) -> None:
    td = dsl.dsl()
    sel = {r["id"]: r for r in fakes.pilot_selection(sl.K12)}
    for line, meta in zip(built["opus55"], built["opus55_meta"], strict=True):
        rec = sel[meta["proof_id"]]
        steps = dsl.reference_steps(rec["prompt"], rec["proof"])
        text = line["params"]["messages"][0]["content"]
        for s in steps:
            assert f"state: {s.state_ascii}" in text
            assert td.tokens(s.state_ascii) == s.state
            assert td.tokens(s.action_ascii) == s.action
        assert meta["ref_actions"] == [s.action_ascii for s in steps]
        assert meta["n_decision"] == sum(s.kind == "decision" for s in steps)


def test_distractors_appear_under_their_step(tmp_path: Path) -> None:
    sel = fakes.pilot_selection(sl.K12)[:1]
    rq.write_jsonl(tmp_path / "sel.jsonl", sel)
    bad = "have n5 : ( ( P > ( ( ~ P ) & R ) ) & ( ~ S ) ) := <| n1 , n2 |> ;"
    rq.write_jsonl(
        tmp_path / "d.jsonl",
        [{"proof_id": sel[0]["id"], "step": 3, "distractors": [{"action": bad, "why": "x"}]}],
    )
    (p,) = rq.build(
        tmp_path / "sel.jsonl",
        tmp_path / "req",
        ["claude-opus-5-5"],
        distractors=tmp_path / "d.jsonl",
    )
    text = rq.read_jsonl(p)[0]["params"]["messages"][0]["content"]
    after = text.split("step 3\n")[1].split("\n\nstep 4")[0]
    assert "likely mistakes" in after and bad in after
    assert text.count("likely mistakes") == 1


def test_solve_mode_gives_only_the_first_state(tmp_path: Path) -> None:
    rq.write_jsonl(tmp_path / "sel.jsonl", fakes.pilot_selection(sl.K12)[:2])
    (p,) = rq.build(tmp_path / "sel.jsonl", tmp_path / "req", ["claude-opus-5-5"], mode="solve")
    for line in rq.read_jsonl(p):
        text = line["params"]["messages"][0]["content"]
        assert text.startswith("Mode: solve.") and text.count("<st>") == 1
        assert line["custom_id"].endswith("-opus55-s1")


def test_prices_are_the_verified_ones() -> None:
    o, s, h = (
        models.MODELS[x]
        for x in ("claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-4-5-20251001")
    )
    assert (o.input, o.output, o.cache_write_5m, o.cache_write_1h, o.cache_read) == (
        4,
        20,
        5,
        8,
        0.20,
    )
    assert (s.input, s.output, s.cache_write_5m, s.cache_write_1h, s.cache_read) == (
        2,
        10,
        2.5,
        4,
        0.20,
    )
    assert (h.input, h.output, h.cache_write_5m, h.cache_write_1h, h.cache_read) == (
        1,
        5,
        1.25,
        2,
        0.10,
    )
    assert (o.cache_min_tokens, s.cache_min_tokens, h.cache_min_tokens) == (512, 512, 4096)
    assert o.price("output") == 10.0 and h.price("input") == 0.5


def test_estimate_arithmetic() -> None:
    m = models.MODELS["claude-opus-5-5"]
    system = "x " * 600
    p = rq.params(m, system, [{"role": "user", "content": "y " * 100}], max_tokens=1000)
    line = {"custom_id": "a-opus55-r1", "params": p}
    meta = {"custom_id": "a-opus55-r1", "attempt": 1, "est_output": 400, "est_extra_input": 0}
    est = rq.estimate(
        [line],
        [meta],
        assume_thinking=600,
        assume_attempts=1.0,
        assume_cache_hit=0.5,
        input_safety=1.0,
    )
    s, u = (
        rq.est_tokens(p["system"][0]["text"]),
        rq.est_tokens("y " * 100) + rq.est_tokens(json.dumps(rq.SCHEMA)),
    )
    assert s >= m.cache_min_tokens
    exp = (s * (0.5 * 0.10 + 0.5 * 4.0) + u * 2.0 + 1000 * 10.0) / 1e6
    worst = (s * 4.0 + u * 2.0 + 1000 * 10.0) / 1e6
    assert est["expected_usd"] == pytest.approx(exp)
    assert est["worst_usd"] == pytest.approx(worst)


def test_billed_usage_and_cost() -> None:
    u = {
        "input_tokens": 100,
        "output_tokens": 50,
        "cache_read_input_tokens": 1000,
        "cache_creation_input_tokens": 300,
        "cache_creation": {"ephemeral_5m_input_tokens": 100, "ephemeral_1h_input_tokens": 200},
    }
    t = models.billed(u)
    assert t == {
        "input": 100,
        "cache_read": 1000,
        "cache_write_5m": 100,
        "cache_write_1h": 200,
        "output": 50,
    }
    c = models.cost(t, models.MODELS["claude-sonnet-5-5"])
    assert c == pytest.approx((100 * 1.0 + 1000 * 0.10 + 100 * 1.25 + 200 * 2.0 + 50 * 5.0) / 1e6)
    assert models.billed({"cache_creation_input_tokens": 7})["cache_write_1h"] == 7
