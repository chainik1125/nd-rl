"""End to end with a fake Message Batches server: select -> build -> submit -> collect ->
validate -> retry generation, plus result parsing and per-model stats (gen/collect.py)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

from code.thinking.gen import batch as bt
from code.thinking.gen import collect as cl
from code.thinking.gen import fakes
from code.thinking.gen import requests as rq
from code.thinking.gen import select as sl
from code.thinking.gen.client import HttpBatchClient

FORK = Path(os.environ.get("ND_TAKEHOME", "/scratch/cowifornia/nd-rl/think/nd-takehome"))
pytestmark = pytest.mark.skipif(
    not (FORK / "state_env.py").exists() or not sl.K12.exists(), reason="needs Dan's fork and K12"
)
KEY = "sk-ant-collect-test"
BROKEN = {
    "k12_16472": "false_look",
    "k12_22530": "truncated",
    "k12_89161": "errored",
    "k12_53241": "always_bad",
}


def submit_round(run: Path, client: HttpBatchClient) -> list[str]:
    done = {
        r["requests_file"]
        for r in bt.ledger_rows(run / "ledger.jsonl")
        if r.get("kind") == "submit"
    }
    sent = []
    for p in sorted((run / "requests").glob("*.jsonl")):
        if p.name.endswith(".manifest.jsonl") or str(p) in done:
            continue
        b = bt.submit(p, run, client=client, send=True, max_usd=100)["batch"]
        bt.poll(b["id"], run, client)
        sent.append(p.name)
    return sent


@pytest.fixture(scope="module")
def e2e(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    mp = pytest.MonkeyPatch()
    mp.setenv("ANTHROPIC_API_KEY", KEY)
    run = tmp_path_factory.mktemp("e2e")
    src = run / "src.jsonl"
    with open(sl.K12) as f, open(src, "w") as g:
        want = {r["row"] for r in fakes.pilot_selection(sl.K12)}
        for i, line in enumerate(f):
            if i in want:
                g.write(line)
            if i > max(want):
                break
    path, _ = sl.run(run, src, n=6, seed=0, excludes={})
    sel = sl.read_jsonl(path)
    ids = {r["prompt"]: r["id"] for r in fakes.pilot_selection(sl.K12)}
    for r in sel:
        r["id"] = ids[r["prompt"]]
    rq.write_jsonl(path, sel)
    rq.build(path, run / "requests", ["claude-opus-5-5", "claude-sonnet-5-5"])
    api = fakes.FakeBatchAPI(fakes.pilot_responder(BROKEN), expect_key=KEY)
    client = HttpBatchClient(transport=api)
    rounds = []
    for _ in range(4):
        sent = submit_round(run, client)
        st = cl.collect(run)
        rounds.append({"sent": sent, "stats": st})
    mp.undo()
    return {"run": run, "rounds": rounds, "api": api}


def test_retry_files_follow_failures(e2e: dict[str, Any]) -> None:
    r = e2e["rounds"]
    assert r[0]["sent"] == ["opus55.r1.jsonl", "sonnet55.r1.jsonl"]
    assert r[1]["sent"] == ["opus55.r2.jsonl", "sonnet55.r2.jsonl"]
    assert r[2]["sent"] == ["opus55.r3.jsonl", "sonnet55.r3.jsonl"]
    assert r[3]["sent"] == [] and r[3]["stats"]["retry_files"] == []
    run = e2e["run"]
    a2 = {x["custom_id"]: x for x in rq.read_jsonl(run / "requests" / "opus55.r2.jsonl")}
    assert set(a2) == {f"{p}-opus55-r2" for p in BROKEN}
    fb = a2["k12_16472-opus55-r2"]["params"]["messages"]
    assert [m["role"] for m in fb] == ["user", "assistant", "user"]
    first = {x["custom_id"]: x for x in rq.read_jsonl(run / "requests" / "opus55.r1.jsonl")}
    assert fb[0] == first["k12_16472-opus55-r1"]["params"]["messages"][0]
    res = {x["custom_id"]: x for f in (run / "results").glob("*.jsonl") for x in rq.read_jsonl(f)}
    assert fb[1]["content"] == res["k12_16472-opus55-r1"]["result"]["message"]["content"]
    assert fb[1]["content"][0]["type"] == "thinking"
    assert fb[2]["content"].startswith(
        "The checker rejected this trace: step 0: false look: n1 : ( P & Q )"
    )
    assert "∧" not in fb[2]["content"]
    trunc = a2["k12_22530-opus55-r2"]["params"]
    assert len(trunc["messages"]) == 1 and trunc["max_tokens"] == 24000
    assert len(a2["k12_89161-opus55-r2"]["params"]["messages"]) == 1
    a3 = rq.read_jsonl(run / "requests" / "opus55.r3.jsonl")
    assert [x["custom_id"] for x in a3] == ["k12_53241-opus55-r3"]
    assert len(a3[0]["params"]["messages"]) == 5
    metas = rq.read_jsonl(run / "requests" / "opus55.r3.manifest.jsonl")
    assert metas[0]["parent"] == "k12_53241-opus55-r2" and metas[0]["attempt"] == 3


def test_accepted_traces_are_valid_pilot_records(e2e: dict[str, Any]) -> None:
    from code.thinking.gen import dsl

    run = e2e["run"]
    acc = rq.read_jsonl(run / "collect" / "accepted.opus55.jsonl")
    pilot = dsl.pilot_records()
    assert sorted(r["id"] for r in acc) == sorted(
        set(pilot) - {"k12_53241"} - {k for k in pilot if k.startswith("tb_")}
    )
    for r in acc:
        assert {"id", "source", "mode", "deviates", "prompt", "steps", "gen"} <= set(r)
        assert r["mode"] == "reference-guided"
        assert r["steps"] == pilot[r["id"]]["steps"]
        assert r["deviates"] == pilot[r["id"]]["deviates"]
        assert dsl.dsl().validate(
            {k: v for k, v in r.items() if k != "gen"}, lean=False, strict=True
        )["ok"]
        assert r["gen"]["attempts"] == (2 if r["id"] in BROKEN else 1)
    (rej,) = rq.read_jsonl(run / "collect" / "rejected.opus55.jsonl")
    assert rej["id"] == "k12_53241" and rej["attempts"] == 3
    assert rej["category"] == "false_look" and len(rej["errors"]) == 3


def test_stats_report_billed_tokens_per_accepted_trace(e2e: dict[str, Any]) -> None:
    st = e2e["rounds"][-1]["stats"]
    o = st["by_model"]["claude-opus-5-5"]
    assert o["n_proofs"] == 6 and o["accepted"] == 5
    assert o["accepted_pct"] == pytest.approx(500 / 6)
    assert o["first_pass_pct"] == pytest.approx(200 / 6)
    assert o["accepted_by_attempt"] == {1: 2, 2: 3}
    # fake usage per succeeded attempt: 1200 in, 3000 cache read, 900 out; 10 succeeded attempts
    # (11 attempts, one errored and unbilled)
    assert o["attempts"] == 11
    assert o["billed_tokens"] == {
        "input": 12000,
        "cache_read": 30000,
        "cache_write_5m": 0,
        "cache_write_1h": 0,
        "output": 9000,
    }
    assert o["billed_tokens_per_accepted"]["output"] == pytest.approx(9000 / 5)
    assert o["billed_tokens_total_per_accepted"] == pytest.approx(51000 / 5)
    assert o["cost_usd"] == pytest.approx((12000 * 2.0 + 30000 * 0.10 + 9000 * 10.0) / 1e6)
    assert o["error_categories_all_attempts"] == {"false_look": 4, "max_tokens": 1, "api_error": 1}
    sv = st["sonnet_vs_opus"]
    assert sv["billed_tokens_per_accepted_ratio"] == pytest.approx(1.0)
    # cache reads cost the same on both models ($0.10 / MTok batch), so not exactly 0.5
    sonnet = (12000 * 1.0 + 30000 * 0.10 + 9000 * 5.0) / 1e6
    assert sv["cost_per_accepted_ratio"] == pytest.approx(sonnet / o["cost_usd"])
    assert sv["sonnet_under_2x_tokens"] is True
    report = (e2e["run"] / "collect" / "report.md").read_text()
    assert "Sonnet qualifies" in report and "83.3% (5)" in report


def test_ledger_records_submissions_and_billed_usage(e2e: dict[str, Any]) -> None:
    rows = bt.ledger_rows(e2e["run"] / "ledger.jsonl")
    subs = [r for r in rows if r["kind"] == "submit"]
    usage = {r["batch_id"]: r for r in rows if r["kind"] == "usage"}
    assert len(subs) == 6 and set(usage) == {s["batch_id"] for s in subs}
    first = usage[subs[0]["batch_id"]]
    assert first["by_model"]["claude-opus-5-5"]["tokens"]["output"] == 5 * 900
    assert first["by_model"]["?"]["results"] == {"errored": 1}
    assert all(s["estimated_usd"] > 0 and s["n_requests"] >= 1 for s in subs)


def test_collect_is_idempotent(e2e: dict[str, Any]) -> None:
    run = e2e["run"]
    before = sorted(p.name for p in (run / "requests").iterdir())
    st = cl.collect(run)
    assert (
        st["retry_files"] == [] and sorted(p.name for p in (run / "requests").iterdir()) == before
    )
    assert json.dumps(st["by_model"]) == json.dumps(e2e["rounds"][-1]["stats"]["by_model"])
    assert st["ledger_usage_rows"] == 0


@pytest.mark.parametrize(
    ("msg", "err"),
    [
        (
            {"stop_reason": "refusal", "stop_details": {"category": "bio"}, "content": []},
            "refusal (bio)",
        ),
        (
            {"stop_reason": "max_tokens", "content": [{"type": "text", "text": '{"st'}]},
            "max_tokens",
        ),
        ({"stop_reason": "end_turn", "content": [{"type": "text", "text": "nope"}]}, "json:"),
        (
            {"stop_reason": "end_turn", "content": [{"type": "text", "text": '{"steps": []}'}]},
            "schema:",
        ),
        (
            {
                "stop_reason": "end_turn",
                "content": [{"type": "text", "text": '{"steps": [{"think": 1, "action": ""}]}'}],
            },
            "schema:",
        ),
    ],
)
def test_parse_message_failures(msg: dict[str, Any], err: str) -> None:
    steps, e = cl.parse_message(msg)
    assert steps is None and e is not None and e.startswith(err)


def test_refusal_is_rejected_without_retry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", KEY)
    rq.write_jsonl(tmp_path / "selection.jsonl", fakes.pilot_selection(sl.K12)[:1])
    rq.build(tmp_path / "selection.jsonl", tmp_path / "requests", ["claude-opus-5-5"])
    pid = fakes.pilot_selection(sl.K12)[0]["id"]
    client = HttpBatchClient(
        transport=fakes.FakeBatchAPI(fakes.pilot_responder({pid: "refusal"}), expect_key=KEY)
    )
    submit_round(tmp_path, client)
    st = cl.collect(tmp_path)
    assert st["retry_files"] == []
    assert st["by_model"]["claude-opus-5-5"]["final_error_categories"] == {"refusal": 1}


def test_bare_have_line_is_a_rejection_not_a_crash() -> None:
    from code.thinking.gen import dsl

    rec = dsl.pilot_records()["k12_37120"]
    steps = [dict(s) for s in rec["steps"]]
    steps[3]["think"] = "have"
    meta = {"proof_id": rec["id"], "mode": "reference", "prompt": rec["prompt"], "ref_actions": []}
    result = {"result": fakes.message("claude-opus-5-5", json.dumps({"steps": steps}))}
    out = cl.judge(meta, result, lean=False)
    assert out["status"] == "retry_feedback" and not out["report"]["ok"]
    assert cl.category(out["error"]) != "validator_crash"
