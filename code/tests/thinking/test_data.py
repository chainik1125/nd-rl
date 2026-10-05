"""The training-pair builder: pilot traces, plain K12 records, loss masks, eval exclusion."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest
import torch

from code.thinking.dan import trace_dsl
from code.thinking.data import (
    EvalIndex,
    build,
    class_keys,
    heldout_records,
    load_pairs,
    plain_pairs,
    save,
    to_fork,
)
from code.thinking.tok import END_THINK, THINK, D, ThinkTokenizer
from code.thinking.train import make_batch_w

from ._pipeline_helpers import HELDOUT, PILOT, k12_head, needs_k12, pilot, pilot_rec


def _jsonl(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def test_build_from_pilot_traces() -> None:
    t = ThinkTokenizer()
    data = build(traces=[PILOT], lean=False, tok=t)
    rep = data["report"]
    assert rep["traces_in"] == rep["traces_valid"] == 8
    assert len(data["records"]) == 8 and rep["lengths"]["pairs"] == 70
    assert rep["lengths"]["steps_with_think"] == 39
    assert data["has_traces"] and data["mode"] == "lean_thinkn" and data["itos"] == t.itos
    T = trace_dsl()
    for rec, pairs in zip(pilot(), data["records"], strict=True):
        v = T.validate(rec, lean=False, strict=True)
        assert [tl for _, _, tl in pairs] == v["think_tokens"]  # block lengths, tags included
        assert max(len(p) + len(q) for p, q, _ in pairs) == v["pair_tokens_max"]
        for (p, q, tl), st in zip(pairs, rec["steps"], strict=True):
            toks = [t.itos[i] for i in q]
            assert toks[0] == THINK and toks[tl - 1] == END_THINK and toks[-1] == "<eos>"
            assert toks[tl:-1] == T.tokens(st["action"])
            assert t.itos[p[-1]] == "<act>"


def test_loss_mask_covers_exactly_think_action_eos() -> None:
    t = ThinkTokenizer()
    recs = build(traces=[PILOT], lean=False, tok=t)["records"]
    ids, plen, _, lens = D.state_train_best.to_device(to_fork(recs), "cpu")
    tlen = torch.tensor([tl for r in recs for _, _, tl in r])
    idxs = torch.arange(ids.shape[0])
    T = ids.shape[1]
    x, m = D.state_train_best.make_batch(ids, plen, lens, idxs, T, t, None, shift=False)
    _, mw = make_batch_w(ids, plen, tlen, lens, idxs, T, t, None, 0.25, shift=False)
    flat = [(p, q, tl) for r in recs for p, q, tl in r]
    for row, (p, q, tl) in enumerate(flat):
        on = m[row].nonzero().squeeze(1).tolist()
        assert on == list(range(len(p), len(p) + len(q)))  # targets: think block + action + <eos>
        assert x[row, on].tolist() == list(q)
        w = mw[row].tolist()
        assert all(v == 0.0 for v in w[: len(p)]) and all(v == 0.0 for v in w[len(p) + len(q) :])
        assert all(v == 0.25 for v in w[len(p) : len(p) + tl])
        assert all(v == 1.0 for v in w[len(p) + tl : len(p) + len(q)])
    assert torch.equal(
        make_batch_w(ids, plen, tlen, lens, idxs, T, t, None, 1.0, shift=False)[1], m.float()
    )


@needs_k12
def test_none_and_empty_modes() -> None:
    t = ThinkTokenizer()
    rec = k12_head(1)[0]
    e = plain_pairs(rec, t, "empty")
    n = plain_pairs(rec, t, "none")
    steps, _, _ = D.state_env.decompose(rec["prompt"], rec["proof"], canon=True)
    assert len(e) == len(n) == len(steps)
    for (pe, qe, te), (pn, qn, tn), (s, a, _) in zip(e, n, steps, strict=True):
        assert pe == pn == bytes(t.encode_toks(s))
        assert te == 2 and tn == 0
        assert list(qe) == [t.think, t.end_think] + t.encode_toks(a) + [t.eos]
        assert list(qn) == t.encode_toks(a) + [t.eos]
    d = build(plain=[], traces=[PILOT], no_trace_mode="none", lean=False)
    assert d["no_trace_mode"] == "none" and d["has_traces"]


def _rename(prompt: str, perm: dict[str, str]) -> str:
    return " ".join(perm.get(x, x) for x in prompt.split())


def test_eval_exclusion_by_renaming_class(tmp_path: Path) -> None:
    rec = pilot_rec("k12_22530")
    swapped = _rename(rec["prompt"], {"R": "Q", "S": "P"})
    assert class_keys(swapped)[0] == class_keys(rec["prompt"])[0]
    ev = _jsonl(tmp_path / "evalset.jsonl", [{"prompt": swapped}])
    data = build(traces=[PILOT], exclude=[ev], lean=False)
    assert data["report"]["traces_valid"] == 7
    assert data["report"]["excluded_by_set"] == {"evalset": 1}
    assert rec["prompt"] not in {m["prompt"] for m in data["meta"]}


def test_eval_exclusion_by_premise_order(tmp_path: Path) -> None:
    rec = pilot_rec("k12_37120")  # THM A , B SEQ C PRF
    body = rec["prompt"][4:-4]
    prem, concl = body.split(" SEQ ")
    a, b = prem.split(" , ")
    ev = _jsonl(tmp_path / "flip.jsonl", [{"prompt": f"THM {b} , {a} SEQ {concl} PRF"}])
    assert EvalIndex([ev]).hit(rec["prompt"]) == ["flip"]
    assert build(traces=[PILOT], exclude=[ev], lean=False)["report"]["traces_valid"] == 7


@needs_k12
def test_class_key_matches_the_fork_key_field() -> None:
    for r in k12_head(300):
        assert class_keys(r["prompt"])[0] == r["key"]


def test_invalid_traces_dropped_with_reasons(tmp_path: Path) -> None:
    bad = copy.deepcopy(pilot_rec("k12_37120"))
    bad["id"] = "bad"
    bad["steps"][3]["think"] = "n1 : S"  # a false look
    broken = {"id": "broken", "prompt": bad["prompt"]}  # no steps
    path = _jsonl(tmp_path / "t.jsonl", [*pilot(), bad, broken])
    data = build(traces=[path], lean=False)
    rep = data["report"]
    assert rep["traces_in"] == 10 and rep["traces_valid"] == 8
    errs = {x["id"]: x["error"] for x in rep["invalid_traces"]}
    assert "false look" in errs["bad"] and "KeyError" in errs["broken"]
    assert sum(rep["invalid_trace_reasons"].values()) == 2


@needs_k12
def test_plain_record_replaced_by_its_trace_and_trace_weight(tmp_path: Path) -> None:
    rows = k12_head(20)
    tr = pilot_rec("k12_37120")
    rows.append({"prompt": tr["prompt"], "proof": "N1 P : PR ; QED", "n_lines": 1})
    plain = _jsonl(tmp_path / "plain.jsonl", rows)
    d = build(traces=[PILOT], plain=[plain], lean=False, trace_weight=3)
    rep = d["report"]
    assert rep["plain_skipped"] == {"replaced_by_trace": 1}
    assert rep["plain_used"] == 20 and len(d["records"]) == 8 * 3 + 20
    assert sum(m["source"] == "trace" for m in d["meta"]) == 24
    d2 = build(traces=[f"{PILOT}:2"], lean=False)  # per-file weight
    assert len(d2["records"]) == 8 * 2


def test_later_trace_file_skips_prompts_an_earlier_file_covered() -> None:
    d = build(traces=[f"{PILOT}:2", PILOT], lean=False)  # LLM:2 then PROG, same proofs
    assert len(d["records"]) == 8 * 2
    assert d["report"]["traces_covered_by_earlier_file"] == 8
    assert d["report"]["traces_valid"] == 8


def test_empty_frac_empties_whole_copies_deterministically() -> None:
    d = build(traces=[f"{PILOT}:20"], lean=False, empty_frac=0.2)
    rep = d["report"]
    assert 0 < rep["trace_copies_emptied"] < 8 * 20 * 0.4
    for r, m in zip(d["records"], d["meta"], strict=True):
        lens = [n for _, _, n in r]
        if m["source"] == "trace_emptied":
            assert lens == [2] * len(r)  # <think> </think> on every step
    full = [r for r, m in zip(d["records"], d["meta"], strict=True) if m["source"] == "trace"]
    assert any(n > 2 for r in full for _, _, n in r)
    again = build(traces=[f"{PILOT}:20"], lean=False, empty_frac=0.2)
    assert again["report"]["trace_copies_emptied"] == rep["trace_copies_emptied"]


def test_save_load_round_trip_and_report(tmp_path: Path) -> None:
    t = ThinkTokenizer()
    d = build(traces=[PILOT], lean=False, max_len=200, tok=t)
    over = [len(p) + len(q) for r in d["records"] for p, q, _ in r if len(p) + len(q) > 200]
    assert d["report"]["lengths"]["pairs_over_max_len"] == len(over) > 0
    assert d["report"]["lengths"]["pair_tokens"]["max"] == 264  # the pilot's largest pair
    rp = save(d, tmp_path / "p.pkl")
    back = load_pairs(tmp_path / "p.pkl", t)
    assert back["records"] == d["records"]
    assert json.loads(rp.read_text())["traces_valid"] == 8
    old = D.lean_tok.LeanTokenizer("lean_staten")
    with pytest.raises(AssertionError):
        load_pairs(tmp_path / "p.pkl", old)


@needs_k12
def test_heldout_pairs_follow_the_format() -> None:
    t = ThinkTokenizer()
    h = heldout_records(HELDOUT, t, "empty", limit=5)
    assert len(h) == 5 and all(tl == 2 for r in h for _, _, tl in r)
