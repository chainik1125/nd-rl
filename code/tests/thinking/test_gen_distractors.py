"""Base-model distractor sampling (code/thinking/gen/distractors.py): canonical renaming, the
keep/drop rules, and a CPU run of the real stage-1 checkpoint on two decision states."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from code.thinking.gen import distractors as ds
from code.thinking.gen import dsl, fakes
from code.thinking.gen import requests as rq
from code.thinking.gen import select as sl

FORK = Path(os.environ.get("ND_TAKEHOME", "/scratch/cowifornia/nd-rl/think/nd-takehome"))
pytestmark = pytest.mark.skipif(
    not (FORK / "state_env.py").exists() or not sl.K12.exists(), reason="needs Dan's fork and K12"
)
C = "( ( P > ( ( ~ P ) & R ) ) & ( ~ S ) )"


def state3() -> dsl.RefStep:
    rec = next(r for r in fakes.pilot_selection(sl.K12) if r["id"] == "k12_37120")
    s = dsl.reference_steps(rec["prompt"], rec["proof"], with_env=True)[3]
    assert s.kind == "decision"
    return s


def toks(a: str) -> list[str]:
    return dsl.dsl().tokens(a)


def test_canonical_renames_defined_names_only() -> None:
    s = state3()
    assert ds.canonical(s.env, toks(f"have n9 : {C} := <| n2 , n1 |> ;")) == s.action
    box = ds.canonical(s.env, toks("have n9 : ( Q > Q ) := ( fun ( n12 : Q ) => by"))
    assert box[1] == "n5" and box[box.index(":=") + 4] == "n6"
    assert ds.canonical(s.env, toks("exact n3")) == toks("exact n3")


def test_judge_samples_keeps_distinct_rejected_actions() -> None:
    s = state3()
    swapped = toks(f"have n7 : {C} := <| n1 , n2 |> ;")  # step_check rejects; renamed to n5
    unbound = toks(f"have n5 : {C} := <| n2 , n9 |> ;")  # env rejects: n9 not in scope
    other_ok = toks("have n5 : ( ( ~ S ) v Q ) := Or.inl n1 ;")
    junk = toks("have")
    samples = [s.action, swapped, swapped, unbound, other_ok, junk, swapped]
    j = ds.judge_samples(s.env, s.action, samples, {"env", "step_check"})
    keys = j["kept"]
    assert keys[0] == " ".join(ds.canonical(s.env, swapped)) and j["counts"][keys[0]] == 3
    assert j["why"][keys[0]].startswith("step_check")
    assert j["why"][keys[1]].startswith("env") and len(keys) == 2
    assert j["n_valid"] == 2 and j["n_distinct"] == 5
    assert j["why"]["have"].startswith("unusable")


def test_kept_distractors_are_refutable_candidates() -> None:
    """A kept distractor, written as a candidate and rejected, must pass the validator."""
    s = state3()
    j = ds.judge_samples(
        s.env, s.action, [toks(f"have n7 : {C} := <| n1 , n2 |> ;")], {"env", "step_check"}
    )
    (d,) = j["kept"]
    rec = dict(dsl.pilot_records()["k12_37120"])
    rec["steps"] = [dict(x) for x in rec["steps"]]
    rec["steps"][3]["think"] = f"{dsl.ascii_toks(d.split(' '))}\nn1 : ( ~ S )\n<no>"
    assert dsl.dsl().validate(rec, lean=False, strict=True)["ok"]


def test_cpu_sampling_with_stage1_checkpoint(tmp_path: Path) -> None:
    if not Path(ds.STAGE1).exists():
        pytest.skip("no stage-1 checkpoint")
    rq.write_jsonl(
        tmp_path / "sel.jsonl", [r for r in fakes.pilot_selection(sl.K12) if r["id"] == "k12_37120"]
    )
    t0 = time.time()
    meta = ds.run(
        tmp_path / "sel.jsonl", tmp_path / "d.jsonl", device="cpu", k=8, limit_states=2, seed=1
    )
    assert time.time() - t0 < 60
    rows = rq.read_jsonl(tmp_path / "d.jsonl")
    assert meta["n_states"] == 2 and len(rows) == 2 and meta["device"] == "cpu"
    for r in rows:
        assert r["proof_id"] == "k12_37120" and r["n_samples"] == 8
        assert r["n_ended"] + 0 <= 8 and r["n_distinct"] >= 1
        assert len(r["distractors"]) <= 3
        for d in r["distractors"]:
            assert d["why"].split(":")[0] in ("env", "step_check")
            assert d["action"] != r["ref_action"]
    on_disk = json.loads((tmp_path / "d.meta.json").read_text())
    assert on_disk["k"] == 8 and on_disk["temperature"] == 0.8
    assert not (FORK / "__pycache__").exists()
