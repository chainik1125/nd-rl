"""Tests of authors_launch: names, the shown-tag rule and the loader, plans and refusals, the stop rule, the job runner.
Nothing here touches a GPU (the job runner runs `python -c` steps with the GPU check stubbed).

    CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 /scratch/cowifornia/nd-rl/think/nd-rl/code/.venv/bin/python \
        -m pytest --confcutdir=. -p no:cacheprovider -q test_authors_launch.py
"""

# ruff: noqa: E501

from __future__ import annotations

import argparse
import gzip
import itertools
import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import authors_launch as AL  # noqa: E402
import authors_sets as AS  # noqa: E402

T, TL, TM = AL.T, AL.TL, AL.TM
OFFICIAL = Path("/scratch/cowifornia/nd-rl/two_authors/sets")
SRC = Path("/scratch/cowifornia/nd-rl/knockout_grpo/dan_data/train_k12dose1.jsonl.gz")


@pytest.fixture()
def ta(tmp_path, monkeypatch):
    """A fresh TA tree (with the official sets linked in) for plans, refusals and the job runner."""
    monkeypatch.setattr(AL, "TA", tmp_path)
    monkeypatch.setattr(T, "AT", tmp_path)
    monkeypatch.setattr(AS, "OUT", OFFICIAL)
    return tmp_path


def corpus(path: Path, recs: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")


def src_records(n: int) -> list[dict]:
    with gzip.open(SRC, "rt") as f:
        return [json.loads(x) for x in itertools.islice(f, n)]


def test_names():
    assert AL.parse_base("authors_p10_s0") == ("authors_p10", 0)
    assert AL.parse_base("stage1_control_half_s2.pt") == ("control_half", 2)
    with pytest.raises(SystemExit):
        AL.parse_base("authors_p50_s0")
    assert AL.line_name("authors_p10", 1) == "rl_authors_p10_s1"
    assert AL.line_name("control_half", 1, 0) == "rl_control_half_b0_s1"
    assert AL.parse_line("rl_control_half_b0_s1") == ("control_half", 0, 1)
    assert AL.parse_line("rl_authors_p10_s2") == ("authors_p10", 2, 2)
    L = T.LINES["rl_authors_p10_s2"]
    assert L == {"tag": "none", "seed": 2, "base": "stage1_authors_p10_s2.pt", "ladder": "la_dose1_s2",
                 "corpus": "authors_p10", "base_seed": 2}  # fmt: skip
    assert T.LINES["rl_control_half_b0_s1"]["base"] == "stage1_control_half_s0.pt"
    assert TM.C.LINES["dd1_s2"]["seed"] == 2  # the recipe the line borrows exists


def test_shown_tag_rule():
    assert AL.shown_tag({"author_tag": "A"}) == "A"
    assert AL.shown_tag({"author_tag": "B", "author": "B"}) == "B"
    for t in (None, "", "none"):
        assert AL.shown_tag({"author_tag": t, "author": "A"}) == "none"
    assert AL.shown_tag({"author": "B", "tag_shown": True}) == "B"
    assert AL.shown_tag({"author_tag": "A", "author": "A", "tag_shown": False}) == "none"
    with pytest.raises(T.TagError):
        AL.shown_tag({"author": "A"})  # no shown-tag field at all
    with pytest.raises(T.TagError):
        AL.shown_tag({"author_tag": "C"})


def test_corpus_census_problems(tmp_path):
    recs = src_records(6)
    good = [
        {
            **r,
            "author": "A" if i % 2 else "B",
            "author_tag": ("A" if i % 2 else "B") if i < 3 else None,
        }
        for i, r in enumerate(recs)
    ]
    corpus(tmp_path / "good.jsonl.gz", good)
    c = AL.corpus_census(tmp_path / "good.jsonl.gz")
    assert c["problems"] == [] and c["records"] == 6 and c["shown"] == 3
    corpus(tmp_path / "all.jsonl.gz", [{**r, "author": "A", "author_tag": "A"} for r in recs])
    assert any(
        "every record shows" in p for p in AL.corpus_census(tmp_path / "all.jsonl.gz")["problems"]
    )
    corpus(tmp_path / "none.jsonl.gz", [{**r, "author": "A", "author_tag": None} for r in recs])
    assert any(
        "no record shows" in p for p in AL.corpus_census(tmp_path / "none.jsonl.gz")["problems"]
    )
    corpus(tmp_path / "wrong.jsonl.gz", [{**r, "author": "A", "author_tag": "B"} for r in recs])
    assert any(
        "its author is A" in p for p in AL.corpus_census(tmp_path / "wrong.jsonl.gz")["problems"]
    )


def test_authors_load_tags_states(tmp_path):
    """The loader encodes a shown-A record under <tagA>, a shown-B under <tagB>, an unshown one untagged, and the
    held-out file untagged; actions never carry a tag."""
    import tag_pretrain as TP

    cls = T.install_dan()
    tok = cls()
    recs = src_records(3)
    fn = tmp_path / "c.jsonl.gz"
    corpus(fn, [{**recs[0], "author": "A", "author_tag": "A"}, {**recs[1], "author": "B", "author_tag": "B"},
                {**recs[2], "author": "A", "author_tag": None}])  # fmt: skip
    TP._cfg.update(data=str(fn), limit=0, pairs={})
    data, skipped = AL.authors_load(str(fn), tok, 12)
    assert skipped == 0 and len(data) == 3
    tag_ids = {v: k for k, v in tok.tag_ids.items()}
    for pairs, want in zip(data, ("A", "B", "none"), strict=True):
        for pre, act in pairs:
            pre, act = list(pre), list(act)
            assert pre[-1] == tok.act_id
            assert tag_ids.get(pre[-2], "none") == want
            assert not any(x in tag_ids for x in act)
    assert TP._cfg["pairs"]["train"]["none"] == len(data[2]) and TP._cfg["pairs"]["train"][
        "A"
    ] == len(data[0])
    held = tmp_path / "h.jsonl"
    held.write_text(json.dumps(recs[0]) + "\n")
    hdata, _ = AL.authors_load(str(held), tok, 0, limit=1)
    assert all(list(p)[-2] not in tag_ids for p, _ in hdata[0])
    assert TP._cfg["pairs"]["heldout"] == {"A": 0, "B": 0, "none": len(hdata[0])}


def ns(**kw) -> argparse.Namespace:
    d = {"cmd": "pretrain", "gpu": 0, "corpus": "authors_half", "seed": 0, "base": None, "tags": "A,B,none",
         "sets": "marker_plain,route_unseen,marker_heldout_class", "k": 32, "line": None, "seg": 1, "resume": None,
         "skip_done": False, "snap_k": 8, "snap_sets": "marker_plain,route_unseen", "stop_on_held": False,
         "lab": False, "allow_unreadable": False, "dry_run": True}  # fmt: skip
    d.update(kw)
    return argparse.Namespace(**d)


@pytest.mark.skipif(not (OFFICIAL / "manifest.json").exists(), reason="official sets not built")
def test_pretrain_plan_and_refusals(ta):
    job, steps, problems, notes = AL.plan(ns())
    assert job == "pretrain_authors_half_s0" and [s["name"] for s in steps] == [
        "pretrain_authors_half_s0",
        "sanity_authors_half_s0",
    ]
    assert any("no corpus" in p for p in problems)
    assert steps[0]["argv"][-1] == str(ta / "bases" / "stage1_authors_half_s0.pt")
    assert steps[0]["timeout_s"] == 1.5 * 3600
    recs = src_records(10)
    corpus(
        ta / "data" / "authors_half.jsonl.gz",
        [{**r, "author": "A", "author_tag": "A" if i < 5 else None} for i, r in enumerate(recs)],
    )
    _, _, problems, notes = AL.plan(ns())
    assert problems == [] and any("25,607 steps" in n for n in notes)
    (ta / "bases").mkdir()
    (ta / "bases" / "stage1_authors_half_s0.pt").write_text("x")
    _, _, problems, _ = AL.plan(ns())
    assert any("never overwritten" in p for p in problems)
    with pytest.raises(SystemExit):
        AL.plan(ns(corpus="authors_p50"))


@pytest.mark.skipif(not (OFFICIAL / "manifest.json").exists(), reason="official sets not built")
def test_measure_plan_and_refusals(ta):
    a = ns(cmd="measure", base="authors_p10_s1")
    job, steps, problems, notes = AL.plan(a)
    assert job == "measure_authors_p10_s1" and any("no base" in p for p in problems)
    assert "--k" in steps[0]["argv"] and steps[0]["argv"][steps[0]["argv"].index("--k") + 1] == "32"
    assert any("105,600 attempts" in n for n in notes)
    (ta / "bases").mkdir()
    (ta / "bases" / "stage1_authors_p10_s1.pt").write_text("x")
    (ta / "measure" / "authors_p10_s1").mkdir(parents=True)
    (ta / "measure" / "authors_p10_s1" / "route_unseen__B.summary.json").write_text("{}")
    _, _, problems, _ = AL.plan(a)
    assert len(problems) == 1 and "never overwritten" in problems[0]
    _, steps, problems, _ = AL.plan(ns(cmd="measure", base="authors_p10_s1", skip_done=True))
    assert problems == [] and "--resume" in steps[0]["argv"]
    _, _, problems, _ = AL.plan(ns(cmd="measure", base="authors_p10_s1", sets="marker_plain,nope"))
    assert any("unknown set nope" in p for p in problems)


@pytest.mark.skipif(not (OFFICIAL / "manifest.json").exists(), reason="official sets not built")
def test_rl_plan_and_refusals(ta):
    a = ns(cmd="rl", line="rl_control_p10_s0")
    job, steps, problems, notes = AL.plan(a)
    assert job == "rl_rl_control_p10_s0_seg1" and any(
        "no " in p and "stage1_control_p10_s0" in p for p in problems
    )
    assert steps[0]["timeout_s"] == 10 * 3600 and "_line" in steps[0]["argv"]
    (ta / "bases").mkdir()
    (ta / "bases" / "stage1_control_p10_s0.pt").write_text("x")
    (ta / "bases" / "stage1_control_p10_s0.tag.json").write_text(
        json.dumps({"train_tokens": 470535415, "author_tags": {}})
    )
    _, _, problems, notes = AL.plan(a)
    assert (
        problems == []
        and any("D 470,535,415" in n for n in notes)
        and any("reached-and-held stop off" in n for n in notes)
    )
    (ta / "runs" / "rl_control_p10_s0" / "seg1").mkdir(parents=True)
    _, _, problems, _ = AL.plan(a)
    assert any("never re-run" in p for p in problems)
    _, _, problems, _ = AL.plan(ns(cmd="rl", line="rl_control_p10_s0", seg=2, resume="/tmp/x.pt"))
    assert any("is not a rl_control_p10_s0_it<i>_rolling|alarm.pt" in p for p in problems)
    _, _, problems, _ = AL.plan(ns(cmd="rl", line="rl_control_p10_s0", seg=2))
    assert any("needs --resume" in p for p in problems)
    (ta / "bases" / "stage1_control_p10_s0.tag.json").write_text(
        json.dumps({"train_tokens": 1, "author_tags": {"variant": "p10"}})
    )
    _, _, problems, _ = AL.plan(ns(cmd="rl", line="rl_control_p10_s0", seg=2))
    assert any("variant" in p for p in problems)


@pytest.mark.skipif(AL.HOST not in AL.TL.LAB, reason="runs on a lab host only")
def test_lab_host_refuses_without_lab_flag_and_dry_run_writes_nothing(ta, capsys):
    ok, why, _ = AL.host_gate(0, lab=False, allow_unreadable=False)
    assert not ok and "lab host" in why[0]
    AL.launch(ns(cmd="measure", base="authors_p10_s0"))
    out = capsys.readouterr().out
    assert "dry run" in out and "A REAL LAUNCH REFUSES" in out and "gate: FAIL" in out
    assert list(ta.iterdir()) == []  # nothing written
    with pytest.raises(SystemExit, match="refused"):
        AL.launch(ns(cmd="measure", base="authors_p10_s0", dry_run=False))
    assert list(ta.iterdir()) == []


def test_stop_rule():
    v = {"x_dan": 0.4, "gpuh": 1.0, "curve": [], "cross_dan": None}
    assert AL.stop_reason(v) is None
    assert "compute cap" in AL.stop_reason({**v, "x_dan": 1.5})
    assert "GPU-h cap" in AL.stop_reason({**v, "gpuh": AL.GPUH_CAP})
    held = {**v, "cross_dan": {"x_dan": 0.3}, "curve": [{"it": 10, "x_dan": 0.3, "first32": 14}, {"it": 20, "x_dan": 0.45, "first32": 15},
                                                         {"it": 30, "x_dan": 0.7, "first32": 16}]}  # fmt: skip
    assert TM.M.held(held["curve"]) is not None
    assert AL.stop_reason(held) is None  # held does not stop our lines by default
    assert "held" in AL.stop_reason(held, on_held=True)


def test_snap_cmd():
    c = AL.snap_cmd(
        "rl_authors_p10_s0", 2, Path("/x/s.pt"), 40, 8, "marker_plain,route_unseen", False, 0, 0
    )
    assert (
        c[c.index("--ckpt_id") + 1] == "rl_authors_p10_s0@it40"
        and c[c.index("--device") + 1] == "cuda"
    )
    assert c[c.index("--out_dir") + 1].endswith("runs/rl_authors_p10_s0/seg2/authors/it0040")
    assert c[c.index("--tags") + 1] == "A,B,none" and "--resume" in c and "--limit" not in c


def test_run_job_lock_timeout_log(ta, monkeypatch):
    """tag_launch.run_job, as the detached job runs it: the lock, each step under timeout, the log and the status,
    stopping at the first failing step, the lock released."""
    monkeypatch.setattr(TL.G, "free_for_us", lambda host, gpu: True)
    log = ta / "logs" / "job_x.log"
    log.parent.mkdir(parents=True)
    py = sys.executable
    spec = {"job": "job_x", "gpu": 0, "log": str(log), "status": str(log.with_suffix(".status.json")),
            "steps": [AL.step("ok", [py, "-c", "print('hello')"], 0.01, 0.0),
                      AL.step("slow", [py, "-c", "import time; time.sleep(30)"], 1 / 3600, 0.0),
                      AL.step("never", [py, "-c", "print('no')"], 0.01, 0.0)]}  # fmt: skip
    sp = ta / "logs" / "job_x.spec.json"
    sp.write_text(json.dumps(spec))
    rc = TL.run_job(sp)
    assert rc == 124  # the second step hit its timeout
    st = json.loads(log.with_suffix(".status.json").read_text())
    assert [(s["name"], s["rc"]) for s in st["steps"]] == [("ok", 0), ("slow", 124)]
    text = log.read_text()
    assert "hello" in text and "step slow: end rc 124 (timeout)" in text and "lock released" in text
    assert not (ta / "locks" / f"{AL.HOST}_gpu0.lock").exists()
    (ta / "locks" / f"{AL.HOST}_gpu0.lock").mkdir(parents=True)
    assert TL.run_job(sp) == 3  # a held lock refuses


def test_pretrain_refuses_existing_base(tmp_path):
    out = tmp_path / "stage1_x_s0.pt"
    out.with_suffix(".tag.json").write_text("{}")
    with pytest.raises(SystemExit, match="never overwritten"):
        AL.pretrain_main(
            ["--corpus", str(tmp_path / "c.jsonl.gz"), "--seed", "0", "--out", str(out)]
        )


DATA = Path("/scratch/cowifornia/nd-rl/two_authors/data")


def test_shown_tag_on_the_corpus_format():
    """authors_corpus.py's records: `tag_shown` decides, `author_tag` is "none" when hidden."""
    assert AL.shown_tag({"author": "A", "tag_shown": False, "author_tag": "none"}) == "none"
    assert AL.shown_tag({"author": "B", "tag_shown": True, "author_tag": "B"}) == "B"


@pytest.mark.skipif(not (DATA / "authors_half.jsonl.gz").exists(), reason="corpora not built")
def test_census_and_loader_on_the_real_corpora(tmp_path):
    import tag_pretrain as TP

    for c in AL.CORPORA:
        cen = AL.corpus_census(DATA / f"{c}.jsonl.gz", limit=4000)
        assert cen["problems"] == [], (c, cen)
        assert 0.4 < cen["shown"] / cen["records"] < 0.6, (c, cen)
    with gzip.open(DATA / "authors_half.jsonl.gz", "rt") as f:
        recs = [json.loads(x) for x in itertools.islice(f, 40)]
    pick = [next(r for r in recs if r["author_tag"] == t) for t in ("A", "B", "none")]
    fn = tmp_path / "c.jsonl.gz"
    corpus(fn, pick)
    cls = T.install_dan()
    tok = cls()
    TP._cfg.update(data=str(fn), limit=0, pairs={})
    data, skipped = AL.authors_load(str(fn), tok, 12)
    tag_ids = {v: k for k, v in tok.tag_ids.items()}
    assert skipped == 0
    for pairs, want in zip(data, ("A", "B", "none"), strict=True):
        assert {tag_ids.get(list(p)[-2], "none") for p, _ in pairs} == {want}
