"""Tests of lh_launch (version 2): W2's launcher pointed at the c-corpora and LH (names, lines, plans and refusals of
pretrain / measure / prep2 / stage2 / rl), the loader's line cap 16, the step rule, the stage-2 scheduler and its
dependency, the offline-arm and reward-model steps with stand-in trainers, `wait`, and W2's job runner on LH. Nothing
here touches a GPU.

    CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 /scratch/cowifornia/nd-rl/think/nd-rl/code/.venv/bin/python \
        -m pytest --confcutdir=. -p no:cacheprovider -q test_lh_launch.py
"""

# ruff: noqa: E501, I001

from __future__ import annotations

import argparse
import os
import gzip
import itertools
import json
import re
import sys
import time
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import lh_core as C  # noqa: E402
import lh_launch as LL  # noqa: E402

AL, T, TL, TM = LL.AL, LL.T, LL.TL, LL.TM
import authors_sets as AS  # noqa: E402

PY = sys.executable
SETS = C.LH / "sets"
REAL = C.LH / "data"  # the installed corpora (the `lh` fixture points C.LH elsewhere)
HAS_C24 = (REAL / "c24.jsonl.gz").exists() and C.HABIT_TABLE.exists()


@pytest.fixture()
def lh(tmp_path, monkeypatch):
    """A fresh LH tree (the installed sets and tables linked in) for plans, refusals and the job runner."""
    for x in ("data", "sets", "bases"):
        (tmp_path / x).mkdir()
    for f in ("note_table.json", "habit_table.json", "rewarded_class.json"):
        if (C.LH / "data" / f).exists():
            (tmp_path / "data" / f).symlink_to(C.LH / "data" / f)
    for s in C.SETS:
        (tmp_path / "sets" / f"{s}.jsonl").symlink_to(SETS / f"{s}.jsonl")
    monkeypatch.setattr(AL, "TA", tmp_path)
    monkeypatch.setattr(T, "AT", tmp_path)
    monkeypatch.setattr(LL, "LH", tmp_path)
    monkeypatch.setattr(C, "LH", tmp_path)
    monkeypatch.setattr(AS, "OUT", tmp_path / "sets")
    monkeypatch.setattr(LL, "STEPS_FILE", tmp_path / "data" / "pretrain_steps.json")
    for k in ("NOTE_TABLE", "REWARDED", "HABIT_TABLE"):
        monkeypatch.setattr(C, k, tmp_path / "data" / getattr(C, k).name)
    monkeypatch.setattr(LL, "detector_problems", lambda: [])
    return tmp_path


def ns(**kw) -> argparse.Namespace:
    d = {"cmd": "pretrain", "gpu": 0, "corpus": "c24", "seed": 0, "base": None, "tags": "A,B,none", "sets": "marker_plain,route_unseen",
         "k": 8, "line": None, "seg": 1, "resume": None, "skip_done": False, "snap_k": 8, "snap_sets": "marker_plain,route_unseen",
         "stop_on_held": False, "lab": False, "allow_unreadable": False, "dry_run": True, "tasks": None, "parallel": 2, "stagger_s": 120.0}  # fmt: skip
    d.update(kw)
    return argparse.Namespace(**d)


def corpus(path: Path, recs: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")


def real_records(c: str, n: int, pred=lambda r: True) -> list[dict]:
    with gzip.open(REAL / f"{c}.jsonl.gz", "rt") as f:
        return list(itertools.islice((r for r in map(json.loads, f) if pred(r)), n))


def steps_file(lh: Path, corpora=("c24", "c9", "c4", "c0")) -> dict:
    doc = LL.steps_doc(
        {
            "authors_half": {"tokens": 1000},
            **{c: {"tokens": 1700 + i} for i, c in enumerate(corpora)},
        }
    )
    (lh / "data" / "pretrain_steps.json").write_text(json.dumps(doc))
    return doc


# ------------------------------------------------------------------------------------------------ names
def test_names_and_lines():
    assert AL.parse_base("c24_s0") == ("c24", 0) and AL.parse_base("stage1_c0_s2.pt") == ("c0", 2)
    assert AL.parse_base("c0n_s0") == ("c0n", 0) and "c0n" in AL.CORPORA
    assert AL.parse_base("control_half_s0") == ("control_half", 0) and AL.parse_base(
        "authors_half_s0"
    ) == ("authors_half", 0)
    with pytest.raises(SystemExit):
        AL.parse_base("l19_s0" if "l19" not in C.CORPORA else "zz_s0")
    assert (
        AL.parse_line("rl_c24_rlhfatom_s0") == ("c24", 0, 0)
        and AL.line_variant("rl_c24_anc_s0") == "anc"
    )
    ours = sorted(x for x in T.LINES if x.startswith("rl_") and x.endswith("_s0"))
    assert ours == sorted(C.all_lines(0))
    if C.CORPORA == ("c24", "c9", "c4", "c0"):
        assert len(ours) == 6 and not [
            x for x in T.LINES if "rlhfatom" in x or "_c9_" in x
        ]  # Amendment 4: no atom GRPO line; Amendment 6: none on c9
    assert not [x for x in T.LINES if "authors_" in x or "control_" in x]  # W2's lines are not ours
    L = T.LINES[C.line_name(C.CORPORA[0], "anc")]
    assert (
        L["tag"] == "none"
        and L["rlvar"] == "anc"
        and L["base"] == f"stage1_{C.CORPORA[0]}_s0.pt"
        and L["lh"]
    )
    assert AL.TA == C.LH and T.AT == C.LH and AS.TA == C.LH


def test_linked_of(monkeypatch, tmp_path):
    p = tmp_path / "habit_table.json"
    p.write_text(
        json.dumps(
            {"linked": ["route", "order", "reit", "conj", "disjunct", "note_atom", "note_~a"]}
        )
    )
    monkeypatch.setattr(C, "HABIT_TABLE", p)
    assert C.linked_of("c24") == [
        "route",
        "order",
        "reit",
        "conj",
        "disjunct",
        "note_atom",
        "note_~a",
    ]
    assert C.linked_of("c9") == ["route", "order", "reit", "conj", "disjunct"]
    assert C.linked_of("c4") == ["route", "order", "reit", "conj"] and C.linked_of("c0") == []


# ------------------------------------------------------------------------------------------------ pretrain
@pytest.mark.skipif(not HAS_C24, reason="c24 not built")
def test_pretrain_plan_and_refusals(lh):
    corpus(lh / "data" / "c24.jsonl.gz", real_records("c24", 50))
    _, steps, problems, _ = AL.plan(ns())
    assert any("no scaled step count" in p for p in problems)
    doc = steps_file(lh)
    _, steps, problems, notes = AL.plan(ns())
    a = steps[0]["argv"]
    assert (
        a[2] == "_pretrain"
        and a[a.index("--steps") + 1] == str(doc["steps"]["0"])
        and steps[0]["timeout_s"] == 3 * 3600
    )
    assert (
        not problems and any("scaled to" in n for n in notes) and any("census" in n for n in notes)
    )
    assert "lh_measure.py" in steps[1]["argv"][1]  # the sanity read through our measurement
    steps_file(lh, ("c9",))
    assert any("computed without c24" in p for p in AL.plan(ns())[2])
    with pytest.raises(SystemExit, match="never retrained"):
        AL.plan(ns(corpus="control_half"))
    # the extra base c0n: its own entry under `extra` in the same step file (PREREG Amendment 3, point 9)
    corpus(lh / "data" / "c0n.jsonl.gz", real_records("c0", 50))
    assert any("no scaled step count for c0n" in p for p in AL.plan(ns(corpus="c0n"))[2])
    e = LL.steps_doc({"authors_half": {"tokens": 126_508_088}, "c0n": {"tokens": 137_753_401}})
    LL.add_extra(lh / "data" / "pretrain_steps.json", "c0n", e)
    _, steps, problems, notes = AL.plan(ns(corpus="c0n"))
    a = steps[0]["argv"]
    assert (
        not problems
        and a[a.index("--steps") + 1] == "28000"
        and a[a.index("--corpus") + 1] == "c0n"
    )


def test_census_cap(tmp_path):
    r = {
        "prompt": "THM P SEQ P PRF",
        "proof": "N1 P : PR ; QED",
        "n_lines": 16,
        "notes": [],
        "author": "A",
        "lh_corpus": "c24",
    }
    fn = tmp_path / "c.jsonl.gz"
    corpus(fn, [r, {**r, "n_lines": 13}])
    cen = LL.census(fn)
    assert (
        not cen["problems"]
        and cen["over_12"] == 2
        and cen["n_lines_max"] == 16
        and cen["lh_corpus"] == {"c24": 2}
    )
    corpus(fn, [r, {**r, "n_lines": 17}])
    assert any("over 16 lines" in p for p in LL.census(fn)["problems"])


@pytest.mark.skipif(not HAS_C24, reason="c24 not built")
def test_loader_cap_16_and_no_drop(tmp_path):
    import tag_pretrain as TP

    long = real_records("c24", 3, lambda r: r["n_lines"] >= 15)
    bad = {"prompt": "THM P SEQ P PRF", "proof": "N1 P : PR ; N2 ( P & Q ) : ANDI N1 N1 ; QED", "n_lines": 2, "notes": [],
           "author": "A", "author_tag": "A", "tag_shown": True, "key": "bad"}  # fmt: skip
    fn = tmp_path / "c.jsonl.gz"
    corpus(fn, long)
    tok = T.install_dan()()
    TP._cfg.update(data=str(fn), limit=0, pairs={})
    with pytest.raises(AssertionError):
        LL._W2_LOAD(str(fn), tok, 12)  # W2's loader at tag_pretrain's cap refuses these records
    out, skipped = LL.lh_load(str(fn), tok, 12)
    assert len(out) == 3 and skipped == 0
    corpus(fn, [*long, bad])
    TP._cfg.update(data=str(fn), limit=0, pairs={})
    with pytest.raises(RuntimeError, match="no record may be dropped"):
        LL.lh_load(str(fn), tok, 12)
    with pytest.raises(RuntimeError, match="cap 6"):
        LL.lh_load(str(fn), tok, 6)


def test_pretrain_main_refusals(tmp_path, lh):
    with pytest.raises(SystemExit, match="never retrained"):
        LL.pretrain_main(
            ["--corpus", "authors_half", "--seed", "0", "--out", str(tmp_path / "x.pt")]
        )
    with pytest.raises(SystemExit, match="no scaled step count"):
        LL.pretrain_main(["--corpus", C.CORPORA[0], "--seed", "0", "--out", str(tmp_path / "x.pt")])
    steps_file(lh, C.CORPORA)
    with pytest.raises(SystemExit, match="trains"):
        LL.pretrain_main(
            [
                "--corpus",
                C.CORPORA[0],
                "--seed",
                "0",
                "--out",
                str(tmp_path / "x.pt"),
                "--steps",
                "150",
            ]
        )


def test_steps_rule():
    d = LL.steps_doc(
        {
            "authors_half": {"tokens": 1000},
            "c24": {"tokens": 1735},
            "c9": {"tokens": 1700},
            "c0": {"tokens": 1600},
        }
    )
    assert d["factor"] == 1.735 and d["line_cap"] == 16 and d["steps"]["0"] % 200 == 0
    assert (
        d["steps"]["0"] == 44600
        and d["steps"]["0"] >= 25607 * 1.735
        and d["old_steps"]["0"] == 25607
    )
    assert set(d["factor_per_corpus"]) == {"c24", "c9", "c0"}


def test_steps_extra_entry(tmp_path):
    """c0n: its own factor (equal passes over its own data), added under `extra`; the four corpora untouched."""
    f = tmp_path / "pretrain_steps.json"
    with pytest.raises(SystemExit, match="comes first"):
        LL.add_extra(f, "c0n", {})
    top = LL.steps_doc({"authors_half": {"tokens": 1000}, "c24": {"tokens": 1735}})
    f.write_text(json.dumps(top))
    e = LL.steps_doc(
        {"authors_half": {"tokens": 126_508_088}, "c0n": {"tokens": 137_753_401}}
    )  # the corpus agent's counts
    assert e["factor"] == pytest.approx(1.08889, abs=1e-6) and e["steps"]["0"] == 28000
    LL.add_extra(f, "c0n", e)
    doc = json.loads(f.read_text())
    assert doc["steps"] == top["steps"] and doc["extra"]["c0n"]["steps"]["0"] == 28000
    assert (
        LL.scaled_steps(0, "c0n", f)["steps"] == 28000
        and LL.scaled_steps(0, "c24", f)["steps"] == 44600
    )
    LL.add_extra(f, "c0n", e)  # the same entry again: nothing changes
    with pytest.raises(SystemExit, match="never overwritten"):
        LL.add_extra(
            f, "c0n", LL.steps_doc({"authors_half": {"tokens": 1000}, "c0n": {"tokens": 1200}})
        )


# ------------------------------------------------------------------------------------------------ measure, prep2, stage2, rl
def test_measure_plan(lh):
    job, steps, problems, notes = AL.plan(ns(cmd="measure", base="c24_s0"))
    assert job == "measure_c24_s0" and [s["name"] for s in steps] == [
        "measure_c24_s0",
        "strict106_c24_s0",
    ]
    assert steps[0]["argv"][1].endswith("lh_measure.py") and steps[0]["argv"][2] == "run"
    assert steps[1]["argv"][2:4] == ["strict106", "--ckpt"] and steps[1]["timeout_s"] == 3600
    assert any("no base" in p for p in problems)
    _, _, problems, _ = AL.plan(ns(cmd="measure", base="control_half_s0"))
    assert any("lh_launch.py nov14" in p for p in problems)
    (lh / "measure" / "c24_s0").mkdir(parents=True)
    (lh / "measure" / "c24_s0" / "strict106__none.jsonl").write_text("{}\n")
    assert any(
        "strict106__none.jsonl exists" in p for p in AL.plan(ns(cmd="measure", base="c24_s0"))[2]
    )
    _, steps, problems, _ = AL.plan(ns(cmd="measure", base="c24_s0", skip_done=True))
    assert not any("strict106" in p for p in problems) and "--resume" in steps[1]["argv"]


def test_prep2_plan(lh):
    c = C.CORPORA[0]
    job, steps, problems, _ = AL.plan(ns(cmd="prep2", corpus=c))
    assert job == f"prep2_{c}_s0" and [s["name"] for s in steps][:2] == [
        f"anchor_{c}",
        f"sample_{c}",
    ]
    assert [s["name"] for s in steps][2:] == [f"build_{c}_{r}" for r in C.raters_of(c)]
    for s in steps[2:]:  # probe twins only where a reward model is fitted (Amendments 4 and 6)
        r = s["name"].split("_")[-1]
        assert ("--no_probe" in s["argv"]) == (not C.has_rm(c, r))
    if "c9" in C.CORPORA:
        (s9,) = [
            s for s in AL.plan(ns(cmd="prep2", corpus="c9"))[1] if s["name"].startswith("build_")
        ]
        assert (
            "--no_probe" in s9["argv"]
        )  # no GRPO line on c9, so no reward model and no probe twins
    assert any("no base" in p for p in problems)
    C.anchor_path(c).parent.mkdir(parents=True, exist_ok=True)
    C.anchor_path(c).write_text("{}\n")
    (C.pairs_dir(c)).mkdir(parents=True)
    (C.pairs_dir(c) / "sample.json").write_text("{}")
    assert [s["name"] for s in AL.plan(ns(cmd="prep2", corpus=c))[1]] == [
        f"build_{c}_{r}" for r in C.raters_of(c)
    ]
    for r in C.raters_of(c):
        C.pairs_dir(c, r).mkdir(parents=True)
        (C.pairs_dir(c, r) / "summary.json").write_text("{}")
    assert any("everything exists" in p for p in AL.plan(ns(cmd="prep2", corpus=c))[2])


def test_stage2_plan(lh, monkeypatch):
    c = C.CORPORA[0]
    monkeypatch.setenv("ND_GUARD", "off")
    job, (s,), problems, _ = AL.plan(ns(cmd="stage2", corpus=c))
    tasks = C.tasks_of(c)
    assert (
        job == f"stage2_{c}_s0"
        and s["argv"][2] == "_tasks"
        and s["argv"][s["argv"].index("--tasks") + 1] == ",".join(tasks)
    )
    # the backstop: twice the estimate + 3 h, at least the longest task's cap + 1 h, at most every task at its cap
    # (GRPO lines alone, the rest two at a time)
    alone = [x for x in tasks if x.startswith("rl_")]
    rest = [x for x in tasks if x not in alone]
    est = sum(LL.task_est(x) for x in alone) + sum(LL.task_est(x) for x in rest) * 1.25 / 2
    longest = max(LL.task_hours(x) for x in tasks)
    worst = (
        sum(LL.task_hours(x) for x in alone)
        + sum(LL.task_hours(x) for x in rest) / 2
        + longest
        + 0.5
    )
    want = round(min(worst, max(2 * est + 3, longest + 1)), 1)
    assert s["timeout_s"] == int(want * 3600) and longest + 1 <= want <= worst
    drv = float(re.search(r"MAX_H_STAGE2:-([0-9.]+)", (HERE / "lh_driver.sh").read_text()).group(1))
    assert want <= drv - 0.5  # the driver's default wait for stage 2 stays above the job's cap
    assert {p.split(":")[0] for p in problems} == set(
        tasks
    )  # no base, no pairs: every task says why
    with pytest.raises(SystemExit, match="not a stage-2 task"):
        AL.plan(ns(cmd="stage2", corpus=c, tasks="rl_zz_anc_s0"))
    monkeypatch.delenv("ND_GUARD")
    assert any("ND_GUARD=off" in p for p in AL.plan(ns(cmd="stage2", corpus=c))[2])
    # the fine-tune pair without its normalizer is refused at its turn only: never a reason to refuse the launch
    monkeypatch.setenv("ND_GUARD", "off")
    sft = C.arm_name("sft", c, "reit")
    monkeypatch.setattr(
        LL, "task_problems", lambda x: [f"{x}: no norm_tokens{LL.AT_RUN}"] if x == sft else []
    )
    _, _, problems, notes = AL.plan(ns(cmd="stage2", corpus=c))
    assert problems == [] and any(n.startswith(f"WILL BE REFUSED: {sft}") for n in notes)


def test_rl_plan(lh, monkeypatch):
    c = C.CORPORA[0]
    monkeypatch.setenv("ND_GUARD", "off")
    line = C.line_name(c, "rlhfreit")
    _, (s,), problems, notes = AL.plan(ns(cmd="rl", line=line))
    assert (
        s["argv"][2] == "_line"
        and s["argv"][s["argv"].index("--iters") + 1] == "40"
        and s["timeout_s"] == 3 * 3600
    )
    assert any("40 iterations" in n for n in notes) and not any("60 iterations" in n for n in notes)
    assert any("no anchor file" in p for p in problems) and any(
        p.startswith("no reward model") for p in problems
    )
    assert any("step limit 51" in n for n in notes)
    with pytest.raises(SystemExit, match="not a line"):
        AL.plan(ns(cmd="rl", line="rl_c24_styA_s0"))


def test_snap_cmd_and_seg_class():
    cmd = AL.snap_cmd("rl_x", 1, Path("/c.pt"), 10, 8, "marker_plain,route_unseen", False, 0, 0)
    assert (
        cmd[1] == str(HERE / "lh_measure.py")
        and cmd[2] == "run"
        and cmd[cmd.index("--tags") + 1] == "none"
        and cmd[cmd.index("--sets") + 1] == "marker_plain"
    )  # untagged, marker_plain only (Amendment 8 point 6)
    Seg, TLN = AL.make_seg_class()
    assert Seg.__name__ == "LHSeg"
    seg = Seg.__new__(Seg)  # env() only: tag_line's fixed values, then the task's share of the CPUs
    seg.a = argparse.Namespace(smoke=True, gpu=0)
    seg.dir, seg.tag = Path("/x"), "none"
    for k in ("LH_TASK_WORKERS", "LH_TASK_OMP"):
        os.environ.pop(k, None)
    assert seg.env()["LEAN_GATE_WORKERS"] == "8"
    os.environ.update(LH_TASK_WORKERS="3", LH_TASK_OMP="1")
    try:
        e = seg.env()
        assert (
            e["LEAN_GATE_WORKERS"] == "3"
            and e["OMP_NUM_THREADS"] == "1"
            and LL.gpu_env(0, False)["LEAN_GATE_WORKERS"] == "3"
        )
    finally:
        for k in ("LH_TASK_WORKERS", "LH_TASK_OMP"):
            del os.environ[k]


def test_stage2_estimate_runs_lines_alone(lh, monkeypatch):
    c = C.CORPORA[0]
    monkeypatch.setattr(LL, "task_problems", lambda x: [])
    monkeypatch.setenv("ND_GUARD", "off")
    ns = argparse.Namespace(corpus=c, gpu=0, parallel=2, stagger_s=120.0, tasks=None)
    _, steps, _, notes = LL.stage2_plan(ns)
    lines = [x for x in C.tasks_of(c) if x.startswith("rl_")]
    rest = [x for x in C.tasks_of(c) if not x.startswith("rl_")]
    want = sum(LL.task_est(x) for x in lines) + 1.25 * sum(LL.task_est(x) for x in rest) / 2
    assert abs(steps[0]["est_h"] - round(want, 2)) < 0.011 and "alone: " in notes[0]
    assert LL.task_weight(lines[0], 2) == 2 and LL.task_weight(rest[0], 2) == 1


def test_task_threads(monkeypatch):
    assert 1 <= LL.cpu_quota() <= os.cpu_count()
    monkeypatch.setattr(LL, "cpu_quota", lambda: 7)
    t = LL.task_threads(
        2, {"LEAN_GATE_WORKERS": "8", "OMP_NUM_THREADS": "8"}
    )  # a pod: W2's job sets 8, the split wins
    assert t == {
        "LEAN_GATE_WORKERS": "3",
        "OMP_NUM_THREADS": "3",
        "LH_TASK_WORKERS": "3",
        "LH_TASK_OMP": "3",
    }
    assert (
        LL.task_threads(1, {})["LEAN_GATE_WORKERS"] == "7"
        and LL.task_threads(4, {})["LEAN_GATE_WORKERS"] == "2"
    )
    assert (
        LL.task_threads(2, {"OMP_NUM_THREADS": "1"})["OMP_NUM_THREADS"] == "1"
    )  # never raised (serrano's CPU limit)
    assert LL.task_threads(2, {"LH_TASK_THREADS": "5"})["LEAN_GATE_WORKERS"] == "5"


@pytest.mark.parametrize("kind", ["anc", "rlhfreit"])
def test_line_dry_run_routes_the_grpo_process(kind, capsys, monkeypatch):
    monkeypatch.setenv("ND_GUARD", "off")
    line = C.line_name(C.CORPORA[0], kind)
    import lh_rlvar  # noqa: F401
    import tag_line as TLN

    # W2's line_main wraps tag_line's argv builders in its process (once per process); undo it after the test
    monkeypatch.setattr(TLN, "grpo_argv", TLN.grpo_argv)
    monkeypatch.setattr(TLN, "grpo_cmd", TLN.grpo_cmd)

    _, (st,), _, _ = AL.plan(
        LL.rl_ns(argparse.Namespace(gpu=0), line)
    )  # the line step as the task runs it
    i = st["argv"].index("_line")
    assert (
        st["argv"][st["argv"].index("--iters") + 1] == "40"
    )  # Amendment 6 (W2's variant lines: 60)
    AL.line_main([*st["argv"][i + 1 :], "--dry-run"])
    info = json.loads(capsys.readouterr().out)
    g = info["grpo"]
    assert g[g.index("--iters") + 1] == "40"
    assert g[1] == str(HERE / "lh_rlvar.py") and g[2:5] == ["grpo", "--var", kind]
    assert g[g.index("--max_steps") + 1] == "51" and g[g.index("--kl_anchor_coef") + 1] == "0.02"
    assert (
        g[g.index("--snap_iters") + 1] == "10,20,30,40" and g[g.index("--unlikely_beta") + 1] == "0"
    )
    assert (
        g[g.index("--kl_anchor_data") + 1] == str(C.anchor_path(C.CORPORA[0]))
        and "--rollback_on_alarm" not in g
    )


# ------------------------------------------------------------------------------------------------ stage-2 steps
def test_trainer_and_fit_argv(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "LH", tmp_path)
    c = C.CORPORA[0]
    grid = ",".join(map(str, C.SNAP_STEPS))
    a = LL.trainer_argv(C.arm_name("dpo", c, "reit", 1))
    assert (
        a[1].endswith("lh_dpo.py")
        and a[a.index("--pairs") + 1].endswith("reit/train.jsonl")
        and "--swap-random" not in a
    )
    assert (
        a[a.index("--kl-states") + 1] == str(C.kl_states_path(c))
        and a[a.index("--device") + 1] == "cuda"
        and a[a.index("--max-tokens") + 1] == "32768"  # two jobs on one 24 GB card (TUNING.md)
    )
    assert (
        a[a.index("--lr") + 1] == str(C.OFFLINE_LR) == "3e-05"  # Amendment 7 point 1
        and a.count("--snap-steps") == a.count("--max-steps") == 1
    )
    assert (
        C.SNAP_STEPS[:5] == (5, 10, 15, 20, 25) and C.SNAP_STEPS[-1] == C.OFFLINE_MAX_STEPS == 400
    )
    assert a[a.index("--seed") + 1] == "1" and a[a.index("--base") + 1] == str(
        C.base_path(c, 0)
    )  # every run from the s0 base
    assert (
        a[a.index("--snap-steps") + 1] == grid
        and a[a.index("--max-steps") + 1] == "400"
        and "--match-steps" not in a
    )
    s = LL.trainer_argv(C.arm_name("dposwap", c, "reit"), smoke=True)
    assert (
        "--swap-random" in s
        and "--max-tokens" not in s
        and s[s.index("--device") + 1] == "cpu"
        and s[s.index("--snap-steps") + 1] == grid
    )
    f = LL.trainer_argv(C.arm_name("sft", c, "reit"))
    g = LL.trainer_argv(C.arm_name("sftsam", c, "reit"))
    assert f[1].endswith("lh_sft.py") and f[f.index("--data") + 1].endswith("reit/sft_pref.jsonl")
    assert g[1].endswith("lh_sft.py") and g[g.index("--data") + 1].endswith(
        "reit/sft_sampled.jsonl"
    )
    r = LL.rm_argv(c, "reit")
    assert (
        r[1].endswith("lh_rm.py")
        and r[2] == "fit"
        and r[r.index("--out") + 1] == str(C.rm_path(c, "reit"))
        and r[r.index("--max-tokens") + 1] == "32768"
    )
    p = LL.rm_probe_argv(c, "reit")
    assert (
        p[2] == "probe"
        and p[p.index("--pairs") + 1].endswith("reit/probe_twins.jsonl")
        and p[p.index("--out") + 1] == str(C.rm_probe_path(c, "reit"))
    )


def test_swap_seed_and_norm_tokens(tmp_path, monkeypatch):
    """Amendment 7 points 3 and 4: dposwap's --swap-seed is the run's seed; sft and sftsam get the same --norm-tokens
    and --batch from the pair build's summary; without them (an older build, or lh_sft without the flag) refused."""
    monkeypatch.setattr(C, "LH", tmp_path)
    c = C.CORPORA[0]
    for s_ in (0, 1):
        a = LL.trainer_argv(C.arm_name("dposwap", c, "reit", s_))
        assert a[a.index("--swap-seed") + 1] == str(s_) == a[a.index("--seed") + 1]
    assert "--swap-seed" not in LL.trainer_argv(C.arm_name("dpo", c, "reit", 1))
    assert "--norm-tokens" not in LL.trainer_argv(C.arm_name("sft", c, "reit"))  # no summary yet
    sm = LL.pair_files(c, "reit")["summary"]
    sm.parent.mkdir(parents=True)
    sm.write_text(json.dumps({"norm_tokens": {"N": 1234, "batch": 64}}))
    f, g = (
        LL.trainer_argv(C.arm_name("sft", c, "reit")),
        LL.trainer_argv(C.arm_name("sftsam", c, "reit")),
    )
    for x in (f, g):
        assert x[x.index("--norm-tokens") + 1] == "1234" and x[x.index("--batch") + 1] == "64"
    monkeypatch.setattr(LL, "detector_problems", lambda: [])
    has = "--norm-tokens" in (HERE / "lh_sft.py").read_text()
    assert any(
        "has no --norm-tokens" in p for p in LL.task_problems(C.arm_name("sft", c, "reit"))
    ) == (not has)
    sm.write_text(json.dumps({"pairs": 3}))
    p = [q for q in LL.task_problems(C.arm_name("sftsam", c, "reit")) if "no norm_tokens" in q]
    assert p and p[0].endswith(LL.AT_RUN)  # refused at its turn; it never stops the stage-2 launch


def test_step0_check_stops_a_dpo_run(lh, monkeypatch):
    """Amendment 7 point 7: a DPO run whose first training row is not 0.6931 +- 0.001 is stopped, marked and listed in
    <c>.FAILED with the reason; a run that never logged a row is an ordinary failure."""
    c = C.CORPORA[0]
    monkeypatch.setattr(C, "POD", lh / "pod")
    arm = C.arm_name("dpo", c, "reit")

    def trainer(loss):
        row = json.dumps({"step": 1, "loss": loss, "train_margin_rel": 0.3})
        return lambda a, smoke=False: [PY, "-c", f"import sys; from pathlib import Path; d = Path({str(C.arm_dir(a))!r}); "
                                                  f"(d / 'train.jsonl').write_text({row!r} + '\\n'); sys.exit(4)"]  # fmt: skip

    monkeypatch.setattr(LL, "trainer_argv", trainer(0.7012))
    assert LL.offline_main(["--arm", arm, "--gpu", "0"]) == 5
    rec = json.loads((C.arm_dir(arm) / "step0_failed.json").read_text())
    assert "loss 0.7012" in rec["reason"] and rec["train_rc"] == 4
    failed = (lh / "pod" / f"{c}.FAILED").read_text()
    assert f"stage2 {arm} STOPPED: step-0 check: first training row (step 1) loss 0.7012" in failed
    assert not (C.arm_dir(arm) / "probe").exists()
    assert LL.step0_problem(arm) is not None
    (C.arm_dir(arm) / "step0_failed.json").unlink()
    (C.arm_dir(arm) / "train.jsonl").write_text(json.dumps({"step": 1, "loss": 0.69318}) + "\n")
    assert LL.step0_problem(arm) is None  # within 0.001
    # the trainer's own verdict (lh_dpo: step0_ok on the first row, its step0_failed.json) is read and never overwritten
    (C.arm_dir(arm) / "train.jsonl").write_text(
        json.dumps({"step": 1, "loss": 0.6935, "step0_loss": 0.6935, "step0_ok": False}) + "\n"
    )
    assert "the trainer's step-0 check failed" in LL.step0_problem(arm)
    (C.arm_dir(arm) / "train.jsonl").write_text("")
    (C.arm_dir(arm) / "step0_failed.json").write_text('{"by": "lh_dpo"}')
    assert "lh_dpo's" not in LL.step0_problem(arm) and LL.step0_problem(arm).endswith(
        "step0_failed.json)"
    )
    arm2 = C.arm_name("dposwap", c, "reit")
    monkeypatch.setattr(
        LL, "trainer_argv", lambda a, smoke=False: [PY, "-c", "import sys; sys.exit(1)"]
    )
    assert (
        LL.offline_main(["--arm", arm2, "--gpu", "0"]) == 1
        and not (C.arm_dir(arm2) / "step0_failed.json").exists()
    )


FAKE_TRAINER = """
import json, sys
from pathlib import Path
a = sys.argv[1:]
d = Path(a[a.index('--out') + 1])
for f in ('snap_step25.pt', 'snap_step50.pt', 'final.pt'):
    (d / f).write_text('x')
(d / 'done.json').write_text(json.dumps({'final': {'step': 60}}))
"""


def test_offline_runs_train_and_only_rated_runs_probe(lh, monkeypatch):
    c = C.CORPORA[0]
    arm, ctl = C.arm_name("dpo", c, "reit"), C.arm_name("dposwap", c, "reit")
    calls = []
    monkeypatch.setattr(
        LL,
        "trainer_argv",
        lambda a, smoke=False: [PY, "-c", FAKE_TRAINER, "--out", str(C.arm_dir(a))],
    )

    def fake_probe(a, snaps, smoke):
        calls.append((a, [x["label"] for x in snaps]))
        return [
            PY,
            "-c",
            f"open({str(C.arm_dir(a) / 'probe' / 'probe.json')!r}, 'w').write('{{}}')",
        ]

    monkeypatch.setattr(LL, "probe_argv", fake_probe)
    assert LL.offline_main(["--arm", arm, "--gpu", "0"]) == 0
    d = C.arm_dir(arm)
    assert calls == [(arm, ["step25", "step50", "final"])] and (d / "probe" / "probe.json").exists()
    rec = json.loads((d / "lh_offline.json").read_text())
    assert rec["train_rc"] == 0 and rec["probe_rc"] == 0 and 100 in rec["grid_missing"]
    assert [x["step"] for x in LL.snapshots_of(arm)] == [25, 50, 60]
    calls.clear()
    assert (
        LL.offline_main(["--arm", arm, "--gpu", "0"]) == 0 and calls == []
    )  # trained and probed: nothing again
    assert any(": already trained and probed" in p for p in LL.task_problems(arm))
    assert (
        LL.offline_main(["--arm", ctl, "--gpu", "0"]) == 0 and calls == []
    )  # a no-preference run is never probed
    assert "not probed" in json.loads((C.arm_dir(ctl) / "lh_offline.json").read_text())["probe"]
    assert any(p.endswith(f": already trained ({C.arm_dir(ctl)})") for p in LL.task_problems(ctl))
    arm2 = C.arm_name("sft", c, "reit")
    C.arm_dir(arm2).mkdir(parents=True)
    (C.arm_dir(arm2) / "metrics.jsonl").write_text("{}\n")
    assert LL.offline_main(["--arm", arm2, "--gpu", "0"]) == 3  # a partial run is refused


def test_probe_argv_stop_rule(lh):
    c = C.CORPORA[0]
    arm = C.arm_name("dpo", c, "reit")
    snaps = [{"label": "step25", "path": Path("a.pt")}, {"label": "final", "path": Path("b.pt")}]
    assert "--stop_above" not in LL.probe_argv(
        arm, snaps, False
    )  # no base share: every snapshot is probed
    (lh / "measure" / f"{c}_s0").mkdir(parents=True)
    LL.base_summary(c).write_text(json.dumps({"lh_totals": {"a_share": {"reit": 40.0}}}))
    a = LL.probe_argv(arm, snaps, False)
    assert (
        a[a.index("--stop_habit") + 1] == "reit"
        and a[a.index("--stop_above") + 1] == "70.0"
        and a[a.index("--k") + 1] == "8"
    )


def test_select_steps():
    sel = LL.select_steps(40.0, [(25, 45.0), (50, 60.0), (100, 75.0), (2000, 95.0)], 2000)
    assert (
        sel["reached"]
        and sel["bracket"] == [50, 100]
        and sel["bracket_moves"] == [20.0, 35.0]
        and sel["measure"] == [50, 100, 2000]
    )
    sel = LL.select_steps(40.0, [(25, 70.0), (50, 71.0), (100, 75.0)], 100)
    assert sel["bracket"] == [25, 50] and sel["measure"] == [
        25,
        50,
        100,
    ]  # exactly +30 is not "more than 30"
    sel = LL.select_steps(40.0, [(25, 80.0), (50, 90.0)], 50)
    assert sel["bracket"] == [0, 25] and sel["measure"] == [
        25,
        50,
    ]  # crossed before the first snapshot: the base below
    grid = list(C.SNAP_STEPS)
    sel = LL.select_steps(40.0, [(s, 45.0) for s in grid], 400, steps=grid)
    assert (
        not sel["reached"]
        and sel["bracket"] == [300, 400]
        and sel["measure"] == [300, 400]
        and sel["max_move"] == 5.0
    )
    assert (
        grid == [5, 10, 15, 20, 25, 30, 40, 50, 75, 100, 150, 200, 300, 400]
        and C.OFFLINE_MAX_STEPS == 400
    )
    # the probe's stop on the new grid: past +30 at step 15 -> bracket [10, 15], the last (400) beside it
    sel = LL.select_steps(40.0, [(5, 50.0), (10, 62.0), (15, 75.0), (400, 95.0)], 400, steps=grid)
    assert sel["bracket"] == [10, 15] and sel["measure"] == [10, 15, 400]


def test_full_task(lh, monkeypatch):
    c = C.CORPORA[0]
    sh = {
        25: 45.0,
        50: 60.0,
        100: 75.0,
        150: None,
    }  # the probe stopped after step 100 (75 > 40 + 30); 150 not probed

    def setup(task: str) -> tuple[str, str]:
        rated, ctl = C.depends_on(task)
        for arm in (rated, ctl):
            d = C.arm_dir(arm)
            d.mkdir(parents=True)
            for st in sh:
                (d / f"snap_step{st}.pt").write_text("x")
            (d / "final.pt").write_text("x")
            (d / "done.json").write_text(json.dumps({"final": {"step": 2000}}))
        (C.arm_dir(rated) / "probe").mkdir()
        snaps = {
            f"step{k}": {"a_share": {"reit": v, "note_atom": v}}
            for k, v in sh.items()
            if v is not None
        }
        (C.arm_dir(rated) / "probe" / "probe.json").write_text(
            json.dumps(
                {"snapshots": {**snaps, "final": {"a_share": {"reit": 95.0, "note_atom": 95.0}}}}
            )
        )
        return rated, ctl

    task = C.full_name("dpo", c, "reit", 0)
    rated, ctl = setup(task)
    assert any("not measured" in p for p in LL.task_problems(task))
    assert LL.full_main(["--task", task]) == 3  # no base share yet
    (lh / "measure" / f"{c}_s0").mkdir(parents=True)
    LL.base_summary(c).write_text(
        json.dumps({"lh_totals": {"a_share": {"reit": 40.0, "note_atom": 40.0}}})
    )
    calls = []
    monkeypatch.setattr(
        LL,
        "measure_argv",
        lambda ck, cid, out, raw, smoke, tags="A,B,none", sets=C.SETS: (
            calls.append((cid, tags)),
            sets_seen.add(tuple(sets)),
            [PY, "-c", "pass"],
        )[2],
    )
    sets_seen: set = set()
    assert LL.full_main(["--task", task]) == 0
    # tags A, B, none on the UPPER bracketing snapshot (Amendment 7 point 2); marker_plain only (Amendment 8 point 6)
    assert calls == [(f"{rated}@step50", "none"), (f"{rated}@step100", "A,B,none"), (f"{rated}@final", "none"),
                     (f"{ctl}@step50", "none"), (f"{ctl}@step100", "none"), (f"{ctl}@final", "none")]  # fmt: skip
    assert sets_seen == {("marker_plain",)}
    rec = json.loads(LL.full_path(task).read_text())
    assert (
        rec["done"]
        and rec["bracket"] == [50, 100]
        and rec["base_share"] == 40.0
        and rec["tags"][f"{rated}@step100"] == "A,B,none"
    )
    assert any(": already measured" in p for p in LL.task_problems(task))
    calls.clear()
    assert LL.full_main(["--task", task]) == 0 and calls == []  # measured before
    task2 = C.full_name("dpo", c, "atom", 0)  # another rater: untagged everywhere
    setup(task2)
    assert LL.full_main(["--task", task2]) == 0 and {t for _, t in calls} == {"none"}


def test_scheduler_and_dependency(lh, monkeypatch):
    """At most --parallel tasks at once; a full measurement starts only after both of its runs ended; every rc."""
    trace = lh / "trace.txt"
    c = C.CORPORA[0]
    tasks = [C.line_name(c, "anc"), C.arm_name("dpo", c, "reit"), C.arm_name("dposwap", c, "reit"),
             C.full_name("dpo", c, "reit", 0), C.arm_name("sft", c, "reit")]  # fmt: skip
    rcs = dict.fromkeys(tasks, 0) | {tasks[4]: 5}
    dur = {tasks[0]: 0.3, tasks[1]: 1.0, tasks[2]: 0.3, tasks[3]: 0.3, tasks[4]: 0.3}
    monkeypatch.setattr(LL, "task_problems", lambda x: [])

    def fake_cmd(x, gpu, extra=""):
        code = (f"import time; open({str(trace)!r}, 'a').write('s {x} %f\\n' % time.time()); time.sleep({dur[x]}); "
                f"open({str(trace)!r}, 'a').write('e {x} %f\\n' % time.time()); raise SystemExit({rcs[x]})")  # fmt: skip
        return [PY, "-c", code]

    monkeypatch.setattr(LL, "task_cmd", fake_cmd)
    args = ["--tasks", ",".join(tasks), "--gpu", "0", "--stagger_s", "0.05", "--poll_s", "0.05"]
    assert LL.tasks_main([*args, "--parallel", "2"]) == 1  # one task ended rc 5
    ev = sorted(
        (float(t), k, x) for k, x, t in (ln.split() for ln in trace.read_text().splitlines())
    )
    running, peak = 0, 0
    for _, k, _ in ev:
        running += 1 if k == "s" else -1
        peak = max(peak, running)
    assert peak == 2
    t = {(k, x): tt for tt, k, x in ev}
    # the GRPO line ran alone (lh_core.LINE_ALONE): nothing started before it ended
    assert all(tt >= t[("e", tasks[0])] for tt, k, x in ev if k == "s" and x != tasks[0])
    assert t[("s", tasks[3])] >= max(
        t[("e", tasks[1])], t[("e", tasks[2])]
    )  # the full measurement after both runs
    assert (
        t[("s", tasks[4])] < t[("s", tasks[3])]
    )  # sft went ahead while the full measurement waited
    assert (lh / "logs" / f"{tasks[0]}.tasks.out").exists()
    trace.unlink()
    rcs[tasks[4]] = 0
    assert LL.tasks_main([*args, "--parallel", "1"]) == 0
    trace.unlink()
    monkeypatch.setattr(
        C, "LINE_ALONE", False
    )  # a line may share: it runs beside the first offline run
    assert LL.tasks_main([*args, "--parallel", "2"]) == 0
    ev = sorted(
        (float(tt), k, x) for k, x, tt in (ln.split() for ln in trace.read_text().splitlines())
    )
    t = {(k, x): tt for tt, k, x in ev}
    assert t[("s", tasks[1])] < t[("e", tasks[0])]
    monkeypatch.setattr(LL, "task_problems", lambda x: ["refused"] if x == tasks[1] else [])
    trace.unlink()
    assert LL.tasks_main([*args, "--parallel", "2"]) == 1
    assert (
        f"s {tasks[3]}" in trace.read_text()
    )  # its rated run was refused (ended at once): the full task still runs


def test_task_line_fits_and_probes_the_rm_first(lh, monkeypatch):
    calls = []
    monkeypatch.setattr(LL, "rm_main", lambda argv: (calls.append(("rm", argv[1], argv[3])), 0)[1])
    monkeypatch.setattr(
        LL, "rm_probe_main", lambda c, r, gpu, smoke: (calls.append(("probe", c, r)), 1)[1]
    )  # fails: not a gate
    monkeypatch.setattr(
        LL, "plan", lambda a: ("j", [AL.step("line", [PY, "-c", "pass"], 1.0, 0)], [], [])
    )
    monkeypatch.setattr(
        LL,
        "run_step",
        lambda name, argv, hours, env=None, log=None: (calls.append(("run", name, env)), 0)[1],
    )
    c = C.CORPORA[0]
    assert LL.task_main(["--task", C.line_name(c, "rlhfreit"), "--gpu", "0"]) == 0
    assert (
        calls[0] == ("rm", c, "reit")
        and calls[1] == ("probe", c, "reit")
        and calls[2][0] == "run"
        and calls[2][2] == {"ND_GUARD": "off"}
    )
    calls.clear()
    assert LL.task_main(["--task", C.line_name(c, "anc"), "--gpu", "0"]) == 0 and [
        x[0] for x in calls
    ] == ["run"]
    monkeypatch.setattr(LL, "rm_main", lambda argv: 4)
    assert (
        LL.task_main(["--task", C.line_name(c, "rlhfreit"), "--gpu", "0"]) == 4
    )  # no line without its reward model


def test_rm_under_090_is_not_run_and_never_retried(lh, monkeypatch):
    """lh_rm.py fit exits 3 with RM.pt.failed under 0.90: the rlhf line is decided not run (a record), the task ends 0,
    no refit on a rerun, task_problems says so, the scheduler and the driver list it apart from failures."""
    c = C.CORPORA[0]
    line = C.line_name(c, "rlhfreit")
    rm = C.rm_path(c, "reit")
    fits = []

    def fake_run(name, argv, hours, env=None, log=None):
        if name.startswith("rm fit"):
            fits.append(argv)
            Path(f"{rm}.failed").write_text("x")
            (rm.parent / "rm.pt.json").write_text(json.dumps({"heldout_acc": 0.81}))
            return 3
        raise AssertionError(f"no line may run: {name}")

    monkeypatch.setattr(LL, "run_step", fake_run)
    monkeypatch.setattr(
        LL, "rm_probe_main", lambda *x: (_ for _ in ()).throw(AssertionError("no probe"))
    )
    assert LL.task_main(["--task", line, "--gpu", "0"]) == 0 and len(fits) == 1
    rec = json.loads(C.not_run_path(line).read_text())
    assert rec["heldout_acc"] == 0.81 and "0.9" in rec["reason"] and rec["fit_rc"] == 3
    assert LL.task_main(["--task", line, "--gpu", "0"]) == 0 and len(fits) == 1  # never refitted
    assert any(": already decided not to run" in p for p in LL.task_problems(line))
    # a crash is not a decision: no record, the next run fits again
    C.not_run_path(line).unlink()
    Path(f"{rm}.failed").unlink()
    monkeypatch.setattr(
        LL, "run_step", lambda name, argv, hours, env=None, log=None: (fits.append(argv), 1)[1]
    )
    assert LL.rm_main(["--corpus", c, "--rater", "reit"]) == 1 and not C.not_run_path(line).exists()


def test_merge_argv_never_repeats_an_option():
    a = ["x", "--max-steps", "2000", "--snap-steps", "2,5", "--device", "cpu"]
    m = LL.merge_argv(
        a, ["--max-steps", "6", "--snap-steps", "2,4", "--batch", "8", "--swap-random"]
    )
    assert m == [
        "x",
        "--max-steps",
        "6",
        "--snap-steps",
        "2,4",
        "--device",
        "cpu",
        "--batch",
        "8",
        "--swap-random",
    ]
    assert m.count("--max-steps") == 1 and LL.merge_argv(a, []) == a


def test_rm_probe_step(lh, monkeypatch):
    c = C.CORPORA[0]
    assert LL.rm_probe_main(c, "reit", 0, False) == 1  # no probe twins: not probed
    tw = LL.pair_files(c, "reit")["probe_twins"]
    tw.parent.mkdir(parents=True)
    tw.write_text('{"habit": "note_atom"}\n')
    out = C.rm_probe_path(c, "reit")
    out.parent.mkdir(parents=True)
    monkeypatch.setattr(
        LL,
        "rm_probe_argv",
        lambda c, r, seed=0, smoke=False: [PY, "-c", f"open({str(out)!r}, 'w').write('{{}}')"],
    )
    assert LL.rm_probe_main(c, "reit", 0, False) == 0 and out.exists()


# ------------------------------------------------------------------------------------------------ wait, dry run, job runner
def test_status_with_the_driver_lock_directory(lh, capsys, monkeypatch):
    """The operator's bug: `status` read LH/pod/<c>.driver.lock, a directory (IsADirectoryError)."""
    monkeypatch.setattr(C, "POD", lh / "pod")
    (lh / "pod" / "c24.driver.lock").mkdir(parents=True)
    (lh / "pod" / "c24.driver.lock" / "pid").write_text("123")
    (lh / "pod" / "c24.GO").write_text("now")
    LL.status()
    out = capsys.readouterr().out
    assert "pod dir c24.driver.lock/ (1 entries)" in out and "pod file c24.GO: now" in out


def test_wait(lh):
    (lh / "logs").mkdir()
    log = lh / "logs" / "measure_c24_s0_20261127T010203Z.log"
    log.write_text("x\n")
    hb = lh / "pod" / "c24.heartbeat.json"
    t0 = time.time()
    assert (
        LL.wait_main(
            [
                "--job",
                "measure_c24_s0",
                "--max_h",
                str(2 / 3600),
                "--every",
                "0.5",
                "--heartbeat",
                str(hb),
                "--stage",
                "measure",
            ]
        )
        == 124
    )
    assert time.time() - t0 < 10 and json.loads(hb.read_text())["stage"] == "measure"
    log.write_text("x\n2026-11-27T01:30:00Z job end rc 0; lock released\n")
    (lh / "logs" / "measure_c24_s0_20261126T010203Z.log").write_text(
        "job end rc 1; lock released\n"
    )  # older: ignored
    assert LL.wait_main(["--job", "measure_c24_s0", "--every", "0.1"]) == 0
    log.write_text("job end rc 124; lock released\n")
    assert LL.wait_main(["--job", "measure_c24_s0", "--every", "0.1"]) == 124
    assert LL.wait_main(["--job", "pretrain_c24_s0", "--every", "0.1", "--no_log_s", "0.3"]) == 2


def test_dry_run_writes_nothing_and_lab_host_refuses(lh, capsys):
    before = sorted(str(p) for p in lh.rglob("*"))
    AL.launch(ns(cmd="measure", base="c24_s0"))
    out = capsys.readouterr().out
    assert "nothing is launched or written" in out and ("lab host" in out or AL.HOST not in TL.LAB)
    assert sorted(str(p) for p in lh.rglob("*")) == before


def test_run_job_lock_timeout_log(lh, monkeypatch):
    monkeypatch.setattr(TL.G, "free_for_us", lambda host, gpu: True)
    log = lh / "logs" / "job_x.log"
    log.parent.mkdir(parents=True)
    spec = {"job": "job_x", "gpu": 0, "log": str(log), "status": str(log.with_suffix(".status.json")),
            "steps": [AL.step("ok", [PY, "-c", "print('hello')"], 0.01, 0.0),
                      AL.step("slow", [PY, "-c", "import time; time.sleep(30)"], 1 / 3600, 0.0)]}  # fmt: skip
    sp = lh / "logs" / "job_x.spec.json"
    sp.write_text(json.dumps(spec))
    assert TL.run_job(sp) == 124
    st = json.loads(log.with_suffix(".status.json").read_text())
    assert [s["rc"] for s in st["steps"]] == [0, 124] and "job end rc 124" in log.read_text()


def test_nov14_copies(tmp_path, monkeypatch):
    src = tmp_path / "src"
    src.mkdir()
    for c in C.NOV14:
        for s in (".pt", ".tag.json", ".pt.args.json"):
            (src / f"stage1_{c}_s0{s}").write_text(c + s)
    monkeypatch.setattr(C, "LH", tmp_path / "lh")
    assert len(LL.nov14_main(["--src", str(src)])) == 6
    assert (
        tmp_path / "lh" / "bases" / "stage1_control_half_s0.pt"
    ).read_text() == "control_half.pt"
    assert LL.nov14_main(["--src", str(src)]) == []
    (src / "stage1_authors_half_s0.pt").write_text("changed")
    with pytest.raises(SystemExit, match="never overwritten"):
        LL.nov14_main(["--src", str(src)])


# ------------------------------------------------------------------------------------------------ the second base pass
FROZEN = Path(
    "/scratch/cowifornia/nd-rl/worktrees/linked-habits-frozen2/code/experiments/current/linked_habits_20261127"
)
NAMES_SCRIPT = r"""
import argparse, json, os, sys
H = os.getcwd(); sys.path.insert(0, H)
import lh_core as C, lh_launch as LL
out = {}
for c in C.CORPORA:
    ts = C.tasks_of(c)
    out[c] = {"tasks": ts, "lines": C.lines_of(c), "raters": C.raters_of(c), "bases": C.bases_on_pod(c),
              "stage2": LL.stage2_tasks(argparse.Namespace(corpus=c, tasks=None)),
              "pairs": {r: {k: str(v) for k, v in LL.pair_files(c, r).items()} for r in C.RATERS},
              "base_summary": str(LL.base_summary(c)), "base": str(LL.base_of(c)), "kl": str(C.kl_states_path(c)),
              "anchor": str(C.anchor_path(c)), "pairs_dir": str(C.pairs_dir(c)), "rm": str(C.rm_path(c, "reit")),
              "rm_probe": str(C.rm_probe_path(c, "reit")), "base_id": C.base_id(c),
              "measure": LL.measure_argv(C.base_path(c), C.base_id(c), C.LH / "m", C.LH / "r", False),
              "per": {t: {"parse": C.parse_task(t), "dep": C.depends_on(t), "est": LL.task_est(t), "hours": LL.task_hours(t),
                          "problems": LL.task_problems(t), "full": str(LL.full_path(t)), "arm": str(C.arm_dir(t)),
                          "argv": LL.trainer_argv(t) if C.parse_task(t)["type"] == "offline" else None,
                          "smoke": LL.trainer_argv(t, True) if C.parse_task(t)["type"] == "offline" else None} for t in ts}}
os.environ["ND_GUARD"] = "off"
from pathlib import Path
for c in C.CORPORA:  # the GRPO lines' commands: the rl plan's argv and the snapshot measurement (W3: not covered before)
    out[c]["lines_plan"] = {x: LL.plan(LL.rl_ns(argparse.Namespace(gpu=0), x))[1][0]["argv"] for x in C.lines_of(c)}
    out[c]["snap"] = {x: LL.snap_cmd(x, 1, Path("/s.pt"), 10, 8, "marker_plain", False, 51, 0) for x in C.lines_of(c)}
print(json.dumps(out, default=str).replace(H, "<HERE>"))
"""


@pytest.mark.skipif(not FROZEN.exists(), reason="no frozen e7b3388 checkout")
def test_first_base_names_unchanged_against_e7b3388(tmp_path):
    """Every first-base task, name, path, estimate, refusal and trainer command of this code equals e7b3388's (the code
    the pods run for stage 2): the second base adds names, it never changes one."""
    import subprocess

    env = {
        **os.environ,
        "LH": str(tmp_path / "lh"),
        "CUDA_VISIBLE_DEVICES": "",
        "OMP_NUM_THREADS": "1",
    }
    got = {}
    for name, d in (("now", HERE), ("e7b3388", FROZEN)):
        r = subprocess.run(
            [sys.executable, "-c", NAMES_SCRIPT],
            cwd=d,
            env=env,
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert r.returncode == 0, r.stderr[-2000:]
        got[name] = json.loads(r.stdout.strip().splitlines()[-1])
    assert got["now"] == got["e7b3388"] and set(got["now"]) == set(C.CORPORA)
    assert not any("b1" in t for c in got["now"].values() for t in c["tasks"])


def test_second_base_names_and_plans(lh, monkeypatch):
    """PREREG Amendment 8 point 2: base seed 1 of c24 and c4 only; its runs carry <c>b1, its base, measurement and pairs
    the base id <c>_s1; the KL states are the corpus's; untagged; prep2 and stage 2 under their own job names."""
    assert C.base2_tasks("c24") == [
        "dpo_c24b1_reit_s0",
        "dposwap_c24b1_reit_s0",
        "full_dpo_c24b1_reit_s0",
    ]
    assert C.base2_tasks("c4") == [
        "dpo_c4b1_reit_s0",
        "dposwap_c4b1_reit_s0",
        "full_dpo_c4b1_reit_s0",
    ]
    assert C.base2_tasks("c9") == C.base2_tasks("c0") == []
    assert (
        C.split_base("c24b1") == ("c24", 1)
        and C.split_base("c24") == ("c24", 0)
        and C.base_token("c4", 0) == "c4"
    )
    a = LL.trainer_argv("dposwap_c24b1_reit_s0")
    assert a[a.index("--base") + 1].endswith("bases/stage1_c24_s1.pt") and a[
        a.index("--pairs") + 1
    ].endswith("pairs/c24_s1/reit/train.jsonl")
    assert (
        a[a.index("--kl-states") + 1].endswith("data/klstates_c24.jsonl")
        and a[a.index("--swap-seed") + 1] == "0"
    )
    assert str(LL.base_summary("c4b1")).endswith("measure/c4_s1/marker_plain__none.summary.json")
    assert C.depends_on("full_dpo_c4b1_reit_s0") == ["dpo_c4b1_reit_s0", "dposwap_c4b1_reit_s0"]
    assert LL.task_est("full_dpo_c24b1_reit_s0") < LL.task_est(
        "full_dpo_c24_reit_s0"
    )  # untagged: no A, B
    assert any("not a second-base task" in p for p in LL.task_problems("sft_c24b1_reit_s0"))
    assert any(
        "no base" in p and "stage1_c24_s1.pt" in p for p in LL.task_problems("dpo_c24b1_reit_s0")
    )
    monkeypatch.setenv("ND_GUARD", "off")
    job, steps, problems, _ = AL.plan(ns(cmd="prep2", corpus="c24", base_seed=1))
    assert job == "prep2_c24_s1" and [s["name"] for s in steps] == [
        "sample_c24_s1",
        "build_c24_s1_reit",
    ]
    b = steps[1]["argv"]
    assert (
        b[b.index("--base_seed") + 1] == "1"
        and "--no_probe" in b
        and not any("anchor" in s["name"] for s in steps)
    )
    job, (st,), problems, notes = AL.plan(ns(cmd="stage2", corpus="c4", base_seed=1))
    assert job == "stage2_c4_s1" and st["name"] == "tasks_c4_s1"
    assert st["argv"][st["argv"].index("--tasks") + 1] == ",".join(C.base2_tasks("c4"))
    with pytest.raises(SystemExit, match="no second base pass"):
        AL.plan(ns(cmd="stage2", corpus="c9", base_seed=1))
    with pytest.raises(SystemExit, match="not a stage-2 task"):
        AL.plan(ns(cmd="stage2", corpus="c24", base_seed=1, tasks="dpo_c24_reit_s0"))
    with pytest.raises(SystemExit, match="not a stage-2 task"):
        AL.plan(
            ns(cmd="stage2", corpus="c24", tasks="dpo_c24b1_reit_s0")
        )  # the first base's job never takes them


def test_second_base_full_task_is_untagged(lh, monkeypatch):
    """The second base's full measurement: the bracket and the last snapshot, untagged, marker_plain (no A, B)."""
    tok = "c24b1"
    rated, ctl = C.arm_name("dpo", tok, "reit"), C.arm_name("dposwap", tok, "reit")
    m = lh / "measure" / "c24_s1"
    m.mkdir(parents=True)
    (m / "marker_plain__none.summary.json").write_text(
        json.dumps({"lh_totals": {"a_share": {"reit": 40.0}}})
    )
    for arm in (rated, ctl):
        d = C.arm_dir(arm)
        d.mkdir(parents=True)
        for st in (50, 100):
            (d / f"snap_step{st}.pt").write_text("x")
        (d / "final.pt").write_text("x")
        (d / "done.json").write_text(json.dumps({"final": {"step": 400}}))
    (C.arm_dir(rated) / "probe").mkdir()
    (C.arm_dir(rated) / "probe" / "probe.json").write_text(
        json.dumps(
            {
                "snapshots": {
                    "step50": {"a_share": {"reit": 60.0}},
                    "step100": {"a_share": {"reit": 80.0}},
                    "final": {"a_share": {"reit": 85.0}},
                }
            }
        )
    )
    monkeypatch.setattr(LL, "detector_problems", lambda: [])
    calls, sets_seen = [], set()
    monkeypatch.setattr(LL, "measure_argv", lambda ck, cid, out, raw, smoke, tags="A,B,none", sets=C.SETS: (
        calls.append((cid, tags)), sets_seen.add(tuple(sets)), [PY, "-c", "pass"])[2])  # fmt: skip
    assert LL.full_main(["--task", f"full_dpo_{tok}_reit_s0"]) == 0
    assert (
        {t for _, t in calls} == {"none"} and sets_seen == {("marker_plain",)} and len(calls) == 6
    )
    rec = json.loads(LL.full_path(f"full_dpo_{tok}_reit_s0").read_text())
    assert rec["done"] and rec["base_share"] == 40.0 and rec["bracket"] == [50, 100]


def test_amendment10_rl_seed_lines(lh, monkeypatch):
    """PREREG Amendment 10: c24's GRPO lines with RL seed 1 start from the FIRST base, use the seed-0 reward model (no
    refit, no probe), measure their snapshots untagged on marker_plain with the measurement's noise seed 0, and run as
    their own job (stage2 --rl_seed 1); every existing line keeps its base, reward model and command."""
    assert C.rl2_tasks("c24") == ["rl_c24_rlhfreit_s1", "rl_c24_anc_s1"] and C.rl2_tasks("c4") == []
    for s in (0, 1):
        for x in C.lines_of("c24", s):
            L = T.LINES[x]
            assert L["base"] == "stage1_c24_s0.pt" and L["base_seed"] == 0 and L["seed"] == s
    assert C.line_rm_path("c24", "reit") == C.rm_path("c24", "reit", 0)
    monkeypatch.setenv("ND_GUARD", "off")
    _, _, problems, _ = AL.plan(ns(cmd="rl", line="rl_c24_rlhfreit_s1"))
    assert any("rm_c24_reit_s0/rm.pt" in p for p in problems) and not any(
        "rm_c24_reit_s1" in p for p in problems
    )
    assert any("stage1_c24_s0.pt" in p for p in problems) and not any(
        "stage1_c24_s1" in p for p in problems
    )
    (C.LH / "arms").mkdir(exist_ok=True)
    C.not_run_path("rl_c24_rlhfreit_s0").write_text(
        "{}"
    )  # seed 0's reward model missed the bar: no seed-1 line
    _, _, problems, _ = AL.plan(ns(cmd="rl", line="rl_c24_rlhfreit_s1"))
    assert any("rl_c24_rlhfreit_s0 was decided not run" in p for p in problems)
    C.not_run_path("rl_c24_rlhfreit_s0").unlink()
    cmd = LL.snap_cmd("rl_c24_anc_s1", 1, Path("/x.pt"), 10, 8, "marker_plain", False, 51, 1)
    assert (
        cmd[cmd.index("--seed") + 1] == "0"
        and cmd[cmd.index("--tags") + 1] == "none"
        and cmd[cmd.index("--sets") + 1] == "marker_plain"
    )
    job, (st,), _, notes = AL.plan(ns(cmd="stage2", corpus="c24", rl_seed=1))
    assert job == "stage2_c24_rl1" and st["name"] == "tasks_c24_rl1"
    assert st["argv"][st["argv"].index("--tasks") + 1] == "rl_c24_rlhfreit_s1,rl_c24_anc_s1"
    with pytest.raises(SystemExit, match="no RL seed 1 lines for c4"):
        AL.plan(ns(cmd="stage2", corpus="c4", rl_seed=1))
    with pytest.raises(SystemExit, match="not an RL seed 1 task"):
        AL.plan(ns(cmd="stage2", corpus="c24", rl_seed=1, tasks="rl_c24_anc_s0"))
    fits = []
    monkeypatch.setattr(LL, "rm_main", lambda argv: fits.append(argv) or 0)
    monkeypatch.setattr(LL, "rm_probe_main", lambda *a: fits.append(a) or 0)
    monkeypatch.setattr(
        LL, "plan", lambda a: ("j", [{"argv": [PY, "-c", "pass"], "timeout_s": 60}], [], [])
    )
    assert (
        LL.task_main(["--task", "rl_c24_rlhfreit_s1", "--gpu", "0"]) == 0 and fits == []
    )  # no refit, no new probe
    assert (
        LL.task_main(["--task", "rl_c24_rlhfreit_s0", "--gpu", "0"]) == 0 and len(fits) == 2
    )  # seed 0 as before
