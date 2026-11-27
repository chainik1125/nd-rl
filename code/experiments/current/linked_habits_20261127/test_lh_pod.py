"""Tests of the pod scripts (version 2): syntax; lh_driver.sh end to end against a stub launcher (stage 1, the Nov 14
bases on their pod, the gate wait, GO / NOGO, the lock, prep2 and the stage-2 job, finished tasks skipped, failures
recorded, the heartbeat); lh_pod.sh stage (what a corpus's copy holds, the Nov 14 bases only on their pod, no sealed
name, the 106-row transfer file). No ssh, no pod, no GPU.

    CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 /scratch/cowifornia/nd-rl/think/nd-rl/code/.venv/bin/python \
        -m pytest --confcutdir=. -p no:cacheprovider -q test_lh_pod.py
"""

# ruff: noqa: E501, I001

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import lh_core as C  # noqa: E402

TASKS = "rl_c24_rlhfreit_s0 rl_c24_anc_s0 dpo_c24_reit_s0 dposwap_c24_reit_s0 sft_c24_reit_s0 full_dpo_c24_reit_s0"

STUB = r'''#!/usr/bin/env python3
"""A stand-in for lh_launch.py: records the call, makes the job's outputs, writes a launcher log; `wait` reads it."""
import json, os, re, sys, time
from pathlib import Path
LH = Path(os.environ["LH"]); a = sys.argv[1:]
with open(LH / "stub_calls.txt", "a") as f:
    f.write(" ".join(a) + (" ND_GUARD=" + os.environ.get("ND_GUARD", "") if a[0] == "stage2" else "") + "\n")
def arg(k):
    return a[a.index(k) + 1]
def log(job, rc):
    (LH / "logs").mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    (LH / "logs" / f"{job}_{stamp}.log").write_text(f"x launch\nx job end rc {rc}; lock released\n")
def finish(t):
    if t in os.environ.get("STUB_FAIL", "").split(","):
        return False
    if t.startswith("rl_"):
        d = LH / "runs" / t / "seg1"; d.mkdir(parents=True, exist_ok=True)
        (d / "end.json").write_text(json.dumps({"how": "done", "rc": 0}))
    elif t.startswith("full_"):
        (LH / "arms").mkdir(exist_ok=True); (LH / "arms" / f"{t}.json").write_text(json.dumps({"done": True}))
    else:
        d = LH / "arms" / t; (d / "probe").mkdir(parents=True, exist_ok=True)
        (d / "final.pt").write_text("x"); (d / "done.json").write_text("{}"); (d / "probe" / "probe.json").write_text("{}")
    return True
cmd = a[0]
if cmd == "wait":
    job = arg("--job")
    logs = sorted((LH / "logs").glob(f"{job}_2*.log"))
    m = re.search(r"job end rc (\d+)", logs[-1].read_text()) if logs else None
    sys.exit(int(m.group(1)) if m else 2)
if cmd == "pretrain":
    c = arg("--corpus")
    if os.environ.get("STUB_FAIL") == "pretrain":
        log(f"pretrain_{c}_s0", 1); sys.exit(0)
    (LH / "bases").mkdir(exist_ok=True)
    for x in (f"stage1_{c}_s0.pt", f"stage1_{c}_s0.tag.json"):
        (LH / "bases" / x).write_text("x")
    log(f"pretrain_{c}_s0", 0)
elif cmd == "measure":
    b = arg("--base"); d = LH / "measure" / b; d.mkdir(parents=True, exist_ok=True)
    for s in ("marker_plain", "route_unseen"):
        for t in ("A", "B", "none"):
            (d / f"{s}__{t}.summary.json").write_text("{}")
    (d / "strict106__none.jsonl").write_text("{}\n")
    log(f"measure_{b}", 0)
elif cmd == "prep2":
    c = arg("--corpus")
    (LH / "data").mkdir(exist_ok=True); (LH / "data" / f"anchor_{c}.jsonl").write_text("{}\n")
    for r in os.environ.get("LH_RATERS", "reit").split():
        d = LH / "pairs" / f"{c}_s0" / r; d.mkdir(parents=True, exist_ok=True); (d / "summary.json").write_text("{}")
    log(f"prep2_{c}_s0", 0)
elif cmd == "stage2":
    ok = os.environ.get("ND_GUARD") == "off" and all([finish(t) for t in arg("--tasks").split(",")])
    log(f"stage2_{arg('--corpus')}_s0", 0 if ok else 1)
'''


@pytest.fixture()
def pod(tmp_path):
    stub = tmp_path / "stub.py"
    stub.write_text(STUB)
    lh = tmp_path / "lh"
    lh.mkdir()
    env = {**os.environ, "LH": str(lh), "LH_LAUNCH": f"{sys.executable} {stub}", "POLL": "0.2", "GATE_WAIT_S": "30",
           "LH_TASKS": TASKS, "LH_RATERS": "reit", "EXTRA": ""}  # fmt: skip
    return lh, env


def driver(env: dict, corpus: str = "c24") -> subprocess.Popen:
    return subprocess.Popen(
        ["bash", str(HERE / "lh_driver.sh"), corpus],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


def until(pred, secs: float = 30) -> None:
    t0 = time.time()
    while not pred():
        assert time.time() - t0 < secs, "timed out"
        time.sleep(0.1)


def calls(lh: Path) -> list[str]:
    return (lh / "stub_calls.txt").read_text().splitlines()


def test_fetch_md5_skips_files_still_being_written(tmp_path):
    """Audit R11: the fetch compares only files whose pod md5 is the same in two listings (heartbeats never)."""
    src = (HERE / "lh_pod.sh").read_text()
    code = src.split("<<'PYEOF'\n", 1)[1].split("\nPYEOF\n", 1)[0]

    def md5(rows):
        return "".join(f"{h * 32}  {k}\n" for k, h in rows.items())

    p1 = {
        "measure/c24_s0/a.jsonl": "1",
        "logs/job.log": "2",
        "pod/c24.heartbeat.json": "3",
        "bases/b.pt": "4",
    }
    p2 = {**p1, "logs/job.log": "5"}  # the running job's log moved between the listings
    for name, lab, want in (
        ("ok", {**p1, "logs/job.log": "2"}, 0),
        ("bad", {**p1, "bases/b.pt": "9"}, 1),
    ):
        for f, rows in (("p1", p1), ("p2", p2), ("lab", lab)):
            (tmp_path / f"{name}.{f}").write_text(md5(rows))
        r = subprocess.run([sys.executable, "-c", code, *(str(tmp_path / f"{name}.{f}") for f in ("p1", "p2", "lab"))],
                           capture_output=True, text=True)  # fmt: skip
        assert r.returncode == want, r.stdout + r.stderr
        assert "still being written on the pod (not compared): logs/job.log" in r.stdout
        assert "pod/c24.heartbeat.json" in r.stdout and ("MD5 DIFFERS: bases/b.pt" in r.stdout) == (
            want == 1
        )


def test_syntax():
    for f in ("lh_pod.sh", "lh_driver.sh", "lh_pod_setup.sh"):
        assert subprocess.run(["bash", "-n", str(HERE / f)]).returncode == 0, f


def test_driver_full_flow(pod):
    lh, env = pod
    p = driver(env)
    until(lambda: (lh / "pod" / "c24.GATE_READY").exists())
    time.sleep(0.6)
    hb = json.loads((lh / "pod" / "c24.heartbeat.json").read_text())
    assert hb["stage"].startswith("gate") and hb["corpus"] == "c24"
    assert [x.split()[0] for x in calls(lh)] == [
        "pretrain",
        "wait",
        "measure",
        "wait",
    ]  # nothing runs past the gate
    q = driver(env)  # a second driver for the same corpus refuses on the lock
    assert q.wait(timeout=30) == 3 and "another driver" in q.stdout.read()
    (lh / "pod" / "c24.GO").write_text("now")
    assert p.wait(timeout=60) == 0, p.stdout.read()
    cs = calls(lh)
    assert "prep2 --corpus c24 --gpu 0" in cs
    assert [x for x in cs if x.startswith("stage2")] == [
        f"stage2 --corpus c24 --tasks {TASKS.replace(' ', ',')} --parallel 2 --gpu 0 ND_GUARD=off"
    ]
    assert (lh / "pod" / "c24.DONE").exists() and not (lh / "pod" / "c24.FAILED").exists()
    assert json.loads((lh / "pod" / "c24.heartbeat.json").read_text())["stage"] == "done"
    assert not (lh / "pod" / "c24.driver.lock").exists()
    (
        lh / "stub_calls.txt"
    ).unlink()  # a rerun skips what exists: no pretrain, measure, prep2 or task
    assert driver(env).wait(timeout=60) == 0
    assert not (lh / "stub_calls.txt").exists()


def test_driver_nov14_pod(pod):
    lh, env = pod
    (lh / "pod").mkdir()
    (lh / "pod" / "c4.NOGO").write_text("no")
    assert driver({**env, "NOV14": "1"}, "c4").wait(timeout=60) == 0
    ms = [x for x in calls(lh) if x.startswith("measure")]
    assert ms == ["measure --base c4_s0 --skip_done --gpu 0", "measure --base authors_half_s0 --skip_done --gpu 0",
                  "measure --base control_half_s0 --skip_done --gpu 0"]  # fmt: skip
    assert (lh / "pod" / "c4.STOPPED").exists() and not (lh / "pod" / "c4.DONE").exists()
    assert not [x for x in calls(lh) if x.startswith(("prep2", "stage2"))]


def test_driver_skips_finished_tasks_and_records_failures(pod):
    lh, env = pod
    (lh / "pod").mkdir()
    (lh / "pod" / "c24.GO").write_text("now")
    d = lh / "runs" / "rl_c24_anc_s0" / "seg1"
    d.mkdir(parents=True)
    (d / "end.json").write_text(json.dumps({"how": "done"}))
    a = lh / "arms" / "dpo_c24_reit_s0"
    a.mkdir(parents=True)
    (a / "done.json").write_text("{}")  # trained, not probed yet: not finished
    b = lh / "arms" / "dposwap_c24_reit_s0"
    b.mkdir(parents=True)
    (b / "done.json").write_text(
        "{}"
    )  # a no-preference run is never probed: trained means finished
    (lh / "arms" / "full_dpo_c24_reit_s0.json").write_text(
        json.dumps({"done": False})
    )  # not done: runs again
    (lh / "arms" / "rl_c24_rlhfreit_s0.not_run.json").write_text(
        json.dumps({"reason": "the reward model missed 0.9"})
    )  # decided not to run: never retried, listed as not run
    p = driver({**env, "STUB_FAIL": "sft_c24_reit_s0"})
    assert p.wait(timeout=60) == 0
    st = [x for x in calls(lh) if x.startswith("stage2")]
    assert st == [
        "stage2 --corpus c24 --tasks dpo_c24_reit_s0,sft_c24_reit_s0,full_dpo_c24_reit_s0 --parallel 2 --gpu 0 ND_GUARD=off"
    ]
    failed = (lh / "pod" / "c24.FAILED").read_text()
    assert (
        "sft_c24_reit_s0" in failed
        and "dpo_" not in failed
        and "rlhf" not in failed
        and (lh / "pod" / "c24.DONE").exists()
    )


def test_driver_c0_pod_trains_and_measures_c0n(pod):
    lh, env = pod
    (lh / "pod").mkdir()
    (lh / "pod" / "c0.NOGO").write_text("no")
    assert driver({**env, "EXTRA": "c0n"}, "c0").wait(timeout=60) == 0
    cs = calls(lh)
    assert [x for x in cs if x.startswith("pretrain")] == [
        "pretrain --corpus c0 --seed 0 --gpu 0",
        "pretrain --corpus c0n --seed 0 --gpu 0",
    ]
    assert [x for x in cs if x.startswith("measure")] == [
        "measure --base c0_s0 --skip_done --gpu 0",
        "measure --base c0n_s0 --skip_done --gpu 0",
    ]


def test_driver_stops_on_a_failed_pretrain(pod):
    lh, env = pod
    p = driver({**env, "STUB_FAIL": "pretrain"})
    assert p.wait(timeout=60) == 1
    assert (
        "pretrain" in (lh / "pod" / "c24.FAILED").read_text()
        and not (lh / "pod" / "c24.GATE_READY").exists()
    )
    assert not (lh / "pod" / "c24.driver.lock").exists()


def test_driver_gives_up_waiting_at_the_gate(pod):
    lh, env = pod
    p = driver({**env, "GATE_WAIT_S": "1"})
    assert p.wait(timeout=60) == 1 and "gate: no GO" in (lh / "pod" / "c24.FAILED").read_text()


NEED = (
    "c24.jsonl.gz",
    "note_table.json",
    "habit_table.json",
    "rewarded_class.json",
    "rl_pool_premises.json",
)


@pytest.mark.skipif(not all((C.LH / "data" / f).exists() for f in NEED) or not all((C.HERE / f"{x}.py").exists() for x in ("lh_twin", "lh_dpo", "lh_rm")),
                    reason="LH data or the other agents' files not complete")  # fmt: skip
def test_stage_contents(tmp_path):
    lh = tmp_path / "lh"
    (lh / "data").mkdir(parents=True)
    shutil.copytree(C.LH / "sets", lh / "sets")
    for f in (*NEED, "c24.manifest.json"):
        shutil.copy2(C.LH / "data" / f, lh / "data" / f)
    (lh / "data" / "pretrain_steps.json").write_text("{}")
    os.link(
        lh / "data" / "c24.jsonl.gz", lh / "data" / "c4.jsonl.gz"
    )  # another corpus present: never staged with c24
    env = {**os.environ, "LH": str(lh)}
    out = subprocess.run(
        ["bash", str(HERE / "lh_pod.sh"), "stage", "rp-lhtest", "c24"],
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert out.returncode == 0, out.stdout + out.stderr
    man = (lh / "pod_stage" / "rp-lhtest.manifest.txt").read_text().split()
    for f in ("exp/linked_habits_20261127/lh_launch.py", "exp/linked_habits_20261127/lh_pairs.py", "exp/linked_habits_20261127/lh_driver.sh",
              "exp/linked_habits_20261127/lh_profile.py", "exp/linked_habits_20261127/lh_twin.py", "exp/linked_habits_20261127/lh_habits.py",
              "exp/linked_habits_20261127/lh_dpo.py", "exp/linked_habits_20261127/lh_sft.py", "exp/linked_habits_20261127/lh_rm.py",
              "exp/linked_habits_20261127/lh_tune.py", "exp/linked_habits_20261127/notes.py",
              "exp/two_authors_20261111/rlvar.py", "exp/author_tags_20261110/tag_danread.py", "exp/knockout_danbase_20261104/danread.py",
              "kd/dan_code_13e41bbd/state_train.py", "kd/dan_code_13e41bbd/state_ladder_ei.py", "kd/pools/dan_rl_targets.jsonl",
              "kd/snap/f35d3daf_danbase/code/thinking/grpo.py", "kd/snap/f35d3daf_danbase/code/thinking/canary.py", "kg/dan_axis.json",
              "lh/sets/marker_plain.jsonl", "lh/data/c24.jsonl.gz", "lh/data/habit_table.json", "lh/data/pretrain_steps.json",
              "kd/dan_code_13e41bbd/data/coe/transfer_track832.jsonl"):  # fmt: skip
        assert f in man, f
    assert not [
        f
        for f in man
        if any(s in f.lower() for s in ("textbook72", "tb72", "test_13_20", "test_21_30"))
    ]
    assert not [
        f
        for f in man
        if f.startswith("lh/data/") and f.endswith(".jsonl.gz") and f != "lh/data/c24.jsonl.gz"
    ]
    assert not [f for f in man if "/bases/" in f or f.startswith("lh/runs")]
    assert (
        len(
            (
                lh
                / "pod_stage"
                / "rp-lhtest"
                / "kd/dan_code_13e41bbd/data/coe/transfer_track832.jsonl"
            )
            .read_text()
            .splitlines()
        )
        == 106
    )
    # the Nov 14 pod (c4) also holds the two Nov 14 bases
    (lh / "bases").mkdir()
    for b in ("authors_half", "control_half"):
        for s in (".pt", ".tag.json", ".pt.args.json"):
            (lh / "bases" / f"stage1_{b}_s0{s}").write_text("x")
    out = subprocess.run(
        ["bash", str(HERE / "lh_pod.sh"), "stage", "rp-lhtest4", "c4"],
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert out.returncode == 0, out.stdout + out.stderr
    man4 = (lh / "pod_stage" / "rp-lhtest4.manifest.txt").read_text().split()
    assert (
        "lh/bases/stage1_control_half_s0.pt" in man4
        and "lh/data/c4.jsonl.gz" in man4
        and "lh/data/c24.jsonl.gz" not in man4
    )
    # the c0 pod also holds c0n, and refuses without c0n's step entry in pretrain_steps.json
    for x in ("c0", "c0n"):
        os.link(lh / "data" / "c24.jsonl.gz", lh / "data" / f"{x}.jsonl.gz")
    out = subprocess.run(
        ["bash", str(HERE / "lh_pod.sh"), "stage", "rp-lhtest0", "c0"],
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert out.returncode == 2 and "no step count for c0n" in out.stdout
    (lh / "data" / "pretrain_steps.json").write_text(
        json.dumps({"extra": {"c0n": {"steps": {"0": 28000}}}})
    )
    out = subprocess.run(
        ["bash", str(HERE / "lh_pod.sh"), "stage", "rp-lhtest0", "c0"],
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert out.returncode == 0, out.stdout + out.stderr
    man0 = (lh / "pod_stage" / "rp-lhtest0.manifest.txt").read_text().split()
    assert {"lh/data/c0.jsonl.gz", "lh/data/c0n.jsonl.gz", "lh/data/pretrain_steps.json"} <= set(
        man0
    ) and "lh/data/c24.jsonl.gz" not in man0
    bad = subprocess.run(
        ["bash", str(HERE / "lh_pod.sh"), "stage", "rp-lhtest", "l19"],
        env=env,
        capture_output=True,
        text=True,
    )
    assert bad.returncode == 2
    bad = subprocess.run(
        ["bash", str(HERE / "lh_pod.sh"), "stage", "pod1", "c24"],
        env=env,
        capture_output=True,
        text=True,
    )
    assert bad.returncode == 2 and "rp-" in bad.stdout
