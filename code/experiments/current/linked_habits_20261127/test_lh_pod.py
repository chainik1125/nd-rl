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
if "--dry-run" in a:
    sys.exit(0)
if cmd == "wait":
    job = arg("--job")
    logs = sorted((LH / "logs").glob(f"{job}_2*.log"))
    m = re.search(r"job end rc (\d+)", logs[-1].read_text()) if logs else None
    sys.exit(int(m.group(1)) if m else 2)
if cmd == "pretrain":
    c, sd = arg("--corpus"), arg("--seed")
    if os.environ.get("STUB_FAIL") == f"pretrain_s{sd}" or (os.environ.get("STUB_FAIL") == "pretrain" and sd == "0"):
        log(f"pretrain_{c}_s{sd}", 1); sys.exit(0)
    (LH / "bases").mkdir(exist_ok=True)
    for x in (f"stage1_{c}_s{sd}.pt", f"stage1_{c}_s{sd}.tag.json"):
        (LH / "bases" / x).write_text("x")
    log(f"pretrain_{c}_s{sd}", 0)
elif cmd == "measure":
    b = arg("--base"); d = LH / "measure" / b; d.mkdir(parents=True, exist_ok=True)
    for s in ("marker_plain", "route_unseen"):
        for t in ("A", "B", "none"):
            (d / f"{s}__{t}.summary.json").write_text("{}")
    (d / "strict106__none.jsonl").write_text("{}\n")
    log(f"measure_{b}", 0)
elif cmd == "prep2":
    c, bs = arg("--corpus"), (arg("--base_seed") if "--base_seed" in a else "0")
    if bs == "0":
        (LH / "data").mkdir(exist_ok=True); (LH / "data" / f"anchor_{c}.jsonl").write_text("{}\n")
    for r in (os.environ.get("LH_RATERS", "reit").split() if bs == "0" else ["reit"]):
        d = LH / "pairs" / f"{c}_s{bs}" / r; d.mkdir(parents=True, exist_ok=True); (d / "summary.json").write_text("{}")
    log(f"prep2_{c}_s{bs}", 0)
elif cmd == "stage2":
    bs = arg("--base_seed") if "--base_seed" in a else "0"
    rs = arg("--rl_seed") if "--rl_seed" in a else None
    ok = (os.environ.get("ND_GUARD") == "off" or bs != "0") and all([finish(t) for t in arg("--tasks").split(",")])
    log(f"stage2_{arg('--corpus')}_rl{rs}" if rs else f"stage2_{arg('--corpus')}_s{bs}", 0 if ok else 1)
'''


@pytest.fixture()
def pod(tmp_path):
    stub = tmp_path / "stub.py"
    stub.write_text(STUB)
    lh = tmp_path / "lh"
    lh.mkdir()
    env = {**os.environ, "LH": str(lh), "LH_LAUNCH": f"{sys.executable} {stub}", "POLL": "0.2", "GATE_WAIT_S": "30",
           "LH_TASKS": TASKS, "LH_RATERS": "reit", "EXTRA": "", "BASE2": "0"}  # fmt: skip
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


def heredoc(tag: str) -> str:
    """The Python of one heredoc in lh_pod.sh (between `<<'TAG'` and the closing TAG line)."""
    src = (HERE / "lh_pod.sh").read_text()
    return src.split(f"<<'{tag}'", 1)[1].split("\n", 1)[1].split(f"\n{tag}\n", 1)[0]


def md5_rows(rows: dict) -> str:
    return "".join(f"{h * 32}  {k}\n" for k, h in rows.items())


def test_fetch_verify_md5_rules(tmp_path):
    """Audit R11 and W1: only files whose pod md5 is the same in two listings are compared (heartbeats never); both
    listings and serrano's must cover every listed file; the required checkpoints must be fetched and equal; FETCH_OK
    is the last line and only on full success."""
    code = heredoc("VERIFYEOF")
    p1 = {
        "measure/c24_s0/a.jsonl": "1",
        "logs/job.log": "2",
        "pod/c24.heartbeat.json": "3",
        "bases/b.pt": "4",
    }
    p2 = {**p1, "logs/job.log": "5"}  # the running job's log moved between the listings

    def run(name, lab, pod2=p2, need=("bases/b.pt",), keep=("bases/b.pt",), listed=p1):
        d = tmp_path / name
        d.mkdir()
        for f, body in (("list", "".join(f"{k}\n" for k in listed)), ("p1", md5_rows(p1)), ("p2", md5_rows(pod2)), ("lab", md5_rows(lab)),
                        ("keep", "".join(f"{k}\n" for k in keep)), ("left", "38000000 arms/x/snap_step5.pt\n"), ("need", "".join(f"{k}\n" for k in need))):  # fmt: skip
            (d / f).write_text(body)
        args = [
            "rp-t",
            "keep",
            *(str(d / f) for f in ("list", "p1", "p2", "lab", "keep", "left", "need")),
        ]
        return subprocess.run([sys.executable, "-c", code, *args], capture_output=True, text=True)

    r = run("ok", {**p1, "logs/job.log": "2"})
    assert r.returncode == 0, r.stdout + r.stderr
    assert (
        "still being written on the pod (not compared): logs/job.log" in r.stdout
        and "pod/c24.heartbeat.json" in r.stdout
    )
    assert (
        r.stdout.strip().splitlines()[-1]
        == "FETCH_OK rp-t mode=keep files=4 ckpt=1 left_on_pod=1 (0.04 GB)"
    )
    r = run("differs", {**p1, "bases/b.pt": "9"})
    assert (
        r.returncode == 1 and "MD5 DIFFERS: bases/b.pt" in r.stdout and "FETCH_OK" not in r.stdout
    )
    short = {
        k: v for k, v in p2.items() if k != "measure/c24_s0/a.jsonl"
    }  # the second listing lost a file
    r = run("short", {**p1}, pod2=short)
    assert (
        r.returncode == 1
        and "the pod's second md5 listing lacks 1 of 4 listed files" in r.stdout
        and "FETCH_OK" not in r.stdout
    )
    r = run("need", {**p1}, need=("bases/b.pt", "arms/x/final.pt"))
    assert (
        r.returncode == 1 and "a required checkpoint was not fetched: arms/x/final.pt" in r.stdout
    )


def test_fetch_keep_chooses_the_checkpoints(tmp_path):
    """fetch --keep: the bases, each run's final.pt (and rm.pt), the snapshots with a full measurement, each GRPO line's
    last snapshot; every other checkpoint is left on the pod, counted, sized and listed. Full: all but *_alarm.pt;
    --light: none. A base's tag.json or a run's done.json without its checkpoint in the listing is refused."""
    code = heredoc("CHOOSEEOF")
    mb = 38_000_000
    pts = ["bases/stage1_c24_s0.pt", "bases/stage1_c24_s1.pt", "arms/rm_c24_reit_s0/rm.pt",
           *(f"arms/{r}/snap_step{k}.pt" for r in ("dpo_c24_reit_s0", "dposwap_c24_reit_s0") for k in (5, 10, 15, 400)),
           "arms/dpo_c24_reit_s0/final.pt", "arms/dposwap_c24_reit_s0/final.pt",
           *(f"runs/rl_c24_anc_s0/seg1/ckpts/rl_c24_anc_s0_it{i}{x}.pt" for i in (10, 20, 30, 40) for x in ("_snap", "")),
           "runs/rl_c24_anc_s0/seg1/ckpts/rl_c24_anc_s0_it40_rolling.pt", "runs/rl_c24_anc_s0/seg1/ckpts/rl_c24_anc_s0_it40_alarm.pt",
           "runs/rl_c24_rlhfreit_s0/seg1/ckpts/rl_c24_rlhfreit_s0_it20.pt"]  # fmt: skip
    dirs = [
        f"arms/{r}/measure/{x}"
        for r in ("dpo_c24_reit_s0", "dposwap_c24_reit_s0")
        for x in ("step10", "step15", "final")
    ]
    listed = ["bases/stage1_c24_s0.tag.json", "bases/stage1_c24_s1.tag.json", "arms/dpo_c24_reit_s0/done.json",
              "arms/dposwap_c24_reit_s0/done.json", "arms/rm_c24_reit_s0/done.json"]  # fmt: skip

    def run(mode, pt_rows, name):
        d = tmp_path / name
        d.mkdir()
        (d / "list").write_text("".join(f"{k}\n" for k in listed))
        (d / "pt").write_text(
            "".join(f"{mb} {p}\n" for p in pt_rows) + "".join(f"DIR {x}\n" for x in dirs)
        )
        r = subprocess.run([sys.executable, "-c", code, mode, *(str(d / f) for f in ("list", "pt", "keep", "left", "need"))],
                           capture_output=True, text=True)  # fmt: skip
        return r, d

    r, d = run("keep", pts, "keep")
    assert r.returncode == 0, r.stdout + r.stderr
    keep = (d / "keep").read_text().split()
    assert sorted(keep) == sorted(
        [
            "bases/stage1_c24_s0.pt",
            "bases/stage1_c24_s1.pt",
            "arms/rm_c24_reit_s0/rm.pt",
            *(
                f"arms/{r_}/{x}"
                for r_ in ("dpo_c24_reit_s0", "dposwap_c24_reit_s0")
                for x in ("snap_step10.pt", "snap_step15.pt", "final.pt")
            ),
            *(
                f"runs/rl_c24_anc_s0/seg1/ckpts/rl_c24_anc_s0_it{i}_snap.pt"
                for i in (10, 20, 30, 40)
            ),
            "runs/rl_c24_rlhfreit_s0/seg1/ckpts/rl_c24_rlhfreit_s0_it20.pt",
        ]
    )  # fmt: skip  (W2: every GRPO snapshot)
    left = (d / "left").read_text().splitlines()
    assert (
        len(pts) == 24
        and len(keep) == 14
        and len(left) == 10
        and all(ln.startswith(f"{mb} ") for ln in left)
    )
    assert (
        f"LEFT ON THE POD (lost when it is deleted, reproducible from the seeds): 10 checkpoints, {10 * mb / 1e9:.2f} GB"
        in r.stdout
    )
    assert sorted((d / "need").read_text().split()) == sorted(
        p for p in pts if p.startswith("bases/") or p.endswith("/final.pt")
    )
    r, d = run("full", pts, "full")
    assert (
        r.returncode == 0 and len((d / "keep").read_text().split()) == 23
    )  # all but the alarm checkpoint
    r, d = run("light", pts, "light")
    assert (
        r.returncode == 0
        and (d / "keep").read_text() == ""
        and len((d / "left").read_text().splitlines()) == 24
    )
    r, _ = run("keep", [p for p in pts if p != "arms/dposwap_c24_reit_s0/final.pt"], "nofinal")
    assert (
        r.returncode == 1
        and "the pod has arms/dposwap_c24_reit_s0/done.json but its checkpoint listing has no arms/dposwap_c24_reit_s0/final.pt"
        in r.stdout
    )
    r, _ = run(
        "keep", [], "zero"
    )  # zero checkpoints while the pod's records name bases and finished runs
    assert r.returncode == 1 and "no bases/stage1_c24_s0.pt" in r.stdout


FAKE_SSH = r"""#!/bin/bash
# a pod on this machine: the remote command runs locally with /workspace/lh -> $FAKEPOD. FAKE_FAIL / FAKE_CUT / FAKE_EMPTY
# = <kind>:<n> make the n-th listing of a kind (list, pt, md5) exit 255, lose its last line, or send only a 0 sentinel.
while [[ $1 == -o ]]; do shift 2; done
shift
cmd="$*"
cmd=${cmd//\/workspace\/lh/$FAKEPOD}
case "$cmd" in *md5sum*) kind=md5 ;; *"-name '*.pt' -printf"*) kind=pt ;; *) kind=list ;; esac
n=$(( $(cat "$FAKEPOD.$kind.n" 2>/dev/null || echo 0) + 1 )); echo "$n" > "$FAKEPOD.$kind.n"
[[ ${FAKE_FAIL:-} == "$kind:$n" ]] && exit 255
if [[ ${FAKE_CUT:-} == "$kind:$n" ]]; then bash -c "$cmd" | head -n -1; exit 0; fi
if [[ ${FAKE_EMPTY:-} == "$kind:$n" ]]; then cat > /dev/null; echo "LISTING_END 0"; exit 0; fi
exec bash -c "$cmd"
"""
FAKE_RSYNC = r"""#!/bin/bash
args=(); skip=0
for a in "$@"; do
  if (( skip )); then skip=0; continue; fi
  if [[ $a == -e ]]; then skip=1; continue; fi
  args+=("${a//rp-test:\/workspace\/lh/$FAKEPOD}")
done
"$REAL_RSYNC" "${args[@]}"; rc=$?
[[ -n ${FAKE_RM_AFTER:-} ]] && rm -f "$FAKE_RM_AFTER"
exit $rc
"""
POD_FILES = {"bases/stage1_c24_s0.pt": "b", "bases/stage1_c24_s0.tag.json": "{}", "measure/c24_s0/marker_plain__none.jsonl": "{}",
             "arms/dpo_c24_reit_s0/snap_step5.pt": "5", "arms/dpo_c24_reit_s0/snap_step10.pt": "10",
             "arms/dpo_c24_reit_s0/final.pt": "f", "arms/dpo_c24_reit_s0/done.json": "{}",
             "arms/dpo_c24_reit_s0/measure/step10/DONE": "x", "arms/dpo_c24_reit_s0/x_alarm.pt": "a",
             "runs/rl_c24_anc_s0/seg1/ckpts/rl_c24_anc_s0_it10_snap.pt": "s10", "runs/rl_c24_anc_s0/seg1/ckpts/rl_c24_anc_s0_it20_snap.pt": "s20",
             "runs/rl_c24_anc_s0/seg1/ckpts/rl_c24_anc_s0_it20_rolling.pt": "r", "pod/c24.heartbeat.json": "{}"}  # fmt: skip


def fake_pod(tmp_path: Path) -> tuple[Path, Path]:
    pod, fake = tmp_path / "pod", tmp_path / "bin"
    fake.mkdir()
    for n, body in (("ssh", FAKE_SSH), ("rsync", FAKE_RSYNC)):
        (fake / n).write_text(body)
        (fake / n).chmod(0o755)
    for k, v in POD_FILES.items():
        (pod / k).parent.mkdir(parents=True, exist_ok=True)
        (pod / k).write_text(v)
    return pod, fake


def fetch(
    tmp_path: Path, pod: Path, fake: Path, name: str, mode: str = "--keep", **faults: str
) -> subprocess.CompletedProcess:
    for f in pod.parent.glob(f"{pod.name}.*.n"):
        f.unlink()
    env = {**os.environ, "PATH": f"{fake}:{os.environ['PATH']}", "FAKEPOD": str(pod), "REAL_RSYNC": shutil.which("rsync"),
           "LH": str(tmp_path / f"lh_{name}"), **faults}  # fmt: skip
    return subprocess.run(["bash", str(HERE / "lh_pod.sh"), "fetch", "rp-test", *([mode] if mode else [])], env=env,
                          capture_output=True, text=True, timeout=120)  # fmt: skip


def fetched_pts(lh: Path) -> list[str]:
    return sorted(str(p.relative_to(lh)) for p in lh.rglob("*.pt") if "pod_stage" not in p.parts)


@pytest.mark.skipif(not shutil.which("rsync"), reason="no rsync")
def test_fetch_modes_end_to_end_against_a_local_pod(tmp_path):
    """lh_pod.sh fetch, run for real against a pod tree on this machine (ssh and rsync replaced on PATH): full fetches
    every checkpoint but *_alarm.pt, --light none, --keep the chosen ones; each ends in its FETCH_OK line; --keep also
    prints the count and size of what it left on the pod and lists it."""
    pod, fake = fake_pod(tmp_path)
    allpt = sorted(k for k in POD_FILES if k.endswith(".pt") and "_alarm" not in k)
    keep = ["arms/dpo_c24_reit_s0/final.pt", "arms/dpo_c24_reit_s0/snap_step10.pt", "bases/stage1_c24_s0.pt",
            "runs/rl_c24_anc_s0/seg1/ckpts/rl_c24_anc_s0_it10_snap.pt", "runs/rl_c24_anc_s0/seg1/ckpts/rl_c24_anc_s0_it20_snap.pt"]  # fmt: skip
    for mode, want, last in (
        ("--keep", keep, "ckpt=5 left_on_pod=3"),
        ("--light", [], "ckpt=0 left_on_pod=8"),
        ("", allpt, "ckpt=7 left_on_pod=1"),
    ):
        r = fetch(tmp_path, pod, fake, mode or "full", mode)
        lh = tmp_path / f"lh_{mode or 'full'}"
        assert r.returncode == 0, r.stdout + r.stderr
        assert fetched_pts(lh) == want and (lh / "measure/c24_s0/marker_plain__none.jsonl").exists()
        files = len((lh / "pod_stage" / "rp-test.fetch.list").read_text().splitlines())
        m = mode.lstrip("-") or "full"
        assert (
            r.stdout.strip()
            .splitlines()[-1]
            .startswith(f"FETCH_OK rp-test mode={m} files={files} {last} (")
        )
    r = fetch(tmp_path, pod, fake, "keep2")
    assert (
        "LEFT ON THE POD (lost when it is deleted, reproducible from the seeds): 3 checkpoints"
        in r.stdout
    )
    left = [
        ln.split(" ", 1)[1]
        for ln in (tmp_path / "lh_keep2" / "pod_stage" / "rp-test.fetch.left")
        .read_text()
        .splitlines()
    ]
    assert left == ["arms/dpo_c24_reit_s0/snap_step5.pt", "arms/dpo_c24_reit_s0/x_alarm.pt",
                    "runs/rl_c24_anc_s0/seg1/ckpts/rl_c24_anc_s0_it20_rolling.pt"]  # fmt: skip
    r = fetch(tmp_path, pod, fake, "all", "--all")
    assert r.returncode == 2 and "unknown mode --all" in r.stdout


@pytest.mark.skipif(not shutil.which("rsync"), reason="no rsync")
@pytest.mark.parametrize(("faults", "why"), [
    ({"FAKE_FAIL": "list:1"}, "ssh exited 255 while listing rp-test.fetch.list"),
    ({"FAKE_FAIL": "pt:1"}, "ssh exited 255 while listing rp-test.fetch.pt"),
    ({"FAKE_FAIL": "md5:1"}, "ssh exited 255 while listing rp-test.fetch.pod.md5"),
    ({"FAKE_FAIL": "md5:2"}, "ssh exited 255 while listing rp-test.fetch.pod2.md5"),
    ({"FAKE_CUT": "list:1"}, "the listing rp-test.fetch.list is cut off (no LISTING_END line)"),
    ({"FAKE_CUT": "md5:2"}, "the listing rp-test.fetch.pod2.md5 is cut off (no LISTING_END line)"),
    ({"FAKE_EMPTY": "list:1"}, "the pod lists no file under /workspace/lh"),
    ({"FAKE_EMPTY": "pt:1"}, "the pod has bases/stage1_c24_s0.tag.json but its checkpoint listing has no bases/stage1_c24_s0.pt"),
    ({"FAKE_EMPTY": "md5:2"}, "the pod's second md5 listing lacks 10 of 10 listed files"),
])  # fmt: skip
def test_fetch_fails_on_a_bad_listing(tmp_path, faults, why):
    """Audit W1: a failed, cut-off or empty listing (the files, the checkpoints, either md5 listing) is a hard failure
    with a reason and never a FETCH_OK line, in every mode it concerns."""
    pod, fake = fake_pod(tmp_path)
    r = fetch(tmp_path, pod, fake, "bad", **faults)
    assert r.returncode == 1 and why in r.stdout, r.stdout + r.stderr
    assert "FETCH_OK" not in r.stdout and "every stable fetched file equal" not in r.stdout


@pytest.mark.skipif(not shutil.which("rsync"), reason="no rsync")
def test_fetch_fails_on_a_missing_final_or_a_vanished_file(tmp_path):
    """A finished run (done.json) whose final.pt is gone is refused (full and --keep; --light fetches no checkpoint);
    a file that vanishes between the listing and the md5 listing fails the fetch."""
    pod, fake = fake_pod(tmp_path)
    (pod / "arms/dpo_c24_reit_s0/final.pt").unlink()
    for mode in ("--keep", ""):
        r = fetch(tmp_path, pod, fake, f"nofinal{mode}", mode)
        assert (
            r.returncode == 1
            and "no arms/dpo_c24_reit_s0/final.pt" in r.stdout
            and "FETCH_OK" not in r.stdout
        )
    assert fetch(tmp_path, pod, fake, "nofinal_light", "--light").returncode == 0
    (pod / "arms/dpo_c24_reit_s0/final.pt").write_text("f")
    (fake / "md5sum").write_text(
        f'#!/bin/bash\nrm -f {pod}/measure/c24_s0/marker_plain__none.jsonl\nexec /usr/bin/md5sum "$@"\n'
    )
    (fake / "md5sum").chmod(0o755)
    r = fetch(tmp_path, pod, fake, "vanished")
    assert r.returncode == 1 and "FETCH_OK" not in r.stdout, r.stdout + r.stderr
    assert (
        "listed files vanished on the pod before their md5 (a job still running? fetch again after it): measure/c24_s0/marker_plain__none.jsonl"
        in r.stdout
    )
    (fake / "md5sum").unlink()
    (pod / "measure/c24_s0/marker_plain__none.jsonl").write_text("{}")
    lh = tmp_path / "lh_rmafter"
    r = fetch(
        tmp_path, pod, fake, "rmafter", FAKE_RM_AFTER=str(lh / "arms/dpo_c24_reit_s0/final.pt")
    )  # audit W7
    assert (
        r.returncode == 1
        and "serrano's md5 of the fetched files failed" in r.stdout
        and "final.pt" in r.stdout
    )


def test_syntax():
    for f in ("lh_pod.sh", "lh_driver.sh", "lh_pod_setup.sh", "lh_rl2.sh"):
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
    for f in ("exp/linked_habits_20261127/lh_launch.py", "exp/linked_habits_20261127/lh_pairs.py", "exp/linked_habits_20261127/lh_driver.sh", "exp/linked_habits_20261127/lh_rl2.sh",
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


B2TASKS = "dpo_c24b1_reit_s0 dposwap_c24b1_reit_s0 full_dpo_c24b1_reit_s0"


def test_driver_second_base_pass(pod):
    """PREREG Amendment 8 point 2: after stage 2 (DONE), base seed 1: pretrain, measure, prep2 and stage 2 of its tasks,
    then BASE2_DONE. A rerun after DONE never enters the first base's stage 2 again and skips what the pass finished."""
    lh, env = pod
    env = {**env, "BASE2": "1", "LH_BASE2_TASKS": B2TASKS}
    (lh / "pod").mkdir()
    (lh / "pod" / "c24.GO").write_text("now")
    assert driver(env).wait(timeout=90) == 0
    cs = [x for x in calls(lh) if not x.startswith("wait")]
    assert cs == ["pretrain --corpus c24 --seed 0 --gpu 0", "measure --base c24_s0 --skip_done --gpu 0", "prep2 --corpus c24 --gpu 0",
                  f"stage2 --corpus c24 --tasks {TASKS.replace(' ', ',')} --parallel 2 --gpu 0 ND_GUARD=off",
                  "pretrain --corpus c24 --seed 1 --gpu 0", "measure --base c24_s1 --skip_done --gpu 0",
                  "prep2 --corpus c24 --base_seed 1 --gpu 0",
                  f"stage2 --corpus c24 --base_seed 1 --tasks {B2TASKS.replace(' ', ',')} --parallel 2 --gpu 0 ND_GUARD="]  # fmt: skip
    pod_ = lh / "pod"
    assert (
        (pod_ / "c24.DONE").exists()
        and (pod_ / "c24.BASE2_DONE").exists()
        and not (pod_ / "c24.FAILED").exists()
    )
    assert json.loads((pod_ / "c24.heartbeat.json").read_text())["stage"] == "base2 done"
    assert (lh / "pairs" / "c24_s1" / "reit" / "summary.json").exists() and (
        lh / "bases" / "stage1_c24_s1.pt"
    ).exists()
    (lh / "stub_calls.txt").unlink()
    assert (
        driver(env).wait(timeout=60) == 0 and not (lh / "stub_calls.txt").exists()
    )  # everything skipped


def test_driver_second_base_after_an_old_stage_2(pod):
    """The operator's path: stage 2 ended under the old driver (DONE, one first-base task failed); `start` with the new
    code runs only the pass (the failed task is not retried while DONE exists) and records the pass's own failures."""
    lh, env = pod
    pod_ = lh / "pod"
    pod_.mkdir()
    (pod_ / "c24.GO").write_text("now")
    assert (
        driver({**env, "STUB_FAIL": "sft_c24_reit_s0"}).wait(timeout=60) == 0
    )  # the old run: BASE2 off
    assert "stage2 sft_c24_reit_s0 did not finish" in (pod_ / "c24.FAILED").read_text()
    (lh / "stub_calls.txt").unlink()
    env2 = {**env, "BASE2": "1", "LH_BASE2_TASKS": B2TASKS, "STUB_FAIL": "dposwap_c24b1_reit_s0"}
    assert driver(env2).wait(timeout=90) == 0
    cs = [x for x in calls(lh) if not x.startswith("wait")]
    assert not any(
        x.startswith("stage2") and "--base_seed" not in x for x in cs
    )  # stage 2 not entered again
    assert cs[0] == "pretrain --corpus c24 --seed 1 --gpu 0" and (pod_ / "c24.BASE2_DONE").exists()
    failed = (pod_ / "c24.FAILED").read_text()
    assert (
        "base2 stage2 dposwap_c24b1_reit_s0 did not finish" in failed
        and "base2 stage2 dpo_c24b1_reit_s0" not in failed
    )


def test_driver_second_base_pretrain_failure_and_switch(pod):
    """A failed pretrain of the second base stops the pass (FAILED, BASE2_DONE, nothing after it); a pod with the switch
    off (c9, c0) never runs it."""
    lh, env = pod
    pod_ = lh / "pod"
    pod_.mkdir()
    (pod_ / "c24.GO").write_text("now")
    assert (
        driver({**env, "BASE2": "1", "LH_BASE2_TASKS": B2TASKS, "STUB_FAIL": "pretrain_s1"}).wait(
            timeout=90
        )
        == 0
    )
    assert (
        "base2 pretrain c24_s1 failed" in (pod_ / "c24.FAILED").read_text()
        and (pod_ / "c24.BASE2_DONE").exists()
    )
    assert not [
        x
        for x in calls(lh)
        if x.startswith(("measure --base c24_s1", "prep2 --corpus c24 --base_seed"))
    ]
    (pod_ / "c0.GO").write_text("now")
    assert (
        driver({**env, "EXTRA": ""}, "c0").wait(timeout=60) == 0
    )  # BASE2=0 (the fixture), as lh_core says for c0
    assert (pod_ / "c0.DONE").exists() and not (pod_ / "c0.BASE2_DONE").exists()


def test_driver_second_base_switch_from_lh_core(pod):
    """Without BASE2 in the environment the driver asks lh_core.BASE2_CORPORA: on for c24, off for c9."""
    lh, env = pod
    env = {k: v for k, v in env.items() if k != "BASE2"}
    pod_ = lh / "pod"
    pod_.mkdir()
    for c in ("c24", "c9"):
        (pod_ / f"{c}.GO").write_text("now")
    assert driver({**env, "LH_BASE2_TASKS": B2TASKS}).wait(timeout=90) == 0
    assert (pod_ / "c24.BASE2_DONE").exists()
    assert driver(env, "c9").wait(timeout=60) == 0
    assert (pod_ / "c9.DONE").exists() and not (pod_ / "c9.BASE2_DONE").exists()


RL2 = "rl_c24_rlhfreit_s1 rl_c24_anc_s1"


def rl2(env: dict, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", str(HERE / "lh_rl2.sh"), "c24", *args], env={**env, "LH_RL2_TASKS": RL2},
                          capture_output=True, text=True, timeout=120)  # fmt: skip


def test_rl2_runs_after_the_second_base_pass(pod):
    """PREREG Amendment 10's one-off: only after c24.BASE2_DONE and with no driver running; one stage-2 job of the two
    RL seed-1 lines (guard off), c24.RL2_DONE at the end, a line that did not finish in c24.FAILED; a rerun skips
    finished lines; --dry-run prints the job's plan and each line's command and launches nothing."""
    lh, env = pod
    pod_ = lh / "pod"
    pod_.mkdir()
    r = rl2(env)
    assert (
        r.returncode == 3
        and "no " in r.stdout
        and "c24.BASE2_DONE: the second base pass has not ended; not started" in r.stdout
    )
    (pod_ / "c24.BASE2_DONE").write_text("x")
    (pod_ / "c24.driver.lock").mkdir()
    r = rl2(env)
    assert (
        r.returncode == 3 and "a driver holds" in r.stdout and not (lh / "stub_calls.txt").exists()
    )
    (pod_ / "c24.driver.lock").rmdir()
    r = rl2(env, "--dry-run")
    assert r.returncode == 0 and calls(lh) == ["stage2 --corpus c24 --rl_seed 1 --parallel 2 --gpu 0 --dry-run ND_GUARD=off",
                                               "rl --line rl_c24_rlhfreit_s1 --gpu 0 --dry-run", "rl --line rl_c24_anc_s1 --gpu 0 --dry-run"]  # fmt: skip
    assert not (pod_ / "c24.RL2_DONE").exists()
    (lh / "stub_calls.txt").unlink()
    r = rl2(env)
    assert r.returncode == 0, r.stdout + r.stderr
    cs = calls(lh)
    assert (
        cs[0]
        == "stage2 --corpus c24 --rl_seed 1 --tasks rl_c24_rlhfreit_s1,rl_c24_anc_s1 --parallel 2 --gpu 0 ND_GUARD=off"
    )
    assert cs[1].startswith("wait --job stage2_c24_rl1 --max_h 8")
    assert (
        (pod_ / "c24.RL2_DONE").exists()
        and not (pod_ / "c24.FAILED").exists()
        and not (pod_ / "c24.driver.lock").exists()
    )
    (lh / "stub_calls.txt").unlink()
    assert (
        rl2(env).returncode == 0 and not (lh / "stub_calls.txt").exists()
    )  # both lines finished: nothing launched


def test_rl2_records_a_line_that_did_not_finish(pod):
    lh, env = pod
    (lh / "pod").mkdir()
    (lh / "pod" / "c24.BASE2_DONE").write_text("x")
    r = rl2({**env, "STUB_FAIL": "rl_c24_anc_s1"})
    assert r.returncode == 0 and (lh / "pod" / "c24.RL2_DONE").exists()
    failed = (lh / "pod" / "c24.FAILED").read_text()
    assert "rl2 rl_c24_anc_s1 did not finish" in failed and "rl_c24_rlhfreit_s1" not in failed
