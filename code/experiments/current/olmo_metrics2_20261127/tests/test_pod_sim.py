"""Local simulation of the pod side (m2_job.sh, m2_status.py, m2_stop.sh) and the serrano-side watcher / push guards,
with a temporary directory standing in for /workspace/m2. No pod, no GPU."""

import gzip
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

X = Path(__file__).resolve().parents[1]
POD = X / "pod"
sys.path.insert(0, str(POD))
import m2_status  # noqa: E402


@pytest.fixture
def root(tmp_path):
    (tmp_path / "meta").mkdir()
    (tmp_path / "meta" / "gpu_status").write_text("GPU_OK 1\n")
    return tmp_path


_BIN = Path(__file__).resolve().parent / "_stubbin"


def _default_stub():  # the host's real nvidia-smi (serrano has lab GPUs) must never decide a test: report 0 MiB
    _BIN.mkdir(exist_ok=True)
    f = _BIN / "nvidia-smi"
    f.write_text(
        '#!/usr/bin/env bash\nif [[ "$*" == *utilization* ]]; then echo "0, 0"; else echo 0; fi\n'
    )
    f.chmod(0o755)
    return f"{_BIN}:{os.environ['PATH']}"


def env_for(root, **kw):
    e = dict(os.environ, M2_ROOT=str(root), HB_INTERVAL="1", PATH=_default_stub())
    e.update({k: str(v) for k, v in kw.items()})
    return e


def start(root, job, to, cmd, **kw):
    return subprocess.Popen(
        ["bash", str(POD / "m2_job.sh"), job, str(to), "--", *cmd],
        env=env_for(root, **kw),
        start_new_session=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def wait_for(pred, t=15):
    end = time.time() + t
    while time.time() < end:
        if pred():
            return True
        time.sleep(0.2)
    return False


def st(root, job, name):
    return root / "runs" / job / "state" / name


def status(root, job):
    return m2_status.parse_line(m2_status.status_lines(str(root), [job])[0])


def alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def test_job_runs_and_done(root):
    sample = "mkdir -p runs/j1/out && echo '{\"a\":1}' | gzip > runs/j1/out/samples.jsonl.gz && sleep 3 && echo fin"
    p = start(root, "j1", 60, ["bash", "-c", sample])
    assert wait_for(lambda: st(root, "j1", "heartbeat").exists())
    s = status(root, "j1")
    assert s["state"] == "RUNNING" and s["hb"].endswith("s")
    hb = st(root, "j1", "heartbeat").read_text()
    assert all(
        k in hb for k in ("ts=", "job=j1", "newest=", "newest_size=", "gpu_mem=", "gpu_util=")
    )
    assert p.wait(30) == 0
    assert st(root, "j1", "DONE").exists() and not st(root, "j1", "RUNNING").exists()
    s = status(root, "j1")
    assert s["state"] == "DONE" and s["rc"] == "0" and s["progress"] == "1"
    assert "fin" in (root / "runs/j1/logs/job.log").read_text()


def test_done_refused_unless_force(root):
    assert start(root, "j2", 30, ["true"]).wait(20) == 0
    assert start(root, "j2", 30, ["true"]).wait(20) == 4
    assert "REFUSED_DONE" in (root / "runs/j2/logs/runner.log").read_text()
    assert start(root, "j2", 30, ["true"], FORCE=1).wait(20) == 0


def test_second_job_refused_while_locked(root):
    a = start(root, "ja", 60, ["sleep", "30"])
    assert wait_for(lambda: st(root, "ja", "RUNNING").exists())
    b = start(root, "jb", 60, ["true"])
    assert b.wait(20) == 3
    assert "LOCKED" in (root / "runs/jb/logs/runner.log").read_text()
    assert status(root, "jb")["state"] == "PENDING"
    # a download-type job (NOGPU) has its own lock
    c = start(root, "jc", 60, ["true"], NOGPU=1)
    assert c.wait(20) == 0
    subprocess.run(
        ["bash", str(POD / "m2_stop.sh"), "ja"], env=env_for(root), capture_output=True, timeout=60
    )
    a.wait(60)


def test_failing_job_records_rc(root):
    p = start(root, "jf", 60, ["bash", "-c", "echo oops; exit 7"])
    assert p.wait(20) == 7
    assert "rc=7" in st(root, "jf", "FAILED").read_text() and not st(root, "jf", "DONE").exists()
    s = status(root, "jf")
    assert s["state"] == "FAILED" and s["rc"] == "7"


def test_timeout_kills_job(root):
    p = start(root, "jt", 2, ["sleep", "30"])
    assert p.wait(30) == 124
    assert "timeout" in st(root, "jt", "FAILED").read_text()


def test_stop_kills_only_that_jobs_group(root, tmp_path_factory):
    other = tmp_path_factory.mktemp("other")
    (other / "meta").mkdir()
    (other / "meta" / "gpu_status").write_text("GPU_OK 1\n")
    a = start(root, "stopme", 120, ["bash", "-c", "sleep 71 & sleep 72; wait"])
    c = start(other, "keepme", 120, ["sleep", "73"])
    bystander = subprocess.Popen(["sleep", "74"], start_new_session=True)
    try:
        assert wait_for(
            lambda: st(root, "stopme", "jobpg").exists() and st(other, "keepme", "jobpg").exists()
        )
        time.sleep(0.5)
        r = subprocess.run(
            ["bash", str(POD / "m2_stop.sh"), "stopme"],
            env=env_for(root),
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert "STOPPED stopme" in r.stdout, r.stdout + r.stderr
        assert a.wait(30) != 0
        assert wait_for(lambda: st(root, "stopme", "FAILED").exists())
        assert "stopped" in st(root, "stopme", "FAILED").read_text()
        ps = subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True).stdout
        assert "sleep 71" not in ps and "sleep 72" not in ps  # children of the stopped job are gone
        assert "sleep 73" in ps and "sleep 74" in ps  # another job and a bystander survive
        assert c.poll() is None and bystander.poll() is None
        assert status(other, "keepme")["state"] == "RUNNING"
    finally:
        for p in (c, bystander):
            p.kill()
        subprocess.run(
            ["bash", str(POD / "m2_stop.sh"), "keepme"], env=env_for(other), capture_output=True
        )


def test_status_parses_heartbeat_and_progress(root):
    run = root / "runs" / "jh"
    (run / "state").mkdir(parents=True)
    (run / "state" / "RUNNING").write_text("x")
    (run / "state" / "pid").write_text(f"{os.getpid()} {os.getpgrp()} now jh\n")
    (run / "state" / "heartbeat").write_text(
        f"ts={int(time.time()) - 125} job=jh newest=a newest_mtime=1 newest_size=2 "
        "gpu_mem=1000 gpu_util=5\n"
    )
    (run / "state" / "expected").write_text("40")
    (run / "s").mkdir()
    with gzip.open(run / "s" / "samples.jsonl.gz", "wt") as f:
        f.write("{}\n" * 7)
    s = status(root, "jh")
    assert s["state"] == "RUNNING" and 120 <= int(s["hb"][:-1]) <= 135 and s["progress"] == "7/40"
    assert s["newest"].endswith("samples.jsonl.gz") and int(s["newest_size"]) > 0
    # dead runner pid with RUNNING marker -> STALE
    (run / "state" / "pid").write_text("999999 999999 now jh\n")
    assert status(root, "jh")["state"] == "STALE"
    assert m2_status.status_lines(str(root), ["nojob"])[0] == "nojob state=ABSENT"


def fake_pod(tmp_path, root):
    f = tmp_path / "fakepod.sh"
    f.write_text(
        f'#!/usr/bin/env bash\n# usage: status <alias> <job>\nshift\nshift\nexec python3 {POD}/m2_status.py --root {root} "$@"\n'
    )
    return f


def watch(tmp_path, root, job, exp, **kw):
    e = dict(
        os.environ,
        M2_DATA=str(tmp_path / "data"),
        POLL="1",
        M2_POD_CMD=str(fake_pod(tmp_path, root)),
    )
    e.update({k: str(v) for k, v in kw.items()})
    return subprocess.Popen(
        ["bash", str(POD / "m2_watch.sh"), "rp-test", job, str(exp)],
        env=e,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def wlog(tmp_path):
    p = tmp_path / "data" / "logs" / "watch_rp-test.log"
    return p.read_text() if p.exists() else ""


def mk_running(root, job, hb_ts, runner_pid):
    run = root / "runs" / job
    (run / "state").mkdir(parents=True)
    (run / "logs").mkdir()
    (run / "logs" / "job.log").write_text("x")
    (run / "state" / "RUNNING").write_text("x")
    (run / "state" / "pid").write_text(f"{runner_pid} 1 now {job}\n")
    (run / "state" / "heartbeat").write_text(
        f"ts={hb_ts} job={job} newest=logs/job.log newest_mtime=1 newest_size=1 "
        "gpu_mem=na gpu_util=na\n"
    )
    return run


def test_watcher_stall_overrun_exit(root, tmp_path):
    sleeper = subprocess.Popen(["sleep", "60"])
    try:
        run = mk_running(root, "jw", int(time.time()) - 1200, sleeper.pid)  # heartbeat 20 min old
        os.utime(
            run / "state" / "RUNNING", (time.time() - 300, time.time() - 300)
        )  # started 5 min ago, expected 1 min
        w = watch(tmp_path, root, "jw", 1)
        assert wait_for(
            lambda: "STALL heartbeat" in wlog(tmp_path) and "OVERRUN" in wlog(tmp_path), 20
        ), wlog(tmp_path)
        assert w.poll() is None
        (run / "state" / "RUNNING").unlink()
        (run / "state" / "DONE").write_text("now rc=0\n")
        assert w.wait(20) == 0
        log = wlog(tmp_path)
        assert "EXIT state=DONE" in log and "PT)" in log and "Z (" in log
        assert os.kill(sleeper.pid, 0) is None  # the watcher never killed anything
    finally:
        sleeper.kill()


def test_watcher_no_growth_stall(root, tmp_path):
    sleeper = subprocess.Popen(["sleep", "60"])
    try:
        run = mk_running(
            root, "jg", int(time.time()) + 100000, sleeper.pid
        )  # heartbeat looks fresh
        w = watch(tmp_path, root, "jg", 600, GROW_STALL=3)
        assert wait_for(lambda: "STALL newest output file" in wlog(tmp_path), 20), wlog(tmp_path)
        assert "STALL heartbeat" not in wlog(tmp_path) and "OVERRUN" not in wlog(tmp_path)
        (run / "state" / "RUNNING").unlink()
        (run / "state" / "FAILED").write_text("now rc=3\n")
        assert w.wait(20) == 0 and "EXIT state=FAILED rc=3" in wlog(tmp_path)
    finally:
        sleeper.kill()


def test_push_refuses_held_out_and_credential_names(tmp_path):
    for bad in ("textbook72_prompts.jsonl", "pids_tb72.txt", "test_13_20.json", "id_runpod"):
        data = tmp_path / bad.replace(".", "_")
        (data / "inputs").mkdir(parents=True)
        (data / "inputs" / bad).write_text("x")
        r = subprocess.run(
            ["bash", str(POD / "m2_pod.sh"), "push", "rp-t1", "--dry-run"],
            env=dict(os.environ, M2_DATA=str(data)),
            capture_output=True,
            text=True,
        )
        assert r.returncode == 1 and "REFUSED" in r.stdout and bad in r.stdout, r.stdout
        assert "rsync" not in r.stdout
    ok = tmp_path / "ok"
    (ok / "inputs").mkdir(parents=True)
    (ok / "inputs" / "prompts_a.jsonl").write_text("x")
    r = subprocess.run(
        ["bash", str(POD / "m2_pod.sh"), "push", "rp-t1", "--dry-run"],
        env=dict(os.environ, M2_DATA=str(ok)),
        capture_output=True,
        text=True,
    )
    assert (
        r.returncode == 0
        and "rsync" in r.stdout
        and "pod/oe_setup.sh" in (ok / "pod_stage" / "rp-t1.manifest.txt").read_text()
    )


def test_verify_weights_and_download_skip(tmp_path):
    dan = tmp_path / "dan" / "olmo_ifcurve"
    dan.mkdir(parents=True)
    (dan / "checkpoints.csv").write_text(
        "name,repo,revision,step,commit,n_shards,weight_bytes\nck,o/r,main,0,abc123,2,10\n"
    )
    md = tmp_path / "models" / "ck"
    md.mkdir(parents=True)
    for i in range(2):
        (md / f"m{i}.safetensors").write_bytes(b"12345")
    (md / ".sha").write_text("abc123\n")
    args = [
        "python3",
        str(POD / "m2_verify_weights.py"),
        "ck",
        "--models",
        str(tmp_path / "models"),
        "--dan",
        str(tmp_path / "dan"),
    ]
    assert subprocess.run(args, capture_output=True).returncode == 1  # no .complete yet
    (md / ".complete").write_text("")
    r = subprocess.run(args, capture_output=True, text=True)
    assert r.returncode == 0 and '"ok": true' in r.stdout
    (md / "m1.safetensors").write_bytes(b"1")  # wrong bytes
    assert subprocess.run(args, capture_output=True).returncode == 1
    (md / "m1.safetensors").write_bytes(b"12345")
    # m2_download.sh with .complete present: skips the fetch (none is installed here), verifies, records the json
    e = dict(
        os.environ,
        DAN_ROOT=str(tmp_path / "dan"),
        MODELS_ROOT=str(tmp_path / "models"),
        M2_ROOT=str(tmp_path / "m2"),
    )
    (tmp_path / "m2" / "runs" / "download_ck").mkdir(parents=True)
    r = subprocess.run(
        ["bash", str(POD / "m2_download.sh"), "ck"], env=e, capture_output=True, text=True
    )
    assert r.returncode == 0 and "already complete" in r.stdout
    assert '"ok": true' in (tmp_path / "m2/runs/download_ck/download_verify.json").read_text()


# ---- GPU-free gate, grandchild kill, chains, stage of temp/rank1/steer


def stub_nvidia(tmp_path, mib):
    b = tmp_path / "bin"
    b.mkdir(exist_ok=True)
    f = b / "nvidia-smi"
    f.write_text(
        f'#!/usr/bin/env bash\nif [[ "$*" == *utilization* ]]; then echo "{mib}, 90"; else echo {mib}; fi\n'
    )
    f.chmod(0o755)
    return dict(PATH=f"{b}:{os.environ['PATH']}")  # overrides env_for's default stub


def test_gpu_busy_refuses(root, tmp_path):
    p = start(root, "jb", 30, ["true"], GPU_WAIT=2, GPU_POLL=1, **stub_nvidia(tmp_path, 5000))
    assert p.wait(30) == 5
    assert "GPU_BUSY" in (root / "runs/jb/logs/runner.log").read_text()
    assert "rc=5" in st(root, "jb", "FAILED").read_text() and not st(root, "jb", "RUNNING").exists()
    assert status(root, "jb")["state"] == "FAILED"
    # free GPU: starts; --nogpu jobs skip the gate
    assert (
        start(root, "jok", 30, ["true"], GPU_WAIT=2, GPU_POLL=1, **stub_nvidia(tmp_path, 100)).wait(
            30
        )
        == 0
    )
    assert (
        start(root, "jng", 30, ["true"], NOGPU=1, GPU_WAIT=2, **stub_nvidia(tmp_path, 5000)).wait(
            30
        )
        == 0
    )


def test_gpu_gate_waits_until_free(root, tmp_path):
    env = stub_nvidia(tmp_path, 5000)
    p = start(root, "jw2", 30, ["true"], GPU_WAIT=20, GPU_POLL=1, **env)
    time.sleep(2)
    assert p.poll() is None  # still waiting
    stub_nvidia(tmp_path, 100)  # memory freed
    assert p.wait(30) == 0


GRANDCHILD = "bash -c 'trap \"\" TERM; sleep 76' & echo $! > {pidfile}; sleep 77; wait"


def test_stop_kills_term_ignoring_grandchild(root, tmp_path):
    pf = tmp_path / "gc.pid"
    a = start(root, "gk", 120, ["bash", "-c", GRANDCHILD.format(pidfile=pf)], STOP_GRACE=3)
    assert wait_for(lambda: pf.exists() and st(root, "gk", "jobpg").exists())
    time.sleep(0.5)
    gc = int(pf.read_text())
    assert alive(gc)
    r = subprocess.run(
        ["bash", str(POD / "m2_stop.sh"), "gk"],
        env=env_for(root, STOP_GRACE=3),
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert "STOPPED gk" in r.stdout and "SIGKILL command" in r.stdout, r.stdout
    a.wait(30)
    assert wait_for(
        lambda: (
            not subprocess.run(["ps", "-o", "stat=", "-p", str(gc)], capture_output=True, text=True)
            .stdout.strip()
            .replace("Z", "")
        ),
        10,
    )


def test_runner_sweeps_grandchild_after_normal_exit(root, tmp_path):
    pf = tmp_path / "gc2.pid"
    cmd = "bash -c 'trap \"\" TERM; sleep 78' & echo $! > " + str(pf) + "; sleep 1"
    p = start(root, "gs", 60, ["bash", "-c", cmd], KILL_GRACE=2)
    assert p.wait(30) == 0
    gc = int(pf.read_text())
    assert wait_for(
        lambda: (
            not subprocess.run(["ps", "-o", "stat=", "-p", str(gc)], capture_output=True, text=True)
            .stdout.strip()
            .replace("Z", "")
        ),
        10,
    )
    assert "SIGKILL to command group" in (root / "runs/gs/logs/runner.log").read_text()


def chain_start(root, name, lines, **kw):
    d = root / "runs" / "_chains" / name
    d.mkdir(parents=True)
    (d / "chain.txt").write_text("\n".join(lines) + "\n")
    return subprocess.Popen(
        ["bash", str(POD / "m2_chain.sh"), name],
        env=env_for(root, **kw),
        start_new_session=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def chain_state(root, name):
    return m2_status.parse_line(m2_status.status_lines(str(root), [f"chain:{name}"])[0])


def test_chain_runs_in_order(root):
    p = chain_start(
        root,
        "c1",
        [
            "# comment",
            "",
            f"a 30 0 -- echo A >> {root}/order.txt",
            f"b 30 0 -- echo B >> {root}/order.txt",
            f"c 30 0 -- echo C >> {root}/order.txt",
        ],
    )
    assert p.wait(60) == 0
    assert (root / "order.txt").read_text().split() == ["A", "B", "C"]
    s = chain_state(root, "c1")
    assert s["state"] == "DONE" and s["progress"] == "3/3" and s["rcs"] == "a:0,b:0,c:0"
    assert all(st(root, j, "DONE").exists() for j in "abc")
    # re-run resumes: everything is skipped
    (root / "runs/_chains/c1/state/DONE").unlink()
    p = subprocess.Popen(
        ["bash", str(POD / "m2_chain.sh"), "c1"], env=env_for(root), stdout=subprocess.DEVNULL
    )
    assert p.wait(60) == 0 and (root / "order.txt").read_text().split() == ["A", "B", "C"]
    assert "SKIP_DONE" in (root / "runs/_chains/c1/log/chain.log").read_text()


def test_chain_stops_on_non_continue_code(root):
    p = chain_start(
        root, "c2", ["a 30 0 -- true", "b 30 0 -- exit 3", f"c 30 0 -- touch {root}/c_ran"]
    )
    assert p.wait(60) == 1
    assert not (root / "c_ran").exists() and not (root / "runs/c/state").exists()
    s = chain_state(root, "c2")
    assert s["state"] == "FAILED" and s["rcs"] == "a:0,b:3" and "b_rc=3" in s["stopped"]


def test_chain_continue_code_11_proceeds(root):
    p = chain_start(root, "c3", ["a 30 0,11 -- exit 11", f"b 30 0 -- touch {root}/b_ran"])
    assert p.wait(60) == 0 and (root / "b_ran").exists()
    assert chain_state(root, "c3")["rcs"] == "a:11,b:0"


def test_chain_stop_flag_stops_current_and_blocks_rest(root):
    p = chain_start(root, "c4", ["a 60 0 -- sleep 50", f"b 30 0 -- touch {root}/b4_ran"])
    assert wait_for(lambda: st(root, "a", "jobpg").exists())
    r = subprocess.run(
        ["bash", str(POD / "m2_stop.sh"), "--chain", "c4"],
        env=env_for(root, STOP_GRACE=5),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert "STOP flag set" in r.stdout and "STOPPED a" in r.stdout, r.stdout
    assert p.wait(30) == 1 and not (root / "b4_ran").exists()
    assert chain_state(root, "c4")["state"] == "FAILED"


def test_nogpu_job_runs_beside_gpu_job(root):
    a = start(root, "gpujob", 60, ["sleep", "30"])
    assert wait_for(lambda: st(root, "gpujob", "RUNNING").exists())
    p = chain_start(root, "c5", [f"cpu 30 0 --nogpu -- touch {root}/cpu_ran"])
    assert p.wait(30) == 0 and (root / "cpu_ran").exists()
    assert start(root, "gpu2", 30, ["true"]).wait(20) == 3  # a second GPU job is still refused
    subprocess.run(
        ["bash", str(POD / "m2_stop.sh"), "gpujob"],
        env=env_for(root, STOP_GRACE=5),
        capture_output=True,
        timeout=60,
    )
    a.wait(30)


def test_chain_command_validation_and_dry_run(tmp_path):
    good = tmp_path / "good.txt"
    good.write_text("# c\na 30 0 -- echo hi\nb 60 0,11 --nogpu -- bash -c 'x; y'\n")
    run = lambda f: subprocess.run(
        ["bash", str(POD / "m2_pod.sh"), "chain", "rp-t1", "cx", str(f), "--dry-run"],  # noqa: E731
        env=dict(os.environ, M2_DATA=str(tmp_path / "d")),
        capture_output=True,
        text=True,
    )
    r = run(good)
    assert r.returncode == 0 and "m2_chain.sh cx" in r.stdout and "2 jobs" in r.stdout, (
        r.stdout + r.stderr
    )
    for bad in ("a x 0 -- echo", "a 30 0 echo hi", "a 30 -- echo", "a 30 0 --gpu -- echo"):
        f = tmp_path / "bad.txt"
        f.write_text(bad + "\n")
        assert run(f).returncode == 2, bad


def test_stage_includes_temp_rank1_steer(tmp_path):
    fake = tmp_path / "X"
    shutil_copy = __import__("shutil").copytree
    shutil_copy(POD, fake / "pod", ignore=lambda d, n: [x for x in n if x == "__pycache__"])
    (fake / "steer").mkdir()
    (fake / "steer" / "steer_hf.py").write_text("x = 1\n")
    (fake / "temp").mkdir()
    (fake / "temp" / "t.py").write_text("x = 1\n")
    r = subprocess.run(
        ["bash", str(fake / "pod" / "m2_pod.sh"), "push", "rp-t1", "--dry-run"],
        env=dict(os.environ, M2_DATA=str(tmp_path / "d")),
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stdout + r.stderr
    man = (tmp_path / "d" / "pod_stage" / "rp-t1.manifest.txt").read_text().split()
    assert (
        "steer/steer_hf.py" in man
        and "temp/t.py" in man
        and "pod/m2_chain.sh" in man
        and "rank1" not in " ".join(man)
    )
