"""Quiet Codex orchestration: no model calls, SSH, or paid pods in these tests."""

import contextlib
import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest

SOURCE = Path(__file__).parents[1] / "experiments/current/autoresearch"


@pytest.fixture
def loop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    spec = importlib.util.spec_from_file_location("research_loop", SOURCE / "loop.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "HERE", tmp_path / "research")
    monkeypatch.setattr(module, "ART", tmp_path / "artifacts")
    monkeypatch.setattr(module, "ROOT", tmp_path)
    module.HERE.mkdir()
    module.ART.mkdir()
    return module


def launch_fixture(loop: Any, session: Path, pid: int = 12345678) -> Path:
    session.mkdir(exist_ok=True)
    logs = session / f"run-{pid}"
    logs.mkdir(exist_ok=True)
    (session / f"launch-{pid}.json").write_text(
        json.dumps(
            {
                "name": "013_test",
                "seeds": [0, 1, 2],
                "pid": pid,
                "pods": [f"013_test_s{s}-{session.name}" for s in range(3)],
                "logs": str(logs),
            }
        )
    )
    directory = loop.HERE / "results/013_test"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "params.json").write_text(
        json.dumps({"title": "Test learning rate", "baseline_metric": 100})
    )
    return logs


def record_fixture(loop: Any, status: str = "keep", metric: int | None = 110) -> None:
    (loop.HERE / "results/013_test/results.json").write_text(
        json.dumps(
            {"status": status, "metric": metric, "seed_metrics": {"0": 105, "1": 110, "2": 115}}
        )
    )
    (loop.ROOT / "session/completed.json").write_text(
        json.dumps({"name": "013_test", "commit": "fake-outcome-commit"})
    )


def test_only_milestones_partial_lines_and_no_replay(
    loop: Any, tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    session = tmp_path / "session"
    logs = launch_fixture(loop, session)
    log = logs / "013_test_s0.driver.log"
    monitor = loop.Monitor(session)
    log.write_text(
        "[01:00:00] pools: train 154990\nsecret command dump\n"
        "[01:00:01] pretrain step 50 loss 0.8\n"
        "[01:05:00] pretrain held-out greedy 0.1\n"
        "[01:06:00] ei r1: acc 0.25, +12 proofs, solved 8/1500,"
    )
    monitor.poll()
    first = capsys.readouterr().out
    assert "Pretrain started" in first and "RL started" in first
    assert "secret" not in first and "loss" not in first and "round 1" not in first
    with log.open("a") as stream:
        stream.write(" L*=7, 100 samples/s, round 60s\n")
    monitor.poll()
    assert "RL round 1 finished" in capsys.readouterr().out
    monitor.poll()
    assert capsys.readouterr().out == ""
    record_fixture(loop)
    monitor.finish()
    final = capsys.readouterr().out
    assert "NEW BEST" in final and "110" in final and "+10" in final
    assert "105" in final and "115" in final


@pytest.mark.parametrize("status,metric", [("discard", 110), ("keep", 100), ("crash", None)])
def test_no_false_new_best(
    loop: Any, tmp_path: Path, capsys: pytest.CaptureFixture[str], status: str, metric: int | None
):
    session = tmp_path / "session"
    launch_fixture(loop, session)
    record_fixture(loop, status, metric)
    loop.Monitor(session).finish()
    output = capsys.readouterr().out
    assert "NEW BEST" not in output
    assert status in output


def test_missing_or_stale_result_stops_iteration(loop: Any, tmp_path: Path):
    session = tmp_path / "session"
    launch_fixture(loop, session)
    with pytest.raises(RuntimeError, match="recorded result"):
        loop.Monitor(session).finish()
    record_fixture(loop)
    os.utime(loop.HERE / "results/013_test/results.json", (1, 1))
    with pytest.raises(RuntimeError, match="recorded result"):
        loop.Monitor(session).finish()


def test_unpublished_outcome_stops_iteration(loop: Any, tmp_path: Path):
    session = tmp_path / "session"
    launch_fixture(loop, session)
    record_fixture(loop)
    receipt = session / "completed.json"
    receipt.unlink(missing_ok=True)
    with pytest.raises(loop.ExperimentBlocked, match="not committed and pushed"):
        loop.Monitor(session).finish()


def test_cleanup_only_owned_pods_even_without_state_file(
    loop: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    session = tmp_path / "session"
    launch_fixture(loop, session)
    calls = []

    def fake_run(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append(argv)
        if "list" in argv:
            payload = {
                "pods": [
                    {"id": "ours", "name": "autoresearch-013_test_s0-session"},
                    {"id": "theirs", "name": "autoresearch-klaus"},
                    {"id": "similar", "name": "autoresearch-013_test_s0-another-session"},
                ]
            }
            return subprocess.CompletedProcess(argv, 0, json.dumps(payload), "")
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(loop.subprocess, "run", fake_run)
    loop.cleanup_pods(session)
    kills = [argv for argv in calls if "kill" in argv]
    assert len(kills) == 1
    assert kills[0][-4:] == ["kill", "ours", "--pod", "013_test_s0-session"]


def test_cleanup_failure_is_visible(
    loop: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    session = tmp_path / "session"
    launch_fixture(loop, session)
    monkeypatch.setattr(
        loop.subprocess,
        "run",
        lambda *a, **kw: subprocess.CompletedProcess(a[0], 1, "", "SECRET verbose failure"),
    )
    assert not loop.cleanup_pods(session)
    output = capsys.readouterr().out
    assert "ERROR" in output and "cleanup" in output.lower()
    assert "SECRET" not in output


def test_real_process_group_termination_including_child(loop: Any, tmp_path: Path):
    child_file = tmp_path / "child.pid"
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import subprocess,sys,time,pathlib; "
            "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); "
            f"pathlib.Path({str(child_file)!r}).write_text(str(p.pid)); time.sleep(60)",
        ],
        start_new_session=True,
    )
    try:
        deadline = time.monotonic() + 5
        while not child_file.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert child_file.exists()
        loop.stop_processes(process, tmp_path)
        assert process.wait(timeout=5) != 0
        # A killed orphan can briefly remain a zombie until the OS reaps it.
        state = subprocess.run(
            ["ps", "-o", "stat=", "-p", child_file.read_text()], capture_output=True, text=True
        ).stdout.strip()
        assert not state or state.startswith("Z")
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def test_process_signal_permission_error_does_not_abort_cleanup(
    loop: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    session = tmp_path / "session"
    launch_fixture(loop, session)
    signals = []

    def denied(group: int, sig: signal.Signals) -> None:
        signals.append(sig)
        raise PermissionError("Operation not permitted")

    monkeypatch.setattr(loop.os, "killpg", denied)
    loop.stop_processes(None, session)
    assert signals == [signal.SIGTERM, signal.SIGKILL]
    assert "could not signal process group" in capsys.readouterr().out


def test_process_wait_after_kill_is_bounded(
    loop: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    class UnresponsiveProcess:
        pid = 12345678

        def wait(self, *, timeout: int) -> None:
            assert timeout <= 3
            raise subprocess.TimeoutExpired("fake process", timeout)

    monkeypatch.setattr(loop.os, "killpg", lambda *args: None)
    loop.stop_processes(UnresponsiveProcess(), tmp_path)
    assert "did not exit" in capsys.readouterr().out


def test_startup_message_precedes_codex_launch(
    loop: Any, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    before_launch = []

    def fake_start(*args: Any, **kwargs: Any) -> None:
        before_launch.append(capsys.readouterr().out)
        raise OSError("fake launch failure")

    monkeypatch.setattr(loop.subprocess, "Popen", fake_start)
    assert loop.run_loop(["codex"], "one experiment") == 1
    assert len(before_launch) == 1
    assert "Starting Codex | checking experiment state" in before_launch[0]
    assert "diagnostics:" in before_launch[0]


def test_blocked_agent_explains_why_without_streaming_other_events(
    loop: Any, tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    reason = (
        "Blocked before preregistration. The checkout contains unfinished runner/launcher "
        "changes, and make -C code check fails with 64 existing Ruff errors in pretrain_poc."
    )
    events = [
        {"type": "item.completed", "item": {"type": "agent_message", "text": "Initial plan"}},
        {"type": "item.completed", "item": {"type": "reasoning", "text": "PRIVATE thoughts"}},
        {
            "type": "item.completed",
            "item": {"type": "command_execution", "aggregated_output": "PRIVATE tool output"},
        },
        {"type": "item.completed", "item": {"type": "agent_message", "text": reason}},
        {"type": "turn.completed"},
    ]
    fake = tmp_path / "blocked_codex.py"
    fake.write_text(f"print({chr(10).join(json.dumps(e) for e in events)!r})\n")
    assert loop.run_loop([sys.executable, str(fake)], "one experiment") == 1
    output = capsys.readouterr().out
    assert "BLOCKED | No experiment launched." in output
    assert reason in output
    assert "PRIVATE" not in output and "Initial plan" not in output
    assert "must finish exactly one" not in output
    assert output.count("Starting Codex") == 1


def test_agent_explanation_is_one_line_bounded_and_handles_partial_logs(loop: Any, tmp_path: Path):
    log = tmp_path / "codex.jsonl"
    assert loop.agent_explanation(tmp_path) == "No agent explanation was recorded."
    event = {
        "type": "item.completed",
        "item": {"type": "agent_message", "text": "Blocked.\n\n" + "detail " * 1000},
    }
    log.write_text("not JSON\nnull\n" + json.dumps(event) + '\n{"truncated":')
    explanation = loop.agent_explanation(tmp_path)
    assert explanation.startswith("Agent: Blocked. detail")
    assert "\n" not in explanation and len(explanation) <= 400


def test_multiple_experiments_remain_an_error_not_a_blocker(loop: Any, tmp_path: Path):
    session = tmp_path / "session"
    launch_fixture(loop, session)
    entry = json.loads(next(session.glob("launch-*.json")).read_text())
    entry["name"] = "014_another"
    (session / "launch-second.json").write_text(json.dumps(entry))
    with pytest.raises(RuntimeError, match="found 2") as caught:
        loop.Monitor(session).finish()
    assert not isinstance(caught.value, loop.ExperimentBlocked)


def test_two_fresh_sessions_then_failure_without_terminal_noise(
    loop: Any, tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
):
    fake = tmp_path / "fake_codex.py"
    fake.write_text("""import json, os, pathlib, sys
root = pathlib.Path.cwd()
counter = root / "calls.json"
calls = json.loads(counter.read_text()) if counter.exists() else []
session = pathlib.Path(os.environ["ND_AUTORESEARCH_SESSION"])
calls.append({"argv": sys.argv[1:], "session": str(session)})
counter.write_text(json.dumps(calls))
print('PRIVATE agent reasoning')
print('PRIVATE stderr', file=sys.stderr)
if len(calls) == 3:
    sys.exit(9)
name = f"{len(calls):03d}_fake"
logs = session / "driver"
logs.mkdir()
(session / "launch-1.json").write_text(json.dumps({
    "name": name, "seeds": [0], "pid": None, "pods": [], "logs": str(logs)}))
directory = root / "research" / "results" / name
directory.mkdir(parents=True)
(directory / "params.json").write_text(json.dumps({"baseline_metric": 100}))
(directory / "results.json").write_text(json.dumps({
    "status": "discard", "metric": 90, "seed_metrics": {"0": 90}}))
(session / "completed.json").write_text(json.dumps({"name": name, "commit": "fake-commit"}))
""")
    monkeypatch.setattr(loop, "POLL_SECS", 0.01)
    assert loop.run_loop([sys.executable, str(fake)], "one experiment") == 1
    calls = json.loads((tmp_path / "calls.json").read_text())
    assert len(calls) == 3
    assert len({call["session"] for call in calls}) == 3
    assert all("resume" not in call["argv"] and "--json" in call["argv"] for call in calls)
    output = capsys.readouterr().out
    assert "PRIVATE" not in output
    assert output.count("Starting Codex") == 3
    assert output.count("Experiment finished") == 2
    assert "ERROR" in output and "9" in output
    assert any("PRIVATE" in path.read_text() for path in loop.ART.glob("loop/*/codex.jsonl"))


@pytest.mark.parametrize("cancel_signal", [signal.SIGINT, signal.SIGTERM])
def test_cancel_entire_launcher_with_detached_runner_and_fake_pods(
    loop: Any, tmp_path: Path, cancel_signal: signal.Signals
):
    """Real OS signals/processes; the only pod implementation is a local fake."""
    (loop.HERE / "iteration.md").write_text("Do one experiment")
    runner = tmp_path / "detached_runner.py"
    runner.write_text("""import json, os, pathlib, time
root = pathlib.Path.cwd()
session = pathlib.Path(os.environ["ND_AUTORESEARCH_SESSION"])
name = "owned-" + session.name
logs = session / "driver"
logs.mkdir()
entry = {"name": "013_test", "seeds": [0], "pid": os.getpid(),
         "pods": [name], "logs": str(logs)}
(session / "launch-1.json").write_text(json.dumps(entry))
(root / "ready.json").write_text(json.dumps({"pid": os.getpid(), "pod": name}))
time.sleep(60)
""")
    fake = tmp_path / "codex"
    fake.write_text(f"""#!{sys.executable}
import pathlib, subprocess, sys, time
print("PRIVATE commentary", flush=True)
subprocess.Popen([sys.executable, str(pathlib.Path.cwd() / "detached_runner.py")],
                 start_new_session=True)
time.sleep(60)
""")
    fake.chmod(0o700)
    (loop.HERE / "pod.py").write_text("""import json, pathlib, sys
root = pathlib.Path.cwd()
ready = json.loads((root / "ready.json").read_text())
if sys.argv[1] == "list":
    print(json.dumps({"pods": [{"name": "autoresearch-" + ready["pod"], "id": "ours"},
                              {"name": "autoresearch-klaus", "id": "theirs"}]}))
else:
    (root / "cleanup-called.json").write_text(json.dumps(sys.argv[1:]))
""")
    bootstrap = f"""import importlib.util, pathlib, sys
spec = importlib.util.spec_from_file_location("research_loop", {str(SOURCE / "loop.py")!r})
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
m.ROOT = pathlib.Path({str(tmp_path)!r})
m.HERE = m.ROOT / "research"
m.ART = m.ROOT / "artifacts"
m.POLL_SECS = 0.01
sys.argv = ["loop.py"]
sys.exit(m.main())
"""
    process = subprocess.Popen(
        [sys.executable, "-c", bootstrap],
        cwd=tmp_path,
        env={**os.environ, "PATH": f"{tmp_path}:{os.environ['PATH']}", "RUNPOD_API_KEY": "fake"},
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    ready = tmp_path / "ready.json"
    try:
        deadline = time.monotonic() + 10
        while not ready.exists() and process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.01)
        assert ready.exists(), process.communicate(timeout=2)
        process.send_signal(cancel_signal)
        output, errors = process.communicate(timeout=10)
        assert process.returncode == 130, (output, errors)
        assert "Cancelled" in output and "Cleanup | terminated" in output
        assert "PRIVATE" not in output and errors == ""
        assert json.loads((tmp_path / "cleanup-called.json").read_text())[:2] == ["kill", "ours"]
        state = subprocess.run(
            ["ps", "-o", "stat=", "-p", str(json.loads(ready.read_text())["pid"])],
            capture_output=True,
            text=True,
        ).stdout.strip()
        assert not state or state.startswith("Z")
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        if ready.exists():
            with contextlib.suppress(ProcessLookupError):
                os.killpg(json.loads(ready.read_text())["pid"], signal.SIGKILL)


def test_runner_registers_ownership_before_starting_threads(tmp_path: Path):
    session = tmp_path / "session-test"
    session.mkdir()
    script = f"""import json, pathlib, sys
sys.path.insert(0, {str(SOURCE)!r})
import run_idea as r
r.register = lambda *args: None
r.ensure_idle = lambda *args: None
r.ART = {str(tmp_path / "artifacts")!r}
r.time.sleep = lambda _: None
session = pathlib.Path({str(session)!r})
def seed(name, s, logs, pod):
    manifest = list(session.glob("launch-*.json"))[0]
    entry = json.loads(manifest.read_text())
    assert entry["pid"] is not None and pod in entry["pods"]
    assert str(s) in pod
    pathlib.Path(logs, f"seed-{{s}}").touch()
r.one = seed
keys = ["status", "metric", "metric_rl", "seed_metrics", "seed_status"]
r.merge = lambda *args: dict.fromkeys(keys)
sys.argv = ["run_idea.py", "013_test", "--seeds", "0,1,2"]
r.main()
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env={**os.environ, "ND_AUTORESEARCH_SESSION": str(session)},
        timeout=10,
    )
    assert completed.returncode == 0, completed.stderr
    entry = json.loads(next(session.glob("launch-*.json")).read_text())
    assert entry["pid"] is None
    assert entry["pods"] == [f"013_test_s{s}-session-test" for s in range(3)]
    assert len(list(Path(entry["logs"]).glob("seed-*"))) == 3


def pending_fixture(loop: Any, name: str = "015_beta2-099") -> Path:
    directory = loop.HERE / "results" / name
    directory.mkdir(parents=True)
    (directory / "params.json").write_text(json.dumps({"config": {"seed": [0, 1, 2]}}))
    return directory


def test_resume_validates_before_instructing_agent(loop: Any, monkeypatch: pytest.MonkeyPatch):
    pending_fixture(loop)
    calls = []

    def checked(argv: list[str], **kwargs: Any):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, "ok", "")

    monkeypatch.setattr(loop.subprocess, "run", checked)
    prompt = loop.iteration_prompt("base instructions")
    assert prompt.startswith("base instructions")
    assert "RESUME 015_beta2-099" in prompt
    assert "--seeds 0,1,2" in prompt
    assert calls[0][-4:] == ["015_beta2-099", "--seeds", "0,1,2", "--check-only"]
    assert calls[0][0] == str(loop.ROOT / "code/.venv/bin/python")


def test_invalid_resume_never_launches_codex(
    loop: Any, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    pending_fixture(loop)
    monkeypatch.setattr(
        loop.subprocess,
        "run",
        lambda argv, **kw: subprocess.CompletedProcess(argv, 1, "", "pretrain.py sha mismatch"),
    )
    monkeypatch.setattr(loop.subprocess, "Popen", lambda *a, **kw: pytest.fail("Codex launched"))
    assert loop.run_loop(["codex"], "instructions") == 1
    assert (
        "BLOCKED | Cannot resume 015_beta2-099: pretrain.py sha mismatch" in capsys.readouterr().out
    )


def test_multiple_pending_experiments_block(loop: Any):
    pending_fixture(loop)
    pending_fixture(loop, "016_other")
    with pytest.raises(loop.ExperimentBlocked, match="Multiple unfinished"):
        loop.iteration_prompt("instructions")


def test_resume_then_next_fresh_proposal(
    loop: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    pending_fixture(loop)
    # Validation is tested separately; this exercises real fresh processes and the loop transition.
    monkeypatch.setattr(
        loop.subprocess,
        "run",
        lambda argv, **kw: subprocess.CompletedProcess(argv, 0, "ok", ""),
    )
    fake = tmp_path / "fake_codex.py"
    fake.write_text("""import json, os, pathlib, sys
root = pathlib.Path.cwd()
counter = root / "calls.json"
calls = json.loads(counter.read_text()) if counter.exists() else []
session = pathlib.Path(os.environ["ND_AUTORESEARCH_SESSION"])
calls.append({"prompt": sys.argv[-1], "session": str(session)})
counter.write_text(json.dumps(calls))
if len(calls) == 2:
    sys.exit(9)
name = "015_beta2-099"
(session / "launch-1.json").write_text(json.dumps({
    "name": name, "seeds": [], "pid": None, "pods": [], "logs": str(session)}))
(root / "research/results" / name / "results.json").write_text(json.dumps({
    "status": "discard", "metric": 90}))
(session / "completed.json").write_text(json.dumps({"name": name, "commit": "fake-commit"}))
""")
    monkeypatch.setattr(loop, "POLL_SECS", 0.01)
    assert loop.run_loop([sys.executable, str(fake)], "base instructions") == 1
    calls = json.loads((tmp_path / "calls.json").read_text())
    assert len(calls) == 2 and calls[0]["session"] != calls[1]["session"]
    assert "RESUME 015_beta2-099" in calls[0]["prompt"]
    assert calls[1]["prompt"] == "base instructions"
