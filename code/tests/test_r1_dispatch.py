"""CPU-only ownership cleanup and immutable-input regressions."""

import json
import os
import subprocess
import sys
import time
from unittest.mock import patch

from r1_cot import dispatch, submit


def test_exited_leader_descendants_stopped_without_touching_other_group(tmp_path):
    ready = tmp_path / "descendant.pid"
    script = (
        "import os,signal,time,pathlib;"
        "signal.signal(signal.SIGTERM,signal.SIG_IGN);"
        f"pathlib.Path({str(ready)!r}).write_text(str(os.getpid()));"
        "time.sleep(30)"
    )
    leader_script = f"import subprocess,sys;subprocess.Popen([sys.executable,'-c',{script!r}])"
    outsider = subprocess.Popen(
        [sys.executable, "-c", "import time;time.sleep(30)"], start_new_session=True
    )
    leader = subprocess.Popen([sys.executable, "-c", leader_script], start_new_session=True)
    try:
        leader.wait(timeout=5)
        deadline = time.monotonic() + 5
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert ready.exists()
        descendant = int(ready.read_text())
        assert os.getpgid(descendant) == leader.pid
        dispatch.stop(leader, grace_seconds=0.05)
        status = None
        while time.monotonic() < deadline:
            result = subprocess.run(
                ["ps", "-p", str(descendant), "-o", "stat="],
                capture_output=True,
                text=True,
                timeout=1,
            )
            status = result.stdout.strip()
            if not status or status.startswith("Z"):
                break
            time.sleep(0.02)
        assert not status or status.startswith("Z")
        assert outsider.poll() is None
    finally:
        dispatch.stop(leader, grace_seconds=0)
        outsider.kill()
        outsider.wait(timeout=5)


def test_cleanup_never_signals_supervisor_group():
    class Child:
        pid = os.getpgrp()

    with patch.object(dispatch.os, "killpg") as kill:
        try:
            dispatch.stop(Child(), grace_seconds=0)
        except RuntimeError:
            pass
        else:
            raise AssertionError("Supervisor group must be rejected")
        kill.assert_not_called()


def test_nvidia_queries_have_bounded_shared_timeout():
    calls = []

    def query(command, **kwargs):
        calls.append(kwargs["timeout"])
        if "--query-compute-apps=gpu_uuid,pid" in command:
            return ""
        return "3, GPU-fixture, 0, 0\n"

    with patch.object(dispatch.subprocess, "check_output", side_effect=query):
        assert dispatch.state(timeout=0.2)[3]["pids"] == []
    assert len(calls) == 2
    assert 0 < calls[1] <= calls[0] <= 0.2


def test_pipeline_configs_and_existing_referenced_inputs_are_frozen(tmp_path):
    data = tmp_path / "data.jsonl"
    checkpoint = tmp_path / "parent.pt"
    guard = tmp_path / "guard.json"
    for path in (data, checkpoint, guard):
        path.write_text("fixture")
    config = tmp_path / "pipeline.json"
    config.write_text(
        json.dumps(
            {
                "stages": [
                    {
                        "kind": "command",
                        "command": [
                            "python",
                            "run.py",
                            "--data=data.jsonl",
                            "--checkpoint",
                            "parent.pt",
                            "--guard-json",
                            "guard.json",
                        ],
                    },
                    {
                        "kind": "release_development",
                        "checkpoints": ["future.pt", "parent.pt"],
                        "data": ["data.jsonl"],
                        "completed_rl_checkpoint": "future.pt",
                    },
                ]
            }
        )
    )
    result = submit.collect_input_hashes(
        ["python", "pipeline.py", "--config", str(config)], tmp_path
    )
    assert set(result) == {str(p) for p in (config, data, checkpoint, guard)}
    before = result[str(data)]
    data.write_text("changed")
    result = submit.collect_input_hashes(
        ["python", "pipeline.py", "--config", str(config)], tmp_path
    )
    assert result[str(data)] != before
