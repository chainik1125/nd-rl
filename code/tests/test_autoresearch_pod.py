"""pod.py run_detached: a dropped ssh connection must not kill or garble the remote job."""

import importlib.util
from pathlib import Path
from typing import Any

import pytest

SOURCE = Path(__file__).parents[1] / "experiments/current/autoresearch/pod.py"

# Runs the "remote" command locally. The first streaming connection (`tail -c`) is killed after
# 1 s and exits 255, like an ssh client losing the network mid-run.
FAKE_SSH = """#!/bin/bash
while [ $# -gt 0 ] && [[ "$1" != root@* ]]; do shift; done; shift
cmd="$*"
if [[ "$cmd" == *"tail -c"* ]] && [ -f "$DROP" ]; then
  rm "$DROP"
  set -m
  bash -c "$cmd" & pid=$!
  sleep 1; kill -- -$pid; exit 255
fi
exec bash -c "$cmd"
"""


@pytest.fixture
def pod(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name, body in (("ssh", FAKE_SSH), ("setsid", '#!/bin/bash\nexec "$@"\n')):
        (bin_dir / name).write_text(body)
        (bin_dir / name).chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}:{Path('/usr/bin')}:{Path('/bin')}")
    monkeypatch.setenv("DROP", str(tmp_path / "drop"))
    spec = importlib.util.spec_from_file_location("pod", SOURCE)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("drop", [False, True])
def test_run_detached_streams_each_line_once_across_drops(
    pod: Any, tmp_path: Path, capfd: pytest.CaptureFixture[str], drop: bool
):
    if drop:
        (tmp_path / "drop").touch()
    job = "for i in 1 2 3 4 5 6; do echo line$i; sleep 0.4; done; exit 3"
    rc = pod.run_detached({"port": 1, "ip": "pod"}, job, str(tmp_path / "art/x"), retry_secs=0.1)
    out = capfd.readouterr().out
    assert rc == 3
    assert [line for line in out.splitlines() if line.startswith("line")] == [
        f"line{i}" for i in range(1, 7)
    ]
    assert ("ssh dropped" in out) == drop
