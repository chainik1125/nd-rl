"""Serrano-only: re-grading one of Dan's stored sample files with tools/grade_job.sh reproduces his stored graded file."""

import subprocess
from pathlib import Path

import pytest

X = Path(__file__).resolve().parents[1]
D = Path("/scratch/cowifornia/nd-rl/olmo_elicit/ext/nd-takehome/artifacts/olmo_ifcurve")


@pytest.mark.skipif(
    not (D / "samples" / "X_s0050.jsonl.gz").exists(), reason="Dan's artifacts not on this machine"
)
def test_regrade_reproduces_stored_grades(tmp_path):
    out = tmp_path / "g.jsonl"
    r = subprocess.run(
        [
            "bash",
            str(X / "tools" / "grade_job.sh"),
            str(D / "samples" / "X_s0050.jsonl.gz"),
            str(out),
        ],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stdout + r.stderr
    assert out.read_bytes() == (D / "graded" / "X_s0050.jsonl").read_bytes()
    assert not list(
        (D.parent.parent.parent / "olmo_ifomega").rglob("__pycache__")
    )  # his tree stayed clean
