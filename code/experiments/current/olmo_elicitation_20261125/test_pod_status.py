# ruff: noqa: E501, SIM115
"""CPU tests for pod/oe_status.py: status lines from a fake runs tree, the parser, the driver's verdict."""

from __future__ import annotations

import gzip
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "pod"))
import oe_status as S  # noqa: E402


def fake_run(
    root: Path, ck: str, stages: str, done: list[str], running: str | None, pid: int, now: float
) -> None:
    st = root / "runs" / ck / "state"
    st.mkdir(parents=True)
    (root / "runs" / ck / "logs").mkdir()
    (st / "chain.pid").write_text(f"{pid} {pid} 2026-11-25T10:00:00Z {stages}\n")
    for s in done:
        (st / f"{s}.DONE").write_text("x rc=0\n")
    if running:
        (st / f"{running}.RUNNING").write_text("x\n")
        (st / "current").write_text(running + "\n")
    (st / "heartbeat").write_text(f"{now - 30:.0f} {running}\n")
    (st / "H.expected").write_text("384\n")
    (root / "runs" / ck / "H").mkdir()
    with gzip.open(root / "runs" / ck / "H" / "samples.jsonl.gz", "wt") as f:
        for i in range(10):
            f.write(json.dumps({"pid": "x", "i": i}) + "\n")
    (root / "runs" / ck / "logs" / "edl.log").write_text("step 3 of 64 loss 1.23\n")


def test_status_roundtrip(tmp_path):
    now = 1_800_000_000.0
    fake_run(
        tmp_path, "s0500", "edl,T,H", ["download", "edl", "T"], "H", os.getpid(), now
    )  # alive chain
    fake_run(
        tmp_path, "s1100", "edl,T,H", ["download"], "edl", 2**22 + 12345, now
    )  # dead pid -> STALE
    lines = S.status_lines(tmp_path / "runs", None, tmp_path / "models", now=now)
    d = S.parse("garbage line\n" + "\n".join(lines))
    a = d["s0500"]
    assert a["stages"]["edl"]["state"] == "DONE" and a["stages"]["H"]["state"] == "RUNNING"
    assert a["stages"]["H"]["progress"] == "10/384"
    assert a["stages"]["rep"]["state"] == "SKIP" and a["stages"]["H"]["hb"] == 30
    assert a["chain"]["alive"] == 1 and a["chain"]["current"] == "H"
    assert not S.finished(a, ["edl", "T", "H"])
    b = d["s1100"]
    assert b["stages"]["edl"]["state"] == "STALE" and b["chain"]["alive"] == 0
    assert b["stages"]["edl"]["progress"].startswith("log:step_3_of_64")
    v = S.verdict("\n".join(lines), "s0500", ["edl", "T", "H"])
    assert v.startswith("RUNNING hb=30 alive=1") and "H:RUNNING" in v


def test_finished_and_failed(tmp_path):
    now = 1_800_000_000.0
    fake_run(tmp_path, "dpo", "edl,rep", ["download", "rep"], None, 2**22 + 777, now)
    st = tmp_path / "runs" / "dpo" / "state"
    (st / "edl.FAILED").write_text("x rc=1\n")
    (st / "chain.DONE").write_text("x failed_stages=1\n")
    lines = S.status_lines(tmp_path / "runs", ["dpo"], tmp_path / "models", now=now)
    d = S.parse("\n".join(lines))["dpo"]
    assert d["chain"]["failed"] == 1 and d["chain"]["done"] == 1
    assert S.finished(d, ["edl", "rep"])
    assert S.verdict("\n".join(lines), "dpo", ["edl", "rep"]).startswith("FINISHED")
    assert S.verdict("", "dpo", ["edl"]) == "NOSTATUS hb=na"
