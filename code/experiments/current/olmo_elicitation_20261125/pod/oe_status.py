#!/usr/bin/env python3
# ruff: noqa: E501, SIM115
"""Status of the olmo elicitation chains (stdlib only).

POD side:   python3 oe_status.py [--runs /workspace/oe/runs] [<ckpt> ...]
            prints one line per (checkpoint, stage), then one `chain` line per checkpoint:
              <ckpt> <stage> state=<S> progress=<P> hb=<age>s gpu=<MiB used> disk=<GB free>
              <ckpt> chain alive=<0|1> pid=<pid> current=<stage> done=<0|1> failed=<n> weights=<present|deleted|absent>
            S: PENDING (not started), RUNNING, DONE, FAILED, STALE (RUNNING marker but no live chain), SKIP (not asked).
SERRANO side: `parse(text)` turns those lines into dicts (olmo_chain.sh decides done / failed / stalled from them).
"""

from __future__ import annotations

import argparse
import contextlib
import gzip
import os
import re
import shutil
import subprocess
import sys
import time
import zlib
from pathlib import Path

STAGES = ["download", "edl", "T", "H", "rep"]
LINE = re.compile(r"^(?P<ckpt>\S+) (?P<stage>\S+) (?P<kv>.*)$")


def count_gz_rows(p: Path) -> int:
    """Complete lines of a gzip file that may still be written to (a truncated tail is tolerated)."""
    n = 0
    try:
        with gzip.open(p, "rb") as f:
            for _ in f:
                n += 1
    except (EOFError, OSError, zlib.error):
        pass
    return n


def count_rows(p: Path) -> int:
    try:
        with open(p, "rb") as f:
            return sum(1 for _ in f)
    except OSError:
        return 0


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def gpu_used() -> str:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=20,
        ).stdout.split()
        return "+".join(out) or "na"
    except (OSError, subprocess.TimeoutExpired):
        return "na"


def dir_bytes(d: Path) -> int:
    t = 0
    for root, _, files in os.walk(d):
        for f in files:
            with contextlib.suppress(OSError):
                t += os.path.getsize(os.path.join(root, f))
    return t


def stage_state(st: Path, stage: str, asked: set[str] | None, alive: bool) -> str:
    if (st / f"{stage}.DONE").exists():
        return "DONE"
    if (st / f"{stage}.FAILED").exists():
        return "FAILED"
    if (st / f"{stage}.RUNNING").exists():
        return "RUNNING" if alive else "STALE"
    if asked is not None and stage != "download" and stage not in asked:
        return "SKIP"
    return "PENDING"


def progress(run: Path, stage: str, models: Path, expected_bytes: int | None) -> str:
    st = run / "state"
    if stage == "download":
        md = models / run.name
        if (md / ".complete").exists():
            return "complete"
        if expected_bytes and md.exists():
            return f"{100 * dir_bytes(md) / expected_bytes:.0f}%"
        return "-"
    if stage == "edl":
        lg = run / "logs" / "edl.log"
        try:
            last = [ln for ln in lg.read_text(errors="replace").splitlines() if ln.strip()][-1]
        except (OSError, IndexError):
            return "-"
        return "log:" + re.sub(r"\s+", "_", last.strip())[:70]
    exp = (
        (st / f"{stage}.expected").read_text().strip()
        if (st / f"{stage}.expected").exists()
        else "?"
    )
    s = count_gz_rows(run / stage / "samples.jsonl.gz")
    g = count_rows(run / stage / "graded.jsonl")
    return f"{s}/{exp}" + (f",graded={g}" if g else "")


def status_lines(
    runs: Path,
    ckpts: list[str] | None = None,
    models: Path = Path("/workspace/models"),
    now: float | None = None,
) -> list[str]:
    now = time.time() if now is None else now
    out = []
    try:
        disk = f"{shutil.disk_usage(runs).free / 1e9:.0f}"
    except OSError:
        disk = "na"
    gpu = gpu_used()
    exp_bytes = {}
    csvp = Path("/workspace/nd-takehome/olmo_ifcurve/checkpoints.csv")
    if csvp.exists():
        for ln in csvp.read_text().splitlines()[1:]:
            f = ln.split(",")
            exp_bytes[f[0]] = int(f[6])
    if ckpts:
        names = ckpts
    else:
        names = (
            sorted(p.name for p in runs.iterdir() if (p / "state").is_dir())
            if runs.exists()
            else []
        )
    for ck in names:
        run = runs / ck
        st = run / "state"
        pid, asked = 0, None
        if (st / "chain.pid").exists():
            f = (st / "chain.pid").read_text().split()
            pid = int(f[0])
            asked = set(f[-1].split(",")) if len(f) >= 4 else None
        alive = bool(pid) and pid_alive(pid) and not (st / "chain.DONE").exists()
        hb = "na"
        if (st / "heartbeat").exists():
            with contextlib.suppress(ValueError, IndexError):
                hb = str(int(now - float((st / "heartbeat").read_text().split()[0])))
        failed = 0
        for s in STAGES:
            state = stage_state(st, s, asked, alive)
            failed += state == "FAILED"
            out.append(
                f"{ck} {s} state={state} progress={progress(run, s, models, exp_bytes.get(ck))} hb={hb}s "
                f"gpu={gpu} disk={disk}"
            )
        cur = (st / "current").read_text().strip() if (st / "current").exists() else "-"
        w = (
            "deleted"
            if (st / "weights_deleted").exists()
            else ("present" if (models / ck).exists() else "absent")
        )
        out.append(
            f"{ck} chain alive={int(alive)} pid={pid} current={cur} done={int((st / 'chain.DONE').exists())} "
            f"failed={failed} weights={w}"
        )
    return out


def parse(text: str) -> dict[str, dict]:
    """{ckpt: {"stages": {stage: {state, progress, hb, gpu, disk}}, "chain": {alive, pid, current, done, failed,
    weights}}} from status_lines output (other lines ignored)."""
    res: dict[str, dict] = {}
    for ln in text.splitlines():
        m = LINE.match(ln.strip())
        if not m:
            continue
        kv = dict(x.split("=", 1) for x in m["kv"].split() if "=" in x)
        if not kv:
            continue
        d = res.setdefault(m["ckpt"], {"stages": {}, "chain": {}})
        if m["stage"] == "chain":
            d["chain"] = {k: (int(v) if v.lstrip("-").isdigit() else v) for k, v in kv.items()}
        elif m["stage"] in STAGES:
            hb = kv.get("hb", "nas").rstrip("s")
            kv["hb"] = int(hb) if hb.isdigit() else None
            d["stages"][m["stage"]] = kv
    return res


def finished(entry: dict, stages: list[str]) -> bool:
    """A chain is finished when it is not alive and wrote chain.DONE (or every asked stage is DONE / FAILED)."""
    ch = entry.get("chain", {})
    if ch.get("alive") == 1:
        return False
    if ch.get("done") == 1:
        return True
    return all(
        entry["stages"].get(s, {}).get("state") in ("DONE", "FAILED") for s in ["download", *stages]
    )


def verdict(text: str, ckpt: str, stages: list[str]) -> str:
    """One word + details for the serrano driver: FINISHED / RUNNING / NOSTATUS, then hb=<age or na> and a
    stage:state summary."""
    d = parse(text).get(ckpt)
    if not d or not d["stages"]:
        return "NOSTATUS hb=na"
    hbs = [v["hb"] for v in d["stages"].values() if v.get("hb") is not None]
    hb = str(min(hbs)) if hbs else "na"
    summ = ",".join(
        f"{s}:{d['stages'].get(s, {}).get('state', '?')}" for s in ["download", *stages]
    )
    word = "FINISHED" if finished(d, stages) else "RUNNING"
    return f"{word} hb={hb} alive={d['chain'].get('alive', '?')} {summ}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", nargs="*")
    ap.add_argument("--runs", default="/workspace/oe/runs")
    ap.add_argument("--models", default="/workspace/models")
    ap.add_argument(
        "--verdict",
        metavar="CKPT",
        help="serrano side: read status lines on stdin, print a verdict",
    )
    ap.add_argument("--stages", default="edl,T,H")
    a = ap.parse_args()
    if a.verdict:
        print(verdict(sys.stdin.read(), a.verdict, a.stages.split(",")))
        return 0
    print("\n".join(status_lines(Path(a.runs), a.ckpts or None, Path(a.models))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
