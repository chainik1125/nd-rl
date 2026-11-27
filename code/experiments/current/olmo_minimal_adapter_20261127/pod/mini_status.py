#!/usr/bin/env python3
# ruff: noqa: E501, SIM115
"""Status of the minimal-adapter chains (stdlib only), modelled on the 2026-11-25 oe_status.py.

POD side:   python3 mini_status.py [--runs $MINI_ROOT/runs] [--models /workspace/models] [<ckpt> ...]
  one line per (checkpoint, stage that exists in state/ or was asked by the chain):
    <ckpt> <stage> state=<S> progress=<P> hb=<age>s gpu=<MiB used> disk=<GB free>
  one line per training cell found under runs/<ckpt>/train/cells/:
    <ckpt> cell:<id> state=<DONE|RUNNING|STALE> age=<s since the newest file in its folder>s last=<last log line>
  one chain line:
    <ckpt> chain alive=<0|1> pid=<pid> current=<stage> done=<0|1> failed=<n> weights=<present|deleted|absent>
  Stages: download, train, gate:<b>, sample:<b>, plain:<name>. Stage names are state-file names (gate:b1 is gate_b1). S: PENDING, RUNNING, DONE, FAILED, STALE (RUNNING marker, no
  live chain), SKIP (not asked). Progress: download = % of bytes; train = cells done/started; gate = per-cell verdict
  counts; sample = rows/expected draws (and graded rows).
Helpers used by mini_chain.sh:
  --verify-batch <dir>                 md5-check <dir>/cells_manifest.json against the staged files; exit 1 on a difference
  --gate-summary <gate.json>           one line "pass=a uninformative=b fail=c fail_cells=..." (exit 0)
  --expected-draws <gate.json> <pids>  draws a sample stage must produce: non-FAIL cells x sum of the pids file's counts
SERRANO side: `parse(text)` turns the status lines into dicts; `verdict` / `finished` as in the old script.
"""

from __future__ import annotations

import argparse
import contextlib
import gzip
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import zlib
from pathlib import Path

LINE = re.compile(r"^(?P<ckpt>\S+) (?P<stage>\S+) (?P<kv>.*)$")
FIXED = ["download", "train"]
MARKERS = (".RUNNING", ".DONE", ".FAILED", ".expected")
NON_STAGE = {"chain", "current", "heartbeat", "weights_deleted"}


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


def newest_age(d: Path, now: float) -> int | None:
    m = None
    for root, _, files in os.walk(d):
        for f in files:
            with contextlib.suppress(OSError):
                t = os.path.getmtime(os.path.join(root, f))
                m = t if m is None else max(m, t)
    return None if m is None else max(0, int(now - m))


def last_line(d: Path) -> str:
    """Last non-empty line of the newest *.log / *.txt / progress* file in a cell folder ('-' if none)."""
    cand = (
        [
            p
            for p in d.iterdir()
            if p.is_file() and (p.suffix in (".log", ".txt") or p.name.startswith("progress"))
        ]
        if d.exists()
        else []
    )
    if not cand:
        return "-"
    p = max(cand, key=lambda x: x.stat().st_mtime)
    try:
        ln = [x for x in p.read_text(errors="replace").splitlines() if x.strip()][-1]
    except (OSError, IndexError):
        return "-"
    return re.sub(r"\s+", "_", ln.strip())[:70]


def stage_state(st: Path, stage: str, asked: set[str] | None, alive: bool) -> str:
    n = stage.replace(":", "_")
    if (st / f"{n}.DONE").exists():
        return "DONE"
    if (st / f"{n}.FAILED").exists():
        return "FAILED"
    if (st / f"{n}.RUNNING").exists():
        return "RUNNING" if alive else "STALE"
    if asked is not None and stage != "download" and stage not in asked:
        return "SKIP"
    return "PENDING"


def gate_summary(gj: Path) -> dict:
    d = json.loads(gj.read_text())
    v = d["verdicts"]
    return {k: sorted(c for c, x in v.items() if x == k) for k in ("PASS", "UNINFORMATIVE", "FAIL")}


def gate_summary_line(gj: Path) -> str:
    g = gate_summary(gj)
    return f"pass={len(g['PASS'])} uninformative={len(g['UNINFORMATIVE'])} fail={len(g['FAIL'])} fail_cells={','.join(g['FAIL']) or '-'}"


def expected_draws(gj: Path, pids_file: Path) -> int:
    g = gate_summary(gj)
    n_cells = len(g["PASS"]) + len(g["UNINFORMATIVE"])
    per = 0
    for tok in pids_file.read_text().split():
        _, _, n = tok.partition(":")
        per += int(n) if n else 0
    return n_cells * per


def verify_batch(batch: Path) -> tuple[bool, list[str]]:
    man = json.loads((batch / "cells_manifest.json").read_text())
    bad = []
    for c, m in man["cells"].items():
        for rel, dg in m["files"].items():
            p = batch / c / rel
            try:
                h = hashlib.md5()  # noqa: S324 (integrity check)
                with open(p, "rb") as f:
                    for b in iter(lambda: f.read(1 << 24), b""):
                        h.update(b)
                ok = h.hexdigest() == dg["md5"]
            except OSError:
                ok = False
            if not ok:
                bad.append(f"{c}/{rel}")
    return not bad, bad


def progress(run: Path, stage: str, models: Path, expected_bytes: int | None) -> str:
    st = run / "state"
    n = stage.replace(":", "_")
    kind, _, b = stage.partition(":")
    if kind == "download":
        md = models / run.name
        if (md / ".complete").exists():
            return "complete"
        if expected_bytes and md.exists():
            return f"{100 * dir_bytes(md) / expected_bytes:.0f}%"
        return "-"
    if kind == "train":
        cells = run / "train" / "cells"
        dirs = [p for p in cells.iterdir() if p.is_dir()] if cells.exists() else []
        done = sum((p / "DONE").exists() for p in dirs)
        return f"cells_done={done}/{len(dirs)}"
    if kind == "gate":
        gj = run / "gate" / f"{b}.json"
        if gj.exists():
            with contextlib.suppress(OSError, ValueError, KeyError):
                return gate_summary_line(gj).replace(" ", ",")
        return "-"
    if kind == "plain":
        s = count_gz_rows(run / f"plain_{b}" / "samples.jsonl.gz")
        g = count_rows(run / f"plain_{b}" / "graded.jsonl")
        exp = (st / f"{n}.expected").read_text().strip() if (st / f"{n}.expected").exists() else "?"
        return f"{s}/{exp}" + (f",graded={g}" if g else "")
    if kind == "sample":
        exp = (st / f"{n}.expected").read_text().strip() if (st / f"{n}.expected").exists() else "?"
        s = sum(count_gz_rows(p) for p in sorted((run / "sample" / b).glob("*/samples.jsonl.gz")))
        g = sum(count_rows(p) for p in sorted((run / "sample" / b).glob("*/graded.jsonl")))
        return f"{s}/{exp}" + (f",graded={g}" if g else "")
    return "-"


def stage_names(st: Path, asked: set[str] | None) -> list[str]:
    seen: list[str] = list(FIXED)
    names = set()
    for p in st.iterdir() if st.exists() else []:
        for m in MARKERS:
            if p.name.endswith(m):
                base = p.name[: -len(m)]
                if base not in NON_STAGE and not base.startswith("chain"):
                    names.add(base)
    ordered = [s for s in sorted(asked or []) if s not in seen]
    for s in ordered:
        seen.append(s)
    for base in sorted(names):
        # a state-file name gate_b1 -> stage gate:b1
        stage = re.sub(r"^(gate|sample|plain)_", r"\1:", base)
        if stage not in seen:
            seen.append(stage)
    return seen


def status_lines(
    runs: Path,
    ckpts: list[str] | None = None,
    models: Path = Path("/workspace/models"),
    now: float | None = None,
    dan: Path = Path("/workspace/nd-takehome"),
) -> list[str]:
    now = time.time() if now is None else now
    out = []
    try:
        disk = f"{shutil.disk_usage(runs).free / 1e9:.0f}"
    except OSError:
        disk = "na"
    gpu = gpu_used()
    exp_bytes = {}
    csvp = dan / "olmo_ifcurve" / "checkpoints.csv"
    if csvp.exists():
        for ln in csvp.read_text().splitlines()[1:]:
            f = ln.split(",")
            exp_bytes[f[0]] = int(f[6])
    names = ckpts or (
        sorted(p.name for p in runs.iterdir() if (p / "state").is_dir()) if runs.exists() else []
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
        for s in stage_names(st, asked):
            state = stage_state(st, s, asked, alive)
            failed += state == "FAILED"
            out.append(
                f"{ck} {s} state={state} progress={progress(run, s, models, exp_bytes.get(ck))} hb={hb}s gpu={gpu} disk={disk}"
            )
        cells = run / "train" / "cells"
        for cd in sorted(p for p in cells.iterdir() if p.is_dir()) if cells.exists() else []:
            cs = "DONE" if (cd / "DONE").exists() else ("RUNNING" if alive else "STALE")
            age = newest_age(cd, now)
            out.append(
                f"{ck} cell:{cd.name} state={cs} age={'na' if age is None else age}s last={last_line(cd)}"
            )
        cur = (st / "current").read_text().strip() if (st / "current").exists() else "-"
        w = (
            "deleted"
            if (st / "weights_deleted").exists()
            else ("present" if (models / ck).exists() else "absent")
        )
        out.append(
            f"{ck} chain alive={int(alive)} pid={pid} current={cur} done={int((st / 'chain.DONE').exists())} failed={failed} weights={w}"
        )
    return out


def parse(text: str) -> dict[str, dict]:
    """{ckpt: {"stages": {stage: kv}, "cells": {cell: kv}, "chain": {...}}} from status_lines output."""
    res: dict[str, dict] = {}
    for ln in text.splitlines():
        m = LINE.match(ln.strip())
        if not m:
            continue
        kv = dict(x.split("=", 1) for x in m["kv"].split() if "=" in x)
        if not kv:
            continue
        d = res.setdefault(m["ckpt"], {"stages": {}, "cells": {}, "chain": {}})
        if m["stage"] == "chain":
            d["chain"] = {k: (int(v) if v.lstrip("-").isdigit() else v) for k, v in kv.items()}
        elif m["stage"].startswith("cell:"):
            d["cells"][m["stage"][5:]] = kv
        else:
            hb = kv.get("hb", "nas").rstrip("s")
            kv["hb"] = int(hb) if hb.isdigit() else None
            d["stages"][m["stage"]] = kv
    return res


def finished(entry: dict, stages: list[str]) -> bool:
    ch = entry.get("chain", {})
    if ch.get("alive") == 1:
        return False
    if ch.get("done") == 1:
        return True
    return all(
        entry["stages"].get(s, {}).get("state") in ("DONE", "FAILED") for s in ["download", *stages]
    )


def verdict(text: str, ckpt: str, stages: list[str]) -> str:
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
    ap.add_argument("--runs", default=os.environ.get("MINI_ROOT", "/workspace/mini") + "/runs")
    ap.add_argument("--models", default="/workspace/models")
    ap.add_argument(
        "--verdict",
        metavar="CKPT",
        help="serrano side: read status lines on stdin, print a verdict",
    )
    ap.add_argument("--stages", default="train")
    ap.add_argument("--verify-batch", metavar="DIR")
    ap.add_argument("--gate-summary", metavar="JSON")
    ap.add_argument("--expected-draws", nargs=2, metavar=("GATE_JSON", "PIDS_FILE"))
    a = ap.parse_args()
    if a.verify_batch:
        ok, bad = verify_batch(Path(a.verify_batch))
        print("BATCH_MD5 ok" if ok else f"BATCH_MD5 MISMATCH {bad}")
        return 0 if ok else 1
    if a.gate_summary:
        print(gate_summary_line(Path(a.gate_summary)))
        return 0
    if a.expected_draws:
        print(expected_draws(Path(a.expected_draws[0]), Path(a.expected_draws[1])))
        return 0
    if a.verdict:
        print(verdict(sys.stdin.read(), a.verdict, a.stages.split(",")))
        return 0
    print("\n".join(status_lines(Path(a.runs), a.ckpts or None, Path(a.models))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
