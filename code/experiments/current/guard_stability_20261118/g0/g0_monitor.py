# ruff: noqa
"""Collapse monitor + auto-fetch for the guard-off author-tag reruns (2026-11-18). Runs ON SERRANO under nohup.

    python3 g0_monitor.py [--hours 10]

Every ~2 min, per line, reads the per-iteration canary values from the line's GRPO logs on its pod
(`canary it N: nll ... P(:) x% ... wrong y%`). Pre-specified rule (a supervisor action, never a training change):
COLLAPSED at the first iteration where canary P(':') > 3e-3 (0.3 %) on 2 consecutive canary reads, or top wrong-token
P > 0.05 (5 %) on 2 consecutive reads, or a non-finite stop. On a trigger for a tag line: STOP file in the segment
(tag_line ends GRPO and finishes its reads), then, if GRPO is still alive after 45 s, kill its PID (from status.json).
The two-authors lines (rp-ta5/rp-ta6) are watched with the same rule but only a COLLAPSE line is logged.
When a pod's driver has written DONE_<line>, fetches the line (g0_pod.sh fetch, md5-checked).
Log: /scratch/cowifornia/nd-rl/author_tags/runs_g0/monitor.log; state: runs_g0/monitor_state.json.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = Path("/scratch/cowifornia/nd-rl/author_tags/runs_g0")
LOG = OUT / "monitor.log"
STATE = OUT / "monitor_state.json"
TAG_LINES = {"tA_p10_s0": "rp-stab4", "tB_p10_s0": "rp-stab6", "tA_p10_s1": "rp-stab5",
             "tB_p10_s1": "rp-tag1", "tA_p10_s2": "rp-stab7", "tB_p10_s2": "rp-stab2"}  # fmt: skip
TA_LINES = {"rl_authors_half_s0": "rp-ta5", "rl_control_half_s0": "rp-ta6"}
P_COLON, P_WRONG = 0.3, 5.0  # in percent, as the log prints them
CAN = re.compile(r"canary it (\d+): nll (\S+) P\(:\) (\S+)% .*?wrong (\S+)%")
SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15"]


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def say(msg: str) -> None:
    with open(LOG, "a") as f:
        f.write(f"{now()} {msg}\n")


def ssh(host: str, cmd: str, timeout: int = 90) -> str | None:
    try:
        p = subprocess.run([*SSH, host, cmd], capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None
    return p.stdout if p.returncode == 0 else None


def fnum(s: str) -> float:
    try:
        return float(s)
    except ValueError:
        return math.nan


def parse(text: str) -> dict[int, tuple[float, float, float]]:
    """iteration -> (nll, P(:) %, wrong %); a later segment's value of an iteration replaces an earlier one."""
    out: dict[int, tuple[float, float, float]] = {}
    for line in text.splitlines():
        m = CAN.search(line)
        if m:
            out[int(m.group(1))] = (fnum(m.group(2)), fnum(m.group(3)), fnum(m.group(4)))
    return out


def rule(can: dict[int, tuple[float, float, float]]) -> tuple[int, int, str] | None:
    """(trigger iteration, first iteration of the two exceeding reads, why)."""
    its = sorted(i for i in can if i > 0)
    prev = None
    for i in its:
        nll, pc, pw = can[i]
        if not all(math.isfinite(x) for x in (nll, pc, pw)):
            return i, i, f"non-finite canary at it {i}: nll {nll} P(:) {pc}% wrong {pw}%"
        if prev is not None:
            _, pc0, pw0 = can[prev]
            if pc0 > P_COLON and pc > P_COLON:
                return i, prev, f"P(:) {pc0}% (it {prev}), {pc}% (it {i}) > {P_COLON}%"
            if pw0 > P_WRONG and pw > P_WRONG:
                return i, prev, f"wrong {pw0}% (it {prev}), {pw}% (it {i}) > {P_WRONG}%"
        prev = i
    return None


def tag_probe(line: str) -> str:
    return (f"cat /workspace/atg0/logs/{line}_seg*.log 2>/dev/null | grep -E 'canary it|ALARM|non-finite|nonfinite'; "
            f"echo '@@SEGS'; ls -d /workspace/atg0/runs/{line}/seg[0-9]* 2>/dev/null; echo '@@END'; "
            f"cat /workspace/atg0/runs/{line}/seg*/end.json 2>/dev/null | grep -o '\"how\": \"[a-z]*\"'; "
            f"echo '@@NB'; ps -eo args | grep '[t]ag_grpo.py' | grep -vc atg0; "
            f"echo '@@DONE'; test -e /workspace/atg0/DONE_{line} && echo yes; true")  # fmt: skip


def kill_line(host: str, line: str, why: str) -> None:
    out = ssh(host, f"ls -d /workspace/atg0/runs/{line}/seg[0-9]* | sort -V | tail -1") or ""
    seg = out.strip()
    if not seg:
        say(f"{line}: KILL wanted but no segment folder found")
        return
    st = ssh(host, f"touch {seg}/STOP; cat {seg}/status.json") or "{}"
    try:
        pid = int(json.loads(st)["pid"])
    except Exception:
        pid = None
    say(f"{line}: STOP written in {seg} ({why}); GRPO pid {pid}")
    time.sleep(45)
    if pid:
        r = ssh(
            host,
            f"if kill -0 {pid} 2>/dev/null; then kill -TERM {pid} && echo killed {pid}; else echo gone {pid}; fi",
        )
        say(f"{line}: {(r or 'ssh failed').strip()}")


def pre_reads(host: str, line: str, c: dict) -> str:
    """For a collapsed line: the cross reads must be of the last snapshot before the collapse (PREREG amendment),
    i.e. the newest snapshot with iteration < the first of the two exceeding canary reads."""
    seed = line[-1]
    out = (
        ssh(
            host,
            f"ls /workspace/atg0/runs/{line}/seg[0-9]*/ckpts/{line}_it*_snap.pt 2>/dev/null; echo @@; "
            f"cat /workspace/atg0/reads/cross_g0_p10_s{seed}/final_{line}.json 2>/dev/null; true",
        )
        or "@@"
    )
    names, _, fin = out.partition("@@")
    snaps = sorted((int(re.search(r"_it(\d+)_snap", p).group(1)), p) for p in names.split())
    first = int(c.get("first", c["it"]))
    pre = [x for x in snaps if x[0] < first]
    if not pre:
        return "none: no snapshot before the collapse"
    it, snap = pre[-1]
    try:
        fit = int(json.loads(fin)["it"])
    except Exception:
        fit = None
    if fit == it:
        return f"the driver's final snapshot it {it} is already pre-collapse"
    lw, rf = ("6", "0.15") if host in ("rp-stab1", "rp-stab2") else ("8", "0.3")
    ssh(
        host,
        f"G0_LEAN_WORKERS={lw} G0_READ_FRAC={rf} setsid nohup bash /workspace/g0exp/g0_reads.sh {line} {seed} {snap} {it} "
        f"> /workspace/atg0/logs/{line}.prereads.log 2>&1 < /dev/null & sleep 1; echo ok",
    )
    return "launched"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=10.0)
    ap.add_argument("--every", type=float, default=120.0)
    a = ap.parse_args()
    t_end = time.time() + a.hours * 3600
    st: dict = json.loads(STATE.read_text()) if STATE.exists() else {}
    st.setdefault("tag", {}); st.setdefault("ta", {})  # fmt: skip
    say(f"monitor start (pid {subprocess.os.getpid()}), {a.hours} h")
    while time.time() < t_end:
        for line, host in TAG_LINES.items():
            s = st["tag"].setdefault(
                line, {"host": host, "collapsed": None, "fetched": False, "progress": []}
            )
            if s["fetched"]:
                continue
            txt = ssh(host, tag_probe(line))
            if txt is None:
                say(f"{line}: {host} did not answer")
                continue
            body, _, rest = txt.partition("@@SEGS")
            segs, _, rest = rest.partition("@@END")
            ends, _, rest = rest.partition("@@NB")
            nb, _, done = rest.partition("@@DONE")
            try:
                s.setdefault("neighbors", []).append(
                    [time.time(), int(nb.split()[0]), s.get("last_it")]
                )
            except (ValueError, IndexError):
                pass
            can = parse(body)
            s["alarms"] = sum(1 for x in body.splitlines() if x.startswith("ALARM it"))
            s["segs"] = len(segs.split())
            s["ends"] = re.findall(r'"how": "([a-z]+)"', ends)
            if can:
                last = max(can)
                s["last_it"] = last
                s["max_colon_pct"] = (
                    max(v[1] for i, v in can.items() if i > 0) if last > 0 else None
                )
                s["max_wrong_pct"] = (
                    max(v[2] for i, v in can.items() if i > 0) if last > 0 else None
                )
                if not s["progress"] or s["progress"][-1][1] != last:
                    s["progress"].append([time.time(), last])
            if s["collapsed"] is None:
                hit = rule(can)
                if hit is None and "nonfinite" in s["ends"]:
                    hit = (
                        s.get("last_it", -1),
                        s.get("last_it", -1),
                        "non-finite stop (end.json how=nonfinite)",
                    )
                if hit is not None:
                    it, first, why = hit
                    s["collapsed"] = {"it": it, "first": first, "why": why, "utc": now()}
                    say(
                        f"COLLAPSE {line} on {host} at it {it} (first exceeding read it {first}): {why}"
                    )
                    STATE.write_text(json.dumps(st, indent=1))
                    ssh(
                        host,
                        f"echo '{json.dumps(s['collapsed'])}' > /workspace/atg0/runs/{line}/COLLAPSE.json",
                    )
                    if "nonfinite" not in s["ends"]:
                        kill_line(host, line, why)
            if "yes" in done and s["collapsed"] and not s.get("pre_reads"):
                s["pre_reads"] = pre_reads(host, line, s["collapsed"])
                say(f"{line}: collapsed line, pre-collapse reads: {s['pre_reads']}")
            if s.get("pre_reads") == "launched" and "yes" not in (
                ssh(host, f"test -e /workspace/atg0/DONE2_{line} && echo yes; true") or ""
            ):
                continue
            if "yes" in done:
                r = subprocess.run(["bash", str(HERE / "g0_pod.sh"), "fetch", host, line], capture_output=True,
                                   text=True, timeout=3600)  # fmt: skip
                say(f"{line}: fetch rc {r.returncode}: {(r.stdout + r.stderr).strip()[-400:]}")
                if r.returncode == 0:
                    s["fetched"] = now()
        for line, host in TA_LINES.items():
            s = st["ta"].setdefault(line, {"host": host, "collapsed": None})
            txt = ssh(
                host, f"grep -E 'canary it' /workspace/ta/logs/{line}_seg*.log 2>/dev/null; true"
            )
            if txt is None:
                continue
            can = parse(txt)
            if can:
                s["last_it"] = max(can)
                s["max_colon_pct"] = max((v[1] for i, v in can.items() if i > 0), default=None)
                s["max_wrong_pct"] = max((v[2] for i, v in can.items() if i > 0), default=None)
            if s["collapsed"] is None:
                hit = rule(can)
                if hit is not None:
                    s["collapsed"] = {"it": hit[0], "why": hit[1], "utc": now()}
                    say(
                        f"COLLAPSE (log only, not killed) {line} on {host} at it {hit[0]}: {hit[1]}"
                    )
        st["utc"] = now()
        STATE.write_text(json.dumps(st, indent=1))
        if all(v.get("fetched") for v in st["tag"].values()) and len(st["tag"]) == len(TAG_LINES):
            say("all six lines fetched: monitor exits")
            return
        time.sleep(a.every)
    say("monitor time cap reached: exits")


if __name__ == "__main__":
    main()
