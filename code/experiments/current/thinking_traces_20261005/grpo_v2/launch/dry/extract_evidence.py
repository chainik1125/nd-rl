#!/usr/bin/env python3
# ruff: noqa: E501  (literal command lines and log lines, compared byte for byte)
# pyright: basic
"""Pull the launcher evidence out of the 2026-10-05..11 session transcripts into evidence.json.

    python3 extract_evidence.py [OUT.json]

Read-only on the transcripts (JSONL on NFS). Each item names its source (file, line, timestamp).
  ps_hb        ps/proc capture of the live grpo_think_h_b s1 L3 GRPO process (2026-10-11 ~10:41Z)
  old_full     serrano harness, 09:07Z: full L3 GRPO command of grpo_think_h_b s1 (final line.sh) and of
               grpo_think_h s1 (the 9,579-byte line.sh before the think-ban edit)
  rec_vs3_s0   serrano harness, 05:07Z: every recorded call of grpo_nothink_vs3 s0 normal on the 8,873-byte line.sh
  rec_h_s1     serrano harness, 08:29Z: first 20 recorded calls of grpo_think_h s1 normal on the 9,579-byte line.sh
  ps_lines     other ps captures of v2 processes (truncated by `cut`)
  log_lines    launcher log lines (launched / exit / read / line done / queue / refusing)
  old_counts   serrano harness per-arm call counts
"""

import json
import os
import re
import sys

BASE = "/home/cowifornia/.claude/projects/-home-cowifornia-code-research-nd-rl/99f17bd8-830c-4f1b-9192-98b5fa5336e1"
FILES = {"main": BASE + ".jsonl"}
for f in sorted(os.listdir(BASE + "/subagents")):
    if f.endswith(".jsonl"):
        FILES[f[:-6]] = f"{BASE}/subagents/{f}"
SELF = "agent-ac677da5640a3c6bc"  # this reconstruction's own transcript: never evidence


def load():
    calls, results = [], {}
    for key, fp in FILES.items():
        if key == SELF:
            continue
        with open(fp) as f:
            for i, line in enumerate(f):
                try:
                    d = json.loads(line)
                except ValueError:
                    continue
                if d.get("type") not in ("assistant", "user"):
                    continue
                c = d.get("message", {}).get("content")
                if not isinstance(c, list):
                    continue
                for b in c:
                    if not isinstance(b, dict):
                        continue
                    if b.get("type") == "tool_use":
                        calls.append(
                            dict(
                                src=key,
                                line=i,
                                ts=d.get("timestamp") or "",
                                id=b.get("id"),
                                name=b.get("name"),
                                input=b.get("input") or {},
                            )
                        )
                    elif b.get("type") == "tool_result":
                        cc = b.get("content")
                        if isinstance(cc, list):
                            cc = "\n".join(x.get("text", "") for x in cc if isinstance(x, dict))
                        results[b.get("tool_use_id")] = cc if isinstance(cc, str) else ""
    calls.sort(key=lambda c: c["ts"])
    return calls, results


def at(calls, results, src, line):
    for c in calls:
        if c["src"].startswith(src) and c["line"] == line:
            return c, results.get(c["id"], "")
    raise KeyError((src, line))


def main() -> None:
    out = (
        sys.argv[1]
        if len(sys.argv) > 1
        else os.path.join(os.path.dirname(os.path.abspath(__file__)), "evidence.json")
    )
    calls, results = load()
    ev = {}
    c, r = at(calls, results, "main", 17371)
    ps = [
        ln
        for ln in r.splitlines()
        if re.match(r"^\s*1259980 ", ln) or re.match(r"^\s*1259978 ", ln)
    ]
    launched = [ln for ln in r.splitlines() if "launched GRPO grpo_think_h_b_s1_L3" in ln]
    c2, r2 = at(calls, results, "main", 17374)
    ev["ps_hb"] = dict(
        src=f"main:{c['line']} {c['ts']} + main:{c2['line']} {c2['ts']}",
        ps=ps,
        launched=launched,
        proc=r2.splitlines(),
    )
    c, r = at(calls, results, "agent-ad6a202caf48dd21", 476)
    ev["old_full"] = dict(
        src=f"agent-ad6a202caf48dd21:{c['line']} {c['ts']}",
        lines=[ln for ln in r.splitlines() if ln.startswith("21600 PY -m code.thinking.grpo")],
        stdout=[ln for ln in r.splitlines() if ln.startswith("00:00Z ")],
    )
    c, r = at(calls, results, "agent-aeb7753436e4d895", 116)
    c3, r3 = at(calls, results, "agent-aeb7753436e4d895", 113)  # the same trial's stdout
    ev["rec_vs3_s0"] = dict(
        src=f"agent-aeb7753436e4d895:{c['line']} {c['ts']} (stdout: :{c3['line']})",
        cut=900,
        lines=[
            ln
            for ln in r.splitlines()
            if re.match(r"^(git|py|python3|nvidia-smi|sleep|rm|timeout)\t", ln)
        ],
        stdout=[ln for ln in r3.splitlines() if ln.startswith("00:00Z ")],
    )
    c, r = at(calls, results, "agent-ad6a202caf48dd21", 96)
    ev["rec_h_s1"] = dict(
        src=f"agent-ad6a202caf48dd21:{c['line']} {c['ts']}",
        cut=600,
        prefix="/tmp/claude-10001/-home-cowifornia-code-research-nd-rl/99f17bd8-830c-4f1b-9192-98b5fa5336e1/scratchpad/dry",
        lines=[
            ln
            for ln in r.splitlines()
            if re.match(r"^(git|py|python3|nvidia-smi|sleep|rm|timeout)\t", ln)
        ][:20],
    )
    pl = []
    for key in (("main", 14113), ("main", 17349)):
        c, r = at(calls, results, *key)
        for ln in r.splitlines():
            if re.search(r"code\.thinking\.(grpo|ei|eval) |watchdog\.py ", ln) and (
                "grpo_v2" in ln or "logs/v2_" in ln
            ):
                pl.append(dict(src=f"{key[0]}:{c['line']} {c['ts']}", line=ln))
    ev["ps_lines"] = pl
    pat = re.compile(
        r"(\d\d:\d\dZ (?:launched (?:GRPO|EI) [^\n|]*|read [a-z0-9]+_[^\n|]*|grpo_v2 line done[^\n|]*|"
        r"(?:GRPO|EI) [a-z_0-9]+_L[23] (?:exit|failed)[^\n|]*|[^\n|]*refusing[^\n|]*|queue GPU [^\n|]*))"
    )
    seen, ll = set(), []
    for c in calls:
        if c["ts"] < "2026-10-10T03:00" or c["ts"] > "2026-10-11T11:48":
            continue
        s = json.dumps(c["input"])
        if "dry" in s or "smoke_auto" in s or "v2auto" in s:
            continue
        for m in pat.finditer(results.get(c["id"], "")):
            x = m.group(1).split('","')[0]
            if x not in seen:
                seen.add(x)
                ll.append(dict(ts=c["ts"][:19], src=f"{c['src']}:{c['line']}", line=x))
    ev["log_lines"] = ll
    oc = []
    for c in calls:
        r = results.get(c["id"], "")
        if c["src"].startswith(("agent-aeb7753436e4d895", "agent-ad6a202caf48dd21")) or (
            c["src"] == "main" and c["line"] == 17536
        ):
            for m in re.finditer(
                r"^\s*(?:(normal|smoke)\s+s(\d) (\S+)\s+exit (\d)\s+calls (\d+)|(\S+) s(\d) exit (\d), calls (\d+))",
                r,
                re.M,
            ):
                oc.append(dict(src=f"{c['src']}:{c['line']}", text=m.group(0).strip()))
            for m in re.finditer(
                r"^\s*(normal|smoke)\s+s(\d) (\S+_b)\s+exit (\d)\s+evals (\d+)\s+L2-only calls dropped:([^\n]*?) ok$",
                r,
                re.M,
            ):
                oc.append(dict(src=f"{c['src']}:{c['line']}", text=m.group(0).strip()))
    ev["old_counts"] = oc
    with open(out, "w") as f:
        json.dump(ev, f, indent=1)
    print(
        f"wrote {out}: "
        + ", ".join(
            f"{k} {len(v) if isinstance(v, list) else len(v.get('lines', v.get('ps', [])))}"
            for k, v in ev.items()
        )
    )


if __name__ == "__main__":
    main()
