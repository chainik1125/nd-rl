"""Share of checker-passing answers that are unpunctuated word strings, per cell (orchestrator, post hoc, read only).

Answer = text after the last </think>. "Unpunctuated" = no . ! ? in the answer (the auditor's R1 definition).
Rows are joined to graded rows by (pid, chunk, i).
"""

import gzip
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

OE = Path("/scratch/cowifornia/nd-rl/olmo_elicit")
DAN = OE / "ext/nd-takehome"
RUNS = OE / "metrics2/runs"
G = DAN / "artifacts/olmo_ifcurve"
GO = DAN / "artifacts/olmo_ifomega"
reach = {l.strip() for l in open(DAN / "olmo_ifcurve/pids_reach.txt") if l.strip()}
H = {l.split(":")[0].strip() for l in open(OE / "metrics2/inputs/rank1/pids_H.txt") if l.strip()}
man = json.load(open(OE / "metrics2/inputs/steer/manifest.json"))["sets"]
M3H, DEV = set(man["held"]), set(man["dev"])


def key(r):
    return (r["pid"], r.get("chunk", 0), r.get("i", 0))


def load(gp, sp, pids, n=None):
    g = {}
    for l in open(gp):
        r = json.loads(l)
        if r["pid"] in pids:
            g[key(r)] = r
    txt = {}
    with gzip.open(sp, "rt") as f:
        for l in f:
            try:
                r = json.loads(l)
            except json.JSONDecodeError:
                break
            if r["pid"] in pids:
                txt[key(r)] = r.get("text", "")
    if n:  # first n per prompt in (chunk, i) order
        by = defaultdict(list)
        for k in sorted(g):
            by[k[0]].append(k)
        keep = {k for p in by for k in by[p][:n]}
        g = {k: v for k, v in g.items() if k in keep}
    return g, txt


def ans(t):
    i = t.rfind("</think>")
    return t[i + 8 :].strip() if i >= 0 else ""


def stats(g, txt):
    rows = [(ans(txt[k]),) for k, r in g.items() if r["strict"] and k in txt]
    if not rows:
        return 0, float("nan"), float("nan")
    a = [x[0] for x in rows]
    nop = np.mean([not re.search(r"[.!?]", x) for x in a]) * 100
    return len(a), float(np.median([len(x.split()) for x in a])), nop


cells = [
    ("dpo stored, H (16)", GO / "graded/F_dpo_ifb.jsonl", GO / "samples/F_dpo_ifb.jsonl.gz", H, 16),
    (
        "s2300 stored, H (8)",
        GO / "graded/F_s2300_ifb.jsonl",
        GO / "samples/F_s2300_ifb.jsonl.gz",
        H,
        8,
    ),
    ("A1, H", RUNS / "A1/out/graded.jsonl", RUNS / "A1/out/samples_complete.jsonl.gz", H, None),
    ("A1u, H", RUNS / "A1u/out/graded.jsonl", RUNS / "A1u/out/samples.jsonl.gz", H, None),
    ("A2, H", RUNS / "A2/out/graded.jsonl", RUNS / "A2/out/samples.jsonl.gz", H, None),
    ("k16, H", RUNS / "M2cond/out/graded.jsonl", RUNS / "M2cond/out/samples.jsonl.gz", H, None),
    ("ExPO g3.5, H", RUNS / "Xheld/out/graded.jsonl", RUNS / "Xheld/out/samples.jsonl.gz", H, None),
    (
        "ExPO g3.5, dev",
        RUNS / "Xdev35/out/graded.jsonl",
        RUNS / "Xdev35/out/samples.jsonl.gz",
        None,
        None,
    ),
    (
        "ExPO g4.6, dev",
        RUNS / "Xdev46/out/graded.jsonl",
        RUNS / "Xdev46/out/samples.jsonl.gz",
        None,
        None,
    ),
    (
        "s0500 stored, reach (32; job M_s0500)",
        G / "graded/M_s0500.jsonl",
        G / "samples/M_s0500.jsonl.gz",
        reach,
        32,
    ),
    (
        "s2300 stored, reach (32)",
        G / "graded/B_s2300.jsonl",
        G / "samples/B_s2300.jsonl.gz",
        reach,
        32,
    ),
    (
        "k16, reach",
        RUNS / "M2cond/out/graded.jsonl",
        RUNS / "M2cond/out/samples.jsonl.gz",
        reach,
        None,
    ),
    (
        "ExPO g3.5, reach",
        RUNS / "Xheld/out/graded.jsonl",
        RUNS / "Xheld/out/samples.jsonl.gz",
        reach,
        None,
    ),
    (
        "C1 dpo card, arm A",
        RUNS / "temp_dpo_C1/out/graded.jsonl",
        RUNS / "temp_dpo_C1/out/samples.jsonl.gz",
        None,
        None,
    ),
    (
        "C2 dpo T0.8, arm A",
        RUNS / "temp_dpo_C2/out/graded.jsonl",
        RUNS / "temp_dpo_C2/out/samples.jsonl.gz",
        None,
        None,
    ),
    (
        "S5R s0500 card, reach",
        RUNS / "temp_s0500_C1/out/graded.jsonl",
        RUNS / "temp_s0500_C1/out/samples.jsonl.gz",
        reach,
        None,
    ),
    (
        "E1 steer a2, M3 held-out",
        RUNS / "m3_e1/out/graded.jsonl",
        RUNS / "m3_e1/out/samples.jsonl.gz",
        M3H,
        None,
    ),
    (
        "dpo stored, M3 held-out (16)",
        GO / "graded/F_dpo_ifb.jsonl",
        GO / "samples/F_dpo_ifb.jsonl.gz",
        M3H,
        16,
    ),
]
print(f"{'cell':32s} passes  median-words  unpunctuated%")
for nm, gp, sp, P, n in cells:
    if not Path(sp).exists() or not Path(gp).exists():
        print(f"{nm:32s} missing {sp if not Path(sp).exists() else gp}")
        continue
    if P is None:
        P = {json.loads(l)["pid"] for l in open(gp)}
    g, t = load(gp, sp, P, n)
    k, mw, nop = stats(g, t)
    print(f"{nm:32s} {k:6d}  {mw:8.0f}      {nop:6.1f}")
