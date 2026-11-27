"""Matched passing answers, ExPO g3.5 (Xheld) and stored s2300 (B_s2300, first 32), same reach prompt (orchestrator, read only).
First passing draw of each, in (chunk, i) order."""

import gzip
import json
import sys
from pathlib import Path

OE = Path("/scratch/cowifornia/nd-rl/olmo_elicit")
DAN = OE / "ext/nd-takehome"
RUNS = OE / "metrics2/runs"
G = DAN / "artifacts/olmo_ifcurve"
prob = {}
for l in open(DAN / "olmo_ifomega/problems.jsonl"):
    r = json.loads(l)
    prob[r["pid"]] = r


def key(r):
    return (r.get("chunk", 0), r.get("i", 0))


def first_pass(gp, sp, pid, n=None):
    g = sorted([json.loads(l) for l in open(gp) if f'"{pid}"' in l], key=key)
    g = [r for r in g if r["pid"] == pid][: n or None]
    passk = [key(r) for r in g if r["strict"]]
    if not passk:
        return None, 0, len(g)
    with gzip.open(sp, "rt") as f:
        for l in f:
            if f'"{pid}"' not in l:
                continue
            r = json.loads(l)
            if r["pid"] == pid and key(r) == passk[0]:
                t = r["text"]
                i = t.rfind("</think>")
                return t[i + 8 :].strip(), len(passk), len(g)
    return None, len(passk), len(g)


for pid in sys.argv[1].split(","):
    p = prob[pid]
    print("=" * 90)
    print(pid, p.get("instruction_id_list"))
    print("PROMPT:", p["prompt"][:600].replace("\n", " "))
    for lab, gp, sp, n in (
        ("ExPO", RUNS / "Xheld/out/graded.jsonl", RUNS / "Xheld/out/samples.jsonl.gz", None),
        ("s2300", G / "graded/B_s2300.jsonl", G / "samples/B_s2300.jsonl.gz", 32),
    ):
        a, k, m = first_pass(gp, sp, pid, n)
        print(
            f"--- {lab}: {k}/{m} pass; first passing answer ({len(a.split()) if a else 0} words):"
        )
        print((a or "")[:700])
