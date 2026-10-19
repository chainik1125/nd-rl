"""Read-only analysis of the ndbench store: per-axis pass@1 / pass@k and where/why attempts fail,
for Dan's cap-12 seed-0 pair (lean_v2 runs). Writes nothing into the store."""

import glob
import gzip
import json
import os
import statistics as st
from collections import Counter, defaultdict

STORE = "/scratch/cowifornia/nd-rl/bench/out/store"
SUITES = "/scratch/cowifornia/nd-rl/bench/out/suites"


def load_suite(name):
    fs = sorted(glob.glob(f"{SUITES}/{name}/v*/items.jsonl"))
    out = {}
    for f in fs:
        for l in open(f):
            r = json.loads(l)
            out[r["pid"]] = r
    return out


def nprem(it):
    return len(it["sequent"]["premises"])


def runs_for(suite, model):
    rs = [json.loads(l) for l in open(f"{STORE}/runs.jsonl")]
    return [
        r["run_id"]
        for r in rs
        if r.get("event") == "start"
        and r.get("suite") == suite
        and r.get("model") == model
        and r.get("judge_id") == "lean_v2"
    ]


def samples(run_id):
    f = f"{STORE}/samples/{run_id}.jsonl.gz"
    if not os.path.exists(f):
        return []
    return [json.loads(l) for l in gzip.open(f, "rt")]


def bucket_reason(s):
    if s["ok"]:
        return "ok"
    nat = json.loads(s["native"]) if s.get("native") else {}
    end = nat.get("end") or ""
    if end == "done":
        return "lean_rejected"
    if end.startswith("syntax:"):
        return end[7:]
    return end or s.get("reason", "?")


def analyse(suite, model, keyfn, run_id=None, filt=None):
    items = load_suite(suite)
    rid = run_id or runs_for(suite, model)[0]
    ss = samples(rid)
    per = defaultdict(list)
    for s in ss:
        it = items.get(s["pid"])
        if it is None:
            continue
        if filt and not filt(it):
            continue
        per[s["pid"]].append(s)
    groups = defaultdict(
        lambda: {
            "items": 0,
            "p1": [],
            "solved": 0,
            "reasons": Counter(),
            "fail_steps": [],
            "early": 0,
            "fails": 0,
            "n": [],
        }
    )
    for pid, lst in per.items():
        it = items[pid]
        g = groups[keyfn(it)]
        g["items"] += 1
        c = sum(x["ok"] for x in lst)
        g["p1"].append(c / len(lst))
        g["n"].append(len(lst))
        g["solved"] += c > 0
        m = nprem(it)
        for x in lst:
            if x["ok"]:
                continue
            g["fails"] += 1
            b = bucket_reason(x)
            g["reasons"][b] += 1
            nat = json.loads(x["native"]) if x.get("native") else {}
            stp = nat.get("steps")
            if stp is not None:
                own = stp - (it.get("prefix") or {}).get("j", 0) if False else stp
                g["fail_steps"].append(stp - m)  # actions after the premise copies
                if stp - m <= 0:
                    g["early"] += 1
    return rid, groups


def show(title, rid, groups, order=None):
    print(f"### {title}  [{rid}]")
    print(
        "key | items | k | pass@1 | pass@k | lean_rej% | formula% | top env reasons (share of fails) | median fail step after premises | fails at <=0 own steps %"
    )
    keys = order or sorted(groups, key=lambda k: (str(type(k)), k))
    for k in keys:
        if k not in groups:
            continue
        g = groups[k]
        F = max(g["fails"], 1)
        top = ", ".join(f"{r} {100 * c / F:.0f}%" for r, c in g["reasons"].most_common(4))
        med = st.median(g["fail_steps"]) if g["fail_steps"] else None
        kk = st.median(g["n"])
        print(
            f"{k} | {g['items']} | {kk:.0f} | {100 * st.mean(g['p1']):.1f}% | {100 * g['solved'] / g['items']:.1f}% | {100 * g['reasons']['lean_rejected'] / F:.0f}% | {100 * g['reasons']['formula'] / F:.0f}% | {top} | {med} | {100 * g['early'] / F:.0f}%"
        )
    print()


if __name__ == "__main__":
    le3 = lambda it: nprem(it) <= 3
    for model in ["dan12-t1-s0", "dan12-fz-s0"]:
        print(f"======== {model}")
        rid, g = analyse(
            "laws",
            model,
            lambda it: it["tags"]["s"],
            filt=lambda it: le3(it) and not it.get("seen_train"),
        )
        show("laws by substitution size s (<=3 premises, unseen)", rid, g)
        rid, g = analyse(
            "laws",
            model,
            lambda it: (
                it["tags"]["schema"],
                "lo" if it["tags"]["s"] <= 2 else ("mid" if it["tags"]["s"] <= 4 else "hi"),
            ),
            filt=lambda it: le3(it) and not it.get("seen_train"),
        )
        show("laws by schema x s band", rid, g)
        rid, g = analyse(
            "families",
            model,
            lambda it: (it["tags"]["family"], it["tags"]["n"]),
            filt=lambda it: le3(it) and not it.get("seen_train"),
        )
        show("families by (family, n), <=3 premises, unseen", rid, g)
        rid, g = analyse(
            "padding", model, lambda it: (it["tags"]["pad_type"], it["tags"]["m"]), filt=le3
        )
        show("padding by (pad_type, m), <=3 premises", rid, g)
        rid, g = analyse("skill_pool_sub", model, lambda it: it["tags"]["skill_group"], filt=le3)
        show("skill_pool_sub", rid, g)
        rid, g = analyse("glue", model, lambda it: it["tags"]["op"], filt=le3)
        show("glue by op (<=3 premises)", rid, g)
        rid, g = analyse("glue", model, lambda it: ">=4" if nprem(it) >= 4 else "<=3")
        show("glue by premise count", rid, g)
