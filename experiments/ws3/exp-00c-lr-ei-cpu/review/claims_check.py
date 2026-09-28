"""Phase 2: recompute the extra numeric claims in results.md and the verify_* reports that the
Recount did not already cover. Own code; run from code/:

    uv run python ../experiments/ws3/exp-00c-lr-ei-cpu/review/claims_check.py

Writes claims_check.json.
"""

import json
import os
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
REPO = os.path.abspath(os.path.join(EXP, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "code", "ws3"))
import ndpath  # noqa: E402,F401

from strict_verify import safe_verify  # noqa: E402

sys.path.insert(0, HERE)
from recount_lib import read_jsonl, renumber, split_prompt  # noqa: E402

ATOMS = ("P", "Q", "R", "S")
OUT = {}


def dump(model):
    d = os.path.join(EXP, "run_lr3e-5", "baseline", "dumps") if model == "base" else os.path.join(EXP, f"run_{model}", "evaluate", "dumps")
    return read_jsonl(os.path.join(d, "long_pool.jsonl.gz"))


V = {}


def ver(p, c):
    if (p, c) not in V:
        V[(p, c)] = safe_verify(p, c)
    return V[(p, c)]


rows = {m: dump(m) for m in ("base", "lr3e-5", "lr1e-4", "lr3e-4", "ei")}
rate = {}
acc = {}
for m, rs in rows.items():
    g = defaultdict(list)
    a = defaultdict(list)
    for r in rs:
        ok = ver(r["prompt"], r["completion"])[0]
        g[r["name"]].append(ok)
        if ok:
            a[r["name"]].append(r)
    rate[m] = {n: sum(v) / len(v) for n, v in g.items()}
    acc[m] = a
L = {r["name"]: r["length"] for r in rows["base"]}
P = {r["name"]: r["prompt"] for r in rows["base"]}

# 1. distinct accepted proofs per newly solved problem (frozen 0/32, trained >= 1/32)
div = {}
for m in ("lr3e-5", "lr1e-4", "lr3e-4", "ei"):
    new = [n for n in rate[m] if rate["base"][n] == 0 and rate[m][n] > 0]
    c = Counter(len({renumber(r["completion"]) for r in acc[m][n]}) for n in new)
    craw = Counter(len({r["completion"] for r in acc[m][n]}) for n in new)
    div[m] = {"newly_solved": len(new), "distinct_after_renumber": dict(sorted(c.items())), "distinct_raw_text": dict(sorted(craw.items()))}
OUT["diversity_newly_solved"] = div

# 2. EI SFT data: duplicates after renumbering, distinct proofs overall
sft = [read_jsonl(os.path.join(EXP, "run_ei", "sft_data", f"round{k}.jsonl.gz")) for k in (1, 2, 3)]
per_round = []
allk = set()
for recs in sft:
    keys = [(r["prompt"], renumber(r["completion"])) for r in recs]
    per_round.append({"records": len(recs), "distinct_after_renumber": len(set(keys)), "duplicates": len(keys) - len(set(keys)),
                      "distinct_raw": len({(r["prompt"], r["completion"]) for r in recs})})
    allk |= set(keys)
OUT["sft"] = {"per_round": per_round, "distinct_all_rounds_after_renumber": len(allk), "total_records": sum(len(x) for x in sft)}
# EI eval accepted proofs appearing in SFT (exact text, and after renumbering; SFT prompts are pool prompts, eval prompts are
# disjoint, so any match would need the same prompt)
sft_exact = {(r["prompt"], r["completion"]) for recs in sft for r in recs}
sft_ren = allk
ei_acc = [r for n in acc["ei"] for r in acc["ei"][n]]
OUT["ei_eval_accepted_in_sft"] = {"accepted": len(ei_acc),
                                   "exact": sum((r["prompt"], r["completion"]) in sft_exact for r in ei_acc),
                                   "renumbered": sum((r["prompt"], renumber(r["completion"])) in sft_ren for r in ei_acc),
                                   "proof_body_renumbered_any_prompt": sum(renumber(r["completion"]) in {k[1] for k in sft_ren} for r in ei_acc)}


# 3. renumbering to N1 changes verdicts?
def renum_from1(c):
    return renumber(c)


ch = {}
for m in ("lr3e-5", "ei", "lr1e-4", "lr3e-4", "base"):
    n = d = 0
    for r in rows[m]:
        ok = ver(r["prompt"], r["completion"])[0]
        ok2 = safe_verify(r["prompt"], renum_from1(r["completion"]))[0]
        n += 1
        d += ok != ok2
    ch[m] = {"rows": n, "verdict_changes": d}
OUT["renumber_from_N1_verdict_changes"] = ch


# 4. eval prompts whose renamed conclusion equals a pool prompt's renamed conclusion
def ckey(s):
    m, out = {}, []
    for t in s.split():
        if t in ATOMS:
            if t not in m:
                m[t] = ATOMS[len(m)]
            out.append(m[t])
        else:
            out.append(t)
    return " ".join(out)


pool = read_jsonl(os.path.join(EXP, "pool.jsonl"))
pc = {ckey(split_prompt(r["prompt"])[1]) for r in pool}
hit = [n for n in P if ckey(split_prompt(P[n])[1]) in pc]
rest = [n for n in P if n not in set(hit)]
OUT["conclusion_overlap_with_pool"] = {
    "n": len(hit),
    **{m: {"hit_avg": round(sum(rate[m][n] for n in hit) / len(hit), 3), "rest_avg": round(sum(rate[m][n] for n in rest) / len(rest), 3)} for m in ("base", "lr3e-5", "ei")},
}


# 5. box depth >= 3
def maxdepth(c):
    best = 0
    for line in c.split(" ; "):
        t = line.split()
        k = 0
        for tok in t[1:]:
            if tok == "|":
                k += 1
            else:
                break
        best = max(best, k)
    return best


dep = {}
for m in ("base", "ei"):
    accd = sum(1 for r in rows[m] if ver(r["prompt"], r["completion"])[0] and maxdepth(r["completion"]) >= 3)
    alld = sum(1 for r in rows[m] if maxdepth(r["completion"]) >= 3)
    dep[m] = {"accepted_depth>=3": accd, "all_rows_depth>=3": alld}
OUT["depth3"] = dep


# 6. accepted proofs citing no premise line
def no_premise_cite(c):
    lines = [l.split() for l in c.split(" ; ") if l.strip() and l.strip() != "QED"]
    pr = {t[0] for t in lines if t and t[-1] == "PR"}
    if not pr:
        return True
    cited = set()
    for t in lines:
        if ":" in t:
            cited |= {x for x in t[t.index(":") + 2 :] if x.startswith("N")}
    return not (cited & pr)


OUT["accepted_no_premise_cited"] = {m: sum(no_premise_cite(r["completion"]) for n in acc[m] for r in acc[m][n]) for m in ("lr3e-5", "ei", "base")}

# 7. accepted line counts by L_true
lc = {}
for m in ("lr3e-5", "ei", "base", "lr1e-4", "lr3e-4"):
    c = Counter()
    for n in acc[m]:
        for r in acc[m][n]:
            c[f"{L[n]}->{ver(r['prompt'], r['completion'])[2]}"] += 1
    lc[m] = dict(sorted(c.items()))
OUT["accepted_lines_by_L_true"] = lc


# 8. EI L8 skeleton share
def skel(c):
    out = []
    for line in c.split(" ; "):
        t = line.split()
        if not t or t[0] == "QED":
            continue
        d = sum(1 for x in t[1:] if x == "|")
        rule = t[t.index(":") + 1] if ":" in t else "?"
        out.append(f"{rule}{'|' * d}")
    return " ".join(out)


l8 = [r for n in acc["ei"] if L[n] == 8 for r in acc["ei"][n]]
sk = Counter(skel(r["completion"]) for r in l8)
OUT["ei_L8_accepted"] = len(l8)
OUT["ei_L8_top_skeletons"] = sk.most_common(12)


def three_nested(s):
    toks = s.split()
    return any(t.endswith("|||") for t in toks) and toks[-3:] == ["IMPI||", "IMPI|", "IMPI"] and sum(t.startswith("AS") for t in toks) == 3


OUT["ei_L8_three_nested_one_step_share"] = f"{sum(v for s, v in sk.items() if three_nested(s))}/{len(l8)}"

# 9. EI vs lr3e-5 overlap of solved-at-least-once
ov = {}
for Ln in (7, 8):
    ns = [n for n in L if L[n] == Ln]
    e = {n for n in ns if rate["ei"][n] > 0}
    g = {n for n in ns if rate["lr3e-5"][n] > 0}
    ov[Ln] = {"ei": len(e), "lr3e-5": len(g), "both": len(e & g), "ei_only": len(e - g), "grpo_only": len(g - e)}
OUT["ei_vs_lr3e-5_solved"] = ov
OUT["ei_solves_every_frozen_solved"] = all(rate["ei"][n] > 0 for n in L if rate["base"][n] > 0)

# 10. textbook ref>=7 any per model (for results.md table row)
TB = {r["name"]: r for r in read_jsonl(os.path.join(ndpath.TEXTBOOK72, "textbook_dev.jsonl"))}
t7 = {}
for m in ("base", "lr3e-5", "lr1e-4", "lr3e-4", "ei"):
    d = os.path.join(EXP, "run_lr3e-5", "baseline", "dumps") if m == "base" else os.path.join(EXP, f"run_{m}", "evaluate", "dumps")
    s = set()
    for r in read_jsonl(os.path.join(d, "textbook72.jsonl.gz")):
        if r["name"] in TB and TB[r["name"]]["reference_lines"] >= 7 and ver(r["prompt"], r["completion"])[0]:
            s.add(r["name"])
    t7[m] = sorted(s)
OUT["textbook_ref7_solved"] = t7

# 11. wall-clock from stages.jsonl
st = {}
for m in ("lr3e-5", "lr1e-4", "lr3e-4", "ei"):
    st[m] = {r["stage"]: round(r["secs"]) for r in read_jsonl(os.path.join(EXP, f"run_{m}", "stages.jsonl"))}
OUT["stage_secs"] = st
OUT["filter_secs"] = round(json.load(open(os.path.join(EXP, "filter_summary.json")))["secs"])

print(json.dumps(OUT, indent=1, default=str))
json.dump(OUT, open(os.path.join(HERE, "claims_check.json"), "w"), indent=1, default=str)
