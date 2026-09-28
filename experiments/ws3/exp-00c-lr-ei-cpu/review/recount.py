"""Reviewer recount for exp-00c (plan section 6). Independent of the executor's eval package and
of analyze.py (not opened).

Run from code/:  uv run python ../experiments/ws3/exp-00c-lr-ei-cpu/review/recount.py
Writes recount.json and recount_stdout.txt next to this file. Imports only
strict_verify.safe_verify (checker of record), the vendored nd_data.canonical_theorem (second
opinion key) and ndpath (sys.path). Nothing from code/ws3/eval or code/ws3/data.py.
"""

import gzip
import hashlib
import json
import math
import os
import random
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
REPO = os.path.abspath(os.path.join(EXP, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "code", "ws3"))
import ndpath  # noqa: E402,F401

from nd_data import canonical_theorem  # noqa: E402
from strict_verify import safe_verify  # noqa: E402

ARMS = ["lr3e-5", "lr1e-4", "lr3e-4", "ei"]
GRPO = ["lr3e-5", "lr1e-4", "lr3e-4"]
ATOMS = ("P", "Q", "R", "S")
OUT = {}
LOG = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)


def read_jsonl(path):
    if not os.path.exists(path) and os.path.exists(path + ".gz"):
        path = path + ".gz"
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as f:
        return [json.loads(l) for l in f if l.strip()]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def split_prompt(prompt):
    t = prompt.split()
    assert t[0] == "THM" and t[-1] == "PRF", prompt
    i = t.index("SEQ")
    prems, cur = [], []
    for tok in t[1:i]:
        if tok == ",":
            prems.append(" ".join(cur))
            cur = []
        else:
            cur.append(tok)
    if cur:
        prems.append(" ".join(cur))
    return prems, " ".join(t[i + 1 : -1])


def thm_of_prompt(prompt):
    prems, con = split_prompt(prompt)
    return (" , ".join(prems) + " |- " + con) if prems else ("|- " + con)


def canon_key(prompt):
    m, out = {}, []
    for tok in thm_of_prompt(prompt).split():
        if tok in ATOMS:
            if tok not in m:
                m[tok] = ATOMS[len(m)]
            out.append(m[tok])
        else:
            out.append(tok)
    return " ".join(out)


def ctheorem(prompt):
    return canonical_theorem(prompt)


def is_trivial(prompt):
    prems, con = split_prompt(prompt)
    return con in prems or "F" in prems


VCACHE = {}


def verify(prompt, comp):
    k = (prompt, comp)
    if k not in VCACHE:
        VCACHE[k] = safe_verify(prompt, comp)
    return VCACHE[k]


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def boot_ci(xs, seed=0, B=10000):
    if not xs:
        return [float("nan")] * 2
    rng = random.Random(seed)
    n = len(xs)
    ms = sorted(sum(xs[rng.randrange(n)] for _ in range(n)) / n for _ in range(B))
    return [ms[int(0.025 * B)], ms[int(0.975 * B) - 1]]


def t_ci(xs):
    n = len(xs)
    if n < 2:
        return [float("nan")] * 2
    m = mean(xs)
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))
    h = 1.96 * sd / math.sqrt(n)
    return [max(0.0, m - h), min(1.0, m + h)]


def sign_test(up, down):
    n = up + down
    if n == 0:
        return 1.0
    k = min(up, down)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / 2**n
    return min(1.0, 2 * p)


def r4(x):
    return None if x is None else round(x, 4)


# ------------------------------------------------------------------ fixed data
FD = ndpath.FORK_DATA
rl = read_jsonl(os.path.join(FD, "rl_targets.jsonl"))
heldout = read_jsonl(os.path.join(FD, "heldout.jsonl"))
transfer = read_jsonl(os.path.join(FD, "transfer.jsonl"))
val36 = read_jsonl(os.path.join(FD, "validation_36.jsonl"))
val36r = read_jsonl(os.path.join(FD, "validation_36_reference_proofs.jsonl"))
tshort = read_jsonl(os.path.join(FD, "test_short_prompts.jsonl"))
tlong = read_jsonl(os.path.join(FD, "test_long_prompts.jsonl"))
tb_dev = read_jsonl(os.path.join(ndpath.TEXTBOOK72, "textbook_dev.jsonl"))
tb_tr = read_jsonl(os.path.join(ndpath.TEXTBOOK72, "textbook_train.jsonl"))
TB = {r["name"]: dict(r, slice="dev58") for r in tb_dev}
TB.update({r["name"]: dict(r, slice="train14") for r in tb_tr})
say("textbook", len(tb_dev), len(tb_tr), "dev ref>=7:", sum(r["reference_lines"] >= 7 for r in tb_dev))

# canon_key reproduces the fork's key field?
mm = sum(canon_key(r["prompt"]) != r["key"] for r in rl + heldout + transfer)
say("canon_key mismatches vs fork key field (rl_targets+heldout+transfer):", mm)
OUT["canon_key_mismatch_fork"] = mm

# ------------------------------------------------------------------ split
tr_half = read_jsonl(os.path.join(EXP, "split/rl_targets_train.jsonl.gz"))
ev_half = read_jsonl(os.path.join(EXP, "split/rl_targets_eval.jsonl.gz"))
eval_set = read_jsonl(os.path.join(EXP, "eval_set.jsonl"))
cands = read_jsonl(os.path.join(EXP, "candidates.jsonl.gz"))
pool = read_jsonl(os.path.join(EXP, "pool.jsonl"))
sp = {}
sp["n_train_half"], sp["n_eval_half"] = len(tr_half), len(ev_half)
sp["train_half_by_L"] = dict(sorted(Counter(int(r["L_true"]) for r in tr_half).items()))
sp["eval_half_by_L"] = dict(sorted(Counter(int(r["L_true"]) for r in ev_half).items()))
ktr = {canon_key(r["prompt"]) for r in tr_half}
kev = {canon_key(r["prompt"]) for r in ev_half}
sp["halves_canon_key_overlap"] = len(ktr & kev)
sp["halves_ctheorem_overlap"] = len({ctheorem(r["prompt"]) for r in tr_half} & {ctheorem(r["prompt"]) for r in ev_half})
names_rl = sorted(r["name"] for r in rl)
sp["halves_union_equals_rl_targets_by_name"] = sorted(r["name"] for r in tr_half + ev_half) == names_rl
rl_by_name = {r["name"]: r for r in rl}
# make_split overwrites `source` with "rl_targets" and adds key/length; every other field must match
sp["half_records_match_rl_targets_fields_except_source"] = all(
    all(rl_by_name[r["name"]][k] == r[k] for k in rl_by_name[r["name"]] if k != "source") for r in tr_half + ev_half
)
sp["half_key_equals_canon_key"] = all(r["key"] == canon_key(r["prompt"]) for r in tr_half + ev_half)
sp["half_length_equals_L_true"] = all(r["length"] == int(r["L_true"]) for r in tr_half + ev_half)
# half assignment rule
def half_of(key):
    return "train" if int(hashlib.sha256(key.encode()).hexdigest(), 16) % 2 == 0 else "eval"
sp["half_rule_reproduced"] = all(half_of(r["key"]) == "train" for r in tr_half) and all(half_of(r["key"]) == "eval" for r in ev_half)
ev_names = {r["name"] for r in ev_half}
sp["eval_set_n"] = len(eval_set)
sp["eval_set_by_L"] = dict(sorted(Counter(int(r["L_true"]) for r in eval_set).items()))
sp["eval_set_all_in_eval_half"] = all(r["name"] in ev_names for r in eval_set)
sp["candidates_n"] = len(cands)
sp["candidates_by_L_source"] = dict(Counter(f"{r['length']}:{r['source']}" for r in cands))
sp["pool_n"] = len(pool)
sp["pool_by_L"] = dict(sorted(Counter(r["length"] for r in pool).items()))
cand_prompts = {r["prompt"] for r in cands}
sp["pool_subset_of_candidates"] = all(r["prompt"] in cand_prompts for r in pool)
tr_prompts = {r["prompt"] for r in tr_half}
sp["pool_L78_all_in_train_half"] = all(r["prompt"] in tr_prompts for r in pool if r["length"] >= 7)
OUT["split"] = sp
say("split", json.dumps(sp))

# ------------------------------------------------------------------ filter recount
fd = read_jsonl(os.path.join(EXP, "filter_dump.jsonl.gz"))
fcount = defaultdict(int)
fk = Counter()
fdis = 0
for r in fd:
    ok = verify(r["prompt"], r["completion"])[0]
    fdis += ok != bool(r["ok"])
    fcount[r["prompt"]] += ok
    fk[r["prompt"]] += 1
flt = {"rows": len(fd), "verdict_disagreements": fdis, "k_values": dict(Counter(fk.values()))}
bylen = defaultdict(Counter)
cl = {r["prompt"]: r["length"] for r in cands}
for p, n in fk.items():
    c = fcount[p]
    bylen[cl[p]]["zero" if c == 0 else "all" if c == 8 else "mixed"] += 1
flt["by_length"] = {L: dict(v) for L, v in sorted(bylen.items())}
flt["pool_frozen_filter_ok_matches"] = all(fcount[r["prompt"]] == r["frozen_filter_ok"] for r in pool)
flt["pool_all_mixed"] = all(0 < fcount[r["prompt"]] < 8 for r in pool)
mixed = Counter(cl[p] for p in fk if 0 < fcount[p] < 8)
flt["mixed_by_L"] = dict(mixed)
flt["pool_contains_every_mixed_L7_L8"] = all(
    p in {r["prompt"] for r in pool} for p in fk if 0 < fcount[p] < 8 and cl[p] >= 7
)
OUT["filter"] = flt
say("filter", json.dumps(flt))

# ------------------------------------------------------------------ dumps
PARTS = {"long": "long_pool.jsonl.gz", "greedy": "greedy_heldout.jsonl.gz", "tb": "textbook72.jsonl.gz"}
base_dir = os.path.join(EXP, "run_lr3e-5", "baseline", "dumps")
copies = {}
for a in ARMS:
    for part, fn in PARTS.items():
        # compare decompressed content (gzip headers carry the compression time)
        copies[f"{a}/{part}"] = read_jsonl(os.path.join(EXP, f"run_{a}", "baseline", "dumps", fn)) == read_jsonl(
            os.path.join(base_dir, fn)
        )
OUT["baseline_copies_identical"] = copies
say("baseline copies identical:", all(copies.values()))

MODELS = {"base": os.path.join(EXP, "run_lr3e-5", "baseline", "dumps")}
for a in ARMS:
    MODELS[a] = os.path.join(EXP, f"run_{a}", "evaluate", "dumps")

reverify = {}
per = {}  # model -> part -> name -> [ok,...]
meta = {}  # name -> (length, slice)
kt = {}
for m, d in MODELS.items():
    per[m] = {}
    for part, fn in PARTS.items():
        rows = read_jsonl(os.path.join(d, fn))
        dis = nld = 0
        acc = 0
        g = defaultdict(list)
        ks = Counter()
        for r in rows:
            ok, reason, n = verify(r["prompt"], r["completion"])
            dis += ok != bool(r["ok"])
            if ok and int(r["n_lines_written"]) != n:
                nld += 1
            acc += ok
            g[r["name"]].append(ok)
            meta[r["name"]] = (r["length"], r["slice"], r["prompt"])
            ks[(r["k"], r["temperature"])] += 1
        per[m][part] = dict(g)
        kt[f"{m}/{part}"] = {str(k): v for k, v in ks.items()}
        reverify[f"{m}/{part}"] = {"rows": len(rows), "accepted": acc, "disagree": dis, "nlines_disagree": nld}
OUT["reverify_eval_dumps"] = reverify
OUT["k_temperature"] = kt
for k, v in reverify.items():
    say("reverify", k, v)

# same prompt sets across models
OUT["same_prompt_sets"] = {
    part: all(set(per[m][part]) == set(per["base"][part]) for m in MODELS) for part in PARTS
}
es_names = {r["name"] for r in eval_set}
OUT["long_dump_equals_eval_set"] = set(per["base"]["long"]) == es_names
es_len = {r["name"]: int(r["L_true"]) for r in eval_set}
OUT["long_dump_length_mismatch"] = sum(meta[n][0] != es_len[n] for n in es_names)
OUT["textbook_dump_names_equal_72"] = set(per["base"]["tb"]) == set(TB)
say("same prompt sets", OUT["same_prompt_sets"], "long==eval_set", OUT["long_dump_equals_eval_set"],
    "len mismatch", OUT["long_dump_length_mismatch"], "tb names", OUT["textbook_dump_names_equal_72"])


def rate(v):
    return sum(v) / len(v)


def by_length_table(part, key_len):
    res = {}
    for m in MODELS:
        d = per[m][part]
        groups = defaultdict(list)
        for n, v in d.items():
            groups[key_len(n)].append(n)
        res[m] = {}
        for L, ns in sorted(groups.items(), key=lambda x: (str(type(x[0])), x[0])):
            rs = [rate(d[n]) for n in ns]
            res[m][str(L)] = {
                "n": len(ns),
                "k": len(d[ns[0]]),
                "avg": r4(mean(rs)),
                "ci95_t": [r4(x) for x in t_ci(rs)],
                "ci95_boot": [r4(x) for x in boot_ci(rs)],
                "any": sum(r > 0 for r in rs),
                "reliable50": sum(r >= 0.5 for r in rs),
                "reliable90": sum(r >= 0.9 for r in rs),
                "expected_solves": r4(sum(rs)),
            }
    return res


# 1. long pool by length
OUT["long_by_length"] = by_length_table("long", lambda n: meta[n][0])
all_long = {m: mean([rate(v) for v in per[m]["long"].values()]) for m in MODELS}
OUT["long_all_avg"] = {m: r4(v) for m, v in all_long.items()}
say("\nLONG POOL (eval_set) avg / any / reliable50 by L")
for m in MODELS:
    for L, v in OUT["long_by_length"][m].items():
        say(f"  {m:7s} L{L}: n={v['n']} k={v['k']} avg={v['avg']} ci_t={v['ci95_t']} boot={v['ci95_boot']} any={v['any']} rel50={v['reliable50']} rel90={v['reliable90']}")

# per-problem base vs trained on long pool
pp = {}
for a in ARMS:
    res = {}
    for L in (7, 8):
        ns = [n for n in per["base"]["long"] if meta[n][0] == L]
        b = {n: rate(per["base"]["long"][n]) for n in ns}
        t = {n: rate(per[a]["long"][n]) for n in ns}
        newcap = [n for n in ns if b[n] == 0 and t[n] >= 0.5]
        newany = [n for n in ns if b[n] == 0 and t[n] > 0]
        sharp = [n for n in ns if b[n] > 0 and t[n] > b[n]]
        worse = [n for n in ns if b[n] > 0 and t[n] < b[n]]
        lost = [n for n in ns if b[n] > 0 and t[n] == 0]
        lost_rel = [n for n in ns if b[n] >= 0.5 and t[n] < 0.5]
        solved = [n for n in ns if b[n] > 0]
        up = sum(t[n] > b[n] for n in solved)
        down = sum(t[n] < b[n] for n in solved)
        same = sum(t[n] == b[n] for n in solved)
        diffs_s = [t[n] - b[n] for n in solved]
        diffs_all = [t[n] - b[n] for n in ns]
        res[str(L)] = {
            "n": len(ns),
            "base_solved_once": len(solved),
            "new_capability": len(newcap),
            "new_capability_names": newcap,
            "new_any": len(newany),
            "sharpened(base>0,up)": len(sharp),
            "worse(base>0,down)": len(worse),
            "lost_any(base>0,trained 0)": len(lost),
            "lost_reliable(base>=.5,trained<.5)": len(lost_rel),
            "paired_on_base_solved": {"up": up, "down": down, "same": same, "sign_p": r4(sign_test(up, down))},
            "mean_diff_base_solved": r4(mean(diffs_s)),
            "mean_diff_base_solved_boot95": [r4(x) for x in boot_ci(diffs_s)],
            "mean_diff_all": r4(mean(diffs_all)),
            "mean_diff_all_boot95": [r4(x) for x in boot_ci(diffs_all)],
        }
    pp[a] = res
OUT["per_problem_long"] = pp
say("\nPER PROBLEM long pool")
for a in ARMS:
    for L, v in pp[a].items():
        vv = {k: x for k, x in v.items() if k != "new_capability_names"}
        say(f"  {a} L{L}:", json.dumps(vv))

# greedy heldout
OUT["greedy_by_length"] = by_length_table("greedy", lambda n: meta[n][0])
g = {}
for m in MODELS:
    d = per[m]["greedy"]
    g[m] = {"avg": r4(mean([rate(v) for v in d.values()])), "solved": sum(v[0] for v in d.values())}
    if m != "base":
        b = per["base"]["greedy"]
        g[m]["lost"] = sum(b[n][0] and not d[n][0] for n in d)
        g[m]["gained"] = sum((not b[n][0]) and d[n][0] for n in d)
        g[m]["avg_drop_vs_base"] = r4(mean([rate(v) for v in b.values()]) - mean([rate(v) for v in d.values()]))
OUT["greedy"] = g
say("\nGREEDY heldout 2-6", json.dumps(g))
for m in MODELS:
    say("  ", m, {L: (v["avg"], v["n"]) for L, v in OUT["greedy_by_length"][m].items()})

# 2. textbook
tbres = {}
for m in MODELS:
    d = per[m]["tb"]
    rates = {n: rate(v) for n, v in d.items()}
    sl = {
        "dev58": [n for n in d if TB[n]["slice"] == "dev58"],
        "train14": [n for n in d if TB[n]["slice"] == "train14"],
        "all72": list(d),
        "dev_ref>=7 (43)": [n for n in d if TB[n]["slice"] == "dev58" and TB[n]["reference_lines"] >= 7],
    }
    tbres[m] = {}
    for s, ns in sl.items():
        rs = [rates[n] for n in ns]
        tbres[m][s] = {
            "n": len(ns),
            "k": len(d[ns[0]]),
            "avg": r4(mean(rs)),
            "ci95_t": [r4(x) for x in t_ci(rs)],
            "any": sum(r > 0 for r in rs),
            "reliable50": sum(r >= 0.5 for r in rs),
            "reliable90": sum(r >= 0.9 for r in rs),
            "expected_solves": r4(sum(rs)),
            "solved_names": sorted(n for n in ns if rates[n] > 0) if s == "dev_ref>=7 (43)" else None,
        }
    tbres[m]["dump_length_matches_reference"] = sum(
        meta[n][0] != TB[n].get("reference_lines", meta[n][0]) for n in d if TB[n]["slice"] == "dev58"
    ) == 0
OUT["textbook"] = tbres
say("\nTEXTBOOK")
for m in MODELS:
    for s in ("dev58", "train14", "all72", "dev_ref>=7 (43)"):
        v = tbres[m][s]
        say(f"  {m:7s} {s:16s} n={v['n']} k={v['k']} avg={v['avg']} ci={v['ci95_t']} any={v['any']} rel50={v['reliable50']} exp={v['expected_solves']} {v['solved_names'] or ''}")
# textbook per problem changes (any-level)
tbchg = {}
for a in ARMS:
    b, t = per["base"]["tb"], per[a]["tb"]
    tbchg[a] = {
        "gained_any": sorted(n for n in b if rate(b[n]) == 0 and rate(t[n]) > 0),
        "lost_any": sorted(n for n in b if rate(b[n]) > 0 and rate(t[n]) == 0),
        "new_capability": sorted(n for n in b if rate(b[n]) == 0 and rate(t[n]) >= 0.5),
    }
OUT["textbook_changes"] = tbchg
say("textbook changes", json.dumps(tbchg))

# ------------------------------------------------------------------ 4. reward sanity
rs_out = {}
for a in ARMS:
    comps = read_jsonl(os.path.join(EXP, f"run_{a}", "completions.jsonl.gz"))
    dis = nld = acc = 0
    vals = Counter()
    for r in comps:
        ok, reason, n = verify(r["prompt"], r["completion"])
        vals[r["reward"]] += 1
        dis += ok != (float(r["reward"]) >= 0.5)
        if ok and int(r["n_lines"]) != n:
            nld += 1
        acc += ok
    o = {"rows": len(comps), "accepted": acc, "reward_disagree": dis, "nlines_disagree": nld,
         "reward_values": {str(k): v for k, v in vals.items()}}
    if a in GRPO:
        G = 16
        bad = 0
        nz = 0
        groups = 0
        step_nz = defaultdict(list)
        for i in range(0, len(comps), G):
            grp = comps[i : i + G]
            if len({r["prompt"] for r in grp}) != 1 or len(grp) != G or len({r["step"] for r in grp}) != 1:
                bad += 1
                continue
            oks = [verify(r["prompt"], r["completion"])[0] for r in grp]
            groups += 1
            mixed_ = 0 < sum(oks) < G
            nz += mixed_
            step_nz[grp[0]["step"]].append(mixed_)
        o.update({"groups": groups, "malformed_groups": bad, "frac_groups_nonzero_std": r4(nz / max(groups, 1)),
                  "steps": len(step_nz),
                  "frac_nonzero_std_first5_steps": r4(mean([mean(step_nz[s]) for s in sorted(step_nz)[:5]])),
                  "frac_nonzero_std_last5_steps": r4(mean([mean(step_nz[s]) for s in sorted(step_nz)[-5:]])),
                  "steps_with_zero_mixed_groups": sum(not any(v) for v in step_nz.values())})
        # reward first/last 5 steps
        by_step = defaultdict(list)
        for r in comps:
            by_step[r["step"]].append(verify(r["prompt"], r["completion"])[0])
        ss = sorted(by_step)
        o["reward_first5"] = r4(mean([mean(by_step[s]) for s in ss[:5]]))
        o["reward_last5"] = r4(mean([mean(by_step[s]) for s in ss[-5:]]))
        # prompts per step distinct
        o["prompts_per_step"] = dict(Counter(len({r["prompt"] for r in comps if r["step"] == s}) for s in ss))
    else:
        by_round = defaultdict(list)
        for r in comps:
            by_round[r["round"]].append(verify(r["prompt"], r["completion"])[0])
        o["reward_by_round"] = {k: r4(mean(v)) for k, v in sorted(by_round.items())}
        o["rows_by_round"] = {k: len(v) for k, v in sorted(by_round.items())}
        # sft data: every record is a verified completion of a pool prompt, sampled that round
        sft = {}
        for rnd in (1, 2, 3):
            recs = read_jsonl(os.path.join(EXP, "run_ei", "sft_data", f"round{rnd}.jsonl.gz"))
            ver = sum(verify(r["prompt"], r["completion"])[0] for r in recs)
            pairs = {(r["prompt"], r["completion"]) for r in comps if r["round"] == rnd}
            sft[rnd] = {"records": len(recs), "verified": ver,
                        "all_in_round_completions": all((r["prompt"], r["completion"]) in pairs for r in recs),
                        "max_per_prompt": max(Counter(r["prompt"] for r in recs).values()),
                        "prompts": len({r["prompt"] for r in recs})}
        o["sft"] = sft
    # prompts used
    o["distinct_prompts"] = len({r["prompt"] for r in comps})
    o["prompts_subset_of_pool"] = {r["prompt"] for r in comps} <= {r["prompt"] for r in pool}
    rs_out[a] = o
    say("reward sanity", a, json.dumps(o))
OUT["reward_sanity"] = rs_out

# trivial and short accepted proofs
Ltrue = {r["prompt"]: int(r["L_true"]) for r in rl}
triv = {}
short = []
for m, d in MODELS.items():
    for part, fn in PARTS.items():
        rows = read_jsonl(os.path.join(d, fn))
        nacc = nt = 0
        for r in rows:
            ok, _, n = verify(r["prompt"], r["completion"])
            if not ok:
                continue
            nacc += 1
            nt += is_trivial(r["prompt"])
            if r["prompt"] in Ltrue and n < Ltrue[r["prompt"]]:
                short.append({"model": m, "part": part, "name": r["name"], "n": n, "L_true": Ltrue[r["prompt"]]})
        triv[f"{m}/{part}"] = f"{nt}/{nacc}"
for a in ARMS:
    comps = read_jsonl(os.path.join(EXP, f"run_{a}", "completions.jsonl.gz"))
    nacc = nt = 0
    for r in comps:
        ok, _, n = verify(r["prompt"], r["completion"])
        if ok:
            nacc += 1
            nt += is_trivial(r["prompt"])
            if r["prompt"] in Ltrue and n < Ltrue[r["prompt"]]:
                short.append({"model": a, "part": "train", "n": n, "L_true": Ltrue[r["prompt"]]})
    triv[f"{a}/train_completions"] = f"{nt}/{nacc}"
OUT["trivial_accepted"] = triv
OUT["accepted_shorter_than_L_true"] = short
# textbook: accepted shorter than reference_lines (reference need not be minimal)
tbshort = Counter()
for m, d in MODELS.items():
    for r in read_jsonl(os.path.join(d, PARTS["tb"])):
        ok, _, n = verify(r["prompt"], r["completion"])
        rl_ = TB[r["name"]].get("reference_lines")
        if ok and rl_ and n < rl_:
            tbshort[(m, r["name"], n, rl_)] += 1
OUT["textbook_accepted_shorter_than_reference"] = [list(k) + [v] for k, v in tbshort.items()]
say("trivial", triv, "short-than-Ltrue", len(short), "tb shorter than ref:", len(tbshort))

# ------------------------------------------------------------------ periodic eval files
per_ev = {}
for a in ARMS:
    p = os.path.join(EXP, f"run_{a}", "eval_periodic.jsonl")
    rows = read_jsonl(p)
    dis = sum(verify(r["prompt"], r["completion"])[0] != bool(r["ok"]) for r in rows)
    per_ev[a] = {"rows": len(rows), "disagree": dis}
OUT["periodic_reverify"] = per_ev
say("periodic", per_ev)

# ------------------------------------------------------------------ metrics coverage and curves
REQ = ["reward", "reward_std", "kl", "entropy", "reward/unparseable_frac", "reward/invalid_frac",
       "completions/mean_length", "completions/max_length", "completions/clipped_ratio", "grad_norm",
       "learning_rate", "rollout/completions_per_sec", "utc", "step"]
REQ_EI = ["reward", "reward/unparseable_frac", "reward/invalid_frac", "ei/prompts_solved_frac",
          "ei/kept_this_round", "ei/train_records", "sft/loss_last", "rollout/total_completions",
          "rollout/completions_per_sec", "utc", "step"]
mc = {}
for a in ARMS:
    rows = read_jsonl(os.path.join(EXP, f"run_{a}", "metrics.jsonl"))
    train_rows = [r for r in rows if "reward" in r]
    ev_rows = [r for r in rows if any(k.startswith("eval/pass_rate_len") for k in r)]
    req = REQ if a in GRPO else REQ_EI
    missing = {k: sum(k not in r for r in train_rows) for k in req}
    o = {"lines": len(rows), "train_lines": len(train_rows), "eval_lines": len(ev_rows),
         "eval_steps": [r["step"] for r in ev_rows],
         "missing_counts": {k: v for k, v in missing.items() if v},
         "all_lines_have_utc_step": all("utc" in r and "step" in r for r in rows),
         "first_utc": min(r["utc"] for r in rows), "last_utc": max(r["utc"] for r in rows)}
    if a in GRPO:
        kl = [r["kl"] for r in train_rows]
        ent = [r["entropy"] for r in train_rows]
        rw = [r["reward"] for r in train_rows]
        o.update({"kl_max": r4(max(kl)), "kl_last": r4(kl[-1]), "entropy_first": r4(ent[0]), "entropy_last": r4(ent[-1]),
                  "entropy_min": r4(min(ent)), "reward_first5": r4(mean(rw[:5])), "reward_last5": r4(mean(rw[-5:])),
                  "frac_zero_std_mean": r4(mean([r["frac_reward_zero_std"] for r in train_rows])),
                  "mean_len_first": r4(train_rows[0]["completions/mean_length"]),
                  "mean_len_last": r4(train_rows[-1]["completions/mean_length"]),
                  "clipped_max": max(r["completions/clipped_ratio"] for r in train_rows),
                  "grad_norm_max": r4(max(r["grad_norm"] for r in train_rows)),
                  "lr_values": sorted({r["learning_rate"] for r in train_rows})[:3] + ["..."]})
    else:
        o["reward_by_round"] = [r4(r["reward"]) for r in train_rows]
        o["kept"] = [r["ei/kept_this_round"] for r in train_rows]
        o["total_completions"] = [r["rollout/total_completions"] for r in train_rows]
        o["sft_loss_last"] = [r4(r["sft/loss_last"]) for r in train_rows]
    o["periodic_len7"] = [(r["step"], r.get("eval/solved_len7"), r.get("eval/n_len7")) for r in ev_rows]
    o["periodic_len8"] = [(r["step"], r.get("eval/solved_len8"), r.get("eval/n_len8")) for r in ev_rows]
    mc[a] = o
    say("metrics", a, json.dumps(o))
OUT["metrics"] = mc

# ------------------------------------------------------------------ 6. selection rule
sel = {}
for a in GRPO:
    drop = g[a]["avg_drop_vs_base"]
    kl_final = mc[a]["kl_last"]
    excl = []
    if drop > 0.02:
        excl.append(f"greedy drop {drop}")
    if kl_final > 1.0:
        excl.append(f"final KL {kl_final}")
    sel[a] = {"mean_diff_L7_base_solved": pp[a]["7"]["mean_diff_base_solved"], "greedy_drop": drop,
              "kl_final": kl_final, "excluded": excl}
qual = [a for a in GRPO if not sel[a]["excluded"]]
sel["selected"] = max(qual, key=lambda a: sel[a]["mean_diff_L7_base_solved"]) if qual else "none qualifies"
OUT["selection"] = sel
say("\nSELECTION", json.dumps(sel))

json.dump(OUT, open(os.path.join(HERE, "recount.json"), "w"), indent=1)
open(os.path.join(HERE, "recount_stdout.txt"), "w").write("\n".join(LOG) + "\n")
say("verifications cached:", len(VCACHE))
