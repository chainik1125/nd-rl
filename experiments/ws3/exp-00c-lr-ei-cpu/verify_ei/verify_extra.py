"""Checks 7-9 for run_ei: SFT data audit, length-8 detail, EI vs lr3e-5 GRPO comparison.
Run from code/: uv run python ../experiments/ws3/exp-00c-lr-ei-cpu/verify_ei/verify_extra.py"""
import hashlib, os, sys
from collections import Counter, defaultdict
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from verify import (rd, norm, parts, body_lines, canon_proof, E, safe_verify,  # noqa
                    canonical_theorem, canon_formula)
OUT = []
def say(*a):
    s = " ".join(str(x) for x in a); print(s); OUT.append(s)

def skel(c):
    return " ".join(l["rule"] + "|" * l["depth"] for l in body_lines(c))

ev = rd(os.path.join(E, "eval_set.jsonl")); pool = rd(os.path.join(E, "pool.jsonl"))
split_ev = rd(os.path.join(E, "split/rl_targets_eval.jsonl.gz"))
Lt = {norm(r["prompt"]): int(r["L_true"]) for r in ev}
poolP = {norm(r["prompt"]) for r in pool}
poolL = {norm(r["prompt"]): int(r["length"]) for r in pool}
evC = {canonical_theorem(r["prompt"]) for r in ev}
sevC = {canonical_theorem(r["prompt"]) for r in split_ev}
sevP = {norm(r["prompt"]) for r in split_ev}
h = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]
say("baseline copy identical to run_lr3e-5 baseline:",
    h(os.path.join(E, "run_ei/baseline/dumps/long_pool.jsonl")) == h(os.path.join(E, "run_lr3e-5/baseline/dumps/long_pool.jsonl")))

# ---- 7 ----
say("\n== 7. SFT data audit ==")
sft_all = []
for k in (1, 2, 3):
    recs = rd(os.path.join(E, f"run_ei/sft_data/round{k}.jsonl"))
    sft_all += recs
    notpool = sum(norm(r["prompt"]) not in poolP for r in recs)
    inev = sum(norm(r["prompt"]) in Lt or norm(r["prompt"]) in sevP for r in recs)
    incan = sum(canonical_theorem(r["prompt"]) in evC or canonical_theorem(r["prompt"]) in sevC for r in recs)
    bad = sum(not safe_verify(r["prompt"], r["completion"])[0] for r in recs)
    per = Counter(norm(r["prompt"]) for r in recs)
    dup_mod_num = sum(len(v) - len({canon_proof(c) for c in v}) for v in
                      [[r["completion"] for r in recs if norm(r["prompt"]) == p] for p in per])
    lens = Counter(poolL[norm(r["prompt"])] for r in recs if norm(r["prompt"]) in poolL)
    say(f"round{k}: records {len(recs)} prompts {len(per)} max/prompt {max(per.values())} not-in-pool {notpool} "
        f"exact-in-eval/eval-split {inev} canonical-in-eval/eval-split {incan} fail-reverify {bad} "
        f"duplicates-modulo-renumbering {dup_mod_num} records-by-pool-length {dict(sorted(lens.items()))}")
# length-8 pool prompts and their SFT data
l8pool = [r["prompt"] for r in pool if int(r["length"]) == 8]
say("\n4 length-8 pool prompts and their SFT proofs:")
sft_by = defaultdict(set)
for r in sft_all:
    sft_by[norm(r["prompt"])].add(canon_proof(r["completion"]))
for p in l8pool:
    say(f"  POOL8 {p}\n    SFT proofs {len(sft_by[norm(p)])}; skeletons {Counter(skel(c) for c in sft_by[norm(p)])}")
sft_skel = Counter(skel(c) for s in sft_by.values() for c in s)
sft_skel_len = Counter(len(body_lines(c)) for s in sft_by.values() for c in s)
say(f"distinct SFT proofs {sum(len(s) for s in sft_by.values())}; by line count {dict(sorted(sft_skel_len.items()))}")

# ---- 8 ----
say("\n== 8. length-8 newly solved detail ==")
fz = rd(os.path.join(E, "run_ei/baseline/dumps/long_pool.jsonl"))
ei = rd(os.path.join(E, "run_ei/evaluate/dumps/long_pool.jsonl"))
gr = rd(os.path.join(E, "run_lr3e-5/evaluate/dumps/long_pool.jsonl"))
def acc(rows):
    d = defaultdict(list)
    for r in rows:
        if r["ok"]: d[norm(r["prompt"])].append(r["completion"])
    return d
A_f, A_e, A_g = acc(fz), acc(ei), acc(gr)
new8 = sorted(p for p in Lt if Lt[p] == 8 and p in A_e and p not in A_f)
say(f"length-8 solved by EI: {sum(1 for p in Lt if Lt[p]==8 and p in A_e)}; newly vs frozen {len(new8)}")
l8skel = Counter()
short = 0
for p in new8:
    c = A_e[p][0]; n = len(body_lines(c))
    short += n < 8
    for cc in A_e[p]: l8skel[skel(cc)] += 1
    say(f"  [{len(A_e[p])}/32, {n} lines vs L_true 8] {p}\n      {canon_proof(c)}")
say(f"length-8 accepted EI proofs shorter than 8: {short}; all length-8 accepted skeletons ({sum(l8skel.values())} proofs, {len(l8skel)} distinct):")
for s, n in l8skel.most_common():
    # template-extension test: dropping one line from the skeleton gives a 7-line SFT skeleton?
    toks = s.split()
    ext = any(" ".join(toks[:i] + toks[i+1:]) in sft_skel for i in range(len(toks)))
    say(f"   {n:4d}  in-SFT-skeletons={s in sft_skel}  one-line-deletion-matches-SFT-skeleton={ext}  {s}")
# does the proof body (formulas) of any length-8 eval proof appear in SFT data verbatim modulo numbering/renaming?
def canon_body(c):
    b = canon_proof(c)
    return canon_formula(b)
sft_bodies = {canon_body(c) for s in sft_by.values() for c in s}
say(f"length-8 EI accepted proofs whose renamed body appears verbatim in SFT data: "
    f"{sum(canon_body(c) in sft_bodies for p in new8 for c in A_e[p])}")
all_ei_bodies = [canon_body(c) for p in A_e for c in A_e[p]]
say(f"all EI accepted eval proofs whose renamed body appears verbatim in SFT data: "
    f"{sum(b in sft_bodies for b in all_ei_bodies)}/{len(all_ei_bodies)}")

# ---- 9 ----
say("\n== 9. EI vs GRPO lr3e-5 ==")
for L in (7, 8):
    E_s = {p for p in A_e if Lt[p] == L}; G_s = {p for p in A_g if Lt[p] == L}; F_s = {p for p in A_f if Lt[p] == L}
    say(f"L{L}: EI {len(E_s)} GRPO {len(G_s)} both {len(E_s & G_s)} EI-only {len(E_s - G_s)} GRPO-only {len(G_s - E_s)} "
        f"frozen {len(F_s)} frozen-solved-but-EI-not {len(F_s - E_s)}")
for name, A in (("EI", A_e), ("GRPO", A_g)):
    d = Counter(len({canon_proof(c) for c in v}) for v in A.values())
    tot = sum(k * v for k, v in d.items()) / sum(d.values())
    say(f"{name}: distinct accepted proofs per solved problem {dict(sorted(d.items()))} mean {tot:.2f}")
open(os.path.join(HERE, "verify_extra_out.txt"), "w").write("\n".join(OUT) + "\n")
