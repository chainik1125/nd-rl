"""Independent skeptical verification of exp-00c (RUN env var, default run_ei) (read-only, no model, no training).

Run from code/:  uv run python ../experiments/ws3/exp-00c-lr-ei-cpu/verify_ei/verify.py
Writes verify_out.json and verify_out.txt next to this file.
Only imports: strict_verify.safe_verify (checker of record), nd_data.canonical_theorem (key),
nd_verify.parse_proof_tokens / verify / parse_formula (to re-check prefixes and renumbered proofs).
"""
import gzip
import itertools
import json
import os
import re
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
E = os.path.dirname(HERE)
ROOT = os.path.abspath(os.path.join(E, "../../.."))
CODE = os.path.join(ROOT, "code")
sys.path.insert(0, os.path.join(CODE, "ws3"))
import ndpath  # noqa: E402,F401
sys.path.insert(0, os.path.join(CODE, "experiments/current/r1_cot/vendor"))
sys.path.insert(0, os.path.join(CODE, "ws3/fork_pin"))
from strict_verify import safe_verify  # noqa: E402
from nd_data import canonical_theorem  # noqa: E402
from nd_verify import parse_proof_tokens  # noqa: E402
from nd_verify.verify import verify as raw_verify, parse_formula  # noqa: E402

OUT = []
RUN = os.environ.get("RUN", "run_ei")


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    OUT.append(s)


def rd(path):
    if not os.path.exists(path) and os.path.exists(path + ".gz"):
        path += ".gz"
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as f:
        return [json.loads(l) for l in f if l.strip()]


def norm(s):
    return " ".join(s.split())


def parts(prompt):
    p = norm(prompt)
    assert p.startswith("THM ") and p.endswith(" PRF"), p
    body = p[4:-4]
    pre, con = body.split("SEQ")
    prem = [norm(x) for x in pre.split(",") if x.strip()]
    return prem, norm(con)


PERMS = [str.maketrans("PQRS", "".join(p)) for p in itertools.permutations("PQRS")]


def canon_formula(f):
    return min(f.translate(t) for t in PERMS)


def parse_prompt_formulas(prompt):
    prem, con = parts(prompt)
    P = [parse_formula(x.split(), 0)[0] for x in prem]
    C = parse_formula(con.split(), 0)[0]
    return P, C


def body_lines(completion):
    """Parse the proof up to the first QED; None if unparsable."""
    toks = completion.split()
    try:
        return parse_proof_tokens(toks)
    except Exception:
        return None


def ftext(f):
    if f[0] == "atom":
        return f[1]
    if f == ("bot",):
        return "F"
    if f[0] == "not":
        return f"( ~ {ftext(f[1])} )"
    op = {"and": "&", "or": "v", "imp": ">"}[f[0]]
    return f"( {ftext(f[1])} {op} {ftext(f[2])} )"


def renumber(completion, start=1):
    toks = completion.split()
    ls = parse_proof_tokens(toks)
    off = ls[0]["idx"] - start
    out = []
    for t in toks:
        if re.fullmatch(r"N\d+", t):
            out.append(f"N{int(t[1:]) - off}")
        else:
            out.append(t)
        if t == "QED":
            break
    return " ".join(out)


def canon_proof(completion):
    """Proof text renumbered from N1 (distinctness modulo start index)."""
    try:
        return renumber(completion, 1)
    except Exception:
        return completion


def valid_prefix(prompt, completion):
    """Largest k such that the first k lines, ended at a depth-0 non-AS line, verify as a proof
    of that line's formula. Independent 'closest attempt' measure."""
    ls = body_lines(completion)
    if not ls:
        return 0
    P, _ = parse_prompt_formulas(prompt)
    toks = completion.split()
    # locate token boundaries of each line (';' ends a line)
    ends = [i for i, t in enumerate(toks) if t == ";"]
    best = 0
    for k in range(1, min(len(ls), len(ends)) + 1):
        ln = ls[k - 1]
        if ln["depth"] != 0 or ln["rule"] == "AS":
            continue
        pref = toks[: ends[k - 1] + 1] + ["QED"]
        try:
            ok, _ = raw_verify(P, ln["formula"], pref)
        except Exception:
            ok = False
        if ok:
            best = k
    return best


def used_lines(ls):
    """Set of line idxs reachable from the final line through refs (boxes: all lines in them)."""
    by = {l["idx"]: l for l in ls}
    seen = set()
    stack = [ls[-1]["idx"]]
    while stack:
        i = stack.pop()
        if i in seen or i not in by:
            continue
        seen.add(i)
        l = by[i]
        refs = l["refs"]
        if l["rule"] in ("IMPI", "NEGI"):
            s, e = refs
            stack += list(range(s, e + 1))
        elif l["rule"] == "ORE":
            j, s1, e1, s2, e2 = refs
            stack += [j] + list(range(s1, e1 + 1)) + list(range(s2, e2 + 1))
        else:
            stack += refs
    return seen


def main():
    ev = rd(os.path.join(E, "eval_set.jsonl"))
    pool = rd(os.path.join(E, "pool.jsonl"))
    prompts_used = rd(os.path.join(E, RUN + "/prompts.jsonl"))
    comps = rd(os.path.join(E, RUN + "/completions.jsonl"))
    split_tr = rd(os.path.join(E, "split/rl_targets_train.jsonl.gz"))
    split_ev = rd(os.path.join(E, "split/rl_targets_eval.jsonl.gz"))
    frozen = rd(os.path.join(E, RUN + "/baseline/dumps/long_pool.jsonl"))
    trained = rd(os.path.join(E, RUN + "/evaluate/dumps/long_pool.jsonl"))
    raw_rl = rd(os.path.join(CODE, "ws3/fork_pin/data/rl_targets.jsonl"))
    tb72 = rd(os.path.join(CODE, "ws3/textbook72/textbook_dev.jsonl")) + rd(
        os.path.join(CODE, "ws3/textbook72/textbook_train.jsonl"))
    say(f"sizes: eval_set {len(ev)} pool {len(pool)} prompts_used {len(prompts_used)} comps {len(comps)} "
        f"split_tr {len(split_tr)} split_ev {len(split_ev)} raw_rl {len(raw_rl)} tb72 {len(tb72)} "
        f"frozen rows {len(frozen)} trained rows {len(trained)}")
    Ltrue = {norm(r["prompt"]): int(r["L_true"]) for r in ev}
    res = {}

    # ---------- check 1: re-verify every sample ----------
    say("\n== 1. re-verify every dump sample with strict_verify.safe_verify ==")
    for name, rows in (("frozen", frozen), ("trained", trained)):
        dis = []
        for r in rows:
            ok, reason, _ = safe_verify(r["prompt"], r["completion"])
            if bool(ok) != bool(r["ok"]):
                dis.append((r["prompt"], r["completion"], r["ok"], ok, reason))
        per = defaultdict(list)
        for r in rows:
            per[norm(r["prompt"])].append(bool(r["ok"]))
        cnt = Counter(len(v) for v in per.values())
        notin = [p for p in per if p not in Ltrue]
        stat = {}
        for L in (7, 8):
            ps = [p for p in per if Ltrue.get(p) == L]
            stat[L] = dict(n=len(ps), avg=sum(sum(per[p]) / len(per[p]) for p in ps) / max(1, len(ps)),
                           solved=sum(any(per[p]) for p in ps))
        say(f"{name}: rows {len(rows)} disagreements {len(dis)} prompts {len(per)} samples/prompt {dict(cnt)} "
            f"prompts not in eval_set {len(notin)} missing eval prompts {len(set(Ltrue) - set(per))}")
        say(f"  recomputed: L7 avg {stat[7]['avg']:.4f} solved {stat[7]['solved']}/{stat[7]['n']}; "
            f"L8 avg {stat[8]['avg']:.4f} solved {stat[8]['solved']}/{stat[8]['n']}")
        say(f"  accepted rows {sum(r['ok'] for r in rows)}; temps {Counter(r.get('temperature') for r in rows)}")
        for d in dis[:5]:
            say("  DISAGREE", d)
        res[f"check1_{name}"] = dict(disagreements=len(dis), stats=stat)
    # training rewards
    dis_c = 0
    for r in comps:
        ok, _, _ = safe_verify(r["prompt"], r["completion"])
        if (1.0 if ok else 0.0) != float(r["reward"]) and not (ok and r["reward"] > 0):
            dis_c += 1
    say(f"training completions: {len(comps)} rows; reward vs re-verify disagreements {dis_c}; "
        f"reward values {Counter(r['reward'] for r in comps)}")
    res["check1_train_rewards_disagree"] = dis_c

    # ---------- check 2: triviality / loophole scan ----------
    say("\n== 2. triviality / loophole scan of accepted proofs ==")
    rulehist = {}
    for name, rows in (("frozen", frozen), ("trained", trained)):
        acc = [r for r in rows if r["ok"]]
        c = Counter()
        rh = Counter()
        short = []
        starts = Counter()
        tails = 0
        for r in acc:
            prem, con = parts(r["prompt"])
            ls = body_lines(r["completion"])
            if ls is None:
                c["unparsable_accepted"] += 1
                continue
            starts[ls[0]["idx"]] += 1
            if con in prem:
                c["conclusion_in_premises"] += 1
            if "F" in prem:
                c["F_in_premises"] += 1
            if ftext(ls[-1]["formula"]) != con:
                c["final_line_text_differs"] += 1
            if ls[-1]["depth"] != 0:
                c["final_line_in_box"] += 1
            u = used_lines(ls)
            pr_idx = [l["idx"] for l in ls if l["rule"] == "PR"]
            if pr_idx and not (set(pr_idx) & u):
                c["uses_no_premise"] += 1
            elif pr_idx and not set(pr_idx) <= u:
                c["leaves_some_premise_unused"] += 1
            if len(u) < len(ls):
                c["has_dead_lines"] += 1
            keys = [(l["depth"], l["formula"], l["rule"], tuple(l["refs"])) for l in ls]
            if len(set(keys)) < len(keys):
                c["repeated_identical_line"] += 1
            fs = [(l["depth"], l["formula"]) for l in ls]
            if len(set(fs)) < len(fs):
                c["repeated_formula_same_depth"] += 1
            for l in ls:
                rh[l["rule"]] += 1
            L = Ltrue[norm(r["prompt"])]
            if len(ls) < L:
                short.append((L, len(ls), r["prompt"], r["completion"]))
            after = r["completion"].split("QED", 1)
            if len(after) == 2 and after[1].strip():
                tails += 1
        tot = sum(rh.values())
        rulehist[name] = {k: (v, round(v / tot, 4)) for k, v in sorted(rh.items())}
        say(f"{name}: accepted {len(acc)}; flags {dict(c)}; text after QED {tails}")
        say(f"  start-index distribution (top) {starts.most_common(8)}")
        say(f"  rule histogram (count, frac): {rulehist[name]}")
        say(f"  proofs shorter than L_true: {len(short)}")
        for s in short:
            say("   SHORT", s)
        res[f"check2_{name}"] = dict(flags=dict(c), short=len(short), rulehist=rulehist[name],
                                     starts=starts.most_common(10), tails=tails)
    # accepted line-count distribution vs L_true
    for name, rows in (("frozen", frozen), ("trained", trained)):
        d = Counter()
        for r in rows:
            if r["ok"]:
                d[(Ltrue[norm(r["prompt"])], len(body_lines(r["completion"])))] += 1
        say(f"{name}: (L_true, accepted n_lines) counts {sorted(d.items())}")

    # ---------- check 3: leaks ----------
    say("\n== 3. leak checks ==")
    evP = {norm(r["prompt"]) for r in ev}
    sets = {
        "prompts.jsonl": {norm(r["prompt"]) for r in prompts_used},
        "pool.jsonl": {norm(r["prompt"]) for r in pool},
        "split_train": {norm(r["prompt"]) for r in split_tr},
        "completions prompts": {norm(r["prompt"]) for r in comps},
    }
    for k, s in sets.items():
        say(f"(a) exact prompt overlap eval_set ∩ {k}: {len(evP & s)} (|{k}|={len(s)})")
    say(f"    prompts.jsonl == pool.jsonl prompts: {sets['prompts.jsonl'] == sets['pool.jsonl']}; "
        f"completions prompts ⊆ prompts.jsonl: {sets['completions prompts'] <= sets['prompts.jsonl']}")
    evC = {canonical_theorem(r["prompt"]) for r in ev}
    trC = {canonical_theorem(r["prompt"]) for r in split_tr}
    poolC = {canonical_theorem(r["prompt"]) for r in pool}
    compC = {canonical_theorem(p) for p in sets["completions prompts"]}
    splevC = {canonical_theorem(r["prompt"]) for r in split_ev}
    tbC = {canonical_theorem(r["prompt"]) for r in tb72}
    say(f"(b) canonical overlap eval_set∩split_train {len(evC & trC)}, eval_set∩pool {len(evC & poolC)}, "
        f"eval_set∩completion prompts {len(evC & compC)}; split_eval∩split_train (whole halves) {len(splevC & trC)}")
    say(f"    textbook72 ∩ pool {len(tbC & poolC)}, ∩ completions prompts {len(tbC & compC)}, "
        f"∩ split_train {len(tbC & trC)}, ∩ eval_set {len(tbC & evC)}")
    # also check the length-6 train.jsonl pool prompts against the whole eval split canonically
    six = [r for r in pool if r.get("source") == "train"]
    say(f"    pool length-6 (train.jsonl) prompts {len(six)}: canonical ∩ whole eval split "
        f"{len({canonical_theorem(r['prompt']) for r in six} & splevC)}")
    # (c) conclusion-level structural near-duplicates
    def ccon(p):
        return canon_formula(parts(p)[1])
    trcon = defaultdict(list)
    for r in split_tr:
        trcon[ccon(r["prompt"])].append(r["prompt"])
    poolcon = defaultdict(list)
    for r in pool:
        poolcon[ccon(r["prompt"])].append(r["prompt"])
    hits = [(r["prompt"], trcon[ccon(r["prompt"])]) for r in ev if ccon(r["prompt"]) in trcon]
    hitsp = [(r["prompt"], poolcon[ccon(r["prompt"])]) for r in ev if ccon(r["prompt"]) in poolcon]
    say(f"(c) eval_set prompts whose renamed conclusion equals a train-split conclusion: {len(hits)}/200; "
        f"equals a pool (trained-on) conclusion: {len(hitsp)}/200")
    # conclusion-only coincidence rate in the eval split itself, for scale
    concl_ev_self = Counter(ccon(r["prompt"]) for r in split_ev)
    say(f"    for scale: distinct renamed conclusions in eval split {len(concl_ev_self)} over {len(split_ev)} records")
    for p, m in hitsp[:5]:
        say(f"    EVAL {p}\n      POOL {m[0]}")
    for p, m in hits[:5]:
        say(f"    EVAL {p}\n      TRAIN-SPLIT {m[0]}")
    # does the near-duplicate subset explain the gain?
    per_f, per_t = defaultdict(list), defaultdict(list)
    for r in frozen:
        per_f[norm(r["prompt"])].append(bool(r["ok"]))
    for r in trained:
        per_t[norm(r["prompt"])].append(bool(r["ok"]))
    hp = {norm(p) for p, _ in hitsp}
    for lab, S in (("conclusion-in-pool", [p for p in evP if p in hp]),
                   ("not", [p for p in evP if p not in hp])):
        S7 = [p for p in S if Ltrue[p] == 7]
        if S7:
            say(f"    L7 {lab}: n {len(S7)} frozen avg {sum(sum(per_f[p]) / 32 for p in S7) / len(S7):.3f} "
                f"trained avg {sum(sum(per_t[p]) / 32 for p in S7) / len(S7):.3f}")
    # (d)
    splevP = {norm(r["prompt"]) for r in split_ev}
    say(f"(d) eval_set prompts in split_eval file: {len(evP & splevP)}/{len(evP)}; "
        f"eval_set prompts in raw rl_targets: {len(evP & {norm(r['prompt']) for r in raw_rl})}; "
        f"split_tr+split_ev == raw rl_targets count: {len(split_tr) + len(split_ev) == len(raw_rl)}")
    res["check3"] = dict(exact={k: len(evP & s) for k, s in sets.items()},
                         canon_eval_train=len(evC & trC), canon_eval_pool=len(evC & poolC),
                         canon_halves=len(splevC & trC), tb72_pool=len(tbC & poolC),
                         concl_hits_train=len(hits), concl_hits_pool=len(hitsp))

    # ---------- check 4: newly solved ----------
    say("\n== 4. newly solved (frozen 0/32, trained >=1/32) ==")
    new = sorted(p for p in evP if not any(per_f[p]) and any(per_t[p]))
    lost = sorted(p for p in evP if any(per_f[p]) and not any(per_t[p]))
    say(f"newly solved {len(new)} (L7 {sum(Ltrue[p] == 7 for p in new)}, L8 {sum(Ltrue[p] == 8 for p in new)}); "
        f"lost {len(lost)}")
    tacc = defaultdict(list)
    for r in trained:
        if r["ok"]:
            tacc[norm(r["prompt"])].append(r["completion"])
    fall = defaultdict(list)
    for r in frozen:
        fall[norm(r["prompt"])].append(r)
    examples = []
    for p in new[:8]:
        fs = fall[p]
        best = max(fs, key=lambda r: valid_prefix(p, r["completion"]))
        vp = valid_prefix(p, best["completion"])
        examples.append(dict(prompt=p, L_true=Ltrue[p], trained_proof=tacc[p][0],
                             n_trained_ok=len(tacc[p]), frozen_best=best["completion"],
                             frozen_best_valid_prefix=vp, frozen_best_reason=best["reason"],
                             frozen_dump_n_valid_prefix=best.get("n_valid_prefix")))
        say(f"\nPROMPT (L_true {Ltrue[p]}, trained {len(tacc[p])}/32): {p}")
        say(f"  TRAINED: {tacc[p][0]}")
        say(f"  FROZEN closest (my valid prefix {vp}, dump n_valid_prefix {best.get('n_valid_prefix')}, "
            f"reason {best['reason']}): {best['completion']}")
    res["check4"] = dict(newly_solved=len(new), lost=len(lost), examples=examples)

    # ---------- check 5: diversity ----------
    say("\n== 5. diversity on newly solved ==")
    dcount = {p: len({canon_proof(c) for c in tacc[p]}) for p in new}
    say(f"distinct accepted proofs per newly-solved problem: {sorted(Counter(dcount.values()).items())}; "
        f"mean {sum(dcount.values()) / max(1, len(dcount)):.2f}")
    skel = Counter()
    for p in new:
        for c in tacc[p]:
            skel[tuple((l["rule"], l["depth"]) for l in body_lines(c))] += 1
    say(f"distinct (rule,depth) skeletons across all newly-solved accepted proofs: {len(skel)} "
        f"over {sum(skel.values())} proofs; top 5:")
    for s, n in skel.most_common(5):
        say(f"   {n}  {' '.join(r + ('|' * d) for r, d in s)}")
    schema = {norm(r["prompt"]): r.get("schema") for r in ev}
    say(f"schemas of newly solved: {Counter(schema[p] for p in new).most_common()}")
    say(f"schemas of eval_set: {Counter(schema[p] for p in evP).most_common()}")
    res["check5"] = dict(distinct=dict(Counter(dcount.values())), n_skeletons=len(skel))

    # ---------- check 6: misc ----------
    say("\n== 6. misc ==")
    # renumbering does not change verdicts
    flip = 0
    checked = 0
    for r in trained[:6400]:
        ls = body_lines(r["completion"])
        if not ls:
            continue
        try:
            rc = renumber(r["completion"], 1)
        except Exception:
            continue
        a = safe_verify(r["prompt"], r["completion"])[0]
        b = safe_verify(r["prompt"], rc)[0]
        checked += 1
        flip += a != b
    say(f"renumber-to-N1 verdict flips (trained dump): {flip}/{checked}")
    # dump prompt fields vs eval_set metadata
    bad_len = sum(int(r["length"]) != Ltrue[norm(r["prompt"])] for r in trained + frozen)
    say(f"dump rows whose length field != eval_set L_true: {bad_len}")
    # conclusion-in-premise or F-in-premise for eval set prompts themselves
    say(f"eval_set prompts with conclusion among premises: "
        f"{sum(parts(r['prompt'])[1] in parts(r['prompt'])[0] for r in ev)}; with F premise "
        f"{sum('F' in parts(r['prompt'])[0] for r in ev)}")
    # trained model copying: is any accepted trained proof identical (mod numbering) to a training rollout?
    comp_acc = defaultdict(set)
    for r in comps:
        if r["reward"] > 0:
            comp_acc[canonical_theorem(r["prompt"])].add(canon_proof(r["completion"]))
    say(f"accepted training rollouts: {sum(r['reward'] > 0 for r in comps)} over "
        f"{len(comp_acc)} distinct prompts")

    json.dump(res, open(os.path.join(HERE, "verify_out.json"), "w"), indent=1, default=str)
    open(os.path.join(HERE, "verify_out.txt"), "w").write("\n".join(OUT) + "\n")


if __name__ == "__main__":
    main()
