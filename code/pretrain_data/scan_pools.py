"""Stage-1 pools built from the recent-literature scan (ideas.md § I), all at K12's size (155,000 records).

Sub-commands (the first three are slow and run detached; `assemble` is quick):

  distract   K12 rows with 1–2 unused *distractor premises* inserted at random positions (I10 Morishita, I17 Saparov).
             A distractor is a random formula over P Q R S that keeps the premises satisfiable, is no line of the
             proof, and does not shorten the minimal proof (checked with `minlen` at bound = minimal length + k - 1,
             on the distracted prompt). Rows come from `k12_minimal` so the minimal length is known.
  pairs      counterfactual premise pairs (I16 Vompa & Tammet): a K12 theorem with one *critical* premise replaced by
             a random formula such that the goal still follows, the minimal proof (`minlen`, bound 12) is at least as
             long, and its rule multiset differs. The sibling joins the pool next to the original.
  aux        long-but-easy auxiliary proofs (I18 Cai et al.): 13–30 lines of ∧I / ∧E / →E forward chains with 3–6
             premises, no boxes, every premise cited, no unused lines, no ∧I of two ∧E siblings, no ∧E of an ∧I line.
  subst      formula substitution (I11 Xia et al.): every atom of a K12 row is replaced with probability 0.7 by a
             random depth-1 formula over P Q R S; the proof is unchanged otherwise and re-verified.
  wrong      spurious-reward control (Charles, 2026-10-07; cf. Shao et al. 2025 "Spurious rewards"): K12 proofs with ONE
             local error (a wrong citation, a wrong rule label, or a changed formula on a derived or assumed line), kept
             only if the pinned verifier REJECTS the proof. Format-perfect, logically wrong. The pool is the control for
             "does Stage-1 teach correctness or only surface structure"; it is never a source of correct proofs.
  assemble   builds the pools below from K12, the join pool, the depth-3 side pool and the sub-command outputs,
             and appends a section to docs/pretrain-data/pools.md.

Pools (every one 155,000 records; records 0–1,023 are K12's first 1,024 rows in every pool, because the pinned
trainer holds those out as its val slice, so val_loss is comparable across pools):

  k12_dial_1e4 / 1e3 / 1e2 / 1e1   E8: K12 with a nested 10⁻⁴ … 10⁻¹ slice of one pattern K12 never shows:
                                   ↔-elimination (∧E on (a→b)∧(b→a), then →E), from the join pool's bicond_elim family.
  k12_tail10 / k12_tail30          K12 with a 10 % / 30 % slice of depth-3+ records (the size ladder's side pool).
  k12_distract10 / k12_distract30  K12 with 10 % / 30 % of rows given distractor premises (original proofs).
  k12_minimal_distract10           k12_minimal with 10 % of rows given distractor premises (minimal proofs).
  k12_subst30 / k12_subst100       K12 with 30 % / 100 % of rows formula-substituted at rate 0.7.
  k12_aux10                        K12 with a 10 % slice of long easy auxiliary proofs.
  k12_pairs1                       K12 with a 1 % slice of counterfactual siblings (originals stay in the pool).
  k12_wrong10 / k12_wrong100       K12 with 10 % / 100 % of its proofs replaced by a rejected one-error version.

    cd code && nohup uv run python pretrain_data/scan_pools.py distract --n 50000 --workers 4 > ../data/pretrain_data/scan/distract.log 2>&1 &
    cd code && nohup uv run python pretrain_data/scan_pools.py pairs --n 2200 --workers 4 > ../data/pretrain_data/scan/pairs.log 2>&1 &
    cd code && uv run python pretrain_data/scan_pools.py aux --n 17000
    cd code && uv run python pretrain_data/scan_pools.py subst
    cd code && uv run python pretrain_data/scan_pools.py wrong
    cd code && uv run python pretrain_data/scan_pools.py assemble

Every record is checked by the pinned verifier (`strict_verify.safe_verify`) and every new theorem is audited against the
evaluation sets and the sealed 72 by renaming class before it enters a pool. Nothing here trains anything.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pool_common as PC  # noqa: E402
import logic as LG  # noqa: E402
from minlen import minlen  # noqa: E402
from patterns import fstr, lines_to_text, parse  # noqa: E402
from strict_verify import safe_verify  # noqa: E402

SCAN = os.path.join(PC.DATA, "scan")
OUT_DIR = os.path.join(PC.DATA, "pools", "scan")
DOC = os.path.join(PC.ROOT, "docs", "pretrain-data", "pools.md")
ATOMS = ["P", "Q", "R", "S"]
VAL_N = 1024          # the pinned trainer's held-out head
MAX_TOKENS = 900      # the pinned model's max_len is 1,024
MAX_LINES = 60        # the tokenizer's MAXN is 64
N_POOL = 155_000


# ---------------------------------------------------------------- small helpers

def rand_formula(rng, depth, atoms=ATOMS):
    if depth == 0 or (depth < 2 and rng.random() < 0.35):
        return ("atom", rng.choice(atoms))
    op = rng.choice(["not", "and", "or", "imp", "imp"])
    if op == "not":
        return ("not", rand_formula(rng, depth - 1, atoms))
    return (op, rand_formula(rng, depth - 1, atoms), rand_formula(rng, depth - 1, atoms))


def prompt_text(prem, goal):
    p = " , ".join(fstr(x) for x in prem)
    return f"THM {p + ' ' if p else ''}SEQ {fstr(goal)} PRF"


def n_tokens(rec):
    return len(rec["prompt"].split()) + len(rec["proof"].split())


def rule_multiset(proof):
    return tuple(sorted(ln["rule"] for ln in parse(proof) if ln["rule"] not in ("PR", "AS")))


def insert_premises(lines, new_prem_order, prem_formulas):
    """lines of a proof whose PR lines are `prem_formulas` in order; returns renumbered proof text with the PR lines
    replaced by `new_prem_order` (a list of ('old', i) or ('new', formula)) followed by the non-PR lines."""
    pr = [ln for ln in lines if ln["rule"] == "PR"]
    assert [fstr(ln["formula"]) for ln in pr] == [fstr(f) for f in prem_formulas]
    out = []
    for j, (kind, x) in enumerate(new_prem_order):
        if kind == "old":
            out.append(pr[x])
        else:
            out.append(dict(idx=-(j + 1), depth=0, formula=x, rule="PR", refs=[]))
    out.extend(ln for ln in lines if ln["rule"] != "PR")
    return lines_to_text(out)


def load_minimal():
    rows = PC.read(os.path.join(PC.DATA, "k12_minimal.jsonl.gz"))
    return rows


def iff_elim_rate(rows):
    """per 1,000 records: proofs with an →E citing a line obtained by ∧E from an ↔-shaped line"""
    n = 0
    for r in rows:
        lines = parse(r["proof"])
        if not lines:
            continue
        by = {ln["idx"]: ln for ln in lines}
        src = {ln["idx"] for ln in lines if ln["rule"] in ("ANDE1", "ANDE2") and ln["refs"]
               and ln["refs"][0] in by and LG.is_iff(by[ln["refs"][0]]["formula"])}
        if any(ln["rule"] == "IMPE" and set(ln["refs"]) & src for ln in lines):
            n += 1
    return round(1000 * n / max(1, len(rows)), 2)


def unused_prem_share(rows):
    n = 0
    for r in rows:
        lines = parse(r["proof"])
        if not lines:
            continue
        cited = set()
        for ln in lines:
            cited.update(ln["refs"])
        if any(ln["rule"] == "PR" and ln["idx"] not in cited for ln in lines):
            n += 1
    return round(n / max(1, len(rows)), 4)


# ---------------------------------------------------------------- distract

def _work_distract(args):
    r, orig_proof, secs, seed = args
    rng = random.Random(seed)
    prem, goal = LG.parse_prompt(r["prompt"])
    lines_min = parse(r["proof"])
    lines_orig = parse(orig_proof)
    n_min = r["n_lines"]
    line_strs = {fstr(ln["formula"]) for ln in lines_orig} | {fstr(ln["formula"]) for ln in lines_min} | {fstr(goal)}
    k = 1 if rng.random() < 0.6 else 2
    for attempt in range(4):
        ds = []
        tries = 0
        while len(ds) < k and tries < 50:
            tries += 1
            d = rand_formula(rng, 2)
            s = fstr(d)
            if s in line_strs or any(fstr(x) == s for x in ds) or len(s.split()) > 9:
                continue
            if not r.get("contra_prem") and not LG.satisfiable(prem + ds + [d]):
                continue
            ds.append(d)
        if len(ds) < k:
            continue
        order = [("old", i) for i in range(len(prem))]
        for d in ds:
            order.insert(rng.randint(0, len(order)), ("new", d))
        new_prem = [prem[x] if kind == "old" else x for kind, x in order]
        prompt2 = prompt_text(new_prem, goal)
        proof_min = insert_premises(lines_min, order, prem)
        proof_orig = insert_premises(lines_orig, order, prem)
        rec_min, why = PC.make_record(prompt2, proof_min, "distract_min", n_distract=k, base_prompt=r["prompt"])
        rec_orig, why2 = PC.make_record(prompt2, proof_orig, "distract", n_distract=k, base_prompt=r["prompt"])
        if rec_min is None or rec_orig is None:
            return dict(status=f"reject:{(why or why2)[:30]}")
        # shortcut check: no proof of the distracted problem shorter than the minimal proof plus the k premise lines
        try:
            u = minlen(prompt2, bound=n_min + k - 1, time_limit=secs)
        except Exception as e:  # noqa: BLE001
            return dict(status=f"error:{type(e).__name__}")
        if u.get("timeout"):
            status = "unknown"
        elif u.get("min_lines_ub") is not None:
            continue  # the distractor opened a shortcut; try other distractors
        else:
            status = "minimal"
        return dict(status=status, rec_min=dict(rec_min, shortcut_check=status), rec_orig=dict(rec_orig, shortcut_check=status))
    return dict(status="shortcut")


def cmd_distract(a):
    rng = random.Random(a.seed)
    k12 = PC.load_k12()
    mn = load_minimal()
    assert all(x["prompt"] == y["prompt"] for x, y in zip(k12, mn))
    cands = [(i, m) for i, m in enumerate(mn) if m["minlen_status"] in ("shorter", "same", "none") and i >= VAL_N]
    rng.shuffle(cands)
    cands = cands[: a.n]
    print("candidates", len(cands), flush=True)
    os.makedirs(SCAN, exist_ok=True)
    out_min, out_orig = [], []
    stats = collections.Counter()
    t0 = time.time()
    with ProcessPoolExecutor(a.workers) as ex:
        it = ex.map(_work_distract, ((m, k12[i]["proof"], a.secs, a.seed * 1_000_003 + i) for i, m in cands), chunksize=16)
        for j, ((i, m), res) in enumerate(zip(cands, it)):
            stats[res["status"].split(":")[0]] += 1
            if "rec_min" in res:
                out_min.append(dict(res["rec_min"], k12_index=i))
                out_orig.append(dict(res["rec_orig"], k12_index=i))
            if (j + 1) % 2000 == 0:
                print(f"{j + 1} {time.time() - t0:.0f}s {dict(stats)}", flush=True)
    idx = PC.EvalIndex()
    out_min, rep_min = idx.audit(out_min)
    keep = {r["k12_index"] for r in out_min}
    out_orig = [r for r in out_orig if r["k12_index"] in keep]
    PC.write(out_min, os.path.join(SCAN, "distract_min.jsonl.gz"))
    PC.write(out_orig, os.path.join(SCAN, "distract_orig.jsonl.gz"))
    summary = dict(candidates=len(cands), status=dict(stats), audit=rep_min, kept=len(out_min), secs=round(time.time() - t0),
                   by_k=dict(collections.Counter(r["n_distract"] for r in out_min)),
                   by_check=dict(collections.Counter(r["shortcut_check"] for r in out_min)),
                   by_prem=dict(sorted(collections.Counter(r["n_prem"] for r in out_min).items())))
    json.dump(summary, open(os.path.join(SCAN, "distract.stats.json"), "w"), indent=1)
    print(json.dumps(summary, indent=1))


# ---------------------------------------------------------------- pairs

def _work_pairs(args):
    r, secs, seed = args
    rng = random.Random(seed)
    prem, goal = LG.parse_prompt(r["prompt"])
    if not prem:
        return dict(status="no_prem")
    n0 = r["n_lines"]
    rules0 = rule_multiset(r["proof"])
    critical = [i for i in range(len(prem)) if not LG.classically_valid(prem[:i] + prem[i + 1:], goal)]
    if not critical:
        return dict(status="no_critical")
    for attempt in range(a_tries):
        i = rng.choice(critical)
        rest = prem[:i] + prem[i + 1:]
        d = rand_formula(rng, 2)
        s = fstr(d)
        if s == fstr(prem[i]) or len(s.split()) > 9 or s == fstr(goal):
            continue
        new_prem = rest[:i] + [d] + rest[i:]
        if not LG.satisfiable(new_prem):
            continue
        if not LG.classically_valid(new_prem, goal):
            continue
        prompt2 = prompt_text(new_prem, goal)
        try:
            u = minlen(prompt2, bound=12, time_limit=secs)
        except Exception as e:  # noqa: BLE001
            return dict(status=f"error:{type(e).__name__}")
        if u.get("timeout") or not u.get("proof") or u.get("min_lines_ub") is None:
            continue
        if u["min_lines_ub"] < n0:
            continue
        if rule_multiset(u["proof"]) == rules0:
            continue
        rec, why = PC.make_record(prompt2, u["proof"], "pairs", pair_of=r["prompt"], swapped=i, orig_n_lines=n0)
        if rec is None:
            return dict(status=f"reject:{why[:30]}")
        return dict(status="ok", rec=rec)
    return dict(status="no_sibling")


a_tries = 6


def cmd_pairs(a):
    rng = random.Random(a.seed)
    mn = load_minimal()
    k12_keys = set()
    for r in mn:
        k12_keys |= PC.BR.keys(r["prompt"])
    cands = [(i, m) for i, m in enumerate(mn) if m["minlen_status"] in ("shorter", "same", "none") and m["n_prem"] >= 1
             and m["n_lines"] >= 4 and i >= VAL_N]
    rng.shuffle(cands)
    cands = cands[: a.n]
    print("candidates", len(cands), flush=True)
    os.makedirs(SCAN, exist_ok=True)
    out = []
    stats = collections.Counter()
    t0 = time.time()
    with ProcessPoolExecutor(a.workers) as ex:
        it = ex.map(_work_pairs, ((m, a.secs, a.seed * 7_919 + i) for i, m in cands), chunksize=8)
        for j, ((i, m), res) in enumerate(zip(cands, it)):
            stats[res["status"].split(":")[0]] += 1
            if "rec" in res:
                rec = res["rec"]
                if PC.BR.keys(rec["prompt"]) & k12_keys:
                    stats["in_k12"] += 1
                    continue
                out.append(dict(rec, k12_index=i))
            if (j + 1) % 200 == 0:
                print(f"{j + 1} {time.time() - t0:.0f}s {dict(stats)}", flush=True)
    out = PC.dedup(out)
    idx = PC.EvalIndex()
    out, rep = idx.audit(out)
    PC.write(out, os.path.join(SCAN, "pairs.jsonl.gz"))
    summary = dict(candidates=len(cands), status=dict(stats), audit=rep, kept=len(out), secs=round(time.time() - t0),
                   longer=sum(1 for r in out if r["n_lines"] > r["orig_n_lines"]),
                   by_len=dict(sorted(collections.Counter(r["n_lines"] for r in out).items())))
    json.dump(summary, open(os.path.join(SCAN, "pairs.stats.json"), "w"), indent=1)
    print(json.dumps(summary, indent=1))


# ---------------------------------------------------------------- aux (long, easy)

def gen_aux(rng, target):
    lines = []
    derived = {}
    origin = {}

    def add(f, rule, refs):
        s = fstr(f)
        if s in derived or len(s.split()) > 25:
            return None
        idx = len(lines) + 1
        lines.append(dict(idx=idx, depth=0, formula=f, rule=rule, refs=list(refs)))
        derived[s] = idx
        return idx

    for _ in range(rng.randint(1, 3)):
        add(rand_formula(rng, rng.choice([0, 1, 1, 2])), "PR", [])
    n_imp = 0
    steps = 0
    while len(lines) < target and steps < 300:
        steps += 1
        choice = rng.choices(["impe", "ande", "andi"], weights=[4, 3, 3])[0]
        if choice == "impe" and n_imp < 5:
            a = rng.choice(lines[-6:])
            b = rand_formula(rng, rng.choice([0, 1, 1, 2]))
            if fstr(b) in derived:
                continue
            imp = ("imp", a["formula"], b)
            pi = add(imp, "PR", [])
            if pi is None:
                continue
            n_imp += 1
            bi = add(b, "IMPE", [pi, a["idx"]])
            origin[bi] = ("impe",)
        elif choice == "ande":
            cands = [ln for ln in lines if ln["formula"][0] == "and" and origin.get(ln["idx"], ("",))[0] != "andi"]
            if not cands:
                continue
            src = rng.choice(cands)
            which = rng.choice([1, 2, 12])
            if which in (1, 12):
                i1 = add(src["formula"][1], "ANDE1", [src["idx"]])
                if i1:
                    origin[i1] = ("ande", src["idx"])
            if which in (2, 12):
                i2 = add(src["formula"][2], "ANDE2", [src["idx"]])
                if i2:
                    origin[i2] = ("ande", src["idx"])
        else:
            if len(lines) < 2:
                continue
            x, y = rng.sample(lines[-8:], 2)
            ox, oy = origin.get(x["idx"], ("",)), origin.get(y["idx"], ("",))
            if ox[0] == "ande" and oy[0] == "ande" and ox[1] == oy[1]:
                continue
            ai = add(("and", x["formula"], y["formula"]), "ANDI", [x["idx"], y["idx"]])
            if ai:
                origin[ai] = ("andi",)
    if not lines or lines[-1]["rule"] == "PR":
        return None
    # prune to the goal's ancestors, drop uncited premises, premises first
    by = {ln["idx"]: ln for ln in lines}
    keep = set()
    stack = [lines[-1]["idx"]]
    while stack:
        i = stack.pop()
        if i in keep:
            continue
        keep.add(i)
        stack.extend(by[i]["refs"])
    kept = [ln for ln in lines if ln["idx"] in keep]
    pr = [ln for ln in kept if ln["rule"] == "PR"]
    rest = [ln for ln in kept if ln["rule"] != "PR"]
    if not (13 <= len(pr) + len(rest) <= 30) or not (3 <= len(pr) <= 6):
        return None
    goal = lines[-1]["formula"]
    if any(fstr(goal) == fstr(p["formula"]) for p in pr):
        return None
    proof = lines_to_text(pr + rest)
    prompt = prompt_text([p["formula"] for p in pr], goal)
    return prompt, proof


def cmd_aux(a):
    rng = random.Random(a.seed)
    os.makedirs(SCAN, exist_ok=True)
    out = []
    stats = collections.Counter()
    seen = set()
    t0 = time.time()
    quota = collections.Counter()
    per_len = a.n // 18 + 1  # flat over 13–30 lines as far as the generator supplies
    while len(out) < a.n and stats["tries"] < a.n * 60:
        stats["tries"] += 1
        g = gen_aux(rng, rng.randint(16, 60))
        if g is None:
            stats["shape"] += 1
            continue
        prompt, proof = g
        n_lines = proof.count(" ; ") + 1
        if quota[n_lines] >= per_len:
            stats["quota_full"] += 1
            continue
        rec, why = PC.make_record(prompt, proof, "aux")
        if rec is None:
            stats["verify_fail"] += 1
            continue
        if n_tokens(rec) > MAX_TOKENS or rec["n_lines"] > MAX_LINES:
            stats["too_long"] += 1
            continue
        quota[rec["n_lines"]] += 1
        ks = PC.BR.keys(prompt)
        if ks & seen:
            stats["dup"] += 1
            continue
        seen |= ks
        out.append(rec)
        if len(out) % 2000 == 0:
            print(len(out), f"{time.time() - t0:.0f}s", dict(stats), flush=True)
    idx = PC.EvalIndex()
    out, rep = idx.audit(out)
    PC.write(out, os.path.join(SCAN, "aux.jsonl.gz"))
    summary = dict(status=dict(stats), audit=rep, kept=len(out), secs=round(time.time() - t0),
                   by_len=dict(sorted(collections.Counter(r["n_lines"] for r in out).items())),
                   by_prem=dict(sorted(collections.Counter(r["n_prem"] for r in out).items())),
                   mean_tokens=round(sum(n_tokens(r) for r in out) / max(1, len(out)), 1),
                   max_tokens=max(n_tokens(r) for r in out) if out else 0)
    json.dump(summary, open(os.path.join(SCAN, "aux.stats.json"), "w"), indent=1)
    print(json.dumps(summary, indent=1))


# ---------------------------------------------------------------- subst

def substitute(rec, rng, rate=0.7):
    present = [t for t in ATOMS if t in rec["prompt"].split()]
    mapping = {}
    for t in present:
        if rng.random() < rate:
            f = rand_formula(rng, 1)
            if f[0] == "atom":
                f = ("not", f)
            mapping[t] = fstr(f)
    if not mapping:
        return None, "no_change"
    p2 = PC.rename_text(rec["prompt"], mapping)
    q2 = PC.rename_text(rec["proof"], mapping)
    rec2, why = PC.make_record(p2, q2, "subst", subst=mapping, base_prompt=rec["prompt"])
    if rec2 is None:
        return None, f"reject:{why[:30]}"
    if n_tokens(rec2) > MAX_TOKENS:
        return None, "too_long"
    if rec2["n_lines"] != rec["n_lines"]:
        return None, "lines_changed"
    return rec2, "ok"


def cmd_subst(a):
    rng = random.Random(a.seed)
    k12 = PC.load_k12()
    os.makedirs(SCAN, exist_ok=True)
    out = []
    stats = collections.Counter()
    t0 = time.time()
    for i, r in enumerate(k12):
        if i < VAL_N:
            continue
        rec, st = substitute(r, rng, a.rate)
        stats[st.split(":")[0]] += 1
        if rec is not None:
            out.append(dict(rec, k12_index=i))
        if (i + 1) % 20000 == 0:
            print(i + 1, f"{time.time() - t0:.0f}s", dict(stats), flush=True)
    idx = PC.EvalIndex()
    out, rep = idx.audit(out)
    PC.write(out, os.path.join(SCAN, "subst.jsonl.gz"))
    summary = dict(status=dict(stats), audit=rep, kept=len(out), secs=round(time.time() - t0),
                   mean_tokens=round(sum(n_tokens(r) for r in out) / max(1, len(out)), 1),
                   k12_mean_tokens=round(sum(n_tokens(r) for r in k12) / len(k12), 1),
                   max_tokens=max(n_tokens(r) for r in out) if out else 0)
    json.dump(summary, open(os.path.join(SCAN, "subst.stats.json"), "w"), indent=1)
    print(json.dumps(summary, indent=1))


# ---------------------------------------------------------------- wrong (spurious-reward control)

SAME_ARITY = {1: ["ANDE1", "ANDE2", "ORI1", "ORI2", "DN", "R"], 2: ["ANDI", "IMPE", "NEGE", "IMPI", "NEGI"]}


def _mutate_formula(f, rng):
    """a different formula of the same shape: swap operands, flip an atom, or add/remove a negation"""
    atoms_here = sorted(LG.atoms(f))
    choice = rng.choice(["swap", "atom", "neg"])
    if choice == "swap" and f[0] in ("and", "or", "imp") and fstr(f[1]) != fstr(f[2]):
        return (f[0], f[2], f[1])
    if choice == "neg":
        return f[1] if f[0] == "not" else ("not", f)
    if atoms_here:
        old = rng.choice(atoms_here)
        new = rng.choice([x for x in ATOMS if x != old])

        def ren(g):
            if g[0] == "atom":
                return ("atom", new) if g[1] == old else g
            if g[0] == "bot":
                return g
            if g[0] == "not":
                return ("not", ren(g[1]))
            return (g[0], ren(g[1]), ren(g[2]))
        return ren(f)
    return ("not", f)


def corrupt(rec, rng):
    """-> (proof text with one local error that the verifier rejects, kind) or (None, reason)"""
    lines0 = parse(rec["proof"])
    if not lines0:
        return None, "parse"
    targets = [i for i, ln in enumerate(lines0) if ln["rule"] != "PR"]
    if not targets:
        return None, "no_target"
    for _ in range(12):
        lines = [dict(ln, refs=list(ln["refs"])) for ln in lines0]
        i = rng.choice(targets)
        ln = lines[i]
        kinds = ["formula"]
        if ln["refs"]:
            kinds.append("cite")
        if ln["rule"] in SAME_ARITY.get(len(ln["refs"]), []) and ln["rule"] != "AS":
            kinds.append("rule")
        kind = rng.choice(kinds)
        if kind == "cite":
            j = rng.randrange(len(ln["refs"]))
            others = [x["idx"] for x in lines[:i] if x["idx"] != ln["refs"][j]]
            if not others:
                continue
            ln["refs"][j] = rng.choice(others)
        elif kind == "rule":
            ln["rule"] = rng.choice([r for r in SAME_ARITY[len(ln["refs"])] if r != ln["rule"]])
        else:
            f2 = _mutate_formula(ln["formula"], rng)
            if fstr(f2) == fstr(ln["formula"]):
                continue
            ln["formula"] = f2
        proof = lines_to_text(lines)
        if proof == rec["proof"]:
            continue
        ok, reason, n = safe_verify(rec["prompt"], proof)
        if ok:
            continue  # the change happened to be valid; try another
        return proof, kind
    return None, "no_reject"


def cmd_wrong(a):
    rng = random.Random(a.seed)
    k12 = PC.load_k12()
    os.makedirs(SCAN, exist_ok=True)
    out = []
    stats = collections.Counter()
    t0 = time.time()
    for i, r in enumerate(k12):
        if i < VAL_N:
            continue
        proof, kind = corrupt(r, rng)
        if proof is None:
            stats[kind] += 1
            continue
        stats[kind] += 1
        rules = collections.Counter(ln["rule"] for ln in parse(proof) if ln["rule"] not in ("PR", "AS"))
        out.append(dict(r, proof=proof, rules=dict(sorted(rules.items())), pool="wrong", verified=False, corruption=kind,
                        orig_proof=r["proof"], k12_index=i))
        if (i + 1) % 20000 == 0:
            print(i + 1, f"{time.time() - t0:.0f}s", dict(stats), flush=True)
    # every record must be rejected by the pinned verifier
    assert all(not safe_verify(r["prompt"], r["proof"])[0] for r in rng.sample(out, min(3000, len(out))))
    PC.write(out, os.path.join(SCAN, "wrong.jsonl.gz"))
    summary = dict(status=dict(stats), kept=len(out), secs=round(time.time() - t0))
    json.dump(summary, open(os.path.join(SCAN, "wrong.stats.json"), "w"), indent=1)
    print(json.dumps(summary, indent=1))


# ---------------------------------------------------------------- assemble

def replace_positions(base, positions, new_rows, tag):
    out = list(base)
    for i, r in zip(positions, new_rows):
        out[i] = dict(r, slice=tag)
    return out


def nested_slice(src, need, rng, tag):
    """shuffled source rows, then shuffled verified atom permutations, then repeats; prefixes are nested"""
    rows = rng.sample(src, len(src))
    if len(rows) >= need:
        return rows[:need], dict(source=len(src), perms=0, repeats=0)
    perms = []
    for r in src:
        perms.extend(p for p in PC.permutations_of(r) if p["prompt"] != r["prompt"])
    rng.shuffle(perms)
    rows.extend(perms[: need - len(rows)])
    reps = 0
    while len(rows) < need:
        rows.append(dict(rng.choice(src), repeat=True))
        reps += 1
    return rows, dict(source=len(src), perms=min(len(perms), need - len(src)), repeats=reps)


def cmd_assemble(a):
    rng = random.Random(a.seed)
    k12 = PC.load_k12()
    N = len(k12)
    assert N == N_POOL
    mn = k12[:VAL_N] + [dict(r, pool="k12_minimal") for r in load_minimal()[VAL_N:]]
    free = list(range(VAL_N, N))
    pools, notes = {}, {}

    def positions(n):
        return rng.sample(free, n)

    # E8 fraction dial: ↔-elimination, nested slices
    joins = PC.read(os.path.join(PC.DATA, "joins.jsonl.gz"))
    bic = [r for r in joins if r.get("family") == "bicond_elim"]
    need_max = int(round(0.1 * N))
    dial_rows, note = nested_slice(bic, need_max, rng, "dial")
    dial_pos = positions(need_max)
    for name, frac in [("1e4", 1e-4), ("1e3", 1e-3), ("1e2", 1e-2), ("1e1", 1e-1)]:
        n = int(round(frac * N))
        pools[f"k12_dial_{name}"] = replace_positions(k12, dial_pos[:n], dial_rows[:n], "dial")
        notes[f"k12_dial_{name}"] = dict(slice=n, **note)
    # depth tail
    deep = PC.read(os.path.join(PC.DATA, "pools", "size", "gen_depth3.jsonl.gz"))
    by_len = collections.defaultdict(list)
    for r in deep:
        if r["n_lines"] >= 7:
            by_len[r["n_lines"]].append(r)
    for L in by_len:
        rng.shuffle(by_len[L])
    for frac in (0.10, 0.30):
        n = int(round(frac * N))
        per = n // len(by_len)
        rows = []
        for L in sorted(by_len):
            rows.extend(by_len[L][:per])
        rows.extend(by_len[max(by_len)][per: per + n - len(rows)])
        rng.shuffle(rows)
        name = f"k12_tail{int(frac * 100)}"
        pools[name] = replace_positions(k12, positions(n), rows, "tail")
        notes[name] = dict(source=len(deep), per_length=per, depth_min=3)
    # distractors: replace the same rows
    p_orig = os.path.join(SCAN, "distract_orig.jsonl.gz")
    if os.path.exists(p_orig):
        d_orig = PC.read(p_orig)
        d_min = PC.read(os.path.join(SCAN, "distract_min.jsonl.gz"))
        rng.shuffle(d_orig)
        by_idx = {r["k12_index"]: r for r in d_min}
        for frac in (0.10, 0.30):
            n = int(round(frac * N))
            rows = d_orig[:n]
            name = f"k12_distract{int(frac * 100)}"
            pools[name] = replace_positions(k12, [r["k12_index"] for r in rows], rows, "distract")
            notes[name] = dict(source=len(d_orig), used=len(rows))
        n = int(round(0.10 * N))
        rows = [by_idx[r["k12_index"]] for r in d_orig[:n]]
        pools["k12_minimal_distract10"] = replace_positions(mn, [r["k12_index"] for r in rows], rows, "distract")
        notes["k12_minimal_distract10"] = dict(source=len(d_min), used=len(rows))
    # substitution: replace the same rows
    p_sub = os.path.join(SCAN, "subst.jsonl.gz")
    if os.path.exists(p_sub):
        subs = PC.read(p_sub)
        rng.shuffle(subs)
        for frac in (0.30, 1.0):
            n = min(len(subs), int(round(frac * N)))
            rows = subs[:n]
            name = f"k12_subst{int(frac * 100)}"
            pools[name] = replace_positions(k12, [r["k12_index"] for r in rows], rows, "subst")
            notes[name] = dict(source=len(subs), used=n)
    # auxiliary long easy proofs
    p_aux = os.path.join(SCAN, "aux.jsonl.gz")
    if os.path.exists(p_aux):
        aux = PC.read(p_aux)
        n = int(round(0.10 * N))
        rows, note = nested_slice(aux, n, rng, "aux")
        pools["k12_aux10"] = replace_positions(k12, positions(n), rows, "aux")
        notes["k12_aux10"] = note
    # spurious-reward control: the same rows with a rejected one-error proof
    p_wrong = os.path.join(SCAN, "wrong.jsonl.gz")
    if os.path.exists(p_wrong):
        wrong = PC.read(p_wrong)
        rng.shuffle(wrong)
        for frac in (0.10, 1.0):
            n = min(len(wrong), int(round(frac * N)))
            rows = wrong[:n]
            name = f"k12_wrong{int(frac * 100)}"
            pools[name] = replace_positions(k12, [r["k12_index"] for r in rows], rows, "wrong")
            notes[name] = dict(source=len(wrong), used=n, verifier_accepts=round((N - n) / N, 4),
                               by_corruption=dict(collections.Counter(r["corruption"] for r in rows)))
    # counterfactual siblings: never replace the originals
    p_pairs = os.path.join(SCAN, "pairs.jsonl.gz")
    if os.path.exists(p_pairs):
        pairs = PC.read(p_pairs)
        n = int(round(0.01 * N))
        rows, note = nested_slice(pairs, n, rng, "pairs")
        protected = {r["k12_index"] for r in rows if "k12_index" in r}
        pos = rng.sample([i for i in free if i not in protected], n)
        pools["k12_pairs1"] = replace_positions(k12, pos, rows, "pairs")
        notes["k12_pairs1"] = note
    # write + stats + doc
    os.makedirs(OUT_DIR, exist_ok=True)
    pieces = PC.hard_pieces()
    st = []
    base = PC.stats(k12, "k12", pieces)
    base.update(iff_elim=iff_elim_rate(k12), unused_prem=unused_prem_share(k12), mean_tokens=round(sum(n_tokens(r) for r in k12) / N, 1),
                max_tokens=max(n_tokens(r) for r in k12))
    st.append(base)
    for name, rows in pools.items():
        print("stats", name, flush=True)
        assert len(rows) == N and all(rows[i] is k12[i] or rows[i]["prompt"] == k12[i]["prompt"] for i in range(VAL_N)), name
        s = PC.stats(rows, name, pieces)
        s.update(iff_elim=iff_elim_rate(rows), unused_prem=unused_prem_share(rows), mean_tokens=round(sum(n_tokens(r) for r in rows) / N, 1),
                 max_tokens=max(n_tokens(r) for r in rows), note=notes.get(name, {}))
        st.append(s)
        if not a.no_write:
            PC.write(rows, os.path.join(OUT_DIR, f"{name}.jsonl.gz"))
    json.dump(dict(pools=st, notes=notes), open(os.path.join(OUT_DIR, "stats.json"), "w"), indent=1, ensure_ascii=False)
    write_doc(st)
    print("wrote", OUT_DIR)


def write_doc(st):
    head = "## Scan pools (ideas.md § I): the literature-scan ideas at K12's size"
    W = [f"\n{head}\n",
         "Built by `scan_pools.py`. Every pool has 155,000 records; records 0–1,023 are K12's first 1,024 rows in every pool "
         "(the pinned trainer holds them out as its val slice, so val_loss is comparable across pools). "
         "`↔-elim /1k` = records whose proof applies →E to a conjunct taken by ∧E from an ↔-shaped line (the E8 dial pattern; K12: 0). "
         "`unused prem` = share of records with a premise no line cites. Tokens = prompt + proof symbols (model max 1,024). "
         "The `k12_wrong*` pools are the spurious-reward control: their replaced proofs each carry one local error and are REJECTED by the "
         "pinned verifier (share accepted is in the composition notes); they are a control, never a source of correct proofs.\n",
         "| pool | n | mean lines | prem ≥4 | depth ≥3 | ↔ rows | ↔-elim /1k | unused prem | mean tokens | max tokens | ∧E share | ∨E share | hard pieces | unknown pieces |",
         "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for s in st:
        W.append(f"| {s['label']} | {s['n']:,} | {s['mean_lines']} | {s['prem_ge4']:.4f} | {s['depth_ge3']:.4f} | {s['bicond']:.4f} | {s['iff_elim']} | "
                 f"{s['unused_prem']:.4f} | {s['mean_tokens']} | {s['max_tokens']} | {s['ande_share']:.3f} | {s['ore_share']:.3f} | "
                 f"{s.get('hard_pieces_covered', '')} | {s.get('unknown_pieces_covered', '')} |")
    W += ["", "Join rates per 1,000 citations:", ""]
    cols = list(PC.JOINS)
    W.append("| pool | " + " | ".join(cols) + " |")
    W.append("| --- | " + " | ".join("---" for _ in cols) + " |")
    for s in st:
        W.append(f"| {s['label']} | " + " | ".join(str(s["joins"][c]) for c in cols) + " |")
    W += ["", "Composition notes:", ""]
    for s in st:
        if s.get("note"):
            W.append(f"- `{s['label']}`: {json.dumps(s['note'])}")
    W.append("")
    text = open(DOC).read()
    if head in text:  # replace an earlier section of ours, leave the others
        i = text.index(head)
        j = text.find("\n## ", i + len(head))
        text = text[:i].rstrip("\n") + "\n" + (text[j:] if j > 0 else "")
    with open(DOC, "w") as f:
        f.write(text.rstrip("\n") + "\n" + "\n".join(W) + "\n")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("distract")
    p.add_argument("--n", type=int, default=50_000)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--secs", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=0)
    p = sub.add_parser("pairs")
    p.add_argument("--n", type=int, default=2_200)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--secs", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=0)
    p = sub.add_parser("aux")
    p.add_argument("--n", type=int, default=17_000)
    p.add_argument("--seed", type=int, default=0)
    p = sub.add_parser("subst")
    p.add_argument("--rate", type=float, default=0.7)
    p.add_argument("--seed", type=int, default=0)
    p = sub.add_parser("wrong")
    p.add_argument("--seed", type=int, default=0)
    p = sub.add_parser("assemble")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no_write", action="store_true")
    a = ap.parse_args()
    {"distract": cmd_distract, "pairs": cmd_pairs, "aux": cmd_aux, "subst": cmd_subst, "wrong": cmd_wrong, "assemble": cmd_assemble}[a.cmd](a)


if __name__ == "__main__":
    main()
