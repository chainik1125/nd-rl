"""P-join: small problems built around the rule joins the hard textbook proofs need (ideas.md A11).

Each template is a sequent over metavariables A–D with a family tag and the joins its minimal proof should
contain. Metavariables are filled with random small formulas over P Q R S (atoms, negated atoms, one binary
connective), the instance is proved by the fork's minimal-proof search, checked by the pinned verifier, and kept
only when its minimal proof actually contains one of the template's target joins (so a template whose join can
be bypassed by a shorter proof contributes nothing). Records are deduplicated by curry-aware renaming class and
audited against every evaluation set and the sealed 72 before they are written.

    cd code && uv run python pretrain_data/joins.py --per_template 600 --out ../data/pretrain_data/joins.jsonl.gz
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
from hard_problems import bigrams  # noqa: E402
from minlen import minlen  # noqa: E402
from patterns import parse, prune_lines  # noqa: E402

ATOMS = ["P", "Q", "R", "S"]
ANY = None  # any join of the family counts


def N(a):
    return ("not", a)


def AND(a, b):
    return ("and", a, b)


def OR(a, b):
    return ("or", a, b)


def IMP(a, b):
    return ("imp", a, b)


def IFF(a, b):
    return AND(IMP(a, b), IMP(b, a))


BOT = ("bot",)

# (family, premises, goal, target joins)  — metavariables are strings "A".."D"
TEMPLATES = [
    # ↔-elimination: ∧E feeding →E
    ("bicond_elim", [IFF("A", "B"), "A"], "B", {"→E←∧E"}),
    ("bicond_elim", [IFF("A", "B"), "B"], "A", {"→E←∧E"}),
    ("bicond_elim", [IFF("A", "B"), AND("A", "C")], "B", {"→E←∧E"}),
    ("bicond_elim", [AND("C", IFF("A", "B")), "A"], OR("B", "D"), {"→E←∧E"}),
    ("bicond_elim", [IFF("A", "B"), IFF("B", "C"), "A"], "C", {"→E←∧E"}),
    ("bicond_elim", [IFF("A", "B"), N("B")], N("A"), {"→E←∧E"}),
    ("bicond_elim", [IFF("A", "B"), IMP("B", "C"), "A"], "C", {"→E←∧E"}),
    # ∧E feeding →E without a full ↔
    ("ande_impe", [AND(IMP("A", "B"), "C"), "A"], "B", {"→E←∧E"}),
    ("ande_impe", [AND("C", IMP("A", "B")), AND("A", "D")], "B", {"→E←∧E"}),
    ("ande_impe", [AND(IMP("A", "B"), IMP("C", "D")), "A", "C"], AND("B", "D"), {"→E←∧E"}),
    ("ande_impe", [AND(IMP("A", "B"), IMP("B", "C")), "A"], "C", {"→E←∧E"}),
    ("ande_impe", [AND("A", IMP("A", "B"))], "B", {"→E←∧E"}),
    # nested ∧E
    ("nested_ande", [AND("A", AND("B", "C"))], "B", {"∧E←∧E"}),
    ("nested_ande", [AND(AND("A", "B"), "C")], "B", {"∧E←∧E"}),
    ("nested_ande", [AND("A", AND("B", "C"))], AND("C", "A"), {"∧E←∧E"}),
    ("nested_ande", [AND(AND("A", "B"), AND("C", "D"))], AND("B", "D"), {"∧E←∧E"}),
    ("nested_ande", [AND("A", AND(AND("B", "C"), "D"))], "C", {"∧E←∧E"}),
    ("nested_ande", [AND(AND("A", "B"), "C")], AND("A", AND("B", "C")), {"∧E←∧E"}),
    ("nested_ande", [AND("A", AND("B", IMP("B", "C")))], "C", {"∧E←∧E", "→E←∧E"}),
    # →E chains and curried premises
    ("impe_chain", [IMP("A", "B"), IMP("B", "C"), IMP("C", "D"), "A"], "D", {"→E←→E"}),
    ("impe_chain", [IMP("A", IMP("B", "C")), "A", "B"], "C", {"→E←→E"}),
    ("impe_chain", [IMP("A", IMP("B", "C")), IMP("A", "B"), "A"], "C", {"→E←→E"}),
    ("impe_chain", [IMP("A", "B"), IMP("B", "C"), "A"], AND("C", "B"), {"→E←→E"}),
    ("impe_chain", [IMP("A", "B"), IMP("B", "C"), IMP("C", "D")], IMP("A", "D"), {"→E←→E"}),
    ("impe_chain", [IMP("A", "B"), IMP("B", N("C")), IMP("D", "C"), "A"], N("D"), {"→E←→E", "¬E←→E"}),
    # negation joins: ¬I feeding ¬E / →E / ∨I, ¬E fed by →E
    ("negation", [N(OR("A", "B")), IMP(N("A"), "C")], "C", {"→E←¬I"}),
    ("negation", [N(OR("A", "B")), IMP(N("A"), "C"), N("C")], "D", {"→E←¬I", "¬E←→E"}),
    ("negation", [N(OR("A", "B"))], AND(N("A"), N("B")), ANY),
    ("negation", [N(AND("A", "B"))], OR(N("A"), N("B")), {"∨I←¬I"}),
    ("negation", [N(AND("A", "B")), "A"], N("B"), ANY),
    ("negation", [IMP("A", "B"), N("B")], N("A"), {"¬E←→E"}),
    ("negation", [IMP("A", "B"), IMP("A", N("B"))], N("A"), {"¬E←→E"}),
    ("negation", [IMP("A", "B"), N("B"), OR("A", "C")], "C", {"¬E←→E"}),
    ("negation", [N(OR("A", "B")), OR("A", "C")], "C", ANY),
    ("negation", [N(IMP("A", "B"))], AND("A", N("B")), ANY),
    ("negation", [N(OR("A", "B")), IMP("C", "A")], N("C"), {"¬E←→E"}),
    ("negation", [N(AND("A", "B")), IMP("C", "A"), IMP("C", "B")], N("C"), ANY),
    ("negation", [IMP("A", "B"), IMP("B", "C"), N("C")], N("A"), {"¬E←→E", "→E←→E"}),
    ("negation", [N(OR("A", "B")), IMP(N("A"), N("C")), IMP("D", "C")], N("D"), {"→E←¬I"}),
    # ↔-introduction: two →I boxes feeding ∧I
    ("bicond_intro", [IMP("A", "B"), IMP("B", "C"), IMP("C", "A")], IFF("A", "C"), {"∧I←→I"}),
    ("bicond_intro", [AND("A", "B")], IFF("A", "B"), {"∧I←→I"}),
    ("bicond_intro", [IFF("A", "B"), IFF("B", "C")], IFF("A", "C"), {"∧I←→I", "→E←∧E"}),
    ("bicond_intro", [IFF("A", "B")], IFF("B", "A"), ANY),
    ("bicond_intro", [IFF("A", "B")], IFF(N("A"), N("B")), {"∧I←→I"}),
    ("bicond_intro", [IMP("A", "B"), IMP("B", "A"), IMP("A", "C"), IMP("C", "A")], IFF("B", "C"), {"∧I←→I"}),
    ("bicond_intro", [N("A"), N("B")], IFF("A", "B"), {"∧I←→I"}),
    ("bicond_intro", [IFF("A", "B")], IFF(AND("A", "C"), AND("B", "C")), {"∧I←→I"}),
    ("bicond_intro", [IFF("A", "B")], IFF(OR("A", "C"), OR("B", "C")), {"∧I←→I"}),
    # nested ∨E
    ("nested_ore", [OR(OR("A", "B"), "C"), IMP("A", "D"), IMP("B", "D"), IMP("C", "D")], "D", {"∨E←∨E"}),
    ("nested_ore", [OR(OR("A", "B"), "C"), N("A"), N("B")], "C", {"∨E←∨E"}),
    ("nested_ore", [OR("A", OR("B", "C"))], OR(OR("A", "B"), "C"), {"∨E←∨E"}),
    ("nested_ore", [OR("A", OR("B", "C")), IMP("B", "A"), IMP("C", "A")], "A", {"∨E←∨E"}),
    ("nested_ore", [OR(OR("A", "B"), "C"), IMP("A", "B"), IMP("C", "B")], "B", {"∨E←∨E"}),
    ("nested_ore", [OR("A", "B"), OR("C", "D")], OR(AND("A", "C"), OR(AND("A", "D"), OR(AND("B", "C"), AND("B", "D")))), {"∨E←∨E"}),
    # structural: four premises, box depth ≥ 3, with no special join
    ("four_prem", ["A", "B", "C", "D"], AND(AND("A", "B"), AND("C", "D")), ANY),
    ("four_prem", [IMP("A", "B"), IMP("B", "C"), "A", "D"], AND("C", "D"), ANY),
    ("four_prem", ["A", "B", IMP("A", "C"), IMP("B", "D")], AND("C", "D"), ANY),
    ("four_prem", [IMP("A", "B"), IMP("C", "D"), OR("A", "C"), N("B")], "D", ANY),
    ("four_prem", ["A", "B", IMP(AND("A", "B"), "C"), IMP("C", "D")], "D", ANY),
    ("four_prem", [IMP("A", "B"), IMP("A", "C"), IMP(AND("B", "C"), "D"), "A"], "D", ANY),
    ("depth3", [], IMP("A", IMP("B", IMP("C", "A"))), ANY),
    ("depth3", [], IMP(IMP("A", "B"), IMP(IMP("B", "C"), IMP("A", "C"))), ANY),
    ("depth3", [IMP("A", IMP("B", "C"))], IMP("B", IMP("A", "C")), ANY),
    ("depth3", [], IMP(IMP("A", IMP("B", "C")), IMP(AND("A", "B"), "C")), ANY),
    ("depth3", [], IMP(IMP(AND("A", "B"), "C"), IMP("A", IMP("B", "C"))), ANY),
    ("depth3", [IMP("A", "B")], IMP(IMP("C", "A"), IMP("C", "B")), ANY),
    ("depth3", [], IMP(N(AND("A", "B")), IMP("A", N("B"))), ANY),
    ("depth3", [], IMP(IMP("A", "B"), IMP(N("B"), N("A"))), ANY),
    ("depth3", [], IMP(IMP(IMP("A", "B"), "A"), "A"), ANY),
    ("depth3", [], IMP(N(IMP("A", "B")), AND("A", N("B"))), ANY),
    ("depth3", [], IMP(N(AND("A", "B")), OR(N("A"), N("B"))), ANY),
]
assert all(j is ANY or j <= set(PC.JOINS) for _, _, _, j in TEMPLATES)


def fillers(rng, k, allow_compound=True):
    """k distinct small formulas over P Q R S (atoms first)"""
    atoms = ATOMS[:]
    rng.shuffle(atoms)
    out = []
    for i in range(k):
        a = atoms[i % 4]
        r = rng.random()
        if not allow_compound or r < 0.55:
            f = ("atom", a)
        elif r < 0.7:
            f = N(("atom", a))
        else:
            b = ("atom", atoms[(i + 1 + rng.randrange(3)) % 4])
            op = rng.choice(["and", "or", "imp"])
            f = (op, ("atom", a), b)
        out.append(f)
    return out


def subst(f, env):
    if isinstance(f, str):
        return env[f]
    if f[0] in ("atom", "bot"):
        return f
    if f[0] == "not":
        return ("not", subst(f[1], env))
    return (f[0], subst(f[1], env), subst(f[2], env))


def instantiate(template, rng, allow_compound):
    fam, prem, goal, joins = template
    mv = sorted({x for f in prem + [goal] for x in metavars(f)})
    env = dict(zip(mv, fillers(rng, len(mv), allow_compound)))
    import transcribe as T
    return T.make_prompt([subst(p, env) for p in prem], subst(goal, env))


def metavars(f):
    if isinstance(f, str):
        return {f}
    if f[0] in ("atom", "bot"):
        return set()
    if f[0] == "not":
        return metavars(f[1])
    return metavars(f[1]) | metavars(f[2])


def work(args):
    fam, prompt, joins, bound, secs = args
    u = minlen(prompt, bound=bound, time_limit=secs)
    if not u.get("proof") or u.get("min_lines_ub") is None:
        return fam, prompt, None, "no_proof" if not u.get("timeout") else "timeout"
    lines = parse(u["proof"])
    if lines is None:
        return fam, prompt, None, "unparsed"
    bg = bigrams(prune_lines(lines))
    present = {name for name, pairs in PC.JOINS.items() if any(bg.get(p, 0) for p in pairs)}
    if joins is not ANY and not (present & joins):
        return fam, prompt, None, "join_bypassed"
    return fam, prompt, u["proof"], sorted(present)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(PC.DATA, "joins.jsonl.gz"))
    ap.add_argument("--per_template", type=int, default=600)
    ap.add_argument("--bound", type=int, default=14)
    ap.add_argument("--secs", type=float, default=5.0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    # instances: the atom-only instance of each template first, then random fills, deduplicated by class
    seen = set()
    jobs = []
    for t in TEMPLATES:
        fam, prem, goal, joins = t
        cands = []
        for i in range(a.per_template * 3):
            cands.append(instantiate(t, rng, allow_compound=i > 0))
        kept = 0
        for p in cands:
            ks = PC.BR.keys(p)
            if ks & seen:
                continue
            seen |= ks
            jobs.append((fam, p, joins, a.bound, a.secs))
            kept += 1
            if kept >= a.per_template:
                break
    print("templates", len(TEMPLATES), "instances to prove", len(jobs), flush=True)
    t0 = time.time()
    rows, rep = [], collections.Counter()
    fam_rep = collections.defaultdict(collections.Counter)
    with ProcessPoolExecutor(a.workers) as ex:
        for fam, prompt, proof, info in ex.map(work, jobs, chunksize=16):
            if proof is None:
                rep[info] += 1
                fam_rep[fam][info] += 1
                continue
            rec, why = PC.make_record(prompt, proof, "joins", family=fam, joins=info)
            if rec is None:
                rep["verify_fail"] += 1
                fam_rep[fam]["verify_fail"] += 1
                continue
            rows.append(rec)
            rep["kept"] += 1
            fam_rep[fam]["kept"] += 1
    print(f"proved in {time.time() - t0:.0f}s: {dict(rep)}", flush=True)
    idx = PC.EvalIndex()
    rows, audit = idx.audit(rows)
    print("audit:", audit, flush=True)
    PC.write(rows, a.out)
    summary = dict(templates=len(TEMPLATES), instances=len(jobs), results=dict(rep), audit=audit,
                   by_family={f: dict(c) for f, c in fam_rep.items()},
                   by_lines=dict(sorted(collections.Counter(r["n_lines"] for r in rows).items())),
                   by_prem=dict(sorted(collections.Counter(r["n_prem"] for r in rows).items())))
    with open(a.out.replace(".jsonl.gz", ".stats.json"), "w") as f:
        json.dump(summary, f, indent=1, ensure_ascii=False)
    print(json.dumps(summary, indent=1, ensure_ascii=False))
    print("wrote", a.out, len(rows))


if __name__ == "__main__":
    main()
