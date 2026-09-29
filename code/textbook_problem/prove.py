"""Reference proofs, shortest lengths and shape tags for transcribed textbook candidates.

For each row of candidates.jsonl:
  1. the fork's minlen search (bound 40 lines, --secs) for the shortest proof it can find;
  2. if that times out: conjunct splitting for a conjunction goal, else one longer search
     (--long_secs). Both give a verified upper bound;
  3. a restricted search (case splits only over premises, up to 4 lines past the shortest) to
     test whether a case split on a derived disjunction is needed;
  4. strict_verify on the proof that is kept.
Then the family, length bin and tags of docs/textbook-problem/plan.md section 2, and the named-law
match against the fork's textbook schemata.

    cd code && uv run python textbook_problem/prove.py \
        --inp ../data/textbook_problem/candidates/candidates.jsonl \
        --out ../data/textbook_problem/candidates/candidates_proved.jsonl --procs 8
"""

import argparse
import collections
import itertools
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [
    HERE,
    os.path.join(HERE, "../ws3/fork_pin"),
    os.path.join(HERE, "fork_tools"),
    os.path.join(HERE, "../experiments/current/r1_cot/vendor"),
]

import logic as L  # noqa: E402
from minlen import minlen  # noqa: E402
from nd_verify import verify_text  # noqa: E402
from patterns import classify, prune_lines  # noqa: E402
from patterns import parse as parse_proof  # noqa: E402
from patterns2 import classify2  # noqa: E402
from strict_verify import safe_verify  # noqa: E402
from textbook_pool import SCHEMATA  # noqa: E402


def N(a):
    return ("not", a)


def AND(a, b):
    return ("and", a, b)


def OR(a, b):
    return ("or", a, b)


def IMP(a, b):
    return ("imp", a, b)


# Classics the fork's schemata do not cover. Same signature as SCHEMATA.
EXTRA = {
    "identity": lambda A, B, C, D: ([A], A),
    "modus_ponens": lambda A, B, C, D: ([IMP(A, B), A], B),
    "commute_or": lambda A, B, C, D: ([OR(A, B)], OR(B, A)),
    "commute_and": lambda A, B, C, D: ([AND(A, B)], AND(B, A)),
    "weakening": lambda A, B, C, D: ([A], IMP(B, A)),
    "linearity": lambda A, B, C, D: ([], OR(IMP(A, B), IMP(B, A))),
    "non_contradiction": lambda A, B, C, D: ([], N(AND(A, N(A)))),
    "material_implication": lambda A, B, C, D: ([OR(N(A), B)], IMP(A, B)),
    "material_implication_conv": lambda A, B, C, D: ([IMP(A, B)], OR(N(A), B)),
    "imp_to_or": lambda A, B, C, D: ([IMP(N(A), B)], OR(A, B)),
    "and_intro_imp": lambda A, B, C, D: ([IMP(A, B), IMP(A, C)], IMP(A, AND(B, C))),
    "or_elim_imp": lambda A, B, C, D: ([IMP(A, C), IMP(B, C)], IMP(OR(A, B), C)),
    "destructive_dilemma": lambda A, B, C, D: (
        [IMP(A, C), IMP(B, D), OR(N(C), N(D))],
        OR(N(A), N(B)),
    ),
    "absurd_conditional": lambda A, B, C, D: ([IMP(A, B), IMP(A, N(B))], N(A)),
}
MV = [("mv", x) for x in "ABCD"]


def _match(p, f, sub):
    if p[0] == "mv":
        if p in sub:
            return sub if sub[p] == f else None
        return {**sub, p: f}
    if p[0] != f[0] or len(p) != len(f):
        return None
    if p[0] in ("atom", "bot"):
        return sub if p == f else None
    for pc, fc in zip(p[1:], f[1:], strict=True):
        sub = _match(pc, fc, sub)
        if sub is None:
            return None
    return sub


def _match_seq(sp, sg, prem, goal):
    if len(sp) != len(prem):
        return False
    for perm in itertools.permutations(prem):
        sub = {}
        for a, b in zip(sp, perm, strict=True):
            sub = _match(a, b, sub)
            if sub is None:
                break
        if sub is not None and _match(sg, goal, sub) is not None:
            return True
    return False


def named_laws(prem, goal):
    """-> sorted names; suffixes mark the theorem form, conjoined premises, or a biconditional."""
    hits = []
    for table in (SCHEMATA, EXTRA):
        for name, fn in table.items():
            sp, sg = fn(*MV)
            if _match_seq(sp, sg, prem, goal):
                hits.append(name)
            elif not prem and sp:
                curried = sg
                for p in reversed(sp):
                    curried = IMP(p, curried)
                conj = sp[0]
                for p in sp[1:]:
                    conj = AND(conj, p)
                if _match_seq([], curried, [], goal) or (
                    len(sp) > 1 and _match_seq([], IMP(conj, sg), [], goal)
                ):
                    hits.append(f"{name} (theorem form)")
            if len(prem) == 1 and len(sp) > 1 and prem[0][0] == "and":
                parts, stack = [], [prem[0]]
                while stack:
                    x = stack.pop()
                    if x[0] == "and":
                        stack += [x[2], x[1]]
                    else:
                        parts.append(x)
                if len(parts) == len(sp) and _match_seq(sp, sg, parts, goal):
                    hits.append(f"{name} (conjoined premises)")
    if not prem and L.is_iff(goal):
        x, y = goal[1][1], goal[1][2]
        for table in (SCHEMATA, EXTRA):
            for name, fn in table.items():
                sp, sg = fn(*MV)
                if len(sp) == 1 and (_match_seq(sp, sg, [x], y) or _match_seq(sp, sg, [y], x)):
                    hits.append(f"{name} (biconditional)")
    return sorted(set(hits))


def _identity_proof(prompt, prem, goal):
    lines = [f"N{i + 1} {L.to_tokens(p)} : PR ;" for i, p in enumerate(prem)]
    if prem[-1] != goal:
        lines.append(f"N{len(prem) + 1} {L.to_tokens(goal)} : R N{prem.index(goal) + 1} ;")
    text = " ".join(lines) + " QED"
    ok, _, n = verify_text(prompt + " " + text)
    return (n, text) if ok else (None, None)


def _shift(proof, n_prem, off):
    """Proof body -> its non-premise lines with every index > n_prem moved up by off."""
    out, lines = [], proof.replace(" QED", "").split(" ; ")
    for ln in lines[n_prem:]:
        toks = [
            f"N{int(t[1:]) + off}" if t[0] == "N" and t[1:].isdigit() and int(t[1:]) > n_prem else t
            for t in ln.strip(" ;").split()
        ]
        out.append(" ".join(toks) + " ;")
    return out


def split_search(prompt, prem, goal, bound, secs):
    """Prove each conjunct (recursively), then ANDI. Nothing is shared between the conjuncts, so
    the length is a looser upper bound. Verified."""
    if goal[0] != "and":
        u = minlen(prompt, bound, secs)
        return u["min_lines_ub"], u["proof"]
    head = "THM " + " , ".join(L.to_tokens(p) for p in prem) + (" " if prem else "")
    parts = []
    for g in (goal[1], goal[2]):
        n, proof = split_search(f"{head}SEQ {L.to_tokens(g)} PRF", prem, g, bound, secs)
        if proof is None:
            return None, None
        parts.append((g, proof))
    k = len(prem)
    lines = [f"N{i + 1} {L.to_tokens(p)} : PR ;" for i, p in enumerate(prem)]
    a = _shift(parts[0][1], k, 0) or [
        f"N{k + 1} {L.to_tokens(parts[0][0])} : R N{prem.index(parts[0][0]) + 1} ;"
    ]
    b = _shift(parts[1][1], k, len(a)) or [
        f"N{k + len(a) + 1} {L.to_tokens(parts[1][0])} : R N{prem.index(parts[1][0]) + 1} ;"
    ]
    end_a, end_b = k + len(a), k + len(a) + len(b)
    lines += a + b + [f"N{end_b + 1} {L.to_tokens(goal)} : ANDI N{end_a} N{end_b} ;"]
    text = " ".join(lines) + " QED"
    ok, _, n = verify_text(prompt + " " + text)
    return (n, text) if ok else (None, None)


def work(a):
    """One problem: shortest proof (with fallbacks), derived-case-split test, strict check."""
    prompt, bound, secs, long_secs = a
    prem, goal = L.parse_prompt(prompt)
    t0 = time.time()
    u = minlen(prompt, bound, secs)
    n, proof, method = u["min_lines_ub"], u["proof"], "whole search"
    if n is None and goal in prem:
        n, proof = _identity_proof(prompt, prem, goal)
        method = "identity"
    if n is None and u["timeout"]:
        if goal[0] == "and":
            n, proof = split_search(prompt, prem, goal, bound, secs)
            method = "conjunct split"
        else:
            v = minlen(prompt, bound, long_secs)
            n, proof, method = v["min_lines_ub"], v["proof"], f"whole search, {long_secs:.0f} s"
    derived = None
    if u["min_lines_ub"] is not None:
        rb = min(bound, u["min_lines_ub"] + 4)
        r = minlen(prompt, rb, secs, forbid=("ORE_DERIVED_LOOSE",))
        derived = {"restricted_len": r["min_lines_ub"], "timeout": r["timeout"], "bound": rb}
    strict = safe_verify(prompt, proof) if proof else (False, "no proof", 0)
    return {
        "n": n,
        "proof": proof,
        "method": method if n is not None else f"{method}: no proof",
        "first_timeout": u["timeout"],
        "derived": derived,
        "strict_ok": bool(strict[0]),
        "strict_reason": strict[1],
        "secs": round(time.time() - t0, 2),
    }


def proof_features(proof):
    lines = parse_proof(proof)
    pr = prune_lines(lines)
    rules = collections.Counter(ln["rule"] for ln in pr)
    c, c2 = classify(proof), classify2(proof)
    return {
        "rules": dict(sorted(rules.items())),
        "rule_set": sorted(r for r in rules if r not in ("PR", "AS")),
        "n_boxes": rules["AS"],
        "max_box_depth": c2["max_depth"],
        "patterns": sorted(
            k
            for k, v in {
                "reductio": c["reductio"],
                "derived_dn": c["derived_dn"],
                "derived_ore": c["derived_ore"],
                "depth3": c["depth3"],
                "depth4": c2["depth4"],
                "impe_chain": c2["impe_chain"],
                "ore_nesting": c2["ore_nesting"],
                "impi_ore": c2["impi_ore"],
                "ori_ore": c2["ori_ore"],
            }.items()
            if v
        ),
    }


BINS = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 15), (16, 10**9)]


def length_bin(n):
    for lo, hi in BINS:
        if lo <= n <= hi:
            return f"{lo}+" if hi == 10**9 else f"{lo}-{hi}"
    return None


def families(rec, goal):
    """plan.md section 2, in table order. `family` is the first match; `tags` all matches."""
    f = rec["proof_features"]
    rs = set(f["rule_set"]) if f else set()
    intu = not rec["classical_only"]
    derived = rec["requires_derived_ore"]
    checks = [
        ("direct", f is not None and f["n_boxes"] == 0),
        ("conditional", "IMPI" in rs and not rs & {"ORE", "NEGI", "DN"} and intu),
        ("case-split", "ORE" in rs and not derived),
        ("case-split-derived", derived),
        ("negation", "NEGI" in rs and "ORE" not in rs and intu),
        ("classical", not intu),
        ("refutation", goal == L.BOT or "BOTE" in rs),
        ("biconditional", L.is_iff(goal)),
        ("nested", f is not None and f["max_box_depth"] >= 3),
    ]
    tags = [name for name, ok in checks if ok]
    return (tags[0] if tags else None), tags


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bound", type=int, default=40)
    ap.add_argument("--secs", type=float, default=300)
    ap.add_argument("--long_secs", type=float, default=1800)
    ap.add_argument("--procs", type=int, default=8)
    args = ap.parse_args()

    with open(args.inp) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    t0 = time.time()
    jobs = [(r["prompt"], args.bound, args.secs, args.long_secs) for r in rows]
    with ProcessPoolExecutor(args.procs) as ex:
        found = []
        for i, res in enumerate(ex.map(work, jobs), 1):
            found.append(res)
            print(f"{i}/{len(rows)} {res['method']} n={res['n']} {res['secs']}s", flush=True)

    out = []
    for r, res in zip(rows, found, strict=True):
        prem, goal = L.parse_prompt(r["prompt"])
        r = dict(r)
        ok = res["n"] is not None and res["strict_ok"]
        r["reference_proof"] = res["proof"] if ok else None
        r["reference_lines"] = res["n"] if ok else None
        r["min_lines"] = res["n"] if ok else None
        r["min_lines_method"] = res["method"]
        r["search_secs"] = res["secs"]
        r["strict_verify"] = {"ok": res["strict_ok"], "reason": res["strict_reason"]}
        d = res["derived"]
        r["requires_derived_ore"] = bool(d and d["restricted_len"] is None and not d["timeout"])
        r["derived_ore_test"] = d
        r["proof_features"] = proof_features(res["proof"]) if ok else None
        r["length_bin"] = length_bin(res["n"]) if ok else None
        r["family"], r["tags"] = families(r, goal) if ok else (None, [])
        laws = named_laws(prem, goal)
        r["schema"] = laws
        r["schema_in_rl_pool"] = any(name.split(" (")[0] in SCHEMATA for name in laws)
        out.append(r)

    with open(args.out, "w") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    have = [r for r in out if r["reference_lines"] is not None]
    summary = {
        "n": len(out),
        "with_proof": len(have),
        "no_proof": [r["name"] for r in out if r["reference_lines"] is None],
        "methods": dict(collections.Counter(r["min_lines_method"] for r in out)),
        "length_bin": {
            f"{lo}-{hi}" if hi < 10**9 else f"{lo}+": sum(
                r["length_bin"] == (f"{lo}-{hi}" if hi < 10**9 else f"{lo}+") for r in have
            )
            for lo, hi in BINS
        },
        "family": dict(collections.Counter(r["family"] for r in have)),
        "tags": dict(collections.Counter(t for r in have for t in r["tags"])),
        "ref_lines_ge7": sum(r["reference_lines"] >= 7 for r in have),
        "requires_derived_ore": sum(r["requires_derived_ore"] for r in have),
        "schema_in_rl_pool": sum(r["schema_in_rl_pool"] for r in out),
        "wall_secs": round(time.time() - t0, 1),
        "settings": {k: getattr(args, k) for k in ("bound", "secs", "long_secs", "procs")},
    }
    with open(os.path.splitext(args.out)[0] + "_summary.json", "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
