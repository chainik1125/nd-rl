"""Stage 2 driver: sample cut composites of pieces into a target length range.

    python build_composites.py --pieces work/pieces_all.jsonl --out work/comps.jsonl --lo 21 \
        --hi 30 --per_len 450 --seed 2130

A composite starts from a "main" piece (a generator theorem with an exact minimal proof) and applies
1 to --max_cuts cuts: each picks a premise p of the current theorem and replaces it by a "lemma"
piece whose conclusion is p up to atom renaming, together with the lemma's minimal derivation
(`compose.compose`: rename, cut, `ndproof.simplify`, `ndproof.sink_all`), keeping <= 3 premises.
Cutting a premise a previous cut introduced gives a chain; cutting another premise of the main gives
a tree.

Which cuts are allowed (small-scale checks, README): the lemma's minimal proof must end with ANDE1/2
or IMPE, or, when the cut formula is a literal (atom or negated atom), with ORE; it must not use ex
falso (BOTE). A lemma ending with an introduction (IMPI, NEGI, ANDI, ORI) is a detour the main then
eliminates (derive A > B in a box, then apply it): such composites were exactly minimal in 8% of
cases at 12-16 lines. A lemma that ends with a case split (ORE) on a compound formula lets the split
move to the use site (33-53% exact); BOTE lemmas leave a 1-line shortcut in half the composites;
DN-ending lemmas were mixed and are left out. The allowed cuts were 93-100% exact.

When a cut would leave more than 3 premises, another premise A may first be discharged into the
conclusion (Gamma, A |- C becomes Gamma |- A > C; `ndproof.discharge` keeps A-independent lines
outside the new box): +1 line, one premise slot freed.

Sampling is length-directed: draw a target length T from the bins still short of --per_len (weighted
by deficit), a main, then lemmas whose lengths bring the current length towards T (a cut adds about
lemma L - 1 lines). The composite is kept when its actual length (after simplification) is in [lo,
hi], its bin is not full and its renaming class is new, and it passes nd_verify and the filters:
consistent premises, every premise semantically necessary, conclusion not a tautology, every premise
used by the reference proof, format limits (`compose.shape`). Rows: {prompt, thm, proof, lines, key,
parts [(role, piece key, L)], cuts [formula], construction, target, shape}.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import random
import sys
import time

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from compose import class_key, compose, load_piece, shape  # noqa: E402
from ndproof import discharge, fclass, fdepth, fstr, semantic_flags, unfold_discharged  # noqa: E402
from state_env import decompose  # noqa: E402  (fork, read-only; on sys.path via ndproof)

ELIM = {"ANDE1", "ANDE2", "IMPE", "ORE", "DN"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pieces", required=True, help="prep_pieces output (comma-separated files)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--lo", type=int, default=21)
    ap.add_argument("--hi", type=int, default=30)
    ap.add_argument("--per_len", type=int, default=450)
    ap.add_argument("--seed", type=int, default=2130)
    ap.add_argument("--max_cuts", type=int, default=3)
    ap.add_argument("--min_main_L", type=int, default=5)
    ap.add_argument("--min_lemma_L", type=int, default=4)
    ap.add_argument("--max_piece_L", type=int, default=13)
    ap.add_argument("--max_uses", type=int, default=3, help="max composites a piece may appear in")
    ap.add_argument(
        "--lemma_rules",
        default="ANDE1,ANDE2,IMPE",
        help="allowed last rules of a lemma (any conclusion)",
    )
    ap.add_argument(
        "--lemma_rules_literal",
        default="ORE",
        help="further last rules allowed when the lemma's conclusion is a literal",
    )
    ap.add_argument("--allow_bote_lemma", action="store_true")
    ap.add_argument(
        "--overlap",
        type=float,
        default=0.7,
        help="probability that a cut renames the lemma to share as many premises as possible",
    )
    ap.add_argument(
        "--discharge",
        type=float,
        default=0.5,
        help=(
            "when a cut would exceed 3 premises: probability of discharging another premise into "
            "the conclusion first"
        ),
    )
    ap.add_argument("--max_fdepth", type=int, default=5)
    ap.add_argument("--max_prompt_tok", type=int, default=90)
    ap.add_argument("--max_box", type=int, default=3)
    ap.add_argument("--max_state", type=int, default=386)
    ap.add_argument("--max_name", type=int, default=32)
    ap.add_argument("--max_tries", type=int, default=3_000_000)
    ap.add_argument(
        "--prior",
        default=None,
        help="earlier composite files: their piece uses and classes count here",
    )
    ap.add_argument("--max_secs", type=float, default=3000)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    lemma_rules = set(a.lemma_rules.split(","))
    st = collections.Counter()
    pieces = []
    seen_piece = set()
    for f in a.pieces.split(","):
        with open(f) as fh:
            for line in fh:
                r = json.loads(line)
                if r["key"] in seen_piece or not r["consistent"] or r["L"] > a.max_piece_L:
                    continue
                seen_piece.add(r["key"])
                pieces.append((r, load_piece(r)))
    lit_rules = set(a.lemma_rules_literal.split(","))
    lem_ok = [
        i
        for i, (r, p) in enumerate(pieces)
        if r["L"] >= a.min_lemma_L
        and (r["last_rule"] in lemma_rules or (r["concl_lit"] and r["last_rule"] in lit_rules))
        and (a.allow_bote_lemma or not r["has_bote"])
        and 1 <= r["n_prem"] <= 3
    ]
    by_class: dict = collections.defaultdict(lambda: collections.defaultdict(list))
    for i in lem_ok:
        r, p = pieces[i]
        by_class[fclass(p.concl)[0]][r["L"]].append(i)
    mains = [
        i
        for i, (r, p) in enumerate(pieces)
        if r["L"] >= a.min_main_L and p.prem and any(fclass(q)[0] in by_class for q in p.prem)
    ]
    print(
        f"pieces {len(pieces)}, mains {len(mains)}, lemmas {len(lem_ok)} in {len(by_class)} "
        "conclusion classes",
        flush=True,
    )

    targets = list(range(a.lo, a.hi + 1))
    have = collections.Counter()
    seen = set()
    uses = collections.Counter()
    if a.prior:  # piece uses and classes of earlier builds count against --max_uses / dedup
        kidx = {r["key"]: i for i, (r, p) in enumerate(pieces)}
        for f in a.prior.split(","):
            with open(f) as fh:
                for line in fh:
                    try:
                        q = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    seen.add(q["key"])
                    for _, k, _ in q["parts"]:
                        if k in kidx:
                            uses[kidx[k]] += 1
    t0 = time.time()
    tries = 0
    with open(a.out, "w") as fo:
        while (
            tries < a.max_tries
            and time.time() - t0 < a.max_secs
            and any(have[L] < a.per_len for L in targets)
        ):
            tries += 1
            deficit = [max(0, a.per_len - have[L]) for L in targets]
            T = rng.choices(targets, weights=deficit)[0]
            mi = rng.choice(mains)
            if uses[mi] >= a.max_uses:
                continue
            mr, cur = pieces[mi]
            parts = [("main", mr["key"], mr["L"])]
            cuts, used, ops = [], [mi], []
            for _ in range(a.max_cuts):
                need = T - len(cur.lines)
                if need < 3:
                    break
                cands = [j for j, q in enumerate(cur.prem) if fclass(q)[0] in by_class]
                if not cands:
                    break
                j = rng.choice(cands)
                byL = by_class[fclass(cur.prem[j])[0]]
                want = need + 1
                if want in byL and rng.random() < 0.6:
                    L = want
                else:
                    Ls = sorted(byL)
                    under = [x for x in Ls if x <= want]
                    L = rng.choice(under) if under else Ls[0]
                li = None
                for _ in range(8):  # a few draws from the bucket for a lemma not used up
                    x = rng.choice(byL[L])
                    if x not in used and uses[x] < a.max_uses:
                        li = x
                        break
                if li is None:
                    st["lemma_busy"] += 1
                    continue
                lr, lem = pieces[li]
                ov = rng.random() < a.overlap
                nxt = compose(lem, cur, j, rng, overlap=ov)
                if nxt is None and len(cur.prem) >= 2 and rng.random() < a.discharge:
                    # free a premise slot: discharge the smallest other premise into the conclusion
                    # (deduction)
                    ks = sorted(
                        (fdepth(q), rng.random(), k) for k, q in enumerate(cur.prem) if k != j
                    )
                    k = ks[0][2]
                    d = discharge(cur, k)
                    if d is not None:
                        j2 = j - (1 if k < j else 0)
                        nxt = compose(lem, d, j2, rng, overlap=ov)
                        if nxt is not None:
                            ops.append(("discharge", fstr(cur.prem[k])))
                            st["discharged"] += 1
                if nxt is None:
                    st["compose_none"] += 1
                    continue
                cuts.append(fstr(cur.prem[j]))
                parts.append(("lemma", lr["key"], lr["L"]))
                used.append(li)
                cur = nxt
                if len(cur.lines) >= T:
                    break
            if not cuts:
                st["no_cut"] += 1
                continue
            c = cur
            L = len(c.lines)
            if L not in targets or have[L] >= a.per_len:
                st["len_skip"] += 1
                continue
            ok, why = c.verify()
            if not ok:
                st["VERIFY_FAIL"] += 1
                continue
            key = class_key(c)
            if key in seen:
                st["dup_class"] += 1
                continue
            sf = semantic_flags(c.prem, c.concl)
            if not sf["consistent"]:
                st["reject_inconsistent"] += 1
                continue
            if not sf["all_necessary"]:
                st["reject_redundant_premise"] += 1
                continue
            if ops:  # a discharged antecedent counts as a premise: it must be necessary too
                up, uc = unfold_discharged(c.prem, c.concl, len(ops))
                if not semantic_flags(up, uc)["all_necessary"]:
                    st["reject_vacuous_antecedent"] += 1
                    continue
            if sf["concl_tautology"] and c.prem:
                st["reject_tautology"] += 1
                continue
            sh = shape(c, decompose)
            bad = None
            if not sh["all_prem_used"]:
                bad = "unused_prem"
            elif sh["fdepth"] > a.max_fdepth:
                bad = "fdepth"
            elif sh["prompt_tok"] > a.max_prompt_tok:
                bad = "prompt_tok"
            elif sh["box_depth"] > a.max_box:
                bad = "box_depth"
            elif "decomp_err" in sh:
                bad = "decomp"
            elif sh["max_state"] > a.max_state:
                bad = "max_state"
            elif sh["max_name"] > a.max_name:
                bad = "max_name"
            if bad:
                st["reject_" + bad] += 1
                continue
            seen.add(key)
            have[L] += 1
            for i in used:
                uses[i] += 1
            row = dict(
                prompt=c.prompt(),
                thm=c.thm(),
                proof=c.body(),
                key=key,
                parts=parts,
                cuts=cuts,
                ops=ops,
                construction=sum(x[2] for x in parts) - len(cuts),
                target=T,
                **sh,
            )
            fo.write(json.dumps(row) + "\n")
            if sum(have.values()) % 500 == 0:
                print(
                    f"{sum(have.values())} kept, tries {tries}, {time.time() - t0:.0f}s, "
                    f"{dict(sorted(have.items()))}",
                    flush=True,
                )
    summ = dict(
        args=vars(a),
        stats=dict(st),
        have=dict(sorted(have.items())),
        tries=tries,
        secs=round(time.time() - t0, 1),
        n_pieces=len(pieces),
        n_mains=len(mains),
        n_lemmas=len(lem_ok),
    )
    with open(a.out + ".summary.json", "w") as fo:
        json.dump(summ, fo, indent=1)
    print(json.dumps(summ))


if __name__ == "__main__":
    main()
