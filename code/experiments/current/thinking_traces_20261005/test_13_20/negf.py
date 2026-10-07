"""Shorten references that prove ~F with a reiterated hypothesis, a blind spot of `minlen`'s search.

    cd nd-rl && python -m code.experiments.current.thinking_traces_20261005.test_13_20.negf \
        --set test_13_20.jsonl --out test_13_20.negf.jsonl --lo 13 --hi 20

`minlen` (and `fastlen`, which inherits its rules) proves ( ~ F ) with a three-line box
`| F : AS ; | F : R ; ( ~ F ) : NEGI`, but `| F : AS ; ( ~ F ) : NEGI s s` is a valid two-line box
(nd_verify and Lean accept it; the model found it, calibration 2026-10-07).  Each such box in a
reference is rewritten (the R line dropped), the proof is re-verified with nd_verify and Lean, and
`reference_lines` / `knob` / shape fields / `minimality` are updated (`L_ub` and `L_lb` minus the
number of boxes, `negf_adjusted` recorded).  Rows whose new length leaves [--lo, --hi] are dropped.
Names and class keys are unchanged.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys

T2130 = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "test_21_30")
sys.path.insert(0, T2130)
sys.dont_write_bytecode = True

from compose import shape  # noqa: E402
from ndproof import Line, Proof, parse  # noqa: E402

from code.thinking.dan import bare, load  # noqa: E402
from code.thinking.lean import lean_check  # noqa: E402

D = load()


def shorten(p: Proof) -> tuple[Proof, int]:
    """Drop every `R` line that ends a NEGI box whose hypothesis is F and only reiterates it."""
    drop = set()
    for ln in p.lines:
        if ln.rule == "NEGI":
            s, e = ln.refs
            le = p.lines[e]
            if p.lines[s].f == ("bot",) and e != s and le.rule == "R" and le.refs == [s]:
                drop.add(e)
    if not drop:
        return p, 0
    remap = {}
    for i, ln in enumerate(p.lines):
        if i in drop:
            remap[i] = ln.refs[0]
    pos, out = {}, []
    for i, ln in enumerate(p.lines):
        if i in drop:
            continue
        pos[i] = len(out)
        out.append(Line(ln.depth, ln.f, ln.rule, list(ln.refs)))
    for ln in out:
        ln.refs = [pos[remap.get(r, r)] for r in ln.refs]
    return Proof(list(p.prem), p.concl, out), len(drop)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lo", type=int, required=True)
    ap.add_argument("--hi", type=int, required=True)
    a = ap.parse_args()
    with bare():
        tok = D.lean_tok.LeanTokenizer("lean_staten")
    with open(a.set) as f:
        rows = [json.loads(x) for x in f]
    st = collections.Counter()
    changed = []
    for r in rows:
        p = parse(r["prompt"], r["reference_proof"])
        q, k = shorten(p)
        if not k:
            continue
        ok, why = q.verify()
        if not ok:
            st["verify_fail"] += 1
            continue
        changed.append((r, q, k))
    items = []
    for r, q, _ in changed:
        with bare():
            _, _, env = D.state_env.decompose(r["prompt"], q.body(), canon=True)
        items.append((tok.statement(r["prompt"]), tok.text(env.text)))
    oks = lean_check(items) if items else []
    upd = {}
    for (r, q, k), ok in zip(changed, oks, strict=True):
        if not ok:
            st["lean_reject"] += 1
            continue
        upd[r["name"]] = (q, k)
    with open(a.out, "w") as fo:
        for r in rows:
            if r["name"] in upd:
                q, k = upd[r["name"]]
                L = len(q.lines)
                st[f"shortened_by_{k}"] += 1
                if not (a.lo <= L <= a.hi):
                    st["dropped_out_of_range"] += 1
                    continue
                sh = shape(q, D.state_env.decompose)
                r["reference_proof_before_negf"] = r["reference_proof"]
                r["reference_proof"] = q.body()
                r["reference_lines"] = r["knob"] = L
                for key in ("max_state", "max_action", "n_actions", "max_name"):
                    r[key] = sh[key]
                r["box_depth"], r["rules"] = sh["box_depth"], sh["rules"]
                m = r["minimality"]
                m["L_ub"] = L
                if m.get("L_lb") is not None:
                    # exact in minlen's space -> exact with the 2-line box too (one box saves one
                    # line); a bare lower bound loses one line per box
                    exact = m.get("cert") in ("exact", "exact_replaced")
                    m["L_lb"] = L if exact else m["L_lb"] - k
                m["negf_adjusted"] = k
            fo.write(json.dumps(r) + "\n")
    print(json.dumps(dict(n_in=len(rows), candidates=len(changed), **st)))


if __name__ == "__main__":
    main()
