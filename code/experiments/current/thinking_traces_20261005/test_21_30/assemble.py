"""Stage 4: select a length-balanced set from refined composites, and write the final set after the
checks.

    # 1. select candidates (balanced over reference lengths)
    python assemble.py select --in work/comps_ref.jsonl --out work/selected.jsonl --lo 21 --hi 30 \
        --per_len 330 --seed 0
    # 2. (lower-bound search: minimality.py; Lean: lean_set.py; both on work/selected.jsonl)
    # 3. write the set
    python assemble.py write --in work/selected_lb_lean.jsonl --out test_21_30.jsonl --per_len 300 \
        --seed 0

`write` keeps a candidate only if Lean accepts its reference proof and the lower-bound search found
no proof of <= its bound (`minsearch.min_found` is null), then takes up to --per_len per reference
length (seeded), and writes rows in the OOD-set format `code.thinking.eval --sets` reads (`name`,
`prompt`, `reference_lines`, `family`, `knob`, `reference_proof`, `max_state`, `max_action`,
`formula_depth`, `in_train_class`) plus provenance and minimality fields (see README).
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import sys


def select(a: argparse.Namespace) -> None:
    """--in may list several composite files (comma-separated): rows are de-duplicated by renaming
    class (first occurrence kept), then up to --per_len are drawn per reference length."""
    rows, seen = [], set()
    for f in a.inp.split(","):
        with open(f) as fh:
            for line in fh:
                r = json.loads(line)
                if r["key"] in seen:
                    continue
                seen.add(r["key"])
                rows.append(r)
    rng = random.Random(a.seed)
    by = collections.defaultdict(list)
    for r in rows:
        if r.get("vocab", {}).get("status") == "error":
            continue
        by[len(r["proof"].split(";")) - 1].append(r)
    out = []
    avail = {}
    for L in range(a.lo, a.hi + 1):
        xs = by.get(L, [])
        avail[L] = len(xs)
        rng.shuffle(xs)
        out += xs[: a.per_len]
    with open(a.out, "w") as fo:
        for r in out:
            fo.write(json.dumps(r) + "\n")
    print(json.dumps(dict(available=avail, selected=len(out))))


def write(a: argparse.Namespace) -> None:
    from code.thinking.data import class_keys  # run as a module from the repo root

    with open(a.inp) as fi:
        rows = [json.loads(line) for line in fi]
    rng = random.Random(a.seed)
    st = collections.Counter()
    by = collections.defaultdict(list)
    seen_k, seen_pk = set(), set()
    for r in rows:
        k, pk = class_keys(r["prompt"])
        if k in seen_k or (pk is not None and pk in seen_pk):
            st["duplicate_class_or_premise_permutation"] += 1
            continue
        seen_k.add(k)
        if pk is not None:
            seen_pk.add(pk)
        r["pkey"] = pk
        r["key"] = k  # data.class_keys form (the sidecar and EvalIndex use it)
        if not r.get("lean_ok"):
            st["lean_reject"] += 1
            continue
        ms = r.get("minsearch", {})
        if ms.get("min_found") is not None:
            st["shorter_proof_found"] += 1
            continue
        if ms.get("timeout"):
            st["lb_timeout_kept"] += 1
        L = len(r["proof"].split(";")) - 1
        by[L].append(r)
    out = []
    for L in sorted(by):
        if not (a.lo <= L <= a.hi):
            continue
        xs = by[L]
        rng.shuffle(xs)
        out += xs[: a.per_len]
    rng.shuffle(out)
    out.sort(key=lambda r: len(r["proof"].split(";")))
    with open(a.out, "w") as fo:
        for i, r in enumerate(out):
            L = len(r["proof"].split(";")) - 1
            ms = r.get("minsearch", {})
            row = dict(
                name=f"t2130_{i:04d}",
                prompt=r["prompt"],
                reference_lines=L,
                family="cut_composite",
                knob=L,
                reference_proof=r["proof"],
                max_state=r["max_state"],
                max_action=r["max_action"],
                formula_depth=r["fdepth"],
                in_train_class=False,
                n_prem=r["n_prem"],
                n_atoms=r["n_atoms"],
                box_depth=r["box_depth"],
                n_ore=r["n_ore"],
                n_actions=r["n_actions"],
                max_name=r["max_name"],
                prompt_tok=r["prompt_tok"],
                rules=r["rules"],
                key=r["key"],
                pkey=r["pkey"],
                minimality=dict(
                    L_ub=L,
                    L_lb=(ms.get("lb", 0) + 1) if ms else None,
                    lb_search_bound=ms.get("bound"),
                    lb_timeout=bool(ms.get("timeout")),
                    construction_lines=r.get("vocab", {}).get("frm", L),
                    vocab_search=r.get("vocab", {}).get("status"),
                    pieces_exact=True,
                ),
                provenance=dict(
                    parts=r["parts"], cuts=r["cuts"], ops=r.get("ops", []), target=r.get("target")
                ),
                lean_ok=True,
            )
            fo.write(json.dumps(row) + "\n")
    hist = collections.Counter(len(r["proof"].split(";")) - 1 for r in out)
    print(json.dumps(dict(n=len(out), hist=dict(sorted(hist.items())), stats=dict(st))))


def merge(a: argparse.Namespace) -> None:
    """--in BASE --refined R1,R2 --lb LB: BASE rows (selection order) with proof / shape / vocab
    taken from the refined files and `minsearch` from the lower-bound file, joined by class key."""
    ref = {}
    for f in a.refined.split(","):
        with open(f) as fh:
            for line in fh:
                r = json.loads(line)
                ref[r["key"]] = r
    lb = {}
    with open(a.lb) as fh:
        for line in fh:
            r = json.loads(line)
            lb[r["key"]] = r["minsearch"]
    n = miss_ref = miss_lb = 0
    with open(a.out, "w") as fo, open(a.inp) as fh:
        for line in fh:
            r = json.loads(line)
            if r["key"] in ref:
                q = ref[r["key"]]
                for k, v in q.items():
                    if k not in ("lean_ok", "lean_err", "minsearch"):
                        r[k] = v
            else:
                miss_ref += 1
            if r["key"] in lb:
                r["minsearch"] = lb[r["key"]]
            else:
                miss_lb += 1
            r.pop("lean_ok", None)
            fo.write(json.dumps(r) + "\n")
            n += 1
    print(json.dumps(dict(n=n, without_refine=miss_ref, without_lb=miss_lb)))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("select", "write"):
        s = sub.add_parser(name)
        s.add_argument("--in", dest="inp", required=True)
        s.add_argument("--out", required=True)
        s.add_argument("--lo", type=int, default=21)
        s.add_argument("--hi", type=int, default=30)
        s.add_argument("--per_len", type=int, default=300)
        s.add_argument("--seed", type=int, default=0)
    m = sub.add_parser("merge")
    m.add_argument("--in", dest="inp", required=True)
    m.add_argument("--refined", required=True)
    m.add_argument("--lb", required=True)
    m.add_argument("--out", required=True)
    a = ap.parse_args()
    {"select": select, "write": write, "merge": merge}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
