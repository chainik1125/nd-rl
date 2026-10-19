"""Pre-flight for the GRPO v2 lines: the curriculum pools exist, have their lengths, and share no
theorem with any evaluation set or with each other.

    python -m code.experiments.current.thinking_traces_20261005.grpo_v2.preflight POOL_L2 POOL_L3 \
        [POOL_L4]

Overlap is checked three ways, and any hit fails: by normalised prompt; by renaming-class key where
both sides carry one (`key` / `pkey`, as the pool builders write them); and by class, for every row
of every set: the renaming class (atoms P-S permuted, premises as a set) of the theorem and of its
curried normal form (leading antecedents of the goal moved into the premises), Charles's audit rule
of 2026-09-30 (`classes`, copied from `charles_eval/build_clean.py` on leon/thinking, 591d785).
Exits 1 on any overlap or missing file.

A third pool (L4, 16-18 lines) is optional. When it is given, it gets the same band, pairwise and
evaluation-set checks, and the L4 test split joins the evaluation sets for every pool; with two
pools the output is unchanged (line.sh and build_mix_ei.py count its `ok` lines). Passing the L4
test split as the third pool checks it against every other evaluation set (the comparison with
itself is skipped).
"""

from __future__ import annotations

import itertools
import json
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

R = Path("/scratch/cowifornia/nd-rl/think/runs")
TB = Path("/scratch/cowifornia/nd-rl/think/nd-takehome/data/eval_only/textbook72")
CH = R / "pools" / "charles_1002"
CURR = R / "pools" / "curriculum"
EVAL_SETS: dict[str, list[Path]] = {
    "test_13_20 (sealed)": [R / "pools/test_13_20/test_13_20.jsonl"],
    "test_21_30 (sealed)": [R / "pools/test_21_30/test_21_30.jsonl"],
    "t1320_n1000 (read)": [R / "eval/matched/sets/t1320_n1000.jsonl"],
    "t2130_n1000 (read)": [R / "eval/matched/sets/t2130_n1000.jsonl"],
    "hard12 test": [R / "restart2_hard12/hard12_test_1k.jsonl"],
    "hard712 test_1k / dev_500": [
        R / "pools/hard712/test_1k.jsonl",
        R / "pools/hard712/dev_500.jsonl",
    ],
    "textbook72": [TB / "textbook_dev.jsonl", TB / "textbook_train.jsonl"],
    "charles173": [CH / "candidate_v0.jsonl", CH / "candidate_v1.jsonl", CH / "batch3.jsonl"],
    "hand_proved_14": [CH / "hand_proved_14.jsonl"],
    "curriculum L2 test_1k": [CURR / "L2/test_1k.jsonl"],
    "curriculum L3 test_1k": [CURR / "L3/test_1k.jsonl"],
}
BANDS = {"L2": (10, 12), "L3": (13, 15), "L4": (16, 18)}
#: Joins EVAL_SETS only when an L4 pool is checked.
L4_TEST = {"curriculum L4 test_1k": [CURR / "L4/test_1k.jsonl"]}
ATOMS = ("P", "Q", "R", "S")
Formula = tuple[str, ...]
Key = tuple[tuple[str, ...], str]


# --- classes: from charles_eval/build_clean.py (leon/thinking 591d785); only `m` is annotated ---


def parse(prompt: str) -> tuple[list[Formula], Formula]:
    t = prompt.split()
    assert t[0] == "THM" and t[-1] == "PRF", prompt
    i = t.index("SEQ")
    prems: list[Formula] = []
    cur: list[str] = []
    depth = 0
    for tok in t[1:i]:  # premises, split on top-level commas
        if tok == "," and depth == 0:
            prems.append(tuple(cur))
            cur = []
            continue
        depth += (tok == "(") - (tok == ")")
        cur.append(tok)
    if cur:
        prems.append(tuple(cur))
    return prems, tuple(t[i + 1 : -1])


def split_imp(f: Formula) -> tuple[Formula, Formula] | None:
    """`( A > B )` at the top level -> (A, B), else None."""
    if len(f) < 5 or f[0] != "(" or f[-1] != ")":
        return None
    depth = 0
    for j, tok in enumerate(f[1:-1], 1):
        depth += (tok == "(") - (tok == ")")
        if tok == ">" and depth == 0:
            return f[1:j], f[j + 1 : -1]
    return None


def curry(prems: list[Formula], goal: Formula) -> tuple[list[Formula], Formula]:
    prems = list(prems)
    while (s := split_imp(goal)) is not None:
        prems.append(s[0])
        goal = s[1]
    return prems, goal


def canon(prems: list[Formula], goal: Formula) -> Key:
    best: Key | None = None
    for perm in itertools.permutations(ATOMS):
        m: dict[str, str] = dict(zip(ATOMS, perm, strict=True))
        key = (
            tuple(sorted(" ".join(m.get(x, x) for x in p) for p in set(prems))),
            " ".join(m.get(x, x) for x in goal),
        )
        if best is None or key < best:
            best = key
    assert best is not None
    return best


def classes(prompt: str) -> set[Key]:
    prems, goal = parse(prompt)
    return {canon(prems, goal), canon(*curry(prems, goal))}


# ------------------------------------------------------------------------------------------------


def class_set(rs: list[dict[str, Any]]) -> set[Key]:
    out: set[Key] = set()
    for r in rs:
        out |= classes(r["prompt"])
    return out


def rows(paths: Iterable[Path]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for p in paths:
        with open(p) as f:
            out += [json.loads(x) for x in f if x.strip()]
    return out


def norm(p: str) -> str:
    return " ".join(p.split())


def keys(rs: list[dict[str, Any]]) -> set[str]:
    return {str(r[k]) for r in rs for k in ("key", "pkey") if r.get(k)}


def main(argv: list[str]) -> int:
    pools: dict[str, Path] = dict(zip(("L2", "L3", "L4"), map(Path, argv), strict=False))
    eval_sets = {**EVAL_SETS, **(L4_TEST if "L4" in pools else {})}
    bad = 0
    loaded: dict[str, list[dict[str, Any]]] = {}
    for name, p in pools.items():
        if not p.exists():
            print(f"MISSING pool {name}: {p}")
            return 1
        rs = rows([p])
        loaded[name] = rs
        raw = [r.get("min_lines", r.get("n_lines")) for r in rs]
        ls = [x for x in raw if isinstance(x, int)]
        lo, hi = BANDS[name]
        out_band = len(raw) - sum(lo <= x <= hi for x in ls)
        span = f"{min(ls)}-{max(ls)}" if ls else "none"
        print(f"pool {name}: {len(rs)} rows, lengths {span} (band {lo}-{hi})")
        if out_band:
            print(f"  BAD: {out_band} rows outside {lo}-{hi}")
            bad += 1
        if len(rs) != 5000:
            print(f"  NOTE: {len(rs)} rows, not 5,000 (one EI round samples every row k times)")
    pp = {n: {norm(r["prompt"]) for r in rs} for n, rs in loaded.items()}
    pk = {n: keys(rs) for n, rs in loaded.items()}
    pc = {n: class_set(rs) for n, rs in loaded.items()}
    for x, y in itertools.combinations(loaded, 2):
        if pp[x] & pp[y] or pk[x] & pk[y] or pc[x] & pc[y]:
            print(
                f"BAD: {x} and {y} share {len(pp[x] & pp[y])} prompts, "
                f"{len(pk[x] & pk[y])} keys, {len(pc[x] & pc[y])} classes"
            )
            bad += 1
    for ename, paths in eval_sets.items():
        have = [p for p in paths if p.exists()]
        if not have:
            print(f"(skip {ename}: not built)")
            continue
        es = rows(have)
        ep, ek, ec = {norm(r["prompt"]) for r in es}, keys(es), class_set(es)
        for n in loaded:
            if pools[n].resolve() in {q.resolve() for q in have}:  # a test split checked as a pool
                print(f"(self {n} vs {ename}: the same file)")
                continue
            a, b, c = len(pp[n] & ep), len(pk[n] & ek), len(pc[n] & ec)
            flag = "BAD" if a or b or c else "ok "
            print(
                f"{flag} {n} vs {ename} ({len(es)}): {a} prompts, {b} keys, "
                f"{c} renaming / curried classes in common"
            )
            bad += bool(a or b or c)
    print("PREFLIGHT", "FAILED" if bad else "PASSED")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
