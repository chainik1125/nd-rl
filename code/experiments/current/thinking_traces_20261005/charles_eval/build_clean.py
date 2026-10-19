"""Charles's textbook release 2026-10-02 as a held-out evaluation set, minus problems that share a
class with our training pools.

    nice python3 code/experiments/current/thinking_traces_20261005/charles_eval/build_clean.py

A problem's class is its renaming class (atoms P-S permuted, premises as a set) and that of its
curried normal form (leading antecedents of the goal moved into the premises), Charles's audit rule
of 2026-09-30. Pools checked: the K12 Stage-1 theorems, rl_hard12 and the whole hard712 pool (its
samples a and b trained the round-2 and round-3 models); extra pool files can be passed as
arguments. Writes `charles_clean.jsonl` (candidate_v0 + candidate_v1 + batch3, overlaps removed)
and `charles_clean.report.json` next to the release copy. Eval only: never train on these.
"""

from __future__ import annotations

import itertools
import json
import sys
from collections.abc import Iterator
from pathlib import Path

R = Path("/scratch/cowifornia/nd-rl/think/runs")
REL = R / "pools/charles_1002"
ATOMS = ("P", "Q", "R", "S")
Formula = tuple[str, ...]
Key = tuple[tuple[str, ...], str]


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
        m = dict(zip(ATOMS, perm, strict=True))
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


def rows(path: Path) -> Iterator[dict]:
    with open(path) as f:
        for line in f:
            yield json.loads(line)


def main() -> None:
    charles = {
        (f, r["name"]): classes(r["prompt"])
        for f in ("candidate_v0", "candidate_v1", "batch3", "hand_proved_14")
        for r in rows(REL / f"{f}.jsonl")
    }
    pools = {
        "K12 (Stage 1)": Path("/scratch/cowifornia/nd-rl/bench/dan_ckpts/data/train_k12.jsonl"),
        "rl_hard12": R / "pools/rl_hard12.jsonl",
        "hard712 pool (all)": R / "pools/hard712/pool.jsonl",
    }
    pools |= {a: Path(a) for a in sys.argv[1:]}
    report: dict[str, object] = {}
    hit_any: set[tuple[str, str]] = set()
    for name, path in pools.items():
        cls: set[Key] = set()
        for r in rows(path):
            try:
                cls |= classes(r["prompt"])
            except (AssertionError, ValueError):  # a prompt outside the THM ... SEQ ... PRF form
                continue
        hits = sorted(k for k, c in charles.items() if c & cls)
        by: dict[str, int] = {}
        for f, _ in hits:
            by[f] = by.get(f, 0) + 1
        print(f"{name:22s} overlapping Charles problems: {len(hits)} {by}")
        report[name] = {"n": len(hits), "by_file": by, "names": [n for _, n in hits]}
        hit_any |= set(hits)
    out = []
    for f in ("candidate_v0", "candidate_v1", "batch3"):
        for r in rows(REL / f"{f}.jsonl"):
            if (f, r["name"]) not in hit_any:
                out.append(r | {"charles_file": f})
    with open(REL / "charles_clean.jsonl", "w") as g:
        g.writelines(json.dumps(r) + "\n" for r in out)
    report["kept"] = len(out)
    report["removed"] = len({k for k in hit_any if k[0] != "hand_proved_14"})
    with open(REL / "charles_clean.report.json", "w") as g:
        json.dump(report, g, indent=1)
    print(f"kept {len(out)} of 173 (candidate_v0 + candidate_v1 + batch3)")


if __name__ == "__main__":
    main()
