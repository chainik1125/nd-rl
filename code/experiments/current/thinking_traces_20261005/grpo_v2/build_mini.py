"""The targets of the GRPO mini runs (line.sh ARM_mini, Leon 2026-10-16): fixed subsets of the
RL pools, a fifth of each, with their preflight.

    python -m code.experiments.current.thinking_traces_20261005.grpo_v2.build_mini [--out DIR]

A mini run is the recipe's two stages, 10 iterations each, from the Stage-1 base, on a fifth of
the data: `N_L2` (1,000) of the L2 pool's 5,000 rows, `N_L3` (1,000) of the L3 pool's 5,000, and
for the L3 stage's extra targets the 3-premise rewrites (`curriculum_3p/L3_3p.jsonl`) of the
chosen L3 rows. Every L3_3p row rewrites one L3 pool row (`orig_name`, one to one: 1,843 of the
5,000 have one), and in the full recipe every 3-premise target's original is an L3 target, so the
L3 subset is drawn in two strata in the pool's proportions: round(`N_L3` x 1,843 / 5,000) = 369
rows with a rewrite and the rest from the 3,157 without. The 3-premise subset is those 369 rows'
rewrites, the same fifth of L3_3p. `SEED` is never a run seed: every seed and knob of the mini
runs trains on the same rows. Rows are copied byte for byte, in their file's order.

The pool-size knob (line.sh `_kpool250`, `_kpool500`; 2026-10-16) trains on nested subsets of those:
`NESTED` rows (250, 500) of each 1,000, the first n of one fixed order of the subset (`SEED` + 2
for L2; for L3 `SEED` + 3 over the rows with a rewrite and `SEED` + 4 over the others, in the same
two strata: round(n x 369 / 1,000) with a rewrite, so 92 and 184), so 250 ⊂ 500 ⊂ 1,000, with the
chosen L3 rows' rewrites as their 3-premise subsets (92, 184: the same fraction).

Then the preflights line.sh runs on the targets (`grpo_v2/preflight.py`): the two subsets (their
bands, no prompt, renaming key or renaming / curried class shared with each other or with any
evaluation set, the curriculum test_1k splits the mini runs read included), and the L2 subset
with the 3-premise subset, where only the band line may fail (the rewrites are shorter than the
L3 band by construction). Each needs all 22 evaluation-set lines `ok` (a set that is not built
prints a skip line, which fails). Writes into DIR (default runs/pools/mini):

    L2_mini.jsonl  L3_mini.jsonl  L3_3p_mini.jsonl   the subsets (and *_mini250, *_mini500)
    mini.json                                       counts, the seeds, sources and subsets' md5s
    preflight*.log                                  the preflights' output, per size
    preflight_pass.txt                              only if all passed: `PREFLIGHT PASSED <md5>
                                                    <file>` for each subset

Everything is written to DIR.tmp first and moved in file by file (each move atomic), the pass record
last, so a line starting meanwhile sees the old record or the new one, never a half-written file.

line.sh refuses the mini arms unless the pass record names each subset's md5 as it is. Exits 1 if
a preflight fails (the subsets and logs stay, for a look; no pass record). Written 2026-10-16.
"""

from __future__ import annotations

import argparse
import collections
import contextlib
import hashlib
import io
import json
import random
import re
import sys
from pathlib import Path

from . import preflight

R = preflight.R
POOL_L2 = R / "pools/curriculum/L2/pool.jsonl"
POOL_L3 = R / "pools/curriculum/L3/pool.jsonl"
L3_3P = R / "pools/curriculum_3p/L3_3p.jsonl"
OUT = R / "pools/mini"
N_L2, N_L3 = 1000, 1000  # a fifth of each 5,000-row pool
SEED = 20261016  # the L2 draw; the L3 draw uses SEED + 1, the nested orders SEED + 2 .. + 4
NESTED = (250, 500)  # the pool-size knob's subset sizes, per stage
N_OK = 22  # the preflight's evaluation-set lines: 11 sets x 2 files


def md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def lines(p: Path) -> list[str]:
    return [x for x in p.read_text().split("\n") if x.strip()]


def pick_l2(n_rows: int, n: int, seed: int) -> list[int]:
    """`n` of `n_rows` row indices, without replacement, in file order."""
    return sorted(random.Random(seed).sample(range(n_rows), n))


def pick_l3(names: list[str], rewritten: set[str], n: int, seed: int) -> list[int]:
    """`n` row indices of the L3 pool (`names`), without replacement, in file order, in two strata
    in the pool's proportions: round(n x share) rows whose name has a rewrite, the rest without."""
    yes = [i for i, x in enumerate(names) if x in rewritten]
    no = [i for i, x in enumerate(names) if x not in rewritten]
    n_yes = round(n * len(yes) / len(names))
    rng = random.Random(seed)
    return sorted(rng.sample(yes, n_yes) + rng.sample(no, n - n_yes))


def nested(n_rows: int, strata: list[list[int]], n: int, seeds: list[int]) -> list[int]:
    """n of a subset's `n_rows` row positions, in file order: the first round(n x share) of each
    stratum (positions) in a fixed order (`seeds`, one per stratum), the last stratum taking the
    rest; so a smaller n is a subset of a larger one."""
    orders = [random.Random(sd).sample(st, len(st)) for st, sd in zip(strata, seeds, strict=True)]
    takes = [round(n * len(st) / n_rows) for st in strata[:-1]]
    takes.append(n - sum(takes))
    return sorted(i for o, t in zip(orders, takes, strict=True) for i in o[:t])


def passed(log: str, band_may_fail: bool) -> bool:
    """line.sh's rules: no BAD, MISSING or skip line (with `band_may_fail`, the indented band
    lines aside: the extra-targets rule), and >= N_OK `ok` lines."""
    ls = log.split("\n")
    rx = r"^(BAD|MISSING|\(skip)" if band_may_fail else r"^\s*(BAD|MISSING|\(skip)"
    bad = [x for x in ls if re.match(rx, x)]
    return not bad and sum(x.startswith("ok ") for x in ls) >= N_OK


def run_preflight(a: Path, b: Path) -> tuple[int, str]:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = preflight.main([str(a), str(b)])
    return rc, buf.getvalue()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args(argv)
    l2, l3, x3 = lines(POOL_L2), lines(POOL_L3), lines(L3_3P)
    r3 = [json.loads(x) for x in l3]
    rx = [json.loads(x) for x in x3]
    names = [r["name"] for r in r3]
    assert len(set(names)) == len(names), "L3 pool names must be unique"
    orig = [r["orig_name"] for r in rx]
    assert len(set(orig)) == len(orig) and set(orig) <= set(names), "L3_3p: one rewrite per row"
    i2 = pick_l2(len(l2), N_L2, SEED)
    i3 = pick_l3(names, set(orig), N_L3, SEED + 1)
    chosen = {names[i] for i in i3}
    ix = [j for j, o in enumerate(orig) if o in chosen]
    sub2 = [l2[i] for i in i2]
    sub3 = [l3[i] for i in i3]
    files = {"L2_mini.jsonl": sub2, "L3_mini.jsonl": sub3, "L3_3p_mini.jsonl": [x3[j] for j in ix]}
    has = [names[i] in set(orig) for i in i3]  # per L3 subset row: it has a rewrite
    pos3 = [[k for k, h in enumerate(has) if h], [k for k, h in enumerate(has) if not h]]
    for n in NESTED:  # the pool-size knob's subsets: 250 within 500 within 1,000
        k2 = nested(len(sub2), [list(range(len(sub2)))], n, [SEED + 2])
        k3 = nested(len(sub3), pos3, n, [SEED + 3, SEED + 4])
        picked = {names[i3[k]] for k in k3}
        files[f"L2_mini{n}.jsonl"] = [sub2[k] for k in k2]
        files[f"L3_mini{n}.jsonl"] = [sub3[k] for k in k3]
        files[f"L3_3p_mini{n}.jsonl"] = [x3[j] for j in ix if orig[j] in picked]
    tmp = Path(f"{a.out}.tmp")
    tmp.mkdir(parents=True, exist_ok=True)
    for f, rows in files.items():
        (tmp / f).write_text("".join(x + "\n" for x in rows))
    runs: dict[str, dict[str, object]] = {}
    for sz in ("", *map(str, NESTED)):
        f2, f3, fx = (tmp / f"{x}_mini{sz}.jsonl" for x in ("L2", "L3", "L3_3p"))
        rc, log = run_preflight(f2, f3)
        rcx, logx = run_preflight(f2, fx)
        (tmp / f"preflight{sz}.log").write_text(log)
        (tmp / f"preflight{sz}_extra.log").write_text(logx)
        runs[sz or "1000"] = {
            "exit": rc,
            "passed": passed(log, False) and rc == 0,
            "extra_exit": rcx,
            "extra_passed": passed(logx, True),
        }
    ok = all(r["passed"] and r["extra_passed"] for r in runs.values())
    srcs = ((POOL_L2, len(l2)), (POOL_L3, len(l3)), (L3_3P, len(x3)))
    subsets = {f: {"rows": len(rows), "md5": md5(tmp / f)} for f, rows in files.items()}
    stats = {
        "out": str(a.out),
        "seed": {"L2": SEED, "L3": SEED + 1, "nested": [SEED + 2, SEED + 3, SEED + 4]},
        "subsets": subsets,
        "l3_with_rewrite": sum(has),
        "min_lines": {  # shortest-proof lines per subset
            f: dict(sorted(collections.Counter(json.loads(x)["min_lines"] for x in rows).items()))
            for f, rows in files.items()
        },
        "sources": {str(p): {"rows": n, "md5": md5(p)} for p, n in srcs},
        "preflight": runs,
    }
    (tmp / "mini.json").write_text(json.dumps(stats, indent=1) + "\n")
    a.out.mkdir(parents=True, exist_ok=True)
    pp = a.out / "preflight_pass.txt"
    if not ok:  # no record of subsets that failed
        pp.unlink(missing_ok=True)
    for f in tmp.iterdir():  # each move atomic; the pass record last
        f.replace(a.out / f.name)
    tmp.rmdir()
    if ok:
        (a.out / "preflight_pass.txt.tmp").write_text(
            "".join(f"PREFLIGHT PASSED {x['md5']} {f}\n" for f, x in subsets.items())
            + "preflight.py on each size's L2 and L3 subsets: passed; on its L2 and 3-premise "
            + f"subsets: only the band line failed; {N_OK} evaluation-set lines ok in each "
            + "(preflight*.log)\n"
        )
        (a.out / "preflight_pass.txt.tmp").replace(pp)
    print(json.dumps(stats, indent=1))
    print("mini preflight", "PASSED" if ok else "FAILED", f"(see {a.out})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
