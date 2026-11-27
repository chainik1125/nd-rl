# ruff: noqa: E501
"""Amendment 6(a) step-2 reading of the pool x seed square, as fixed in PREREG Amendments 6 and 8 (SERRANO side).

Input: the contrasts table written by analyze_cells.py (`--contrasts-out X` -> X.json, rows with a, b, diff, lo, hi in
points, two-level paired bootstrap). Corners (cell ids):
  A  = the 25 Nov adapter              dpo__final__OLD_A16__n64              (batch_old, 64.8 %)
  A' = our trainer's replicate of A    dpo__final25__A16__n64__r1p_s523373504 (batch_a6s2; 25 Nov pool, order and seed)
  B  = 25 Nov pool, our seed           dpo__final25__A16__n64__r1p           (batch_a6s2)
  C  = our pool, 25 Nov seed           dpo__final__A16__n64__r1p_s523373504  (batch_a6s2)
  D  = R-final, our pool and seed      dpo__final__A16__n64__r1p             (batch R, 40.6 %; NOT the dev-rule cell's 43.2 %)
Primary square: corner A := A' (every corner from our trainer; Amendment 8). Secondary: corner A := the 25 Nov adapter.

Readings, checked in the order 1, 2, 4, 3, 5 (Amendment 8): "above 0" = 95 % lower bound > 0; "includes 0" = lo <= 0 <= hi.
  1 pool:     B-D and A-C above 0; C-D and A-B include 0
  2 seed:     C-D and A-B above 0; B-D and A-C include 0
  4 combination only: A-B, A-C, A-D above 0; B-C, B-D, C-D include 0
  3 both:     at least one of B-D, A-C and at least one of C-D, A-B above 0
  5 unresolved: anything else
Replicate check: A'-A includes 0 -> "A reproduced"; otherwise "A not reproduced" and both squares are reported.
Seed-spread X for the stage-1 relabelling: the larger seed-contrast point estimate if reading 2 or 3, and at least |A'-A|
when A'-A excludes 0 (run-to-run spread); null when neither applies.

  a6_reading.py --contrasts .../v2/a6/step2_contrasts.json --out .../v2/a6/step2_reading.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

CORNERS = {
    "A": "dpo__final__OLD_A16__n64",
    "A'": "dpo__final25__A16__n64__r1p_s523373504",
    "B": "dpo__final25__A16__n64__r1p",
    "C": "dpo__final__A16__n64__r1p_s523373504",
    "D": "dpo__final__A16__n64__r1p",
}
# the exact --contrasts list for analyze_cells.py (no id contains "-"; never add an lr1e-3 cell here)
PAIRS = [("B", "D"), ("A", "C"), ("C", "D"), ("A", "B"), ("B", "C"), ("A", "D"),
         ("A'", "C"), ("A'", "B"), ("A'", "D"), ("A'", "A")]  # fmt: skip


def contrasts_arg() -> str:
    return ",".join(f"{CORNERS[a]}-{CORNERS[b]}" for a, b in PAIRS)


def table(rows: list[dict]) -> dict[tuple[str, str], dict]:
    inv = {v: k for k, v in CORNERS.items()}
    return {(inv[r["a"]], inv[r["b"]]): r for r in rows if r["a"] in inv and r["b"] in inv}


def above(r: dict) -> bool:
    return r["lo"] > 0


def includes0(r: dict) -> bool:
    return r["lo"] <= 0 <= r["hi"]


def reading(t: dict[tuple[str, str], dict], a: str) -> dict:
    """The square's reading with corner A taken from corner `a` ("A" or "A'")."""

    def c(x: str, y: str) -> dict:
        return t[(a if x == "A" else x, a if y == "A" else y)]

    pool = [c("B", "D"), c("A", "C")]
    seed = [c("C", "D"), c("A", "B")]
    bcd = [c("B", "C"), c("B", "D"), c("C", "D")]
    if all(map(above, pool)) and all(map(includes0, seed)):
        r = "1 pool"
    elif all(map(above, seed)) and all(map(includes0, pool)):
        r = "2 seed"
    elif all(above(c("A", y)) for y in "BCD") and all(map(includes0, bcd)):
        r = "4 combination only"
    elif any(map(above, pool)) and any(map(above, seed)):
        r = "3 both"
    else:
        r = "5 unresolved"
    x = max(s["diff"] for s in seed) if r in ("2 seed", "3 both") else None
    return {
        "corner_A": CORNERS[a],
        "reading": r,
        "pool_contrasts": {"B-D": c("B", "D"), "A-C": c("A", "C")},
        "seed_contrasts": {"C-D": c("C", "D"), "A-B": c("A", "B")},
        "others": {"B-C": c("B", "C"), "A-D": c("A", "D")},
        "seed_X": x,
    }


def read_all(rows: list[dict]) -> dict:
    t = table(rows)
    missing = [f"{a}-{b}" for a, b in PAIRS if (a, b) not in t]
    if missing:
        raise SystemExit(f"contrasts missing: {missing}")
    rep = t[("A'", "A")]
    reproduced = includes0(rep)
    prim, sec = reading(t, "A'"), reading(t, "A")
    xs = [v for v in (prim["seed_X"], None if reproduced else abs(rep["diff"])) if v is not None]
    return {
        "primary": prim,
        "secondary_25nov_adapter_as_A": sec,
        "replicate": {"A'-A": rep, "A_reproduced": reproduced},
        "readings_agree": prim["reading"] == sec["reading"],
        "relabel_X": max(xs) if xs else None,
        "order": "1, 2, 4, 3, 5",
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--contrasts", type=Path, required=False)
    ap.add_argument("--out", type=Path, required=False)
    ap.add_argument("--print-contrasts-arg", action="store_true")
    a = ap.parse_args()
    if a.print_contrasts_arg:
        print(contrasts_arg())
        return
    out = read_all(json.loads(a.contrasts.read_text()))
    a.out.write_text(json.dumps(out, indent=1) + "\n")
    for k in ("primary", "secondary_25nov_adapter_as_A"):
        print(f"{k}: {out[k]['reading']}")
    rep = out["replicate"]["A'-A"]
    print(
        f"A'-A {rep['diff']:+.1f} [{rep['lo']:+.1f}, {rep['hi']:+.1f}] reproduced={out['replicate']['A_reproduced']}; relabel X={out['relabel_X']}"
    )


if __name__ == "__main__":
    main()
