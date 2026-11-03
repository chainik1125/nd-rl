"""σ content in the thinking traces of each corpus (coordinator's trace-leak check, 2026-11-03).

    python -m code.experiments.current.knockout_grpo_20261103.trace_leak   # -> OUT/trace_leak.json

The traces are the search's narration (searchtrace.py): goal lines `ctx |- G := SKEL`, dead ends
with countermodels (`<no>`), backtracking. Per corpus (full = S1g3, ko, d01, d1) and per record, at
the effort level the pickle drew for it (`data.draw_level`, level 0 = empty think) and also over
the whole e64 / e256 files, it counts:
  * `bycontra`  -- `Classical.byContradiction` written in a think block (the classical detour, the
    DN step's Lean form, as a candidate or a plan);
  * `bycontra_dead` -- that candidate marked dead (`<no>` on the same line);
  * `reductio_plan` -- a think block with the detour followed by its NEGI box,
    `|- ( ~ ( ~ G ) ) := fun ( n : ( ~ G ) )`: a reductio planned in the narration;
  * `dneg` -- any `( ~ ( ~` in a think block;
  * `dneg_goal` -- a goal line whose target is a double negation, `|- ( ~ ( ~ ...`;
  * `act_bycontra` -- an action with `Classical.byContradiction` (a DN step in the proof itself;
    in ko only DN of a premise or hypothesis, which Dan's knockout keeps);
  * `act_bycontra_derived` -- a DN action whose argument is not a premise / hypothesis name (the
    proof's own σ+, which the proof labels already count).
"""

from __future__ import annotations

import collections
import json
import re
from pathlib import Path

from .common import OUT, TRACES

DETOUR = "Classical.byContradiction"
PLAN = re.compile(r"\|- \( ~ \( ~ (.+?) \) \) := fun \( n\d+ : \( ~ ")
GOAL_DNEG = re.compile(r"\|- \( ~ \( ~ ")


def draw_levels(ids: list[str]) -> dict[str, int]:
    from code.thinking.data import draw_level  # unchanged since cd089700 (the pickles' commit)

    specs = [(0, None, 0.2), (64, "e64", 0.4), (256, "e256", 0.4)]
    return {i: int(draw_level(i, specs)) for i in ids}


def scan_think(text: str) -> collections.Counter:
    c: collections.Counter = collections.Counter()
    if not text:
        return c
    for line in text.split("\n"):
        if DETOUR in line:
            c["bycontra"] += line.count(DETOUR)
            if "<no>" in line:
                c["bycontra_dead"] += 1
        if "( ~ ( ~" in line:
            c["dneg"] += 1
        if GOAL_DNEG.search(line):
            c["dneg_goal"] += 1
    if DETOUR in text and PLAN.search(text):
        c["reductio_plan"] += 1
    return c


def scan_action(a: str, premises: set[str]) -> collections.Counter:
    c: collections.Counter = collections.Counter()
    if DETOUR in a:
        c["act_bycontra"] += 1
        arg = a.split(DETOUR, 1)[1].strip().split()
        name = arg[0].strip("()") if arg else ""
        if name not in premises and not name.startswith("h"):
            c["act_bycontra_derived"] += 1
    return c


def corpus_files(arm: str) -> dict[str, Path]:
    if arm == "full":
        return {"e64": TRACES / "e64.jsonl", "e256": TRACES / "e256.jsonl"}
    d = OUT / "corpora" / arm
    return {"e64": d / "e64.jsonl", "e256": d / "e256.jsonl"}


def scan(arm: str) -> dict:
    files = corpus_files(arm)
    recs = {k: {json.loads(x)["id"]: json.loads(x) for x in open(p)} for k, p in files.items()}  # noqa: SIM115
    ids = list(recs["e256"])
    lev = draw_levels(ids)
    out: dict = {"records": len(ids), "levels": dict(collections.Counter(lev.values()))}
    for scope in ("trained", "e64_all", "e256_all"):
        tot: collections.Counter = collections.Counter()
        recs_with: collections.Counter = collections.Counter()
        for i in ids:
            if scope == "trained":
                if lev[i] == 0:
                    continue
                r = recs["e64" if lev[i] == 64 else "e256"][i]
            else:
                r = recs[scope.split("_")[0]][i]
            c: collections.Counter = collections.Counter()
            for st in r["steps"]:
                c += scan_think(st.get("think", ""))
            tot += c
            for k in c:
                recs_with[k] += 1
        out[scope] = {"counts": dict(tot), "records_with": dict(recs_with)}
    act: collections.Counter = collections.Counter()
    act_recs: collections.Counter = collections.Counter()
    for i in ids:
        c = collections.Counter()
        for st in recs["e256"][i]["steps"]:
            c += scan_action(st["action"], set())
        act += c
        for k in c:
            act_recs[k] += 1
    out["actions"] = {"counts": dict(act), "records_with": dict(act_recs)}
    return out


def main() -> None:
    res = {arm: scan(arm) for arm in ("full", "ko", "d01", "d1")}
    (OUT / "trace_leak.json").write_text(json.dumps(res, indent=1))
    for arm, r in res.items():
        t = r["trained"]["records_with"]
        print(
            arm,
            r["records"],
            "trained-level records with:",
            t,
            "| actions:",
            r["actions"]["records_with"],
        )


if __name__ == "__main__":
    main()
