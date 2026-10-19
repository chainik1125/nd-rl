"""Held-out problem sets the curriculum pools must avoid, compared up to curried normal form.

    python -m code.experiments.current.thinking_traces_20261005.curriculum.heldout FILE.jsonl ...

Two held-out sources (coordinator, 2026-10-09):

* Charles's textbook release of 2026-10-02 (a held-out evaluation set): `candidate_v0`,
  `candidate_v1`, `batch3`, `hand_proved_14` and `derived` under `runs/pools/charles_1002/`
  (412 rows, copied from origin/charles/main; commit in `source_commit.txt`);
* textbook72 (`textbook_dev` + `textbook_train`).

A pool row is excluded when it shares a renaming class with a held-out problem either as written
or in **curried normal form**: every leading antecedent of the goal is moved into the premises
(`|- A > (B > C)` and `A, B |- C` are one problem; Charles's audit rule of 2026-09-30), repeated
premises are dropped. Both sides are put in that form, and the class keys are
`hard712/run.py class_keys` (atom renaming with premise order kept, and the premise-order-
invariant key, which exists up to 5 premises).

The curriculum also applies the curried comparison to the sealed test sets (`SEALED`), within
each band (no two rows of a band share a curried class, so pool and test cannot) and across
bands.
"""

from __future__ import annotations

import argparse
import json
from typing import Any

from code.experiments.current.thinking_traces_20261005.hard712 import ext
from code.ndbench.core.keys import canon_key, pkey, prompt_to_thm, split_thm

GC = ext.gap_common()
MAX_PREM_FOR_PKEY = 5


def class_keys(prompt: str) -> tuple[str, str | None]:
    """As `hard712/run.py class_keys` (not imported from there: `run.py generate --skip` imports
    this module)."""
    thm = prompt_to_thm(prompt)
    prem, _ = split_thm(thm)
    return canon_key(thm), (pkey(thm) if len(prem) <= MAX_PREM_FOR_PKEY else None)


CH = "/scratch/cowifornia/nd-rl/think/runs/pools/charles_1002"
TB = "/scratch/cowifornia/nd-rl/think/nd-takehome/data/eval_only/textbook72"
R = "/scratch/cowifornia/nd-rl/think/runs"
FILES = {
    **{f: f"{CH}/{f}.jsonl" for f in ("candidate_v0", "candidate_v1", "batch3", "hand_proved_14")},
    "derived": f"{CH}/derived.jsonl",
    "textbook_dev": f"{TB}/textbook_dev.jsonl",
    "textbook_train": f"{TB}/textbook_train.jsonl",
}
#: The sealed test sets, compared the same way (the renaming-class exclusion index already holds
#: them as written; this adds their curried forms).
SEALED = {
    "test_13_20": f"{R}/pools/test_13_20/test_13_20.jsonl",
    "test_21_30": f"{R}/pools/test_21_30/test_21_30.jsonl",
    "hard12_test_1k": f"{R}/restart2_hard12/hard12_test_1k.jsonl",
    "hard712_test_1k": f"{R}/pools/hard712/test_1k.jsonl",
    "hard712_dev_500": f"{R}/pools/hard712/dev_500.jsonl",
}


def curried(prompt: str) -> str:
    prem, goal = GC.parse_prompt(prompt)
    prem = list(prem)
    while goal[0] == "imp":
        prem.append(goal[1])
        goal = goal[2]
    seen: list[Any] = []
    for p in prem:
        if p not in seen:
            seen.append(p)
    ps = " , ".join(GC.fstr(p) for p in seen)
    return f"THM {ps} SEQ {GC.fstr(goal)} PRF" if seen else f"THM SEQ {GC.fstr(goal)} PRF"


def keys(prompt: str) -> set[str]:
    out: set[str] = set()
    for p in (prompt, curried(prompt)):
        k, pk = class_keys(p)
        out.add("k:" + k)
        if pk is not None:
            out.add("p:" + pk)
    return out


def heldout_index(files: dict[str, str] | None = None) -> dict[str, set[str]]:
    """key -> the held-out files holding it (default: Charles's release and textbook72; pass
    `{**FILES, **SEALED}` to add the sealed test sets)."""
    idx: dict[str, set[str]] = {}
    for name, path in (FILES if files is None else files).items():
        with open(path) as f:
            for line in f:
                if line.strip():
                    for k in keys(json.loads(line)["prompt"]):
                        idx.setdefault(k, set()).add(name)
    return idx


def hits(idx: dict[str, set[str]], prompt: str) -> set[str]:
    out: set[str] = set()
    for k in keys(prompt):
        out |= idx.get(k, set())
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--sealed", action="store_true", help="also the sealed test sets")
    a = ap.parse_args()
    idx = heldout_index({**FILES, **SEALED} if a.sealed else None)
    for p in a.files:
        n = bad = 0
        by: dict[str, int] = {}
        with open(p) as f:
            for line in f:
                if not line.strip():
                    continue
                n += 1
                h = hits(idx, json.loads(line)["prompt"])
                bad += bool(h)
                for x in h:
                    by[x] = by.get(x, 0) + 1
        print(json.dumps(dict(file=p, rows=n, heldout_hits=bad, by_file=by)))


if __name__ == "__main__":
    main()
