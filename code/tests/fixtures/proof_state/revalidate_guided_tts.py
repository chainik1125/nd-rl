"""Re-run run ``guided-tts``'s gate on the step checker with the port's ``check_term``.

    uv run --project code --no-sync python -m \\
        code.tests.fixtures.proof_state.revalidate_guided_tts artifacts/gt/eval/*.steps.jsonl.gz

The inputs are the fork's step dumps (``artifacts/gt/eval/*_plain.steps.jsonl.gz`` of run
``guided-tts``, in ``hf://buckets/dan-pandori/nd-rl/guided-tts/``): each (state, step) pair as a
standalone Lean theorem (``src``) and the fork checker's verdict. There, Lean agreed with every
verdict (``step_check_validate.py``: 314,541 distinct pairs per seed, 0 false rejects, 0 misses),
so a port that returns the same verdicts agrees with Lean on them too. The dumps' prompts include
the sealed textbook72 problems, so they are not committed here.

Each ``src`` is parsed back into the premises, the hypotheses in scope, the declared formula and
the term; files are grouped by seed and deduplicated by ``src``, as the fork's gate did.
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
from collections import Counter
from pathlib import Path

from code.shared_code.lean_seq import Formula, parse_lean_formula
from code.shared_code.step_check import check_term

_SPACED = str.maketrans({c: f" {c} " for c in "()¬⟨⟩,"})
_GLUED = re.compile(r"^(n\d+)(\.1|\.2|\.elim)$")


def symbols(src: str) -> list[str]:
    """Lean source -> ``lean_seq`` symbols (parentheses spaced, ``n3.1`` -> ``n3 .1``)."""
    out: list[str] = []
    for w in src.translate(_SPACED).split():
        m = _GLUED.match(w)
        out += [m.group(1), m.group(2)] if m else [w]
    return out


def parse_src(src: str) -> tuple[list[Formula], dict[str, Formula], Formula, list[str]]:
    """``theorem t (P Q R S : Prop) (h1 : A) .. (n3 : B) .. : F := by have z : F := <term> ;
    exact z`` -> (premises, scope, F, term)."""
    t = symbols(src)
    head = symbols("theorem t ( P Q R S : Prop )")
    assert t[: len(head)] == head, src
    i = len(head)
    premises: list[Formula] = []
    scope: dict[str, Formula] = {}
    while t[i] == "(":
        name = t[i + 1]
        assert t[i + 2] == ":", src
        f, i = parse_lean_formula(t, i + 3)
        assert t[i] == ")", src
        i += 1
        if name[0] == "h":
            assert name == f"h{len(premises) + 1}", src
            premises.append(f)
        else:
            scope[name] = f
    assert t[i] == ":", src
    declared, i = parse_lean_formula(t, i + 1)
    assert t[i : i + 5] == [":=", "by", "have", "z", ":"], src
    again, i = parse_lean_formula(t, i + 5)
    assert again == declared and t[i] == ":=" and t[-3:] == [";", "exact", "z"], src
    return premises, scope, declared, t[i + 1 : -3]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("files", nargs="+", type=Path)
    a = ap.parse_args()
    groups: dict[str, dict[str, str | None]] = {}
    for path in a.files:
        seed = re.search(r"_s(\d+)_", path.name)
        group = groups.setdefault(seed.group(1) if seed else path.name, {})
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                group.setdefault(d["src"], d["verdict"])
    total: Counter[str] = Counter()
    for seed, pairs in sorted(groups.items()):
        table: Counter[str] = Counter()
        for src, verdict in pairs.items():
            got = check_term(*parse_src(src))
            table["same" if got == verdict else f"fork {verdict} / port {got}"] += 1
            table["fork rejects" if verdict else "fork passes"] += 1
        print(f"seed {seed}: {len(pairs):,} pairs, {dict(table)}")
        total.update(table)
    print(f"all: {sum(len(p) for p in groups.values()):,} pairs, {dict(total)}")


if __name__ == "__main__":
    main()
