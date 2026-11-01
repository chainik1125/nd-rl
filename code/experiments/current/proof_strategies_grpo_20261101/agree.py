"""Agreement of Dan's classifier with the reviewer's blind reimplementation (`review_ps/cls.py`) on
every distinct model proof this experiment labelled, and on the S1g3 corpus.

    cd /home/cowifornia/code/research/nd-rl
    ND_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 nice code/.venv/bin/python -m \\
        code.experiments.current.proof_strategies_grpo_20261101.agree

Compared, as in the reviewer's `cmp.py`: the 13 "uses" labels, the four working / genuine flags,
the primary and the curried primary. Writes agree.json next to this file.
"""

from __future__ import annotations

import collections
import gzip
import json
from typing import Any

from . import common as C
from .classify import READS, read_name

STRATS = [
    "direct",
    "and_intro",
    "cond_proof",
    "or_intro",
    "cases",
    "disj_syll",
    "neg_intro",
    "mod_tollens",
    "ex_falso",
    "contradiction",
    "dne",
    "lem",
    "lemma_reuse",
]
WORKSUB = {
    "cond_proof": "cond_proof.working",
    "cases": "cases.genuine",
    "neg_intro": "neg_intro.working",
    "contradiction": "contradiction.working",
}


def compare(
    rev: Any, prompt: str, nd: str, lab: dict[str, Any], diff: collections.Counter, ex: dict
) -> None:
    import nd as rnd  # type: ignore[import-not-found]  # the reviewer's parser

    prem, concl = rnd.parse_prompt(prompt)
    P = rnd.Proof(prem, concl, nd)
    o = rev.classify(prem, concl, nd, pruned=True, P=P)
    keys = []
    for s in STRATS:
        if (s in o["uses"]) != (s in lab["u"]):
            keys.append(f"uses {s} reviewer={s in o['uses']}")
    for s, w in WORKSUB.items():
        if (s in o["work"]) != (w in lab["sub"]):
            keys.append(f"work {s} reviewer={s in o['work']}")
    if o["primary"] != lab["pri"]:
        keys.append(f"primary {lab['pri']}->{o['primary']}")
    if o["curried"] != lab["pric"]:
        keys.append(f"curried {lab['pric']}->{o['curried']}")
    for k in keys:
        diff[k] += 1
        ex.setdefault(k, []).append((prompt, nd))
    diff["_records_with_any_difference"] += bool(keys)


def main() -> int:
    rev = C.reviewer_cls()
    C.dan_strategies()
    seen: set[tuple[str, str]] = set()
    diff: collections.Counter[str] = collections.Counter()
    ex: dict[str, list[tuple[str, str]]] = {}
    n = 0
    for st, stem, seed, k in READS:
        p = C.OUT / "reads" / f"{read_name(st, stem, seed, k)}.json.gz"
        with gzip.open(p, "rt") as f:
            rows = json.load(f)
        for r in rows.values():
            for nd, _, lab in r["proofs"]:
                key = (r["prompt"], nd)
                if key in seen:
                    continue
                seen.add(key)
                compare(rev, r["prompt"], nd, lab, diff, ex)
                n += 1
    n_model = n
    # the corpus: every record
    with (
        open(C.TRACES / "plain_e256.jsonl") as fp,
        gzip.open(C.OUT / "labels/corpus_s1g3.jsonl.gz", "rt") as fl,
    ):
        cdiff: collections.Counter[str] = collections.Counter()
        cex: dict[str, list[tuple[str, str]]] = {}
        nc = 0
        for a, b in zip(fp, fl, strict=True):
            r, lab = json.loads(a), json.loads(b)
            compare(rev, r["prompt"], r["proof"], lab, cdiff, cex)
            nc += 1
    out = {
        "model_distinct_proofs": n_model,
        "model_differences": dict(diff),
        "model_examples": {k: v[:3] for k, v in ex.items()},
        "corpus_records": nc,
        "corpus_differences": dict(cdiff),
        "corpus_examples": {k: v[:3] for k, v in cex.items()},
    }
    (C.HERE / "agree.json").write_text(json.dumps(out, indent=1))
    print(f"model proofs: {n_model} distinct, differences {dict(diff) or 'none'}")
    print(f"corpus: {nc} records, differences {dict(cdiff) or 'none'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
