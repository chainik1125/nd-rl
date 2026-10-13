"""Tests for ``code.shared_code.proof_gen``.

The golden fixtures are the take-home fork's ``gen.py`` output for four seeds and settings,
verbatim, and the fork's K12 filters (``kh_assemble.py``) on the K12 sample and on a pool drawn as
cap-horizon drew K12's long bins (``fixtures/proof_state/make_golden.py``). The fork checked
every generated proof with its ND verifier; here Lean checks them (``--runslow``; skipped when
Lean is not installed).
"""

import gzip
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from code.shared_code.lean_check import find_lean, judge
from code.shared_code.lean_seq import LeanSeqTokenizer, parse_proof
from code.shared_code.proof_gen import (
    Generator,
    GenFail,
    eligible,
    generate,
    max_box_depth,
    prune,
    pruned_length,
    renaming_class,
    sample_one,
)

FIXTURES = Path(__file__).parent / "fixtures" / "proof_state"


def load(name: str) -> list[dict[str, Any]]:
    with gzip.open(FIXTURES / name, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


SETTINGS = load("gen.jsonl.gz")
FILTERS = load("filters.jsonl.gz")
RECORDS = {r["i"]: r for r in load("k12_sample.jsonl.gz")}


def run(args: list[str], stats: Counter[str]) -> list[str]:
    """``gen.py``'s command line, through ``generate``."""
    flags = [a for a in args if a != "--long"]
    opt = dict(zip(flags[::2], flags[1::2], strict=True))
    records = generate(
        int(opt["--n"]),
        min_lines=int(opt["--min"]),
        max_lines=int(opt["--max"]),
        seed=int(opt["--seed"]),
        long="--long" in args,
        per_length=int(opt["--per_len"]) if "--per_len" in opt else None,
        max_prem=int(opt.get("--max_prem", 3)),
        stats=stats,
    )
    return [json.dumps(r) for r in records]


@pytest.mark.parametrize("setting", SETTINGS, ids=[s["setting"] for s in SETTINGS])
def test_output_matches_the_fork(setting: dict[str, Any]) -> None:
    stats: Counter[str] = Counter()
    lines = run(setting["args"], stats)
    # Byte for byte, in order. A proof of a theorem in a sealed (textbook72) renaming class is
    # stored only as the SHA-256 of the fork's line.
    stored = setting["lines"]
    got = [
        {"sha256": hashlib.sha256(line.encode("utf-8")).hexdigest()}
        if isinstance(want, dict)
        else line
        for line, want in zip(lines, stored, strict=True)
    ]
    assert got == stored
    per_len = Counter(json.loads(line)["n_lines"] for line in lines)
    assert dict(stats) == setting["summary"]["stats"]  # the same draws failed for the same reasons
    assert {str(k): v for k, v in sorted(per_len.items())} == setting["summary"]["per_len"]


def test_filters_match_the_fork() -> None:
    exclude = {renaming_class(t) for t in FILTERS[0]["exclude_thms"]}
    bad = []
    for row in FILTERS[1:]:
        r = RECORDS[row["i"]] if "i" in row else row
        lines = parse_proof(r["proof"].split())
        got = {
            "key": renaming_class(r["thm"]),
            "written_depth": max_box_depth(r["proof"]),
            "pruned_depth3": max((ln.depth for ln in prune(lines, keep_premises=True)), default=0)
            >= 3,
            "pruned_length": pruned_length(r["proof"]),
            "pruned_length_with_premises": len(prune(lines, keep_premises=True)),
            "eligible": eligible(r["thm"], r["proof"], exclude=exclude),
        }
        if got != {k: row[k] for k in got}:
            bad.append((r["thm"], got))
    assert not bad, bad[:3]
    pool = [row for row in FILTERS[1:] if "i" not in row]
    assert sum(row["written_depth"] >= 3 for row in pool) > 30  # both rules were exercised
    assert sum(row["key"] in exclude for row in pool) > 30


def test_renaming_class() -> None:
    assert renaming_class("( R > S ) , R |- S") == "( P > Q ) , P |- Q"
    assert (
        renaming_class("( S v R ) , ( Q & P ) |- ( P > F )") == "( P v Q ) , ( R & S ) |- ( S > F )"
    )
    assert renaming_class(" |- ( Q > Q )") == "|- ( P > P )"  # no premises


def test_pruning_and_depth() -> None:
    proof = (
        "N1 P : PR ; N2 Q : PR ; N3 | R : AS ; N4 | | P : AS ; N5 | | P : R N1 ; "
        "N6 | ( P > P ) : IMPI N4 N5 ; N7 ( R > ( P > P ) ) : IMPI N3 N6 ; QED"
    )
    lines = parse_proof(proof.split())
    assert [ln.idx for ln in prune(lines)] == [1, 3, 4, 5, 6, 7]  # N2 is never cited
    assert [ln.idx for ln in prune(lines, keep_premises=True)] == [1, 2, 3, 4, 5, 6, 7]
    assert pruned_length(proof) == 6
    assert max_box_depth(proof) == 2
    thm = "P , Q |- ( R > ( P > P ) )"
    assert eligible(thm, proof, exclude=set())
    assert not eligible(thm, proof, exclude={renaming_class("Q , R |- ( P > ( Q > Q ) )")})
    assert not eligible(thm, proof, exclude=set(), box_depth=1)


def test_generator_is_reproducible() -> None:
    """Two generators with one seed draw the same proofs and fail on the same draws."""
    g1, g2 = Generator(random.Random(3)), Generator(random.Random(3))
    for _ in range(50):
        try:
            a = sample_one(g1, g1.rng)
        except GenFail as e:
            with pytest.raises(GenFail, match=str(e)):
                sample_one(g2, g2.rng)
        else:
            assert sample_one(g2, g2.rng) == a


@pytest.mark.slow
def test_lean_accepts_generated_proofs() -> None:
    if find_lean() is None:
        pytest.skip("Lean 4 is not installed")
    tok = LeanSeqTokenizer()
    records = [json.loads(line) for s in SETTINGS for line in s["lines"] if isinstance(line, str)]
    verdicts = judge(
        tok, [r["prompt"] for r in records], [tok.encode_proof(r["proof"]) for r in records]
    )
    assert len(records) == 1178  # 1,190 less the 12 stored as digests
    assert all(v.ok for v in verdicts), [v for v in verdicts if not v.ok][:3]
