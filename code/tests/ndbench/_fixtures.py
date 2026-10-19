"""Shared WP0 test fixtures: 60 verified proofs and a deterministic proof-corruption generator.

Kept independent of `code.ndbench.judge.nd.NDJudge`'s own implementation (it re-parses proof text
by simple token slicing here, not via `core.proof`), so `test_wp0_judge_nd.py`'s agreement check
is a real cross-check, not a tautology.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import TypedDict

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
VALID_PROOFS_PATH = FIXTURES_DIR / "valid_proofs.jsonl"


class FixtureProof(TypedDict):
    name: str
    prompt: str
    thm: str
    n_lines: int
    n_prem: int | None
    rules: list[str] | None
    proof: str


def load_valid_proofs() -> list[FixtureProof]:
    """60 minlen-verified proofs drawn from `ab2/data/reserve.jsonl`'s `minlen_proof` field
    (extracted once; see the module docstring of `valid_proofs.jsonl`'s generation in the WP0
    report -- committed here so tests never depend on that external, non-versioned path)."""
    rows: list[FixtureProof] = []
    with open(VALID_PROOFS_PATH) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _split_lines(body: str) -> list[str]:
    """Independent re-implementation of line-splitting (not `core.proof.split_lines`)."""
    toks = body.split()
    lines: list[str] = []
    cur: list[str] = []
    for t in toks:
        if t == "QED":
            break
        cur.append(t)
        if t == ";":
            lines.append(" ".join(cur))
            cur = []
    return lines


_RULES = [
    "PR",
    "AS",
    "R",
    "ANDI",
    "ANDE1",
    "ANDE2",
    "IMPE",
    "IMPI",
    "ORI1",
    "ORI2",
    "ORE",
    "NEGE",
    "NEGI",
    "BOTE",
    "DN",
]
_ATOMS = ["P", "Q", "R", "S"]


def corrupt_drop_qed(body: str, rng: random.Random) -> str:
    del rng
    toks = body.split()
    return " ".join(toks[:-1]) if toks and toks[-1] == "QED" else body


def corrupt_truncate(body: str, rng: random.Random) -> str:
    """Drop the last 1-2 lines and the QED: a proof that runs out of tokens before finishing."""
    lines = _split_lines(body)
    if len(lines) < 2:
        return corrupt_drop_qed(body, rng)
    drop = rng.randint(1, min(2, len(lines) - 1))
    return " ".join(lines[: len(lines) - drop])


def corrupt_wrong_rule(body: str, rng: random.Random) -> str:
    """Swap one non-PR/AS line's rule name for a different one (often changes its arity too)."""
    lines = _split_lines(body)
    candidates = [
        i for i, ln in enumerate(lines) if ":" in ln.split() and _rule_of(ln) not in ("PR", "AS")
    ]
    if not candidates:
        return corrupt_drop_qed(body, rng)
    i = rng.choice(candidates)
    toks = lines[i].split()
    pos = toks.index(":") + 1
    current = toks[pos]
    other = rng.choice([r for r in _RULES if r not in (current, "PR", "AS")])
    toks[pos] = other
    lines[i] = " ".join(toks)
    return " ".join(lines) + " QED"


def _rule_of(line: str) -> str:
    toks = line.split()
    return toks[toks.index(":") + 1]


def corrupt_wrong_ref(body: str, rng: random.Random) -> str:
    """Change one citation's target line index to a different (still in-range) one."""
    lines = _split_lines(body)
    candidates = []
    for i, ln in enumerate(lines):
        toks = ln.split()
        pos = toks.index(":") + 1 + 1  # first ref token, if any, is right after the rule name
        if pos < len(toks) - 1 and toks[pos].startswith("N") and toks[pos][1:].isdigit():
            candidates.append((i, pos))
    if not candidates:
        return corrupt_drop_qed(body, rng)
    i, pos = rng.choice(candidates)
    lines_i = lines[i].split()
    own_idx = lines_i[0]
    other_indices = [ln.split()[0] for j, ln in enumerate(lines) if j != i]
    if not other_indices:
        return corrupt_drop_qed(body, rng)
    new_ref = rng.choice([r for r in other_indices if r != own_idx] or other_indices)
    lines_i[pos] = new_ref
    lines[i] = " ".join(lines_i)
    return " ".join(lines) + " QED"


def corrupt_bad_index(body: str, rng: random.Random) -> str:
    """Skip a line number partway through (breaks the consecutive-index invariant)."""
    lines = _split_lines(body)
    if len(lines) < 2:
        return corrupt_drop_qed(body, rng)
    i = rng.randint(1, len(lines) - 1)
    toks = lines[i].split()
    idx = int(toks[0][1:])
    toks[0] = f"N{idx + 1}"
    lines[i] = " ".join(toks)
    return " ".join(lines) + " QED"


def corrupt_premise(body: str, rng: random.Random) -> str:
    """Change a PR line's formula by flipping one atom (breaks the premise-match check)."""
    lines = _split_lines(body)
    pr_lines = [i for i, ln in enumerate(lines) if _rule_of(ln) == "PR"]
    if not pr_lines:
        return corrupt_drop_qed(body, rng)
    i = rng.choice(pr_lines)
    toks = lines[i].split()
    atom_positions = [k for k, t in enumerate(toks) if t in _ATOMS]
    if not atom_positions:
        return corrupt_drop_qed(body, rng)
    k = rng.choice(atom_positions)
    toks[k] = rng.choice([a for a in _ATOMS if a != toks[k]])
    lines[i] = " ".join(toks)
    return " ".join(lines) + " QED"


def corrupt_early_qed(body: str, rng: random.Random) -> str:
    """Truncate right after an AS line and close with QED: ends inside an open subproof."""
    lines = _split_lines(body)
    as_positions = [i for i, ln in enumerate(lines) if _rule_of(ln) == "AS"]
    if not as_positions:
        return corrupt_drop_qed(body, rng)
    i = rng.choice(as_positions)
    return " ".join(lines[: i + 1]) + " QED"


CORRUPTIONS = [
    corrupt_drop_qed,
    corrupt_truncate,
    corrupt_wrong_rule,
    corrupt_wrong_ref,
    corrupt_bad_index,
    corrupt_premise,
    corrupt_early_qed,
]


def generate_corruptions(
    proofs: list[FixtureProof], n: int, seed: int = 0
) -> list[tuple[FixtureProof, str]]:
    """`n` `(fixture, corrupted_body)` pairs, deterministic given `seed`, cycling through
    `CORRUPTIONS` and the fixture list so every corruption kind is exercised many times over."""
    rng = random.Random(seed)
    out: list[tuple[FixtureProof, str]] = []
    i = 0
    while len(out) < n:
        proof = proofs[i % len(proofs)]
        corrupt_fn = CORRUPTIONS[i % len(CORRUPTIONS)]
        corrupted = corrupt_fn(proof["proof"], random.Random(rng.random()))
        out.append((proof, corrupted))
        i += 1
    return out
