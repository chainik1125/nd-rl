"""Deterministic rejected-draft cold starts from a supplied cap-limited proof.

The verifier is used only to construct training data. No verifier message or
repair is supplied during model sampling. This module performs no data I/O.
"""

from __future__ import annotations

import hashlib
import json
import random
from typing import TypedDict

from strict_verify import safe_verify

VERSION = "single-token-rejected-draft-v1"
ATOMS = ("P", "Q", "R", "S", "F")
BINARY = ("&", "v", ">")
RULES = (
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
)


class Mutation(TypedDict):
    token_index: int
    before: str
    after: str
    kind: str


class CorruptionProvenance(TypedDict):
    version: str
    seed: int
    seed_material_sha256: str
    source_prompt_sha256: str
    correct_proof_sha256: str
    correct_proof_lines: int
    original_proof_cap: int
    draft_sha256: str
    draft_strict_reward: int
    strict_rejection_reason: str
    candidates_tried: int
    mutation: Mutation


class CorruptionResult(TypedDict):
    draft: str
    provenance: CorruptionProvenance


def digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def corrupt_draft(
    prompt: str, body: str, *, seed: int = 0, max_proof_lines: int = 6
) -> CorruptionResult:
    """Change exactly one formula/rule token, requiring strict rejection.

    References, headers, delimiters, scope bars, and QED are never mutated. Some
    mutations remain correct (for example, an unused line); those are skipped.
    The original valid proof is returned to the caller unchanged by construction.
    """
    prompt, body = " ".join(prompt.split()), " ".join(body.split())
    words = body.split()
    ok, reason, lines = safe_verify(prompt, body)
    if not ok or not words or words[-1] != "QED" or words.count("QED") != 1:
        raise ValueError("Correction source must be a valid terminal proof: " + reason)
    if not 1 <= lines <= max_proof_lines:
        raise ValueError(f"Correction source must contain 1..{max_proof_lines} lines")
    candidates = []
    in_formula = True
    for index, word in enumerate(words):
        if word == ";":
            in_formula = True
        elif word == ":":
            in_formula = False
            rule_at = index + 1
            candidates.extend((rule_at, rule, "rule") for rule in RULES if rule != words[rule_at])
        elif in_formula and word in ATOMS:
            candidates.extend((index, atom, "formula_atom") for atom in ATOMS if atom != word)
        elif in_formula and word in BINARY:
            candidates.extend(
                (index, operator, "formula_operator") for operator in BINARY if operator != word
            )
    seed_material = json.dumps([seed, prompt, body], separators=(",", ":"))
    seed_digest = digest(seed_material)
    random.Random(int(seed_digest, 16)).shuffle(candidates)
    for tried, (position, replacement, kind) in enumerate(candidates, 1):
        mutated = list(words)
        mutated[position] = replacement
        draft = " ".join(mutated)
        draft_ok, rejection, _ = safe_verify(prompt, draft)
        if draft_ok:
            continue
        return {
            "draft": draft,
            "provenance": {
                "version": VERSION,
                "seed": seed,
                "seed_material_sha256": seed_digest,
                "source_prompt_sha256": digest(prompt),
                "correct_proof_sha256": digest(body),
                "correct_proof_lines": lines,
                "original_proof_cap": max_proof_lines,
                "draft_sha256": digest(draft),
                "draft_strict_reward": 0,
                "strict_rejection_reason": rejection,
                "candidates_tried": tried,
                "mutation": {
                    "token_index": position,
                    "before": words[position],
                    "after": replacement,
                    "kind": kind,
                },
            },
        }
    raise ValueError("No strictly rejected single-token draft mutation found")
