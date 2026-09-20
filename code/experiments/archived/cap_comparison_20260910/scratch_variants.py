"""Cold-start plans from the supplied short proof only; no data or model I/O.

dependency-goal-v1 visits the final goal before its prerequisites, retains
branch assumptions and scope depths, and omits prompt premises and irrelevant
lines. It emits ordinary scratch vocabulary strings, not a proof certificate.
The caller must strictly verify the original proof before using its plan.
"""

from __future__ import annotations

import re

VERSION = "dependency-goal-v1"


def _parse(body, max_proof_lines):
    words = body.split()
    if not words or words[-1] != "QED" or words.count("QED") != 1:
        raise ValueError("A single terminal QED is required")
    segments = " ".join(words[:-1]).split(";")
    if segments[-1].strip():
        raise ValueError("Every proof line must end with a semicolon")
    lines = {}
    for segment in segments[:-1]:
        row = segment.split()
        if not row or not re.fullmatch(r"N[1-9][0-9]*", row[0]) or row.count(":") != 1:
            raise ValueError("Malformed proof line")
        idx, colon = int(row[0][1:]), row.index(":")
        if idx in lines or (lines and idx <= next(reversed(lines))):
            raise ValueError("Line indices must be unique and increasing")
        depth = 0
        while 1 + depth < colon and row[1 + depth] == "|":
            depth += 1
        formula = row[1 + depth : colon]
        if not formula or "|" in formula or colon + 1 >= len(row):
            raise ValueError("Malformed formula/rule boundary")
        rule = row[colon + 1]
        refs = []
        for token in row[colon + 2 :]:
            if not re.fullmatch(r"N[1-9][0-9]*", token):
                raise ValueError("Malformed citation")
            ref = int(token[1:])
            if ref not in lines:
                raise ValueError("Every citation must name a preceding line")
            refs.append(ref)
        lines[idx] = {"formula": formula, "rule": rule, "refs": refs, "depth": depth}
    if not lines or len(lines) > max_proof_lines:
        raise ValueError(f"Cold-start source must contain 1..{max_proof_lines} lines")
    return lines


def plan(body: str, max_proof_lines: int = 6) -> list[str]:
    """Return a compact goal-first dependency plan in the existing vocabulary.

    Each entry is RULE [scope bars] FORMULA ;. AS leaves are explicit; PR
    leaves are already supplied by the theorem prompt. Identical formulas at
    distinct line identities remain separate because their contexts may differ.
    Generated plans need not obey this deterministic construction during RL.
    """
    lines = _parse(body, max_proof_lines)
    emitted, visited = [], set()

    def visit(idx):
        if idx in visited:
            return
        visited.add(idx)
        line = lines[idx]
        if line["rule"] == "R" and len(line["refs"]) == 1:
            cited = lines[line["refs"][0]]
            if cited["depth"] == line["depth"] and cited["formula"] == line["formula"]:
                visit(line["refs"][0])
                return
        if line["rule"] != "PR":
            emitted.extend([line["rule"], *(["|"] * line["depth"]), *line["formula"], ";"])
        for ref in line["refs"]:
            visit(ref)

    root = next(reversed(lines))
    visit(root)
    if not emitted:
        # Preserve a target even when the whole proof is a premise/reiteration.
        line = lines[root]
        emitted = ["PR", *line["formula"], ";"]
    return emitted
