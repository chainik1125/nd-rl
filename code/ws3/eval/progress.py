"""How far a proof attempt gets before its first invalid line (a continuous companion to pass@1).

The fixed verifier (`fork_pin/nd_verify/verify.py`) only checks individual rule applications once
the final line already matches the goal, so its reason string says nothing about most failed
attempts. `valid_prefix` walks the lines forward and applies the same per-line conditions the
verifier applies, in the same order, stopping at the first line that breaks one:

    consecutive indices, premise block (k-th PR line is the k-th declared premise), box depth,
    rule arity, citations pointing to earlier citable lines or closed boxes (the strict wrapper's
    rule), and the rule's formula check.

It is a diagnostic, never the reward. Its agreement with the verifier is tested: every accepted
proof is fully valid here, and on attempts the verifier rejects with a line-numbered rule error,
the first invalid line here is that line.
"""

from __future__ import annotations

import ndpath  # noqa: F401

# isort: split
from nd_verify.verify import BOT, ParseError, parse_formula, parse_proof_tokens  # fork_pin

ARITY = {"ANDI": 2, "ANDE1": 1, "ANDE2": 1, "IMPE": 2, "IMPI": 2, "ORI1": 1, "ORI2": 1,
         "ORE": 5, "NEGE": 2, "NEGI": 2, "BOTE": 1, "DN": 1, "R": 1}


def parse_prompt(prompt: str):
    """(premises, conclusion) as verifier formulas from 'THM p1 , p2 SEQ c PRF'."""
    toks = prompt.split()
    if not toks or toks[0] != "THM" or toks[-1] != "PRF":
        raise ParseError("malformed prompt")
    i, premises = 1, []
    if toks[i] != "SEQ":
        while True:
            f, i = parse_formula(toks, i)
            premises.append(f)
            if toks[i] == ",":
                i += 1
                continue
            break
    if toks[i] != "SEQ":
        raise ParseError("missing SEQ")
    concl, i = parse_formula(toks, i + 1)
    return premises, concl


def parse_lines(body: str):
    """Lines of the body up to its first QED, stopping at the first line that does not parse."""
    toks = body.split()
    if "QED" in toks:
        toks = toks[: toks.index("QED")]
    lines, seg = [], []
    for t in toks:
        seg.append(t)
        if t == ";":
            try:
                lines += parse_proof_tokens(seg + ["QED"])
            except (ParseError, IndexError, KeyError):
                return lines, True
            seg = []
    return lines, bool(seg)  # trailing tokens without ';' count as an unparseable last line


def _rule_ok(rule, G, refs, F):
    if rule == "R":
        return G == F(refs[0])
    if rule == "ANDI":
        return G == ("and", F(refs[0]), F(refs[1]))
    if rule == "ANDE1":
        return F(refs[0])[0] == "and" and F(refs[0])[1] == G
    if rule == "ANDE2":
        return F(refs[0])[0] == "and" and F(refs[0])[2] == G
    if rule == "IMPE":
        return F(refs[0]) == ("imp", F(refs[1]), G)
    if rule == "IMPI":
        return G == ("imp", F(refs[0]), F(refs[1]))
    if rule == "ORI1":
        return G[0] == "or" and G[1] == F(refs[0])
    if rule == "ORI2":
        return G[0] == "or" and G[2] == F(refs[0])
    if rule == "ORE":
        j, s1, e1, s2, e2 = refs
        return F(j) == ("or", F(s1), F(s2)) and F(e1) == G and F(e2) == G
    if rule == "NEGE":
        return F(refs[1]) == ("not", F(refs[0])) and G == BOT
    if rule == "NEGI":
        return F(refs[1]) == BOT and G == ("not", F(refs[0]))
    if rule == "BOTE":
        return F(refs[0]) == BOT
    if rule == "DN":
        return F(refs[0]) == ("not", ("not", G))
    return False


def valid_prefix(prompt: str, body: str) -> dict:
    """{'n_lines', 'n_valid', 'first_error', 'goal_derived'} for one attempt.

    `n_lines` counts lines that parse; `n_valid` is the number of leading lines that pass every
    per-line check; `first_error` is None if all parsed lines are valid and nothing unparseable
    follows; `goal_derived` is True if a valid top-level, non-assumption line states the goal.
    """
    out = {"n_lines": 0, "n_valid": 0, "first_error": None, "goal_derived": False}
    try:
        premises, concl = parse_prompt(prompt)
    except (ParseError, IndexError, KeyError):
        out["first_error"] = "malformed prompt"
        return out
    lines, trailing_junk = parse_lines(body)
    out["n_lines"] = len(lines)
    stack, ctx, fml = [], {}, {}
    box_start, box_lines, box_depth, as_lines = {}, {}, {}, {}
    next_box, n_pr, in_premises = 0, 0, True

    def line_citable(j, i):
        return j < i and j in ctx and ctx[j] == ctx[i][: len(ctx[j])]

    def box_citable(s, e, i):
        if s not in as_lines:
            return False
        b = as_lines[s]
        if b in ctx[i] or e != box_lines.get(b) or e < s:
            return False  # still open, or e is not the box's last line
        if depth[e] != box_depth[b]:
            return False
        parent = ctx[s][:-1]
        return parent == ctx[i][: len(parent)]

    depth = {}
    idx0 = lines[0]["idx"] if lines else 1
    for k, ln in enumerate(lines):
        idx, d, rule, refs, G = ln["idx"], ln["depth"], ln["rule"], ln["refs"], ln["formula"]
        err = None
        if idx0 < 1:
            err = "line index must be >= 1"
        elif idx != idx0 + k:
            err = f"line index mismatch at position {k + 1}"
        elif rule == "PR":
            if not in_premises or d != 0:
                err = f"PR misplaced at line {idx}"
            elif n_pr >= len(premises) or G != premises[n_pr]:
                err = f"premise block does not match declared premises (line {idx})"
        elif in_premises and n_pr != len(premises):
            err = f"premise block does not match declared premises (line {idx})"
        if err is None and rule == "AS" and (d < 1 or d > len(stack) + 1):
            err = f"bad AS depth at line {idx}"
        if err is None and rule != "AS" and d > len(stack):
            err = f"depth jump at line {idx}"
        if err is not None:
            out["first_error"] = err
            return out
        # structure: same bookkeeping as the verifier's first pass
        if rule == "PR":
            n_pr += 1
        else:
            in_premises = False
        if rule == "AS":
            del stack[d - 1:]
            next_box += 1
            stack.append(next_box)
            box_start[next_box], box_depth[next_box], as_lines[idx] = idx, d, next_box
        else:
            del stack[d:]
        for b in stack:
            box_lines[b] = idx
        ctx[idx], fml[idx], depth[idx] = tuple(stack), G, d
        # rule application, citations only to earlier lines (strict wrapper)
        if rule in ("PR", "AS"):
            if refs:
                err = f"{rule} takes no refs (line {idx})"
        elif len(refs) != ARITY[rule]:
            err = f"wrong ref count for {rule} (line {idx})"
        elif any(not (idx0 <= r < idx) for r in refs):
            err = f"ref out of range (line {idx})"
        elif rule in ("IMPI", "NEGI"):
            if not box_citable(refs[0], refs[1], idx):
                err = f"bad box cite (line {idx})"
        elif rule == "ORE":
            j, s1, e1, s2, e2 = refs
            if not line_citable(j, idx):
                err = f"bad line cite (line {idx})"
            elif not box_citable(s1, e1, idx) or not box_citable(s2, e2, idx):
                err = f"bad box cite (line {idx})"
        elif not all(line_citable(r, idx) for r in refs):
            err = f"bad line cite (line {idx})"
        if err is None and rule not in ("PR", "AS"):
            try:
                ok = _rule_ok(rule, G, refs, fml.__getitem__)
            except (IndexError, TypeError, KeyError):
                ok = False
            if not ok:
                err = f"rule check failed: {rule} (line {idx})"
        if err is not None:
            out["first_error"] = err
            return out
        out["n_valid"] += 1
        if d == 0 and rule != "AS" and G == concl:
            out["goal_derived"] = True
    if trailing_junk:
        out["first_error"] = "unparseable line"
    elif not lines:
        out["first_error"] = "empty proof"
    return out


def valid_prefix_frac(prompt: str, body: str, ref_len: int) -> float:
    """Valid leading lines as a fraction of the reference length, capped at 1."""
    if ref_len is None or ref_len <= 0:
        return float("nan")
    return min(valid_prefix(prompt, body)["n_valid"] / ref_len, 1.0)
