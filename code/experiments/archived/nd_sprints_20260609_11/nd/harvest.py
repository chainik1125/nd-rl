"""Hindsight prefix harvesting: from any proof attempt (valid or not),
extract the longest verifier-valid prefix ending at a top-level line, and
relabel it as a proof of that line's formula.

Every harvested record is re-verified end-to-end with the independent
verifier before being returned.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nd.verify import verify_text


def _split_lines(body_toks):
    """Split proof-body tokens into per-line token lists (each ends with ';')."""
    lines, cur = [], []
    for t in body_toks:
        if t == 'QED':
            break
        cur.append(t)
        if t == ';':
            lines.append(cur)
            cur = []
    return lines


def _line_parts(line_toks):
    """Return (depth, formula_toks, rule) for one line's tokens, or None."""
    if len(line_toks) < 4 or not line_toks[0].startswith('N'):
        return None
    i = 1
    depth = 0
    while i < len(line_toks) and line_toks[i] == '|':
        depth += 1
        i += 1
    try:
        c = line_toks.index(':')
    except ValueError:
        return None
    formula = line_toks[i:c]
    rule = line_toks[c + 1] if c + 1 < len(line_toks) else None
    return depth, formula, rule


def harvest_prefix(full_text, reserved_ids, min_lines=5):
    """Longest valid top-level prefix of a proof attempt, relabeled.

    Returns {'thm', 'text', 'n_lines'} or None. Skips prefixes whose
    relabeled theorem is in reserved_ids (held-out sets) or whose conclusion
    is just a premise/reiterated premise.
    """
    toks = full_text.split()
    try:
        i_prf = toks.index('PRF')
        i_seq = toks.index('SEQ')
    except ValueError:
        return None
    prem_toks = toks[1:i_seq]
    prem_strs = [s.strip() for s in ' '.join(prem_toks).split(' , ')] \
        if prem_toks else []
    prefix_head = toks[:i_seq + 1]  # THM <premises> SEQ
    lines = _split_lines(toks[i_prf + 1:])
    if len(lines) < min_lines:
        return None

    for k in range(len(lines), min_lines - 1, -1):
        parts = _line_parts(lines[k - 1])
        if parts is None:
            continue
        depth, formula, rule = parts
        if depth != 0 or rule in ('PR', 'AS'):
            continue
        concl = ' '.join(formula)
        if concl in prem_strs:
            continue
        thm = (' , '.join(prem_strs) + ' |- ' + concl) if prem_strs \
            else (' |- ' + concl)
        if thm in reserved_ids:
            continue
        body = [t for ln in lines[:k] for t in ln]
        cand = ' '.join(prefix_head + formula + ['PRF'] + body + ['QED'])
        ok, _, nl = verify_text(cand)
        if ok:
            return {'thm': thm, 'text': cand, 'n_lines': nl}
    return None
