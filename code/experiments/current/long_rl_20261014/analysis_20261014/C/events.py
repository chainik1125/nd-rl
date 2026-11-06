"""Turn per-token scores into ND decision events.
event = (rec index, kind, rule, token positions). kinds:
  WHAT   : 'have' + asserted formula of a non-premise line   (what to assert next)
  HOW    : proof-term tokens of that line except box contents (rule head, cited names, rule keywords)
  HYP    : the assumed formula of a box
  CLOSE  : the box-closing 'exact' + cited name (rule = CLOSE_IMPI/CLOSE_NEGI/CLOSE_ORE)
  FINAL  : the final 'exact' + name + <eos>
  PREM   : premise restating
mechanical tokens (DEF, PUNCT, BOXOPEN) are attached to nothing.
"""
import annot
from collections import defaultdict


def events(proof):
    toks, lines = annot.annotate(proof)
    ev = defaultdict(list)  # (kind, owner, rule, occurrence) -> positions
    box_count = defaultdict(int)
    for j, (t, role, own, rule, d) in enumerate(toks):
        if role == 'PREM': ev[('PREM', own, 'PR')].append(j)
        elif role == 'LSTART' and rule.startswith('CLOSE_'):
            # two closes for ORE: distinguish by occurrence
            box_count[own] += 1; ev[('CLOSE', own, rule, box_count[own])].append(j)
        elif role == 'CLOSEREF' and rule.startswith('CLOSE_'):
            ev[('CLOSE', own, rule, box_count[own])].append(j)
        elif rule == 'FINAL' or role == 'EOS': ev[('FINAL', None, 'FINAL')].append(j)
        elif role in ('LSTART', 'FORM'): ev[('WHAT', own, rule)].append(j)
        elif role in ('HEAD', 'REF', 'RULEKW'): ev[('HOW', own, rule)].append(j)
        elif role == 'HYP': ev[('HYP', own, rule, box_count[own])].append(j)
    out = []
    for k, pos in ev.items():
        out.append({'kind': k[0], 'own': k[1], 'rule': k[2], 'pos': pos})
    out.sort(key=lambda e: e['pos'][0])
    return toks, lines, out
