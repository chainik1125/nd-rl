"""Annotated mirror of overlay/lean_tok.proof_tokens: every Lean token of a gold proof gets
(tok, role, owner ND idx, owner rule, depth). Names numbered by first appearance like tok.encode_proof (no shift).
Roles:
  PREM      tokens of a premise-restating `have nK : F := hj ;`
  LSTART    first token of a statement slot: 'have' (continue) or 'exact' (close box / finish)
  DEF       the new name after `have`
  FORM      formula tokens of the asserted formula of a have
  HEAD      the first token of the proof term (rule choice; for R/ANDE/IMPE/NEGE this is a cited name)
  REF       other cited names
  RULEKW    rule syntax inside the term (.1 .2 , ⟩ Or.inl ...) after the head
  BOXOPEN   ( fun ( nS : ... ) => by  wrapper tokens (not the hyp formula)
  HYP       the assumed formula of a box
  CLOSEREF  the name cited by a box's/proof's closing exact
  PUNCT     : := ; ) etc that are forced
  EOS
"""
import os, sys
sys.path.insert(0, "/private/tmp/claude-501/-Users-robbie-repos-nd-rl/a1d02fb4-ba43-48c4-838b-e227293b6c2a/scratchpad/fork51604b3")
from nd_verify.verify import parse_proof_tokens, ATOM_NAMES
ATOMS = ['P', 'Q', 'R', 'S', 'T', 'U', 'W', 'X', 'Y', 'Z']
ATOM_NAMES.update(ATOMS)


def ftoks(f):
    t = f[0]
    if t == 'atom': return [f[1]]
    if t == 'bot': return ['False']
    if t == 'not': return ['(', '¬'] + ftoks(f[1]) + [')']
    op = {'and': '∧', 'or': '∨', 'imp': '→'}[t]
    return ['('] + ftoks(f[1]) + [op] + ftoks(f[2]) + [')']


def annotate(proof):
    lines = parse_proof_tokens(proof.split())
    rule_of = {ln['idx']: ln['rule'] for ln in lines}
    stack, hyp, last = [[]], [None], [None]
    closed = {}

    def close():
        s = stack.pop(); h = hyp.pop(); l = last.pop()
        closed[h[0]] = (h[1], s, l)

    def T(tok, role, own, rule, d):
        return (tok, role, own, rule, d)

    def box(s, e, own, rule, d, neg=False):
        hf, st, l = closed[s]
        assert l == e
        out = [T('(', 'HEAD' if not st_head[0] else 'BOXOPEN', own, rule, d)]
        st_head[0] = True
        out += [T('fun', 'BOXOPEN', own, rule, d), T('(', 'BOXOPEN', own, rule, d), T(('n', s), 'BOXOPEN', own, rule, d),
                T(':', 'BOXOPEN', own, rule, d)]
        out += [T(x, 'HYP', own, rule, d) for x in ftoks(hf)]
        out += [T(')', 'BOXOPEN', own, rule, d), T('=>', 'BOXOPEN', own, rule, d), T('by', 'BOXOPEN', own, rule, d)]
        out += st
        out += [T('exact', 'LSTART', own, 'CLOSE_' + rule, d + 1)]
        if neg:
            out += [T('(', 'PUNCT', own, 'CLOSE_' + rule, d + 1), T(('n', e), 'CLOSEREF', own, 'CLOSE_' + rule, d + 1),
                    T(':', 'PUNCT', own, 'CLOSE_' + rule, d + 1), T('False', 'PUNCT', own, 'CLOSE_' + rule, d + 1),
                    T(')', 'PUNCT', own, 'CLOSE_' + rule, d + 1)]
        else:
            out += [T(('n', e), 'CLOSEREF', own, 'CLOSE_' + rule, d + 1)]
        out += [T(')', 'PUNCT', own, 'CLOSE_' + rule, d + 1)]
        return out

    st_head = [False]
    for ln in lines:
        i, d, f, rule, refs = ln['idx'], ln['depth'], ln['formula'], ln['rule'], ln['refs']
        if rule == 'AS':
            while len(stack) - 1 >= d: close()
            stack.append([]); hyp.append((i, f)); last.append(i)
            continue
        while len(stack) - 1 > d: close()
        n = lambda j: ('n', j)
        st_head[0] = False
        if rule == 'PR':
            k = sum(1 for l2 in lines if l2['rule'] == 'PR' and l2['idx'] <= i)
            toks = ['have', n(i), ':'] + ftoks(f) + [':=', f'h{k}', ';']
            stack[-1] += [T(x, 'PREM', i, 'PR', d) for x in toks]
            last[-1] = i
            continue
        if rule == 'R': term = [(n(refs[0]), 'HEAD')]
        elif rule == 'ANDI': term = [('⟨', 'HEAD'), (n(refs[0]), 'REF'), (',', 'RULEKW'), (n(refs[1]), 'REF'), ('⟩', 'RULEKW')]
        elif rule == 'ANDE1': term = [(n(refs[0]), 'HEAD'), ('.1', 'RULEKW')]
        elif rule == 'ANDE2': term = [(n(refs[0]), 'HEAD'), ('.2', 'RULEKW')]
        elif rule == 'IMPE': term = [(n(refs[0]), 'HEAD'), (n(refs[1]), 'REF')]
        elif rule == 'NEGE': term = [(n(refs[1]), 'HEAD'), (n(refs[0]), 'REF')]
        elif rule == 'ORI1': term = [('Or.inl', 'HEAD'), (n(refs[0]), 'REF')]
        elif rule == 'ORI2': term = [('Or.inr', 'HEAD'), (n(refs[0]), 'REF')]
        elif rule == 'BOTE': term = [('False', 'HEAD'), ('.elim', 'RULEKW'), (n(refs[0]), 'REF')]
        elif rule == 'DN': term = [('Classical.byContradiction', 'HEAD')] + [(x, 'RULEKW') for x in ['(', 'fun', 'hh', '=>']] + [(n(refs[0]), 'REF'), ('hh', 'RULEKW'), (')', 'RULEKW')]
        else: term = None
        head = [T('have', 'LSTART', i, rule, d), T(n(i), 'DEF', i, rule, d), T(':', 'PUNCT', i, rule, d)]
        head += [T(x, 'FORM', i, rule, d) for x in ftoks(f)] + [T(':=', 'PUNCT', i, rule, d)]
        if term is not None:
            body = [T(x, r, i, rule, d) for x, r in term]
        elif rule in ('IMPI', 'NEGI'):
            body = box(refs[0], refs[1], i, rule, d, neg=(rule == 'NEGI'))
        elif rule == 'ORE':
            body = [T('Or.elim', 'HEAD', i, rule, d), T(n(refs[0]), 'REF', i, rule, d)]
            st_head[0] = True
            body += box(refs[1], refs[2], i, rule, d) + box(refs[3], refs[4], i, rule, d)
        else:
            raise ValueError(rule)
        stack[-1] += head + body + [T(';', 'PUNCT', i, rule, d)]
        last[-1] = i
    assert len(stack) == 1
    out = stack[0] + [T('exact', 'LSTART', lines[-1]['idx'], 'FINAL', 0), T(n(lines[-1]['idx']), 'CLOSEREF', lines[-1]['idx'], 'FINAL', 0)]
    # number names by first appearance
    order = {}
    res = []
    for tok, role, own, rule, d in out:
        if isinstance(tok, tuple):
            k = order.setdefault(tok[1], len(order) + 1)
            tok = f'n{k}'
        res.append((tok, role, own, rule, d))
    res.append(('<eos>', 'EOS', None, 'EOS', 0))
    return res, lines
