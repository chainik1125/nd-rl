"""Independent first-order Fitch verifier. Parses the token format and checks
every line, including quantifier rules with eigenvariable side conditions.
Shares no rule logic with the sampler — soundness is independent.

Line format (after PRF): N<i> ['|']*depth <formula> : <RULE> [N<r>...] ;
Quantifier rules:
  ALLE Nj      : conclusion is an instance of the (all x A) at Nj
  EXI  Nj      : (exists x A) generalizing the witness at Nj
  ALLI Nj      : (all x A) from A[x:=p] at Nj; p a parameter fresh in the
                 assumptions Nj depends on and not in the conclusion
  EXE  Nj Ns Ne: from (exists x A) at Nj and box [assume A[x:=p] .. derive C],
                 conclude C; p fresh in (exists x A), C, and outer assumptions
"""
PREDS = {'P': 1, 'Q': 1, 'R': 2, 'S': 2}
VARSET = set('xyzw')
CONSTSET = set('abcde')
PROP_RULES = {'ANDI', 'ANDE1', 'ANDE2', 'IMPE', 'IMPI', 'ORI1', 'ORI2', 'ORE',
              'NEGE', 'NEGI', 'BOTE', 'DN', 'R', 'PR', 'AS'}
Q_RULES = {'ALLI', 'ALLE', 'EXI', 'EXE'}
RULE_NAMES = PROP_RULES | Q_RULES
BOT = ('bot',)


class ParseError(Exception):
    pass


def _is_var(tok):
    return tok[0] in VARSET


def parse_term(toks, i):
    t = toks[i]
    if t in CONSTSET or (t and t[0] in CONSTSET):
        return ('c', t), i + 1
    if _is_var(t):
        return ('v', t), i + 1
    raise ParseError(f'bad term {t!r}')


def parse_formula(toks, i):
    if i >= len(toks):
        raise ParseError('eof')
    t = toks[i]
    if t == 'F':
        return BOT, i + 1
    if t in PREDS:
        if i + 1 >= len(toks) or toks[i + 1] != '(':
            raise ParseError('pred needs (')
        j = i + 2
        terms = []
        while j < len(toks) and toks[j] != ')':
            tm, j = parse_term(toks, j)
            terms.append(tm)
        if j >= len(toks):
            raise ParseError('pred missing )')
        if len(terms) != PREDS[t]:
            raise ParseError(f'arity {t}')
        return ('atom', t, tuple(terms)), j + 1
    if t == '(':
        nx = toks[i + 1] if i + 1 < len(toks) else None
        if nx == '~':
            sub, j = parse_formula(toks, i + 2)
            if j >= len(toks) or toks[j] != ')':
                raise ParseError('missing ) after ~')
            return ('not', sub), j + 1
        if nx in ('A', 'E'):
            var = toks[i + 2]
            if not _is_var(var):
                raise ParseError('quantifier needs var')
            body, j = parse_formula(toks, i + 3)
            if j >= len(toks) or toks[j] != ')':
                raise ParseError('missing ) after quantifier')
            return (('all' if nx == 'A' else 'ex'), var, body), j + 1
        left, j = parse_formula(toks, i + 1)
        if j >= len(toks) or toks[j] not in ('&', 'v', '>'):
            raise ParseError('missing binop')
        op = {'&': 'and', 'v': 'or', '>': 'imp'}[toks[j]]
        right, k = parse_formula(toks, j + 1)
        if k >= len(toks) or toks[k] != ')':
            raise ParseError('missing )')
        return (op, left, right), k + 1
    raise ParseError(f'bad formula token {t!r}')


# --- substitution helpers (independent reimpl) -----------------------------
def _consts(f):
    k = f[0]
    if k == 'bot':
        return set()
    if k == 'atom':
        return {t[1] for t in f[2] if t[0] == 'c'}
    if k == 'not':
        return _consts(f[1])
    if k in ('and', 'or', 'imp'):
        return _consts(f[1]) | _consts(f[2])
    return _consts(f[2])


def _match_instance(A, x, C):
    found = [None]

    def walk(a, cc, bound):
        if a[0] != cc[0]:
            return False
        k = a[0]
        if k == 'bot':
            return True
        if k == 'atom':
            if a[1] != cc[1] or len(a[2]) != len(cc[2]):
                return False
            for ta, tc in zip(a[2], cc[2]):
                if ta == ('v', x) and x not in bound:
                    if found[0] is None:
                        found[0] = tc
                    elif found[0] != tc:
                        return False
                elif ta != tc:
                    return False
            return True
        if k == 'not':
            return walk(a[1], cc[1], bound)
        if k in ('and', 'or', 'imp'):
            return walk(a[1], cc[1], bound) and walk(a[2], cc[2], bound)
        if k in ('all', 'ex'):
            return a[1] == cc[1] and walk(a[2], cc[2], bound | {a[1]})
        return False

    if walk(A, C, frozenset()):
        return found[0] if found[0] is not None else ('v', x)
    return None


def parse_proof_tokens(toks):
    lines = []
    i = 0
    while i < len(toks):
        if toks[i] == 'QED':
            return lines
        t = toks[i]
        if not (t.startswith('N') and t[1:].isdigit()):
            raise ParseError(f'expected line index, got {t!r}')
        idx = int(t[1:])
        i += 1
        depth = 0
        while i < len(toks) and toks[i] == '|':
            depth += 1
            i += 1
        formula, i = parse_formula(toks, i)
        if i >= len(toks) or toks[i] != ':':
            raise ParseError('missing :')
        i += 1
        if i >= len(toks) or toks[i] not in RULE_NAMES:
            raise ParseError(f'bad rule {toks[i] if i < len(toks) else "eof"!r}')
        rule = toks[i]
        i += 1
        refs = []
        while i < len(toks) and toks[i].startswith('N') and toks[i][1:].isdigit():
            refs.append(int(toks[i][1:]))
            i += 1
        if i >= len(toks) or toks[i] != ';':
            raise ParseError('missing ;')
        i += 1
        lines.append({'idx': idx, 'depth': depth, 'formula': formula,
                      'rule': rule, 'refs': refs})
    raise ParseError('missing QED')


def verify(premises, conclusion, proof_toks):
    try:
        lines = parse_proof_tokens(proof_toks)
    except (ParseError, IndexError, RecursionError, ValueError) as e:
        return False, f'parse: {type(e).__name__}'
    if not lines:
        return False, 'empty'
    n = len(lines)
    idx0 = lines[0]['idx']
    for k, ln in enumerate(lines):
        if ln['idx'] != idx0 + k:
            return False, f'index mismatch at {k+1}'

    # box tracking (same scheme as propositional verifier)
    stack, nb = [], [0]
    ctx, box_start, box_lines, box_depth, as_lines = {}, {}, {}, {}, {}
    in_premise = True
    for ln in lines:
        idx, d, rule = ln['idx'], ln['depth'], ln['rule']
        if rule == 'PR':
            if not in_premise or d != 0:
                return False, f'PR misplaced {idx}'
        else:
            in_premise = False
        if rule == 'AS':
            if d < 1 or d > len(stack) + 1:
                return False, f'bad AS depth {idx}'
            del stack[d - 1:]
            nb[0] += 1
            b = nb[0]
            stack.append(b)
            box_start[b] = idx
            box_depth[b] = d
            as_lines[idx] = b
        else:
            if d > len(stack):
                return False, f'depth jump {idx}'
            del stack[d:]
        for b in stack:
            box_lines[b] = idx
        ctx[idx] = tuple(stack)

    pr = [ln for ln in lines if ln['rule'] == 'PR']
    if [p['formula'] for p in pr] != list(premises):
        return False, 'premise mismatch'
    last = lines[-1]
    if last['depth'] != 0 or last['formula'] != conclusion:
        return False, 'final line not conclusion at depth 0'
    if last['rule'] in ('AS',):
        return False, 'conclusion is assumption'

    fml = {ln['idx']: ln['formula'] for ln in lines}
    prem_consts = set().union(*[_consts(p) for p in premises]) if premises else set()

    def line_citable(j, i):
        return 1 <= j < i and ctx[j] == ctx[i][:len(ctx[j])]

    def box_citable(s, e, i):
        if s not in as_lines:
            return False
        b = as_lines[s]
        if e != box_lines.get(b) or lines[e - idx0]['depth'] != box_depth[b]:
            return False
        if b in ctx[i]:
            return False
        return ctx[s][:-1] == ctx[i][:len(ctx[s]) - 1]

    def active_consts(i):
        """constants in premises + AS-hypotheses of boxes open at line i."""
        cs = set(prem_consts)
        for b in ctx[i]:
            cs |= _consts(fml[box_start[b]])
        return cs

    arity = {'ANDI': 2, 'ANDE1': 1, 'ANDE2': 1, 'IMPE': 2, 'IMPI': 2, 'ORI1': 1,
             'ORI2': 1, 'ORE': 5, 'NEGE': 2, 'NEGI': 2, 'BOTE': 1, 'DN': 1,
             'R': 1, 'ALLE': 1, 'EXI': 1, 'ALLI': 1, 'EXE': 3}

    for ln in lines:
        idx, rule, refs, G = ln['idx'], ln['rule'], ln['refs'], ln['formula']
        if rule in ('PR', 'AS'):
            if refs:
                return False, f'{rule} takes no refs {idx}'
            continue
        if len(refs) != arity[rule]:
            return False, f'arity {rule} {idx}'
        if any(not (idx0 <= r <= idx0 + n - 1) for r in refs):
            return False, f'ref range {idx}'
        F = lambda r: fml[r]

        # citability
        if rule in ('IMPI', 'NEGI'):
            s, e = refs
            if not box_citable(s, e, idx):
                return False, f'bad box {idx}'
        elif rule == 'ORE':
            j, s1, e1, s2, e2 = refs
            if not line_citable(j, idx) or not box_citable(s1, e1, idx) or not box_citable(s2, e2, idx):
                return False, f'bad ORE cite {idx}'
        elif rule == 'EXE':
            j, s, e = refs
            if not line_citable(j, idx) or not box_citable(s, e, idx):
                return False, f'bad EXE cite {idx}'
        else:
            for r in refs:
                if not line_citable(r, idx):
                    return False, f'bad cite {idx}'

        ok = True
        if rule == 'R':
            ok = G == F(refs[0])
        elif rule == 'ANDI':
            ok = G == ('and', F(refs[0]), F(refs[1]))
        elif rule == 'ANDE1':
            ok = F(refs[0])[0] == 'and' and F(refs[0])[1] == G
        elif rule == 'ANDE2':
            ok = F(refs[0])[0] == 'and' and F(refs[0])[2] == G
        elif rule == 'IMPE':
            ok = F(refs[0]) == ('imp', F(refs[1]), G)
        elif rule == 'IMPI':
            s, e = refs
            ok = G == ('imp', F(s), F(e))
        elif rule == 'ORI1':
            ok = G[0] == 'or' and G[1] == F(refs[0])
        elif rule == 'ORI2':
            ok = G[0] == 'or' and G[2] == F(refs[0])
        elif rule == 'ORE':
            j, s1, e1, s2, e2 = refs
            ok = F(j) == ('or', F(s1), F(s2)) and F(e1) == G and F(e2) == G
        elif rule == 'NEGE':
            ok = F(refs[1]) == ('not', F(refs[0])) and G == BOT
        elif rule == 'NEGI':
            s, e = refs
            ok = F(e) == BOT and G == ('not', F(s))
        elif rule == 'BOTE':
            ok = F(refs[0]) == BOT
        elif rule == 'DN':
            ok = F(refs[0]) == ('not', ('not', G))
        elif rule == 'ALLE':
            P = F(refs[0])
            ok = P[0] == 'all' and _match_instance(P[2], P[1], G) is not None
        elif rule == 'EXI':
            ok = G[0] == 'ex' and _match_instance(G[2], G[1], F(refs[0])) is not None
        elif rule == 'ALLI':
            if G[0] != 'all':
                return False, f'allI-shape {idx}'
            x, A = G[1], G[2]
            p = _match_instance(A, x, F(refs[0]))
            if p is None:
                return False, f'allI-not-instance {idx}'
            if p[0] != 'v':  # has a real eigen-parameter; check freshness
                pc = p[1]
                if pc in active_consts(refs[0]) or pc in _consts(G):
                    return False, f'allI-eigenvar {idx}'
            ok = True
        elif rule == 'EXE':
            j, s, e = refs
            P = F(j)
            if P[0] != 'ex':
                return False, f'exE-shape {idx}'
            if F(e) != G:
                return False, f'exE-concl {idx}'
            x, A = P[1], P[2]
            p = _match_instance(A, x, F(s))
            if p is None:
                return False, f'exE-not-instance {idx}'
            if p[0] != 'v':
                pc = p[1]
                outer = active_consts(idx)
                if pc in _consts(P) or pc in _consts(G) or pc in outer:
                    return False, f'exE-eigenvar {idx}'
            ok = True
        if not ok:
            return False, f'rule check {rule} {idx}'

    return True, 'ok'


def verify_text(text):
    toks = text.split() if isinstance(text, str) else list(text)
    try:
        if not toks or toks[0] != 'THM':
            return False, 'missing THM', 0
        i = 1
        premises = []
        if toks[i] != 'SEQ':
            while True:
                f, i = parse_formula(toks, i)
                premises.append(f)
                if i < len(toks) and toks[i] == ',':
                    i += 1
                    continue
                break
        if i >= len(toks) or toks[i] != 'SEQ':
            return False, 'missing SEQ', 0
        concl, i = parse_formula(toks, i + 1)
        if i >= len(toks) or toks[i] != 'PRF':
            return False, 'missing PRF', 0
        body = toks[i + 1:]
        ok, reason = verify(premises, concl, body)
        nl = sum(1 for t in body if t == ';')
        return ok, reason, nl
    except (ParseError, IndexError, RecursionError, ValueError) as e:
        return False, f'parse: {type(e).__name__}', 0
