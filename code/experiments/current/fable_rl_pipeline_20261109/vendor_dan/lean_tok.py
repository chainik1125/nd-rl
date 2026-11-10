"""Lean 4 as the training surface form for the from-scratch model (run lean-format, proposal 8). One symbol per token.

The text the model reads and writes is the nd2lean.py rendering of an ND proof, linearised on one line with `;` between
tactics (no indentation tokens):

  prompt : theorem t ( P Q R S : Prop ) ( h1 : F1 ) ... : C := by
  proof  : have n1 : F1 := h1 ; have n7 : ( P → R ) := ( fun ( n3 : P ) => by have n4 : Q := n1 n3 ; exact n4 ) ; exact n7 <eos>

Terms (exactly nd2lean's): PR `hK`; R `nA`; ANDI `⟨ nA , nB ⟩`; ANDE `nA .1|.2`; IMPE / NEGE `nF nA`; ORI `Or.inl|Or.inr nA`;
BOTE `nA .elim`; DN `Classical.byContradiction ( fun hh => nA hh )`; IMPI `( fun ( nS : A ) => by ... ; exact nE )`;
NEGI the same ending `exact ( nE : False )`; ORE `Or.elim nJ <box> <box>`.  `.1 .2 .elim` are glued to the name in the text.

Hypothesis names are LABELS, not line indices.  64 name tokens n1..n64; two naming schemes (stored in the ckpt as the mode):
  lean_rand : every training presentation draws a random injective map names -> n1..n64 (names carry no order)
  lean_seq  : names are numbered in order of first appearance in the Lean text, plus a random start offset
              (the analogue of the token format's `abs` start-index shift)
decode() parses the sampled tokens with the strict grammar above and returns the ND proof (spec.md format, N1..) it
denotes — the "nd2lean inverse" — so nd_verify, prune, normalize and every analysis script work unchanged; a sample
outside the grammar decodes to 'LEANPARSE <reason>' (rejected by nd_verify).  The literal Lean text of the last decoded
sample is kept in `last_text` and is what lean_gate.py sends to Lean.
"""
import random, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nd_verify.verify import parse_proof_tokens, parse_formula
from nd2lean import parse_prompt, translate

MAXN = 64
MAXH = 8
FSYMS = ['(', ')', '¬', '∧', '∨', '→', 'P', 'Q', 'R', 'S', 'False']
PSYMS = ['theorem', 't', ':', 'Prop', ':=', 'by']
TSYMS = ['have', 'exact', ';', 'fun', '=>', '⟨', '⟩', ',', '.1', '.2', '.elim', 'Or.inl', 'Or.inr', 'Or.elim', 'Classical.byContradiction', 'hh']
# run state-env (proposal 13): the only tokens the state-conditioned modes add on top of lean_seq's vocabulary.
STATE_SYMS = ['<st>', '<nl>', '⊢', '<act>']
STATE_MODES = ('lean_state', 'lean_stateh', 'lean_staten')


class ParseFail(Exception):
    pass


def ftoks(f):
    """formula tuple -> Lean tokens (fully parenthesised, as nd2lean.lf)"""
    t = f[0]
    if t == 'atom': return [f[1]]
    if t == 'bot': return ['False']
    if t == 'not': return ['(', '¬'] + ftoks(f[1]) + [')']
    op = {'and': '∧', 'or': '∨', 'imp': '→'}[t]
    return ['('] + ftoks(f[1]) + [op] + ftoks(f[2]) + [')']


def fnd(f):
    """formula tuple -> ND tokens (spec.md)"""
    t = f[0]
    if t == 'atom': return f[1]
    if t == 'bot': return 'F'
    if t == 'not': return f'( ~ {fnd(f[1])} )'
    op = {'and': '&', 'or': 'v', 'imp': '>'}[t]
    return f'( {fnd(f[1])} {op} {fnd(f[2])} )'


def prompt_tokens(prompt):
    prem, concl = parse_prompt(prompt)
    if len(prem) > MAXH:
        raise ValueError('too many premises')
    out = ['theorem', 't', '(', 'P', 'Q', 'R', 'S', ':', 'Prop', ')']
    for j, p in enumerate(prem):
        out += ['(', f'h{j+1}', ':'] + ftoks(p) + [')']
    return out + [':'] + ftoks(concl) + [':=', 'by']


def proof_tokens(proof):
    """ND proof body -> Lean tokens with names as ('n', ND index).  Mirrors nd2lean.translate (which must have accepted
    the proof); a closed box that no rule cites is dropped, as in nd2lean."""
    lines = parse_proof_tokens(proof.split())
    stack = [[]]            # token lists of the open boxes' statements; stack[0] = top level
    hyp = [None]            # (idx, formula) of each open box
    last = [None]
    closed = {}             # start idx -> (hyp formula, stmts tokens, last idx)
    n_pr = 0

    def close():
        s = stack.pop(); h = hyp.pop(); l = last.pop()
        closed[h[0]] = (h[1], s, l)

    def box(s, e, neg=False):
        hf, st, l = closed[s]
        assert l == e, 'box cite'
        ex = ['exact', '(', ('n', e), ':', 'False', ')'] if neg else ['exact', ('n', e)]
        return ['(', 'fun', '(', ('n', s), ':'] + ftoks(hf) + [')', '=>', 'by'] + st + ex + [')']

    for ln in lines:
        i, d, f, rule, refs = ln['idx'], ln['depth'], ln['formula'], ln['rule'], ln['refs']
        if rule == 'AS':
            while len(stack) - 1 >= d: close()
            stack.append([]); hyp.append((i, f)); last.append(i)
            continue
        while len(stack) - 1 > d: close()
        n = lambda j: ('n', j)
        if rule == 'PR': n_pr += 1; term = [f'h{n_pr}']
        elif rule == 'R': term = [n(refs[0])]
        elif rule == 'ANDI': term = ['⟨', n(refs[0]), ',', n(refs[1]), '⟩']
        elif rule == 'ANDE1': term = [n(refs[0]), '.1']
        elif rule == 'ANDE2': term = [n(refs[0]), '.2']
        elif rule == 'IMPE': term = [n(refs[0]), n(refs[1])]
        elif rule == 'NEGE': term = [n(refs[1]), n(refs[0])]
        elif rule == 'ORI1': term = ['Or.inl', n(refs[0])]
        elif rule == 'ORI2': term = ['Or.inr', n(refs[0])]
        elif rule == 'BOTE': term = [n(refs[0]), '.elim']
        elif rule == 'DN': term = ['Classical.byContradiction', '(', 'fun', 'hh', '=>', n(refs[0]), 'hh', ')']
        elif rule == 'IMPI': term = box(refs[0], refs[1])
        elif rule == 'NEGI': term = box(refs[0], refs[1], neg=True)
        elif rule == 'ORE': term = ['Or.elim', n(refs[0])] + box(refs[1], refs[2]) + box(refs[3], refs[4])
        else: raise ValueError(rule)
        stack[-1] += ['have', n(i), ':'] + ftoks(f) + [':='] + term + [';']
        last[-1] = i
    assert len(stack) == 1
    return stack[0] + ['exact', ('n', lines[-1]['idx'])]


class LeanTokenizer:
    def __init__(self, mode='lean_rand'):
        assert mode in ('lean_rand', 'lean_seq') + STATE_MODES
        self.mode = mode
        self.state_mode = mode in STATE_MODES       # names are still lean_seq's; the extra tokens render the tactic state
        self.with_history = mode == 'lean_stateh'
        self.canon = mode == 'lean_staten'      # names a step introduces are `max index in scope + 1` (run state-env, arm SN)
        self.itos = ['<pad>', '<eos>'] + FSYMS + PSYMS + TSYMS + (STATE_SYMS if self.state_mode else []) \
            + [f'h{k}' for k in range(1, MAXH + 1)]
        self.ref0 = len(self.itos)
        self.itos += [f'n{k}' for k in range(1, MAXN + 1)]
        self.stoi = {s: i for i, s in enumerate(self.itos)}
        self.pad, self.eos = 0, 1
        self.shift = True
        self.last_text = None

    @property
    def vocab_size(self):
        return len(self.itos)

    def encode_prompt(self, prompt):
        return [self.stoi[t] for t in prompt_tokens(prompt)]

    def encode_proof(self, body):
        """ND body -> ids with names numbered by first appearance (n1, n2, ...); <eos> included."""
        order = {}
        out = []
        for t in proof_tokens(body):
            if isinstance(t, tuple):
                k = order.setdefault(t[1], len(order) + 1)
                if k > MAXN: raise ValueError('too many names')
                out.append(self.ref0 + k - 1)
            else:
                out.append(self.stoi[t])
        return out + [self.eos]

    def shift_abs(self, ids, rng):
        """name augmentation (the interface name is train.py's)."""
        if not self.shift:
            return ids
        mx = max((x - self.ref0 + 1 for x in ids if x >= self.ref0), default=0)
        if mx == 0:
            return ids
        if self.mode != 'lean_rand':
            s = rng.randint(0, MAXN - mx)
            return [x + s if x >= self.ref0 else x for x in ids]
        perm = rng.sample(range(MAXN), mx)
        return [self.ref0 + perm[x - self.ref0] if x >= self.ref0 else x for x in ids]

    # ---- the state-conditioned modes (run state-env) ----
    def encode_toks(self, toks):
        return [self.stoi[t] for t in toks]

    def shift_pair(self, ids_a, ids_b, rng):
        """one random name offset applied to BOTH halves of a (state, action) pair (lean_seq's augmentation)."""
        if not self.shift:
            return ids_a, ids_b
        mx = max((x - self.ref0 + 1 for x in list(ids_a) + list(ids_b) if x >= self.ref0), default=0)
        if mx == 0:
            return ids_a, ids_b
        s = rng.randint(0, MAXN - mx)
        f = lambda ids: [x + s if x >= self.ref0 else x for x in ids]
        return f(ids_a), f(ids_b)

    def decode_action(self, ids):
        """sampled ids of ONE action -> (token strings, ended).  `ended` is False if <eos> never came (truncation)."""
        toks = []
        for x in ids:
            if x == self.pad:
                continue
            if x == self.eos:
                return toks, True
            toks.append(self.itos[x])
        return toks, False

    # ---- text ----
    def text(self, toks):
        """tokens (strings) -> Lean source text of the tactic block (one line)."""
        out = []
        for t in toks:
            if t in ('.1', '.2', '.elim') and out:
                out[-1] += t
            else:
                out.append(t)
        return ' '.join(out)

    def statement(self, prompt):
        return ' '.join(prompt_tokens(prompt))

    # ---- inverse ----
    def decode(self, ids):
        toks = []
        ended = False
        for x in ids:
            if x == self.pad: continue
            if x == self.eos: ended = True; break
            toks.append(self.itos[x])
        self.last_text = self.text(toks) if ended else None
        if not ended:
            return 'LEANPARSE no-eos'
        try:
            return inverse(toks)
        except ParseFail as e:
            return f'LEANPARSE {e}'


def inverse(toks):
    """Lean tokens (strict nd2lean grammar) -> ND proof body 'N1 ... ; ... QED'.  Names resolve lexically (latest binding wins,
    as in Lean).  Raises ParseFail outside the grammar: PR lines first and in order, a box's `exact` cites the box's last line,
    the final `exact` cites the last top-level line."""
    pos = [0]
    out = []                 # ND lines (strings)
    forms = {}               # ND idx -> formula tuple
    n_pr = [0]; seen_non_pr = [False]

    def peek():
        return toks[pos[0]] if pos[0] < len(toks) else None

    def eat(x=None):
        t = peek()
        if t is None or (x is not None and t != x):
            raise ParseFail(f'expected {x}')
        pos[0] += 1
        return t

    def name_tok():
        t = eat()
        if not (t[0] == 'n' and t[1:].isdigit()):
            raise ParseFail('name')
        return t

    def formula():
        # re-use the ND formula parser on a token-mapped slice
        m = {'(': '(', ')': ')', '¬': '~', '∧': '&', '∨': 'v', '→': '>', 'P': 'P', 'Q': 'Q', 'R': 'R', 'S': 'S', 'False': 'F'}
        j = pos[0]; nd = []
        while j < len(toks) and toks[j] in m:
            nd.append(m[toks[j]]); j += 1
        try:
            f, used = parse_formula(nd + ['$'], 0)
        except Exception:
            raise ParseFail('formula')
        pos[0] += used
        return f

    def ref(scope):
        t = name_tok()
        if t not in scope:
            raise ParseFail('unbound')
        return scope[t]

    def emit(depth, f, rule, refs):
        i = len(out) + 1
        out.append(f'N{i} ' + '| ' * depth + f'{fnd(f)} : {rule}' + ''.join(f' N{r}' for r in refs) + ' ;')
        forms[i] = f
        return i

    def box(scope, depth):
        """'( fun ( n : A ) => by stmts exact .. )' -> (start idx, end idx, neg)"""
        eat('('); eat('fun'); eat('('); nm = name_tok(); eat(':'); hf = formula(); eat(')'); eat('=>'); eat('by')
        s = emit(depth + 1, hf, 'AS', [])
        sc = dict(scope); sc[nm] = s
        lastl = stmts(sc, depth + 1, s)
        eat('exact')
        if peek() == '(':
            eat('('); e = ref(sc); eat(':'); eat('False'); eat(')'); neg = True
        else:
            e = ref(sc); neg = False
        if e != lastl:
            raise ParseFail('exact not last')
        eat(')')
        return s, e, neg

    def stmts(scope, depth, lastl):
        while peek() == 'have':
            eat('have'); nm = name_tok(); eat(':'); f = formula(); eat(':=')
            t = peek()
            if t is None: raise ParseFail('eof')
            if t[0] == 'h' and t[1:].isdigit():
                eat(); k = int(t[1:])
                if depth != 0 or seen_non_pr[0] or k != n_pr[0] + 1: raise ParseFail('PR position')
                n_pr[0] += 1
                i = emit(0, f, 'PR', [])
            else:
                seen_non_pr[0] = True
                if t == '⟨':
                    eat(); a = ref(scope); eat(','); b = ref(scope); eat('⟩'); rule, refs, pre = 'ANDI', [a, b], None
                elif t in ('Or.inl', 'Or.inr'):
                    eat(); a = ref(scope); rule, refs = ('ORI1' if t == 'Or.inl' else 'ORI2'), [a]
                elif t == 'Classical.byContradiction':
                    eat(); eat('('); eat('fun'); eat('hh'); eat('=>'); a = ref(scope); eat('hh'); eat(')'); rule, refs = 'DN', [a]
                elif t == '(':
                    s, e, neg = box(scope, depth); rule, refs = ('NEGI' if neg else 'IMPI'), [s, e]
                elif t == 'Or.elim':
                    eat(); j = ref(scope); s1, e1, g1 = box(scope, depth); s2, e2, g2 = box(scope, depth)
                    if g1 or g2: raise ParseFail('ORE box')
                    rule, refs = 'ORE', [j, s1, e1, s2, e2]
                else:
                    a = ref(scope); t2 = peek()
                    if t2 == '.1': eat(); rule, refs = 'ANDE1', [a]
                    elif t2 == '.2': eat(); rule, refs = 'ANDE2', [a]
                    elif t2 == '.elim': eat(); rule, refs = 'BOTE', [a]
                    elif t2 == ';': rule, refs = 'R', [a]
                    else:
                        b = ref(scope)
                        if forms[a][0] == 'not': rule, refs = 'NEGE', [b, a]
                        else: rule, refs = 'IMPE', [a, b]
                i = emit(depth, f, rule, refs)
            eat(';')
            scope[nm] = i
            lastl = i
        return lastl

    top = {}
    lastl = stmts(top, 0, None)
    eat('exact')
    e = ref(top)
    if pos[0] != len(toks):
        raise ParseFail('trailing tokens')
    if lastl is None or e != lastl:
        raise ParseFail('exact not last')
    return ' '.join(out) + ' QED'
