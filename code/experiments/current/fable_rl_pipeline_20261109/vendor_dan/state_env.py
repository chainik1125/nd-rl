#!/usr/bin/env python3
"""state_env.py -- an AlphaProof-style step environment for the `lean_seq` fragment (run `state-env`, proposal 13).

What the policy sees is the **tactic state of the focused goal**: every hypothesis in scope as `name : formula`
(premises `h1..hm`, `have`s completed at this or an enclosing level, box binders), then `⊢ goal`.  Nothing else --
no theorem statement (the theorem *is* the initial state), no proof history (arm S; arm SH prepends the history).

What the policy does is one **action**, the literal `lean_seq` text of one step:

  have n<k> : F := <atomic term> ;                                  a `have` line
  have n<k> : F := ( fun ( n<j> : A ) => by                         opening an implication / negation box
  have n<k> : F := Or.elim n<i> ( fun ( n<j> : A ) => by            opening an Or.elim, first branch
  exact n<k>                                                        closing the focused goal

The environment applies the action syntactically and renders it back into `lean_seq` text.  `render(action)` is the
action itself except for `exact`, where the environment supplies the tokens that **close the box** -- they encode the
box structure, not a proof step, and the environment is what knows the structure:

  top level    exact n            -> `exact n`                                      (the proof is finished)
  imp box      exact n            -> `exact n ) ;`
  neg box      exact n            -> `exact ( n : False ) ) ;`
  Or.elim b1   exact n            -> `exact n ) ( fun ( n<next> : B ) => by`        (second branch opened)
  Or.elim b2   exact n            -> `exact n ) ;`

`n<next>` is `max name index used in this attempt so far + 1`, which is exactly `lean_seq`'s "numbered in order of
first appearance", so replaying a control proof's actions reproduces its text **byte for byte** (gate 1).

Everything is syntax: scope, hypotheses and the goal follow from the actions because every `have` declares its
formula.  There is **no symbolic type check** -- a wrong term is caught by Lean on the finished proof, exactly as in
the whole-proof loop (`lean_gate` / `lean_judge`; Lean alone decides, Dan 2026-09-27).  The checks the environment
does make are the ones `lean_tok.inverse`'s strict grammar makes, so an accepted attempt always parses:
cited names in scope; `exact n` only if `n` is the focused frame's last statement and its declared formula is the
goal; premise lines (`:= h<k>`) only at depth 0, before any other line, in order; a box binder whose formula is the
one its `have` declares.
"""
import functools, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nd_verify.verify import parse_formula
from nd2lean import parse_prompt
from lean_tok import ftoks, proof_tokens, inverse, ParseFail, MAXN

NDMAP = {'(': '(', ')': ')', '¬': '~', '∧': '&', '∨': 'v', '→': '>', 'P': 'P', 'Q': 'Q', 'R': 'R', 'S': 'S', 'False': 'F'}
BOT = ('bot',)


def is_name(t):
    return bool(t) and t[0] == 'n' and t[1:].isdigit() and 1 <= int(t[1:]) <= MAXN


def parse_ftoks(toks, i):
    """Lean formula tokens at toks[i:] -> (formula tuple, n tokens used)."""
    j = i
    nd = []
    while j < len(toks) and toks[j] in NDMAP:
        nd.append(NDMAP[toks[j]]); j += 1
    if not nd:
        raise ParseFail('formula')
    try:
        f, used = parse_formula(nd + ['$'], 0)
    except Exception:
        raise ParseFail('formula')
    return f, used


def seq_tokens(nd_body):
    """ND proof body -> the `lean_seq` token strings (names n1.. in order of first appearance), as encode_proof does."""
    order = {}
    out = []
    for t in proof_tokens(nd_body):
        if isinstance(t, tuple):
            k = order.setdefault(t[1], len(order) + 1)
            if k > MAXN:
                raise ValueError('too many names')
            out.append(f'n{k}')
        else:
            out.append(t)
    return out


# --------------------------------------------------------------------------- decomposition
def split_actions(toks, sink=None):
    """`lean_seq` token list -> the list of model-visible actions (see the module docstring).  Raises ParseFail
    outside the strict grammar.  The Or.elim second-branch opener is NOT an action: the environment supplies it;
    `sink`, if given, collects its binder name, in the order the branch-1 `exact` actions occur."""
    pos = [0]
    acts = []

    def peek():
        return toks[pos[0]] if pos[0] < len(toks) else None

    def eat(x=None):
        t = peek()
        if t is None or (x is not None and t != x):
            raise ParseFail(f'expected {x}')
        pos[0] += 1
        return t

    def name():
        t = eat()
        if not is_name(t):
            raise ParseFail('name')
        return t

    def formula():
        f, used = parse_ftoks(toks, pos[0])
        out = toks[pos[0]:pos[0] + used]
        pos[0] += used
        return out

    def box_head():
        out = [eat('('), eat('fun'), eat('('), name(), eat(':')] + formula() + [eat(')'), eat('=>'), eat('by')]
        return out

    def close_box():
        eat('exact')
        if peek() == '(':
            eat('('); n = name(); eat(':'); eat('False'); eat(')')
        else:
            n = name()
        eat(')')
        return ['exact', n]

    def stmts():
        while peek() == 'have':
            a = [eat('have'), name(), eat(':')] + formula() + [eat(':=')]
            t = peek()
            if t == '(':
                acts.append(a + box_head()); stmts(); acts.append(close_box()); eat(';')
            elif t == 'Or.elim':
                a += [eat('Or.elim'), name()]
                acts.append(a + box_head()); stmts(); acts.append(close_box())
                h2 = box_head()                 # environment-supplied second branch opener
                if sink is not None:
                    sink.append(h2[3])
                stmts(); acts.append(close_box()); eat(';')
            else:
                while peek() is not None and peek() != ';':
                    a.append(eat())
                a.append(eat(';'))
                acts.append(a)

    stmts()
    eat('exact')
    acts.append(['exact', name()])
    if pos[0] != len(toks):
        raise ParseFail('trailing tokens')
    return acts


# --------------------------------------------------------------------------- the environment
class Frame:
    __slots__ = ('kind', 'goal', 'pending', 'ore_right', 'htoks', 'names', 'last')

    def __init__(self, kind, goal, pending=None, ore_right=None):
        self.kind = kind            # top | imp | neg | or1 | or2
        self.goal = goal
        self.pending = pending      # (name, formula) of the parent `have` this box proves
        self.ore_right = ore_right
        self.htoks = []             # this frame's own hypothesis lines, each `name : F <nl>`
        self.names = {}             # name -> formula, this frame's own bindings
        self.last = None            # name of this frame's last statement (a box binder counts)


@functools.lru_cache(maxsize=8192)
def prompt_parts(prompt):
    """(premises, conclusion, the top frame's hypothesis lines) -- cached: k attempts share one theorem."""
    prem, concl = parse_prompt(prompt)
    lines = tuple(tuple([f'h{j + 1}', ':'] + ftoks(p) + ['<nl>']) for j, p in enumerate(prem))
    return prem, concl, lines


class Env:
    """One proof attempt.  `apply(action_tokens)` -> (ok, reason)."""

    def __init__(self, prompt, canon=False, base=0, assign=False):
        prem, concl, lines = prompt_parts(prompt)
        self.canon = canon
        self.base = base            # canonical names start at n<base+1> (the train-time name offset, drawn per attempt)
        self.assign = assign        # arm SN: the environment, not the model, names what a step introduces
        self.renamed = 0            # defining names the environment overrode (assign=True)
        self.defined = 0
        self.prompt = prompt
        self.prem = prem
        self.concl = concl
        top = Frame('top', concl)
        top.htoks = [list(x) for x in lines]
        self.frames = [top]
        self.text = []              # rendered lean_seq tokens
        self.hist = []              # model-visible action tokens, concatenated (arm SH)
        self.n_pr = 0
        self.seen_non_pr = False
        self.maxname = 0
        self.done = False
        self.failed = None
        self.steps = 0

    # ---- scope ----
    def lookup(self, n):
        for f in reversed(self.frames):
            if n in f.names:
                return f.names[n]
        return None

    def next_name(self, reserve=(), frames=None):
        """the canonical next name: `max index in scope (+ the pending `have`s) + 1`.  Under canonical naming the
        action's name token is a function of the state, which the global first-appearance numbering is not."""
        mx = self.base
        for f in (self.frames if frames is None else frames):
            for n in f.names:
                mx = max(mx, int(n[1:]))
            if f.pending:
                mx = max(mx, int(f.pending[0][1:]))
        for r in reserve:
            mx = max(mx, int(r))
        return mx + 1

    def add_hyp(self, fr, n, f):
        fr.htoks.append([n, ':'] + ftoks(f) + ['<nl>'])
        fr.names[n] = f
        fr.last = n

    # ---- observation ----
    def state_tokens(self):
        out = ['<st>']
        for f in self.frames:
            for line in f.htoks:
                out += line
        return out + ['⊢'] + ftoks(self.frames[-1].goal) + ['<act>']

    # ---- transition ----
    def apply(self, act):
        self.steps += 1
        try:
            self._apply(list(act))
        except ParseFail as e:
            self.failed = str(e)
            return False, str(e)
        except (IndexError, KeyError, ValueError) as e:     # any other malformed action ends the attempt, never the run
            self.failed = 'malformed'
            return False, 'malformed'
        self.hist += list(self._eff)
        return True, ''

    def _bump(self, n):
        self.maxname = max(self.maxname, int(n[1:]))

    def _apply(self, act):
        self._eff = act
        if not act:
            raise ParseFail('empty action')
        if act[0] == 'exact':
            if len(act) != 2 or not is_name(act[1]):
                raise ParseFail('exact form')
            n = act[1]
            fr = self.frames[-1]
            f = self.lookup(n)
            if f is None:
                raise ParseFail('unbound')
            if fr.last != n:
                raise ParseFail('exact not last')
            if f != fr.goal:
                raise ParseFail('exact goal mismatch')
            self._close(n)
            return
        if act[0] != 'have':
            raise ParseFail('action head')
        i = 1
        if i >= len(act) or not is_name(act[i]):
            raise ParseFail('have name')
        nm = act[i]; i += 1
        if i >= len(act) or act[i] != ':':
            raise ParseFail('colon')
        i += 1
        f, used = parse_ftoks(act, i); i += used
        if i >= len(act) or act[i] != ':=':
            raise ParseFail('assign')
        i += 1
        t = act[i] if i < len(act) else None
        if self.assign:
            act = self._assign_names(act, i)
            nm = act[1]
            self._eff = act
        if t is None:
            raise ParseFail('empty term')
        fr = self.frames[-1]
        if t[0] == 'h' and t[1:].isdigit():                       # premise line
            if i + 2 != len(act) or act[i + 1] != ';':
                raise ParseFail('PR form')
            k = int(t[1:])
            if len(self.frames) != 1 or self.seen_non_pr or k != self.n_pr + 1 or k > len(self.prem):
                raise ParseFail('PR position')
            self.n_pr += 1
            self._bump(nm)
            self.add_hyp(fr, nm, f)
            self.text += act
            return
        if t == '(':                                              # implication / negation box
            head = self._box_head(act, i)
            if head is None:
                raise ParseFail('box head')
            x, xf, end = head
            if end != len(act):
                raise ParseFail('box trailing')
            if f[0] == 'imp':
                kind, goal, want = 'imp', f[2], f[1]
            elif f[0] == 'not':
                kind, goal, want = 'neg', BOT, f[1]
            else:
                raise ParseFail('box on non-arrow')
            if xf != want:
                raise ParseFail('binder mismatch')
            self.seen_non_pr = True
            self._bump(nm); self._bump(x)
            self.text += act
            nf = Frame(kind, goal, pending=(nm, f))
            nf.htoks.append([x, ':'] + ftoks(xf) + ['<nl>'])
            nf.names[x] = xf
            nf.last = x
            self.frames.append(nf)
            return
        if t == 'Or.elim':                                        # Or.elim, first branch
            if i + 1 >= len(act) or not is_name(act[i + 1]):
                raise ParseFail('Or.elim name')
            j = act[i + 1]
            jf = self.lookup(j)
            if jf is None:
                raise ParseFail('unbound')
            if jf[0] != 'or':
                raise ParseFail('Or.elim on non-or')
            head = self._box_head(act, i + 2)
            if head is None:
                raise ParseFail('box head')
            x, xf, end = head
            if end != len(act):
                raise ParseFail('box trailing')
            if xf != jf[1]:
                raise ParseFail('binder mismatch')
            self.seen_non_pr = True
            self._bump(nm); self._bump(x)
            self.text += act
            nf = Frame('or1', f, pending=(nm, f), ore_right=jf[2])
            nf.htoks.append([x, ':'] + ftoks(xf) + ['<nl>'])
            nf.names[x] = xf
            nf.last = x
            self.frames.append(nf)
            return
        # atomic term
        if act[-1] != ';':
            raise ParseFail('no semicolon')
        self._check_atomic(act[i:-1])
        self.seen_non_pr = True
        self._bump(nm)
        self.add_hyp(fr, nm, f)
        self.text += act
        return

    def _assign_names(self, act, i):
        """arm SN: replace the name(s) this `have` action introduces by the canonical ones (`max in scope + 1`, from
        `self.base`).  Citations are untouched -- they name hypotheses the state shows."""
        a = list(act)
        nh = f'n{self.next_name()}'
        self.defined += 1; self.renamed += a[1] != nh
        a[1] = nh
        j = None
        if i < len(a) and a[i] == '(' and i + 3 < len(a):
            j = i + 3
        elif i < len(a) and a[i] == 'Or.elim' and i + 5 < len(a):
            j = i + 5
        if j is not None and is_name(a[j]):
            nb = f'n{self.next_name(reserve=[nh[1:]])}'
            self.defined += 1; self.renamed += a[j] != nb
            a[j] = nb
        if int(nh[1:]) > MAXN or (j is not None and int(a[j][1:]) > MAXN):
            raise ParseFail('names exhausted')
        return a

    def _box_head(self, act, i):
        """act[i:] == '( fun ( X : A ) => by' -> (X, A, end index) or None."""
        try:
            if act[i] != '(' or act[i + 1] != 'fun' or act[i + 2] != '(' or not is_name(act[i + 3]) or act[i + 4] != ':':
                return None
            x = act[i + 3]
            xf, used = parse_ftoks(act, i + 5)
            j = i + 5 + used
            if act[j] != ')' or act[j + 1] != '=>' or act[j + 2] != 'by':
                return None
            return x, xf, j + 3
        except (IndexError, ParseFail):
            return None

    def _check_atomic(self, tm):
        """the nine atomic term shapes of the grammar; every cited name must be in scope."""
        def ref(n):
            if not is_name(n) or self.lookup(n) is None:
                raise ParseFail('unbound')
        if not tm:
            raise ParseFail('empty term')
        if len(tm) == 1:
            ref(tm[0]); return
        if tm[0] == '⟨':
            if len(tm) != 5 or tm[2] != ',' or tm[4] != '⟩':
                raise ParseFail('term')
            ref(tm[1]); ref(tm[3]); return
        if tm[0] in ('Or.inl', 'Or.inr'):
            if len(tm) != 2:
                raise ParseFail('term')
            ref(tm[1]); return
        if tm[0] == 'Classical.byContradiction':
            if len(tm) != 8 or tm[1:5] != ['(', 'fun', 'hh', '=>'] or tm[6] != 'hh' or tm[7] != ')':
                raise ParseFail('term')
            ref(tm[5]); return
        if len(tm) == 2:
            ref(tm[0])
            if tm[1] in ('.1', '.2', '.elim'):
                return
            ref(tm[1]); return
        raise ParseFail('term')

    def _close(self, n):
        fr = self.frames[-1]
        if fr.kind == 'top':
            self.text += ['exact', n]
            self.done = True
            return
        if fr.kind == 'or1':
            if self.canon:
                k = self.next_name(frames=self.frames[:-1] + [Frame('x', None, pending=fr.pending)])
            else:
                k = self.maxname + 1
            if k > MAXN:
                raise ParseFail('names exhausted')
            self.maxname = max(self.maxname, k)
            b = f'n{k}'
            rt = ftoks(fr.ore_right)
            self.text += ['exact', n, ')', '(', 'fun', '(', b, ':'] + rt + [')', '=>', 'by']
            nf = Frame('or2', fr.goal, pending=fr.pending)
            nf.htoks.append([b, ':'] + rt + ['<nl>'])
            nf.names[b] = fr.ore_right
            nf.last = b
            self.frames[-1] = nf
            return
        if fr.kind == 'neg':
            self.text += ['exact', '(', n, ':', 'False', ')', ')', ';']
        else:                                                     # imp | or2
            self.text += ['exact', n, ')', ';']
        self.frames.pop()
        pn, pf = fr.pending
        self.add_hyp(self.frames[-1], pn, pf)

    # ---- result ----
    def nd(self):
        """the ND proof the rendered text denotes, or 'LEANPARSE <reason>'."""
        if not self.done:
            return 'LEANPARSE env: ' + (self.failed or 'incomplete')
        try:
            return inverse(self.text)
        except ParseFail as e:
            return f'LEANPARSE {e}'


def canonicalise(prompt, nd_body):
    """`lean_seq` tokens rewritten so that every name a step *introduces* is `max index in scope + 1` -- a function of
    the state, unlike the global first-appearance index, which the state stops determining once a box has closed and
    taken its names out of scope (3.00 % of the control set's `have` actions).  The result is an alpha-variant: the ND
    proof `lean_tok.inverse` returns is unchanged, and Lean's verdict is invariant to hypothesis renaming."""
    toks = seq_tokens(nd_body)
    sink = []
    acts = split_actions(toks, sink)
    env = Env(prompt, canon=True)
    m = {}
    si = 0
    out = []
    for a in acts:
        if a[0] == 'exact':
            b2 = None
            if env.frames[-1].kind == 'or1':
                b2 = sink[si]; si += 1
            na = ['exact', m[a[1]]]
            ok, why = env.apply(na)
            if not ok:
                raise ParseFail(f'canon: {why}')
            if b2 is not None:
                m[b2] = env.frames[-1].last
        else:
            j = a.index(':=')
            na = list(a)
            nh = f'n{env.next_name()}'
            na[1] = nh
            if a[j + 1] == '(':
                nb = f'n{env.next_name(reserve=[nh[1:]])}'
                na[j + 4] = nb
                m[a[j + 4]] = nb
            elif a[j + 1] == 'Or.elim':
                na[j + 2] = m[a[j + 2]]
                nb = f'n{env.next_name(reserve=[nh[1:]])}'
                na[j + 6] = nb
                m[a[j + 6]] = nb
            else:
                for i in range(j + 1, len(a)):
                    if is_name(a[i]):
                        na[i] = m[a[i]]
            m[a[1]] = nh
            ok, why = env.apply(na)
            if not ok:
                raise ParseFail(f'canon: {why}')
        out.append(na)
    if not env.done:
        raise ParseFail('canon: did not finish')
    return env.text


def decompose(prompt, nd_body, canon=False):
    """-> (steps, tokens, env) where steps is a list of (state tokens, action tokens, history tokens).
    Raises ParseFail if the actions do not replay or do not reassemble the text byte for byte.  With `canon=True` the
    text is first rewritten by `canonicalise` (an alpha-variant denoting the same ND proof)."""
    toks = canonicalise(prompt, nd_body) if canon else seq_tokens(nd_body)
    acts = split_actions(toks)
    env = Env(prompt, canon=canon)
    steps = []
    for a in acts:
        st = env.state_tokens()
        hs = list(env.hist)
        ok, why = env.apply(a)
        if not ok:
            raise ParseFail(f'replay: {why}')
        steps.append((st, a, hs))
    if not env.done:
        raise ParseFail('replay did not finish')
    if env.text != toks:
        raise ParseFail('reassembly mismatch')
    return steps, toks, env
