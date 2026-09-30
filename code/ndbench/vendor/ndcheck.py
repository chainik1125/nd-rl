"""Per-sample outcome of a sampled proof: nd_verify's verdict plus where the proof first went wrong.

    classify(prompt, body) -> dict(ok, kind, line, reason, n_lines, start)

  ok       nd_verify.verify_text(prompt + ' ' + body)[0]   (the official verdict; n_ok counts this)
  kind     'ok'        the incremental checker accepts every line and QED closes a valid proof
           'bad_line'  line `line` (1-based position in the proof) is the first line the checker rejects;
                       `reason` says why (parse, index, index_overflow, premise, depth, arity, range, cite, rule)
           'early_qed' every line is valid but QED came after `line` lines without finishing the proof
                       (reason: inside_subproof, not_conclusion, premises_incomplete, empty, ends_with_as)
           'no_qed'    every complete line is valid but the sample ran out of tokens (max_new) after `line` lines
  n_lines  number of complete lines (';') in the body
  start    the first line's index N<start> (None if unparseable); the 'abs' tokenizer has only N1..N64, so a
           proof that starts at N<s> cannot have more than 65 - s lines ('index_overflow')

The incremental checker is run 2's LineChecker (ab2/nodes/n092/rl.py), which accepts a line exactly when
nd_verify would accept it given the lines before it, extended with reason codes. One known difference:
nd_verify accepts an IMPI/NEGI/ORE citation of a box that starts *after* the citing line; the checker rejects
it (refs must point backwards). `ok` always comes from nd_verify; disagreements are counted by the evaluator.
"""
import os, sys

_LIB = '/scratch/cowifornia/nd-rl/ab3/frozen/lib'
if _LIB not in sys.path:
    sys.path.insert(0, _LIB)
from nd_verify import verify_text, parse_formula as _pf, ParseError as _PE, RULE_NAMES as _RN  # noqa: E402
from nd_verify.verify import BOT as _BOT  # noqa: E402

MAXN = 64
_GARITY = {'ANDI': 2, 'ANDE1': 1, 'ANDE2': 1, 'IMPE': 2, 'IMPI': 2, 'ORI1': 1, 'ORI2': 1, 'ORE': 5, 'NEGE': 2,
           'NEGI': 2, 'BOTE': 1, 'DN': 1, 'R': 1}


def sequent(prompt):
    toks = prompt.split()
    prem, i = [], 1
    if toks[i] != 'SEQ':
        while True:
            f, i = _pf(toks, i)
            prem.append(f)
            if toks[i] == ',':
                i += 1
                continue
            break
    concl, _ = _pf(toks, i + 1)
    return prem, concl


class LineChecker:
    """Incremental nd_verify. add_line(toks) -> None if nd_verify would accept the line given the lines before
    it, else a reason code (and the line is not added). finish() -> None if QED here gives a valid proof."""

    def __init__(self, prem, concl):
        self.prem, self.concl = prem, concl
        self.lines, self.stack, self.nb = [], [], 0
        self.ctx, self.box_lines, self.box_depth, self.as_lines, self.fml = {}, {}, {}, {}, {}
        self.n_pr, self.in_pr = 0, True

    def add_line(self, toks):
        try:
            t = toks[0]
            if not (t.startswith('N') and t[1:].isdigit()):
                return 'index'       # the line does not start with a line index
            idx, i, d = int(t[1:]), 1, 0
            while i < len(toks) and toks[i] == '|':
                d += 1
                i += 1
            G, i = _pf(toks, i)
            if toks[i] != ':' or toks[i + 1] not in _RN:
                return 'parse'
            rule, i = toks[i + 1], i + 2
            refs = []
            while i < len(toks) and toks[i].startswith('N') and toks[i][1:].isdigit():
                refs.append(int(toks[i][1:]))
                i += 1
            if i != len(toks) - 1 or toks[i] != ';':
                return 'parse'
        except (_PE, IndexError, KeyError):
            return 'parse'
        L = self.lines
        idx0 = L[0]['idx'] if L else idx
        if idx < 1 or idx != idx0 + len(L):
            return 'index'
        stack = list(self.stack)
        if rule == 'PR':
            if not self.in_pr or d != 0 or self.n_pr >= len(self.prem) or G != self.prem[self.n_pr] or refs:
                return 'premise'
        elif self.n_pr != len(self.prem):
            return 'premise'
        new_box = None
        if rule == 'AS':
            if refs:
                return 'arity'
            if d < 1 or d > len(stack) + 1:
                return 'depth'
            del stack[d - 1:]
            new_box = self.nb + 1
            stack.append(new_box)
        else:
            if d > len(stack):
                return 'depth'
            del stack[d:]
        ci = tuple(stack)
        if rule not in ('PR', 'AS'):
            if len(refs) != _GARITY[rule]:
                return 'arity'
            if any(not (idx0 <= r < idx) for r in refs):
                return 'range'
            ctx, F = self.ctx, self.fml

            def lc(j):
                return ctx[j] == ci[:len(ctx[j])]

            def bc(s, e):
                if s not in self.as_lines:
                    return False
                b = self.as_lines[s]
                if e != self.box_lines.get(b) or L[e - idx0]['depth'] != self.box_depth[b] or e < s or b in ci:
                    return False
                par = ctx[s][:-1]
                return par == ci[:len(par)]
            if rule in ('IMPI', 'NEGI'):
                if not bc(refs[0], refs[1]):
                    return 'cite'
            elif rule == 'ORE':
                if not lc(refs[0]) or not bc(refs[1], refs[2]) or not bc(refs[3], refs[4]):
                    return 'cite'
            elif not all(lc(r) for r in refs):
                return 'cite'
            f = lambda k: F[refs[k]]
            try:
                if rule == 'R':
                    ok = G == f(0)
                elif rule == 'ANDI':
                    ok = G == ('and', f(0), f(1))
                elif rule == 'ANDE1':
                    ok = f(0)[0] == 'and' and f(0)[1] == G
                elif rule == 'ANDE2':
                    ok = f(0)[0] == 'and' and f(0)[2] == G
                elif rule == 'IMPE':
                    ok = f(0) == ('imp', f(1), G)
                elif rule == 'IMPI':
                    ok = G == ('imp', f(0), f(1))
                elif rule == 'ORI1':
                    ok = G[0] == 'or' and G[1] == f(0)
                elif rule == 'ORI2':
                    ok = G[0] == 'or' and G[2] == f(0)
                elif rule == 'ORE':
                    ok = f(0) == ('or', f(1), f(3)) and f(2) == G and f(4) == G
                elif rule == 'NEGE':
                    ok = f(1) == ('not', f(0)) and G == _BOT
                elif rule == 'NEGI':
                    ok = f(1) == _BOT and G == ('not', f(0))
                elif rule == 'BOTE':
                    ok = f(0) == _BOT
                else:  # DN
                    ok = f(0) == ('not', ('not', G))
            except (TypeError, IndexError):
                ok = False
            if not ok:
                return 'rule'
        # commit
        if rule == 'PR':
            self.n_pr += 1
        else:
            self.in_pr = False
        if new_box is not None:
            self.nb = new_box
            self.box_depth[new_box] = d
            self.as_lines[idx] = new_box
        self.stack = stack
        for b in stack:
            self.box_lines[b] = idx
        self.ctx[idx] = ci
        self.fml[idx] = G
        L.append({'idx': idx, 'depth': d, 'rule': rule})
        return None

    def finish(self):
        if not self.lines:
            return 'empty'
        if self.n_pr != len(self.prem):
            return 'premises_incomplete'
        last = self.lines[-1]
        if last['depth'] != 0:
            return 'inside_subproof'
        if last['rule'] == 'AS':
            return 'ends_with_as'
        if self.fml[last['idx']] != self.concl:
            return 'not_conclusion'
        if last['rule'] == 'PR' and self.concl not in self.prem:
            return 'not_conclusion'
        return None


def _overflow(start, p):
    """Line p (1-based) of a proof that starts at N<start> would need an index token beyond N64."""
    return start is not None and start + p - 1 > MAXN


def classify(prompt, body, seq=None):
    """Outcome of one sampled proof body (decoded text after PRF, QED-terminated unless it ran out)."""
    ok = bool(verify_text(prompt + ' ' + body)[0])
    toks = body.split()
    n_lines = toks.count(';')
    start = None
    if toks and toks[0].startswith('N') and toks[0][1:].isdigit():
        start = int(toks[0][1:])
    prem, concl = seq if seq is not None else sequent(prompt)
    ck = LineChecker(prem, concl)
    has_qed = 'QED' in toks
    if has_qed:
        toks = toks[:toks.index('QED')]
    out = dict(ok=ok, n_lines=n_lines, start=start)
    cur, pos = [], 0
    for t in toks:
        cur.append(t)
        if t == ';':
            pos += 1
            why = ck.add_line(cur)
            if why is not None:
                return dict(out, kind='bad_line', line=pos, reason='index_overflow' if _overflow(start, pos) else why)
            cur = []
    if cur:           # a partial line: right before QED (malformed), or cut off by max_new
        if has_qed or _overflow(start, pos + 1):
            return dict(out, kind='bad_line', line=pos + 1, reason='index_overflow' if _overflow(start, pos + 1) else 'parse')
    if not has_qed:   # generation stops only at QED or max_new
        return dict(out, kind='no_qed', line=pos, reason='max_new')
    why = ck.finish()
    if why is not None:
        if start is not None and start + pos - 1 >= MAXN:
            why = 'index_limit'      # stopped at N64: there is no N65 token to continue with
        return dict(out, kind='early_qed', line=pos, reason=why)
    return dict(out, kind='ok', line=pos, reason='')
