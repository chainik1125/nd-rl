"""Self-written goal annotations for Fitch-style natural-deduction proofs (ab3/goals).

A proof body is the text after PRF: lines `N<i> [|]*d <formula> : <RULE> [N<r> ...] ;` ended by QED.
An annotated body may contain annotation groups `GOAL <formula>` at line boundaries (right after PRF, i.e. at
the start of the body, or right after a `;`). The verifier never sees them: strip() removes them first.

    parse_body(prompt, body)       -> Proof (lines, contexts, boxes) of a proof that parses (no validity check)
    obligations(proof)             -> the goal agenda of a VERIFIED proof (see DESIGN.md, "The agenda")
    displays(proof, policy)        -> {boundary: goal formula} (boundary = number of lines written before it)
    annotate(prompt, body, policy) -> annotated body (canonical annotation; body must verify)
    strip(body)                    -> body without well-formed annotation groups at line boundaries
    read_annotations(body)         -> (plain body, {boundary: [formula strings]}, n_malformed)
    consistency(prompt, body)      -> how the model's own annotations compare with the canonical ones of its
                                      (stripped, verified) proof

Boundaries: boundary k is the point after k lines (k = 0: before the first line). The canonical annotation
writes at most one group per boundary and never at boundary n (after the last line, right before QED).

Nothing here imports torch.
"""
import os, sys

_LIB = '/scratch/cowifornia/nd-rl/ab3/frozen/lib'
if _LIB not in sys.path:      # appended: a harness copy's own lib/ (same files) stays first
    sys.path.append(_LIB)
from nd_verify import verify_text, parse_formula, ParseError  # noqa: E402
from nd_verify.verify import parse_proof_tokens, BOT  # noqa: E402

GOAL = 'GOAL'
POLICIES = ('goals', 'start', 'every', 'boxes', 'naive')
DEFAULT_POLICY = 'goals'


# ---------------------------------------------------------------- formulas

def fstr(f):
    """Formula tuple -> canonical token string (the dataset's fully parenthesised format)."""
    k = f[0]
    if k == 'atom':
        return f[1]
    if k == 'bot':
        return 'F'
    if k == 'not':
        return f'( ~ {fstr(f[1])} )'
    op = {'and': '&', 'or': 'v', 'imp': '>'}[k]
    return f'( {fstr(f[1])} {op} {fstr(f[2])} )'


def sequent(prompt):
    """'THM a , b SEQ c PRF' -> (premise tuples, conclusion tuple)."""
    toks = prompt.split()
    prem, i = [], 1
    if toks[i] != 'SEQ':
        while True:
            f, i = parse_formula(toks, i)
            prem.append(f)
            if toks[i] == ',':
                i += 1
                continue
            break
    concl, _ = parse_formula(toks, i + 1)
    return prem, concl


# ---------------------------------------------------------------- proof structure

class Proof:
    """Parsed proof body: per line (0-based position p) idx, depth, formula, rule, refs (as positions), ctx
    (tuple of open box ids after the line, as nd_verify computes it); boxes: id -> dict(s, e, depth, level)."""

    def __init__(self, prompt, lines):
        self.prompt = prompt
        self.prem, self.concl = sequent(prompt)
        self.lines = lines
        self.n = len(lines)
        idx0 = lines[0]['idx'] if lines else 1
        self.idx0 = idx0
        stack, nb, ctx, boxes = [], 0, [], {}
        for p, ln in enumerate(lines):
            d = ln['depth']
            if ln['rule'] == 'AS':
                del stack[d - 1:]
                nb += 1
                stack.append(nb)
                boxes[nb] = {'s': p, 'depth': d}
            else:
                del stack[d:]
            ctx.append(tuple(stack))
            for b in stack:
                boxes[b]['e'] = p
        self.ctx = ctx
        for b, B in boxes.items():
            c = ctx[B['s']]
            B['level'] = [p for p in range(B['s'], B['e'] + 1) if ctx[p] == c]
        self.boxes = boxes
        self.box_of_as = {B['s']: b for b, B in boxes.items()}
        self.pos = [[r - idx0 for r in ln['refs']] for ln in lines]   # refs as positions
        self.n_prem_lines = 0
        for ln in lines:
            if ln['rule'] != 'PR':
                break
            self.n_prem_lines += 1


def parse_body(prompt, body):
    """Proof of a body that parses (nd_verify's parser + context tracking); raises ParseError otherwise."""
    toks = body.split()
    if 'QED' not in toks:
        raise ParseError('missing QED')
    lines = parse_proof_tokens(toks)
    for ln in lines:
        d = ln['depth']
        if ln['rule'] == 'AS' and d < 1:
            raise ParseError('AS at depth 0')
    return Proof(prompt, lines)


# ---------------------------------------------------------------- the agenda

class Ob:
    """One obligation: formula `f` to be derived in context `ctx`; pushed at position `push` (the line after which
    it becomes owed; -1 = before the first line), achieved at position `ach`. kind: root / box / sub."""
    __slots__ = ('f', 'ctx', 'push', 'ach', 'kind', 'box', 'parent')

    def __init__(self, f, ctx, push, ach, kind, box=None, parent=None):
        self.f, self.ctx, self.push, self.ach, self.kind, self.box, self.parent = f, ctx, push, ach, kind, box, parent

    def __repr__(self):
        return f'Ob({self.kind} {fstr(self.f)} push={self.push} ach={self.ach})'


def _available(P, p_push, ctx, f):
    """Is formula f already written on a line citable from context ctx at (and including) position p_push?"""
    for p in range(0, p_push + 1):
        c = P.ctx[p]
        if c == ctx[:len(c)] and P.lines[p]['formula'] == f:
            return True
    return False


def _split(P, o, out, splits=True):
    """Append o's pending conjunct sub-goals (recursively) to out, when o is achieved by ANDI."""
    if not splits:
        return
    a = o.ach
    if a < 0 or P.lines[a]['rule'] != 'ANDI':
        return
    seen = []
    for q in P.pos[a]:
        c = P.lines[q]['formula']
        if c in seen:
            continue
        seen.append(c)
        if _available(P, o.push, o.ctx, c):
            continue
        ach = next((p for p in range(o.push + 1, a) if P.ctx[p] == o.ctx and P.lines[p]['formula'] == c), None)
        if ach is None:      # cannot happen for a verified proof (the cited line is on o's level after the push)
            raise ValueError(f'sub-goal {fstr(c)} of {fstr(o.f)} has no achieving line')
        s = Ob(c, o.ctx, o.push, ach, 'sub', parent=o)
        out.append(s)
        _split(P, s, out, splits)


def obligations(P, splits=True):
    """All obligations of a verified proof, in push order. The root (the theorem's conclusion) is pushed after the
    premise block and achieved by the last line. Each box's goal is the formula of its last line on the box's own
    level (for a discharged box: the IMPI consequent, F for NEGI, the ORE case target); it is pushed at its AS line
    and achieved by that last level line. With splits, an obligation achieved by ANDI owes those conjuncts that
    are not already written on a citable line when it is pushed; each such conjunct is achieved by the first
    line on the same level that writes it."""
    obs = []
    if P.n == 0:
        return obs
    root = Ob(P.concl, (), P.n_prem_lines - 1, P.n - 1, 'root')
    obs.append(root)
    _split(P, root, obs, splits)
    for b in sorted(P.boxes):
        B = P.boxes[b]
        g = B['level'][-1]
        o = Ob(P.lines[g]['formula'], P.ctx[B['s']], B['s'], g, 'box', box=b)
        obs.append(o)
        _split(P, o, obs, splits)
    return obs


def agenda_trace(P, splits=True):
    """Per boundary k (0..n; boundary k = after k lines): (next, ready, box_end, pending) where
      pending  obligations pushed at a position < k and not yet achieved (achieving position >= k)
      next     the pending obligation achieved first (ties: the most recently pushed), or None
      ready    next has conjunct sub-goals and all of them are already achieved (only the ANDI step is left)
      box_end  line k-1 is the last line of a box, so line k must close that box (a discharge or a sibling AS)"""
    obs = obligations(P, splits)
    kids = {}
    for o in obs:
        if o.parent is not None:
            kids.setdefault(id(o.parent), []).append(o)
    ends = {B['e'] for B in P.boxes.values()}
    out = []
    for k in range(P.n + 1):
        pend = [o for o in obs if o.push < k and o.ach >= k]
        nxt = min(pend, key=lambda o: (o.ach, -o.push)) if pend else None
        ready = nxt is not None and id(nxt) in kids and all(s.ach < k for s in kids[id(nxt)])
        out.append((nxt, ready, (k - 1) in ends, pend))
    return out


def displays(P, policy=DEFAULT_POLICY):
    """{boundary k: goal formula tuple} of the canonical annotation under `policy` (see DESIGN.md). At boundary k
    (after the premise block, before the last line) the candidate is `next`, the pending obligation achieved first.
      goals    (the design) write it when it differs from the obligation written last: a new box's goal after its
               AS line, the next conjunct after one is achieved, the goal the proof returns to after a lemma box
               closes; except at a box end (the next line is forced: it closes the box), when it is ready (its
               conjuncts are all written: the next line is the ANDI), and the theorem's conclusion at the start
               (the prompt states it; it is written when the proof returns to it)
      start    as goals, and the theorem's conclusion is also written after the premise block
      every    at every boundary where start may write, the goal is written even if unchanged
      boxes    as goals without conjunct splits (box goals, and the conclusion when the proof returns to it)
      naive    first draft (kept for the record): after the premise block, every AS line and every line that
               achieves an obligation, without the box-end / ready suppression"""
    splits = policy != 'boxes'
    tr = agenda_trace(P, splits=splits)
    obs = obligations(P, True) if policy == 'naive' else None
    out, last = {}, None
    k0 = P.n_prem_lines
    for k in range(k0, P.n):
        nxt, ready, box_end, pend = tr[k]
        if nxt is None:
            continue
        if policy == 'naive':
            if k == k0 or P.lines[k - 1]['rule'] == 'AS' or any(o.ach == k - 1 and o.push < k - 1 for o in obs):
                out[k] = nxt.f
            continue
        if box_end or ready:
            continue
        if policy == 'every':
            out[k] = nxt.f
            continue
        if nxt is last:
            continue
        last = nxt
        if policy in ('goals', 'boxes') and nxt.kind == 'root' and k == k0:
            continue
        out[k] = nxt.f
    return out


def body_lines(body):
    """Token lists of the complete lines of a plain body (up to QED)."""
    toks = body.split()
    if 'QED' in toks:
        toks = toks[:toks.index('QED')]
    lines, cur = [], []
    for t in toks:
        cur.append(t)
        if t == ';':
            lines.append(cur)
            cur = []
    return lines, cur


def render(body, disp):
    """Insert `GOAL <formula>` groups into a plain body at the given boundaries ({k: formula tuple or string})."""
    lines, rest = body_lines(body)
    out = []
    for k, ln in enumerate(lines):
        if k in disp:
            g = disp[k]
            out += [GOAL] + (fstr(g) if isinstance(g, tuple) else g).split()
        out += ln
    if len(lines) in disp:
        g = disp[len(lines)]
        out += [GOAL] + (fstr(g) if isinstance(g, tuple) else g).split()
    out += rest
    if 'QED' in body.split():
        out.append('QED')
    return ' '.join(out)


def annotate(prompt, body, policy=DEFAULT_POLICY, check=True):
    """Canonical annotation of a verified plain proof body."""
    if check:
        ok, reason, _ = verify_text(prompt + ' ' + body)
        if not ok:
            raise ValueError(f'annotate: proof does not verify ({reason})')
    P = parse_body(prompt, body)
    return render(body, displays(P, policy))


# ---------------------------------------------------------------- stripping

def _is_line_start(t):
    return t.startswith('N') and t[1:].isdigit()


def read_annotations(body):
    """Split an annotated body into (plain body, {boundary: [goal strings]}, n_malformed).
    A well-formed group is GOAL followed by one complete formula, at a line boundary (the start of the body or
    right after ';'); any number of groups may follow each other there. Anything else containing GOAL (a GOAL
    inside a line, or a GOAL whose formula does not parse) is left in the plain body, where the verifier rejects
    it like any other malformed token; n_malformed counts those GOAL tokens."""
    toks = body.split()
    out, ann, bad = [], {}, 0
    i, k, at_boundary = 0, 0, True
    while i < len(toks):
        t = toks[i]
        if t == GOAL and at_boundary:
            try:
                f, j = parse_formula(toks, i + 1)
            except (ParseError, IndexError, KeyError):
                f = None
            if f is not None:
                ann.setdefault(k, []).append(' '.join(toks[i + 1:j]))
                i = j
                continue
            bad += 1
            out.append(t)
            at_boundary = False
            i += 1
            continue
        if t == GOAL:
            bad += 1
        out.append(t)
        if t == ';':
            k += 1
            at_boundary = True
        elif t == 'QED':
            at_boundary = False
        else:
            at_boundary = False
        i += 1
    return ' '.join(out), ann, bad


def strip(body):
    """Annotated body -> plain body (well-formed annotation groups at line boundaries removed)."""
    return read_annotations(body)[0]


# ---------------------------------------------------------------- consistency of the model's own annotations

def consistency(prompt, body, policy=DEFAULT_POLICY):
    """Compare the annotations the model wrote in `body` with the canonical annotation of its own stripped proof.
    Only meaningful for a proof that verifies after stripping (returns None otherwise).
      exact        the annotated text equals the canonical annotation exactly
      n_written    annotation groups written; n_canon: canonical groups
      matched      canonical boundaries where the model wrote exactly the canonical goal (one group)
      pos_match    canonical boundaries where the model wrote some group (any formula)
      extra        groups at boundaries where the canonical annotation has none
      valid_goal   written groups whose formula is a goal on the canonical agenda pending at that boundary
                   (any of the pending obligations, not only the next one)"""
    plain, ann, bad = read_annotations(body)
    ok, _, _ = verify_text(prompt + ' ' + plain)
    if not ok:
        return None
    P = parse_body(prompt, plain)
    can = {k: fstr(f) for k, f in displays(P, policy).items()}
    obs = obligations(P, splits=(policy != 'boxes'))
    n_written = sum(len(v) for v in ann.values())
    matched = sum(1 for k, g in can.items() if ann.get(k) == [g])
    pos_match = sum(1 for k in can if k in ann)
    extra = sum(len(v) for k, v in ann.items() if k not in can)
    valid = 0
    for k, gs in ann.items():
        pend = {fstr(o.f) for o in obs if o.push < k and o.ach >= k}
        valid += sum(1 for g in gs if g in pend)
    return {'exact': (render(plain, can) == ' '.join(body.split())), 'n_written': n_written, 'n_canon': len(can),
            'matched': matched, 'pos_match': pos_match, 'extra': extra, 'valid_goal': valid, 'malformed': bad}
