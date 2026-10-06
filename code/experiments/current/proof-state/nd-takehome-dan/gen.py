#!/usr/bin/env python3
"""Procedural random generator of verifier-valid natural-deduction proofs.

How it samples (see writeup.md, Stage 1):
  * Forward mode: start from 0-3 random premises, apply randomly chosen rules to the
    lines that are currently citable, open/close subproof boxes at random (IMPI, NEGI,
    ORE all appear). When a rule needs a formula that is not available (the antecedent
    of an implication, the negation for NEGE, ...) a *premise is introduced lazily*
    with some probability; this is what makes the premises of a theorem fit together.
  * Goal mode: sample a random conclusion and *complete* it with `reach()`, a randomised
    routine that decomposes the goal by its main connective (ANDI / ORI / IMPI-box /
    NEGI-box) and otherwise uses an available line, ex falso, DN, or a lazy premise
    `(Z > G)`. It never fails and never backtracks, so it is a generator, not a prover.
  * The second branch of every ORE is completed with `reach()` to the first branch's end.
  * Every proof is dependency-pruned (only lines the conclusion transitively cites are
    kept; unused premises are dropped), then checked with nd_verify.verify_text.
    Anything the verifier rejects is discarded and counted.

Theorem key: `thm` = "p1 , p2 |- c" (formula token strings). Renaming class key:
atoms relabelled in order of first appearance.

  python gen.py --n 200000 --min 2 --max 6 --out data/raw_cap6.jsonl --seed 1
  python gen.py --n 20000 --min 7 --max 16 --out data/raw_long.jsonl --seed 2 --long
"""
import argparse, json, random, sys, os, collections, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nd_verify import verify_text

ATOMS = ['P', 'Q', 'R', 'S']
BOT = ('bot',)


def fstr(f):
    if f[0] == 'atom':
        return f[1]
    if f[0] == 'bot':
        return 'F'
    if f[0] == 'not':
        return f'( ~ {fstr(f[1])} )'
    op = {'and': '&', 'or': 'v', 'imp': '>'}[f[0]]
    return f'( {fstr(f[1])} {op} {fstr(f[2])} )'


def fsize(f):
    if f[0] in ('atom', 'bot'):
        return 1
    if f[0] == 'not':
        return 1 + fsize(f[1])
    return 1 + fsize(f[1]) + fsize(f[2])


class Line:
    __slots__ = ('depth', 'f', 'rule', 'refs', 'idx', 'keep', 'lazy')

    def __init__(self, depth, f, rule, refs=()):
        self.depth, self.f, self.rule, self.refs = depth, f, rule, list(refs)
        self.idx = None
        self.keep = False
        self.lazy = False   # premise added after reset() (a lazy premise), for the shape table only


class Fail(Exception):
    pass


class Gen:
    def __init__(self, rng, max_prem=3, max_depth=3, fdepth=2, atoms=ATOMS, bot_p=0.02):
        self.rng = rng
        self.max_prem = max_prem
        self.max_depth = max_depth
        self.fdepth = fdepth
        self.atoms = atoms
        self.bot_p = bot_p
        self.ore_steps = 3
        self.strict = False   # long pools: no lazy (Z>G)/G fallback, no F, fewer garbage intros
        # ds-generator knobs (2026-09-22). Defaults reproduce the take-home generator byte for byte.
        self.ore_boxes = False  # G1: ORE branch 1 takes full step()s (boxes / nested ORE allowed under max_depth) instead of step_local()
        self.floor = 0          # box depth below which step() may not 'close' (the open ORE branch); 0 = take-home behaviour

    # ---------- formulas ----------
    def rf(self, depth=None):
        rng = self.rng
        if depth is None:
            depth = self.fdepth
        if depth == 0 or rng.random() < 0.45:
            if rng.random() < self.bot_p:
                return BOT
            return ('atom', rng.choice(self.atoms))
        r = rng.random()
        if r < 0.28:
            return ('not', self.rf(depth - 1))
        op = rng.choice(['and', 'or', 'imp'])
        return (op, self.rf(depth - 1), self.rf(depth - 1))

    # ---------- state ----------
    def reset(self, n_prem):
        self.prem = []          # premise Line objects (depth 0, rule PR)
        self.body = []          # all non-premise lines in order
        # levels[0] = depth-0 derived lines; levels[d] = lines inside the open box at depth d
        self.levels = [[]]
        self.boxes = []         # open boxes: (AS line)
        self.closed = [[]]      # per level: closed boxes (s, e) citable at that level
        for _ in range(n_prem):
            self.add_premise(self.rf(), lazy=False)

    @property
    def depth(self):
        return len(self.boxes)

    def add_premise(self, f, lazy=True):
        for p in self.prem:
            if p.f == f:
                return p
        if len(self.prem) >= self.max_prem:
            raise Fail('premise cap')
        ln = Line(0, f, 'PR')
        ln.lazy = lazy
        self.prem.append(ln)
        return ln

    def avail(self):
        out = list(self.prem)
        for lv in self.levels:
            out.extend(lv)
        return out

    def find(self, f):
        for ln in self.avail():
            if ln.f == f:
                return ln
        return None

    def emit(self, f, rule, refs=()):
        ln = Line(self.depth, f, rule, refs)
        self.body.append(ln)
        self.levels[-1].append(ln)
        return ln

    def open_box(self, hyp):
        if self.depth >= self.max_depth:
            raise Fail('depth cap')
        ln = Line(self.depth + 1, hyp, 'AS')
        self.body.append(ln)
        self.boxes.append(ln)
        self.levels.append([ln])
        self.closed.append([])
        return ln

    def close_box(self, rule=None):
        """Close the innermost box with IMPI (or NEGI if it ends in F). Returns new line."""
        s = self.boxes.pop()
        lines = self.levels.pop()
        self.closed.pop()
        e = lines[-1]
        if rule is None:
            rule = 'NEGI' if (e.f == BOT and self.rng.random() < 0.8) else 'IMPI'
        if rule == 'NEGI' and e.f != BOT:
            rule = 'IMPI'
        G = ('not', s.f) if rule == 'NEGI' else ('imp', s.f, e.f)
        self.closed[-1].append((s, e))
        return self.emit(G, rule, [s, e])

    def last_at_level(self):
        return self.levels[-1][-1] if self.levels[-1] else None

    # ---------- goal completion ----------
    def reach(self, G, budget):
        """Append lines so that G is the last line at the current level. Never fails."""
        rng = self.rng
        last = self.last_at_level()
        if last is not None and last.f == G:
            return last
        have = self.find(G)
        if have is not None:
            return self.emit(G, 'R', [have])
        opts = []
        if budget > 0:
            if G[0] == 'and':
                opts.append('andi')
            if G[0] == 'or':
                opts.append('ori')
            if G[0] == 'imp' and self.depth < self.max_depth:
                opts.append('impi')
            if G[0] == 'not' and self.depth < self.max_depth:
                opts.append('negi')
        bot = self.find(BOT)
        if bot is not None and G != BOT:
            opts.append('bote')
        if G == BOT:
            for ln in self.avail():
                if ln.f[0] == 'not' and self.find(ln.f[1]) is not None:
                    opts.append(('nege', self.find(ln.f[1]), ln))
            if not any(isinstance(o, tuple) and o[0] == 'nege' for o in opts) and self.avail() and len(self.prem) < self.max_prem:
                opts.append('nege_lazy')
        elif budget > 0 and self.depth < self.max_depth and rng.random() < 0.5:
            opts.append('raa')
        dn = self.find(('not', ('not', G)))
        if dn is not None:
            opts.append('dn')
        for ln in self.avail():
            if ln.f[0] == 'imp' and ln.f[2] == G and self.find(ln.f[1]) is not None:
                opts.append(('impe', ln))
            if ln.f[0] == 'and' and G in (ln.f[1], ln.f[2]):
                opts.append(('ande', ln))
        if len(self.prem) < self.max_prem:
            opts.append('lazy')
        if not opts:
            opts.append('lazy')  # will raise Fail via premise cap
        o = rng.choice(opts)
        if o == 'andi':
            a = self.reach(G[1], budget - 1)
            b = self.reach(G[2], budget - 1)
            return self.emit(G, 'ANDI', [a, b])
        if o == 'ori':
            side = 1 if self.find(G[1]) is not None else (2 if self.find(G[2]) is not None else rng.choice([1, 2]))
            a = self.reach(G[side], budget - 1)
            return self.emit(G, 'ORI1' if side == 1 else 'ORI2', [a])
        if o == 'impi':
            self.open_box(G[1])
            self.reach(G[2], budget - 1)
            return self.close_box('IMPI')
        if o == 'negi':
            self.open_box(G[1])
            self.reach(BOT, budget - 1)
            return self.close_box('NEGI')
        if o == 'bote':
            return self.emit(G, 'BOTE', [bot])
        if o == 'raa':
            self.open_box(('not', G))
            self.reach(BOT, budget - 1)
            nn = self.close_box('NEGI')
            return self.emit(G, 'DN', [nn])
        if isinstance(o, tuple) and o[0] == 'nege':
            return self.emit(BOT, 'NEGE', [o[1], o[2]])
        if o == 'nege_lazy':
            x = last if (last is not None and rng.random() < 0.7) else rng.choice(self.avail())
            if x.f[0] == 'not' and rng.random() < 0.5:
                p = self.add_premise(x.f[1])
                return self.emit(BOT, 'NEGE', [p, x])
            p = self.add_premise(('not', x.f))
            return self.emit(BOT, 'NEGE', [x, p])
        if o == 'dn':
            return self.emit(G, 'DN', [dn])
        if isinstance(o, tuple) and o[0] == 'impe':
            ln = o[1]
            return self.emit(G, 'IMPE', [ln, self.find(ln.f[1])])
        if isinstance(o, tuple) and o[0] == 'ande':
            ln = o[1]
            return self.emit(G, 'ANDE1' if ln.f[1] == G else 'ANDE2', [ln])
        # lazy premise: (Z > G) for some available Z (prefer the current box's last line)
        if self.strict:
            raise Fail('strict: no lazy goal premise')
        av = self.avail()
        if av and rng.random() < 0.85:
            z = last if (last is not None and rng.random() < 0.6) else rng.choice(av)
            p = self.add_premise(('imp', z.f, G))
            return self.emit(G, 'IMPE', [p, z])
        p = self.add_premise(G)
        return self.emit(G, 'R', [p])

    # ---------- forward step ----------
    def step(self):
        rng = self.rng
        av = self.avail()
        acts = []
        if self.depth < self.max_depth:
            acts += ['as'] * 3
        if self.depth > self.floor and len(self.levels[-1]) >= 1:
            acts += ['close'] * 3
        if av:
            if self.strict:
                acts += ['andi', 'ori'] + ['impe'] * 4 + ['nege'] * 3 + ['ande'] * 3 + ['dn'] * 2 + ['bote'] + ['ore'] * 6
            else:
                acts += ['andi'] * 2 + ['ori'] * 2 + ['impe'] * 4 + ['nege'] * 3 + ['ande'] * 2 + ['dn'] * 2 + ['bote'] + ['ore'] * 5
            if self.depth > 0:
                acts.append('r')
        else:
            acts += ['as'] * 2
        a = rng.choice(acts)
        if a == 'as':
            hyp = None
            r = rng.random()
            imps = [l for l in av if l.f[0] == 'imp']
            negs = [l for l in av if l.f[0] == 'not']
            if r < 0.3 and imps:
                hyp = rng.choice(imps).f[1]
            elif r < 0.5 and negs:
                hyp = rng.choice(negs).f[1]
            elif r < 0.6:
                hyp = ('not', ('not', self.rf(1)))
            elif r < 0.7 and av:
                hyp = ('not', rng.choice(av).f)
            if hyp is None:
                hyp = self.rf()
            self.open_box(hyp)
            return
        if a == 'close':
            self.close_box()
            return
        if a == 'r':
            outer = list(self.prem)
            for lv in self.levels[:-1]:
                outer.extend(lv)
            if outer:
                self.emit_new(rng.choice(outer).f, 'R', [rng.choice(outer)])
            return
        if a == 'andi':
            x, y = rng.choice(av), rng.choice(av)
            self.emit_new(('and', x.f, y.f), 'ANDI', [x, y])
            return
        if a == 'ori':
            x = rng.choice(av)
            other = self.rf()
            if rng.random() < 0.5:
                self.emit_new(('or', x.f, other), 'ORI1', [x])
            else:
                self.emit_new(('or', other, x.f), 'ORI2', [x])
            return
        if a == 'ande':
            cs = [l for l in av if l.f[0] == 'and']
            if cs:
                c = rng.choice(cs)
                if rng.random() < 0.5:
                    self.emit_new(c.f[1], 'ANDE1', [c])
                else:
                    self.emit_new(c.f[2], 'ANDE2', [c])
            return
        if a == 'dn':
            ds = [l for l in av if l.f[0] == 'not' and l.f[1][0] == 'not']
            if ds:
                d = rng.choice(ds)
                self.emit_new(d.f[1][1], 'DN', [d])
            return
        if a == 'bote':
            b = self.find(BOT)
            if b is not None:
                self.emit_new(self.rf(), 'BOTE', [b])
            return
        if a == 'impe':
            imps = [l for l in av if l.f[0] == 'imp']
            if imps and rng.random() < 0.7:
                im = rng.choice(imps)
                ant = self.find(im.f[1])
                if ant is None:
                    if rng.random() < 0.6 and len(self.prem) < self.max_prem:
                        ant = self.add_premise(im.f[1])
                    else:
                        return
                self.emit_new(im.f[2], 'IMPE', [im, ant])
            elif not self.strict:
                x = rng.choice(av)
                if len(self.prem) < self.max_prem and rng.random() < 0.6:
                    im = self.add_premise(('imp', x.f, self.rf()))
                    self.emit_new(im.f[2], 'IMPE', [im, x])
            return
        if a == 'nege':
            for l in av:
                if l.f[0] == 'not':
                    pos = self.find(l.f[1])
                    if pos is not None:
                        self.emit_new(BOT, 'NEGE', [pos, l])
                        return
            x = rng.choice(av)
            if x.f[0] == 'not' and rng.random() < 0.5:
                if len(self.prem) < self.max_prem and rng.random() < 0.5:
                    p = self.add_premise(x.f[1])
                    self.emit_new(BOT, 'NEGE', [p, x])
            elif len(self.prem) < self.max_prem and rng.random() < 0.5:
                p = self.add_premise(('not', x.f))
                self.emit_new(BOT, 'NEGE', [x, p])
            return
        if a == 'ore':
            ors = [l for l in av if l.f[0] == 'or']
            if not ors:
                if len(self.prem) < self.max_prem and rng.random() < 0.5:
                    ors = [self.add_premise(('or', self.rf(), self.rf()))]
                else:
                    return
            if self.depth >= self.max_depth:
                return
            d = rng.choice(ors)
            A, B = d.f[1], d.f[2]
            s1 = self.open_box(A)
            if self.ore_boxes:
                # G1: full forward steps inside the branch (AS/close/ORE allowed under the global depth cap); the branch box
                # itself cannot be closed by step() (floor), and boxes still open at the end are closed here.
                fl = self.floor; self.floor = self.depth
                for _ in range(rng.randint(0, self.ore_steps)):
                    self.step()
                while self.depth > self.floor:
                    self.close_box()
                self.floor = fl
            else:
                for _ in range(rng.randint(0, self.ore_steps)):
                    self.step_local()
            e1 = self.levels[-1][-1]
            G = e1.f
            self.boxes.pop(); self.levels.pop(); self.closed.pop()
            s2 = self.open_box(B)
            self.reach(G, 2)
            e2 = self.levels[-1][-1]
            self.boxes.pop(); self.levels.pop(); self.closed.pop()
            self.emit(G, 'ORE', [d, s1, e1, s2, e2])
            return

    def step_local(self):
        """A forward step that does not change box structure (used inside ORE branch 1)."""
        d = self.depth
        for _ in range(4):
            self.step_no_box()
            if self.depth != d:
                break

    def step_no_box(self):
        rng = self.rng
        av = self.avail()
        a = rng.choice(['andi', 'ori', 'impe', 'nege', 'ande', 'dn', 'bote', 'impe', 'ande'])
        saved = (self.depth, self.max_depth)
        self.max_depth = self.depth  # prevents 'as'/'ore' from opening boxes
        try:
            # reuse step() but with box-changing actions disabled by depth cap
            if a in ('andi', 'ori', 'impe', 'nege', 'ande', 'dn', 'bote'):
                self._local(a, av)
        finally:
            self.max_depth = saved[1]

    def _local(self, a, av):
        # thin wrapper so step()'s local branches can be reused
        rng = self.rng
        if a == 'andi':
            x, y = rng.choice(av), rng.choice(av)
            self.emit_new(('and', x.f, y.f), 'ANDI', [x, y])
        elif a == 'ori':
            x = rng.choice(av)
            other = self.rf()
            if rng.random() < 0.5:
                self.emit_new(('or', x.f, other), 'ORI1', [x])
            else:
                self.emit_new(('or', other, x.f), 'ORI2', [x])
        elif a == 'ande':
            cs = [l for l in av if l.f[0] == 'and']
            if cs:
                c = rng.choice(cs)
                side = rng.random() < 0.5
                self.emit_new(c.f[1] if side else c.f[2], 'ANDE1' if side else 'ANDE2', [c])
        elif a == 'dn':
            ds = [l for l in av if l.f[0] == 'not' and l.f[1][0] == 'not']
            if ds:
                d = rng.choice(ds)
                self.emit_new(d.f[1][1], 'DN', [d])
        elif a == 'bote':
            b = self.find(BOT)
            if b is not None:
                self.emit_new(self.rf(), 'BOTE', [b])
        elif a == 'impe':
            imps = [l for l in av if l.f[0] == 'imp']
            if imps:
                im = rng.choice(imps)
                ant = self.find(im.f[1])
                if ant is None and len(self.prem) < self.max_prem and rng.random() < 0.6:
                    ant = self.add_premise(im.f[1])
                if ant is not None:
                    self.emit_new(im.f[2], 'IMPE', [im, ant])
            else:
                x = rng.choice(av)
                if len(self.prem) < self.max_prem and rng.random() < 0.6:
                    im = self.add_premise(('imp', x.f, self.rf()))
                    self.emit_new(im.f[2], 'IMPE', [im, x])
        elif a == 'nege':
            for l in av:
                if l.f[0] == 'not':
                    pos = self.find(l.f[1])
                    if pos is not None:
                        self.emit_new(BOT, 'NEGE', [pos, l])
                        return
            x = rng.choice(av)
            if len(self.prem) < self.max_prem and rng.random() < 0.4:
                p = self.add_premise(('not', x.f))
                self.emit_new(BOT, 'NEGE', [x, p])

    def emit_new(self, f, rule, refs):
        """Emit unless f is already citable (avoid useless duplicates); also skip huge formulas."""
        if self.find(f) is not None or fsize(f) > 15:
            return None
        return self.emit(f, rule, refs)

    # ---------- whole proofs ----------
    def forward(self, n_steps, n_prem):
        self.reset(n_prem)
        for _ in range(n_steps):
            self.step()
        while self.depth > 0:
            self.close_box()
        if not self.body:
            raise Fail('empty')
        return self.finish()

    def goal(self, n_prem, gdepth, budget):
        self.reset(n_prem)
        G = self.rf(gdepth)
        if G[0] in ('atom', 'bot') and self.rng.random() < 0.7:
            G = self.rf(gdepth)
        self.reach(G, budget)
        while self.depth > 0:
            self.close_box()
        if not self.body:
            raise Fail('empty')
        return self.finish()

    def finish(self):
        last = self.body[-1]
        if last.rule == 'AS' or last.depth != 0:
            raise Fail('bad last')
        # dependency pruning
        stack = [last]
        while stack:
            ln = stack.pop()
            if ln.keep:
                continue
            ln.keep = True
            stack.extend(ln.refs)
        prem = [p for p in self.prem if p.keep]
        body = [l for l in self.body if l.keep]
        lines = prem + body
        for i, ln in enumerate(lines):
            ln.idx = i + 1
        if last.f in [p.f for p in prem]:
            raise Fail('trivial: conclusion is a premise')
        if any(p.f == BOT for p in prem):
            raise Fail('trivial: F premise')
        if self.strict:
            if last.rule in ('ANDI', 'ORI1', 'ORI2', 'BOTE', 'R'):
                raise Fail('strict: intro conclusion')
            if any('F' in fstr(p.f) for p in prem) or 'F' in fstr(last.f):
                raise Fail('strict: F in theorem')
        toks = []
        for ln in lines:
            toks.append(f'N{ln.idx}')
            toks.extend(['|'] * ln.depth)
            toks.append(fstr(ln.f))
            toks.append(':')
            toks.append(ln.rule)
            toks.extend(f'N{r.idx}' for r in ln.refs)
            toks.append(';')
        toks.append('QED')
        prems = ' , '.join(fstr(p.f) for p in prem)
        prompt = f'THM {prems} SEQ {fstr(last.f)} PRF' if prem else f'THM SEQ {fstr(last.f)} PRF'
        thm = f'{prems} |- {fstr(last.f)}'
        body_s = ' '.join(toks)
        rules = sorted(set(l.rule for l in body))
        contra = any(('not', p.f) in [q.f for q in prem] for p in prem)
        return {'contra_prem': contra, 'thm': thm, 'prompt': prompt, 'proof': body_s, 'n_lines': len(lines), 'rules': rules,
                'n_prem': len(prem), 'gen_last_rule': last.rule, 'n_lazy_prem': sum(1 for p in prem if p.lazy)}


def canon_key(thm):
    """Atom-renaming class: relabel atoms by first appearance in the theorem string."""
    m = {}
    out = []
    for t in thm.split():
        if t in ATOMS:
            if t not in m:
                m[t] = ATOMS[len(m)]
            out.append(m[t])
        else:
            out.append(t)
    return ' '.join(out)


KNOBS_G1 = {'ore_steps': 3, 'ore_boxes': True}
KNOBS_G2 = {'ore_steps': 3, 'ore_boxes': True, 'goal_only': True, 'strict': True, 'budgets': [2, 3, 4], 'gdepth': [2, 2, 3]}


def sample_one(g, rng, long=False, knobs=None):
    """knobs (ds-generator, 2026-09-22): None = the take-home generator (control path, byte-identical). Keys:
    ore_steps (int, default 1 at cap 6), ore_boxes (bool), goal_only (bool: 100 % goal mode), strict (bool: Gen.strict, no lazy
    ( Z > G ) / G fallback, no F in the theorem, no intro-rule conclusion; bot_p 0 as in the strict long generator),
    budgets (reach budgets for goal mode), gdepth (goal formula depths)."""
    if knobs:
        g.ore_steps = knobs.get('ore_steps', 1)
        g.ore_boxes = bool(knobs.get('ore_boxes', False))
        g.strict = bool(knobs.get('strict', False))
        g.bot_p = 0.0 if g.strict else 0.02
        if knobs.get('ladder'):
            # the strict long generator that built the ladder pools (sample_one(long=True)), with the knobs above applied
            mode = 'goal' if rng.random() < 0.5 else 'forward'
            if mode == 'forward':
                return g.forward(rng.randint(6, 22), rng.choice([0, 1, 1, 2, 2, 3]))
            return g.goal(rng.choice([0, 0, 1, 1, 2]), rng.choice([2, 3, 3]), rng.choice([3, 4, 5]))
        budgets = knobs.get('budgets', [1, 2, 3]); gdepth = knobs.get('gdepth', [1, 2, 2, 3])
        mode = 'goal' if (knobs.get('goal_only') or rng.random() < 0.45) else 'forward'
        if mode == 'forward':
            return g.forward(rng.randint(1, 10), rng.choice([0, 1, 1, 2, 2, 3]))
        return g.goal(rng.choice([0, 0, 1, 1, 2]), rng.choice(gdepth), rng.choice(budgets))
    g.ore_boxes = False
    g.ore_steps = 3 if long else 1
    g.strict = long
    g.bot_p = 0.0 if long else 0.02
    if long:
        mode = 'goal' if rng.random() < 0.5 else 'forward'
        if mode == 'forward':
            return g.forward(rng.randint(6, 22), rng.choice([0, 1, 1, 2, 2, 3]))
        return g.goal(rng.choice([0, 0, 1, 1, 2]), rng.choice([2, 3, 3]), rng.choice([3, 4, 5]))
    mode = 'goal' if rng.random() < 0.45 else 'forward'
    if mode == 'forward':
        return g.forward(rng.randint(1, 10), rng.choice([0, 1, 1, 2, 2, 3]))
    return g.goal(rng.choice([0, 0, 1, 1, 2]), rng.choice([1, 2, 2, 3]), rng.choice([1, 2, 3]))


def knobs_from_args(a):
    """None unless a ds-generator flag is set (so the control path is untouched)."""
    k = {}
    if a.ore_steps is not None: k['ore_steps'] = a.ore_steps
    if a.ore_boxes: k['ore_boxes'] = True
    if a.goal_only: k['goal_only'] = True
    if a.strict: k['strict'] = True
    if a.budgets: k['budgets'] = [int(x) for x in a.budgets.split(',')]
    if a.gdepth: k['gdepth'] = [int(x) for x in a.gdepth.split(',')]
    if getattr(a, 'ladder', False): k['ladder'] = True; k['strict'] = True; k.setdefault('ore_steps', 3)
    if getattr(a, 'drop_contra', False): k['drop_contra'] = True
    if k and 'ore_steps' not in k: k['ore_steps'] = 1
    return k or None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=1000, help='number of accepted, deduplicated proofs to emit')
    ap.add_argument('--min', type=int, default=2)
    ap.add_argument('--max', type=int, default=6)
    ap.add_argument('--out', required=True)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--long', action='store_true', help='settings for 7-16 line pools')
    ap.add_argument('--per_len', type=int, default=None, help='cap per pruned length (flattens the histogram)')
    ap.add_argument('--max_prem', type=int, default=3)
    ap.add_argument('--ore_steps', type=int, default=None, help='ds-generator knob: steps in ORE branch 1 (take-home: 1 at cap 6, 3 with --long)')
    ap.add_argument('--ore_boxes', action='store_true', help='ds-generator knob G1: boxes / nested ORE allowed inside ORE branch 1')
    ap.add_argument('--goal_only', action='store_true', help='ds-generator knob G2: 100 %% goal mode')
    ap.add_argument('--strict', action='store_true', help='ds-generator knob G2: Gen.strict (no lazy ( Z > G ) / G fallback, no F, no intro conclusion)')
    ap.add_argument('--budgets', default=None, help='ds-generator knob: reach budgets, e.g. 2,3,4')
    ap.add_argument('--gdepth', default=None, help='ds-generator knob: goal formula depths, e.g. 2,2,3')
    ap.add_argument('--ladder', action='store_true', help='ds-generator knob: the strict long generator that built the ladder pools (50 %% goal / 50 %% forward, strict, ore_steps 3, budgets 3-5, gdepth 2-3, forward 6-22 steps) at whatever --min/--max')
    a = ap.parse_args()
    knobs = knobs_from_args(a)
    rng = random.Random(a.seed)
    g = Gen(rng, max_prem=a.max_prem)
    seen = set()
    seen_str = set()
    stats = collections.Counter()
    per_len = collections.Counter()
    t0 = time.time()
    n_out = 0
    with open(a.out, 'w') as fo:
        while n_out < a.n:
            stats['tries'] += 1
            try:
                r = sample_one(g, rng, a.long, knobs)
            except (Fail, RecursionError) as e:
                stats['fail:' + str(e).split(':')[0]] += 1
                continue
            L = r['n_lines']
            if L < a.min or L > a.max:
                stats['len_out_of_range'] += 1
                continue
            if a.per_len and per_len[L] >= a.per_len:
                stats['len_full'] += 1
                continue
            ok, reason, nl = verify_text(r['prompt'] + ' ' + r['proof'])
            if not ok:
                stats['VERIFIER_REJECT'] += 1
                print('REJECT', reason, r['prompt'], r['proof'], file=sys.stderr)
                continue
            assert nl == L
            key = canon_key(r['thm'])
            if key in seen:
                stats['dup_theorem'] += 1
                if r['thm'] not in seen_str:
                    stats['dup_renaming_only'] += 1   # same class, different atom names
                    seen_str.add(r['thm'])
                continue
            seen.add(key)
            seen_str.add(r['thm'])
            r['key'] = key
            r['text'] = r['prompt'] + ' ' + r['proof']
            fo.write(json.dumps(r) + '\n')
            n_out += 1
            per_len[L] += 1
            if n_out % 20000 == 0:
                print(f'{n_out} written, {time.time()-t0:.0f}s, {dict(stats)}', file=sys.stderr)
    print(json.dumps({'stats': dict(stats), 'per_len': dict(sorted(per_len.items())), 'secs': time.time() - t0}))


if __name__ == '__main__':
    main()
