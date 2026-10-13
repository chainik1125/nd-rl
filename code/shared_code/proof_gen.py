"""Random natural-deduction proofs, and the filters the training sets were built with.

How it samples (``Generator``):

- **Forward mode** (``forward``): start from 0-3 random premises, apply randomly chosen rules to
  the lines that are currently citable, and open and close boxes at random (IMPI, NEGI and ORE all
  appear). When a rule needs a formula that is not available (the antecedent of an implication,
  the negation for NEGE, ...) a premise is introduced lazily, with some probability; this is what
  makes a theorem's premises fit together.
- **Goal mode** (``goal``): draw a conclusion and complete it with ``reach``, a randomised routine
  that decomposes the goal by its main connective (ANDI, ORI, an IMPI or NEGI box) or uses an
  available line, ex falso, DN or a lazy premise ``( Z > G )``. It never backtracks: it is a
  generator, not a prover. The second branch of every ORE is completed the same way.
- **Dependency pruning** (``Generator.finish``): only the lines the conclusion transitively cites
  are kept, and unused premises are dropped, so a proof's length is its pruned length.
- ``long=True`` is the strict variant that made the 7-16-line pools: no lazy ``( Z > G )`` or
  ``G`` premise, no ``F`` in the theorem, no conclusion by an introduction rule.

``generate`` is the take-home's ``gen.py`` loop: keep proofs whose length is in range, cap the
count per length if asked (``per_length``, which flattens the length histogram), and keep one
theorem per renaming class (``renaming_class``: atoms relabelled in order of first appearance).

The K12 training set (``data/kh/train_k12.jsonl``) took its 7-12-line proofs from this generator
(short mode, ``max_prem=3``), keeping those with box depth at most 2 (``max_box_depth``) whose
renaming class is in none of the evaluation pools and not in the cap-6 training set
(``eligible``).

Provenance: the take-home fork ``dan-pandori/nd-takehome``, ``gen.py`` and ``prune.py`` at
4dcd0e4f; the K12 rules from ``kh_assemble.py`` at 2e8b1914 (run ``cap-horizon``). The random
draws are the fork's, call for call, so a seed gives the same proofs as there
(``tests/test_proof_gen.py``). Formulas are ``lean_seq``'s. Not ported: the fork's check of every
proof with its ND verifier (it rejected none of 160,000), and the ``ds-generator`` knobs (G1/G2).
"""

from __future__ import annotations

import random
from collections import Counter
from collections.abc import Container, Iterator, Sequence
from typing import TypedDict

from code.shared_code.lean_seq import (
    ATOMS,
    Atom,
    BinOp,
    BinOpName,
    Bot,
    Formula,
    NDLine,
    Not,
    nd_formula_text,
    parse_proof,
)

FALSE = Bot()


def formula_size(f: Formula) -> int:
    """The number of nodes of a formula (atoms, F, connectives)."""
    if isinstance(f, Atom | Bot):
        return 1
    if isinstance(f, Not):
        return 1 + formula_size(f.sub)
    return 1 + formula_size(f.left) + formula_size(f.right)


class GenFail(Exception):
    """This draw produced no proof (a cap was hit, or the proof is trivial)."""


class Line:
    """A proof line under construction; ``refs`` are the lines it cites."""

    __slots__ = ("depth", "f", "rule", "refs", "idx", "keep", "lazy")

    def __init__(self, depth: int, f: Formula, rule: str, refs: Sequence[Line] = ()) -> None:
        self.depth = depth
        self.f = f
        self.rule = rule
        self.refs = list(refs)
        self.idx = 0  # the line number, assigned by ``finish``
        self.keep = False  # cited by the conclusion, transitively
        self.lazy = False  # a premise added after ``reset`` (a lazy premise)


class GenRecord(TypedDict):
    """One generated proof, with the fork's keys in the fork's order."""

    contra_prem: bool  # some premise is the negation of another
    thm: str  # "p1 , p2 |- c"
    prompt: str  # "THM p1 , p2 SEQ c PRF"
    proof: str  # "N1 ... ; QED"
    n_lines: int
    rules: list[str]  # the rules of the non-premise lines, sorted
    n_prem: int
    gen_last_rule: str
    n_lazy_prem: int
    key: str  # renaming_class(thm), set by ``generate``
    text: str  # prompt + " " + proof, set by ``generate``


class Generator:
    """Draws one proof per ``forward`` or ``goal`` call, with ``rng`` its only randomness."""

    def __init__(
        self,
        rng: random.Random,
        max_prem: int = 3,
        max_depth: int = 3,
        fdepth: int = 2,
        atoms: Sequence[str] = ATOMS,
        bot_p: float = 0.02,
    ) -> None:
        self.rng = rng
        self.max_prem = max_prem
        self.max_depth = max_depth
        self.fdepth = fdepth
        self.atoms = list(atoms)
        self.bot_p = bot_p
        self.ore_steps = 3
        self.strict = False  # long pools: see the module docstring
        self.prem: list[Line] = []
        self.body: list[Line] = []
        self.levels: list[list[Line]] = [[]]  # [0]: depth-0 lines; [d]: the open box at depth d
        self.boxes: list[Line] = []  # the AS lines of the open boxes

    # formulas

    def rf(self, depth: int | None = None) -> Formula:
        """A random formula of depth at most ``depth`` (default ``fdepth``)."""
        rng = self.rng
        if depth is None:
            depth = self.fdepth
        if depth == 0 or rng.random() < 0.45:
            if rng.random() < self.bot_p:
                return FALSE
            return Atom(rng.choice(self.atoms))
        if rng.random() < 0.28:
            return Not(self.rf(depth - 1))
        op: BinOpName = rng.choice(["and", "or", "imp"])
        left = self.rf(depth - 1)
        return BinOp(op, left, self.rf(depth - 1))

    # state

    def reset(self, n_prem: int) -> None:
        self.prem = []
        self.body = []
        self.levels = [[]]
        self.boxes = []
        for _ in range(n_prem):
            self.add_premise(self.rf(), lazy=False)

    @property
    def depth(self) -> int:
        return len(self.boxes)

    def add_premise(self, f: Formula, lazy: bool = True) -> Line:
        for p in self.prem:
            if p.f == f:
                return p
        if len(self.prem) >= self.max_prem:
            raise GenFail("premise cap")
        ln = Line(0, f, "PR")
        ln.lazy = lazy
        self.prem.append(ln)
        return ln

    def avail(self) -> list[Line]:
        """The citable lines: premises, then each open level's lines."""
        out = list(self.prem)
        for lv in self.levels:
            out.extend(lv)
        return out

    def find(self, f: Formula) -> Line | None:
        for ln in self.avail():
            if ln.f == f:
                return ln
        return None

    def emit(self, f: Formula, rule: str, refs: Sequence[Line] = ()) -> Line:
        ln = Line(self.depth, f, rule, refs)
        self.body.append(ln)
        self.levels[-1].append(ln)
        return ln

    def emit_new(self, f: Formula, rule: str, refs: Sequence[Line]) -> Line | None:
        """``emit``, unless ``f`` is already citable or has more than 15 nodes."""
        if self.find(f) is not None or formula_size(f) > 15:
            return None
        return self.emit(f, rule, refs)

    def open_box(self, hyp: Formula) -> Line:
        if self.depth >= self.max_depth:
            raise GenFail("depth cap")
        ln = Line(self.depth + 1, hyp, "AS")
        self.body.append(ln)
        self.boxes.append(ln)
        self.levels.append([ln])
        return ln

    def close_box(self, rule: str | None = None) -> Line:
        """Close the innermost box with IMPI, or NEGI (if it ends in F); return the new line."""
        s = self.boxes.pop()
        e = self.levels.pop()[-1]
        if rule is None:
            rule = "NEGI" if (e.f == FALSE and self.rng.random() < 0.8) else "IMPI"
        if rule == "NEGI" and e.f != FALSE:
            rule = "IMPI"
        g = Not(s.f) if rule == "NEGI" else BinOp("imp", s.f, e.f)
        return self.emit(g, rule, [s, e])

    def _drop_box(self) -> None:
        """Forget the innermost box (an ORE branch, which ``ORE`` cites instead of closing)."""
        self.boxes.pop()
        self.levels.pop()

    def last_at_level(self) -> Line | None:
        return self.levels[-1][-1] if self.levels[-1] else None

    # goal completion

    def reach(self, g: Formula, budget: int) -> Line:
        """Append lines so that ``g`` is the last line at the current level. Never backtracks."""
        rng = self.rng
        last = self.last_at_level()
        if last is not None and last.f == g:
            return last
        have = self.find(g)
        if have is not None:
            return self.emit(g, "R", [have])
        opts: list[str | tuple[str, Line] | tuple[str, Line, Line]] = []
        if budget > 0:
            if isinstance(g, BinOp) and g.op == "and":
                opts.append("andi")
            if isinstance(g, BinOp) and g.op == "or":
                opts.append("ori")
            if isinstance(g, BinOp) and g.op == "imp" and self.depth < self.max_depth:
                opts.append("impi")
            if isinstance(g, Not) and self.depth < self.max_depth:
                opts.append("negi")
        bot = self.find(FALSE)
        if bot is not None and g != FALSE:
            opts.append("bote")
        if g == FALSE:
            for ln in self.avail():
                if isinstance(ln.f, Not):
                    pos = self.find(ln.f.sub)
                    if pos is not None:
                        opts.append(("nege", pos, ln))
            has_nege = any(isinstance(o, tuple) and o[0] == "nege" for o in opts)
            if not has_nege and self.avail() and len(self.prem) < self.max_prem:
                opts.append("nege_lazy")
        elif budget > 0 and self.depth < self.max_depth and rng.random() < 0.5:
            opts.append("raa")
        dn = self.find(Not(Not(g)))
        if dn is not None:
            opts.append("dn")
        for ln in self.avail():
            f = ln.f
            if (
                isinstance(f, BinOp)
                and f.op == "imp"
                and f.right == g
                and self.find(f.left) is not None
            ):
                opts.append(("impe", ln))
            if isinstance(f, BinOp) and f.op == "and" and g in (f.left, f.right):
                opts.append(("ande", ln))
        if len(self.prem) < self.max_prem:
            opts.append("lazy")
        if not opts:
            opts.append("lazy")  # raises GenFail: the premise cap
        o = rng.choice(opts)
        if o == "andi":
            assert isinstance(g, BinOp)
            a = self.reach(g.left, budget - 1)
            b = self.reach(g.right, budget - 1)
            return self.emit(g, "ANDI", [a, b])
        if o == "ori":
            assert isinstance(g, BinOp)
            if self.find(g.left) is not None:
                side = 1
            elif self.find(g.right) is not None:
                side = 2
            else:
                side = rng.choice([1, 2])
            a = self.reach(g.left if side == 1 else g.right, budget - 1)
            return self.emit(g, "ORI1" if side == 1 else "ORI2", [a])
        if o == "impi":
            assert isinstance(g, BinOp)
            self.open_box(g.left)
            self.reach(g.right, budget - 1)
            return self.close_box("IMPI")
        if o == "negi":
            assert isinstance(g, Not)
            self.open_box(g.sub)
            self.reach(FALSE, budget - 1)
            return self.close_box("NEGI")
        if o == "bote":
            assert bot is not None
            return self.emit(g, "BOTE", [bot])
        if o == "raa":
            self.open_box(Not(g))
            self.reach(FALSE, budget - 1)
            nn = self.close_box("NEGI")
            return self.emit(g, "DN", [nn])
        if isinstance(o, tuple) and o[0] == "nege":
            assert len(o) == 3
            return self.emit(FALSE, "NEGE", [o[1], o[2]])
        if o == "nege_lazy":
            x = last if (last is not None and rng.random() < 0.7) else rng.choice(self.avail())
            if isinstance(x.f, Not) and rng.random() < 0.5:
                p = self.add_premise(x.f.sub)
                return self.emit(FALSE, "NEGE", [p, x])
            p = self.add_premise(Not(x.f))
            return self.emit(FALSE, "NEGE", [x, p])
        if o == "dn":
            assert dn is not None
            return self.emit(g, "DN", [dn])
        if isinstance(o, tuple) and o[0] == "impe":
            ln = o[1]
            assert isinstance(ln.f, BinOp)
            ante = self.find(ln.f.left)
            assert ante is not None
            return self.emit(g, "IMPE", [ln, ante])
        if isinstance(o, tuple) and o[0] == "ande":
            ln = o[1]
            assert isinstance(ln.f, BinOp)
            return self.emit(g, "ANDE1" if ln.f.left == g else "ANDE2", [ln])
        # a lazy premise: ( Z > G ) for an available Z (preferring the box's last line), or G
        if self.strict:
            raise GenFail("strict: no lazy goal premise")
        av = self.avail()
        if av and rng.random() < 0.85:
            z = last if (last is not None and rng.random() < 0.6) else rng.choice(av)
            p = self.add_premise(BinOp("imp", z.f, g))
            return self.emit(g, "IMPE", [p, z])
        p = self.add_premise(g)
        return self.emit(g, "R", [p])

    # forward steps

    def step(self) -> None:
        """One random forward step."""
        rng = self.rng
        av = self.avail()
        acts: list[str] = []
        if self.depth < self.max_depth:
            acts += ["as"] * 3
        if self.depth > 0 and len(self.levels[-1]) >= 1:
            acts += ["close"] * 3
        if av:
            if self.strict:
                acts += ["andi", "ori", *["impe"] * 4, *["nege"] * 3, *["ande"] * 3]
                acts += [*["dn"] * 2, "bote", *["ore"] * 6]
            else:
                acts += [*["andi"] * 2, *["ori"] * 2, *["impe"] * 4, *["nege"] * 3]
                acts += [*["ande"] * 2, *["dn"] * 2, "bote", *["ore"] * 5]
            if self.depth > 0:
                acts.append("r")
        else:
            acts += ["as"] * 2
        a = rng.choice(acts)
        if a == "as":
            hyp = None
            r = rng.random()
            imps = [ln.f for ln in av if isinstance(ln.f, BinOp) and ln.f.op == "imp"]
            negs = [ln.f for ln in av if isinstance(ln.f, Not)]
            if r < 0.3 and imps:
                f = rng.choice(imps)
                assert isinstance(f, BinOp)
                hyp = f.left
            elif r < 0.5 and negs:
                f = rng.choice(negs)
                assert isinstance(f, Not)
                hyp = f.sub
            elif r < 0.6:
                hyp = Not(Not(self.rf(1)))
            elif r < 0.7 and av:
                hyp = Not(rng.choice(av).f)
            if hyp is None:
                hyp = self.rf()
            self.open_box(hyp)
        elif a == "close":
            self.close_box()
        elif a == "r":
            outer = list(self.prem)
            for lv in self.levels[:-1]:
                outer.extend(lv)
            if outer:
                # Never emits: an outer line's formula is always citable. Kept for the draws.
                self.emit_new(rng.choice(outer).f, "R", [rng.choice(outer)])
        elif a == "ore":
            self._ore(av)
        elif a == "impe":
            imps = [ln for ln in av if isinstance(ln.f, BinOp) and ln.f.op == "imp"]
            if imps and rng.random() < 0.7:
                im = rng.choice(imps)
                assert isinstance(im.f, BinOp)
                ante = self.find(im.f.left)
                if ante is None:
                    if rng.random() < 0.6 and len(self.prem) < self.max_prem:
                        ante = self.add_premise(im.f.left)
                    else:
                        return
                self.emit_new(im.f.right, "IMPE", [im, ante])
            elif not self.strict:
                x = rng.choice(av)
                if len(self.prem) < self.max_prem and rng.random() < 0.6:
                    im = self.add_premise(BinOp("imp", x.f, self.rf()))
                    assert isinstance(im.f, BinOp)
                    self.emit_new(im.f.right, "IMPE", [im, x])
        elif a == "nege":
            for ln in av:
                if isinstance(ln.f, Not):
                    pos = self.find(ln.f.sub)
                    if pos is not None:
                        self.emit_new(FALSE, "NEGE", [pos, ln])
                        return
            x = rng.choice(av)
            if isinstance(x.f, Not) and rng.random() < 0.5:
                if len(self.prem) < self.max_prem and rng.random() < 0.5:
                    p = self.add_premise(x.f.sub)
                    self.emit_new(FALSE, "NEGE", [p, x])
            elif len(self.prem) < self.max_prem and rng.random() < 0.5:
                p = self.add_premise(Not(x.f))
                self.emit_new(FALSE, "NEGE", [x, p])
        else:  # andi, ori, ande, dn, bote
            self._local(a, av)

    def _ore(self, av: list[Line]) -> None:
        """An ORE on an available disjunction (or a lazy premise): branch 1 takes up to
        ``ore_steps`` local steps; branch 2 is completed to branch 1's last formula."""
        rng = self.rng
        ors = [ln for ln in av if isinstance(ln.f, BinOp) and ln.f.op == "or"]
        if not ors:
            if len(self.prem) < self.max_prem and rng.random() < 0.5:
                ors = [self.add_premise(BinOp("or", self.rf(), self.rf()))]
            else:
                return
        if self.depth >= self.max_depth:
            return
        d = rng.choice(ors)
        assert isinstance(d.f, BinOp)
        s1 = self.open_box(d.f.left)
        for _ in range(rng.randint(0, self.ore_steps)):
            self.step_local()
        e1 = self.levels[-1][-1]
        self._drop_box()
        s2 = self.open_box(d.f.right)
        self.reach(e1.f, 2)
        e2 = self.levels[-1][-1]
        self._drop_box()
        self.emit(e1.f, "ORE", [d, s1, e1, s2, e2])

    def step_local(self) -> None:
        """A forward step that does not change the box structure (ORE branch 1)."""
        d = self.depth
        for _ in range(4):
            self.step_no_box()
            if self.depth != d:
                break

    def step_no_box(self) -> None:
        rng = self.rng
        av = self.avail()
        a = rng.choice(["andi", "ori", "impe", "nege", "ande", "dn", "bote", "impe", "ande"])
        saved = self.max_depth
        self.max_depth = self.depth  # no box can open
        try:
            self._local(a, av)
        finally:
            self.max_depth = saved

    def _local(self, a: str, av: list[Line]) -> None:
        """A rule that adds one line at the current level. ``step`` has its own IMPE and NEGE,
        which draw differently."""
        rng = self.rng
        if a == "andi":
            x, y = rng.choice(av), rng.choice(av)
            self.emit_new(BinOp("and", x.f, y.f), "ANDI", [x, y])
        elif a == "ori":
            x = rng.choice(av)
            other = self.rf()
            if rng.random() < 0.5:
                self.emit_new(BinOp("or", x.f, other), "ORI1", [x])
            else:
                self.emit_new(BinOp("or", other, x.f), "ORI2", [x])
        elif a == "ande":
            cs = [ln for ln in av if isinstance(ln.f, BinOp) and ln.f.op == "and"]
            if cs:
                c = rng.choice(cs)
                assert isinstance(c.f, BinOp)
                first = rng.random() < 0.5
                self.emit_new(c.f.left if first else c.f.right, "ANDE1" if first else "ANDE2", [c])
        elif a == "dn":
            ds = [ln for ln in av if isinstance(ln.f, Not) and isinstance(ln.f.sub, Not)]
            if ds:
                d = rng.choice(ds)
                assert isinstance(d.f, Not) and isinstance(d.f.sub, Not)
                self.emit_new(d.f.sub.sub, "DN", [d])
        elif a == "bote":
            b = self.find(FALSE)
            if b is not None:
                self.emit_new(self.rf(), "BOTE", [b])
        elif a == "impe":
            imps = [ln for ln in av if isinstance(ln.f, BinOp) and ln.f.op == "imp"]
            if imps:
                im = rng.choice(imps)
                assert isinstance(im.f, BinOp)
                ante = self.find(im.f.left)
                if ante is None and len(self.prem) < self.max_prem and rng.random() < 0.6:
                    ante = self.add_premise(im.f.left)
                if ante is not None:
                    self.emit_new(im.f.right, "IMPE", [im, ante])
            else:
                x = rng.choice(av)
                if len(self.prem) < self.max_prem and rng.random() < 0.6:
                    im = self.add_premise(BinOp("imp", x.f, self.rf()))
                    assert isinstance(im.f, BinOp)
                    self.emit_new(im.f.right, "IMPE", [im, x])
        elif a == "nege":
            for ln in av:
                if isinstance(ln.f, Not):
                    pos = self.find(ln.f.sub)
                    if pos is not None:
                        self.emit_new(FALSE, "NEGE", [pos, ln])
                        return
            x = rng.choice(av)
            if len(self.prem) < self.max_prem and rng.random() < 0.4:
                p = self.add_premise(Not(x.f))
                self.emit_new(FALSE, "NEGE", [x, p])

    # whole proofs

    def forward(self, n_steps: int, n_prem: int) -> GenRecord:
        self.reset(n_prem)
        for _ in range(n_steps):
            self.step()
        while self.depth > 0:
            self.close_box()
        if not self.body:
            raise GenFail("empty")
        return self.finish()

    def goal(self, n_prem: int, gdepth: int, budget: int) -> GenRecord:
        self.reset(n_prem)
        g = self.rf(gdepth)
        if isinstance(g, Atom | Bot) and self.rng.random() < 0.7:
            g = self.rf(gdepth)
        self.reach(g, budget)
        while self.depth > 0:
            self.close_box()
        if not self.body:
            raise GenFail("empty")
        return self.finish()

    def finish(self) -> GenRecord:
        """Prune to the lines the last line cites, number them, and render the record."""
        last = self.body[-1]
        if last.rule == "AS" or last.depth != 0:
            raise GenFail("bad last")
        stack = [last]
        while stack:
            ln = stack.pop()
            if ln.keep:
                continue
            ln.keep = True
            stack.extend(ln.refs)
        prem = [p for p in self.prem if p.keep]
        body = [ln for ln in self.body if ln.keep]
        lines = prem + body
        for i, ln in enumerate(lines):
            ln.idx = i + 1
        if last.f in [p.f for p in prem]:
            raise GenFail("trivial: conclusion is a premise")
        if any(p.f == FALSE for p in prem):
            raise GenFail("trivial: F premise")
        if self.strict:
            if last.rule in ("ANDI", "ORI1", "ORI2", "BOTE", "R"):
                raise GenFail("strict: intro conclusion")
            texts = [nd_formula_text(p.f) for p in prem] + [nd_formula_text(last.f)]
            if any("F" in t for t in texts):
                raise GenFail("strict: F in theorem")
        toks: list[str] = []
        for ln in lines:
            toks += [f"N{ln.idx}", *["|"] * ln.depth, nd_formula_text(ln.f), ":", ln.rule]
            toks += [*(f"N{r.idx}" for r in ln.refs), ";"]
        toks.append("QED")
        prems = " , ".join(nd_formula_text(p.f) for p in prem)
        concl = nd_formula_text(last.f)
        prem_fs = [q.f for q in prem]
        return {
            "contra_prem": any(Not(p.f) in prem_fs for p in prem),
            "thm": f"{prems} |- {concl}",
            "prompt": f"THM {prems} SEQ {concl} PRF" if prem else f"THM SEQ {concl} PRF",
            "proof": " ".join(toks),
            "n_lines": len(lines),
            "rules": sorted({ln.rule for ln in body}),
            "n_prem": len(prem),
            "gen_last_rule": last.rule,
            "n_lazy_prem": sum(1 for p in prem if p.lazy),
            "key": "",
            "text": "",
        }


def sample_one(g: Generator, rng: random.Random, long: bool = False) -> GenRecord:
    """One draw with the take-home's settings: forward mode 55 % of the time (1-10 steps), else
    goal mode; ``long``: the strict generator, half each, longer runs and deeper goals."""
    g.ore_steps = 3 if long else 1
    g.strict = long
    g.bot_p = 0.0 if long else 0.02
    if long:
        if rng.random() >= 0.5:
            return g.forward(rng.randint(6, 22), rng.choice([0, 1, 1, 2, 2, 3]))
        return g.goal(rng.choice([0, 0, 1, 1, 2]), rng.choice([2, 3, 3]), rng.choice([3, 4, 5]))
    if rng.random() >= 0.45:
        return g.forward(rng.randint(1, 10), rng.choice([0, 1, 1, 2, 2, 3]))
    return g.goal(rng.choice([0, 0, 1, 1, 2]), rng.choice([1, 2, 2, 3]), rng.choice([1, 2, 3]))


def renaming_class(thm: str) -> str:
    """A theorem's renaming class: its atoms relabelled ``P, Q, R, S`` by first appearance."""
    m: dict[str, str] = {}
    out: list[str] = []
    for t in thm.split():
        if t in ATOMS and t not in m:
            m[t] = ATOMS[len(m)]
        out.append(m.get(t, t))
    return " ".join(out)


def generate(
    n: int,
    *,
    min_lines: int = 2,
    max_lines: int = 6,
    seed: int = 0,
    long: bool = False,
    per_length: int | None = None,
    max_prem: int = 3,
    stats: Counter[str] | None = None,
) -> Iterator[GenRecord]:
    """``n`` proofs of ``min_lines``-``max_lines`` (pruned) lines, one per renaming class, at most
    ``per_length`` of each length: the take-home ``gen.py`` with ``--n --min --max --seed --long
    --per_len --max_prem``. Draws that fail or are filtered out are counted in ``stats``."""
    rng = random.Random(seed)
    g = Generator(rng, max_prem=max_prem)
    stats = Counter() if stats is None else stats
    seen: set[str] = set()
    seen_thm: set[str] = set()
    counts: Counter[int] = Counter()
    n_out = 0
    while n_out < n:
        stats["tries"] += 1
        try:
            r = sample_one(g, rng, long)
        except (GenFail, RecursionError) as e:
            stats["fail:" + str(e).split(":")[0]] += 1
            continue
        length = r["n_lines"]
        if length < min_lines or length > max_lines:
            stats["len_out_of_range"] += 1
            continue
        if per_length and counts[length] >= per_length:
            stats["len_full"] += 1
            continue
        key = renaming_class(r["thm"])
        if key in seen:
            stats["dup_theorem"] += 1
            if r["thm"] not in seen_thm:
                stats["dup_renaming_only"] += 1  # same class, other atoms
                seen_thm.add(r["thm"])
            continue
        seen.add(key)
        seen_thm.add(r["thm"])
        r["key"] = key
        r["text"] = r["prompt"] + " " + r["proof"]
        yield r
        n_out += 1
        counts[length] += 1


# --------------------------------------------------------------------------------------------------
# Pruning and the K12 filters


def prune(lines: Sequence[NDLine], *, keep_premises: bool = False) -> list[NDLine]:
    """The lines the last line transitively cites, in order (citing a box cites its first and last
    lines); ``keep_premises`` keeps every PR line too, so the result proves the same prompt."""
    if not lines:
        return []
    by_idx = {ln.idx: ln for ln in lines}
    keep = {ln.idx for ln in lines if ln.rule == "PR"} if keep_premises else set[int]()
    stack = [lines[-1].idx]
    while stack:
        i = stack.pop()
        if i in keep or i not in by_idx:
            continue
        keep.add(i)
        stack.extend(by_idx[i].refs)
    return [ln for ln in lines if ln.idx in keep]


def pruned_length(body: str) -> int:
    """The number of lines an ND proof's conclusion transitively cites. Raises ``NDParseError``
    if the body does not parse (the fork returned the written length instead)."""
    return len(prune(parse_proof(body.split())))


def max_box_depth(body: str) -> int:
    """The deepest box nesting of an ND proof as written (the most ``|`` on one line)."""
    return max((ln.depth for ln in parse_proof(body.split())), default=0)


def eligible(thm: str, body: str, *, exclude: Container[str], box_depth: int = 2) -> bool:
    """K12's rule for a generated proof: box depth at most ``box_depth`` and a renaming class
    not in ``exclude`` (e.g. ``{renaming_class(t) for t in pool_theorems}``)."""
    return renaming_class(thm) not in exclude and max_box_depth(body) <= box_depth
