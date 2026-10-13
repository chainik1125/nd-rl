"""The proof-state environment: a ``lean_seq`` proof as a sequence of (tactic state, action) steps.

A whole-proof model writes a ``lean_seq`` proof in one pass. Here the policy instead sees the
tactic state of the focused goal and writes one step::

    <st> h1 : ( P ∧ Q ) <nl> n1 : ( P ∧ Q ) <nl> ⊢ P <act>      ->     have n2 : P := n1 .1 ;

The state lists the hypotheses in scope, in Lean's order (the premises ``h1..hm``, the ``have``\\ s
completed at this or an enclosing level, the box binders), then ``⊢`` and the goal; formulas are
``lean_formula_tokens``' rendering. Nothing else: the theorem is the initial state. An action is
the literal ``lean_seq`` text of one step:

    have n<k> : F := <atomic term> ;                           an ordinary line
    have n<k> : F := ( fun ( n<j> : A ) => by                  opening an implication / negation box
    have n<k> : F := Or.elim n<i> ( fun ( n<j> : A ) => by     opening an Or.elim, first branch
    exact n<k>                                                 closing the focused goal

The environment applies an action syntactically and appends it to the proof text (``Env.text``).
It completes ``exact``, because the tokens that close a box encode box structure, not a step:

    top level    exact n  ->  exact n                                  (the proof is finished)
    imp box      exact n  ->  exact n ) ;
    neg box      exact n  ->  exact ( n : False ) ) ;
    Or.elim b1   exact n  ->  exact n ) ( fun ( n<next> : B ) => by    (branch 2 opened)
    Or.elim b2   exact n  ->  exact n ) ;

Names. By default the action writes every name it introduces, and ``n<next>`` above is the largest
name used so far plus one: ``lean_seq``'s numbering by first appearance, so replaying a proof's
actions reproduces its text byte for byte (``decompose``). That numbering is not a function of the
state once a box has closed and taken its names out of scope. With ``canon=True`` every name a step
introduces is instead the largest index in scope plus one, counting from ``base``; ``canonicalise``
rewrites a proof that way (an alpha-variant: the same ND proof, the same Lean verdict). With
``assign=True`` as well, the environment overrides the names the action introduces with those
(the fork's arm SN, "SN-v2", sampled with ``base`` drawn from 0..32 per attempt); citations are
the policy's.

There is no type check: a wrong term is Lean's to catch on the finished proof (``step_check``
catches the certain errors one step at a time). The environment checks what the strict grammar of
``lean_seq.inverse`` checks, so a finished attempt always parses: cited names in scope; ``exact n``
only for the focused frame's last statement, whose declared formula is the goal; premise lines
``:= h<k>`` only at depth 0, in order, before any other line; a box binder of the formula its
``have`` implies. A rejected action ends the attempt; malformed input never raises.

Provenance: the take-home fork ``dan-pandori/nd-takehome``, ``state_env.py`` at commit 4dcd0e4f
(runs ``state-env`` and ``best-state``), on ``lean_seq``'s types instead of formula tuples. There,
the decomposition replayed and reassembled all 155,000 cap-6 training proofs byte for byte, and
the rendered state matched Lean's ``trace_state`` (hypotheses and goal) at 4,300 cut points with
0 mismatches. ``tests/test_state_env.py`` checks this port against the fork's own output
(``tests/fixtures/proof_state/``).
"""

from __future__ import annotations

import functools
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Literal, NamedTuple

from code.shared_code.lean_seq import (
    MAX_NAMES,
    BinOp,
    Bot,
    Formula,
    LeanParseError,
    Not,
    inverse,
    lean_formula_tokens,
    parse_lean_formula,
    parse_prompt,
    proof_symbols,
)

STATE_SYMBOLS = ("<st>", "<nl>", "⊢", "<act>")
FALSE = Bot()


def is_name(t: str) -> bool:
    """A hypothesis name the environment accepts: ``n1`` to ``n64``."""
    return len(t) > 1 and t[0] == "n" and t[1:].isdigit() and 1 <= int(t[1:]) <= MAX_NAMES


# --------------------------------------------------------------------------------------------------
# Splitting a proof into actions


class _Splitter:
    """Recursive-descent parser for the strict ``lean_seq`` grammar, emitting actions."""

    def __init__(self, toks: Sequence[str]) -> None:
        self.toks = toks
        self.pos = 0
        self.actions: list[list[str]] = []
        self.second_binders: list[str] = []

    def peek(self) -> str | None:
        return self.toks[self.pos] if self.pos < len(self.toks) else None

    def eat(self, expected: str | None = None) -> str:
        t = self.peek()
        if t is None or (expected is not None and t != expected):
            raise LeanParseError(f"expected {expected}")
        self.pos += 1
        return t

    def name(self) -> str:
        t = self.eat()
        if not is_name(t):
            raise LeanParseError("name")
        return t

    def formula(self) -> list[str]:
        start = self.pos
        _, self.pos = parse_lean_formula(self.toks, self.pos)
        return list(self.toks[start : self.pos])

    def box_head(self) -> list[str]:
        head = [self.eat("("), self.eat("fun"), self.eat("("), self.name(), self.eat(":")]
        return [*head, *self.formula(), self.eat(")"), self.eat("=>"), self.eat("by")]

    def close_box(self) -> list[str]:
        self.eat("exact")
        if self.peek() == "(":
            self.eat("(")
            n = self.name()
            for x in (":", "False", ")"):
                self.eat(x)
        else:
            n = self.name()
        self.eat(")")
        return ["exact", n]

    def stmts(self) -> None:
        while self.peek() == "have":
            act = [self.eat("have"), self.name(), self.eat(":"), *self.formula(), self.eat(":=")]
            t = self.peek()
            if t == "(":
                self.actions.append(act + self.box_head())
                self.stmts()
                self.actions.append(self.close_box())
                self.eat(";")
            elif t == "Or.elim":
                act += [self.eat("Or.elim"), self.name()]
                self.actions.append(act + self.box_head())
                self.stmts()
                self.actions.append(self.close_box())
                self.second_binders.append(self.box_head()[3])  # the environment opens branch 2
                self.stmts()
                self.actions.append(self.close_box())
                self.eat(";")
            else:
                while self.peek() is not None and self.peek() != ";":
                    act.append(self.eat())
                act.append(self.eat(";"))
                self.actions.append(act)

    def parse(self) -> list[list[str]]:
        self.stmts()
        self.eat("exact")
        self.actions.append(["exact", self.name()])
        if self.pos != len(self.toks):
            raise LeanParseError("trailing tokens")
        return self.actions


def split_actions(toks: Sequence[str]) -> tuple[list[list[str]], list[str]]:
    """``lean_seq`` symbols -> the actions that write them, and the binder names of the
    ``Or.elim`` second branches in order (the environment opens those, so they are not actions).

    Raises ``LeanParseError`` outside the strict grammar.
    """
    splitter = _Splitter(toks)
    return splitter.parse(), splitter.second_binders


# --------------------------------------------------------------------------------------------------
# The environment

FrameKind = Literal["top", "imp", "neg", "or1", "or2"]


@dataclass(eq=False, slots=True)
class Frame:
    """One open goal: the top level, an implication or negation box, or an ``Or.elim`` branch."""

    kind: FrameKind
    goal: Formula
    pending: tuple[str, Formula] | None = None  # (name, formula) of the ``have`` this box proves
    ore_right: Formula | None = None  # Or.elim branch 1: the second disjunct, branch 2's binder
    lines: list[list[str]] = field(default_factory=list)  # own hypotheses, ``n : F <nl>`` each
    names: dict[str, Formula] = field(default_factory=dict)  # own bindings
    last: str | None = None  # the name of this frame's last statement (a box binder counts)


@functools.lru_cache(maxsize=8192)
def _theorem(prompt: str) -> tuple[tuple[Formula, ...], Formula, tuple[tuple[str, ...], ...]]:
    """(premises, conclusion, the top frame's hypothesis lines); cached, as attempts share one."""
    premises, conclusion = parse_prompt(prompt)
    lines = tuple(
        (f"h{j}", ":", *lean_formula_tokens(p), "<nl>") for j, p in enumerate(premises, start=1)
    )
    return tuple(premises), conclusion, lines


class Env:
    """One proof attempt. ``apply(action)`` -> (accepted, reason); see the module docstring."""

    def __init__(
        self, prompt: str, *, canon: bool = False, base: int = 0, assign: bool = False
    ) -> None:
        premises, conclusion, lines = _theorem(prompt)
        self.prompt = prompt
        self.premises = premises
        self.conclusion = conclusion
        self.canon = canon
        self.base = base  # canonical names start at n<base + 1>
        self.assign = assign  # the environment names what a step introduces
        self.defined = 0  # names the environment assigned (``assign``)
        self.renamed = 0  # ... of which differed from the action's own
        top = Frame("top", conclusion)
        top.lines = [list(line) for line in lines]
        self.frames: list[Frame] = [top]
        self.text: list[str] = []  # the rendered lean_seq symbols
        self.history: list[str] = []  # the accepted actions, concatenated (as the policy wrote
        # them, with assigned names)
        self.n_premises = 0
        self.seen_non_premise = False
        self.max_name = 0
        self.done = False
        self.failed: str | None = None
        self.steps = 0

    # scope

    def lookup(self, name: str) -> Formula | None:
        """The formula a name in scope stands for (the innermost binding), or None."""
        for fr in reversed(self.frames):
            if name in fr.names:
                return fr.names[name]
        return None

    def scope(self) -> dict[str, Formula]:
        """Every name in scope and its formula, in the state's order (inner bindings win)."""
        out: dict[str, Formula] = {}
        for fr in self.frames:
            out.update(fr.names)
        return out

    def next_name(self, reserve: Iterable[int] = (), frames: Sequence[Frame] | None = None) -> int:
        """The canonical next name index: the largest in scope (with the pending ``have``\\ s, and
        ``reserve``), counting from ``base``, plus one."""
        mx = self.base
        for fr in self.frames if frames is None else frames:
            for n in fr.names:
                mx = max(mx, int(n[1:]))
            if fr.pending:
                mx = max(mx, int(fr.pending[0][1:]))
        for r in reserve:
            mx = max(mx, r)
        return mx + 1

    def _add_hyp(self, fr: Frame, name: str, f: Formula) -> None:
        fr.lines.append([name, ":", *lean_formula_tokens(f), "<nl>"])
        fr.names[name] = f
        fr.last = name

    def _bump(self, name: str) -> None:
        self.max_name = max(self.max_name, int(name[1:]))

    # observation

    def state_tokens(self) -> list[str]:
        """``<st> <hypotheses, each "name : F <nl>"> ⊢ <goal> <act>``."""
        out = ["<st>"]
        for fr in self.frames:
            for line in fr.lines:
                out += line
        return [*out, "⊢", *lean_formula_tokens(self.frames[-1].goal), "<act>"]

    # transition

    def apply(self, action: Sequence[str]) -> tuple[bool, str]:
        """Apply one action: (True, "") or (False, reason). A rejection sets ``failed``."""
        self.steps += 1
        try:
            effective = self._apply(list(action))
        except LeanParseError as e:
            self.failed = str(e)
            return False, str(e)
        except (IndexError, KeyError, ValueError):
            self.failed = "malformed"
            return False, "malformed"
        self.history += effective
        return True, ""

    def _apply(self, act: list[str]) -> list[str]:
        """Apply ``act``; return it as applied (with assigned names), or raise."""
        if not act:
            raise LeanParseError("empty action")
        if act[0] == "exact":
            if len(act) != 2 or not is_name(act[1]):
                raise LeanParseError("exact form")
            fr = self.frames[-1]
            f = self.lookup(act[1])
            if f is None:
                raise LeanParseError("unbound")
            if fr.last != act[1]:
                raise LeanParseError("exact not last")
            if f != fr.goal:
                raise LeanParseError("exact goal mismatch")
            self._close(act[1])
            return act
        if act[0] != "have":
            raise LeanParseError("action head")
        if len(act) < 2 or not is_name(act[1]):
            raise LeanParseError("have name")
        if len(act) < 3 or act[2] != ":":
            raise LeanParseError("colon")
        f, i = parse_lean_formula(act, 3)
        if i >= len(act) or act[i] != ":=":
            raise LeanParseError("assign")
        i += 1
        t = act[i] if i < len(act) else None
        if self.assign:
            act = self._assign_names(act, i)
        name = act[1]
        if t is None:
            raise LeanParseError("empty term")
        fr = self.frames[-1]
        if t[0] == "h" and t[1:].isdigit():  # a premise line
            if i + 2 != len(act) or act[i + 1] != ";":
                raise LeanParseError("PR form")
            k = int(t[1:])
            if (
                len(self.frames) != 1
                or self.seen_non_premise
                or k != self.n_premises + 1
                or k > len(self.premises)
            ):
                raise LeanParseError("PR position")
            self.n_premises += 1
            self._bump(name)
            self._add_hyp(fr, name, f)
            self.text += act
            return act
        if t == "(":  # an implication or negation box
            x, xf = self._box_head(act, i)
            if isinstance(f, BinOp) and f.op == "imp":
                kind, goal, want = "imp", f.right, f.left
            elif isinstance(f, Not):
                kind, goal, want = "neg", FALSE, f.sub
            else:
                raise LeanParseError("box on non-arrow")
            if xf != want:
                raise LeanParseError("binder mismatch")
            self._open(act, Frame(kind, goal, pending=(name, f)), x, xf)
            return act
        if t == "Or.elim":  # Or.elim, first branch
            if i + 1 >= len(act) or not is_name(act[i + 1]):
                raise LeanParseError("Or.elim name")
            jf = self.lookup(act[i + 1])
            if jf is None:
                raise LeanParseError("unbound")
            if not (isinstance(jf, BinOp) and jf.op == "or"):
                raise LeanParseError("Or.elim on non-or")
            x, xf = self._box_head(act, i + 2)
            if xf != jf.left:
                raise LeanParseError("binder mismatch")
            self._open(act, Frame("or1", f, pending=(name, f), ore_right=jf.right), x, xf)
            return act
        if act[-1] != ";":  # an atomic term
            raise LeanParseError("no semicolon")
        self._check_atomic(act[i:-1])
        self.seen_non_premise = True
        self._bump(name)
        self._add_hyp(fr, name, f)
        self.text += act
        return act

    def _assign_names(self, act: list[str], i: int) -> list[str]:
        """``act`` with the name(s) it introduces replaced by the canonical ones; ``act[i]`` is the
        first token of the term. Citations are left alone: they name what the state shows."""
        a = list(act)
        nh = f"n{self.next_name()}"
        self.defined += 1
        self.renamed += a[1] != nh
        a[1] = nh
        j = None  # the box binder's position
        if i < len(a) and a[i] == "(" and i + 3 < len(a):
            j = i + 3
        elif i < len(a) and a[i] == "Or.elim" and i + 5 < len(a):
            j = i + 5
        if j is not None and is_name(a[j]):
            nb = f"n{self.next_name(reserve=[int(nh[1:])])}"
            self.defined += 1
            self.renamed += a[j] != nb
            a[j] = nb
        if int(nh[1:]) > MAX_NAMES or (j is not None and int(a[j][1:]) > MAX_NAMES):
            raise LeanParseError("names exhausted")
        return a

    def _box_head(self, act: Sequence[str], i: int) -> tuple[str, Formula]:
        """``act[i:]`` must be exactly ``( fun ( x : A ) => by``; return (x, A)."""
        try:
            if (
                act[i] != "("
                or act[i + 1] != "fun"
                or act[i + 2] != "("
                or not is_name(act[i + 3])
                or act[i + 4] != ":"
            ):
                raise LeanParseError("box head")
            xf, j = parse_lean_formula(act, i + 5)
            if act[j] != ")" or act[j + 1] != "=>" or act[j + 2] != "by":
                raise LeanParseError("box head")
        except (IndexError, LeanParseError):
            raise LeanParseError("box head") from None
        if j + 3 != len(act):
            raise LeanParseError("box trailing")
        return act[i + 3], xf

    def _open(self, act: list[str], frame: Frame, binder: str, f: Formula) -> None:
        assert frame.pending is not None
        self.seen_non_premise = True
        self._bump(frame.pending[0])
        self._bump(binder)
        self.text += act
        self._add_hyp(frame, binder, f)
        self.frames.append(frame)

    def _check_atomic(self, term: Sequence[str]) -> None:
        """The atomic term shapes of the grammar, every cited name in scope."""

        def ref(n: str) -> None:
            if not is_name(n) or self.lookup(n) is None:
                raise LeanParseError("unbound")

        if not term:
            raise LeanParseError("empty term")
        if len(term) == 1:  # n
            ref(term[0])
        elif term[0] == "⟨":  # ⟨ a , b ⟩
            if len(term) != 5 or term[2] != "," or term[4] != "⟩":
                raise LeanParseError("term")
            ref(term[1])
            ref(term[3])
        elif term[0] in ("Or.inl", "Or.inr"):
            if len(term) != 2:
                raise LeanParseError("term")
            ref(term[1])
        elif term[0] == "Classical.byContradiction":  # ( fun hh => a hh )
            if (
                len(term) != 8
                or list(term[1:5]) != ["(", "fun", "hh", "=>"]
                or term[6] != "hh"
                or term[7] != ")"
            ):
                raise LeanParseError("term")
            ref(term[5])
        elif len(term) == 2:  # a .1, a .2, a .elim, f a
            ref(term[0])
            if term[1] not in (".1", ".2", ".elim"):
                ref(term[1])
        else:
            raise LeanParseError("term")

    def _close(self, name: str) -> None:
        """``exact name`` closes the focused frame."""
        fr = self.frames[-1]
        if fr.kind == "top":
            self.text += ["exact", name]
            self.done = True
            return
        assert fr.pending is not None
        if fr.kind == "or1":
            assert fr.ore_right is not None
            if self.canon:
                k = self.next_name(reserve=[int(fr.pending[0][1:])], frames=self.frames[:-1])
            else:
                k = self.max_name + 1
            if k > MAX_NAMES:
                raise LeanParseError("names exhausted")
            self.max_name = max(self.max_name, k)
            b = f"n{k}"
            right = lean_formula_tokens(fr.ore_right)
            self.text += ["exact", name, ")", "(", "fun", "(", b, ":", *right, ")", "=>", "by"]
            branch = Frame("or2", fr.goal, pending=fr.pending)
            self._add_hyp(branch, b, fr.ore_right)
            self.frames[-1] = branch
            return
        if fr.kind == "neg":
            self.text += ["exact", "(", name, ":", "False", ")", ")", ";"]
        else:  # imp, or2
            self.text += ["exact", name, ")", ";"]
        self.frames.pop()
        self._add_hyp(self.frames[-1], *fr.pending)

    # result

    def nd(self) -> str:
        """The ND proof body the rendered text denotes, or ``"LEANPARSE <reason>"``."""
        if not self.done:
            return "LEANPARSE env: " + (self.failed or "incomplete")
        try:
            return inverse(self.text)
        except LeanParseError as e:
            return f"LEANPARSE {e}"


# --------------------------------------------------------------------------------------------------
# Whole proofs


def canonicalise(prompt: str, body: str) -> list[str]:
    """An ND proof's ``lean_seq`` symbols with every name a step introduces renamed to the largest
    index in scope plus one (``canon=True``'s naming). An alpha-variant: ``inverse`` returns the
    same ND proof, and Lean's verdict does not depend on hypothesis names."""
    actions, second_binders = split_actions(proof_symbols(body))
    env = Env(prompt, canon=True)
    rename: dict[str, str] = {}
    n_second = 0
    for a in actions:
        if a[0] == "exact":
            opens_second = env.frames[-1].kind == "or1"
            ok, why = env.apply(["exact", rename[a[1]]])
            if not ok:
                raise LeanParseError(f"canon: {why}")
            if opens_second:
                binder = env.frames[-1].last
                assert binder is not None
                rename[second_binders[n_second]] = binder
                n_second += 1
            continue
        j = a.index(":=")
        na = list(a)
        nh = f"n{env.next_name()}"
        na[1] = nh
        if a[j + 1] == "(":
            nb = f"n{env.next_name(reserve=[int(nh[1:])])}"
            na[j + 4] = nb
            rename[a[j + 4]] = nb
        elif a[j + 1] == "Or.elim":
            na[j + 2] = rename[a[j + 2]]
            nb = f"n{env.next_name(reserve=[int(nh[1:])])}"
            na[j + 6] = nb
            rename[a[j + 6]] = nb
        else:
            for k in range(j + 1, len(a)):
                if is_name(a[k]):
                    na[k] = rename[a[k]]
        rename[a[1]] = nh
        ok, why = env.apply(na)
        if not ok:
            raise LeanParseError(f"canon: {why}")
    if not env.done:
        raise LeanParseError("canon: did not finish")
    return env.text


class Step(NamedTuple):
    """One step of a decomposed proof: the state before it, the action, the earlier actions."""

    state: list[str]
    action: list[str]
    history: list[str]


def decompose(prompt: str, body: str, *, canon: bool = False) -> tuple[list[Step], list[str], Env]:
    """An ND proof -> (its steps, its ``lean_seq`` symbols, the finished environment).

    The actions are replayed in a fresh ``Env``; raises ``LeanParseError`` unless they all apply
    and reassemble the symbols byte for byte. With ``canon=True`` the proof is first rewritten by
    ``canonicalise``.
    """
    toks = canonicalise(prompt, body) if canon else proof_symbols(body)
    actions, _ = split_actions(toks)
    env = Env(prompt, canon=canon)
    steps: list[Step] = []
    for a in actions:
        state = env.state_tokens()
        history = list(env.history)
        ok, why = env.apply(a)
        if not ok:
            raise LeanParseError(f"replay: {why}")
        steps.append(Step(state, a, history))
    if not env.done:
        raise LeanParseError("replay did not finish")
    if env.text != toks:
        raise LeanParseError("reassembly mismatch")
    return steps, toks, env
