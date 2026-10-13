"""Lean 4 ``lean_seq`` rendering of natural-deduction proofs: a tokenizer and its inverse parser.

The take-home writes Fitch-style natural-deduction (ND) proofs in a token format::

    THM ( P > Q ) , P SEQ Q PRF      N1 ( P > Q ) : PR ; N2 P : PR ; N3 Q : IMPE N1 N2 ; QED

``lean_seq`` renders the same proof as one line of Lean 4 tactic text, one symbol per token::

    theorem t ( P Q R S : Prop ) ( h1 : ( P → Q ) ) ( h2 : P ) : Q := by
    have n1 : ( P → Q ) := h1 ; have n2 : P := h2 ; have n3 : Q := n1 n2 ; exact n3

Each ND rule maps to one Lean term:

- ``PR``: ``hK``; ``R``: ``nA``; ``ANDI``: ``⟨ nA , nB ⟩``
- ``ANDE1`` / ``ANDE2``: ``nA .1`` / ``nA .2``
- ``IMPE`` / ``NEGE``: ``nF nA``; ``ORI1`` / ``ORI2``: ``Or.inl nA`` / ``Or.inr nA``; ``BOTE``:
  ``nA .elim``; ``DN``: ``Classical.byContradiction ( fun hh => nA hh )``
- ``IMPI``: a box ``( fun ( nS : A ) => by ... ; exact nE )``; ``NEGI``: the same box ending
  ``exact ( nE : False )``; ``ORE``: ``Or.elim nJ <box> <box>``

In the text, ``.1``, ``.2`` and ``.elim`` are glued to the name.

Hypothesis names are labels, numbered by first appearance in the Lean text (``n1``, ``n2``, ...).
In training, a random offset is added to all names in a proof (``shift_names``), the analogue of the
token format's start-index shift. ``inverse`` parses the strict grammar back to the ND proof it
denotes, so ND tooling (verifiers, pruning, analysis) works unchanged on model samples.

Provenance and compatibility: ported from the take-home fork ``dan-pandori/nd-takehome``
(``lean_tok.py``, mode ``lean_seq``), with the same 107-token vocabulary in the same order, so a
checkpoint trained there decodes here unchanged. ``tests/test_lean_seq.py`` checks this against 68
golden examples produced by the original. The fork's other modes are not included: ``lean_rand``
(random name permutations, which trained worse) and the vocabularies of the state-conditioned
modes (``lean_state*``); their states and actions are ``state_env``'s.

Checking: the rendered text is ordinary Lean 4 core (no Mathlib); ``statement(prompt) + " " + text``
is a complete theorem. Which checker decides whether a proof counts is up to the caller; this module
only translates.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, TypeAlias

# --------------------------------------------------------------------------------------------------
# Formulas


@dataclass(frozen=True)
class Atom:
    name: str


@dataclass(frozen=True)
class Bot:
    pass


@dataclass(frozen=True)
class Not:
    sub: Formula


BinOpName: TypeAlias = Literal["and", "or", "imp"]


@dataclass(frozen=True)
class BinOp:
    op: BinOpName
    left: Formula
    right: Formula


Formula: TypeAlias = Atom | Bot | Not | BinOp

ATOMS = ("P", "Q", "R", "S")
ND_BINOPS: dict[str, BinOpName] = {"&": "and", "v": "or", ">": "imp"}
LEAN_BINOPS: dict[BinOpName, str] = {"and": "∧", "or": "∨", "imp": "→"}
ND_OF_BINOP: dict[BinOpName, str] = {"and": "&", "or": "v", "imp": ">"}
RULES = frozenset(
    {"ANDI", "ANDE1", "ANDE2", "IMPE", "IMPI", "ORI1", "ORI2", "ORE"}
    | {"NEGE", "NEGI", "BOTE", "DN", "PR", "AS", "R"}
)


class NDParseError(ValueError):
    """The ND text is not in the take-home's token format."""


class LeanParseError(ValueError):
    """The Lean tokens are outside the strict ``lean_seq`` grammar."""


def parse_formula(toks: Sequence[str], i: int) -> tuple[Formula, int]:
    """Parse one ND formula starting at ``toks[i]``; return it and the index after it."""
    if i >= len(toks):
        raise NDParseError("eof in formula")
    t = toks[i]
    if t in ATOMS:
        return Atom(t), i + 1
    if t == "F":
        return Bot(), i + 1
    if t != "(":
        raise NDParseError(f"bad formula token {t!r}")
    if i + 1 < len(toks) and toks[i + 1] == "~":
        sub, j = parse_formula(toks, i + 2)
        if j >= len(toks) or toks[j] != ")":
            raise NDParseError("missing ) after ~")
        return Not(sub), j + 1
    left, j = parse_formula(toks, i + 1)
    if j >= len(toks) or toks[j] not in ND_BINOPS:
        raise NDParseError("missing binary operator")
    right, k = parse_formula(toks, j + 1)
    if k >= len(toks) or toks[k] != ")":
        raise NDParseError("missing )")
    return BinOp(ND_BINOPS[toks[j]], left, right), k + 1


@dataclass(frozen=True)
class NDLine:
    idx: int
    depth: int
    formula: Formula
    rule: str
    refs: tuple[int, ...]


def parse_proof(toks: Sequence[str]) -> list[NDLine]:
    """Parse an ND proof body (``N1 ... ; ... QED``) into lines. Syntax only; no rule checking."""
    lines: list[NDLine] = []
    i = 0
    while i < len(toks):
        if toks[i] == "QED":
            return lines
        t = toks[i]
        if not (t.startswith("N") and t[1:].isdigit()):
            raise NDParseError(f"expected line index, got {t!r}")
        idx = int(t[1:])
        i += 1
        depth = 0
        while i < len(toks) and toks[i] == "|":
            depth += 1
            i += 1
        formula, i = parse_formula(toks, i)
        if i >= len(toks) or toks[i] != ":":
            raise NDParseError("missing :")
        i += 1
        if i >= len(toks) or toks[i] not in RULES:
            raise NDParseError("bad rule name")
        rule = toks[i]
        i += 1
        refs: list[int] = []
        while i < len(toks) and toks[i].startswith("N") and toks[i][1:].isdigit():
            refs.append(int(toks[i][1:]))
            i += 1
        if i >= len(toks) or toks[i] != ";":
            raise NDParseError("missing ;")
        i += 1
        lines.append(NDLine(idx, depth, formula, rule, tuple(refs)))
    raise NDParseError("missing QED")


def parse_prompt(prompt: str) -> tuple[list[Formula], Formula]:
    """``THM A , B SEQ C PRF`` -> (premises, conclusion)."""
    toks = prompt.split()
    if not toks or toks[0] != "THM" or "SEQ" not in toks or toks[-1] != "PRF":
        raise NDParseError("bad prompt")
    i = 1
    premises: list[Formula] = []
    if toks[i] != "SEQ":
        while True:
            f, i = parse_formula(toks, i)
            premises.append(f)
            if i < len(toks) and toks[i] == ",":
                i += 1
                continue
            break
    if i >= len(toks) or toks[i] != "SEQ":
        raise NDParseError("bad prompt")
    conclusion, i = parse_formula(toks, i + 1)
    if i != len(toks) - 1:
        raise NDParseError("bad prompt")
    return premises, conclusion


def lean_formula_tokens(f: Formula) -> list[str]:
    """A formula as Lean tokens, fully parenthesised."""
    if isinstance(f, Atom):
        return [f.name]
    if isinstance(f, Bot):
        return ["False"]
    if isinstance(f, Not):
        return ["(", "¬", *lean_formula_tokens(f.sub), ")"]
    return [
        "(",
        *lean_formula_tokens(f.left),
        LEAN_BINOPS[f.op],
        *lean_formula_tokens(f.right),
        ")",
    ]


def nd_formula_text(f: Formula) -> str:
    """A formula in the ND token format."""
    if isinstance(f, Atom):
        return f.name
    if isinstance(f, Bot):
        return "F"
    if isinstance(f, Not):
        return f"( ~ {nd_formula_text(f.sub)} )"
    return f"( {nd_formula_text(f.left)} {ND_OF_BINOP[f.op]} {nd_formula_text(f.right)} )"


# --------------------------------------------------------------------------------------------------
# ND -> Lean tokens

MAX_NAMES = 64
MAX_PREMISES = 8
FORMULA_SYMBOLS = ("(", ")", "¬", "∧", "∨", "→", "P", "Q", "R", "S", "False")
PROMPT_SYMBOLS = ("theorem", "t", ":", "Prop", ":=", "by")
TACTIC_SYMBOLS = (
    "have",
    "exact",
    ";",
    "fun",
    "=>",
    "⟨",
    "⟩",
    ",",
    ".1",
    ".2",
    ".elim",
    "Or.inl",
    "Or.inr",
    "Or.elim",
    "Classical.byContradiction",
    "hh",
)


@dataclass(frozen=True)
class Name:
    """A hypothesis name, identified by the ND line index it stands for."""

    idx: int


Token: TypeAlias = str | Name


def prompt_tokens(prompt: str) -> list[str]:
    """The Lean theorem statement for an ND prompt, as tokens."""
    premises, conclusion = parse_prompt(prompt)
    if len(premises) > MAX_PREMISES:
        raise ValueError(f"more than {MAX_PREMISES} premises")
    out = ["theorem", "t", "(", "P", "Q", "R", "S", ":", "Prop", ")"]
    for j, p in enumerate(premises, start=1):
        out += ["(", f"h{j}", ":", *lean_formula_tokens(p), ")"]
    return [*out, ":", *lean_formula_tokens(conclusion), ":=", "by"]


def proof_tokens(body: str) -> list[Token]:
    """An ND proof body as Lean tokens, hypothesis names still as ``Name`` (ND line index).

    A closed box that no rule cites is dropped, as the Lean rendering has no place for it.
    """
    lines = parse_proof(body.split())
    if not lines:
        raise NDParseError("empty proof")
    stack: list[list[Token]] = [[]]  # statements of each open box; stack[0] is the top level
    hyps: list[tuple[int, Formula] | None] = [None]  # (AS line, formula) of each open box
    lasts: list[int | None] = [None]  # last line of each open box
    closed: dict[int, tuple[Formula, list[Token], int | None]] = {}  # AS line -> the closed box
    n_premises = 0

    def close() -> None:
        body_toks = stack.pop()
        hyp = hyps.pop()
        last = lasts.pop()
        if hyp is None:
            raise NDParseError("unbalanced boxes")
        closed[hyp[0]] = (hyp[1], body_toks, last)

    def box(start: int, end: int, neg: bool = False) -> list[Token]:
        if start not in closed or closed[start][2] != end:
            raise NDParseError(f"box N{start}-N{end} is not a closed box")
        hf, body_toks, _ = closed[start]
        head: list[Token] = [
            "(",
            "fun",
            "(",
            Name(start),
            ":",
            *lean_formula_tokens(hf),
            ")",
            "=>",
            "by",
        ]
        ending: list[Token] = (
            ["exact", "(", Name(end), ":", "False", ")"] if neg else ["exact", Name(end)]
        )
        return [*head, *body_toks, *ending, ")"]

    def need(ln: NDLine, k: int) -> tuple[int, ...]:
        if len(ln.refs) != k:
            raise NDParseError(f"N{ln.idx}: {ln.rule} cites {len(ln.refs)} lines, expected {k}")
        return ln.refs

    for ln in lines:
        if ln.rule == "AS":
            while len(stack) - 1 >= ln.depth:
                close()
            stack.append([])
            hyps.append((ln.idx, ln.formula))
            lasts.append(ln.idx)
            continue
        while len(stack) - 1 > ln.depth:
            close()
        term: list[Token]
        if ln.rule == "PR":
            n_premises += 1
            term = [f"h{n_premises}"]
        elif ln.rule == "R":
            term = [Name(need(ln, 1)[0])]
        elif ln.rule == "ANDI":
            a, b = need(ln, 2)
            term = ["⟨", Name(a), ",", Name(b), "⟩"]
        elif ln.rule in ("ANDE1", "ANDE2"):
            term = [Name(need(ln, 1)[0]), ".1" if ln.rule == "ANDE1" else ".2"]
        elif ln.rule == "IMPE":
            a, b = need(ln, 2)
            term = [Name(a), Name(b)]
        elif ln.rule == "NEGE":
            a, b = need(ln, 2)
            term = [Name(b), Name(a)]
        elif ln.rule in ("ORI1", "ORI2"):
            term = ["Or.inl" if ln.rule == "ORI1" else "Or.inr", Name(need(ln, 1)[0])]
        elif ln.rule == "BOTE":
            term = [Name(need(ln, 1)[0]), ".elim"]
        elif ln.rule == "DN":
            a = need(ln, 1)[0]
            term = ["Classical.byContradiction", "(", "fun", "hh", "=>", Name(a), "hh", ")"]
        elif ln.rule in ("IMPI", "NEGI"):
            s, e = need(ln, 2)
            term = box(s, e, neg=ln.rule == "NEGI")
        elif ln.rule == "ORE":
            j, s1, e1, s2, e2 = need(ln, 5)
            term = ["Or.elim", Name(j), *box(s1, e1), *box(s2, e2)]
        else:
            raise NDParseError(f"unknown rule {ln.rule}")
        stack[-1] += ["have", Name(ln.idx), ":", *lean_formula_tokens(ln.formula), ":=", *term, ";"]
        lasts[-1] = ln.idx
    if len(stack) != 1:
        raise NDParseError("the proof ends inside a box")
    return [*stack[0], "exact", Name(lines[-1].idx)]


def proof_symbols(body: str) -> list[str]:
    """An ND proof body as ``lean_seq`` symbols, names ``n1``, ``n2``, ... by first appearance."""
    order: dict[int, int] = {}
    out: list[str] = []
    for t in proof_tokens(body):
        if isinstance(t, Name):
            k = order.setdefault(t.idx, len(order) + 1)
            if k > MAX_NAMES:
                raise ValueError(f"more than {MAX_NAMES} names")
            out.append(f"n{k}")
        else:
            out.append(t)
    return out


# --------------------------------------------------------------------------------------------------
# The tokenizer


class LeanSeqTokenizer:
    """Token ids for the ``lean_seq`` rendering. Vocabulary: 107 symbols, the fork's order.

    Besides its own methods it implements the tokenizer interface of the ND experiment pipeline
    (``ProofTokenizer`` in ``experiments/current/r1_cot/vendor/nd_model.py``): ``pad_id``,
    ``eos_id``, ``prf_id``, ``len()``, ``encode_prompt``, ``encode_record``, ``decode_body`` and
    ``state_dict`` / ``from_state_dict``, so ``encode_records``, ``train_steps``,
    ``generate_bodies`` and ``save_checkpoint`` work with it unchanged. A training sequence is the
    prompt ids followed by the proof ids (ending in ``<eos>``), the fork's layout; there is no
    ``<BOS>``.
    """

    VERSION = "lean-seq-v1"

    def __init__(self) -> None:
        self.itos: list[str] = [
            "<pad>",
            "<eos>",
            *FORMULA_SYMBOLS,
            *PROMPT_SYMBOLS,
            *TACTIC_SYMBOLS,
            *(f"h{k}" for k in range(1, MAX_PREMISES + 1)),
            *(f"n{k}" for k in range(1, MAX_NAMES + 1)),
        ]
        self.stoi: dict[str, int] = {s: i for i, s in enumerate(self.itos)}
        self.pad_id = self.stoi["<pad>"]
        self.eos_id = self.stoi["<eos>"]
        # A prompt ends with its first ``by`` (``... : C := by``), so the pipeline finds the prompt
        # length as ``ids.index(prf_id) + 1``, as it does with ``PRF`` in the ND token format.
        self.prf_id = self.stoi["by"]
        self.first_name = self.stoi["n1"]

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def __len__(self) -> int:
        return len(self.itos)

    def state_dict(self) -> dict[str, object]:
        """What a checkpoint stores to identify the tokenizer (the pipeline's convention)."""
        return {"version": self.VERSION, "tokens": list(self.itos)}

    @classmethod
    def from_state_dict(cls, state: dict[str, object]) -> LeanSeqTokenizer:
        """The tokenizer a checkpoint was trained with; raises ``ValueError`` for any other."""
        tok = cls()
        if state != tok.state_dict():
            raise ValueError("not a lean_seq tokenizer state (version or vocabulary differs)")
        return tok

    def encode_prompt(self, prompt: str) -> list[int]:
        """The theorem statement as ids."""
        return [self.stoi[t] for t in prompt_tokens(prompt)]

    def encode_proof(self, body: str) -> list[int]:
        """An ND proof body as ids, names numbered by first appearance, ``<eos>`` appended."""
        return [*(self.stoi[t] for t in proof_symbols(body)), self.eos_id]

    def encode_record(self, record: str | dict[str, str]) -> list[int]:
        """One training sequence: prompt ids, then proof ids ending in ``<eos>``.

        ``record`` is the ND text ``"THM ... PRF <proof>"``, a dict with that text under ``"text"``
        (the pipeline's record format), or a dict with ``"prompt"`` and ``"proof"``.
        """
        if isinstance(record, dict) and "prompt" in record:
            prompt, proof = record["prompt"], record["proof"]
        else:
            words = (record if isinstance(record, str) else record["text"]).split()
            if "PRF" not in words:
                raise NDParseError("record has no PRF")
            k = words.index("PRF") + 1
            prompt, proof = " ".join(words[:k]), " ".join(words[k:])
        return self.encode_prompt(prompt) + self.encode_proof(proof)

    def shift_names(self, ids: Sequence[int], rng: random.Random) -> list[int]:
        """Add one random offset to every name id (training augmentation); names stay in n1..n64."""
        top = max((x - self.first_name + 1 for x in ids if x >= self.first_name), default=0)
        if top == 0:
            return list(ids)
        s = rng.randint(0, MAX_NAMES - top)
        return [x + s if x >= self.first_name else x for x in ids]

    def tokens(self, ids: Sequence[int]) -> tuple[list[str], bool]:
        """Ids up to ``<eos>`` as symbols, and whether ``<eos>`` was reached (skips padding)."""
        toks: list[str] = []
        for x in ids:
            if x == self.pad_id:
                continue
            if x == self.eos_id:
                return toks, True
            toks.append(self.itos[x])
        return toks, False

    def lean_text(self, ids: Sequence[int]) -> str | None:
        """The Lean tactic text of a sample, or None if it never reached ``<eos>``."""
        toks, ended = self.tokens(ids)
        return text(toks) if ended else None

    def statement(self, prompt: str) -> str:
        """The Lean theorem statement for an ND prompt, as text."""
        return " ".join(prompt_tokens(prompt))

    def decode(self, ids: Sequence[int]) -> str:
        """A sample as the ND proof body it denotes, or ``"LEANPARSE <reason>"`` if ungrammatical.

        The ``LEANPARSE`` string is the take-home convention: ND verifiers reject it, so it can flow
        through ND tooling as an ordinary failed sample.
        """
        toks, ended = self.tokens(ids)
        if not ended:
            return "LEANPARSE no-eos"
        try:
            return inverse(toks)
        except LeanParseError as e:
            return f"LEANPARSE {e}"

    def decode_body(self, ids: Sequence[int]) -> str:
        """``decode`` under the pipeline's name: the ND body a sample denotes, or a ``LEANPARSE``
        marker. The body is only what the sample means; whether it counts is Lean's call
        (``code.shared_code.lean_check.judge``)."""
        return self.decode(ids)


def text(toks: Sequence[str]) -> str:
    """Lean symbols -> one line of Lean source (``.1``, ``.2`` and ``.elim`` glued to the name)."""
    out: list[str] = []
    for t in toks:
        if t in (".1", ".2", ".elim") and out:
            out[-1] += t
        else:
            out.append(t)
    return " ".join(out)


# --------------------------------------------------------------------------------------------------
# Lean tokens -> ND (the inverse)

_TO_ND = {
    "(": "(",
    ")": ")",
    "¬": "~",
    "∧": "&",
    "∨": "v",
    "→": ">",
    "P": "P",
    "Q": "Q",
    "R": "R",
    "S": "S",
    "False": "F",
}


def parse_lean_formula(toks: Sequence[str], i: int) -> tuple[Formula, int]:
    """Parse one formula in ``lean_formula_tokens``' form starting at ``toks[i]``; return it and the
    index after it. Raises ``LeanParseError`` if there is none."""
    nd: list[str] = []
    j = i
    while j < len(toks) and toks[j] in _TO_ND:
        nd.append(_TO_ND[toks[j]])
        j += 1
    try:
        f, used = parse_formula([*nd, "$"], 0)
    except NDParseError as e:
        raise LeanParseError("formula") from e
    return f, i + used


def _is_name(t: str) -> bool:
    return len(t) > 1 and t[0] == "n" and t[1:].isdigit()


def _is_premise(t: str) -> bool:
    return len(t) > 1 and t[0] == "h" and t[1:].isdigit()


class _Inverse:
    """Recursive-descent parser for the strict ``lean_seq`` grammar, emitting ND lines."""

    def __init__(self, toks: Sequence[str]) -> None:
        self.toks = toks
        self.pos = 0
        self.lines: list[str] = []
        self.forms: dict[int, Formula] = {}
        self.n_premises = 0
        self.seen_non_premise = False

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
        if not _is_name(t):
            raise LeanParseError("name")
        return t

    def formula(self) -> Formula:
        f, self.pos = parse_lean_formula(self.toks, self.pos)
        return f

    def ref(self, scope: dict[str, int]) -> int:
        t = self.name()
        if t not in scope:
            raise LeanParseError("unbound")
        return scope[t]

    def emit(self, depth: int, f: Formula, rule: str, refs: Sequence[int]) -> int:
        i = len(self.lines) + 1
        cites = "".join(f" N{r}" for r in refs)
        self.lines.append(f"N{i} " + "| " * depth + f"{nd_formula_text(f)} : {rule}{cites} ;")
        self.forms[i] = f
        return i

    def box(self, scope: dict[str, int], depth: int) -> tuple[int, int, bool]:
        """``( fun ( n : A ) => by stmts exact .. )`` -> (AS line, last line, ends in False)."""
        for x in ("(", "fun", "("):
            self.eat(x)
        nm = self.name()
        self.eat(":")
        hf = self.formula()
        for x in (")", "=>", "by"):
            self.eat(x)
        start = self.emit(depth + 1, hf, "AS", [])
        inner = dict(scope)
        inner[nm] = start
        last = self.stmts(inner, depth + 1, start)
        self.eat("exact")
        if self.peek() == "(":
            self.eat("(")
            end = self.ref(inner)
            for x in (":", "False", ")"):
                self.eat(x)
            neg = True
        else:
            end = self.ref(inner)
            neg = False
        if end != last:
            raise LeanParseError("exact not last")
        self.eat(")")
        return start, end, neg

    def term(self, scope: dict[str, int], depth: int) -> tuple[str, list[int]]:
        t = self.peek()
        if t == "⟨":
            self.eat()
            a = self.ref(scope)
            self.eat(",")
            b = self.ref(scope)
            self.eat("⟩")
            return "ANDI", [a, b]
        if t in ("Or.inl", "Or.inr"):
            self.eat()
            return ("ORI1" if t == "Or.inl" else "ORI2"), [self.ref(scope)]
        if t == "Classical.byContradiction":
            for x in ("Classical.byContradiction", "(", "fun", "hh", "=>"):
                self.eat(x)
            a = self.ref(scope)
            self.eat("hh")
            self.eat(")")
            return "DN", [a]
        if t == "(":
            s, e, neg = self.box(scope, depth)
            return ("NEGI" if neg else "IMPI"), [s, e]
        if t == "Or.elim":
            self.eat()
            j = self.ref(scope)
            s1, e1, neg1 = self.box(scope, depth)
            s2, e2, neg2 = self.box(scope, depth)
            if neg1 or neg2:
                raise LeanParseError("ORE box")
            return "ORE", [j, s1, e1, s2, e2]
        a = self.ref(scope)
        t2 = self.peek()
        if t2 == ".1":
            self.eat()
            return "ANDE1", [a]
        if t2 == ".2":
            self.eat()
            return "ANDE2", [a]
        if t2 == ".elim":
            self.eat()
            return "BOTE", [a]
        if t2 == ";":
            return "R", [a]
        b = self.ref(scope)
        return ("NEGE", [b, a]) if isinstance(self.forms[a], Not) else ("IMPE", [a, b])

    def stmts(self, scope: dict[str, int], depth: int, last: int | None) -> int | None:
        while self.peek() == "have":
            self.eat("have")
            nm = self.name()
            self.eat(":")
            f = self.formula()
            self.eat(":=")
            t = self.peek()
            if t is None:
                raise LeanParseError("eof")
            if _is_premise(t):
                self.eat()
                k = int(t[1:])
                if depth != 0 or self.seen_non_premise or k != self.n_premises + 1:
                    raise LeanParseError("PR position")
                self.n_premises += 1
                i = self.emit(0, f, "PR", [])
            else:
                self.seen_non_premise = True
                rule, refs = self.term(scope, depth)
                i = self.emit(depth, f, rule, refs)
            self.eat(";")
            scope[nm] = i
            last = i
        return last

    def parse(self) -> str:
        top: dict[str, int] = {}
        last = self.stmts(top, 0, None)
        self.eat("exact")
        end = self.ref(top)
        if self.pos != len(self.toks):
            raise LeanParseError("trailing tokens")
        if last is None or end != last:
            raise LeanParseError("exact not last")
        return " ".join(self.lines) + " QED"


def inverse(toks: Sequence[str]) -> str:
    """Lean symbols in the strict ``lean_seq`` grammar -> the ND proof body they denote.

    Names resolve lexically (the latest binding wins, as in Lean). Raises ``LeanParseError`` outside
    the grammar: premise lines come first and in order, a box's ``exact`` cites the box's last line,
    and the final ``exact`` cites the last top-level line.
    """
    return _Inverse(toks).parse()


# --------------------------------------------------------------------------------------------------
# Term size

# Lean's elaborated term size per ND line of the ``lean_seq`` rendering (see ``term_size``).
TERM_SIZE = {
    "PR": 0,
    "AS": 0,
    "R": 0,
    "ANDI": 1,
    "ANDE1": 1,
    "ANDE2": 1,
    "IMPE": 1,
    "NEGE": 1,
    "ORI1": 1,
    "ORI2": 1,
    "BOTE": 1,
    "IMPI": 1,
    "NEGI": 1,
    "ORE": 3,
    "DN": 3,
}


def term_size(toks: Sequence[str]) -> int:
    """The size of the proof term Lean elaborates from ``lean_seq`` symbols (a proof's text, e.g.
    ``proof_symbols(body)``); raises ``LeanParseError`` outside the strict grammar.

    The fork's definition (``lean_check.py``): the inference nodes of the elaborated value, 1 per
    application headed by an inference constant (``And.intro``, ``Or.inl``, ``False.elim``,
    ``Classical.byContradiction``, ...), 1 per argument of an application headed by a hypothesis,
    1 per ``fun`` binder and 1 per projection; a ``have`` adds nothing, and names and types count
    0. On this fragment that is a weight per line (``TERM_SIZE``): a premise or a reiteration is
    free, ``Or.elim`` with its two binders is 3 and ``Classical.byContradiction ( fun hh => a hh )``
    3. One exception: ``n .elim`` on a conjunction is 2, as Lean eta-expands it to
    ``fun f => And.elim f n``. Unlike a line count, it does not depend on how a proof is split into
    lines. ``tests/test_lean_seq.py`` checks it against Lean's own count.
    """
    lines = parse_proof(inverse(toks).split())
    formulas = {ln.idx: ln.formula for ln in lines}
    size = 0
    for ln in lines:
        size += TERM_SIZE[ln.rule]
        if ln.rule == "BOTE":
            cited = formulas[ln.refs[0]]
            size += isinstance(cited, BinOp) and cited.op == "and"
    return size
