#!/usr/bin/env python3
"""Premise notes: habits 5 to 19 of the linked-habits experiment (linked_habits_20261127, SPEC.md).

    python notes.py table [--force-check]    # draw the form table (seed 20261127) -> LH/data/note_table.json
    python notes.py show                     # print the table
    python notes.py read "<prompt>" "<proof>"

A **note** is one extra top-level line per premise, written right after the PR block, in premise order, before any
other line. It is derived from that premise's PR line `Nj` (formula X) by one menu rule and is never cited:

    copy  X : R Nj                       or2  ( X v X ) : ORI1 Nj
    and2  ( X & X ) : ANDI Nj Nj         orF  ( X v F ) : ORI1 Nj       For  ( F v X ) : ORI2 Nj
    orN   ( X v ( ~ X ) ) : ORI1 Nj      Nor  ( ( ~ X ) v X ) : ORI2 Nj

(`NOTE_MENU`; CORPUS.md says whether the doubling forms were kept.) The premise's **class** (`premise_class`) is its
shape: `atom` (P, Q, R, S or F), `~a` / `~c` (negation of an atom / a compound), and for each of `&`, `v`, `>` the
four `aa`, `ac`, `ca`, `cc` (left and right side atomic or compound), e.g. `&ac`, `v cc` written `vcc`, `>ca`.

The **form table** (`load_table`, LH/data/note_table.json) gives each class two distinct menu forms, A's and B's,
drawn once with seed 20261127 under the balance rule: each author uses every one of the 7 menu forms for 2 classes
(one form for 3, since 15 = 7 x 2 + 1), so every form belongs to both authors; A's and B's form differ in every class.

API (Part B imports these; names and return shapes are fixed):

    premise_class(formula_tokens) -> class name            (tokens: list or space-separated string)
    NOTE_MENU                     {form: {"formula": template with X, "rule": ...}}, 7 forms in menu order
    load_table(path=TABLE_PATH)   -> {class: {"A": form, "B": form}}
    add_notes(prompt, proof, form_of_class) -> proof       form_of_class: {class: form name}; renumbers N labels
    read_notes(prompt, proof, table=None) -> [{premise_index, cls, form, side, line, pos}, ...]
    note_reading(prompt, proof, table=None) -> {ok, err, n_prem, notes, late, late_forms, unnoted, unnoted_classes}
    strip_notes(prompt, proof) -> proof                    (the header notes removed, renumbered)

`premise_index` is 0-based in premise order; `side` is "A", "B" or "other" (a menu form that is neither author's
form for that class); `line` is the note's N number, `pos` its 0-based position among the proof's notes.
"""

# ruff: noqa: E501
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Any

SEED = 20261127
LH = Path(os.environ.get("LH", "/scratch/cowifornia/nd-rl/linked_habits"))
TABLE_PATH = LH / "data" / "note_table.json"

ATOMS = ("P", "Q", "R", "S", "F")
OPS = {"&": "&", "v": "v", ">": ">"}
CLASSES = (
    "atom", "~a", "~c",
    "&aa", "&ac", "&ca", "&cc",
    "vaa", "vac", "vca", "vcc",
    ">aa", ">ac", ">ca", ">cc",
)  # fmt: skip

# The menu: form -> the note's formula (X = the premise's formula) and rule. ANDI cites the premise twice, the others
# once. Order is the menu order of SPEC.md.
NOTE_MENU: dict[str, dict[str, str]] = {
    "copy": {"formula": "X", "rule": "R"},
    "and2": {"formula": "( X & X )", "rule": "ANDI"},
    "or2": {"formula": "( X v X )", "rule": "ORI1"},
    "orF": {"formula": "( X v F )", "rule": "ORI1"},
    "For": {"formula": "( F v X )", "rule": "ORI2"},
    "orN": {"formula": "( X v ( ~ X ) )", "rule": "ORI1"},
    "Nor": {"formula": "( ( ~ X ) v X )", "rule": "ORI2"},
}
# The junk-only replacements SPEC.md names for the doubling forms (used only if the context check requires them).
JUNK_FORMS: dict[str, dict[str, str]] = {
    "orNF": {"formula": "( X v ( ~ F ) )", "rule": "ORI1"},
    "NFor": {"formula": "( ( ~ F ) v X )", "rule": "ORI2"},
    "orFF": {"formula": "( X v ( F > F ) )", "rule": "ORI1"},
    "FFor": {"formula": "( ( F > F ) v X )", "rule": "ORI2"},
}
NOTE_RULES = frozenset(m["rule"] for m in NOTE_MENU.values())
LABEL = re.compile(r"^N(\d+)$")


# ============================================================================ formulas and classes
def _toks(formula_tokens: str | list[str] | tuple[str, ...]) -> list[str]:
    return formula_tokens.split() if isinstance(formula_tokens, str) else list(formula_tokens)


def _operand(toks: list[str], i: int) -> int:
    """End (exclusive) of the formula starting at toks[i]: an atom or a parenthesised formula."""
    if toks[i] != "(":
        return i + 1
    depth = 0
    for j in range(i, len(toks)):
        depth += toks[j] == "("
        depth -= toks[j] == ")"
        if depth == 0:
            return j + 1
    raise ValueError(f"unbalanced formula: {' '.join(toks)}")


def _is_atom(toks: list[str]) -> bool:
    return len(toks) == 1 and toks[0] in ATOMS


def premise_class(formula_tokens: str | list[str] | tuple[str, ...]) -> str:
    """The shape class of a premise formula (one of CLASSES). Raises ValueError for a formula it cannot read."""
    t = _toks(formula_tokens)
    if _is_atom(t):
        return "atom"
    if len(t) < 4 or t[0] != "(" or t[-1] != ")" or _operand(t, 0) != len(t):
        raise ValueError(f"not a formula: {' '.join(t)}")
    inner = t[1:-1]
    if inner[0] == "~":
        sub = inner[1:]
        if _operand(sub, 0) != len(sub):
            raise ValueError(f"not a formula: {' '.join(t)}")
        return "~a" if _is_atom(sub) else "~c"
    e = _operand(inner, 0)
    if e >= len(inner) or inner[e] not in OPS:
        raise ValueError(f"not a formula: {' '.join(t)}")
    left, right = inner[:e], inner[e + 1 :]
    if not right or _operand(right, 0) != len(right):
        raise ValueError(f"not a formula: {' '.join(t)}")
    return OPS[inner[e]] + ("a" if _is_atom(left) else "c") + ("a" if _is_atom(right) else "c")


def note_formula(form: str, x: str | list[str]) -> list[str]:
    """The note formula of menu (or junk) form `form` for premise formula X, as tokens."""
    m = NOTE_MENU.get(form) or JUNK_FORMS[form]
    xs = _toks(x)
    out: list[str] = []
    for tok in m["formula"].split():
        out.extend(xs if tok == "X" else [tok])
    return out


def note_rule(form: str) -> str:
    return (NOTE_MENU.get(form) or JUNK_FORMS[form])["rule"]


# ============================================================================ proof lines (text level)
class Line:
    """One proof line: label number, box depth, formula tokens, rule, cited labels."""

    __slots__ = ("n", "depth", "f", "rule", "refs")

    def __init__(self, n: int, depth: int, f: list[str], rule: str, refs: list[int]):
        self.n, self.depth, self.f, self.rule, self.refs = n, depth, f, rule, refs

    def toks(self) -> list[str]:
        return [
            f"N{self.n}",
            *["|"] * self.depth,
            *self.f,
            ":",
            self.rule,
            *[f"N{r}" for r in self.refs],
            ";",
        ]


def parse_lines(proof: str) -> tuple[list[Line], list[str]]:
    """Proof text -> (lines, the tokens after the last ';', normally ['QED']). Raises ValueError when a line is not
    `N<i> |* formula : RULE N<j>* ;`."""
    toks = proof.split()
    lines: list[Line] = []
    cur: list[str] = []
    for t in toks:
        if t != ";":
            cur.append(t)
            continue
        m = LABEL.match(cur[0]) if cur else None
        if m is None or ":" not in cur:
            raise ValueError(f"bad line: {' '.join(cur)}")
        k = 1
        while k < len(cur) and cur[k] == "|":
            k += 1
        c = cur.index(":")
        if c <= k or c + 1 >= len(cur):
            raise ValueError(f"bad line: {' '.join(cur)}")
        refs = []
        for r in cur[c + 2 :]:
            mr = LABEL.match(r)
            if mr is None:
                raise ValueError(f"bad citation in: {' '.join(cur)}")
            refs.append(int(mr.group(1)))
        lines.append(Line(int(m.group(1)), k - 1, cur[k:c], cur[c + 1], refs))
        cur = []
    return lines, cur


def render_lines(lines: list[Line], tail: list[str]) -> str:
    """Lines -> proof text, relabelled N1, N2, ... in order, with every citation following its line."""
    new = {ln.n: i + 1 for i, ln in enumerate(lines)}
    if len(new) != len(lines):
        raise ValueError("duplicate line labels")
    out: list[str] = []
    for i, ln in enumerate(lines):
        out.extend(Line(i + 1, ln.depth, ln.f, ln.rule, [new[r] for r in ln.refs]).toks())
    return " ".join(out + tail)


def _pr_block(lines: list[Line]) -> int:
    k = 0
    while k < len(lines) and lines[k].rule == "PR":
        k += 1
    return k


# ============================================================================ table
def draw_table(seed: int = SEED, menu: tuple[str, ...] | None = None) -> dict[str, dict[str, str]]:
    """{class: {"A": form, "B": form}} under the balance rule (module docstring): each author's 15 forms are the 7 menu
    forms twice plus one form drawn at random, shuffled over the classes; drawn again (same rng) until A's and B's form
    differ in every class. Deterministic in `seed`."""
    menu = tuple(menu or NOTE_MENU)
    rng = random.Random(seed)
    while True:
        cols = {}
        for side in ("A", "B"):
            pool = [f for f in menu for _ in range(2)]
            pool += rng.sample(menu, len(CLASSES) - len(pool))
            rng.shuffle(pool)
            cols[side] = pool
        if all(a != b for a, b in zip(cols["A"], cols["B"], strict=True)):
            return {c: {"A": cols["A"][i], "B": cols["B"][i]} for i, c in enumerate(CLASSES)}


def check_table(table: dict[str, dict[str, str]], menu: tuple[str, ...] | None = None) -> list[str]:
    """Problems with a table under the balance rule ([] if none)."""
    menu = tuple(menu or NOTE_MENU)
    bad = []
    if sorted(table) != sorted(CLASSES):
        bad.append("classes")
    for side in ("A", "B"):
        cnt = {f: sum(table[c][side] == f for c in table) for f in menu}
        if any(f not in menu for c in table for f in (table[c][side],)):
            bad.append(f"{side}: form outside the menu")
        if min(cnt.values()) < 2 or max(cnt.values()) > 3:
            bad.append(f"{side}: counts {cnt}")
    if any(table[c]["A"] == table[c]["B"] for c in table):
        bad.append("A == B in a class")
    return bad


def table_doc(table: dict[str, dict[str, str]], seed: int = SEED) -> dict[str, Any]:
    menu = list(NOTE_MENU)
    return {
        "seed": seed,
        "menu": {f: NOTE_MENU[f] for f in menu},
        "classes": list(CLASSES),
        "table": table,
        "counts": {s: {f: sum(table[c][s] == f for c in table) for f in menu} for s in ("A", "B")},
        "rule": "notes.draw_table: per author, the 7 menu forms twice plus one drawn at random (random.Random(seed)."
        "sample), shuffled over CLASSES in order; redrawn with the same rng until A != B in every class",
    }


def write_table(path: Path = TABLE_PATH, seed: int = SEED) -> dict[str, Any]:
    """Draw the table and write it; refuses (ValueError) if `path` holds a different table. -> the document."""
    table = draw_table(seed)
    bad = check_table(table)
    if bad:
        raise ValueError(f"table fails the balance rule: {bad}")
    doc = table_doc(table, seed)
    path = Path(path)
    if path.exists():
        old = json.loads(path.read_text())
        if old.get("table") != table or old.get("menu") != doc["menu"] or old.get("seed") != seed:
            raise ValueError(f"{path} holds a different table; refusing to overwrite")
        return old
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(doc, indent=1) + "\n")
    tmp.replace(path)
    return doc


_TABLE: dict[str, Any] = {}


def load_table(path: Path | str = TABLE_PATH) -> dict[str, dict[str, str]]:
    """{class: {"A": form, "B": form}} from the written table (cached per path)."""
    key = str(path)
    if key not in _TABLE:
        doc = json.loads(Path(path).read_text())
        if list(doc["menu"]) != list(NOTE_MENU):
            raise ValueError(
                f"{path}: menu {list(doc['menu'])} differs from notes.NOTE_MENU {list(NOTE_MENU)}"
            )
        _TABLE[key] = doc["table"]
    return _TABLE[key]


def table_sha256(path: Path | str = TABLE_PATH) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ============================================================================ write, read, strip
def premises(proof: str) -> list[list[str]]:
    """The premise formulas (tokens) of a proof's PR block."""
    lines, _ = parse_lines(proof)
    return [ln.f for ln in lines[: _pr_block(lines)]]


def add_notes(prompt: str, proof: str, form_of_class: dict[str, str]) -> str:
    """The proof with one note per premise after the PR block, in premise order; premise j of class c gets form
    form_of_class[c] (KeyError if missing). Every N label and citation after the PR block is renumbered. A proof
    without premises comes back unchanged. `prompt` is unused (the proof carries its premises)."""
    del prompt
    lines, tail = parse_lines(proof)
    k = _pr_block(lines)
    if k == 0:
        return proof
    if k == len(lines):
        raise ValueError(
            "a proof of only premise lines cannot take a note (it would become the last line)"
        )
    top = max(ln.n for ln in lines)
    notes = []
    for i, pr in enumerate(lines[:k]):
        form = form_of_class[premise_class(pr.f)]
        rule = note_rule(form)
        refs = [pr.n, pr.n] if rule == "ANDI" else [pr.n]
        notes.append(Line(top + 1 + i, 0, note_formula(form, pr.f), rule, refs))
    return render_lines(lines[:k] + notes + lines[k:], tail)


def _menu_form(line: Line, prem: dict[int, list[str]]) -> str | None:
    """The menu form `line` is, as a derivation from one PR line (`prem`: label -> formula), or None."""
    if line.rule not in NOTE_RULES or not line.refs or line.refs[0] not in prem:
        return None
    if line.rule == "ANDI":
        if len(line.refs) != 2 or line.refs[0] != line.refs[1]:
            return None
    elif len(line.refs) != 1:
        return None
    x = prem[line.refs[0]]
    for form, m in NOTE_MENU.items():
        if m["rule"] == line.rule and note_formula(form, x) == line.f:
            return form  # forms never coincide for a premise other than F (no F premise in the corpora)
    return None


def note_reading(
    prompt: str, proof: str, table: dict[str, dict[str, str]] | None = None
) -> dict[str, Any]:
    """The note reading of a proof.

    A note is a depth-0 line in the header block (after the last PR line, before the first line that is not a note),
    derived from one PR line by a menu form (rule and formula), never cited, and not the last line.
    -> {ok, err, n_prem, notes: [{premise_index, cls, form, side, line, pos}], late (menu-form dead lines outside the
    header, any depth), late_forms ([{premise_index, cls, form, side, line, depth}]), unnoted (premises with no header
    note), unnoted_classes, header_end (number of lines up to the end of the header)}.
    `table` defaults to load_table(); `prompt` is unused."""
    del prompt
    try:
        lines, _tail = parse_lines(proof)
    except ValueError as e:
        return {"ok": False, "err": str(e), "n_prem": 0, "notes": [], "late": 0, "late_forms": [], "unnoted": 0,
                "unnoted_classes": [], "header_end": 0}  # fmt: skip
    table = load_table() if table is None else table
    k = _pr_block(lines)
    prem = {ln.n: ln.f for ln in lines[:k]}
    pidx = {ln.n: i for i, ln in enumerate(lines[:k])}
    cls: dict[int, str | None] = {}
    for n, f in prem.items():
        try:
            cls[n] = premise_class(f)
        except ValueError:
            cls[n] = None
    cited = {r for ln in lines for r in ln.refs}
    last = len(lines) - 1

    def entry(i: int, form: str) -> dict[str, Any]:
        ln = lines[i]
        c = cls[ln.refs[0]]
        row = table.get(c, {}) if c else {}
        side = "A" if row.get("A") == form else "B" if row.get("B") == form else "other"
        return {
            "premise_index": pidx[ln.refs[0]],
            "cls": c,
            "form": form,
            "side": side,
            "line": ln.n,
        }

    notes: list[dict[str, Any]] = []
    i = k
    while i < last:
        ln = lines[i]
        form = _menu_form(ln, prem) if ln.depth == 0 and ln.n not in cited else None
        if form is None:
            break
        notes.append({**entry(i, form), "pos": len(notes)})
        i += 1
    header_end = i
    late_forms = []
    for j in range(header_end, last):
        ln = lines[j]
        if ln.n in cited:
            continue
        form = _menu_form(ln, prem)
        if form is not None:
            late_forms.append({**entry(j, form), "depth": ln.depth})
    noted = {x["premise_index"] for x in notes}
    unnoted = [i for i in range(k) if i not in noted]
    return {
        "ok": True,
        "err": None,
        "n_prem": k,
        "notes": notes,
        "late": len(late_forms),
        "late_forms": late_forms,
        "unnoted": len(unnoted),
        "unnoted_classes": [cls[lines[i].n] for i in unnoted],
        "header_end": header_end,
    }


def read_notes(
    prompt: str, proof: str, table: dict[str, dict[str, str]] | None = None
) -> list[dict[str, Any]]:
    """The proof's notes: [{premise_index, cls, form, side, line, pos}] in written order (note_reading's `notes`)."""
    return note_reading(prompt, proof, table)["notes"]


def strip_notes(prompt: str, proof: str) -> str:
    """The proof with its header notes (note_reading) removed and every label renumbered; the inverse of add_notes.
    A proof that does not parse comes back unchanged."""
    try:
        lines, tail = parse_lines(proof)
    except ValueError:
        return proof
    rd = note_reading(prompt, proof, table={})
    if not rd["notes"]:
        return proof
    k = _pr_block(lines)
    return render_lines(lines[:k] + lines[rd["header_end"] :], tail)


# ============================================================================ CLI
def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser(
        "table", help="draw and write LH/data/note_table.json (refuses a different existing table)"
    )
    p.add_argument("--path", default=str(TABLE_PATH))
    p = sp.add_parser("show")
    p.add_argument("--path", default=str(TABLE_PATH))
    p = sp.add_parser("read")
    p.add_argument("prompt")
    p.add_argument("proof")
    a = ap.parse_args(argv)
    if a.cmd == "table":
        doc = write_table(Path(a.path))
        print(json.dumps(doc, indent=1))
        print("sha256", table_sha256(a.path), file=sys.stderr)
    elif a.cmd == "show":
        t = load_table(a.path)
        for c in CLASSES:
            print(f"{c:4s} A {t[c]['A']:5s} B {t[c]['B']}")
    else:
        print(json.dumps(note_reading(a.prompt, a.proof), indent=1))


if __name__ == "__main__":
    main()
