"""Two more sources read for the candidate sets (docs/textbook-problem/diversification/sources.md,
status 2026-09-30):

- Pelletier (1986), "Seventy-five problems for testing automatic theorem provers", propositional
  problems 1-17. The paper is behind a paywall, so the statements are read from two open
  transcriptions that cite it by number: Isabelle's ``src/FOL/ex/Classical.thy`` (all 17, as
  single formulas) and TPTP v9.3.1 ``SYN040+1`` ... ``SYN047+1`` (6 of them, with problem 10 in its
  original premises-and-conclusion form). Where both exist they are cross-checked.
- The Oxford *Logic Manual* course: Halbach's *Exercises Booklet* (2018/19) exercise 6.1 and
  Fritz's *More Exercises* (2014) exercise 6.1, the propositional natural deduction exercises.
  Read with pypdf; the PDFs print numerals through a font whose glyph names come out as
  ``/one.oldstyle`` and the turnstile as ``/uni22A2``.

Each reader yields ``(where, number, original, text, notes)`` like ``pdf_sources``.
"""

import os
import re

from pdf_sources import FORMULA, to_ascii

# ---------------------------------------------------------------- Pelletier

ISABELLE_SYMS = {
    r"\<longrightarrow>": "→",
    r"\<longleftrightarrow>": "↔",
    r"\<and>": "∧",
    r"\<or>": "∨",
    r"\<not>": "¬",
}
TPTP_SYMS = {"<=>": "↔", "=>": "→", "&": "∧", "|": "∨", "~": "¬"}
PEL_TEXT = re.compile(r"text\\<open>(\d+)[^\\]*\\<close>\s*\nlemma \\<open>(.*?)\\<close>", re.S)
PEL_SKIPPED = [("problems 18-75", "first-order")]


def _isabelle_formula(s):
    for k, v in ISABELLE_SYMS.items():
        s = s.replace(k, v)
    return re.sub(r"\s+", " ", s).strip()


def _tptp_sequent(path):
    """(premises, goal) from the fof lines of a TPTP problem file."""
    with open(path) as f:
        text = f.read()
    fofs = re.findall(r"fof\((\w+),(axiom|conjecture),\s*(.*?)\)\s*\.", text, re.S)
    prem, goal = [], None
    for _, role, f in fofs:
        f = re.sub(r"\s+", " ", f).strip()
        for k, v in TPTP_SYMS.items():
            f = f.replace(k, v)
        f = re.sub(r"\b[a-z]\b", lambda m: m.group().upper(), f)  # TPTP writes p, q, r
        if f.startswith("(") and f.endswith(")"):
            f = f[1:-1].strip()
        if role == "axiom":
            prem.append(f)
        else:
            goal = f
    return prem, goal


def pelletier_items(isabelle_thy):
    """Problems 1-17 from Classical.thy; problem 10 from TPTP in its premises form when present."""
    tptp_dir = os.path.join(os.path.dirname(isabelle_thy), "..", "tptp")
    tptp = {}
    for fn in sorted(os.listdir(tptp_dir)) if os.path.isdir(tptp_dir) else []:
        p = os.path.join(tptp_dir, fn)
        with open(p) as f:
            m = re.search(r"Problem\s*:\s*Pelletier Problem (\d+)", f.read())
        if m:
            tptp[int(m.group(1))] = p
    with open(isabelle_thy) as f:
        thy = f.read()
    for num, body in PEL_TEXT.findall(thy):
        n = int(num)
        if n > 17:
            break
        formula = _isabelle_formula(body)
        notes = ["read from Isabelle src/FOL/ex/Classical.thy (BSD-3), which cites Pelletier 1986"]
        original = f"⊢ {formula}"
        if n in tptp:
            prem, goal = _tptp_sequent(tptp[n])
            if prem:
                original = ", ".join(prem) + " ⊢ " + goal
                notes.append(
                    f"stated with premises as in TPTP {os.path.basename(tptp[n])}; Isabelle "
                    "writes the same problem as one conditional"
                )
            else:
                notes.append(f"cross-checked against TPTP {os.path.basename(tptp[n])}")
        where = f"Pelletier 1986, problem {n}"
        text = to_ascii(original) if re.fullmatch(FORMULA + "*", original) else None
        yield where, f"P{n}", original, text, notes


# ---------------------------------------------------------------- Oxford

OLDSTYLE = {
    "/zero.oldstyle": "0",
    "/one.oldstyle": "1",
    "/two.oldstyle": "2",
    "/three.oldstyle": "3",
    "/four.oldstyle": "4",
    "/five.oldstyle": "5",
    "/six.oldstyle": "6",
    "/seven.oldstyle": "7",
    "/eight.oldstyle": "8",
    "/nine.oldstyle": "9",
}
OX_ITEM = re.compile(r"^\((i|ii|iii|iv|v|vi|vii|viii|ix|x)\)\s*(.*)$")
OX_SKIPPED = [
    ("booklet 6.2", "one argument with six premises and subscripted letters (7 letters)"),
    ("booklet 6.3-6.5, Fritz 6.2", "first-order"),
    ("Fritz 6.3-6.4", "find the mistakes; questions about the rules"),
]


def _oxford_text(pdf):
    from pypdf import PdfReader

    out = []
    for p in PdfReader(pdf).pages:
        t = p.extract_text() or ""
        for k, v in OLDSTYLE.items():
            t = t.replace(k, v)
        out.append(t.replace("/uni22A2", "⊢"))
    return "\n".join(out).split("\n")


def _oxford_section(lines, start_pat, stop_pat):
    on = False
    for line in lines:
        s = line.strip()
        if re.search(start_pat, s):
            on = True
            continue
        if on and re.search(stop_pat, s):
            return
        if on:
            m = OX_ITEM.match(s)
            if m:
                yield m.group(1), m.group(2)


def oxford_items(booklet_pdf):
    fritz_pdf = os.path.join(os.path.dirname(booklet_pdf), "exercises_peter.pdf")
    parts = [
        (
            booklet_pdf,
            "Halbach, Exercises Booklet 2018/19, exercise 6.1",
            r"6\.1\.Establishthefollowingclaims",
            r"6\.2\.",
        ),
        (
            fritz_pdf,
            "Fritz, More Exercises 2014, exercise 6.1",
            r"^Exercise 6\.1\.",
            r"^Exercise 6\.2\.",
        ),
    ]
    for pdf, label, start, stop in parts:
        for roman, body in _oxford_section(_oxford_text(pdf), start, stop):
            body = re.sub(r"\s+", " ", body).strip()
            notes = ["read from the PDF text (pypdf); ⊢ decoded from /uni22A2"]
            text = to_ascii(body) if re.fullmatch(FORMULA + "*", body) else None
            if text is None:
                notes = ["subscripted letters or symbols outside the propositional alphabet"]
            yield f"{label} ({roman})", roman, body, text, notes
