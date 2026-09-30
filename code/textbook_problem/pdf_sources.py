"""Proof exercises read out of two PDF sources (S1 pass 3, docs/textbook-problem/diversification/
people_scan.md):

- Roy, *Symbolic Logic: An Accessible Introduction to Serious Mathematical Logic*, section 6.2
  (sentential natural deduction, systems NDs and NDs+). Text is extracted with pdfminer, whose
  output keeps the MathTime symbol font's unmapped glyphs as ``(cid:N)``; the map below decodes
  them. pypdf silently drops the same glyphs (the tilde and the right bracket), so it is not used.
- Carr, *The Natural Deduction Pack* (for Halbach's *Logic Manual*), section 4 practice problems.
  pypdf extracts its text cleanly.

Each reader yields ``(where, number, original, text, notes)``: ``original`` is the item as printed
(decoded to Unicode connectives), ``text`` is the same in the transcriber's ASCII notation
(``~ /\\ \\/ -> <-> :|-:``), ``notes`` are encoding notes to attach.
"""

import os
import re

FORMULA = r"[A-Z¬∧∨→↔⊢()\[\],\s]"
ASCII = str.maketrans({"¬": "~", "∧": "/\\", "∨": "\\/", "→": "->", "↔": "<->", "⊢": ":|-:"})


def to_ascii(s):
    return re.sub(r"\s+", " ", s.translate(ASCII)).strip()


# ---------------------------------------------------------------- Roy

# 0-based PDF page indices holding section 6.2 (printed pages 207–262; PDF page = printed + 12).
ROY_PAGES = range(218, 276)
# Exercises that ask for a derivation of a stated sequent, with the label used in provenance.
# The theorem lists T6.1–T6.20 and T6.21–T6.30 are what E6.16, E6.19 and E6.20e ask to derive.
ROY_EXERCISES = {
    "E6.7": "sec. 6.2.1 (→ and ∧), NDs",
    "E6.10": "sec. 6.2.2 (∼ and ∨), NDs",
    "E6.13": "sec. 6.2.3 (↔), NDs",
    "E6.15": "sec. 6.2.4 Strategy, NDs",
    "E6.18": "sec. 6.2.4 Strategy (contradiction), NDs",
    "E6.20": "sec. 6.2.4 Strategy, 'particularly challenging', NDs",
    "E6.21": "sec. 6.2.5 The system NDs+ (derived rules)",
}
ROY_THEOREMS = {
    range(1, 21): "theorems T6.1–T6.20 (E6.16), NDs",
    range(21, 31): "theorems T6.21–T6.30 (E6.19, E6.20e), NDs",
}
ROY_SKIPPED = [
    ("E6.1–E6.4", "the game system NP (E6.1e is not valid) and conceptual questions"),
    ("E6.5, E6.8, E6.11, E6.17", "fill in justifications for given derivations"),
    ("E6.6, E6.9, E6.12", "explain why given derivations are illegitimate"),
    ("E6.14, E6.30", "name the strategy (no derivation asked)"),
    ("E6.22", "translation exercises"),
    ("E6.23", "essays"),
    ("E6.24 onward", "quantificational"),
    ("chapter 3", "axiomatic system AD, not natural deduction"),
]
ROY_CHARS = str.maketrans(
    {".": "(", "/": ")", "Œ": "[", "!": "→", "^": "∧", "_": "∨", "$": "↔", "`": "⊢", ";": ","}
)
ROY_CIDS = {"(cid:24)": "¬", "(cid:141)": "]"}
ROY_ITEM = re.compile(r"^(\*?)([a-z])\. ?(.*)$")
ROY_THM = re.compile(r"^(\*?)T6\.(\d+)\. ?(.*)$")
ROY_EX = re.compile(r"^\*?(E6\.\d+)\.")
ROY_STOP = re.compile(r"^(=== PAGE|CHAPTER 6\.|\d+$|6\.\d)")


def roy_text(pdf):
    """pdfminer text of the sentential chapter, cached next to the PDF."""
    cache = pdf + ".pm.txt"
    if not os.path.exists(cache):
        from pdfminer.high_level import extract_text
        from pdfminer.layout import LAParams

        # the default layout settings interleave the goals of neighbouring items on some pages
        lp = LAParams(char_margin=4.0, line_margin=0.2)
        with open(cache, "w") as f:
            for i in ROY_PAGES:
                f.write(f"\n=== PAGE {i + 1} ===\n")
                f.write(extract_text(pdf, page_numbers=[i], laparams=lp))
    with open(cache) as f:
        return f.read().split("\n")


def roy_decode(s):
    for k, v in ROY_CIDS.items():
        s = s.replace(k, v)
    s = re.sub(r"\bNDs\+?", " ", s)
    return s.translate(ROY_CHARS)


def roy_items(pdf):
    lines = roy_text(pdf)
    exercise, page, n_items = None, None, 0
    seen_theorems = set()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("=== PAGE"):
            page = int(line.split()[2])
        m_ex = ROY_EX.match(line)
        if m_ex:
            exercise, n_items = m_ex.group(1), 0
            i += 1
            continue
        m_it, m_th = ROY_ITEM.match(line), ROY_THM.match(line)
        label, body, star = None, None, ""
        if m_th and any(int(m_th.group(2)) in r for r in ROY_THEOREMS):
            n = int(m_th.group(2))
            if n in seen_theorems:  # later mentions come with their demonstrations
                i += 1
                continue
            seen_theorems.add(n)
            label = next(v for r, v in ROY_THEOREMS.items() if n in r)
            number, body, star = f"T6.{n}", m_th.group(3), m_th.group(1)
        elif m_it and exercise in ROY_EXERCISES:
            label = ROY_EXERCISES[exercise]
            number, body, star = f"{exercise}{m_it.group(2)}", m_it.group(3), m_it.group(1)
            n_items += 1
        if label is None:
            # prose after an exercise's items ends the exercise; the derivation displays that
            # follow use the same "a." line labels
            if exercise and n_items and line and not ROY_STOP.match(line):
                exercise = None
            i += 1
            continue
        # an item continues over following lines that hold only formula text (or 'NDs')
        raw = [body]
        j = i + 1
        while j < len(lines):
            nxt = lines[j].strip()
            if not nxt:
                j += 1
                continue
            if ROY_ITEM.match(nxt) or ROY_THM.match(nxt) or ROY_EX.match(nxt):
                break
            if ROY_STOP.match(nxt):
                j += 1
                continue
            dec = roy_decode(nxt)
            # formula text, possibly followed by a theorem's name in prose
            if re.fullmatch(FORMULA + r"+(\s[a-z].*)?", dec):
                raw.append(nxt)
                j += 1
                continue
            break
        text = roy_decode(" ".join(raw))
        notes = ["read from the PDF text (pdfminer); glyphs decoded: ∼ → ¬, ⊢NDs → ⊢"]
        # a theorem may carry a name after the formula ("principle of excluded middle")
        m_tail = re.search(r"\s+[a-z].*$", text)
        if m_tail:
            notes.append(f"trailing text dropped: {m_tail.group().strip()!r}")
            text = text[: m_tail.start()]
        if "(cid:" in text:
            text = f"UNDECODED {text}"
        if star:
            notes.append(
                "starred: the author's answer is published in Answers to Selected Exercises"
            )
        original = re.sub(r"\s+", " ", text).strip()
        yield f"PDF p. {page}, {label}", number, original, to_ascii(original), notes
        i = j


# ---------------------------------------------------------------- Carr

CARR_PAGES = {
    10: "4.1 Core",
    11: "4.1 Core",
    12: "4.2 Conjunction",
    13: "4.3 Implication",
    14: "4.4 Disjunction",
    15: "4.5 Biconditional",
    16: "4.6 Negation",
    17: "4.6 Negation",
    24: "4.10 Additional challenges, Contraposition",
}
CARR_WORKED = {4: "3.1 Worked example"}
CARR_SKIPPED = [
    ("4.1 items 2, 3, 6, 8–10, 13, 14; 4.7–4.9", "first-order or identity"),
    (
        "4.10 Admissible rules; Contraposition 3–11",
        "questions about the system, or proofs in the alternate system ⊢C",
    ),
    (
        "bonus challenges",
        "kept as plain sequents; the extra constraint (step limit, no ¬Elim) is dropped",
    ),
]
CARR_SUBSCRIPTS = {"P1": "U", "P2": "V", "P3": "W", "P4": "Z", "P5": "T", "Q1": "X", "Q2": "Y"}
CARR_ITEM = re.compile(r"^(\d+)\. (.*?)\s*\((\d+)\)$")
CARR_BONUS = re.compile(r"^[A-Z0-9¬∧∨→↔⊣()\s]*⊣[A-Z0-9¬∧∨→↔⊣()\s]*$")


def carr_text(pdf):
    from pypdf import PdfReader

    return [p.extract_text() or "" for p in PdfReader(pdf).pages]


def carr_items(pdf):
    pages = carr_text(pdf)
    for idx, label in {**CARR_PAGES, **CARR_WORKED}.items():
        sub = ""
        for line in pages[idx].split("\n"):
            line = line.strip()
            if re.match(r"^[A-Z][a-z]", line) and "⊣" not in line:
                sub = line if len(line) < 45 else sub
            m = CARR_ITEM.match(line)
            steps, number = None, None
            if m:
                number, body, steps = m.group(1), m.group(2), int(m.group(3))
            elif CARR_BONUS.match(line) and "⊣" in line and idx in (13, 17, 24):
                number, body = "bonus", line
            elif idx in CARR_WORKED and line.startswith("⊣"):
                number, body = "worked", line
            else:
                continue
            if re.search(r"[∀∃=a-z]", body):
                yield f"PDF p. {idx + 1}, {label}", number, body, None, ["first-order or identity"]
                continue
            if "⊣C" in body:
                yield (
                    f"PDF p. {idx + 1}, {label}",
                    number,
                    body,
                    None,
                    ["proof in the alternate system ⊢C"],
                )
                continue
            notes = ["read from the PDF text (pypdf); ⊣ is the pack's turnstile"]
            if steps is not None:
                notes.append(f"author's solution length: {steps} steps")
            if sub and idx not in CARR_WORKED:
                notes.append(f"under the heading {sub!r}")
            text = body
            for k, v in CARR_SUBSCRIPTS.items():
                if k in text:
                    text = text.replace(k, v)
                    notes.append(f"subscripted letter {k} read as {v}")
            text = text.replace("⊣", "⊢")
            yield f"PDF p. {idx + 1}, {label}", number, body, to_ascii(text), notes
