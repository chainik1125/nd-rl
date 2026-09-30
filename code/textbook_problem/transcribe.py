"""Transcribe textbook exercises into the textbook-72 JSONL format and audit them for overlap.

Sources and the exercise blocks read from each are listed in SOURCES and PARTS, chosen by hand
after reading every practice section (docs/textbook-problem/diversification/sources.md). Only
blocks that ask for a proof are read; the rest are listed in SKIPPED_PARTS with the reason.

Output rows match `code/ws3/textbook72/textbook_dev.jsonl` (canonical_hash, name, prompt, role),
plus the provenance fields from docs/textbook-problem/plan.md section 3. No proof search is run
here, so rows carry no `reference_lines` yet.

Overlap with the sealed sets (the textbook 72 and the test files) is reported as a count only:
matching exercises are dropped without being written anywhere.

Run from code/:
    uv run python textbook_problem/transcribe.py \
        --raw ../data/textbook_problem/raw --out ../data/textbook_problem/candidates
"""

import argparse
import collections
import hashlib
import html
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [
    HERE,
    os.path.join(HERE, "../ws3"),
    os.path.join(HERE, "../ws3/fork_pin"),
    os.path.join(HERE, "../experiments/current/r1_cot/vendor"),
]
import ndpath  # noqa: E402,F401
from data import canon_key, load_pool, load_textbook72, thm_of_prompt  # noqa: E402
from logic import atoms, classically_valid, intuitionistically_valid, to_tokens  # noqa: E402
from nd_data import canonical_theorem, theorem_hash  # noqa: E402
from pdf_sources import CARR_SKIPPED, ROY_SKIPPED, carr_items, roy_items  # noqa: E402
from tokenizer import SYMS  # noqa: E402

BOT = ("bot",)
ATOM_NAMES = "PQRS"
PROMPT_TOKEN_CAP = 1024 - 400  # model max_len minus the evaluator's max_new
TODAY = "2026-09-28"

SOURCES = {
    "forallx-calgary": {
        "book": "forall x: Calgary (Thomas-Bolduc & Zach, after Magnus and Button)",
        "repo": "https://github.com/rzach/forallx-yyc",
        "commit": "c5cb41544eb61302fa2d06b8a86c76ed3ba52a7c",
        "licence": "CC BY 4.0",
        "priority": 0,
    },
    "forallx-syracuse": {
        "book": "forall x: Syracuse (Rieppel, after Button's Cambridge edition)",
        "repo": "https://github.com/mrieppel/forallxsyr",
        "commit": "46841216b602e7e9d0fa4069593dfb47e1af0a91",
        "licence": "CC BY-SA 4.0",
        "priority": 1,
    },
    "forallx-msu": {
        "book": "forall x: Mississippi State (Johnson, after Calgary)",
        "repo": "https://github.com/loighic/forallx-msu",
        "commit": "1c4200da893a75bac85d37f97f904a9e2e058a00",
        "licence": "CC BY 4.0",
        "priority": 2,
    },
    "carnap-book": {
        "book": "The Carnap Book (Leach-Krouse & Ehrlich)",
        "repo": "https://carnap.io/book",
        "commit": None,
        "licence": "CC BY 4.0",
        "priority": 3,
    },
    # PDF sources (S1 pass 3). 'commit' is None; the PDF's sha256 is in the summary's raw_sha256.
    "roy": {
        "book": "Symbolic Logic: An Accessible Introduction to Serious Mathematical Logic (Roy), "
        "2023 print edition, third revision July 2025",
        "repo": "https://tonyroyphilosophy.net/wp-content/uploads/2023/06/SymLogFullTxt.pdf",
        "commit": None,
        "licence": "CC BY-NC-SA 4.0",
        "priority": 4,
        "accessed": "2026-09-29",
    },
    "carr": {
        "book": "The Natural Deduction Pack (Carr), March 2018, for Halbach's The Logic Manual",
        "repo": "https://logicmanual.philosophy.ox.ac.uk/carr/NDpack.pdf",
        "commit": None,
        "licence": "none stated (author to be asked)",
        "priority": 5,
        "accessed": "2026-09-29",
    },
}

CAL, SYR = "forallx-yyc-prooftfl.tex", "forallx-04-tfl-prf.tex"
MSU_A, MSU_B = "forallx-msu-part4a--proof_tfl.tex", "forallx-msu-part4b-sound_complete.tex"
PTC = "ch. Proof-theoretic concepts"

# (source, file, first line, last line, kind, label). kind: 'items' reads every \item as a
# sequent, theorem or interderivable pair; 'valid_or_not' does the same but keeps only valid
# ones (the exercise asks to prove the valid ones); 'either' reads "Show that ..." sentences.
PARTS = [
    ("forallx-calgary", CAL, 785, 803, "items", "ch. Basic rules for TFL"),
    ("forallx-calgary", CAL, 1219, 1231, "items", "ch. Constructing proofs"),
    ("forallx-calgary", CAL, 1235, 1245, "items", "ch. Constructing proofs"),
    ("forallx-calgary", CAL, 1246, 1258, "items", "ch. Constructing proofs"),
    ("forallx-calgary", CAL, 1463, 1473, "items", "ch. Additional rules"),
    ("forallx-calgary", CAL, 1570, 1578, "items", PTC),
    ("forallx-calgary", CAL, 1579, 1587, "items", PTC),
    ("forallx-calgary", CAL, 1588, 1598, "pairs", PTC),
    ("forallx-calgary", CAL, 1990, 2006, "either", "ch. Soundness and completeness"),
    ("forallx-calgary", CAL, 2007, 2022, "either", "ch. Soundness and completeness"),
    ("forallx-syracuse", SYR, 351, 361, "items", "sec. Additional assumptions"),
    ("forallx-syracuse", SYR, 362, 372, "items", "sec. Additional assumptions"),
    ("forallx-syracuse", SYR, 448, 459, "items", "sec. Theorems and reiterating"),
    ("forallx-syracuse", SYR, 521, 536, "items", "sec. Biconditional rules"),
    ("forallx-syracuse", SYR, 648, 670, "items", "sec. Negation rules"),
    ("forallx-syracuse", SYR, 799, 819, "items", "sec. Disjunction rules"),
    ("forallx-syracuse", SYR, 910, 926, "items", "sec. Proof strategies"),
    ("forallx-syracuse", SYR, 931, 942, "items", "sec. Proof strategies"),
    ("forallx-syracuse", SYR, 944, 953, "items", "sec. Proof strategies"),
    ("forallx-syracuse", SYR, 1130, 1144, "items", "sec. Derived rules"),
    ("forallx-msu", MSU_A, 1235, 1263, "items", "ch. Rules of derivation"),
    ("forallx-msu", MSU_A, 1264, 1278, "items", "ch. Rules of derivation"),
    ("forallx-msu", MSU_A, 1800, 1822, "items", "ch. Theorems"),
    ("forallx-msu", MSU_A, 1973, 1992, "items", PTC),
    ("forallx-msu", MSU_B, 111, 125, "valid_or_not", "ch. Soundness"),
]

SKIPPED_PARTS = [
    ("forallx-calgary", "718-742", "explain the mistakes in two incorrect proofs"),
    ("forallx-calgary", "743-783", "add citations to given proofs"),
    ("forallx-calgary", "1232-1234", "formulate strategies (no formula)"),
    ("forallx-calgary", "1414-1462", "add citations to given proofs"),
    ("forallx-calgary", "1599-1606", "conceptual questions"),
    ("forallx-calgary", "1782-1813", "proof schemes with metavariables, or conceptual"),
    ("forallx-syracuse", "672-708", "proofs in a modified rule system (DNE, or EX+LEM)"),
    ("forallx-syracuse", "869-908", "add citations to given proofs"),
    ("forallx-syracuse", "1082-1128", "add citations to given proofs"),
    ("forallx-syracuse", "1147-1148", "proof templates with metavariables"),
    ("forallx-msu", "part4a 1288-1799, 1823-1972, 1993-", "answer sections"),
    ("carnap-book", "ch. 9-12", "symbolization, truth tables, first-order logic"),
    *[("roy", a, b) for a, b in ROY_SKIPPED],
    *[("carr", a, b) for a, b in CARR_SKIPPED],
]

# ---------------------------------------------------------------- formula parsing

LATEX_TOKENS = [
    (r"\\leftproves\s*\\proves|\\pequiv|\\dashv\s*\\vdash|\\leftrightarrows", "PAIR"),
    (r"\\proves|\\vdash|\\therefore|:\|-:", "TURN"),
    (r"\\eiff|\\leftrightarrow|\\liff|<->", "IFF"),
    (r"\\eif|\\rightarrow|\\to\b|\\supset|->", "IMP"),
    (r"\\eand|\\land|\\wedge|/\\", "AND"),
    (r"\\eor|\\lor|\\vee|\\/", "OR"),
    (r"\\enot|\\lnot|\\neg|~|-", "NOT"),
    (r"\\ered|\\bot", "BOT"),
    (r"\(|\[|\\\{|\{", "LP"),
    (r"\)|\]|\\\}|\}", "RP"),
    (r",", "COMMA"),
    (r"[A-Z]", "ATOM"),
]
TOKEN_RE = re.compile("|".join(f"(?P<{n}{i}>{p})" for i, (p, n) in enumerate(LATEX_TOKENS)))
NOISE_RE = re.compile(r"\\(smallskip|medskip|bigskip|quad|qquad|,|;|!|\s)|\\ |\$")


class ParseError(ValueError):
    pass


def lex(s, latex=True):
    out, i = [], 0
    if latex:
        s = NOISE_RE.sub(" ", s)
    while i < len(s):
        if s[i].isspace():
            i += 1
            continue
        m = TOKEN_RE.match(s, i)
        if not m:
            raise ParseError(f"unexpected text at {s[i : i + 20]!r}")
        kind = re.sub(r"\d+$", "", m.lastgroup)
        out.append((kind, m.group()))
        i = m.end()
    return out


class Parser:
    """Connectives: ~ binds tightest; ∧/∨ bind tighter than →/↔ (Carnap's JI rule); a chain of the
    same junction groups to the left; mixed junctions or chained →/↔ without brackets are an
    error. `notes` records every convention actually used."""

    def __init__(self, toks):
        self.t, self.i, self.notes = toks, 0, set()

    def peek(self):
        return self.t[self.i][0] if self.i < len(self.t) else None

    def eat(self, kind):
        if self.peek() != kind:
            raise ParseError(f"expected {kind}, got {self.peek()}")
        self.i += 1
        return self.t[self.i - 1]

    def formula(self):
        left = self.junction()
        if self.peek() in ("IMP", "IFF"):
            op = self.eat(self.peek())[0]
            right = self.junction()
            if self.peek() in ("IMP", "IFF"):
                raise ParseError("unbracketed chain of →/↔")
            if self._bare(left, "J") or self._bare(right, "J"):
                self.notes.add("read ∧/∨ as binding tighter than →/↔")
            return ("IFF" if op == "IFF" else "imp", left, right)
        return left

    def _bare(self, f, _):
        return isinstance(f, tuple) and len(f) == 4  # junction built without brackets

    def junction(self):
        left = self.unary()
        op = None
        while self.peek() in ("AND", "OR"):
            k = self.eat(self.peek())[0]
            if op and k != op:
                raise ParseError("unbracketed mix of ∧ and ∨")
            if op:
                self.notes.add("grouped a chain of the same junction to the left")
            op = k
            right = self.unary()
            left = ({"AND": "and", "OR": "or"}[k], _strip(left), right, "bare")
        return left

    def unary(self):
        if self.peek() == "NOT":
            self.eat("NOT")
            return ("not", _strip(self.unary()))
        if self.peek() == "ATOM":
            return ("atom", self.eat("ATOM")[1])
        if self.peek() == "BOT":
            self.eat("BOT")
            return BOT
        if self.peek() == "LP":
            self.eat("LP")
            f = self.formula()
            self.eat("RP")
            return _strip(f)
        raise ParseError(f"unexpected {self.peek()}")


def _strip(f):
    """Drop the 'bare' marker and expand ↔ (encoding rule: A ↔ B as (A → B) ∧ (B → A))."""
    if f[0] == "atom" or f == BOT:
        return f
    if f[0] == "not":
        return ("not", _strip(f[1]))
    a, b = _strip(f[1]), _strip(f[2])
    if f[0] == "IFF":
        return ("and", ("imp", a, b), ("imp", b, a))
    return (f[0], a, b)


def parse_list(p):
    fs = [_strip(p.formula())]
    while p.peek() == "COMMA":
        p.eat("COMMA")
        fs.append(_strip(p.formula()))
    return fs


def parse_item(text, latex=True):
    """Returns (kind, [(premises, goal)], notes). kind: sequent | theorem | pair."""
    toks = lex(text, latex)
    kinds = [k for k, _ in toks]
    p = Parser(toks)
    if "PAIR" in kinds:
        a = _strip(p.formula())
        p.eat("PAIR")
        b = _strip(p.formula())
        out = ("pair", [([a], b), ([b], a)])
    elif "TURN" in kinds:
        prem = [] if p.peek() == "TURN" else parse_list(p)
        p.eat("TURN")
        out = ("sequent" if prem else "theorem", [(prem, _strip(p.formula()))])
    else:
        out = ("theorem", [([], _strip(p.formula()))])
    if p.i != len(toks):
        raise ParseError(f"trailing tokens: {toks[p.i :]}")
    return out[0], out[1], sorted(p.notes)


# ---------------------------------------------------------------- extraction


def clean_item(line):
    line = re.sub(r"(?<!\\)%.*", "", line)
    line = re.sub(r"\\item(\[[^]]*\])?", "", line, count=1)
    line = re.split(r"\\\\|\\hfill|\\textit|\\emph", line)[0]
    return line.strip().rstrip(".").strip()


def tex_items(path, lo, hi):
    with open(path, encoding="utf-8") as f:
        lines = f.read().split("\n")
    n = 0
    for no in range(lo, hi + 1):
        raw = lines[no - 1]
        if re.match(r"\s*%", raw) or r"\item" not in raw:
            continue
        n += 1
        yield no, n, raw.strip(), clean_item(raw)


EITHER = [
    (r"Show that (?:the sentence )?(.+?) is a theorem$", "theorem"),
    (r"Show that (.+?) is valid$", "sequent"),
    (r"Show that (?:the sentence )?(.+?) is equivalent to (?:the sentence )?(.+)$", "pair"),
    (r"Show that the sentences (.+?) are jointly inconsistent$", "inconsistent"),
    (r"Show that (?:the sentence )?(.+?) is a contradiction$", "contradiction"),
]


def parse_either(text):
    t = re.sub(r"\s+", " ", text.replace("$", "")).strip()
    if re.search(r"\bnot\b|\binvalid\b|contingent|jointly consistent", t):
        raise ParseError("not a proof task (asks to show a negative or contingency)")
    for pat, kind in EITHER:
        m = re.match(pat, t)
        if not m:
            continue
        if kind == "pair":
            k, seqs, notes = parse_item(f"{m.group(1)} \\pequiv {m.group(2)}")
            return k, seqs, notes + ["equivalence as two sequents"]
        if kind in ("inconsistent", "contradiction"):
            p = Parser(lex(m.group(1)))
            prem = parse_list(p)
            if p.i != len(p.t):
                raise ParseError("trailing tokens")
            return "sequent", [(prem, BOT)], [f"{kind} shown by deriving ⊥"]
        return parse_item(m.group(1) if kind == "theorem" else m.group(1))
    raise ParseError("unrecognised 'Show that' sentence")


def carnap_items(path, chapter):
    with open(path, encoding="utf-8") as f:
        text = f.read()
    # attribute values contain ">" (e.g. "P->Q"), so match quoted attributes, not [^>]*
    for tag in re.findall(r'<div((?:\s+[\w-]+="[^"]*")*)\s*>', text):
        attrs = dict(re.findall(r'data-carnap-([a-z]+)="([^"]*)"', tag))
        if attrs.get("type") != "proofchecker" or attrs.get("system") != "prop":
            continue
        label = attrs.get("submission", "").replace("saveAs:", "")
        yield label, html.unescape(attrs.get("goal", "")), chapter


# ---------------------------------------------------------------- encoding and records


def rename(prem, goal):
    order = []
    for f in prem + [goal]:
        for a in _atoms_in_order(f):
            if a not in order:
                order.append(a)
    if len(order) > 4:
        raise ParseError(f"{len(order)} atoms (the tokenizer has 4)")
    m = {a: ATOM_NAMES[i] for i, a in enumerate(order)}

    def sub(f):
        if f == BOT:
            return f
        if f[0] == "atom":
            return ("atom", m[f[1]])
        if f[0] == "not":
            return ("not", sub(f[1]))
        return (f[0], sub(f[1]), sub(f[2]))

    return [sub(p) for p in prem], sub(goal), {a: m[a] for a in order}


def _atoms_in_order(f):
    if f == BOT:
        return []
    if f[0] == "atom":
        return [f[1]]
    return [a for g in f[1:] for a in _atoms_in_order(g)]


def has_iff_source(text):
    return bool(re.search(r"\\eiff|\\leftrightarrow|<->|\\liff|↔", text))


def make_prompt(prem, goal):
    head = "THM " + " , ".join(to_tokens(p) for p in prem) + (" " if prem else "")
    return head + "SEQ " + to_tokens(goal) + " PRF"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    candidates, exclusions = [], []

    def add(src, where, number, original, kind, seqs, notes, keep_invalid_as, direction=None):
        for d, (prem, goal) in enumerate(seqs):
            rec_notes = list(notes)
            try:
                p2, g2, amap = rename(prem, goal)
            except ParseError as e:
                exclusions.append(
                    dict(source=src, where=where, number=number, original=original, reason=str(e))
                )
                continue
            if any(a != b for a, b in amap.items()):
                rec_notes.append(
                    "sentence letters renamed: " + ", ".join(f"{a}→{b}" for a, b in amap.items())
                )
            if has_iff_source(original):
                rec_notes.append("↔ written as (A → B) ∧ (B → A)")
            if not classically_valid(p2, g2):
                exclusions.append(
                    dict(
                        source=src,
                        where=where,
                        number=number,
                        original=original,
                        reason=keep_invalid_as,
                    )
                )
                continue
            prompt = make_prompt(p2, g2)
            ntok = len(prompt.split())
            assert all(t in SYMS for t in prompt.split()), prompt
            h = theorem_hash(prompt)
            part = None
            if kind == "pair":
                part = "left to right" if d == 0 else "right to left"
                rec_notes.append(f"interderivable pair, {part}")
            candidates.append(
                {
                    "canonical_hash": h,
                    "name": "textbook_" + h[:20],
                    "prompt": prompt,
                    "role": "textbook_candidate",
                    "source": {
                        **{k: SOURCES[src][k] for k in ("book", "repo", "commit", "licence")},
                        "id": src,
                        "where": where,
                        "number": number,
                        "direction": part,
                        "accessed": SOURCES[src].get("accessed", TODAY),
                    },
                    "original": original,
                    "encoding_notes": rec_notes,
                    "form": "theorem" if not p2 else "sequent",
                    "n_premises": len(p2),
                    "n_atoms": len({a for f in p2 + [g2] for a in atoms(f)}),
                    "classical_only": not intuitionistically_valid(p2, g2),
                    "n_prompt_tokens": ntok,
                    "within_token_cap": ntok <= PROMPT_TOKEN_CAP,
                }
            )

    for src, fn, lo, hi, kind, label in PARTS:
        path = os.path.join(args.raw, src, fn)
        for no, n, raw, text in tex_items(path, lo, hi):
            where = f"{fn}:{no} ({label}, block at line {lo}, item {n})"
            try:
                if kind == "either":
                    k, seqs, notes = parse_either(text)
                elif kind == "pairs":
                    # Calgary's interderivable pairs are printed "A, B"
                    p = Parser(lex(text))
                    fs = parse_list(p)
                    if len(fs) != 2 or p.i != len(p.t):
                        raise ParseError("expected a pair of sentences")
                    k, seqs, notes = "pair", [([fs[0]], fs[1]), ([fs[1]], fs[0])], sorted(p.notes)
                else:
                    k, seqs, notes = parse_item(text)
            except ParseError as e:
                exclusions.append(
                    dict(
                        source=src,
                        where=where,
                        number=n,
                        original=raw,
                        reason=f"not transcribed: {e}",
                    )
                )
                continue
            invalid_reason = (
                "invalid (the exercise asks to prove only the valid ones)"
                if kind == "valid_or_not"
                else "invalid: check the transcription"
            )
            add(
                src,
                where,
                n,
                raw,
                k,
                seqs,
                notes + (["exercise: prove if valid"] if kind == "valid_or_not" else []),
                invalid_reason,
            )

    for ch in (3, 4, 5, 6, 7, 8):
        path = os.path.join(args.raw, "carnap-book", f"book-{ch}.html")
        for label, goal, _ in carnap_items(path, ch):
            where = f"carnap.io/book/{ch}, exercise {label}"
            if goal.startswith("["):
                exclusions.append(
                    dict(
                        source="carnap-book",
                        where=where,
                        number=label,
                        original="(goal stored encoded in the page source)",
                        reason="not transcribed: goal is hidden; read it from the "
                        "rendered page by hand",
                    )
                )
                continue
            try:
                k, seqs, notes = parse_item(goal, latex=False)
            except ParseError as e:
                exclusions.append(
                    dict(
                        source="carnap-book",
                        where=where,
                        number=label,
                        original=goal,
                        reason=f"not transcribed: {e}",
                    )
                )
                continue
            add(
                "carnap-book",
                where,
                label,
                goal.strip(),
                k,
                seqs,
                notes,
                "invalid: check the transcription",
            )

    for src, fn, reader in (
        ("roy", "roy/SymLogFullTxt.pdf", roy_items),
        ("carr", "carr/NDpack.pdf", carr_items),
    ):
        for where, number, original, text, notes0 in reader(os.path.join(args.raw, fn)):
            if text is None:  # the reader marks items outside scope with a reason
                exclusions.append(
                    dict(
                        source=src, where=where, number=number, original=original, reason=notes0[0]
                    )
                )
                continue
            try:
                k, seqs, notes = parse_item(text, latex=False)
            except ParseError as e:
                exclusions.append(
                    dict(
                        source=src,
                        where=where,
                        number=number,
                        original=original,
                        reason=f"not transcribed: {e}",
                    )
                )
                continue
            add(
                src,
                where,
                number,
                original,
                k,
                seqs,
                notes0 + notes,
                "invalid: check the transcription",
            )

    # ------------------------------------------------------------ audit
    def keysets(rows):
        return (
            {canonical_theorem(r["prompt"]) for r in rows},
            {canon_key(thm_of_prompt(r["prompt"])) for r in rows},
            len(rows),
        )

    sealed = {
        "textbook72": keysets(load_textbook72()),
        "test_files": keysets(load_pool("test_short_prompts") + load_pool("test_long_prompts")),
    }
    named = {
        n: keysets(load_pool(n))
        for n in ("validation_36", "heldout", "transfer", "rl_targets", "train")
    }

    # 1) duplicates within the set, keeping the first by source priority
    candidates.sort(key=lambda r: SOURCES[r["source"]["id"]]["priority"])
    unique, seen = [], {}
    audit = collections.Counter()
    for r in candidates:
        ct = canonical_theorem(r["prompt"])
        if ct in seen:
            seen[ct]["also_in"].append(r["source"]["where"])
            audit["duplicate within the set (kept the first by source priority)"] += 1
            continue
        r["also_in"] = []
        seen[ct] = r
        unique.append(r)

    # 2) overlap audit on distinct classes. Sealed hits are counted by set and source only.
    kept = []
    by_source = collections.defaultdict(collections.Counter)
    for r in unique:
        src = r["source"]["id"]
        ct, ck = canonical_theorem(r["prompt"]), canon_key(thm_of_prompt(r["prompt"]))
        s_hits = [n for n, (cts, cks, _) in sealed.items() if ct in cts or ck in cks]
        if s_hits:
            for n in s_hits:
                audit[f"sealed: {n} (dropped, not recorded)"] += 1
            by_source[src]["sealed"] += 1
            continue
        hits = [n for n, (cts, cks, _) in named.items() if ct in cts or ck in cks]
        if hits:
            for n in hits:
                audit[f"overlap with {n}"] += 1
            by_source[src]["pool overlap"] += 1
            exclusions.append(
                dict(
                    source=src,
                    where=r["source"]["where"],
                    number=r["source"]["number"],
                    original=r["original"],
                    reason="renaming of an item in " + ", ".join(hits),
                )
            )
            continue
        by_source[src]["kept"] += 1
        kept.append(r)

    with open(os.path.join(args.out, "candidates.jsonl"), "w") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(os.path.join(args.out, "exclusions.jsonl"), "w") as f:
        for r in exclusions:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    by_src = collections.Counter(r["source"]["id"] for r in kept)
    ex_reason = collections.Counter(
        re.sub(r": .*", "", e["reason"])
        if e["reason"].startswith("not transcribed")
        else e["reason"]
        for e in exclusions
    )
    summary = {
        "date": TODAY,
        "n_candidates_written": len(kept),
        "n_transcribed_before_audit": len(candidates),
        "candidates_by_source": dict(by_src),
        "candidates_by_form": dict(collections.Counter(r["form"] for r in kept)),
        "candidates_by_atoms": dict(
            sorted(collections.Counter(r["n_atoms"] for r in kept).items())
        ),
        "classical_only": sum(r["classical_only"] for r in kept),
        "over_prompt_token_cap": sum(not r["within_token_cap"] for r in kept),
        "transcribed_by_source": dict(collections.Counter(r["source"]["id"] for r in candidates)),
        "audit": dict(audit),
        "audit_by_source": {s: dict(c) for s, c in by_source.items()},
        "audit_sets": {
            "sealed": {n: v[2] for n, v in sealed.items()},
            **{n: v[2] for n, v in named.items()},
        },
        "audit_method": "renaming class by canonical_theorem (renaming + premise order) or "
        "canon_key (renaming by first appearance); either match counts",
        "exclusions_by_reason": dict(ex_reason),
        "skipped_parts": [dict(source=s, lines=ln, reason=w) for s, ln, w in SKIPPED_PARTS],
        "raw_sha256": {
            os.path.relpath(os.path.join(d, fn), args.raw): _sha(os.path.join(d, fn))
            for d, _, fns in os.walk(args.raw)
            for fn in sorted(fns)
        },
    }
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1, ensure_ascii=False)
    print(json.dumps({k: summary[k] for k in list(summary)[:10]}, indent=1, ensure_ascii=False))
    print(
        "exclusions_by_reason:",
        json.dumps(summary["exclusions_by_reason"], ensure_ascii=False, indent=1),
    )


def _sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


if __name__ == "__main__":
    main()
