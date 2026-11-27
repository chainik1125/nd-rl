"""The detector of the 24 linked habits (linked habits version 2, SPEC2.md "Detector"; CPU only).

API (the runner imports these; names, signatures and return shapes are fixed):

    HABITS                      ordered habit names: the 9 natural habits, then the 15 note habits
                                ("route", "order", "reit", "conj", "split", "exfalso", "disjunct", "halves", "restate",
                                 "note_atom", "note_~a", ..., "note_>cc")  (notes.CLASSES order)
    load_tables(lh=None)        -> {"habit": the habit table (LH/data/habit_table.json), "note": the note table
                                    (LH/data/note_table.json), "linked": [habits in the linked set],
                                    "a_form": {habit: 1 | 2 (natural) or a note form name}}
    profile(prompt, proof, tables=None) -> dict, never raises:
        ok          bool: the proof was read (False: `unreadable` True and `err` says why; every count is 0)
        unreadable  bool
        err         str | None
        lines       lines of the written proof;  dead: dead lines dropped before the natural habits were read
        habits      {habit: {"sites": n, "a": n, "b": n, "f1": n, "f2": n}} for every name in HABITS
                    sites = sites with room; a / b = sites in author A's / B's form (by the habit table); f1 / f2 the
                    same by form number (natural habits; for notes f1 = a, f2 = b). For a note habit (premise class
                    c): sites = premises of class c, a / b = those noted in A's / B's form, plus "other" (a menu
                    form that is neither) and "none" (no note).
        seq         [[position, habit, "A" | "B"], ...] every site with a form, in proof order (position = the N
                    number of the site's line in the written proof), for first-site / later-site readings
        notes       notes.note_reading of the proof (the note habits' raw reading)
        near        near-miss counters (audit F9; added 2026-11-27, all 0 when unreadable):
                      ande_apart         an unused ANDE whose other half is the conjunction's only used ANDE in
                                         that scope, not in the builder's place (right after it, or right before it
                                         when it is its scope's last line) or citing another line; such a site is
                                         also counted under `halves` as "other" (in `sites`, in neither form)
                      dead_r_in_box      an R line inside a box that nothing cites (pruned before reading reit)
                      cited_header_copy  a premise whose menu-form line sits in the header run (the top-level
                                         menu-form lines right after the PR block, cited or not) but is cited, so it
                                         is not a note, while the premise has no uncited line in the run and some
                                         other premise has one (so habit 9's copies after the notes, or in a proof
                                         without notes, are not counted)
                      double_note        a premise with two or more notes (each is counted under its habit)
                      late_note          a menu-form dead line outside the header (notes.note_reading's `late`)
        Every natural habit also has "other" = sites - f1 - f2 (an order that is neither form, a halves near miss).

Reading order (SPEC2): the header notes are read first (notes.note_reading) and their lines removed from the tree,
dead lines are dropped (habits.prune; habit 8's unused half is read before that), habit 9's copies are collapsed,
then the other natural habits are read on the rest (lh_habits.read_tree). Habit definitions: lh_habits.py and
CORPUS2.md. Habits 5 (split) and 6 (exfalso) are read but not linked (load_tables()["linked"]); split's form-2 sites
cannot be read, so its `b` / `f2` are always 0.
"""

# ruff: noqa: E501
from __future__ import annotations

import collections
import contextlib
import json
import os
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import lh_habits as L  # noqa: E402  (puts W2, the tag experiment and Dan's tree on sys.path)
import notes as N  # noqa: E402

H = L.H

HABITS = L.HABITS
NATURAL = L.NATURAL
NOTE_HABITS = L.NOTE_HABITS
LH = Path(os.environ.get("LH", "/scratch/cowifornia/nd-rl/linked_habits"))
HABIT_TABLE = LH / "data" / "habit_table.json"
_CACHE: dict[str, Any] = {}


def load_tables(lh: Path | str | None = None) -> dict[str, Any]:
    """{"habit": habit_table.json, "note": note_table.json's table, "linked": [...], "a_form": {habit: 1 | 2 | form}}
    from LH/data (cached per directory)."""
    d = Path(lh) / "data" if lh is not None else HABIT_TABLE.parent
    key = str(d)
    if key not in _CACHE:
        ht = json.loads((d / "habit_table.json").read_text())
        nt = N.load_table(d / "note_table.json")
        a_form = dict(ht["a_form"])
        for c in N.CLASSES:
            a_form[f"note_{c}"] = nt[c]["A"]
        _CACHE[key] = {"habit": ht, "note": nt, "linked": list(ht["linked"]), "a_form": a_form}
    return _CACHE[key]


def provisional_tables(a_form: dict[str, int] | None = None) -> dict[str, Any]:
    """Tables for use before habit_table.json exists (the builder's balance step): the note table, and form 1 = A
    for every natural habit unless `a_form` says otherwise."""
    nt = N.load_table()
    af = {h: 1 for h in NATURAL}
    af.update(a_form or {})
    for c in N.CLASSES:
        af[f"note_{c}"] = nt[c]["A"]
    return {"habit": None, "note": nt, "linked": list(HABITS), "a_form": af}


def _blank(err: str | None, lines: int) -> dict[str, Any]:
    hab: dict[str, dict[str, int]] = {
        h: {"sites": 0, "a": 0, "b": 0, "f1": 0, "f2": 0} for h in NATURAL
    }
    for h in NOTE_HABITS:
        hab[h] = {"sites": 0, "a": 0, "b": 0, "f1": 0, "f2": 0, "other": 0, "none": 0}
    for h in NATURAL:
        hab[h]["other"] = 0
    return {"ok": err is None, "unreadable": err is not None, "err": err, "lines": lines, "dead": 0, "habits": hab,
            "seq": [], "notes": None, "near": dict.fromkeys(NEAR, 0)}  # fmt: skip


NEAR = ("ande_apart", "dead_r_in_box", "cited_header_copy", "double_note", "late_note")


def _header_near(proof: str, rd: dict[str, Any]) -> dict[str, int]:
    """cited_header_copy, double_note and late_note (module docstring) from the note reading and the text."""
    lines, _tail = N.parse_lines(proof)
    k = 0
    while k < len(lines) and lines[k].rule == "PR":
        k += 1
    prem = {ln.n: ln.f for ln in lines[:k]}
    cited = {r for ln in lines for r in ln.refs}
    noted = collections.Counter(x["premise_index"] for x in rd["notes"])
    run = []
    j = k
    while j < len(lines) and lines[j].depth == 0 and N._menu_form(lines[j], prem) is not None:
        run.append(lines[j])
        j += 1
    free = {
        ln.refs[0] for ln in run if ln.n not in cited
    }  # premises with an uncited (note-like) line in the run
    lost = {ln.refs[0] for ln in run if ln.n in cited and ln.refs[0] not in free} if free else set()
    return {
        "cited_header_copy": len(lost),
        "double_note": sum(v >= 2 for v in noted.values()),
        "late_note": rd["late"],
    }


def profile(prompt: str, proof: str, tables: dict[str, Any] | None = None) -> dict[str, Any]:
    """The 24-habit reading of one proof (module docstring). Never raises."""
    lines = proof.count(" ;") if isinstance(proof, str) else 0
    try:
        tb = (
            tables
            if tables is not None
            else (load_tables() if HABIT_TABLE.exists() else provisional_tables())
        )
        rd = N.note_reading(prompt, proof, tb["note"])
        if not rd["ok"]:
            return _blank(f"notes: {rd['err']}", lines)
        T = H.build(proof)
        note_ids = {x["line"] for x in rd["notes"]}
        nat = L.read_tree(T, note_ids, proof)
    except Exception as e:  # noqa: BLE001  (a model-written proof the tree or a detector cannot read)
        return _blank(f"{type(e).__name__}: {e}", lines)
    out = _blank(None, lines)
    out["dead"] = nat["dead"]
    out["notes"] = rd
    out["route_flags"] = nat["route_flags"]
    seq = []
    af = tb["a_form"]
    for h in NATURAL:
        x = nat[h]
        a = af.get(h, 1)
        out["habits"][h] = {"sites": x["sites"], "f1": x["f1"], "f2": x["f2"],
                            "a": x["f1"] if a == 1 else x["f2"], "b": x["f2"] if a == 1 else x["f1"],
                            "other": x["sites"] - x["f1"] - x["f2"]}  # fmt: skip
        for pos, f in x["pos"]:
            if f in (1, 2):
                seq.append([pos, h, "A" if f == a else "B"])
    prem = N.premises(proof)
    for f in prem:
        try:
            c = N.premise_class(f)
        except ValueError:
            continue
        out["habits"][f"note_{c}"]["sites"] += 1
    noted = set()
    for x in rd["notes"]:
        h = f"note_{x['cls']}"
        noted.add(x["premise_index"])
        k = {"A": "a", "B": "b"}.get(x["side"], "other")
        out["habits"][h][k] += 1
        if x["side"] in ("A", "B"):
            seq.append([x["line"], h, x["side"]])
    for i, f in enumerate(prem):
        if i not in noted:
            with contextlib.suppress(ValueError):
                out["habits"][f"note_{N.premise_class(f)}"]["none"] += 1
    for h in NOTE_HABITS:
        d = out["habits"][h]
        d["f1"], d["f2"] = d["a"], d["b"]
    out["seq"] = sorted(seq)
    near = dict.fromkeys(NEAR, 0)
    near.update(nat.get("near", {}))
    with contextlib.suppress(Exception):  # the counters never stop a reading
        near.update(_header_near(proof, rd))
    out["near"] = near
    return out


def reading(p: dict[str, Any], habits: tuple[str, ...] = HABITS) -> dict[str, tuple[int, int, int]]:
    """{habit: (sites, f1, f2)}: the part of a profile the round trip and the twin compare."""
    return {
        h: (p["habits"][h]["sites"], p["habits"][h]["f1"], p["habits"][h]["f2"]) for h in habits
    }
