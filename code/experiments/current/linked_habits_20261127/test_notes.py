"""Tests for notes.py (premise notes, linked_habits_20261127).  PY -m pytest -q test_notes.py"""

# ruff: noqa: E501
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
for _p in (
    str(HERE.parent / "author_tags_20261110"),
    str(HERE.parent / "two_authors_20261111"),
    str(HERE),
):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import notes as N  # noqa: E402

# A real corpus record (authors_half, record 2): two premises, a box.
PROMPT = "THM ( ~ P ) , ( ( R & R ) & ( Q v Q ) ) SEQ ( ( ~ ( ~ P ) ) > ( ( Q v Q ) & ( ( ~ P ) & ( ( R & R ) & ( Q v Q ) ) ) ) ) PRF"
PROOF = (
    "N1 ( ~ P ) : PR ; N2 ( ( R & R ) & ( Q v Q ) ) : PR ; N3 | ( ~ ( ~ P ) ) : AS ; N4 | ( Q v Q ) : ANDE2 N2 ; "
    "N5 | ( ( ~ P ) & ( ( R & R ) & ( Q v Q ) ) ) : ANDI N1 N2 ; N6 | ( ( Q v Q ) & ( ( ~ P ) & ( ( R & R ) & ( Q v Q ) ) ) ) "
    ": ANDI N4 N5 ; N7 ( ( ~ ( ~ P ) ) > ( ( Q v Q ) & ( ( ~ P ) & ( ( R & R ) & ( Q v Q ) ) ) ) ) : IMPI N3 N6 ; QED"
)
# Three premises, two of one class (vaa), an ORE with two boxes.
PROMPT3 = "THM ( P v Q ) , ( Q v P ) , R SEQ ( Q v P ) PRF"
PROOF3 = (
    "N1 ( P v Q ) : PR ; N2 ( Q v P ) : PR ; N3 R : PR ; N4 | P : AS ; N5 | ( Q v P ) : ORI2 N4 ; N6 | Q : AS ; "
    "N7 | ( Q v P ) : ORI1 N6 ; N8 ( Q v P ) : ORE N1 N4 N5 N6 N7 ; QED"
)
TABLE = {c: {"A": "copy", "B": "and2"} for c in N.CLASSES}
TABLE.update(
    {
        "vaa": {"A": "orN", "B": "Nor"},
        "atom": {"A": "For", "B": "orF"},
        "~a": {"A": "or2", "B": "orF"},
    }
)


def dan():
    import tag_core as T

    T.install_dan()
    from nd_verify import verify_text
    from state_env import decompose

    return verify_text, decompose


# ---------------------------------------------------------------------------- classes and menu
@pytest.mark.parametrize(
    "f,c",
    [
        ("P", "atom"), ("F", "atom"), ("( ~ Q )", "~a"), ("( ~ ( P & Q ) )", "~c"), ("( ~ ( ~ P ) )", "~c"),
        ("( P & Q )", "&aa"), ("( P & ( ~ Q ) )", "&ac"), ("( ( P v Q ) & R )", "&ca"), ("( ( ~ P ) & ( ~ Q ) )", "&cc"),
        ("( P v Q )", "vaa"), ("( P v ( Q > R ) )", "vac"), ("( ( P > Q ) v R )", "vca"), ("( ( P > Q ) v ( R & S ) )", "vcc"),
        ("( P > F )", ">aa"), ("( P > ( Q > R ) )", ">ac"), ("( ( P & Q ) > R )", ">ca"), ("( ( ~ P ) > ( ~ Q ) )", ">cc"),
    ],
)  # fmt: skip
def test_premise_class(f, c):
    assert N.premise_class(f) == c
    assert N.premise_class(f.split()) == c


@pytest.mark.parametrize(
    "f", ["", "( P & Q", "( P Q )", "( P & Q ) & R", "( ~ P Q )", "X", "( P & )"]
)
def test_premise_class_rejects(f):
    with pytest.raises((ValueError, IndexError)):
        N.premise_class(f)


def test_fifteen_classes_and_seven_forms():
    assert len(N.CLASSES) == 15 and len(set(N.CLASSES)) == 15
    assert list(N.NOTE_MENU) == ["copy", "and2", "or2", "orF", "For", "orN", "Nor"]
    x = "( P > Q )"
    want = {
        "copy": ("( P > Q )", "R"), "and2": ("( ( P > Q ) & ( P > Q ) )", "ANDI"),
        "or2": ("( ( P > Q ) v ( P > Q ) )", "ORI1"), "orF": ("( ( P > Q ) v F )", "ORI1"),
        "For": ("( F v ( P > Q ) )", "ORI2"), "orN": ("( ( P > Q ) v ( ~ ( P > Q ) ) )", "ORI1"),
        "Nor": ("( ( ~ ( P > Q ) ) v ( P > Q ) )", "ORI2"),
    }  # fmt: skip
    for f, (formula, rule) in want.items():
        assert " ".join(N.note_formula(f, x)) == formula and N.note_rule(f) == rule


# ---------------------------------------------------------------------------- add / read / strip
def test_parse_render_roundtrip():
    for p in (PROOF, PROOF3):
        lines, tail = N.parse_lines(p)
        assert N.render_lines(lines, tail) == p


def test_add_notes_renumbers():
    q = N.add_notes(PROMPT, PROOF, {"~a": "Nor", "&cc": "and2"})
    assert q == (
        "N1 ( ~ P ) : PR ; N2 ( ( R & R ) & ( Q v Q ) ) : PR ; N3 ( ( ~ ( ~ P ) ) v ( ~ P ) ) : ORI2 N1 ; "
        "N4 ( ( ( R & R ) & ( Q v Q ) ) & ( ( R & R ) & ( Q v Q ) ) ) : ANDI N2 N2 ; N5 | ( ~ ( ~ P ) ) : AS ; "
        "N6 | ( Q v Q ) : ANDE2 N2 ; N7 | ( ( ~ P ) & ( ( R & R ) & ( Q v Q ) ) ) : ANDI N1 N2 ; "
        "N8 | ( ( Q v Q ) & ( ( ~ P ) & ( ( R & R ) & ( Q v Q ) ) ) ) : ANDI N6 N7 ; "
        "N9 ( ( ~ ( ~ P ) ) > ( ( Q v Q ) & ( ( ~ P ) & ( ( R & R ) & ( Q v Q ) ) ) ) ) : IMPI N5 N8 ; QED"
    )
    q3 = N.add_notes(PROMPT3, PROOF3, {"vaa": "copy", "atom": "orF"})
    assert q3 == (
        "N1 ( P v Q ) : PR ; N2 ( Q v P ) : PR ; N3 R : PR ; N4 ( P v Q ) : R N1 ; N5 ( Q v P ) : R N2 ; "
        "N6 ( R v F ) : ORI1 N3 ; N7 | P : AS ; N8 | ( Q v P ) : ORI2 N7 ; N9 | Q : AS ; N10 | ( Q v P ) : ORI1 N9 ; "
        "N11 ( Q v P ) : ORE N1 N7 N8 N9 N10 ; QED"
    )
    with pytest.raises(KeyError):
        N.add_notes(PROMPT3, PROOF3, {"vaa": "copy"})


def test_no_premise_no_note():
    p = "N1 | P : AS ; N2 ( P > P ) : IMPI N1 N1 ; QED"
    assert N.add_notes("THM SEQ ( P > P ) PRF", p, {}) == p
    assert N.read_notes("THM SEQ ( P > P ) PRF", p, TABLE) == []


def test_read_notes_sides_and_positions():
    q3 = N.add_notes(PROMPT3, PROOF3, {"vaa": "Nor", "atom": "copy"})
    rd = N.note_reading(PROMPT3, q3, TABLE)
    assert rd["ok"] and rd["late"] == 0 and rd["unnoted"] == 0 and rd["n_prem"] == 3
    assert [(x["premise_index"], x["cls"], x["form"], x["side"], x["line"], x["pos"]) for x in rd["notes"]] == [
        (0, "vaa", "Nor", "B", 4, 0), (1, "vaa", "Nor", "B", 5, 1), (2, "atom", "copy", "other", 6, 2)]  # fmt: skip
    assert N.read_notes(PROMPT3, q3, TABLE) == rd["notes"]


def test_read_notes_every_form_roundtrip():
    for f in N.NOTE_MENU:
        q = N.add_notes(PROMPT, PROOF, {"~a": f, "&cc": f})
        got = N.read_notes(PROMPT, q, TABLE)
        assert [x["form"] for x in got] == [f, f] and [x["premise_index"] for x in got] == [0, 1]
        assert N.strip_notes(PROMPT, q) == PROOF


def test_cited_or_last_or_late_lines_are_not_notes():
    # a menu-form line that is cited is a real step, not a note: the header ends there
    p = "N1 P : PR ; N2 ( P v F ) : ORI1 N1 ; N3 ( P v F ) : R N2 ; QED"
    rd = N.note_reading("THM P SEQ ( P v F ) PRF", p, TABLE)
    assert rd["notes"] == [] and rd["unnoted"] == 1 and rd["late"] == 0
    # the last line (the conclusion) is never a note
    p = "N1 P : PR ; N2 ( P v F ) : ORI1 N1 ; QED"
    assert N.note_reading("THM P SEQ ( P v F ) PRF", p, TABLE)["notes"] == []
    # a dead menu-form line after a real line is late, not a note
    p = "N1 P : PR ; N2 Q : PR ; N3 ( P & Q ) : ANDI N1 N2 ; N4 ( Q v F ) : ORI1 N2 ; N5 ( Q & P ) : ANDI N2 N1 ; QED"
    rd = N.note_reading("THM P , Q SEQ ( Q & P ) PRF", p, TABLE)
    assert rd["notes"] == [] and rd["late"] == 1 and rd["unnoted"] == 2
    assert rd["late_forms"][0]["form"] == "orF" and rd["late_forms"][0]["premise_index"] == 1
    # a note for the second premise only; ANDI citing two different lines is not and2; ORI2 ( X v X ) is not a menu form
    p = "N1 P : PR ; N2 Q : PR ; N3 ( Q v F ) : ORI1 N2 ; N4 ( Q & P ) : ANDI N2 N1 ; QED"
    rd = N.note_reading("THM P , Q SEQ ( Q & P ) PRF", p, TABLE)
    assert [(x["premise_index"], x["form"], x["side"]) for x in rd["notes"]] == [
        (1, "orF", "B")
    ] and rd["unnoted"] == 1
    p = "N1 P : PR ; N2 Q : PR ; N3 ( P & Q ) : ANDI N1 N2 ; N4 ( Q v Q ) : ORI2 N2 ; N5 ( Q & P ) : ANDI N2 N1 ; QED"
    rd = N.note_reading("THM P , Q SEQ ( Q & P ) PRF", p, TABLE)
    assert rd["notes"] == [] and rd["late"] == 0
    # a boxed menu-form line right after the PR block is not a note (depth 0 only); it counts as late if dead
    p = "N1 P : PR ; N2 | Q : AS ; N3 | ( P v F ) : ORI1 N1 ; N4 | P : R N1 ; N5 ( Q > P ) : IMPI N2 N4 ; QED"
    rd = N.note_reading("THM P SEQ ( Q > P ) PRF", p, TABLE)
    assert rd["notes"] == [] and rd["late"] == 1


def test_garbage_is_not_read():
    rd = N.note_reading("THM P SEQ P PRF", "N1 P : PR ; garbage ; QED", TABLE)
    assert not rd["ok"] and rd["notes"] == []
    assert N.strip_notes("", "N1 P : PR ; garbage ; QED") == "N1 P : PR ; garbage ; QED"


def test_strip_notes_inverse():
    for forms in ({"~a": "copy", "&cc": "orN"}, {"~a": "For", "&cc": "or2"}):
        assert N.strip_notes(PROMPT, N.add_notes(PROMPT, PROOF, forms)) == PROOF
    assert N.strip_notes(PROMPT, PROOF) == PROOF


# ---------------------------------------------------------------------------- Dan's checks on every form
def test_every_form_passes_nd_verify_and_state_env():
    verify, decompose = dan()
    for f in (*N.NOTE_MENU, *N.JUNK_FORMS):
        for prompt, proof, cls in (
            (PROMPT, PROOF, ("~a", "&cc")),
            (PROMPT3, PROOF3, ("vaa", "atom")),
        ):
            q = N.add_notes(prompt, proof, dict.fromkeys(cls, f))
            ok, why, _ = verify(prompt + " " + q)
            assert ok, (f, why)
            steps, _t, _e = decompose(prompt, q, canon=True)
            n = len(N.premises(q))
            n_old = len(decompose(prompt, proof, canon=True)[0])
            assert len(steps) == n_old + n
            # each note's hypothesis is in every state after its step (formula as Lean tokens)
            from lean_tok import ftoks
            from nd_verify.verify import parse_formula

            for k, x in enumerate(N.premises(q)):
                note = " ".join(ftoks(parse_formula(N.note_formula(f, x), 0)[0]))
                first = (
                    n + k + 1
                )  # n premise copies, then the notes in order; state i comes before action i
                assert all(f" : {note} <nl>" in " ".join(st) for st, _a, _h in steps[first:]), (
                    f,
                    k,
                )


def test_profile_ignores_notes():
    import authors_profile as AP

    for prompt, proof, cls in ((PROMPT, PROOF, ("~a", "&cc")), (PROMPT3, PROOF3, ("vaa", "atom"))):
        a = AP.profile(prompt, proof)
        for f in N.NOTE_MENU:
            b = AP.profile(prompt, N.add_notes(prompt, proof, dict.fromkeys(cls, f)))
            for k in ("order", "reit", "conj", "conj_all", "route", "sigma", "bundle", "joint"):
                assert a[k] == b[k], (f, k)
            assert b["dead"] == len(N.premises(proof))


# ---------------------------------------------------------------------------- table
def test_draw_table_balance_and_determinism():
    t = N.draw_table()
    assert t == N.draw_table(N.SEED)
    assert N.check_table(t) == []
    for side in ("A", "B"):
        counts = sorted(sum(t[c][side] == f for c in t) for f in N.NOTE_MENU)
        assert counts == [2, 2, 2, 2, 2, 2, 3]
    assert all(t[c]["A"] != t[c]["B"] for c in t)
    assert N.draw_table(1) != t


def test_write_table_refuses_a_different_table(tmp_path):
    p = tmp_path / "t.json"
    doc = N.write_table(p)
    assert json.loads(p.read_text())["table"] == doc["table"] == N.draw_table()
    assert N.write_table(p)["table"] == doc["table"]  # the same table: fine
    with pytest.raises(ValueError):
        N.write_table(p, seed=1)


def test_written_table_is_the_seeded_draw():
    if not N.TABLE_PATH.exists():
        pytest.skip("no table written yet")
    assert N.load_table() == N.draw_table(N.SEED)
    assert N.check_table(N.load_table()) == []
