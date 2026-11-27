"""Tests for lh_profile.py (the 24-habit detector).  PY -m pytest -q test_lh_profile.py"""

# ruff: noqa: E501
from __future__ import annotations

import gzip
import inspect
import itertools
import json
import random
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import lh_profile as LP  # noqa: E402
import notes as N  # noqa: E402

DATA = LP.LH / "data"
needs_tables = pytest.mark.skipif(not LP.HABIT_TABLE.exists(), reason="no habit table")


def corpus(c: str) -> Path:
    return DATA / f"{c}.jsonl.gz"


def test_api():
    assert len(LP.HABITS) == 24 and len(set(LP.HABITS)) == 24
    assert LP.HABITS[:9] == (
        "route",
        "order",
        "reit",
        "conj",
        "split",
        "exfalso",
        "disjunct",
        "halves",
        "restate",
    )
    assert LP.HABITS[9:] == tuple(f"note_{c}" for c in N.CLASSES)
    assert list(inspect.signature(LP.profile).parameters) == ["prompt", "proof", "tables"]
    assert list(inspect.signature(LP.load_tables).parameters) == ["lh"]


@needs_tables
def test_load_tables():
    t = LP.load_tables()
    assert set(t) >= {"habit", "note", "linked", "a_form"}
    assert set(t["a_form"]) == set(LP.HABITS)
    assert "split" not in t["linked"] and "exfalso" not in t["linked"] and len(t["linked"]) == 22
    assert t["a_form"]["note_atom"] == N.load_table()["atom"]["A"]


GARBAGE = [
    "", "QED", ";", "N1", "N1 P : PR", "N1 P : PR ; garbage ; QED", "N1 P : PR ; N2 Q : ANDE1 N9 ; QED",
    "N1 | | | P : AS ; QED", "N1 P : PR ; N2 | Q : AS ; QED", "N1 ( P & : PR ; QED", "N2 P : PR ; N1 Q : R N2 ; QED",
    "N1 P : PR ; N2 P : R N2 ; QED", "N1 P : PR ; N2 ( P v P ) : ORI1 N1 N1 N1 ; QED", "LEANPARSE exact form",
    "N1 P : PR ; N2 | Q : AS ; N3 ( Q > P ) : IMPI N2 N9 ; QED", "N1 P : PR ; N2 P : FOO N1 ; QED",
]  # fmt: skip


@pytest.mark.parametrize("proof", GARBAGE)
def test_never_raises(proof):
    p = LP.profile("THM P SEQ P PRF", proof)
    assert set(p) >= {"ok", "unreadable", "err", "habits", "seq"}
    assert p["ok"] != p["unreadable"]
    if not p["ok"]:
        assert all(v["sites"] == 0 for v in p["habits"].values())


def test_never_raises_on_random_token_soup():
    rng = random.Random(0)
    toks = [
        "N1",
        "N2",
        "N3",
        "|",
        "(",
        ")",
        "P",
        "Q",
        "&",
        "v",
        ">",
        "~",
        "F",
        ":",
        "PR",
        "R",
        "AS",
        "IMPI",
        "ORE",
        "ANDI",
        ";",
        "QED",
    ]
    for _ in range(300):
        p = LP.profile(
            "THM P SEQ P PRF", " ".join(rng.choice(toks) for _ in range(rng.randint(1, 30)))
        )
        assert isinstance(p["ok"], bool)
    assert LP.profile("", None)["unreadable"]


def test_reading_of_a_noted_proof():
    t = (
        "N1 ( ~ P ) : PR ; N2 ( ( R & R ) & ( Q v Q ) ) : PR ; N3 ( ( ~ ( ~ P ) ) v ( ~ P ) ) : ORI2 N1 ; "
        "N4 ( ( ( R & R ) & ( Q v Q ) ) v F ) : ORI1 N2 ; N5 | ( ~ ( ~ P ) ) : AS ; N6 | ( Q v Q ) : ANDE2 N2 ; "
        "N7 | ( ( ~ P ) & ( ( R & R ) & ( Q v Q ) ) ) : ANDI N1 N2 ; N8 | ( ( Q v Q ) & ( ( ~ P ) & ( ( R & R ) & ( Q v Q ) ) ) ) : ANDI N6 N7 ; "
        "N9 ( ( ~ ( ~ P ) ) > ( ( Q v Q ) & ( ( ~ P ) & ( ( R & R ) & ( Q v Q ) ) ) ) ) : IMPI N5 N8 ; QED"
    )
    p = LP.profile("", t)
    assert p["ok"] and p["dead"] == 0
    assert p["habits"]["reit"]["sites"] == 2 and p["habits"]["halves"]["sites"] == 1
    assert p["habits"]["note_~a"]["sites"] == 1 and p["habits"]["note_&cc"]["sites"] == 1
    assert [x[1] for x in p["seq"]][:2] == ["note_~a", "note_&cc"]
    assert p["seq"] == sorted(p["seq"])
    # without notes the premises read as un-noted
    q = N.strip_notes("", t)
    assert LP.profile("", q)["habits"]["note_~a"]["none"] == 1


@needs_tables
@pytest.mark.skipif(
    not all(corpus(c).exists() for c in ("c24", "c9", "c4", "c0")), reason="corpora not built"
)
@pytest.mark.parametrize("c", ["c24", "c9", "c4", "c0"])
def test_round_trip_on_corpus_records(c):
    """The detector agrees with the record's `realized` (the build stored the detector's reading after checking it
    against the build's own reading of the natural habits and the written notes)."""
    with gzip.open(corpus(c), "rt") as f:
        for line in itertools.islice(f, 400):
            r = json.loads(line)
            p = LP.profile(r["prompt"], r["proof"])
            assert p["ok"]
            got = {
                h: [p["habits"][h]["sites"], p["habits"][h]["a"], p["habits"][h]["b"]]
                for h in LP.HABITS
            }
            assert got == r["realized"]
            for h, side in r["forms"].items():
                s, a, b = r["realized"][h]
                if (
                    side in ("A", "B")
                    and s
                    and h not in (r.get("skipped") or {})
                    and h in LP.load_tables()["linked"]
                ):
                    if h == "conj" and "conjunct" in (r.get("skipped") or {}):
                        continue
                    assert (a if side == "A" else b) == s, (h, side, r["realized"][h])


# ---------------------------------------------------------------------------- near misses (design audit F9 a-d)
BOX = "( ( ~ ( ~ P ) ) > ( ( Q v Q ) & ( ( ~ P ) & ( ( R & R ) & ( Q v Q ) ) ) ) )"
NPR = "THM ( ~ P ) , ( ( R & R ) & ( Q v Q ) ) SEQ " + BOX + " PRF"
HEAD = "N1 ( ~ P ) : PR ; N2 ( ( R & R ) & ( Q v Q ) ) : PR ; N3 | ( ~ ( ~ P ) ) : AS ; "


def _verify(prompt: str, proof: str) -> bool:
    import tag_core as T

    T.install_dan()
    from nd_verify import verify_text

    return bool(verify_text(prompt + " " + proof)[0])


def test_near_a_halves_apart_reads_other():
    """F9a: the unused other half one line later than the builder puts it is a halves site in neither form."""
    import lh_twin as TW

    builder = HEAD + (
        "N4 | ( Q v Q ) : ANDE2 N2 ; N5 | ( R & R ) : ANDE1 N2 ; N6 | ( ( ~ P ) & ( ( R & R ) & ( Q v Q ) ) ) : ANDI N1 N2 ; "
        "N7 | ( ( Q v Q ) & ( ( ~ P ) & ( ( R & R ) & ( Q v Q ) ) ) ) : ANDI N4 N6 ; N8 "
        + BOX
        + " : IMPI N3 N7 ; QED"
    )
    apart = HEAD + (
        "N4 | ( Q v Q ) : ANDE2 N2 ; N5 | ( ( ~ P ) & ( ( R & R ) & ( Q v Q ) ) ) : ANDI N1 N2 ; N6 | ( R & R ) : ANDE1 N2 ; "
        "N7 | ( ( Q v Q ) & ( ( ~ P ) & ( ( R & R ) & ( Q v Q ) ) ) ) : ANDI N4 N5 ; N8 "
        + BOX
        + " : IMPI N3 N7 ; QED"
    )
    assert _verify(NPR, builder) and _verify(NPR, apart)
    b, a = LP.profile(NPR, builder), LP.profile(NPR, apart)
    assert (
        b["habits"]["halves"]["sites"],
        b["habits"]["halves"]["f1"],
        b["habits"]["halves"]["other"],
    ) == (1, 1, 0)
    assert b["near"]["ande_apart"] == 0
    h = a["habits"]["halves"]
    assert (h["sites"], h["a"], h["b"], h["other"]) == (1, 0, 0, 1)
    assert a["near"]["ande_apart"] == 1
    assert not any(x[1] == "halves" for x in a["seq"])
    assert TW.twin(NPR, apart, "halves") is None


def test_near_b_dead_r_in_a_box():
    """F9b: an R copy in the box that the use does not cite is dead, pruned, and the site reads not reiterated."""
    t = HEAD + (
        "N4 | ( ~ P ) : R N1 ; N5 | ( Q v Q ) : ANDE2 N2 ; N6 | ( ( ~ P ) & ( ( R & R ) & ( Q v Q ) ) ) : ANDI N1 N2 ; "
        "N7 | ( ( Q v Q ) & ( ( ~ P ) & ( ( R & R ) & ( Q v Q ) ) ) ) : ANDI N5 N6 ; N8 "
        + BOX
        + " : IMPI N3 N7 ; QED"
    )
    assert _verify(NPR, t)
    p = LP.profile(NPR, t)
    assert p["near"]["dead_r_in_box"] == 1 and p["dead"] == 1
    assert (p["habits"]["reit"]["sites"], p["habits"]["reit"]["f1"]) == (2, 0)
    assert (
        p["near"]["late_note"] == 1
    )  # a dead copy of a premise is also a menu-form dead line outside the header


def test_near_c_cited_header_copy():
    """F9c: an A proof whose copy note (>cc's A form) gets cited loses the note and gains a restate copy."""
    pr = "THM ( ( P & Q ) > ( R v S ) ) , ( P & Q ) SEQ ( R v S ) PRF"
    good = (
        "N1 ( ( P & Q ) > ( R v S ) ) : PR ; N2 ( P & Q ) : PR ; N3 ( ( P & Q ) > ( R v S ) ) : R N1 ; "
        "N4 ( ( P & Q ) v ( ~ ( P & Q ) ) ) : ORI1 N2 ; N5 ( R v S ) : IMPE N1 N2 ; QED"
    )
    probe = good.replace("IMPE N1 N2", "IMPE N3 N2")
    assert _verify(pr, good) and _verify(pr, probe)
    g, p = LP.profile(pr, good), LP.profile(pr, probe)
    assert g["habits"]["note_>cc"]["a"] == 1 and g["near"] == dict.fromkeys(LP.NEAR, 0)
    assert p["habits"]["note_>cc"]["none"] == 1 and p["near"]["cited_header_copy"] == 1
    assert p["near"]["late_note"] == 1  # the &aa note now follows a line that is not a note
    assert (p["habits"]["restate"]["f1"], p["habits"]["restate"]["f2"]) == (1, 1)
    # habit 9's copies after the notes, or in a proof without notes, are not counted
    t = N.strip_notes(pr, good).replace(
        "N3 ( R v S ) : IMPE N1 N2",
        "N3 ( ( P & Q ) > ( R v S ) ) : R N1 ; N4 ( R v S ) : IMPE N3 N2",
    )
    assert _verify(pr, t) and LP.profile(pr, t)["near"]["cited_header_copy"] == 0


def test_near_d_double_note():
    """F9d: two notes for one premise count under the habit twice and once as double_note."""
    t = (
        "N1 ( ~ P ) : PR ; N2 ( ( R & R ) & ( Q v Q ) ) : PR ; N3 ( F v ( ~ P ) ) : ORI2 N1 ; N4 ( ( ~ ( ~ P ) ) v ( ~ P ) ) : ORI2 N1 ; "
        "N5 | ( ~ ( ~ P ) ) : AS ; N6 | ( Q v Q ) : ANDE2 N2 ; N7 | ( ( ~ P ) & ( ( R & R ) & ( Q v Q ) ) ) : ANDI N1 N2 ; "
        "N8 | ( ( Q v Q ) & ( ( ~ P ) & ( ( R & R ) & ( Q v Q ) ) ) ) : ANDI N6 N7 ; N9 "
        + BOX
        + " : IMPI N5 N8 ; QED"
    )
    assert _verify(NPR, t)
    nt = N.load_table()
    assert {nt["~a"]["A"], nt["~a"]["B"]} == {"For", "Nor"}
    p = LP.profile(NPR, t)
    assert p["near"]["double_note"] == 1
    h = p["habits"]["note_~a"]
    assert (h["sites"], h["a"], h["b"]) == (1, 1, 1)


def test_near_on_unreadable_is_zero():
    assert LP.profile("", "garbage")["near"] == dict.fromkeys(LP.NEAR, 0)
