"""Tests for lh_habits.py (the natural habits' readings and rewrites).  PY -m pytest -q test_lh_habits.py"""

# ruff: noqa: E501
from __future__ import annotations

import gzip
import itertools
import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import lh_habits as L  # noqa: E402

H = L.H
TA_AUTHORS = Path("/scratch/cowifornia/nd-rl/two_authors/data/authors_half.jsonl.gz")


def dan_verify():
    import tag_core as T

    T.install_dan()
    from nd_verify import verify_text

    return verify_text


def read(text: str) -> dict:
    return L.read_tree(H.build(text), set(), text)


def forms(text: str, h: str) -> tuple[int, int, int]:
    x = read(text)[h]
    return x["sites"], x["f1"], x["f2"]


def setter(text: str, h: str, form: int) -> str:
    T = H.build(text)
    L.SETTERS[h](T, form)
    return H.render(T)


def roundtrip(text: str, h: str, prompt: str) -> None:
    """text has its sites in form 2: set form 1, check, set form 2 again, get text back; both forms verify."""
    verify = dan_verify()
    s, f1, f2 = forms(text, h)
    assert s > 0 and f2 == s
    one = setter(text, h, 1)
    assert one != text
    assert forms(one, h) == (s, s, 0)
    assert setter(one, h, 2) == text
    for t in (text, one):
        ok, why, _ = verify(prompt + " " + t)
        assert ok, why
    # every other habit's reading is unchanged
    a, b = read(text), read(one)
    for g in L.NATURAL:
        if g != h:
            assert (a[g]["sites"], a[g]["f1"], a[g]["f2"]) == (
                b[g]["sites"],
                b[g]["f1"],
                b[g]["f2"],
            ), g


# ---------------------------------------------------------------------------- habit 8: halves
def test_halves_after_and_before_a_last_line():
    roundtrip(
        "N1 ( P & Q ) : PR ; N2 P : ANDE1 N1 ; N3 ( P v R ) : ORI1 N2 ; QED",
        "halves",
        "THM ( P & Q ) SEQ ( P v R ) PRF",
    )
    one = setter("N1 ( P & Q ) : PR ; N2 P : ANDE1 N1 ; N3 ( P v R ) : ORI1 N2 ; QED", "halves", 1)
    assert (
        one
        == "N1 ( P & Q ) : PR ; N2 P : ANDE1 N1 ; N3 Q : ANDE2 N1 ; N4 ( P v R ) : ORI1 N2 ; QED"
    )
    # the needed half is the proof's last line: the other half goes right before it
    t = "N1 ( P & Q ) : PR ; N2 Q : ANDE2 N1 ; QED"
    roundtrip(t, "halves", "THM ( P & Q ) SEQ Q PRF")
    assert setter(t, "halves", 1) == "N1 ( P & Q ) : PR ; N2 P : ANDE1 N1 ; N3 Q : ANDE2 N1 ; QED"


def test_halves_not_a_site():
    # both halves used; a conjunction of equal sides
    assert (
        forms(
            "N1 ( P & Q ) : PR ; N2 P : ANDE1 N1 ; N3 Q : ANDE2 N1 ; N4 ( Q & P ) : ANDI N3 N2 ; QED",
            "halves",
        )[0]
        == 0
    )
    assert (
        forms("N1 ( P & P ) : PR ; N2 P : ANDE1 N1 ; N3 ( P v R ) : ORI1 N2 ; QED", "halves")[0]
        == 0
    )


def test_halves_inside_a_box_through_a_reiteration():
    t = "N1 ( P & Q ) : PR ; N2 | R : AS ; N3 | ( P & Q ) : R N1 ; N4 | P : ANDE1 N3 ; N5 ( R > P ) : IMPI N2 N4 ; QED"
    roundtrip(t, "halves", "THM ( P & Q ) SEQ ( R > P ) PRF")
    assert forms(t, "reit") == (1, 1, 0)


# ---------------------------------------------------------------------------- habit 9: restate
def test_restate_first_top_level_use():
    t = "N1 P : PR ; N2 Q : PR ; N3 ( P & Q ) : ANDI N1 N2 ; QED"
    roundtrip(t, "restate", "THM P , Q SEQ ( P & Q ) PRF")
    assert (
        setter(t, "restate", 1)
        == "N1 P : PR ; N2 Q : PR ; N3 P : R N1 ; N4 Q : R N2 ; N5 ( P & Q ) : ANDI N3 N4 ; QED"
    )
    # only the first use is a site; later uses cite the premise in either form
    t = "N1 P : PR ; N2 ( P v Q ) : ORI1 N1 ; N3 ( ( P v Q ) & P ) : ANDI N2 N1 ; QED"
    assert forms(t, "restate") == (1, 0, 1)
    roundtrip(t, "restate", "THM P SEQ ( ( P v Q ) & P ) PRF")


def test_restate_copy_cited_twice_is_not_a_site():
    t = "N1 P : PR ; N2 P : R N1 ; N3 ( P v Q ) : ORI1 N2 ; N4 ( ( P v Q ) & P ) : ANDI N3 N2 ; QED"
    assert forms(t, "restate")[0] == 0


def test_restate_copies_are_collapsed_before_habits_2_to_4():
    t = "N1 P : PR ; N2 Q : PR ; N3 ( P & Q ) : ANDI N1 N2 ; QED"
    one = setter(t, "restate", 1)
    # the copies would be two independent units of one ANDI (a conjunct site) if they were read
    assert read(one)["conj"]["sites"] == read(t)["conj"]["sites"] == 0


# ---------------------------------------------------------------------------- habit 7: disjunct
def test_disjunct_free_site():
    t = "N1 P : PR ; N2 Q : PR ; N3 ( P & Q ) : ANDI N1 N2 ; N4 ( P v Q ) : ORI1 N1 ; N5 ( ( P & Q ) & ( P v Q ) ) : ANDI N3 N4 ; QED"
    roundtrip(t, "disjunct", "THM P , Q SEQ ( ( P & Q ) & ( P v Q ) ) PRF")
    assert "( P v Q ) : ORI2 N2" in setter(t, "disjunct", 1)


def test_disjunct_swap_that_moves_another_habit_is_not_a_site():
    # swapping would leave premise P without a top-level use (restate) -> not free
    t = "N1 P : PR ; N2 Q : PR ; N3 ( P v Q ) : ORI1 N1 ; N4 ( ( P v Q ) & Q ) : ANDI N3 N2 ; QED"
    assert forms(t, "disjunct")[0] == 0
    # the cited line would die
    t = "N1 P : PR ; N2 Q : PR ; N3 ( P & Q ) : ANDI N1 N2 ; N4 P : ANDE1 N3 ; N5 ( P v Q ) : ORI1 N4 ; N6 ( ( P v Q ) & ( P & Q ) ) : ANDI N5 N3 ; QED"
    assert forms(t, "disjunct")[0] == 0


# ---------------------------------------------------------------------------- the rest of the reading
def test_route_split_exfalso_readings():
    red = "N1 ( ( ~ P ) > F ) : PR ; N2 | ( ~ P ) : AS ; N3 | F : IMPE N1 N2 ; N4 ( ~ ( ~ P ) ) : NEGI N2 N3 ; N5 P : DN N4 ; QED"
    assert forms(red, "route") == (1, 1, 0)
    assert forms("N1 F : PR ; N2 P : BOTE N1 ; QED", "exfalso") == (1, 0, 1)
    vac = "N1 F : PR ; N2 | ( ~ P ) : AS ; N3 | F : R N1 ; N4 ( ~ ( ~ P ) ) : NEGI N2 N3 ; N5 P : DN N4 ; QED"
    assert forms(vac, "exfalso") == (1, 1, 0)
    split = "N1 ( P v Q ) : PR ; N2 R : PR ; N3 | P : AS ; N4 | R : R N2 ; N5 | Q : AS ; N6 | R : R N2 ; N7 R : ORE N1 N3 N4 N5 N6 ; QED"
    assert forms(split, "split") == (1, 1, 0)


@pytest.mark.skipif(not TA_AUTHORS.exists(), reason="no TA corpus")
def test_detect_tree_is_authors_corpus_detect():
    """On TA proofs without restate copies, the reading of habits 2-4 is W2's (the record's `realized`)."""
    n = 0
    with gzip.open(TA_AUTHORS, "rt") as f:
        for line in itertools.islice(f, 1500):
            r = json.loads(line)
            rd = read(r["proof"])
            det = L.AC.detect(r["proof"])
            if rd["restate"]["f1"] == 0:
                assert (
                    rd["detect"]
                    == {k: det[k] for k in ("order", "conjunct", "reit")}
                    == r["realized"]
                )
                n += 1
            assert rd["route"]["sites"] == int(r["route"])
    assert n > 1400


@pytest.mark.skipif(not TA_AUTHORS.exists(), reason="no TA corpus")
def test_setters_on_corpus_proofs():
    """Every habit with room set to each form and back, on TA proofs: forms read as set; Lean-free checks."""
    verify = dan_verify()
    seen = {h: 0 for h in L.REWRITABLE}
    with gzip.open(TA_AUTHORS, "rt") as f:
        for line in itertools.islice(f, 1500):
            r = json.loads(line)
            for h in L.REWRITABLE:
                s, f1, f2 = forms(r["proof"], h)
                if not s:
                    continue
                for form in (1, 2):
                    t = setter(r["proof"], h, form)
                    got = forms(t, h)
                    if got[0] == s:
                        assert got[form] == s
                    ok, why, _ = verify(r["prompt"] + " " + t)
                    assert ok, (h, form, why)
                seen[h] += 1
    assert all(v > 0 for v in seen.values()), seen
