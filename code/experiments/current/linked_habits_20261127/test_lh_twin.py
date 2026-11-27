"""Tests for lh_twin.py (minimal pairs).  PY -m pytest -q test_lh_twin.py"""

# ruff: noqa: E501
from __future__ import annotations

import gzip
import inspect
import itertools
import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import lh_profile as LP  # noqa: E402
import lh_twin as TW  # noqa: E402
import notes as N  # noqa: E402

DATA = LP.LH / "data"
needs_tables = pytest.mark.skipif(not LP.HABIT_TABLE.exists(), reason="no habit table")
PROMPT = "THM ( ~ P ) , ( ( R & R ) & ( Q v Q ) ) SEQ ( ( ~ ( ~ P ) ) > ( ( Q v Q ) & ( ( ~ P ) & ( ( R & R ) & ( Q v Q ) ) ) ) ) PRF"
PROOF = (
    "N1 ( ~ P ) : PR ; N2 ( ( R & R ) & ( Q v Q ) ) : PR ; N3 | ( ~ ( ~ P ) ) : AS ; N4 | ( Q v Q ) : ANDE2 N2 ; "
    "N5 | ( ( ~ P ) & ( ( R & R ) & ( Q v Q ) ) ) : ANDI N1 N2 ; N6 | ( ( Q v Q ) & ( ( ~ P ) & ( ( R & R ) & ( Q v Q ) ) ) ) "
    ": ANDI N4 N5 ; N7 ( ( ~ ( ~ P ) ) > ( ( Q v Q ) & ( ( ~ P ) & ( ( R & R ) & ( Q v Q ) ) ) ) ) : IMPI N3 N6 ; QED"
)


def test_api():
    assert list(inspect.signature(TW.twin).parameters) == ["prompt", "proof", "habit", "tables"]
    assert "reit" in TW.FLIPPABLE and "note_atom" in TW.FLIPPABLE
    assert "order" not in TW.FLIPPABLE and "route" not in TW.FLIPPABLE


@needs_tables
def test_note_twin_and_back():
    nt = N.load_table()
    q = N.add_notes(PROMPT, PROOF, {"~a": nt["~a"]["A"], "&cc": nt["&cc"]["B"]})
    t = TW.twin(PROMPT, q, "note_~a")
    assert t is not None and t != q
    assert N.read_notes(PROMPT, t)[0]["side"] == "B" and N.read_notes(PROMPT, t)[1]["side"] == "B"
    assert TW.twin(PROMPT, t, "note_~a") == q
    assert TW.twin(PROMPT, q, "note_atom") is None  # no atom premise


@needs_tables
def test_reit_twin_and_back():
    t = TW.twin(PROMPT, PROOF, "reit")
    assert t is not None
    p0, p1 = LP.profile(PROMPT, PROOF), LP.profile(PROMPT, t)
    assert (p1["habits"]["reit"]["a"], p1["habits"]["reit"]["b"]) == (
        p0["habits"]["reit"]["b"],
        p0["habits"]["reit"]["a"],
    )
    assert TW.twin(PROMPT, t, "reit") == PROOF


@needs_tables
def test_fail_closed():
    assert TW.twin(PROMPT, "garbage", "reit") is None
    assert TW.twin(PROMPT, None, "reit") is None
    assert TW.twin(PROMPT, PROOF, "order") is None
    assert TW.twin(PROMPT, PROOF, "route") is None
    assert TW.twin(PROMPT, PROOF, "no_such_habit") is None
    assert TW.twin(PROMPT, PROOF, "disjunct") is None  # no site
    # a note in a third menu form: the note habit has a site not in A's or B's form
    nt = N.load_table()
    other = next(f for f in N.NOTE_MENU if f not in (nt["~a"]["A"], nt["~a"]["B"]))
    q = N.add_notes(PROMPT, PROOF, {"~a": other, "&cc": nt["&cc"]["A"]})
    assert TW.twin(PROMPT, q, "note_~a") is None
    # a premise without a note
    assert TW.twin(PROMPT, PROOF, "note_~a") is None
    # a box not right before its discharging line (not the renderer's layout)
    odd = "N1 P : PR ; N2 | Q : AS ; N3 | P : R N1 ; N4 ( P v R ) : ORI1 N1 ; N5 ( Q > P ) : IMPI N2 N3 ; N6 ( ( Q > P ) & ( P v R ) ) : ANDI N5 N4 ; QED"
    assert TW.twin("THM P SEQ ( ( Q > P ) & ( P v R ) ) PRF", odd, "reit") is None


@needs_tables
@pytest.mark.skipif(not (DATA / "c24.jsonl.gz").exists(), reason="corpora not built")
def test_twins_on_corpus_records():
    n = {h: 0 for h in ("reit", "note_atom", "restate", "halves")}
    with gzip.open(DATA / "c24.jsonl.gz", "rt") as f:
        for line in itertools.islice(f, 300):
            r = json.loads(line)
            for h in n:
                if not r["realized"][h][0]:
                    continue
                t = TW.twin(r["prompt"], r["proof"], h)
                if t is None:
                    continue
                n[h] += 1
                assert TW.twin(r["prompt"], t, h) == r["proof"]
    assert all(v > 10 for v in n.values()), n


@pytest.mark.skipif(not (DATA / "twin_check.json").exists(), reason="no twin check yet")
def test_twin_check_results():
    d = json.loads((DATA / "twin_check.json").read_text())["habits"]
    for h in ("reit", "note_atom"):
        x = d[h]
        assert x["twins"] >= 1000 and x["yield_pct"] >= 90
        assert x["back_equal"] == x["others_equal"] == x["flipped"] == x["twins"]
        assert x["lean"]["checked"] == 500 and x["lean"]["accepted"] == 500
    for h, x in d.items():
        assert x["others_equal"] == x["flipped"] == x["twins"], h
        assert x["lean"]["accepted"] == x["lean"]["checked"], h


@pytest.mark.skipif(not (DATA / "twin_check.json").exists(), reason="no twin check yet")
def test_twin_check_counts_mixed_apart():
    """Audit F10: mixed proofs (both forms) are counted apart; the count check is binding on the others."""
    d = json.loads((DATA / "twin_check.json").read_text())["habits"]
    for h, x in d.items():
        assert x["mixed_twins"] + x["unmixed_twins"] == x["twins"], h
        assert x["unmixed_flipped"] == x["unmixed_twins"], h
        assert x["mixed_with_site"] >= x["mixed_twins"] >= x["mixed_twins_a_eq_b"], h
