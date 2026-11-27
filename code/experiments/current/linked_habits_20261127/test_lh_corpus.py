"""Tests for lh_corpus.py (the four version-2 corpora).  PY -m pytest -q test_lh_corpus.py

Unit tests run on the first source records; the corpus tests read the built corpora and manifests (skipped before
the build)."""

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
import lh_corpus as LC  # noqa: E402
import lh_habits as L  # noqa: E402
import notes as N  # noqa: E402

DATA = LC.DATA
built = pytest.mark.skipif(
    not all((DATA / f"{c}.manifest.json").exists() for c in LC.CORPORA), reason="corpora not built"
)
needs_table = pytest.mark.skipif(not LC.HABIT_TABLE.exists(), reason="no habit table")


def man(c: str) -> dict:
    return json.loads((DATA / f"{c}.manifest.json").read_text())


def head(c: str, n: int) -> list[dict]:
    with gzip.open(DATA / f"{c}.jsonl.gz", "rt") as f:
        return [json.loads(x) for x in itertools.islice(f, n)]


def test_phase1_one_record():
    recs = N.LH and LC.NC.read_source("authors_half", 40)
    for i, r in enumerate(recs):
        p = LC.phase1_one((i, r["prompt"], r["proof"]))
        assert p["canonical"]
        assert len(p["combos"]) == 2 ** len(p["room"])
        for key, c in p["combos"].items():
            assert LC.n_lines(c["text"]) + p["n_prem"] <= LC.MAX_LINES or c["text"] == r["proof"]
            for k in range(0, len(key), 2):
                h = {"d": "disjunct", "r": "restate", "h": "halves"}[key[k]]
                form = int(key[k + 1])
                s, f1, f2 = c["reading"][h]
                if h not in c["skipped"] and s:
                    assert (f1 if form == 1 else f2) == s


def test_coins_shared_and_author_forms():
    assert LC.author_forms("A", {"disjunct": 1, "restate": 2, "halves": 2}) == {
        "disjunct": 1,
        "restate": 2,
        "halves": 2,
    }
    assert LC.author_forms("B", {"disjunct": 1, "restate": 2, "halves": 2}) == {
        "disjunct": 2,
        "restate": 1,
        "halves": 1,
    }
    r = {"key": "k1", "author": "A"}
    q = {h: 0.5 for h in LC.NEW}
    assert LC.new_forms("c4", r, {h: 1 for h in LC.NEW}, q) == LC.new_forms(
        "c0", r, {h: 1 for h in LC.NEW}, q
    )
    assert LC.note_sides("c9", r, ["atom"], {"atom": 0.5}) == LC.note_sides(
        "c0", r, ["atom"], {"atom": 0.5}
    )
    assert LC.note_sides("c24", r, ["atom", ">cc"], {}) == {"atom": "A", ">cc": "A"}


@needs_table
def test_habit_table():
    t = json.loads(LC.HABIT_TABLE.read_text())
    assert t["within_tolerance"] and abs(t["chosen"]["diff"]) <= LC.LENGTH_TOL
    assert {h: t["a_form"][h] for h in ("route", "order", "reit", "conj")} == {
        "route": 1,
        "order": 1,
        "reit": 1,
        "conj": 1,
    }
    assert t["linked"] == [h for h in L.HABITS if h not in L.NOT_LINKED] and len(t["linked"]) == 22
    assert len(t["candidates"]) == 8 and t["candidates"][0] == t["chosen"]


@built
@pytest.mark.parametrize("c", list(LC.CORPORA))
def test_manifest_validation(c):
    m = man(c)
    assert m["records"] == 155000 and m["sha256"] == LC.NC.sha256(Path(m["file"]))
    for k in LC.CHECKS:
        assert m["validation"][k]["fail"] == 0, (k, m["validation"][k])
    lean = m["lean"]
    assert lean["rejected"] == 0 and lean["sampled"] >= LC.LEAN_MIN
    for cell, (acc, checked, avail) in lean["per_cell_accepted_checked_available"].items():
        assert acc == checked and checked >= min(avail, LC.LEAN_PER_CELL), cell
    for h, x in m["habits"].items():
        if h in L.NOT_LINKED:
            continue
        assert x["assigned_agree"] == x["assigned_checked"], (h, x)


@built
def test_form_rates_match_across_corpora():
    ms = {c: man(c) for c in LC.CORPORA}
    for h in L.HABITS:
        if h in L.NOT_LINKED or h == "route":
            continue
        shares = {c: ms[c]["habits"][h]["a_share"] for c in ms}
        sites = min(ms[c]["habits"][h]["sites"] for c in ms)
        tol = max(0.01, 4 * (0.25 / max(1, sites)) ** 0.5)
        assert max(shares.values()) - min(shares.values()) <= 2 * tol, (h, shares)


@built
def test_lengths_balanced_where_linked():
    for c in ("c24", "c9"):
        L_ = man(c)["lengths"]["mean_lines_by_author"]
        assert abs(L_["A"] - L_["B"]) <= LC.LENGTH_TOL, (c, L_)
        assert (
            man(c)["lengths"]["lines"]
            and max(int(k) for k in man(c)["lengths"]["lines"]) <= LC.MAX_LINES
        )


@built
def test_associations():
    m24, m0 = man("c24"), man("c0")
    for h in ("order", "conj", "halves", "restate", "note_atom", "note_>cc"):
        assert m24["assoc_author_tag"][h]["author"]["phi"] > 0.9, h
        assert abs(m0["assoc_author_tag"][h]["author"]["phi"]) < 0.05, h
    m4 = man("c4")
    for h in ("halves", "restate", "note_atom"):
        assert abs(m4["assoc_author_tag"][h]["author"]["phi"]) < 0.05, h


@built
def test_records_across_corpora():
    a, b, c, d = (head(x, 2000) for x in ("c24", "c9", "c4", "c0"))
    for r24, r9, r4, r0 in zip(a, b, c, d, strict=True):
        assert r24["key"] == r9["key"] == r4["key"] == r0["key"]
        assert N.strip_notes("", r24["proof"]) == N.strip_notes("", r9["proof"])
        assert r9["note_forms"] == r4["note_forms"] == r0["note_forms"]
        assert r24["author"] == r9["author"] == r4["author"] and r24["tag_shown"] == r0["tag_shown"]
        for r in (r24, r9, r4, r0):
            assert set(r) >= {
                "author",
                "tag_shown",
                "author_tag",
                "forms",
                "realized",
                "route",
                "notes",
                "src_proof",
                "lh_corpus",
            }
            assert (
                set(r["forms"]) == set(L.HABITS)
                and r["n_lines"] == LC.n_lines(r["proof"]) <= LC.MAX_LINES
            )
        assert set(r24["note_forms"].values()) <= {r24["author"]}


# ---------------------------------------------------------------------------- c0n (design audit F7)
c0n_built = pytest.mark.skipif(not (DATA / "c0n.manifest.json").exists(), reason="c0n not built")


@c0n_built
def test_c0n_manifest():
    m = man("c0n")
    assert m["records"] == 155000 and m["sha256"] == LC.NC.sha256(Path(m["file"]))
    for k in LC.C0N_CHECKS:
        assert m["validation"][k]["fail"] == 0, (k, m["validation"][k])
    assert m["lean"]["rejected"] == 0 and m["lean"]["sampled"] >= LC.LEAN_MIN
    lc = m["launcher_counts"]
    assert lc["records"] == 155000 and lc["skipped"] == 0 and lc["sha256"] == m["sha256"]
    ep = m["equal_passes"]
    assert ep["factor_vs_authors_half"] == round(lc["tokens"] / ep["authors_half_tokens"], 6)
    assert (
        lc["tokens"] < ep["c0_for_comparison"]["tokens"] and m["total_lines"] < ep["c0_total_lines"]
    )
    for h in L.NOTE_HABITS:
        assert m["habits"][h]["a_sites"] == m["habits"][h]["b_sites"] == 0
    m0 = man("c0")
    for h in L.NATURAL:
        for k in ("sites", "a_sites", "b_sites", "room_records"):
            assert m["habits"][h][k] == m0["habits"][h][k], (h, k)


@c0n_built
def test_c0n_records_are_c0_without_notes():
    for r0, rn in zip(head("c0", 2000), head("c0n", 2000), strict=True):
        assert rn["proof"] == N.strip_notes(r0["prompt"], r0["proof"])
        assert (
            rn["n_lines"] == r0["n_lines"] - len(r0["notes"])
            and rn["notes"] == []
            and rn["lh_corpus"] == "c0n"
        )
        for k in (
            "key",
            "author",
            "tag_shown",
            "author_tag",
            "route",
            "src_proof",
            "ta_proof",
            "skipped",
        ):
            assert rn.get(k) == r0.get(k), k
        assert {h: rn["realized"][h] for h in L.NATURAL} == {
            h: r0["realized"][h] for h in L.NATURAL
        }
        assert all(rn["forms"][h] == "none" for h in L.NOTE_HABITS)
        assert all(rn["forms"][h] == r0["forms"][h] for h in L.NATURAL)


@pytest.mark.skipif(not (DATA / "roundtrip_check.json").exists(), reason="no round-trip check yet")
def test_roundtrip_check():
    d = json.loads((DATA / "roundtrip_check.json").read_text())
    for c in ("c24", "c9", "c4", "c0", "c0n"):
        x = d[c]
        assert x["records"] == x["readable"] == x["equal"] == 155000, (c, x)
        assert x["halves_other"] == 0
        assert x["file_sha256"] == LC.NC.sha256(DATA / f"{c}.jsonl.gz")
