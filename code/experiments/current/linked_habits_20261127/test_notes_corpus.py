"""Tests for notes_corpus.py (linked_habits_20261127).  PY -m pytest -q test_notes_corpus.py

The unit tests run on the first records of the TA sources; the corpus tests check the built corpora and their
manifests when they exist (skipped otherwise)."""

# ruff: noqa: E501
from __future__ import annotations

import collections
import gzip
import itertools
import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import notes as N  # noqa: E402
import notes_corpus as NC  # noqa: E402

needs_table = pytest.mark.skipif(not N.TABLE_PATH.exists(), reason="no form table yet")
needs_sources = pytest.mark.skipif(
    not all(p.exists() for p in NC.SOURCES.values()), reason="no TA sources"
)


def built(c: str) -> Path:
    return NC.DATA / f"{c}.jsonl.gz"


needs_corpora = pytest.mark.skipif(
    not all(built(c).exists() and (NC.DATA / f"{c}.manifest.json").exists() for c in NC.CORPORA),
    reason="corpora not built",
)


def head(path: Path, n: int) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [json.loads(x) for x in itertools.islice(f, n)]


# ---------------------------------------------------------------------------- draws
def test_coin_is_fair_shared_and_independent_of_class():
    keys = [f"k{i}" for i in range(20000)]
    a = [NC.coin(k, "&aa") == "A" for k in keys]
    b = [NC.coin(k, "vcc") == "A" for k in keys]
    assert abs(sum(a) / len(a) - 0.5) < 0.015 and abs(sum(b) / len(b) - 0.5) < 0.015
    both = sum(x and y for x, y in zip(a, b, strict=True)) / len(keys)
    assert abs(both - 0.25) < 0.015
    assert [NC.coin(k, "&aa") for k in keys[:50]] == [NC.coin(k, "&aa") for k in keys[:50]]
    rec = {"key": "k7", "author": "B"}
    assert NC.sides_for("l4", rec, ["&aa", "vcc"]) == NC.sides_for("l0", rec, ["&aa", "vcc"])


def test_l19_follows_the_author():
    assert NC.sides_for("l19", {"key": "x", "author": "A"}, ["atom", ">cc"]) == {
        "atom": "A",
        ">cc": "A",
    }
    assert NC.sides_for("l19", {"key": "x", "author": "B"}, ["atom"]) == {"atom": "B"}
    with pytest.raises(ValueError):
        NC.sides_for("l19", {"key": "x", "author": "none"}, ["atom"])


# ---------------------------------------------------------------------------- one record end to end
@needs_table
@needs_sources
@pytest.mark.parametrize("corpus", list(NC.CORPORA))
def test_process_first_records(corpus):
    src = NC.CORPORA[corpus][0]
    recs = NC.read_source(src, 60)
    table = N.load_table()
    for i, r in enumerate(recs):
        slim = {
            "prompt": r["prompt"],
            "proof": r["proof"],
            "key": r["key"],
            "author": r["author"],
            "tag": NC.shown_tag(r),
        }
        x = NC.process((corpus, i, slim, True))
        assert all(
            x["chk"][k] for k in ("state_env", "nd_verify", "premises_kept", "read_ok", "strip_ok")
        ), x["chk"]
        p = x["chk"]["profile"]
        assert all(p.values()), p
        n = len(N.premises(r["proof"]))
        assert len(x["notes"]) == n
        assert x["proof"].split().count(";") == r["n_lines"] + n
        for note in x["notes"]:
            assert note["form"] == table[note["cls"]][note["side"]]
            if corpus == "l19":
                assert note["side"] == r["author"]
            else:
                assert note["side"] == NC.coin(r["key"], note["cls"])
        out = NC.out_record(r, x, corpus)
        assert (
            out["ta_proof"] == r["proof"]
            and out["src_proof"] == r["src_proof"]
            and out["lh_corpus"] == corpus
        )
        if (
            "text" in r
        ):  # 70,455 of the 155,000 source records carry `text` (prompt + proof); it follows the proof
            assert out["text"] == r["prompt"] + " " + x["proof"]
        assert set(out) == set(r) | {"notes", "note_forms", "lh_corpus", "ta_proof"}
        assert all(
            out[k] == r[k]
            for k in ("author", "tag_shown", "author_tag", "forms", "realized", "route", "key")
        )


@needs_table
@needs_sources
def test_lean_sample_covers_every_cell():
    recs = NC.read_source("authors_half", 400)
    res = []
    for i, r in enumerate(recs):
        slim = {
            "prompt": r["prompt"],
            "proof": r["proof"],
            "key": r["key"],
            "author": r["author"],
            "tag": NC.shown_tag(r),
        }
        x = NC.process(("l4", i, slim, False))
        res.append(x)
    ids = NC.lean_sample("l4", res)
    assert ids == NC.lean_sample("l4", res)  # seeded
    cells = collections.Counter(
        (n["cls"], n["form"])
        for x in res
        for n in {(m["cls"], m["form"]): m for m in x["notes"]}.values()
    )
    got = collections.Counter(
        (c, f) for i in ids for c, f in {(m["cls"], m["form"]) for m in res[i]["notes"]}
    )
    for cell, n in cells.items():
        assert got[cell] >= min(n, NC.LEAN_PER_CELL)
    assert len(ids) == len(
        {x["i"] for x in res if x["notes"]}
    )  # fewer than LEAN_MIN noted records: all of them


# ---------------------------------------------------------------------------- the built corpora
@needs_corpora
@pytest.mark.parametrize("corpus", list(NC.CORPORA))
def test_manifest(corpus):
    man = json.loads((NC.DATA / f"{corpus}.manifest.json").read_text())
    assert man["sha256"] == NC.sha256(built(corpus))
    assert man["records"] == 155000
    assert man["table"] == N.load_table() and man["table_sha256"] == N.table_sha256()
    v = man["validation"]
    for k in ("state_env", "nd_verify", "premises_kept", "read_ok", "strip_ok"):
        assert v[k]["fail"] == 0, (k, v[k])
    assert v["profile"]["checked"] == 155000 and v["profile"]["all_equal"] == 155000
    lean = man["lean"]
    assert (
        lean["sampled"] >= NC.LEAN_MIN
        and lean["rejected"] == 0
        and lean["min_per_class_form"] >= NC.LEAN_PER_CELL
    )
    assert lean["cells"] == 30
    lo, hi = man["a_share_notes_class_range"]
    assert 0 < lo <= hi < 1
    if (
        corpus == "l19"
    ):  # every note in its author's form: a class's A-share is the author-A share of its records
        assert man["a_share_by_author_label"] == {"A": 1.0, "B": 0.0}
    else:
        for c in N.CLASSES:  # a fair coin per (record, class): within 4 standard errors of one half
            d = man["per_class"][c]
            assert (
                abs(d["a_share_record_classes"] - 0.5) <= 4 * (0.25 / d["record_classes"]) ** 0.5
            ), (c, d)
        for name, a in man["assoc"].items():
            assert a["phi"] is not None and abs(a["phi"]) < 0.02, (name, a)
    assert sum(man["per_class"][c]["notes"] for c in N.CLASSES) == man["notes_total"]
    assert (
        man["lengths"]["pairs_over_ctx_nominal_after"] == 0
        and man["lengths"]["actions_over_max_action_after"] == 0
    )


@needs_corpora
def test_corpora_agree_record_by_record():
    n = 3000
    l19, l4, l0 = (head(built(c), n) for c in ("l19", "l4", "l0"))
    srcs = {s: NC.read_source(s, n) for s in NC.SOURCES}
    for a, b, c, sa, sc in zip(
        l19, l4, l0, srcs["authors_half"], srcs["control_half"], strict=True
    ):
        assert a["key"] == b["key"] == c["key"] == sa["key"] == sc["key"]
        assert a["ta_proof"] == b["ta_proof"] == sa["proof"] and c["ta_proof"] == sc["proof"]
        assert (
            N.strip_notes(a["prompt"], a["proof"])
            == sa["proof"]
            == N.strip_notes(b["prompt"], b["proof"])
        )
        assert N.strip_notes(c["prompt"], c["proof"]) == sc["proof"]
        assert b["note_forms"] == c["note_forms"]  # one coin for l4 and l0
        assert set(a["note_forms"].values()) <= {a["author"]}
        assert (
            [x["cls"] for x in a["notes"]]
            == [x["cls"] for x in b["notes"]]
            == [x["cls"] for x in c["notes"]]
        )
        for r in (a, b, c):
            assert (
                r["n_lines"]
                == r["proof"].split().count(";")
                == len(N.premises(r["proof"])) + r["ta_proof"].split().count(";")
            )
            got = [(x["cls"], x["form"], x["side"]) for x in N.read_notes(r["prompt"], r["proof"])]
            assert got == [(x["cls"], x["form"], x["side"]) for x in r["notes"]]


@needs_corpora
def test_gzip_is_deterministic(tmp_path):
    recs = head(built("l4"), 200)
    p1, p2 = tmp_path / "a.jsonl.gz", tmp_path / "b.jsonl.gz"
    lines = [json.dumps(r) + "\n" for r in recs]
    NC.write_gz(p1, lines)
    NC.write_gz(p2, lines)
    assert p1.read_bytes() == p2.read_bytes()
