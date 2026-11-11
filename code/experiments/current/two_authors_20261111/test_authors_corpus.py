"""Tests for authors_corpus.py (two_authors_20261111).

  CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 /scratch/cowifornia/nd-rl/think/nd-rl/code/.venv/bin/python \
      -m pytest --confcutdir=. -p no:cacheprovider -q test_authors_corpus.py

Two builds of the first N = 1,500 source records (13 route records among them) go to a temporary folder, each with its
own fresh Lean cache, so every rewritten proof is checked by Lean in the test itself (about 20 s in all)."""

# ruff: noqa: E501
from __future__ import annotations

import gzip
import json
import math
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import authors_corpus as AC  # noqa: E402

H = AC.H
N = 1500

# goal-driven fixture of test_habits.py: the box opens first, Q is derived inside it
GD = (
    "THM ( P > Q ) , ( Q > S ) , P SEQ ( R > S ) PRF",
    "N1 ( P > Q ) : PR ; N2 ( Q > S ) : PR ; N3 P : PR ; N4 | R : AS ; N5 | Q : IMPE N1 N3 ; N6 | S : IMPE N2 N5 ; "
    "N7 ( R > S ) : IMPI N4 N6 ; QED",
)
# two independent conjuncts, left first
CONJ = (
    "THM ( P > Q ) , ( P > R ) , P SEQ ( Q & R ) PRF",
    "N1 ( P > Q ) : PR ; N2 ( P > R ) : PR ; N3 P : PR ; N4 Q : IMPE N1 N3 ; N5 R : IMPE N2 N3 ; "
    "N6 ( Q & R ) : ANDI N4 N5 ; QED",
)


def _build(tmp: Path) -> dict:
    AC.limit_cores(8)  # serrano: CPU work on at most 8 cores (Lean processes are multi-threaded)
    return AC.build(AC.SRC, tmp, N, procs=8, workers=8, cache=tmp / "lean.jsonl")


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    d = tmp_path_factory.mktemp("b1")
    summary = _build(d)
    corp = {}
    for c in AC.CORPORA:
        with gzip.open(d / f"{c}.jsonl.gz", "rt") as f:
            corp[c] = [json.loads(x) for x in f]
    mans = {c: json.loads((d / f"{c}.manifest.json").read_text()) for c in AC.CORPORA}
    src, items = AC.read_source(AC.SRC, N)
    return {"dir": d, "summary": summary, "corp": corp, "mans": mans, "src": src, "items": items}


def valid(prompt, body):
    ok, why, _ = AC._tools()["verify"](prompt + " " + body)
    assert ok, (why, body)


# ---------------------------------------------------------------------------- unit tests on hand-made proofs
def test_strip_and_add_reit_round_trip():
    p, b = GD
    P = H.build(b)
    A = AC.add_reit(
        P
    )  # the box cites N2 (and, goal-driven, N1 N3) from outside: one R per outer line
    ta = H.render(A)
    valid(p, ta)
    assert AC.n_lines(ta) == AC.n_lines(b) + 3
    rs = H.reit_sites(A)
    assert rs and all(x[3] for x in rs)
    S = H.build(ta)
    assert AC.strip_reit(S) == 3
    assert H.render(S) == b  # stripping undoes the reiteration exactly


def test_order_forms_and_detector():
    p, b = GD
    P = H.build(b)
    assert AC.order_form(P) == "B"
    pd = H.render(AC.normal_form(P, "pd"))
    valid(p, pd)
    assert AC.order_form(H.build(pd)) == "A"
    # reiteration does not change the order reading (R copies are stripped before it)
    assert AC.order_form(AC.add_reit(H.build(pd))) == "A"
    assert AC.order_form(AC.add_reit(P)) == "B"


def test_capped_sink_keeps_depth_two():
    # a box at top level whose only user is inside another box: habits.sink_all would put it at depth 2 inside the
    # depth-1 box's body only if its own box then sits at depth <= 2
    prompt = "THM ( P > Q ) SEQ ( R > ( P > Q ) ) PRF"
    body = "N1 ( P > Q ) : PR ; N2 | P : AS ; N3 | Q : IMPE N1 N2 ; N4 ( P > Q ) : IMPI N2 N3 ; N5 | R : AS ; N6 | ( P > Q ) : R N4 ; N7 ( R > ( P > Q ) ) : IMPI N5 N6 ; QED"
    valid(prompt, body)
    P = H.build(body)
    gd = AC.normal_form(P, "gd")
    assert H.max_depth(gd) <= 2
    valid(prompt, H.render(gd))
    deep = (
        "THM ( P > Q ) SEQ ( R > ( S > ( P > Q ) ) ) PRF",
        "N1 ( P > Q ) : PR ; N2 | P : AS ; N3 | Q : IMPE N1 N2 ; N4 ( P > Q ) : IMPI N2 N3 ; N5 | R : AS ; N6 | | S : AS ; "
        "N7 | | ( P > Q ) : R N4 ; N8 | ( S > ( P > Q ) ) : IMPI N6 N7 ; N9 ( R > ( S > ( P > Q ) ) ) : IMPI N5 N8 ; QED",
    )
    valid(*deep)
    D = H.build(deep[1])
    assert H.max_depth(H.normal_form(D, "gd")) == 3  # habits' own sink goes past the cap here
    assert H.max_depth(AC.normal_form(D, "gd")) == 2  # the capped one does not
    valid(deep[0], H.render(AC.normal_form(D, "gd")))


def test_conjunct_forms():
    p, b = CONJ
    P = H.build(b)
    sites = AC.conj_site_ids(P)
    assert len(sites) == 1
    A = AC.apply_conj(P, sites, "A")
    ta = H.render(A)
    valid(p, ta)
    assert ta.index("R : IMPE") < ta.index("Q : IMPE")
    assert AC.detect(ta)["conjunct"] == [1, 1, 1, 1]
    assert AC.detect(b)["conjunct"] == [1, 0, 1, 0]
    r = AC.rewrite(p, b, ("A", "A", "A"))
    assert r["det"]["conjunct"][3] == 1 and not r["skipped"]


def test_rewrite_all_forms_on_fixture():
    p, b = GD
    ra = AC.rewrite(p, b, AC.ALL_A)
    rb = AC.rewrite(p, b, AC.ALL_B)
    valid(p, ra["text"])
    valid(p, rb["text"])
    assert ra["det"]["order"] == "A" and rb["det"]["order"] == "B"
    s, a = ra["det"]["reit"]
    assert s and a == s
    s, a = rb["det"]["reit"]
    assert s and a == 0
    assert rb["text"] == b  # the fixture is already author B's proof


def test_lean_negative_control(tmp_path):
    p = "THM P , ( ~ P ) SEQ Q PRF"
    good = "N1 P : PR ; N2 ( ~ P ) : PR ; N3 F : NEGE N1 N2 ; N4 Q : BOTE N3 ; QED"
    bad = "N1 P : PR ; N2 ( ~ P ) : PR ; N3 F : NEGE N1 N2 ; N4 Q : DN N3 ; QED"
    AC.limit_cores(8)
    v = AC.lean_verdicts([(p, good), (p, bad)], 2, tmp_path / "c.jsonl")
    assert v[AC._lean_key(p, good)] and not v[AC._lean_key(p, bad)]
    # the cache gives the same verdicts
    v2 = AC.lean_verdicts([(p, good), (p, bad)], 2, tmp_path / "c.jsonl")
    assert v2 == v


def test_lean_fallback_drops_last_stage(monkeypatch):
    """A Lean rejection of a final proof drops its last applied stage and the record is rebuilt."""
    p, b = GD
    recs = [{"prompt": p, "proof": b}]
    ph1 = [AC.process({"_i": 0, "prompt": p, "proof": b})]
    ta = ph1[0]["texts"][ph1[0]["res"]["AAA"]["t"]]
    assert ph1[0]["res"]["AAA"]["applied"][-1] == "reit"

    def fake(pairs, workers, cache):
        return {AC._lean_key(x, y): y != ta for x, y in pairs}

    monkeypatch.setattr(AC, "lean_verdicts", fake)
    log: dict = {}
    fin = AC.resolve(recs, ph1, {(0, "AAA")}, 1, None, log)[(0, "AAA")]
    assert fin["banned"] == ["reit"] and fin["skipped"] == {"reit": "lean"}
    assert fin["det"]["order"] == "A" and fin["det"]["reit"][1] == 0
    assert log["lean_failed_stage"]["reit"] == 1


# ---------------------------------------------------------------------------- the built corpora
def test_determinism(built, tmp_path):
    _build(tmp_path)
    for c in AC.CORPORA:
        assert (tmp_path / f"{c}.jsonl.gz").read_bytes() == (
            built["dir"] / f"{c}.jsonl.gz"
        ).read_bytes(), c


def test_no_record_lost_or_duplicated_statements_unchanged(built):
    src = built["src"]
    for c, rows in built["corp"].items():
        assert len(rows) == len(src) == N, c
        assert len({r["key"] for r in rows}) == N
        for r, s in zip(rows, src, strict=True):
            for k in ("key", "thm", "prompt", "n_prem"):
                assert r[k] == s[k], (c, k)
            assert r["src_proof"] == s["proof"]
            valid(r["prompt"], r["proof"])


def test_limits_and_preamble(built):
    tok = AC._tools()["tok"]
    for c, rows in built["corp"].items():
        for r in rows:
            assert r["n_lines"] == AC.n_lines(r["proof"]) <= 12, c
            P = H.build(r["proof"])
            assert H.max_depth(P) <= 2 and len(P.prem) <= 3
            # the premise lines (the environment's preamble) are untouched, byte for byte
            pr = [s for s in r["proof"].split(" ; ") if s.split()[-1] == "PR"]
            pr0 = [s for s in r["src_proof"].split(" ; ") if s.split()[-1] == "PR"]
            assert pr == pr0
            assert r["rules"] == AC.rules_of(r["proof"])
            if "text" in r:
                assert r["text"] == r["prompt"] + " " + r["proof"]
            steps, _t, _e = AC._tools()["decompose"](r["prompt"], r["proof"], canon=True)
            for st, act, _h in steps:
                tok.encode_toks(st)
                tok.encode_toks(act)


def test_route_records_under_A_and_still_sigma_plus(built):
    knock = AC._tools()["knock"]
    dose = [d for _k, d in built["items"][:N]]
    assert sum(dose) == 13
    for c, rows in built["corp"].items():
        for r, d in zip(rows, dose, strict=True):
            drop, _fl = knock(r["proof"])
            assert bool(drop) == d == r["route"], (c, r["key"])
            if c.startswith("authors") and d:
                assert r["author"] == "A"
    for c in ("authors_half", "authors_p10"):
        assert all(r["author"] == "A" for r in built["corp"][c] if r["route"])
        assert built["mans"][c]["per_author"]["B"]["route"] == 0


def test_author_draws_are_the_existing_rules(built):
    items = built["items"]
    half = [AC.T.corpus_tag(k, d) for k, d in items[:N]]
    p10, info = AC.TC.route_share_tags(items, 0.10)
    assert [r["author"] for r in built["corp"]["authors_half"]] == half
    assert [r["author"] for r in built["corp"]["authors_p10"]] == p10[:N]
    assert p10.count("A") == 15500 and info["ordinary_A"] == 13950


def test_forms_as_assigned(built):
    """Every record's stored reading equals a fresh detector run, and every habit with room is in the assigned form
    unless the record says why it was skipped."""
    for c, rows in built["corp"].items():
        for r in rows:
            det = AC.detect(r["proof"])
            assert r["realized"] == det, (c, r["key"])
            f, sk = r["forms"], r.get("skipped", {})
            if det["order"] != "none" and "order" not in sk:
                assert det["order"] == f["order"], (c, r["key"])
            n, a, nf, af = det["conjunct"]
            if nf and "conjunct" not in sk:
                assert af == (nf if f["conjunct"] == "A" else 0), (c, r["key"])
            s, ra = det["reit"]
            if s and f["reit"] == "B":
                assert ra == 0, (c, r["key"])
            if s and f["reit"] == "A" and "reit" not in sk:
                assert ra == s, (c, r["key"])
            if c.startswith("authors"):
                assert set(f.values()) == {r["author"]}


def test_record_fields(built):
    for c, rows in built["corp"].items():
        for r in rows:
            assert r["author_tag"] == (r["author"] if r["tag_shown"] else "none")
            assert r["ta_corpus"] == c
            assert r["tag_shown"] == AC.tag_shown(r["key"])


def test_tag_shown_on_about_half_of_each_author(built):
    for c, rows in built["corp"].items():
        for a in ("A", "B"):
            xs = [r["tag_shown"] for r in rows if r["author"] == a]
            z = (sum(xs) - len(xs) / 2) / (math.sqrt(len(xs)) / 2)
            assert abs(z) < 3, (c, a, sum(xs), len(xs))


def test_control_independence(built):
    """In the controls the assigned forms and the tag label come from separate hashes: on this sample no pair of them
    (nor any of them and the route) is associated beyond chance (|phi| < 3.5 / sqrt(n)), and the realized per-site
    shares are near the authors corpus's."""
    for v in ("half", "p10"):
        rows = built["corp"][f"control_{v}"]
        var = {h: [1 if r["forms"][h] == "A" else 0 for r in rows] for h in AC.HABITS}
        var["author"] = [1 if r["author"] == "A" else 0 for r in rows]
        var["route"] = [1 if r["route"] else 0 for r in rows]
        for x in var:
            for y in var:
                if x < y:
                    s = AC.assoc(var[x], var[y])
                    if s["phi"] is not None:
                        assert abs(s["phi"]) < 3.5 / math.sqrt(s["n"]), (v, x, y, s)
        man = built["mans"][f"control_{v}"]
        tgt = man["calibration"]["target"]
        for h in ("order", "reit"):
            assert abs(man["realized_share"][h] - tgt[h]) < 0.05, (v, h, man["realized_share"], tgt)


def test_authors_shares(built):
    """Author A writes every habit in A's form where it had room and fit, author B never does."""
    for v in ("half", "p10"):
        m = built["mans"][f"authors_{v}"]["habits"]
        for a, want in (("A", 1.0), ("B", 0.0)):
            assert m["order"][a]["sites"] > 0 and m["order"][a]["a_share"] == want
            if m["conjunct"][a][
                "free_sites"
            ]:  # a few records hold two sites that need opposite orders
                assert abs(m["conjunct"][a]["free_a_share"] - want) <= 0.01
        assert m["reit"]["B"]["a_share"] == 0.0
        assert (
            m["reit"]["A"]["a_share"] > 0.4
        )  # the 12-line cap skips many of A's long proofs (CORPUS.md)
        sk = m["reit"]["A"]["skipped"]
        assert set(sk) <= {"lines>12", "route_detector", "lean"}


def test_manifest_matches_files(built):
    for c in AC.CORPORA:
        m = built["mans"][c]
        p = built["dir"] / f"{c}.jsonl.gz"
        assert m["out_sha256"] == AC.T.sha256(p)
        assert m["records"] == N
        rows = built["corp"][c]
        for a in ("A", "B"):
            pa = m["per_author"][a]
            assert pa["records"] == sum(r["author"] == a for r in rows)
            assert pa["tag_shown"] == sum(r["author"] == a and r["tag_shown"] for r in rows)
            assert pa["route"] == sum(r["author"] == a and r["route"] for r in rows)


def test_limited_build_is_a_prefix_of_the_full_build(built):
    """The authors corpora do not depend on --limit or on the hash seed (pytest does not pin it, the CLI pins 0): the
    first N lines of the full build are this build's lines, byte for byte. (The controls differ: their calibration
    is over the records built.)"""
    for c in ("authors_half", "authors_p10"):
        full = AC.DATA / f"{c}.jsonl.gz"
        if not full.exists():
            pytest.skip("no full build in TA/data")
        with gzip.open(full, "rt") as f:
            head = [next(f) for _ in range(N)]
        with gzip.open(built["dir"] / f"{c}.jsonl.gz", "rt") as f:
            assert head == f.readlines(), c
