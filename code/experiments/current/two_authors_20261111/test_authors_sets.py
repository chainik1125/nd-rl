"""Tests of authors_sets: no corpus or RL-pool theorem in any set, the stated criteria, determinism, write-once.

    CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 /scratch/cowifornia/nd-rl/think/nd-rl/code/.venv/bin/python \
        -m pytest --confcutdir=. -p no:cacheprovider -q test_authors_sets.py
"""

# ruff: noqa: E501

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import authors_profile as AP  # noqa: E402
import authors_sets as AS  # noqa: E402

OFFICIAL = Path("/scratch/cowifornia/nd-rl/two_authors/sets")
SMALL = {"marker_plain": 12, "route_unseen": 6, "marker_heldout_class": 6}


def test_pkey_premise_order_and_renaming():
    a = "( P > Q ) , ( Q > R ) |- ( P > R )"
    b = "( S > P ) , ( R > S ) |- ( R > P )"  # premises swapped, atoms renamed
    assert AS.pkey(a) == AS.pkey(b)
    assert AS.pkey(a) != AS.pkey("( P > Q ) , ( Q > R ) |- ( R > P )")
    assert AS.pkey(" |- ( P > P )") == AS.pkey("|- ( Q > Q )")
    assert AS.split_thm("( P v Q ) , ( ~ ( P & Q ) ) |- P") == (
        ["( P v Q )", "( ~ ( P & Q ) )"],
        "P",
    )
    assert AS.rename(["( S & P )", "( R v S )"]) == ["( P & Q )", "( R v P )"]


def test_quotas():
    assert AS.quotas(500) == {7: 83, 8: 83, 9: 83, 10: 83, 11: 84, 12: 84}
    assert sum(AS.quotas(300).values()) == 300


@pytest.fixture(scope="module")
def exclusions():
    return AS.exclusion_keys(AS.exclusion_files())


@pytest.mark.skipif(not (OFFICIAL / "manifest.json").exists(), reason="official sets not built")
def test_official_sets_exclude_corpus_and_rl_pool(exclusions):
    keys, pkeys, info = exclusions
    assert str(AS.SRC_CORPUS) in info and str(AS.RL_POOL) in info and str(AS.TRANSFER) in info
    assert (
        info[str(AS.SRC_CORPUS)]["records"] == 155000 and info[str(AS.RL_POOL)]["records"] == 4495
    )
    for s in AS.SETS:
        rows = AS.load_set(s, OFFICIAL)
        assert len(rows) == AS.SIZES[s]
        assert not any(r["key"] in keys for r in rows), s
        assert not any(AS.pkey(r["thm"]) in pkeys for r in rows), s


@pytest.mark.skipif(not (OFFICIAL / "manifest.json").exists(), reason="official sets not built")
def test_official_sets_criteria_and_manifest():
    man = json.loads((OFFICIAL / "manifest.json").read_text())
    assert man["seed"] == AS.SEED and man["checks"]["ok"]
    r = AS.check(OFFICIAL)
    assert r["ok"], r
    assert all(v == 0 for v in r["pairwise_overlap"].values())
    mp = AS.load_set("marker_plain", OFFICIAL)
    ru = AS.load_set("route_unseen", OFFICIAL)
    mh = AS.load_set("marker_heldout_class", OFFICIAL)
    assert all(not x["needs_route"] and not x["ref_route"] and not x["has_F"] for x in mp)
    assert all(x["has_F"] and not x["needs_route"] for x in mh)
    assert all(x["needs_route"] and x["ref_route"] and "( ~ ( ~" not in x["thm"] for x in ru)
    assert all(
        x["ref_room"]["order"] == 1 for x in mp + ru + mh
    )  # step order, the primary marker, has room
    # the stated criteria, recomputed from the reference proofs with the official profile
    for x in mp[:60] + ru[:30] + mh[:30]:
        p = AP.profile(x["prompt"], x["ref_proof"])
        assert {h: p[h]["sites"] for h in AP.HABITS} == x["ref_room"]
        assert p["route"] == x["ref_route"] and p["reit_fits"] == x["ref_reit_fits"]
        assert AS.intuit_provable(x["prompt"]) == (not x["needs_route"])
        assert p["order"]["sites"] == 1 and p["n_room"] >= (1 if x["set"] == "route_unseen" else 2)
        assert p["lines"] <= 12 and p["depth"] <= 2 and x["n_prem"] <= 3
    # length strata 7-12, as the manifest says
    for s, rows in (("marker_plain", mp), ("route_unseen", ru), ("marker_heldout_class", mh)):
        assert {str(k): v for k, v in man["sets"][s]["per_length"].items()} == {
            str(L): sum(1 for x in rows if x["n_lines"] == L)
            for L in range(7, 13)
            if any(x["n_lines"] == L for x in rows)
        }


@pytest.mark.skipif(not (OFFICIAL / "manifest.json").exists(), reason="official sets not built")
def test_official_build_refuses_a_second_write():
    with pytest.raises(SystemExit, match="written once"):
        AS.build(OFFICIAL, procs=1, sizes=SMALL, limit=10)


def test_build_is_deterministic(tmp_path):
    """Two builds with the same seed give byte-identical set files (a small build on the first 30,000 records of the
    seeded order)."""
    m1 = AS.build(tmp_path / "a", procs=4, sizes=SMALL, limit=30000)
    m2 = AS.build(tmp_path / "b", procs=2, sizes=SMALL, limit=30000)
    for s in AS.SETS:
        assert (tmp_path / "a" / f"{s}.jsonl").read_bytes() == (
            tmp_path / "b" / f"{s}.jsonl"
        ).read_bytes()
        assert m1["sets"][s]["sha256"] == m2["sets"][s]["sha256"]
        assert m1["sets"][s]["n"] == SMALL[s]
    assert m1["checks"]["ok"]
    with pytest.raises(SystemExit):
        AS.build(tmp_path / "a", procs=1, sizes=SMALL, limit=100)
