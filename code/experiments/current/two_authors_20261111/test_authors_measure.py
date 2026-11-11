"""Tests of authors_measure: per-theorem counts, ids, the strict-106 profile of a Dan read, plans and refusals.
(The sampling path itself ran in the CPU smoke; RUNNER.md.)

    CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 /scratch/cowifornia/nd-rl/think/nd-rl/code/.venv/bin/python \
        -m pytest --confcutdir=. -p no:cacheprovider -q test_authors_measure.py
"""

# ruff: noqa: E501

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import authors_measure as AM  # noqa: E402
import test_authors_profile as TP  # noqa: E402  (the hand-written proofs)

OFFICIAL = Path("/scratch/cowifornia/nd-rl/two_authors/sets")


def test_ckpt_id_of():
    assert AM.ckpt_id_of("/x/bases/stage1_authors_p10_s0.pt") == "authors_p10_s0"
    assert (
        AM.ckpt_id_of("/x/runs/rl_authors_p10_s0/seg1/ckpts/rl_authors_p10_s0_it0040_snap.pt")
        == "rl_authors_p10_s0@it40"
    )
    assert AM.ckpt_id_of("/x/other.pt") == "other"


def test_theorem_row_counts():
    thm = {
        "name": "t",
        "prompt": TP.GD[0],
        "key": "k",
        "n_lines": 7,
        "needs_route": False,
        "has_F": False,
    }
    row = AM.theorem_row(
        thm,
        [TP.GD[1], TP.PD[1], TP.PD_REIT[1], "N1 junk : PR ; QED"],
        8,
        {"ckpt": "c", "tag": "none", "set": "s"},
    )
    assert (row["attempts"], row["accepted"], row["unparsed"], row["route"]) == (8, 4, 1, 0)
    assert row["habits"]["order"] == {"sites": 3, "a": 2, "b": 1}
    assert row["habits"]["reit"] == {
        "sites": 3 + 2 + 2,
        "a": 2,
        "b": 5,
    }  # GD cites N1, N2, N3 in its box
    assert row["bundle"] == {"A": 1, "B": 1, "mixed": 1, "none": 0}
    assert row["joint"] == {"oB.rB.c-": 1, "oA.rB.c-": 1, "oA.rA.c-": 1}
    assert row["split"]["route"]["proofs"] == 0 and row["split"]["plain"]["proofs"] == 3
    assert "proof" not in json.dumps(row).replace('"proofs"', "")  # counts only, no proof text


def test_theorem_row_route_split():
    thm = {"name": "r", "prompt": TP.ROUTE[0]}
    row = AM.theorem_row(thm, [TP.ROUTE[1], TP.CONJ_L[1]], 2, {})
    assert row["route"] == 1
    assert row["split"]["route"]["proofs"] == 1 and row["split"]["plain"]["proofs"] == 1
    assert row["split"]["plain"]["habits"]["conj"] == {"sites": 1, "a": 0, "b": 1}


def test_danread_profile(tmp_path):
    names = AM.strict106_names()
    assert len(names) == 106
    rp = tmp_path / "reads" / "it0010.json"
    rp.parent.mkdir()
    rp.write_text(json.dumps({"ckpt": "/x/runs/rl_a/seg1/ckpts/rl_a_it0010_snap.pt", "k": 64, "noise_seed": 510,
                              "author_tag": "none", "sigma": {names[0]: {"n_ok": 5}}}))  # fmt: skip
    rp.with_suffix(".proofs.jsonl").write_text(
        json.dumps({"name": names[0], "proofs": [TP.ROUTE[1], TP.PD[1]]}) + "\n"
    )
    out = AM.danread(argparse.Namespace(read=str(rp), ckpt_id=None, out=None))
    rows = [json.loads(x) for x in out.read_text().splitlines()]
    assert len(rows) == 106 and {r["set"] for r in rows} == {"strict106"}
    r0 = rows[0]
    assert (
        r0["ckpt"] == "rl_a@it10"
        and r0["unit"] == "distinct"
        and r0["accepted"] == 2
        and r0["route"] == 1
    )
    assert r0["accepted_attempts"] == 5 and r0["attempts"] == 64
    assert all(r["accepted"] == 0 for r in rows[1:])
    with pytest.raises(SystemExit, match="exists"):
        AM.danread(argparse.Namespace(read=str(rp), ckpt_id=None, out=None))


def _run_args(**kw) -> argparse.Namespace:
    d = {"ckpt": "/x/stage1_authors_p10_s0.pt", "ckpt_id": None, "tags": "A,B,none", "sets": "marker_plain,route_unseen",
         "sets_root": str(OFFICIAL), "k": 4, "seed": 0, "batch": 64, "device": "cpu", "limit": 3, "out_dir": None,
         "raw_dir": None, "resume": False, "dry_run": True}  # fmt: skip
    d.update(kw)
    return argparse.Namespace(**d)


@pytest.mark.skipif(not (OFFICIAL / "manifest.json").exists(), reason="official sets not built")
def test_run_plan_and_refusal(tmp_path):
    info = AM.run(_run_args(out_dir=str(tmp_path / "m"), raw_dir=str(tmp_path / "r")))
    assert (
        len(info["pairs"]) == 6
        and info["attempts"] == 6 * 3 * 4
        and info["ckpt_id"] == "authors_p10_s0"
    )
    assert not (tmp_path / "m").exists()  # a dry run writes nothing
    (tmp_path / "m").mkdir()
    (tmp_path / "m" / "marker_plain__A.summary.json").write_text("{}")
    with pytest.raises(SystemExit, match="never overwritten"):
        AM.run(_run_args(out_dir=str(tmp_path / "m"), raw_dir=str(tmp_path / "r")))
    info = AM.run(_run_args(out_dir=str(tmp_path / "m"), raw_dir=str(tmp_path / "r"), resume=True))
    assert info["skipped"] == [("marker_plain", "A")] and len(info["pairs"]) == 5


@pytest.mark.skipif(not (OFFICIAL / "manifest.json").exists(), reason="official sets not built")
def test_estimate_arithmetic():
    e = AM.estimate(
        32,
        ["marker_plain", "route_unseen", "marker_heldout_class"],
        ["A", "B", "none"],
        0.5,
        1.0,
        OFFICIAL,
    )
    assert e["theorems"] == 1100 and e["attempts"] == 1100 * 32 * 3
    assert e["min_3090"] == pytest.approx(e["attempts"] / AM.RATE_3090 / 60, abs=0.06)
    assert e["min_A40"] == pytest.approx(e["min_3090"] / AM.A40_OVER_3090, abs=0.06)
    half = AM.estimate(32, ["marker_plain"], ["A"], 0.5, 1.0, OFFICIAL, share=0.5)
    full = AM.estimate(32, ["marker_plain"], ["A"], 0.5, 1.0, OFFICIAL)
    assert half["min_3090"] == pytest.approx(2 * full["min_3090"], abs=0.11)


def test_theorem_row_length_and_route_fields():
    """Reference length, proof lengths, reiteration split by whether A's full form fits in 12 lines, the route with R
    followed (and Dan's knock kept for comparison)."""
    thm = {"name": "t", "prompt": TP.GD[0], "ref_proof": TP.GD[1]}
    row = AM.theorem_row(thm, [TP.PD_REIT[1], TP.PD[1]], 2, {})
    assert row["ref_lines"] == 7 and row["len_hist"] == {"9": 1, "7": 1}
    assert row["reit_fit"]["fits"] == {"proofs": 2, "sites": 4, "a": 2}
    assert row["reit_fit"]["nofit"] == {"proofs": 0, "sites": 0, "a": 0}
    assert row["reit_by_lines"] == {"9": [2, 2], "7": [2, 0]}
    rc = AM.theorem_row({"name": "r", "prompt": TP.RCOPY[0]}, [TP.RCOPY[1], TP.ROUTE[1]], 2, {})
    assert rc["route"] == 1 and rc["route_knock"] == 2 and rc["sigma"] == 1
