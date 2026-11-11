"""Tests of authors_gate's arithmetic on synthetic rows (hand-computed cases and a simulated authors/control pair).

    CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 /scratch/cowifornia/nd-rl/think/nd-rl/code/.venv/bin/python \
        -m pytest --confcutdir=. -p no:cacheprovider -q test_authors_gate.py
"""

# ruff: noqa: E501

from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import authors_gate as AG  # noqa: E402
import authors_measure as AM  # noqa: E402

HAB = ("order", "reit", "conj")


def counts(
    sites: dict | None = None, a: dict | None = None, joint: dict | None = None, proofs: int = 0
) -> dict:
    sites, a = sites or {}, a or {}
    return {"proofs": proofs, "habits": {h: {"sites": sites.get(h, 0), "a": a.get(h, 0), "b": sites.get(h, 0) - a.get(h, 0)} for h in HAB},
            "bundle": {"A": 0, "B": 0, "mixed": 0, "none": 0}, "joint": joint or {}}  # fmt: skip


def row(thm: str, all_: dict, route: dict | None = None, plain: dict | None = None) -> dict:
    return {"theorem": thm, "attempts": 8, "accepted": all_["proofs"], "route": (route or {"proofs": 0})["proofs"],
            "habits": all_["habits"], "joint": all_["joint"], "bundle": all_["bundle"],
            "split": {"route": route or counts(), "plain": plain or counts()}}  # fmt: skip


def write(root: Path, ckpt: str, tag: str, s: str, rows: list[dict]) -> None:
    d = root / ckpt
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{s}__{tag}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_parse_joint():
    assert AG.parse_joint("oA.rB.c-") == {"order": "A", "reit": "B", "conj": None}
    assert AG.parse_joint("oM.r-.cA") == {"order": "M", "reit": None, "conj": "A"}


def test_q1_by_hand():
    rows_a = {
        ("s", "t1"): row("t1", counts({"order": 2}, {"order": 2})),
        ("s", "t2"): row("t2", counts({"order": 2}, {"order": 0})),
    }
    rows_b = {
        ("s", "t1"): row("t1", counts({"order": 2}, {"order": 0})),
        ("s", "t2"): row("t2", counts({"order": 2}, {"order": 0})),
    }
    ids = sorted(rows_a)
    W = AG.boot_weights(len(ids), 500, 0)
    q = AG.q1_tag(rows_a, rows_b, ids, W)
    assert q["order"]["tagA"] == pytest.approx(0.5) and q["order"]["tagB"] == 0.0
    assert q["order"]["diff"] == pytest.approx(0.5)
    assert q["all"]["diff"] == pytest.approx(0.5)  # no other habit has sites
    lo, hi = q["order"]["ci"]
    assert 0.0 <= lo <= 0.5 <= hi <= 1.0
    assert np.isnan(q["reit"]["diff"])  # no sites: undefined, never 0


def test_q2_by_hand_bundled_and_independent():
    bundled = {("s", "t"): {"joint": {"oA.rA.c-": 3, "oB.rB.c-": 1}}}
    q, _ = AG.q2_agree(bundled, [("s", "t")], AG.boot_weights(1, 10, 0))
    assert q["n"]["point"] == 4
    assert q["observed"]["point"] == pytest.approx(1.0)
    assert q["independent"]["point"] == pytest.approx(9 / 16 + 1 / 16)
    assert q["excess"]["point"] == pytest.approx(1 - 10 / 16)
    assert q["phi_order_reit"]["point"] == pytest.approx(1.0)
    indep = {("s", "t"): {"joint": {"oA.rA.c-": 1, "oA.rB.c-": 1, "oB.rA.c-": 1, "oB.rB.c-": 1}}}
    q, _ = AG.q2_agree(indep, [("s", "t")], AG.boot_weights(1, 10, 0))
    assert q["observed"]["point"] == pytest.approx(0.5) and q["independent"][
        "point"
    ] == pytest.approx(0.5)
    assert q["excess"]["point"] == pytest.approx(0.0) and q["phi_order_reit"][
        "point"
    ] == pytest.approx(0.0)
    # proofs with room for one habit, or none, are not in the >= 2 population; M forms count as disagreement
    mixed = {("s", "t"): {"joint": {"oA.r-.c-": 5, "o-.r-.c-": 7, "oM.rA.c-": 2, "oA.rA.cA": 2}}}
    q, _ = AG.q2_agree(mixed, [("s", "t")], AG.boot_weights(1, 10, 0))
    assert q["n"]["point"] == 4 and q["observed"]["point"] == pytest.approx(0.5)
    # within pattern (order, reit): pA(order) 0, pA(reit) 1 -> expected 0; pattern (order, reit, conj): all 1 -> 1
    assert q["independent"]["point"] == pytest.approx((2 * 0 + 2 * 1) / 4)


def test_q3_by_hand():
    r1 = row("t1", counts({"reit": 4}, {"reit": 2}, proofs=2), route=counts({"reit": 2}, {"reit": 2}, proofs=1),
             plain=counts({"reit": 2}, {"reit": 0}, proofs=1))  # fmt: skip
    r2 = row(
        "t2",
        counts({"reit": 2}, {"reit": 2}, proofs=1),
        plain=counts({"reit": 2}, {"reit": 2}, proofs=1),
    )  # no route proof
    rows = {("s", "t1"): r1, ("s", "t2"): r2}
    q, d = AG.q3_route(rows, sorted(rows), AG.boot_weights(2, 200, 0))
    assert q["theorems"] == 1 and q["route_proofs"] == 1 and q["plain_proofs"] == 1
    assert q["reit"]["route"] == 1.0 and q["reit"]["plain"] == 0.0 and q["reit"]["diff"] == 1.0
    assert q["all"]["diff"] == 1.0


def simulate(
    rng: random.Random, n_thm: int, bundled: bool, tag_follow: bool, route_tied: bool
) -> dict[str, list[dict]]:
    """Rows for one base under tags A, B, none: each proof has room for order and reit; an author is drawn per proof
    (tag A: mostly A, tag B: mostly B, none: 50/50 when the base follows the tag, else 50/50 always); a bundled base
    writes every habit in the proof's author's form (90 %), an unbundled one draws each habit independently (50 %).
    A route proof comes from author A (route_tied) at 10 %, else at random."""
    out: dict[str, list[dict]] = {}
    for tag in ("A", "B", "none"):
        pa = {"A": 0.9, "B": 0.1, "none": 0.5}[tag] if tag_follow else 0.5
        rows = []
        for t in range(n_thm):
            allc, rt, pl = AM.empty_counts(), AM.empty_counts(), AM.empty_counts()
            for _ in range(8):
                au = "A" if rng.random() < pa else "B"
                forms = {}
                for h in ("order", "reit"):
                    if bundled:
                        forms[h] = au if rng.random() < 0.9 else ("B" if au == "A" else "A")
                    else:
                        forms[h] = "A" if rng.random() < 0.5 else "B"
                route = (au == "A" and rng.random() < 0.2) if route_tied else rng.random() < 0.1
                p = {
                    h: {
                        "sites": 1 if h in forms else 0,
                        "a": int(forms.get(h) == "A"),
                        "b": int(forms.get(h) == "B"),
                    }
                    for h in HAB
                }
                p["bundle"] = (
                    "A"
                    if all(f == "A" for f in forms.values())
                    else "B"
                    if all(f == "B" for f in forms.values())
                    else "mixed"
                )
                p["joint"] = f"o{forms['order']}.r{forms['reit']}.c-"
                AM.add_profile(allc, p)
                AM.add_profile(rt if route else pl, p)
            rows.append({"theorem": f"t{t}", "attempts": 8, "accepted": 8, "route": rt["proofs"], "habits": allc["habits"],
                         "joint": allc["joint"], "bundle": allc["bundle"], "split": {"route": rt, "plain": pl}})  # fmt: skip
        out[tag] = rows
    return out


def test_gate_on_simulated_bases(tmp_path):
    rng = random.Random(0)
    for ck, kw in (("authors_x_s0", {"bundled": True, "tag_follow": True, "route_tied": True}),
                   ("control_x_s0", {"bundled": False, "tag_follow": False, "route_tied": False})):  # fmt: skip
        for tag, rows in simulate(rng, 150, **kw).items():
            write(tmp_path, ck, tag, "marker_plain", rows)
    res = AG.gate(tmp_path, "authors_x_s0", "control_x_s0", ["marker_plain"], B=400, seed=1)
    q1 = res["q1"]["marker_plain"]
    assert q1["authors_x_s0"]["all"]["diff"] > 0.5 and q1["authors_x_s0"]["all"]["ci"][0] > 0.4
    lo, hi = q1["control_x_s0"]["all"]["ci"]
    assert lo < 0 < hi
    q2 = res["q2"]["marker_plain"]
    assert (
        q2["authors_x_s0"]["excess"]["point"] > 0.2 and q2["authors_x_s0"]["excess"]["ci"][0] > 0.1
    )
    lo, hi = q2["control_x_s0"]["excess"]["ci"]
    assert lo < 0 < hi
    assert q2["excess_diff"]["ci"][0] > 0.1
    q3 = res["q3"]["marker_plain"]
    assert q3["authors_x_s0"]["all"]["diff"] > 0.3 and q3["authors_x_s0"]["all"]["ci"][0] > 0.15
    lo, hi = q3["control_x_s0"]["all"]["ci"]
    assert lo < 0 < hi
    text = AG.report(res)
    assert "(1) The base follows the tag" in text and " pp [" in text and "%" in text
    json.dumps(res, default=float)  # serialisable for --json


def test_missing_rows_refuse(tmp_path):
    with pytest.raises(SystemExit, match="no rows"):
        AG.gate(tmp_path, "authors_x_s0", "control_x_s0", ["marker_plain"], B=10)


def test_q1_reit_fits():
    def r(t, a):
        x = row(t, counts({"reit": 2}, {"reit": a}))
        x["reit_fit"] = {
            "fits": {"proofs": 1, "sites": 2, "a": a},
            "nofit": {"proofs": 0, "sites": 0, "a": 0},
        }
        return x

    ra, rb = {("s", "t"): r("t", 2)}, {("s", "t"): r("t", 0)}
    q = AG.q1_tag(ra, rb, [("s", "t")], AG.boot_weights(1, 10, 0))
    assert q["reit_fits"]["diff"] == 1.0 and q["reit_fits"]["sites_tagA"] == 2
