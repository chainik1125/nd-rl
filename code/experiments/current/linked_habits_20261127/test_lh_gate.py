"""Tests of lh_gate (version 2) and lh_stats: hand-made rows with known answers, and the whole gate on rows built from
real c24 proofs (a base that writes its corpus: notes and linked habits all in the author's form).

    CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 /scratch/cowifornia/nd-rl/think/nd-rl/code/.venv/bin/python \
        -m pytest --confcutdir=. -p no:cacheprovider -q test_lh_gate.py
"""

# ruff: noqa: E501, I001

from __future__ import annotations

import gzip
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import lh_core as C  # noqa: E402
import lh_gate as G  # noqa: E402
import lh_measure as LM  # noqa: E402
import lh_stats as S  # noqa: E402

REAL = C.LH / "data"
needs = pytest.mark.skipif(
    not (REAL / "c24.jsonl.gz").exists() or not C.HABIT_TABLE.exists(), reason="c24 not built"
)


def write_sets(root: Path, n: dict[str, int]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for s, k in n.items():
        (root / f"{s}.jsonl").write_text(
            "".join(json.dumps({"name": f"{s[:2]}{i}"}) + "\n" for i in range(k))
        )


def write_rows(d: Path, tag: str, s: str, rows: list[dict]) -> None:
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{s}__{tag}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_stats_known_answers(tmp_path):
    write_sets(tmp_path / "sets", {"marker_plain": 4})
    ids = S.set_ids(["marker_plain"], tmp_path / "sets")
    rows = [{"theorem": f"ma{i}", "attempts": 8, "accepted": 8 if i < 2 else 0,
             "lh": {"seen": 8, "step_cap": 1, "proofs": 8 if i < 2 else 0, "lines": 80 if i < 2 else 0,
                    "hab": {"reit": [10, 6, 2, 0, 0], "note_atom": [4, 1, 1, 1, 1]},
                    "first": {"reit": [3, 1]}, "later": {"reit": [3, 1]}, "link": {"notes|notes": [3, 1, 1, 3]}}}
            for i in range(4)]  # fmt: skip
    write_rows(tmp_path / "m", "none", "marker_plain", rows)
    m = S.load(tmp_path / "m", "none", ["marker_plain"], ids)
    s = m.sums(np.ones(4))
    assert float(S.a_share(s, "reit")) == 75.0 and float(S.a_share(s, "reit", "f")) == 75.0
    assert float(S.note_rate(s, ["note_atom"])) == 75.0 and float(
        S.other_rate(s, ["note_atom"])
    ) == pytest.approx(100 / 3)
    assert float(S.link(s, "link notes|notes")) == 50.0
    p = {k: float(v) for k, v in S.perf(s).items()}
    assert p == {
        "accepted_pct": 50.0,
        "pass1": 50.0,
        "pass8": 50.0,
        "mean_lines": 10.0,
        "step_cap_pct": 12.5,
    }
    W = S.boot_weights(4, 200, 0)
    e = S.est(lambda x: S.a_share(x["m"], "reit"), {"m": m}, W)
    assert e["point"] == 75.0 and e["ci"] == [75.0, 75.0]  # identical theorems: no spread
    assert S.load(tmp_path / "m", "A", ["marker_plain"], ids) is None


def rows_from(recs: list[dict], names: list[str], k: int = 8) -> list[dict]:
    """One row per theorem name: the k records given for it, all 'accepted' (a base that writes its corpus)."""
    out = []
    for i, n in enumerate(names):
        pr = recs[i * k : (i + 1) * k]
        lh = LM.empty_lh()
        for r in pr:
            LM.add_proof(lh, r["prompt"], r["proof"])
        lh.update(seen=k, step_cap=0)
        out.append({"theorem": n, "attempts": k, "accepted": len(pr), "lh": lh})
    return out


@needs
def test_gate_on_a_base_that_writes_c24(tmp_path, monkeypatch):
    with gzip.open(REAL / "c24.jsonl.gz", "rt") as f:
        recs = [json.loads(x) for x in itertools.islice(f, 3000)]
    by = {
        "A": [r for r in recs if r["author"] == "A"],
        "B": [r for r in recs if r["author"] == "B"],
    }
    write_sets(tmp_path / "sets", {"marker_plain": 30, "route_unseen": 10})
    meas = tmp_path / "measure"
    for s, n in (("marker_plain", 30), ("route_unseen", 10)):
        names = [f"{s[:2]}{i}" for i in range(n)]
        off = 0 if s == "marker_plain" else 30 * 8
        for tag, src in (("A", by["A"][off:]), ("B", by["B"][off:]), ("none", recs[off:])):
            rows = rows_from(src, names)
            for bid in ("c24_s0", "authors_half_s0"):
                write_rows(meas / bid, tag, s, rows)
    monkeypatch.setattr(C, "NOV14", ("authors_half", "control_half"))
    res = G.gate(meas, ["c24_s0"], 200, 0, 20, tmp_path / "sets")
    g = res["bases"]["pooled"]["c24_s0"]
    assert (
        set(g["G2"]["all_sites"]) == set(G.LINKS)
        and g["G2"]["all_sites"]["notes|notes"]["point"] == 100.0
    )
    assert not any(
        k.split("|")[0] == k.split("|")[1] for k in g["G2"]["habit_pairs"]
    )  # same-habit pairs are not read
    assert not any("split" in k or "exfalso" in k for k in g["G2"]["habit_pairs"])
    natural = [h for h in G.habit_lists()[1] if h not in G.NOT_POOLED]
    assert g["G1"]["pooled_natural"]["A"] == S.point(
        lambda x: S.pooled_share(x["A"], natural),
        G.base_mats(meas, "c24_s0", list(C.SETS), S.set_ids(list(C.SETS), tmp_path / "sets")),
        40,
    )
    assert set(g["near_per_proof"]) == set(LM.NEAR)
    assert g["G0"]["point"] == 100.0  # every premise carries a note in the corpus
    assert (
        g["G1"]["pooled_notes"]["diff"]["point"] == 100.0
        and g["G1"]["note_atom"]["A"] == 100.0
        and g["G1"]["note_atom"]["B"] == 0.0
    )
    assert (
        g["G1"]["split"]["diff"]["point"] is None or abs(g["G1"]["split"]["diff"]["point"]) < 100
    )  # split is not linked
    assert g["G2"]["notes|notes"]["point"] == 100.0 and g["G2"]["notes|notes"]["ci"] == [
        100.0,
        100.0,
    ]
    assert (
        g["G2"]["habit_pairs_summary"]["pairs"] > 0 and g["perf"]["accepted_pct"]["point"] == 100.0
    )
    monkeypatch.setattr(G, "G0_BASES", ("c24_s0",))
    assert G.decision(res)["reading"] == "stage 2 runs"
    assert (
        res["decision"]["reading"].startswith("not complete: lacks c9_s0") or "c9" not in C.CORPORA
    )  # c9, c4, c0 not measured
    f = res["perf_flag"]["marker_plain"]["c24_s0-authors_half_s0"]
    assert f["pass1_diff"]["point"] == 0.0 and f["pass8_diff"]["point"] == 0.0 and not f["degraded"]
    txt = G.report(res)
    assert (
        "c24_s0: G0 +100.0" in txt
        and "Bases not measured yet: c9_s0" in txt
        and "c24's G2 so far: stage 2 runs" in txt
    )


def test_pass1_flag(tmp_path, monkeypatch):
    """PREREG Amendment 3, point 9: the flag is on pass@1, more than 5 points lower with the interval excluding 0."""
    write_sets(tmp_path / "sets", {"marker_plain": 40, "route_unseen": 4})
    meas = tmp_path / "measure"
    for bid, acc in (("c0_s0", 3), ("control_half_s0", 6), ("c0n_s0", 3)):
        for s, n in (("marker_plain", 40), ("route_unseen", 4)):
            rows = [{"theorem": f"{s[:2]}{i}", "attempts": 8, "accepted": acc + (i % 2), "lh": {"seen": 8, "step_cap": 0, "proofs": 0}}
                    for i in range(n)]  # fmt: skip
            for tag in G.TAGS:
                write_rows(meas / bid, tag, s, rows)
    monkeypatch.setattr(C, "NOV14", ("authors_half", "control_half"))
    res = G.gate(meas, ["c0_s0", "c0n_s0"], 200, 0, 20, tmp_path / "sets")
    fl = res["perf_flag"]["marker_plain"]
    assert (
        fl["c0_s0-control_half_s0"]["degraded"]
        and fl["c0_s0-control_half_s0"]["pass1_diff"]["point"] == -37.5
    )
    assert not fl["c0_s0-c0n_s0"]["degraded"] and fl["c0_s0-c0n_s0"]["pass1_diff"]["point"] == 0.0
    assert "c24_s0-authors_half_s0" not in fl and "DEGRADED (PREREG flag, pass@1)" in G.report(res)


def test_decision_rule(monkeypatch):
    monkeypatch.setattr(G, "G0_BASES", ("c24_s0", "c9_s0", "c4_s0", "c0_s0"))

    def res(p, ci, g0=None, drop=()):
        g0 = g0 or {}
        b = {
            f"{c}_s0": {"G0": {"point": g0.get(c, 80.0)}}
            for c in ("c24", "c9", "c4", "c0")
            if c not in drop
        }
        if "c24" not in drop:
            b["c24_s0"]["G2"] = {"notes|notes": {"point": p, "ci": ci}}
        return {"bases": {"pooled": b}}

    assert G.decision(res(35.0, [31.0, 40.0]))["reading"] == "stage 2 runs"
    assert G.decision(res(35.0, [-1.0, 40.0]))["reading"] == "no stage 2"
    assert G.decision(res(15.0, [5.0, 25.0]))["reading"] == "stage 2 runs; the link is weak"
    assert G.decision(res(5.0, [1.0, 9.0]))["reading"] == "no stage 2"
    # Amendment 8 point 1: G0 >= 50 on every c base first; a base under it stops for repair whatever G2 says
    d = G.decision(res(35.0, [31.0, 40.0], {"c4": 42.0}))
    assert (
        d["reading"] == "stop for repair (G0): c4_s0 42.0 under 50"
        and d["G2_reading"] == "stage 2 runs"
    )
    # bases still pretraining on other pods: named, no crash; a G0 failure among those measured still stops
    d = G.decision(res(35.0, [31.0, 40.0], drop=("c24", "c9")))
    assert d["reading"] == "not complete: lacks c24_s0, c9_s0" and d["missing"] == [
        "c24_s0",
        "c9_s0",
    ]
    assert G.decision(res(35.0, [31.0, 40.0], {"c0": 10.0}, drop=("c9",)))["reading"].startswith(
        "stop for repair (G0): c0_s0"
    )
    assert G.decision(res(35.0, [31.0, 40.0], drop=("c9",)))["reading"].startswith(
        "not complete: lacks c9_s0 (G0 so far >= 50; c24's G2 so far: stage 2 runs"
    )
    assert G.decision({"bases": {}})["reading"].startswith("not complete: lacks c24_s0")


def test_noted_share_counts_premises_not_notes(tmp_path):
    """R1: a premise with two notes (lh_profile double_note) counts once in G0; note_rate counts notes."""
    write_sets(tmp_path / "sets", {"marker_plain": 2})
    ids = S.set_ids(["marker_plain"], tmp_path / "sets")
    # 10 premises: 4 with no note (none), 6 noted, one of them with two notes (A + B + other = 7 notes)
    rows = [
        {
            "theorem": f"ma{i}",
            "attempts": 8,
            "accepted": 8,
            "lh": {"proofs": 8, "hab": {"note_atom": [10, 4, 2, 1, 4]}},
        }
        for i in range(2)
    ]
    write_rows(tmp_path / "m", "none", "marker_plain", rows)
    s = S.load(tmp_path / "m", "none", ["marker_plain"], ids).sums(np.ones(2))
    assert (
        float(S.noted_share(s, ["note_atom"])) == 60.0
        and float(S.note_rate(s, ["note_atom"])) == 70.0
    )
