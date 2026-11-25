# ruff: noqa: E501, SIM115
"""CPU tests for Stage 0 and Stage 2 inputs (run: python -m pytest test_stage0.py -q, from this folder).

Need Dan's clone and fetched artifacts under /scratch/cowifornia/nd-rl/olmo_elicit/ext/ and the stage2 inputs
(make_stage2.py); a test skips when its data are absent.
"""

from __future__ import annotations

import gzip
import importlib.util
import json
import math

import analyze_arms as AA
import analyze_rep as AR
import dan_curve as dc
import make_stage2 as M2
import numpy as np
import pytest

ROWS = dc.DAN / "artifacts" / "olmo_ifcurve" / "rows"
need_rows = pytest.mark.skipif(not ROWS.exists(), reason="Dan's rows not fetched")


def his_cells():
    spec = importlib.util.spec_from_file_location("dan_cells", dc.DAN / "olmo_ifcurve" / "cells.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_constants_equal_his():
    c = his_cells()
    assert dc.CKPTS == c.CKPTS
    assert dc.SOURCES == c.SOURCES
    assert all(dc.N_A.get(k, 8) == c.N_A.get(k, 8) for k in dc.CKPTS)
    assert dc.N_B == c.N_B


def test_passk_estimator():
    assert dc.passk(8, 0, 1) == 0.0
    assert dc.passk(8, 8, 8) == 1.0
    assert dc.passk(8, 2, 1) == pytest.approx(0.25)
    assert dc.passk(4, 1, 2) == pytest.approx(1 - math.comb(3, 2) / math.comb(4, 2))
    assert math.isnan(dc.passk(4, 1, 8))
    # unbiased: mean over all k-subsets of "any success"
    n, c, k = 6, 2, 3
    from itertools import combinations

    subsets = list(combinations([1] * c + [0] * (n - c), k))
    assert dc.passk(n, c, k) == pytest.approx(np.mean([any(s) for s in subsets]))


@need_rows
def test_passk_matches_his_csv():
    t = dc.passk_table(dc.Rows(ROWS), dc.arm_a())
    his = dc.dan_passk()
    for c in dc.CKPTS:
        for k in dc.KS:
            h = his[c][k][0]
            if math.isnan(h):
                assert math.isnan(t[c][k])
            else:
                assert abs(t[c][k] - h) < 1e-6, (c, k)


@need_rows
def test_reach_matches_his_results():
    t = dc.reach_table(dc.Rows(ROWS), dc.reach())
    his = dc.dan_reach()
    for c in dc.CKPTS[1:]:
        assert t[c]["draws_min"] == 32
        assert t[c]["solved"] == his[c]["solved_strict"]
        assert t[c]["strict_rate"] == pytest.approx(his[c]["strict_rate"])


def test_stage2_inputs():
    out = M2.OUT
    if not (out / "manifest.json").exists():
        pytest.skip("stage2 inputs not built")
    tp, ao = M2.t_rows()
    rows = [json.loads(ln) for ln in gzip.open(out / "T_s0500.jsonl.gz", "rt")]
    assert len(rows) == 230 and {r["writer"] for r in rows} == {"s0500"}
    assert [r["prefix"] for r in rows[:115]] == [r["prefix"] for r in tp]
    assert [r["user_suffix"] for r in rows[115:]] == [r["user_suffix"] for r in ao]
    assert sum(int(r["n"]) for r in rows) == 1840
    # every field but writer / cond is his
    for a, b in zip(rows[:115], tp, strict=True):
        assert {k: v for k, v in a.items() if k not in ("writer", "cond")} == {
            k: v for k, v in b.items() if k not in ("writer", "cond")
        }
    # H problems: his lines, byte for byte
    his_lines = set(open(M2.SRC["hint16"]).read().splitlines()) | set(
        open(M2.SRC["hint32"]).read().splitlines()
    )
    ours = (out / "problems_H.jsonl").read_text().splitlines()
    assert len(ours) == 48 and set(ours) <= his_lines
    assert [json.loads(x)["pid"] for x in ours] == dc.reach()
    assert (out / "pids_rep.txt").read_text().split() == [f"{p}:2" for p in dc.arm_a()]


@pytest.mark.skipif(
    not (dc.DAN / "artifacts/olmo_answeronly/graded/R_dpo.jsonl").exists(), reason="not fetched"
)
def test_arms_reproduce_his_answeronly_summary():
    # his olmo_answeronly/summary.md: reach dpo T1 0.139, A1 0.633; s2300 T2 0.935, A1 0.952 (strict, 41 prompts)
    D = AA.D
    for path, cond, want in (
        (D / "olmo_transplant/graded/R_dpo.jsonl", "T1", 13.9),
        (D / "olmo_answeronly/graded/R_dpo.jsonl", "A1", 63.3),
        (D / "olmo_transplant/graded/R_s2300.jsonl", "T2", 93.5),
        (D / "olmo_answeronly/graded/R_s2300.jsonl", "A1", 95.2),
    ):
        s = AA.summarize(AA.t_cells(path, {cond}), 8, 41, reps=200)
        assert s["n_prompts"] == 41 and s["n_draws"] == 920
        assert abs(s["strict_pct"] - want) < 0.06, (path.name, cond, s["strict_pct"])


@need_rows
def test_rep_compare_on_his_own_draws(tmp_path):
    # his F_s2300_ifb draws as "ours": first 2 per prompt vs his 8-draw cell -> small difference, PASS
    g = dc.DAN / "artifacts/olmo_ifomega/graded/F_s2300_ifb.jsonl"
    keep, n = [], {}
    for ln in sorted(
        open(g), key=lambda x: (json.loads(x)["pid"], json.loads(x)["chunk"], json.loads(x)["i"])
    ):
        r = json.loads(ln)
        if n.get(r["pid"], 0) < 2:
            n[r["pid"]] = n.get(r["pid"], 0) + 1
            keep.append(ln)
    f = tmp_path / "graded.jsonl"
    f.write_text("".join(keep))
    res = AR.compare(AR.ours(f), dc.Rows(ROWS), "s2300", dc.arm_a(), reps=500)
    assert res["prompts"] == 218 and res["our_draws"] == 436
    assert res["diff_lo95"] <= 0 <= res["diff_hi95"] and res["verdict"].startswith("PASS")
