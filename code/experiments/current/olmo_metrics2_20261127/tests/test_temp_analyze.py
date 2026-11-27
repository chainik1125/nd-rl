"""End-to-end on a synthetic C1: graded rows made by re-drawing Dan's stored dpo draws, so Delta must be about 0."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

X = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(X / "tools"))
sys.path.insert(0, str(X / "temp"))
# load temp/analyze.py under its own name: rank1/analyze.py is also importable as `analyze` in the same pytest run
# (orchestrator fix, 2026-11-27: the full-suite run picked up rank1's module from sys.modules)
_spec = importlib.util.spec_from_file_location("temp_analyze", X / "temp" / "analyze.py")
analyze = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(analyze)
import temp_common as tc  # noqa: E402
import temp_triggers as tt  # noqa: E402

pytestmark = pytest.mark.skipif(
    not (tc.DAN / "olmo_ifcurve" / "pids_armA.txt").exists(), reason="Dan's clone absent"
)


def synth(tmp_path, k=8, seed=0, boost=0.0):
    rng = np.random.default_rng(seed)
    st = tc.stored_arm_a("dpo")
    reach = tc.dc.reach()
    rows = []
    reach_set = set(reach)
    for p in tc.dc.arm_a():
        if p in reach_set:
            continue
        for c in range(2):
            src = st[p][int(rng.integers(0, 16))]
            s = int(src["strict"] or rng.random() < boost)
            rows.append(
                {
                    "pid": p,
                    "chunk": c,
                    "i": 0,
                    "strict": s,
                    "n_tok": src["n_tok"],
                    "finish": "stop",
                    "think_end": 1,
                }
            )
    for p in reach:
        for c in range(k):
            rows.append(
                {
                    "pid": p,
                    "chunk": c,
                    "i": 0,
                    "strict": 0,
                    "n_tok": 4000,
                    "finish": "stop",
                    "think_end": 1,
                }
            )
    path = tmp_path / "c1_graded.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    # shared 42 reach prompts need arm-A scoring from their first 2 draws: they have strict 0 by construction here
    return path


def test_analyze_runs_and_null_is_null(tmp_path):
    g = synth(tmp_path)
    analyze.main(["--c1", str(g), "--out-dir", str(tmp_path / "out")])
    res = json.load(open(tmp_path / "out" / "summary.json"))
    c1 = res["arm_a"]["C1"]
    assert c1["n_prompts"] == 218
    # the 42 shared prompts are forced to 0 passes (baseline ~0 too), the rest redrawn from the baseline: Delta about 0
    assert abs(c1["delta_pts"]) < 4 and c1["delta_lo95"] < 0 < c1["delta_hi95"]
    assert (
        abs(res["stored"]["dpo_pass1_pct"] - 29.0) < 0.05
        and abs(res["stored"]["D_pts"] - 39.3) < 0.1
    )
    assert res["split_dpo_draws_1_8"]["ever_passed"]["n_prompts"] == 100
    assert res["reach_C1b"]["passes"] == 0 and res["reach_C1b"]["baseline"].startswith("proxy")
    assert "share_RL" in c1 and set(c1["split"]) == {"ever_passed", "never_passed"}
    assert (tmp_path / "out" / "summary.csv").exists() and c1["families"]


def test_trigger_evaluate_null_and_boost(tmp_path):
    stored = json.load(
        open("/scratch/cowifornia/nd-rl/olmo_elicit/metrics2/inputs/temp/stored_rates.json")
    )
    g = synth(tmp_path)
    rows = tc.jl(g)
    res = tt.evaluate(rows, {}, 8, stored, None, False, {"accepted": False})
    assert (
        not res["trigger_A"]["fires"]
        and res["trigger_A"]["cost_usd"] == 6.3
        and "nothing" in res["decides"]
    )
    g2 = synth(tmp_path, boost=0.25, seed=1)
    res2 = tt.evaluate(tc.jl(g2), {}, 8, stored, None, False, {"accepted": False})
    assert (
        res2["trigger_A"]["fires"]
        and res2["trigger_A"]["sign"] == "positive"
        and res2["arm_a_delta"]["excludes_zero"]
    )


def test_reach_passes_fire_trigger_via_reach_rule(tmp_path):
    stored = json.load(
        open("/scratch/cowifornia/nd-rl/olmo_elicit/metrics2/inputs/temp/stored_rates.json")
    )
    rows = tc.jl(synth(tmp_path))
    reach = tc.dc.reach()
    for r in rows:
        if r["pid"] in reach[:2] and r["chunk"] == 5:
            r["strict"] = 1
    for r in rows:
        if r["pid"] == reach[2] and r["chunk"] in (6, 7):
            r["strict"] = 1
    res = tt.evaluate(rows, {}, 8, stored, None, False, {"accepted": False})
    assert (
        res["reach_rule"]["c1b"]["passes"] == 4
        and res["reach_rule"]["fires"]
        and res["trigger_A"]["via_reach"]
    )
