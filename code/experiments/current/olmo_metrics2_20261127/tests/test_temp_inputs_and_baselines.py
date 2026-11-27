import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

X = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(X / "tools"))
import temp_common as tc  # noqa: E402
import temp_make_inputs as mi  # noqa: E402

DAN = tc.DAN
pytestmark = pytest.mark.skipif(
    not (DAN / "olmo_ifladder" / "sample.py").exists(), reason="Dan's clone absent"
)


def dan_parse_spec():
    spec = importlib.util.spec_from_file_location("dan_s", DAN / "olmo_ifladder" / "sample.py")
    m = importlib.util.module_from_spec(spec)
    sys.dont_write_bytecode = True
    spec.loader.exec_module(m)
    return m.parse_spec


def test_input_files_parse_with_dans_parse_spec_and_count(tmp_path):
    mi.main(tmp_path)
    ps = dan_parse_spec()
    c1 = ps("@" + str(tmp_path / "pids_C1_k8.txt"), 2)
    assert len(c1) == 224 and sum(c1.values()) == 736
    reach = set(mi.pids("pids_reach.txt"))
    assert {p for p, n in c1.items() if n == 8} == reach
    assert sum(n for p, n in c1.items() if p not in reach) == 352
    c1_4 = ps("@" + str(tmp_path / "pids_C1_k4.txt"), 2)
    assert sum(c1_4.values()) == 544
    assert sum(ps("@" + str(tmp_path / "pids_reach_k8.txt"), 2).values()) == 384
    assert sum(ps("@" + str(tmp_path / "pids_reach_k4.txt"), 2).values()) == 192
    a2 = ps("@" + str(tmp_path / "pids_armA_n2.txt"), 2)
    assert len(a2) == 218 and set(a2.values()) == {2}
    arm = set(mi.pids("pids_armA.txt"))
    assert arm >= set(c1) - (reach - arm) and len(reach & arm) == 42 and len(reach - arm) == 6
    # every arm-A prompt gets at least 2 draws in C1 (the 42 shared ones from the reach draws)
    assert all(c1[p] >= 2 for p in arm)


def stored_pass1(ck):
    st = tc.stored_arm_a(ck)
    return 100 * np.mean([np.mean([r["strict"] for r in v]) for v in st.values()])


def test_section4_baselines_reproduce():
    assert round(stored_pass1("dpo"), 1) == 29.0
    assert round(stored_pass1("s0500"), 1) == 41.2
    assert round(stored_pass1("s2300"), 1) == 68.3


def test_split_by_draws_1_8_with_baseline_9_16():
    st = tc.stored_arm_a("dpo")
    ever = [p for p, v in st.items() if any(r["strict"] for r in v[:8])]
    never = [p for p in st if p not in set(ever)]
    assert (len(ever), len(never)) == (100, 118)
    assert (
        round(100 * np.mean([np.mean([r["strict"] for r in st[p][8:]]) for p in ever]), 1) == 61.5
    )
    assert (
        round(100 * np.mean([np.mean([r["strict"] for r in st[p][8:]]) for p in never]), 1) == 1.4
    )
    reach = set(tc.dc.reach())
    assert reach & set(tc.dc.arm_a()) <= set(never)


def test_stored_reach_baselines():
    assert (
        round(
            100
            * np.mean(
                [np.mean([r["strict"] for r in v]) for v in tc.stored_reach("s2300").values()]
            ),
            1,
        )
        == 69.4
    )
    assert (
        round(
            100
            * np.mean(
                [np.mean([r["strict"] for r in v]) for v in tc.stored_reach("s0500").values()]
            ),
            1,
        )
        == 4.5
    )
    d = tc.stored_reach("dpo")
    assert len(d) == 48 and all(len(v) == 16 for v in d.values())
    assert sum(r["strict"] for v in d.values() for r in v) == 0


def test_stored_rates_file_is_consistent_with_frozen_detector():
    p = Path("/scratch/cowifornia/nd-rl/olmo_elicit/metrics2/inputs/temp/stored_rates.json")
    if p.exists():
        j = json.load(open(p))
        assert j["detector"] == "temp_repeat_flag v1" and j["dpo"]["n"] == 3488
