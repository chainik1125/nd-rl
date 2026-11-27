"""Tests of steer/steer_trigger.py (grade-based E2 trigger incl. d6) and steer/steer_analyze.py (tables, bootstrap, guard).
Run: pytest -q tests/test_steer_trigger_analyze.py"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "steer"))
import steer_analyze as A  # noqa: E402
import steer_trigger as T  # noqa: E402

REACH = [f"r{i}" for i in range(48)]
HELD = [f"h{i}" for i in range(80)]
DEV = [f"d{i}" for i in range(64)]


def rows(pid_n, passes, n_tok=4000, finish="stop"):
    """pid_n: {pid: draws}; passes: {pid: number of strict passes among them}."""
    out = []
    for p, n in pid_n.items():
        for i in range(n):
            out.append(
                {
                    "pid": p,
                    "chunk": 0,
                    "i": i,
                    "strict": i < passes.get(p, 0),
                    "n_tok": n_tok,
                    "finish": finish,
                    "think_end": True,
                }
            )
    return out


def baseline(rate_held, rate_dev=0.3, long_every=0):
    def lst(rate, n=16):
        k = int(round(rate * n))
        return [
            [i < k, 4000 if not (long_every and i % long_every == 0) else 20000, "stop"]
            for i in range(n)
        ]

    return {"held": {p: lst(rate_held) for p in HELD}, "dev": {p: lst(rate_dev) for p in DEV}}


def write_inputs(tmp_path, base):
    d = tmp_path / "inputs"
    d.mkdir()
    (d / "manifest.json").write_text(
        json.dumps({"sets": {"reach": REACH, "held": HELD, "dev": DEV}})
    )
    (d / "dpo_baseline.json").write_text(json.dumps(base))
    return d


def test_trigger_reach_rule_scaled_by_e0p_draws_d6():
    base = baseline(0.25)["held"]
    held_same = {p: 2 for p in HELD}
    e1 = rows(
        {**{p: 8 for p in REACH}, **held_same}, {"r0": 1, "r1": 1, **{p: 1 for p in HELD[:40]}}
    )  # 2 reach passes; held pass@1 = 0.25
    e0_full = rows({p: 8 for p in REACH}, {})
    t = T.decide(e1, e0_full, base, set(REACH), set(HELD))
    assert (
        t["trigger"]
        and t["reach_hit"]
        and t["e1_reach_pass"] == 2
        and t["e0p_reach_draws"] == 384
        and t["scale_s"] == 1.0
    )
    e0_cut = rows(
        {p: 4 for p in REACH}, {"r5": 1}
    )  # E0' cut to 192 draws with 1 pass: s = 2 -> need >= 4
    assert not T.decide(e1, e0_cut, base, set(REACH), set(HELD))["trigger"]
    e1_more = rows(
        {**{p: 8 for p in REACH}, **held_same}, {"r0": 2, "r1": 2, **{p: 1 for p in HELD[:40]}}
    )
    t2 = T.decide(e1_more, e0_cut, base, set(REACH), set(HELD))
    assert t2["scale_s"] == 2.0 and t2["trigger"] and t2["reach_diff_scaled"] == 2.0


def test_trigger_heldout_rule_both_directions_and_null_case():
    base = baseline(0.25)["held"]
    e0 = rows({p: 8 for p in REACH}, {})

    def e1_with(k):
        return rows({**{p: 8 for p in REACH}, **{p: 2 for p in HELD}}, {p: 1 for p in HELD[:k]})

    # held-out pass@1 = k/160 draws: baseline 0.25 -> 40 passes of 160 draws is a null; +10/160 = +6.25 points triggers
    assert not T.decide(e1_with(40), e0, base, set(REACH), set(HELD))["trigger"]
    up = T.decide(e1_with(50), e0, base, set(REACH), set(HELD))
    assert (
        up["held_hit"] and up["trigger"] and up["held_diff_points"] == pytest.approx(6.25, abs=1e-6)
    )
    down = T.decide(e1_with(30), e0, base, set(REACH), set(HELD))
    assert (
        down["held_hit"] and down["held_diff_points"] < 0
    )  # a loss also runs E2 (needs a control)
    with pytest.raises(ValueError):
        T.decide([], e0, base, set(REACH), set(HELD))


def test_trigger_cli_writes_json_and_prints_flag(tmp_path, capsys):
    d = write_inputs(tmp_path, baseline(0.25))
    for name, rr in (
        ("e1", rows({**{p: 8 for p in REACH}, **{p: 2 for p in HELD}}, {"r3": 3})),
        ("e0p", rows({p: 8 for p in REACH}, {})),
    ):
        (tmp_path / f"{name}.jsonl").write_text("".join(json.dumps(x) + "\n" for x in rr))
    rc = T.main(
        [
            "--e1",
            str(tmp_path / "e1.jsonl"),
            "--e0p",
            str(tmp_path / "e0p.jsonl"),
            "--inputs",
            str(d),
            "--out",
            str(tmp_path / "t.json"),
        ]
    )
    out = capsys.readouterr().out
    assert (
        rc == 0
        and "E2_TRIGGER=1" in out
        and json.loads((tmp_path / "t.json").read_text())["trigger"] is True
    )


def test_analyze_reach_heldout_dev_guard_and_tables(tmp_path):
    base = baseline(
        0.25, rate_dev=0.30, long_every=8
    )  # every 8th stored dev draw is 20k tokens long
    inp = write_inputs(tmp_path, base)
    runs = tmp_path / "runs"

    def cell(name, rr):
        (runs / name).mkdir(parents=True, exist_ok=True)
        (runs / name / "graded.jsonl").write_text("".join(json.dumps(x) + "\n" for x in rr))

    cell("e0p", rows({p: 8 for p in REACH}, {"r0": 1}))
    cell(
        "e1",
        rows(
            {**{p: 8 for p in REACH}, **{p: 2 for p in HELD}},
            {**{p: 2 for p in REACH[:6]}, **{p: 1 for p in HELD[:40]}},
        ),
    )
    for k in (1, 2, 3):
        cell(
            f"e2_s{k}",
            rows(
                {**{p: [3, 3, 2][k - 1] for p in REACH}, **{p: 2 for p in HELD[k - 1 :: 3]}},
                {"r1": 1},
            ),
        )
    cell("dev_guard", rows({p: 2 for p in DEV}, {p: 1 for p in DEV[:40]}, n_tok=5000))
    res = A.analyze(runs, inp, B=500)
    r1, r0 = res["reach"]["e1"], res["reach"]["e0p"]
    assert r1["draws"] == 384 and r1["passes"] == 12 and r0["passes"] == 1
    assert (
        r1["pass1"]["pct"] == pytest.approx(100 * 12 / 384)
        and r1["pass1"]["lo"] <= r1["pass1"]["pct"] <= r1["pass1"]["hi"]
    )
    assert r1["k_common"] == 8 and r1["diff_pass1_vs_e0p"]["pts"] == pytest.approx(
        100 * (12 - 1) / 384
    )
    assert r1["prompts_with_a_pass"] == 6
    assert (
        res["reach"]["e2_pooled"]["draws"] == 384
        and "e2_pooled" in res["held"]
        and res["held"]["e2_pooled"]["draws"] == 160
    )
    h = res["held"]["e1"]
    assert h["pass1"]["pct"] == pytest.approx(100 * 40 / 160) and res["held"]["dpo_stored"][
        "pass1"
    ]["pct"] == pytest.approx(25.0)
    assert h["diff_vs_dpo_stored"]["pts"] == pytest.approx(0.0, abs=1e-9)
    dg = res["dev_guard"]
    # cap-12k scoring on BOTH sides: 5000-token draws all count; stored draws longer than 12k never pass (20k tokens) -> baseline lowered
    # 40 of 64 prompts with 1 of 2 draws passing -> 31.25 %; stored: 5 passes of 16 per prompt, one of them 20k tokens long -> 4/16 = 25 %
    assert dg["pass1"]["pct"] == pytest.approx(31.25) and dg["dpo_stored_pass1"][
        "pct"
    ] == pytest.approx(25.0)
    assert dg["diff"]["pts"] == pytest.approx(6.25, abs=1e-9)
    assert dg["dpo_mean_len_capped"] == pytest.approx(5000.0) and dg[
        "length_ratio"
    ] == pytest.approx(1.0)
    assert dg["guard"]["cap_hits_le_10pct"] and dg["guard"]["pass"] is True
    A.write_tables(res, tmp_path / "out")
    for f in ("analysis.json", "reach.csv", "heldout.csv", "checkpoint_equivalents.csv"):
        assert (tmp_path / "out" / f).exists()
    assert res["checkpoint_equivalents_pct_reach"]["s0500"] == 4.5


def test_dev_guard_fails_on_loops_and_on_a_large_drop(tmp_path):
    base = baseline(0.25, rate_dev=0.30)
    inp = write_inputs(tmp_path, base)
    runs = tmp_path / "runs"
    (runs / "dev_guard").mkdir(parents=True)
    rr = rows(
        {p: 2 for p in DEV}, {}, n_tok=12000, finish="length"
    )  # every draw loops to the cap and fails
    (runs / "dev_guard" / "graded.jsonl").write_text("".join(json.dumps(x) + "\n" for x in rr))
    res = A.analyze(runs, inp, B=200)
    g = res["dev_guard"]["guard"]
    assert (
        not g["cap_hits_le_10pct"]
        and not g["pass1_not_more_than_10pts_below"]
        and not g["length_ratio_0.7_1.3"]
        and not g["pass"]
    )


def test_bootstrap_helpers_are_seeded_and_paired():
    x = np.array([0.0, 0.5, 1.0, 0.25] * 12)
    a = A.boot_mean(x, 300)
    assert a == A.boot_mean(x, 300) and a["lo"] < a["pct"] < a["hi"]
    y = x.copy()
    d = A.boot_diff(x, y, 300)
    assert (
        d["pts"] == 0.0 and d["lo"] == 0.0 and d["hi"] == 0.0
    )  # identical prompts paired -> zero width


def test_figure_is_written_through_ndbench_style(tmp_path):
    res = {
        "reach": {
            "e0p": {"pass1": {"pct": 0.3, "lo": 0.0, "hi": 0.8}},
            "e1": {"pass1": {"pct": 3.0, "lo": 1.0, "hi": 5.0}},
            "e2_pooled": {"pass1": {"pct": 0.5, "lo": 0.0, "hi": 1.2}},
        }
    }
    paths = A.figure(res, tmp_path)
    assert sorted(p.suffix for p in paths) == [".pdf", ".png", ".svg"] and all(
        p.exists() for p in paths
    )
