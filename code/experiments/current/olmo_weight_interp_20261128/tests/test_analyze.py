"""The analysis on synthetic per-draw files: every verdict of the reading rule, cut cells with their
bounds, the half-gain point, answer style, and the figure."""

import json

import analyze as an
import numpy as np
import pytest
from wi_fixtures import by_counts, spec48, uniform, write_cell

SPEC = spec48()


def make_data(tmp_path, cells: dict, **kw):
    data = tmp_path / "data"
    (data / "inputs").mkdir(parents=True)
    (data / "inputs" / "spec_reach4.txt").write_text(
        "\n".join(f"{p}:{n}" for p, n in SPEC.items()) + "\n"
    )
    for tag, passes in cells.items():
        write_cell(data / "runs" / f"cell_{tag}" / "out", tag, passes, **kw.get(tag, {}))
    return data


def run(tmp_path, cells, reps=2000, **kw):
    data = make_data(tmp_path, cells, **kw)
    return an.run(data, data / "analysis", reps), data


LINEAR = {
    "a100": uniform(3),
    "a050": by_counts([2, 1] * 24),
    "a075": uniform(2),
    "a025": by_counts([1, 0] * 24),
}


def test_verdict_about_proportional(tmp_path):
    res, data = run(tmp_path, LINEAR)
    r = res["ratios"]["a050"]["sampled"]
    assert res["cells"]["a100"]["pass1"] == 0.75 and res["cells"]["a050"]["pass1"] == 0.375
    assert r["r"] == pytest.approx(0.5) and 0.2 <= r["ci"][0] <= 0.5 <= r["ci"][1] <= 0.8
    assert res["reading"]["verdict"] == "about proportional" and res["reading"]["cells_complete"]
    assert res["half_gain"]["alpha"] == pytest.approx(0.5) and res["half_gain"]["monotone"]
    txt = (data / "analysis" / "results.txt").read_text()
    assert (
        "Reading (rule fixed in PREREG.md, on the alpha 0.5 share of the alpha 1.0 rate): "
        "about proportional (both cells have every planned draw)." in txt
    )
    # both lines are always printed; without stored rates the second says so
    assert "Smooth reference (PREREG Amendment 1): not evaluated" in txt
    assert res["reference_reading"] == "not evaluated" and "nothing is said about a switch" in txt
    assert "alpha 0.5: 37.5 % [" in txt and "share of the alpha 1.0 rate 50.0 % [" in txt
    man = json.loads((data / "analysis" / "cells_manifest.json").read_text())
    assert set(man) == set(LINEAR) and man["a050"]["rows"] == 192 and man["a050"]["complete"]
    assert (
        man["a050"]["samples"].endswith("cell_a050/out/samples.jsonl.gz")
        and len(man["a050"]["md5"]) == 2
    )


def test_verdict_below_proportional(tmp_path):
    cells = dict(LINEAR, a050=by_counts([1] * 4 + [0] * 44), a075=by_counts([3, 2] * 24))
    res, _ = run(tmp_path, cells)
    r = res["ratios"]["a050"]["sampled"]
    assert r["r"] < 0.35 and r["ci"][1] < 0.5
    assert res["reading"]["verdict"] == "below proportional at the midpoint"
    assert 0.5 < res["half_gain"]["alpha"] < 0.75
    assert res["predictions"]["orchestrator"]["verdict"]["observed"] == res["reading"]["verdict"]


def test_verdict_above_proportional(tmp_path):
    cells = dict(LINEAR, a050=by_counts([3] * 40 + [2] * 8), a025=uniform(2))
    res, _ = run(tmp_path, cells)
    r = res["ratios"]["a050"]["sampled"]
    assert r["r"] > 0.65 and r["ci"][0] > 0.5 and res["reading"]["verdict"] == "above proportional"
    assert res["half_gain"]["alpha"] < 0.25 and res["half_gain"]["below_first_cell"]


def test_verdict_unclear_when_the_interval_is_wide(tmp_path):
    # the yardstick passes on 12 prompts only; alpha 0.5 on 4 of them: r = 1/3 with a wide interval
    cells = dict(LINEAR, a100=by_counts([4] * 12 + [0] * 36), a050=by_counts([4] * 4 + [0] * 44))
    res, _ = run(tmp_path, cells)
    r = res["ratios"]["a050"]["sampled"]
    assert r["r"] == pytest.approx(1 / 3) and r["ci"][1] >= 0.5
    assert res["reading"]["verdict"] == "unclear at this size"
    # r inside 0.35..0.65 but the interval leaves 0.2..0.8
    cells = dict(LINEAR, a100=by_counts([4] * 8 + [0] * 40), a050=by_counts([4] * 4 + [0] * 44))
    res, _ = run(tmp_path / "b", cells)
    r = res["ratios"]["a050"]["sampled"]
    assert r["r"] == pytest.approx(0.5) and not (r["ci"][0] >= 0.2 and r["ci"][1] <= 0.8)
    assert res["reading"]["verdict"] == "unclear at this size"


def test_rule_boundaries():
    v = an.verdict
    assert v(0.5, 0.2, 0.8) == "about proportional" and v(0.35, 0.21, 0.6) == "about proportional"
    assert (
        v(0.5, 0.19, 0.8) == "unclear at this size" and v(0.5, 0.3, 0.81) == "unclear at this size"
    )
    assert v(0.34, 0.1, 0.49) == "below proportional at the midpoint"
    assert v(0.34, 0.1, 0.5) == "unclear at this size"
    assert v(0.66, 0.51, 0.9) == "above proportional"
    assert v(0.66, 0.5, 0.9) == "unclear at this size"
    assert v(0.0, 0.0, 0.0) == "below proportional at the midpoint"
    assert v(float("nan"), 0.1, 0.2) == "no reading"
    seen = {
        v(r, lo, hi)
        for r in np.linspace(0, 1.2, 25)
        for lo in np.linspace(0, r, 7)
        for hi in np.linspace(r, 1.3, 7)
    }
    assert seen == set(an.VERDICTS)  # the four verdicts are reachable and nothing else comes out


def test_cut_cell_is_reported_with_bounds_and_makes_the_reading_unclear(tmp_path):
    p = by_counts([2, 1] * 24)
    for i in range(30):  # 30 prompts keep one of their four draws (the second one)
        p[f"p{i:02d}"] = p[f"p{i:02d}"][1:2]
    res, data = run(tmp_path, dict(LINEAR, a050=p), a050={"truncate": True})
    c = res["cells"]["a050"]
    assert not c["complete"] and c["missing"] == 90 and c["finished"] == 102 and c["planned"] == 192
    assert c["pass1_all_fail"] < c["pass1"] < c["pass1_all_pass"]
    assert c["pass1_all_fail"] == pytest.approx(c["passes"] / 192)
    assert c["pass1_all_pass"] == pytest.approx((c["passes"] + 90) / 192)
    rr = res["ratios"]["a050"]
    assert rr["low"]["r"] < rr["sampled"]["r"] < rr["high"]["r"]
    assert res["reading"]["verdict"] == "unclear at this size (cell cut)"
    assert len(set(res["reading"]["by_bound"].values())) > 1
    txt = (data / "analysis" / "results.txt").read_text()
    assert "INCOMPLETE: 90 of 192 draws have no row (0 prompts have none" in txt
    assert "every missing draw a fail" in txt and "every missing draw a pass" in txt
    assert c["prompts_without_rows"] == 0
    man = json.loads((data / "analysis" / "cells_manifest.json").read_text())
    assert (
        man["a050"]["samples"].endswith("samples_complete.jsonl.gz") and not man["a050"]["complete"]
    )


def test_a_small_cut_keeps_the_verdict_when_all_bounds_agree(tmp_path):
    p = by_counts([1] * 4 + [0] * 44)
    p["p47"] = p["p47"][:3]  # one draw missing
    top = uniform(3)
    top["p00"] = top["p00"][:3]  # the yardstick lost one draw too
    res, _ = run(
        tmp_path, dict(LINEAR, a050=p, a100=top), a050={"truncate": True}, a100={"truncate": True}
    )
    assert res["cells"]["a050"]["missing"] == 1 and res["cells"]["a100"]["missing"] == 1
    assert set(res["reading"]["by_bound"].values()) == {"below proportional at the midpoint"}
    assert res["reading"]["verdict"] == "below proportional at the midpoint"
    assert not res["reading"]["cells_complete"]
    hi = res["ratios"]["a050"]["high"]["r"]  # numerator all-pass over denominator all-fail
    assert hi == pytest.approx((5 / 192) / (144 / 192))  # the yardstick's lost draw was a fail
    lo = res["ratios"]["a050"]["low"]["r"]  # numerator all-fail over denominator all-pass
    assert lo == pytest.approx((4 / 192) / (145 / 192))


def test_no_reading_without_a_yardstick_or_without_alpha_half(tmp_path):
    res, _ = run(tmp_path / "zero", dict(LINEAR, a100=uniform(0)))
    assert res["reading"]["verdict"] == "no reading" and "0 %" in res["reading"]["why"]
    res, _ = run(tmp_path / "absent", {k: v for k, v in LINEAR.items() if k != "a100"})
    assert res["reading"]["verdict"] == "no reading" and res["absent"] == ["a100"]
    res, data = run(tmp_path / "nohalf", {k: v for k, v in LINEAR.items() if k != "a050"})
    assert res["reading"]["verdict"] == "no reading" and "alpha 0.5" in res["reading"]["why"]
    assert "alpha 0.5: not run" in (data / "analysis" / "results.txt").read_text()
    assert res["half_gain"]["cells_used"] == ["a025", "a075", "a100"]


def test_settings_problem_marks_the_cell_incomplete_but_keeps_its_rows(tmp_path):
    res, data = run(tmp_path, LINEAR, a050={"meta": {"temperature": 0.6}})
    c = res["cells"]["a050"]
    assert (
        not c["complete"] and c["missing"] == 0 and any("temperature" in p for p in c["problems"])
    )
    assert not res["reading"]["cells_complete"] and "settings check failed" in res["reading"]["why"]
    assert "a check failed" in (data / "analysis" / "results.txt").read_text()


def test_half_gain_point():
    h = an.half_gain
    assert h({0.25: 0.0, 0.5: 0.0, 0.75: 0.4, 1.0: 0.8})["alpha"] == pytest.approx(0.75)
    assert h({0.25: 0.0, 0.5: 0.2, 0.75: 0.6, 1.0: 0.8})["alpha"] == pytest.approx(0.625)
    a = h({0.25: 0.6, 0.5: 0.7, 0.75: 0.7, 1.0: 0.8})
    assert a["alpha"] == pytest.approx(0.25 * 0.4 / 0.6) and a["below_first_cell"]
    b = h({0.25: 0.5, 0.5: 0.1, 0.75: 0.2, 1.0: 0.8})  # not monotone: the first crossing counts
    assert b["alpha"] == pytest.approx(0.2) and not b["monotone"]
    assert h({0.25: 0.0, 0.5: 0.0, 0.75: 0.0, 1.0: 0.0})["alpha"] is None
    assert h({0.5: 0.1, 1.0: 0.8})["alpha"] == pytest.approx(0.5 + 0.5 * 0.3 / 0.7)


def test_answer_style_and_token_counts(tmp_path):
    cells = dict(LINEAR)
    res, _ = run(
        tmp_path, cells, a050={"answer": "apple banana cherry"}, a100={"answer": "Yes. It is done!"}
    )
    s50, s100 = res["cells"]["a050"]["style"], res["cells"]["a100"]["style"]
    assert s50 == {"passes": 72, "unpunctuated_share": 1.0, "median_words": 3.0}
    assert s100 == {"passes": 144, "unpunctuated_share": 0.0, "median_words": 4.0}
    assert res["cells"]["a050"]["mean_tokens"] == pytest.approx(101.5)
    assert (
        res["cells"]["a050"]["cut_at_cap"] == 0 and res["cells"]["a050"]["no_end_of_thinking"] == 0
    )
    assert an.answer_of("a</think>b</think> c ") == "c" and an.answer_of("no tag") == ""
    assert an.answer_style([]) == {"passes": 0, "unpunctuated_share": None, "median_words": None}


def test_yardstick_refs_writes_and_predictions(tmp_path):
    data = make_data(tmp_path, LINEAR)
    (data / "refs").mkdir()
    stored = dict.fromkeys(SPEC, 0.5)
    (data / "refs" / "refs_reach.json").write_text(
        json.dumps(
            {"per_prompt": {"s2300": stored}, "stored_points": {"dpo_stored": {"rate": 0.0}}}
        )
    )
    w = data / "runs" / "write_a050" / "out"
    w.mkdir(parents=True)
    all_ = {
        "realised_fraction": 0.43,
        "length_ratio": 0.5,
        "off_line_share": 0.1,
        "at_dpo_share_of_differing": 0.3,
    }
    cls = {"matrix": {"realised_fraction": 0.42}, "norm": {"realised_fraction": 0.5}}
    chk = {"tensors_checked": 707, "all": all_, "by_class": cls}
    (w / "check.json").write_text(json.dumps({"ok": True, "check": chk}))
    res = an.run(data, data / "analysis", 500)
    assert "half_gain_realised" not in res  # only one of the three writes is known
    assert res["writes"]["a050"]["realised_fraction_by_class"] == {"matrix": 0.42, "norm": 0.5}
    y = res["yardstick"]
    assert (
        y["same_engine"] == 0.75 and y["stored"] == 0.5 and y["difference"] == pytest.approx(0.25)
    )
    assert not y["flag"] and res["writes"]["a050"]["realised_fraction"] == 0.43
    txt = (data / "analysis" / "results.txt").read_text()
    assert "realised fraction along the line 0.430 (more than 0.05 from the nominal alpha)" in txt
    assert "difference +25.0 points" in txt
    pr = res["predictions"]
    assert (
        pr["coordinator"]["a050"]["inside"] is False
        and pr["orchestrator"]["a100"]["inside"] is True
    )
    assert pr["coordinator"]["a025"]["observed"] == pytest.approx(0.125)
    # Amendment 2: both predicted "below proportional at the midpoint", "compatible with a smooth
    # path"
    for who in ("orchestrator", "coordinator"):
        assert pr[who]["verdict"] == {
            "predicted": "below proportional at the midpoint",
            "observed": "about proportional",
        }
        assert pr[who]["reference"]["predicted"] == "compatible with a smooth path"
        assert pr[who]["reference"]["observed"] == res["reference_reading"]
    assert (
        'verdict "below proportional at the midpoint"; second line "compatible with a smooth' in txt
    )
    # share 0.5 against a reference of about 0.05: above it, and nothing is said about a switch
    assert res["reference_reading"] == "earlier than the smooth reference"
    assert "earlier than a gradual change in each prompt's log-odds would give" in txt
    assert "switch, as in" not in txt
    json.loads((data / "analysis" / "results.json").read_text())  # valid JSON, no NaN tokens needed


def test_figure_is_written_in_three_formats(tmp_path):
    import fig_interp

    p = by_counts([2, 1] * 24)
    for i in range(10):
        p[f"p{i:02d}"] = p[f"p{i:02d}"][:2]
    res, data = run(tmp_path, dict(LINEAR, a050=p), reps=300, a050={"truncate": True})
    paths = fig_interp.draw(
        json.loads((data / "analysis" / "results.json").read_text()), data / "analysis" / "fig"
    )
    assert [q.suffix for q in paths] == [".pdf", ".svg", ".png"] and all(
        q.stat().st_size > 2000 for q in paths
    )
    assert (
        fig_interp.main(
            ["--results", str(data / "analysis" / "results.json"), "--out", str(tmp_path / "f2")]
        )
        == 0
    )


def test_a_prompt_with_no_finished_draw_is_counted_not_dropped(tmp_path):
    p = by_counts([1] * 48)
    for i in range(40, 48):
        p[f"p{i:02d}"] = []  # eight prompts never got a draw back
    res, data = run(tmp_path, dict(LINEAR, a050=p), a050={"truncate": True})
    c = res["cells"]["a050"]
    assert c["prompts_without_rows"] == 8 and c["missing"] == 32 and c["finished"] == 160
    assert c["pass1"] == pytest.approx(0.25)  # as sampled: over the 40 prompts that have rows
    assert c["pass1_all_fail"] == pytest.approx(40 / 192) and c["pass1_all_pass"] == pytest.approx(
        72 / 192
    )
    assert (
        "(8 prompts have none and are outside the as-sampled rate)"
        in (data / "analysis" / "results.txt").read_text()
    )


def test_smooth_reference_and_how_the_observed_share_is_set_against_it():
    stored = np.array(
        [1.0] * 20 + [0.75] * 12 + [0.5] * 10 + [0.0625] * 6
    )  # shaped like the stored s2300 rates
    ref = an.smooth_reference(stored)
    a, b = (ref[k] for k in an.REFERENCE_P0)
    for v in (a, b):
        r = v["rates"]
        assert r["0.25"] < r["0.5"] < r["0.75"] < r["1.0"] and len(v["curve"]["rate"]) == 41
        assert v["curve"]["rate"][0] < 0.003 and v["curve"]["rate"][-1] == pytest.approx(r["1.0"])
        # the audit's first finding: a smooth path lands inside the rule's lowest band
        assert 0.05 < v["r050"] < 0.35 and 0.5 < v["half_gain"] < 0.8
    assert b["r050"] < a["r050"]  # a lower starting rate bends the curve further
    one = an.smooth_reference(np.array([0.5]))[next(iter(an.REFERENCE_P0))]
    ps = 16.5 / 33
    want = 1 / (1 + np.exp(-(0.5 * np.log((1 / 384) / (383 / 384)) + 0.5 * np.log(ps / (1 - ps)))))
    assert one["rates"]["0.5"] == pytest.approx(want)
    assert an.against_reference(0.0, b["r050"] - 0.01, ref) == "sharper than the smooth reference"
    assert an.against_reference(a["r050"] + 0.01, 0.6, ref) == "earlier than the smooth reference"
    assert an.against_reference(0.05, 0.3, ref) == "compatible with a smooth path"
    assert an.against_reference(None, None, ref) == "not evaluated"
    assert an.SMALL_PROVERS["r050"] == pytest.approx(0.0163, abs=1e-3)


def test_reference_sensitivity_and_realised_axis_in_the_report(tmp_path):
    cells = dict(LINEAR, a050=by_counts([1] * 4 + [0] * 44), a075=by_counts([3, 2] * 24))
    data = make_data(tmp_path, cells)
    (data / "refs").mkdir()
    stored = {p: (1.0 if i < 30 else 0.25) for i, p in enumerate(SPEC)}
    (data / "refs" / "refs_reach.json").write_text(json.dumps({"per_prompt": {"s2300": stored}}))
    for tag, real in (("a025", 0.20), ("a050", 0.50), ("a075", 0.80)):
        w = data / "runs" / f"write_{tag}" / "out"
        w.mkdir(parents=True)
        (w / "check.json").write_text(
            json.dumps({"ok": True, "check": {"all": {"realised_fraction": real}}})
        )
    res = an.run(data, data / "analysis", 400)
    assert res["reading"]["verdict"] == "below proportional at the midpoint"
    assert res["reference_reading"] == "sharper than the smooth reference"  # 2.8 % of the yardstick
    assert res["both_lines"] == "Both lines together: a switch, as in the small provers."
    assert set(res["ratios_vs_stored"]) == set(cells)
    assert res["ratios_vs_stored"]["a100"]["r"] == pytest.approx(
        0.75 / np.mean(list(stored.values()))
    )
    h, hr = res["half_gain"], res["half_gain_realised"]
    assert hr["axis"] == {"a025": 0.2, "a050": 0.5, "a075": 0.8, "a100": 1.0}
    # rates 12.5, 2.1, 62.5, 75 %: the first crossing of 37.5 % is between alpha 0.5 and 0.75
    assert h["alpha"] == pytest.approx(0.5 + 0.25 * (0.375 - 4 / 192) / (0.625 - 4 / 192))
    assert hr["alpha"] == pytest.approx(0.5 + 0.30 * (0.375 - 4 / 192) / (0.625 - 4 / 192))
    txt = (data / "analysis" / "results.txt").read_text()
    for part in (
        "Smooth reference (PREREG Amendment 1",
        "Against that reference the observed share is: sharper than the smooth reference. "
        "Both lines together: a switch, as in the small provers. In the small provers",
        "In the small provers the alpha 0.5 share was 1.6 %",
        "Sensitivity (not used by the rule): with the stored s2300 rate",
        "On the realised axis",
        "0.25 -> 0.200",
    ):
        assert part in txt, part


def test_a_share_near_a_band_edge_is_said_to_be_near_the_boundary(tmp_path):
    res, data = run(tmp_path / "far", LINEAR)  # share 0.5
    assert res["reading"]["near_boundary"] is False
    assert "near the boundary" not in (data / "analysis" / "results.txt").read_text()
    # 1 of 4 against 3 of 4: share 0.333, in the lowest band and 0.017 from the edge at 0.35
    res, data = run(tmp_path / "near", dict(LINEAR, a050=uniform(1)))
    assert res["reading"]["verdict"] == "below proportional at the midpoint"
    assert res["reading"]["near_boundary"] is True
    assert "near the boundary of the rule" in (data / "analysis" / "results.txt").read_text()


def test_a_switch_may_be_named_only_with_both_lines():
    assert an.VERDICTS == (
        "about proportional",
        "below proportional at the midpoint",
        "above proportional",
        "unclear at this size",
    )
    name = an.VERDICTS[1]
    second = (an.SHARPER, an.COMPATIBLE, an.EARLIER, an.NOT_EVALUATED)
    said = {(v, r): an.both_lines(v, r) for v in an.VERDICTS for r in second}
    with_switch = [k for k, text in said.items() if an.SWITCH_SENTENCE in text]
    assert with_switch == [(name, an.SHARPER)]  # the only combination that may name a switch
    assert said[(name, an.SHARPER)] == "Both lines together: a switch, as in the small provers."
    for v in an.VERDICTS:
        assert said[(v, an.COMPATIBLE)] == (
            "The curve is what a gradual change in each prompt's log-odds would give."
        )
    assert "no switch is claimed" in said[("unclear at this size", an.SHARPER)]
    assert "earlier than a gradual change" in said[("above proportional", an.EARLIER)]
    assert not any("switch" in v for v in an.VERDICTS)  # no verdict name speaks of a switch


def test_old_verdict_names_are_gone_from_code_tests_and_runbook():
    old = ("late " + "switch", "about " + "linear", '"ear' + 'ly"', "'ear" + "ly'")
    root = an.Path(an.__file__).resolve().parent
    files = [*root.glob("*.py"), *root.glob("tests/*.py"), *root.glob("pod/*"), root / "RUNBOOK.md"]
    assert len(files) > 15
    for f in files:
        text = f.read_text()
        assert not [o for o in old if o in text], f.name
