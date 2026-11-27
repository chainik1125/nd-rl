import json

import numpy as np
import rank1_fixtures  # noqa: F401  (puts rank1/ on sys.path)

# isort: split
import analyze as an  # noqa: E402
import lora_gate as lg  # noqa: E402
import rules  # noqa: E402
import stats  # noqa: E402
from common import collapse_stats, missing_bound, prompt_rates  # noqa: E402


def mk_rows(pids, n, rate, rng, finish="stop", think_end=True, drop=0):
    rows = []
    for p in pids:
        for i in range(n):
            rows.append(
                {
                    "pid": p,
                    "i": i,
                    "strict": bool(rng.random() < rate),
                    "finish": finish,
                    "think_end": think_end,
                    "n_tok": 100,
                }
            )
    return rows[: len(rows) - drop] if drop else rows


def refs_for(pids_by_set, rates, rng):
    refs = {}
    for ck, rt in rates.items():
        refs[ck] = {
            t: {p: float(np.clip(rt[t] + 0.1 * rng.standard_normal(), 0, 1)) for p in ps}
            for t, ps in pids_by_set.items()
        }
    return refs


def setup(tmp_path, f_true_h, f_true_r, seed=0):
    rng = np.random.default_rng(seed)
    D = [f"d{i}" for i in range(64)]
    H = [f"h{i}" for i in range(112)]
    R = [f"r{i}" for i in range(48)]
    pids = {"D": D, "H": H, "R": R}
    rates = {
        "dpo": {"D": 0.4, "H": 0.32, "R": 0.0},
        "s0500": {"D": 0.6, "H": 0.44, "R": 0.05},
        "s2300": {"D": 0.78, "H": 0.65, "R": 0.69},
    }
    refs = refs_for(pids, rates, rng)
    inp = tmp_path / "inp"
    inp.mkdir()
    (inp / "refs.json").write_text(json.dumps(refs))
    for t, ps in pids.items():
        (inp / f"pids_{t}.txt").write_text("\n".join(ps) + "\n")
    h_rate = 0.32 + f_true_h * 0.33
    r_rate = f_true_r * 0.69
    rows = mk_rows(H, 2, h_rate, rng) + mk_rows(R, 4, r_rate, rng)
    gp = tmp_path / "g.jsonl"
    gp.write_text("\n".join(json.dumps(r) for r in rows))
    return pids, refs, inp, gp, rows


def test_collapse_union_and_killed_draws(tmp_path):
    rng = np.random.default_rng(0)
    ps = [f"p{i}" for i in range(10)]
    rows = mk_rows(ps, 10, 0.5, rng)
    for r in rows[:2]:
        r["finish"] = "length"
        r["think_end"] = False  # capped AND no end-of-thinking: counted once (union)
    cs = collapse_stats(rows, 100)
    assert (
        cs["bad_union"] == 2
        and cs["capped"] == 2
        and cs["no_end"] == 2
        and cs["collapsed"] is False
    )
    cs2 = collapse_stats(
        rows[:90], 100
    )  # the last prompt's 10 draws were killed in flight: they count as bad
    assert cs2["missing"] == 10 and cs2["bad_union"] == 12 and cs2["collapsed"] is True
    rate = prompt_rates(rows[:90], ps, 10)
    bound = missing_bound(rows[:90], ps, 10)
    assert rate["p9"] == 0.0 and bound["p9"] == 1.0 and all(bound[p] >= rate[p] for p in ps)


def test_recovered_fraction_known_value(tmp_path):
    pids, refs, inp, gp, rows = setup(tmp_path, 1.0, 1.0)
    cell = prompt_rates(rows, pids["H"], 2)
    f = stats.recovered_fraction(cell, refs["dpo"]["H"], refs["s2300"]["H"], 500, 0)
    assert 0.6 < f["f"] < 1.5 and f["f_lo"] < f["f"] < f["f_hi"] and f["n_prompts"] == 112
    d = stats.pass_diff(refs["dpo"]["H"], refs["dpo"]["H"], 200, 0)
    assert d["diff"] == 0 and d["lo"] == 0 == d["hi"]


def test_release_rule_clean_null_and_success(tmp_path):
    pids, refs, inp, gp, rows = setup(tmp_path, 0.0, 0.0)
    res = rules.release(rows, refs, pids["H"], pids["R"], 2, 4, reps=300)
    assert res["clean_null"] and "SKIP" in res["decision"]
    assert (
        rules.main(
            [
                "release",
                "--a1",
                str(gp),
                "--refs",
                str(inp / "refs.json"),
                "--h",
                str(inp / "pids_H.txt"),
                "--r",
                str(inp / "pids_R.txt"),
            ]
        )
        == 10
    )
    pids, refs, inp, gp, rows = (
        setup(tmp_path / "s", 0.9, 0.9) if (tmp_path / "s").mkdir() is None else None
    )
    assert (
        rules.main(
            [
                "release",
                "--a1",
                str(gp),
                "--refs",
                str(inp / "refs.json"),
                "--h",
                str(inp / "pids_H.txt"),
                "--r",
                str(inp / "pids_R.txt"),
            ]
        )
        == 0
    )


def test_release_rule_collapse_runs_a2_and_a1u(tmp_path):
    pids, refs, inp, gp, rows = setup(tmp_path, 0.0, 0.0)
    res = rules.release(
        rows[:300], refs, pids["H"], pids["R"], 2, 4, reps=200
    )  # most draws killed in flight
    assert (
        res["collapse"]["collapsed"]
        and not res["clean_null"]
        and res["decision"] == "RUN A2 then A1u"
    )


def test_gamma_choice_rule(tmp_path):
    rng = np.random.default_rng(3)
    D = [f"d{i}" for i in range(64)]
    r35 = mk_rows(D, 2, 0.5, rng)
    r46_same = mk_rows(D, 2, 0.5, rng)
    r46_big = mk_rows(D, 2, 0.9, rng)
    assert (
        rules.gamma_choice(r46_same, r35, D, 2, 300)["chosen_gamma"] == 3.5
    )  # no lead beyond one SE
    big = rules.gamma_choice(r46_big, r35, D, 2, 300)
    assert big["chosen_gamma"] == 4.6 and big["lead_46_minus_35"] > big["paired_se"]
    collapsed = rules.gamma_choice(
        mk_rows(D, 2, 0.9, rng, finish="length", think_end=False), r35, D, 2, 300
    )
    assert collapsed["chosen_gamma"] == 3.5 and collapsed["collapse46"]["collapsed"]
    out = tmp_path / "c.json"
    g46, g35 = tmp_path / "46.jsonl", tmp_path / "35.jsonl"
    g46.write_text("\n".join(json.dumps(r) for r in r46_big))
    g35.write_text("\n".join(json.dumps(r) for r in r35))
    (tmp_path / "d.txt").write_text("\n".join(D))
    code = rules.main(
        [
            "gamma",
            "--g46",
            str(g46),
            "--g35",
            str(g35),
            "--d",
            str(tmp_path / "d.txt"),
            "--out",
            str(out),
        ]
    )
    assert (
        code == 1
        and json.loads(out.read_text())["chosen_gamma"] == 4.6
        and "utc" in json.loads(out.read_text())
    )


def test_gate_verdict_functions():
    rng = np.random.default_rng(0)
    base = [rng.uniform(0.5, 3, 200) for _ in range(2)]
    delta = [rng.normal(0.4, 0.3, 200) for _ in range(2)]
    r16 = [b + d for b, d in zip(base, delta)]
    r1_ok = [b + d * 1.001 for b, d in zip(base, delta)]
    assert lg.powered(base, r16)
    v = lg.consistency_verdict(base, base, r1_ok, r16)
    assert v["pass"]
    r1_bad = [b + d * 0.8 for b, d in zip(base, delta)]  # wrong scale
    assert not lg.consistency_verdict(base, base, r1_bad, r16)["pass"]
    r1_noise = [b + rng.normal(0.4, 0.3, 200) for b in base]  # uncorrelated change
    assert not lg.consistency_verdict(base, base, r1_noise, r16)["pass"]
    tiny_change = [b + 1e-4 for b in base]
    assert not lg.powered(base, tiny_change)
    assert lg.applied_verdict(base, base, r16, r16)["pass"]
    assert not lg.applied_verdict(base, base, base, base)[
        "pass"
    ]  # adapter not applied: zero change
    assert not lg.applied_verdict(base, base, r16, [x + 0.01 for x in r16])[
        "pass"
    ]  # not deterministic
    # Amendment 3: small scattered LoRA-path noise (sum 0.01 % of base) repeats; a 0.57 % shift above does not
    noisy = [x + rng.normal(0, 0.003, x.size) for x in r16]
    v = lg.applied_verdict(base, base, r16, noisy)
    assert v["pass"] and not v["examples"][0]["repeats_exact"]
    # an adapter whose summed change is within noise but which changes most tokens (A2-like) counts as changed
    a2 = [b + rng.normal(0, 0.003, b.size) for b in base]
    a2b = [b + rng.normal(0, 0.003, b.size) for b in base]
    assert lg.applied_verdict(base, base, a2, a2b)["pass"]
    # ... but not when base itself is not exactly repeatable
    assert not lg.applied_verdict(base, [x + 1e-3 for x in base], a2, a2b)["pass"]
    # Amendment 3a: a second call that dropped the adapter (equal to base) fails even for a weak adapter
    assert not lg.applied_verdict(base, base, a2, [x.copy() for x in base])["pass"]


def test_analyze_end_to_end(tmp_path):
    pids, refs, inp, gp, rows = setup(tmp_path, 0.8, 0.8)
    cells = {
        "cells": [
            {"name": "A1", "graded": str(gp), "n": {"H": 2, "R": 4}, "kind": "adapter"},
            {"name": "X_held", "graded": str(gp), "n": {"H": 2, "R": 4}, "kind": "expo"},
        ]
    }
    (tmp_path / "cells.json").write_text(json.dumps(cells))
    out = tmp_path / "out"
    assert (
        an.main(
            [
                "--cells",
                str(tmp_path / "cells.json"),
                "--inputs",
                str(inp),
                "--out",
                str(out),
                "--reps",
                "300",
            ]
        )
        == 0
    )
    res = json.loads((out / "results.json").read_text())
    c = res["cells"]["A1"]["sets"]["H"]
    assert "f" in c and "vs_dpo" in c and "vs_s2300" in c and "f_prime_vs_s0500_gap" not in c
    assert "f_prime_vs_s0500_gap" in res["cells"]["X_held"]["sets"]["R"]
    assert (out / "results.csv").exists()
    for ext in ("pdf", "svg", "png"):
        assert (out / f"rank1_recovered_fraction.{ext}").exists()
