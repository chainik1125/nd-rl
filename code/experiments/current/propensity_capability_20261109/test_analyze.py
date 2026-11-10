"""Tests of analyze.py (and a smoke test of fig.py) on synthetic result files in SPEC.md's schema. CPU only.

From the repository root:

    code/.venv/bin/python -m pytest -c code/pyproject.toml code/experiments/current/propensity_capability_20261109/test_analyze.py
    code/.venv/bin/python code/experiments/current/propensity_capability_20261109/test_analyze.py DIR   # writes a demo set

Every case is built so its outcome is known before the analysis runs: one per reading (elicitation, candidate
capability gain, mix, neither, unresolved), one per router verdict (router, default shift, unresolved, no rise to
locate, and the fallback when helpful / idle are null), a direction that is never applicable, a sampled propensity of
exactly 0 with a finite exact probability, a missing checkpoint, and the hard schema errors.
"""

# ruff: noqa: E501, B905
from __future__ import annotations

import json
import math
import sys
import zlib
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyze as A  # noqa: E402

K, KF = 32, 16
DIRS_ALL = ("reductio", "negation", "conditional", "cases")


# ----------------------------------------------------------------------------- the synthetic generator
def _u(*parts) -> float:
    """A deterministic number in [0, 1) from its parts."""
    return zlib.crc32("|".join(map(str, parts)).encode()) / 2**32


def _rng(*parts) -> np.random.Generator:
    return np.random.default_rng(zlib.crc32("|".join(map(str, parts)).encode()))


def theorem_rows(
    s: str, name: str, app: dict, spec: dict, cls: dict | None = None, spread: float = 0.5
) -> list[dict]:
    """Rows of one theorem at one checkpoint. spec: d -> {"P": mean first-move probability, "C": forced solve rate,
    "use": rate of use anywhere (default P)}, or exact per-theorem overrides "p", "c" (solved of KF), "first", "used",
    "capped", "refused"."""
    meta = {
        "set": s,
        "name": name,
        "n_lines": 3 + int(10 * _u(name, "n")),
        "schema": "sA" if _u(name, "s") < 0.5 else "sB",
        "source": "syn",
    }
    rows, firsts = [], 0
    for d in DIRS_ALL:
        base = {
            **meta,
            "direction": d,
            "applicable": bool(app.get(d)),
            "dir_class": (cls or {}).get(d),
            "tokens": None,
            "x": None,
        }
        if not app.get(d):
            rows.append(
                {
                    **base,
                    "arm": "U",
                    "k": K,
                    "solved": 0,
                    "used": 0,
                    "first": 0,
                    "capped": None,
                    "refused": None,
                    "logp_open": None,
                }
            )
            rows.append(
                {
                    **base,
                    "arm": "F",
                    "k": 0,
                    "solved": 0,
                    "used": None,
                    "first": None,
                    "capped": None,
                    "refused": None,
                    "logp_open": None,
                }
            )
            rows.append(
                {
                    **base,
                    "arm": "E",
                    "k": 0,
                    "solved": None,
                    "used": None,
                    "first": None,
                    "capped": None,
                    "refused": None,
                    "logp_open": None,
                }
            )
            continue
        sp = spec[d]
        p = (
            sp["p"]
            if "p" in sp
            else min(sp["P"] * (1 - spread + 2 * spread * _u(name, d, "p")), 0.999)
        )
        c = min(sp.get("C", 0.0) * (1 - spread + 2 * spread * _u(name, d, "c")), 1.0)
        solved_f = sp["c"] if "c" in sp else int(_rng(name, d, "C", sp.get("C")).binomial(KF, c))
        cap = min(sp.get("capped", 0), KF - solved_f)
        ref = min(sp.get("refused", 0), KF - solved_f - cap)
        first = sp["first"] if "first" in sp else int(_rng(name, d, "first", p).binomial(K, p))
        use = sp.get("use", sp.get("P", p))
        used = (
            sp["used"]
            if "used" in sp
            else max(first, int(_rng(name, d, "use", use).binomial(K, min(use, 1.0))))
        )
        u_solved = int(_rng(name, d, "us", used, c).binomial(used, c))
        firsts += first
        rows.append(
            {
                **base,
                "arm": "U",
                "k": K,
                "solved": u_solved,
                "used": used,
                "first": first,
                "capped": None,
                "refused": None,
                "logp_open": None,
            }
        )
        rows.append(
            {
                **base,
                "arm": "F",
                "k": KF,
                "solved": solved_f,
                "used": None,
                "first": None,
                "capped": cap,
                "refused": ref,
                "logp_open": None,
            }
        )
        rows.append({**base, "arm": "E", "k": 0, "solved": None, "used": None, "first": None, "capped": None, "refused": None,
                     "logp_open": math.log(p) if p > 0 else -1e300})  # fmt: skip
    any_solved = int(
        _rng(name, "any", json.dumps(spec, sort_keys=True, default=str)).binomial(K, 0.25)
    )
    rows.append({**meta, "arm": "U", "direction": "any", "applicable": True, "k": K, "solved": any_solved, "used": None, "first": None,
                 "capped": 1, "refused": 2, "logp_open": None, "dir_class": None, "tokens": 1000})  # fmt: skip
    rows.append({**meta, "arm": "U", "direction": "direct", "applicable": True, "k": K, "solved": 0, "used": None, "first": max(0, K - firsts),
                 "capped": None, "refused": None, "logp_open": None, "dir_class": None, "tokens": None})  # fmt: skip
    for r in rows:
        r.pop("x", None)
    return rows


def write_ckpt(
    d: Path, ckpt: str, dose: str, seed: int, stage: str, x: float | None, rows: list[dict]
) -> None:
    d.mkdir(parents=True, exist_ok=True)
    with open(d / f"{ckpt}.jsonl", "w") as f:
        for r in rows:
            f.write(
                json.dumps(
                    {"ckpt": ckpt, "dose": dose, "seed": seed, "stage": stage, "x_dan": x, **r}
                )
                + "\n"
            )


def ckpt_rows(theorems: list[tuple[str, str, dict, dict | None]], spec_for) -> list[dict]:
    """theorems: (set, name, applicable, classes); spec_for(set, name, index) -> spec."""
    out = []
    for i, (s, n, app, cls) in enumerate(theorems):
        out += theorem_rows(s, n, app, spec_for(s, n, i), cls)
    return out


def thms(
    n: int,
    s: str = "strict106",
    app: dict | None = None,
    cls: dict | None = None,
    prefix: str = "t",
) -> list:
    app = app or {"reductio": True}
    return [(s, f"{prefix}{s[:2]}{i:03d}", app, cls) for i in range(n)]


def one_line(tmp: Path, spec_b: dict, spec_e: dict, n: int = 60, dose: str = "1", seed: int = 0, x: float = 0.5,
             theorems: list | None = None) -> dict:  # fmt: skip
    """A base and an RL end on the same theorems; returns the analysis."""
    ts = theorems or thms(n)
    write_ckpt(
        tmp, f"b{dose}_{seed}", dose, seed, "base", 0.0, ckpt_rows(ts, lambda s, nm, i: spec_b)
    )
    write_ckpt(
        tmp, f"e{dose}_{seed}", dose, seed, "rl_end", x, ckpt_rows(ts, lambda s, nm, i: spec_e)
    )
    return A.analyse(tmp)


def cellof(res: dict, dose="1", seed=0, s="strict106", d="reductio") -> dict:
    return res["lines"][f"{dose}|{seed}"]["cells"][f"{s}|{d}"]["primary"]


# ----------------------------------------------------------------------------- rules
def test_state_rule_boundaries():
    assert A.state(0.20, 0.05, 0.35, 0.10, 0.16) == "up"
    assert (
        A.state(0.14, 0.01, 0.27, 0.10, 0.14) == "flat"
    )  # real but below DELTA, interval inside ±FLAT
    assert (
        A.state(0.20, -0.05, 0.45, 0.10, 0.16) == "unresolved"
    )  # interval holds 0 and passes FLAT
    assert A.state(-0.20, -0.35, -0.05, 0.10, 0.06) == "down"
    assert A.state(2.0, 1.5, 2.5, 1e-6, 1e-4) == "rare"  # x100, but below 1 % at both checkpoints
    assert A.state(0.0, -0.29, 0.29, 0.5, 0.5) == "flat"
    assert A.state(0.0, -0.31, 0.29, 0.5, 0.5) == "unresolved"


def test_reading_table():
    assert A.reading("up", "flat") == "elicitation"
    assert A.reading("up", "up") == "mix"
    assert A.reading("flat", "up") == "candidate capability gain"
    assert A.reading("down", "up") == "candidate capability gain"
    assert A.reading("rare", "up") == "candidate capability gain"
    assert A.reading("flat", "flat") == "neither"
    assert A.reading("rare", "down") == "neither"
    assert A.reading("up", "rare") == "other (outside the table)"
    assert A.reading("up", "down") == "other (outside the table)"
    assert A.reading("up", "unresolved") == "unresolved"
    assert A.reading("unresolved", "flat") == "unresolved"


def test_dose_reading_rule():
    assert A.dose_reading(["mix", "mix", "mix"]) == "mix"
    assert A.dose_reading(["mix", "mix", "unresolved"]) == "mix (2 of 3)"
    assert A.dose_reading(["mix", "mix", "no reading: E disagrees with U"]) == "mix (2 of 3)"
    assert A.dose_reading(["mix", "mix", "elicitation"]) == "seeds disagree"
    assert A.dose_reading(["mix", "elicitation", "unresolved"]) == "seeds disagree"
    assert (
        A.dose_reading(["mix", "mix", None]) == "mix"
    )  # a missing checkpoint: both read seeds agree
    assert A.dose_reading(["mix", None, None]) == "too few seeds"
    assert A.dose_reading(["router", "router", "unresolved"], A.ROUTES) == "router (2 of 3)"
    assert (
        A.dose_reading(["unresolved"] * 3) == "unresolved"
    )  # the same non-reading on every seed is reported as such
    assert A.dose_reading(["no rise to locate"] * 3, A.ROUTES) == "no rise to locate"


# ----------------------------------------------------------------------------- one case per reading
def test_elicitation(tmp_path):
    res = one_line(
        tmp_path, {"reductio": {"P": 0.02, "C": 0.3}}, {"reductio": {"P": 0.2, "C": 0.3}}
    )
    c = cellof(res)
    assert c["P"]["state"] == "up" and c["C"]["state"] == "flat"
    assert c["C"]["dlog10"] == 0.0  # same counts at both checkpoints
    assert c["reading"] == "elicitation"


def test_candidate_capability_gain(tmp_path):
    res = one_line(
        tmp_path, {"reductio": {"P": 1e-6, "C": 0.10}}, {"reductio": {"P": 1e-7, "C": 0.35}}
    )
    c = cellof(res)
    assert c["P"]["state"] == "rare" and c["C"]["state"] == "up"
    assert c["reading"] == "candidate capability gain"


def test_mix_and_its_size(tmp_path):
    res = one_line(
        tmp_path, {"reductio": {"P": 0.02, "C": 0.10}}, {"reductio": {"P": 0.2, "C": 0.35}}
    )
    c = cellof(res)
    assert c["reading"] == "mix"
    sz = c["size"]
    assert (
        abs(sz["dPC_from_P"] + sz["dPC_from_C"] - sz["dPC"]) < 1e-12
    )  # the Shapley split is exact
    assert 0.5 < sz["share_P"] < 1.0  # ×10 in P against ×3.5 in C


def test_neither(tmp_path):
    res = one_line(tmp_path, {"reductio": {"P": 0.1, "C": 0.3}}, {"reductio": {"P": 0.1, "C": 0.3}})
    c = cellof(res)
    assert (c["P"]["state"], c["C"]["state"], c["reading"]) == ("flat", "flat", "neither")


def test_unresolved(tmp_path):
    """Capability: half the theorems go 0 -> 16 of 16, half 16 -> 0, so the interval is wide around 0."""
    ts = thms(24)
    b = {n: {"reductio": {"P": 0.1, "c": 16 if i % 2 else 0}} for i, (_, n, _, _) in enumerate(ts)}
    e = {n: {"reductio": {"P": 0.1, "c": 0 if i % 2 else 16}} for i, (_, n, _, _) in enumerate(ts)}
    e[ts[0][1]]["reductio"]["c"] = 12  # a little net change
    write_ckpt(tmp_path, "b", "1", 0, "base", 0.0, ckpt_rows(ts, lambda s, nm, i: b[nm]))
    write_ckpt(tmp_path, "e", "1", 0, "rl_end", 0.5, ckpt_rows(ts, lambda s, nm, i: e[nm]))
    c = cellof(A.analyse(tmp_path))
    assert c["C"]["state"] == "unresolved" and c["reading"] == "unresolved"


# ----------------------------------------------------------------------------- the awkward cases
def test_never_applicable_direction(tmp_path):
    res = one_line(tmp_path, {"reductio": {"P": 0.1, "C": 0.3}}, {"reductio": {"P": 0.1, "C": 0.3}})
    c = cellof(res, d="cases")
    assert c["n_applicable"] == 0 and c["reading"] == "no reading: never applicable"
    assert res["doses"]["1|strict106|cases"]["reading"] == "never applicable"
    lv = res["levels"]["b1_0"]["strict106"]["cases"]
    assert lv["n"] == 0 and "P_first" not in lv
    assert "never applicable" in res["lines"]["1|0"]["router"]["cases"]["verdict"]
    A.tables(res)  # renders without the missing pieces


def test_zero_sampled_propensity_with_finite_exact(tmp_path):
    """U never opens with the reductio in 32 attempts, E says 1e-7: the analysis uses E, and 0 of k is not 0."""
    spec = {"reductio": {"p": 1e-7, "first": 0, "used": 0, "C": 0.3}}
    res = one_line(tmp_path, spec, {"reductio": {"p": 1e-7, "first": 0, "used": 0, "C": 0.6}})
    lv = res["levels"]["b1_0"]["strict106"]["reductio"]
    assert lv["P_first_sampled"]["est"] == 0.0
    assert lv["P_first"]["est"] == pytest.approx(1e-7)
    assert lv["P_first"]["ci"][0] > 0
    assert not lv["E_vs_U"]["fail"]
    c = cellof(res)
    assert math.isfinite(c["P"]["dlog10"]) and c["P"]["state"] == "rare"
    assert c["reading"] == "candidate capability gain"
    sec = res["lines"]["1|0"]["cells"]["strict106|reductio"]["uses_anywhere"]
    assert (
        sec["P"]["base"]["est"] == 0.0
        and math.isfinite(sec["P"]["dlog10"])
        and sec["P"]["state"] == "rare"
    )
    assert "1.0e-05" in A.tables(res)  # 1e-7 as a percentage, not "0.0"


def test_missing_checkpoint(tmp_path):
    ts = thms(40)
    for seed in (0, 1, 2):
        write_ckpt(
            tmp_path,
            f"b{seed}",
            "1",
            seed,
            "base",
            0.0,
            ckpt_rows(ts, lambda s, n, i: {"reductio": {"P": 0.02, "C": 0.3}}),
        )
        if seed != 2:  # seed 2's RL end is missing
            write_ckpt(
                tmp_path,
                f"e{seed}",
                "1",
                seed,
                "rl_end",
                0.5,
                ckpt_rows(ts, lambda s, n, i: {"reductio": {"P": 0.2, "C": 0.3}}),
            )
    res = A.analyse(tmp_path)
    assert res["lines"]["1|2"]["missing"] == "rl_end"
    dr = res["doses"]["1|strict106|reductio"]
    assert dr["readings"] == ["elicitation", "elicitation", None]
    assert dr["reading"] == "elicitation"
    assert "missing rl_end" in A.tables(res)


def test_E_disagrees_with_U(tmp_path):
    res = one_line(
        tmp_path,
        {"reductio": {"p": 1e-4, "first": 20, "used": 20, "C": 0.3}},
        {"reductio": {"P": 0.2, "C": 0.3}},
    )
    assert res["levels"]["b1_0"]["strict106"]["reductio"]["E_vs_U"]["fail"]
    assert cellof(res)["reading"] == "no reading: E disagrees with U"
    assert res["informative"]["c_E_fails_in_over_a_quarter"]


def test_forced_attempts_mostly_refused(tmp_path):
    res = one_line(
        tmp_path,
        {"reductio": {"P": 0.02, "C": 0.1, "refused": 12}},
        {"reductio": {"P": 0.2, "C": 0.3}},
    )
    assert cellof(res)["reading"] == "no reading: forced attempts mostly refused or capped"


def test_too_few_applicable(tmp_path):
    ts = thms(30, app={"reductio": True}) + thms(
        8, app={"reductio": True, "negation": True}, prefix="neg"
    )
    sp = {"reductio": {"P": 0.1, "C": 0.3}, "negation": {"P": 0.5, "C": 0.5}}
    write_ckpt(tmp_path, "b", "1", 0, "base", 0.0, ckpt_rows(ts, lambda s, n, i: sp))
    write_ckpt(tmp_path, "e", "1", 0, "rl_end", 0.5, ckpt_rows(ts, lambda s, n, i: sp))
    c = cellof(A.analyse(tmp_path), d="negation")
    assert c["n"] == 8 and c["reading"] == "no reading: n = 8 < 20"


def test_largest_x_end_is_primary(tmp_path):
    ts = thms(30)
    write_ckpt(
        tmp_path,
        "b",
        "0",
        0,
        "base",
        0.0,
        ckpt_rows(ts, lambda s, n, i: {"reductio": {"P": 0.1, "C": 0.3}}),
    )
    write_ckpt(
        tmp_path,
        "e_it500",
        "0",
        0,
        "rl_end",
        3.97,
        ckpt_rows(ts, lambda s, n, i: {"reductio": {"P": 0.1, "C": 0.3}}),
    )
    write_ckpt(
        tmp_path,
        "e_it580",
        "0",
        0,
        "rl_end",
        5.07,
        ckpt_rows(ts, lambda s, n, i: {"reductio": {"P": 0.1, "C": 0.6}}),
    )
    res = A.analyse(tmp_path)
    assert res["lines"]["0|0"]["end"] == "e_it580"
    assert res["matched_and_secondary"]["e_it500"]["kind"] == "secondary end"
    assert res["lines"]["0|0"]["x_end"] == 5.07


def test_schema_errors(tmp_path):
    ts = thms(25)
    rows = ckpt_rows(ts, lambda s, n, i: {"reductio": {"P": 0.1, "C": 0.3}})
    write_ckpt(tmp_path, "b", "1", 0, "base", 0.0, rows)
    bad = [dict(r) for r in rows]
    for r in bad:  # the E arm says the reductio is not applicable on one theorem, U and F say it is
        if r["name"] == ts[0][1] and r["arm"] == "E" and r["direction"] == "reductio":
            r["applicable"] = False
    write_ckpt(tmp_path, "e", "1", 0, "rl_end", 0.5, bad)
    with pytest.raises(A.SchemaError, match="applicability"):
        A.analyse(tmp_path)
    write_ckpt(tmp_path, "e", "1", 0, "rl_end", 0.5, rows + rows[:1])
    with pytest.raises(A.SchemaError, match="duplicate"):
        A.analyse(tmp_path)
    over = [dict(r) for r in rows]
    over[0]["solved"] = 99
    write_ckpt(tmp_path, "e", "1", 0, "rl_end", 0.5, over)
    with pytest.raises(A.SchemaError):
        A.analyse(tmp_path)


def test_bootstrap_is_deterministic_and_brackets_the_estimate(tmp_path):
    spec_b, spec_e = {"reductio": {"P": 0.02, "C": 0.10}}, {"reductio": {"P": 0.2, "C": 0.35}}
    r1 = one_line(tmp_path, spec_b, spec_e)
    r2 = A.analyse(tmp_path)
    c1, c2 = cellof(r1), cellof(r2)
    assert c1["P"]["ci"] == c2["P"]["ci"] and c1["C"]["ci"] == c2["C"]["ci"]
    for f in ("P", "C"):
        lo, hi = c1[f]["ci"]
        assert lo <= c1[f]["dlog10"] <= hi


# ----------------------------------------------------------------------------- router or default shift
def router_case(
    tmp: Path,
    req: tuple,
    idle: tuple,
    helpful: tuple | None = None,
    labelled: bool = True,
    idle_p_end=None,
) -> dict:
    """reductio: 30 required theorems (the 106) and 40 optional ones (transfer_other): 20 helpful, 20 idle, or all
    unlabelled (labelled=False). Each tuple: (P at base, P at end)."""
    req_t = thms(30, cls={"reductio": "required"})
    hl = [
        (
            "transfer_other",
            f"h{i:03d}",
            {"reductio": True},
            {"reductio": "helpful" if labelled else None},
        )
        for i in range(20)
    ]
    il = [
        (
            "transfer_other",
            f"i{i:03d}",
            {"reductio": True},
            {"reductio": "idle" if labelled else None},
        )
        for i in range(20)
    ]
    helpful = helpful or idle

    def spec(stage):
        def f(s, n, i):
            j = 0 if stage == "b" else 1
            if s == "strict106":
                return {"reductio": {"P": req[j], "C": 0.3}}
            if n.startswith("h"):
                return {"reductio": {"P": helpful[j], "C": 0.3}}
            if stage == "e" and idle_p_end is not None:
                return {"reductio": {"p": idle_p_end(i - 50), "C": 0.3}}
            return {"reductio": {"P": idle[j], "C": 0.3}}

        return f

    ts = req_t + hl + il
    write_ckpt(tmp, "b", "1", 0, "base", 0.0, ckpt_rows(ts, spec("b")))
    write_ckpt(tmp, "e", "1", 0, "rl_end", 0.5, ckpt_rows(ts, spec("e")))
    return A.analyse(tmp)["lines"]["1|0"]["router"]["reductio"]


def test_router(tmp_path):
    r = router_case(tmp_path, req=(0.01, 0.30), idle=(0.05, 0.05))
    assert r["mode"] == "classes" and r["classes"]["required"]["state"] == "up"
    assert r["classes"]["idle"]["state"] == "flat" and r["verdict"] == "router"
    assert abs(r["spillover_f"]) < 0.1


def test_default_shift(tmp_path):
    r = router_case(tmp_path, req=(0.01, 0.30), idle=(0.01, 0.30))
    assert r["classes"]["idle"]["state"] == "up" and r["verdict"] == "default shift"
    assert 0.8 < r["spillover_f"] < 1.25


def test_router_unresolved(tmp_path):
    """The idle class rises through one theorem only, so its interval spans both sides of the split."""
    r = router_case(
        tmp_path, req=(0.01, 0.30), idle=(0.02, 0.02), idle_p_end=lambda j: 0.9 if j == 0 else 0.02
    )
    assert r["classes"]["idle"]["state"] == "unresolved"
    lo, hi = r["d_minus_half_ci"]
    assert lo < 0 < hi and r["verdict"] == "unresolved"


def test_no_rise_to_locate(tmp_path):
    r = router_case(tmp_path, req=(0.05, 0.05), idle=(0.01, 0.30))
    assert r["classes"]["required"]["state"] == "flat" and r["verdict"] == "no rise to locate"


def test_router_fallback_when_helpful_and_idle_are_null(tmp_path):
    r = router_case(tmp_path, req=(0.01, 0.30), idle=(0.05, 0.05), labelled=False)
    assert r["mode"] == "fallback" and set(r["classes"]) == {"required", "optional"}
    assert r["classes"]["optional"]["n"] == 40 and r["verdict"] == "router"


def test_router_rare_not_needed_class_is_flagged(tmp_path):
    r = router_case(tmp_path, req=(0.01, 0.30), idle=(1e-6, 1e-4))  # x100, but never at 1 %
    assert r["classes"]["idle"]["state"] == "rare" and r["verdict"] == "router"
    assert r["flag"] == "not-needed class stays below 1 %"


def test_class_must_not_change_between_checkpoints(tmp_path):
    ts = thms(25, cls={"reductio": "required"})
    write_ckpt(
        tmp_path,
        "b",
        "1",
        0,
        "base",
        0.0,
        ckpt_rows(ts, lambda s, n, i: {"reductio": {"P": 0.1, "C": 0.3}}),
    )
    ts2 = thms(25, cls={"reductio": "idle"})
    write_ckpt(
        tmp_path,
        "e",
        "1",
        0,
        "rl_end",
        0.5,
        ckpt_rows(ts2, lambda s, n, i: {"reductio": {"P": 0.1, "C": 0.3}}),
    )
    with pytest.raises(A.SchemaError, match="dir_class"):
        A.analyse(tmp_path)


# ----------------------------------------------------------------------------- end to end, and the figure
def make_demo(out: Path) -> None:
    """A synthetic read in the full layout (the 106 and 240, three doses x three seeds, Dan's full base), for the
    figure and the end-to-end test. Its numbers are invented and mean nothing."""
    strict = [
        (
            "strict106",
            f"s{i:03d}",
            {"reductio": True, "negation": i < 10, "cases": i % 4 == 0},
            {"reductio": "required"},
        )
        for i in range(106)
    ]
    other = [("transfer_other", f"o{i:03d}",
              {"reductio": True, "negation": i % 4 == 0, "conditional": i % 3 == 0, "cases": i % 5 == 0},
              {"reductio": "helpful" if i % 2 else "idle"}) for i in range(240)]  # fmt: skip
    ts = strict + other

    def spec(dose, seed, stage):
        j = 0 if stage == "base" else 1
        jit = 1 + 0.3 * (_u(dose, seed, "jit") - 0.5)

        def f(s, n, i):
            if dose == "full":
                red = (0.002, 0.2)
            elif dose == "1":
                red = ((0.001 * jit, 0.13 * jit), (0.12 * jit, 0.3 * jit))[j]
            else:
                red = ((3e-5 * jit, 0.10 * jit), (4e-8 * jit, 0.30 * jit))[j]
            if s == "transfer_other" and dose == "1" and j == 1:
                red = (red[0] * (0.05 if n[-1] in "02468" else 0.4), 0.45 * jit)
            elif s == "transfer_other":
                red = (red[0] * 0.5, 0.45 * jit if j else 0.4 * jit)
            return {"reductio": {"P": red[0], "C": red[1]},
                    "negation": {"P": (0.5, 0.75)[j] * jit, "C": 0.6},
                    "conditional": {"P": (0.9, 0.95)[j], "C": (0.7, 0.72)[j] * jit},
                    "cases": {"P": (0.08, 0.10)[j] * jit, "C": (0.2, 0.35)[j] * jit}}  # fmt: skip

        return f

    xs = {"0": (5.07, 5.93, 5.94), "0.1": (4.42, 5.43, 5.63), "1": (0.41, 0.45, 0.69)}
    write_ckpt(out, "full_s3", "full", 3, "base", None, ckpt_rows(ts, spec("full", 3, "base")))
    for dose, xx in xs.items():
        for seed, x in enumerate(xx):
            write_ckpt(
                out,
                f"base_{dose}_s{seed}",
                dose,
                seed,
                "base",
                0.0,
                ckpt_rows(ts, spec(dose, seed, "base")),
            )
            write_ckpt(
                out,
                f"end_{dose}_s{seed}",
                dose,
                seed,
                "rl_end",
                x,
                ckpt_rows(ts, spec(dose, seed, "rl_end")),
            )


@pytest.fixture(scope="module")
def demo(tmp_path_factory) -> tuple[Path, Path]:
    d = tmp_path_factory.mktemp("demo")
    make_demo(d / "results")
    out = d / "out"
    sys.argv = ["analyze.py", "--results", str(d / "results"), "--out", str(out)]
    A.main()
    return d, out


def test_end_to_end_writes_tables_and_json(demo):
    _, out = demo
    res = json.loads((out / "results.json").read_text())
    md = (out / "tables.md").read_text()
    assert res["informative"]["verdict"] in ("informative", "not informative")
    assert res["doses"]["0|strict106|reductio"]["reading"].startswith("candidate capability gain")
    assert res["doses"]["1|strict106|reductio"]["reading"].startswith("mix")
    assert res["doses"]["1|strict106|conditional"]["reading"] == "never applicable"
    assert res["doses"]["1|strict106|negation"]["reading"] == "no reading (n < 20)"
    assert res["router_doses"]["1|reductio"]["verdict"].startswith("router")
    assert "full_s3" in res["levels"]
    for section in ("Dose-level readings", "Per seed", "Levels per checkpoint", "Unforced success", "uses d anywhere",
                    "familiar subset", "router or default shift", "by schema"):  # fmt: skip
        assert section in md, section


def test_figure_renders(demo):
    try:
        import fig  # noqa: PLC0415
    except ImportError as e:  # needs the repository root on the path for code.ndbench
        pytest.skip(f"fig needs code.ndbench: {e}")
    d, out = demo
    paths = fig.render(
        json.loads((out / "results.json").read_text()), d / "fig" / "propensity_capability"
    )
    for p in paths:
        assert p.exists() and p.stat().st_size > 1000


if __name__ == "__main__":
    make_demo(Path(sys.argv[1]))
    print(f"wrote a synthetic read to {sys.argv[1]}")
