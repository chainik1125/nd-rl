"""Tests of posthoc_anywhere.py on synthetic result rows in SPEC.md's schema. CPU only.

From the repository root:

    code/.venv/bin/python -m pytest -c code/pyproject.toml code/experiments/current/propensity_capability_20261109/test_posthoc_anywhere.py

Each case sets, per theorem, exact counts out of k = 32: attempts that use the reductio anywhere (`used`), accepted
attempts (U/any `solved`), and accepted attempts whose proof goes through the reductio (U/reductio `solved`, as
pc_run.py writes it), so the outcome is known before the analysis runs: a router (uses rise on required theorems
only), a default shift (uses rise on required and idle alike), no rise (required flat), a rise carried only by proofs
through the reductio (the "excluding proofs through d" sensitivity then sees none), detours in accepted proofs, the
split of required by set, the dose agreement rule over three seeds, the matched-compute checkpoint kept apart, and the
files written end to end.
"""

# ruff: noqa: E501, B905
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyze as A  # noqa: E402
import posthoc_anywhere as H  # noqa: E402

K, KF = 32, 16
OTHER = ("negation", "conditional", "cases")


def rows_for(
    s: str, name: str, cls: str | None, used: int, solved_any: int, solved_red: int
) -> list[dict]:
    """One theorem at one checkpoint: reductio applicable (the only direction), with exact U counts. solved_red:
    accepted attempts whose proof goes through the reductio."""
    meta = {"set": s, "name": name, "n_lines": 5, "schema": None, "source": "syn", "tokens": None}
    out = [
        {**meta, "arm": "U", "direction": "reductio", "applicable": True, "dir_class": cls, "k": K, "solved": solved_red,
         "used": used, "first": 0, "solved_first": 0, "capped": None, "refused": None, "logp_open": None},
        {**meta, "arm": "F", "direction": "reductio", "applicable": True, "dir_class": cls, "k": KF, "solved": 4,
         "used": None, "first": None, "capped": 0, "refused": 0, "logp_open": None},
        {**meta, "arm": "E", "direction": "reductio", "applicable": True, "dir_class": cls, "k": 0, "solved": None,
         "used": None, "first": None, "capped": None, "refused": None, "logp_open": math.log(1e-4)},
        {**meta, "arm": "U", "direction": "any", "applicable": True, "dir_class": None, "k": K, "solved": solved_any,
         "used": None, "first": None, "capped": 0, "refused": 0, "logp_open": None},
        {**meta, "arm": "U", "direction": "direct", "applicable": True, "dir_class": None, "k": K, "solved": 0,
         "used": None, "first": K, "capped": None, "refused": None, "logp_open": None},
    ]  # fmt: skip
    for d in OTHER:
        out += [
            {**meta, "arm": "U", "direction": d, "applicable": False, "dir_class": None, "k": K, "solved": 0, "used": 0,
             "first": 0, "capped": None, "refused": None, "logp_open": None},
            {**meta, "arm": "F", "direction": d, "applicable": False, "dir_class": None, "k": 0, "solved": 0, "used": None,
             "first": None, "capped": None, "refused": None, "logp_open": None},
            {**meta, "arm": "E", "direction": d, "applicable": False, "dir_class": None, "k": 0, "solved": None, "used": None,
             "first": None, "capped": None, "refused": None, "logp_open": None},
        ]  # fmt: skip
    return out


def write(
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


def theorems(
    n_req_strict: int = 40, n_req_other: int = 0, n_idle: int = 40, n_helpful: int = 4
) -> list[tuple[str, str, str]]:
    return ([("strict106", f"r{i:03d}", "required") for i in range(n_req_strict)]
            + [("transfer_other", f"q{i:03d}", "required") for i in range(n_req_other)]
            + [("transfer_other", f"i{i:03d}", "idle") for i in range(n_idle)]
            + [("transfer_other", f"h{i:03d}", "helpful") for i in range(n_helpful)])  # fmt: skip


def counts(rate: float, i: int) -> int:
    """A per-theorem count out of K around `rate`, varying over theorems so the bootstrap has spread."""
    return max(0, min(K, round(K * rate * (0.5 + (i % 5) / 4))))


def ckpt(ts, spec) -> list[dict]:
    """spec(set, name, cls, i) -> (used, solved_any, solved_red)."""
    out = []
    for i, (s, n, c) in enumerate(ts):
        out += rows_for(s, n, c, *spec(s, n, c, i))
    return out


def simple(use_req: float, use_idle: float, acc_req: float = 0.0):
    """Required theorems use the reductio at rate use_req, idle ones at use_idle. On required theorems a share acc_req
    of attempts is accepted, each of them using the reductio (as on the strict 106); idle ones are accepted at 50 %
    and their accepted attempts never use it."""

    def f(s, n, c, i):
        if c == "required":
            u = counts(use_req, i)
            a = min(u, counts(acc_req, i))
            return u, a, a
        return counts(use_idle, i), K // 2, 0

    return f


def run(tmp: Path, base_spec, end_spec, ts=None, seeds=(0,), dose="1", matched_spec=None) -> dict:
    ts = ts or theorems()
    res_dir = tmp / "rows"
    for sd in seeds:
        write(res_dir, f"b{dose}_{sd}", dose, sd, "base", 0.0, ckpt(ts, base_spec))
        write(res_dir, f"e{dose}_{sd}", dose, sd, "rl_end", 0.5, ckpt(ts, end_spec))
    if matched_spec is not None:
        write(res_dir, f"m{dose}_0", dose, 0, "matched", 0.3, ckpt(ts, matched_spec))
    return H.posthoc(res_dir, None)


def frozen_cmp(res: dict, line: str = "1|0") -> dict:
    return res["lines"][line]["router"]["reductio"]["comparisons"]["frozen classes, all attempts"]


# ----------------------------------------------------------------------------- the three verdicts
def test_router(tmp_path):
    res = run(tmp_path, simple(0.02, 0.04), simple(0.40, 0.04))
    r = res["lines"]["1|0"]["router"]["reductio"]
    assert r["classes"]["required"]["all"]["state"] == "up"
    assert r["classes"]["idle"]["all"]["state"] == "flat"
    c = frozen_cmp(res)
    assert c["verdict"] == "router" and abs(c["spillover_f"]) < 0.1
    assert c["diff_points"]["est"] > 0.3 and c["diff_points"]["ci"][0] > 0
    # helpful (n = 4) is reported with no state
    assert r["classes"]["helpful"]["all"]["state"].startswith("not read")


def test_default_shift(tmp_path):
    res = run(tmp_path, simple(0.02, 0.02), simple(0.40, 0.40))
    r = res["lines"]["1|0"]["router"]["reductio"]
    assert (
        r["classes"]["required"]["all"]["state"] == "up"
        and r["classes"]["idle"]["all"]["state"] == "up"
    )
    c = frozen_cmp(res)
    assert c["verdict"] == "default shift" and 0.8 < c["spillover_f"] < 1.25


def test_no_rise(tmp_path):
    res = run(tmp_path, simple(0.05, 0.02), simple(0.05, 0.40))
    r = res["lines"]["1|0"]["router"]["reductio"]
    assert r["classes"]["required"]["all"]["state"] == "flat"
    assert frozen_cmp(res)["verdict"] == "no rise to locate"


# ----------------------------------------------------------------------------- the split by acceptance
def test_rise_carried_by_accepted_attempts_only(tmp_path):
    """Required: uses rise 5 % -> 40 % only because accepted attempts (all of which use it) rise 0 -> 35 %;
    attempts not accepted use it at about 5 % throughout. The all-attempts rule sees a router; the not-accepted
    sensitivity sees no rise."""

    def spec(stage):
        def f(s, n, c, i):
            if c == "required":
                a = 0 if stage == "b" else counts(0.35, i)
                rej_use = round(0.05 * (K - a))
                return a + rej_use, a, a
            return 1, K // 2, 0

        return f

    res = run(tmp_path, spec("b"), spec("e"))
    r = res["lines"]["1|0"]["router"]["reductio"]
    req = r["classes"]["required"]
    assert req["all"]["state"] == "up"
    assert req["excl_through"]["state"] in ("flat", "rare")
    assert req["among_accepted"]["end"]["est"] == 1.0  # every accepted attempt goes through it
    assert req["cells_accepted_not_through"] == 0
    # all = through + not through, exactly
    for stage in ("base", "end"):
        assert (
            abs(
                req["all"][stage]["est"]
                - req["through"][stage]["est"]
                - req["not_through"][stage]["est"]
            )
            < 1e-12
        )
    cmp = r["comparisons"]
    assert cmp["frozen classes, all attempts"]["verdict"] == "router"
    assert cmp["frozen classes, excluding proofs through d"]["verdict"] == "no rise to locate"


def test_detours_in_accepted_proofs(tmp_path):
    """As in the real rows: an accepted proof can open the reductio without going through it (U/reductio solved counts
    proofs through it), so 'opens it' can exceed 'not accepted'. Such theorems must not break the split, and they count
    in 'uses, excluding proofs through d'."""

    def spec(stage):
        def f(s, n, c, i):
            if c == "required":
                return counts(0.02 if stage == "b" else 0.40, i), 0, 0
            if (
                i % 4 == 0
            ):  # idle theorem: every attempt accepted, every one opens a reductio as a detour
                return K, K, 0
            return 1, K // 2, 0

        return f

    res = run(tmp_path, spec("b"), spec("e"))
    idle = res["lines"]["1|0"]["router"]["reductio"]["classes"]["idle"]
    assert idle["cells_accepted_not_through"] > 0
    assert (
        abs(idle["excl_through"]["base"]["est"] - idle["all"]["base"]["est"]) < 1e-12
    )  # nothing goes through it
    assert idle["through"]["base"]["est"] == 0


def test_required_split_by_set(tmp_path):
    """The other set's required theorems stay flat while the strict ones rise: the frozen class rises (diluted), the
    strict-only side rises more, the other-set side gives no rise."""
    ts = theorems(n_req_strict=40, n_req_other=24, n_idle=40)

    def spec(stage):
        def f(s, n, c, i):
            if c == "required" and s == "strict106":
                return counts(0.02 if stage == "b" else 0.40, i), 0, 0
            if c == "required":
                return counts(0.10, i), K // 2, 0
            return counts(0.03, i), K // 2, 0

        return f

    res = run(tmp_path, spec("b"), spec("e"), ts=ts)
    r = res["lines"]["1|0"]["router"]["reductio"]
    assert r["classes"]["required"]["all"]["n"] == 64
    assert (
        r["classes"]["required_strict106"]["all"]["n"] == 40
        and r["classes"]["required_other"]["all"]["n"] == 24
    )
    assert (
        r["classes"]["required_strict106"]["all"]["dlogodds10"]["est"]
        > r["classes"]["required"]["all"]["dlogodds10"]["est"]
    )
    cmp = r["comparisons"]
    assert cmp["strict 106 only, all attempts"]["verdict"] == "router"
    assert cmp["30 classical-only of the other set, all attempts"]["verdict"] == "no rise to locate"


# ----------------------------------------------------------------------------- dose level, pooling, matched
def test_dose_agreement_and_pooled_has_no_verdict(tmp_path):
    res = run(tmp_path, simple(0.02, 0.04), simple(0.40, 0.04), seeds=(0, 1, 2))
    dv = res["doses"]["1"]["router"]["reductio"]
    assert dv["verdicts"]["frozen classes, all attempts"]["per_seed"] == ["router"] * 3
    assert dv["verdict"] == "router"
    pooled = dv["pooled"]["comparisons"]["frozen classes, all attempts"]
    assert pooled["verdict"].startswith("none")
    # identical seeds: the pooled rate equals each seed's rate
    one = res["lines"]["1|0"]["router"]["reductio"]["classes"]["required"]["all"]
    assert abs(dv["pooled"]["classes"]["required"]["all"]["end"]["est"] - one["end"]["est"]) < 1e-12


def test_matched_is_kept_apart(tmp_path):
    res = run(tmp_path, simple(0.02, 0.04), simple(0.40, 0.04), matched_spec=simple(0.02, 0.40))
    assert "m1_0" in res["matched"]
    m = res["matched"]["m1_0"]["router"]["reductio"]
    assert m["comparisons"]["frozen classes, all attempts"]["verdict"] == "no rise to locate"
    # the dose verdict uses the RL end only
    assert res["doses"]["1"]["router"]["reductio"]["verdicts"]["frozen classes, all attempts"][
        "per_seed"
    ] == ["router"]


def test_other_directions_have_no_router_reading(tmp_path):
    """negation, conditional, cases: dir_class null, so the fallback: no required class, no verdict."""
    ts = theorems()

    def with_neg(spec):
        def rows(s, n, c, i):
            return spec(s, n, c, i)

        return rows

    res_dir = tmp_path / "rows"
    for stage, sp, x in (("base", simple(0.02, 0.04), 0.0), ("rl_end", simple(0.4, 0.04), 0.5)):
        rows = ckpt(ts, with_neg(sp))
        for r in rows:  # make negation applicable everywhere, with a rise in use
            if r["direction"] == "negation":
                r["applicable"] = True
                if r["arm"] == "U":
                    r["used"] = 8 if stage == "base" else 24
                if r["arm"] == "F":
                    r["k"] = KF
                if r["arm"] == "E":
                    r["logp_open"] = math.log(0.2)
        write(res_dir, f"{stage}_0", "1", 0, stage, x, rows)
    res = H.posthoc(res_dir, None)
    r = res["lines"]["1|0"]["router"]["negation"]
    assert r["mode"] == "fallback" and r["verdict"].startswith("no router reading")
    assert r["classes"]["optional"]["all"]["state"] == "up" and not r["comparisons"]


def test_same_theorems_and_levels_as_the_frozen_router(tmp_path):
    """The frozen router cell's 'uses anywhere' column is the measure here: same classes, same n, same levels."""
    ts = theorems(n_req_strict=30, n_req_other=0, n_idle=30, n_helpful=0)
    res_dir = tmp_path / "rows"
    write(res_dir, "b", "1", 0, "base", 0.0, ckpt(ts, simple(0.02, 0.04)))
    write(res_dir, "e", "1", 0, "rl_end", 0.5, ckpt(ts, simple(0.40, 0.04, acc_req=0.2)))
    fr = A.analyse(res_dir)["lines"]["1|0"]["router"]["reductio"]["classes"]
    mine = H.posthoc(res_dir, None)["lines"]["1|0"]["router"]["reductio"]["classes"]
    for g in ("required", "idle"):
        assert fr[g]["n"] == mine[g]["all"]["n"]
        assert abs(fr[g]["used_base"] - mine[g]["all"]["base"]["est"]) < 1e-12
        assert abs(fr[g]["used_end"] - mine[g]["all"]["end"]["est"]) < 1e-12


def test_inconsistent_counts_are_refused(tmp_path):
    def bad(s, n, c, i):
        return 2, 1, 3  # accepted-and-uses above uses

    res_dir = tmp_path / "rows"
    write(res_dir, "b", "1", 0, "base", 0.0, ckpt(theorems(), bad))
    write(res_dir, "e", "1", 0, "rl_end", 0.5, ckpt(theorems(), bad))
    with pytest.raises((H.PosthocError, A.SchemaError)):
        H.posthoc(res_dir, None)


def test_end_to_end_writes_files(tmp_path):
    ts = theorems()
    res_dir = tmp_path / "rows"
    for sd in (0, 1, 2):
        write(res_dir, f"b_{sd}", "1", sd, "base", 0.0, ckpt(ts, simple(0.02, 0.04)))
        write(
            res_dir,
            f"e_{sd}",
            "1",
            sd,
            "rl_end",
            0.4 + sd / 10,
            ckpt(ts, simple(0.40, 0.04, acc_req=0.2)),
        )
    frozen = tmp_path / "frozen" / "results.json"
    frozen.parent.mkdir()
    frozen.write_text(json.dumps(A.analyse(res_dir), default=float))
    out = tmp_path / "out"
    sys.argv = [
        "posthoc_anywhere.py",
        "--results",
        str(res_dir),
        "--frozen",
        str(frozen),
        "--out",
        str(out),
    ]
    try:
        H.main()
    except ImportError as e:  # the figure needs code/ndbench/figures/style.py
        pytest.skip(f"figure style not importable: {e}")
    res = json.loads((out / "posthoc_anywhere.json").read_text())
    md = (out / "posthoc_anywhere.md").read_text()
    assert res["checks"]["inputs_match_frozen_results_json"] is True
    assert res["checks"]["max_abs_diff_vs_frozen_router_uses_anywhere_column"] < 1e-12
    assert res["checks"]["uses_anywhere_secondary_recomputed_equal"] is True
    assert res["doses"]["1"]["router"]["reductio"]["verdict"] == "router"
    for section in ("Reductio: dose verdicts", "Reductio per seed", "pooled per dose", "carried by accepted proofs",
                    "no router reading", "elicitation against capability"):  # fmt: skip
        assert section in md, section
    for ext in ("pdf", "svg", "png"):
        p = out / f"posthoc_anywhere.{ext}"
        assert p.exists() and p.stat().st_size > 1000
