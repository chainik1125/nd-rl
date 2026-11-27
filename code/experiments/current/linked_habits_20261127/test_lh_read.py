"""Tests of lh_read (PREREG Amendments 1-6) on a synthetic run tree whose answers are known: the +30 read by
interpolation between bracketing snapshots (from the full task's record, or recomputed), the mean over two DPO seeds,
D with the thin rule, P1, S_before / S_after, the site-level split, the pooled first-site notes, the corpus contrasts,
the GRPO chain prediction and reward-model marks, the fine-tune composition prediction, the form-type groups, the
readings table and the report. Every theorem is identical, so every interval equals its point.

    CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 /scratch/cowifornia/nd-rl/think/nd-rl/code/.venv/bin/python \
        -m pytest --confcutdir=. -p no:cacheprovider -q test_lh_read.py
"""

# ruff: noqa: E501, I001

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import lh_core as C  # noqa: E402
import lh_read as LR  # noqa: E402
import lh_stats as S  # noqa: E402

HABITS, NATURAL, NOTES = LR.habit_lists()
N_MP, N_RU = 10, 4
PER, FIRST = (
    100,
    10,
)  # sites with a form per habit per theorem (exfalso: 2, thin), of which the proofs' first sites


def lh(
    shares: dict[str, float],
    cond: dict | None = None,
    condn: list | None = None,
    after: dict | None = None,
) -> dict:
    """A theorem's `lh`: PER sites per habit at the given A-shares (default 50), FIRST of them first sites; `after`
    {h: (sites, A)} of h's sites after the first reit site (lh_measure `pair reit|h`); cond / condn for reit."""
    hab, first, later = {}, {}, {}
    for h in HABITS:
        n, f = (2, 1) if h == "exfalso" else (PER, FIRST)
        sh = shares.get(h, 50)
        a, fa = round(n * sh / 100), round(f * sh / 100)
        hab[h] = [n, a, n - a, 0, 0]
        first[h], later[h] = [fa, f - fa], [a - fa, n - a - (f - fa)]
    pair = {f"reit|{h}": [a1, n1 - a1, 0, 0] for h, (n1, a1) in (after or {}).items()}
    return {"seen": 8, "step_cap": 1, "proofs": 4, "lines": 40, "hab": hab, "first": first, "later": later, "link": {}, "linko": {},
            "pair": pair, "cond": {"reit": cond or {}}, "condn": {"reit": condn or [0, 0]}, "near": {"ande_apart": 2, "late_note": 0}}  # fmt: skip


def write(d: Path, none: dict[str, float], **kw) -> None:
    d.mkdir(parents=True, exist_ok=True)
    tags = {
        "none": lh(none, **kw),
        "A": lh(dict.fromkeys(HABITS, 90.0)),
        "B": lh(dict.fromkeys(HABITS, 10.0)),
    }
    for s, n in (("marker_plain", N_MP), ("route_unseen", N_RU)):
        for t, x in tags.items():
            rows = [
                {"theorem": f"{s[:2]}{i}", "attempts": 8, "accepted": 4, "lh": x} for i in range(n)
            ]
            (d / f"{s}__{t}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (d / "DONE").write_text("x")


def run(lh_: Path, arm: str, by_step: dict[int, dict[str, float]], final: int = 2000) -> None:
    a = lh_ / "arms" / arm
    for st, sh in by_step.items():
        write(a / "measure" / ("final" if st == final else f"step{st}"), sh)
    (a / "done.json").write_text(json.dumps({"final": {"step": final}}))
    (a / "metrics.jsonl").write_text(
        "".join(json.dumps({"step": st, "kl": st / 1000}) + "\n" for st in by_step)
    )


NOTES60 = dict.fromkeys(NOTES, 60.0)


@pytest.fixture(scope="module")
def tree(tmp_path_factory):
    t = tmp_path_factory.mktemp("lh")
    (t / "sets").mkdir()
    for s, n in (("marker_plain", N_MP), ("route_unseen", N_RU)):
        (t / "sets" / f"{s}.jsonl").write_text(
            "".join(json.dumps({"name": f"{s[:2]}{i}"}) + "\n" for i in range(n))
        )
    (t / "data").mkdir()
    if (C.LH / "data" / "note_table.json").exists():
        shutil.copy(C.LH / "data" / "note_table.json", t / "data")
    base_cond = {"note_atom": [80, 20, 20, 80]}  # the base: P(atom note A | reit A) 80, | reit B 20
    for c in ("c24", "c4", "c0"):
        write(t / "measure" / f"{c}_s0", {}, cond=base_cond, condn=[50, 50])
    # DPO: reit 70 at step 50 (+20), 90 at step 100 (+40), 95 at the end: +30 at w = 0.5 between 50 and 100
    lift = {
        "c24": (60, 80),
        "c4": (60, 70),
        "c0": (50, 50),
    }  # order and conj in the rated run at steps 50 and 100
    for c, (o1, o2) in lift.items():
        for seed in (0, 1):
            d = (
                4 if (c == "c24" and seed == 1) else 0
            )  # seed 1 in c24: order and conj 4 points higher
            notes = NOTES60 if c == "c24" else {}
            run(t, C.arm_name("dpo", c, "reit", seed), {50: {"reit": 70, "order": o1 + d, "conj": o1 + d, **notes},
                                                        100: {"reit": 90, "order": o2 + d, "conj": o2 + d, **notes},
                                                        2000: {"reit": 95}})  # fmt: skip
            run(t, C.arm_name("dposwap", c, "reit", seed), {50: {}, 100: {}, 2000: {}})
    (t / "arms" / "full_dpo_c24_reit_s0.json").write_text(
        json.dumps({"reached": True, "bracket": [50, 100], "done": True})
    )
    probe = {
        "snapshots": {
            f"step{k}": {"a_share": {"reit": v}} for k, v in ((25, 55.0), (50, 70.0), (100, 90.0))
        }
    }
    (t / "arms" / "dpo_c24_reit_s0" / "probe").mkdir()
    (t / "arms" / "dpo_c24_reit_s0" / "probe" / "probe.json").write_text(json.dumps(probe))
    # the fine-tune pair in c24 and its composition
    run(
        t,
        C.arm_name("sft", "c24", "reit", 0),
        {50: {"reit": 70, "order": 55}, 100: {"reit": 90, "order": 65}, 2000: {"reit": 95}},
    )
    run(t, C.arm_name("sftsam", "c24", "reit", 0), {50: {}, 100: {}, 2000: {}})
    sp = t / "pairs" / "c24_s0" / "reit"
    sp.mkdir(parents=True)
    sp.joinpath("summary.json").write_text(json.dumps({"a_share": {"sft_pref": {"reit": 100.0, "order": 55.0}, "sft_sampled": {"reit": 30.0, "order": 50.0},
                                                                  "base_sampled": {"reit": 30.0}}}))  # fmt: skip
    # GRPO in c24: reit 60 at it5, 80 at it10 (matched); the reward line writes A notes beyond the chain prediction
    for it, sh, cond, condn in ((5, {"reit": 60}, {"note_atom": [72, 18, 2, 8]}, [90, 10]), (10, {"reit": 80, "restate": 70},
                                                                                           {"note_atom": [80, 10, 2, 8]}, [90, 10])):  # fmt: skip
        write(
            t / "runs" / "rl_c24_rlhfreit_s0" / "seg1" / "authors" / f"it{it:04d}",
            sh,
            cond=cond,
            condn=condn,
            after={"restate": (40, 36)},
        )
        write(t / "runs" / "rl_c24_anc_s0" / "seg1" / "authors" / f"it{it:04d}", {}, cond={"note_atom": [40, 10, 10, 40]}, condn=[50, 50],
              after={"restate": (40, 20)})  # fmt: skip
    rm = t / "arms" / "rm_c24_reit_s0"
    rm.mkdir(parents=True)
    rm.joinpath("probe.json").write_text(
        json.dumps(
            {
                "habits": {
                    "note_atom": {"n": 50, "a_higher_share": 0.62},
                    "order": {"n": 40, "a_higher_share": 0.5},
                }
            }
        )
    )
    return t


@pytest.fixture(scope="module")
def res(tree):
    return LR.readout(tree, ["c24", "c4", "c0"], 200, 0, 50, 30.0)


def test_interpolated_read_and_seed_mean(res):
    e = res["compare"]["dpo:reit"]["c24"]
    rd = e["read"]["at_target"]
    assert (
        rd["steps"] == [[50, 100], [50, 100]]
        and rd["w"] == [0.5, 0.5]
        and rd["move"] == 30.0
        and rd["reached"]
    )
    assert rd["D"]["order"]["point"] == pytest.approx(22) and rd["D"]["order"]["ci"] == [
        pytest.approx(22),
        pytest.approx(22),
    ]
    assert rd["per_seed"][0]["D"]["order"]["point"] == pytest.approx(20) and rd["per_seed"][1]["D"][
        "order"
    ]["point"] == pytest.approx(24)
    assert rd["D"]["reit"]["point"] == pytest.approx(30) and rd["D_per_point"][
        "order"
    ] == pytest.approx(22 / 30, abs=1e-3)
    assert (
        rd["D"]["note_atom"]["point"] == pytest.approx(10)
        and "exfalso" in rd["thin"]
        and "exfalso" not in rd["D"]
    )
    assert rd["kl_rated"] == [[0.05, 0.1], [0.05, 0.1]] and rd["precondition_met"]
    assert [m["move"] for m in e["seeds"][0]["probe"]] == [5.0, 20.0, 40.0]
    last = e["read"]["last"]
    assert last["steps"] == [[2000], [2000]] and last["move"] == 45.0
    # c4 has no full-task record: the bracket is recomputed from the full measurements
    assert res["compare"]["dpo:reit"]["c4"]["read"]["at_target"]["D"]["order"][
        "point"
    ] == pytest.approx(15)


def test_prereg_dpo(res):
    pr = res["prereg"]["dpo:reit:at_target"]
    assert pr["P1_M(c24)"]["point"] == pytest.approx(22) and pr["P1_M(c4)"][
        "point"
    ] == pytest.approx(15)
    assert pr["P1_M(c24)-M(c4)"]["point"] == pytest.approx(7)
    sb = pr["P2_S_before(c24)"]
    assert set(sb["habits"]) == {*NOTES, "order"} and sb["point"] == pytest.approx(
        (15 * 10 + 22) / 16, abs=0.01
    )
    sa = pr["P2_S_after(c24)"]
    assert set(sa["habits"]) <= set(LR.AFTER) and "conj" in sa["habits"]
    assert sa["point"] == pytest.approx(22 / len(sa["habits"]), abs=0.01)
    assert set(pr["P2_S_before(c4)"]["habits"]) == {"order"} and pr["P2_first_notes_pooled(c24)"][
        "point"
    ] == pytest.approx(10)
    assert pr["P2_S_after(c4)"]["habits"] == ["conj"] and pr["preconditions"] == {
        "c24": True,
        "c4": True,
        "c0": True,
    }
    b = res["between"]["dpo:reit:at_target"]
    assert b["c24-c0"]["order"]["point"] == pytest.approx(22) and b["c24-c4"]["conj"][
        "point"
    ] == pytest.approx(7)
    g = pr["notes_by_form_type"]
    if res["form_groups"]:
        assert set(g) == {"copy", "or2", "longer", "shorter"} and g["copy"]["c24-c0"][
            "point"
        ] == pytest.approx(10)
    rdg = res["readings"]
    assert (
        rdg["reading"] == "the author moves as a unit with 24 links and not with 4"
        and not rdg["invalid"]
    )
    assert (
        rdg["per_seed"]["M(c24)"] == [pytest.approx(20), pytest.approx(24)]
        and "second_base_needed" not in rdg
    )


def test_grpo_chain_and_reward_marks(res):
    e = res["compare"]["rlhf:reit"]["c24"]
    rd = e["read"][
        "at_target"
    ]  # the primary GRPO read at matched movement, the last iteration beside it (Amendment 8)
    assert e["primary"] == "at_target" == res["compare"]["dpo:reit"]["c24"]["primary"]
    assert rd["steps"] == [[10]] and rd["move"] == 30.0 and "chain" in e["read"]["last"]
    ch = rd["chain"][
        "note_atom"
    ]  # predicted 0.9 x 80 + 0.1 x 20 = 74, observed 82; anchor line 50 against 50
    assert ch["reward_line"]["point"] == pytest.approx(8) and ch["anchor_line"][
        "point"
    ] == pytest.approx(0)
    assert ch["difference"]["point"] == pytest.approx(8)
    assert e["reward_driven"] is None and e["rm_probe"]["rule"].startswith(
        "pre-Amendment 7"
    )  # an old output: no marks
    sp = rd[
        "D_split"
    ]  # restate after the first reit site: 36 / 40 A in the reward line, 20 / 40 in the anchor line
    assert sp["after"]["restate"]["point"] == pytest.approx(40) and sp["before"]["restate"][
        "point"
    ] == pytest.approx((70 - 36) / 60 * 100 - 30 / 60 * 100, abs=0.01)


def test_c9_grpo_not_run_by_design(tree):
    r = LR.readout(tree, ["c24", "c9"], 20, 0, 50, 30.0)
    assert r["compare"]["rlhf:reit"]["c9"]["not_run"].startswith("by design")
    assert "c24" in r["compare"]["rlhf:reit"] and "c9: NOT RUN by design" in LR.report(r)


def test_sft_beside_its_prediction(res):
    e = res["compare"]["sft:reit"]["c24"]
    assert e["composition_prediction"] == {"reit": 70.0, "order": 5.0}
    assert e["read"]["at_target"]["D"]["order"]["point"] == pytest.approx(10)  # (55 + 65) / 2 - 50


def test_split_share_counts():
    cols = {"reit|x": None}
    del cols
    r = {
        "attempts": 8,
        "accepted": 8,
        "lh": {"hab": {"x": [100, 70, 30, 0, 0]}, "pair": {"reit|x": [30, 10, 0, 0]}},
    }
    m = S.Mat({"t": r}, ["t"], ("reit",))
    s = m.sums(__import__("numpy").ones(1))
    assert float(S.split_share(s, "reit", "x", "after")) == 75.0 and float(
        S.split_share(s, "reit", "x", "before")
    ) == pytest.approx(40 / 60 * 100)
    assert float(S.split_sites(s, "reit", "x", "before")) == 60.0
    assert (
        "pair reit|x:aa" in S.Mat({"t": r}, ["t"], ("reit",)).cols
        and "pair reit|x:aa" not in S.Mat({"t": r}, ["t"], ("note_atom",)).cols
    )


def test_untagged_only_snapshot(tree, tmp_path):
    """A tuned snapshot measured untagged only (PREREG Amendment 4, point 2) is read; a folder without untagged rows is not."""
    d = tmp_path / "m"
    write(d, {"reit": 80})
    for f in d.glob("*__[AB].jsonl"):
        f.unlink()
    ids = S.set_ids(LR.READ_SET, tree / "sets")
    m = LR.mats(d, LR.READ_SET, ids)
    assert (
        set(m) == {"none"}
        and LR.seeds_of("dpo", "atom") == (0,)
        and LR.seeds_of("sftsam", "atom") == ()
    )
    assert (
        LR.seeds_of("dposwap", "reit") == (0, 1)
        and LR.seeds_of("rlhf", "atom") == ()
        and LR.seeds_of("rlhf", "reit") == (0,)
    )
    for f in d.glob("*__none.jsonl"):
        f.unlink()
    assert LR.mats(d, LR.READ_SET, ids) is None


def test_probe_bracket_not_confirmed(tree, tmp_path):
    """The probe's bracket read by the full measurements: both snapshots under +30 -> w clamped, not reached, flagged."""
    t = tmp_path / "lh"
    shutil.copytree(tree, t)
    for seed in (0, 1):
        arm = C.arm_name("dpo", "c0", "reit", seed)
        shutil.rmtree(t / "arms" / arm / "measure" / "step100")
        write(
            t / "arms" / arm / "measure" / "step100", {"reit": 75, "order": 50, "conj": 50}
        )  # +25: under +30
        (t / "arms" / f"full_dpo_c0_reit_s{seed}.json").write_text(
            json.dumps({"reached": True, "bracket": [50, 100], "done": True})
        )
    r = LR.readout(t, ["c24", "c4", "c0"], 50, 0, 50, 30.0)
    rd = r["compare"]["dpo:reit"]["c0"]["read"]["at_target"]
    assert rd["steps"] == [[50, 100], [50, 100]] and rd["w"] == [1.0, 1.0] and rd["move"] == 25.0
    assert not rd["reached"] and rd["bracketed"] == [False, False]
    assert "do not bracket +30" in LR.report(r)


def test_tables_and_report(res):
    t = res["tables"]["dpo_c24_reit_s0"]["step100"]["read"]["none"]
    assert (
        t["a_share"]["reit"] == 90.0
        and t["near_per_proof"]["ande_apart"] == 0.5
        and t["other_pct"]["reit"] == 0.0
    )
    assert (
        t["accepted_pct"] == 50.0
        and t["pass8"] == 100.0
        and t["step_cap_pct"] == 12.5
        and t["mean_lines"] == 10.0
    )
    assert res["tables"]["c24_s0"]["base"]["read"]["A"]["a_share"]["order"] == 90.0
    txt = LR.report(res)
    assert (
        "[dpo:reit]" in txt
        and "P1_M(c24)-M(c4) +7.0" in txt
        and "the author moves as a unit" in txt
        and "chain" in txt
    )


# ------------------------------------------------------------------------------------------------ audit 2b cases
# The auditor's synthetic trees (discourse/design_audit.md, R3 and R4; its scripts read_cases.py and read_case_e.py):
# base 50 everywhere, every theorem identical, so every interval equals its point.
def case_tree(tmp_path, name):
    t = tmp_path / name
    (t / "sets").mkdir(parents=True)
    for s, n in (("marker_plain", N_MP), ("route_unseen", N_RU)):
        (t / "sets" / f"{s}.jsonl").write_text(
            "".join(json.dumps({"name": f"{s[:2]}{i}"}) + "\n" for i in range(n))
        )
    (t / "data").mkdir()
    if (C.LH / "data" / "note_table.json").exists():
        shutil.copy(C.LH / "data" / "note_table.json", t / "data")
    for c in ("c24", "c4", "c0"):
        write(t / "measure" / f"{c}_s0", {})
    return t


def case_dpo(t, c, seed, by_step, full=None):
    run(t, C.arm_name("dpo", c, "reit", seed), by_step)
    run(t, C.arm_name("dposwap", c, "reit", seed), {k: {} for k in by_step})
    if full is not None:
        (t / "arms" / f"full_dpo_{c}_reit_s{seed}.json").write_text(json.dumps(full))


def case_read(t):
    r = LR.readout(t, ["c24", "c4", "c0"], 20, 0, 50, 30.0)
    return r, r["readings"]["reading"]


def test_case_a_never_reaches_plus_20(tmp_path):
    t = case_tree(tmp_path, "A")
    for c in ("c24", "c4", "c0"):
        for s in (0, 1):
            case_dpo(
                t,
                c,
                s,
                {50: {"reit": 55}, 100: {"reit": 60}, 2000: {"reit": 60}},
                {"reached": False, "bracket": [100, 2000], "done": True},
            )
    r, row = case_read(t)
    assert row.startswith("not read: c24: seed 0: +30 not reached") and "habits edited" not in row
    assert "precondition +20 not met" in row


def test_case_b_past_plus_30_at_the_first_snapshot_is_read(tmp_path):
    t = case_tree(tmp_path, "B")
    for c in ("c24", "c4", "c0"):
        for s in (0, 1):
            o = 70 if c == "c24" else 50
            case_dpo(t, c, s, {2: {"reit": 90, "order": o, "conj": o}, 2000: {"reit": 95, "order": o, "conj": o}},
                     {"reached": True, "bracket": [0, 2], "done": True})  # fmt: skip
    r, row = case_read(t)
    rd = r["compare"]["dpo:reit"]["c24"]["read"]["at_target"]
    assert rd["w"] == [0.75, 0.75] and rd["D"]["order"]["point"] == pytest.approx(
        15
    )  # 0.75 x 20, by hand
    assert (
        not row.startswith("not read")
        and r["compare"]["dpo:reit"]["c24"]["valid"]["at_target"]["ok"]
    )


def test_case_c_seeds_disagree(tmp_path):
    t = case_tree(tmp_path, "C")
    for c in ("c24", "c4", "c0"):
        for s in (0, 1):
            o = (70 if s == 0 else 30) if c == "c24" else 50
            case_dpo(t, c, s, {50: {"reit": 70, "order": o, "conj": o}, 100: {"reit": 90, "order": o, "conj": o}, 2000: {"reit": 95}},
                     {"reached": True, "bracket": [50, 100], "done": True})  # fmt: skip
    r, row = case_read(t)
    assert row.startswith("not read: seeds disagree (M(c24) 20.0 / -20.0")
    assert r["readings"]["per_seed"]["M(c4)"] == [0.0, 0.0]


def test_case_d_only_c24_moves_the_rated_habit(tmp_path):
    t = case_tree(tmp_path, "D")
    for c in ("c24", "c4", "c0"):
        for s in (0, 1):
            if c == "c24":
                up = {"order": 70, "conj": 70, **dict.fromkeys(NOTES, 70.0)}
                case_dpo(
                    t,
                    c,
                    s,
                    {50: {"reit": 70, **up}, 100: {"reit": 90, **up}, 2000: {"reit": 95}},
                    {"reached": True, "bracket": [50, 100], "done": True},
                )
            else:
                case_dpo(
                    t,
                    c,
                    s,
                    {50: {"reit": 55}, 100: {"reit": 55}, 2000: {"reit": 55}},
                    {"reached": False, "bracket": [100, 2000], "done": True},
                )
    r, row = case_read(t)
    assert (
        row.startswith("not read: c4: seed 0: +30 not reached")
        and "c0: seed 0" in row
        and "moves as a unit" not in row
    )


def test_case_e_missing_bracket_measurement_is_not_replaced(tmp_path):
    t = case_tree(tmp_path, "E")
    for c in ("c24", "c4", "c0"):
        for s in (0, 1):
            late = 90 if c == "c24" else 50
            case_dpo(t, c, s, {50: {"reit": 70}, 100: {"reit": 90}, 2000: {"reit": 95, "order": late, "conj": late}},
                     {"reached": True, "bracket": [50, 100], "done": True})  # fmt: skip
    shutil.rmtree(t / "arms" / "dposwap_c24_reit_s0" / "measure" / "step100")
    r, row = case_read(t)
    seeds = r["compare"]["dpo:reit"]["c24"]["seeds"]
    assert (
        seeds[0]["at_target_missing"]
        == "bracket [50, 100]: no finished measurement of dposwap_c24_reit_s0@100"
    )
    assert row.startswith("not read: c24: seed 0: bracket [50, 100]: no finished measurement")
    rd = r["compare"]["dpo:reit"]["c24"]["read"]["at_target"]
    assert rd["steps"] == [
        [50, 100]
    ]  # seed 1 only: the last snapshot never stands in for seed 0's +30 read


def test_rm_probe_marks_from_the_probe(tmp_path):
    """lh_rm.py probe's own marks (per-habit `status`, PREREG Amendment 7 point 6) are read, not recomputed."""
    d = tmp_path / "arms" / "rm_c24_reit_s0"
    d.mkdir(parents=True)
    (d / "probe.json").write_text(json.dumps({"rule": "Amendment 7 point 6", "band": [-0.01, 0.01], "habits": {
        "order": {"kind": "habit", "n": 150, "bonus_diff": 0.002, "status": "within_band", "reward_driven": False},
        "probe:length": {"kind": "control", "n": 120, "bonus_diff": 0.08, "status": "reward_driven", "reward_driven": True},
        "conj": {"kind": "habit", "n": 30, "status": "not_probed", "reward_driven": None}}}))  # fmt: skip
    pr = LR.rm_probe(tmp_path, "c24", "reit")
    assert (
        pr["reward_driven"] == ["probe:length"]
        and pr["not_probed"] == ["conj"]
        and pr["habits"]["order"]["n"] == 150
    )
    (d / "probe.json").write_text(
        json.dumps({"habits": {"order": {"a_higher_share": 0.62}}})
    )  # an old output
    assert LR.rm_probe(tmp_path, "c24", "reit")["reward_driven"] is None


def test_chain_weight_from_site_counts():
    """Audit R7: q_h = (AA + AB) / (AA + AB + BA + BB) from the same cond counts as the conditionals."""
    r = {
        "attempts": 8,
        "accepted": 8,
        "lh": {
            "hab": {},
            "cond": {"reit": {"order": [30, 10, 5, 55]}},
            "condn": {"reit": [90, 10]},
        },
    }
    m = S.Mat({"t": r}, ["t"], ("reit",))
    s = m.sums(__import__("numpy").ones(1))
    assert float(S.cond_q(s, "reit", "order")) == pytest.approx(0.4) and float(
        S.first_form_share(s, "reit")
    ) == pytest.approx(0.9)


# ------------------------------------------------------------------------------------------------ second bases
# PREREG Amendment 8 points 2-3. First bases: every run reaches +30 at w 0.5 between steps 50 and 100; c24's order,
# conj and notes at `o24` in both trainer seeds, c4 and c0 flat. Second bases (base seed 1): c24b1 at `o24b`, c4b1 flat.
FULL = {"reached": True, "bracket": [50, 100], "done": True}


def two_base_tree(tmp_path, name, o24=70, o24b=None, b2=("c24", "c4"), b2_reit=90, ended=()):
    t = case_tree(tmp_path, name)
    for c in ("c24", "c4", "c0"):
        o = o24 if c == "c24" else 50
        up = {"order": o, "conj": o, **dict.fromkeys(NOTES, float(o))}
        for s in (0, 1):
            case_dpo(
                t, c, s, {50: {"reit": 70, **up}, 100: {"reit": 90, **up}, 2000: {"reit": 95}}, FULL
            )
    for c in b2:
        o = (o24b if c == "c24" else 50) if o24b is not None else 50
        up = {"order": o, "conj": o, **dict.fromkeys(NOTES, float(o))}
        write(t / "measure" / f"{c}_s1", {})
        tok = C.base_token(c, 1)
        case_dpo(
            t,
            tok,
            0,
            {
                50: {"reit": 70, **up},
                100: {"reit": b2_reit, **up},
                400: {"reit": 95 if b2_reit > 80 else b2_reit},
            },
            FULL if b2_reit > 80 else {"reached": False, "bracket": [100, 400], "done": True},
        )
    for c in ended:
        (t / "pod").mkdir(exist_ok=True)
        (t / "pod" / f"{c}.BASE2_DONE").write_text("x")
    return t


def test_second_base_pending_reads_the_first_base(tmp_path):
    r, row = case_read(two_base_tree(tmp_path, "P", b2=()))
    rd = r["readings"]
    assert rd["basis"] == "first base only (second base pending: c24b1, c4b1)" and rd[
        "second_bases"
    ] == {"c24": "pending", "c4": "pending"}
    assert row == "the author moves as a unit with 24 links and not with 4" == rd["first_base_only"]
    assert rd["per_base_seed"]["M(c24)"] == [pytest.approx(20), None]


def test_second_bases_averaged(tmp_path):
    t = two_base_tree(tmp_path, "AV", o24=80, o24b=60)  # M(c24) +30 on base 0, +10 on base 1
    r, row = case_read(t)
    rd, pr = r["readings"], r["prereg"]["dpo:reit:at_target"]
    assert rd["basis"] == "average over the two base seeds (Amendment 8 point 3)" and rd[
        "second_bases"
    ] == {"c24": "done", "c4": "done"}
    assert rd["per_base_seed"]["M(c24)"] == [pytest.approx(30), pytest.approx(10)] and rd[
        "per_base_seed"
    ]["M(c4)"] == [0.0, 0.0]
    assert pr["base_avg"]["M(c24)"]["point"] == pytest.approx(20) and pr["base_avg"][
        "S_before(c24)"
    ]["point"] == pytest.approx(20)
    assert (
        row == "the author moves as a unit with 24 links and not with 4"
        and not rd["bases_disagree"]
    )
    assert (
        r["compare"]["dpo:reit"]["c24b1"]["valid"]["at_target"]["ok"]
        and r["base2"]["c4"]["state"] == "done"
    )
    assert "c24_s1" in r["tables"] and "dpo_c24b1_reit_s0" in r["tables"]
    assert "Second base c24b1 (base seed 1): done" in LR.report(r)


def test_second_bases_disagree(tmp_path):
    r, row = case_read(
        two_base_tree(tmp_path, "DIS", o24=80, o24b=42)
    )  # D +30 on base 0, -8 on base 1
    rd = r["readings"]
    assert rd["bases_disagree"] == ["M(c24)-M(c4)", "S_before(c24)"]
    assert row.startswith(
        "unresolved (bases disagree): M(c24)-M(c4) 30.0 / -8.0; S_before(c24) 30.0 / -8.0 (the average would read: "
    )
    # "either value outside +-5": +30 / -4 disagrees; +30 / +4 (the same sign) is averaged
    r1, row1 = case_read(two_base_tree(tmp_path, "DIS4", o24=80, o24b=46))
    assert r1["readings"]["bases_disagree"] == [
        "M(c24)-M(c4)",
        "S_before(c24)",
    ] and row1.startswith("unresolved (bases disagree)")
    r2, row2 = case_read(two_base_tree(tmp_path, "AGR", o24=80, o24b=54))
    assert (
        not r2["readings"]["bases_disagree"]
        and row2 == "the author moves as a unit with 24 links and not with 4"
    )


def test_second_base_ended_without_its_measurement(tmp_path):
    r, row = case_read(two_base_tree(tmp_path, "END", b2=("c24",), ended=("c4",)))
    assert r["readings"]["second_bases"] == {"c24": "done", "c4": "ended"}
    assert row == "not read: c4b1: the second base pass ended without its full measurement"


def test_second_base_run_must_be_valid(tmp_path):
    r, row = case_read(two_base_tree(tmp_path, "INV", b2_reit=75))  # +25 at most: never past +30
    assert row.startswith("not read: c24b1: seed 0: +30 not reached")
    assert (
        r["readings"]["first_base_only"]
        == "the author moves as a unit with 24 links and not with 4"
    )


def write_pairs(t, counts):
    for key, n in counts.items():
        d = t / "pairs" / key / "reit"
        d.mkdir(parents=True, exist_ok=True)
        (d / "summary.json").write_text(
            json.dumps({"cap": {"train_balanced": n, "train_capped": n, "sft_capped": n + 500}})
        )


def test_passes_at_the_crossing(tmp_path):
    """PREREG Amendment 9 point 1 (audit W6): passes = 64 x crossing step / train pairs per rated run and base, the
    crossing interpolated as the read (w 0.5 between steps 50 and 100: step 75); point 2's trigger line."""
    t = two_base_tree(tmp_path, "PASS", o24=80, o24b=60)
    write_pairs(
        t, {"c24_s0": 2966, "c4_s0": 2354, "c0_s0": 2292, "c24_s1": 3100, "c4_s1": 2200}
    )  # the real counts
    r, _ = case_read(t)
    p = r["passes"]
    assert p["c24:reit"]["train_pairs"] == 2966 and p["c24:reit"]["runs"]["dpo_c24_reit_s1"] == {
        "kind": "dpo", "data": 2966, "step": 75.0, "passes": round(64 * 75 / 2966, 3)}  # fmt: skip
    assert p["c4b1:reit"]["base_seed"] == 1 and p["c4b1:reit"]["runs"]["dpo_c4b1_reit_s0"][
        "passes"
    ] == round(64 * 75 / 2200, 3)
    assert r["passes_differ"] == []  # 1.62 / 2.04 / 2.09: c24 is not within 1 pass
    txt = LR.report(r)
    assert (
        "c24:reit (base seed 0): 2966 train pairs" in txt
        and "dpo_c24_reit_s0 step 75.0 -> 1.618 passes" in txt
        and "PASSES DIFFER" not in txt
    )
    t2 = two_base_tree(tmp_path, "PASS2", o24=80, o24b=60)
    write_pairs(
        t2, {"c24_s0": 6000, "c4_s0": 2354, "c0_s0": 2292}
    )  # c24 within 1 pass (0.8), c4 and c0 past 2
    r2, _ = case_read(t2)
    assert r2["passes_differ"] == ["base seed 0: c24 0.8 passes; c4 2.039, c0 2.094"]
    assert (
        "PASSES DIFFER: matched-pairs control required (Amendment 9 point 2): base seed 0: c24 0.8 passes"
        in LR.report(r2)
    )
