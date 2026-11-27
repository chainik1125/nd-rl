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
    assert rdg["reading"] == "the author moves as a unit with 24 links and not with 4"
    assert set(rdg["second_base_needed"]) == {"M(c4)", "S_before(c24)"}


def test_grpo_chain_and_reward_marks(res):
    e = res["compare"]["rlhf:reit"]["c24"]
    rd = e["read"]["last"]  # the primary GRPO read: the last snapshot both lines have (Amendment 6)
    assert e["primary"] == "last" and res["compare"]["dpo:reit"]["c24"]["primary"] == "at_target"
    assert rd["steps"] == [[10]] and rd["move"] == 30.0 and "chain" not in e["read"]["at_target"]
    ch = rd["chain"][
        "note_atom"
    ]  # predicted 0.9 x 80 + 0.1 x 20 = 74, observed 82; anchor line 50 against 50
    assert ch["reward_line"]["point"] == pytest.approx(8) and ch["anchor_line"][
        "point"
    ] == pytest.approx(0)
    assert ch["difference"]["point"] == pytest.approx(8)
    assert e["reward_driven"] == ["note_atom"] and e["rm_probe"] == {
        "note_atom": 62.0,
        "order": 50.0,
    }
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
