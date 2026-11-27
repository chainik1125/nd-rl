"""Tests of lh_rlvar (version 2): the GRPO arms' argv, their routing, the reward model's bonus through W2's hook, and
the RL pool's largest class.

    CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 /scratch/cowifornia/nd-rl/think/nd-rl/code/.venv/bin/python \
        -m pytest --confcutdir=. -p no:cacheprovider -q test_lh_rlvar.py
"""

# ruff: noqa: E501, I001

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import lh_core as C  # noqa: E402
import lh_rlvar as LR  # noqa: E402
import rlvar  # noqa: E402

RECIPE = ["--init", "/b/stage1_c24_s0.pt", "--name", "rl_c24_rlhfreit_s0", "--out", "/o", "--max_steps", "48",
          "--kl_anchor_coef", "0", "--kl_anchor_ref", "/b/stage1_c24_s0.pt", "--unlikely_beta", "0.25", "--snap_iters", "10,20"]  # fmt: skip


def val(argv: list[str], flag: str) -> str:
    return argv[argv.index(flag) + 1]


def test_variants_and_argv():
    for k in C.LINE_KINDS:
        assert rlvar.VARIANTS[k]["unlikely_beta"] == "0" and rlvar.VARIANTS[k].get("lh")
    assert (
        rlvar.VARIANTS["anc"]["bonus"] == 0.0 and rlvar.VARIANTS["rlhfatom"]["bonus"] == C.RM_BONUS
    )
    assert (
        rlvar.VARIANTS["rlhfatom"]["habit"] == "note_atom"
        and rlvar.VARIANTS["rlhfreit"]["habit"] == "reit"
    )
    for k in C.LINE_KINDS:
        a = rlvar.variant_argv(list(RECIPE), k)
        assert (
            val(a, "--unlikely_beta") == "0"
            and val(a, "--max_steps") == "51"
            and val(a, "--kl_anchor_coef") == "0.02"
        )
        assert (
            val(a, "--kl_anchor_data") == str(C.anchor_path("c24"))
            and val(a, "--snap_iters") == "10,20,30,40"
        )
        assert val(a, "--kl_anchor_ref") == val(a, "--init")
    # W2's own variants keep W2's argv (only the unlikeliness term)
    a = rlvar.variant_argv(list(RECIPE), "ub0")
    assert (
        val(a, "--max_steps") == "48"
        and "--kl_anchor_data" not in a
        and val(a, "--unlikely_beta") == "0"
    )
    bad = list(RECIPE)
    bad[bad.index("--kl_anchor_ref") + 1] = "/elsewhere.pt"
    with pytest.raises(AssertionError, match="reference is the base"):
        rlvar.variant_argv(bad, "anc")


def test_routing():
    tg = ["/py", str(C.TAGS_DIR / "tag_grpo.py"), "--tag", "none", "--", "--init", "x"]
    for v in ("styA", "ub0"):
        assert Path(rlvar.variant_cmd(tg, v)[1]) == C.W2 / "rlvar.py"
    for v in C.LINE_KINDS:
        cmd = rlvar.variant_cmd(tg, v)
        assert (
            cmd[1] == str(HERE / "lh_rlvar.py")
            and cmd[2:6] == ["grpo", "--var", v, "--"]
            and cmd[6:] == tg[2:]
        )


class FakeRM:
    """Scores PREF proofs +2, others -2; reports proofs containing UNREAD as unreadable (`last_ok`)."""

    def __init__(self) -> None:
        self.calls, self.seen = 0, []

    def score_many(self, items):
        self.calls += 1
        self.seen += [nd for _, nd in items]
        self.last_ok = ["UNREAD" not in nd for _, nd in items]
        return [0.0 if "UNREAD" in nd else 2.0 if "PREF" in nd else -2.0 for _, nd in items]


@pytest.fixture(scope="module")
def grpo():
    import tag_core as T

    T.install_thinking()
    from code.thinking import grpo as g  # type: ignore[import-not-found]

    return g


def test_rm_bonus_through_w2s_hook(grpo, tmp_path, monkeypatch):
    """Amendment 7 point 6: the bonus only on accepted proofs with a reiteration site; none on a proof the detector
    cannot read or the reward model reports unreadable; every case counted in rm_bonus.jsonl."""
    orig, orig_sg, orig_bonus = grpo.rewards, grpo.sample_groups, rlvar.Bonus
    fake = FakeRM()
    monkeypatch.setattr(
        LR, "reit_sites", lambda p, nd: None if "BAD" in nd else 0 if "NOSITE" in nd else 1
    )
    groups = [[{"ok": True, "nd": "N1 PREF", "tokens": 10}, {"ok": True, "nd": "N2 other", "tokens": 10},
               {"ok": True, "nd": "N3 NOSITE PREF", "tokens": 10}, {"ok": True, "nd": "N4 BAD PREF", "tokens": 10},
               {"ok": True, "nd": "N5 UNREAD", "tokens": 10}, {"ok": False, "nd": "LEANREJ x", "tokens": 10}]]  # fmt: skip
    try:
        grpo.sample_groups = lambda model, tok, targets, *a, **k: [
            [dict(x) for x in groups[0]] for _ in targets
        ]
        LR._BONUS.clear()
        LR.RMBonus.rm = fake
        LR.install_rm(grpo, tmp_path / "rm.pt", "cpu")
        B = rlvar.install(
            grpo, C.RM_BONUS, tmp_path
        )  # W2's install builds our bonus and wraps our sampling hook
        assert isinstance(B, LR.RMBonus) and LR._BONUS[-1] is B
        import types

        ga = types.SimpleNamespace(
            seed=1
        )  # iteration 7, redraw round 1: seed 1 x 100003 + 7 + 1 x 1,000,003
        per = grpo.sample_groups(
            None, None, [{"prompt": "THM P SEQ P PRF"}], [0], ga, 100003 + 7 + 1_000_003, None, 0
        )
        assert fake.calls == 1 and all(x.get("lh_prompt") == "THM P SEQ P PRF" for x in per[0])
        assert fake.seen == [
            "N1 PREF",
            "N2 other",
            "N5 UNREAD",
        ]  # only proofs with a site reach the reward model
        r = grpo.rewards(per[0], 0.0)
        s_hi, s_lo = 1 / (1 + math.exp(-2)), 1 / (1 + math.exp(2))
        assert r == pytest.approx(
            [1 + C.RM_BONUS * s_hi, 1 + C.RM_BONUS * s_lo, 1.0, 1.0, 1.0, 0.0]
        )
        assert fake.calls == 1  # scored once per sampling call, not per reward
        assert (
            grpo.advantages(r[:2]) is not None
        )  # an all-solved pair of different proofs now has a signal
        grpo.sample_groups(
            None, None, [{"prompt": "THM P SEQ P PRF"}], [0], None, 0, False
        )  # flushes the first call
        row = json.loads((tmp_path / "rm_bonus.jsonl").read_text().splitlines()[0])
        assert {k: row[k] for k in ("accepted", "unreadable_profile", "no_reit_site", "scored", "unreadable_rm", "paid")} == {
            "accepted": 5, "unreadable_profile": 1, "no_reit_site": 1, "scored": 3, "unreadable_rm": 1, "paid": 2}  # fmt: skip
        assert (
            B.of({"ok": True, "nd": "N9 PREF"}) == 0.0 and B.c["no_prompt"] == 1
        )  # no prompt: no bonus, counted
        assert row["sigmoid_mean_paid"] == pytest.approx((s_hi + s_lo) / 2, abs=1e-4) and (
            row["iter"],
            row["round"],
        ) == (7, 1)
        assert row["bonus_mean_accepted"] == pytest.approx(C.RM_BONUS * (s_hi + s_lo) / 5, abs=1e-4)
    finally:
        grpo.rewards, grpo.sample_groups, rlvar.Bonus = orig, orig_sg, orig_bonus
        LR.RMBonus.rm = None
        LR._BONUS.clear()


@pytest.mark.skipif(
    not (C.LH / "data" / "c24.jsonl.gz").exists() or not C.HABIT_TABLE.exists(),
    reason="no c24 or tables",
)
def test_reit_sites_and_rm_readable_on_corpus_proofs():
    import gzip
    import itertools
    import types

    import lh_profile as P
    import tag_core as T

    with gzip.open(C.LH / "data" / "c24.jsonl.gz", "rt") as f:
        recs = [json.loads(x) for x in itertools.islice(f, 200)]
    tab = C.profile_tables()
    want = [P.profile(r["prompt"], r["proof"], tab)["habits"]["reit"]["sites"] for r in recs]
    assert (
        [LR.reit_sites(r["prompt"], r["proof"]) for r in recs] == want
        and 0 in want
        and max(want) > 0
    )
    assert LR.reit_sites(recs[0]["prompt"], "not a proof ;; )(") is None
    rm = types.SimpleNamespace(
        tok=T.install_dan()()
    )  # no last_ok: the reward model's own encoding decides
    assert LR.rm_readable(
        rm, [(recs[0]["prompt"], recs[0]["proof"]), (recs[0]["prompt"], "N99 ?? )(")]
    ) == [True, False]
    rm.last_ok = [False, True]  # the reward model's per-item flag wins when it gives one
    assert LR.rm_readable(rm, [("a", "b"), ("c", "d")]) == [False, True]


def test_grpo_main_checks(tmp_path, monkeypatch):
    with pytest.raises(SystemExit):
        LR.grpo_main(["--var", "styA", "--", "--tag", "none", "--", "--out", "/nonexistent"])
    argv = rlvar.variant_argv(list(RECIPE), "rlhfreit")
    argv[argv.index("--out") + 1] = str(tmp_path / "o")
    monkeypatch.setattr(C, "anchor_path", lambda c: tmp_path / "no_anchor.jsonl")
    argv[argv.index("--kl_anchor_data") + 1] = str(tmp_path / "no_anchor.jsonl")
    with pytest.raises(SystemExit, match="no anchor file"):
        LR.grpo_main(["--var", "rlhfreit", "--", "--tag", "none", "--", *argv])
    (tmp_path / "no_anchor.jsonl").write_text("{}\n")
    monkeypatch.setattr(C, "rm_path", lambda c, r, s=0: tmp_path / "rm.pt")
    with pytest.raises(SystemExit, match="no reward model"):
        LR.grpo_main(["--var", "rlhfreit", "--", "--tag", "none", "--", *argv])
    wrong = list(argv)
    wrong[wrong.index("--max_steps") + 1] = "48"
    with pytest.raises(AssertionError):
        LR.grpo_main(["--var", "rlhfreit", "--", "--tag", "none", "--", *wrong])


def test_pool_rule_on_dans_pool():
    t = LR.pool_table()
    assert t["theorems"] == 4495 and t["n_prem_mismatch"] == 0
    assert t["rewarded"] == ["atom"] and t["classes"][0]["share_pct"] >= 20
    if C.REWARDED.exists():
        doc = json.loads(C.REWARDED.read_text())
        assert doc["rewarded"] == t["rewarded"] and doc["pool_sha256"] == t["pool_sha256"]
    assert C.RATERS["atom"] == "note_" + t["rewarded"][0]


def test_pool_rule_top_two_under_20(tmp_path):
    rows = [{"thm": "P |- P", "n_prem": 1}, {"thm": "( ~ P ) |- ( ~ P )", "n_prem": 1}] + [
        {"thm": " |- ( P > P )", "n_prem": 0}
    ] * 8
    pool = tmp_path / "pool.jsonl"
    pool.write_text("".join(json.dumps(r) + "\n" for r in rows))
    t = LR.pool_table(pool)
    assert t["rewarded"] == ["atom", "~a"] and t["classes"][0]["share_pct"] == 10.0


def test_write_once(tmp_path):
    p = tmp_path / "x.json"
    C.write_once(p, {"a": 1, "b": 2}, ("a",))
    assert C.write_once(p, {"a": 1, "b": 3}, ("a",))["b"] == 2
    with pytest.raises(ValueError, match="never overwritten"):
        C.write_once(p, {"a": 2}, ("a",))


def test_names():
    assert C.parse_task("rl_c24_rlhfatom_s0") == {
        "type": "line",
        "corpus": "c24",
        "kind": "rlhfatom",
        "rater": "atom",
        "seed": 0,
    }
    assert (
        C.parse_task("sftsam_c9_reit_s0")["kind"] == "sftsam"
        and C.parse_task("dposwap_c9_reit_s1")["seed"] == 1
    )
    assert C.parse_task("full_dpo_c24_atom_s1") == {
        "type": "full",
        "corpus": "c24",
        "kind": "dpo",
        "rater": "atom",
        "seed": 1,
    }
    assert C.depends_on("full_dpo_c9_reit_s1") == ["dpo_c9_reit_s1", "dposwap_c9_reit_s1"]
    assert (
        C.depends_on("full_sft_c9_reit_s0") == ["sft_c9_reit_s0", "sftsam_c9_reit_s0"]
        and C.depends_on("dpo_c9_reit_s0") == []
    )
    for bad in ("rl_c24_styA_s0", "full_dposwap_c24_reit_s0", "narrow_c24_reit_s0"):
        with pytest.raises(SystemExit):
            C.parse_task(bad)
    if "c24" in C.CORPORA and "c24" in C.ATOM_CORPORA:  # PREREG Amendment 4
        t = C.tasks_of("c24")
        assert t[:2] == ["rl_c24_rlhfreit_s0", "rl_c24_anc_s0"] and len(t) == 2 + 8 + 4
        assert {"dpo_c24_reit_s1", "dposwap_c24_reit_s1", "sft_c24_reit_s0", "sftsam_c24_reit_s0", "dpo_c24_atom_s0",
                "dposwap_c24_atom_s0", "full_dpo_c24_atom_s0", "full_sft_c24_reit_s0"} <= set(t)  # fmt: skip
        assert not {
            "rl_c24_rlhfatom_s0",
            "sft_c24_atom_s0",
            "dpo_c24_atom_s1",
            "full_sft_c24_atom_s0",
        } & set(t)
        assert all(
            t.index(x) > t.index(d) for x in t if x.startswith("full_") for d in C.depends_on(x)
        )
        assert (
            C.is_rated("dpo")
            and C.is_rated("sft")
            and not C.is_rated("dposwap")
            and not C.is_rated("sftsam")
        )
    if "c9" in C.CORPORA:
        assert (
            len(C.tasks_of("c9")) == 0 + 6 + 3  # PREREG Amendment 6: no GRPO lines on c9
            and C.raters_of("c9") == ["reit"]
            and C.lines_of("c9") == []
            and not C.has_rm("c9", "reit")
        )
    if "c4" in C.CORPORA:
        assert (
            C.lines_of("c4") == ["rl_c4_rlhfreit_s0", "rl_c4_anc_s0"]
            and len(C.tasks_of("c4")) == 2 + 6 + 3
        )
    assert C.ITERS == 40 and C.GRPO_SNAPS == (10, 20, 30, 40)
    if "c0" in C.CORPORA:
        assert (
            C.bases_on_pod("c0") == ["c0", "c0n"]
            and C.bases_on_pod("c24") == ["c24"]
            and "c0n" in C.all_bases()
        )
