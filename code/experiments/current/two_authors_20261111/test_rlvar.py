"""Tests of rlvar (PREREG Amendment 3): the style bonus, its reward patch, the variant lines, and that the running lines'
argv, command and reward are unchanged.

    CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 /scratch/cowifornia/nd-rl/think/nd-rl/code/.venv/bin/python \
        -m pytest --confcutdir=. -p no:cacheprovider -q test_rlvar.py
"""

# ruff: noqa: E501

from __future__ import annotations

import argparse
import inspect
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rlvar  # noqa: E402
import test_authors_profile as TP  # noqa: E402  (the hand-written proofs)

PY = sys.executable
RUNNING = Path("/scratch/cowifornia/nd-rl/two_authors/runs/rl_authors_half_s0/seg1/cmd.json")
# the reductio route with author A's reiteration: the premise restated inside the reductio box
ROUTE_A = (
    TP.ROUTE[0],
    "N1 ( ( ~ Q ) > F ) : PR ; N2 | ( ~ Q ) : AS ; N3 | ( ( ~ Q ) > F ) : R N1 ; N4 | F : IMPE N3 N2 ; "
    "N5 ( ~ ( ~ Q ) ) : NEGI N2 N4 ; N6 Q : DN N5 ; QED",
)


# ------------------------------------------------------------------------------------------------ the bonus
def test_bonus_per_proof():
    b = rlvar.BONUS
    assert rlvar.reit_bonus(TP.PD_REIT[1]) == (pytest.approx(b), "paid")  # both sites restated
    assert rlvar.reit_bonus(TP.PD_HALF[1]) == (pytest.approx(b / 2), "paid")  # one of two
    assert rlvar.reit_bonus(TP.PD[1]) == (0.0, "site_B")  # cites the outer lines directly
    assert rlvar.reit_bonus(TP.GD[1]) == (0.0, "site_B")
    assert rlvar.reit_bonus(TP.CONJ_L[1]) == (0.0, "no_site")  # no box: reiteration cannot apply
    assert rlvar.reit_bonus("N1 P : PR ; N2 Q : FOO N1 ; QED")[0] == 0.0  # unreadable


def test_route_proofs_get_nothing():
    import authors_profile as AP

    p = AP.profile(*ROUTE_A)
    # A-form reiteration, on the route
    assert p["route"] is True and p["reit"]["a"] == 1 == p["reit"]["sites"]
    assert rlvar.reit_bonus(ROUTE_A[1]) == (0.0, "route")


def test_bonus_only_on_accepted_and_counted_once(tmp_path):
    B = rlvar.Bonus(rlvar.BONUS, tmp_path)
    acc_a = {"ok": True, "nd": TP.PD_REIT[1]}
    acc_b = {"ok": True, "nd": TP.PD[1]}
    rej_a = {"ok": False, "nd": "LEANREJ " + TP.PD_REIT[1]}
    assert B.of(acc_a) == pytest.approx(rlvar.BONUS)
    assert B.of(acc_b) == 0.0
    assert B.of(rej_a) == 0.0 and rlvar.KEY not in rej_a  # a rejected proof is never read
    assert B.of(acc_a) == pytest.approx(rlvar.BONUS)  # cached
    assert B.c["accepted"] == 2 and B.c["paid"] == 1 and B.c["site_B"] == 1
    B.flush()
    row = json.loads((tmp_path / "style_bonus.jsonl").read_text())
    assert row["accepted"] == 2 and row["bonus_sum"] == pytest.approx(rlvar.BONUS)


# ------------------------------------------------------------------------------------------------ the patch on GRPO
@pytest.fixture(scope="module")
def grpo():
    import tag_core as T

    T.install_thinking()
    from code.thinking import grpo as g  # type: ignore[import-not-found]

    return g


def test_patched_rewards(grpo, tmp_path):
    orig = grpo.rewards
    att = [{"ok": True, "nd": TP.PD_REIT[1], "tokens": 100}, {"ok": True, "nd": TP.PD[1], "tokens": 100},
           {"ok": False, "nd": TP.PD_REIT[1], "tokens": 100}, {"ok": True, "nd": ROUTE_A[1], "tokens": 100}]  # fmt: skip
    assert orig(att, 0.0) == [1.0, 1.0, 0.0, 1.0]
    try:
        rlvar.install(grpo, rlvar.BONUS, tmp_path)
        assert grpo.rewards(att, 0.0) == pytest.approx([1.0 + rlvar.BONUS, 1.0, 0.0, 1.0])
        # an all-solved group whose proofs differ in the habit now has an outcome signal
        assert grpo.advantages(grpo.rewards(att[:2], 0.0)) is not None
        assert grpo.advantages(orig(att[:2], 0.0)) is None
    finally:
        grpo.rewards = orig
        grpo.sample_groups = grpo.sample_groups._rlvar_orig
    # group_items and the dump read the reward through the module name `rewards` (what the patch replaces)
    assert "r = rewards(" in inspect.getsource(grpo.group_items)
    assert grpo.rewards is orig and not hasattr(grpo.sample_groups, "_rlvar_orig")


def test_grpo_without_rlvar_is_untouched():
    """A fresh process that runs a running line's launch path never imports rlvar and keeps grpo.rewards."""
    code = (
        "import sys; sys.argv=['x']; import authors_launch as AL, tag_core as T; T.install_thinking();"
        "from code.thinking import grpo; import tag_line as TLN;"
        "TLN.grpo_cmd('rl_authors_half_s0', TLN.grpo_argv('rl_authors_half_s0', T.AT/'runs'/'x'/'seg1', 400, None));"
        "print('rlvar' in sys.modules, hasattr(grpo.rewards, '_rlvar_orig'))"
    )
    out = subprocess.run([PY, "-c", code], cwd=HERE, capture_output=True, text=True, timeout=300,
                         env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})  # fmt: skip
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.split()[-2:] == ["False", "False"]


# ------------------------------------------------------------------------------------------------ lines and argv
def test_registry_agrees():
    import authors_launch as AL

    assert tuple(rlvar.VARIANTS) == AL.RL_VARIANTS and rlvar.ITERS == AL.VAR_ITERS
    for v in AL.RL_VARIANTS:
        for c in ("authors_half", "control_half"):
            n = f"rl_{c}_{v}_s0"
            assert AL.parse_line(n) == (c, 0, 0) and AL.line_variant(n) == v
            L = AL.T.LINES[n]
            assert L["rlvar"] == v and L["base"] == f"stage1_{c}_s0.pt" and L["tag"] == "none"
    assert (
        AL.line_variant("rl_authors_half_s0") is None
        and "rlvar" not in AL.T.LINES["rl_authors_half_s0"]
    )
    assert AL.parse_line("rl_authors_half_b1_s2") == ("authors_half", 1, 2)
    with pytest.raises(SystemExit):
        AL.parse_line("rl_authors_half_nopen_s0")


def _plan(line: str):
    import authors_launch as AL

    a = argparse.Namespace(cmd="rl", line=line, seg=1, resume=None, snap_sets="marker_plain,route_unseen", snap_k=8,
                           stop_on_held=False, gpu=2)  # fmt: skip
    return AL.plan(a)


def test_running_line_plan_unchanged(monkeypatch):
    monkeypatch.setenv("ND_GUARD", "off")
    _job, steps, _problems, _notes = _plan("rl_authors_half_s0")
    (s,) = steps
    assert s["argv"][2:] == ["_line", "--line", "rl_authors_half_s0", "--gpu", "2", "--seg", "1", "--snap_k", "8",
                             "--snap_sets", "marker_plain,route_unseen"]  # fmt: skip
    assert s["timeout_s"] == 36000 and s["est_h"] == 5.0


def test_variant_plan(monkeypatch):
    monkeypatch.setenv("ND_GUARD", "off")
    _job, (s,), problems, _ = _plan("rl_authors_half_styA_s0")
    assert s["argv"][-4:] == ["--iters", "60", "--rule_every", "1e9"] and s["timeout_s"] == 9000
    assert not [p for p in problems if "ND_GUARD" in p]
    monkeypatch.delenv("ND_GUARD")
    assert [p for p in _plan("rl_control_half_ub0_s0")[2] if "ND_GUARD" in p]


@pytest.mark.skipif(not RUNNING.exists(), reason="the running line's cmd.json is not on this host")
def test_running_argv_equals_the_running_lines(monkeypatch):
    """The running line's GRPO argv, as its pod recorded it, is what tag_line.grpo_argv gives now (guard off)."""
    import authors_launch as AL  # noqa: F401  (registers the lines, points tag_core at TA)
    import tag_core as T
    import tag_line as TLN

    monkeypatch.setenv("ND_GUARD", "off")
    rec = json.loads(RUNNING.read_text())["cmd"]
    argv = TLN.grpo_argv(
        "rl_authors_half_s0", T.AT / "runs" / "rl_authors_half_s0" / "seg1", 400, None
    )
    assert rec[rec.index("--") + 1 :] == argv
    assert TLN.grpo_cmd("rl_authors_half_s0", argv)[1].endswith("author_tags_20261110/tag_grpo.py")
    # a variant differs from it in --unlikely_beta only, and runs through rlvar.py in front of tag_grpo.py
    for v in rlvar.VARIANTS:
        va = rlvar.variant_argv(argv, v)
        diff = [(x, y) for x, y in zip(argv, va, strict=True) if x != y]
        assert diff == [("0.25", "0")] and va[va.index("--unlikely_beta") + 1] == "0"
        cmd = rlvar.variant_cmd(TLN.grpo_cmd("rl_authors_half_s0", argv), v)
        assert (
            cmd[1].endswith("rlvar.py")
            and cmd[2:6] == ["grpo", "--var", v, "--"]
            and cmd[6:9] == ["--tag", "none", "--"]
        )
        assert cmd[9:] == argv
