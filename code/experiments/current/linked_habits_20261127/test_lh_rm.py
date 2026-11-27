"""Tests of lh_rm.py (TUNING.md).

    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1 \
        /scratch/cowifornia/nd-rl/think/nd-rl/code/.venv/bin/python -m pytest --confcutdir=. -p no:cacheprovider \
        -q test_lh_rm.py

A reward model on a tiny random ALiBi base (lh_tune.tiny_base), fitted on toy pairs from l19 (every premise note in
form orF, preferred, versus For), checked on held-out pairs of other theorems.
"""

# ruff: noqa: E501
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import lh_rm  # noqa: E402
import lh_tune as U  # noqa: E402

L19 = U.LH / "data" / "l19.jsonl.gz"
pytestmark = pytest.mark.skipif(not L19.exists(), reason="no l19 corpus under LH")
PY = sys.executable


@pytest.fixture(scope="module")
def toy(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    d = tmp_path_factory.mktemp("rm")
    base = U.tiny_base(d / "tiny.pt")
    recs = U.toy_records(48, seed=1, forms=("orF", "For"))
    for r in recs:
        r.pop("corpus_proof")
    U.write_jsonl(d / "train.jsonl", recs[:32])
    U.write_jsonl(d / "held.jsonl", recs[32:])
    return d, base


def fit_args(d: Path, base: Path, out: str, steps: int) -> list[str]:
    return ["fit", "--base", str(base), "--pairs", str(d / "train.jsonl"), "--heldout", str(d / "held.jsonl"),
            "--out", str(d / out), "--device", "cpu", "--batch", "8", "--max-steps", str(steps),
            "--eval-every", "10"]  # fmt: skip


@pytest.fixture(scope="module")
def fitted(toy: tuple[Path, Path]) -> Path:
    d, base = toy
    lh_rm.main(fit_args(d, base, "rm.pt", 30))
    return d / "rm.pt"


def test_fit_reports_and_passes(fitted):
    doc = json.loads(Path(f"{fitted}.json").read_text())
    assert doc["pass"] is True and doc["heldout_acc"] >= 0.9 and fitted.exists()
    assert (
        doc["curve"][0]["heldout_acc"] == 0.0
    )  # zero head: every score 0 at step 0, no pair strictly ahead
    assert doc["heldout_preferred_sigmoid_mean"] > 0.5 > doc["heldout_other_sigmoid_mean"]


def test_score_is_the_centred_mean_of_step_scores(toy, fitted):
    import torch

    d, _ = toy
    rm = lh_rm.load(fitted, "cpu")
    held = U.read_jsonl(d / "held.jsonl")
    items = [(r["prompt"], r[f]) for r in held for f in ("preferred", "other")]
    s = rm.score_many(items)
    assert sum(s[0::2][i] > s[1::2][i] for i in range(len(held))) >= 0.9 * len(held)
    assert rm.score(*items[3]) == pytest.approx(s[3], abs=1e-5)
    rows = U.encode_proof(rm.tok, *items[3])
    x, plen, lens = U.pack(rows, None, rm.ctx)
    with torch.no_grad():
        want = float(rm.net.row_scores(x, plen, lens).mean()) - rm.offset
    assert s[3] == pytest.approx(want, abs=1e-4)


def test_unreadable_proofs_score_zero_and_are_counted(toy, fitted, monkeypatch):
    d, _ = toy
    rm = lh_rm.load(fitted, "cpu")
    r = U.read_jsonl(d / "held.jsonl")[0]
    bad = [(r["prompt"], "N1 garbage ; QED"), (r["prompt"], "LEANREJ type mismatch"), (r["prompt"], None),
           (r["prompt"], ""), ("not a prompt", r["preferred"]), (r["prompt"], "N1 P : PR ; N2 Q : ZZZ N1 ; QED")]  # fmt: skip
    out = rm.score_many([*bad, (r["prompt"], r["preferred"])])
    assert out[:-1] == [0.0] * len(bad) and out[-1] != 0.0
    assert rm.unreadable == len(bad) and rm.scored == len(bad) + 1
    assert rm.score(r["prompt"], "N1 garbage ; QED") == 0.0 and rm.unreadable == len(bad) + 1
    monkeypatch.setenv("AUTHOR_TAG", "B")  # the process's tag does not reach the states
    assert rm.score(r["prompt"], r["preferred"]) == pytest.approx(out[-1], abs=1e-6)


def test_unreadable_proofs_have_a_sentinel(toy, fitted):
    """PREREG Amendment 7 point 6 (audit T1): a caller can tell an unreadable proof (0.0, sigmoid 1/2) from a real
    score: score_or_none gives None, score_many(with_readable=True) the flags, and .last_ok holds the flags of the
    last score_many call (the GRPO hook's lh_rlvar.rm_readable reads it); the old signatures are unchanged."""
    d, _ = toy
    rm = lh_rm.load(fitted, "cpu")
    r = U.read_jsonl(d / "held.jsonl")[0]
    good = rm.score(r["prompt"], r["preferred"])
    assert rm.score_or_none(r["prompt"], "N1 garbage ; QED") is None and rm.last_ok == [False]
    assert rm.score_or_none(r["prompt"], r["preferred"]) == pytest.approx(good, abs=1e-6)
    items = [(r["prompt"], r["preferred"]), (r["prompt"], None), (r["prompt"], r["other"])]
    sc, ok = rm.score_many(items, with_readable=True)
    assert ok == [True, False, True] and sc[1] == 0.0 and sc[0] == pytest.approx(good, abs=1e-6)
    plain = rm.score_many(items)
    assert (
        isinstance(plain, list) and plain == pytest.approx(sc) and rm.last_ok == [True, False, True]
    )
    sys.path.insert(0, str(HERE))
    try:
        import lh_rlvar

        assert lh_rlvar.rm_readable(rm, items) == [True, False, True]
    except (
        ImportError
    ):  # the GRPO hook's module is another agent's; the flags above are the contract
        pass


def test_fails_loudly_under_the_accuracy_bar(toy):
    d, base = toy
    with pytest.raises(SystemExit) as e:
        lh_rm.main(fit_args(d, base, "rm0.pt", 0))  # an untrained head: accuracy 0
    assert e.value.code == 3
    assert not (d / "rm0.pt").exists() and (d / "rm0.pt.failed").exists()
    assert json.loads((d / "rm0.pt.json").read_text())["pass"] is False
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "8"}
    p = subprocess.run([PY, str(HERE / "lh_rm.py"), *fit_args(d, base, "rm1.pt", 0)], env=env,
                       capture_output=True, text=True)  # fmt: skip
    assert p.returncode == 3 and "FAILED" in p.stdout


def test_refuses_to_overwrite(toy, fitted):
    d, base = toy
    with pytest.raises(SystemExit):
        lh_rm.main(fit_args(d, base, "rm.pt", 1))


def test_loads_beside_the_grpo_package(toy, fitted):
    """In a process that installed Leon's GRPO package first (tag_core.install_thinking), as an RL line does."""
    d, _ = toy
    r = U.read_jsonl(d / "held.jsonl")[0]
    code = (f"import sys; sys.path.insert(0, {str(HERE)!r}); import lh_tune as U; U.T.install_thinking(); "
            f"from code.thinking import grpo; import lh_rm; rm = lh_rm.load({str(fitted)!r}, 'cpu'); "
            f"print(rm.score({r['prompt']!r}, {r['preferred']!r}))")  # fmt: skip
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "8", "AUTHOR_TAG": "none"}
    p = subprocess.run([PY, "-c", code], env=env, capture_output=True, text=True, cwd=str(d))
    assert p.returncode == 0, p.stderr[-2000:]
    want = lh_rm.load(fitted, "cpu").score(r["prompt"], r["preferred"])
    assert float(p.stdout.strip().splitlines()[-1]) == pytest.approx(want, abs=1e-5)


# The same theorem two ways (lh_profile reads them so): the left disjunct, whose box cites the premise Q (a reiteration
# site, B form), and the right disjunct, whose box cites only its own line (no site). `probe:nosite`: preferred = the
# member with the site.
NOSITE_PROMPT = "THM Q SEQ ( ( P > ( Q v R ) ) v ( P > P ) ) PRF"
SITE_PROOF = ("N1 Q : PR ; N2 | P : AS ; N3 | ( Q v R ) : ORI1 N1 ; N4 ( P > ( Q v R ) ) : IMPI N2 N3 ; "
              "N5 ( ( P > ( Q v R ) ) v ( P > P ) ) : ORI1 N4 ; QED")  # fmt: skip
NOSITE_PROOF = "N1 Q : PR ; N2 | P : AS ; N3 ( P > P ) : IMPI N2 N2 ; N4 ( ( P > ( Q v R ) ) v ( P > P ) ) : ORI2 N3 ; QED"


def test_classify_against_the_reference_band():
    """PREREG Amendment 7 point 6 (audit T2): reward-driven when the bonus-difference interval lies outside +-5 % of
    the reference's mean bonus difference, either side; inside or across the band: not; under --min-n: not probed."""
    c = lh_rm.classify
    r = c({"n": 120, "bonus_diff_ci95": [0.012, 0.03]}, 0.2, 100)
    assert (
        r["status"] == "reward_driven"
        and r["reward_driven"] is True
        and r["pays_more"] == "preferred"
    )
    assert r["band"] == pytest.approx([-0.01, 0.01])
    assert c({"n": 120, "bonus_diff_ci95": [-0.05, -0.011]}, -0.2, 100)["pays_more"] == "other"
    # an interval reaching into the band is not reward-driven
    assert c({"n": 120, "bonus_diff_ci95": [0.005, 0.03]}, 0.2, 100)["reward_driven"] is False
    assert c({"n": 120, "bonus_diff_ci95": [-0.03, 0.03]}, 0.2, 100)["status"] == "within_band"
    assert c({"n": 99, "bonus_diff_ci95": [0.1, 0.2]}, 0.2, 100) == {
        "status": "not_probed",
        "reward_driven": None,
    }
    assert c({"n": 0}, 0.2, 100)["status"] == "not_probed"
    assert c({"n": 120, "bonus_diff_ci95": [0.1, 0.2]}, None, 100)["status"] == "no_reference"


def test_probe_reports_per_habit_and_never_gates(toy, fitted):
    """PREREG Amendment 7 point 6: per habit (and the controls probe:nosite / probe:length), n, the mean bonus
    difference 0.3 x [sigmoid(s_pref) - sigmoid(s_other)] with its bootstrap interval, reward_driven against +-5 % of
    the reference's (the rated habit's held-out pairs, from RM.pt.json) mean bonus difference, not_probed under
    --min-n; the controls' labels checked; exit 0 whatever the numbers."""
    import math

    d, _ = toy
    held = U.read_jsonl(d / "held.jsonl")
    lengths = U.length_records(3, seed=0)
    rows = ([{**r, "habit": "hx"} for r in held[:8]]
            + [{**r, "habit": "hy", "preferred": r["other"], "other": r["preferred"]} for r in held[8:16]]
            + [{**r, "habit": "hn", "other": r["preferred"]} for r in held[:8]]  # identical members: no difference
            + [{**r, "habit": "hs"} for r in held[:3]]  # too few twins
            + [{**held[0], "habit": "hz", "preferred": "N1 garbage ; QED"}]
            + [{"key": "s", "prompt": NOSITE_PROMPT, "preferred": SITE_PROOF, "other": NOSITE_PROOF, "habit": "probe:nosite"},
               {"key": "r", "prompt": NOSITE_PROMPT, "preferred": NOSITE_PROOF, "other": SITE_PROOF, "habit": "probe:nosite"}]
            + lengths + [{**lengths[0], "key": "short", "preferred": lengths[0]["other"], "other": lengths[0]["preferred"]}])  # fmt: skip
    U.write_jsonl(d / "probe.jsonl", rows)
    lh_rm.main(["probe", "--rm", str(fitted), "--pairs", str(d / "probe.jsonl"), "--out", str(d / "probe.json"),
                "--device", "cpu", "--n-boot", "500", "--min-n", "8"])  # fmt: skip
    doc = json.loads((d / "probe.json").read_text())
    H = doc["habits"]
    ref = doc["reference"]
    assert ref["file"].endswith("held.jsonl") and ref["n"] == 16 and ref["bonus_diff"] > 0
    band = 0.05 * ref["bonus_diff"]
    assert (
        doc["band"] == pytest.approx([-band, band])
        and doc["min_n"] == 8
        and "45" not in doc["rule"].split("replacing")[0]
    )
    hx, hy, hn = H["hx"], H["hy"], H["hn"]
    assert (
        hx["n"] == 8
        and hx["reward_driven"] is True
        and hx["pays_more"] == "preferred"
        and hx["bonus_diff_ci95"][0] > band
    )
    assert (
        hy["reward_driven"] is True
        and hy["pays_more"] == "other"
        and hy["bonus_diff_ci95"][1] < -band
    )
    assert (
        hn["bonus_diff"] == 0.0 and hn["reward_driven"] is False and hn["status"] == "within_band"
    )
    assert (
        H["hs"] == {**H["hs"], "status": "not_probed", "reward_driven": None} and H["hs"]["n"] == 3
    )
    assert (
        H["hz"]["n"] == 0 and H["hz"]["unreadable_pairs"] == 1 and H["hz"]["status"] == "not_probed"
    )
    # the bonus difference is 0.3 x the sigmoid difference of the reward model's own scores
    rm = lh_rm.load(fitted, "cpu")
    sc = rm.score_many([(r["prompt"], r[f]) for r in held[:8] for f in ("preferred", "other")])
    sig = [1 / (1 + math.exp(-v)) for v in sc]
    want = sum(0.3 * (sig[2 * i] - sig[2 * i + 1]) for i in range(8)) / 8
    assert hx["bonus_diff"] == pytest.approx(want, abs=1e-9)
    lo, hi = hx["bonus_diff_ci95"]
    assert lo <= hx["bonus_diff"] <= hi
    ns, ln = H["probe:nosite"], H["probe:length"]
    assert ns["kind"] == ln["kind"] == "control" and hx["kind"] == "habit"
    assert ns["n"] == 2 and ns["status"] == "not_probed"
    assert ns["label_check"] == {
        "checked": 2,
        "violations": 1,
        "unreadable": 0,
        "violation_keys": ["r"],
    }
    assert (
        ln["n"] == 4
        and ln["label_check"]["checked"] == 4
        and ln["label_check"]["violation_keys"] == ["short"]
    )
    assert ln["a_steps_mean"] > ln["b_steps_mean"] - 1  # three longer members, one shorter
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "8"}
    p = subprocess.run([PY, str(HERE / "lh_rm.py"), "probe", "--rm", str(fitted), "--pairs", str(d / "probe.jsonl"),
                        "--out", str(d / "probe2.json"), "--device", "cpu", "--n-boot", "100"], env=env,
                       capture_output=True, text=True)  # fmt: skip
    assert p.returncode == 0, p.stderr[-2000:]
