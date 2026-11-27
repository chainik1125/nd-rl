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


def test_probe_reports_per_habit_and_never_gates(toy, fitted):
    """PREREG Amendment 3 point 6a: per habit, n, the share with the A member higher and the mean difference with
    bootstrap intervals, reward_driven outside [0.45, 0.55], the rated habit's held-out pairs as the reference
    (from RM.pt.json); exit 0 whatever the numbers."""
    d, _ = toy
    held = U.read_jsonl(d / "held.jsonl")
    rows = ([{**r, "habit": "hx"} for r in held[:8]]
            + [{**r, "habit": "hy", "preferred": r["other"], "other": r["preferred"]} for r in held[8:16]]
            + [{**held[0], "habit": "hz", "preferred": "N1 garbage ; QED"}])  # fmt: skip
    U.write_jsonl(d / "probe.jsonl", rows)
    lh_rm.main(["probe", "--rm", str(fitted), "--pairs", str(d / "probe.jsonl"), "--out", str(d / "probe.json"),
                "--device", "cpu", "--n-boot", "500"])  # fmt: skip
    doc = json.loads((d / "probe.json").read_text())
    hx, hy, hz = (doc["habits"][h] for h in ("hx", "hy", "hz"))
    assert (
        hx["n"] == 8
        and hx["a_higher_share"] >= 0.9
        and hx["reward_driven"] is True
        and hx["mean_diff"] > 0
    )
    assert (
        hy["n"] == 8
        and hy["a_higher_share"] <= 0.1
        and hy["reward_driven"] is True
        and hy["mean_diff"] < 0
    )
    for h in (hx, hy):
        lo, hi = h["mean_diff_ci95"]
        assert lo <= h["mean_diff"] <= hi and h["a_higher_share_ci95"][0] <= h["a_higher_share"]
    assert hz["n"] == 0 and hz["unreadable_pairs"] == 1 and hz["reward_driven"] is None
    ref = doc["reference"]
    assert ref["file"].endswith("held.jsonl") and ref["n"] == 16 and ref["a_higher_share"] >= 0.9
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "8"}
    p = subprocess.run([PY, str(HERE / "lh_rm.py"), "probe", "--rm", str(fitted), "--pairs", str(d / "probe.jsonl"),
                        "--out", str(d / "probe2.json"), "--device", "cpu", "--n-boot", "100"], env=env,
                       capture_output=True, text=True)  # fmt: skip
    assert p.returncode == 0, p.stderr[-2000:]
