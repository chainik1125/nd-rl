"""Tests of steer/steer_preflight.py summarize logic (PASS / FAIL) on synthetic log-probs.  Run: pytest -q tests/test_steer_preflight.py"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "steer"))
import steer_preflight as F  # noqa: E402

RNG = np.random.RandomState(0)
SEQ_N = [1000, 5000]
SEQS = [{"tid": f"pf{k}", "n": n} for k, n in enumerate(SEQ_N)]


def lp(delta=0.0, base=None):
    base = base if base is not None else [RNG.randn(n - 1) for n in SEQ_N]
    return [(b + delta * RNG.randn(len(b))).tolist() for b in base]


def make(noise=0.002, steer_shift=0.5):
    """HF refs and vLLM runs where vLLM = HF + small noise, and steering moves log-probs by `steer_shift` on average."""
    base = [RNG.randn(n - 1) for n in SEQ_N]
    shift = [steer_shift * RNG.randn(n - 1) for n in SEQ_N]
    hf = {
        "stock": [b.tolist() for b in base],
        "chosen": [(b + s).tolist() for b, s in zip(base, shift)],
        "chosen_half": [(b + s / 2).tolist() for b, s in zip(base, shift)],
        "scramble2001": [(b + 0.7 * np.roll(s, 1)).tolist() for b, s in zip(base, shift)],
    }

    def noisy(ref):
        return [(np.asarray(x) + noise * RNG.randn(len(x))).tolist() for x in ref]

    runs = {
        "stock": {"logprobs": noisy(hf["stock"]), "repeat_seq0": None},
        "chosen": {"logprobs": noisy(hf["chosen"])},
        "chosen_eager": {"logprobs": noisy(hf["chosen"])},
        "scramble2001": {"logprobs": noisy(hf["scramble2001"])},
    }
    runs["chosen_eager"]["logprobs"] = [
        (np.asarray(a) + 0.0005 * RNG.randn(len(a))).tolist() for a in runs["chosen"]["logprobs"]
    ]
    runs["stock"]["repeat_seq0"] = list(runs["stock"]["logprobs"][0])
    runs["zero"] = {
        "logprobs": [list(x) for x in runs["stock"]["logprobs"]]
    }  # patched zero == stock, bit for bit
    runs["stock_eager"] = {"logprobs": noisy(hf["stock"])}
    runs["zero_eager"] = {
        "logprobs": [list(x) for x in runs["stock_eager"]["logprobs"]]
    }  # Amendment 1: identity in eager
    refs = {"seqs": SEQS, "refs": hf}
    sha = {
        "chosen_sha": "aa",
        "scramble_sha": "bb",
        "half_sha": "cc",
        "zero_sha": "00",
        "layer": 24,
    }
    act = {
        k: [{"layer": 24, "vec_sha256": sha[v]}]
        for k, v in (
            ("chosen", "chosen_sha"),
            ("chosen_eager", "chosen_sha"),
            ("scramble2001", "scramble_sha"),
            ("zero", "zero_sha"),
            ("zero_eager", "zero_sha"),
        )
    }
    return refs, runs, act, sha


def test_all_checks_pass_on_consistent_runs():
    refs, runs, act, sha = make()
    rep = F.summarize_runs(refs, runs, act, sha)
    assert rep["pass"], rep["checks"]
    for k in (
        "zero_identity",
        "chosen_matches_hf",
        "chosen_drop_detected",
        "scramble2001_drop_detected",
        "scramble2001_matches_hf",
        "compile_vs_eager",
        "prefix_cache_repeat",
        "chosen_steer_active",
        "scramble2001_steer_active",
        "no_missing_required_runs",
    ):
        assert rep["checks"][k], k
    assert (
        rep["stats"]["chosen_vs_hf_beyond_4096"]["n"] > 0
    )  # the 5,000-token sequence crosses the sliding window
    assert rep["tolerance"]["mean"] >= F.FLOOR_MEAN and "power_factor" not in rep["tolerance"]
    assert (
        rep["pass_chosen"]
        and rep["stats"]["zero_identity_path"] == "exact"
        and rep["stats"]["prefix_cache_repeat_path"] == "exact"
    )
    assert (
        rep["stats"]["chosen_effect_over_tol_mean"] > 1
        and rep["stats"]["scramble2001_effect_over_tol_mean"] > 1
    )


def test_fails_when_steering_in_vllm_differs_from_the_hf_hook():
    refs, runs, act, sha = make()
    runs["chosen"]["logprobs"] = [
        (np.asarray(x) + 0.3 * RNG.randn(len(x))).tolist() for x in runs["chosen"]["logprobs"]
    ]  # e.g. wrong layer
    rep = F.summarize_runs(refs, runs, act, sha)
    assert not rep["pass"] and not rep["checks"]["chosen_matches_hf"]


def test_tiny_steering_effect_means_drop_not_detected():
    refs, runs, act, sha = make(
        steer_shift=0.0005
    )  # steering effect below the tolerance: a dropped vector would go unseen
    rep = F.summarize_runs(refs, runs, act, sha)
    assert not rep["checks"]["chosen_drop_detected"] and not rep["pass"] and not rep["pass_chosen"]
    refs2, runs2, act2, sha2 = make()
    runs2["chosen"]["logprobs"] = [
        list(x) for x in runs2["stock"]["logprobs"]
    ]  # patch inactive in the engine: output == stock
    rep2 = F.summarize_runs(refs2, runs2, act2, sha2)
    assert not rep2["checks"]["chosen_matches_hf"] and not rep2["pass"]


def test_scramble_only_failure_passes_chosen_only():
    refs, runs, act, sha = make()
    runs["scramble2001"]["logprobs"] = [
        (np.asarray(x) + 0.3 * RNG.randn(len(x))).tolist() for x in runs["scramble2001"]["logprobs"]
    ]
    rep = F.summarize_runs(refs, runs, act, sha)
    assert not rep["checks"]["scramble2001_matches_hf"] and not rep["pass"] and rep["pass_chosen"]
    refs, runs, act, sha = make()
    act["scramble2001"] = []
    rep = F.summarize_runs(refs, runs, act, sha)
    assert not rep["pass"] and rep["pass_chosen"]


def test_eager_zero_identity_paths():
    # eager stocks identical but zero differs -> exact_failed
    refs, runs, act, sha = make()
    runs["stock_eager2"] = {"logprobs": [list(x) for x in runs["stock_eager"]["logprobs"]]}
    runs["zero_eager"]["logprobs"] = [
        (np.asarray(x) + 0.001 * RNG.randn(len(x))).tolist() for x in runs["zero_eager"]["logprobs"]
    ]
    rep = F.summarize_runs(refs, runs, act, sha)
    assert (
        not rep["checks"]["zero_identity"] and rep["stats"]["zero_identity_path"] == "exact_failed"
    )
    assert rep["stats"]["eager_stock_repeat_exact"] and not rep["pass"]
    # stock_eager2 missing and zero differs -> exact_failed
    refs, runs, act, sha = make()
    runs["zero_eager"]["logprobs"][1][10] += 1e-6
    rep = F.summarize_runs(refs, runs, act, sha)
    assert rep["stats"]["zero_identity_path"] == "exact_failed" and not rep["pass"]
    # eager stocks differ, zero within 1.5 x spread -> fallback pass; beyond -> fail
    refs, runs, act, sha = make()
    se = runs["stock_eager"]["logprobs"]
    runs["stock_eager2"] = {
        "logprobs": [(np.asarray(x) + 0.004 * RNG.randn(len(x))).tolist() for x in se]
    }
    runs["zero_eager"]["logprobs"] = [
        (np.asarray(x) + 0.004 * RNG.randn(len(x))).tolist() for x in se
    ]
    rep = F.summarize_runs(refs, runs, act, sha)
    assert (
        rep["stats"]["zero_identity_path"] == "fallback_eager_spread"
        and not rep["stats"]["eager_stock_repeat_exact"]
    )
    assert rep["checks"]["zero_identity"] and rep["pass"], rep["stats"]
    runs["zero_eager"]["logprobs"] = [
        (np.asarray(x) + 0.02 * RNG.randn(len(x))).tolist() for x in se
    ]
    rep = F.summarize_runs(refs, runs, act, sha)
    assert (
        rep["stats"]["zero_identity_path"] == "fallback_eager_spread"
        and not rep["checks"]["zero_identity"]
        and not rep["pass"]
    )


def test_prefix_cache_repeat_fallback_to_tolerance():
    refs, runs, act, sha = make()
    runs["stock"]["repeat_seq0"][5] += 0.002  # above the exact 1e-3, far inside tol_mean / tol_p99
    rep = F.summarize_runs(refs, runs, act, sha)
    assert (
        rep["checks"]["prefix_cache_repeat"]
        and rep["stats"]["prefix_cache_repeat_path"] == "fallback_tol"
        and rep["pass"]
    )
    refs, runs, act, sha = make()
    runs["stock"]["repeat_seq0"] = [x + 0.5 for x in runs["stock"]["repeat_seq0"]]
    rep = F.summarize_runs(refs, runs, act, sha)
    assert (
        not rep["checks"]["prefix_cache_repeat"]
        and rep["stats"]["prefix_cache_repeat_path"] == "failed"
    )


def test_zero_vector_must_be_bit_identical_to_stock():
    """Amendment 1: bit identity in eager mode; compiled zero vs compiled stock within tolerance."""
    refs, runs, act, sha = make()
    runs["zero_eager"]["logprobs"][1][10] += 1e-6
    rep = F.summarize_runs(refs, runs, act, sha)
    assert not rep["checks"]["zero_identity"] and not rep["pass"]
    refs, runs, act, sha = make()
    runs["zero"]["logprobs"] = [
        (np.asarray(x) + 0.002 * RNG.randn(len(x))).tolist() for x in runs["zero"]["logprobs"]
    ]  # cross-process noise
    rep = F.summarize_runs(refs, runs, act, sha)
    assert rep["pass"] and not rep["stats"]["zero_identity_compiled_exact"]
    runs["zero"]["logprobs"] = [
        (np.asarray(x) + 0.3 * RNG.randn(len(x))).tolist() for x in runs["zero"]["logprobs"]
    ]  # patch alters the model
    rep = F.summarize_runs(refs, runs, act, sha)
    assert not rep["checks"]["zero_compiled_within_tol"] and not rep["pass"]
    refs, runs, act, sha = make()
    del runs["zero_eager"]
    rep = F.summarize_runs(refs, runs, act, sha)
    assert "zero_eager" in rep["missing_runs"] and not rep["pass"]


def test_missing_required_run_prefix_cache_and_steer_active_failures():
    refs, runs, act, sha = make()
    r2 = {k: v for k, v in runs.items() if k != "chosen_eager"}
    rep = F.summarize_runs(refs, r2, act, sha)
    assert rep["missing_runs"] == ["chosen_eager"] and not rep["pass"]
    assert F.summarize_runs(refs, r2, act, sha, require_eager=False)["pass"]
    refs, runs, act, sha = make()
    runs["stock"]["repeat_seq0"] = [x + 0.5 for x in runs["stock"]["repeat_seq0"]]
    assert not F.summarize_runs(refs, runs, act, sha)["checks"]["prefix_cache_repeat"]
    refs, runs, act, sha = make()
    act["chosen"] = [{"layer": 24, "vec_sha256": "WRONG"}]
    assert not F.summarize_runs(refs, runs, act, sha)["checks"]["chosen_steer_active"]
    act["chosen"] = []
    assert not F.summarize_runs(refs, runs, act, sha)["checks"]["chosen_steer_active"]


def test_compile_vs_eager_disagreement_fails_and_half_alpha_checked_when_present():
    refs, runs, act, sha = make()
    runs["chosen_eager"]["logprobs"] = [
        (np.asarray(x) + 0.2 * RNG.randn(len(x))).tolist() for x in runs["chosen"]["logprobs"]
    ]
    assert not F.summarize_runs(refs, runs, act, sha)["checks"]["compile_vs_eager"]
    refs, runs, act, sha = make()
    runs["chosen_half"] = {
        "logprobs": [
            (np.asarray(x) + 0.001 * RNG.randn(len(x))).tolist()
            for x in refs["refs"]["chosen_half"]
        ]
    }
    assert F.summarize_runs(refs, runs, act, sha)["checks"]["chosen_half_matches_hf"]


def test_cli_summarize_exit_codes(tmp_path):
    refs, runs, act, sha = make()
    p3, d = tmp_path / "p3", tmp_path / "pf"
    p3.mkdir()
    d.mkdir()
    (p3 / "preflight_refs.json").write_text(json.dumps(refs))
    (p3 / "chosen.json").write_text(
        json.dumps(
            {
                "layer": 24,
                "sha256": {"chosen": "aa", "scramble2001": "bb", "chosen_half": "cc", "zero": "00"},
            }
        )
    )
    for name, r in runs.items():
        (d / f"pf_{name}.json").write_text(json.dumps(r))
        (d / f"active_{name}.jsonl").write_text(
            "".join(json.dumps(x) + "\n" for x in act.get(name, []))
        )
    assert (
        F.main(["summarize", "--p3", str(p3), "--dir", str(d), "--out", str(tmp_path / "r.json")])
        == 0
    )
    assert json.loads((tmp_path / "r.json").read_text())["pass"]
    (d / "pf_chosen_eager.json").unlink()
    assert (
        F.main(["summarize", "--p3", str(p3), "--dir", str(d), "--out", str(tmp_path / "r2.json")])
        == 1
    )
    (d / "pf_chosen_eager.json").write_text(json.dumps(runs["chosen_eager"]))
    bad = [
        (np.asarray(x) + 0.3 * RNG.randn(len(x))).tolist() for x in runs["scramble2001"]["logprobs"]
    ]
    (d / "pf_scramble2001.json").write_text(json.dumps({"logprobs": bad}))
    assert (
        F.main(["summarize", "--p3", str(p3), "--dir", str(d), "--out", str(tmp_path / "r3.json")])
        == 4
    )


def test_suffix_option_is_accepted():
    import contextlib
    import io

    with contextlib.redirect_stderr(io.StringIO()) as e, pytest.raises(SystemExit):
        F.main(
            ["run", "--mode", "stock", "--eager", "--suffix", "2", "--model", "m", "--p3", "p"]
        )  # only --out missing
    err = e.getvalue().strip().splitlines()[-1]
    assert "required: --out" in err and "unrecognized" not in err
