# ruff: noqa: E501
"""Tests for footprint.py: concentration and top-k arithmetic, and an end-to-end run on a tiny synthetic npz."""

import json

import footprint as fp
import numpy as np


def test_concentration_toy():
    nll = np.array([10.0, 5.0] + [0.5] * 98)  # N = 100, total 64
    assert abs(fp.concentration(nll, 0.01) - 10 / 64) < 1e-12
    assert abs(fp.concentration(nll, 0.05) - (15 + 3 * 0.5) / 64) < 1e-12
    assert abs(fp.concentration(nll, 0.10) - (15 + 8 * 0.5) / 64) < 1e-12


def test_concentration_weights_match_repetition():
    rng = np.random.default_rng(0)
    nll = np.sort(rng.exponential(size=50))[::-1]
    w = rng.integers(0, 4, size=50)
    rep = np.repeat(nll, w)
    for q in fp.QS:
        assert abs(fp.concentration_sorted(nll, w, q) - fp.concentration(rep, q)) < 1e-9


def test_topk_share():
    rank = np.array([0, 0, 1, 4, 5, 19, 0, 2])
    assert fp.topk_share(rank, 1) == 3 / 8
    assert fp.topk_share(rank, 5) == 6 / 8
    w = np.array([1, 1, 0, 0, 0, 0, 0, 0])
    assert fp.topk_share(rank, 1, w) == 1.0


def _write_npz(path, seed):
    rng = np.random.default_rng(seed)
    n_tok = np.array([30, 40, 25, 35])
    think_end = np.array([20, 30, 10, 25], dtype=np.int32)
    off = np.concatenate([[0], np.cumsum(n_tok)]).astype(np.int64)
    n = int(off[-1])
    np.savez_compressed(
        path,
        pid=np.array(["a", "b", "c", "d"]),
        set=np.array(["train", "test", "test", "reach"]),
        think_end=think_end,
        n_tok=n_tok.astype(np.int32),
        offset=off,
        token=rng.integers(0, 100, n).astype(np.int32),
        nll=rng.exponential(0.5, n).astype(np.float32),
        rank=rng.integers(0, 8, n).astype(np.int32),
        entropy=np.zeros(n, dtype=np.float16),
        topk_ids=np.zeros((n, 20), dtype=np.int32),
        topk_logp=np.zeros((n, 20), dtype=np.float16),
    )


def test_end_to_end(tmp_path):
    runs = tmp_path / "runs"
    for i, (ck, _st) in enumerate(fp.CKPTS):
        (runs / ck / "edl").mkdir(parents=True)
        _write_npz(runs / ck / "edl" / "step0_tokens.npz", i)
    out = tmp_path / "out"
    assert fp.main(["--runs", str(runs), "--out-dir", str(out), "--B", "20", "--jobs", "1"]) == 0
    s = json.loads((out / "footprint_summary.json").read_text())
    r = s["checkpoints"]["dpo"]["segments"]
    d = fp.load_npz(runs / "dpo" / "edl" / "step0_tokens.npz")
    assert r["total"]["n_positions"] == 130
    assert r["think"]["n_positions"] + r["answer"]["n_positions"] == 130
    assert r["think"]["n_positions"] == 85
    assert abs(r["total"]["nll"] - d["nll"].mean()) < 1e-9
    assert abs(r["total"]["top1"] - (d["rank"] == 0).mean()) < 1e-12
    lo, hi = r["total"]["top5_ci"]
    assert lo <= hi
    p = s["paired"]["s2300-dpo"]["segments"]["total"]["top1"]
    assert (
        abs(
            p["diff"]
            - (s["checkpoints"]["s2300"]["segments"]["total"]["top1"] - r["total"]["top1"])
        )
        < 1e-12
    )
    assert "s2300-s1100" in s["paired"] and "s0100-dpo" in s["paired"]
    assert (out / "footprint_summary_paired.csv").exists()
    for ext in ("pdf", "svg", "png"):
        assert (out / f"footprint.{ext}").exists()
    assert (out / "footprint_summary.csv").exists()
