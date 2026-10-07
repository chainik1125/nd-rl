"""`code.thinking.adaptivity`: the stratified set, the row join and the analysis, on synthetic rows
with known answers."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from code.thinking import adaptivity as ad


def row(name: str, ok: list[int], think: list[float], draws: list[float]) -> dict[str, Any]:
    return dict(name=name, ok=ok, think=think, draws=draws, forced=[0] * len(ok))


def labels_for(t0: list[float], n_lines: list[int], extra: int = 0) -> dict[str, dict[str, Any]]:
    lab: dict[str, dict[str, Any]] = {
        f"t{i}": dict(group="rl", t0_pass1=p, n_lines=n)
        for i, (p, n) in enumerate(zip(t0, n_lines, strict=True))
    }
    for j in range(extra):
        lab[f"b{j}"] = dict(group="textbook_dev", t0_pass1=None, n_lines=None)
    return lab


def test_bins() -> None:
    assert [ad.t0_bin(p) for p in (0.0, 0.0625, 0.25, 0.3125, 0.5, 0.5625, 0.625, 0.7)] == [
        "0", "(0,.25]", "(0,.25]", "(.25,.5]", "(.25,.5]", "(.5,.625]", "(.5,.625]", None,
    ]  # fmt: skip
    assert ad.t0_bin(None) is None
    assert [ad.len_bin(n) for n in (7, 12, 13, 20, 21)] == ["<=12", "<=12", "13-20", "13-20", None]
    assert [ad.own_bin(p) for p in (0.0, 0.25, 0.5, 0.75, 1.0)] == list(ad.OWN_BINS)


def test_ranks_and_spearman() -> None:
    x = np.array([3.0, 1.0, 2.0, 2.0])
    assert ad.ranks(x).tolist() == [4.0, 1.0, 2.5, 2.5]
    a = np.arange(10, dtype=np.float64)
    assert ad.spearman(a, a**3) == pytest.approx(1.0)
    assert ad.spearman(a, -a) == pytest.approx(-1.0)
    assert math.isnan(ad.spearman(a, np.ones(10)))
    # ties: Pearson on average ranks
    y = np.array([1.0, 1.0, 2.0, 3.0])
    z = np.array([1.0, 2.0, 2.0, 4.0])
    rz, ry = ad.ranks(z), ad.ranks(y)
    assert ad.spearman(y, z) == pytest.approx(float(np.corrcoef(ry, rz)[0, 1]))
    e = ad.spearman_ci(a, a + 0.1 * np.sin(a), draws=200)
    assert e["est"] == pytest.approx(1.0) and e["n"] == 10 and e["lo"] <= e["est"] <= e["hi"]


def test_pass_at_k() -> None:
    c = np.array([0, 1, 8, 16], dtype=np.int64)
    p1 = ad.pass_at_k(16, c, 1)
    assert p1.tolist() == pytest.approx([0.0, 1 / 16, 0.5, 1.0])
    p5 = ad.pass_at_k(16, c, 5)
    assert p5[1] == pytest.approx(1 - math.comb(15, 5) / math.comb(16, 5))
    assert p5[3] == 1.0 and p5[0] == 0.0


def test_mean_ci() -> None:
    v = np.array([1.0, 1.0, 1.0, np.nan])
    e = ad.mean_ci(v, draws=100)
    assert (e["est"], e["lo"], e["hi"], e["n"]) == (1.0, 1.0, 1.0, 3)
    assert ad.mean_ci(np.array([np.nan]))["n"] == 0


def synthetic(T: int = 40, k: int = 8) -> tuple[dict[int, list[dict[str, Any]]], dict[str, Any]]:
    """Harder targets (lower T0 pass rate) think longer per block and gain more from e256."""
    rng = np.random.default_rng(0)
    t0 = [round(float(x), 4) for x in np.linspace(0.0, 0.625, T)]
    n_lines = [8 + (i % 2) * 8 for i in range(T)]
    rows: dict[int, list[dict[str, Any]]] = {0: [], 256: []}
    for i, p in enumerate(t0):
        n0 = int(round(p * k))
        ok0 = [1] * n0 + [0] * (k - n0)
        rows[0].append(row(f"t{i}", ok0, [0.0] * k, [5.0] * k))
        n256 = min(k, n0 + (2 if p < 0.3 else 0))
        ok256 = [1] * n256 + [0] * (k - n256)
        blk = 100.0 - 100.0 * p  # tokens per block falls with T0 pass rate
        think = [blk * 5 + float(rng.integers(0, 3)) for _ in range(k)]
        rows[256].append(row(f"t{i}", ok256, think, [5.0] * k))
    return rows, labels_for(t0, n_lines)


def test_make_table_aligns_and_checks() -> None:
    rows, lab = synthetic()
    rows[256] = list(reversed(rows[256]))  # a level in another order
    t = ad.make_table(rows, lab)
    assert t.k == 8 and len(t.names) == 40
    assert t.levels[256].ok[0].tolist() == rows[0][0]["ok"][:0] + [1, 1] + [0] * 6
    with pytest.raises(ValueError, match="without labels"):
        ad.make_table(rows, {k: v for k, v in lab.items() if k != "t3"})
    bad = {0: rows[0], 256: rows[256][:-1]}
    with pytest.raises(ValueError, match="different targets"):
        ad.make_table(bad, lab)


def test_analyze_synthetic() -> None:
    rows, lab = synthetic()
    res = ad.analyze(ad.make_table(rows, lab), draws=300)
    # pass@1 at e0 equals the mean of n_ok / k
    exp = 100 * np.mean([sum(r["ok"]) / 8 for r in rows[0]])
    assert res["passk"]["0"]["all"]["pass@1"]["est"] == pytest.approx(exp)
    # think per block falls with T0 pass rate: rho near -1, CI below 0
    c = res["corr"]["256"]["think_block ~ t0_pass1 (RL)"]
    assert c["est"] < -0.95 and c["hi"] < 0
    # gain: +2/8 on T0 < 0.3, 0 elsewhere
    g = res["gain"]["256"]
    assert g["T0 0"]["gain"]["est"] == pytest.approx(0.25)
    assert g["T0 (.5,.625]"]["gain"]["est"] == pytest.approx(0.0)
    assert g["T0 (.5,.625]"]["gain"]["lo"] == pytest.approx(0.0)
    # think tables: per block about 100 - 100 p, per attempt 5 blocks of it
    th = res["think"]["256"]["T0 0"]
    assert th["think_block"]["est"] == pytest.approx(100.0, abs=1.0)
    assert th["blocks"] == pytest.approx(5.0)
    md = ad.render(res, "synthetic")
    assert "T0 (.25,.5] / len 13-20" in md and "own e0 0 (split half)" in md
    cmp = ad.compare({"a": res, "b": res})
    assert "| T0 0 |" in cmp and "think tokens per block" in cmp


def test_own_e0_split_half() -> None:
    """Bins come from e0's first half, the gain's e0 side from its second half."""
    k = 8
    lab = labels_for([0.0, 0.0], [8, 8])
    rows = {
        0: [row("t0", [0] * 4 + [1] * 4, [0.0] * k, [1.0] * k),
            row("t1", [1] * 4 + [0] * 4, [0.0] * k, [1.0] * k)],
        256: [row("t0", [1] * k, [10.0] * k, [1.0] * k),
              row("t1", [1] * k, [10.0] * k, [1.0] * k)],
    }  # fmt: skip
    g = ad.analyze(ad.make_table(rows, lab), draws=50)["gain"]["256"]
    assert g["own e0 0 (split half)"]["p_e0"] == pytest.approx(1.0)  # t0's second half
    assert g["own e0 0 (split half)"]["gain"]["est"] == pytest.approx(0.0)
    assert g["own e0 (.75,1] (split half)"]["gain"]["est"] == pytest.approx(1.0)


def test_by_outcome() -> None:
    k = 4
    lab = labels_for([0.25, 0.25], [8, 8])
    rows = {
        0: [row("t0", [0] * k, [0.0] * k, [2.0] * k), row("t1", [0] * k, [0.0] * k, [2.0] * k)],
        # t0: accepted attempts think 10 per block, failed 30; t1 all fail (not within-target)
        256: [row("t0", [1, 1, 0, 0], [20.0, 20.0, 60.0, 60.0], [2.0] * k),
              row("t1", [0] * k, [8.0] * k, [2.0] * k)],
    }  # fmt: skip
    o = ad.analyze(ad.make_table(rows, lab), draws=50)["outcome"]["256"]
    assert (o["attempts_ok"], o["attempts_fail"]) == (2, 6)
    assert o["think_block_ok"] == pytest.approx(10.0)
    assert o["think_block_fail"] == pytest.approx((30 * 2 + 4 * 4) / 6)
    w = o["think_block_within_fail_minus_ok"]
    assert (w["est"], w["n"]) == (pytest.approx(20.0), 1)
    assert sum(q["n"] for q in o["pass_by_think_block_quartile"]) == 8


def test_build_set(tmp_path: Path) -> None:
    pool = []
    i = 0
    for p in (0.0, 0.125, 0.375, 0.5625, 0.75):
        for n in (9, 15, 25):
            for _ in range(3):
                pool.append(dict(name=f"rh_{i}", prompt=f"THM P{i}", t0_pass1=p, n_lines=n))
                i += 1
    tb = {"textbook_dev": [dict(name="tb0", prompt="THM Q", reference_lines=7)],
          "textbook_train": [dict(name="tb1", prompt="THM R")]}  # fmt: skip
    recs, lab = ad.build_set(pool, tb, per_cell=2, seed=0)
    assert len(recs) == 8 * 2 + 2
    cells = {(v["t0_bin"], v["len_bin"]) for v in lab.values() if v["group"] == "rl"}
    assert len(cells) == 8
    assert all(v["t0_pass1"] <= 0.625 and v["n_lines"] <= 20 for v in lab.values() if v["t0_bin"])
    assert recs[-2]["reference_lines"] == 7 and lab["tb1"]["group"] == "textbook_train"
    assert recs == ad.build_set(pool, tb, per_cell=2, seed=0)[0]  # deterministic
    with pytest.raises(ValueError, match="cell"):
        ad.build_set(pool, tb, per_cell=4)
    # the CLI round trip: set + labels, then analysis over written rows
    p = tmp_path / "pool.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in pool))
    d = tmp_path / "tb"
    d.mkdir()
    for g, rs in tb.items():
        (d / f"{g}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rs))
    s, lf = tmp_path / "set.jsonl", tmp_path / "set.labels.json"
    argv = ["build-set", "--pool", str(p), "--textbook", str(d), "--per_cell", "2"]
    assert ad.main(argv + ["--set", str(s), "--labels", str(lf)]) == 0
    names = [json.loads(x)["name"] for x in s.read_text().splitlines()]
    assert set(names) == set(ad.load_labels(lf))
    run = tmp_path / "run"
    for B in (0, 64):
        e = run / "eval" / f"B{B}"
        e.mkdir(parents=True)
        e.joinpath(f"rows_plain_B{B}.jsonl").write_text(
            "".join(
                json.dumps(row(n, [1, 0], [B * 1.0, B * 2.0], [1.0, 2.0])) + "\n" for n in names
            )
        )
    assert ad.main(["--sets", str(s), "--labels", str(lf), "--out", str(run), "--analyze_only",
                    "--draws", "50"]) == 0  # fmt: skip
    res = json.loads((run / "adaptivity.json").read_text())
    assert res["levels"] == [0, 64] and res["n_targets"] == 18
    assert res["think"]["64"]["all"]["think_block"]["est"] == pytest.approx(64.0)
