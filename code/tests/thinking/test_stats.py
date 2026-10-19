"""`code.thinking.stats`: the hierarchical bootstrap, paired differences, matched-token pass@k, the
analytic variance components and power, on synthetic reads with known answers."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from code.thinking import figures as F
from code.thinking import stats as st

EVAL = Path("/scratch/cowifornia/nd-rl/think/runs/eval")
needs_eval = pytest.mark.skipif(
    not (EVAL / "tb72_T0_s0").exists(), reason="Stage-1 eval outputs absent"
)


def make_arm(c: np.ndarray, n: int, label: str = "A", tokens: np.ndarray | None = None) -> st.Arm:
    """An in-memory arm: c is S x T successes out of n attempts each."""
    S, T = c.shape
    return st.Arm(
        label,
        [f"t{i:04d}" for i in range(T)],
        ["set"] * T,
        np.full((S, T), n, dtype=np.int64),
        c.astype(np.int64),
        tokens,
        [],
        [None] * S,
    )


def write_eval_dir(
    d: Path, ok: dict[str, list[int]], tokens: int, cell: str = "plain_B0", ckpt: str = ""
) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    with open(d / f"rows_{cell}.jsonl", "a") as f:
        for name, o in ok.items():
            row = {"file": "set", "name": name, "k": len(o), "n_ok": sum(o), "ok": o,
                   "tokens": [tokens] * len(o)}  # fmt: skip
            f.write(json.dumps(row) + "\n")
    summ = d / "summary.json"
    cells = json.loads(summ.read_text())["cells"] if summ.exists() else {}
    cells[cell] = {}
    summ.write_text(json.dumps({"ckpt": ckpt or str(d), "cells": cells}))
    return d


# ----------------------------------------------------------------------------- the estimator


def test_passk_curve_is_the_unbiased_estimator() -> None:
    n = np.array([[256, 16], [8, 8]])
    c = np.array([[3, 0], [8, 2]])
    cur = st.passk_curve(n, c, 8)
    for (i, j), nn in np.ndenumerate(n):
        for k in range(1, 9):
            assert cur[i, j, k - 1] == pytest.approx(F.pass_at_k(int(nn), int(c[i, j]), k))
    with pytest.raises(ValueError):
        st.passk_curve(n, c, 9)


def test_known_p_is_covered_at_the_nominal_rate() -> None:
    """Theorems drawn from a population with p in {0.1, 0.5}: the 95% interval of pass@1 (true
    0.3) and pass@5 (true mean of 1 - (1 - p)^5) covers the truth in about 95% of reads."""
    rng = np.random.default_rng(1)
    truth = {1: 0.3, 5: (1 - 0.9**5 + 1 - 0.5**5) / 2}
    hits = {1: 0, 5: 0}
    reps = 160
    for r in range(reps):
        p = rng.choice([0.1, 0.5], size=40)
        c = rng.binomial(16, np.broadcast_to(p, (3, 40)))
        arm = make_arm(c, 16)
        for k in hits:
            e = st.ci(arm, k, draws=400, seed=r)
            hits[k] += e.lo <= truth[k] <= e.hi
    for k in hits:
        assert 0.88 <= hits[k] / reps <= 1.0, (k, hits[k] / reps)


def test_point_estimate_and_k_none() -> None:
    c = np.array([[1, 0, 4], [2, 0, 4]])
    arm = make_arm(c, 4)
    e = st.ci(arm, 1, draws=200)
    assert e.est == pytest.approx(np.mean(c / 4)) and e.lo <= e.est <= e.hi
    # k None is pass@n: the any-success rate
    assert st.ci(arm, None, draws=50).est == pytest.approx(2 / 3)
    with pytest.raises(ValueError, match="pass@5"):
        st.ci(arm, 5)


def test_one_seed_resamples_theorems_only_and_levels() -> None:
    c = np.array([[0, 1, 2, 3, 4, 4, 4, 0]])
    arm = make_arm(c, 4)
    both = st.ci(arm, 1, draws=500)
    th = st.ci(arm, 1, draws=500, levels="theorems")
    assert both.lo == th.lo and both.hi == th.hi  # one seed: nothing to resample
    seeds_only = st.ci(arm, 1, draws=500, levels="seeds")
    assert seeds_only.lo == seeds_only.hi == pytest.approx(seeds_only.est)


# ---------------------------------------------------------------------------- paired differences


def test_identical_arms_give_a_zero_centred_interval() -> None:
    rng = np.random.default_rng(2)
    c = rng.binomial(32, rng.uniform(0, 1, size=(3, 50)))
    a, b = make_arm(c, 32, "A"), make_arm(c.copy(), 32, "B")
    e = st.paired_diff(a, b, 1, draws=2000)
    assert e.est == 0.0
    assert e.lo < 0 < e.hi
    assert abs(e.lo + e.hi) < 0.15 * (e.hi - e.lo)
    assert e.p_boot is not None and e.p_boot > 0.5
    assert (e.n_pos, e.n_neg, e.p_sign) == (0, 0, 1.0)
    # the same checkpoints read twice: seeds paired, the difference is exactly 0
    p = st.paired_diff(a, b, 1, draws=500, paired_seeds=True)
    assert p.paired_seeds and p.lo == p.hi == 0.0


def test_pairing_on_theorems_resolves_a_shift_marginal_intervals_hide() -> None:
    """Arm B solves 2 more of 32 attempts on every theorem: the arms' own intervals overlap
    (theorem difficulty varies a lot) but the paired difference excludes 0."""
    rng = np.random.default_rng(3)
    c = np.broadcast_to(rng.binomial(30, rng.uniform(0, 1, size=60)), (3, 60)).copy()
    a, b = make_arm(c, 32, "A"), make_arm(c + 2, 32, "B")
    ea, eb = st.ci(a, 1, draws=1000), st.ci(b, 1, draws=1000)
    assert eb.lo < ea.hi  # overlapping
    d = st.paired_diff(b, a, 1, draws=1000)
    assert d.est == pytest.approx(2 / 32)
    assert d.lo > 0 and d.p_boot is not None and d.p_boot < 0.01
    assert (d.n_pos, d.n_neg) == (60, 0)
    assert d.p_sign is not None and d.p_sign < 1e-15


def test_same_process_arms_cover_zero() -> None:
    rng = np.random.default_rng(4)
    base = rng.uniform(0.0, 0.8, size=40)
    hits_b = hits_t = 0
    reps = 100
    for r in range(reps):
        arms = [make_arm(rng.binomial(16, np.broadcast_to(base, (3, 40))), 16, lab) for lab in "AB"]
        e = st.paired_diff(arms[0], arms[1], 1, draws=400, seed=r)
        cp = st.components(arms[0], arms[1], 1)
        hits_b += e.lo <= 0 <= e.hi
        hits_t += cp.lo <= 0 <= cp.hi
    assert hits_b / reps >= 0.88 and hits_t / reps >= 0.88


def test_sign_test_is_exact() -> None:
    assert st.sign_test([1.0] * 9 + [-1.0]) == (9, 1, pytest.approx(2 * 11 / 1024))
    assert st.sign_test([0.0, 0.0]) == (0, 0, 1.0)
    assert st.sign_test([1.0, -1.0])[2] == 1.0


def test_boot_p() -> None:
    assert st.boot_p(np.array([1.0, 2.0, 3.0])) == 0.0
    assert st.boot_p(np.array([-1.0, 1.0, 2.0, 3.0])) == pytest.approx(0.5)
    assert st.boot_p(np.zeros(5)) == 1.0


# ------------------------------------------------------------------------------- matched tokens


def test_matched_tokens_interpolate_in_log_tokens(tmp_path: Path) -> None:
    ok = {"a": [1, 0, 0, 0, 0, 0, 0, 0], "b": [1, 1, 0, 0, 0, 0, 0, 0], "c": [0] * 8}
    dirs = [
        write_eval_dir(tmp_path / "s0", ok, tokens=100),
        write_eval_dir(tmp_path / "s1", ok, tokens=200),
    ]
    arm = st.load_arm("A", dirs)
    assert list(arm.tokens_per_attempt()) == [100.0, 200.0]

    def passk(k: int) -> float:
        return float(np.mean([F.pass_at_k(8, sum(o), k) for o in ok.values()]))

    def logi(k: float, k0: int, k1: int) -> float:
        w = math.log(k / k0) / math.log(k1 / k0)
        return passk(k0) * (1 - w) + passk(k1) * w

    # 600 tokens: seed 0 draws 6 attempts, seed 1 draws 3
    e = st.at_tokens(arm, [600.0], draws=50)[0]
    assert e.est == pytest.approx((passk(6) + passk(3)) / 2) and e.in_range
    # 500 tokens: k* = 5 and 2.5, the latter between 2 and 3 in log k
    e = st.at_tokens(arm, [500.0], draws=50)[0]
    assert e.est == pytest.approx((passk(5) + logi(2.5, 2, 3)) / 2)
    # the tokens figure's grid: k = 1, 2, 4, 8
    e = st.at_tokens(arm, [600.0], grid="pow2", draws=50)[0]
    assert e.est == pytest.approx((logi(6, 4, 8) + logi(3, 2, 4)) / 2)
    # at a figure point (x = k * tokens per attempt) the value is the figure's y, seed by seed
    data = F.tokens_data([F.Group("A", [str(d) for d in dirs])], arms=["plain"], sets=["set"])
    x0, y0 = data["series"][0]["seeds"][0]["sets"]["set"]["pts"][2]  # seed 0, k = 4
    one = st.load_arm("A0", dirs[:1])
    assert st.at_tokens(one, [x0], draws=10)[0].est == pytest.approx(y0)
    # outside [1, n] attempts: clamped, flagged
    lo, hi = st.at_tokens(arm, [50.0, 1600.0], draws=50)
    assert not lo.in_range and lo.est == pytest.approx((passk(1) + passk(1)) / 2)
    assert not hi.in_range and hi.est == pytest.approx((passk(8) + passk(8)) / 2)
    assert lo.clamped == 1.0


def test_matched_token_differences_are_paired(tmp_path: Path) -> None:
    ok = {f"t{i}": [int(j < i) for j in range(8)] for i in range(8)}
    a = st.load_arm("A", [write_eval_dir(tmp_path / "a", ok, tokens=100)])
    b = st.load_arm("B", [write_eval_dir(tmp_path / "b", ok, tokens=100)])
    d = st.paired_diff_at_tokens(a, b, [300.0], draws=200)[0]
    assert d.est == 0.0 and d.lo == d.hi == 0.0  # one seed each, same data
    # B spends twice the tokens per attempt: at 400 tokens A has pass@4, B pass@2
    c = st.load_arm("C", [write_eval_dir(tmp_path / "c", ok, tokens=200)])
    d = st.paired_diff_at_tokens(a, c, [400.0], draws=200)[0]
    p4 = np.mean([F.pass_at_k(8, sum(o), 4) for o in ok.values()])
    p2 = np.mean([F.pass_at_k(8, sum(o), 2) for o in ok.values()])
    assert d.est == pytest.approx(p4 - p2) and d.lo > 0
    assert d.n_neg == 0 and d.n_pos is not None and d.n_pos > 0


# ------------------------------------------------------------------------ loaders and clusters


def test_load_arm_aligns_and_detects_shared_checkpoints(tmp_path: Path) -> None:
    ok = {"x": [1, 0], "y": [0, 0]}
    d0 = write_eval_dir(tmp_path / "s0", ok, 10, "plain_B0", ckpt="ck0")
    write_eval_dir(tmp_path / "s0", ok, 10, "plain_B256", ckpt="ck0")
    b0 = st.load_arm("B0", [d0 / "rows_plain_B0.jsonl"])
    b256 = st.load_arm("B256", [d0], cell="plain_B256")
    assert b0.models == ["ck0"] and st.same_models(b0, b256)
    other = st.load_arm("O", [write_eval_dir(tmp_path / "o", ok, 10, ckpt="ck1")])
    assert not st.same_models(b0, other)
    short = st.load_arm("S", [write_eval_dir(tmp_path / "short", {"x": [1, 0]}, 10)])
    with pytest.raises(ValueError, match="different theorems"):
        st.paired_diff(b0, short, 1)
    with pytest.raises(ValueError, match="other theorems"):
        st.load_arm("bad", [d0 / "rows_plain_B0.jsonl", tmp_path / "short"])
    assert st.load_arm("X", [d0 / "rows_plain_B0.jsonl"], exclude=["nope"]).T == 2
    with pytest.raises(ValueError, match="no theorems left"):
        st.load_arm("X", [d0 / "rows_plain_B0.jsonl"], sets=["nope"])


def test_clusters_are_resampled_as_units(tmp_path: Path) -> None:
    sets = tmp_path / "gen.jsonl"
    rows = [{"name": f"g{i}", "family": "f", "knob": i // 3} for i in range(6)]
    sets.write_text("".join(json.dumps(r) + "\n" for r in rows))
    cl = st.clusters_from_sets([sets], ["family", "knob"])
    assert cl["g0"] == cl["g2"] == "gen:family=f,knob=0" and cl["g3"] != cl["g0"]
    names = [f"g{i}" for i in range(6)] + ["solo"]
    idx = st.cluster_index(names, cl)
    assert idx is not None and list(idx) == [0, 0, 0, 1, 1, 1, 2]
    dr = st.make_draws(7, draws=300, clusters=idx)
    assert (dr.W[:, 0] == dr.W[:, 1]).all() and (dr.W[:, 3] == dr.W[:, 5]).all()
    assert set(dr.W.sum(1)) <= {3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0}
    # theorem-level weights always sum to T, and are shared between calls
    w1 = st.make_draws(7, draws=50, seed=3).W
    assert (w1.sum(1) == 7).all() and (w1 == st.make_draws(7, draws=50, seed=3).W).all()


# ------------------------------------------------------------------------------ analytic + power


def test_t_distribution() -> None:
    for q, df, want in [(0.975, 2, 4.302653), (0.975, 4, 2.776445), (0.975, 30, 2.042272),
                        (0.9, 10, 1.372184), (0.5, 3, 0.0)]:  # fmt: skip
        assert st.t_ppf(q, df) == pytest.approx(want, abs=1e-5)
    assert st.t_ppf(0.975, math.inf) == pytest.approx(1.959964, abs=1e-6)
    assert st.t_cdf(-1.3, 7) == pytest.approx(1 - st.t_cdf(1.3, 7))


def test_components_recover_known_variances() -> None:
    """A noiseless crossed design: V = theorem effect + seed shift, no residual."""
    th = np.linspace(0.1, 0.9, 50)
    shift_a = np.array([-0.02, 0.0, 0.02])
    Va = th[None, :] + shift_a[:, None]
    cp = st.components(make_arm(np.zeros((3, 50), int), 1), None, 1)  # all zero: var 0
    assert cp.se == 0.0 and cp.p == 1.0
    a = make_arm(np.zeros((3, 50), int), 1)
    a._curve = Va[:, :, None]
    b = make_arm(np.zeros((3, 50), int), 1, "B")
    b._curve = np.broadcast_to(th, (3, 50))[:, :, None].copy()
    cp = st.components(a, b, 1)
    assert cp.seed_var[0] == pytest.approx(np.var(shift_a, ddof=1))
    assert cp.seed_var[1] == pytest.approx(0.0) and cp.res_ms[0] == pytest.approx(0.0)
    assert cp.theorem_var == pytest.approx(0.0)
    assert cp.se == pytest.approx(math.sqrt(np.var(shift_a, ddof=1) / 3))
    assert cp.df == pytest.approx(2.0)  # all variance from 3 seeds
    # power: only seeds help; more theorems never reach a target below the seed term
    assert st.mde(cp, S=12) == pytest.approx(st.mde(cp) / 2)
    assert st.theorems_needed(cp, 0.01) is None
    s = st.seeds_needed(cp, 0.01)
    assert s is not None and st.mde(cp, S=s) <= 0.01 < st.mde(cp, S=s - 1)
    pooled = st.pooled_components([a, b], [(a, b)], 1)
    assert pooled.seed_var == [pytest.approx(np.var(shift_a, ddof=1) / 2)] * 2


def test_component_levels() -> None:
    rng = np.random.default_rng(10)
    arm = make_arm(rng.binomial(16, rng.uniform(0, 1, size=(4, 30))), 16)
    V = arm.values(1)
    seeds = st.components(arm, None, 1, levels="seeds")
    assert seeds.se == pytest.approx(np.std(V.mean(1), ddof=1) / 2)
    assert seeds.df == pytest.approx(3.0)
    thms = st.components(arm, None, 1, levels="theorems")
    assert thms.se == pytest.approx(np.std(V.mean(0), ddof=1) / math.sqrt(30))
    both = st.components(arm, None, 1)
    assert both.se >= thms.se and both.var() == pytest.approx(seeds.var())
    with pytest.raises(ValueError):
        st.components(arm, None, 1, levels="nope")


def test_cluster_robust_components() -> None:
    """Ten clusters of five identical theorems: the clustered SE is that of a mean of ten cluster
    values, and the design effect is about the cluster size."""
    rng = np.random.default_rng(9)
    u = rng.normal(0, 0.1, size=10)
    a = make_arm(np.zeros((3, 50), int), 1)
    a._curve = np.broadcast_to(np.repeat(0.5 + u, 5), (3, 50))[:, :, None].copy()
    b = make_arm(np.zeros((3, 50), int), 1, "B")
    b._curve = np.full((3, 50, 1), 0.5)
    clusters = {nm: str(i // 5) for i, nm in enumerate(a.names)}
    flat = st.components(a, b, 1)
    cl = st.components(a, b, 1, clusters=clusters)
    assert cl.clusters == 10 and flat.clusters is None
    assert cl.se**2 == pytest.approx(np.var(u, ddof=1) / 10)
    assert cl.deff > 4 and cl.var() == pytest.approx(cl.se**2)
    assert cl.df == pytest.approx(9.0)


def test_theorems_needed_when_theorem_noise_dominates() -> None:
    rng = np.random.default_rng(5)
    d = rng.normal(0, 0.1, size=40)
    a = make_arm(np.zeros((3, 40), int), 1)
    a._curve = np.broadcast_to(0.5 + d, (3, 40))[:, :, None].copy()
    b = make_arm(np.zeros((3, 40), int), 1, "B")
    b._curve = np.full((3, 40, 1), 0.5)
    cp = st.components(a, b, 1)
    assert cp.seed_var == [pytest.approx(0.0, abs=1e-20)] * 2
    t = st.theorems_needed(cp, 0.03)
    assert t is not None and st.mde(cp, T=t) <= 0.03 < st.mde(cp, T=t - 1)
    want = (2.8015852 * np.std(d, ddof=1) / 0.03) ** 2
    assert t == pytest.approx(want, rel=0.02)


# ------------------------------------------------------------------------------------------ CLI


def test_table_cli_writes_markdown_and_json(tmp_path: Path) -> None:
    rng = np.random.default_rng(6)
    names = [f"t{i}" for i in range(12)]
    groups = []
    for arm, tok in (("A", 50), ("B", 120)):
        paths = []
        for s in range(2):
            ok = {nm: [int(x) for x in rng.random(16) < 0.4] for nm in names}
            paths.append(str(write_eval_dir(tmp_path / f"{arm}{s}", ok, tok)))
        groups.append(f"{arm}=" + ",".join(paths))
    out, js = tmp_path / "t.md", tmp_path / "t.json"
    argv = ["table", "--group", groups[0], "--group", groups[1], "--k", "1,5,n"]
    argv += ["--tokens", "400,1000", "--draws", "300", "--out", str(out), "--json", str(js)]
    assert st.main(argv) == 0
    md = out.read_text()
    for head in ("| arm | seeds | pass@1 | pass@5 | pass@16 |", "| B − A | pass@1 |",
                 "| difference (pp) | 400 | 1k |", "| pass@16 |", "MDE"):  # fmt: skip
        assert head in md, head
    data = json.loads(js.read_text())
    assert len(data["arms"]) == 2 and len(data["diffs"]) == 3
    assert data["informative"]["pass@1"]["n"] == 12


@needs_eval
def test_real_reads_match_the_evaluators_summary() -> None:
    paths = [EVAL / f"tb72_T0_s{s}" for s in range(3)]
    arm = st.load_arm("T0", paths)
    want = [json.loads((p / "summary.json").read_text())["cells"]["plain_B0"]["all"] for p in paths]
    for k in (1, 4, 256):
        got = arm.values(k).mean(1)
        assert got == pytest.approx([w["pass_at_k_pct"][str(k)] / 100 for w in want])


# ------------------------------------------------------------------- figures.py --ci (CI bands)


def test_figure_ci_bands(tmp_path: Path) -> None:
    from .test_figures import OK, write_tb72
    from .test_figures import write_eval_dir as fig_eval_dir

    refs = F.reference_lengths(write_tb72(tmp_path / "tb"))
    rng = np.random.default_rng(8)
    dirs = []
    for s in range(3):
        ok = {nm: [int(x) for x in rng.random(16) < 0.2 + 0.1 * i] for i, nm in enumerate(OK)}
        dirs.append(fig_eval_dir(tmp_path / f"s{s}", {"plain_B0": ok}, tokens=10 + s))
    g = F.Group("A", [str(d) for d in dirs])
    lb = F.lengthbin_data([g], refs=refs, check_sizes=None, ci=True, ci_draws=300)
    band = lb["groups"][0]["ci"]
    for b in F.BIN_ORDER:
        assert band[b][0] <= lb["groups"][0]["mean"][b] <= band[b][1]
    assert "hierarchical bootstrap" in F.lengthbin_caption(lb)
    assert "no CI" in F.lengthbin_caption(F.lengthbin_data([g], refs=refs, check_sizes=None))
    tk = F.tokens_data([g], arms=["plain"], sets=["all", "textbook_dev"], ci=True, ci_draws=300)
    ser = tk["series"][0]
    for s in ("all", "textbook_dev"):
        assert len(ser["ci"][s]) == len(ser["mean"][s])
        for (lo, hi), (_x, y) in zip(ser["ci"][s], ser["mean"][s], strict=True):
            assert lo <= y <= hi
    assert "hierarchical bootstrap" in F.tokens_caption(tk)
    # the tokens CLI with --ci; the length-bin figure drawn directly (the look-alike's bins are
    # not textbook72's sizes, which the lengthbin CLI checks)
    out = tmp_path / "figs" / "fig_tokens"
    argv = ["tokens", "--out", str(out), "--group", f"A={','.join(map(str, dirs))}"]
    assert F.main([*argv, "--sets", "all", "--ci", "--ci-draws", "200"]) == 0
    assert out.with_suffix(".png").stat().st_size > 0
    assert "ci" in json.loads(out.with_suffix(".json").read_text())["series"][0]
    with F.S.paper_style():
        fig = F.fig_lengthbin(lb)
        files = F._write(fig, tmp_path / "figs" / "fig_lb", lb, F.lengthbin_caption(lb))  # noqa: SLF001
    assert all(f.stat().st_size > 0 for f in files)
