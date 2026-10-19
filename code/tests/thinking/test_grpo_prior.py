"""GRPO's unseen-pair prior (`--unseen_prior`, Leon 2026-10-13): without it, and with `half`, the
runs are 0de954e's (digests of tiny-model runs, the earlier pilot options on too); `mean` puts an
unseen pair at its budget's mean pass rate over the seen pairs, with the same Beta strength for
seen ones, falls back to 1/2 until PRIOR_MIN_SEEN are seen, leaves unseen pairs out of the
--max_pass filter, and logs the unseen share of the draws and of the weight, all recomputed here
from a run's sampled groups."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from code.thinking import grpo

from ._grpo_digest import CONFIGS, run_digest, setup_go
from .test_grpo_guard import _same, _weights
from .test_grpo_pilots import EXTRA as PILOT_EXTRA
from .test_grpo_pilots import GOLDEN as C4_GOLDEN
from .test_grpo_pilots import Pilot

TIMING = ("secs", "secs_sample", "attempts_per_s")
EXTRA: dict[str, Callable[[Path], list[str]]] = {
    # 0de954e's pilot options together (groups of 2, so some lack a signal): refill rounds, pairs
    # over --max_pass, the unlikeliness reward
    "pilots": lambda iso: [
        *("--iters", "4", "--group", "2", "--refill", "--max_pass", "0.6"),
        *("--unlikely_beta", "0.25"),
        *("--overlong", "fail", "--monitor", "--seed", "3"),
    ],
}
# `_grpo_digest.run_digest` on a clean export of 0de954e (`git archive 0de954e code`, loaded first
# on sys.path); its default path is c4b8548's (test_grpo_pilots.py's GOLDEN)
GOLDEN = {
    "pilots": {
        "rollouts": "1351edff212225d72492c8f4bdddccae8af675b10b4bd1b42527f019ac1a0c00",
        "rows": "833fcb3dd6185a58ef4ff6c77b39cbf8ed009bc884e091a35bb2eb27367e3567",
        "lines": "3790d0f1fe8910a642a3e3785a9131fc9461a960572f700ecc3e0aacd7ce44f5",
        "weights": "1fe033667bc3a8002451dfec6c004a40d3211a9437d30b5d13bfb6cb48b7c2b6",
        "args": "c6940ae69fa3b80f072eea9e61bee68ab1de58447bac9beff1b702ed317841fc",
    },
}


# --- 0de954e's runs -----------------------------------------------------------------------------


def test_the_pilot_options_are_0de954e(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(CONFIGS, "pilots", EXTRA["pilots"])
    got = run_digest(tmp_path, monkeypatch, "pilots")
    assert got == GOLDEN["pilots"]
    rows = [json.loads(x) for x in next(tmp_path.rglob("metrics.jsonl")).read_text().splitlines()]
    assert sum(r["refill"]["rounds"] for r in rows) > 0  # the config exercises what it pins
    assert any(r["max_pass"]["over"] for r in rows) and any(r["unlikely"]["groups"] for r in rows)


@pytest.mark.parametrize(
    ("config", "golden"),
    [("plain", C4_GOLDEN["plain"]), ("pilots", GOLDEN["pilots"])],
)
def test_half_is_todays_sampler(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, config: str, golden: dict[str, str]
) -> None:
    """--unseen_prior half: the same rollouts and weights; the rows and lines add its record."""
    base = {**PILOT_EXTRA, **EXTRA}[config]
    monkeypatch.setitem(CONFIGS, "half", lambda iso: [*base(iso), "--unseen_prior", "half"])
    got = run_digest(tmp_path, monkeypatch, "half")
    assert (got["rollouts"], got["weights"]) == (golden["rollouts"], golden["weights"])
    assert got["rows"] != golden["rows"]


def test_the_option_is_left_out_unless_given() -> None:
    base = ["--init", "x", "--name", "g", "--out", "o", "--targets", "t"]
    assert "unseen_prior" not in vars(grpo.parser().parse_args(base))
    a = grpo.parser().parse_args([*base, "--unseen_prior", "mean"])
    assert a.unseen_prior == "mean"
    with pytest.raises(SystemExit):
        grpo.parser().parse_args([*base, "--unseen_prior", "target"])


# --- the estimate -------------------------------------------------------------------------------


def _cur(prior: str, n: int = 40) -> grpo.Curriculum:
    return grpo.Curriculum(n, [0, 64], 0.03, 0.7, seed=5, prior=prior)


def test_half_is_the_beta_1_1_mean() -> None:
    c = _cur("half")
    c.update((0, 0), 3, 4)
    c.update((0, 0), 4, 4)  # s = 0.7 x 3 + 4, n = 0.7 x 4 + 4
    assert c.p(c.index[(0, 0)]) == (0.7 * 3 + 4 + 1.0) / (0.7 * 4 + 4 + 2.0)
    assert c.p(c.index[(1, 0)]) == 0.5
    assert {b: m for b, (m, _) in c.prior_means().items()} == {0: 0.5, 64: 0.5}


def test_mean_prior_per_budget_with_its_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(grpo, "PRIOR_MIN_SEEN", 3)
    c = _cur("mean")
    h = _cur("half")
    for cc in (c, h):
        for t, k in ((0, 4), (1, 3), (2, 2)):  # budget 0: three seen pairs at 1, 3/4, 1/2
            cc.update((t, 0), k, 4)
        cc.update((0, 64), 0, 4)  # budget 64: one seen pair, under the minimum
    m0 = (1.0 + 0.75 + 0.5) / 3
    assert c.prior_means() == {0: (pytest.approx(m0), 3), 64: (0.5, 1)}
    assert c.p(c.index[(5, 0)]) == pytest.approx(m0)  # unseen: the budget's mean
    assert c.p(c.index[(1, 0)]) == pytest.approx((3 + 2 * m0) / (4 + 2))  # pulled toward m0
    # at the 1/2 fallback 'mean' is 'half' bit for bit
    for k in ((0, 64), (5, 64)):
        assert c.p(c.index[k]) == h.p(h.index[k])
    st = c.prior_stats()
    assert (st["prior_mean"]["0"], st["prior_mean"]["64"]) == (pytest.approx(m0), 0.5)
    assert st["fallback"] == [64] and (st["seen"], st["unseen"]) == (4, 76)
    w = [c.weight(i) for i in range(len(c.keys))]
    un = [c.n[i] == 0 for i in range(len(c.keys))]
    w_un = sum(x for x, u in zip(w, un, strict=True) if u)
    assert st["unseen_weight_share"] == pytest.approx(w_un / sum(w))
    assert st["w_unseen"] == pytest.approx(w_un / 76)
    assert st["w_seen"] == pytest.approx((sum(w) - w_un) / 4)
    # the cache follows updates and restores
    c.update((3, 64), 4, 4)
    c.update((4, 64), 4, 4)  # budget 64: 3 seen at 0, 1, 1
    assert c.prior_means()[64] == (pytest.approx(2 / 3), 3)
    snap = c.state()
    c.update((5, 64), 0, 4)
    assert c.prior_means()[64][1] == 4
    c.load_state(snap)
    assert c.prior_means()[64] == (pytest.approx(2 / 3), 3)


def test_draw_uses_the_prior_and_the_same_random_numbers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(grpo, "PRIOR_MIN_SEEN", 2)
    c, h = _cur("mean", 10), _cur("half", 10)
    for cc in (c, h):
        for t in range(4):
            cc.update((t, 0), 4, 4)  # budget 0's seen pairs all solved: m = 1
    assert c.weight(c.index[(9, 0)]) == pytest.approx(0.03)  # m (1 - m) + floor
    assert h.weight(h.index[(9, 0)]) == pytest.approx(0.28)
    dc, dh = c.draw(20), h.draw(20)
    assert sorted(dc) == sorted(dh) and dc != dh  # another order, the same pairs
    assert c.rng.getstate() == h.rng.getstate()
    # --max_pass leaves the unseen pairs (at m = 1 here) in; only the seen ones over it are out
    full = c.draw(20)
    c.rng.setstate(h.rng.getstate())
    kept = c.draw(20, max_pass=0.9)
    assert len(kept) == 16 and set(full) - set(kept) == {(t, 0) for t in range(4)}
    assert (c.over(0.9), h.over(0.9), h.over(0.8)) == (4, 0, 4)  # half: solved pairs at 5/6


# --- a run --------------------------------------------------------------------------------------


@pytest.fixture
def pilot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Pilot:
    monkeypatch.setattr(grpo, "PRIOR_MIN_SEEN", 2)
    return Pilot(setup_go(tmp_path, monkeypatch), monkeypatch)


def _replay(calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """From the sampled groups alone, the curriculum before each call: per pair (s, n)."""
    s: dict[tuple[str, int], float] = {}
    n: dict[tuple[str, int], float] = {}
    before = []
    for c in calls:
        before.append((dict(s), dict(n)))
        for p, b, grp in zip(c["targets"], c["budgets"], c["per"], strict=True):
            k = (p, b)
            s[k] = s.get(k, 0.0) * 0.7 + sum(bool(x["ok"]) for x in grp)
            n[k] = n.get(k, 0.0) * 0.7 + len(grp)
    return [{"s": a, "n": b} for a, b in before]


def test_a_mean_prior_run_logs_what_its_draws_saw(
    pilot: Pilot, capsys: pytest.CaptureFixture[str]
) -> None:
    argv = ["--iters", "4", "--prompts", "4", "--overlong", "fail", "--seed", "1"]
    hist, out = pilot([*argv, "--unseen_prior", "mean"], out="mean")
    calls = pilot.calls
    assert json.loads((out / "args.json").read_text())["unseen_prior"] == "mean"
    prompts = [json.loads(x)["prompt"] for x in pilot.path.read_text().split("\n") if x]
    keys = [(p, b) for p in prompts for b in (0, 16)]
    printed = capsys.readouterr().out
    for h, c, st in zip(hist, calls, _replay(calls), strict=True):
        q = h["prior"]
        drawn = list(zip(c["targets"], c["budgets"], strict=True))
        assert q["drawn"] == len(drawn) == 4 and q["mode"] == "mean"
        assert q["unseen_drawn"] == sum(k not in st["n"] for k in drawn)
        # the prior and the weights at the draw, from the sampled groups alone
        mu = {}
        for b in (0, 16):
            r = [st["s"][k] / st["n"][k] for k in keys if k[1] == b and k in st["n"]]
            mu[b] = sum(r) / len(r) if len(r) >= 2 else 0.5
        assert q["prior_mean"] == pytest.approx({str(b): m for b, m in mu.items()})
        assert q["fallback"] == [b for b in (0, 16) if sum(k[1] == b for k in st["n"]) < 2]

        p = {k: (st["s"].get(k, 0.0) + 2 * mu[k[1]]) / (st["n"].get(k, 0.0) + 2) for k in keys}
        w = {k: x * (1 - x) + 0.03 for k, x in p.items()}
        w_un = sum(w[k] for k in keys if k not in st["n"])
        assert q["unseen_weight_share"] == pytest.approx(w_un / sum(w.values()))
        assert (q["seen"], q["unseen"]) == (len(st["n"]), 12 - len(st["n"]))
        assert f"prior it {h['iter']}: mean, prior mean" in printed
    # iteration 1 sees no pair yet: the 1/2 prior, the half run's draw and rollouts
    half, _ = pilot([*argv, "--unseen_prior", "half"], out="half")
    assert pilot.calls[0]["per"] == calls[0]["per"] and half[0]["prior"]["fallback"] == []
    assert [c["targets"] for c in pilot.calls] != [c["targets"] for c in calls]  # then it moves


def test_resume_keeps_the_prior(pilot: Pilot) -> None:
    argv = ["--iters", "4", "--prompts", "4", "--unseen_prior", "mean", "--save_every", "1"]
    straight, s_out = pilot(argv, out="straight")
    resumed, r_out = pilot([*argv, "--resume", str(s_out / "ckpts" / "g_it2.pt")], out="resumed")
    no = lambda r: {k: v for k, v in r.items() if k not in TIMING}  # noqa: E731
    assert [no(h) for h in straight[2:]] == [no(h) for h in resumed]
    assert _same(_weights(s_out / "ckpts" / "g_it4.pt"), _weights(r_out / "ckpts" / "g_it4.pt"))
