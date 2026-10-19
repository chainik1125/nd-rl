"""GRPO's pilot options (Leon 2026-10-13): refill (dynamic sampling), the pass-rate filter and the
unlikeliness reward (He, Fried & Welleck 2025, arXiv 2506.02355). Without them the runs are
c4b8548's: digests of tiny-model runs (rollouts, rows, log lines, weights, args.json and the batch
dumps), next to test_grpo_sched.py's. Refill draws distinct pairs until --prompts groups have an
outcome signal, trains on those in draw order, and logs the sampled compute (per iteration and
for the run, through rollbacks and resumes); without a skipped group it changes nothing. The
filter drops pairs over its threshold. The unlikeliness reward is the paper's formula (and its
released code's), reads the PPO ratio's log-probs, leaves the skip rule and the step penalty
alone, and is recomputed here from the batch dumps of a run."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import torch

from code.thinking import grpo
from code.thinking.grpo_io import load_batch

from ._grpo_digest import CONFIGS, run_digest, setup_go
from ._pipeline_helpers import pilot_rec
from .test_grpo_guard import Go, _same, _weights
from .test_grpo_verifier import P as REJ_P
from .test_grpo_verifier import T
from .test_grpo_verifier import _att as _vatt

TIMING = ("secs", "secs_sample", "attempts_per_s")


def _dump_digest(d: Path) -> str:
    """sha256 of every batch dump in d (tensors by dtype, shape and bytes; the rest as JSON)."""
    h = hashlib.sha256()

    def feed(x: Any) -> None:
        if isinstance(x, torch.Tensor):
            h.update(str((x.dtype, tuple(x.shape))).encode())
            t = x.detach().cpu().contiguous()
            h.update(t.view(torch.uint8).numpy().tobytes() if t.numel() else b"")
        elif isinstance(x, dict):
            for k in sorted(x, key=str):
                h.update(str(k).encode())
                feed(x[k])
        elif isinstance(x, (list, tuple)):
            h.update(b"[")
            for v in x:
                feed(v)
            h.update(b"]")
        else:
            h.update(json.dumps(x).encode())

    for p in sorted(d.glob("*.pt")):
        h.update(p.name.encode())
        feed(torch.load(str(p), map_location="cpu", weights_only=False))
    return h.hexdigest()


def _mixed(*lp: str) -> Callable[[Path], list[str]]:
    """--overlong mask, both length terms (`lp`: --len_penalty's value, or none), the extra
    targets, the batch dumps and the monitor: the code paths the refill loop rewrote."""
    return lambda iso: [
        *("--iters", "3", "--overlong", "mask", "--len_target_coef", "0.2"),
        *(("--len_penalty", *lp) if lp else ()),
        *("--targets_extra", str(iso / "extra.jsonl")),
        *("--targets_extra_frac", "0.5", "--prompts", "4", "--dump_batches", str(iso / "dumps")),
        *("--monitor", "--seed", "1"),
    ]


EXTRA: dict[str, Callable[[Path], list[str]]] = {
    "plain": lambda iso: ["--iters", "3"],
    "mixed0": _mixed(),  # 2026-10-16: "mixed" without its length penalty
}
# `_grpo_digest.run_digest` (+ `_dump_digest`) on a clean export of c4b8548 (`git archive c4b8548
# code`, loaded first on sys.path), the commit the live v2 runs import; its default path is
# fa28f6e's, so test_grpo_sched.py's GOLDEN digests hold here too
GOLDEN = {
    "plain": {
        "rollouts": "97ee54d7a8861c1ac4d344fad0a0ff87906eff229ec71f585cef62cb8650918f",
        "rows": "facb4a09ac328c90b0906edaf3e4ae63f3c6f0fac52c06e3dbfca1028feccc76",
        "lines": "3e6f3d4d38a104f0ef2be544c406f2b5d3095ecc0ffb5227aefe73be90fefee6",
        "weights": "e70b343d15e22df571f140802a8a8d53131ff6e0161e6d27a4c2a40a9bbc3052",
        "args": "6028cb12ef8e9a6b11421c549bff14a093e9599870ded0f99fe1d652e798e808",
    },
    "mixed0": {
        "rollouts": "a3588a61e81a1f296da3ed5234b54d36808b1e16279d7ac930cd6a1692ed8522",
        "rows": "0bd4f7c7a19af2720cdd158c52326d11dd18f9412b77e8867eff9b96372e7878",
        "lines": "669458118865a5f5632afaf4f5234c816978c7036aa3515f3d76cd82a30811cd",
        "weights": "fdbf480827c6df5614ade1dc957432afe95a7e764c8c4d6c4e726e19a96b295a",
        "args": "89274ce7c4c75d1dacdef9cebcb8b585ad94395d6a3bfee6f679953601f84187",
        "dumps": "9242eec08792c673f06ec2bfeade4dd9ace6478b87a0a5de68df8e9502a5023f",
    },
}
# "mixed" (with --len_penalty 0.05): c4b8548's digests until 2026-10-16, when a length-only group
# (all solved: its rewards differ by the penalty alone) lost its outcome signal and went on top of
# the loss's divisor (test_grpo_len_penalty.py). The sampler's results and args.json are still
# c4b8548's; the rows (`len_only`), lines (`len it`), weights and dumps are this commit's.
MIXED_C4B = {
    "rollouts": "b57a404df572e548e9ac8597fcf220f0754001ddc82c18a284bcc25af98b608e",
    "rows": "6eecf38a5eaa4b84aa15f5759f02359ded070b897c8ed546836c5914aafb1661",
    "lines": "aac9201e2a2947a1636d978ece2cf729e369d9a5a96523f543881ef1ae4691d7",
    "weights": "36187b5167d34a322b42cf9696246b9522d89422cf10db16f6a63cf01159e61c",
    "args": "ceb91c7ef8a696e3fb1924fd1ad759e75cb0044ee16d3ab0879dcecb9304709b",
    "dumps": "53075c218fdc0c8bfa1e5ea897fafcfc632f48de2e60b5abc043f0a191035bb8",
}
MIXED_LEN = {
    **{k: MIXED_C4B[k] for k in ("rollouts", "args")},
    "rows": "a59d3d4497466e7c1816b63ffa713371a546708cdace51851bdbf3411046b654",
    "lines": "ef698b58ae0c495dd1231a26f59ab1f7d30ae5225bf434a5fc297e81b2afac1f",
    "weights": "8eaec7c21671b91ac7aac5ea6779edf1b3408d471345a363e79b9213d305ba9b",
    "dumps": "2f0dd773fb2ad88d2cd817d3d2d8e7a0f3286d15889653df06591a59b0ba7e22",
}
PILOT_ARGS = ("refill", "refill_max", "max_pass", "unlikely_beta")


# --- the default path ---------------------------------------------------------------------------


@pytest.mark.parametrize("config", sorted(EXTRA))
def test_default_path_is_c4b8548(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, config: str
) -> None:
    monkeypatch.setitem(CONFIGS, config, EXTRA[config])
    got = run_digest(tmp_path, monkeypatch, config)
    dumps = list(tmp_path.rglob("dumps"))
    if dumps:
        got["dumps"] = _dump_digest(dumps[0])
    assert got == GOLDEN[config]


def test_the_length_penalty_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """ "mixed" with --len_penalty 0.05: c4b8548's rollouts and args.json, this commit's rest."""
    monkeypatch.setitem(CONFIGS, "mixed", _mixed("0.05"))
    got = run_digest(tmp_path, monkeypatch, "mixed")
    got["dumps"] = _dump_digest(next(tmp_path.rglob("dumps")))
    assert got == MIXED_LEN
    assert {k: v for k, v in got.items() if v == MIXED_C4B[k]} == {
        k: MIXED_C4B[k] for k in ("rollouts", "args")
    }


@pytest.mark.parametrize(
    ("flags", "same"),
    [
        (["--unlikely_beta", "0"], ("rollouts", "rows", "lines", "weights")),
        (["--max_pass", "1"], ("rollouts", "weights")),  # the rows and lines add its record
    ],
)
def test_beta_zero_and_max_pass_one_change_nothing_but_their_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, flags: list[str], same: tuple[str, ...]
) -> None:
    """--unlikely_beta 0 is the plain run but for args.json; no estimate is above --max_pass 1."""
    monkeypatch.setitem(CONFIGS, "flags", lambda iso: [*EXTRA["plain"](iso), *flags])
    got = run_digest(tmp_path, monkeypatch, "flags")
    assert {k: got[k] for k in same} == {k: GOLDEN["plain"][k] for k in same}
    assert got["args"] != GOLDEN["plain"]["args"]


def _ns(*argv: str) -> Any:
    return grpo.parser().parse_args(
        ["--init", "x", "--name", "g", "--out", "o", "--targets", "t", *argv]
    )


def test_options_are_left_out_unless_given_and_checked() -> None:
    a = _ns()
    assert not set(PILOT_ARGS) & set(vars(a))
    assert grpo.pilot_options(a) == (False, None, 0.0) and not set(PILOT_ARGS) & set(vars(a))
    a = _ns("--refill", "--max_pass", "0.9", "--unlikely_beta", "0.25")
    assert grpo.pilot_options(a) == (True, 0.9, 0.25) and a.refill_max == grpo.REFILL_MAX
    for bad in (
        ["--refill_max", "3"],  # without --refill
        ["--refill", "--refill_max", "0"],
        ["--max_pass", "0.4"],  # would filter the unseen pairs (p = 0.5)
        ["--max_pass", "1.1"],
        ["--unlikely_beta", "-0.1"],
        ["--unlikely_beta", "1.5"],
    ):
        with pytest.raises(AssertionError):
            grpo.pilot_options(_ns(*bad))


# --- the curriculum's draw ------------------------------------------------------------------------


def test_draw_excludes_and_filters_with_the_same_random_numbers() -> None:
    c1, c2 = (grpo.Curriculum(5, [0, 16], 0.03, 0.7, seed=3) for _ in range(2))
    for c in (c1, c2):
        c.update((0, 0), 4, 4)  # p = 5/6
        c.update((2, 16), 1, 4)  # p = 1/3
    full = c1.draw(10)
    part = c2.draw(10, {c2.index[(1, 16)]}, max_pass=0.8)
    assert part == [k for k in full if k not in ((1, 16), (0, 0))]
    assert c1.rng.getstate() == c2.rng.getstate()  # one number per pair either way
    assert (c2.over(0.8), c2.over(0.5), c2.over(0.9)) == (1, 1, 0)  # unseen pairs sit at 0.5
    assert len(c2.draw(10, max_pass=0.5)) == 9  # only the pair over 0.5 is left out


# --- the unlikeliness reward ----------------------------------------------------------------------


def test_frac_ranks() -> None:
    assert grpo.frac_ranks([3.0, 1.0, 2.0]) == [2, 0, 1]
    assert grpo.frac_ranks([1.0, 1.0, 2.0]) == [0.5, 0.5, 2]
    assert grpo.frac_ranks([5.0, 5.0, 5.0]) == [1, 1, 1]


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_the_deduction_is_the_papers_and_its_released_codes(seed: int) -> None:
    """r_i = R(x, y_i) (1 - beta (G - rank(y_i)) / G), rank 1 the most likely of the G (Sec.
    4.1), and the released code's R x (1 - beta x argsort(argsort(log p)) / G)."""
    g, beta = 8, 0.25
    torch.manual_seed(seed)
    lp = (-torch.rand(g) * 3).tolist()
    ok = [bool(v) for v in (torch.rand(g) < 0.6).tolist()]
    att = [{"ok": o} for o in ok]
    pen = grpo.unlikely_penalties(att, lp, [False] * g, beta)
    rank = {j: i + 1 for i, j in enumerate(sorted(range(g), key=lambda j: -lp[j]))}
    paper = [float(o) * (1 - beta * (g - rank[j]) / g) for j, o in enumerate(ok)]
    assert [float(o) - u for o, u in zip(ok, pen, strict=True)] == pytest.approx(paper)
    code_r = (torch.argsort(torch.argsort(torch.tensor(lp))).double() / g).tolist()
    assert pen == pytest.approx([float(o) * beta * r for o, r in zip(ok, code_r, strict=True)])
    # failures pay nothing; masked or unscored attempts are out of the ranking and of n
    sc: list[float | None] = [-1.0, -2.0, None, -0.5, -3.0]
    att5 = [{"ok": True}] * 5
    got = grpo.unlikely_penalties(att5, sc, [False, False, False, False, True], 0.5)
    # ranked: -1 (1 below it), -2 (0), -0.5 (2); n = 3
    assert got == pytest.approx([0.5 / 3, 0.0, 0.0, 1.0 / 3, 0.0])


def test_attempt_logp_is_the_mean_over_the_attempts_own_scored_tokens() -> None:
    src = [(0, "acc"), (0, "acc"), (1, "acc"), (1, "rej"), (2, "acc")]
    old = [
        torch.tensor([-1.0, -2.0]),
        torch.tensor([-3.0]),
        torch.tensor([-0.5, -math.inf]),  # banned in its phase: left out
        torch.tensor([-9.0, -9.0]),  # a redrawn draw: not part of the attempt
        torch.tensor([-math.inf]),
    ]
    assert grpo.attempt_logp(4, src, old) == [-2.0, -0.5, None, None]


def _oatt(i: int, ok: bool, status: str | None = None) -> dict[str, Any]:
    step = {"prompt": bytes([5, 6]), "gen": bytes([10 + i, 11 + i]), "forced": False}
    st = status or ("done" if ok else "fail:S:bad")
    return {"status": st, "ok": ok, "tokens": 2, "think": 0, "trace": [step]}


def _adv(items: list[grpo.Item] | None) -> dict[int, float]:
    return {} if items is None else {g[0] - 10: a for _, g, _, a in items}


def test_group_items_with_the_deductions() -> None:
    mixed = [_oatt(0, True), _oatt(1, True), _oatt(2, False), _oatt(3, False)]
    sig: list[bool] = []
    items, _ = grpo.group_items(mixed, T, 0.0, "fail", signal=sig, unlikely=[0.2, 0.0, 0.0, 0.0])
    # rewards 0.8, 1, 0, 0: mean 0.45
    assert sig == [True] and _adv(items) == pytest.approx({0: 0.35, 1: 0.55, 2: -0.45, 3: -0.45})
    # the skip rule reads the rewards before the deduction: an all-solved group stays skipped
    solved = [_oatt(i, True) for i in range(4)]
    assert (
        grpo.group_items(solved, T, 0.0, "fail", signal=sig, unlikely=[0.1, 0.2, 0, 0])[0] is None
    )
    assert sig == [True, False]
    # --overlong mask: the masked attempt is out of the baseline (its deduction never applies)
    ol = [_oatt(0, True), _oatt(1, False), _oatt(2, False, "fail:step_cap")]
    items, masked = grpo.group_items(ol, T, 0.0, "mask", unlikely=[0.25, 0.0, 0.0])
    assert masked == [False, False, True] and _adv(items) == pytest.approx({0: 0.375, 1: -0.375})
    # --overlong fail: a capped attempt is a failure, in the baseline
    items, _ = grpo.group_items(ol, T, 0.0, "fail", unlikely=[0.25, 0.0, 0.0])
    assert _adv(items) == pytest.approx({0: 0.5, 1: -0.25, 2: -0.25})


def test_the_step_penalty_stays_on_top() -> None:
    """With --rej_penalty the rejected draws keep -P on top of their attempt's advantage; only the
    outcome part moves."""
    att = [_vatt(True, [0]), _vatt(True, []), _vatt(False, [1], end_rejected=True)]
    src: list[tuple[int, str]] = []
    base, _ = grpo.group_items(att, T, 0.0, "fail", rej_penalty=REJ_P, src=src)
    src2: list[tuple[int, str]] = []
    pen = [0.25, 0.0, 0.0]
    got, _ = grpo.group_items(att, T, 0.0, "fail", rej_penalty=REJ_P, src=src2, unlikely=pen)
    assert base is not None and got is not None and src == src2
    # attempt 0: its step, its redrawn draw; 1: its step; 2: its step, the draw that ended it,
    # its redrawn draw
    assert src == [(0, "acc"), (0, "rej"), (1, "acc"), (2, "acc"), (2, "acc"), (2, "rej")]
    assert all(x[:3] == b[:3] for x, b in zip(got, base, strict=True))
    m = (0.75 + 1.0 + 0.0) / 3  # the mean reward after the deduction
    want = [0.75 - m, -REJ_P, 1.0 - m, -m, -m - REJ_P, -REJ_P]
    assert [x[3] for x in got] == pytest.approx(want)
    m0 = 2 / 3
    assert [x[3] for x in base] == pytest.approx([1 - m0, -REJ_P, 1 - m0, -m0, -m0 - REJ_P, -REJ_P])


# --- runs: a tiny model with a judge by target ----------------------------------------------------

SIX = ["k12_37120", "k12_16472", "k12_22530", "k12_89161", "k12_53241", "k12_152829"]


class Pilot:
    """`Go` on six targets (budgets 0 and 16: 12 pairs) with a judge by target: the first `easy`
    targets are always solved (no signal), the others solved by every other attempt (a signal in
    every group). `calls`: per `sample_groups` call (one per round) its targets, budgets and
    groups."""

    def __init__(self, go: Go, mp: pytest.MonkeyPatch, easy: int = 3) -> None:
        self.go = go
        rows = [{"prompt": pilot_rec(r)["prompt"]} for r in SIX]
        self.path = go.iso / "six.jsonl"
        self.path.write_text("".join(json.dumps(r) + "\n" for r in rows))
        self.easy = {r["prompt"] for r in rows[:easy]}
        self.calls: list[dict[str, Any]] = []
        mp.setattr(grpo, "judge", self.judge)
        real = grpo.sample_groups

        def spy(model: Any, tok: Any, targets: Any, budgets: Any, a: Any, *rest: Any) -> Any:
            per = real(model, tok, targets, budgets, a, *rest)
            prompts = [t["prompt"] for t in targets]
            self.calls.append({"targets": prompts, "budgets": list(budgets), "per": per})
            return per

        mp.setattr(grpo, "sample_groups", spy)

    def judge(self, tok: Any, prompts: Any, res: list[dict[str, Any]], lean: Any = None) -> Any:
        for i, (p, r) in enumerate(zip(prompts, res, strict=True)):
            r["ok"] = p in self.easy or i % 2 == 0  # groups are runs of --group (4) attempts
        return res

    def __call__(self, argv: list[str], out: str = "g") -> tuple[list[dict[str, Any]], Path]:
        self.calls = []
        return self.go([*argv, "--targets", str(self.path)], out)


@pytest.fixture
def pilot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Pilot:
    return Pilot(setup_go(tmp_path, monkeypatch), monkeypatch)


def _no(r: dict[str, Any], *keys: str) -> dict[str, Any]:
    return {k: v for k, v in r.items() if k not in (*TIMING, *keys)}


def test_refill_fills_the_batch_and_counts_the_compute(
    pilot: Pilot, capsys: pytest.CaptureFixture[str]
) -> None:
    dumps = pilot.go.iso / "dumps"
    argv = ["--iters", "4", "--refill", "--prompts", "4", "--overlong", "fail", "--seed", "1"]
    hist, out = pilot([*argv, "--dump_batches", str(dumps), "--monitor"])
    assert json.loads((out / "args.json").read_text())["refill_max"] == 2
    k, cum, P = 0, {"groups": 0, "attempts": 0, "tokens": 0}, 4
    printed = capsys.readouterr().out
    for h in hist:
        f = h["refill"]
        calls = pilot.calls[k : k + 1 + f["rounds"]]
        k += len(calls)
        drawn = [len(c["targets"]) for c in calls]
        assert f["drawn"] == drawn and f["rounds"] <= 2 and f["target"] == P
        pairs = [(p, b) for c in calls for p, b in zip(c["targets"], c["budgets"], strict=True)]
        assert len(set(pairs)) == len(pairs)  # distinct within the iteration
        sig = [[p not in pilot.easy for p in c["targets"]] for c in calls]
        assert f["signal"] == [sum(s) for s in sig]
        # each refill round: the deficit at the signal share so far, at most --prompts, at most
        # the pairs not yet drawn this iteration
        for r in range(1, len(calls)):
            have, s = sum(drawn[:r]), sum(f["signal"][:r])
            want = P if s == 0 else min(P, math.ceil((P - s) / (s / have)))
            assert drawn[r] == min(want, 12 - have) and s < P
        # trained: the groups with a signal, in draw order, until --prompts
        flat = [x for s in sig for x in s]
        trained, n = [], 0
        for x in flat:
            trained.append(x and n < P)
            n += trained[-1]
        assert f["signal_groups"] == sum(trained) == min(P, sum(flat))
        assert f["surplus"] == sum(flat) - sum(trained)
        assert f["skip_before"] == pytest.approx(1 - sum(sig[0]) / drawn[0])
        assert f["skip_after"] == pytest.approx(1 - sum(trained) / P)
        d = load_batch(dumps / f"g_it{h['iter']}_batch.pt")
        assert d["group_skipped"].tolist() == [not t for t in trained]
        assert set(d["attempt_group"][d["item_attempt"].long()].tolist()) == {
            i for i, t in enumerate(trained) if t
        }
        # the sampling rows cover every group sampled
        att = [x for c in calls for grp_ in c["per"] for x in grp_]
        assert h["groups_skipped"] == pytest.approx(flat.count(False) / len(flat))
        assert h["reward_pass1"] == pytest.approx(sum(bool(x["ok"]) for x in att) / len(att))
        assert sum(v["groups"] for v in h["monitor"]["by_pass"].values()) == len(flat)
        n_tok = sum(int(x["tokens"]) for x in att)
        assert (f["groups"], f["attempts"], f["tokens"]) == (len(flat), len(att), n_tok)
        assert len(att) == 4 * len(flat)  # --group 4
        for key, v in (("groups", len(flat)), ("attempts", len(att)), ("tokens", n_tok)):
            cum[key] += v
            assert f[f"{key}_cum"] == cum[key]
        assert f"refill it {h['iter']}: {f['rounds']} rounds, drawn {drawn}" in printed
    assert k == len(pilot.calls) and sum(h["refill"]["rounds"] for h in hist) > 0


def test_refill_without_a_skipped_group_is_the_plain_run(pilot: Pilot) -> None:
    pilot.easy = set()  # every group has a signal
    argv = [
        "--iters",
        "3",
        "--prompts",
        "4",
        "--save_every",
        "1",
        "--seed",
        "2",
        "--overlong",
        "fail",
    ]
    plain, p_out = pilot(argv, out="plain")
    p_calls = pilot.calls
    refill, r_out = pilot([*argv, "--refill"], out="refill")
    assert len(pilot.calls) == len(p_calls) == 3
    assert [c["per"] for c in pilot.calls] == [c["per"] for c in p_calls]
    assert [_no(h, "refill") for h in refill] == [_no(h) for h in plain]
    assert all(h["refill"]["rounds"] == 0 for h in refill)
    for t in (1, 2, 3):
        ck = f"g_it{t}.pt"
        assert _same(_weights(p_out / "ckpts" / ck), _weights(r_out / "ckpts" / ck))


def test_refill_totals_survive_rollbacks_and_resumes(
    pilot: Pilot, monkeypatch: pytest.MonkeyPatch
) -> None:
    argv = ["--iters", "4", "--prompts", "4", "--refill", "--save_every", "1", "--seed", "1"]
    straight, s_out = pilot(argv, out="straight")
    resumed, _ = pilot([*argv, "--resume", str(s_out / "ckpts" / "g_it2.pt")], out="resumed")
    assert [h["iter"] for h in resumed] == [3, 4]
    assert [_no(h) for h in straight[2:]] == [_no(h) for h in resumed]
    # a rollback restores the curricula and the weights, not the compute spent
    monkeypatch.setattr(grpo, "alarms", lambda row, prev: ["test"] if row["iter"] == 2 else [])
    hist, _ = pilot([*argv, "--rollback_on_alarm", "--save_rolling", "1"], out="rb")
    assert hist[1]["rollback"]["to_iter"] == 1
    for key in ("groups", "attempts", "tokens"):
        run = [h["refill"][key] for h in hist]
        assert [h["refill"][f"{key}_cum"] for h in hist] == [sum(run[: i + 1]) for i in range(4)]


def test_max_pass_drops_the_pairs_over_it(pilot: Pilot) -> None:
    """An always-solved pair is at p = 5/6 after its first group of 4: over 0.8, never drawn
    again. The others sit at 1/2."""
    hist, out = pilot(["--iters", "5", "--prompts", "4", "--max_pass", "0.8", "--seed", "3"])
    assert json.loads((out / "args.json").read_text())["max_pass"] == 0.8
    seen: set[tuple[str, int]] = set()
    for h, c in zip(hist, pilot.calls, strict=True):
        pairs = list(zip(c["targets"], c["budgets"], strict=True))
        easy = {x for x in pairs if x[0] in pilot.easy}
        assert not easy & seen
        seen |= easy
        assert h["max_pass"] == {"threshold": 0.8, "over": len(seen)}
    assert seen  # the filter had something to do


def test_unlikeliness_advantages_recomputed_from_the_dumps(
    pilot: Pilot, capsys: pytest.CaptureFixture[str]
) -> None:
    """--unlikely_beta 0.25: from each iteration's batch dump (the old log-probs, the attempts and
    their items), the ranks, rewards and advantages, independently of grpo's code. Iteration 1
    samples and scores the same tokens as the plain run; its advantages move."""
    argv = ["--iters", "3", "--prompts", "4", "--overlong", "fail", "--seed", "1"]
    base, _ = pilot([*argv, "--dump_batches", str(pilot.go.iso / "d0")], out="base")
    hist, _ = pilot(
        [*argv, "--unlikely_beta", "0.25", "--dump_batches", str(pilot.go.iso / "d1")], out="ul"
    )
    printed = capsys.readouterr().out
    beta, moved = 0.25, 0
    for h in hist:
        d = load_batch(pilot.go.iso / "d1" / f"g_it{h['iter']}_batch.pt")
        ia, ag = d["item_attempt"].tolist(), d["attempt_group"].tolist()
        ok, oo = d["attempt_ok"].tolist(), d["old_off"].tolist()
        old = d["old_logp"]
        n_ranked = 0
        for g, skipped in enumerate(d["group_skipped"].tolist()):
            A = [j for j, x in enumerate(ag) if x == g]
            if skipped:
                assert len({ok[j] for j in A}) == 1  # no signal: all solved (easy) or failed
                continue
            tot = {j: 0.0 for j in A}
            cnt = {j: 0 for j in A}
            for k, j in enumerate(ia):
                if j in tot:
                    o = old[oo[k] : oo[k + 1]]
                    o = o[torch.isfinite(o)]
                    tot[j] += float(o.sum())
                    cnt[j] += o.numel()
            lp = {j: tot[j] / cnt[j] for j in A}
            rho = {
                j: sum(lp[i] < lp[j] for i in A) + (sum(lp[i] == lp[j] for i in A) - 1) / 2
                for j in A
            }
            r = {j: float(ok[j]) * (1 - beta * rho[j] / len(A)) for j in A}
            mean = sum(r.values()) / len(A)
            assert [d["attempt_reward"][j].item() for j in A] == pytest.approx([r[j] for j in A])
            for k, j in enumerate(ia):
                if j in tot:
                    assert d["adv"][k].item() == pytest.approx(r[j] - mean)
            n_ranked += sum(ok[j] for j in A)
            moved += any(rho[j] > 0 and ok[j] for j in A)
        u = h["unlikely"]
        assert u["beta"] == beta and u["solved"] == n_ranked
        assert u["groups"] == sum(not s for s in d["group_skipped"].tolist())
        assert f"unlikely it {h['iter']}: beta 0.25, {u['groups']} groups" in printed
    assert moved
    d0 = load_batch(pilot.go.iso / "d0" / "g_it1_batch.pt")
    d1 = load_batch(pilot.go.iso / "d1" / "g_it1_batch.pt")
    for key in ("prompt", "gen", "mask", "old_logp", "item_attempt", "group_skipped"):
        assert torch.equal(d0[key], d1[key]), key
    assert not torch.equal(d0["adv"], d1["adv"])
    assert _no(base[0], "unlikely")["reward_pass1"] == hist[0]["reward_pass1"]


def test_refill_draws_the_extra_share_from_the_extra_targets(pilot: Pilot) -> None:
    """--targets_extra_frac f: round(f x m) of a refill round's m pairs from the extra file, while
    it has undrawn pairs (Go's one extra target, always solved here: 2 pairs, both in round 0)."""
    extra = str(pilot.go.iso / "extra.jsonl")
    argv = ["--iters", "2", "--prompts", "4", "--refill", "--overlong", "fail", "--seed", "1"]
    hist, _ = pilot([*argv, "--targets_extra", extra, "--targets_extra_frac", "0.5"])
    x_prompt = json.loads(Path(extra).read_text())["prompt"]
    k = 0
    for h in hist:
        f = h["refill"]
        calls = pilot.calls[k : k + 1 + f["rounds"]]
        k += len(calls)
        n_x = [sum(p == x_prompt for p in c["targets"]) for c in calls]
        assert n_x[0] == 2 and not any(n_x[1:])  # both extra pairs in round 0, none left after
        sig = [sum(p not in pilot.easy for p in c["targets"]) for c in calls]
        assert h["extra_groups"] == 2 and f["signal"] == sig
        assert f["rounds"] >= 1 and f["signal_groups"] == min(4, sum(sig))
