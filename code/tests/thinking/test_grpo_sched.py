"""GRPO's learning-rate schedule (`--lr_schedule`; Leon 2026-10-13): without it the runs are
fa28f6e's (digests of tiny-model runs: rollouts, rows, log lines, weights, args); cosine and wsd
(warmup-stable-decay; linear, cosine and 1-sqrt decay) have their warmup, endpoints and shape; the
rollback multiplier sits on top and the schedule follows the iteration number through rollbacks;
the effective lr is what AdamW steps with; resumes continue the schedule and refuse another."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pytest
import torch

from code.thinking import grpo
from code.thinking.ckpt import load_train_state

from ._grpo_digest import CONFIGS, run_digest, setup_go
from .test_grpo_guard import Go, _same, _weights

BASE_LR = 1e-3  # Go's --lr
COSINE = ["--lr_schedule", "cosine", "--warmup_iters", "2", "--min_lr_frac", "0.1"]
WSD = ["--lr_schedule", "wsd", "--warmup_iters", "1", "--decay_frac", "0.5", "--min_lr_frac", "0.1"]
# line.sh's _w / _wb L3 flags (no --warmup_iters: --warmup's step ramp, as today)
LAUNCH_W = [
    *("--lr_schedule", "wsd", "--decay_frac", "0.2", "--min_lr_frac", "0.3"),
    *("--decay_shape", "sqrt"),
]
SCHEDS = {
    "cosine": COSINE,
    "cosine_steps": ["--lr_schedule", "cosine"],  # the step ramp
    "wsd_linear": [*WSD, "--decay_shape", "linear"],
    "wsd_cosine": [*WSD, "--decay_shape", "cosine"],
    "wsd_sqrt": [*WSD, "--decay_shape", "sqrt"],
    "wsd_steps": [*LAUNCH_W[:2], "--decay_frac", "0.5", *LAUNCH_W[4:]],  # _w's, a short run
}
TIMING = ("secs", "secs_sample", "attempts_per_s")

# `_grpo_digest.run_digest` on a clean export of fa28f6e (`git archive fa28f6e code`, as the
# helper's docstring says for e027354; fa28f6e's code/thinking and code/tests are e027354's, and the
# three CONFIGS digests equal test_grpo_auto.py's on leon/grpo-v2-auto). `rollback` is added here:
# rollbacks that happen (no restore check; the <eos>-biased model raises the think ban alarm every
# iteration) with the default --warmup 10 step ramp, which a rollback restores.
GOLDEN = {
    "guarded": {
        "rollouts": "26bce0a1a4b904364ff844fc851fdee3ec72f1642ea7311ef3b8ced0b019bdb4",
        "rows": "f00e051bf99bc95a56c913cc556957b6a6ee1db069b4da38d54f2f47941245ad",
        "lines": "30738e3973988116d73e13fe5537f9d02f9749e2ba7d1c268ed61e65f7402375",
        "weights": "c5e3c0c060bd913481cb742d90d07ab8a09cc5b00cdfbf181b899ae18a78620e",
        "args": "f1e37f20edb9142f424175253715a729a536b7b32d8d69a4f24b814281f2e4bf",
    },
    "rollback": {
        "rollouts": "e0d3081dd0666d9a1b2209a4daafe47338751226edfe4b4b34032b0a6f343ae2",
        "rows": "596af35c4419faaafc1902126233255fa1d601efbc1ca3df3265ae6ff1d74e37",
        "lines": "2f992523f377f70afdbd83300b96709d3fe13edad70cbbc908bb1880be6d7bd5",
        "weights": "b6c2ccab6a016e2082a6982759dc687854615471436d997506b48120f7a5fce9",
        "args": "c96fb3f9c7aab3764043ce9b9d2c7d16ce0bec0c1f8e2ecc2c761c27cf3208bb",
    },
    "verifier_stop": {
        "rollouts": "a7b347d480893c5c2f3fe646fcc125592f49120262d31115878a206d7589fcd5",
        "rows": "38dc6329f884377893ea3442acd841458cc2c644eca73e392a519fbd89e760ee",
        "lines": "5fd7de2e0afeb36f6927de2bd8625d20dd6bfdfdf53714760654c509d406f982",
        "weights": "f4e31354738a6e0b7e4f990fa7f8d0261bc5af40488fdfeb82a82d15e316f71c",
        "args": "53b771a05a540309fb70769decf19a6b6631b241fd67157f7bffc71770091f78",
    },
    "verifier_redraw": {
        "rollouts": "4a203bd571220088b06d41ee4bebcb47be01fc8121cecdb2243ab3238bb753a1",
        "rows": "8760a1a3a0993dddbfab2f7a195c9064857732ee7cd98e76cd9aa8aaf9b7e71b",
        "lines": "482c3b71607252158aaacc1ddbe3836ab82e29a4e6998f92ed34c9cb5a25aecb",
        "weights": "7c326ff3339f24d81bbe0cb05ab88e2c7692d0bb61774b88ce32c2e7bd3b6220",
        "args": "50651629657b725a0ed35657d2ca732fc9d1ad1413878c2cddca818aea21d166",
    },
}


def _rollback_cfg(iso: Path) -> list[str]:
    return ["--iters", "5", "--rollback_on_alarm", "--save_rolling", "1", "--save_every", "2"]


@pytest.fixture
def go(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Go:
    """test_grpo_guard's `go`: the tiny effort model on an <eos>-biased init, the hash judge."""
    return setup_go(tmp_path, monkeypatch)


def _sched(argv: list[str], iters: int) -> dict[str, Any]:
    a = grpo.parser().parse_args(
        ["--init", "x", "--name", "g", "--out", "o", "--targets", "t", "--iters", str(iters), *argv]
    )
    s = grpo.lr_schedule(a)
    assert s is not None
    return s


def _ms(argv: list[str], iters: int) -> list[float]:
    s = _sched(argv, iters)
    return [grpo.lr_mult(t, s) for t in range(1, iters + 1)]


def _no_timing(r: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in r.items() if k not in TIMING}


# --- the default path ---------------------------------------------------------------------------


@pytest.mark.parametrize("config", [*sorted(CONFIGS), "rollback"])
def test_default_path_is_fa28f6e(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, config: str
) -> None:
    monkeypatch.setitem(CONFIGS, "rollback", _rollback_cfg)
    assert run_digest(tmp_path, monkeypatch, config) == GOLDEN[config]


def test_explicit_constant_is_the_default_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`--lr_schedule constant` given: the same run; args.json records only that option."""
    const = ["--lr_schedule", "constant"]
    monkeypatch.setitem(CONFIGS, "rollback_const", lambda iso: [*_rollback_cfg(iso), *const])
    got = run_digest(tmp_path, monkeypatch, "rollback_const")
    assert {k: v for k, v in got.items() if k != "args"} == {
        k: v for k, v in GOLDEN["rollback"].items() if k != "args"
    }
    args = json.loads((tmp_path / "g" / "args.json").read_text())
    assert args["lr_schedule"] == "constant"
    assert not set(grpo.SCHED_DEFAULTS) & set(args)


def test_args_json_records_the_schedule_only_when_given(go: Go) -> None:
    go(["--iters", "1"], out="const")
    a0 = json.loads((go.iso / "const" / "args.json").read_text())
    assert not {"lr_schedule", *grpo.SCHED_DEFAULTS} & set(a0)
    go(["--iters", "2", "--lr_schedule", "cosine"], out="cos")
    a1 = json.loads((go.iso / "cos" / "args.json").read_text())
    assert {k: a1[k] for k in ("lr_schedule", "warmup_iters", "min_lr_frac")} == {
        "lr_schedule": "cosine",
        "warmup_iters": 0,
        "min_lr_frac": 0.1,
    }
    assert "decay_frac" not in a1 and "decay_shape" not in a1
    hist, _ = go(["--iters", "2", "--lr_schedule", "wsd"], out="wsd")
    a2 = json.loads((go.iso / "wsd" / "args.json").read_text())
    assert (a2["decay_frac"], a2["decay_shape"]) == (0.2, "linear")
    assert hist[0]["alarm_cfg"]["lr_schedule"] == {
        "kind": "wsd",
        "iters": 2,
        "warmup_iters": 0,
        "min_lr_frac": 0.1,
        "decay_frac": 0.2,
        "decay_shape": "linear",
        "warmup_steps": 10,
        "decay_iters": 1,
    }


# --- the schedule's values ----------------------------------------------------------------------


def test_cosine_warmup_endpoints_and_shape() -> None:
    """The launcher's _c: 39 iterations, warmup 3, down to 0.1 (EI's sched over iterations)."""
    m = _ms(["--lr_schedule", "cosine", "--warmup_iters", "3", "--min_lr_frac", "0.1"], 39)
    assert m[:3] == pytest.approx([1 / 3, 2 / 3, 1.0])
    assert m[-1] == pytest.approx(0.1, abs=1e-12)
    assert all(x > y for x, y in zip(m[2:], m[3:], strict=False))  # decreasing after the warmup
    assert m[2 + 18] == pytest.approx(0.55)  # half way through the cosine: (1 + f) / 2
    # train.finetune's formula, at t = s (steps from 1)
    for t, x in enumerate(m, 1):
        if t >= 3:
            want = 0.1 + 0.5 * 0.9 * (1 + math.cos(math.pi * (t - 3) / 36))
            assert x == pytest.approx(want)
    # no warmup: the first iteration is already on the cosine
    m0 = _ms(["--lr_schedule", "cosine"], 10)
    assert m0[0] == pytest.approx(0.1 + 0.45 * (1 + math.cos(math.pi / 10))) and m0[0] < 1
    assert m0[-1] == pytest.approx(0.1)


@pytest.mark.parametrize("shape", grpo.DECAY_SHAPES)
def test_wsd_warmup_stable_decay(shape: str) -> None:
    """The launcher's _w: 39 iterations, warmup 3, the last round(0.2 x 39) = 8 decaying to 0.1."""
    argv = ["--lr_schedule", "wsd", "--warmup_iters", "3", "--decay_frac", "0.2"]
    s = _sched([*argv, "--decay_shape", shape], 39)
    assert s["decay_iters"] == 8
    m = [grpo.lr_mult(t, s) for t in range(1, 40)]
    assert m[:2] == pytest.approx([1 / 3, 2 / 3])
    assert m[2:31] == [1.0] * 29  # iterations 3 .. 31
    dec = m[30:]  # 31 .. 39
    assert all(x > y for x, y in zip(dec, dec[1:], strict=False)) and dec[-1] == pytest.approx(0.1)
    p = [k / 8 for k in range(1, 9)]
    g = {
        "linear": [1 - x for x in p],
        "cosine": [0.5 * (1 + math.cos(math.pi * x)) for x in p],
        "sqrt": [1 - math.sqrt(x) for x in p],
    }[shape]
    assert dec[1:] == pytest.approx([0.1 + 0.9 * x for x in g])
    if shape == "sqrt":  # 1 - sqrt(p) < 1 - p inside the decay: it cools down faster
        lin = [grpo.lr_mult(t, {**s, "decay_shape": "linear"}) for t in range(32, 39)]
        assert all(x < y for x, y in zip(m[31:38], lin, strict=True))


def test_launcher_w_l3_schedule() -> None:
    """_w's L3 over 39 iterations: today's 10-step ramp, m = 1 to iteration 31, then 1 - sqrt
    over the last 8 down to 0.3."""
    s = _sched(LAUNCH_W, 39)
    assert (s["warmup_steps"], s["decay_iters"]) == (10, 8)
    m = [grpo.lr_mult(t, s) for t in range(1, 40)]
    assert m[:31] == [1.0] * 31 and m[-1] == pytest.approx(0.3)
    assert m[31:] == pytest.approx([0.3 + 0.7 * (1 - math.sqrt(k / 8)) for k in range(1, 9)])


def test_decay_iterations_round_half_up_and_at_least_one() -> None:
    def d(frac: str, iters: int) -> int:
        return _sched(["--lr_schedule", "wsd", "--decay_frac", frac], iters)["decay_iters"]

    assert (d("0.2", 39), d("0.25", 10), d("0.2", 2), d("0.01", 5), d("1", 7)) == (8, 3, 1, 1, 7)
    m = _ms(["--lr_schedule", "wsd", "--decay_frac", "1", "--min_lr_frac", "0"], 4)
    assert m == pytest.approx([0.75, 0.5, 0.25, 0.0])  # the whole run decays, linearly


def test_a_run_shorter_than_its_warmup_is_all_warmup() -> None:
    """SMOKE=1's 2 iterations under the launcher's warmup 3: the warmup comes first."""
    assert _ms(["--lr_schedule", "cosine", "--warmup_iters", "3"], 2) == pytest.approx(
        [1 / 3, 2 / 3]
    )
    w = ["--lr_schedule", "wsd", "--warmup_iters", "3", "--decay_frac", "0.2"]
    assert _ms(w, 2) == pytest.approx([1 / 3, 2 / 3])
    # wsd with the warmup ending at the last iteration still decays there; cosine needs W < T
    assert _ms(["--lr_schedule", "wsd", "--warmup_iters", "3"], 3)[-1] == pytest.approx(0.1)
    assert _ms(["--lr_schedule", "cosine", "--warmup_iters", "3"], 3)[-1] == 1.0


@pytest.mark.parametrize(
    "argv",
    [
        ["--warmup_iters", "3"],  # constant: no schedule option
        ["--lr_schedule", "constant", "--min_lr_frac", "0.1"],
        ["--lr_schedule", "cosine", "--decay_frac", "0.2"],  # wsd's options
        ["--lr_schedule", "cosine", "--decay_shape", "sqrt"],
        ["--lr_schedule", "cosine", "--min_lr_frac", "1.5"],
        ["--lr_schedule", "cosine", "--warmup_iters", "-1"],
        ["--lr_schedule", "wsd", "--decay_frac", "0"],
        ["--lr_schedule", "wsd", "--decay_frac", "1.2"],
    ],
)
def test_refused_options(argv: list[str]) -> None:
    a = grpo.parser().parse_args(
        ["--init", "x", "--name", "g", "--out", "o", "--targets", "t", *argv]
    )
    with pytest.raises(AssertionError):
        grpo.lr_schedule(a)


def test_parser_leaves_the_options_out_unless_given() -> None:
    a = grpo.parser().parse_args(["--init", "x", "--name", "g", "--out", "o", "--targets", "t"])
    assert not {"lr_schedule", *grpo.SCHED_DEFAULTS} & set(vars(a))
    assert grpo.lr_schedule(a) is None and not {"lr_schedule", *grpo.SCHED_DEFAULTS} & set(vars(a))
    a = grpo.parser().parse_args(
        ["--init", "x", "--name", "g", "--out", "o", "--targets", "t", "--lr_schedule", "cosine"]
    )
    s = grpo.lr_schedule(a)
    assert s == {
        "kind": "cosine",
        "iters": 280,
        "warmup_iters": 0,
        "min_lr_frac": 0.1,
        "warmup_steps": 10,  # no --warmup_iters: --warmup's step ramp stays
    }
    assert (a.warmup_iters, a.min_lr_frac) == (0, 0.1)  # resolved onto the namespace
    assert _sched(["--lr_schedule", "wsd", "--warmup_iters", "3"], 39)["warmup_steps"] == 0


# --- runs: the optimizer, rollbacks, resumes ----------------------------------------------------


def _walk(hist: list[dict[str, Any]], s: dict[str, Any]) -> tuple[list[float], list[float]]:
    """What the rows imply, independently of grpo's code -> (the lr of every AdamW step, each
    row's lr). Step n of iteration t runs at --lr x m(t) x lr_scale x min(1, n / warmup_steps);
    a row's lr is its last step's, or its next step's when it takes none; a rollback to
    iteration j restores j's step count. Every non-empty minibatch is assumed to step."""
    ws = max(1, s["warmup_steps"])
    n, n_at = 0, {0: 0}
    steps: list[float] = []
    rows: list[float] = []
    for h in hist:
        m, k = grpo.lr_mult(h["iter"], s), h.get("lr_scale", 1.0)
        mine = [BASE_LR * m * k * min(1.0, (n + j) / ws) for j in range(1, h["steps"] + 1)]
        steps += mine
        n += h["steps"]
        rows.append(mine[-1] if mine else BASE_LR * m * k * min(1.0, (n + 1) / ws))
        n_at[h["iter"]] = n
        if h.get("rollback"):
            n = n_at[h["rollback"]["to_iter"]]
            n_at[h["iter"]] = n  # the checkpoints of the iteration hold the restored state
    return steps, rows


def _guard(t: int, lr: float, kind: str, m: float, scale: float, ramp: float) -> str:
    r = f" x ramp {ramp:.2f}" if ramp < 1 else ""
    return f"lr {lr:.3g} = {BASE_LR:g} x {kind} {m:.4f} x {scale:g}{r}"


@pytest.mark.parametrize("name", sorted(SCHEDS))
def test_the_effective_lr_reaches_adamw(
    go: Go, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], name: str
) -> None:
    """Every AdamW step of iteration t runs at --lr x m(t) (x the step ramp without
    --warmup_iters) in every param group; the row's lr, lr_mult, lr_ramp and the guard line say
    so. Go's --updates 2 with --warmup 10: the step ramp covers 5 iterations."""
    seen: list[list[float]] = []
    real_step = torch.optim.AdamW.step

    def spy(self: torch.optim.AdamW, *args: Any, **kw: Any) -> Any:
        seen.append([float(g["lr"]) for g in self.param_groups])
        return real_step(self, *args, **kw)

    monkeypatch.setattr(torch.optim.AdamW, "step", spy)
    hist, _ = go(["--iters", "4", *SCHEDS[name]])
    s = _sched(SCHEDS[name], 4)
    ms = [grpo.lr_mult(t, s) for t in range(1, 5)]
    assert [h["lr_mult"] for h in hist] == ms
    steps, rows = _walk(hist, s)
    assert [h["lr"] for h in hist] == pytest.approx(rows, rel=1e-12)
    assert sum(h["steps"] for h in hist) > 0 and len(seen) == len(steps)
    assert all(
        g == pytest.approx([w] * len(g), rel=1e-12) for g, w in zip(seen, steps, strict=True)
    )
    ramps = [h["lr"] / (BASE_LR * m) for h, m in zip(hist, ms, strict=True)]
    assert [h["lr_ramp"] for h in hist] == pytest.approx(ramps)
    assert (s["warmup_steps"] == 0) == all(r == 1.0 for r in ramps)
    assert hist[-1]["lr_mult"] == pytest.approx(s["min_lr_frac"])
    guard = [x for x in capsys.readouterr().out.splitlines() if x.startswith("guard it ")]
    assert [x.split("; capped")[0] for x in guard] == [
        f"guard it {h['iter']}: " + _guard(h["iter"], h["lr"], s["kind"], m, 1.0, h["lr_ramp"])
        for h, m in zip(hist, ms, strict=True)
    ]


def test_wsd_before_its_decay_is_the_constant_run_bit_for_bit(go: Go) -> None:
    """The launcher's _w L3 (no --warmup_iters: the step ramp as today) over 4 iterations, the
    last one decaying: iterations 1-3 are the constant run's (rows apart from the schedule's keys,
    weights); iteration 4 samples the same attempts and steps at min_lr_frac x its lr."""
    every = ["--iters", "4", "--save_every", "1", "--seed", "2"]  # seed 2: every iteration steps
    const, c_out = go(every, out="const")
    wsd, w_out = go([*every, *LAUNCH_W[:2], "--decay_frac", "0.25", *LAUNCH_W[4:]], out="wsd")
    added = ("lr_mult", "lr_ramp")
    for c, w in zip(const[:3], wsd[:3], strict=True):
        w = {k: v for k, v in w.items() if k not in added}
        w["alarm_cfg"] = {k: v for k, v in w["alarm_cfg"].items() if k != "lr_schedule"}
        assert _no_timing(c) == _no_timing(w)
    for t in (1, 2, 3):
        ck = f"g_it{t}.pt"
        assert _same(_weights(c_out / "ckpts" / ck), _weights(w_out / "ckpts" / ck))
    assert [wsd[3][k] for k in ("reward_pass1", "tokens_per_attempt", "items")] == [
        const[3][k] for k in ("reward_pass1", "tokens_per_attempt", "items")
    ]
    assert wsd[3]["lr_mult"] == pytest.approx(0.3) and wsd[3]["steps"] > 0
    assert wsd[3]["lr"] == pytest.approx(0.3 * const[3]["lr"])
    assert not _same(_weights(c_out / "ckpts" / "g_it4.pt"), _weights(w_out / "ckpts" / "g_it4.pt"))


@pytest.mark.parametrize("name", ["cosine", "wsd_sqrt", "wsd_steps"])
def test_rollbacks_multiply_the_schedule_and_it_follows_the_iteration_number(
    go: Go, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], name: str
) -> None:
    """Alarms at it3 and it5: each restores the newest good checkpoint (iteration 2, then 4) and
    halves the lr from the next iteration. The iterations go on 4, 5, 6 (the restore does not move
    the number back), each at --lr x m(t) x 0.5^rollbacks (x the step ramp, whose step count the
    restore does move back), so the run ends at min_lr_frac x 0.25 at its last iteration."""
    monkeypatch.setattr(grpo, "alarms", lambda row, prev: ["test"] if row["iter"] in (3, 5) else [])
    argv = ["--iters", "6", "--save_rolling", "1", "--save_every", "3", "--rollback_on_alarm"]
    hist, out = go([*argv, *SCHEDS[name]])
    s = _sched(SCHEDS[name], 6)
    assert [h["iter"] for h in hist] == [1, 2, 3, 4, 5, 6]
    assert [(h.get("rollback") or {}).get("to_iter") for h in hist] == [
        None,
        None,
        2,
        None,
        4,
        None,
    ]
    scale = [1.0, 1.0, 1.0, 0.5, 0.5, 0.25]
    assert [(h["lr_scale"], h["rollbacks"]) for h in hist] == list(
        zip(scale, [0, 0, 0, 1, 1, 2], strict=True)
    )
    ms = [grpo.lr_mult(t, s) for t in range(1, 7)]
    assert [h["lr_mult"] for h in hist] == ms
    _, rows = _walk(hist, s)
    assert [h["lr"] for h in hist] == pytest.approx(rows, rel=1e-12)
    assert hist[-1]["lr_mult"] == pytest.approx(s["min_lr_frac"])
    assert hist[-1]["lr"] == pytest.approx(BASE_LR * s["min_lr_frac"] * 0.25 * hist[-1]["lr_ramp"])
    # the rollback's checkpoint holds the restored state; its saved lr scale is the new one
    st = load_train_state(out / "ckpts" / "g_it3.pt")
    assert st is not None and st["iter"] == 3 and st["loop"]["lr_scale"] == 0.5
    assert torch.load(str(out / "ckpts" / "g_it3.pt"))["extra"]["grpo_rollback"]["to_iter"] == 2
    printed = capsys.readouterr().out
    assert "ROLLBACK it 3: restored" in printed and "ROLLBACK it 5: restored" in printed
    last = [x for x in printed.splitlines() if x.startswith("guard it 6:")][0]
    h = hist[-1]
    want = _guard(6, h["lr"], s["kind"], h["lr_mult"], 0.25, h["lr_ramp"])
    assert f"lr x0.25 (2 rollbacks); {want}" in last
    assert f" x {s['kind']} {s['min_lr_frac']:.4f} x 0.25" in last


@pytest.mark.parametrize("name", ["cosine", "wsd_linear"])
def test_resume_continues_the_schedule_exactly(go: Go, name: str) -> None:
    """4 straight iterations, then a resume of the same run from its it2 checkpoint for the last
    2: the same lrs, rows and weights. A 2-iteration run is another schedule (T = 2): its it2
    checkpoint is refused for a 4-iteration resume, and nothing is written."""
    argv = ["--iters", "4", "--save_every", "2", *SCHEDS[name]]
    straight, s_out = go(argv, out="straight")
    _, short = go(["--iters", "2", *SCHEDS[name]], out="short")
    with pytest.raises(AssertionError, match="lr schedule"):
        go([*argv, "--resume", str(short / "ckpts" / "g_it2.pt")], out="refused")
    assert not (go.iso / "refused").exists()
    resumed, r_out = go([*argv, "--resume", str(s_out / "ckpts" / "g_it2.pt")], out="resumed")
    assert [h["iter"] for h in resumed] == [3, 4]
    assert [_no_timing(h) for h in straight[2:]] == [_no_timing(h) for h in resumed]
    assert _same(_weights(s_out / "ckpts" / "g_it4.pt"), _weights(r_out / "ckpts" / "g_it4.pt"))


def test_resume_after_a_rollback_keeps_the_scale_and_the_schedule(
    go: Go, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A rollback at it3, then a resume from it4's rolling checkpoint: it5 and it6 are the
    straight run's (lr scale 0.5 restored, m(t) from the iteration number)."""
    monkeypatch.setattr(grpo, "alarms", lambda row, prev: ["test"] if row["iter"] == 3 else [])
    argv = ["--iters", "6", "--save_rolling", "1", "--save_every", "3", "--rollback_on_alarm"]
    straight, s_out = go([*argv, *SCHEDS["wsd_linear"]], out="straight")
    ck4 = s_out / "ckpts" / "g_it4_rolling.pt"
    resumed, r_out = go([*argv, *SCHEDS["wsd_linear"], "--resume", str(ck4)], out="resumed")
    assert [h["iter"] for h in resumed] == [5, 6]
    assert [(h["lr_scale"], h["rollbacks"]) for h in resumed] == [(0.5, 1), (0.5, 1)]
    assert [_no_timing(h) for h in straight[4:]] == [_no_timing(h) for h in resumed]
    assert _same(_weights(s_out / "ckpts" / "g_it6.pt"), _weights(r_out / "ckpts" / "g_it6.pt"))
    assert resumed[-1]["lr"] == pytest.approx(BASE_LR * 0.1 * 0.5)


@pytest.mark.parametrize(
    ("first", "then"),
    [
        (COSINE, []),  # the schedule dropped
        ([], COSINE),  # a constant run resumed under a schedule
        (COSINE, SCHEDS["wsd_linear"]),  # another kind
        (COSINE, [*COSINE[:-1], "0.2"]),  # another min_lr_frac
    ],
)
def test_resume_refuses_another_schedule(go: Go, first: list[str], then: list[str]) -> None:
    _, out = go(["--iters", "2", *first], out="a")
    ck = out / "ckpts" / "g_it2.pt"
    with pytest.raises(AssertionError, match="lr schedule"):
        go(["--iters", "2", *then, "--resume", str(ck)], out="b")


def test_constant_resume_is_unchecked(go: Go) -> None:
    """Without a schedule a resume may change --iters, as before."""
    _, out = go(["--iters", "2"], out="a")
    hist, _ = go(["--iters", "3", "--resume", str(out / "ckpts" / "g_it2.pt")], out="b")
    assert [h["iter"] for h in hist] == [3] and "lr_mult" not in hist[0]


def test_schedule_namespace_for_saved_args() -> None:
    """The resume check rebuilds the schedule from a checkpoint's saved args (a plain dict)."""
    saved = {"lr_schedule": "wsd", "iters": 39, "warmup_iters": 3, "min_lr_frac": 0.1}
    saved |= {"decay_frac": 0.2, "decay_shape": "linear"}
    s = grpo.lr_schedule(argparse.Namespace(**saved))
    assert s is not None and s["decay_iters"] == 8
