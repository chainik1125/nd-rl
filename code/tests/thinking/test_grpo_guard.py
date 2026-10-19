"""GRPO safeguards (collapse audit 2026-10-08): the capped-attempt share and its alarm, the relative
clip-low and canary alarms, the long-context canary and its file (rows, layout, name shift, the
canary / anchor split), the KL anchor, exact resume from a checkpoint's training state, rollback on
an alarm, the think ban restore check and stop (2026-10-11), the batch dumps, and long proofs mixed
into the pool."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import zlib
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
import torch

from code.thinking import grpo, progtrace
from code.thinking.canary import (
    FORMAT,
    anchor_kl,
    b0_prompt,
    canary_metrics,
    load_canary,
    n_hyp,
    name_base,
    proof_row,
)
from code.thinking.ckpt import TRAIN_STATE, load_train_state, save_ckpt
from code.thinking.data import trace_pairs
from code.thinking.effort import EffortTokenizer, effort_id
from code.thinking.grpo_io import compact_adam, dump_batch, load_batch, unpack_items
from code.thinking.sample import phase_bans
from code.thinking.tok import ThinkTokenizer

from ._pipeline_helpers import bias_logits, isolate, pilot_rec, tiny_model

T = ThinkTokenizer()
E = EffortTokenizer()
# two reference proofs (a K12 training item, 11 lines; a long-pool item, 13 lines), inline so the
# tests need no local data
PROOFS = [
    (
        "k12_fixture",
        "THM ( ( R v P ) v ( S > Q ) ) , ( ( S > Q ) > ( ( R v P ) & ( P > P ) ) ) SEQ "
        "( R > ( ( R v P ) & ( P > P ) ) ) PRF",
        "N1 ( ( R v P ) v ( S > Q ) ) : PR ; N2 ( ( S > Q ) > ( ( R v P ) & ( P > P ) ) ) : PR ; "
        "N3 | R : AS ; N4 | | P : AS ; N5 | ( P > P ) : IMPI N4 N4 ; N6 | | ( R v P ) : AS ; "
        "N7 | | ( ( R v P ) & ( P > P ) ) : ANDI N6 N5 ; N8 | | ( S > Q ) : AS ; "
        "N9 | | ( ( R v P ) & ( P > P ) ) : IMPE N2 N8 ; "
        "N10 | ( ( R v P ) & ( P > P ) ) : ORE N1 N6 N7 N8 N9 ; "
        "N11 ( R > ( ( R v P ) & ( P > P ) ) ) : IMPI N3 N10 ; QED",
    ),
    (
        "rh_171648",
        "THM Q , ( ( ( R v S ) v Q ) v ( S & P ) ) SEQ "
        "( ( ~ ( ~ ( P & R ) ) ) > ( ( R v S ) v Q ) ) PRF",
        "N1 Q : PR ; N2 ( ( ( R v S ) v Q ) v ( S & P ) ) : PR ; N3 | ( ( R v S ) v Q ) : AS ; "
        "N4 | ( ( ( R v S ) v Q ) & Q ) : ANDI N3 N1 ; N5 | ( S & P ) : AS ; N6 | Q : R N1 ; "
        "N7 | ( ( R v S ) v Q ) : ORI2 N6 ; N8 | Q : R N1 ; "
        "N9 | ( ( ( R v S ) v Q ) & Q ) : ANDI N7 N8 ; "
        "N10 ( ( ( R v S ) v Q ) & Q ) : ORE N2 N3 N4 N5 N9 ; N11 | ( ~ ( ~ ( P & R ) ) ) : AS ; "
        "N12 | ( ( R v S ) v Q ) : ANDE1 N10 ; "
        "N13 ( ( ~ ( ~ ( P & R ) ) ) > ( ( R v S ) v Q ) ) : IMPI N11 N12 ; QED",
    ),
]
MIN_HYP = 4  # the fixture proofs are short: their longest states hold 7-9 hypothesis lines


def _canary_file(path: Path, min_hyp: int = MIN_HYP) -> Path:
    rows = [proof_row(n, p, q, min_hyp) for n, p, q in PROOFS]
    assert all(r is not None for r in rows)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


# --- capped attempts -----------------------------------------------------------------------------


def _att(status: str, ok: bool = False) -> dict[str, Any]:
    step = {"prompt": bytes([5, 6]), "gen": bytes([10, 11]), "forced": False}
    return {"status": status, "ok": ok, "tokens": 2, "think": 0, "trace": [step]}


def test_capped_share_counts_every_cap_whatever_overlong_does() -> None:
    groups = [
        [_att("done", True), _att("fail:truncated"), _att("fail:S:bad"), _att("fail:step_cap")],
        [_att("done", True), _att("done"), _att("fail:context"), _att("fail:S:bad")],
        [_att("done", True), _att("fail:S:bad")],
    ]
    m = grpo.capped_metrics(groups, [256, 0, 256])
    assert m["capped_frac"] == pytest.approx(3 / 10)
    assert m["capped_by_budget"] == pytest.approx({"0": 1 / 4, "256": 2 / 6})
    assert m["capped_kinds"] == {"step_cap": 1, "truncated": 1, "context": 1}
    # --overlong fail masks nothing (the old overlong_frac is 0); the capped share is unchanged
    for att in groups:
        _, masked = grpo.group_items(att, T, 0.0, "fail")
        assert not any(masked)
    assert grpo.capped_metrics([[_att("done", True)]], [16])["capped_frac"] == 0.0


# --- alarms --------------------------------------------------------------------------------------


def _row(**kw: Any) -> dict[str, Any]:
    row = dict(
        ban_think=1e-5,
        ban_think_max=1e-4,
        grad_norm_max=0.1,
        clip_low=0.002,
        overlong_frac=0.0,
        nonfinite=0,
        capped_by_budget={"0": 0.0, "256": 0.01},
        clip_low_median=0.0019,
        alarm_cfg={"capped": 0.05, "clip_low_rel": 2.0, "canary_mult": 5.0},
    )
    return {**row, **kw}


BASE = {"nll": 0.2, "wrong_max": 0.004, "colon": 1e-5, "wrong_tok": ")"}


@pytest.mark.parametrize(
    ("kw", "want"),
    [
        ({}, None),
        ({"capped_by_budget": {"0": 0.02, "256": 0.08}}, "capped attempts B256 0.0800 > 0.05"),
        ({"capped_by_budget": {"0": 0.06, "256": 0.3}}, "capped attempts B0 0.0600, B256 0.3000"),
        ({"capped_by_budget": {"256": 0.3}, "alarm_cfg": {"capped": 0.0}}, None),  # off
        ({"clip_low": 0.0066}, "clip-low 0.0066 > 2 x running median 0.0019"),  # seed 0, it48
        ({"clip_low": 0.0030}, None),  # seed 1's largest
        ({"clip_low": 0.0066, "clip_low_median": None}, None),  # warm-up
        ({"clip_low": 0.0008, "clip_low_median": 0.0}, None),  # under the floor
        ({"clip_low": 0.0066, "alarm_cfg": {}}, None),  # a row without thresholds
        ({"canary": {"nll": 0.3, "wrong_max": 0.005, "colon": 2e-5, "base": BASE}}, None),
        (
            {"canary": {"nll": 1.2, "wrong_max": 0.005, "colon": 2e-5, "base": BASE}},
            "canary action NLL 1.2 > 5 x base 0.2",
        ),
        (
            {"canary": {"nll": 0.3, "wrong_max": 0.062, "colon": 2e-5, "base": BASE}},
            "canary top wrong-token P 0.062 > 5 x base 0.004",
        ),
        (
            {"canary": {"nll": 0.3, "wrong_max": 0.005, "colon": 9.2e-5, "base": BASE}},
            "canary P(':') 9.2e-05 > 5 x base 1e-05",
        ),
        ({"canary": {"nll": 9.0, "wrong_max": 1.0, "colon": 1.0, "base": None}}, None),
    ],
)
def test_safeguard_alarms(kw: dict[str, Any], want: str | None) -> None:
    got = grpo.alarms(_row(**kw), [0.1] * 9)
    assert got == [] if want is None else (len(got) == 1 and got[0].startswith(want))


# --- the canary file and metrics ------------------------------------------------------------------


def test_canary_rows_are_the_stage1_data_path() -> None:
    """proof_row keeps the steps with >= min_hyp hypothesis lines; their states and actions are
    the ids `progtrace.trace_for` -> `data.trace_pairs` gives (the audit's data path)."""
    name, prompt, proof = PROOFS[1]
    row = proof_row(name, prompt, proof, MIN_HYP, source="fixture")
    assert row is not None and row["format"] == FORMAT and row["source"] == "fixture"
    assert proof_row(name, prompt, proof, 99) is None
    rec, _ = progtrace.trace_for(name, prompt, proof, progtrace.Cfg.v1(), seed=0)
    pairs = trace_pairs(rec, T)
    assert row["n_steps"] == len(pairs)
    want = [i for i, (p, _, _) in enumerate(pairs) if n_hyp([T.itos[x] for x in p]) >= MIN_HYP]
    assert [s["i"] for s in row["steps"]] == want and want
    for s in row["steps"]:
        p, q, tl = pairs[s["i"]]
        assert T.encode_toks(s["state"].split()) == list(p)
        assert T.encode_toks(s["action"].split()) == list(q[tl:])  # action <eos>
        assert s["n_hyp"] == s["state"].split().count(":") >= MIN_HYP
    assert 0 <= row["base"] <= 32 and row["base"] == name_base(name, 13)


def test_canary_layout_shift_and_split(tmp_path: Path) -> None:
    path = _canary_file(tmp_path / "c.jsonl")
    rows = [json.loads(x) for x in path.read_text().splitlines()]
    n_all = sum(len(r["steps"]) for r in rows)
    cd = load_canary(path, E, True, 3)
    assert len(cd.canary) == 3 and len(cd.canary) + len(cd.anchor) == n_all and cd.n_proofs == 1
    tail = [effort_id(E, 0), E.think, E.end_think]
    for p, t in cd.canary + cd.anchor:
        assert p[-3:] == tail and t[-1] == E.eos and E.end_think not in t
    # the canary's proof is not in the anchor: every step of a proof shares its name base
    by_name = {r["name"]: r for r in rows}
    first = sorted(by_name, key=lambda n: hashlib.sha256(n.encode()).hexdigest())[0]
    r0 = by_name[first]
    want = [
        [x + r0["base"] if x >= E.ref0 else x for x in E.encode_toks(s["state"].split())]
        for s in r0["steps"]
    ]
    assert len(r0["steps"]) >= 3 and [p[:-3] for p, _ in cd.canary] == want[:3]
    # a T0 layout is the state alone; n = 0 gives every step to the anchor
    t0 = load_canary(path, T, False, 0)
    assert not t0.canary and len(t0.anchor) == n_all
    assert b0_prompt(T, [5, 6], False) == [5, 6]
    assert b0_prompt(T, [5], True) == [5, T.think, T.end_think]


def test_canary_metrics_match_a_direct_computation(tmp_path: Path) -> None:
    b = torch.zeros(E.vocab_size)
    b[E.stoi[":"]] = 2.0  # some mass on ':' at wrong positions
    b[list(E.effort_ids)] = 4.0  # banned in an action: never counted
    model = bias_logits(tiny_model(E.vocab_size, seed=5), b)
    cd = load_canary(_canary_file(tmp_path / "c.jsonl"), E, True, 1000)
    _, ban = phase_bans(E, E.vocab_size, True)
    got = canary_metrics(model, cd.canary, E, ban, "cpu", max_tokens=200)  # several chunks
    colon = E.stoi[":"]
    lps, tgts = [], []
    with torch.no_grad():
        for p, t in cd.canary:
            z = model(torch.tensor([p + t])[:, :-1]).float()[0, len(p) - 1 :]
            lps.append(torch.log_softmax(z.masked_fill(ban, -math.inf), -1))
            tgts.append(torch.tensor(t))
    lp, y = torch.cat(lps), torch.cat(tgts)
    P = lp.exp()
    notc = y != colon
    assert got["nll"] == pytest.approx(float(-lp.gather(1, y[:, None]).mean()), rel=1e-5)
    assert got["colon"] == pytest.approx(float(P[notc, colon].mean()), rel=1e-5)
    assert got["colon_logp"] == pytest.approx(float(lp[notc, colon].mean()), rel=1e-5)
    wrong = torch.stack(
        [P[y != v, v].mean() if (y != v).any() else torch.tensor(0.0) for v in range(E.vocab_size)]
    )
    assert got["wrong_max"] == pytest.approx(float(wrong.max()), rel=1e-5)
    assert got["wrong_tok"] == E.itos[int(wrong.argmax())]
    assert got["positions"] == len(y) and got["steps"] == len(cd.canary)
    assert all(float(P[:, k].max()) == 0.0 for k in E.effort_ids)


# --- the KL anchor --------------------------------------------------------------------------------


def _anchor(tmp_path: Path) -> list[Any]:
    return load_canary(_canary_file(tmp_path / "c.jsonl"), E, True, 0).anchor


def test_kl_anchor_is_zero_at_the_reference_and_finite(tmp_path: Path) -> None:
    seqs = _anchor(tmp_path)
    b = torch.zeros(E.vocab_size)
    b[list(E.effort_ids)] = 3.0  # banned: -inf in both, must not give NaN
    model = bias_logits(tiny_model(E.vocab_size, seed=6), b)
    ref = copy.deepcopy(model)
    _, ban = phase_bans(E, E.vocab_size, True)
    model.zero_grad(set_to_none=True)
    kl = anchor_kl(model, ref, seqs, 1.0, ban, "cpu", max_tokens=300, coef=0.5)
    assert abs(kl) < 1e-6
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert max(float(g.abs().max()) for g in grads) < 1e-5  # the KL's minimum


def test_kl_anchor_value_gradient_and_direction(tmp_path: Path) -> None:
    seqs = _anchor(tmp_path)
    _, ban = phase_bans(E, E.vocab_size, True)
    model, ref = tiny_model(E.vocab_size, seed=7), tiny_model(E.vocab_size, seed=8)
    want = 0.0
    n = 0
    with torch.no_grad():
        for p, t in seqs:
            x = torch.tensor([p + t])[:, :-1]
            zs = [m(x).float()[0, len(p) - 1 :] / 0.8 for m in (ref, model)]
            lr_, lp = (torch.log_softmax(z.masked_fill(ban, -math.inf), -1) for z in zs)
            d = torch.where(ban, 0.0, lr_ - lp)
            want += float((lr_.exp() * d).sum())
            n += len(t)
    model.zero_grad(set_to_none=True)
    kl = anchor_kl(model, ref, seqs, 0.8, ban, "cpu", max_tokens=300, coef=2.0)
    assert kl == pytest.approx(want / n, rel=1e-5) and kl > 0
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert all(torch.isfinite(g).all() for g in grads) and any(g.abs().sum() > 0 for g in grads)
    g2 = [p.grad.clone() for p in model.parameters() if p.grad is not None]
    model.zero_grad(set_to_none=True)
    anchor_kl(model, ref, seqs, 0.8, ban, "cpu", max_tokens=10_000, coef=1.0)  # one chunk
    g1 = [p.grad for p in model.parameters() if p.grad is not None]
    assert all(torch.allclose(x, 2 * y, rtol=1e-4, atol=1e-8) for x, y in zip(g2, g1, strict=True))
    with torch.no_grad():  # a small descent step lowers it
        for q in model.parameters():
            if q.grad is not None:
                q -= 1e-3 * q.grad
    after = anchor_kl(model, ref, seqs, 0.8, ban, "cpu", backward=False)
    assert after < kl


# --- end to end ---------------------------------------------------------------------------------


def _crc_judge(tok: Any, prompts: Any, res: list[dict[str, Any]], lean: Any = None) -> Any:
    """A verdict from the attempt's own sampled ids: the same in every process and every call."""
    for r in res:
        r["ok"] = zlib.crc32(b"".join(bytes(s.get("gen") or b"") for s in r["trace"])) % 2 == 0
    return res


class Go:
    """grpo.run on a tiny effort model (as test_grpo._tiny_run: budgets 0 and 16, 2 prompts x 4
    attempts, 2 minibatch steps, an <eos>-biased init) with a content-hash judge; `go(argv,
    out=...)` runs into iso / out with a shared init, targets, extra targets and canary file
    (`go.argv(argv, out)`: the full command line, for `grpo.main`)."""

    def __init__(self, iso: Path) -> None:
        self.iso = iso
        self.init = iso / "init.pt"
        save_ckpt(
            str(self.init), tiny_model(T.vocab_size, seed=2), T.mode, {"think_format": "think"}
        )
        rows = [{"prompt": pilot_rec(r)["prompt"]} for r in ("k12_37120", "k12_16472")]
        (iso / "targets.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        extra = {"prompt": pilot_rec("k12_22530")["prompt"]}
        (iso / "extra.jsonl").write_text(json.dumps(extra) + "\n")
        _canary_file(iso / "canary.jsonl")

    def argv(self, argv: list[str], out: str = "g") -> list[str]:
        base = ["--init", str(self.init), "--name", "g", "--out", str(self.iso / out)]
        base += ["--targets", str(self.iso / "targets.jsonl"), "--budgets", "0,16", "--effort"]
        base += ["--prompts", "2", "--group", "4", "--updates", "2", "--max_action", "8"]
        base += ["--max_steps", "3", "--batch", "8", "--device", "cpu", "--lr", "1e-3"]
        return base + argv

    def __call__(self, argv: list[str], out: str = "g") -> tuple[list[dict[str, Any]], Path]:
        return grpo.run(grpo.parser().parse_args(self.argv(argv, out))), self.iso / out


@pytest.fixture
def go(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Go:
    iso = isolate(tmp_path, monkeypatch)
    real_load = grpo.load_ckpt

    def eos_biased_load(path: Any, device: str = "cpu") -> Any:
        m, tok, x = real_load(path, device)
        b = torch.zeros(m.cfg["vocab"])
        b[tok.eos] = 3.0
        return bias_logits(m, b), tok, x

    monkeypatch.setattr(grpo, "judge", _crc_judge)
    monkeypatch.setattr(grpo, "load_ckpt", eos_biased_load)
    return Go(iso)


def _weights(path: Path) -> dict[str, torch.Tensor]:
    return torch.load(str(path), map_location="cpu")["state"]


def _same(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> bool:
    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


def _opt_tensors(st: dict[str, Any]) -> list[torch.Tensor]:
    return [v for s in st["opt"]["state"].values() for v in s.values()]


TIMING = ("secs", "secs_sample", "attempts_per_s")


def test_capped_alarm_under_overlong_fail(go: Go, capsys: pytest.CaptureFixture[str]) -> None:
    hist, _ = go(["--iters", "2", "--overlong", "fail"])
    assert all(h["overlong_frac"] == 0.0 for h in hist)  # the old metric: blind under 'fail'
    assert any(h["capped_frac"] > 0 for h in hist)  # random actions hit max_action
    for h in hist:
        n = sum(h["capped_kinds"].values())
        assert n == round(h["capped_frac"] * 8)
        over = [b for b, v in h["capped_by_budget"].items() if v > 0.05]
        assert any(m.startswith("capped attempts") for m in h["alarms"]) == bool(over)
    lines = [x for x in capsys.readouterr().out.splitlines() if x.startswith("it ")]
    assert len(lines) == 2 and all(" cap " in x for x in lines)


def test_resume_is_exact(go: Go) -> None:
    """2 iterations, then a resume for 2 more = 4 straight iterations: the same weights, AdamW
    state and metrics, bit for bit."""
    straight, s_out = go(["--iters", "4"], out="straight")
    go(["--iters", "2"], out="resumed")
    r_out = go.iso / "resumed"
    ck2 = r_out / "ckpts" / "g_it2.pt"
    st = load_train_state(ck2)
    assert st is not None and st["iter"] == 2 and st["loop"]["n_step"] == 4
    assert TRAIN_STATE not in torch.load(str(ck2))["extra"]  # never copied along with extra
    # a crash after it2 left a row for it3: the resume moves it aside
    with open(r_out / "metrics.jsonl", "a") as f:
        f.write(json.dumps({"iter": 3, "stale": True}) + "\n")
    resumed, _ = go(["--iters", "4", "--resume", str(ck2)], out="resumed")
    assert [h["iter"] for h in resumed] == [3, 4]
    assert _same(_weights(s_out / "ckpts" / "g_it4.pt"), _weights(r_out / "ckpts" / "g_it4.pt"))
    a, b = (load_train_state(o / "ckpts" / "g_it4.pt") for o in (s_out, r_out))
    assert a is not None and b is not None
    assert all(torch.equal(x, y) for x, y in zip(_opt_tensors(a), _opt_tensors(b), strict=True))
    assert a["curriculum"] == b["curriculum"] and a["rng"] == b["rng"] and a["loop"] == b["loop"]
    for x, y in zip(straight[2:], resumed, strict=True):
        assert {k: v for k, v in x.items() if k not in TIMING} == {
            k: v for k, v in y.items() if k not in TIMING
        }
    rows = [json.loads(x) for x in (r_out / "metrics.jsonl").read_text().splitlines()]
    assert [r["iter"] for r in rows] == [1, 2, 3, 4]
    assert json.loads((r_out / "metrics.pre_resume.jsonl").read_text())["stale"] is True
    assert (r_out / "args.pre_resume.json").exists() and (r_out / "args.json").exists()


def test_rollback_restores_the_parameters_and_halves_the_lr(
    go: Go, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Alarms from it3 on, --max_rollbacks 1: it3 restores the it2 rolling checkpoint (weights,
    AdamW state, step count) and halves the lr; it4 and it5 alarm without a rollback."""
    monkeypatch.setattr(grpo, "alarms", lambda row, prev: ["test"] if row["iter"] >= 3 else [])
    argv = ["--iters", "5", "--save_rolling", "1", "--save_every", "3", "--warmup", "1"]
    hist, out = go([*argv, "--rollback_on_alarm", "--max_rollbacks", "1"])
    ck = out / "ckpts"
    assert (ck / "g_it0_rolling.pt").exists()  # the start: something to roll back to
    assert hist[2]["rollback"] == {
        "to": str(ck / "g_it2_rolling.pt"),
        "to_iter": 2,
        "n": 1,
        "lr_scale": 0.5,
    }
    assert [h["rollback"] for h in hist[3:]] == [None, None]
    assert [(h["lr_scale"], h["rollbacks"]) for h in hist] == [(1.0, 0)] * 3 + [(0.5, 1)] * 2
    # the lr each stepping iteration used (one that skips every group takes no step)
    lrs = [h["lr"] for h in hist if h["steps"]]
    want = [1e-3 if h["iter"] <= 3 else 5e-4 for h in hist if h["steps"]]
    assert lrs == pytest.approx(want) and 5e-4 in want
    # it3's permanent checkpoint is written after the restore: it2's weights and AdamW state
    w2, w3 = _weights(ck / "g_it2_rolling.pt"), _weights(ck / "g_it3.pt")
    assert _same(w2, w3) and not _same(w2, _weights(ck / "g_it3_alarm.pt"))
    s2, s3 = load_train_state(ck / "g_it2_rolling.pt"), load_train_state(ck / "g_it3.pt")
    assert s2 is not None and s3 is not None and s3["loop"]["n_step"] == s2["loop"]["n_step"] == 4
    assert all(torch.equal(x, y) for x, y in zip(_opt_tensors(s2), _opt_tensors(s3), strict=True))
    assert s3["loop"]["lr_scale"] == 0.5 and s3["iter"] == 3
    assert torch.load(str(ck / "g_it3.pt"))["extra"]["grpo_rollback"]["to_iter"] == 2
    printed = capsys.readouterr().out
    assert "ROLLBACK it 3: restored" in printed and "ALARM it 4: no rollback" in printed


# --- the think ban restore check and stop (2026-10-11) -------------------------------------------


@pytest.mark.parametrize(
    ("x", "y", "back"),
    [
        (1.1e-3, 1.2e-3, False),  # the restore point is over the limit too: restoring cannot help
        (1.1e-3, 2.19e-3, False),
        (9e-4, 1.2e-3, True),  # under the limit: restoring helps
        (1e-3, 1.2e-3, True),  # at the limit is not over it (the alarm's own test)
        (1.1e-3, 2.2e-3, True),  # a spike: twice the restore point's mass
        (1.1e-3, 9e-3, True),
        (math.nan, 1.2e-3, True),  # an unusable score rolls back, as without the check
    ],
)
def test_ban_rollback_choice(x: float, y: float, back: bool) -> None:
    assert grpo.ban_rollback(x, y) is back


def test_restore_ban_mass_is_the_loss_ban_think_at_the_restore_weights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`ban_mass` is `policy_loss`'s `ban_think` (whatever the chunking; 0 without think tokens);
    `restore_ban_mass` gives it at a checkpoint's weights from a scratch copy of another model,
    which keeps its weights and gradients."""
    isolate(tmp_path, monkeypatch)
    b = torch.zeros(E.vocab_size)
    b[E.eos] = 2.0  # mass on the think ban
    model = bias_logits(tiny_model(E.vocab_size, seed=3), b)
    ids, bans = (E.think, E.end_think), phase_bans(E, E.vocab_size, True)
    st_, q, a_ = E.stoi["⊢"], E.stoi["Q"], E.stoi["exact"]
    items: list[grpo.Item] = [
        ([st_, effort_id(E, 16), E.think], [q, q, E.end_think, a_, E.eos], [True] * 5, 0.5),
        ([st_, q, effort_id(E, 64), E.think], [q, a_, q, E.end_think, a_, E.eos], [True] * 6, -0.5),
        ([st_, q, effort_id(E, 0), E.think, E.end_think], [a_, E.eos], [True] * 2, -0.5),
    ]
    assert len(grpo._chunks(items, 16)) > 1
    st = grpo.policy_loss(model, items, 0.8, "cpu", think_ids=ids, bans=bans)
    assert st.ban_think > grpo.ALARM_BAN
    for max_tokens in (16, 50_000):
        got = grpo.ban_mass(model, items, 0.8, "cpu", max_tokens, ids, bans)
        assert got == pytest.approx(st.ban_think, rel=1e-6)
    assert grpo.ban_mass(model, items[2:], 0.8, "cpu", 16, ids, bans) == 0.0  # budget 0 only
    path = tmp_path / "restore.pt"
    save_ckpt(str(path), model, E.mode)
    other = bias_logits(tiny_model(E.vocab_size, seed=4), b)
    other.train()
    grpo.policy_loss(other, items, 0.8, "cpu", think_ids=ids, bans=bans)  # gradients, as an update
    w0 = {k: v.clone() for k, v in other.state_dict().items()}
    g0 = [p.grad.clone() for p in other.parameters() if p.grad is not None]
    got = grpo.restore_ban_mass(other, str(path), items, 0.8, "cpu", 16, ids, bans)
    assert got == pytest.approx(st.ban_think, rel=1e-6)
    assert grpo.ban_mass(other, items, 0.8, "cpu", 16, ids, bans) != pytest.approx(got, rel=1e-3)
    assert _same(w0, other.state_dict()) and other.training
    g1 = [p.grad for p in other.parameters() if p.grad is not None]
    assert g0 and all(torch.equal(x, y) for x, y in zip(g0, g1, strict=True))


def _ban_alarms(
    monkeypatch: pytest.MonkeyPatch,
    ys: dict[int, float],
    other: Sequence[int] = (),
    restore: float | None = None,
) -> list[str]:
    """Iteration i's think ban mass is ys[i] (1e-5 for the others): the alarms read it and the row
    keeps it. The alarms are the think ban alarm, plus 'test' at the iterations in `other`. With
    `restore`, the restore check scores every restore point at `restore` -> the restore points it
    scored, one per check."""

    def fake_alarms(row: dict[str, Any], prev: Any) -> list[str]:
        row["ban_think"] = ys.get(row["iter"], 1e-5)
        ban = grpo.ban_alarm(row)
        return ([ban] if ban else []) + (["test"] if row["iter"] in other else [])

    monkeypatch.setattr(grpo, "alarms", fake_alarms)
    scored: list[str] = []
    if restore is not None:

        def fake_restore(model: Any, path: str, *args: Any) -> float:
            scored.append(path)
            return restore

        monkeypatch.setattr(grpo, "restore_ban_mass", fake_restore)
    return scored


ROLL = ["--rollback_on_alarm", "--save_rolling", "1", "--save_every", "3", "--warmup", "1"]
CHECK = [*ROLL, "--ban_restore_check"]


def test_ban_alarm_without_the_check_rolls_back(go: Go, monkeypatch: pytest.MonkeyPatch) -> None:
    """The options off (the default): a think ban alarm rolls back as before, nothing is scored,
    and the row records the options."""
    scored = _ban_alarms(monkeypatch, {3: 1.2e-3}, restore=1.1e-3)
    hist, out = go(["--iters", "4", *ROLL])
    assert scored == [] and not any("ban_check" in h or "stop" in h for h in hist)
    assert hist[2]["rollback"] == {
        "to": str(out / "ckpts" / "g_it2_rolling.pt"),
        "to_iter": 2,
        "n": 1,
        "lr_scale": 0.5,
    }
    assert [(h["lr_scale"], h["rollbacks"]) for h in hist] == [(1.0, 0)] * 3 + [(0.5, 1)]
    assert (out / "ckpts" / "g_it3_alarm.pt").exists()
    cfg = hist[0]["alarm_cfg"]
    assert (cfg["ban_restore_check"], cfg["ban_stop"]) == (False, 0.0)


def test_ban_check_skips_the_rollback_when_the_restore_point_is_over_the_limit(
    go: Go, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """it3 and it4 alarm on the think ban alone, their restore points are over the limit as well
    and they are under twice theirs: no rollback, the alarm logged, no alarm checkpoint, the lr
    kept. it3's permanent checkpoint is good, so it is it4's restore point; it4 rolls."""
    scored = _ban_alarms(monkeypatch, {3: 1.2e-3, 4: 1.3e-3}, restore=1.1e-3)
    hist, out = go(["--iters", "5", *CHECK])
    ck = out / "ckpts"
    assert scored == [str(ck / "g_it2_rolling.pt"), str(ck / "g_it3.pt")]
    assert hist[2]["ban_check"] == {
        "to": str(ck / "g_it2_rolling.pt"),
        "to_iter": 2,
        "restore_mass": 1.1e-3,
        "mass": 1.2e-3,
        "rollback": False,
    }
    assert hist[3]["ban_check"]["to_iter"] == 3 and hist[3]["ban_check"]["rollback"] is False
    assert [h["rollback"] for h in hist[2:4]] == [None, None]
    assert [h["alarms"] for h in hist[2:4]] == [[grpo.ban_alarm(h)] for h in hist[2:4]]
    assert all((h["lr_scale"], h["rollbacks"]) == (1.0, 0) for h in hist)
    assert hist[0]["alarm_cfg"]["ban_restore_check"] is True
    assert not list(ck.glob("*_alarm.pt"))
    names = sorted(p.name for p in ck.glob("g_it*.pt"))
    assert names == sorted(
        ["g_it0.pt", "g_it0_rolling.pt", "g_it1_rolling.pt", "g_it2_rolling.pt", "g_it3.pt"]
        + ["g_it4_rolling.pt", "g_it5.pt"]
    )
    # nothing was restored: it3 trained on from it2 (weights, AdamW steps)
    assert hist[2]["steps"] > 0 and not _same(
        _weights(ck / "g_it2_rolling.pt"), _weights(ck / "g_it3.pt")
    )
    s2, s3 = load_train_state(ck / "g_it2_rolling.pt"), load_train_state(ck / "g_it3.pt")
    assert s2 is not None and s3 is not None
    assert s3["loop"]["n_step"] == s2["loop"]["n_step"] + hist[2]["steps"]
    printed = capsys.readouterr().out
    assert (
        "ALARM it 3: no rollback: the restore point (it 2) has think banned mass 0.0011 > 0.001 on "
        "this iteration's think tokens (this iteration 0.0012 < 2 x 0.0011)"
    ) in printed.splitlines()
    assert "ALARM it 3: think banned mass 0.0012 > 0.001" in printed
    assert "ROLLBACK" not in printed and "--max_rollbacks" not in printed


@pytest.mark.parametrize(("y", "x"), [(1.2e-3, 5e-4), (2.5e-3, 1.1e-3)])  # under; a spike
def test_ban_check_rolls_back_from_under_the_limit_or_a_spike(
    go: Go, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], y: float, x: float
) -> None:
    scored = _ban_alarms(monkeypatch, {3: y}, restore=x)
    hist, out = go(["--iters", "4", *CHECK])
    ck = out / "ckpts"
    assert scored == [str(ck / "g_it2_rolling.pt")]
    to = {"to": str(ck / "g_it2_rolling.pt"), "to_iter": 2}
    assert hist[2]["ban_check"] == {**to, "restore_mass": x, "mass": y, "rollback": True}
    assert hist[2]["rollback"] == {**to, "n": 1, "lr_scale": 0.5}
    assert [(h["lr_scale"], h["rollbacks"]) for h in hist] == [(1.0, 0)] * 3 + [(0.5, 1)]
    assert (ck / "g_it3_alarm.pt").exists()
    assert _same(_weights(ck / "g_it2_rolling.pt"), _weights(ck / "g_it3.pt"))  # restored
    printed = capsys.readouterr().out
    assert (
        f"ALARM it 3: rollback: the restore point (it 2) has think banned mass {x:.3g} on this "
        f"iteration's think tokens (this iteration {y:.3g})"
    ) in printed.splitlines()
    assert "ROLLBACK it 3: restored" in printed


@pytest.mark.parametrize("ys", [{3: 1.2e-3}, {}])
def test_another_alarm_rolls_back_unchecked(
    go: Go, monkeypatch: pytest.MonkeyPatch, ys: dict[int, float]
) -> None:
    """A think ban alarm with another alarm, or another alarm alone: rolled back, not scored."""
    scored = _ban_alarms(monkeypatch, ys, other=[3], restore=1.1e-3)
    hist, _ = go(["--iters", "4", *CHECK])
    assert scored == [] and not any("ban_check" in h for h in hist)
    assert hist[2]["rollback"]["to_iter"] == 2 and hist[3]["lr_scale"] == 0.5


def test_ban_check_after_max_rollbacks(
    go: Go, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """--max_rollbacks reached: no check, the old line, the alarm checkpoint."""
    scored = _ban_alarms(monkeypatch, {3: 1.2e-3, 4: 1.2e-3}, restore=5e-4)
    hist, out = go(["--iters", "4", *CHECK, "--max_rollbacks", "1"])
    assert len(scored) == 1 and hist[2]["ban_check"]["rollback"] is True
    assert hist[3]["rollback"] is None and "ban_check" not in hist[3]
    assert (out / "ckpts" / "g_it4_alarm.pt").exists()
    printed = capsys.readouterr().out
    assert "ALARM it 4: no rollback (--max_rollbacks 1 reached)" in printed.splitlines()
    assert "ALARM it 4: no rollback:" not in printed


def test_ban_check_scores_the_iteration_under_the_restore_point(
    go: Go, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The real check on the <eos>-biased model, whose think ban mass is far over the limit from
    the start: it1 is let through. Its restore mass is `policy_loss`'s `ban_think` of it1's
    trained items (the dump) under the start checkpoint's weights; it2 checks against it1."""
    monkeypatch.setattr(grpo, "alarms", lambda row, prev: [m for m in [grpo.ban_alarm(row)] if m])
    dumps = go.iso / "dumps"
    hist, out = go(["--iters", "2", *CHECK, "--dump_batches", str(dumps)])
    ck = out / "ckpts"
    bc = hist[0]["ban_check"]
    assert (bc["to"], bc["to_iter"]) == (str(ck / "g_it0_rolling.pt"), 0)
    assert bc["mass"] == hist[0]["ban_think"] > grpo.ALARM_BAN
    model, tok, _ = grpo.load_ckpt(ck / "g_it0_rolling.pt")  # the fixture's <eos> bias too
    items, _ = unpack_items(load_batch(dumps / "g_it1_batch.pt"))
    bans = phase_bans(tok, model.cfg["vocab"], True)
    st = grpo.policy_loss(model, items, 1.0, "cpu", think_ids=(tok.think, tok.end_think), bans=bans)
    assert bc["restore_mass"] == pytest.approx(st.ban_think, rel=1e-5)
    assert bc["rollback"] is False and hist[0]["rollback"] is None
    assert hist[1]["ban_check"]["to"] == str(ck / "g_it1_rolling.pt")
    assert not list(ck.glob("*_alarm.pt"))
    assert "ALARM it 1: no rollback: the restore point (it 0)" in capsys.readouterr().out


def test_ban_stop_ends_the_run_with_exit_code_3(
    go: Go, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """it2's think ban mass is over --ban_stop: its row, its usual lines and its alarm checkpoint,
    then the STOP line and exit code 3; no rollback, no check, nothing after. Under the stop,
    the run goes on (exit code 0)."""
    scored = _ban_alarms(monkeypatch, {2: 6e-3}, restore=1.1e-3)
    assert grpo.main(go.argv(["--iters", "4", *CHECK, "--ban_stop", "0.005"])) == 3
    out, ck = go.iso / "g", go.iso / "g" / "ckpts"
    rows = [json.loads(x) for x in (out / "metrics.jsonl").read_text().splitlines()]
    assert [r["iter"] for r in rows] == [1, 2] and scored == []
    assert rows[1]["stop"] == {"ban_think": 6e-3, "ban_stop": 0.005}
    assert rows[1]["rollback"] is None and "stop" not in rows[0]
    assert (
        rows[1]["alarms"] == [grpo.ban_alarm(rows[1])] and rows[1]["alarm_cfg"]["ban_stop"] == 0.005
    )
    assert sorted(p.name for p in ck.glob("g_it*.pt")) == sorted(
        ["g_it0.pt", "g_it0_rolling.pt", "g_it1_rolling.pt", "g_it2_alarm.pt"]
    )
    st = load_train_state(ck / "g_it2_alarm.pt")
    assert st is not None and st["iter"] == 2
    printed = capsys.readouterr().out.splitlines()
    assert printed[-1] == "STOP it 2: think banned mass 0.006 > 0.005: ending the run"
    assert any(x.startswith("it 2: ") for x in printed) and "ALARM it 2: think banned mass" in (
        "\n".join(printed)
    )
    assert not any("ROLLBACK" in x or "no rollback" in x or x.startswith("it 3:") for x in printed)
    assert grpo.main(go.argv(["--iters", "3", "--ban_stop", "0.01"], out="under")) == 0
    assert len((go.iso / "under" / "metrics.jsonl").read_text().splitlines()) == 3


def test_batch_dump_round_trip(tmp_path: Path) -> None:
    items: list[grpo.Item] = [
        ([3, 4, 5], [6, 7, 8], [True, False, True], 2 / 3),
        ([9], [10], [True], -1 / 3),
        ([11, 12], [13, 14, 15, 16], [True] * 4, -1 / 3),
    ]
    old = [torch.tensor([-0.5, -1.25]), torch.tensor([-2.0]), torch.randn(4)]
    model = tiny_model(E.vocab_size, seed=3)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    assert compact_adam(opt, model) == {"step": 0, "names": [], "exp_avg": [], "exp_avg_sq": []}
    model(torch.tensor([[1, 2, 3]])).sum().backward()
    opt.step()
    adam = compact_adam(opt, model)
    attempts = [
        {"group": 0, "status": "done", "ok": True, "reward": 1.0, "masked": False},
        {"group": 0, "status": "fail:S:x", "ok": False, "reward": 0.0, "masked": False},
        {"group": 0, "status": "fail:truncated", "ok": False, "reward": 0.0, "masked": True},
        {"group": 1, "status": "done", "ok": True, "reward": 1.0, "masked": False},
    ]
    groups = [
        {"budget": 16, "target": 4, "source": 0, "skipped": False},
        {"budget": 0, "target": 1, "source": 1, "skipped": True},
    ]
    p = dump_batch(
        tmp_path / "d" / "b.pt",
        it=7,
        items=items,
        old=old,
        item_attempt=[0, 1, 1],
        order=[2, 0, 1],
        updates=2,
        attempts=attempts,
        groups=groups,
        steps=[{"lr": 1e-3, "grad_norm": 0.5, "kl_anchor": 0.0}],
        adam=adam,
        settings={"temperature": 1.0},
        vocab=E.vocab_size,
    )
    d = load_batch(p)  # weights_only: plain types and tensors
    got, got_old = unpack_items(d)
    assert got == items  # advantages in float64: exact
    assert all(torch.equal(a, b.float()) for a, b in zip(got_old, old, strict=True))
    assert d["iter"] == 7 and d["minibatch"].tolist() == [1, 0, 0]  # order[u::2]: [2, 1], [0]
    assert d["budget"].tolist() == [16, 16, 16] and d["prompt"].dtype == torch.uint8
    assert d["attempt_status"] == [x["status"] for x in attempts]
    assert d["group_skipped"].tolist() == [False, True] and d["group_source"].tolist() == [0, 1]
    names = [n for n, _ in model.named_parameters()]
    assert d["adam"]["names"] == names and d["adam"]["step"] == 1
    for n, m1 in zip(names, d["adam"]["exp_avg"], strict=True):
        full = opt.state[dict(model.named_parameters())[n]]["exp_avg"]
        assert m1.dtype == torch.bfloat16 and torch.allclose(m1.float(), full, rtol=1e-2, atol=1e-8)


def test_dumps_hold_the_trained_batch(go: Go) -> None:
    """--dump_batches: the dumped items are the iteration's (count, budgets), their old log-probs
    are the sampling policy's, and iteration 2's AdamW state is the one iteration 1 left."""
    iso = go.iso
    hist, out = go(["--iters", "2", "--save_every", "1", "--dump_batches", str(iso / "dumps")])
    d1, d2 = (load_batch(iso / "dumps" / f"g_it{i}_batch.pt") for i in (1, 2))
    for h, d in zip(hist, (d1, d2), strict=True):
        items, old = unpack_items(d)
        assert len(items) == h["items"] and len(d["attempt_status"]) == 8
        assert d["group_budget"].tolist() and set(d["budget"].tolist()) <= {0, 16}
        ga = d["attempt_group"][d["item_attempt"].long()]
        assert torch.equal(d["budget"], d["group_budget"][ga.long()])
        assert sum(
            int(m.sum()) for m in torch.split(d["mask"], (d["gen_off"].diff()).tolist())
        ) == (sum(len(o) for o in old))
        assert len(d["step_lr"]) == h["steps"] == 2
    # iteration 1 sampled with the effort-expanded init: recompute its old log-probs
    model, tok, _ = grpo.load_ckpt(out / "ckpts" / "g_it0.pt")
    bans = phase_bans(tok, model.cfg["vocab"], True)
    items, old = unpack_items(d1)
    want = grpo.old_logps(model, items, 1.0, "cpu", 50_000, (tok.think, tok.end_think), bans)
    assert all(torch.allclose(a, b, atol=1e-5) for a, b in zip(old, want, strict=True))
    assert d1["adam"]["step"] == 0 and d2["adam"]["step"] == 2
    s1 = load_train_state(out / "ckpts" / "g_it1.pt")
    assert s1 is not None
    full = [s["exp_avg_sq"] for s in s1["opt"]["state"].values()]
    for a, b in zip(d2["adam"]["exp_avg_sq"], full, strict=True):
        assert torch.allclose(a.float(), b, rtol=1e-2, atol=1e-12)


def test_canary_and_kl_anchor_end_to_end(go: Go, capsys: pytest.CaptureFixture[str]) -> None:
    iso = go.iso
    ref = iso / "g" / "ckpts" / "g_it0.pt"  # the effort expansion of the init, written first
    argv = ["--iters", "2", "--canary", str(iso / "canary.jsonl"), "--canary_n", "4"]
    hist, out = go([*argv, "--kl_anchor_coef", "0.5", "--kl_anchor_ref", str(ref)])
    base = json.loads((out / "canary_base.json").read_text())
    assert base["iter"] == 0 and base["steps"] == 4
    for h in hist:
        c = h["canary"]
        assert c["steps"] == 4 and c["base"] == {k: v for k, v in base.items() if k in c["base"]}
        assert c["nll"] > 0 and 0 <= c["colon"] <= c["wrong_max"] + 1e-12  # ':' is one token
        assert math.isfinite(h["kl_anchor"]) and h["kl_anchor"] >= 0
        assert h["kl_anchor_coef"] == 0.5 and h["grad_norm_policy"] > 0
    assert hist[1]["kl_anchor"] > 0  # the policy has moved off pi_ref
    printed = capsys.readouterr().out
    assert "canary it 0:" in printed and "canary it 2:" in printed and "kl anchor" in printed


def test_long_proofs_mixed_into_the_pool(go: Go) -> None:
    iso = go.iso
    argv = ["--iters", "2", "--targets_extra", str(iso / "extra.jsonl")]
    argv += ["--targets_extra_frac", "0.5", "--dump_batches", str(iso / "dumps")]
    hist, _ = go(argv)
    for h in hist:
        assert h["extra_groups"] == 1 and 0.0 <= h["extra_pass1"] <= 1.0
        assert h["curriculum_extra"]["pairs"] == 2 and h["curriculum"]["pairs"] == 4
        d = load_batch(iso / "dumps" / f"g_it{h['iter']}_batch.pt")
        assert d["group_source"].tolist() == [0, 1] and d["group_target"].tolist()[1] == 0
    assert hist[1]["curriculum_extra"]["seen"] >= 1
