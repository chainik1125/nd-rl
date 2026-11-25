# ruff: noqa: E501
"""CPU tests of the Stage 1 EDL code (olmo_edl.py, analyze_edl.py, build_pool.py) on a tiny random Olmo3 model.

Run: /scratch/cowifornia/venvs/olmo-edl/bin/python -m pytest -q test_olmo_edl.py   (torch 2.8.0, transformers 4.57.1,
peft 0.17.1, the pod's versions). A few seconds.
"""

from __future__ import annotations

import base64
import csv
import gzip
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("peft")
transformers = pytest.importorskip("transformers")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
for p in (str(HERE), str(ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

import analyze_edl as an  # noqa: E402
import build_pool as bp  # noqa: E402
import olmo_edl as oe  # noqa: E402
from olmo_template import TEMPLATE  # noqa: E402

V = 64


class FakeTok:
    """apply_chat_template stand-in: deterministic ids from the user turn, with a pad id (0) the encoder must drop."""

    pad_token_id = 0

    def apply_chat_template(
        self, msgs, add_generation_prompt, tokenize, return_dict, chat_template
    ):
        assert chat_template == TEMPLATE and add_generation_prompt and tokenize and not return_dict
        return [2, 3] + [6 + (ord(ch) % 50) for ch in msgs[-1]["content"]] + [0, 5]


def tiny_model(seed: int = 0, sliding: int = 8, max_pos: int = 512):
    from transformers import Olmo3Config, Olmo3ForCausalLM

    torch.manual_seed(seed)
    cfg = Olmo3Config(
        vocab_size=V,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=max_pos,
        sliding_window=sliding,
        layer_types=["sliding_attention", "full_attention"],
        pad_token_id=0,
        eos_token_id=1,
        bos_token_id=None,
    )
    cfg._attn_implementation = "sdpa"
    m = Olmo3ForCausalLM(cfg)
    m.config.use_cache = False
    return m


def make_examples(
    rng: np.random.Generator, k: int, prefix: str, lens=(12, 30), equal_len: int | None = None
) -> list:
    tok = FakeTok()
    out = []
    for j in range(k):
        T = equal_len or int(rng.integers(*lens))
        e = int(rng.integers(2, T - 2))
        resp = rng.integers(6, V, size=T).tolist()
        resp[-1] = 1
        out.append(
            oe.Example(
                f"{prefix}{j}", oe.prompt_ids(tok, f"prompt {prefix} {j}"), resp, e, "genuine"
            )
        )
    return out


def make_sets(seed=0, n=6, m=4, r=3, equal_len=None):
    rng = np.random.default_rng(seed)
    return {
        "train": make_examples(rng, n, "tr", equal_len=equal_len),
        "test": make_examples(rng, m, "te", equal_len=equal_len),
        "reach": make_examples(rng, r, "re", equal_len=equal_len),
    }


POOL_INFO = {"pool": "synthetic", "manifest_sha256": "x" * 64, "max_tok_filter": None}


def run(model, sets, out, **kw):
    args = dict(
        ckpt="tiny",
        ident={"model": "tiny"},
        lrs=[1e-2],
        orders=1,
        seed=0,
        evals=[0, 2, 4, 6],
        reach_evals=[0, 6],
        batch_tokens=64,
        save="none",
        heartbeat_secs=3600,
    )
    args.update(kw)
    return oe.run_edl(model, FakeTok(), sets, POOL_INFO, out, **args)


# ------------------------------------------------------------------------------------------ arithmetic


def test_edl_arithmetic_matches_ndbench():
    from code.ndbench.metrics import edl as nd_edl

    rng = np.random.default_rng(1)
    online = rng.gamma(2.0, 50.0, size=64)
    for n, tm in ((8, 70.0), (16, 65.5), (64, 61.25)):
        assert an.edl_sum(online, tm, n) == pytest.approx(nd_edl(list(online[:n]), tm), rel=1e-12)
    # length-normalised form = rate difference; times online tokens it is the nats form
    tok = rng.integers(100, 900, size=64).astype(float)
    test_n, test_t = 5000.0, 4000.0
    v = an.edl_norm_per_tok(online, tok, test_n, test_t, 32)
    assert v == pytest.approx(online[:32].sum() / tok[:32].sum() - 1.25)


def synth_rows(seed=0, lrs=(1e-4, 3e-4), orders=2, n=8, m=5):
    """Row dicts with known numbers (no model), for the analysis."""
    rng = np.random.default_rng(seed)
    tth, tan = rng.integers(50, 90, size=m), rng.integers(5, 15, size=m)
    rows = []
    for lr in lrs:
        for o in range(orders):
            nth, nan_ = rng.integers(40, 100, size=n), rng.integers(4, 20, size=n)
            on = {
                "pids": [f"p{i}" for i in range(n)],
                "think_nats": list(nth * 0.5),
                "answer_nats": list(nan_ * 0.8),
                "n_think": list(map(int, nth)),
                "n_answer": list(map(int, nan_)),
            }
            ev = {}
            for e in (0, 4, n):
                f = 1.0 - 0.02 * e * (1 + lr * 1e3)
                ev[str(e)] = {
                    "test": {"think_nats": list(tth * 0.5 * f), "answer_nats": list(tan * 0.8 * f)}
                }
            rows.append(
                {
                    "schema": oe.SCHEMA,
                    "lr": lr,
                    "order": o,
                    "status": "ok",
                    "n": n,
                    "online": on,
                    "evals": ev,
                    "test": {
                        "pids": [f"t{j}" for j in range(m)],
                        "n_think": list(map(int, tth)),
                        "n_answer": list(map(int, tan)),
                    },
                }
            )
    return rows


def test_analysis_edl_star_rules():
    from code.ndbench.metrics import edl as nd_edl

    rows = synth_rows()
    c = an.Ckpt("dpo", None, rows)
    res = an.analyze_ckpt(c, np.random.default_rng(0), 200)
    # code-length lr by hand
    cl = {
        lr: np.mean(
            [
                sum(r["online"]["think_nats"]) + sum(r["online"]["answer_nats"])
                for r in rows
                if r["lr"] == lr
            ]
        )
        for lr in (1e-4, 3e-4)
    }
    lr = min(cl, key=cl.get)
    by_order = []
    for r in rows:
        if r["lr"] == lr:
            onl = list(np.add(r["online"]["think_nats"], r["online"]["answer_nats"]))
            tm = float(
                np.mean(
                    np.add(
                        r["evals"]["8"]["test"]["think_nats"],
                        r["evals"]["8"]["test"]["answer_nats"],
                    )
                )
            )
            by_order.append(nd_edl(onl, tm))
    want = float(np.mean(by_order)) + math.log(2)
    got = res["EDL_star"]["sum"]["mdl"]["total"]
    assert got["lr"] == lr and got["nats"] == pytest.approx(want)
    assert got["ci"][0] <= got["nats"] <= got["ci"][1]
    # decomposition at the total's lr: think + answer = EDL*(total) - ln k
    dec = res["EDL_at_total_lr"]["sum"]
    assert dec["think"]["nats"] + dec["answer"]["nats"] == pytest.approx(got["nats"] - math.log(2))
    # the answer segment's own EDL*: its own code-length lr
    cla = {
        x: np.mean([sum(r["online"]["answer_nats"]) for r in rows if r["lr"] == x])
        for x in (1e-4, 3e-4)
    }
    assert res["EDL_star"]["sum"]["mdl"]["answer"]["lr"] == min(cla, key=cla.get)
    # min rule: min over lr of the mean EDL(n) + ln k
    m = min(res["by_lr"][f"{x:g}"]["EDL64"]["sum"]["total"] for x in (1e-4, 3e-4))
    assert res["EDL_star"]["sum"]["min"]["total"]["nats"] == pytest.approx(m + math.log(2))
    for seg in ("total", "think", "answer"):
        cv = res["curve"]["norm"][seg]
        assert cv["n"] == [4, 8] and len(cv["ci"]) == 2
    # per token over n' equals the hand formula for order 0 / order 1 averaged
    lr_t = res["curve"]["sum"]["total"]["lr"]
    vals = []
    for r in rows:
        if r["lr"] == lr_t:
            onl = np.add(r["online"]["think_nats"], r["online"]["answer_nats"])
            tok = np.add(r["online"]["n_think"], r["online"]["n_answer"])
            tm = float(
                np.mean(
                    np.add(
                        r["evals"]["4"]["test"]["think_nats"],
                        r["evals"]["4"]["test"]["answer_nats"],
                    )
                )
            )
            vals.append(nd_edl(list(onl[:4]), tm) / tok[:4].sum())
    assert res["curve"]["sum"]["total"]["per_tok"][0] == pytest.approx(np.mean(vals))


# ------------------------------------------------------------------------------------------ model


def test_token_nll_matches_hf_loss_and_segments():
    model = tiny_model()
    ex = make_sets()["train"][0]
    with torch.no_grad():
        nll = oe.token_nll(model, [(ex.prompt, ex.resp)])[0]
        x = torch.tensor([ex.prompt + ex.resp])
        lab = torch.tensor([[-100] * len(ex.prompt) + ex.resp])
        hf = float(model(input_ids=x, labels=lab).loss)  # Dan's lora.py nll()
    assert nll.shape == (len(ex.resp),)
    assert float(nll.mean()) == pytest.approx(hf, rel=1e-5)
    th, ans = oe.seg_sums(nll, ex.think_end)
    assert th == pytest.approx(float(nll[: ex.think_end].sum()), rel=1e-6)
    assert th + ans == pytest.approx(float(nll.sum()), rel=1e-6)


def test_batched_scoring_matches_batch1():
    model = tiny_model(sliding=8)  # responses (12-30 tokens) run past the sliding window
    exs = make_sets(n=7)["train"]
    a = oe.score(model, exs, batch_tokens=0)
    b = oe.score(model, exs, batch_tokens=400)
    np.testing.assert_allclose(a["think_nats"], b["think_nats"], rtol=1e-4)
    np.testing.assert_allclose(a["answer_nats"], b["answer_nats"], rtol=1e-4)


def test_adapter_reset_restores_step0_exactly():
    model = oe.wrap_lora(tiny_model(), grad_ckpt=True)
    exs = make_sets()["train"]
    oe.reset_lora(model, 1)
    s0 = oe.score(model, exs)
    params = oe.trainable(model)
    opt = torch.optim.AdamW(params, lr=5e-2)
    for ex in exs[:3]:
        oe.train_step(model, opt, params, ex, grad_clip=1.0)
    s1 = oe.score(model, exs)
    assert s1["think_nats"] != s0["think_nats"]  # training changed the model
    oe.reset_lora(model, 2)  # another A init: B = 0 makes it irrelevant
    s2 = oe.score(model, exs)
    assert s2["think_nats"] == s0["think_nats"] and s2["answer_nats"] == s0["answer_nats"]


def test_run_rows_reset_check_and_fields(tmp_path):
    model = oe.wrap_lora(tiny_model(), grad_ckpt=True)
    sets = make_sets()
    assert run(model, sets, tmp_path, lrs=[1e-2, 3e-2]) == 0
    rows = oe.read_rows(tmp_path / "edl_rows.jsonl")
    assert len(rows) == 2 and (tmp_path / "DONE").exists() and (tmp_path / "step0.json").exists()
    for r in rows:
        assert r["status"] == "ok" and r["reset_check"]["abs_diff"] == 0.0
        assert (
            sorted(r["evals"], key=int) == ["0", "2", "4", "6"]
            and "reach" in r["evals"]["6"]
            and "reach" not in r["evals"]["4"]
        )
        assert len(r["online"]["think_nats"]) == 6 and r["online"]["pids"] == [
            sets["train"][k].pid for k in r["perm"]
        ]
        assert r["test"]["pids"] == [e.pid for e in sets["test"]]
    hb = json.loads((tmp_path / "heartbeat.json").read_text())
    assert hb["phase"] == "done"
    s = an.analyze(tmp_path.parent / "nonexistent", B=10)  # no runs: empty, no crash
    assert s["checkpoints"] == {}


def test_null_control_edl_zero(tmp_path):
    """A model whose loss cannot change (zero LM head: uniform logits, zero gradients) has EDL = 0 in the
    length-normalised form, and in the per-example form when train and test responses have equal lengths."""
    for equal_len, sub in ((None, "var"), (20, "eq")):
        model = tiny_model()
        with torch.no_grad():
            model.lm_head.weight.zero_()
        model = oe.wrap_lora(model, grad_ckpt=False)
        sets = make_sets(equal_len=equal_len)
        d = tmp_path / sub / "dpo" / "edl"
        assert run(model, sets, d, ckpt="dpo", lrs=[1e-2], orders=2) == 0
        res = an.analyze(tmp_path / sub, B=50)["checkpoints"]["dpo"]
        norm = res["by_lr"]["0.01"]["EDL64"]["norm"]["total"]
        assert abs(norm) < 1e-9
        for seg in ("total", "think", "answer"):
            assert res["EDL_star"]["norm"]["mdl"][seg]["nats"] == pytest.approx(
                0.0, abs=1e-9
            )  # ln(1 lr) = 0
        r = oe.read_rows(d / "edl_rows.jsonl")[0]
        c = math.log(V)
        assert r["online"]["think_nats"][0] == pytest.approx(
            c * r["online"]["n_think"][0], rel=1e-6
        )
        s = res["by_lr"]["0.01"]["EDL64"]["sum"]["total"]
        if equal_len:
            assert abs(s) < 1e-6
        else:  # the per-example form charges the length difference
            n_tr = sum(e.n_tok for e in sets["train"])
            n_te = np.mean([e.n_tok for e in sets["test"]])
            assert s == pytest.approx(c * (n_tr - len(sets["train"]) * n_te), rel=1e-5)


def test_resume_skips_finished_rows(tmp_path, monkeypatch):
    model = oe.wrap_lora(tiny_model(), grad_ckpt=True)
    sets = make_sets()
    assert run(model, sets, tmp_path, lrs=[1e-2]) == 0
    first = (tmp_path / "edl_rows.jsonl").read_text()
    calls = {"run": 0, "step0": 0}
    real_run, real_s0 = oe.run_one, oe.compute_step0

    def count_run(*a, **k):
        calls["run"] += 1
        return real_run(*a, **k)

    def count_s0(*a, **k):
        calls["step0"] += 1
        return real_s0(*a, **k)

    monkeypatch.setattr(oe, "run_one", count_run)
    monkeypatch.setattr(oe, "compute_step0", count_s0)
    assert run(model, sets, tmp_path, lrs=[1e-2, 3e-2]) == 0
    assert calls == {"run": 1, "step0": 0}
    assert (tmp_path / "edl_rows.jsonl").read_text().startswith(first)
    assert run(model, sets, tmp_path, lrs=[1e-2, 3e-2]) == 0 and calls["run"] == 1
    assert len(oe.read_rows(tmp_path / "edl_rows.jsonl")) == 2
    with pytest.raises(SystemExit):  # rows of another checkpoint in the same out dir
        run(model, sets, tmp_path, ckpt="other", lrs=[1e-2, 3e-2])


def test_nonfinite_row_is_recorded_and_next_row_runs(tmp_path, monkeypatch):
    model = oe.wrap_lora(tiny_model(), grad_ckpt=True)
    sets = make_sets()
    real = oe.train_step
    seen = {"n": 0}

    def flaky(model_, opt, params, ex, **k):
        if opt.param_groups[0]["lr"] == 3e-2:
            seen["n"] += 1
            if seen["n"] == 2:
                raise oe.NonFinite("synthetic NaN")
        return real(model_, opt, params, ex, **k)

    monkeypatch.setattr(oe, "train_step", flaky)
    assert run(model, sets, tmp_path, lrs=[3e-2, 1e-2]) == 0
    rows = {r["lr"]: r for r in oe.read_rows(tmp_path / "edl_rows.jsonl")}
    assert rows[3e-2]["status"] == "nonfinite" and len(rows[3e-2]["online"]["think_nats"]) == 2
    assert rows[1e-2]["status"] == "ok" and len(rows[1e-2]["online"]["think_nats"]) == 6


# ------------------------------------------------------------------------------------------ pool


def _ids(think: int, ans: int, mark: bool = True) -> list[int]:
    return [10] * think + ([524, 27963, 1363] if mark else []) + [20] * ans + [bp.EOS]


def test_pool_builder_rules(tmp_path):
    dan = tmp_path / "dan"
    (dan / "olmo_ifomega").mkdir(parents=True)
    (dan / "olmo_ifcurve").mkdir()
    for d in ("olmo_ladder", "olmo_ifladder", "olmo_transplant", "artifacts/olmo_ifcurve/hand"):
        (dan / d).mkdir(parents=True, exist_ok=True)
    arm = [f"ifbench_{k}" for k in range(1, 13)] + ["ifbench_100"]
    reach = ["ifbench_100", "ifeval_200"]
    probs = [
        {
            "pid": p,
            "task": p.split("_")[0],
            "key": 0,
            "prompt": f"Do {p}.",
            "instruction_id_list": ["x:y"],
            "kwargs": [{}],
        }
        for p in arm + ["ifeval_200"]
    ]
    (dan / "olmo_ifomega/problems.jsonl").write_text("".join(json.dumps(p) + "\n" for p in probs))
    (dan / "olmo_ifcurve/pids_armA.txt").write_text("\n".join(arm) + "\n")
    (dan / "olmo_ifcurve/pids_reach.txt").write_text("\n".join(reach) + "\n")
    job, run_ = "F_s2300_ifb", "olmo_ifomega"
    draws = []  # (pid, chunk, i, ids, finish, strict, answer text)

    def add(pid, c, i, ids, finish="stop", strict=True, ans="ok"):
        draws.append((pid, c, i, ids, finish, strict, ans))

    add("ifbench_1", 0, 0, _ids(10, 3))
    add("ifbench_1", 0, 1, _ids(12, 3), ans="good")  # labelled genuine -> preferred
    add("ifbench_2", 0, 0, _ids(10, 3))  # gamed
    add("ifbench_2", 0, 1, _ids(10, 3), strict=False, ans="other")
    add("ifbench_3", 0, 0, _ids(10, 3, mark=False))  # no </think>
    add("ifbench_4", 0, 0, _ids(10, 3), finish="length")  # length cut: not a pass
    add("ifbench_5", 0, 0, _ids(80, 3))  # over the cap (50)
    add("ifbench_6", 0, 0, _ids(10, 3), ans="same")  # gamed
    add("ifbench_6", 0, 1, _ids(11, 3), ans="same")  # identical answer: inherits gamed
    add("ifbench_7", 0, 0, _ids(10, 3))  # screen gamed over a genuine read: gamed (conflict)
    for k in range(8, 13):
        add(f"ifbench_{k}", 0, 0, _ids(10 + k, 3, mark=True), ans=f"a{k}")
        add(f"ifbench_{k}", 0, 1, _ids(12 + k, 3), ans=f"b{k}")
    add("ifbench_100", 0, 0, _ids(10, 3), ans="r1")
    add("ifbench_100", 0, 1, _ids(14, 3), ans="r2")  # genuine
    add("ifeval_200", 0, 0, _ids(10, 3))  # strict-only: dropped from reach by default
    smp = dan / "artifacts" / run_ / "samples"
    smp.mkdir(parents=True)
    (dan / "artifacts" / run_ / "graded").mkdir()
    (dan / "artifacts" / run_ / "meta").mkdir()
    with (
        gzip.open(smp / f"{job}.jsonl.gz", "wt") as f,
        open(dan / "artifacts" / run_ / "graded" / f"{job}.jsonl", "w") as g,
    ):
        for pid, c, i, ids, fin, st, ans in draws:
            text = ("think </think>\n\n" if 524 in ids else "think ") + ans
            b64 = base64.b64encode(np.asarray(ids, dtype=np.int32).tobytes()).decode()
            f.write(
                json.dumps(
                    {
                        "pid": pid,
                        "model": "s2300",
                        "rev": "main",
                        "job": job,
                        "chunk": c,
                        "i": i,
                        "n_tok": len(ids),
                        "finish": fin,
                        "text": text,
                        "ids": b64,
                    }
                )
                + "\n"
            )
            g.write(
                json.dumps(
                    {
                        "pid": pid,
                        "job": job,
                        "chunk": c,
                        "i": i,
                        "n_tok": len(ids),
                        "finish": fin,
                        "task": pid.split("_")[0],
                        "strict": st,
                    }
                )
                + "\n"
            )
    meta = {
        "job": job,
        "name": "s2300",
        "rev": "main",
        "problems": "olmo_ifomega/problems.jsonl",
        "lora": None,
        "seed": 1,
        "template_sha256": bp.TEMPLATE_SHA256,
    }
    (dan / "artifacts" / run_ / "meta" / f"{job}.json").write_text(json.dumps(meta) + "\n")
    (dan / "artifacts/olmo_iffu/meta").mkdir(parents=True)
    (dan / "artifacts/olmo_iffu/meta/T2_s2300.json").write_text(
        json.dumps({**meta, "job": "T2_s2300", "problems": "olmo_iffu/problems_hint.jsonl"}) + "\n"
    )

    def write_csv(rel, header, rows):
        with open(dan / rel, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows)

    write_csv(
        "olmo_ifcurve/handcheck.csv",
        ["job", "pid", "chunk", "i", "final_label"],
        [
            [job, "ifbench_1", 0, 1, "genuine"],
            [job, "ifbench_2", 0, 0, "gamed"],
            [job, "ifbench_6", 0, 0, "gamed"],
            [job, "ifbench_100", 0, 1, "genuine"],
        ],
    )
    write_csv(
        "olmo_ifomega/handcheck.csv",
        ["job", "pid", "chunk", "i", "final_label"],
        [[job, "ifbench_7", 0, 0, "genuine"]],
    )
    write_csv("olmo_ifladder/handcheck_src.csv", ["job", "pid", "chunk", "i", "final_label"], [])
    write_csv(
        "olmo_transplant/handcheck_src.csv",
        ["key", "final_label"],
        [[f"{job}|ifbench_8|0|1", "genuine"]],
    )
    write_csv(
        "artifacts/olmo_ifcurve/hand/screen.csv",
        ["job", "pid", "chunk", "i", "reason"],
        [[job, "ifbench_7", 0, 0, "rule 1"]],
    )

    out = tmp_path / "pool"
    man = bp.build(dan, out, cap=50, n_train=3, n_test=2, seed=7, tokenizer=None, reach_any=False)
    st = man["prompt_status"]["arm A minus reach"]
    assert st == {
        "ok": 6,
        "only gamed": 3,
        "no </think> in any strict pass": 1,
        "no strict pass": 1,
        "over cap": 1,
    }
    assert [s["job"] for s in man["skipped_s2300_jobs"]] == ["T2_s2300"]
    assert man["draw_stats"]["label_conflicts"] == 1 and man["draw_stats"]["inherited_gamed"] == 1
    sets = {
        k: [json.loads(x) for x in (out / f"{k}.jsonl").read_text().splitlines()]
        for k in ("train", "test", "reach")
    }
    assert len(sets["train"]) == 3 and len(sets["test"]) == 2
    tr, te = {r["pid"] for r in sets["train"]}, {r["pid"] for r in sets["test"]}
    assert not tr & te and (tr | te) <= {"ifbench_1"} | {f"ifbench_{k}" for k in range(8, 13)}
    assert (
        [r["pid"] for r in sets["reach"]] == ["ifbench_100"]
        and sets["reach"][0]["i"] == 1
        and sets["reach"][0]["label"] == "genuine"
    )
    for r in sets["train"] + sets["test"]:
        ids = oe.b64_ids(r["ids"])
        assert (
            ids[r["think_end"] - 3 : r["think_end"]] == [524, 27963, 1363]
            and ids[-1] == bp.EOS
            and r["n_tok"] <= 50
        )
        if r["pid"] == "ifbench_1":
            assert r["i"] == 1 and r["label"] == "genuine"
        if r["pid"] == "ifbench_8":
            assert r["i"] == 1 and r["label"] == "genuine"
    sha = man["files"]["train.jsonl"]["sha256"]
    man2 = bp.build(
        dan,
        tmp_path / "pool2",
        cap=50,
        n_train=3,
        n_test=2,
        seed=7,
        tokenizer=None,
        reach_any=False,
    )
    assert man2["files"]["train.jsonl"]["sha256"] == sha  # deterministic
    man3 = bp.build(
        dan, tmp_path / "pool3", cap=50, n_train=3, n_test=2, seed=7, tokenizer=None, reach_any=True
    )
    assert man3["files"]["reach.jsonl"]["rows"] == 2


def test_extra_epochs_conv_block(tmp_path):
    model = oe.wrap_lora(tiny_model(), grad_ckpt=True)
    sets = make_sets()
    assert run(model, sets, tmp_path, ckpt="dpo", lrs=[1e-2], extra_epochs=2, save="best") == 0
    r = oe.read_rows(tmp_path / "edl_rows.jsonl")[0]
    cv = r["conv"]
    assert (
        cv["status"] == "ok"
        and cv["epochs"] == 2
        and len(cv["train_loss"]) == 2
        and len(cv["train_loss"][0]) == 6
    )
    assert len(cv["test"]["think_nats"]) == 4 and len(cv["reach"]["answer_nats"]) == 3
    assert sum(cv["test"]["think_nats"]) != sum(
        r["evals"]["6"]["test"]["think_nats"]
    )  # more training moved it
    assert (tmp_path / "adapters" / "best_order0" / "adapter_config.json").exists()
    # the saved adapter is theta_n (after the online pass, before the extra epochs)
    from peft import PeftModel

    m2 = PeftModel.from_pretrained(tiny_model(), str(tmp_path / "adapters" / "best_order0"))
    sc = oe.score(m2, sets["test"], batch_tokens=64)
    np.testing.assert_allclose(sc["think_nats"], r["evals"]["6"]["test"]["think_nats"], rtol=1e-5)
    with pytest.raises(SystemExit):  # rows made with other extra_epochs are another setup
        run(model, sets, tmp_path, ckpt="dpo", lrs=[1e-2], extra_epochs=0)


def test_step0_token_stats_sanity(tmp_path):
    model = oe.wrap_lora(tiny_model(), grad_ckpt=True)
    sets = make_sets()
    assert run(model, sets, tmp_path, lrs=[1e-2]) == 0
    s0 = json.loads((tmp_path / "step0.json").read_text())
    ts = oe.load_token_stats(tmp_path / "step0_tokens.npz")
    assert s0["token_stats"]["examples"] == 13 and set(ts) == {
        e.pid for nm in oe.SETS for e in sets[nm]
    }
    for nm in oe.SETS:
        for k, e in enumerate(sets[nm]):
            d = ts[e.pid]
            assert (
                d["set"] == nm
                and d["think_end"] == e.think_end
                and d["n_tok"] == e.n_tok == len(d["nll"])
            )
            assert d["token"].tolist() == e.resp
            # same numbers as the step-0 sums (fp32 per position vs the cross-entropy path)
            assert float(d["nll"].astype(np.float64).sum()) == pytest.approx(
                s0[nm]["think_nats"][k] + s0[nm]["answer_nats"][k], rel=1e-5
            )
            assert float(d["nll"][: e.think_end].astype(np.float64).sum()) == pytest.approx(
                s0[nm]["think_nats"][k], rel=1e-5
            )
            top1 = d["topk_ids"][:, 0] == d["token"]
            assert ((d["rank"] == 0) == top1).all()
            assert (np.diff(d["topk_logp"].astype(np.float32), axis=1) <= 1e-3).all()
            assert (d["entropy"] >= 0).all() and (
                d["entropy"].astype(np.float32) <= math.log(V) + 1e-2
            ).all()
            inside = (
                d["rank"] < oe.TOPK
            )  # where the stored token is in the top-20 its log-prob is there too
            lp = np.take_along_axis(
                d["topk_logp"].astype(np.float32), np.clip(d["rank"], 0, oe.TOPK - 1)[:, None], 1
            )[:, 0]
            np.testing.assert_allclose(lp[inside], -d["nll"][inside], atol=2e-2)
    # a restart reuses step 0 only when the token file exists
    (tmp_path / "step0_tokens.npz").unlink()
    calls = {"n": 0}
    real = oe.compute_step0

    def count(*a, **k):
        calls["n"] += 1
        return real(*a, **k)

    oe.compute_step0 = count
    try:
        assert run(model, sets, tmp_path, lrs=[1e-2]) == 0
    finally:
        oe.compute_step0 = real
    assert calls["n"] == 1 and (tmp_path / "step0_tokens.npz").exists()


def test_conv_rows_best_equals_in_place_continuation(tmp_path):
    sets = make_sets()
    m1 = oe.wrap_lora(tiny_model(), grad_ckpt=True)
    assert (
        run(
            m1,
            sets,
            tmp_path / "best",
            lrs=[1e-2, 3e-2],
            orders=2,
            extra_epochs=2,
            conv_rows="best",
        )
        == 0
    )
    rows = oe.read_rows(tmp_path / "best" / "edl_rows.jsonl")
    main = [r for r in rows if r.get("schema") == oe.SCHEMA]
    conv = [r for r in rows if r.get("schema") == oe.CONV_SCHEMA]
    assert len(main) == 4 and all("conv" not in r for r in main) and len(conv) == 1
    code = {
        r["lr"]: sum(r["online"]["think_nats"]) + sum(r["online"]["answer_nats"])
        for r in main
        if r["order"] == 0
    }
    c = conv[0]
    assert (
        c["lr"] == min(code, key=code.get)
        and c["order"] == 0
        and c["load_check_max_abs_diff_test_nats"] == 0.0
    )
    m2 = oe.wrap_lora(tiny_model(), grad_ckpt=True)
    assert (
        run(m2, sets, tmp_path / "all", lrs=[c["lr"]], orders=1, extra_epochs=2, conv_rows="all")
        == 0
    )
    ref = oe.read_rows(tmp_path / "all" / "edl_rows.jsonl")[0]["conv"]
    np.testing.assert_allclose(
        c["conv"]["test"]["think_nats"], ref["test"]["think_nats"], rtol=1e-6
    )
    np.testing.assert_allclose(
        c["conv"]["reach"]["answer_nats"], ref["reach"]["answer_nats"], rtol=1e-6
    )
    # a restart finds everything done (no new conv row)
    assert (
        run(
            m1,
            sets,
            tmp_path / "best",
            lrs=[1e-2, 3e-2],
            orders=2,
            extra_epochs=2,
            conv_rows="best",
        )
        == 0
    )
    assert len(oe.read_rows(tmp_path / "best" / "edl_rows.jsonl")) == 5
    d = tmp_path / "ana" / "dpo" / "edl"
    d.mkdir(parents=True)
    for f in ("edl_rows.jsonl", "step0.json"):
        (d / f).write_text((tmp_path / "best" / f).read_text())
    res = an.analyze(tmp_path / "ana", B=20)["checkpoints"]["dpo"]
    assert "EDL_conv" in res and res["EDL_conv"]["sum"]["answer"]["lr"] == c["lr"]


def test_ce_checkpoint_same_loss_and_grads():
    ex = make_sets()["train"][0]
    out = []
    for flag in (False, True):
        model = oe.wrap_lora(tiny_model(), grad_ckpt=True)
        oe.reset_lora(model, 3)
        with torch.no_grad():
            for mod in oe.lora_layers(model):
                mod.lora_B["default"].weight.normal_(0, 0.1)  # nonzero, so A gets gradient too
        params = oe.trainable(model)
        opt = torch.optim.AdamW(params, lr=1e-3)
        oe.OPTS["ce_checkpoint"] = flag
        try:
            out.append(oe.train_step(model, opt, params, ex, grad_clip=1e9))
        finally:
            oe.OPTS["ce_checkpoint"] = False
    assert out[0][0] == pytest.approx(out[1][0], rel=1e-6) and out[0][1] == pytest.approx(
        out[1][1], rel=1e-5
    )


def test_analysis_and_figure_on_partial_checkpoints(tmp_path):
    pytest.importorskip("matplotlib")
    import fig_edl

    sets = make_sets()
    for ck, seed, kw in (
        ("dpo", 0, {"lrs": [1e-2, 3e-2], "orders": 2}),
        ("s0500", 1, {"lrs": [1e-2]}),
    ):
        model = oe.wrap_lora(tiny_model(seed), grad_ckpt=True)
        assert run(model, sets, tmp_path / "runs" / ck / "edl", ckpt=ck, **kw) == 0
    s0 = tmp_path / "runs" / "s2300" / "edl"  # only step 0 so far
    s0.mkdir(parents=True)
    (s0 / "step0.json").write_text((tmp_path / "runs" / "s0500" / "edl" / "step0.json").read_text())
    s = an.analyze(tmp_path / "runs", B=30)
    assert (
        list(s["checkpoints"]) == ["dpo", "s0500", "s2300"]
        and "EDL_star" not in s["checkpoints"]["s2300"]
    )
    assert set(s["checkpoints"]["dpo"]["gap_share"]) == {"total", "think", "answer"}
    lm = s["checkpoints"]["dpo"]["L_min_posthoc"][
        "answer"
    ]  # post hoc minimum over the evaluated points
    assert (
        lm["nats_per_tok"] == min(lm["all_points"].values())
        and lm["at"] in lm["all_points"]
        and "post hoc" in lm["note"]
    )
    rows = an.flatten(s["checkpoints"])
    assert any(r["metric"] == "EDL_star_mdl" and r["segment"] == "answer" for r in rows)
    from code.ndbench.figures import style as st

    with st.paper_style():
        fig = fig_edl.make_figure(s)
        paths = st.save(fig, tmp_path / "fig" / "edl_curve")
        st.close(fig)
    assert all(p.exists() and p.stat().st_size > 0 for p in paths)


def _scaled_run_one(monkeypatch, factor: dict, calls: list):
    """Wrap run_one so that a row's online nats (and the code length its callback sees) are scaled by factor[lr]:
    a deterministic way to choose which lr has the shortest code length."""
    real = oe.run_one

    def wrapped(model, sets, step0, **kw):
        f = factor.get(kw["lr"], 1.0)
        calls.append((kw["lr"], kw["order"]))
        cb = kw.get("on_online_done")
        if cb is not None:
            kw["on_online_done"] = lambda m, code, opt, cb=cb: cb(m, code * f, opt)
        fields, model = real(model, sets, step0, **kw)
        for k in ("think_nats", "answer_nats"):
            fields["online"][k] = [x * f for x in fields["online"][k]]
        return fields, model

    monkeypatch.setattr(oe, "run_one", wrapped)


def _sha(p: Path) -> str:
    import hashlib

    return hashlib.sha256(p.read_bytes()).hexdigest()


@pytest.mark.parametrize("scenario", ["new_wins", "new_loses", "best_left_grid"])
def test_grid_shift_extension(tmp_path, monkeypatch, scenario):
    a, b, c, d = 1e-4, 3e-4, 1e-3, 3e-3
    factor = {a: 1.0, b: 0.9, c: 0.8, d: 0.5 if scenario == "new_wins" else 1.2}
    calls: list = []
    _scaled_run_one(monkeypatch, factor, calls)
    sets = make_sets(n=4, m=2, r=1)
    kw = dict(
        orders=2, extra_epochs=1, conv_rows="best", save="best", evals=[0, 2, 4], reach_evals=[0, 4]
    )
    model = oe.wrap_lora(tiny_model(), grad_ckpt=True)
    out = tmp_path / "dpo" / "edl"
    assert run(model, sets, out, ckpt="dpo", lrs=[a, b, c], **kw) == 0
    rows0 = oe.read_rows(out / "edl_rows.jsonl")
    conv0 = [r for r in rows0 if r["schema"] == oe.CONV_SCHEMA]
    assert len(conv0) == 1 and conv0[0]["lr"] == c
    best_dir = out / "adapters" / "best_order0"
    sha0 = _sha(best_dir / "adapter_model.safetensors")
    calls.clear()
    grid = [b, c, d] if scenario != "best_left_grid" else [a, b, d]
    assert run(model, sets, out, ckpt="dpo", lrs=grid, **kw) == 0
    rows1 = oe.read_rows(out / "edl_rows.jsonl")
    assert rows1[: len(rows0)] == rows0  # nothing rewritten
    main_new = [r for r in rows1[len(rows0) :] if r["schema"] == oe.SCHEMA]
    conv_new = [r for r in rows1[len(rows0) :] if r["schema"] == oe.CONV_SCHEMA]
    assert sorted((r["lr"], r["order"]) for r in main_new) == [(d, 0), (d, 1)]
    best_json = json.loads((out / "adapters" / "BEST.json").read_text())
    done = json.loads((out / "DONE").read_text())
    assert done["lrs"] == sorted(grid)
    if scenario == "new_wins":
        assert (
            len(conv_new) == 1
            and conv_new[0]["lr"] == d
            and conv_new[0]["load_check_max_abs_diff_test_nats"] == 0.0
        )
        assert conv_new[0]["supersedes"][0]["lr"] == c and best_json["lr"] == d
        assert _sha(best_dir / "adapter_model.safetensors") != sha0
        assert calls == [(d, 0), (d, 1)]
        want = d
    elif scenario == "new_loses":
        assert (
            conv_new == []
            and best_json["lr"] == c
            and _sha(best_dir / "adapter_model.safetensors") == sha0
        )
        assert calls == [(d, 0), (d, 1)]
        want = c
    else:  # the saved best (c) is outside the new grid: b is the grid's best; its online pass is re-run for the conv row
        assert len(conv_new) == 1 and conv_new[0]["lr"] == b and "re-run" in conv_new[0]["mode"]
        assert (
            conv_new[0]["rerun_code_length_diff"] == pytest.approx(0.0, abs=1e-6)
            and best_json["lr"] == b
        )
        assert calls == [(d, 0), (d, 1), (b, 0)]
        want = b
    calls.clear()
    assert (
        run(model, sets, out, ckpt="dpo", lrs=grid, **kw) == 0 and calls == []
    )  # nothing left to do
    s = an.analyze(tmp_path, B=20, lrs=grid)["checkpoints"]["dpo"]
    assert (
        s["lr_mdl"]["total"] == want
        and s["grid"]["lrs"] == sorted(grid)
        and s["grid"]["ln_k"] == pytest.approx(math.log(3))
    )
    assert set(s["by_lr"]) == {f"{x:g}" for x in grid}
    off = {a, b, c} - set(grid)
    assert set(s["off_grid"]) == {f"{x:g}" for x in off}
    assert s["EDL_conv"]["sum"]["total"]["lr"] == want
    if scenario != "new_loses":
        assert any(x["lr"] == c for x in s["conv_ignored"])


# ------------------------------------------------------------------------------------------ in-context (olmo_icl.py)


class ICLTok(FakeTok):
    """FakeTok whose chat template is the piecewise encoding olmo_icl builds (as the real tokenizer's is, checked on the pool)."""

    eos_token_id = 1

    def encode(self, s, add_special_tokens=False):
        return [6 + (ord(ch) % 50) for ch in s]

    def apply_chat_template(
        self, msgs, add_generation_prompt, tokenize, return_dict, chat_template
    ):
        import olmo_icl as icl

        return (
            self.encode(icl.SYS)
            + self.encode("<|im_start|>user\n" + msgs[-1]["content"] + icl.END)
            + self.encode(icl.HEAD)
            + [0]
        )


def icl_sets(seed=0, n=8, m=3, r=2):
    rng = np.random.default_rng(seed)
    tok = ICLTok()
    out, prompts = {}, {}
    for name, k in (("train", n), ("test", m), ("reach", r)):
        exs = []
        for j in range(k):
            pid = f"{name}{j}"
            prompts[pid] = f"Do {pid}."
            T = int(rng.integers(10, 20))
            resp = rng.integers(6, V, size=T).tolist()
            resp[-1] = 1
            exs.append(
                oe.Example(
                    pid,
                    oe.prompt_ids(tok, prompts[pid]),
                    resp,
                    int(rng.integers(2, T - 2)),
                    "genuine" if j % 2 == 0 or name != "train" else "strict-only",
                )
            )
        out[name] = exs
    return out, prompts


def test_icl_k0_reproduces_step0_and_demos_fixed(tmp_path):
    import olmo_icl as icl

    sets, prompts = icl_sets()
    rows = {}
    for ck, seed in (("dpo", 0), ("s2300", 1)):
        model = oe.wrap_lora(tiny_model(seed, max_pos=2048), grad_ckpt=True)
        edl = tmp_path / ck / "edl"
        assert (
            oe.run_edl(
                model,
                ICLTok(),
                sets,
                POOL_INFO,
                edl,
                ckpt=ck,
                ident={},
                lrs=[1e-2],
                orders=1,
                seed=0,
                evals=[0, 6],
                reach_evals=[0, 6],
                batch_tokens=300,
                heartbeat_secs=3600,
                token_stats_on=False,
            )
            == 0
        )
        step0 = json.loads((edl / "step0.json").read_text())
        oe.reset_lora(model, 5)
        assert (
            icl.run_icl(
                model,
                ICLTok(),
                sets,
                prompts,
                tmp_path / ck / "icl",
                ckpt=ck,
                ks=[0, 1, 2],
                draws=2,
                seed=0,
                step0=step0,
                k0_batch_tokens=300,
                heartbeat_secs=3600,
            )
            == 0
        )
        rows[ck] = {
            (r["k"], r["draw"]): r for r in oe.read_rows(tmp_path / ck / "icl" / "icl_rows.jsonl")
        }
        assert rows[ck][(0, 0)]["k0_check_max_abs_diff"] == {"test": 0.0, "reach": 0.0}
        assert set(rows[ck]) == {(0, 0), (1, 0), (1, 1), (2, 0), (2, 1)}
    targets = {e.pid for e in sets["test"] + sets["reach"]}
    for key in rows["dpo"]:
        a, b = rows["dpo"][key], rows["s2300"][key]
        assert a["demo_pids"] == b["demo_pids"]  # identical demo sets at every checkpoint
        assert not set(a["demo_pids"]) & targets  # no target is its own demo
        assert all(p in {e.pid for e in sets["train"]} for p in a["demo_pids"])
    assert rows["dpo"][(2, 0)]["demo_pids"][:1] == rows["dpo"][(1, 0)]["demo_pids"]  # nested
    assert not set(rows["dpo"][(2, 0)]["demo_pids"]) & set(
        rows["dpo"][(2, 1)]["demo_pids"]
    )  # draws disjoint
    assert all(
        p.startswith("train") and int(p[5:]) % 2 == 0
        for r in rows["dpo"].values()
        for p in r["demo_pids"]
    )  # genuine only
    k1 = rows["dpo"][(1, 0)]["test"]
    assert k1["context_len"][0] > rows["dpo"][(0, 0)]["test"]["context_len"][0]
    # resumable: nothing left to do
    model = oe.wrap_lora(tiny_model(0, max_pos=2048), grad_ckpt=True)
    assert (
        icl.run_icl(
            model,
            ICLTok(),
            sets,
            prompts,
            tmp_path / "dpo" / "icl",
            ckpt="dpo",
            ks=[0, 1, 2],
            draws=2,
            seed=0,
            step0=None,
            k0_batch_tokens=300,
            heartbeat_secs=3600,
        )
        == 0
    )
    assert len(oe.read_rows(tmp_path / "dpo" / "icl" / "icl_rows.jsonl")) == 5
    # analysis: k = 0 equals step 0's L0 rate; share defined except at s2300; gain = L0 - Lk; figure renders
    import analyze_icl as ai

    res = ai.analyze(tmp_path, B=40)
    d = res["checkpoints"]["dpo"]
    step0 = json.loads((tmp_path / "dpo" / "edl" / "step0.json").read_text())
    for seg in ("total", "think", "answer"):
        l0 = float(an.seg_vals(step0["test"], seg).sum() / an.seg_toks(step0["test"], seg).sum())
        assert abs(d["test"][seg]["0"]["L"] - l0) < 1e-12
        e = d["test"][seg]["2"]
        assert (
            abs(e["gain"] - (l0 - e["L"])) < 1e-12 and e["draws"] == 2 and len(e["per_draw"]) == 2
        )
        assert abs(e["L"] - np.mean(e["per_draw"])) < 1e-9  # equal targets in both draws
        assert "share" in e and "share" not in res["checkpoints"]["s2300"]["test"][seg]["2"]
        lf = float(
            an.seg_vals(
                json.loads((tmp_path / "s2300" / "edl" / "step0.json").read_text())["test"], seg
            ).sum()
            / an.seg_toks(step0["test"], seg).sum()
        )
        assert abs(e["share"] - (l0 - e["L"]) / (l0 - lf)) < 1e-9
    assert any(
        r["metric"] == "gap_share" and r["segment"] == "answer"
        for r in ai.flatten(res["checkpoints"])
    )
    if importlib.util.find_spec("matplotlib") is not None:
        import fig_icl

        from code.ndbench.figures import style as st

        with st.paper_style():
            fig = fig_icl.make_figure(res)
            paths = st.save(fig, tmp_path / "fig" / "icl_curve")
            st.close(fig)
        assert all(p.exists() and p.stat().st_size > 0 for p in paths)


def test_icl_prefix_cache_matches_full_forward():
    import olmo_icl as icl

    sets, prompts = icl_sets()
    model = oe.wrap_lora(
        tiny_model(0, sliding=16, max_pos=2048), grad_ckpt=False
    )  # contexts run past the sliding window
    oe.reset_lora(model, 0)
    model.eval()
    pc = icl.pieces(ICLTok())
    dem = icl.demo_order(sets["train"], 0, 0, 2)
    prefix = list(pc.sys)
    for e in dem:
        prefix += pc.user(prompts[e.pid]) + pc.head + e.resp[:-1] + pc.end
    with torch.inference_mode():
        cache = icl.build_cache(model, prefix)
        for e in sets["test"]:
            ctx = icl.context_ids(pc, prompts, dem, prompts[e.pid], 1)
            a = icl.score_one(model, ctx, e, pad_id=0)
            b = icl.score_one(model, ctx, e, pad_id=0, cache=cache, prefix_len=len(prefix))
            np.testing.assert_allclose(a, b, rtol=1e-4)


def test_icl_context_matches_chat_template_text():
    import olmo_icl as icl
    from olmo_template import TEMPLATE as T

    tdir = Path("/scratch/cowifornia/nd-rl/olmo_elicit/tok/s2300")
    pool = Path("/scratch/cowifornia/nd-rl/olmo_elicit/pool")
    if not (tdir.exists() and pool.exists()):
        pytest.skip("real tokenizer / pool not on this machine")
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(tdir))
    sets, _ = oe.load_pool(pool, tok, n=64, m=32, max_tok=None)
    prompts = icl.raw_prompts(pool)
    pc = icl.pieces(tok)
    dem = icl.demo_order(sets["train"], 0, 0, 2)
    tgt = sets["test"][0]
    ctx = icl.context_ids(pc, prompts, dem, prompts[tgt.pid], tok.eos_token_id)
    msgs = []
    for e in dem:
        msgs += [
            {"role": "user", "content": prompts[e.pid]},
            {
                "role": "assistant",
                "content": "<think>" + tok.decode(e.resp[:-1], clean_up_tokenization_spaces=False),
            },
        ]
    msgs.append({"role": "user", "content": prompts[tgt.pid]})
    text = tok.apply_chat_template(
        msgs, add_generation_prompt=True, tokenize=False, chat_template=T
    )
    assert tok.decode(ctx, clean_up_tokenization_spaces=False) == text
    assert pc.sys + pc.user(prompts[tgt.pid]) + pc.head == tgt.prompt


# ------------------------------------------------------------------------------------------ token profile (analyze_tokens.py)


def _full_js(p, q):
    m = 0.5 * (p + q)
    with np.errstate(divide="ignore", invalid="ignore"):
        a = np.where(p > 0, p * np.log(p / m), 0.0).sum()
        b = np.where(q > 0, q * np.log(q / m), 0.0).sum()
    return 0.5 * (a + b)


def test_tokens_js_topk_against_full_vocabulary():
    import analyze_tokens as at

    rng = np.random.default_rng(0)
    for V in (20, 60):
        P = rng.dirichlet(np.full(V, 0.3), size=200)
        Q = 0.6 * P + 0.4 * rng.dirichlet(np.full(V, 0.3), size=200)
        ids = lambda X: np.argsort(-X, axis=1)[:, :20].astype(np.int32)  # noqa: E731
        pi, qi = ids(P), ids(Q)
        pl = np.log(np.take_along_axis(P, pi, 1)).astype(np.float16)
        ql = np.log(np.take_along_axis(Q, qi, 1)).astype(np.float16)
        out = at.js_topk(pi, pl, qi, ql, chunk=64)
        full = np.array([_full_js(P[j], Q[j]) for j in range(200)])
        assert (out["js_lb"] <= full + 2e-3).all()  # a lower bound (fp16 log-probs)
        assert (out["js"] >= 0).all() and (out["js"] <= np.log(2) + 1e-9).all()
        if V == 20:  # top 20 = the whole vocabulary: both equal the exact JS
            np.testing.assert_allclose(out["js"], full, atol=2e-3)
            np.testing.assert_allclose(out["js_lb"], full, atol=2e-3)
        else:  # empirical (not a proven bound): the truncation error stays below the discarded top-20 mass
            miss = np.maximum(1 - out["mass_p"], 1 - out["mass_q"])
            assert (np.abs(out["js"] - full) <= miss + 2e-3).all()
    same = at.js_topk(pi, pl, pi, pl)
    assert np.abs(same["js"]).max() < 1e-12 and np.abs(same["js_lb"]).max() < 1e-12
    v = np.array([5.0, -1.0, 3.0, 1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    assert at.top_share(v, 0.1) == 5 / 10 and at.top_share(v, 0.2) == 8 / 10
    assert at.top_share(np.clip(v, 0, None), 0.2) == 8 / 11
    assert at.top_share(v, 0.1, order_by=-np.arange(10.0)) == 5 / 10  # first position ranked top


def test_tokens_profile_end_to_end(tmp_path):
    import analyze_tokens as at

    sets = make_sets()
    for ck, seed in (("dpo", 0), ("s0500", 3), ("s2300", 1)):
        model = oe.wrap_lora(tiny_model(seed), grad_ckpt=True)
        assert run(model, sets, tmp_path / ck / "edl", ckpt=ck, evals=[0, 6], lrs=[1e-2]) == 0
    res = at.analyze(tmp_path, B=30)
    assert list(res["checkpoints"]) == ["dpo", "s0500", "s2300"]
    ref = res["checkpoints"]["s2300"]["sets"]["all"]
    for seg in ("total", "think", "answer"):
        assert (
            ref[seg]["excess_mean"] == 0
            and ref[seg]["js_mean"] < 1e-12
            and ref[seg]["share_argmax_agree"] == 1
        )
    s0 = {
        c: json.loads((tmp_path / c / "edl" / "step0.json").read_text()) for c in ("dpo", "s2300")
    }
    for r in res["checkpoints"].values():
        assert r["step0_check_max_abs_diff_nats"] < 1e-3
        for seg in ("total", "think", "answer"):
            d = r["sets"]["all"][seg]
            n = sum(int(an.seg_toks(s0["dpo"][s], seg).sum()) for s in ("train", "test", "reach"))
            assert d["n_positions"] == n and sum(b["n"] for b in d["entropy_deciles"]) == n
            assert (
                0 <= d["share_top1"] <= d["share_top5"] <= 1
                and 0 <= d["js_lb_mean"] <= d["js_mean"] + 1e-6
            )
            assert d["js_mean_ci"][0] <= d["js_mean"] <= d["js_mean_ci"][1]
    d = res["checkpoints"]["dpo"]["sets"]["all"]["answer"]

    def tot(c):
        return sum(float(an.seg_vals(s0[c][s], "answer").sum()) for s in ("train", "test", "reach"))

    toks = sum(float(an.seg_toks(s0["dpo"][s], "answer").sum()) for s in ("train", "test", "reach"))
    assert (
        abs(d["excess_mean"] - (tot("dpo") - tot("s2300")) / toks) < 1e-6
    )  # = the L0 gap per token
    assert (
        abs(sum(b["share_of_net_excess"] for b in d["entropy_deciles"]) - 1) < 1e-9
        if d["excess_sum"] > 0
        else True
    )
    assert set(res["checkpoints"]["dpo"]["sets"]) == {"all", "train", "test", "reach"}
    rows = at.flatten(res["checkpoints"])
    assert any(r["metric"] == "excess_share_top5pct_net" and r["segment"] == "answer" for r in rows)
    if importlib.util.find_spec("matplotlib") is not None:
        import fig_tokens

        from code.ndbench.figures import style as st

        with st.paper_style():
            fig = fig_tokens.make_figure(res)
            paths = st.save(fig, tmp_path / "fig" / "tokens_profile")
            st.close(fig)
        assert all(p.exists() and p.stat().st_size > 0 for p in paths)


# ------------------------------------------------------------------------------------------ post hoc readouts (analyze_edl.posthoc)


def test_posthoc_crossing_arithmetic(tmp_path):
    steps = [0, 100, 500, 1100, 2300]
    h = an.half_step(steps, [10, 9, 6, 2, 1], "half_of_first", "down")
    assert h["target"] == 5 and h["bracket"] == [500, 1100] and abs(h["step"] - 650) < 1e-9
    p = an.half_step(steps, [29.0, 31.5, 41.2, 57.7, 68.3], "half_of_change", "up")
    assert (
        p["bracket"] == [500, 1100]
        and abs(p["step"] - (500 + (48.65 - 41.2) / (57.7 - 41.2) * 600)) < 1e-9
    )
    assert an.half_step(steps, [10, 9, 8, 7, 6], "half_of_first")["step"] is None
    early = {"step": 300.0, "bracket": [100.0, 500.0]}
    late = {"step": 1500.0, "bracket": [1100.0, 2300.0]}
    same = {"step": 900.0, "bracket": [500.0, 1100.0]}
    assert an.p4_verdict(early, p) == "pass" and an.p4_verdict(late, p) == "fail"
    assert an.p4_verdict(same, p).startswith("not established") and an.p4_verdict(
        {"step": None}, p
    ).startswith("undetermined")
    l0 = [0.62, 0.61, 0.58, 0.54, 0.50]
    e = an.step_equivalent(steps, l0, 0.52)
    assert abs(e["step"] - 1700) < 1e-9 and e["label"] == "between 1100 and 2300"
    assert an.step_equivalent(steps, l0, 0.49)["label"].startswith("beyond s2300")
    assert an.step_equivalent(steps, l0, 0.63)["label"].startswith("at or above")
    assert not an.first_crossing(steps, [5, 6, 4, 3, 2], 3.5, "down")["curve_monotone"]
    csv_p = tmp_path / "repro_summary.csv"
    csv_p.write_text(
        "ckpt,step,k,his,ours_regrade,ours_own,diff_pts\ndpo,0,1,0.289851,0,0,0\ndpo,0,2,0.3,0,0,0\ns0050,50,8,nan,nan,nan,nan\ns2300,2300,1,0.683486,0,0,0\n"
    )
    p1 = an.load_pass1(csv_p)
    assert (
        set(p1) == {"dpo", "s2300"}
        and abs(p1["dpo"] - 28.9851) < 1e-9
        and abs(p1["s2300"] - 68.3486) < 1e-9
    )


def test_posthoc_paired_null_correction_end_to_end(tmp_path):
    import shutil

    sets = make_sets()
    runs = tmp_path / "runs"
    for ck, seed, k in (("dpo", 0, 2), ("s0500", 3, 1), ("s2300", 1, 1)):
        model = oe.wrap_lora(tiny_model(seed), grad_ckpt=True)
        assert run(model, sets, runs / ck / "edl", ckpt=ck, lrs=[1e-2, 3e-2], orders=k) == 0
    # identical to s2300 with one order (only the test prompts are resampled): a paired difference is exactly 0 in every
    # replicate. With two orders, the orders are resampled independently per checkpoint, by design.
    shutil.copytree(runs / "s2300" / "edl", runs / "s1100" / "edl")
    pass1 = tmp_path / "p.csv"
    pass1.write_text(
        "ckpt,step,k,his\ndpo,0,1,0.29\ns0500,500,1,0.41\ns1100,1100,1,0.58\ns2300,2300,1,0.68\n"
    )
    s = an.analyze(runs, B=40, pass1=pass1)
    assert (
        list(s)[-1] == "posthoc"
        and "POST HOC" in s["posthoc"]["note"]
        and "NOT resampled" in s["posthoc"]["resampled"]
    )
    ph, res = s["posthoc"], s["checkpoints"]
    assert ph["checkpoints"] == ["dpo", "s0500", "s1100", "s2300"]
    for conv in ("sum", "norm"):
        for seg in ("total", "think", "answer"):
            for c in ("s1100", "s2300"):
                d = ph["delta_EDL_star"][c][conv][seg]
                assert (
                    d["nats"] == 0 and d["ci"] == [0.0, 0.0] and d["per_tok_ci"] == [0.0, 0.0]
                )  # paired
            d = ph["delta_EDL_star"]["dpo"][conv][seg]
            a, b = (
                res["dpo"]["EDL_star"][conv]["mdl"][seg],
                res["s2300"]["EDL_star"][conv]["mdl"][seg],
            )
            assert (
                abs(d["nats"] - (a["nats"] - b["nats"])) < 1e-9
                and abs(d["per_tok"] - (a["per_tok"] - b["per_tok"])) < 1e-12
            )
            assert d["ci"][0] <= d["nats"] <= d["ci"][1]
    for key in ("L0_test", "L0_train", "L0_reach", "L64_test"):
        assert ph["delta_L"]["s1100"][key]["total"] == {"nats_per_tok": 0.0, "ci": [0.0, 0.0]} or (
            key == "L64_test" and ph["delta_L"]["s1100"][key]["total"]["ci"] == [0.0, 0.0]
        )
    dl = ph["delta_L"]["dpo"]["L0_test"]["answer"]["nats_per_tok"]
    assert (
        abs(
            dl
            - (
                res["dpo"]["L0"]["test"]["answer"]["nats_per_tok"]
                - res["s2300"]["L0"]["test"]["answer"]["nats_per_tok"]
            )
        )
        < 1e-12
    )
    # P4: the ΔEDL* curve ends at 0, the verdict field exists, the steps match half_step on the stored values
    p4 = ph["P4"]
    vals = p4["delta_EDL_star"]["norm"]["total"]["values_nats"]
    assert vals[-1] == 0 and vals[-2] == 0
    if vals[0] > 0:
        assert (
            p4["delta_EDL_star"]["norm"]["total"]["step"]
            == an.half_step([0, 500, 1100, 2300], vals, "half_of_first")["step"]
        )
    assert (
        abs(
            p4["pass1"]["step"]
            - an.half_step([0, 500, 1100, 2300], [29.0, 41.0, 58.0, 68.0], "half_of_change", "up")[
                "step"
            ]
        )
        < 1e-9
    )
    assert p4["P4_as_written"] is not None and p4["P4_null_corrected"] is not None
    # RL-step equivalent: the same function on the stored L0 curve
    eq = ph["rl_step_equivalent_of_64"]["total"]
    l0 = [eq["L0_test_curve"][c] for c in ("dpo", "s0500", "s1100", "s2300")]
    assert (
        eq["dpo"]["step"]
        == an.step_equivalent([0, 500, 1100, 2300], l0, res["dpo"]["L64"]["total"]["nats_per_tok"])[
            "step"
        ]
    )
    # reach transfer by hand at dpo
    s0d = json.loads((runs / "dpo" / "edl" / "step0.json").read_text())
    s0f = json.loads((runs / "s2300" / "edl" / "step0.json").read_text())
    lr = res["dpo"]["lr_mdl"]["total"]
    rows = [
        r
        for r in oe.read_rows(runs / "dpo" / "edl" / "edl_rows.jsonl")
        if r.get("schema") == oe.SCHEMA and r["lr"] == lr
    ]
    tt = an.seg_toks(s0d["reach"], "total")
    a0 = an.seg_vals(s0d["reach"], "total").sum() / tt.sum()
    a64 = (
        np.mean([an.seg_vals(r["evals"]["6"]["reach"], "total") for r in rows], axis=0).sum()
        / tt.sum()
    )
    af = an.seg_vals(s0f["reach"], "total").sum() / tt.sum()
    rt = ph["reach_transfer"]["dpo"]["total"]
    assert abs(rt["L_reach_0"] - a0) < 1e-12 and abs(rt["L_reach_64"] - a64) < 1e-12
    assert abs(rt["gap_share_reach"] - (a0 - a64) / (a0 - af)) < 1e-9 and "gap_share_test" in rt
    assert any(r["metric"] == "posthoc_delta_EDL_star" for r in an.flatten_posthoc(ph))
    if importlib.util.find_spec("matplotlib") is not None:
        import fig_edl

        from code.ndbench.figures import style as st

        with st.paper_style():
            for make, name in (
                (fig_edl.make_figure, "edl_curve"),
                (fig_edl.make_learning_figure, "edl_learning"),
            ):
                fig = make(s)
                paths = st.save(fig, tmp_path / "fig" / name)
                st.close(fig)
                assert all(p.exists() and p.stat().st_size > 0 for p in paths)
