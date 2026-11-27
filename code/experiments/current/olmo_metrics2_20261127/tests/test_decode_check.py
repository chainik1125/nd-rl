"""CPU tests of tools/decode_check.py with a fake vllm module injected through sys.modules."""

import base64
import hashlib
import importlib.util
import json
import os
import sys
import types
from pathlib import Path

import numpy as np
import pytest

X = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("decode_check", X / "tools" / "decode_check.py")
dc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dc)

CFG = {
    "decode_noise": 0.0,
    "adapter_effect": 0.5,
    "drop_in_decode": False,
    "base_decode_noise": 0.0,
}


def _h(*parts) -> int:
    return int(hashlib.sha256(repr(parts).encode()).hexdigest()[:12], 16)


def _base_lp(ctx, tok):
    return -0.1 - (_h("b", tuple(ctx[-4:]), tok) % 4000) / 1000.0


def _tag(lora_name):
    if lora_name:
        return lora_name
    return "steer" if os.environ.get("STEER_LAYER") else "base"


def _lp(ctx, tok, tag, decode):
    v = _base_lp(ctx, tok)
    if decode and CFG["drop_in_decode"]:
        tag = "base"
    if tag != "base":
        v += CFG["adapter_effect"] * (((_h("a", tuple(ctx[-4:]), tok) % 2000) / 1000.0) - 1.0)
    if decode:
        noise = CFG["base_decode_noise"] if tag == "base" else CFG["decode_noise"]
        v += noise * (((_h("n", tuple(ctx[-4:]), tok) % 2000) / 1000.0) - 1.0)
    return v


class Obj:
    def __init__(self, logprob):
        self.logprob = logprob


class FakeSP:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class FakeLoRA:
    def __init__(self, name, int_id, path):
        self.lora_name, self.lora_int_id, self.lora_path = name, int_id, path


class FakeLLM:
    last_kwargs = None

    def __init__(self, **kw):
        FakeLLM.last_kwargs = kw

    def generate(self, prompts, sp, lora_request=None, use_tqdm=True):
        tag = _tag(lora_request.lora_name if lora_request else None)
        outs = []
        for pr in prompts:
            ids = pr["prompt_token_ids"]
            if getattr(sp, "prompt_logprobs", None) is not None:
                pl = [None] + [
                    {ids[j]: Obj(_lp(ids[:j], ids[j], tag, False))} for j in range(1, len(ids))
                ]
                outs.append(
                    types.SimpleNamespace(
                        prompt_logprobs=pl,
                        outputs=[types.SimpleNamespace(token_ids=[0], logprobs=None)],
                    )
                )
            else:
                ctx, toks, lps = list(ids), [], []
                for _ in range(sp.max_tokens):
                    t = _h("t", tuple(ctx[-4:]), sp.seed, tag if tag != "base" else "base") % 1000
                    lps.append({t: Obj(_lp(ctx, t, tag, True))})
                    toks.append(t)
                    ctx.append(t)
                outs.append(
                    types.SimpleNamespace(
                        prompt_logprobs=None,
                        outputs=[types.SimpleNamespace(token_ids=toks, logprobs=lps)],
                    )
                )
        return outs


@pytest.fixture(autouse=True)
def fake_vllm(monkeypatch):
    CFG.update(decode_noise=0.0, adapter_effect=0.5, drop_in_decode=False, base_decode_noise=0.0)
    v = types.ModuleType("vllm")
    v.LLM, v.SamplingParams = FakeLLM, FakeSP
    lm = types.ModuleType("vllm.lora")
    lr = types.ModuleType("vllm.lora.request")
    lr.LoRARequest = FakeLoRA
    monkeypatch.setitem(sys.modules, "vllm", v)
    monkeypatch.setitem(sys.modules, "vllm.lora", lm)
    monkeypatch.setitem(sys.modules, "vllm.lora.request", lr)
    for k in ("STEER_LAYER", "STEER_VEC", "STEER_ACTIVE_FILE"):
        monkeypatch.delenv(k, raising=False)


@pytest.fixture
def pool(tmp_path):
    d = tmp_path / "pool"
    d.mkdir()
    rows = []
    for i in range(3):
        ids = np.arange(10 + i, 30 + i, dtype=np.int32)
        rows.append({"pid": f"p{i}", "prompt_ids": base64.b64encode(ids.tobytes()).decode()})
    (d / "test.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return d


def run_lora(pool, tmp_path, n=16):
    out = tmp_path / "o.json"
    code = dc.main(
        [
            "lora",
            "--model",
            "M",
            "--tokenizer",
            "T",
            "--adapter",
            str(tmp_path),
            "--adapter-name",
            "A1",
            "--pool",
            str(pool),
            "--out",
            str(out),
            "--n-tokens",
            str(n),
        ]
    )
    return code, json.loads(out.read_text())


def test_lora_pass(pool, tmp_path, capsys):
    CFG["decode_noise"] = 0.01
    code, r = run_lora(pool, tmp_path)
    assert code == 0 and r["status"] == "PASS" and r["verdict"] == "PASS"
    assert all(r["checks"].values())
    assert len(r["tokens_x"]) == 2 and len(r["tokens_x"][0]) == 16 and r["seed"] == 4242
    kw = FakeLLM.last_kwargs
    assert (
        kw["enable_lora"]
        and kw["max_lora_rank"] == 16
        and kw["max_loras"] == 1
        and kw["enable_prefix_caching"]
    )
    assert (
        "enforce_eager" not in kw
        and kw["max_model_len"] == 33321
        and kw["gpu_memory_utilization"] == 0.92
    )
    assert capsys.readouterr().out.strip().splitlines()[-1].startswith("DECODE_CHECK PASS ")


def test_lora_adapter_dropped_in_decode(pool, tmp_path):
    CFG["drop_in_decode"] = True
    code, r = run_lora(pool, tmp_path)
    assert code == 1 and r["status"] == "FAIL"
    assert not r["checks"]["decode_matches_prefill"] and r["checks"]["has_power"]
    assert r["stats"]["decode_x_vs_prefill_base_on_x"]["mean_abs"] < 1e-9


def test_lora_no_power(pool, tmp_path):
    CFG["adapter_effect"] = 0.001
    code, r = run_lora(pool, tmp_path)
    assert code == 2 and r["verdict"] == "NO_POWER" and not r["checks"]["has_power"]


def test_lora_base_mismatch(pool, tmp_path):
    CFG["base_decode_noise"] = 0.5
    code, r = run_lora(pool, tmp_path)
    assert code == 1 and not r["checks"]["base_decode_matches_prefill"]


def test_input_error(tmp_path):
    code = dc.main(
        [
            "lora",
            "--model",
            "M",
            "--adapter",
            "a",
            "--adapter-name",
            "A1",
            "--pool",
            str(tmp_path / "nope"),
            "--out",
            str(tmp_path / "o.json"),
        ]
    )
    assert code == 3


def make_p3(tmp_path):
    p3 = tmp_path / "p3"
    p3.mkdir()
    (p3 / "chosen.json").write_text(
        json.dumps({"layer": 20, "files": {"chosen": "vec.pt"}, "sha256": {"chosen": "abc123"}})
    )
    return p3


def install_stub(monkeypatch, write=True, sha="abc123", layer=20):
    stub = types.ModuleType("steer_patch")

    def install_from_env(env=None, **kw):
        assert os.environ["STEER_LAYER"] == "20" and os.environ["STEER_VEC"] == "vec.pt"
        if write:
            with open(os.environ["STEER_ACTIVE_FILE"], "a") as f:
                f.write(
                    json.dumps({"event": "STEER_ACTIVE", "layer": layer, "vec_sha256": sha}) + "\n"
                )

    stub.install_from_env = install_from_env
    monkeypatch.setitem(sys.modules, "steer_patch", stub)


def steer_pipeline(pool, tmp_path, n=16):
    p3, d = make_p3(tmp_path), tmp_path / "steer_out"
    common = ["--model", "M", "--tokenizer", "T", "--pool", str(pool), "--n-tokens", str(n)]
    assert dc.main(["steer-gen", *common, "--p3", str(p3), "--out", str(d)]) == 0
    assert os.environ.get("STEER_LAYER") == "20"
    os.environ.pop(
        "STEER_LAYER"
    )  # a new process in real life; the stock step must not depend on it
    os.environ["STEER_LAYER"] = "20"  # even if inherited, stock-score must clear it
    assert dc.main(["stock-score", *common, "--dir", str(d)]) == 0
    assert "STEER_LAYER" not in os.environ
    out = tmp_path / "s.json"
    code = dc.main(["steer-summarize", "--dir", str(d), "--p3", str(p3), "--out", str(out)])
    return code, json.loads(out.read_text())


def test_steer_end_to_end_pass(pool, tmp_path, monkeypatch):
    CFG["decode_noise"] = 0.01
    install_stub(monkeypatch)
    code, r = steer_pipeline(pool, tmp_path)
    assert code == 0 and r["status"] == "PASS" and r["checks"]["steer_active"]
    assert "enable_lora" not in r["engine_kwargs"]["steer_gen"]


def test_steer_missing_active(pool, tmp_path, monkeypatch):
    install_stub(monkeypatch, write=False)
    code, r = steer_pipeline(pool, tmp_path)
    assert code == 1 and not r["checks"]["steer_active"] and r["status"] == "FAIL"


def test_steer_wrong_sha(pool, tmp_path, monkeypatch):
    install_stub(monkeypatch, sha="zzz")
    code, r = steer_pipeline(pool, tmp_path)
    assert code == 1 and not r["checks"]["steer_active"]


def test_verdict_function():
    rng = np.random.default_rng(0)
    p = [rng.normal(-2, 1, 200) for _ in range(2)]
    base = [x + 0.3 for x in p]
    ok = dc.verdict([x + 0.001 for x in p], p, base, base, base)
    assert all(ok["checks"].values())
    dropped = dc.verdict(base, p, base, base, base)
    assert not dropped["checks"]["decode_matches_prefill"]
    assert dc.status_of(dropped["checks"]) == ("FAIL", 1)
    weak = dc.verdict(p, p, [x + 0.001 for x in p], p, p)
    assert dc.status_of(weak["checks"]) == ("NO_POWER", 2)
    spiky = [x.copy() for x in p]
    for s in spiky:
        s[:10] += 1.0  # 2.5 % of tokens off by 1 nat: mean small, p99 large
    v = dc.verdict(spiky, p, base, p, p)
    assert (
        v["stats"]["decode_vs_prefill_x"]["p99_abs"] > dc.TOL_P99
        and not v["checks"]["decode_matches_prefill"]
    )
    assert dc.status_of(
        {"decode_matches_prefill": True, "has_power": True, "base_decode_matches_prefill": True}
    ) == ("PASS", 0)
