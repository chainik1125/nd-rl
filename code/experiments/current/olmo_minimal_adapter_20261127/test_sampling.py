# ruff: noqa: E501, SIM115
"""CPU tests for the minimal-adapter sampling side: chain (stub python), lora_gate and multi_sample (stub vLLM), prompt
hashes against the old L_dpo meta, stage_cells, mini_status, analyze_cells (synthetic + the old dpo L cell), fig_cells."""

from __future__ import annotations

import gzip
import json
import os
import shutil
import stat
import subprocess
import sys
import time
import types
import zlib
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "pod"))
import analyze_cells as AC  # noqa: E402
import fig_cells as FC  # noqa: E402
import lora_gate as LG  # noqa: E402
import mini_status as MS  # noqa: E402
import multi_sample as MSP  # noqa: E402
import stage_cells as SC  # noqa: E402

OE = Path("/scratch/cowifornia/nd-rl/olmo_elicit")
DAN = OE / "ext" / "nd-takehome"
OLD_L = OE / "runs" / "dpo" / "L"
TOK = OE / "tok" / "s2300"


# ------------------------------------------------------------------------------------------- fake vLLM
class FakeLogprob:
    def __init__(self, logprob):
        self.logprob = logprob


class FakeLoRARequest:
    def __init__(self, lora_name, lora_int_id, lora_path):
        self.lora_name, self.lora_int_id, self.lora_path = lora_name, lora_int_id, lora_path


class FakeSP:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class FakeComp:
    def __init__(self, index, token_ids):
        self.index, self.token_ids = index, token_ids
        self.finish_reason, self.stop_reason = "stop", None


class FakeOut:
    def __init__(self, request_id, comps):
        self.request_id, self.outputs, self.finished = request_id, comps, True


class FakeEngine:
    def __init__(self):
        self.q, self.added = [], []
        self.vllm_config = types.SimpleNamespace(
            cache_config=types.SimpleNamespace(num_gpu_blocks=7, block_size=16),
            model_config=types.SimpleNamespace(max_model_len=0),
        )

    def add_request(self, rid, prompt, sp, lora_request=None):
        self.q.append((rid, prompt, sp, lora_request))
        self.added.append((rid, lora_request.lora_name if lora_request else None))

    def has_unfinished_requests(self):
        return bool(self.q)

    def step(self):
        rid, prompt, sp, lr = (
            self.q.pop()
        )  # last in, first out: completion order differs from submission
        lid = lr.lora_int_id if lr else 0
        # tokens: 1000 * adapter id + index, so a row shows which adapter served it
        comps = [FakeComp(i, [1000 * lid + i, 5, 6 + i]) for i in range(sp.n)]
        return [FakeOut(rid, comps)]


class FakeLLM:
    last_kwargs: dict = {}
    last_engine: FakeEngine | None = None
    # gate world: {tuple(ids): (peft base nll, peft adapter nll)}, adapters actually applied, noise std, base drift
    world: dict = {}
    applied: set = set()
    noise = 0.01
    drift = 0.0
    calls = 0

    def __init__(self, **kw):
        FakeLLM.last_kwargs = kw
        self.llm_engine = FakeEngine()
        FakeLLM.last_engine = self.llm_engine
        self.kw = kw

    def generate(self, prompts, sp, lora_request=None, use_tqdm=False):
        FakeLLM.calls += 1
        outs = []
        for p in prompts:
            ids = p["prompt_token_ids"]
            base, adapter = self.world[tuple(ids)]
            rng = np.random.default_rng(
                zlib.crc32(repr((ids, lora_request.lora_name if lora_request else None)).encode())
            )
            noise = rng.normal(0, self.noise, len(ids))
            n_resp = len(base)
            nll = (
                adapter
                if lora_request is not None and lora_request.lora_name in self.applied
                else base
            ) + noise[-n_resp:]
            if lora_request is None and self.drift and FakeLLM.calls > 3:
                nll = nll * (1 + self.drift)
            lp = [None] * (len(ids) - n_resp) + [
                {ids[j]: FakeLogprob(-nll[j - (len(ids) - n_resp)])}
                for j in range(len(ids) - n_resp, len(ids))
            ]
            outs.append(types.SimpleNamespace(prompt_logprobs=lp))
        return outs


def install_fake_vllm(monkeypatch):
    FakeLLM.world, FakeLLM.applied, FakeLLM.noise, FakeLLM.drift, FakeLLM.calls = (
        {},
        set(),
        0.01,
        0.0,
        0,
    )
    vllm = types.ModuleType("vllm")
    vllm.LLM, vllm.SamplingParams, vllm.__version__ = FakeLLM, FakeSP, "0.11.0-fake"
    lora = types.ModuleType("vllm.lora")
    req = types.ModuleType("vllm.lora.request")
    req.LoRARequest = FakeLoRARequest
    sp = types.ModuleType("vllm.sampling_params")
    sp.RequestOutputKind = types.SimpleNamespace(FINAL_ONLY="final_only")
    for name, mod in (
        ("vllm", vllm),
        ("vllm.lora", lora),
        ("vllm.lora.request", req),
        ("vllm.sampling_params", sp),
    ):
        monkeypatch.setitem(sys.modules, name, mod)


# ------------------------------------------------------------------------------------------- fixtures
def make_cell(
    root: Path, name: str, rng, delta_mean=-0.05, delta_std=0.1, did_not_train=False, r=16, n_ex=2
):
    """A trained-cell folder with a gate.npz whose PEFT numbers follow base + N(delta_mean, delta_std)."""
    cd = root / name
    (cd / "adapter").mkdir(parents=True)
    (cd / "adapter" / "adapter_config.json").write_text(
        json.dumps({"peft_type": "LORA", "r": r, "target_modules": ["o_proj"], "lora_alpha": 2 * r})
    )
    (cd / "adapter" / "adapter_model.safetensors").write_bytes(name.encode() * 10)
    arr = {"kind": np.array(["dev"] * n_ex), "pid": np.array([f"{name}_p{j}" for j in range(n_ex)])}
    for j in range(n_ex):
        L = 200 + 50 * j
        base = rng.uniform(0.5, 4.0, L)
        arr[f"prompt_ids_{j}"] = rng.integers(10, 500, 30 + j).astype(np.int32)
        arr[f"resp_ids_{j}"] = rng.integers(10, 500, L).astype(np.int32)
        arr[f"base_nll_{j}"] = base
        arr[f"adapter_nll_{j}"] = base + rng.normal(delta_mean, delta_std, L)
    np.savez(cd / "gate.npz", **arr)
    (cd / "cell.json").write_text(
        json.dumps(
            {
                "cell": name,
                "family": "A16",
                "P": 134217728,
                "n": 64,
                "source": "own",
                "selected": {"lr": 3e-4, "step": 64},
                "did_not_train": did_not_train,
                "lora_r": r,
                "mdl_nats": 1.5e5,
                "edl_tok_bits_per_trained_weight": 0.01,
            }
        )
    )
    (cd / "DONE").write_text("ok\n")
    return cd


def load_world(root: Path, names):
    w = {}
    for c in names:
        for e in LG.load_cell(root / c):
            w[tuple(e["prompt"] + e["resp"])] = (e["base"], e["adapter"])
    return w


# ------------------------------------------------------------------------------------------- stage_cells
def test_stage_cells(tmp_path):
    rng = np.random.default_rng(0)
    cells = tmp_path / "runs" / "dpo" / "train" / "cells"
    make_cell(cells, "good", rng)
    make_cell(cells, "dnt", rng, did_not_train=True)
    make_cell(cells, "nodone", rng)
    (cells / "nodone" / "DONE").unlink()
    make_cell(cells, "r32", rng, r=32)
    batch = tmp_path / "b1"
    man, refused = SC.stage(
        tmp_path / "runs", "dpo", ["good", "dnt", "nodone", "r32", "ghost"], batch
    )
    assert set(refused) == {"dnt", "nodone", "r32", "ghost"} and "did_not_train" in refused["dnt"]
    assert list(man["cells"]) == ["good"]
    f = man["cells"]["good"]["files"]["adapter/adapter_model.safetensors"]
    assert len(f["md5"]) == 32 and len(f["sha256"]) == 64 and f["bytes"] == 40
    assert os.stat(batch / "good" / "gate.npz").st_nlink >= 2  # hard link
    man, refused = SC.stage(tmp_path / "runs", "dpo", ["dnt"], batch, force=True)
    assert (
        not refused
        and man["cells"]["dnt"]["forced"]
        and man["cells"]["dnt"]["did_not_train"]
        and "good" in man["cells"]
    )
    assert (
        SC.main(
            [
                "--runs",
                str(tmp_path / "runs"),
                "--ckpt",
                "dpo",
                "--cells",
                "good,nodone",
                "--stage",
                str(tmp_path / "b2"),
            ]
        )
        == 1
    )
    ok, bad = MS.verify_batch(batch)
    assert ok
    (batch / "good" / "cell.json").write_text("{}")
    ok, bad = MS.verify_batch(batch)
    assert not ok and bad == ["good/cell.json"]


# ------------------------------------------------------------------------------------------- lora_gate
def run_gate(tmp_path, monkeypatch, names, applied, deltas, drift=0.0, noise=0.01):
    install_fake_vllm(monkeypatch)
    rng = np.random.default_rng(1)
    root = tmp_path / "batch"
    shutil.rmtree(root, ignore_errors=True)
    for c in names:
        make_cell(root, c, rng, *deltas[c])
    FakeLLM.world = load_world(root, names)
    FakeLLM.applied, FakeLLM.drift, FakeLLM.noise = set(applied), drift, noise
    out = tmp_path / "gate.json"
    rc = LG.main(["--model", "m", "--tokenizer", "t", "--cells-dir", str(root), "--out", str(out)])
    return rc, json.loads(out.read_text())


def test_gate_pass_fail_uninformative(tmp_path, monkeypatch):
    names = ["applied", "notapplied", "tiny"]
    deltas = {"applied": (-0.05, 0.1), "notapplied": (-0.05, 0.1), "tiny": (0.0, 0.004)}
    rc, g = run_gate(tmp_path, monkeypatch, names, ["applied", "tiny"], deltas)
    v = g["verdicts"]
    assert v == {"applied": "PASS", "notapplied": "FAIL", "tiny": "UNINFORMATIVE"}
    assert rc == 1 and g["n_fail"] == 1
    a = g["cells"]["applied"]
    assert a["corr"] > 0.95 and 0.9 < a["ratio"] < 1.1 and a["informative"] and a["base_again"]
    assert g["cells"]["notapplied"]["ratio"] == pytest.approx(0.0, abs=0.2)
    assert g["settings"]["max_lora_rank"] == 16 and g["settings"]["max_loras"] == 3
    kw = FakeLLM.last_kwargs
    assert (
        kw["enable_lora"]
        and kw["max_lora_rank"] == 16
        and kw["max_loras"] == 3
        and kw["enforce_eager"] is True
    )
    assert (
        kw["enable_prefix_caching"] is False
        and kw["generation_config"] == "vllm"
        and kw["dtype"] == "bfloat16"
    )
    longest = max(
        len(e["prompt"]) + len(e["resp"]) for e in LG.load_cell(tmp_path / "batch" / "applied")
    )
    assert kw["max_model_len"] >= longest + 16


def test_gate_all_pass_exit_zero_and_base_drift(tmp_path, monkeypatch):
    deltas = {"a": (-0.05, 0.1), "b": (-0.04, 0.1)}
    rc, g = run_gate(tmp_path, monkeypatch, ["a", "b"], ["a", "b"], deltas)
    assert rc == 0 and set(g["verdicts"].values()) == {"PASS"}
    rc, g = run_gate(tmp_path, monkeypatch, ["a", "b"], ["a", "b"], deltas, drift=0.02)
    assert rc == 1 and set(g["verdicts"].values()) == {"FAIL"} and not g["cells"]["a"]["base_again"]


def test_gate_base_scored_once_per_distinct_example(tmp_path, monkeypatch):
    install_fake_vllm(monkeypatch)
    rng = np.random.default_rng(1)
    root = tmp_path / "batch"
    make_cell(root, "a", rng)
    shutil.copytree(root / "a", root / "b")  # identical examples in two cells
    FakeLLM.world = load_world(root, ["a"])
    FakeLLM.applied = {"a", "b"}
    seen = []
    orig = FakeLLM.generate

    def spy(self, prompts, sp, lora_request=None, use_tqdm=False):
        seen.append((lora_request is None, len(prompts)))
        return orig(self, prompts, sp, lora_request, use_tqdm)

    monkeypatch.setattr(FakeLLM, "generate", spy)
    LG.main(["--model", "m", "--cells-dir", str(root), "--out", str(tmp_path / "g.json")])
    assert [n for is_base, n in seen if is_base] == [
        2,
        2,
    ]  # 2 distinct examples, scored before and after the adapters
    assert [n for is_base, n in seen if not is_base] == [2, 2]


def test_judge_edge_cases():
    r = LG.judge(np.zeros(5), np.zeros(5), np.zeros(5), 10.0, 10.0, 10.0)
    assert r["verdict"] == "UNINFORMATIVE" and np.isnan(r["corr"])


# ------------------------------------------------------------------------------------------- multi_sample
class FakeTok:
    pad_token_id = None

    def apply_chat_template(
        self, msgs, add_generation_prompt, tokenize, return_dict, chat_template
    ):
        return [ord(ch) % 200 + 1 for ch in msgs[-1]["content"]][:12]

    def decode(self, ids, skip_special_tokens=True):
        return "<think>x</think> " + " ".join(map(str, ids))


def make_problems(tmp_path):
    pr = tmp_path / "problems.jsonl"
    pr.write_text(
        "".join(
            json.dumps({"pid": p, "prompt": f"say {p} please", "task": "ifbench"}) + "\n"
            for p in ("p1", "p2", "p3")
        )
    )
    pf = tmp_path / "pids.txt"
    pf.write_text("p1:8\np2:8\np3:4\n")
    return pr, pf


@pytest.mark.skipif(not (DAN / "olmo_ifladder" / "sample.py").exists(), reason="Dan's clone absent")
def test_multi_sample_stub(tmp_path, monkeypatch):
    install_fake_vllm(monkeypatch)
    monkeypatch.setattr(MSP, "load_tokenizer", lambda p: FakeTok())
    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", lambda *a, **k: FakeTok())
    rng = np.random.default_rng(2)
    root = tmp_path / "batch"
    for c in ("c1", "c2", "c3"):
        make_cell(root, c, rng)
    (root / "c3" / "adapter" / "adapter_config.json").write_text(
        json.dumps({"peft_type": "LORA", "r": 1, "target_modules": ["o_proj", "q_proj"]})
    )
    pr, pf = make_problems(tmp_path)
    gate = tmp_path / "gate.json"
    gate.write_text(json.dumps({"verdicts": {"c1": "PASS", "c2": "FAIL", "c3": "UNINFORMATIVE"}}))
    # resume state of c1: p1 chunk 0 complete (marked), p1 chunk 1 partial (2 of 4 rows)
    (tmp_path / "out" / "c1").mkdir(parents=True)
    with gzip.open(tmp_path / "out" / "c1" / "samples.jsonl.gz", "wt") as f:
        for i in range(4):
            f.write(json.dumps({"pid": "p1", "chunk": 0, "i": i, "text": "OLD"}) + "\n")
        for i in range(2):
            f.write(json.dumps({"pid": "p1", "chunk": 1, "i": i, "text": "PARTIAL"}) + "\n")
    common = [
        "--model",
        "m",
        "--tokenizer",
        "t",
        "--name",
        "dpo",
        "--rev",
        "main",
        "--problems",
        str(pr),
        "--pids",
        f"@{pf}",
        "--n",
        "0",
        "--chunk",
        "4",
        "--seed",
        "7",
        "--job",
        "L_test",
    ]
    rc = MSP.main(
        [
            "--dan",
            str(DAN),
            *common,
            "--cells-dir",
            str(root),
            "--gate",
            str(gate),
            "--out",
            str(tmp_path / "out"),
        ]
    )
    assert rc == 0
    out = tmp_path / "out"
    assert not (out / "c2").exists() and json.loads((out / "skipped_cells.json").read_text()) == {
        "c2": "FAIL"
    }
    kw, eng = FakeLLM.last_kwargs, FakeLLM.last_engine
    assert (
        kw["enable_lora"]
        and kw["max_lora_rank"] == 16
        and kw["max_loras"] == 2
        and kw["max_cpu_loras"] == 2
    )
    # requests: right id format, routed to their own adapter, interleaved round-robin over the cells
    assert len(eng.added) == 5 + 5 - 1  # c1: 5 requests left after resume (p1 chunk 0 done), c3: 5
    for rid, lname in eng.added:
        assert rid.split("|")[0] == lname
    first = [lname for _, lname in eng.added[:8]]
    assert (
        first[0] != first[1] and first[:2] == first[2:4]
    )  # c1 c3 c1 c3 ... until c1's remaining requests run out
    # resumed cell: old chunk kept, partial chunk re-run whole, nothing else duplicated
    rows1 = [json.loads(ln) for ln in gzip.open(out / "c1" / "samples.jsonl.gz", "rt")]
    assert sum(r["text"] == "OLD" for r in rows1) == 4 and not any(
        r["text"] == "PARTIAL" for r in rows1
    )
    keyset = [(r["pid"], r["chunk"], r["i"]) for r in rows1]
    assert len(keyset) == len(set(keyset)) == 8 + 8 + 4  # p1 8, p2 8, p3 4 draws
    rows3 = [json.loads(ln) for ln in gzip.open(out / "c3" / "samples.jsonl.gz", "rt")]
    assert len(rows3) == 20 and {r["model"] for r in rows3} == {"c3"}
    assert all(r["ids"] and r["n_tok"] == 3 for r in rows3)
    # each row was served by its own adapter (token 1000 * lora id): c1 -> id 1, c3 -> id 2
    ids3 = {
        int(np.frombuffer(__import__("base64").b64decode(r["ids"]), dtype=np.int32)[0])
        for r in rows3
    }
    ids1_new = {
        int(np.frombuffer(__import__("base64").b64decode(r["ids"]), dtype=np.int32)[0])
        for r in rows1
        if r["text"] != "OLD"
    }
    assert ids3 == {1000 * 2 + i for i in range(4)} and ids1_new == {1000 * 1 + i for i in range(4)}
    # meta: Dan's fields plus ours; prompt hashes cover every pid
    meta = json.loads((out / "c3" / "meta.json").read_text().splitlines()[-1])
    assert (
        meta["cells_in_engine"] == ["c1", "c3"]
        and meta["adapter_r"] == 1
        and meta["adapter_targets"] == ["o_proj", "q_proj"]
    )
    assert len(meta["lora_sha256"]) == 64 and meta["lora_name"] == "c3" and meta["lora_rank"] == 16
    assert (
        set(meta["prompt_sha256_by_pid"]) == {"p1", "p2", "p3"}
        and meta["samples"] == 20
        and meta["vllm"] == "0.11.0-fake"
    )

    # same stub engine through HIS main (no LoRA): row schema, meta fields and LLM arguments must agree
    dan = MSP.load_dan(str(DAN))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sample.py",
            *common[:12],
            "--pids",
            f"@{pf}",
            "--n",
            "0",
            "--chunk",
            "4",
            "--seed",
            "7",
            "--job",
            "L_test",
            "--out",
            str(tmp_path / "dan.jsonl.gz"),
            "--meta",
            str(tmp_path / "dan_meta.json"),
            "--order",
            "given",
        ],
    )
    # his CLI has no --cells-dir etc.; rebuild argv from scratch
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sample.py",
            "--model",
            "m",
            "--tokenizer",
            "t",
            "--name",
            "dpo",
            "--rev",
            "main",
            "--problems",
            str(pr),
            "--pids",
            f"@{pf}",
            "--n",
            "0",
            "--chunk",
            "4",
            "--seed",
            "7",
            "--job",
            "L_test",
            "--out",
            str(tmp_path / "dan.jsonl.gz"),
            "--meta",
            str(tmp_path / "dan_meta.json"),
            "--order",
            "given",
        ],
    )
    dan.main()
    dan_kw = FakeLLM.last_kwargs
    drow = json.loads(next(iter(gzip.open(tmp_path / "dan.jsonl.gz", "rt"))))
    assert list(drow) == list(rows3[0]), "row schema (keys and order) differs from Dan's"
    for k, v in dan_kw.items():
        assert kw[k] == v or k == "max_model_len", f"LLM argument {k} differs"
    assert kw["max_model_len"] == dan_kw["max_model_len"]
    dmeta = json.loads((tmp_path / "dan_meta.json").read_text().splitlines()[-1])
    assert set(dmeta) <= set(meta), set(dmeta) - set(meta)
    assert (
        dmeta["prompt_sha256_by_pid"] == meta["prompt_sha256_by_pid"]
        and dmeta["template_sha256"] == meta["template_sha256"]
    )
    assert (
        dmeta["max_tokens_by_pid"] == meta["max_tokens_by_pid"]
        and dmeta["prompt_tokens_by_pid"] == meta["prompt_tokens_by_pid"]
    )
    # second run: nothing left to do
    assert (
        MSP.main(
            [
                "--dan",
                str(DAN),
                *common,
                "--cells-dir",
                str(root),
                "--gate",
                str(gate),
                "--out",
                str(out),
            ]
        )
        == 0
    )


@pytest.mark.skipif(
    not (OLD_L / "meta.json").exists() or not (TOK / "tokenizer.json").exists(),
    reason="old L_dpo meta or tokenizer absent",
)
def test_prompt_sha256_equals_old_dpo_meta():
    from transformers import AutoTokenizer

    old = json.loads((OLD_L / "meta.json").read_text().splitlines()[0])["prompt_sha256_by_pid"]
    dan = MSP.load_dan(str(DAN))
    probs = dan.load_problems(str(DAN / "olmo_ifomega" / "problems.jsonl"))
    want = dan.parse_spec("@" + str(OE / "stage2" / "pids_L.txt"), 0)
    ids, sha = MSP.prompt_hashes(dan, AutoTokenizer.from_pretrained(str(TOK)), probs, list(want))
    assert set(sha) == set(old) and len(sha) == 48
    assert sha == old
    old_len = json.loads((OLD_L / "meta.json").read_text().splitlines()[0])["prompt_tokens_by_pid"]
    assert {p: len(x) for p, x in ids.items()} == old_len


# ------------------------------------------------------------------------------------------- chain
FAKEPY = """#!{py}
import gzip, json, os, sys
a = sys.argv[1:]
if a and a[0] == "-u":
    a = a[1:]
if not a or a[0] == "-":
    print(json.dumps({{"ok": True}})); sys.exit(0)
script = os.path.basename(a[0])
def opt(k):
    return a[a.index(k) + 1]
open(os.environ["STUB_LOG"], "a").write(script + " " + " ".join(a[1:]) + "\\n")
if script == "mini_status.py":
    os.execv("{py}", ["{py}", *a])
elif script == "mini_train.py":
    for c in ("c1", "c2"):
        d = os.path.join(opt("--out"), "cells", c)
        os.makedirs(d, exist_ok=True)
        open(os.path.join(d, "DONE"), "w").write("ok")
elif script == "lora_gate.py":
    cells = sorted(os.listdir(opt("--cells-dir")))
    cells = [c for c in cells if os.path.isdir(os.path.join(opt("--cells-dir"), c))]
    v = {{c: ("FAIL" if c == "c2" else "PASS") for c in cells}}
    os.makedirs(os.path.dirname(opt("--out")), exist_ok=True)
    json.dump({{"verdicts": v, "n_fail": sum(x == "FAIL" for x in v.values())}}, open(opt("--out"), "w"))
    sys.exit(1 if "FAIL" in v.values() else 0)
elif script == "multi_sample.py":
    v = json.load(open(opt("--gate")))["verdicts"]
    for c, x in v.items():
        if x == "FAIL":
            continue
        os.makedirs(os.path.join(opt("--out"), c), exist_ok=True)
        with gzip.open(os.path.join(opt("--out"), c, "samples.jsonl.gz"), "wt") as f:
            for i in range(8):
                f.write(json.dumps({{"pid": "p1", "chunk": i // 4, "i": i % 4, "text": "x"}}) + "\\n")
elif script == "sample.py":
    assert "--lora" not in a and "--n" in a and opt("--chunk") == "4" and opt("--order") == "given" and opt("--tp") == "1"
    os.makedirs(os.path.dirname(opt("--out")), exist_ok=True)
    with gzip.open(opt("--out"), "wt") as f:
        for i in range(16):
            f.write(json.dumps({{"pid": "p1", "chunk": i // 4, "i": i % 4, "text": "x"}}) + "\\n")
    open(opt("--meta"), "w").write("{{}}\\n")
elif script == "grade.py":
    with open(a[3], "w") as f:
        f.write(json.dumps({{"pid": "p1", "strict": True}}) + "\\n")
"""

CSV_HEAD = "name,repo,revision,step,commit,n_shards,weight_bytes,weights_digest\n"


def chain_env(tmp_path):
    mini, dan, models = tmp_path / "mini", tmp_path / "dan", tmp_path / "models"
    for d in (
        mini / "meta",
        mini / "stage2",
        mini / "plans",
        mini / "pools",
        dan / "olmo_ifcurve",
        dan / "pod" / "olmo_ifomega",
        dan / "olmo_ifomega",
        models,
    ):
        d.mkdir(parents=True, exist_ok=True)
    (mini / "pod").symlink_to(HERE / "pod")
    (mini / "meta" / "gpu_status").write_text("GPU_OK 1\n")
    (mini / "stage2" / "pids_L.txt").write_text("p1:8\np2:8\n")
    (mini / "stage2" / "seeds_L.csv").write_text("ckpt,step,seed_L\ndpo,0,640000\n")
    (mini / "plans" / "p.json").write_text("{}")
    (mini / "stage").mkdir(exist_ok=True)
    (mini / "stage" / "problems_ans.jsonl").write_text("{}\n")
    (mini / "stage" / "pids_ans.txt").write_text("p1:8\np2:8\n")
    (mini / "stage" / "seed_ans.txt").write_text("650000\n")
    (dan / "olmo_ifcurve" / "checkpoints.csv").write_text(
        CSV_HEAD + "dpo,org/repo,main,0,abc123,1,10,dg\n"
    )
    (dan / "olmo_ifomega" / "problems.jsonl").write_text("{}\n")
    fetch = dan / "pod" / "olmo_ifomega" / "fetch.sh"
    fetch.write_text(
        '#!/usr/bin/env bash\nD=$MODELS_ROOT/$3; mkdir -p $D; head -c 10 /dev/zero > $D/model-00001.safetensors; echo "$2" > $D/.sha; touch $D/.complete\n'
    )
    py = tmp_path / "fakepy"
    py.write_text(FAKEPY.format(py=sys.executable))
    py.chmod(py.stat().st_mode | stat.S_IEXEC)
    rng = np.random.default_rng(3)
    cells = tmp_path / "runs" / "dpo" / "train" / "cells"
    for c in ("c1", "c2"):
        make_cell(cells, c, rng)
    SC.stage(tmp_path / "runs", "dpo", ["c1", "c2"], mini / "adapters" / "b1")
    env = {
        **os.environ,
        "MINI_ROOT": str(mini),
        "DAN_ROOT": str(dan),
        "MODELS_ROOT": str(models),
        "PY": str(py),
        "STUB_LOG": str(tmp_path / "stub.log"),
        "PLAN": "plans/p.json",
        "TRAIN_ARGS": "--foo 1",
        "HB_SECONDS": "1",
        "PATH": f"{tmp_path / 'bin'}:{os.environ['PATH']}",
    }
    (tmp_path / "bin").mkdir(exist_ok=True)
    return env, mini, models


def run_chain(env, ck, stages):
    return subprocess.run(
        ["bash", str(HERE / "pod" / "mini_chain.sh"), ck, stages],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )


@pytest.mark.skipif(shutil.which("flock") is None, reason="flock missing")
def test_chain_simulation(tmp_path):
    env, mini, models = chain_env(tmp_path)
    p = run_chain(env, "dpo", "download,train,gate:b1,sample:b1")
    assert p.returncode == 0, p.stdout + p.stderr
    st = mini / "runs" / "dpo" / "state"
    for s in ("download", "train", "gate_b1", "sample_b1"):
        assert (st / f"{s}.DONE").exists() and not (st / f"{s}.RUNNING").exists(), s
    assert (
        "fail=1" in (st / "gate_b1.DONE").read_text()
        and "fail_cells=c2" in (st / "gate_b1.DONE").read_text()
    )
    assert (models / "dpo" / ".complete").exists(), "weights are kept by default"
    assert json.loads((mini / "runs" / "dpo" / "download_verify.json").read_text())["ok"]
    out = mini / "runs" / "dpo" / "sample" / "b1"
    assert (out / "c1" / "samples.jsonl.gz").exists() and (out / "c1" / "graded.jsonl").exists()
    assert not (out / "c2").exists(), "a cell that FAILed the gate is not sampled"
    assert (st / "sample_b1.expected").read_text().strip() == "16"  # 1 cell x (8 + 8)
    assert (st / "chain.DONE").exists()
    log = Path(env["STUB_LOG"]).read_text()
    assert (
        "mini_train.py run --model" in log
        and "--plan" in log
        and "--foo 1" in log
        and "--cells-dir" in log
    )
    assert "--gate" in log and "--seed 640000" in log and "--chunk 4" in log
    # idempotent rerun: everything skipped
    p2 = run_chain(env, "dpo", "download,train,gate:b1,sample:b1")
    assert (
        p2.returncode == 0
        and "skip train (DONE)" in p2.stdout
        and "download already done" in p2.stdout
    )
    # status lines
    lines = MS.status_lines(mini / "runs", None, models, dan=Path(env["DAN_ROOT"]))
    d = MS.parse("\n".join(lines))["dpo"]
    assert d["stages"]["train"]["state"] == "DONE" and d["stages"]["gate:b1"]["state"] == "DONE"
    assert (
        d["stages"]["sample:b1"]["progress"].startswith("8/16")
        and "graded=1" in d["stages"]["sample:b1"]["progress"]
    )
    assert (
        d["chain"]["done"] == 1
        and d["chain"]["weights"] == "present"
        and set(d["cells"]) == {"c1", "c2"}
    )
    assert MS.verdict("\n".join(lines), "dpo", ["train", "gate:b1", "sample:b1"]).startswith(
        "FINISHED"
    )


@pytest.mark.skipif(shutil.which("flock") is None, reason="flock missing")
def test_chain_failures_and_delete_weights(tmp_path):
    env, mini, models = chain_env(tmp_path)
    # a sample stage without a gate result fails, the chain goes on; weights deleted on request
    p = run_chain({**env, "DELETE_WEIGHTS": "1"}, "dpo", "download,sample:b1,gate:nope")
    st = mini / "runs" / "dpo" / "state"
    assert p.returncode == 0, p.stdout + p.stderr
    assert (st / "sample_b1.FAILED").exists() and "no gate result" in (
        st / "sample_b1.FAILED"
    ).read_text()
    assert (st / "gate_nope.FAILED").exists() and "no staged batch" in (
        st / "gate_nope.FAILED"
    ).read_text()
    assert not (models / "dpo").exists() and (st / "weights_deleted").exists()
    assert "failed_stages=2" in (st / "chain.DONE").read_text()
    # unknown stage and unknown checkpoint
    assert run_chain(env, "dpo", "bogus").returncode == 2
    assert run_chain(env, "nope", "download").returncode == 2
    # a failed download stops the chain: other stages FAILED
    env2, mini2, _ = chain_env(tmp_path / "second")
    (tmp_path / "second" / "dan" / "pod" / "olmo_ifomega" / "fetch.sh").write_text("exit 1\n")
    p = run_chain(env2, "dpo", "download,train")
    assert p.returncode == 1 and (mini2 / "runs" / "dpo" / "state" / "train.FAILED").exists()


@pytest.mark.skipif(shutil.which("flock") is None, reason="flock missing")
def test_chain_lock(tmp_path):
    env, mini, _ = chain_env(tmp_path)
    import fcntl

    fh = open(mini / "gpu.lock", "w")
    fcntl.flock(fh, fcntl.LOCK_EX)
    try:
        p = run_chain(env, "dpo", "download")
        assert p.returncode == 3 and "LOCKED" in p.stdout
    finally:
        fh.close()


@pytest.mark.skipif(shutil.which("flock") is None, reason="flock missing")
def test_chain_shared_pod_guards(tmp_path):
    """Amendment 9: on a pod shared with another job the chain can be told to hold that job's GPU lock too and to
    refuse a GPU that is in use. Both are off unless set."""
    import fcntl

    env, mini, _ = chain_env(tmp_path)
    smi = tmp_path / "bin" / "nvidia-smi"
    smi.write_text('#!/bin/bash\ncat "$SMI_USED"\n')
    smi.chmod(smi.stat().st_mode | stat.S_IEXEC)
    used = tmp_path / "used"
    used.write_text("131000\n")
    env = {**env, "SMI_USED": str(used)}
    st = mini / "runs" / "dpo" / "state"
    # off by default: a busy GPU does not stop a chain that was not asked to look
    assert run_chain(env, "dpo", "download").returncode == 0
    p = run_chain({**env, "GPU_FREE_MIB": "2000"}, "dpo", "download,train")
    assert (
        p.returncode == 3
        and "GPU_BUSY: 131000 MiB" in p.stdout
        and not (st / "train.DONE").exists()
    )
    used.write_text("")  # no answer is not "free"
    assert run_chain({**env, "GPU_FREE_MIB": "2000"}, "dpo", "download,train").returncode == 3
    used.write_text("412\n")
    other = tmp_path / "wi" / "gpu.lock"
    # a lock file that does not exist is not created
    p = run_chain({**env, "GPU_FREE_MIB": "2000", "ALSO_LOCK": str(other)}, "dpo", "download,train")
    assert p.returncode == 0 and (st / "train.DONE").exists() and not other.parent.exists()
    other.parent.mkdir()
    other.write_text("")
    before = other.stat().st_mtime_ns
    fh = open(other, "w")
    fcntl.flock(fh, fcntl.LOCK_EX)
    try:
        p = run_chain({**env, "ALSO_LOCK": str(other)}, "dpo", "download,gate:b1")
        assert p.returncode == 3 and f"another job holds {other}" in p.stdout
        assert not (st / "gate_b1.DONE").exists()
    finally:
        fh.close()
    p = run_chain({**env, "ALSO_LOCK": str(other)}, "dpo", "download,gate:b1")
    assert p.returncode == 0 and (st / "gate_b1.DONE").exists()
    assert other.read_text() == "" and other.stat().st_mtime_ns >= before


@pytest.mark.skipif(shutil.which("flock") is None, reason="flock missing")
def test_chain_holds_the_other_jobs_lock_while_it_runs(tmp_path):
    import fcntl

    env, mini, _ = chain_env(tmp_path)
    other = tmp_path / "gpu.lock"
    other.write_text("")
    fetch = tmp_path / "dan" / "pod" / "olmo_ifomega" / "fetch.sh"
    fetch.write_text(
        "#!/usr/bin/env bash\nfor _ in $(seq 100); do [ -f $MODELS_ROOT/go ] && break; sleep 0.1; done\n"
        + fetch.read_text().split("\n", 1)[1]
    )
    p = subprocess.Popen(
        ["bash", str(HERE / "pod" / "mini_chain.sh"), "dpo", "download"],
        env={**env, "ALSO_LOCK": str(other)},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        held = False
        for _ in range(100):  # the chain takes the lock within a moment of starting
            with open(other) as fh:
                try:
                    fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    held = True
                    break
            time.sleep(0.05)
        assert held, "the chain did not take the other job's lock"
    finally:
        (tmp_path / "models" / "go").touch()
        out, _ = p.communicate(timeout=60)
    assert p.returncode == 0, out
    with open(other) as fh:  # released when the chain ends
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)


def test_mini_pod_dry_run(tmp_path):
    env = {
        **os.environ,
        "V2": str(tmp_path / "v2"),
        "OE_DATA": str(tmp_path / "oe"),
        "PATH": os.environ["PATH"],
    }
    (tmp_path / "oe" / "stage2").mkdir(parents=True)
    (tmp_path / "oe" / "stage2" / "pids_L.txt").write_text("p1:8\n")
    (tmp_path / "oe" / "stage2" / "seeds_L.csv").write_text("ckpt,step,seed_L\ndpo,0,1\n")
    (tmp_path / "v2" / "pools").mkdir(parents=True)
    (tmp_path / "v2" / "pools" / "x.jsonl").write_text("{}\n")
    plan = tmp_path / "plan.json"
    plan.write_text("{}")
    batch = tmp_path / "b1"
    batch.mkdir()
    (batch / "cells_manifest.json").write_text("{}")
    sh = str(HERE / "pod" / "mini_pod.sh")

    def go(*a):
        return subprocess.run(
            ["bash", sh, "--dry-run", *a], env=env, capture_output=True, text=True, timeout=60
        )

    p = go("train", "rp-x", "dpo", "--plan", str(plan), "--train-args", "--max-steps 64")
    assert p.returncode == 0, p.stdout + p.stderr
    assert (
        "mini_chain.sh dpo download,train" in p.stdout
        and "PLAN=plans/plan.json" in p.stdout
        and "rsync" in p.stdout
    )
    p = go("sample", "rp-x", "dpo", str(batch))
    assert (
        p.returncode == 0
        and "mini_chain.sh dpo download,gate:b1,sample:b1" in p.stdout
        and "adapters/b1/" in p.stdout
    )
    st = tmp_path / "v2" / "pod_stage" / "rp-x"
    assert (
        (st / "olmo_edl.py").exists()
        and (st / "olmo_template.py").exists()
        and (st / "stage2" / "pids_L.txt").exists()
    )
    assert (st / "pod" / "mini_chain.sh").exists() and not list(st.glob("test_*"))
    for c in (
        ["status", "rp-x"],
        ["fetch", "rp-x", "dpo"],
        ["stop", "rp-x", "dpo"],
        ["setup", "rp-x"],
    ):
        q = go(*c)
        assert q.returncode == 0, (c, q.stdout + q.stderr)
    assert "mini_status.py" in go("status", "rp-x").stdout
    assert "SIGTERM" in go("stop", "rp-x").stdout and "pkill" not in go("stop", "rp-x").stdout
    assert (
        go("status", "bad").returncode == 2
        and go("train", "rp-x", "nope", "--plan", str(plan)).returncode == 2
    )
    assert (
        go("train", "rp-x", "dpo", "--plan", str(plan), "--train-args", "$(rm -rf x)").returncode
        == 2
    )


def test_status_helpers(tmp_path):
    gj = tmp_path / "g.json"
    gj.write_text(json.dumps({"verdicts": {"a": "PASS", "b": "FAIL", "c": "UNINFORMATIVE"}}))
    pf = tmp_path / "pids.txt"
    pf.write_text("x:8\ny:8\n")
    assert MS.gate_summary_line(gj) == "pass=1 uninformative=1 fail=1 fail_cells=b"
    assert MS.expected_draws(gj, pf) == 2 * 16
    cs = tmp_path / "runs" / "dpo" / "train" / "cells" / "run1"
    cs.mkdir(parents=True)
    (cs / "train.log").write_text("step 3 of 64 loss 1.2\n")
    (tmp_path / "runs" / "dpo" / "state").mkdir()
    (tmp_path / "runs" / "dpo" / "state" / "train.RUNNING").write_text("x\n")
    (tmp_path / "runs" / "dpo" / "state" / "chain.pid").write_text(
        f"{os.getpid()} {os.getpid()} t download,train\n"
    )
    d = MS.parse("\n".join(MS.status_lines(tmp_path / "runs", None, tmp_path / "m", dan=tmp_path)))[
        "dpo"
    ]
    assert (
        d["stages"]["train"]["state"] == "RUNNING"
        and d["stages"]["train"]["progress"] == "cells_done=0/1"
    )
    assert d["cells"]["run1"]["state"] == "RUNNING" and d["cells"]["run1"]["last"].startswith(
        "step_3_of_64"
    )


# ------------------------------------------------------------------------------------------- plain stage
@pytest.mark.skipif(shutil.which("flock") is None, reason="flock missing")
def test_chain_plain_stage(tmp_path):
    env, mini, models = chain_env(tmp_path)
    p = run_chain(env, "dpo", "download,plain:ans")
    assert p.returncode == 0, p.stdout + p.stderr
    st = mini / "runs" / "dpo" / "state"
    assert (st / "plain_ans.DONE").exists() and (
        st / "plain_ans.expected"
    ).read_text().strip() == "16"
    o = mini / "runs" / "dpo" / "plain_ans"
    assert (
        (o / "samples.jsonl.gz").exists()
        and (o / "graded.jsonl").exists()
        and (o / "meta.json").exists()
    )
    log = Path(env["STUB_LOG"]).read_text()
    assert (
        "sample.py --model" in log
        and "problems_ans.jsonl" in log
        and "--seed 650000" in log
        and "--job plain_ans_dpo" in log
    )
    grade = [ln for ln in log.splitlines() if ln.startswith("grade.py")][-1]
    assert "problems_ans.jsonl" in grade  # graded against the same problems file
    d = MS.parse(
        "\n".join(MS.status_lines(mini / "runs", None, models, dan=Path(env["DAN_ROOT"])))
    )["dpo"]
    assert d["stages"]["plain:ans"]["state"] == "DONE" and d["stages"]["plain:ans"][
        "progress"
    ].startswith("16/16")
    # missing inputs: the stage fails, the chain goes on
    (mini / "stage" / "seed_ans.txt").unlink()
    (mini / "runs" / "dpo" / "state" / "plain_ans.DONE").unlink()
    assert run_chain(env, "dpo", "plain:ans").returncode == 0
    assert "missing" in (st / "plain_ans.FAILED").read_text()
    assert (
        run_chain(env, "dpo", "plain:bad_name").returncode == 2
    )  # underscore not allowed in a plain name


def test_mini_pod_plain_dry_run(tmp_path):
    env = {**os.environ, "V2": str(tmp_path / "v2"), "OE_DATA": str(tmp_path / "oe")}
    (tmp_path / "oe" / "stage2").mkdir(parents=True)
    (tmp_path / "oe" / "stage2" / "seeds_L.csv").write_text("ckpt,step,seed_L\ndpo,0,1\n")
    prob, pids = tmp_path / "pr.jsonl", tmp_path / "pids.txt"
    prob.write_text("{}\n")
    pids.write_text("p1:8\n")
    sh = str(HERE / "pod" / "mini_pod.sh")

    def go(*a):
        return subprocess.run(
            ["bash", sh, "--dry-run", *a], env=env, capture_output=True, text=True, timeout=60
        )

    p = go(
        "plain",
        "rp-x",
        "dpo",
        "answeronly",
        "--problems",
        str(prob),
        "--pids",
        str(pids),
        "--seed",
        "650000",
    )
    assert p.returncode == 0, p.stdout + p.stderr
    assert "mini_chain.sh dpo download,plain:answeronly" in p.stdout
    st = tmp_path / "v2" / "pod_stage" / "rp-x" / "stage"
    assert (st / "problems_answeronly.jsonl").exists() and (st / "pids_answeronly.txt").exists()
    assert (st / "seed_answeronly.txt").read_text().strip() == "650000"
    assert (
        go(
            "plain", "rp-x", "dpo", "answeronly", "--problems", str(prob), "--pids", str(pids)
        ).returncode
        == 2
    )
    assert (
        go(
            "plain",
            "rp-x",
            "dpo",
            "bad_name",
            "--problems",
            str(prob),
            "--pids",
            str(pids),
            "--seed",
            "1",
        ).returncode
        == 2
    )


# ------------------------------------------------------------------------------------------- prompt control
@pytest.mark.skipif(
    not (DAN / "olmo_answeronly").parent.exists()
    or not (DAN / "artifacts" / "olmo_answeronly" / "requests" / "s2300_reach.jsonl.gz").exists(),
    reason="Dan's artifacts absent",
)
def test_prompt_control_inputs(tmp_path):
    import make_prompt_control as PC

    rows, lines = PC.build()
    assert len(rows) == 48 and lines == (OE / "stage2" / "pids_L.txt").read_text().split()
    # byte-for-byte the wording registered in his A1 request rows
    assert (
        PC.registered_suffix()
        == "\n\nAfter thinking, reply with only your final answer, exactly as drafted, with no introduction or explanation."
    )
    dan = {
        json.loads(ln)["pid"]: json.loads(ln)
        for ln in open(DAN / "olmo_ifomega" / "problems.jsonl")
    }
    for r in rows:
        d = dict(dan[r["pid"]])
        assert r["prompt_sample"] == d["prompt"] + PC.registered_suffix()
        assert {
            k: v for k, v in r.items() if k != "prompt_sample"
        } == d  # every other field equals Dan's row
    # his sampler would send prompt_sample as the user turn
    sam = MSP.load_dan(str(DAN))
    assert sam.prompt_ids  # (the choice is probs[pid].get("messages_sample") or probs[pid].get("prompt_sample", prompt))
    out = tmp_path / "o"
    import sys as _s

    old = _s.argv
    _s.argv = ["x", "--out-dir", str(out)]
    try:
        PC.main()
    finally:
        _s.argv = old
    assert len((out / "problems_answeronly.jsonl").read_text().splitlines()) == 48
    assert (out / "pids_answeronly.txt").read_text().split() == lines


# ------------------------------------------------------------------------------------------- analysis
def synth(rate_by_pid, draws=8, seed=0):
    rng = np.random.default_rng(seed)
    return {p: [(int(rng.random() < r), 5) for _ in range(draws)] for p, r in rate_by_pid.items()}


def test_analyze_cell_synthetic():
    pids = [f"p{i}" for i in range(48)]
    final = synth({p: 0.7 for p in pids}, 32, 1)
    ckpt = synth({p: 0.0 for p in pids}, 16, 2)
    hi = synth({p: 0.9 for p in pids}, 8, 3)
    lo = synth({p: 0.2 for p in pids}, 8, 4)
    near = synth({p: 0.5 for p in pids}, 8, 5)
    a = AC.analyze_cell(hi, final, ckpt, 48, reps=500)
    assert (
        a["reaches"]
        and a["reaches_interval"]
        and a["reaches_strict"]
        and a["g"] > 1.0
        and a["diff_final"] > 0
    )
    b = AC.analyze_cell(lo, final, ckpt, 48, reps=500)
    assert not b["reaches"] and not b["reaches_strict"] and 0.1 < b["gap_share"] < 0.4
    assert b["g_lo"] < b["g"] < b["g_hi"] and b["g"] == pytest.approx(b["gap_share"])
    c = AC.analyze_cell(near, final, ckpt, 48, reps=500)
    assert (
        c["diff_final_lo"] < c["diff_final"] < c["diff_final_hi"]
        and c["final_lo95"] < c["final_pct"] < c["final_hi95"]
    )
    assert c["rate2_lo"] < c["strict_pct"] < c["rate2_hi"]
    assert a["strict_pct"] == pytest.approx(
        100 * np.mean([s for v in hi.values() for s, _ in v]), abs=0.01
    )
    assert a["ckpt_pct"] == 0.0 and a["solved_pct"] <= 100
    assert AC.analyze_cell({}, final, ckpt, 48, reps=100)["reaches"] is None


def test_two_level_interval_is_wider_than_prompt_only():
    pids = [f"p{i}" for i in range(48)]
    cell = synth({p: 0.5 for p in pids}, 8, 7)
    final = synth({p: 0.5 for p in pids}, 32, 8)
    r = AC.analyze_cell(cell, final, synth({p: 0.0 for p in pids}, 16, 9), 48, reps=4000)
    assert (
        r["rate2_hi"] - r["rate2_lo"] > r["hi95"] - r["lo95"] - 0.5
    )  # draw noise adds to prompt noise


def test_contrast_and_resolve():
    pids = [f"p{i}" for i in range(48)]
    a = synth({p: 0.8 for p in pids}, 8, 1)
    b = synth({p: 0.3 for p in pids}, 8, 2)
    d = AC.contrast(a, b, 2000)
    assert d["diff"] > 30 and d["lo"] > 0 and d["excludes_zero"] and d["n_prompts"] == 48
    same = AC.contrast(a, a, 2000)
    assert same["diff"] == 0 and same["lo"] < 0 < same["hi"] and not same["excludes_zero"]
    ids = ["dpo__own__A16__n64", "dpo__unsel__A16__n64", "dpo__final__A16__n64", "plain:answeronly"]
    assert (
        AC.resolve("unsel:A16:64", ids) == "dpo__unsel__A16__n64"
        and AC.resolve("own:A16:64", ids) == "dpo__own__A16__n64"
    )
    assert (
        AC.resolve("own:O1:64", ids) is None
        and AC.resolve("plain:answeronly", ids) == "plain:answeronly"
    )


def test_think_tokens_and_coverage():
    import base64

    ids = np.array([5, 6, 524, 27963, 1363, 7, 524, 27963, 397, 8], dtype=np.int32)
    assert AC.think_tokens(base64.b64encode(ids.tobytes()).decode()) == 6  # the last marker
    assert (
        AC.think_tokens(base64.b64encode(np.array([1, 2, 3], dtype=np.int32).tobytes()).decode())
        is None
    )
    cells = {
        "a": [(1, 1)] * 4,
        "b": [(0, 1)] * 4,
        "c": [(1, 1)] * 2 + [(0, 1)] * 2,
        "d": [(0, 1)] * 4,
    }
    rids = {"a": {"x"}, "b": {"x", "y"}, "c": {"y"}, "d": {"x", "y", "z"}}
    r = AC.coverage_split(cells, {"x", "q"}, rids)
    assert (r["n_cov_all"], r["rate_cov_all"]) == (1, 100.0)  # a
    assert (r["n_cov_some"], r["rate_cov_some"]) == (2, 0.0)  # b, d
    assert (r["n_cov_none"], r["rate_cov_none"]) == (1, 50.0)  # c


def write_cell_dir(d: Path, pids, passes, finish="stop", n_tok=100):
    d.mkdir(parents=True)
    with gzip.open(d / "samples.jsonl.gz", "wt") as f:
        for p in pids:
            for i in range(8):
                ids = np.array([1, 2, 524, 27963, 1363, 9], dtype=np.int32)
                f.write(
                    json.dumps(
                        {
                            "pid": p,
                            "chunk": i // 4,
                            "i": i % 4,
                            "n_tok": n_tok,
                            "finish": finish if i == 0 else "stop",
                            "text": "<think>x</think> a b c",
                            "ids": __import__("base64").b64encode(ids.tobytes()).decode(),
                        }
                    )
                    + "\n"
                )
    (d / "graded.jsonl").write_text(
        "".join(
            json.dumps({"pid": p, "chunk": i // 4, "i": i % 4, "strict": i < passes(p)}) + "\n"
            for p in pids
            for i in range(8)
        )
    )


def test_analyze_tree_and_outputs(tmp_path, monkeypatch):
    pids = [f"p{i}" for i in range(6)]
    runs = tmp_path / "runs"
    ck = runs / "dpo"
    own, unsel = "dpo__own__A16__n64", "dpo__unsel__A16__n64"
    write_cell_dir(ck / "sample" / "b1" / own, pids, lambda p: 6, finish="length", n_tok=100)
    write_cell_dir(ck / "sample" / "b1" / unsel, pids, lambda p: 2)
    write_cell_dir(ck / "plain_answeronly", pids, lambda p: 1)
    (ck / "gate").mkdir()
    (ck / "gate" / "b1.json").write_text(
        json.dumps({"verdicts": {own: "PASS", unsel: "UNINFORMATIVE"}})
    )
    pool = tmp_path / "pools" / "dpo" / "own"
    pool.mkdir(parents=True)
    (pool / "train.jsonl").write_text(
        "".join(
            json.dumps(
                {
                    "idx": i,
                    "pid": f"t{i}",
                    "n_tok": 1000 + i,
                    "instruction_id_list": [["a"], ["b"]][i % 2],
                }
            )
            + "\n"
            for i in range(4)
        )
    )
    for cid, src, lr in ((own, "own", 0.003), (unsel, "unsel", 0.001)):
        cd = ck / "train" / "cells" / cid
        cd.mkdir(parents=True)
        (cd / "cell.json").write_text(
            json.dumps(
                {
                    "cell": cid,
                    "family": "lora:r=16",
                    "P": 655360,
                    "n": 2,
                    "source": src,
                    "pool": str(pool),
                    "selected": {"lr": lr, "step": 4},
                    "did_not_train": False,
                    "lora_r": 16,
                    "mdl_nats": 9.0,
                    "mdl_tokens": 2001,
                    "edl_tok_bits_per_trained_weight": 0.5,
                }
            )
        )
    monkeypatch.setattr(
        AC,
        "references",
        lambda c, reach: (
            synth({p: 0.8 for p in reach}, 32, 1),
            synth({p: 0.0 for p in reach}, 16, 2),
        ),
    )
    rids = {"p0": {"a"}, "p1": {"a", "c"}, "p2": {"c"}, "p3": {"b"}, "p4": {"a"}, "p5": {"a", "b"}}
    rows, crows = AC.analyze(runs, None, reps=300, reach=pids, rids=rids, root=tmp_path)
    by = {r["cell"]: r for r in rows}
    assert set(by) == {own, unsel, "plain:answeronly"}
    r = by[own]
    assert (
        r["P"] == 655360
        and r["n"] == 2
        and r["lr"] == 0.003
        and r["gate"] == "PASS"
        and r["complete"] is True
    )
    assert (
        r["strict_pct"] == 75.0
        and r["solved_pct"] == 100.0
        and r["g"] == pytest.approx(75 / 80, abs=0.1)
    )
    assert r["tokens_one_pass"] == 2001 and r["tokens_seen_est"] == 4002 and r["mdl_tokens"] == 2001
    assert (
        r["med_resp_tokens"] == 100
        and r["med_think_tokens"] == 2
        and r["length_cut_share"] == pytest.approx(1 / 8)
    )
    # train ids of the first 2 examples = {a, b}: p0 {a} all, p1 {a,c} some, p2 {c} none, p3 {b} all, p4 all, p5 all
    assert (r["n_cov_all"], r["n_cov_some"], r["n_cov_none"]) == (4, 1, 1)
    p = by["plain:answeronly"]
    assert (
        p["source"] == "plain"
        and p["gate"] is None
        and p["strict_pct"] == pytest.approx(100 * 1 / 8, abs=0.01)
        and p.get("n_cov_all") is None
    )
    names = {c["name"]: c for c in crows}
    assert set(names) == {
        "own:A16:64-unsel:A16:64",
        f"{own}-plain:answeronly",
        f"{unsel}-plain:answeronly",
    }  # "final:A16:64" is absent: skipped
    assert names[f"{own}-plain:answeronly"]["diff"] == pytest.approx(62.5, abs=0.01)
    assert names["own:A16:64-unsel:A16:64"]["diff"] == pytest.approx(50.0, abs=0.01)
    # an explicit list replaces the defaults
    _, crows2 = AC.analyze(
        runs, None, reps=300, reach=pids, rids=rids, root=tmp_path, contrasts=[f"{own}-{unsel}"]
    )
    assert (
        len(crows2) == 1
        and crows2[0]["diff"] == pytest.approx(50.0, abs=0.01)
        and crows2[0]["excludes_zero"]
    )
    AC.write(rows, tmp_path / "cs")
    AC.write_table(crows, AC.CONTRAST_KEYS, tmp_path / "cc")
    j = json.loads((tmp_path / "cs.json").read_text())
    assert {x["cell"] for x in j} == set(by) and "reaches" in (
        tmp_path / "cs.csv"
    ).read_text().splitlines()[0]
    assert (tmp_path / "cc.csv").read_text().splitlines()[0].startswith("ckpt,name,a,b")


@pytest.mark.skipif(
    not (OLD_L / "graded.jsonl").exists()
    or not (DAN / "artifacts" / "olmo_ifcurve" / "rows" / "B_s2300.csv").exists(),
    reason="old L_dpo run or Dan's artifacts absent",
)
def test_reproduces_old_dpo_adapter_cell():
    reach = AC.dc.reach()
    detail = AC.load_detail(OLD_L / "samples.jsonl.gz", OLD_L / "graded.jsonl", reach)
    cells = AC.to_cells(detail)
    assert cells == AC.load_cell_draws(OLD_L / "samples.jsonl.gz", OLD_L / "graded.jsonl", reach)
    final, plain = AC.references("dpo", reach)
    r = AC.analyze_cell(cells, final, plain, 48)
    assert (
        r["strict_pct"] == pytest.approx(63.5, abs=0.06)
        and r["lo95"] == pytest.approx(53.7, abs=0.06)
        and r["hi95"] == pytest.approx(73.2, abs=0.06)
    )
    assert r["n_draws"] == 384 and r["n_prompts"] == 48
    assert (r["final_pct"], r["final_lo95"], r["final_hi95"]) == pytest.approx(
        (69.4, 60.7, 77.7), abs=0.06
    )
    assert r["ckpt_pct"] == 0.0 and r["reaches_interval"] is True and r["reaches_strict"] is False
    assert (
        0.9 < r["g"] < 0.93 and 0.8 <= r["g_lo"] < 0.9 and r["reaches"] is True
    )  # the audit's "about 0.83"
    assert (
        r["rate2_lo"] < r["lo95"] and r["rate2_hi"] > r["hi95"] - 1
    )  # draw noise widens the interval
    ls = AC.length_stats(detail)
    assert ls["length_cut_share"] == 0.0 and ls["med_think_tokens"] <= ls["med_resp_tokens"]


# ------------------------------------------------------------------------------------------- figure
def fake_summary(tmp_path, with_n_cells=True):
    rows = []
    for src, base in (("own", 0.2), ("final", 0.5)):
        for P, fam in ((256, "M256"), (655360, "O1"), (134217728, "A16")):
            rows.append((src, fam, P, 64, base + 0.1 * len(str(P)) / 9))
        if with_n_cells:
            for n in (1, 4, 16):
                rows.append((src, "A16", 134217728, n, base * n / 20))
    p = tmp_path / "cells_summary.csv"
    with open(p, "w") as f:
        f.write("ckpt,cell,source,family,P,n,g,g_lo,g_hi\n")
        for src, fam, P, n, y in rows:
            f.write(
                f"dpo,dpo__{src}__{fam}__n{n},{src},{fam},{P},{n},{y:.3f},{y - 0.12:.3f},{y + 0.12:.3f}\n"
            )
        f.write("dpo,dpo__own__O1__n64,own,O1,655360,64,,,\n")  # a cell without a G: skipped
        f.write("dpo,plain:answeronly,plain,,,,0.05,-0.02,0.12\n")
    return p


@pytest.mark.parametrize("with_n", [True, False])
def test_fig_cells_renders(tmp_path, with_n):
    p = fake_summary(tmp_path, with_n)
    rows = FC.read_rows(p, "dpo")
    ctl = FC.read_control(p, "dpo")
    assert ctl == (0.05, -0.02, 0.12) and all(r["source"] != "plain" for r in rows)
    paths = FC.render(rows, tmp_path / "fig" / "cells", "dpo", ctl)
    assert [x.suffix for x in paths] == [".pdf", ".svg", ".png"] and all(
        x.stat().st_size > 1000 for x in paths
    )
    assert len(rows) == (12 if with_n else 6)
    from PIL import Image

    w, h = Image.open(paths[2]).size
    assert w == round(5.5 * 300)
    FC.render(rows, tmp_path / "fig2" / "cells", "dpo", None)  # without a control
