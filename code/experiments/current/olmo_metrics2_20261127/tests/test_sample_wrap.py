"""CPU tests of sample_wrap.py with a stub vllm (never installed); Dan's file is only read."""

import gzip
import hashlib
import json
import sys
import types
from pathlib import Path

import pytest

X = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(X))
import sample_wrap  # noqa: E402

DAN_ROOT = Path("/scratch/cowifornia/nd-rl/olmo_elicit/ext/nd-takehome")
DAN_FILE = DAN_ROOT / "olmo_ifladder" / "sample.py"
DAN_SHA_AT_IMPORT = hashlib.sha256(DAN_FILE.read_bytes()).hexdigest()


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


class Recorder:
    events = []
    sps = []


def make_stub():
    rec = Recorder
    rec.events, rec.sps = [], []
    vllm = types.ModuleType("vllm")
    vllm.__version__ = "stub"

    class SamplingParams:
        def __init__(self, **kw):
            self.kw = kw
            self.n = kw["n"]
            rec.sps.append(kw)

    class Comp:
        def __init__(self, i):
            self.index, self.token_ids, self.finish_reason, self.stop_reason = (
                i,
                [5, 6, 7],
                "stop",
                None,
            )

    class Out:
        def __init__(self, rid, n):
            self.request_id, self.finished = rid, True
            self.outputs = [Comp(i) for i in range(n)]

    class Engine:
        def __init__(self):
            self.q = []
            cc = types.SimpleNamespace(num_gpu_blocks=1, block_size=16)
            self.vllm_config = types.SimpleNamespace(
                cache_config=cc, model_config=types.SimpleNamespace(max_model_len=99)
            )

        def add_request(self, rid, prompt, sp, lora_request=None):
            self.q.append(Out(rid, sp.n))

        def has_unfinished_requests(self):
            return bool(self.q)

        def step(self):
            return [self.q.pop(0)]

    class LLM:
        def __init__(self, **kw):
            rec.events.append("LLM")
            self.llm_engine = Engine()

    vllm.LLM, vllm.SamplingParams = LLM, SamplingParams
    vllm.EVENTS = rec.events
    sp_mod = types.ModuleType("vllm.sampling_params")
    sp_mod.RequestOutputKind = types.SimpleNamespace(FINAL_ONLY=1)
    vllm.sampling_params = sp_mod
    return vllm, sp_mod, rec


@pytest.fixture
def stub(monkeypatch):
    vllm, sp_mod, rec = make_stub()
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "vllm.sampling_params", sp_mod)
    import transformers

    class Tok:
        pad_token_id = None

        @classmethod
        def from_pretrained(cls, *a, **k):
            return cls()

        def apply_chat_template(self, msgs, **k):
            return [1, 2, 3]

        def decode(self, ids, **k):
            return "x</think>y"

    monkeypatch.setattr(transformers, "AutoTokenizer", Tok)
    return rec


def run(tmp_path, wrap_flags, extra_dan=()):
    prob = tmp_path / "p.jsonl"
    prob.write_text(json.dumps({"pid": "p1", "task": "ifbench", "prompt": "hi"}) + "\n")
    out, meta = tmp_path / "s.jsonl.gz", tmp_path / "meta.jsonl"
    argv = [
        *wrap_flags,
        "--dan-root",
        str(DAN_ROOT),
        "--",
        "--model",
        "fake",
        "--name",
        "n",
        "--rev",
        "r",
        "--problems",
        str(prob),
        "--n",
        "2",
        "--chunk",
        "2",
        "--seed",
        "1",
        "--job",
        "j",
        "--out",
        str(out),
        "--meta",
        str(meta),
        "--max_tokens",
        "16",
        *extra_dan,
    ]
    return sample_wrap.main(argv), out, meta


def test_override_reaches_sampling_params_and_meta(stub, tmp_path):
    before = sha(DAN_FILE)
    side = tmp_path / "side.json"
    run(tmp_path, ["--temperature", "0.6", "--top-p", "0.95", "--sidecar", str(side)])
    assert stub.sps and all(s["temperature"] == 0.6 and s["top_p"] == 0.95 for s in stub.sps)
    m = json.loads(Path(tmp_path / "meta.jsonl").read_text().splitlines()[-1])
    assert (m["temperature"], m["top_p"]) == (0.6, 0.95)
    s = json.load(open(side))
    assert s["status"] == "done" and s["meta_check"].startswith("ok")
    assert s["temperature"] == 0.6 and s["dan_sample_py_sha256"] == before
    assert s["utc_start"] and s["utc_end"] and s["dan_argv"][0] == "--model"
    with gzip.open(tmp_path / "s.jsonl.gz", "rt") as f:
        assert len(f.readlines()) == 2
    assert sha(DAN_FILE) == before


def test_defaults_are_one_and_one(stub, tmp_path):
    run(tmp_path, [])
    assert stub.sps[0]["temperature"] == 1.0 and stub.sps[0]["top_p"] == 1.0
    m = json.loads(Path(tmp_path / "meta.jsonl").read_text().splitlines()[-1])
    assert (m["temperature"], m["top_p"]) == (1.0, 1.0)


def test_patch_called_before_llm(stub, tmp_path):
    patch = tmp_path / "patch_mod.py"
    patch.write_text(
        "CALLED = []\n"
        "def apply(args):\n"
        "    import vllm\n"
        "    CALLED.append(sorted(args.llm_kwargs))\n"
        "    import builtins\n"
        "    builtins._m2_events = getattr(builtins, '_m2_events', [])\n"
        "    builtins._m2_events.append('apply')\n"
    )
    side = tmp_path / "side.json"
    import builtins

    builtins._m2_events = []
    run(tmp_path, ["--patch", str(patch), "--sidecar", str(side)])
    assert builtins._m2_events == ["apply"]
    assert stub.events == ["LLM"]
    # order: apply ran before the real LLM was constructed
    s = json.load(open(side))
    assert (
        s["patch_sha256"] == sha(patch)
        and s["patch"] == str(patch.resolve())
        and "patch_applied_utc" in s
    )
    del builtins._m2_events


def test_patch_order_strict(stub, tmp_path):
    patch = tmp_path / "patch_order.py"
    patch.write_text(
        "def apply(args):\n    import vllm\n    vllm.ORDER = list(vllm.EVENTS) + ['apply']\n"
    )
    run(tmp_path, ["--patch", str(patch)])
    import vllm

    assert vllm.ORDER == ["apply"]  # LLM() had not been constructed yet when apply ran
    assert stub.events == ["LLM"]


def test_patch_without_apply_refused(stub, tmp_path):
    bad = tmp_path / "bad.py"
    bad.write_text("x = 1\n")
    with pytest.raises(SystemExit):
        run(tmp_path, ["--patch", str(bad)])


def test_meta_mismatch_detected(stub, tmp_path, monkeypatch):
    """If Dan's meta did not carry the overridden values the wrapper must fail (simulated by a main that ignores them)."""
    real = sample_wrap.load_module

    def patched(path, name):
        m = real(path, name)
        if name.startswith("dan_"):
            orig = m.main

            def main2():
                rc = orig()
                lines = Path(tmp_path / "meta.jsonl").read_text().splitlines()
                d = json.loads(lines[-1])
                d["temperature"] = 1.0
                Path(tmp_path / "meta.jsonl").write_text(
                    "\n".join(lines[:-1] + [json.dumps(d)]) + "\n"
                )
                return rc

            m.main = main2
        return m

    monkeypatch.setattr(sample_wrap, "load_module", patched)
    with pytest.raises(AssertionError):
        run(tmp_path, ["--temperature", "0.6"])


def test_dan_file_unchanged():
    assert sha(DAN_FILE) == DAN_SHA_AT_IMPORT  # after every test above ran
