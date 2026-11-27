"""CPU tests of steer/steer_patch.py against a stub Olmo2DecoderLayer (no vLLM on serrano).  Run: pytest -q tests/test_steer_patch.py"""

import json
import os
import sys
import types
from pathlib import Path

import pytest
import torch

torch.set_num_threads(1)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "steer"))
import steer_common as C  # noqa: E402
import steer_patch as P  # noqa: E402

HID = 8


def stub_module():
    """A fake vllm.model_executor.models.olmo2 with the real class's constructor / forward signatures."""

    class Olmo2DecoderLayer(torch.nn.Module):
        def __init__(self, *, vllm_config, prefix=""):
            super().__init__()
            self.prefix = prefix
            self.w = torch.nn.Parameter(torch.ones(1, dtype=torch.bfloat16), requires_grad=False)

        def forward(self, positions, hidden_states):
            return hidden_states * self.w  # stand-in for the block; x * 1.0 is exact

    m = types.SimpleNamespace(Olmo2DecoderLayer=Olmo2DecoderLayer)
    return m


def eli(prefix):
    return int(prefix.split(".")[-1])


CFG = types.SimpleNamespace(
    model_config=types.SimpleNamespace(hf_config=types.SimpleNamespace(hidden_size=HID))
)


def make_vec(path, vec, layer, alpha=2.0, seed=None):
    t = torch.as_tensor(vec, dtype=torch.float32)
    torch.save(
        {
            "vec": t,
            "sha256": C.sha256_bytes(t.numpy().tobytes()),
            "norm": float(t.norm()),
            "layer": layer,
            "alpha": alpha,
            "scramble_seed": seed,
            "vtype": "V_resp",
        },
        path,
    )
    return C.sha256_bytes(t.numpy().tobytes())


def build_layers(m, n=4):
    return [m.Olmo2DecoderLayer(vllm_config=CFG, prefix=f"model.layers.{i}") for i in range(n)]


def test_buffer_added_to_the_output_of_layer_lminus1_only(tmp_path, capsys):
    m = stub_module()
    vec = torch.arange(HID, dtype=torch.float32) / 4
    sha = make_vec(tmp_path / "v.pt", vec, layer=3)
    act = tmp_path / "active.jsonl"
    info = P.install(
        3, str(tmp_path / "v.pt"), olmo2=m, extract_layer_index=eli, active_file=str(act)
    )
    layers = build_layers(m)
    x = torch.randn(5, HID).to(torch.bfloat16)
    pos = torch.arange(5)
    for i, ly in enumerate(layers):
        out = ly(pos, x)
        if i == 2:  # layer index l-1 = 2
            assert torch.equal(out, x + vec.to(torch.bfloat16))
            assert out.dtype == torch.bfloat16 and ly.steer_vec.dtype == torch.bfloat16
            assert (
                "steer_vec" in dict(ly.named_buffers()) and "steer_vec" not in ly.state_dict()
            )  # non-persistent
        else:
            assert torch.equal(out, x) and not hasattr(ly, "steer_vec")
    # STEER_ACTIVE: printed and written, with layer, alpha, scramble seed, vector sha and norm
    printed = [ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("STEER_ACTIVE ")]
    assert len(printed) == 1
    rec = json.loads(printed[0][len("STEER_ACTIVE ") :])
    assert (
        rec["layer"] == 3
        and rec["module_index"] == 2
        and rec["vec_sha256"] == sha == info["sha256"]
    )
    assert (
        rec["alpha"] == 2.0
        and rec["scramble_seed"] is None
        and rec["norm"] == pytest.approx(float(vec.norm()))
    )
    assert json.loads(act.read_text().splitlines()[0])["vec_sha256"] == sha


def test_zero_vector_is_identity_and_still_registered(tmp_path):
    m = stub_module()
    make_vec(tmp_path / "z.pt", torch.zeros(HID), layer=2, alpha=0.0)
    P.install(2, str(tmp_path / "z.pt"), olmo2=m, extract_layer_index=eli)
    ly = build_layers(m)[1]
    assert hasattr(ly, "steer_vec")
    x = torch.randn(7, HID).to(torch.bfloat16)
    assert torch.equal(ly(torch.arange(7), x), x)  # x + 0 is exact in bf16


def test_scramble_metadata_and_layer_mismatch_and_sha_check(tmp_path, capsys):
    m = stub_module()
    make_vec(tmp_path / "s.pt", torch.randn(HID), layer=5, seed=2001)
    with pytest.raises(SystemExit, match="built for layer 5"):
        P.install(3, str(tmp_path / "s.pt"), olmo2=m, extract_layer_index=eli)
    P.install(5, str(tmp_path / "s.pt"), olmo2=m, extract_layer_index=eli)
    build_layers(m, 6)
    rec = json.loads(
        [ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("STEER_ACTIVE ")][0][
            13:
        ]
    )
    assert rec["scramble_seed"] == 2001
    # a vector file whose stored sha256 does not match its tensor is refused
    t = torch.randn(HID)
    torch.save({"vec": t, "sha256": "0" * 64, "layer": 1}, tmp_path / "bad.pt")
    with pytest.raises(SystemExit, match="sha256"):
        P.install(1, str(tmp_path / "bad.pt"), olmo2=stub_module(), extract_layer_index=eli)


def test_wrong_hidden_size_and_double_install_refused(tmp_path):
    m = stub_module()
    make_vec(tmp_path / "v.pt", torch.randn(HID + 1), layer=1)
    P.install(1, str(tmp_path / "v.pt"), olmo2=m, extract_layer_index=eli)
    with pytest.raises(AssertionError, match="hidden size"):
        build_layers(m, 1)
    m2 = stub_module()
    make_vec(tmp_path / "w.pt", torch.randn(HID), layer=1)
    P.install(1, str(tmp_path / "w.pt"), olmo2=m2, extract_layer_index=eli)
    with pytest.raises(SystemExit, match="already patched"):
        P.install(1, str(tmp_path / "w.pt"), olmo2=m2, extract_layer_index=eli)


def test_inactive_without_layer_env_and_error_without_vec(capsys):
    assert P.install_from_env({}) is None
    assert "STEER_INACTIVE" in capsys.readouterr().out
    with pytest.raises(SystemExit, match="STEER_VEC missing"):
        P.install_from_env({"STEER_LAYER": "3"})


def test_apply_sets_inproc_fallback_and_writes_sidecar(tmp_path, monkeypatch):
    monkeypatch.delenv("STEER_LAYER", raising=False)
    monkeypatch.delenv("STEER_ACTIVE_FILE", raising=False)
    monkeypatch.setenv("STEER_INPROC", "1")
    monkeypatch.delenv("VLLM_ENABLE_V1_MULTIPROCESSING", raising=False)
    side = tmp_path / "wrap.json"
    P.apply(types.SimpleNamespace(sidecar=str(side)))
    assert os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] == "0"  # the env switch of the fallback
    spec = json.loads(Path(str(side) + ".steer.json").read_text())
    assert spec["STEER_INPROC"] == "1" and spec["steer"] is None
    assert os.environ["STEER_ACTIVE_FILE"].endswith("wrap.json.steer_active.jsonl")
    monkeypatch.delenv("STEER_INPROC")
    monkeypatch.delenv("VLLM_ENABLE_V1_MULTIPROCESSING")
    P.apply(types.SimpleNamespace(sidecar=str(side)))
    assert (
        "VLLM_ENABLE_V1_MULTIPROCESSING" not in os.environ
    )  # without the switch the default (forked engine) stays
