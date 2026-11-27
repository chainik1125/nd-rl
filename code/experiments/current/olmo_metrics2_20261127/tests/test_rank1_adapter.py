import json
import struct
from pathlib import Path

import numpy as np
import pytest
import torch
from rank1_fixtures import H, make_checkpoints
from safetensors.torch import load_file

# isort: split
import build_adapters as ba  # noqa: E402
import wpass  # noqa: E402
from common import matrix_names, peft_key, random_pair, tensor_seed  # noqa: E402

N0 = "model.layers.0.self_attn.q_proj.weight"
N1 = "model.layers.1.mlp.down_proj.weight"


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("rank1")
    ck = tmp / "models"
    ck.mkdir()
    make_checkpoints(ck)
    w = tmp / "w"
    assert (
        wpass.main(
            [
                "--models-dir",
                str(ck),
                "--out",
                str(w),
                "--layers",
                "2",
                "--device",
                "cpu",
                "--k",
                "16",
            ]
        )
        == 0
    )
    root = tmp / "adapters"
    ba.build(w, root, ["A1", "A1u", "A2", "k16", "gate"])
    delta = {
        n: (
            load_file(str(ck / "s2300" / "model.safetensors"))[n].float()
            - load_file(str(ck / "dpo" / "model.safetensors"))[n].float()
        )
        for n in matrix_names(2)
    }
    return {
        "root": root,
        "w": w,
        "ck": ck,
        "delta": delta,
        "fac": load_file(str(w / "factors_d2300_k16.safetensors")),
        "rows": {r["name"]: r for r in __import__("csv").DictReader(open(w / "w_matrices.csv"))},
    }


def read(root, var):
    t = load_file(str(root / var / "adapter_model.safetensors"))
    cfg = json.loads((root / var / "adapter_config.json").read_text())
    return t, cfg


def prod(t, cfg, n):
    return ba.product(t[peft_key(n, "B")], t[peft_key(n, "A")], cfg["lora_alpha"])


def test_a1_equals_intended_matrix(built):
    t, cfg = read(built["root"], "A1")
    assert (
        cfg["r"] == 1 and cfg["lora_alpha"] == 1 and not cfg["use_rslora"] and not cfg["use_dora"]
    )
    for n in (N0, N1):
        row = built["rows"][n]
        u, v = built["fac"][n + "::U"][:, 0].double(), built["fac"][n + "::V"][:, 0].double()
        want = float(row["fro_d2300"]) * torch.outer(u, v)
        got = prod(t, cfg, n)
        assert torch.linalg.norm(got - want) / torch.linalg.norm(want) < 1e-6
        # the bf16 cast vLLM applies: error of the product relative to its norm (spec 1e-3; per-factor bf16 gives about 1.5e-3)
        Bq, Aq = t[peft_key(n, "B")].bfloat16(), t[peft_key(n, "A")].bfloat16()
        gotq = ba.product(Bq, Aq, 1)
        assert torch.linalg.norm(gotq - want) / torch.linalg.norm(want) < 5e-3
        # the adapter's matrix has the Frobenius norm of the full delta and cosine sigma1/||delta|| with it (catches a transposed u v^T)
        d = built["delta"][n].double()
        cos = float((got * d).sum() / (torch.linalg.norm(got) * torch.linalg.norm(d)))
        assert abs(cos - float(row["sigma1_d2300"]) / float(row["fro_d2300"])) < 1e-3
        assert abs(float(torch.linalg.norm(got)) - float(row["fro_d2300"])) < 1e-4 * float(
            row["fro_d2300"]
        )


def test_a1u_is_sigma1_and_a2_is_random_unit(built):
    t, cfg = read(built["root"], "A1u")
    for n in (N0, N1):
        row = built["rows"][n]
        got = prod(t, cfg, n)
        assert abs(float(torch.linalg.norm(got)) - float(row["sigma1_d2300"])) < 1e-4 * float(
            row["sigma1_d2300"]
        )
    t2, cfg2 = read(built["root"], "A2")
    for n in (N0, N1):
        row = built["rows"][n]
        got = prod(t2, cfg2, n)
        assert abs(float(torch.linalg.norm(got)) - float(row["fro_d2300"])) < 1e-4 * float(
            row["fro_d2300"]
        )
        d = built["delta"][n].double()
        cos = float((got * d).sum() / (torch.linalg.norm(got) * torch.linalg.norm(d)))
        assert (
            abs(cos) < 0.25
        )  # random direction, no alignment (tiny dims: chance level about 1/sqrt(m n))
        # exact draw: g then h from one default_rng(710003 + crc32(name)) stream, fp64
        p, q = random_pair(int(row["out"]), int(row["in"]), tensor_seed(710003, n))
        want = float(row["fro_d2300"]) * torch.outer(torch.from_numpy(p), torch.from_numpy(q))
        assert torch.linalg.norm(got - want) / torch.linalg.norm(want) < 1e-6
        rng = np.random.default_rng(710003 + __import__("zlib").crc32(n.encode()))
        g = rng.standard_normal(int(row["out"]))
        assert np.allclose(p, g / np.linalg.norm(g))


def test_rank16_is_unscaled_truncation(built):
    t, cfg = read(built["root"], "k16")
    assert cfg["r"] == 16 and cfg["lora_alpha"] == 16
    n = N0
    U, S, V = (
        built["fac"][n + "::U"].double(),
        built["fac"][n + "::S"].double(),
        built["fac"][n + "::V"].double(),
    )
    want = U @ torch.diag(S) @ V.T
    assert torch.linalg.norm(prod(t, cfg, n) - want) / torch.linalg.norm(want) < 1e-6
    # fewer than 16 singular directions exist in a tiny matrix? shapes here are 64 x 64, so 16 is fine
    assert torch.linalg.norm(want) < float(built["rows"][n]["fro_d2300"]) * 1.0000001


def test_gate_copies_rank1_equals_rank16_equal_columns(built):
    for m in (1, 4, 16, 64):
        t1, c1 = read(built["root"], f"gate_m{m}_r1")
        t16, c16 = read(built["root"], f"gate_m{m}_r16")
        assert c1["r"] == 1 and c16["r"] == 16 and c16["lora_alpha"] == 16
        for n in (N0, N1):
            want = (
                m
                * float(built["rows"][n]["sigma1_d2300"])
                * torch.outer(
                    built["fac"][n + "::U"][:, 0].double(), built["fac"][n + "::V"][:, 0].double()
                )
            )
            assert torch.linalg.norm(prod(t1, c1, n) - want) / torch.linalg.norm(want) < 1e-6
            assert torch.linalg.norm(prod(t16, c16, n) - want) / torch.linalg.norm(want) < 1e-6
            # identical bf16 mantissas: the r16 factor is the r1 factor divided by 4 (exact), repeated 16 times
            B1, B16 = t1[peft_key(n, "B")].bfloat16(), t16[peft_key(n, "B")].bfloat16()
            assert torch.equal((B1.float() / 4).bfloat16().float(), B16[:, :1].float())
            assert torch.equal(B16[:, 0], B16[:, 15])


def test_names_and_packing_match_vllm_expectations(built):
    t, _ = read(built["root"], "A1")
    keys = set(t)
    assert len(keys) == 2 * 14
    expected = {
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "qkv_proj",
        "gate_up_proj",
    }
    for k in keys:
        assert k.startswith("base_model.model.model.layers.") and k.endswith(
            (".lora_A.weight", ".lora_B.weight")
        )
        assert (
            k.split(".")[-3] in expected
        )  # the module name vLLM maps to its packed qkv_proj / gate_up_proj
    for n in (N0, N1):
        o, i = built["delta"][n].shape
        assert t[peft_key(n, "B")].shape == (o, 1) and t[peft_key(n, "A")].shape == (1, i)
        assert t[peft_key(n, "A")].dtype == torch.float32
    # the 2026-11-25 adapter: same 896 names for 64 layers
    old = next(
        iter(
            Path("/scratch/cowifornia/nd-rl/olmo_elicit/runs").glob(
                "*/edl/adapters/best_order0/adapter_model.safetensors"
            )
        ),
        None,
    )
    if old is not None:
        b = open(old, "rb")
        n = struct.unpack("<Q", b.read(8))[0]
        hdr = json.loads(b.read(n))
        oldkeys = {k for k in hdr if k != "__metadata__"}
        mine = {peft_key(nm, w) for nm in matrix_names(64) for w in ("A", "B")}
        assert oldkeys == mine and len(mine) == 896


def test_peft_loads_and_merges_to_intended_delta(built, tmp_path):
    transformers = pytest.importorskip("transformers")
    peft = pytest.importorskip("peft")
    cfg = transformers.Olmo3Config(
        vocab_size=128,
        hidden_size=H,
        intermediate_size=96,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        layer_types=["sliding_attention", "full_attention"],
        sliding_window=16,
        tie_word_embeddings=False,
    )
    for var, scale_expected in (("A1", 1.0), ("gate_m4_r16", 1.0)):
        torch.manual_seed(0)
        model = transformers.Olmo3ForCausalLM(cfg).float()
        before = {n: model.state_dict()[n].clone() for n in matrix_names(2)}
        pm = peft.PeftModel.from_pretrained(model, str(built["root"] / var))
        layer0 = pm.base_model.model.model.layers[0].self_attn.q_proj
        assert abs(layer0.scaling["default"] - scale_expected) < 1e-9
        merged = pm.merge_and_unload()
        t, c = read(built["root"], var)
        for n in (N0, N1, "model.layers.1.mlp.up_proj.weight"):
            got = (merged.state_dict()[n] - before[n]).double()
            want = prod(t, c, n)
            assert torch.linalg.norm(got - want) / torch.linalg.norm(want) < 1e-4
