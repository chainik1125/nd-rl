import json

import pytest
import torch
from rank1_fixtures import write_sharded
from safetensors import safe_open

# isort: split
import expo_write as ew  # noqa: E402
from common import Checkpoint, sha256_file  # noqa: E402


def tiny(root, seed, shift=0.0):
    g = torch.Generator().manual_seed(seed)
    base = torch.Generator().manual_seed(7)
    t = {}
    for i in range(5):
        w = torch.randn(6, 4, generator=base) * 0.05
        t[f"model.layers.{i}.mlp.up_proj.weight"] = (
            w + shift * torch.randn(6, 4, generator=g) * 0.003
        ).bfloat16()
    t["model.norm.weight"] = (torch.ones(4) + shift * torch.randn(4, generator=g) * 0.01).bfloat16()
    t["rope.inv_freq_int"] = torch.arange(3, dtype=torch.int64)
    write_sharded(root, t, nshards=3)
    return t


def test_write_layout_arithmetic_manifest(tmp_path):
    d, s = tiny(tmp_path / "dpo", 1, 0.0), tiny(tmp_path / "s0500", 2, 1.0)
    man = ew.write(
        tmp_path / "dpo", tmp_path / "s0500", 3.5, tmp_path / "expo", verify=4, min_free_gb=0
    )
    assert man["verify_ok"] and man["tensors"] == 7
    out = Checkpoint(tmp_path / "expo")
    # same shard names and tensor-to-shard map, index copied byte for byte
    assert out.wmap == Checkpoint(tmp_path / "dpo").wmap
    assert sha256_file(tmp_path / "expo" / "model.safetensors.index.json") == sha256_file(
        tmp_path / "dpo" / "model.safetensors.index.json"
    )
    for n, a in d.items():
        o = out.get(n)
        assert o.dtype == a.dtype and o.shape == a.shape
        if a.is_floating_point():
            ref = (a.float() + 3.5 * (s[n].float() - a.float())).bfloat16()
            assert torch.equal(o, ref), n  # fp32 combine, round-to-nearest-even bf16
        else:
            assert torch.equal(o, a)
    assert (tmp_path / "expo" / "config.json").exists() and (
        tmp_path / "expo" / "tokenizer.json"
    ).exists()
    for fn, h in man["files"].items():
        assert sha256_file(tmp_path / "expo" / fn) == h
    with safe_open(
        str(tmp_path / "expo" / "model-00001-of-00003.safetensors"), framework="pt"
    ) as f:
        assert f.metadata() == {"format": "pt"}
    gamma_back = json.loads((tmp_path / "expo" / "MANIFEST.json").read_text())["gamma"]
    assert gamma_back == 3.5


def test_gamma_one_is_s0500_and_zero_is_dpo(tmp_path):
    d, s = tiny(tmp_path / "dpo", 1, 0.0), tiny(tmp_path / "s0500", 2, 1.0)
    ew.write(tmp_path / "dpo", tmp_path / "s0500", 1.0, tmp_path / "g1", verify=1, min_free_gb=0)
    ew.write(tmp_path / "dpo", tmp_path / "s0500", 0.0, tmp_path / "g0", verify=1, min_free_gb=0)
    for n in d:
        assert torch.equal(Checkpoint(tmp_path / "g1").get(n), s[n])
        assert torch.equal(Checkpoint(tmp_path / "g0").get(n), d[n])


def test_refuses_without_disk_room_and_cli_codes(tmp_path):
    tiny(tmp_path / "dpo", 1, 0.0)
    tiny(tmp_path / "s0500", 2, 1.0)
    with pytest.raises(SystemExit):
        ew.write(tmp_path / "dpo", tmp_path / "s0500", 4.6, tmp_path / "x", min_free_gb=1e9)
    assert (
        ew.main(
            [
                "--dpo",
                str(tmp_path / "dpo"),
                "--s0500",
                str(tmp_path / "s0500"),
                "--gamma",
                "4.6",
                "--out",
                str(tmp_path / "x"),
                "--min-free-gb",
                "1e9",
            ]
        )
        == 5
    )
    assert (
        ew.main(
            [
                "--dpo",
                str(tmp_path / "dpo"),
                "--s0500",
                str(tmp_path / "s0500"),
                "--gamma",
                "4.6",
                "--out",
                str(tmp_path / "y"),
                "--min-free-gb",
                "0",
            ]
        )
        == 0
    )


def test_extrapolation_collapses_rounding_to_float32_not_bf16_chain(tmp_path):
    # theta = dpo + gamma (s - dpo) in one fp32 expression, not (gamma s - (gamma-1) dpo) or a bf16 intermediate
    d, s = tiny(tmp_path / "dpo", 1, 0.0), tiny(tmp_path / "s0500", 2, 1.0)
    n = "model.layers.0.mlp.up_proj.weight"
    got = ew.combine(d[n], s[n], 4.6)
    assert torch.equal(got, (d[n].float() + 4.6 * (s[n].float() - d[n].float())).bfloat16())


def test_hybrid_e_copies_matrices_from_s2300_and_rest_from_dpo(tmp_path):
    import hybrid_write as hw
    from rank1_fixtures import make_checkpoints, write_sharded

    ck = tmp_path / "m"
    ck.mkdir()
    make_checkpoints(ck)
    from safetensors.torch import load_file

    d = load_file(str(ck / "dpo" / "model.safetensors"))
    s = load_file(str(ck / "s2300" / "model.safetensors"))
    extra_d = {
        **d,
        "model.embed_tokens.weight": torch.randn(10, 4).bfloat16(),
        "lm_head.weight": torch.randn(10, 4).bfloat16(),
    }
    extra_s = {
        **s,
        "model.embed_tokens.weight": torch.randn(10, 4).bfloat16(),
        "lm_head.weight": torch.randn(10, 4).bfloat16(),
    }
    write_sharded(tmp_path / "dpo", extra_d, 3)
    write_sharded(tmp_path / "s2300", extra_s, 3)
    man = hw.write(
        tmp_path / "dpo", tmp_path / "s2300", tmp_path / "hyb", n_layers=2, min_free_gb=0
    )
    assert man["verify_ok"] and man["matrices_from_s2300"] == 14 and man["others_from_dpo"] == 2
    o = Checkpoint(tmp_path / "hyb")
    assert torch.equal(
        o.get("model.layers.0.self_attn.q_proj.weight"),
        extra_s["model.layers.0.self_attn.q_proj.weight"],
    )
    assert torch.equal(o.get("lm_head.weight"), extra_d["lm_head.weight"]) and not torch.equal(
        o.get("lm_head.weight"), extra_s["lm_head.weight"]
    )
    assert (
        hw.main(
            [
                "--dpo",
                str(tmp_path / "dpo"),
                "--s2300",
                str(tmp_path / "s2300"),
                "--out",
                str(tmp_path / "h2"),
                "--layers",
                "2",
                "--min-free-gb",
                "1e9",
            ]
        )
        == 5
    )


def test_verify_exact_cancellation_is_not_an_error(tmp_path):
    """Orchestrator fix 2026-11-27: 23 b = 18 a gives an exact 0 at gamma 4.6; the first verifier flagged it as 128 ulp."""
    import torch
    from safetensors.torch import save_file

    a = torch.tensor(
        [0.0004825592041015625, 0.00024127960205078125, -0.00019741058349609375, 0.5, 1.0],
        dtype=torch.bfloat16,
    )
    b = torch.tensor(
        [0.000377655029296875, 0.0001888275146484375, -0.0001544952392578125, 0.25, 1.0],
        dtype=torch.bfloat16,
    )
    for d, t in (("dpo", a), ("s0500", b)):
        (tmp_path / d).mkdir()
        save_file(
            {"w": t.reshape(1, 5)},
            str(tmp_path / d / "model-00001-of-00001.safetensors"),
            metadata={"format": "pt"},
        )
        (tmp_path / d / "model.safetensors.index.json").write_text(
            '{"metadata": {"total_size": 10}, "weight_map": {"w": "model-00001-of-00001.safetensors"}}'
        )
    man = ew.write(
        tmp_path / "dpo", tmp_path / "s0500", 4.6, tmp_path / "g46", verify=1, min_free_gb=0
    )
    assert man["verify_ok"], man["verify"]
    from safetensors import safe_open

    with safe_open(str(tmp_path / "g46" / "model-00001-of-00001.safetensors"), framework="pt") as f:
        o = f.get_tensor("w").flatten()
    assert float(o[0]) == 0.0 and float(o[1]) == 0.0 and float(o[2]) == 0.0
    assert (
        ew.main(
            [
                "--dpo",
                str(tmp_path / "dpo"),
                "--s0500",
                str(tmp_path / "s0500"),
                "--gamma",
                "4.6",
                "--out",
                str(tmp_path / "g46"),
                "--verify",
                "1",
                "--verify-only",
            ]
        )
        == 0
    )
    # a corrupted element is still caught
    bad = o.clone()
    bad[3] = bad[3] * 1.5
    save_file(
        {"w": bad.reshape(1, 5)},
        str(tmp_path / "g46" / "model-00001-of-00001.safetensors"),
        metadata={"format": "pt"},
    )
    assert (
        ew.main(
            [
                "--dpo",
                str(tmp_path / "dpo"),
                "--s0500",
                str(tmp_path / "s0500"),
                "--gamma",
                "4.6",
                "--out",
                str(tmp_path / "g46"),
                "--verify",
                "1",
                "--verify-only",
            ]
        )
        == 4
    )
