# ruff: noqa: E501
"""compare_adapters: the low-rank inner products equal the dense ones, and the readings follow Amendment 6(a)."""

import json

import torch
from compare_adapters import compare, reading
from safetensors.torch import save_file


def _adapter(d, A, B, r=2, alpha=4, prefix="base_model.model.model.layers.0.self_attn.q_proj"):
    d.mkdir()
    (d / "adapter_config.json").write_text(json.dumps({"r": r, "lora_alpha": alpha}))
    save_file(
        {
            f"{prefix}.lora_A.weight": A.float().contiguous(),
            f"{prefix}.lora_B.weight": B.float().contiguous(),
        },
        str(d / "adapter_model.safetensors"),
    )


def test_matches_dense(tmp_path):
    g = torch.Generator().manual_seed(0)
    A1, B1 = torch.randn(2, 6, generator=g), torch.randn(5, 2, generator=g)
    A2, B2 = torch.randn(2, 6, generator=g), torch.randn(5, 2, generator=g)
    _adapter(tmp_path / "a", A1, B1)
    _adapter(tmp_path / "b", A2, B2)
    out = compare(tmp_path / "a", tmp_path / "b")
    W1, W2 = 2 * B1.double() @ A1.double(), 2 * B2.double() @ A2.double()
    cos = float((W1 * W2).sum() / (W1.norm() * W2.norm()))
    assert abs(out["total_cos"] - cos) < 1e-6
    assert abs(out["total_norm_a"] - float(W1.norm())) < 1e-5
    assert out["n_common"] == 1


def test_identical_is_match(tmp_path):
    g = torch.Generator().manual_seed(1)
    A, B = torch.randn(2, 6, generator=g), torch.randn(5, 2, generator=g)
    _adapter(tmp_path / "a", A, B)
    _adapter(tmp_path / "b", A, B)
    out = compare(tmp_path / "a", tmp_path / "b")
    assert out["reading"] == "match" and abs(out["total_cos"] - 1) < 1e-9


def test_readings():
    assert reading(0.995, 1.01) == "match"
    assert reading(0.95, 1.01) == "in between"
    assert reading(0.995, 1.05) == "in between"
    assert reading(0.85, 1.0) == "no match"
    assert reading(0.995, 1.12) == "no match"
