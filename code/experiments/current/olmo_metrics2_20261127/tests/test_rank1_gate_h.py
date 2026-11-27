import hashlib
import json

import numpy as np
import rank1_fixtures  # noqa: F401  (puts rank1/ on sys.path)
import torch
from safetensors.torch import load_file, save_file

# isort: split
import lora_gate as lg  # noqa: E402


def mk(seed=0):
    rng = np.random.default_rng(seed)
    base = [rng.uniform(0.5, 3, 300) for _ in range(2)]
    eff = {
        v: [rng.normal(s, 0.2, 300) for _ in range(2)]
        for v, s in (("A1", 0.4), ("A1u", 0.1), ("A2", 0.05))
    }
    return rng, base, eff


def calls_for(rng, base, eff, jitter=0.003):
    out = {}
    for v, e in eff.items():
        out[v] = tuple(
            [b + d + rng.normal(0, jitter, b.size) for b, d in zip(base, e)] for _ in range(2)
        )
    return out


def test_all_exact_pass():
    rng, base, eff = mk()
    ad = calls_for(rng, base, eff)
    code, d = lg.decide(True, [base, base, base], [base, base], ad)
    assert (
        code == 0 and d["base_exact_all"] and d["z_exact_all"] and all(d["a1_repeat_within_10pct"])
    )
    assert d["applied"]["A1"]["examples"][0]["a1_repeat_within_10pct"]
    assert lg.decide(False, [base] * 3, [base] * 2, ad)[0] == 30


def test_base_not_exact_33_fallback():
    rng, base, eff = mk()
    noise = 0.001
    b2 = [b + rng.normal(0, noise, b.size) for b in base]
    b3 = [b + rng.normal(0, noise, b.size) for b in base]
    ad = calls_for(rng, base, eff, jitter=0.0)
    ad = {
        v: tuple([x + 0.05 for x in c] for c in cs) for v, cs in ad.items()
    }  # deviation 0.05+ >> 3 x noise
    ad["A2"] = tuple([b + 0.05 for b in base] for _ in range(2))
    code, d = lg.decide(True, [base, b2, b3], [base, base], ad)
    assert code == 33 and d["fallback"]["mode"] == "noise_floor" and d["fallback"]["factor"] == 3.0
    assert not d["base_exact_all"] and "applied" not in d
    assert d["fallback"]["pass"], d["fallback"]
    assert d["fallback"]["per_example"][0]["n"] > 0
    # one A2 call within the noise -> fallback fails, code stays 33
    ad2 = dict(ad)
    ad2["A2"] = (ad["A2"][0], [b + 1e-4 for b in base])
    code, d = lg.decide(True, [base, b2, b3], [base, base], ad2)
    assert code == 33 and not d["fallback"]["pass"]
    assert not d["fallback"]["per_example"][0]["calls"]["A2_2"]["applied"]


def test_z_not_exact_34():
    rng, base, eff = mk()
    ad = calls_for(rng, base, eff)
    z = [b + 1e-3 for b in base]
    code, d = lg.decide(True, [base] * 3, [z, base], ad)
    assert code == 34 and "fallback" in d and d["base_exact_all"] and not d["z_exact_all"]


def test_a1_half_effect_and_dropped_adapter_32():
    rng, base, eff = mk()
    ad = calls_for(rng, base, eff)
    half = [b + 0.5 * d for b, d in zip(base, eff["A1"])]
    ad1 = dict(ad)
    ad1["A1"] = (ad["A1"][0], half)
    code, d = lg.decide(True, [base] * 3, [base] * 2, ad1)
    assert code == 32 and not all(d["a1_repeat_within_10pct"])
    # rule 3a's own checks alone would not catch a mild change: summed repeat is within 0.5 % of base for A1u
    for v in ("A1", "A1u", "A2"):
        adx = dict(ad)
        adx[v] = (ad[v][0], [b.copy() for b in base])  # adapter dropped in call 2
        assert lg.decide(True, [base] * 3, [base] * 2, adx)[0] == 32, v


def test_a1_repeat_rule_and_stats():
    base = [np.zeros(10)]
    assert lg.a1_repeat_rule(base, [np.ones(10)], [np.ones(10) * 1.05]) == [True]
    assert lg.a1_repeat_rule(base, [np.ones(10)], [np.ones(10) * 1.2]) == [False]
    st = lg.call_stats(base, {"x": [np.concatenate([np.zeros(5), np.ones(5)])]})
    assert (
        st["x"][0]["sum"] == 5
        and st["x"][0]["frac_tokens_diff_vs_base1"] == 0.5
        and st["x"][0]["mean_abs_vs_base1"] == 0.5
    )


def test_make_zero_b_adapter(tmp_path):
    src, dst = tmp_path / "A1", tmp_path / "ctrl" / "Z"
    src.mkdir()
    t = {
        "base_model.model.l0.lora_A.weight": torch.randn(1, 8),
        "base_model.model.l0.lora_B.weight": torch.randn(8, 1),
        "base_model.model.l1.lora_A.weight": torch.randn(1, 8),
        "base_model.model.l1.lora_B.weight": torch.randn(8, 1, dtype=torch.float32).to(
            torch.bfloat16
        ),
    }
    save_file(t, str(src / "adapter_model.safetensors"))
    (src / "adapter_config.json").write_text(json.dumps({"r": 1}))
    info = lg.make_zero_b_adapter(src, dst)
    assert info["n_zeroed"] == 2 and info["n_tensors"] == 4
    assert (
        info["sha256"]
        == hashlib.sha256((dst / "adapter_model.safetensors").read_bytes()).hexdigest()
    )
    out = load_file(str(dst / "adapter_model.safetensors"))
    for k, v in t.items():
        assert out[k].dtype == v.dtype and out[k].shape == v.shape
        if "lora_B" in k:
            assert not out[k].float().abs().sum()
        else:
            assert torch.equal(out[k], v)
    assert json.loads((dst / "adapter_config.json").read_text()) == {"r": 1}
