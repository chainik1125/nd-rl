# ruff: noqa: E501
"""Adapter writer (Turn 5 B.3): PEFT-format LoRA adapters that vLLM 0.11.0's LoRA path reads.

  python3 build_adapters.py --w-dir runs/W/out --out-root runs/build/adapters [--variants A1,A1u,A2,k16,gate]

A rank-r adapter whose product B @ A * (lora_alpha / r) equals the intended matrix, per matrix of the 448 (PyTorch [out, in]
layout, lora_A [r, in], lora_B [out, r], fp32 files; vLLM casts to bf16). Rank 1: r = lora_alpha = 1 (scale 1), balanced
split B = sqrt(c) p, A = sqrt(c) q^T.
  A1   p = u1, q = v1 of Delta_2300, c = ||Delta_2300||_F (Cai Eq. 3, Frobenius reading)
  A1u  same vectors, c = sigma_1 (Leon's literal "top direction only")
  A2   p, q uniform random unit vectors (g then h from one numpy default_rng(710003 + crc32(name)) stream, fp64 normals), c = ||Delta_2300||_F
  k16  unscaled top-16 truncation of Delta_2300: B = U16 S16^(1/2), A = S16^(1/2) V16^T, r = lora_alpha = 16 (Turn 6 D5)
  gate gate_m{1,4,16,64}_r1 and gate_m{1,4,16,64}_r16: the amplified copy m sigma_1 u1 v1^T of A1u, written at r = 1 and as r = 16
       with 16 equal columns sqrt(m c)/4 (lora_alpha = 16); exact in bf16, so both files carry identical mantissas
Writes <out>/<variant>/{adapter_config.json, adapter_model.safetensors, MANIFEST.json (sha256, per-matrix c, seeds)}.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    MATRIX_TYPES,
    SEEDS,
    peft_key,
    random_pair,
    sha256_file,
    tensor_seed,
)

TARGETS = sorted(
    t.split(".")[-1] for t, _ in MATRIX_TYPES
)  # o_proj, down_proj ... as PEFT lists them (a set)


def adapter_config(r: int, alpha: int, base: str = "/workspace/models/dpo") -> dict:
    return {
        "alpha_pattern": {},
        "auto_mapping": None,
        "base_model_name_or_path": base,
        "bias": "none",
        "fan_in_fan_out": False,
        "inference_mode": True,
        "init_lora_weights": True,
        "layer_replication": None,
        "layers_pattern": None,
        "layers_to_transform": None,
        "lora_alpha": alpha,
        "lora_bias": False,
        "lora_dropout": 0.0,
        "modules_to_save": None,
        "peft_type": "LORA",
        "r": r,
        "rank_pattern": {},
        "revision": None,
        "target_modules": sorted({t.split(".")[-1] for t, _ in MATRIX_TYPES}),
        "task_type": "CAUSAL_LM",
        "use_dora": False,
        "use_rslora": False,
    }


def write_adapter(
    out_dir,
    factors: dict[str, tuple[torch.Tensor, torch.Tensor]],
    r: int,
    alpha: int,
    extra: dict | None = None,
    base: str = "/workspace/models/dpo",
) -> Path:
    """factors: tensor name (e.g. model.layers.0.self_attn.q_proj.weight) -> (B [out, r], A [r, in]) float32."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tensors = {}
    for name, (B, A) in factors.items():
        assert B.shape[1] == r and A.shape[0] == r, (name, B.shape, A.shape, r)
        tensors[peft_key(name, "A")] = A.to(torch.float32).contiguous()
        tensors[peft_key(name, "B")] = B.to(torch.float32).contiguous()
    save_file(tensors, str(out / "adapter_model.safetensors"), metadata={"format": "pt"})
    (out / "adapter_config.json").write_text(json.dumps(adapter_config(r, alpha, base), indent=2))
    man = {
        "r": r,
        "lora_alpha": alpha,
        "n_matrices": len(factors),
        "sha256": sha256_file(out / "adapter_model.safetensors"),
        **(extra or {}),
    }
    (out / "MANIFEST.json").write_text(json.dumps(man, indent=1))
    return out


def product(B: torch.Tensor, A: torch.Tensor, alpha: float) -> torch.Tensor:
    """The matrix the adapter adds: B @ A * (alpha / r), computed in float64."""
    r = B.shape[1]
    return (B.double() @ A.double()) * (alpha / r)


def rank1_factors(u: torch.Tensor, v: torch.Tensor, c: float):
    s = float(c) ** 0.5
    return (s * u.reshape(-1, 1).float()), (s * v.reshape(1, -1).float())


def equal_columns_factors(u: torch.Tensor, v: torch.Tensor, c: float, r: int = 16):
    """r equal columns sqrt(c)/sqrt(r) p and rows sqrt(c)/sqrt(r) q^T: product c p q^T at scale lora_alpha / r = 1 (alpha = r)."""
    s = (float(c) / r) ** 0.5
    return (s * u.reshape(-1, 1).float()).repeat(1, r), (s * v.reshape(1, -1).float()).repeat(r, 1)


def load_inputs(w_dir):
    w = Path(w_dir)
    rows = list(csv.DictReader(open(w / "w_matrices.csv")))
    f = load_file(str(w / "factors_d2300_k16.safetensors"))
    return rows, f


def build(
    w_dir,
    out_root,
    variants: list[str],
    names: list[str] | None = None,
    base: str = "/workspace/models/dpo",
) -> dict:
    rows, f = load_inputs(w_dir)
    meta = {r["name"]: r for r in rows}
    names = names or [r["name"] for r in rows]
    out_root = Path(out_root)
    res = {}
    for var in variants:
        fac, cs, seeds = {}, {}, {}
        if var in ("A1", "A1u", "A2"):
            for n in names:
                u, v = f[n + "::U"][:, 0], f[n + "::V"][:, 0]
                fro, s1 = float(meta[n]["fro_d2300"]), float(meta[n]["sigma1_d2300"])
                if var == "A2":
                    sd = tensor_seed(SEEDS["A2"], n)
                    pp, qq = random_pair(int(meta[n]["out"]), int(meta[n]["in"]), sd)
                    p, q = torch.from_numpy(pp), torch.from_numpy(qq)
                    fac[n] = rank1_factors(p, q, fro)
                    seeds[n] = sd
                    cs[n] = fro
                else:
                    c = fro if var == "A1" else s1
                    fac[n] = rank1_factors(u, v, c)
                    cs[n] = c
            write_adapter(
                out_root / var, fac, 1, 1, {"variant": var, "c": cs, "seeds": seeds or None}, base
            )
        elif var == "k16":
            for n in names:
                U, S, V = f[n + "::U"], f[n + "::S"], f[n + "::V"]
                sq = S.sqrt()
                fac[n] = (U * sq, sq.reshape(-1, 1) * V.T)
            write_adapter(out_root / var, fac, 16, 16, {"variant": var}, base)
        elif var == "gate":
            for m in (1, 4, 16, 64):
                f1, f16, cc = {}, {}, {}
                for n in names:
                    u, v = f[n + "::U"][:, 0], f[n + "::V"][:, 0]
                    c = m * float(meta[n]["sigma1_d2300"])
                    f1[n] = rank1_factors(u, v, c)
                    f16[n] = equal_columns_factors(u, v, c, 16)
                    cc[n] = c
                write_adapter(
                    out_root / f"gate_m{m}_r1",
                    f1,
                    1,
                    1,
                    {"variant": "gate_r1", "m": m, "c": cc},
                    base,
                )
                write_adapter(
                    out_root / f"gate_m{m}_r16",
                    f16,
                    16,
                    16,
                    {"variant": "gate_r16", "m": m, "c": cc},
                    base,
                )
        else:
            raise SystemExit(f"unknown variant {var}")
        res[var] = "written"
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--w-dir", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--variants", default="A1,A1u,A2,k16,gate")
    ap.add_argument("--base", default="/workspace/models/dpo")
    a = ap.parse_args(argv)
    torch.set_num_threads(1)
    print(json.dumps(build(a.w_dir, a.out_root, a.variants.split(","), base=a.base)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
