"""Synthetic tiny Olmo3-shaped checkpoints for the rank1 tests (CPU, tiny shapes only)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file

R1 = Path(__file__).resolve().parents[1] / "rank1"
if str(R1) not in sys.path:
    sys.path.insert(0, str(R1))

from common import MATRIX_TYPES, matrix_names  # noqa: E402

H, FF, KV = 64, 96, 32  # hidden, mlp, kv width of the tiny config (4 heads x 16, 2 kv heads)
SHAPES = {
    "self_attn.q_proj": (H, H),
    "self_attn.k_proj": (KV, H),
    "self_attn.v_proj": (KV, H),
    "self_attn.o_proj": (H, H),
    "mlp.gate_proj": (FF, H),
    "mlp.up_proj": (FF, H),
    "mlp.down_proj": (H, FF),
}
NL = 2


def shape_of(name: str):
    for t, _ in MATRIX_TYPES:
        if f".{t}." in name:
            return SHAPES[t]
    raise KeyError(name)


def make_checkpoints(root: Path, signal: float = 1.0, seed: int = 0) -> dict:
    """dpo + three checkpoints whose deltas are planted rank-1 (strength `signal` times the noise) plus bf16 noise.
    Returns the planted unit vectors per matrix."""
    rng = np.random.default_rng(seed)
    names = matrix_names(NL)
    base = {
        n: torch.tensor(rng.standard_normal(shape_of(n)) * 0.05, dtype=torch.float32).bfloat16()
        for n in names
    }
    planted = {}
    ck = {"dpo": base}
    for tag, scale in (("s0100", 0.15), ("s0500", 0.5), ("s2300", 1.0)):
        d = {}
        for n in names:
            m, k = shape_of(n)
            if n not in planted:
                u, v = rng.standard_normal(m), rng.standard_normal(k)
                planted[n] = (u / np.linalg.norm(u), v / np.linalg.norm(v))
            u, v = planted[n]
            delta = signal * scale * 0.5 * np.outer(u, v) * (
                m * k
            ) ** 0.5 * 0.01 + scale * 0.002 * rng.standard_normal((m, k))
            d[n] = (base[n].float() + torch.tensor(delta, dtype=torch.float32)).bfloat16()
        ck[tag] = d
    for tag, d in ck.items():
        (root / tag).mkdir(parents=True, exist_ok=True)
        save_file(dict(d), str(root / tag / "model.safetensors"), metadata={"format": "pt"})
    return planted


def write_sharded(root: Path, tensors: dict[str, torch.Tensor], nshards: int = 2) -> None:
    root.mkdir(parents=True, exist_ok=True)
    names = sorted(tensors)
    wmap, total = {}, 0
    for s in range(nshards):
        sub = {n: tensors[n] for i, n in enumerate(names) if i % nshards == s}
        fn = f"model-{s + 1:05d}-of-{nshards:05d}.safetensors"
        save_file(sub, str(root / fn), metadata={"format": "pt"})
        wmap.update({n: fn for n in sub})
        total += sum(t.numel() * t.element_size() for t in sub.values())
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total}, "weight_map": wmap})
    )
    (root / "config.json").write_text(json.dumps({"architectures": ["Olmo3ForCausalLM"]}))
    (root / "tokenizer.json").write_text("{}")
