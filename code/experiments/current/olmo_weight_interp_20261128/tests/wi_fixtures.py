"""Tiny Olmo-shaped checkpoints and synthetic cells for the tests (CPU only)."""

from __future__ import annotations

import gzip
import json
import os
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

W = Path(__file__).resolve().parents[1]
CUR = W.parent
if str(W) not in sys.path:
    sys.path.insert(0, str(W))

TYPES = {
    "self_attn.q_proj": (8, 8),
    "self_attn.k_proj": (4, 8),
    "self_attn.v_proj": (4, 8),
    "self_attn.o_proj": (8, 8),
    "mlp.gate_proj": (12, 8),
    "mlp.up_proj": (12, 8),
    "mlp.down_proj": (8, 12),
}
LAYERS = 6
CONFIG = {"architectures": ["Olmo3ForCausalLM"], "hidden_size": 8, "use_cache": True}


def names() -> dict[str, tuple[int, ...]]:
    out: dict[str, tuple[int, ...]] = {
        "model.embed_tokens.weight": (10, 8),
        "lm_head.weight": (10, 8),
    }
    for i in range(LAYERS):
        for t, shape in TYPES.items():
            out[f"model.layers.{i}.{t}.weight"] = shape
        out[f"model.layers.{i}.post_attention_layernorm.weight"] = (8,)
        out[f"model.layers.{i}.post_feedforward_layernorm.weight"] = (8,)
    out["model.norm.weight"] = (8,)
    return out


def step(t: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """Move each bf16 element by k bf16 steps (k may be negative); stays on the same side of
    zero."""
    bits = t.view(torch.int16).to(torch.int32)
    mag = (bits & 0x7FFF) + torch.where(bits < 0, -k, k)
    mag = mag.clamp(min=1, max=0x7F00)
    return ((bits & ~0x7FFF) | mag).to(torch.int16).view(torch.bfloat16)


def write_sharded(root: Path, tensors: dict[str, torch.Tensor], nshards: int = 3, config=None):
    root.mkdir(parents=True, exist_ok=True)
    order = sorted(tensors)
    wmap, total = {}, 0
    for s in range(nshards):
        sub = {n: tensors[n] for i, n in enumerate(order) if i % nshards == s}
        fn = f"model-{s + 1:05d}-of-{nshards:05d}.safetensors"
        save_file(sub, str(root / fn), metadata={"format": "pt"})
        wmap.update(dict.fromkeys(sub, fn))
        total += sum(t.numel() * t.element_size() for t in sub.values())
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total}, "weight_map": wmap})
    )
    (root / "config.json").write_text(json.dumps(config or CONFIG))
    (root / "tokenizer.json").write_text("{}")


def make_pair(root: Path, seed: int = 0, max_step: int = 3, frozen=("model.layers.1.",)):
    """dpo and s2300: s2300 = dpo moved by -max_step..max_step bf16 steps per element; tensors whose
    name starts with an entry of `frozen` do not move. Returns (dpo tensors, s2300 tensors)."""
    g = torch.Generator().manual_seed(seed)
    d, s = {}, {}
    for n, shape in names().items():
        base = torch.ones(shape) if "norm" in n else torch.randn(shape, generator=g) * 0.05
        d[n] = (base + torch.randn(shape, generator=g) * 0.01).bfloat16()
        k = torch.randint(-max_step, max_step + 1, shape, generator=g, dtype=torch.int32)
        s[n] = d[n].clone() if n.startswith(tuple(frozen)) else step(d[n], k)
    write_sharded(root / "dpo", d)
    write_sharded(root / "s2300", s, config=dict(CONFIG, use_cache=False))
    return d, s


# ---------------------------------------------------------------------------- synthetic cells
def spec48(n: int = 4, prompts: int = 48) -> dict[str, int]:
    return {f"p{i:02d}": n for i in range(prompts)}


def write_cell(
    d: Path,
    tag: str,
    passes: dict[str, list[bool]],
    *,
    spec: dict[str, int] | None = None,
    answer: str = "A full sentence.",
    truncate: bool = False,
    meta: dict | None = None,
    seed: int = 710012,
) -> None:
    """A cell folder as the pod leaves it: passes[pid] = strict flags of the finished draws (shorter
    than the plan = draws with no row)."""
    spec = spec or spec48()
    d.mkdir(parents=True, exist_ok=True)
    rows, graded = [], []
    for pid, flags in passes.items():
        for c, ok in enumerate(flags):
            base = {
                "pid": pid,
                "chunk": c,
                "i": 0,
                "n_tok": 100 + c,
                "finish": "stop",
                "job": f"cell_{tag}",
            }
            rows.append(dict(base, text=f"thinking</think>{answer if ok else 'no'}"))
            graded.append(dict(base, strict=bool(ok), per=[bool(ok)], think_end=True))
    with gzip.open(d / "samples.jsonl.gz", "wt") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    if truncate:  # what a killed sampler leaves: no gzip end marker
        raw = (d / "samples.jsonl.gz").read_bytes()
        (d / "samples.jsonl.gz").write_bytes(raw[:-12])
        with gzip.open(d / "samples_complete.jsonl.gz", "wt") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
    (d / "graded.jsonl").write_text("".join(json.dumps(r) + "\n" for r in graded))
    n = sum(spec.values())
    model_dir = "/workspace/models/s2300" if tag == "a100" else f"/workspace/wi/models/interp_{tag}"
    m = {
        "job": f"cell_{tag}",
        "seed": seed,
        "model_dir": model_dir,
        "n_requests": n,
        "samples": n,
        "temperature": 1.0,
        "top_p": 1.0,
        "max_tokens": 32768,
        "chunk": 1,
        "chunk0": 0,
        "stop_after": 0,
        "lora": None,
        "fill_context": None,
        "tp": 1,
        "vllm": "0.11.0",
        "tok_per_s": 1500.0,
        "prompt_sha256_by_pid": dict.fromkeys(spec, "x"),
    }
    m.update(meta or {})
    if not truncate:
        (d / "meta.jsonl").write_text(json.dumps(m) + "\n")
        (d / "wrap.json").write_text(
            json.dumps({"status": "done", "temperature_overridden": False})
        )


def uniform(k: int, spec: dict[str, int] | None = None) -> dict[str, list[bool]]:
    """Every prompt passes k of its draws."""
    spec = spec or spec48()
    return {p: [i < k for i in range(n)] for p, n in spec.items()}


def by_counts(counts: list[int], spec: dict[str, int] | None = None) -> dict[str, list[bool]]:
    spec = spec or spec48()
    return {p: [i < counts[j] for i in range(n)] for j, (p, n) in enumerate(spec.items())}


def env_path() -> str:
    return f"{Path(sys.executable).parent}:{os.environ['PATH']}"
