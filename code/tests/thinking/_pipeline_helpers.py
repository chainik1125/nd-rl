"""Shared helpers for the thinking-pipeline tests: tiny and scripted models, an isolated env."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest
import torch

from code.thinking.dan import load

D = load()
EXP = Path(__file__).resolve().parents[2] / "experiments" / "current" / "thinking_traces_20261005"
PILOT = EXP / "pilot_traces.jsonl"
K12 = Path("/scratch/cowifornia/nd-rl/bench/dan_ckpts/data/train_k12.jsonl")
HELDOUT = Path("/scratch/cowifornia/nd-rl/bench/dan_ckpts/data/heldout.jsonl")
STAGE1 = Path(
    "/scratch/cowifornia/nd-rl/bench/dan_ckpts/best-state/stage1/stage1_best12_s0_b1200.pt"
)
LEAN = Path(os.path.expanduser("~/.elan/bin/lean"))
needs_k12 = pytest.mark.skipif(not K12.exists(), reason="local K12 copy not available")
needs_lean = pytest.mark.skipif(not LEAN.exists(), reason="no Lean binary")


def pilot() -> list[dict[str, Any]]:
    with open(PILOT) as f:
        return [json.loads(line) for line in f if line.strip()]


def pilot_rec(rid: str) -> dict[str, Any]:
    return next(r for r in pilot() if r["id"] == rid)


def k12_head(n: int) -> list[dict[str, Any]]:
    out = []
    with open(K12) as f:
        for line in f:
            out.append(json.loads(line))
            if len(out) >= n:
                break
    return out


def tiny_model(vocab: int, seed: int = 0) -> Any:
    torch.manual_seed(seed)
    m = D.best_model.ALiBiGPT(vocab, 2, 32, 4, d_ff=64)
    m.eval()
    return m


def bias_logits(model: Any, bias: torch.Tensor) -> Any:
    """Add a constant vector to the model's logits (a forward hook on the head)."""
    model.head.register_forward_hook(lambda mod, inp, out: out + bias.to(out.dtype))
    return model


class ScriptedModel(torch.nn.Module):
    """Writes a fixed continuation after each known state: all logit mass on the scripted token.

    The per-row history lives in the KV-cache dict as tensors, so the sampler's batch compaction
    keeps it aligned with the rows.  An unknown state gets `<eos>` (an empty action)."""

    def __init__(self, tok: Any, script: dict[bytes, list[int]]) -> None:
        super().__init__()
        self.blocks = torch.nn.ModuleList([torch.nn.Identity()])
        self.dummy = torch.nn.Parameter(torch.zeros(1))
        self.tok, self.script = tok, script
        self.act = tok.stoi["<act>"]

    def forward(
        self, idx: torch.Tensor, pos: Any = None, mask: Any = None, caches: Any = None
    ) -> Any:
        B, T = idx.shape
        c = caches[0]
        if "hist" not in c:
            c["hist"] = torch.full((B, c["max"] + T), -1, dtype=torch.long)
            c["len"] = torch.zeros(B, dtype=torch.long)
        for b in range(B):
            toks = [int(x) for x in idx[b] if T == 1 or int(x) != self.tok.pad]
            n = int(c["len"][b])
            c["hist"][b, n : n + len(toks)] = torch.tensor(toks, dtype=torch.long)
            c["len"][b] = n + len(toks)
        out = torch.full((B, 1, self.tok.vocab_size), -30.0)
        for b in range(B):
            h = c["hist"][b, : int(c["len"][b])].tolist()
            j = len(h) - 1 - h[::-1].index(self.act)
            cont = self.script.get(bytes(h[: j + 1]), [])
            gen = h[j + 1 :]
            nxt = cont[len(gen)] if len(gen) < len(cont) else self.tok.eos
            out[b, 0, nxt] = 30.0
        return out


def script_from_pairs(pairs: list[tuple[bytes, bytes, int]]) -> dict[bytes, list[int]]:
    return {p: list(q) for p, q, _ in pairs}


def isolate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Offline, registry rows and the Lean log under tmp_path (monkeypatch undoes it all)."""
    for k, v in {
        "ND_OFFLINE": "1",
        "ND_REGISTRY_SYNC": "0",
        "ND_REGISTRY_DIR": str(tmp_path / "registry"),
        "LEAN_GATE_LOG": str(tmp_path / "lean_gate.jsonl"),
    }.items():
        monkeypatch.setenv(k, v)
    monkeypatch.delenv("ND_REGISTRY", raising=False)
    monkeypatch.setitem(D.record._state, "file", None)
    return tmp_path
