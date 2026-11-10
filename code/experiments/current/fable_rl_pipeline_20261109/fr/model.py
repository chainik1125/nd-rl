"""The network: a from-scratch decoder-only transformer (RoPE) with a policy head and two scalar
heads read from the same trunk.

- policy: next-token logits over the `lean_staten` vocabulary (the action, token by token).
- verifier (aux[0]): read at the last token of an action; "is this step well typed?".
- value (aux[1]): read at the `<act>` token that ends the state; "will this attempt finish?".

Module names for the trunk follow Dan's `model.GPT`, so his checkpoints load (aux head left at
its initial value) and can be run by the same solver in plain mode.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

_ROPE = {}


def rope_table(hd: int, device, n: int = 8192, base: float = 10000.0):
    key = (hd, str(device), n)
    c = _ROPE.get(key)
    if c is None:
        pos = torch.arange(n, device=device, dtype=torch.float32)
        inv = 1.0 / (base ** (torch.arange(0, hd, 2, device=device, dtype=torch.float32) / hd))
        ang = pos[:, None] * inv[None, :]
        c = _ROPE[key] = (torch.cos(ang), torch.sin(ang))
    return c


def apply_rope(x, cos, sin):
    # x: (B, H, T, hd); cos/sin: (T, hd/2) or (B, T, hd/2)
    x1, x2 = x[..., ::2], x[..., 1::2]
    if cos.dim() == 2:
        cos, sin = cos[None, None], sin[None, None]
    else:
        cos, sin = cos[:, None], sin[:, None]
    return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1).flatten(-2)


class Block(nn.Module):
    def __init__(self, d: int, h: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.fc1 = nn.Linear(d, 4 * d)
        self.fc2 = nn.Linear(4 * d, d)
        self.h = h

    def forward(self, x, cos, sin, mask=None, cache=None):
        B, T, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(B, T, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if cache is not None:
            n0 = cache["n"]
            cache["k"][:, :, n0 : n0 + T] = k
            cache["v"][:, :, n0 : n0 + T] = v
            cache["n"] = n0 + T
            k, v = cache["k"][:, :, : n0 + T], cache["v"][:, :, : n0 + T]
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        else:
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=mask is None)
        x = x + self.proj(y.transpose(1, 2).reshape(B, T, D))
        return x + self.fc2(F.gelu(self.fc1(self.ln2(x))))


class GPT(nn.Module):
    def __init__(self, vocab: int, n_layer: int = 8, d: int = 384, n_head: int = 8, **_):
        super().__init__()
        self.cfg = dict(vocab=vocab, n_layer=n_layer, d=d, n_head=n_head)
        self.emb = nn.Embedding(vocab, d)
        self.blocks = nn.ModuleList([Block(d, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.aux = nn.Linear(d, 2)
        self.hd = d // n_head
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def trunk(self, idx, pos=None, mask=None, caches=None):
        """-> final hidden states (B, T, d)."""
        cos, sin = rope_table(self.hd, idx.device)
        if pos is None:
            cos, sin = cos[: idx.shape[1]], sin[: idx.shape[1]]
        else:
            cos, sin = cos[pos], sin[pos]
        x = self.emb(idx)
        for i, b in enumerate(self.blocks):
            x = b(x, cos, sin, mask, None if caches is None else caches[i])
        return self.ln_f(x)

    def forward(self, idx):
        h = self.trunk(idx)
        return self.head(h), self.aux(h)

    def new_caches(self, B: int, T: int, dtype):
        dev = self.emb.weight.device
        hd = self.hd
        return [
            {
                "k": torch.empty(B, b.h, T, hd, device=dev, dtype=dtype),
                "v": torch.empty(B, b.h, T, hd, device=dev, dtype=dtype),
                "n": 0,
            }
            for b in self.blocks
        ]


def save_ckpt(path: str, model: GPT, extra: dict | None = None):
    torch.save(
        {"cfg": model.cfg, "state": model.state_dict(), "tok_mode": "lean_staten",
         "extra": extra or {}},
        path,
    )  # fmt: skip


def load_ckpt(path: str, device="cpu"):
    """-> (model in eval mode, extra). Also reads Dan's `lean_staten` checkpoints: his trunk has
    the same module names; the aux head then keeps its initial weights (`extra['has_aux']` False)."""
    ck = torch.load(path, map_location=device, weights_only=False)
    assert ck.get("tok_mode", "lean_staten") == "lean_staten", ck.get("tok_mode")
    cfg = {k: ck["cfg"][k] for k in ("vocab", "n_layer", "d", "n_head")}
    m = GPT(**cfg).to(device)
    missing, unexpected = m.load_state_dict(ck["state"], strict=False)
    assert not unexpected, unexpected
    assert set(missing) <= {"aux.weight", "aux.bias"}, missing
    extra = dict(ck.get("extra") or {})
    extra["has_aux"] = not missing
    m.eval()
    return m, extra
