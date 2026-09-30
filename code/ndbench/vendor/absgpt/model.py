"""Small from-scratch decoder-only transformer with RoPE and a KV cache for sampling."""
import math, torch, torch.nn as nn, torch.nn.functional as F


def rope_cache(T, hd, device, base=10000.0):
    pos = torch.arange(T, device=device, dtype=torch.float32)
    inv = 1.0 / (base ** (torch.arange(0, hd, 2, device=device, dtype=torch.float32) / hd))
    ang = pos[:, None] * inv[None, :]
    return torch.cos(ang), torch.sin(ang)  # (T, hd/2)


def apply_rope(x, cos, sin):
    # x: (B, H, T, hd); cos/sin: (T, hd/2) or (B, T, hd/2)
    x1, x2 = x[..., ::2], x[..., 1::2]
    if cos.dim() == 2:
        cos, sin = cos[None, None], sin[None, None]
    else:
        cos, sin = cos[:, None], sin[:, None]
    y1 = x1 * cos - x2 * sin
    y2 = x1 * sin + x2 * cos
    return torch.stack([y1, y2], -1).flatten(-2)


class Block(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.fc1 = nn.Linear(d, 4 * d)
        self.fc2 = nn.Linear(4 * d, d)
        self.h = h

    def attn(self, x, cos, sin, mask, cache):
        B, T, D = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if cache is not None:
            # preallocated KV cache (avoids per-step torch.cat growth / allocator fragmentation)
            if 'k' not in cache:
                cache['k'] = k.new_empty(B, self.h, cache['max'], D // self.h)
                cache['v'] = v.new_empty(B, self.h, cache['max'], D // self.h)
                cache['n'] = 0
            n0, n1 = cache['n'], cache['n'] + T
            cache['k'][:, :, n0:n1] = k
            cache['v'][:, :, n0:n1] = v
            cache['n'] = n1
            k, v = cache['k'][:, :, :n1], cache['v'][:, :, :n1]
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        else:
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=mask is None)
        return self.proj(y.transpose(1, 2).reshape(B, T, D))

    def forward(self, x, cos, sin, mask=None, cache=None):
        x = x + self.attn(self.ln1(x), cos, sin, mask, cache)
        x = x + self.fc2(F.gelu(self.fc1(self.ln2(x))))
        return x


class GPT(nn.Module):
    def __init__(self, vocab, n_layer=4, d=256, n_head=8, max_len=1024):
        super().__init__()
        self.cfg = dict(vocab=vocab, n_layer=n_layer, d=d, n_head=n_head, max_len=max_len)
        self.emb = nn.Embedding(vocab, d)
        self.blocks = nn.ModuleList([Block(d, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.hd = d // n_head
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx, pos=None, mask=None, caches=None):
        """idx: (B,T). pos: (B,T) positions (for left-padded batches) or None -> arange.
        mask: bool (B,1,T,S) attention mask or None (causal). caches: list of dicts per layer."""
        B, T = idx.shape
        if pos is None:
            cos, sin = rope_cache(T, self.hd, idx.device)
        else:
            cos_all, sin_all = rope_cache(int(pos.max().item()) + 1, self.hd, idx.device)
            cos, sin = cos_all[pos], sin_all[pos]
        x = self.emb(idx)
        for i, b in enumerate(self.blocks):
            x = b(x, cos, sin, mask, None if caches is None else caches[i])
        return self.head(self.ln_f(x))


def save_ckpt(path, model, tok_mode, extra=None):
    torch.save({'cfg': model.cfg, 'state': model.state_dict(), 'tok_mode': tok_mode, 'extra': extra or {}}, path)


def load_ckpt(path, device='cpu'):
    from tokenizer import Tokenizer
    ck = torch.load(path, map_location=device)
    tk = Tokenizer(ck['tok_mode'])
    m = GPT(**ck['cfg']).to(device)
    m.load_state_dict(ck['state'])
    m.eval()
    return m, tk, ck.get('extra', {})
