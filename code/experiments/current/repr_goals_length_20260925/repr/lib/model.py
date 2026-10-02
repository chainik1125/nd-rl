"""Small from-scratch decoder-only transformer with a KV cache for sampling: the frozen lib/model.py GPT
(ab3/frozen/lib/model.py) with a choice of positional encoding for the representation experiment (ab3/repr).

Every variant has exactly the frozen parameters and nn.Linear layers (so the compute meter charges the same per
token). Only what the attention sees as "position" changes. The variant is stored in the checkpoint cfg; a key
is present only when it is not the default, so an abs checkpoint has the frozen cfg and loads with frozen code.

  pe='rope'  (default)  RoPE on token positions, base 10000: the frozen code path, bit-identical.
  pe='none'             no positional encoding (NoPE, Kazemnejad et al. 2023): causal mask only. Positions
                        passed in are ignored.
  pe='line'             two-part rotary. A position is a pair (line, off): line = the number of line breaks
                        (';' and 'PRF') before the token (prompt = line 0, first proof line = line 1, QED sits
                        at the start of the line after the last one), off = the token's offset within its line
                        (or within the prompt). Of the hd/2 = 16 rotary pairs per head, the first `line_pairs`
                        (8) rotate by line, the rest by off; each half uses RoPE's geometric spectrum
                        (base 10000) over its own pairs. Positions are (B, T, 2) tensors; with pos=None they are
                        computed from the tokens (rows must start at the sequence start). cfg['seps'] holds the
                        token ids of ';' and 'PRF'.
  pos_train='rand'      (pe='rope' only; read by the training code, not by forward): during training each
                        sequence's positions are a sorted uniform random subset of range(pos_range) (Ruoss et
                        al. 2023); sampling and evaluation use positions 0, 1, 2, ... unless told otherwise.

Position helpers used by every sampler (so the KV-cache decode matches a full forward):
  init_pos(keep)        positions of left-padded prompt tokens ((B, L) or (B, L, 2))
  next_pos(pos, tok)    position of the token that follows token `tok` sitting at `pos`
  seq_positions(ids)    positions of one sequence (list of token ids) from its start, as a list
"""
import math, torch, torch.nn as nn, torch.nn.functional as F

BASE_CFG = ('vocab', 'n_layer', 'd', 'n_head', 'max_len')
DEFAULTS = dict(pe='rope', line_pairs=8, seps=None, pos_train='arange', pos_range=1024)


def rope_cache(T, hd, device, base=10000.0):
    pos = torch.arange(T, device=device, dtype=torch.float32)
    inv = 1.0 / (base ** (torch.arange(0, hd, 2, device=device, dtype=torch.float32) / hd))
    ang = pos[:, None] * inv[None, :]
    return torch.cos(ang), torch.sin(ang)  # (T, hd/2)


def inv_freq(n_pairs, device, base=10000.0):
    """RoPE's spectrum for a rotary block of n_pairs pairs: base ** (-2i / (2 n_pairs)), i = 0..n_pairs-1."""
    return 1.0 / (base ** (torch.arange(0, 2 * n_pairs, 2, device=device, dtype=torch.float32) / (2 * n_pairs)))


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


def line_positions(idx, seps):
    """(B, T) token ids, rows starting at the sequence start -> (B, T, 2) long (line, off)."""
    brk = torch.zeros_like(idx, dtype=torch.bool)
    for s in seps:
        brk |= idx == s
    b = brk.long()
    line = b.cumsum(1) - b                                  # breaks strictly before t
    T = idx.size(1)
    t = torch.arange(T, device=idx.device).expand_as(idx)
    last = torch.where(brk, t, torch.full_like(t, -1)).cummax(1).values   # last break at or before t
    lastb = torch.cat([torch.full_like(last[:, :1], -1), last[:, :-1]], 1)  # ... strictly before t
    off = t - lastb - 1
    return torch.stack([line, off], -1)


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
        if cos is not None:
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
    def __init__(self, vocab, n_layer=4, d=256, n_head=8, max_len=1024, **variant):
        super().__init__()
        bad = set(variant) - set(DEFAULTS)
        if bad:
            raise TypeError(f'unknown GPT variant keys {sorted(bad)}')
        v = dict(DEFAULTS, **variant)
        if isinstance(v['seps'], tuple):
            v['seps'] = list(v['seps'])
        assert v['pe'] in ('rope', 'none', 'line'), v['pe']
        assert v['pos_train'] in ('arange', 'rand'), v['pos_train']
        assert v['pos_train'] == 'arange' or v['pe'] == 'rope', 'pos_train=rand needs pe=rope'
        assert v['pe'] != 'line' or (v['seps'] and 0 < v['line_pairs'] < (d // n_head) // 2), v
        self.cfg = dict(vocab=vocab, n_layer=n_layer, d=d, n_head=n_head, max_len=max_len)
        self.cfg.update({k: x for k, x in v.items() if x != DEFAULTS[k]})
        self.pe, self.pos_train, self.pos_range = v['pe'], v['pos_train'], v['pos_range']
        self.line_pairs, self.seps = v['line_pairs'], tuple(v['seps'] or ())
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

    # ---- positions
    def line_rot(self, pos):
        """(..., 2) long (line, off) -> cos, sin of shape (..., hd/2): line_pairs pairs by line, the rest by off."""
        P, Q = self.line_pairs, self.hd // 2 - self.line_pairs
        dev = pos.device
        ang = torch.cat([pos[..., 0:1].float() * inv_freq(P, dev), pos[..., 1:2].float() * inv_freq(Q, dev)], -1)
        return torch.cos(ang), torch.sin(ang)

    def rotary(self, idx, pos):
        """cos, sin for the forward pass (None, None for pe='none')."""
        if self.pe == 'none':
            return None, None
        if self.pe == 'line':
            if pos is None:
                pos = line_positions(idx, self.seps)
            return self.line_rot(pos)
        T = idx.size(1)
        if pos is None:
            return rope_cache(T, self.hd, idx.device)
        cos_all, sin_all = rope_cache(int(pos.max().item()) + 1, self.hd, idx.device)
        return cos_all[pos], sin_all[pos]

    def init_pos(self, keep):
        """Positions of left-padded prompt rows (keep: (B, L) bool, True on real tokens)."""
        p = (keep.cumsum(1) - 1).clamp(min=0)
        if self.pe == 'line':      # a prompt holds no line break before its final PRF: line 0, off = index
            return torch.stack([torch.zeros_like(p), p], -1)
        return p

    def next_pos(self, pos, tok):
        """Position of the token after `tok` (B,), which sits at `pos` ((B,) or (B, 2))."""
        if self.pe != 'line':
            return pos + 1
        brk = torch.zeros_like(tok, dtype=torch.bool)
        for s in self.seps:
            brk |= tok == s
        return torch.stack([pos[:, 0] + brk.long(), torch.where(brk, torch.zeros_like(pos[:, 1]), pos[:, 1] + 1)], -1)

    def seq_positions(self, ids):
        """Positions of one sequence (list of ids from its start) as a python list (ints, or [line, off] pairs)."""
        if self.pe != 'line':
            return list(range(len(ids)))
        out, line, off = [], 0, 0
        for x in ids:
            out.append([line, off])
            if x in self.seps:
                line, off = line + 1, 0
            else:
                off += 1
        return out

    def forward(self, idx, pos=None, mask=None, caches=None):
        """idx: (B,T). pos: (B,T) positions ((B,T,2) for pe='line') or None -> from the sequence start.
        mask: bool (B,1,T,S) attention mask or None (causal). caches: list of dicts per layer."""
        cos, sin = self.rotary(idx, pos)
        x = self.emb(idx)
        for i, b in enumerate(self.blocks):
            x = b(x, cos, sin, mask, None if caches is None else caches[i])
        return self.head(self.ln_f(x))


def variant_kwargs(cfg):
    """The non-base keys of a checkpoint cfg (the variant), for GPT(vocab, n_layer, d, n_head, **variant_kwargs)."""
    return {k: v for k, v in (cfg or {}).items() if k not in BASE_CFG}


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
