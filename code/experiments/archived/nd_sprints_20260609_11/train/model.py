"""Small GPT for ND proofs, with randomized position offsets and batched
KV-cache sampling.

Randomized offsets: during training each sequence gets position ids
offset + [0..len), offset ~ U[0, block_size - len]. The model therefore sees
every absolute position during training even when all proofs are short, which
removes "unseen position" as an explanation for length-generalization failure.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Attn(nn.Module):
    def __init__(self, d, h, dropout):
        super().__init__()
        self.h = h
        self.dk = d // h
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.dropout = dropout

    def forward(self, x, attn_bias=None, cache=None):
        """x: (B,T,d). cache: dict with 'k','v' past tensors or None.
        attn_bias: optional (B,1,T,Tk) boolean mask, True = masked out.
        Returns out, new_cache_kv (k,v incl past)."""
        B, T, d = x.shape
        q, k, v = self.qkv(x).split(d, dim=2)
        q = q.view(B, T, self.h, self.dk).transpose(1, 2)
        k = k.view(B, T, self.h, self.dk).transpose(1, 2)
        v = v.view(B, T, self.h, self.dk).transpose(1, 2)
        if cache is not None and cache.get('k') is not None:
            k = torch.cat([cache['k'], k], dim=2)
            v = torch.cat([cache['v'], v], dim=2)
        if cache is not None:
            cache['k'], cache['v'] = k, v
        if attn_bias is not None:
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=~attn_bias,
                dropout_p=self.dropout if self.training else 0.0)
        else:
            out = F.scaled_dot_product_attention(
                q, k, v, is_causal=(T > 1),
                dropout_p=self.dropout if self.training else 0.0)
        out = out.transpose(1, 2).reshape(B, T, d)
        return self.proj(out)


class Block(nn.Module):
    def __init__(self, d, h, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = Attn(d, h, dropout)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(
            nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d),
            nn.Dropout(dropout))

    def forward(self, x, attn_bias=None, cache=None):
        x = x + self.attn(self.ln1(x), attn_bias=attn_bias, cache=cache)
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size, block_size=512, n_layer=4, n_head=8,
                 d_model=256, dropout=0.0):
        super().__init__()
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(block_size, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([Block(d_model, n_head, dropout)
                                     for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx, pos_ids=None, targets=None, loss_mask=None):
        """Training/scoring path (no cache)."""
        B, T = idx.shape
        if pos_ids is None:
            pos_ids = torch.arange(T, device=idx.device).unsqueeze(0).expand(B, T)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos_ids))
        for b in self.blocks:
            x = b(x)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            l = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                targets.reshape(-1), reduction='none')
            l = l.view(B, T)
            if loss_mask is not None:
                loss = (l * loss_mask).sum() / loss_mask.sum().clamp(min=1)
            else:
                loss = l.mean()
        return logits, loss

    @torch.no_grad()
    def sample(self, prompts, max_new, temperature=1.0, top_k=None,
               eos_id=None, pad_id=0, device='cuda'):
        """Batched sampling with KV cache. prompts: ragged list of id lists.
        Left-pads prompts; pads are masked out of attention. Returns list of
        generated id lists (continuation only, including eos if produced)."""
        self.eval()
        B = len(prompts)
        lens = [len(p) for p in prompts]
        Lp = max(lens)
        idx = torch.full((B, Lp), pad_id, dtype=torch.long, device=device)
        pos = torch.zeros((B, Lp), dtype=torch.long, device=device)
        pad_mask = torch.ones((B, Lp), dtype=torch.bool, device=device)
        for i, p in enumerate(prompts):
            L = len(p)
            idx[i, Lp - L:] = torch.tensor(p, dtype=torch.long, device=device)
            pos[i, Lp - L:] = torch.arange(L, device=device)
            pad_mask[i, Lp - L:] = False  # False = real token
        caches = [{'k': None, 'v': None} for _ in self.blocks]

        # prefill: causal mask + key padding mask
        causal = torch.triu(torch.ones(Lp, Lp, dtype=torch.bool, device=device), 1)
        bias = causal.unsqueeze(0).unsqueeze(0) | pad_mask[:, None, None, :]
        # rows that are fully masked (pad query positions attending to nothing
        # before them): ensure each query can attend to itself
        eye = torch.eye(Lp, dtype=torch.bool, device=device)
        bias = bias & ~eye.unsqueeze(0).unsqueeze(0)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        for b, c in zip(self.blocks, caches):
            x = b(x, attn_bias=bias, cache=c)
        logits = self.head(self.ln_f(x[:, -1:, :])).squeeze(1)

        done = torch.zeros(B, dtype=torch.bool, device=device)
        out_ids = []
        cur_pos = torch.tensor(lens, device=device)
        key_mask = pad_mask  # (B, Tk) True = masked
        for step in range(max_new):
            lg = logits / max(temperature, 1e-6)
            if top_k is not None:
                v, _ = torch.topk(lg, top_k)
                lg[lg < v[:, [-1]]] = -float('inf')
            if temperature <= 0.001:
                nxt = lg.argmax(-1)
            else:
                nxt = torch.multinomial(F.softmax(lg, dim=-1), 1).squeeze(-1)
            nxt = torch.where(done, torch.full_like(nxt, pad_id), nxt)
            out_ids.append(nxt)
            done = done | (nxt == eos_id)
            if done.all() or Lp + step + 1 >= self.block_size:
                break
            # decode one token
            key_mask = torch.cat(
                [key_mask, torch.zeros((B, 1), dtype=torch.bool, device=device)], 1)
            bias = key_mask[:, None, None, :]
            p = torch.minimum(cur_pos + step,
                              torch.full_like(cur_pos, self.block_size - 1))
            x = self.tok_emb(nxt.unsqueeze(1)) + self.pos_emb(p.unsqueeze(1))
            for b, c in zip(self.blocks, caches):
                x = b(x, attn_bias=bias, cache=c)
            logits = self.head(self.ln_f(x[:, -1:, :])).squeeze(1)
        outs = torch.stack(out_ids, 1).tolist() if out_ids else [[] for _ in range(B)]
        res = []
        for i, row in enumerate(outs):
            seq = []
            for t in row:
                seq.append(t)
                if t == eos_id:
                    break
            # strip trailing pads emitted after done
            while seq and seq[-1] == pad_id:
                seq.pop()
            res.append(seq)
        return res
