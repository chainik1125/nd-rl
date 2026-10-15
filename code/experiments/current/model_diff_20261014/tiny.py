# ruff: noqa: E501,SIM115  (research scripts: long result lines, bare open(), short loop names)
"""Standalone re-implementation of long_rl_20261014's untied model (run 198 trunk: six Peri-LN blocks, ALiBi on half
the heads, NoPE on the rest), so a pod needs only torch and a state dict. Checked against pretrain.LoopedGPT by
`python tiny.py <ckpt>` (max |logit diff| printed; run where nd.py imports).

Tiny(state).resid(x) -> (7, B, T, 384): embedding, then the residual stream after each block.
Tiny(state).logits_from(h, layer) -> logits continuing the forward from resid index `layer`.
Padding: right-pad and ignore pad positions (causal attention never looks right).
"""

import torch
import torch.nn.functional as F

N_LAYER, D, N_HEAD = 6, 384, 8
SLOPES = [2.0 ** (-8.0 * (i + 1) / 4) for i in range(4)] + [0.0] * 4


class Tiny(torch.nn.Module):
    def __init__(self, state):
        super().__init__()
        self.p = {k: v.float() for k, v in state.items()}

    def to(self, dev):
        self.p = {k: v.to(dev) for k, v in self.p.items()}
        return self

    def _ln(self, x, name):
        return F.layer_norm(x, (D,), self.p[name + ".weight"], self.p[name + ".bias"])

    def _lin(self, x, name):
        return F.linear(x, self.p[name + ".weight"], self.p.get(name + ".bias"))

    def bias(self, T, dev):
        pos = torch.arange(T, device=dev)
        rel = (pos[:, None] - pos[None, :]).float()
        s = torch.tensor(SLOPES, device=dev).view(1, N_HEAD, 1, 1)
        return (-s * rel).masked_fill(rel < 0, float("-inf"))

    def block(self, j, x, bias, head_mask=None):
        b = f"blocks.{j}."
        B, T, _ = x.shape
        a = self._ln(x, b + "ln1")
        q, k, v = self._lin(a, b + "qkv").view(B, T, 3, N_HEAD, D // N_HEAD).permute(2, 0, 3, 1, 4)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
        if head_mask is not None:  # (H,) multiplier per head output, for ablations
            y = y * head_mask.view(1, N_HEAD, 1, 1)
        y = self._lin(y.transpose(1, 2).reshape(B, T, D), b + "proj")
        x = x + self._ln(y, b + "ln_attn_out")
        m = self._lin(F.gelu(self._lin(self._ln(x, b + "ln2"), b + "fc1")), b + "fc2")
        return x + self._ln(m, b + "ln_mlp_out")

    @torch.no_grad()
    def resid(self, x):
        bias = self.bias(x.shape[1], x.device)
        h = F.embedding(x, self.p["emb.weight"])
        hs = [h]
        for j in range(N_LAYER):
            h = self.block(j, h, bias)
            hs.append(h)
        return torch.stack(hs)

    @torch.no_grad()
    def logits_from(self, h, layer):
        bias = self.bias(h.shape[1], h.device)
        for j in range(layer, N_LAYER):
            h = self.block(j, h, bias)
        return F.linear(self._ln(h, "ln_f"), self.p["head.weight"])


if __name__ == "__main__":
    import json
    import sys

    import nd

    m, tok = nd.load(sys.argv[1])
    t = Tiny(torch.load(sys.argv[1], map_location="cpu").get("state"))
    r = json.loads(open(nd.E + "/data/rl.jsonl").readline())
    ids, _ = nd.encode(tok, r["prompt"], r["proof"], 5)
    x = torch.tensor([ids, ids[:-7] + [0] * 7])
    with torch.no_grad():
        ref = m(x)
    h = t.resid(x)
    print(
        "max |logit diff|",
        float((t.logits_from(h[-1], 6) - ref).abs().max()),
        "| from layer 3",
        float((t.logits_from(h[3], 3) - ref).abs().max()),
    )
