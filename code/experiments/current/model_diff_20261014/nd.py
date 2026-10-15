# ruff: noqa: E501  (research scripts: long result lines, bare open(), short loop names)
"""Shared helpers for model diffing on long_rl_20261014 checkpoints (untied arm only).

load(path) -> model, tok          (ckpt {'state':..} or a plain state dict)
encode(tok, prompt, proof, shift) -> (ids, n_prompt)
labels(tok, ids, n_prompt) -> per-token dict of structural labels (box depth, token kind, ...)
resid(model, x) -> (n_layer+1, B, T, D) residual stream: embedding, then after each block
Paths: CKPT[name] for the six checkpoints downloaded into ckpt/.
"""

import os
import sys

os.environ.setdefault("ND_ARCH", "untied")
os.environ.setdefault(
    "ND_FORK",
    "/private/tmp/claude-501/-Users-robbie-repos-nd-rl/a1d02fb4-ba43-48c4-838b-e227293b6c2a/scratchpad/fork51604b3",
)
E = os.environ.get("LONG_RL", "/Users/robbie/repos/nd-rl/code/experiments/current/long_rl_20261014")
sys.path.insert(0, E)
import harness as H  # noqa: E402,F401  (puts overlay + fork on sys.path, registers the overlay tokenizer)
import pretrain as P  # noqa: E402
import torch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
_R = f"{HERE}/ckpt/runs/long-rl-20261014-"
CKPT = {
    "g0_pre": _R + "v3_untied-grpo_s0/pretrain.pt",
    "g0_fin": _R + "v3_untied-grpo_s0/final.pt",
    "g1_pre": _R + "v3_untied-grpo_s1/pretrain.pt",
    "g1_fin": _R + "v3_untied-grpo_s1/final.pt",
    "e0_pre": _R + "v2_untied-ei_s0/pretrain.pt",
    "e0_fin": _R + "v2_untied-ei_s0/final.pt",
    "e1_pre": _R + "v2_untied-ei_s1/pretrain.pt",
    "e1_fin": _R + "v2_untied-ei_s1/final.pt",
}


def load(path, device="cpu"):
    ck = torch.load(path, map_location="cpu")
    state = ck.get("state", ck)
    tok = P.make_tokenizer(P.TOK_MODE)
    model = P.LoopedGPT(tok.vocab_size, P.N_LAYER, P.D, P.N_HEAD, 0)
    model.blocks = torch.nn.ModuleList(P.PeriLNBlock(P.D, P.N_HEAD) for _ in range(P.N_LAYER))
    model.load_state_dict(state)
    assert model.untied
    return model.eval().to(device), tok


def encode(tok, prompt, proof, shift=0):
    """prompt+proof ids; name tokens shifted by `shift` (lean_seq's random start offset, 0 = as encoded)."""
    p, q = tok.encode_prompt(prompt), tok.encode_proof(proof)
    q = [x + shift if x >= tok.ref0 else x for x in q]
    return p + q, len(p)


def rand_shift(tok, ids, rng):
    mx = max((x - tok.ref0 + 1 for x in ids if x >= tok.ref0), default=0)
    return rng.randint(0, P.MAXN - mx) if mx else 0


KINDS = {
    "(": "lparen",
    ")": "rparen",
    "fun": "fun",
    "=>": "arrow",
    "by": "by",
    "have": "have",
    "exact": "exact",
    ";": "semi",
    ":": "colon",
    ":=": "assign",
    "Or.elim": "orelim",
    "Or.inl": "orI",
    "Or.inr": "orI",
    "⟨": "anglel",
    "⟩": "angler",
    ",": "comma",
    ".1": "proj",
    ".2": "proj",
    "False": "False",
    ".elim": "elim",
    "Classical.byContradiction": "dn",
    "hh": "hh",
    "<eos>": "eos",
}


def labels(tok, ids, n_prompt):
    """Per proof-token structural labels (index i in ids, i >= n_prompt):
    depth   number of open `( fun ( n : A ) => by` boxes the token sits in (the box's own `(`..`by` header counts as inside)
    kind    token class (KINDS, or 'name', 'atom', 'conn', 'prem')
    in_f    inside a formula (after a `have n :` / box binder `:` and before `:=` / `)`=>)
    next_open  the NEXT token opens a box (this token's position is where the model decides to write `( fun`)
    """
    s = [tok.itos[x] for x in ids]
    out = []
    stack = []  # one entry per open '(' : True if it opened a box
    depth = 0
    in_f = False
    for i in range(n_prompt, len(s)):
        t = s[i]
        nxt = s[i + 1] if i + 1 < len(s) else None
        opens_box = t == "(" and nxt == "fun" and (i + 2 < len(s) and s[i + 2] == "(")
        if opens_box:
            depth += 1
        kind = KINDS.get(t) or (
            "name"
            if t.startswith("n") and t[1:].isdigit()
            else "prem"
            if t.startswith("h") and t[1:].isdigit()
            else "atom"
            if t in P.__dict__.get("ATOMS", ()) or t in "PQRSTUWXYZ"
            else "conn"
        )
        out.append(
            {
                "i": i,
                "tok": t,
                "depth": depth,
                "kind": kind,
                "in_f": in_f,
                "next_open": nxt == "("
                and i + 2 < len(s)
                and s[i + 2] == "fun"
                and i + 3 < len(s)
                and s[i + 3] == "(",
            }
        )
        # update state after this token
        if t == "(":
            stack.append(opens_box)
        elif t == ")" and stack and stack.pop():
            depth -= 1
        if t == ":" and i > 0 and (s[i - 1].startswith("n") or s[i - 1] == "hh"):
            in_f = True
        elif t in (":=",) or (t == ")" and nxt == "=>"):
            in_f = False
    return out


@torch.no_grad()
def resid(model, x):
    """(L+1, B, T, D): embedding output, then the residual stream after each block. Causal, no padding mask:
    pad on the right and ignore pad positions."""
    B, T = x.shape
    pos = torch.arange(T, device=x.device)[None]
    rel = pos[:, None, :, None] - pos[:, None, None, :]
    bias = (-model.slopes * rel.to(model.slopes.dtype)).masked_fill(rel < 0, float("-inf"))
    h = model.emb(x)
    hs = [h]
    for b in model.blocks:
        h = b(h, None, None, bias, None)
        hs.append(h)
    return torch.stack(hs)


@torch.no_grad()
def logits_from(model, h_last):
    return model.head(model.ln_f(h_last))


@torch.no_grad()
def run_from(model, h, layer, x_len):
    """continue the forward from the residual stream after block `layer` (h: B,T,D) -> logits"""
    T = h.shape[1]
    pos = torch.arange(T, device=h.device)[None]
    rel = pos[:, None, :, None] - pos[:, None, None, :]
    bias = (-model.slopes * rel.to(model.slopes.dtype)).masked_fill(rel < 0, float("-inf"))
    for b in model.blocks[layer:]:
        h = b(h, None, None, bias, None)
    return model.head(model.ln_f(h))


@torch.no_grad()
def attn_maps(model, x):
    """(L, H, T, T) attention probabilities of each block (same computation as PeriLNBlock.attn, no cache)."""
    B, T = x.shape
    assert B == 1
    pos = torch.arange(T, device=x.device)[None]
    rel = pos[:, None, :, None] - pos[:, None, None, :]
    bias = (-model.slopes * rel.to(model.slopes.dtype)).masked_fill(rel < 0, float("-inf"))
    h = model.emb(x)
    out = []
    for b in model.blocks:
        a = b.ln1(h)
        D = a.shape[-1]
        q, k, v = b.qkv(a).view(B, T, 3, b.h, D // b.h).permute(2, 0, 3, 1, 4)
        s = (q @ k.transpose(-1, -2)) / (D // b.h) ** 0.5 + bias
        out.append(torch.softmax(s, -1)[0])
        h = b(h, None, None, bias, None)
    return torch.stack(out)
