"""CPU checks on a tiny random Olmo3 (transformers 4.57.1). No real weights. Run: python -I test_steer_cpu.py"""

import torch
from transformers import Olmo3Config, Olmo3ForCausalLM

torch.manual_seed(0)
cfg = Olmo3Config(
    vocab_size=200,
    hidden_size=64,
    intermediate_size=128,
    num_hidden_layers=6,
    num_attention_heads=4,
    num_key_value_heads=2,
    max_position_embeddings=256,
    sliding_window=8,
    layer_types=["sliding_attention"] * 3
    + ["full_attention"]
    + ["sliding_attention"] * 1
    + ["full_attention"],
    tie_word_embeddings=False,
)
m = Olmo3ForCausalLM(cfg).eval()
L = cfg.num_hidden_layers
ids = torch.randint(0, 200, (2, 20))
out = {}
raw = {}
hooks = [
    m.model.layers[i].register_forward_hook(
        lambda mod, a, o, i=i: raw.__setitem__(i, o if torch.is_tensor(o) else o[0])
    )
    for i in range(L)
]
with torch.no_grad():
    r = m(ids, output_hidden_states=True)
hs = r.hidden_states
print("len(hidden_states) =", len(hs), "(expect L+1 =", L + 1, ")")
print("hs[0] is embedding output:", torch.equal(hs[0], m.model.embed_tokens(ids)))
for l in range(1, L):
    assert torch.equal(hs[l], raw[l - 1]), l
print("hs[l] == raw output of layer l-1 for l=1..L-1: True")
print(
    "hs[L] == raw output of layer L-1 ?",
    torch.equal(hs[L], raw[L - 1]),
    "; == final norm of it ?",
    torch.allclose(hs[L], m.model.norm(raw[L - 1]), atol=1e-6),
)


# steering by hook on layer l-1 output (= residual after l layers = hs[l])
def steer_logits(l, vec):
    h = m.model.layers[l - 1].register_forward_hook(lambda mod, a, o: o + vec)
    try:
        with torch.no_grad():
            rr = m(ids, output_hidden_states=True)
    finally:
        h.remove()
    return rr


v = torch.randn(64)
for l in (1, 3, 5):
    seen = {}
    pre = (
        m.model.layers[l].register_forward_pre_hook(
            lambda mod, a, kw: seen.__setitem__("x", a[0] if a else kw["hidden_states"]),
            with_kwargs=True,
        )
        if l < L
        else None
    )
    rr = steer_logits(l, v)
    if pre:
        pre.remove()
    print(
        f"l={l}: HF-recorded hs[l] is the PRE-hook value (unchanged): {torch.equal(rr.hidden_states[l], hs[l])}",
        "| input of layer l (seen by a pre-hook) == hs[l] + v:",
        (torch.allclose(seen["x"], hs[l] + v, atol=1e-6) if pre else "n/a (l=L)"),
        "| logits changed:",
        not torch.allclose(rr.logits, r.logits),
    )
print("alpha=0 identical logits:", torch.equal(steer_logits(3, torch.zeros(64)).logits, r.logits))

# layer 0 = embedding edit is exact: add v to every embedding row == steering at hs[0]
emb = m.model.embed_tokens.weight.data.clone()
m.model.embed_tokens.weight.data += v
with torch.no_grad():
    re = m(ids, output_hidden_states=True)
m.model.embed_tokens.weight.data.copy_(emb)
h0 = m.model.embed_tokens.register_forward_hook(lambda mod, a, o: o + v)
with torch.no_grad():
    rh = m(ids)
h0.remove()
print("embedding-row edit == hook at hs[0]:", torch.allclose(re.logits, rh.logits, atol=1e-5))

# the sublayer output is not constant, so no weight edit gives a constant mid-stream addition:
# show post-norm sublayer output depends on the input (norm is on the output, no bias anywhere)
print("bias params in model:", [n for n, _ in m.named_parameters() if n.endswith("bias")])


# vLLM's decoder layer order (olmo2.py L190-210), re-implemented with its RMSNorm.forward_native (weight applied AFTER the
# cast back to input dtype) against HF's layer (weight applied BEFORE the cast), in bf16.
def vllm_rms(x, w, eps):
    od = x.dtype
    x = x.float()
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return x.to(od) * w


mb = Olmo3ForCausalLM(cfg).to(torch.bfloat16).eval()
with torch.no_grad():
    rb = mb(ids, output_hidden_states=True)
    lay = mb.model.layers[2]
    x = rb.hidden_states[2]
    hf_out = rb.hidden_states[3]
    # vLLM layer = same sub-modules, vllm-style norms; reuse HF attention/mlp for the sublayer bodies
    pos = torch.arange(20)[None].expand(2, -1)
    # recompute HF layer via module to get the sublayer bodies
    resid = x
    att_raw = (
        lay.self_attn
    )  # needs rotary/mask; instead compare only the norm semantics on random tensors:
    t = torch.randn(2, 20, 64, dtype=torch.bfloat16) * 5
    a = lay.post_attention_layernorm(t)
    b = vllm_rms(t, lay.post_attention_layernorm.weight, cfg.rms_norm_eps)
    lay.post_attention_layernorm.weight.data = torch.randn(64).to(torch.bfloat16)
    a = lay.post_attention_layernorm(t)
    b = vllm_rms(t, lay.post_attention_layernorm.weight, cfg.rms_norm_eps)
    print(
        "bf16 norm semantics, HF vs vLLM-native: max abs diff",
        (a.float() - b.float()).abs().max().item(),
        "max rel",
        ((a.float() - b.float()).abs() / a.float().abs().clamp_min(1e-3)).max().item(),
        "(1 bf16 ulp ~ 0.0078 rel)",
    )
