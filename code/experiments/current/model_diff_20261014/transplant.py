# ruff: noqa: E501,SIM115  (research scripts: long result lines, bare open(), short loop names)
"""Which weights carry the depth-3 pop fix? Transplant components between a pretrained model and its RL'd final model
and measure pop accuracy (k = 1, 2) on spine-3 ladder proofs (EI s1's texts), plus proof-token CE on short pretraining
proofs as a 'nothing else broke' check.
  components: block j attention (ln1, qkv, proj, ln_attn_out), block j MLP (ln2, fc1, fc2, ln_mlp_out),
              block j q/k only, block j v/o only; embeddings; ln_f/head
python transplant.py g0   (pair = g0_pre / g0_fin)  -> out/transplant_<pair>.json
"""

import json
import sys

import torch
from tiny import Tiny

torch.set_num_threads(6)
pair = sys.argv[1] if len(sys.argv) > 1 else "g0"
C = torch.load("out/corpus.pt")
CK = {
    "g0": "v3_untied-grpo_s0",
    "g1": "v3_untied-grpo_s1",
    "e0": "v2_untied-ei_s0",
    "e1": "v2_untied-ei_s1",
}
sd = {
    s: torch.load(
        f"ckpt/runs/long-rl-20261014-{CK[pair]}/{'pretrain' if s == 'pre' else 'final'}.pt",
        map_location="cpu",
    )["state"]
    for s in ("pre", "fin")
}
starts = C["starts"].tolist() + [len(C["ids"])]
src = C["src"].tolist()
LAD3, PRE = C["SRCS"].index("lad3"), C["SRCS"].index("pre")
lad = [i for i in range(len(src)) if src[i] == LAD3][:200]
pre = [i for i in range(len(src)) if src[i] == PRE][:300]
ids = C["ids"].long()
pk = C["pop_k"].long()
npr = C["n_prompt"].tolist()


def comp_keys(j, part):
    b = f"blocks.{j}."
    return {
        "attn": [
            b + x
            for x in (
                "ln1.weight",
                "ln1.bias",
                "qkv.weight",
                "qkv.bias",
                "proj.weight",
                "proj.bias",
                "ln_attn_out.weight",
                "ln_attn_out.bias",
            )
        ],
        "mlp": [
            b + x
            for x in (
                "ln2.weight",
                "ln2.bias",
                "fc1.weight",
                "fc1.bias",
                "fc2.weight",
                "fc2.bias",
                "ln_mlp_out.weight",
                "ln_mlp_out.bias",
            )
        ],
    }[part]


def mix(base, donor, keys, qk_only=None):
    s = dict(sd[base])
    for k in keys:
        if qk_only is not None and k.endswith(("qkv.weight", "qkv.bias")):
            t = sd[base][k].clone()
            D = t.shape[0] // 3
            sl = slice(0, 2 * D) if qk_only else slice(2 * D, 3 * D)
            t[sl] = sd[donor][k][sl]
            s[k] = t
        elif (
            qk_only is not None
            and not k.endswith(("qkv.weight", "qkv.bias"))
            and not (not qk_only and "proj" in k)
        ):
            continue
        else:
            s[k] = sd[donor][k]
    return s


@torch.no_grad()
def score(state):
    m = Tiny(state)
    hit = {1: 0, 2: 0}
    tot = {1: 0, 2: 0}
    for i in lad:
        x = ids[starts[i] : starts[i + 1]]
        h = m.resid(x[None])
        lg = m.logits_from(h[-1], 6)[0]
        p = pk[starts[i] : starts[i + 1]]
        for j in (p >= 1).nonzero().flatten().tolist():
            k = min(int(p[j]), 2)
            tot[k] += 1
            hit[k] += int(lg[j].argmax() == x[j + 1])
    ce, nt = 0.0, 0
    for i in pre:
        x = ids[starts[i] : starts[i + 1]]
        lg = m.logits_from(m.resid(x[None])[-1], 6)[0]
        a = npr[i]
        ce += float(torch.nn.functional.cross_entropy(lg[a - 1 : -1], x[a:], reduction="sum"))
        nt += len(x) - a
    return {"pop1": hit[1] / tot[1], "pop2": hit[2] / tot[2], "ce_pre": ce / nt}


res = {}
for base, donor in (("pre", "fin"), ("fin", "pre")):
    res[f"{base}"] = score(sd[base])
    print(f"\n== base {pair}_{base}, donor {pair}_{donor}:  clean {res[base]}", flush=True)
    rows = [("emb", ["emb.weight"]), ("head", ["ln_f.weight", "ln_f.bias", "head.weight"])]
    for j in range(6):
        rows += [(f"L{j} attn", comp_keys(j, "attn")), (f"L{j} mlp", comp_keys(j, "mlp"))]
    for name, keys in rows:
        r = score(mix(base, donor, keys))
        res[f"{base}<-{name}"] = r
        print(
            f"  {name:>10}: pop1 {r['pop1']:.0%} pop2 {r['pop2']:.0%} ce_pre {r['ce_pre']:.4f}",
            flush=True,
        )
    for j in range(6):
        for tag, qk in (("qk", True), ("vo", False)):
            r = score(mix(base, donor, comp_keys(j, "attn"), qk_only=qk))
            res[f"{base}<-L{j} {tag}"] = r
            print(
                f"  L{j} {tag:>5}: pop1 {r['pop1']:.0%} pop2 {r['pop2']:.0%} ce_pre {r['ce_pre']:.4f}",
                flush=True,
            )
json.dump(res, open(f"out/transplant_{pair}.json", "w"), indent=1)
