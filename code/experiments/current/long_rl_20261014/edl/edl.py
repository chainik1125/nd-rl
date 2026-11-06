#!/usr/bin/env python3
# imports the fork (via harness), which is not resolvable here
# pyright: reportMissingImports=false
"""On the pod: excess description length (Donoway, Joren, Roger & Leike 2026, arXiv 2601.04728) of one checkpoint on
the holdout pool's gold proofs, per body-line band.

  ND_ARCH=untied python3 edl/edl.py <ckpt.pt> --tag NAME --out results/NAME.json [--smoke]

--smoke runs on CPU with two proofs per split, one rate and one order: a plumbing check, not a measurement.

Per band, the holdout theorems are split by a key hash into n train and n test. A fresh copy of the checkpoint
learns the train proofs one at a time; each proof is scored (summed -log p of its proof tokens, nats) just before the
step on it, giving online_nats. EDL = sum(online_nats) - n * L_test(theta_n), where L_test is mean nats per test proof.
L_test is also taken after 1/8, 1/4 and 1/2 of the train proofs, for the EDL-vs-n curve.

The learner is the RL fine-tune's own (pretrain.finetune): Muon on the matrices, AdamW on the rest, grad clip 1, train
mode (the looped arm draws its loop count per step) with the atom and name-offset augmentation, here one proof per
step at a constant rate. The rate is a multiple of FT_LR / FT_MUON_LR from LR_MULTS, run for two data orders; the
dashboard picks the multiple with the smallest mean online code length and charges ln(len(LR_MULTS)) for the choice.
Scoring is eval mode at EVAL_LEVEL loops, unaugmented. Names keep their written offset (n1 first), which the model
was trained to treat as a random offset: that costs about ln(room) nats per proof, in both terms, so it cancels.
"""

import argparse
import hashlib
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import harness  # noqa: E402  (overlay tokenizer, fork model code)

torch = harness.torch
AUTOCAST = dict(device_type="cuda", dtype=torch.bfloat16, enabled=harness.DEV == "cuda")
BANDS = {"7-9": 22, "10-14": 64, "15-19": 64, "20-24": 64, "25-34": 64, "35-49": 32, "50+": 48}  # n per split
LR_MULTS = (0.1, 0.3, 1.0, 3.0)
ORDERS = 2


def split(recs, n):
    """n train and n test records, by a hash of the key (fixed across checkpoints)."""
    ranked = sorted(recs, key=lambda r: hashlib.sha1(f"edl:{r['key']}".encode()).hexdigest())
    return ranked[:n], ranked[n:2 * n]


@torch.no_grad()
def nats(agent, model, tok, items):
    """Summed -log p of each item's proof tokens, in eval mode: a list of floats in items' order."""
    model.eval()
    ids, mask, _, lens = agent.to_gpu(items, tok, harness.DEV)
    out = []
    for i in range(0, len(items), 16):
        idxs = torch.arange(i, min(i + 16, len(items)), device=harness.DEV)
        x, m = agent.gpu_batch(ids, mask, None, lens, idxs, tok, None, augment=False)
        with torch.autocast(**AUTOCAST):
            logits = model(x[:, :-1]).float()
        ce = torch.nn.functional.cross_entropy(logits.transpose(1, 2), x[:, 1:], reduction="none")
        out += (ce * m[:, 1:]).sum(1).tolist()
    return out


def online(agent, model, tok, train, test, mult, seed):
    """One online pass at FT rates x mult: (online_nats, {n': L_test(theta_n')})."""
    matrices = [p for p in model.blocks.parameters() if p.ndim == 2] + [model.head.weight]
    matrix_ids = {id(p) for p in matrices}
    auxiliary = [p for p in model.parameters() if id(p) not in matrix_ids]
    opt = torch.optim.AdamW(auxiliary, lr=agent.FT_LR * mult, weight_decay=0.1, betas=(0.9, 0.95))
    muon = agent.MuonStep(matrices, harness.DEV)
    ids, mask, ref_max, lens = agent.to_gpu(train, tok, harness.DEV)
    gen = torch.Generator(device=harness.DEV)
    gen.manual_seed(seed)
    order = list(range(len(train)))
    random.Random(seed).shuffle(order)
    n = len(train)
    checks = {max(1, n // 8), max(1, n // 4), n // 2, n}
    online_nats, l_test = [], {}
    for step, i in enumerate(order, 1):
        online_nats += nats(agent, model, tok, [train[i]])
        model.train()
        x, m = agent.gpu_batch(ids, mask, ref_max, lens, torch.tensor([i], device=harness.DEV), tok, gen)
        with torch.autocast(**AUTOCAST):
            loss = agent.loss_on(model, x, m)
        model.zero_grad(set_to_none=False)  # grads keep fixed addresses for the Muon CUDA graph
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        muon(agent.FT_MUON_LR * mult)
        if step in checks:
            l_test[step] = sum(nats(agent, model, tok, test)) / len(test)
    return online_nats, l_test


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    assert harness.DEV == "cuda" or a.smoke, "no CUDA on this pod"
    agent = __import__("pretrain")
    train_recs = harness.read(f"{harness.DATA}/pretrain.jsonl.gz")
    bands, mults, orders = BANDS, LR_MULTS, ORDERS
    if a.smoke:
        train_recs, bands, mults, orders = train_recs[:3000], {b: 2 for b in BANDS}, LR_MULTS[:1], 1
    model, tok, _ = agent.pretrain(train_recs, 1e-9, harness.DEV, 0, harness.log)  # zero steps: model + tokenizer
    ck = harness.torch.load(a.ckpt, map_location="cpu", weights_only=True)
    state = ck.get("state", ck)  # pretrain.pt/final.pt wrap the state; rl_round_NNN.pt is bare
    setattr(model, harness.KNOB, harness.EVAL_LEVEL)  # eval mode's loop count; train mode draws its own
    holdout = harness.read(f"{harness.DATA}/holdout.jsonl")
    enc = lambda r: (tok.encode_prompt(r["prompt"]), tok.encode_proof(r["proof"]))  # noqa: E731
    out = {"tag": a.tag, "ckpt": os.path.basename(a.ckpt), "sha256": harness.sha(a.ckpt), "lr_mults": mults,
           "bands": {}}
    for band, n in bands.items():
        train, test = split([r for r in holdout if r["band"] == band], n)
        assert len(test) == n, f"{band}: only {len(train) + len(test)} holdout records"
        train, test = [enc(r) for r in train], [enc(r) for r in test]
        model.load_state_dict(state)
        runs, t0 = [], time.time()
        l_test_0 = sum(nats(agent, model, tok, test)) / n
        for mult in mults:
            for order in range(orders):
                model.load_state_dict(state)
                online_nats, l_test = online(agent, model, tok, train, test, mult, seed=1000 * order + 17)
                runs.append({"lr_mult": mult, "order": order, "online_nats": online_nats, "l_test": l_test,
                             "edl": sum(online_nats) - n * l_test[n]})
        out["bands"][band] = {"n": n, "l_test_0": l_test_0, "runs": runs,
                              "proof_tokens": sum(len(q) for _, q in test) / n}
        best = min(runs, key=lambda r: sum(r["online_nats"]))
        harness.log(f"{a.tag} {band}: L_test {l_test_0:.1f} -> {best['l_test'][n]:.1f} nats/proof at x{best['lr_mult']}, "
                    f"EDL {best['edl']:.0f} nats, {time.time() - t0:.0f}s")
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        with open(a.out, "w") as f:
            json.dump(out, f)


if __name__ == "__main__":
    main()
