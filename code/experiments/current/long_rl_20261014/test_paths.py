"""Plumbing test for the paths a smoke run cannot reach with an untrained model: gold proofs encoded as a model would
write them must be accepted by the Lean gate, judged with the record's body length, and train through both RL updates.

    ND_FORK=<fork at 51604b3> ND_RL=grpo python test_paths.py
"""
import os
import random
import sys

import harness as H  # noqa: E402  (sets sys.path: overlay, fork, here)
import pretrain as P  # noqa: E402

pools = H.read(f"{H.DATA}/dev.jsonl")
recs = random.Random(0).sample(pools, 12)
train = H.read(f"{H.DATA}/pretrain.jsonl.gz")[:2000]
model, tok, _ = P.pretrain(train, 5, H.DEV, 0, H.log)
# gold proofs through the gate and judge, exactly as sampled ids would go
per = []
for r in recs:
    ids = tok.encode_proof(r["proof"])
    nd = tok.decode(ids)
    per.append([(ids, nd, tok.last_text)])
gated = H.gate(tok, [r["prompt"] for r in recs], [p[0][1] for p in per], [p[0][2] for p in per])
rows = H.judge(recs, [[(p[0][0], g, not g.startswith("LEAN"))] for p, g in zip(per, gated)])
assert all(r["solved"] for r in rows), [r["name"] for r in rows if not r["solved"]]
assert all(r["written"][0] == rec["body"] for r, rec in zip(rows, recs) if rec["body_exact"]), "body length mismatch"
print("gate + judge: 12/12 gold proofs accepted, body lengths match")
# a corrupted proof must be rejected by Lean
bad = tok.encode_proof(recs[0]["proof"])
j = next(i for i, x in enumerate(bad) if x in tok.atom_ids and i > 20)
bad[j] = tok.atom_ids[(tok.atom_ids.index(bad[j]) + 1) % 10]
g = H.gate(tok, [recs[0]["prompt"]], [tok.decode(bad)], [tok.last_text])
assert g[0].startswith("LEAN"), g[0]
print("gate: corrupted proof rejected:", g[0][:40])
# both RL updates on these proofs
rng = random.Random(0)
items = [(tok.encode_prompt(r["prompt"]), tok.encode_proof(r["proof"])) for r in recs]
adv = [0.5 if i % 2 else -0.5 for i in range(len(items))]
retain = [(tok.encode_prompt(r["prompt"]), tok.encode_proof(r["proof"])) for r in train[:12]]
gen = P.torch.Generator(device=H.DEV)
matrices = [p for p in model.blocks.parameters() if p.ndim == 2] + [model.head.weight]
opt = P.torch.optim.AdamW([p for p in model.parameters() if all(p is not q for q in matrices)], lr=1e-4)
muon = P.MuonStep(matrices, H.DEV)
autocast = P.torch.autocast("cuda", enabled=False)
# PPO surrogate: clipped at 0.8 / 1.28, dual-clipped at 3a for a < 0
r = P.torch.tensor([0.5, 1.0, 1.5, 0.5, 1.0, 1.5, 5.0])
a = P.torch.tensor([1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0])
assert P.torch.allclose(P.ppo_surrogate(r, a), P.torch.tensor([0.5, 1.0, 1.28, -0.8, -1.0, -1.5, -3.0])), P.ppo_surrogate(r, a)
# with zero learning rates the policy never moves, so every ratio is exactly 1 (old log p matches the update's)
lrs = P.GRPO_LR, P.GRPO_MUON_LR
P.GRPO_LR = P.GRPO_MUON_LR = 0.0
before = [p.detach().clone() for p in model.parameters()]
_, steps, ratio = P.pg_update(model, tok, items, adv, retain, opt, muon, 0, gen, rng, autocast, H.DEV)
assert abs(ratio["ratio"] - 1) < 1e-4 and ratio["clip_frac"] == 0, ratio
assert all(P.torch.equal(p, q) for p, q in zip(model.parameters(), before)), "zero lr moved the weights"
P.GRPO_LR, P.GRPO_MUON_LR = lrs
loss, steps, ratio = P.pg_update(model, tok, items, adv, retain, opt, muon, 0, gen, rng, autocast, H.DEV)
assert not all(P.torch.equal(p, q) for p, q in zip(model.parameters(), before)), "pg_update did not move the weights"
print(f"pg_update: {steps} steps, loss {loss:.4f}, {ratio}")
# rollout_stats against a one-rollout-at-a-time Categorical reference
rows = [{"prompt": r["prompt"], "rollouts": [(tok.encode_proof(r["proof"]), i % 2 == 0), (tok.encode_proof(r["proof"]), True)]}
        for i, r in enumerate(recs)]
rs = P.rollout_stats(model, tok, rows, H.DEV)
ref, n = {k: 0.0 for k in ("entropy", "nll", "entropy_t", "nll_t")}, 0
with P.torch.no_grad():
    for row in rows:
        for ids, _ in row["rollouts"]:
            p = tok.encode_prompt(row["prompt"])
            x = P.torch.tensor([p + ids], device=H.DEV)
            logits = model(x[:, :-1])[0, len(p) - 1 :].float()
            for suffix, temp in (("", 1.0), ("_t", P.GRPO_TEMP)):
                dist = P.torch.distributions.Categorical(logits=logits / temp)
                ref["entropy" + suffix] += float(dist.entropy().sum())
                ref["nll" + suffix] -= float(dist.log_prob(x[0, len(p) :]).sum())
            n += len(ids)
assert all(abs(rs[k] - v / n) < 1e-3 for k, v in ref.items()), (rs, {k: v / n for k, v in ref.items()})
assert rs["all_pass"] == len(recs) // 2 and rs["all_fail"] == 0 and rs["tokens"] == n / (2 * len(recs)), rs
print("rollout_stats:", P.fmt_stats(rs))
# KL probes: exact KL against a per-position Categorical reference, zero for identical models
other = P.copy.deepcopy(model)
with P.torch.no_grad():
    for prm in other.parameters():
        prm.add_(0.02 * P.torch.randn_like(prm))
kl = P.kl_pair(model, other, tok, items, H.DEV)
ref, n = {"kl_pq": 0.0, "kl_qp": 0.0}, 0
with P.torch.no_grad():
    for p_ids, q_ids in items:
        x = P.torch.tensor([p_ids + q_ids], device=H.DEV)
        dp, dq = (P.torch.distributions.Categorical(logits=mod(x[:, :-1])[0, len(p_ids) - 1 :].float()) for mod in (model, other))
        ref["kl_pq"] += float(P.torch.distributions.kl_divergence(dp, dq).sum())
        ref["kl_qp"] += float(P.torch.distributions.kl_divergence(dq, dp).sum())
        n += len(q_ids)
assert all(abs(kl[k] - v / n) < 1e-4 for k, v in ref.items()) and kl["kl_pq"] > 0, (kl, ref, n)
same = P.kl_pair(model, model, tok, items, H.DEV)
assert abs(same["kl_pq"]) < 1e-6 and abs(same["kl_qp"]) < 1e-6 and abs(same["nll_p"] - same["nll_q"]) < 1e-6, same
print(f"kl_pair: KL(p||q) {kl['kl_pq']:.5f} KL(q||p) {kl['kl_qp']:.5f} match the reference; 0 for identical models")


class FakeOracle:
    class a:
        out = os.environ.get("TMPDIR", "/tmp")
    gold = recs


probe = P.Probe(model, tok, train, FakeOracle, H.DEV, H.log)
rows5 = [{"prompt": r["prompt"], "rollouts": [(tok.encode_proof(r["proof"]), True)]} for r in recs]
probe.measure(model, rows5, 4)  # unchanged weights: the step KL is 0 and the pretrained reference is the model itself
h = probe.history[-1]
assert h["step"] < 1e-6 and h["pre_roll"] < 1e-6 and h["pre_pt"] < 1e-6, h
P.GRPO_LR = P.GRPO_MUON_LR = 1e-2
P.pg_update(model, tok, items, adv, retain, opt, muon, 100, gen, rng, autocast, H.DEV)
P.GRPO_LR, P.GRPO_MUON_LR = lrs
probe.measure(model, rows5, 5)  # after an update: both KLs > 0, step KL equals the drift (one step from pre) and a checkpoint
h = probe.history[-1]
assert h["step"] > 0 and abs(h["step"] - h["pre_roll_rev"]) < 1e-6 and h["pre_pt"] > 0, h
ck = os.path.join(FakeOracle.a.out, "rl_round_005.pt")
assert set(P.torch.load(ck)) == set(model.state_dict()), "checkpoint keys"
os.remove(ck)
print("probe:", {k: round(v, 5) for k, v in h.items()})
P.FT_RECORDS = 64  # CPU: a few steps are enough to exercise the path
loss, steps = P.finetune(model, tok, items * 4 + retain, rng, H.DEV)
print(f"finetune: {steps} steps, loss {loss:.4f}")
print("OK")
