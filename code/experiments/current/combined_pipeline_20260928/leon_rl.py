# imports the fork via harness_frozen, which is not resolvable here
# pyright: reportMissingImports=false
"""Leon's RL recipe as a drop-in for the frozen harness's stage_rl (same signature and keys).

Source: origin/leon/home-rolled-rl @ 0f749e2,
code/experiments/current/rl_hillclimb_run2_20260924/recipes/rl_champion2_n060_fullpool.py
(GEAR run-2 champion: expert iteration evolved by agents; abs tokenizer, nd_verify reward).

Ported, tokenizer-independent:
  - small rounds: a 400-target window per round from one rotating queue over the targets; once the
    first pass is used up, refills put the hardest first (unsolved, then lowest latest solve rate)
  - knapsack sample allocation (mean k=32 per target, 4..128): unseen targets by a length prior,
    seen ones by their latest solve rate
  - training-time temperature 1.0 (eval stays at the harness's 0.8)
  - per round: fresh AdamW, 135 steps, lr 3e-4 -> 3e-5 cosine after 35 warmup steps, batch 128
  - GEM loss (Li et al. 2025, arXiv 2408.16673), beta 0.7, instead of cross-entropy
  - mix: <= 4 proofs per solved target, repeated max(1, round(8 * (1 - solve rate))) times, plus
    20,000 random cap-6 records (as the harness)
Not ported: the line-level verifier-guided sampler and its CUDA-graph version (they parse abs
proof lines), the abs index cap, hindsight relabelling (a no-op in Leon's run 1), schema cover.

Budget: matched in samples to the harness's stage (RL_ROUNDS * N_TARGETS * K = 192,000), so the
comparison is the recipe, not more compute. That is 15 rounds of 400 x 32.
"""

import collections
import json
import math
import time

import harness_frozen as H
import torch
import torch.nn.functional as F

CFG = dict(
    window=400,
    k_min=4,
    k_max=128,
    k_len_prior={7: 2, 8: 3, 9: 4, 10: 6},  # Leon's weights; 11+ get 8
    k_len_prior_max=8,
    temperature=1.0,
    hard_refill=1200,
    max_per_thm=4,
    rare_weight_max=8,
    ft_steps=135,
    ft_batch=128,
    lr=3e-4,
    min_lr=3e-5,
    warmup=35,
    weight_decay=0.1,
    betas=(0.9, 0.95),
    grad_clip=1.0,
    gem_beta=0.7,
)


def gem_loss_on(model, x, m, beta):
    """Leon's gem_loss_on (h='logsigmoid'), unchanged: -sum_v q(v) w(v) (log p(y) - log p(v)) with
    q = softmax(z / beta) and w = sigmoid(0.01 (z_v - z_y)) both detached; proof tokens only."""
    logits = model(x[:, :-1])
    mk = m[:, 1:]
    z = logits[mk].float()
    y = x[:, 1:][mk]
    with torch.no_grad():
        w = torch.sigmoid(0.01 * (z - z.gather(-1, y[:, None])))
        q = F.softmax(z / beta, -1)
    lp = F.log_softmax(z, -1)
    return -(q * w * (lp.gather(-1, y[:, None]) - lp)).sum(-1).mean()


def alloc_k(targets, k, rate):
    w = []
    for t in targets:
        p = rate.get(t["name"])
        if p is None:
            w.append(CFG["k_len_prior"].get(t["n_lines"], CFG["k_len_prior_max"]))
        else:
            w.append(1 if p >= 0.5 else 4 if p > 0 else 8)
    tot, sw = k * len(targets), sum(w)
    return [max(CFG["k_min"], min(CFG["k_max"], round(tot * x / sw))) for x in w]


def finetune_gem(model, tok, data, steps, rng):
    opt = torch.optim.AdamW(
        model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"], betas=CFG["betas"]
    )
    warm, lr, lo = CFG["warmup"], CFG["lr"], CFG["min_lr"]
    bs = min(CFG["ft_batch"], len(data))
    model.train()
    t0, perm = time.time(), []
    for s in range(1, steps + 1):
        cur = (
            lr * s / warm
            if s < warm
            else lo + 0.5 * (lr - lo) * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm)))
        )
        for g in opt.param_groups:
            g["lr"] = cur
        if len(perm) < bs:
            perm = list(range(len(data)))
            rng.shuffle(perm)
        x, m = H.mk_batch(data, [perm.pop() for _ in range(bs)], tok, rng, H.DEV)
        with H.autocast():
            loss = gem_loss_on(model, x, m, CFG["gem_beta"])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CFG["grad_clip"])
        opt.step()
    model.eval()
    return {"steps": steps, "secs": time.time() - t0, "last_loss": loss.item()}


def stage_rl(a, tok, model, targets, train_recs, rng):
    t0 = time.time()
    budget = a.rl_rounds * len(targets) * a.k  # the harness stage's sample count
    ft_steps = CFG["ft_steps"] if not a.smoke else a.ft_steps
    found = collections.defaultdict(dict)
    rate = {}
    queue = list(targets)  # already a seeded random sample
    rounds, total, refills, r = [], 0, 0, 0
    while total < budget:
        r += 1
        rt = time.time()
        n_win = min(CFG["window"], len(targets))
        if len(queue) < n_win:
            queued = {t["name"] for t in queue}
            rest = [t for t in targets if t["name"] not in queued]
            rng.shuffle(rest)
            rest.sort(key=lambda t: (bool(found[t["name"]]), rate.get(t["name"], 0.0)))
            queue += rest[: CFG["hard_refill"]]
            refills += 1
        win, queue = queue[:n_win], queue[n_win:]
        ks = alloc_k(win, a.k, rate)
        if sum(ks) > budget - total:  # last round: shrink to the samples left
            f = (budget - total) / sum(ks)
            ks = [max(1, int(k * f)) for k in ks]
        prompts = [t["prompt"] for t, k in zip(win, ks, strict=True) for _ in range(k)]
        ts = time.time()
        flat = H.generate(
            model,
            tok,
            prompts,
            greedy=False,
            temperature=CFG["temperature"],
            batch=H.SAMPLE_BATCH,
            seed=a.seed * 1000 + r,
            max_new=a.max_new,
        )
        s_secs = time.time() - ts
        per, i = [], 0
        for k in ks:
            per.append(flat[i : i + k])
            i += k
        rows = H.judge(win, per, "n_lines")
        new = n_ok = 0
        for t, row, k in zip(win, rows, ks, strict=True):
            rate[t["name"]] = row["n_ok"] / k
            n_ok += row["n_ok"]
            for p, wl, pl in zip(
                row["proofs"], row["written_lens"], row["pruned_lens"], strict=False
            ):
                pn = H.norm(p)
                if pn not in found[t["name"]]:
                    found[t["name"]][pn] = {"proof": p, "written": wl, "pruned": pl, "round": r}
                    new += 1
        total += len(prompts)
        solved = {n for n, d in found.items() if d}
        ls, ge = H.lstar(solved, targets)
        mix = []
        for t in targets:
            fs = list(found[t["name"]].values())
            rng.shuffle(fs)
            w = max(1, round(CFG["rare_weight_max"] * (1 - rate.get(t["name"], 0.5))))
            for x in fs[: CFG["max_per_thm"]]:
                mix += [(t["prompt"], x["proof"])] * w
        n_rl = len(mix)
        mix += [
            (x["prompt"], x["proof"])
            for x in rng.sample(train_recs, min(a.retain, len(train_recs)))
        ]
        data = [(tok.encode_prompt(p), tok.encode_proof(q)) for p, q in mix]
        ft = finetune_gem(model, tok, data, ft_steps, rng) if n_rl else {"steps": 0, "secs": 0}
        rounds.append(
            {
                "round": r,
                "window": len(win),
                "refills": refills,
                "sample_acc": n_ok / len(prompts),
                "new_proofs": new,
                "targets_solved": len(solved),
                "lstar": ls,
                "ge": ge,
                "mix_rl_records": n_rl,
                "samples_cum": total,
                "sampling": {
                    "samples": len(prompts),
                    "secs": s_secs,
                    "samples_per_sec": len(prompts) / s_secs,
                },
                "finetune": ft,
                "secs": time.time() - rt,
            }
        )
        H.log(
            f"leon r{r}: acc {n_ok / len(prompts):.4f}, +{new} proofs, "
            f"solved {len(solved)}/{len(targets)}, L*={ls}, {total}/{budget} samples, "
            f"{len(prompts) / s_secs:.0f} samples/s, round {rounds[-1]['secs']:.0f}s"
        )
    H.save_checkpoint(a, model, tok, "rl_final")
    with open(f"{a.out}/found_targets.jsonl", "w") as f:
        for t in targets:
            for x in found[t["name"]].values():
                f.write(
                    json.dumps({"name": t["name"], "thm": t["thm"], "L_true": t["n_lines"], **x})
                    + "\n"
                )
    return {"rounds": rounds, "secs": time.time() - t0, "targets": len(targets), "recipe": "leon"}
