#!/usr/bin/env python3
# imports the fork, the combined pipeline and test_time_scaling on the pod, not resolvable here
# pyright: reportMissingImports=false
"""Test-time RL (AlphaProof's TTRL) on a set of hard targets, or its sampling-only control.

TTRL: every target plus its statement-only variants (variants.py) go into one pool. Each round:
  1. sample k proofs of every active item (guided.generate, Leon's RL settings, T 1.0);
  2. a target with a verified proof is solved: it and all its variants leave the pool;
  3. an item solved in more than --master of its k samples is mastered and leaves the pool;
  4. items solved in 0 < rate <= --master of their samples train: <= 4 distinct proofs each,
     repeated max(1, round(8 (1 - rate))) times, plus 20,000 random cap-6 records (Leon's mix);
     items at rate 0 stay in the pool untrained;
  5. one round of Leon's RL update (finetune_gem: GEM loss, fresh AdamW, 135 steps).
One model is adapted on all targets at once, as AlphaProof does for several targets.

--control: the same sampler on the targets alone, no variants, no training, k per round. With
the same total samples it is the compute-matched "just sample more" baseline.

  python3 ttrl.py --run combo-c005_s0 --pool dev --names-file targets_dev64.txt --out x.json
"""

import argparse
import collections
import json
import random
import time

import harness_frozen as H  # leon_rl's name for the frozen harness; sets up the fork's sys.path
import leon_rl
import torch
from guided import STATS as GUIDED_STATS
from guided import generate as guided_generate
from nd_verify import verify_text
from normalize import norm
from ttscale import load, pool
from variants import parse_thm, prompt, variants

GUIDE = {k: leon_rl.CFG[k] for k in ("guide_retries", "guide_rollbacks", "guide_slack")}
GUIDE["guide_seg_max"] = leon_rl.CFG["guide_seg_max"]


def build_items(recs, depth, control):
    items = []
    for r in recs:
        items.append(
            {"name": r["name"], "group": r["name"], "kind": "target", "prompt": r["prompt"]}
        )
        if control:
            continue
        prems, goal = parse_thm(r["thm"])
        for i, v in enumerate(variants(prems, goal, depth)):
            items.append(
                {
                    "name": f"{r['name']}~{i}",
                    "group": r["name"],
                    "kind": v["kind"],
                    "prompt": prompt(v["prems"], v["goal"]),
                }
            )
    for it in items:
        it |= {"active": True, "rates": [], "proofs": {}, "left": None}
    return items


def sample_round(model, tok, act, a, rnd, seed):
    prompts = [it["prompt"] for it in act for _ in range(a.k)]
    t0 = time.time()
    res = guided_generate(model, tok, prompts, a.temp, H.MAX_NEW, a.batch, seed * 1000 + rnd, GUIDE)
    secs = time.time() - t0
    for j, it in enumerate(act):
        n_ok = 0
        for p, r in zip(
            prompts[j * a.k : (j + 1) * a.k], res[j * a.k : (j + 1) * a.k], strict=True
        ):
            if verify_text(p + " " + r)[0]:
                n_ok += 1
                it["proofs"].setdefault(norm(r), r)
        it["rates"].append(n_ok / a.k)
    return {"samples": len(prompts), "secs": secs}


def train_mix(act, train_recs, a, rng):
    mix = []
    for it in act:
        p = it["rates"][-1]
        if 0 < p <= a.master:
            proofs = list(it["proofs"].values())
            rng.shuffle(proofs)
            weight = max(1, round(leon_rl.CFG["rare_weight_max"] * (1 - p)))
            for q in proofs[: leon_rl.CFG["max_per_thm"]]:
                mix += [(it["prompt"], q)] * weight
    n_rl = len(mix)
    mix += [
        (x["prompt"], x["proof"]) for x in rng.sample(train_recs, min(H.RETAIN, len(train_recs)))
    ]
    return mix, n_rl


def summarize_round(rnd, items, act, samp, ft, n_rl, t_round):
    by_kind = collections.defaultdict(lambda: {"items": 0, "any": 0, "rate_sum": 0.0})
    for it in act:
        kind = (
            "target"
            if it["kind"] == "target"
            else f"d{it['kind'].count('/') + 1}:{it['kind'].split('/')[-1]}"
        )
        b = by_kind[kind]
        b["items"] += 1
        b["any"] += it["rates"][-1] > 0
        b["rate_sum"] += it["rates"][-1]
    targets = [it for it in items if it["kind"] == "target"]
    return {
        "round": rnd,
        "active": len(act),
        "sampling": samp,
        "finetune": ft,
        "mix_rl_records": n_rl,
        "targets_solved": sum(
            it["left"] is not None and it["left"][1] == "solved" for it in targets
        ),
        "by_kind": {
            k: {"items": b["items"], "any": b["any"], "mean_rate": b["rate_sum"] / b["items"]}
            for k, b in sorted(by_kind.items())
        },
        "secs": time.time() - t_round,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--ckpt-dir", help="read source/pretrain.py and final.pt from disk, not HF")
    ap.add_argument("--pool", choices=["holdout", "dev"], default="dev")
    ap.add_argument("--names-file", required=True)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--master", type=float, default=0.75)
    ap.add_argument("--temp", type=float, default=leon_rl.CFG["temperature"])
    ap.add_argument("--control", action="store_true")
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t0 = time.time()
    model, tok, seed = load(a.run, a.ckpt_dir)
    with open(a.names_file) as f:
        names = ",".join(x.strip() for x in f if x.strip())
    recs = pool(a.pool, 0, names)
    items = build_items(recs, a.depth, a.control)
    train_recs = H.read(f"{H.FORK}/data/train.jsonl")
    rng = random.Random(seed)
    torch.manual_seed(seed)
    H.log(f"{len(recs)} targets, {len(items) - len(recs)} variants, control={a.control}")
    rounds = []

    def dump():
        out = {
            "args": vars(a),
            "gpu": torch.cuda.get_device_name(0) if H.DEV == "cuda" else "cpu",
            "guided_stats": dict(GUIDED_STATS),
            "secs": time.time() - t0,
            "rounds": rounds,
            "items": [{k: v for k, v in it.items() if k != "active"} for it in items],
        }
        with open(a.out, "w") as f:
            json.dump(out, f)

    for rnd in range(1, a.rounds + 1):
        tr = time.time()
        act = [it for it in items if it["active"]]
        if not act:
            break
        samp = sample_round(model, tok, act, a, rnd, seed)
        for it in act:
            if it["kind"] == "target" and it["rates"][-1] > 0:
                for x in items:
                    if x["group"] == it["group"] and x["active"]:
                        x["active"], x["left"] = (
                            False,
                            (rnd, "solved" if x is it else "group_solved"),
                        )
        for it in act:
            if it["active"] and it["rates"][-1] > a.master:
                it["active"], it["left"] = False, (rnd, "mastered")
        ft, n_rl = {"steps": 0, "secs": 0.0}, 0
        if not a.control:
            mix, n_rl = train_mix([it for it in act if it["active"]], train_recs, a, rng)
            if n_rl:
                data = [(tok.encode_prompt(p), tok.encode_proof(q)) for p, q in mix]
                ft = leon_rl.finetune_gem(model, tok, data, leon_rl.CFG["ft_steps"], rng)
        rounds.append(summarize_round(rnd, items, act, samp, ft, n_rl, tr))
        r = rounds[-1]
        H.log(
            f"round {rnd}: {r['active']} active, {samp['samples']} samples in {samp['secs']:.0f}s, "
            f"+{n_rl} RL records, targets solved {r['targets_solved']}/{len(recs)}"
        )
        dump()
    dump()


if __name__ == "__main__":
    main()
