#!/usr/bin/env python3
"""Proof of concept for the pretrain workstream loop, "option A at every step".

SUPERSEDED for the agent loop by ../autoresearch/ (harness.py + pretrain.py); see its README for why
the pretrain stage was split out. This file is kept as-is and still runs.


  pretrain  Dan's cap-6 Stage-1 file, `abs` codec, 4-layer d=256 model, wall-clock capped
  RL        plain expert iteration (ladder rung T1) on a fixed subsample of the ladder RL-target
            pool, wall-clock capped; rounds run until the next one would not fit
  eval      frozen Stage-1 model vs final model, equal attempts, on the dev half of the ladder
            transfer pool (split by renaming-class hash); primary number = dev theorems solved
            at L_true >= 9; L*, per-bin rates, held-out greedy and pruned written lengths beside it

Runs on top of a checkout of dan-pandori/nd-takehome (ND_FORK, pinned in remote.sh); imports its
model / tokenizer / sampler / verifier / judge unchanged. Everything this writes goes to --out.

  python3 pipeline.py --out artifacts/run0 --pretrain_secs 300 --rl_secs 300
  python3 pipeline.py --out /tmp/smoke --smoke          # CPU, seconds, catches plumbing bugs
"""

import argparse
import collections
import hashlib
import json
import math
import os
import random
import sys
import time

FORK = os.environ.get("ND_FORK", "/workspace/nd-takehome")
sys.path.insert(0, FORK)
import torch  # noqa: E402
from eval_set import judge, wilson  # noqa: E402
from model import GPT, save_ckpt  # noqa: E402
from normalize import norm  # noqa: E402
from sample import generate  # noqa: E402
from tokenizer import Tokenizer  # noqa: E402
from train import batch as mk_batch  # noqa: E402
from train import load as load_recs  # noqa: E402
from train import loss_on  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def read(fn):
    with open(fn) as stream:
        return [json.loads(line) for line in stream if line.strip()]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def autocast():
    return torch.autocast("cuda", dtype=torch.bfloat16, enabled=(DEV == "cuda"))


# ---------------------------------------------------------------- training (shared by both stages)


def train_timed(model, tok, data, *, max_secs, max_steps, bs, lr, min_lr, warmup, wd, rng, tag):
    """AdamW + warmup + cosine. The cosine runs over *time* when max_secs binds, over steps
    otherwise, so a wall-clock cap is also the schedule rather than a truncation of one."""
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd, betas=(0.9, 0.95))
    model.train()
    t0 = time.time()
    perm, step, last_loss = [], 0, float("nan")
    while True:
        elapsed = time.time() - t0
        frac = max(elapsed / max_secs if max_secs else 0.0, step / max_steps if max_steps else 0.0)
        if frac >= 1.0:
            break
        step += 1
        cur_lr = (
            lr * step / warmup
            if step < warmup
            else min_lr + 0.5 * (lr - min_lr) * (1 + math.cos(math.pi * frac))
        )
        for g in opt.param_groups:
            g["lr"] = cur_lr
        if len(perm) < bs:
            perm = list(range(len(data)))
            rng.shuffle(perm)
        idxs = [perm.pop() for _ in range(bs)]
        x, m = mk_batch(data, idxs, tok, rng, DEV)
        with autocast():
            loss = loss_on(model, x, m)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 200 == 0:
            last_loss = loss.item()
            log(f"{tag} step {step} loss {last_loss:.4f} lr {cur_lr:.2e} {time.time() - t0:.0f}s")
    model.eval()
    secs = time.time() - t0
    return {
        "steps": step,
        "secs": secs,
        "steps_per_sec": step / secs,
        "records_seen": step * bs,
        "last_loss": last_loss,
    }


@torch.no_grad()
def val_loss(model, tok, held, rng):
    model.eval()
    tot, n = 0.0, 0
    for s in range(0, len(held), 256):
        idxs = range(s, min(s + 256, len(held)))
        with autocast():
            tot += loss_on(model, *mk_batch(held, idxs, tok, rng, DEV)).item() * len(idxs)
        n += len(idxs)
    return tot / max(n, 1)


# ---------------------------------------------------------------- bookkeeping


def lstar(solved_names, recs, need=5):
    """L* = largest L with >= `need` distinct theorems solved at L_true >= L.
    Records carry L_true in n_lines.
    """
    ls_solved = [r["n_lines"] for r in recs if r["name"] in solved_names]
    ge = {L: sum(1 for x in ls_solved if x >= L) for L in range(7, 15)}
    return max([L for L, c in ge.items() if c >= need], default=0), ge


def by_bin(solved_names, recs):
    by = collections.defaultdict(lambda: [0, 0])
    for r in recs:
        by[r["n_lines"]][0] += r["name"] in solved_names
        by[r["n_lines"]][1] += 1
    return {
        str(L): {"solved": k, "n": n, "rate": k / n, "ci": list(wilson(k, n))}
        for L, (k, n) in sorted(by.items())
    }


def sample_and_judge(model, tok, recs, k, temperature, seed, batch, max_new):
    prompts = [r["prompt"] for r in recs for _ in range(k)]
    t0 = time.time()
    flat = generate(
        model,
        tok,
        prompts,
        greedy=(k == 1 and temperature == 0),
        temperature=temperature,
        batch=batch,
        seed=seed,
        max_new=max_new,
    )
    secs = time.time() - t0
    rows = judge(recs, [flat[i * k : (i + 1) * k] for i in range(len(recs))], "n_lines")
    return rows, {"samples": len(prompts), "secs": secs, "samples_per_sec": len(prompts) / secs}


def greedy_rate(model, tok, recs, batch, max_new):
    rows, _ = sample_and_judge(model, tok, recs, 1, 0, 0, batch, max_new)
    return sum(r["solved"] for r in rows) / max(len(rows), 1)


# ---------------------------------------------------------------- stages


def stage_pretrain(a, tok, train_data, held, rng):
    model = GPT(tok.vocab_size, a.n_layer, a.d, a.n_head).to(DEV)
    n_params = model.n_params()
    log(f"pretrain: {n_params:,} params, {len(train_data):,} records, cap {a.pretrain_secs}s")
    assert n_params <= a.param_cap, f"{n_params} > param cap {a.param_cap}"
    st = train_timed(
        model,
        tok,
        train_data,
        max_secs=a.pretrain_secs,
        max_steps=a.pretrain_steps,
        bs=a.bs,
        lr=a.lr,
        min_lr=a.lr / 10,
        warmup=200,
        wd=0.1,
        rng=rng,
        tag="pretrain",
    )
    st["n_params"] = n_params
    st["val_loss"] = val_loss(model, tok, held, rng)
    st["heldout_greedy"] = greedy_rate(model, tok, a.heldout_recs, a.batch, a.max_new)
    log(
        f"pretrain done: {st['steps']} steps "
        f"({st['records_seen'] / len(train_data):.1f} epochs), val {st['val_loss']:.4f}, "
        f"held-out greedy {st['heldout_greedy']:.3f}"
    )
    save_ckpt(f"{a.out}/stage1.pt", model, tok.mode, extra=st)
    return model, st


def stage_rl(a, tok, model, targets, train_recs, rng):
    """Plain expert iteration: sample k per target, keep verified proofs (start-index
    normalised, <= max_per_thm per theorem, x rl_weight), fine-tune on them + `retain`
    random Stage-1 records; repeat while a round fits.
    """
    t0 = time.time()
    found = collections.defaultdict(dict)  # name -> norm -> record
    rounds, round_secs = [], None
    r = 0
    while True:
        elapsed = time.time() - t0
        if r > 0 and (elapsed + round_secs > a.rl_secs or r >= a.rl_max_rounds):
            break
        r += 1
        rt = time.time()
        rows, samp = sample_and_judge(
            model, tok, targets, a.k, a.temperature, a.seed * 1000 + r, a.batch, a.max_new
        )
        new = 0
        for t, row in zip(targets, rows, strict=False):
            for p, wl, pl in zip(
                row["proofs"], row["written_lens"], row["pruned_lens"], strict=False
            ):
                pn = norm(p)
                if pn not in found[t["name"]]:
                    found[t["name"]][pn] = {"proof": p, "written": wl, "pruned": pl, "round": r}
                    new += 1
        solved = {n for n, d in found.items() if d}
        ls, ge = lstar(solved, targets)
        mix = []
        for t in targets:
            fs = list(found[t["name"]].values())
            rng.shuffle(fs)
            for x in fs[: a.max_per_thm]:
                mix += [(t["prompt"], x["proof"])] * a.rl_weight
        n_rl = len(mix)
        mix += [
            (x["prompt"], x["proof"])
            for x in rng.sample(train_recs, min(a.retain, len(train_recs)))
        ]
        data = [(tok.encode_prompt(p), tok.encode_proof(q)) for p, q in mix]
        ft = (
            train_timed(
                model,
                tok,
                data,
                max_secs=0,
                max_steps=a.ft_steps,
                bs=a.ft_bs,
                lr=a.ft_lr,
                min_lr=a.ft_lr / 10,
                warmup=50,
                wd=0.1,
                rng=rng,
                tag=f"ei r{r}",
            )
            if n_rl
            else {"steps": 0, "secs": 0}
        )
        round_secs = time.time() - rt
        rounds.append(
            {
                "round": r,
                "sample_acc": sum(x["n_ok"] for x in rows) / samp["samples"],
                "new_proofs": new,
                "targets_solved": len(solved),
                "lstar": ls,
                "ge": ge,
                "mix_rl_records": n_rl,
                "sampling": samp,
                "finetune": ft,
                "secs": round_secs,
            }
        )
        log(
            f"ei r{r}: acc {rounds[-1]['sample_acc']:.4f}, +{new} proofs, "
            f"solved {len(solved)}/{len(targets)}, L*={ls}, ge9={ge[9]}, "
            f"{samp['samples_per_sec']:.0f} samples/s, round {round_secs:.0f}s"
        )
    secs = time.time() - t0
    save_ckpt(f"{a.out}/final.pt", model, tok.mode, extra={"rounds": len(rounds), "secs": secs})
    with open(f"{a.out}/found_targets.jsonl", "w") as f:
        for t in targets:
            for x in found[t["name"]].values():
                f.write(
                    json.dumps({"name": t["name"], "thm": t["thm"], "L_true": t["n_lines"], **x})
                    + "\n"
                )
    return {
        "rounds": rounds,
        "secs": secs,
        "over_budget": secs > a.rl_secs,
        "targets": len(targets),
    }


def stage_eval(a, tok, model, tag, dev_pool, heldout_recs):
    rows, samp = sample_and_judge(
        model, tok, dev_pool, a.eval_k, a.temperature, a.seed * 1000 + 777, a.batch, a.max_new
    )
    solved = {r["name"] for r in rows if r["solved"]}
    ls, ge = lstar(solved, dev_pool)
    pruned_max = max((max(r["pruned_lens"]) for r in rows if r["solved"]), default=0)
    written_max = max((max(r["written_lens"]) for r in rows if r["solved"]), default=0)
    out = {
        "attempts_per_theorem": a.eval_k,
        "solved": len(solved),
        "n": len(dev_pool),
        "solved_ge9": ge[9],
        "lstar": ls,
        "ge": ge,
        "by_bin": by_bin(solved, dev_pool),
        "max_written": written_max,
        "max_pruned": pruned_max,
        "heldout_greedy": greedy_rate(model, tok, heldout_recs, a.batch, a.max_new),
        "sampling": samp,
    }
    log(
        f"eval {tag}: solved {len(solved)}/{len(dev_pool)}, ge9 {ge[9]}, "
        f"L*={ls}, max pruned {pruned_max}, "
        f"held-out greedy {out['heldout_greedy']:.3f}"
    )
    with open(f"{a.out}/eval_{tag}.jsonl", "w") as f:
        for r in rows:
            f.write(
                json.dumps(
                    {
                        k: r[k]
                        for k in (
                            "name",
                            "n_lines",
                            "solved",
                            "n_ok",
                            "proofs",
                            "written_lens",
                            "pruned_lens",
                        )
                    }
                )
                + "\n"
            )
    return out


def dev_split(transfer):
    """Deterministic half of the transfer pool by renaming-class hash.
    The other half is never scored by this loop.
    """
    return [r for r in transfer if int(hashlib.sha1(r["key"].encode()).hexdigest(), 16) % 2 == 0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pretrain_secs", type=float, default=300)
    ap.add_argument("--pretrain_steps", type=int, default=0, help="0 = time-capped only")
    ap.add_argument("--rl_secs", type=float, default=300)
    ap.add_argument("--rl_max_rounds", type=int, default=8)
    ap.add_argument("--param_cap", type=int, default=10_000_000)
    ap.add_argument("--n_layer", type=int, default=4)
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--n_head", type=int, default=8)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--n_targets", type=int, default=1500)
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--eval_k", type=int, default=32)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--max_new", type=int, default=512)
    ap.add_argument("--batch", type=int, default=1024, help="sampling batch")
    ap.add_argument("--ft_steps", type=int, default=600)
    ap.add_argument("--ft_bs", type=int, default=128)
    ap.add_argument("--ft_lr", type=float, default=3e-4)
    ap.add_argument("--retain", type=int, default=20000)
    ap.add_argument("--max_per_thm", type=int, default=4)
    ap.add_argument("--rl_weight", type=int, default=4)
    ap.add_argument(
        "--smoke", action="store_true", help="tiny everything, for a CPU plumbing check"
    )
    a = ap.parse_args()
    if a.smoke:
        a.pretrain_secs, a.rl_secs, a.rl_max_rounds = 15, 20, 2
        a.n_targets, a.k, a.eval_k, a.batch, a.bs, a.ft_steps, a.retain, a.max_new = (
            8,
            2,
            2,
            16,
            16,
            5,
            50,
            120,
        )
    os.makedirs(a.out, exist_ok=True)
    with open(f"{a.out}/args.json", "w") as stream:
        json.dump(vars(a), stream, indent=1)
    torch.manual_seed(a.seed)
    rng = random.Random(a.seed)
    T0 = time.time()

    tok = Tokenizer("abs")
    train_recs = read(f"{FORK}/data/train.jsonl")
    assert all(r["n_lines"] <= 6 for r in train_recs), "pretrain file violates cap 6"
    train_data = load_recs(f"{FORK}/data/train.jsonl", tok, cap=6)
    heldout_all = read(f"{FORK}/data/heldout.jsonl")
    # the file is length-ordered (1,000 each of lengths 2-6): stride it, don't take a prefix
    heldout_all = heldout_all[:: len(heldout_all) // 2000][:2000]
    held = [(tok.encode_prompt(r["prompt"]), tok.encode_proof(r["proof"])) for r in heldout_all]
    a.heldout_recs = heldout_all if not a.smoke else heldout_all[:16]
    targets_all = read(f"{FORK}/data/ladder/rl_targets.jsonl")
    transfer = read(f"{FORK}/data/ladder/transfer.jsonl")
    dev_pool = dev_split(transfer)
    if a.smoke:
        dev_pool = dev_pool[:8]
    targets = random.Random(a.seed).sample(targets_all, min(a.n_targets, len(targets_all)))
    tk, dk, hk = (
        {t["key"] for t in targets},
        {r["key"] for r in dev_pool},
        {r["key"] for r in train_recs},
    )
    assert not (tk & dk) and not (tk & hk) and not (dk & hk), "renaming-class leak between pools"
    log(
        f"pools: train {len(train_recs):,}, targets {len(targets)} (of {len(targets_all)}), "
        f"dev-transfer {len(dev_pool)} "
        f"(of {len(transfer)}), held-out {len(a.heldout_recs)}; device {DEV}"
    )

    model, pre = stage_pretrain(a, tok, train_data, held, rng)
    frozen = stage_eval(a, tok, model, "frozen", dev_pool, a.heldout_recs)
    rl = stage_rl(a, tok, model, targets, train_recs, rng)
    final = stage_eval(a, tok, model, "final", dev_pool, a.heldout_recs)

    summary = {
        "args": {k: v for k, v in vars(a).items() if k != "heldout_recs"},
        "device": DEV,
        "gpu": torch.cuda.get_device_name(0) if DEV == "cuda" else "cpu",
        "pretrain": pre,
        "rl": rl,
        "eval_frozen": frozen,
        "eval_final": final,
        "total_secs": time.time() - T0,
        "primary": {
            "dev_solved_ge9_final": final["solved_ge9"],
            "dev_solved_ge9_frozen": frozen["solved_ge9"],
            "lstar_final": final["lstar"],
            "lstar_frozen": frozen["lstar"],
        },
    }
    with open(f"{a.out}/summary.json", "w") as stream:
        json.dump(summary, stream, indent=1)
    log(
        f"SUMMARY dev-transfer solved at L_true>=9: frozen {frozen['solved_ge9']} -> "
        f"final {final['solved_ge9']} (of {sum(1 for r in dev_pool if r['n_lines'] >= 9)}); "
        f"L* {frozen['lstar']} -> {final['lstar']}; pretrain {pre['secs']:.0f}s, "
        f"RL {rl['secs']:.0f}s ({len(rl['rounds'])} rounds), total {summary['total_secs']:.0f}s"
    )


if __name__ == "__main__":
    main()
