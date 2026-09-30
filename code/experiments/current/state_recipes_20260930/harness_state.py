#!/usr/bin/env python3
# imports the fork (model, tokenizer, train, ...), which is not resolvable here
# pyright: reportMissingImports=false
"""The frozen autoresearch harness (harness_frozen.py, sha 2aa52f23e7ec) with one switch, STATE_ARM:

  C0  the frozen harness unchanged: whole proofs, lean_seq
  SN  Dan's proof-state format with environment-assigned names (lean_staten, his arm SN-v2)
  SH  Dan's proof-state format with the proof so far prepended (lean_stateh, his arm SH)

For SN / SH: every record is split into (state, action) pairs before pretrain() starts its clock
(state_pairs.attach); sampling goes through Dan's step environment (state_sample.env_generate:
the policy sees a tactic state and writes one step, max 256 tokens per step, 48 steps per attempt);
the RL fine-tune trains on the pairs of FT_BS whole proofs per step (Dan's state_train.batch), loss
on action tokens. Rounds, k, targets, mix, fine-tune schedule, eval pools and the metric are the
frozen harness's. `diff harness_frozen.py harness_state.py` is the whole change.

The frozen harness's docstring follows.

FROZEN harness for the pretrain autoresearch loop. The research agent does not edit this file.

One timed run = harness owns the clock, RL and eval; `pretrain.py` (agent-editable) owns
pretraining:

  1. load frozen data (Dan's cap-6 Stage-1 file, ladder RL-target pool, ladder transfer pool)
  2. call pretrain.pretrain(train_recs, PRETRAIN_SECS, ...) -> (model, tok); charged for everything
     including model construction and torch.compile. Overrun past the grace -> status "timeout".
  3. check n_params <= PARAM_CAP and tok.mode == TOK_MODE (Dan's Lean surface form; the fork's
     sample.generate then gates every sample through Lean, so the RL reward and the eval are
     Lean AND nd_verify)
  4. plain expert iteration (ladder rung T1): RL_ROUNDS rounds, fixed in samples, harness optimizer
  5. eval on the dev half of the transfer pool; METRIC = dev theorems solved at L_true >= 7
  6. SIGALRM at TOTAL_SECS from process start kills everything -> status "timeout"

Budget history: 600 s / 240 s pretrain / 2 rounds / k=32 for runs 000-010 (2026-09-22);
then 1200 / 300 / 4 / 64; from 098 (2026-09-24) the Lean protocol: fork 51604b3, tok lean_seq,
1800 / 300 / 4 / 64; from 100: 2400 s total, because 098 took 1620 / 1684 / 1811 s (one
timeout on a slow pod) with Lean checking and ~1.2x longer sequences.

The holdout half of the transfer pool is never scored here (see holdout_split); score it offline
from final.pt when checking in on the loop. Everything is written to --out.

  python3 harness.py --out artifacts/x                 # on the pod (see pod.py / remote.sh)
  ND_FORK=... python3 harness.py --out /tmp/smoke --smoke   # CPU, ~1 min, plumbing check
"""

import argparse
import collections
import hashlib
import importlib
import json
import math
import os
import random
import shutil
import signal
import sys
import time
import traceback
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
FORK = os.environ.get("ND_FORK", "/workspace/nd-takehome")
sys.path.insert(0, HERE)
sys.path.insert(
    0, FORK
)  # fork first: its train.py provides batch/loss_on; the agent's file is pretrain.py
import state_pairs  # noqa: E402
import torch  # noqa: E402
from eval_set import judge, wilson  # noqa: E402
from normalize import norm  # noqa: E402
from sample import generate as proof_generate  # noqa: E402
from state_sample import env_generate  # noqa: E402  (dan_best-state)
from state_train import batch as pair_batch  # noqa: E402  (dan_best-state: pairs of whole proofs)
from tokenizer import make_tokenizer  # noqa: E402
from train import batch as proof_batch  # noqa: E402  (the fork's train.py)
from train import loss_on  # noqa: E402

ARM = os.environ.get("STATE_ARM", "C0")
STATE = ARM != "C0"
MAX_ACTION, MAX_STEPS = 256, 48  # Dan's state-env settings
# live attempts per wave (Dan used 2048). Every step re-encodes the state (SH: history + state),
# and the ALiBi recipes' prefill builds a (batch, heads, L, L) fp32 bias: 2048 OOMed an A40 at
# L ~ 300 on SH and SN (2026-09-30). Weak SH models still reach L > 1000 at 256, so on an OOM the
# batch halves for the rest of the run and the call is redone (generate). A batch change is a
# sampling re-draw only.
ENV_BATCH = {"SN": 512, "SH": 256}.get(ARM, 2048)
ENV_OOMS = []  # (batch before, batch after) per OOM, recorded in the summary
FT_MAX_TOKENS = 200_000  # padded tokens per fine-tune micro-batch (SH's histories are long)


def generate(model, tok, prompts, greedy, temperature, batch, seed, max_new):
    if not STATE:
        return proof_generate(
            model,
            tok,
            prompts,
            greedy=greedy,
            temperature=temperature,
            batch=batch,
            seed=seed,
            max_new=max_new,
        )
    global ENV_BATCH
    while True:
        try:
            return env_generate(
                model,
                tok,
                prompts,
                greedy=greedy,
                temperature=temperature,
                max_action=MAX_ACTION,
                max_steps=MAX_STEPS,
                batch=ENV_BATCH,
                seed=seed,
            )
        except torch.OutOfMemoryError:
            if ENV_BATCH <= 16:
                raise
        torch.cuda.empty_cache()  # outside the except: the traceback's tensors are freed
        ENV_OOMS.append((ENV_BATCH, ENV_BATCH // 2))
        ENV_BATCH //= 2
        log(f"env sampling OOM: batch {ENV_OOMS[-1][0]} -> {ENV_BATCH}, redoing the call")


def mk_batch(data, idxs, tok, rng, dev):
    return (pair_batch if STATE else proof_batch)(data, idxs, tok, rng, dev)


# ---------------------------------------------------------------- the rules
# (hash-checked into every result)
TOTAL_SECS = int(
    os.environ.get("ND_TOTAL_SECS", 2400)
)  # process start -> summary written; SIGALRM past this. env: benchmarks only
PRETRAIN_SECS = 300  # agent's budget, compile included
PRETRAIN_GRACE = 15  # seconds the agent may overrun before the run is a "timeout"
PARAM_CAP = 10_000_000  # unique trainable params, embeddings included
RL_ROUNDS = 4  # expert-iteration rounds, fixed in samples
N_TARGETS, K, TEMP, MAX_NEW, SAMPLE_BATCH = 1500, 32, 0.8, 512, 1024
FT_STEPS, FT_BS, FT_LR, RETAIN, MAX_PER_THM, RL_WEIGHT = 600, 128, 3e-4, 20000, 4, 4
EVAL_K = 64
TOK_MODE = {"C0": "lean_seq", "SN": "lean_staten", "SH": "lean_stateh"}[ARM]
os.environ["ND_TOK_MODE"] = TOK_MODE  # the recipes read it
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def read(fn):
    return [json.loads(line) for line in open(fn) if line.strip()]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sha(fn):
    return hashlib.sha256(open(fn, "rb").read()).hexdigest()[:12]


def autocast():
    return torch.autocast("cuda", dtype=torch.bfloat16, enabled=(DEV == "cuda"))


class Timeout(Exception):
    pass


# ---------------------------------------------------------------- harness-side fine-tuning
# (RL stage only)


def finetune(model, tok, data, *, steps, bs, lr, rng, tag):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1, betas=(0.9, 0.95))
    model.train()
    t0 = time.time()
    perm, warmup = [], 50
    for step in range(1, steps + 1):
        cur_lr = (
            lr * step / warmup
            if step < warmup
            else lr / 10 + 0.5 * (lr - lr / 10) * (1 + math.cos(math.pi * step / steps))
        )
        for g in opt.param_groups:
            g["lr"] = cur_lr
        if len(perm) < bs:
            perm = list(range(len(data)))
            rng.shuffle(perm)
        idxs = [perm.pop() for _ in range(bs)]
        x, m = mk_batch(data, idxs, tok, rng, DEV)
        opt.zero_grad(set_to_none=True)
        # micro-batches weighted by their share of the target tokens: the same gradient as one
        # batch (a C0 batch is always a single micro-batch, weight exactly 1)
        per, n_tgt = max(1, FT_MAX_TOKENS // x.shape[1]), m[:, 1:].sum()
        for s0 in range(0, x.shape[0], per):
            xb, mb = x[s0 : s0 + per], m[s0 : s0 + per]
            with autocast():
                loss = loss_on(model, xb, mb) * (mb[:, 1:].sum() / n_tgt)
            loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 200 == 0:
            log(f"{tag} step {step} loss {loss.item():.4f} {time.time() - t0:.0f}s")
    model.eval()
    return {"steps": steps, "secs": time.time() - t0}


# ---------------------------------------------------------------- bookkeeping


def save_checkpoint(a, model, tok, stage):
    """Publish complete weight snapshots atomically; these are not optimizer-resume files."""
    path = Path(a.out) / f"{stage}.pt"
    pending = path.with_suffix(".pt.tmp")
    try:
        torch.save(
            {
                "state": model.state_dict(),
                "tok_mode": getattr(tok, "mode", None),
                "stage": stage,
                "seed": a.seed,
                "harness_sha": sha(f"{HERE}/harness.py"),
                "pretrain_sha": sha(f"{HERE}/pretrain.py"),
            },
            pending,
        )
        os.replace(pending, path)
    finally:
        pending.unlink(missing_ok=True)


def lstar(solved_names, recs, need=5):
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


def greedy_rate(model, tok, recs):
    rows, _ = sample_and_judge(model, tok, recs, 1, 0, 0, SAMPLE_BATCH, MAX_NEW)
    return sum(r["solved"] for r in rows) / max(len(rows), 1)


def dev_split(transfer):
    return [r for r in transfer if int(hashlib.sha1(r["key"].encode()).hexdigest(), 16) % 2 == 0]


def holdout_split(transfer):
    """Never scored by the loop. Score final.pt on this offline when checking in."""
    return [r for r in transfer if int(hashlib.sha1(r["key"].encode()).hexdigest(), 16) % 2 == 1]


# ---------------------------------------------------------------- stages


def stage_rl(a, tok, model, targets, train_recs, rng):
    t0 = time.time()
    found = collections.defaultdict(dict)
    rounds = []
    for r in range(1, a.rl_rounds + 1):
        rt = time.time()
        rows, samp = sample_and_judge(
            model, tok, targets, a.k, TEMP, a.seed * 1000 + r, SAMPLE_BATCH, a.max_new
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
            for x in fs[:MAX_PER_THM]:
                mix += [(t["prompt"], x["proof"])] * RL_WEIGHT
        n_rl = len(mix)
        mix += [
            (x["prompt"], x["proof"])
            for x in rng.sample(train_recs, min(a.retain, len(train_recs)))
        ]
        # state arms: record -> its (state, action) pairs; a proof that does not replay is dropped
        if STATE:
            recs = [{"prompt": p, "proof": q} for p, q in mix]
            state_pairs.attach(recs, tok)
            data = [r["pairs"] for r in recs if r["pairs"]]
        else:
            data = [(tok.encode_prompt(p), tok.encode_proof(q)) for p, q in mix]
        ft = (
            finetune(
                model, tok, data, steps=a.ft_steps, bs=FT_BS, lr=FT_LR, rng=rng, tag=f"ei r{r}"
            )
            if n_rl
            else {"steps": 0, "secs": 0}
        )
        save_checkpoint(a, model, tok, f"rl_round_{r:02d}")
        rounds.append(
            {
                "round": r,
                "sample_acc": sum(x["n_ok"] for x in rows) / samp["samples"],
                "new_proofs": new,
                "targets_solved": len(solved),
                "lstar": ls,
                "ge": ge,
                "mix_rl_records": n_rl,
                "mix_dropped": len(mix) - len(data),
                "sampling": samp,
                "finetune": ft,
                "secs": time.time() - rt,
            }
        )
        log(
            f"ei r{r}: acc {rounds[-1]['sample_acc']:.4f}, +{new} proofs, "
            f"solved {len(solved)}/{len(targets)}, "
            f"L*={ls}, {samp['samples_per_sec']:.0f} samples/s, round {rounds[-1]['secs']:.0f}s"
        )
    with open(f"{a.out}/found_targets.jsonl", "w") as f:
        for t in targets:
            for x in found[t["name"]].values():
                f.write(
                    json.dumps({"name": t["name"], "thm": t["thm"], "L_true": t["n_lines"], **x})
                    + "\n"
                )
    return {"rounds": rounds, "secs": time.time() - t0, "targets": len(targets)}


def stage_eval(a, tok, model, dev_pool, heldout_recs):
    rows, samp = sample_and_judge(
        model, tok, dev_pool, a.eval_k, TEMP, a.seed * 1000 + 777, SAMPLE_BATCH, a.max_new
    )
    solved = {r["name"] for r in rows if r["solved"]}
    ls, ge = lstar(solved, dev_pool)
    out = {
        "attempts_per_theorem": a.eval_k,
        "solved": len(solved),
        "n": len(dev_pool),
        "solved_ge7": ge[7],
        "lstar": ls,
        "ge": ge,
        "by_bin": by_bin(solved, dev_pool),
        "max_pruned": max((max(r["pruned_lens"]) for r in rows if r["solved"]), default=0),
        "heldout_greedy": greedy_rate(model, tok, heldout_recs),
        "sampling": samp,
    }
    log(
        f"eval: solved {len(solved)}/{len(dev_pool)}, ge7 {ge[7]}, L*={ls}, "
        f"max pruned {out['max_pruned']}, "
        f"held-out greedy {out['heldout_greedy']:.3f}"
    )
    with open(f"{a.out}/eval_dev.jsonl", "w") as f:
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


def lean_gate_totals():
    """Sum the fork lean_gate's per-generate() log lines (remote.sh sets LEAN_GATE_LOG)."""
    fn = os.environ.get("LEAN_GATE_LOG")
    if not fn or not os.path.exists(fn):
        return None
    # dan_best-state's gate logs other keys (Lean alone decides): sum whatever numbers it logs
    tot = collections.Counter()
    for rec in read(fn):
        tot.update({k: v for k, v in rec.items() if isinstance(v, (int, float))})
    return dict(tot)


def write_summary(a, status, **kw):
    s = {
        "status": status,
        "metric": kw.get("eval", {}).get("solved_ge7"),
        "metric_rl": (kw.get("rl", {}).get("rounds") or [{}])[-1].get(
            "targets_solved"
        ),  # secondary: cumulative RL targets solved
        "harness_sha": sha(f"{HERE}/harness.py"),
        "pretrain_sha": sha(f"{HERE}/pretrain.py"),
        "device": DEV,
        "gpu": torch.cuda.get_device_name(0) if DEV == "cuda" else "cpu",
        "torch": torch.__version__,
        "lean_gate": lean_gate_totals(),
        "arm": ARM,
        "env_batch": ENV_BATCH,
        "env_ooms": ENV_OOMS,
        "total_secs": time.time() - a.T0,
        "rules": {
            k: globals()[k]
            for k in (
                "TOTAL_SECS",
                "PRETRAIN_SECS",
                "PARAM_CAP",
                "RL_ROUNDS",
                "N_TARGETS",
                "K",
                "EVAL_K",
                "TOK_MODE",
            )
        },
        **kw,
    }
    with open(f"{a.out}/summary.json", "w") as f:
        json.dump(s, f, indent=1)
    log(
        f"SUMMARY status={status} metric={s['metric']} metric_rl={s['metric_rl']} "
        f"total {s['total_secs']:.0f}s"
    )
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--pretrain-only", action="store_true", help="save pretrain.pt and stop before eval/RL"
    )
    ap.add_argument(
        "--smoke", action="store_true", help="tiny everything, for a CPU plumbing check"
    )
    a = ap.parse_args()
    a.pretrain_secs, a.rl_rounds, a.k, a.eval_k, a.max_new, a.ft_steps, a.retain, a.n_targets = (
        PRETRAIN_SECS,
        RL_ROUNDS,
        K,
        EVAL_K,
        MAX_NEW,
        FT_STEPS,
        RETAIN,
        N_TARGETS,
    )
    if a.smoke:
        a.pretrain_secs, a.k, a.eval_k, a.max_new, a.ft_steps, a.retain, a.n_targets = (
            10,
            2,
            2,
            120,
            5,
            50,
            8,
        )
    os.makedirs(a.out, exist_ok=True)
    # Preserve discarded variants too: weights alone do not specify the model architecture.
    source_dir = Path(a.out) / "source"
    source_dir.mkdir(exist_ok=True)
    for filename in ("pretrain.py", "harness.py", "remote.sh", "state_pairs.py"):
        shutil.copy2(Path(HERE) / filename, source_dir / filename)
    a.T0 = time.time()
    torch.manual_seed(a.seed)
    rng = random.Random(a.seed)

    def on_alarm(signum, frame):
        raise Timeout()

    signal.signal(signal.SIGALRM, on_alarm)
    signal.alarm(TOTAL_SECS if not a.smoke else 300)

    train_recs = read(f"{FORK}/data/train.jsonl")
    assert all(r["n_lines"] <= 6 for r in train_recs), "pretrain file violates cap 6"
    if STATE:  # before pretrain()'s clock, as in Dan's port
        t_pairs, skipped = time.time(), state_pairs.attach(train_recs, make_tokenizer(TOK_MODE))
        n_pairs = sum(len(r["pairs"]) for r in train_recs)
        log(f"{ARM} pairs: {n_pairs:,}, {skipped} records skipped, {time.time() - t_pairs:.0f}s")
    heldout_all = read(f"{FORK}/data/heldout.jsonl")
    # Length-ordered file: take 2000 evenly spaced records, so every length 2-6 is covered
    # (before 105, stride 2 + [:2000] took lengths 2-5 only).
    heldout_all = [heldout_all[i * len(heldout_all) // 2000] for i in range(2000)]
    heldout_recs = heldout_all if not a.smoke else heldout_all[:16]
    targets_all = read(f"{FORK}/data/ladder/rl_targets.jsonl")
    transfer = read(f"{FORK}/data/ladder/transfer.jsonl")
    dev_pool = dev_split(transfer) if not a.smoke else dev_split(transfer)[:8]
    targets = random.Random(a.seed).sample(targets_all, min(a.n_targets, len(targets_all)))
    tk, dk, hk = (
        {t["key"] for t in targets},
        {r["key"] for r in dev_pool},
        {r["key"] for r in train_recs},
    )
    assert not (tk & dk) and not (tk & hk) and not (dk & hk), "renaming-class leak between pools"
    log(
        f"pools: train {len(train_recs):,}, targets {len(targets)}, dev-transfer {len(dev_pool)}, "
        f"held-out {len(heldout_recs)}; "
        f"device {DEV}; harness {sha(f'{HERE}/harness.py')} pretrain {sha(f'{HERE}/pretrain.py')}"
    )

    try:
        agent = importlib.import_module("pretrain")
        t0 = time.time()
        model, tok, pre = agent.pretrain(train_recs, a.pretrain_secs, DEV, a.seed, log)
        if DEV == "cuda":
            torch.cuda.synchronize()
        pre = dict(pre or {})
        pre["secs"] = time.time() - t0
        model.eval()
        n_params = sum(p.numel() for p in model.parameters())
        pre["n_params"] = n_params
        log(
            f"pretrain returned after {pre['secs']:.0f}s (budget {a.pretrain_secs}s), "
            f"{n_params:,} params"
        )
        if pre["secs"] > a.pretrain_secs + PRETRAIN_GRACE:
            write_summary(
                a,
                "timeout",
                pretrain=pre,
                reason=f"pretrain overran budget by {pre['secs'] - a.pretrain_secs:.0f}s",
            )
            return
        if getattr(tok, "mode", None) != TOK_MODE:
            write_summary(a, "crash", pretrain=pre, reason=f"tok.mode {tok.mode!r} != {TOK_MODE!r}")
            return
        if n_params > PARAM_CAP:
            write_summary(a, "crash", pretrain=pre, reason=f"{n_params} params > cap {PARAM_CAP}")
            return
        save_checkpoint(a, model, tok, "pretrain")
        if a.pretrain_only:
            write_summary(a, "ok", stage="pretrain_only", pretrain=pre)
            return
        pre["heldout_greedy"] = greedy_rate(model, tok, heldout_recs)
        log(f"pretrain held-out greedy {pre['heldout_greedy']:.3f}")
        rl = stage_rl(a, tok, model, targets, train_recs, rng)
        ev = stage_eval(a, tok, model, dev_pool, heldout_recs)
        save_checkpoint(a, model, tok, "final")
        write_summary(a, "ok", pretrain=pre, rl=rl, eval=ev)
    except Timeout:
        write_summary(a, "timeout", reason=f"exceeded TOTAL_SECS={TOTAL_SECS}")
    except Exception:
        traceback.print_exc()
        write_summary(a, "crash", reason=traceback.format_exc()[-2000:])
    finally:
        signal.alarm(0)


if __name__ == "__main__":
    main()
