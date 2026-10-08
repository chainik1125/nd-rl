#!/usr/bin/env python3
# imports the fork (model, tokenizer, train, ...), which is not resolvable here
# pyright: reportMissingImports=false
"""Harness for the retry experiment: the test-time-loops harness (autoresearch_loops/harness.py,
sha 2eec1b61666a, the one run 590 was scored with) plus multi-attempt episodes. What changed:

  - `oracle.sample_episodes` gives rl() the failed attempts and their error tokens as well as
    the accepted proofs (the original `oracle.sample` returns accepted proofs only)
  - the retry arms get a second eval: EP_N episodes of up to EP_ATTEMPTS attempts per dev
    theorem, the same 64 proofs as the scored eval's 64 independent samples
  - ND_ARM picks the arm (iid | retry | blank); TOTAL_SECS defaults to 7200

Everything below this paragraph is the original docstring.

FROZEN harness for the test-time-loops autoresearch loop. The research agent does not edit it.

Forked from ../autoresearch/harness.py (sha 2aa52f23e7ec) on 2026-10-07. What changed: the RL
stage belongs to the agent, behind a wall-clock budget and a sampling oracle, and the metric is
measured with the model's test-time compute knob (KNOB) set to EVAL_LEVEL.

One timed run = harness owns the clocks, the oracle and the eval; `pretrain.py` (agent-editable)
owns pretraining and RL:

  1. load frozen data (Dan's cap-6 Stage-1 file, ladder RL-target pool, ladder transfer pool)
  2. call pretrain.pretrain(train_recs, PRETRAIN_SECS, ...) -> (model, tok, stats); charged for
     everything including model construction and torch.compile. Overrun past the grace -> timeout.
  3. check n_params <= PARAM_CAP, tok.mode == TOK_MODE and that the model has the KNOB attribute
  4. call pretrain.rl(model, tok, targets, train_recs, oracle, RL_SECS, ...): expert iteration in
     whatever form the agent likes. The targets are statements only, and `oracle.sample` is the
     only way to turn them into training data: it samples the model's own attempts and returns
     the ones Lean AND nd_verify accept. RL_SECS is wall-clock inside rl() minus the time the
     oracle spent in the Lean gate (pod CPUs differ; GPU work is what is rationed). Overrun past
     the grace, or an oracle call after the budget is spent -> timeout.
  5. eval on the dev half of the transfer pool with the harness setting the model's KNOB:
     METRIC = dev theorems solved at L_true >= 7 with EVAL_K attempts at EVAL_LEVEL.
     The same eval at DIAG_LEVEL is recorded as a diagnostic, not scored.
  6. SIGALRM at TOTAL_SECS from process start kills everything -> status "timeout"

The holdout half of the transfer pool is never scored here. Everything is written to --out.

  python3 harness.py --out artifacts/x                 # on the pod (see pod.py / remote.sh)
  ND_FORK=... python3 harness.py --out /tmp/smoke --smoke   # CPU, ~1 min, plumbing check
"""

import argparse
import collections
import hashlib
import importlib
import json
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
import torch  # noqa: E402
from episodes import sample_episodes  # noqa: E402
from eval_set import judge, wilson  # noqa: E402
from sample import generate  # noqa: E402

# ---------------------------------------------------------------- the rules
# (hash-checked into every result)
TOTAL_SECS = int(
    os.environ.get("ND_TOTAL_SECS", 7200)
)  # process start -> summary written; SIGALRM past this. env: benchmarks only
PRETRAIN_SECS = 300  # agent's pretraining budget, compile included
PRETRAIN_GRACE = 15  # seconds the agent may overrun before the run is a "timeout"
RL_SECS = 1800  # agent's RL budget: wall-clock inside rl() minus the oracle's Lean-gate time
RL_GRACE = 60
PARAM_CAP = 10_000_000  # unique trainable params, embeddings included; checked after RL too
N_TARGETS, TEMP, MAX_NEW, SAMPLE_BATCH = 1500, 0.8, 512, 1024
EVAL_K = 64
# The model's test-time compute knob: an integer attribute the harness sets before each eval and
# rl() sets before oracle calls. A sibling loop for another architecture changes these three
# constants and nothing else in this file.
KNOB = "n_loops"  # loops of the shared core
EVAL_LEVEL = 8  # the metric: the knob's value during the scored eval
DIAG_LEVEL = 2  # the same eval again at this value, recorded but not scored
TOK_MODE = "lean_seq"  # required tokenizer mode (fork lean_tok.py)
ARM = os.environ.get("ND_ARM", "iid")  # iid: run 590 unchanged; retry: error tokens; blank: <fail>
EP_N, EP_ATTEMPTS = 8, 8  # the episode eval: EP_N * EP_ATTEMPTS == EVAL_K proofs per theorem
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def read(fn):
    return [json.loads(line) for line in open(fn) if line.strip()]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sha(fn):
    return hashlib.sha256(open(fn, "rb").read()).hexdigest()[:12]


class Timeout(Exception):
    pass


class BudgetSpent(Timeout):
    """rl() asked the oracle for samples after RL_SECS was used up."""


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


def gate_seconds():
    """Seconds the fork's lean_gate has logged so far (Lean wall-clock plus nd_verify)."""
    fn = os.environ.get("LEAN_GATE_LOG")
    if not fn or not os.path.exists(fn):
        return 0.0
    return sum(rec["lean_wall_s"] + rec["nd_verify_s"] for rec in read(fn))


def sample_and_judge(model, tok, recs, k, temperature, seed, batch, max_new):
    prompts = [r["prompt"] for r in recs for _ in range(k)]
    t0, g0 = time.time(), gate_seconds()
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
    secs, gate = time.time() - t0, gate_seconds() - g0
    rows = judge(recs, [flat[i * k : (i + 1) * k] for i in range(len(recs))], "n_lines")
    return rows, {
        "samples": len(prompts),
        "secs": secs,
        "gate_secs": gate,
        "samples_per_sec": len(prompts) / secs,
    }


def greedy_rate(model, tok, recs):
    rows, _ = sample_and_judge(model, tok, recs, 1, 0, 0, SAMPLE_BATCH, MAX_NEW)
    return sum(r["solved"] for r in rows) / max(len(rows), 1)


def dev_split(transfer):
    return [r for r in transfer if int(hashlib.sha1(r["key"].encode()).hexdigest(), 16) % 2 == 0]


def holdout_split(transfer):
    """Never scored by the loop. Score final.pt on this offline when checking in."""
    return [r for r in transfer if int(hashlib.sha1(r["key"].encode()).hexdigest(), 16) % 2 == 1]


# ---------------------------------------------------------------- the RL oracle


class Oracle:
    """What rl() gets instead of data: the model's own attempts at the RL targets, filtered to
    the ones Lean AND nd_verify accept. It also keeps the RL clock and the record of every call.
    """

    def __init__(self, a, tok, targets):
        self.a, self.tok = a, tok
        self.targets = {t["name"]: t for t in targets}
        self.found = collections.defaultdict(dict)  # name -> {proof text: record}
        self.calls = []
        self.t0 = self.last = time.time()
        self.uncharged = 0.0

    def used(self):
        """RL seconds charged so far: wall-clock since rl() started, minus Lean-gate time."""
        return time.time() - self.t0 - self.uncharged

    def left(self):
        return self.a.rl_secs - self.used()

    def sample(self, model, targets, k, temperature=TEMP, batch=SAMPLE_BATCH):
        """k attempts at each of `targets` (records rl() was given, any subset, in any order) from
        `model` as it is now, in eval mode, at whatever its KNOB is set to. Returns one
        dict per target, in order: name, prompt, n_lines, proofs (the distinct accepted proof
        texts), n_ok (accepted attempts) and n_tried. Rejected attempts are not returned.
        Raises BudgetSpent, which ends the run as a timeout, once the budget is used up."""
        if self.left() <= 0:
            raise BudgetSpent(f"oracle called with {self.used():.0f}s of {self.a.rl_secs}s used")
        recs = [self.targets[t["name"]] for t in targets]
        call = len(self.calls) + 1
        model.eval()
        rows, samp = sample_and_judge(
            model, self.tok, recs, k, temperature, self.a.seed * 1000 + call, batch, self.a.max_new
        )
        self.uncharged += samp["gate_secs"]
        new = 0
        for t, row in zip(recs, rows, strict=True):
            for p, wl, pl in zip(
                row["proofs"], row["written_lens"], row["pruned_lens"], strict=True
            ):
                if p not in self.found[t["name"]]:
                    self.found[t["name"]][p] = {
                        "proof": p,
                        "written": wl,
                        "pruned": pl,
                        "round": call,
                    }
                    new += 1
        solved = {n for n, d in self.found.items() if d}
        ls, ge = lstar(solved, list(self.targets.values()))
        now = time.time()
        self.calls.append(
            {
                "round": call,
                "level": getattr(model, KNOB, None),
                "k": k,
                "n_targets": len(recs),
                "sample_acc": sum(x["n_ok"] for x in rows) / max(samp["samples"], 1),
                "new_proofs": new,
                "targets_solved": len(solved),
                "lstar": ls,
                "ge": ge,
                "mix_rl_records": sum(len(x["proofs"]) for x in rows),  # proofs handed back
                "sampling": samp,
                "secs": now - self.last,  # since the previous call returned: includes rl()'s work
                "rl_secs_used": self.used(),
            }
        )
        self.last = now
        c = self.calls[-1]
        log(
            f"ei r{call}: acc {c['sample_acc']:.4f}, +{new} proofs, "
            f"solved {len(solved)}/{len(self.targets)}, L*={ls}, {KNOB} {c['level']}, "
            f"{len(recs)} targets x {k}, {samp['samples_per_sec']:.0f} samples/s, "
            f"RL clock {c['rl_secs_used']:.0f}/{self.a.rl_secs}s"
        )
        return [
            {
                "name": t["name"],
                "prompt": t["prompt"],
                "n_lines": t["n_lines"],
                "proofs": list(row["proofs"]),
                "n_ok": row["n_ok"],
                "n_tried": row["n_tried"],
            }
            for t, row in zip(recs, rows, strict=True)
        ]

    def sample_episodes(self, model, targets, n_ep, max_attempts, mode):
        """n_ep episodes of up to max_attempts attempts at each of `targets` (episodes.py), failed
        attempts and error tokens included. Charged like `sample`: wall-clock minus Lean-gate
        time. Returns one list of episodes per target, in order."""
        if self.left() <= 0:
            raise BudgetSpent(f"oracle called with {self.used():.0f}s of {self.a.rl_secs}s used")
        recs = [self.targets[t["name"]] for t in targets]
        call = len(self.calls) + 1
        model.eval()
        t0, g0 = time.time(), gate_seconds()
        eps = sample_episodes(
            model,
            self.tok,
            recs,
            n_ep,
            max_attempts,
            mode,
            TEMP,
            self.a.seed * 1000 + call,
            self.a.max_new,
        )
        secs, gate = time.time() - t0, gate_seconds() - g0
        self.uncharged += gate
        new = 0
        for t, row in zip(recs, eps, strict=True):
            for ep in row:
                p = ep["proof"]
                if p and p not in self.found[t["name"]]:
                    self.found[t["name"]][p] = {"proof": p, "round": call}
                    new += 1
        solved = {n for n, d in self.found.items() if d}
        ls, ge = lstar(solved, list(self.targets.values()))
        now = time.time()
        stats = episode_stats(eps, max_attempts)
        self.calls.append(
            {
                "round": call,
                "episodes": n_ep,
                "max_attempts": max_attempts,
                "n_targets": len(recs),
                "new_proofs": new,
                "targets_solved": len(solved),
                "lstar": ls,
                "ge": ge,
                **stats,
                "sampling": {"secs": secs, "gate_secs": gate},
                "secs": now - self.last,
                "rl_secs_used": self.used(),
            }
        )
        self.last = now
        log(
            f"ei r{call}: {len(recs)} targets x {n_ep} episodes, attempts {stats['tried']}, "
            f"accepted {stats['ok']}, +{new} proofs, solved {len(solved)}/{len(self.targets)}, "
            f"L*={ls}, RL clock {self.used():.0f}/{self.a.rl_secs}s"
        )
        return eps


def episode_stats(eps, max_attempts):
    """Per attempt number: how many episodes got that far and how many were accepted there.
    `solved_by[a]`: theorems with an episode accepted at attempt <= a+1. Plus the error codes."""
    tried, ok = [0] * max_attempts, [0] * max_attempts
    first = [None] * len(eps)  # earliest accepted attempt over the theorem's episodes
    codes, toks = collections.Counter(), 0
    for i, row in enumerate(eps):
        for ep in row:
            n = len(ep["n_tok"])
            toks += sum(ep["n_tok"])
            codes.update(ep["codes"])
            for a in range(n):
                tried[a] += 1
            if ep["proof"]:
                ok[n - 1] += 1
                first[i] = n if first[i] is None else min(first[i], n)
    solved_by = [sum(1 for f in first if f is not None and f <= a + 1) for a in range(max_attempts)]
    return {
        "tried": tried,
        "ok": ok,
        "solved_by": solved_by,
        "sampled_tokens": toks,
        "codes": dict(codes.most_common()),
    }


# ---------------------------------------------------------------- stages


def stage_rl(a, agent, tok, model, targets, train_recs):
    oracle = Oracle(a, tok, targets)
    # statements only: what the agent's RL may know about a target
    visible = [{"name": t["name"], "prompt": t["prompt"], "n_lines": t["n_lines"]} for t in targets]
    stats = agent.rl(model, tok, visible, train_recs, oracle, a.rl_secs, DEV, a.seed, log)
    if DEV == "cuda":
        torch.cuda.synchronize()
    used = oracle.used()
    model.eval()
    with open(f"{a.out}/found_targets.jsonl", "w") as f:
        for t in targets:
            for x in oracle.found[t["name"]].values():
                f.write(
                    json.dumps({"name": t["name"], "thm": t["thm"], "L_true": t["n_lines"], **x})
                    + "\n"
                )
    log(f"rl returned: {used:.0f}s of {a.rl_secs}s charged, {oracle.uncharged:.0f}s Lean gate")
    return {
        "rounds": oracle.calls,
        "secs": time.time() - oracle.t0,
        "rl_secs_used": used,
        "gate_secs": oracle.uncharged,
        "targets": len(targets),
        "stats": dict(stats or {}),
    }


def eval_at(a, tok, model, dev_pool, level):
    setattr(model, KNOB, level)
    rows, samp = sample_and_judge(
        model, tok, dev_pool, a.eval_k, TEMP, a.seed * 1000 + 777, SAMPLE_BATCH, a.max_new
    )
    solved = {r["name"] for r in rows if r["solved"]}
    ls, ge = lstar(solved, dev_pool)
    out = {
        "level": level,
        "attempts_per_theorem": a.eval_k,
        "solved": len(solved),
        "n": len(dev_pool),
        "solved_ge7": ge[7],
        "lstar": ls,
        "ge": ge,
        "by_bin": by_bin(solved, dev_pool),
        "sample_acc": sum(r["n_ok"] for r in rows) / max(samp["samples"], 1),
        "max_pruned": max((max(r["pruned_lens"]) for r in rows if r["solved"]), default=0),
        "sampling": samp,
    }
    log(
        f"eval at {KNOB} {level}: solved {len(solved)}/{len(dev_pool)}, ge7 {ge[7]}, L*={ls}, "
        f"per-sample {out['sample_acc']:.4f}, {samp['samples_per_sec']:.0f} samples/s"
    )
    return out, rows


def eval_episodes(a, tok, model, dev_pool):
    """The retry eval: EP_N episodes of up to EP_ATTEMPTS attempts per dev theorem."""
    t0 = time.time()
    n_ep, n_att = (EP_N, EP_ATTEMPTS) if a.eval_k == EVAL_K else (2, 2)  # smoke
    model.eval()
    eps = sample_episodes(
        model,
        tok,
        dev_pool,
        n_ep,
        n_att,
        ARM if ARM == "blank" else "error",
        TEMP,
        a.seed * 1000 + 778,
        a.max_new,
    )
    solved = {
        r["name"] for r, row in zip(dev_pool, eps, strict=True) if any(e["proof"] for e in row)
    }
    ls, ge = lstar(solved, dev_pool)
    out = {
        "episodes": n_ep,
        "max_attempts": n_att,
        "solved": len(solved),
        "solved_ge7": ge[7],
        "lstar": ls,
        "by_bin": by_bin(solved, dev_pool),
        **episode_stats(eps, n_att),
        "secs": time.time() - t0,
    }
    with open(f"{a.out}/eval_dev_episodes.jsonl", "w") as f:
        for r, row in zip(dev_pool, eps, strict=True):
            f.write(
                json.dumps(
                    {
                        "name": r["name"],
                        "n_lines": r["n_lines"],
                        "episodes": [
                            {
                                "texts": [
                                    tok.text([tok.itos[x] for x in ids]) for ids, _ in e["fails"]
                                ],
                                "codes": e["codes"],
                                "n_tok": e["n_tok"],
                                "proof": e["proof"],
                                "errs": [[tok.itos[x] for x in err] for _, err in e["fails"]],
                            }
                            for e in row
                        ],
                    }
                )
                + "\n"
            )
    log(
        f"episode eval: solved {len(solved)}/{len(dev_pool)} with {n_ep} x <= {n_att} attempts; "
        f"tried {out['tried']}, accepted {out['ok']}, solved by attempt {out['solved_by']}"
    )
    return out


def stage_eval(a, tok, model, dev_pool, heldout_recs):
    out, rows = eval_at(a, tok, model, dev_pool, a.eval_level)
    out["heldout_greedy"] = greedy_rate(model, tok, heldout_recs)
    with open(f"{a.out}/eval_dev.jsonl", "w") as f:
        for r in rows:
            keys = ("name", "n_lines", "solved", "n_ok", "proofs", "written_lens", "pruned_lens")
            f.write(json.dumps({k: r[k] for k in keys}) + "\n")
    # k of EVAL_K independent samples accepted, per theorem: all the episode eval is compared with
    out["n_ok"] = [r["n_ok"] for r in rows]
    if ARM != "iid":
        out["episodes"] = eval_episodes(a, tok, model, dev_pool)
        return (
            out  # the untied model ignores the knob, so the diagnostic eval would repeat this one
        )
    diag, _ = eval_at(a, tok, model, dev_pool, a.diag_level)
    out["diag"] = diag
    setattr(model, KNOB, a.eval_level)
    log(
        f"eval: solved {out['solved']}/{len(dev_pool)}, ge7 {out['solved_ge7']} at "
        f"{KNOB} {a.eval_level} ({diag['solved_ge7']} at {a.diag_level}), L*={out['lstar']}, "
        f"max pruned {out['max_pruned']}, held-out greedy {out['heldout_greedy']:.3f}"
    )
    return out


def lean_gate_totals():
    """Sum the fork lean_gate's per-generate() log lines (remote.sh sets LEAN_GATE_LOG)."""
    fn = os.environ.get("LEAN_GATE_LOG")
    if not fn or not os.path.exists(fn):
        return None
    keys = ("samples", "parse_fail", "distinct_checked", "both_ok", "nd_ok_lean_rej")
    tot = dict.fromkeys(keys + ("nd_rej_lean_ok", "lean_wall_s"), 0)
    for rec in read(fn):
        for k in tot:
            tot[k] += rec[k]
    return tot


RULES = (
    "TOTAL_SECS",
    "PRETRAIN_SECS",
    "RL_SECS",
    "PARAM_CAP",
    "N_TARGETS",
    "EVAL_K",
    "KNOB",
    "EVAL_LEVEL",
    "DIAG_LEVEL",
    "TOK_MODE",
)


def write_summary(a, status, **kw):
    s = {
        "status": status,
        "metric": kw.get("eval", {}).get("solved_ge7"),
        "metric_diag": kw.get("eval", {}).get("diag", {}).get("solved_ge7"),
        "metric_episodes": kw.get("eval", {}).get("episodes", {}).get("solved_ge7"),
        "arm": ARM,
        "metric_rl": (kw.get("rl", {}).get("rounds") or [{}])[-1].get(
            "targets_solved"
        ),  # secondary: cumulative RL targets solved
        "harness_sha": sha(f"{HERE}/harness.py"),
        "pretrain_sha": sha(f"{HERE}/pretrain.py"),
        "device": DEV,
        "gpu": torch.cuda.get_device_name(0) if DEV == "cuda" else "cpu",
        "torch": torch.__version__,
        "lean_gate": lean_gate_totals(),
        "total_secs": time.time() - a.T0,
        "rules": {k: globals()[k] for k in RULES},
        **kw,
    }
    with open(f"{a.out}/summary.json", "w") as f:
        json.dump(s, f, indent=1)
    log(
        f"SUMMARY status={status} metric={s['metric']} metric_diag={s['metric_diag']} "
        f"metric_rl={s['metric_rl']} total {s['total_secs']:.0f}s"
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
    a.pretrain_secs, a.rl_secs, a.eval_k, a.max_new, a.n_targets = (
        PRETRAIN_SECS,
        RL_SECS,
        EVAL_K,
        MAX_NEW,
        N_TARGETS,
    )
    a.eval_level, a.diag_level = EVAL_LEVEL, DIAG_LEVEL
    if a.smoke:
        a.pretrain_secs, a.rl_secs, a.eval_k, a.max_new, a.n_targets = 10, 30, 2, 120, 8
    os.makedirs(a.out, exist_ok=True)
    # Preserve discarded variants too: weights alone do not specify the model architecture.
    source_dir = Path(a.out) / "source"
    source_dir.mkdir(exist_ok=True)
    for filename in ("pretrain.py", "harness.py", "remote.sh", "episodes.py", "errors.py"):
        shutil.copy2(Path(HERE) / filename, source_dir / filename)
    a.T0 = time.time()
    torch.manual_seed(a.seed)

    def on_alarm(signum, frame):
        raise Timeout()

    signal.signal(signal.SIGALRM, on_alarm)
    signal.alarm(TOTAL_SECS if not a.smoke else 900)

    train_recs = read(f"{FORK}/data/train.jsonl")
    assert all(r["n_lines"] <= 6 for r in train_recs), "pretrain file violates cap 6"
    heldout_all = read(f"{FORK}/data/heldout.jsonl")
    # Length-ordered file: take 2000 evenly spaced records, so every length 2-6 is covered.
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

    pre = {}
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
        if pre["secs"] > a.pretrain_secs + PRETRAIN_GRACE and not a.smoke:
            over = pre["secs"] - a.pretrain_secs
            write_summary(
                a, "timeout", pretrain=pre, reason=f"pretrain overran budget by {over:.0f}s"
            )
            return
        if getattr(tok, "mode", None) != TOK_MODE:
            write_summary(a, "crash", pretrain=pre, reason=f"tok.mode {tok.mode!r} != {TOK_MODE!r}")
            return
        if n_params > PARAM_CAP:
            write_summary(a, "crash", pretrain=pre, reason=f"{n_params} params > cap {PARAM_CAP}")
            return
        if not isinstance(getattr(model, KNOB, None), int):
            reason = f"model has no integer {KNOB} for the harness to set at eval"
            write_summary(a, "crash", pretrain=pre, reason=reason)
            return
        save_checkpoint(a, model, tok, "pretrain")
        if a.pretrain_only:
            write_summary(a, "ok", stage="pretrain_only", pretrain=pre)
            return
        pre["heldout_greedy"] = greedy_rate(model, tok, heldout_recs)
        log(f"pretrain held-out greedy {pre['heldout_greedy']:.3f}")
        rl = stage_rl(a, agent, tok, model, targets, train_recs)
        if rl["rl_secs_used"] > a.rl_secs + RL_GRACE and not a.smoke:
            over = rl["rl_secs_used"] - a.rl_secs
            reason = f"rl overran budget by {over:.0f}s"
            write_summary(a, "timeout", pretrain=pre, rl=rl, reason=reason)
            return
        n_after = sum(p.numel() for p in model.parameters())
        if n_after > PARAM_CAP:
            reason = f"{n_after} params > cap {PARAM_CAP} after rl"
            write_summary(a, "crash", pretrain=pre, rl=rl, reason=reason)
            return
        save_checkpoint(a, model, tok, "final")
        ev = stage_eval(a, tok, model, dev_pool, heldout_recs)
        write_summary(a, "ok", pretrain=pre, rl=rl, eval=ev)
    except BudgetSpent as e:
        write_summary(a, "timeout", pretrain=pre, reason=f"rl: {e}")
    except Timeout:
        write_summary(a, "timeout", pretrain=pre, reason=f"exceeded TOTAL_SECS={TOTAL_SECS}")
    except Exception:
        traceback.print_exc()
        write_summary(a, "crash", pretrain=pre, reason=traceback.format_exc()[-2000:])
    finally:
        signal.alarm(0)


if __name__ == "__main__":
    main()
