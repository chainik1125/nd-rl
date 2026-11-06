#!/usr/bin/env python3
# imports the fork (model, tokenizer, train, ...), which is not resolvable here
# pyright: reportMissingImports=false
"""Harness for long_rl_20261014: pretrain on <= 6 body lines, then RL on the long-proofs set (body 7 to ~45).

Forked from the test-time-loops harness (autoresearch_loops/harness.py, sha 2eec1b61666a, the one run 590 and 512 were
scored with). What changed (Robbie, 2026-10-14):

  - Lean is the only checker: overlay/lean_gate.py, and a sample counts iff Lean accepts its literal text. nd_verify is
    never called; the decoded ND proof is used only to count lines.
  - Length is body lines (proof lines after the premise block) everywhere: bands, L*, the pretraining cap.
  - Tokenizer: overlay/lean_tok.py (ten atoms, ten premises, 160 names). The overlay directory is first on sys.path.
  - Data (make_data.py): pretraining = Dan's cap-6 file with distractor premises; RL targets / dev / holdout = the
    long-proofs set, band-capped. The old transfer dev pool is evaluated too (EVAL_K_OLD samples), to catch forgetting.
  - The oracle also returns every rollout (token ids and the Lean verdict) for the GRPO arm: sample_rollouts().
  - Sampling batch size follows the model's KV-cache depth (model.kv_layers()), so 8-loop sampling fits an A40.

  1. pretrain.pretrain(train_recs, PRETRAIN_SECS, ...) -> (model, tok, stats), charged for everything
  2. checks: params <= PARAM_CAP, tok.mode == TOK_MODE, model has an integer KNOB
  3. pretrain.rl(model, tok, targets, train_recs, oracle, RL_SECS, ...): the arm's RL. RL_SECS is wall-clock inside rl()
     minus Lean-gate time.
  4. eval: dev (long-proofs set) with EVAL_K samples at EVAL_LEVEL, by band; old transfer dev with EVAL_K_OLD samples
  5. SIGALRM at TOTAL_SECS

  python3 harness.py --out artifacts/x [--seed 0]                    # on the pod (remote.sh)
  ND_FORK=... python3 harness.py --out /tmp/smoke --smoke            # CPU plumbing check with real Lean
"""

import argparse
import collections
import gzip
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
DATA = os.environ.get("ND_DATA", os.path.join(HERE, "data"))
# overlay first (tokenizer, Lean-only gate), then the fork (model, sample, train), then this directory (pretrain.py)
sys.path[:0] = [os.path.join(HERE, "overlay"), FORK, HERE]
import importlib.util  # noqa: E402

import torch  # noqa: E402


def _overlay(name):
    """Load overlay/<name>.py under the module name the fork imports. Fork modules put the fork directory at
    sys.path[0] whenever they are imported (nd2lean does, inside lean_tok), so path order alone is not enough:
    registering the module in sys.modules first is what makes every later `import lean_tok` get the overlay."""
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, "overlay", f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_overlay("lean_tok")
gate = _overlay("lean_gate").gate
from eval_set import wilson  # noqa: E402  (fork; wilson interval only)
from nd_verify.verify import parse_proof_tokens  # noqa: E402  (a parser: line counts only, never a verdict)
from sample import generate_ids  # noqa: E402  (fork)

# ---------------------------------------------------------------- the rules
TOTAL_SECS = int(os.environ.get("ND_TOTAL_SECS", 6.5 * 3600))  # RL_SECS excludes Lean-gate time (~80 s a round)
PRETRAIN_SECS = 600  # Dan's file with distractors is ~1.6x the tokens of the plain cap-6 file
PRETRAIN_GRACE = 30
RL_SECS = int(os.environ.get("ND_RL_SECS", 3 * 3600))
RL_GRACE = 120
PARAM_CAP = 10_000_000
TEMP = 0.8
MAX_NEW = 2304  # proof tokens: approach E's 80-line references run to ~1,900; checked against the pools at load
KV_TOKENS = 1024 * 6 * 1700  # sampling batch * KV layers * tokens: 6 layers x 1,024 x 1,700 is ~16 GB of bf16 KV
EVAL_K, EVAL_K_OLD = 64, 16
KNOB = "n_loops"
EVAL_LEVEL, DIAG_LEVEL = int(os.environ.get("ND_EVAL_LEVEL", 8)), 2
TOK_MODE = "lean_seq"
BANDS = ((0, 6), (7, 9), (10, 14), (15, 19), (20, 24), (25, 34), (35, 49), (50, 200))
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def read(fn):
    op = gzip.open if fn.endswith(".gz") else open
    return [json.loads(line) for line in op(fn, "rt") if line.strip()]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sha(fn):
    return hashlib.sha256(open(fn, "rb").read()).hexdigest()[:12]


def band(body):
    return next(f"{lo}-{hi}" if hi < 200 else f"{lo}+" for lo, hi in BANDS if lo <= body <= hi)


class Timeout(Exception):
    pass


class BudgetSpent(Timeout):
    pass


# ---------------------------------------------------------------- sampling and judging


def body_lines(nd):
    """(written body lines, pruned body lines) of a decoded ND proof: lines after the premise block, and those the
    last line depends on through citations (a box rule cites the box's first and last lines, so the closure keeps a
    needed box's hypothesis and every line its conclusion depends on)."""
    lines = parse_proof_tokens(nd.split())
    by = {ln["idx"]: ln for ln in lines}
    need, todo = set(), [lines[-1]["idx"]]
    while todo:
        i = todo.pop()
        if i not in need:
            need.add(i)
            todo += by[i]["refs"]
    body = [ln for ln in lines if ln["rule"] != "PR"]
    return len(body), sum(1 for ln in body if ln["idx"] in need)


def kv_batch(model, n_tokens):
    layers = model.kv_layers() if hasattr(model, "kv_layers") else len(model.blocks)
    return max(32, min(1024, KV_TOKENS // (layers * n_tokens)))


def new_tokens(body, max_new):
    """Sampling cap for a target of `body` lines: reference proofs run up to ~23 tokens per body line (16 for most), so
    64 + 30 per line leaves >= 1.3x room, and a sample rambling without <eos> no longer holds a whole batch to max_new.
    Rounded up to a multiple of 128 so that a few caps group the samples into full batches."""
    return min(max_new, -(-(64 + 30 * body) // 128) * 128)


def sample(model, tok, recs, k, seed, max_new):
    """k samples per record at temperature TEMP. Returns per record a list of (ids, decoded ND proof, accepted)."""
    prompts = [r["prompt"] for r in recs for _ in range(k)]
    caps = [new_tokens(r["body"], max_new) for r in recs for _ in range(k)]
    ids = [tok.encode_prompt(p) for p in prompts]
    gen = torch.Generator(device=DEV)
    gen.manual_seed(seed)
    outs, nds, texts = [None] * len(prompts), [None] * len(prompts), [None] * len(prompts)
    for cap in sorted(set(caps)):
        order = sorted((i for i in range(len(prompts)) if caps[i] == cap), key=lambda i: len(ids[i]))
        batch = kv_batch(model, len(ids[order[-1]]) + cap)
        for s in range(0, len(order), batch):
            chunk = order[s : s + batch]
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(DEV == "cuda")):
                res = generate_ids(model, tok, [ids[i] for i in chunk], False, TEMP, cap, gen)
            for i, o in zip(chunk, res, strict=True):
                o = o[: o.index(tok.eos) + 1] if tok.eos in o else o
                outs[i], nds[i], texts[i] = o, tok.decode(o), tok.last_text
    if DEV == "cuda":
        torch.cuda.empty_cache()
    gated = gate(tok, prompts, nds, texts)
    flat = [(o, nd, not nd.startswith("LEAN")) for o, nd in zip(outs, gated, strict=True)]
    return [flat[i * k : (i + 1) * k] for i in range(len(recs))]


def judge(recs, per):
    rows = []
    for r, samples in zip(recs, per, strict=True):
        good = {}
        for _, nd, ok in samples:
            if ok and nd not in good:
                good[nd] = body_lines(nd)
        rows.append({"name": r["name"], "body": r["body"], "solved": bool(good), "n_ok": sum(ok for *_, ok in samples),
                     "n_tried": len(samples), "proofs": list(good), "written": [w for w, _ in good.values()],
                     "pruned": [p for _, p in good.values()]})
    return rows


def lstar(solved, recs, need=5):
    ls = [r["body"] for r in recs if r["name"] in solved]
    ge = {L: sum(1 for x in ls if x >= L) for L in range(7, 81)}
    return max([L for L, c in ge.items() if c >= need], default=0), ge


def by_band(solved, recs):
    by = collections.defaultdict(lambda: [0, 0])
    for r in recs:
        by[band(r["body"])][0] += r["name"] in solved
        by[band(r["body"])][1] += 1
    return {b: {"solved": k, "n": n, "rate": k / n, "ci": list(wilson(k, n))} for b, (k, n) in sorted(by.items())}


def gate_seconds():
    fn = os.environ.get("LEAN_GATE_LOG")
    return sum(r["lean_wall_s"] for r in read(fn)) if fn and os.path.exists(fn) else 0.0


# ---------------------------------------------------------------- the RL oracle


class Oracle:
    """What rl() gets instead of data: the model's own attempts at the RL targets, judged by Lean."""

    def __init__(self, a, tok, targets, found=None, call0=0):
        self.a, self.tok = a, tok
        self.targets = {t["name"]: t for t in targets}
        self.found = collections.defaultdict(dict)  # name -> {proof text: {written, pruned, round}}
        self.found.update(found or {})  # a resumed run's earlier proofs
        self.call0 = call0  # a resumed run's earlier oracle calls: call numbers and sampling seeds continue
        self.calls = []
        self.t0 = self.last = time.time()
        self.uncharged = 0.0

    def used(self):
        return time.time() - self.t0 - self.uncharged

    def left(self):
        return self.a.rl_secs - self.used()

    def _sample(self, model, targets, k):
        if self.left() <= 0:
            raise BudgetSpent(f"oracle called with {self.used():.0f}s of {self.a.rl_secs}s used")
        recs = [self.targets[t["name"]] for t in targets]
        call = self.call0 + len(self.calls) + 1
        model.eval()
        t0, g0 = time.time(), gate_seconds()
        per = sample(model, self.tok, recs, k, self.a.seed * 100_000 + call, self.a.max_new)
        secs, gate_s = time.time() - t0, gate_seconds() - g0
        self.uncharged += gate_s
        rows = judge(recs, per)
        new = 0
        for t, row in zip(recs, rows, strict=True):
            for p, w, pl in zip(row["proofs"], row["written"], row["pruned"], strict=True):
                if p not in self.found[t["name"]]:
                    self.found[t["name"]][p] = {"written": w, "pruned": pl, "round": call}
                    new += 1
        solved = {n for n, d in self.found.items() if d}
        ls, _ = lstar(solved, list(self.targets.values()))
        n = len(recs) * k
        self.calls.append({"round": call, "level": getattr(model, KNOB, None), "k": k, "n_targets": len(recs),
                           "sample_acc": sum(x["n_ok"] for x in rows) / max(n, 1), "new_proofs": new,
                           "targets_solved": len(solved), "lstar": ls,
                           "solved_by_band": {b: v["solved"] for b, v in by_band(solved, list(self.targets.values())).items()},
                           "sampling": {"samples": n, "secs": secs, "gate_secs": gate_s},
                           "secs": time.time() - self.last, "rl_secs_used": self.used()})
        self.last = time.time()
        c = self.calls[-1]
        log(f"oracle {call}: acc {c['sample_acc']:.4f}, +{new} proofs, solved {len(solved)}/{len(self.targets)}, "
            f"L*={ls}, {KNOB} {c['level']}, {len(recs)} targets x {k}, {n / secs:.0f} samples/s, "
            f"RL clock {c['rl_secs_used']:.0f}/{self.a.rl_secs}s")
        return recs, per, rows

    def sample(self, model, targets, k):
        """k attempts per target. Returns per target {name, prompt, body, proofs (distinct Lean-accepted ND proofs),
        n_ok, n_tried}. Raises BudgetSpent once the budget is used up."""
        recs, _, rows = self._sample(model, targets, k)
        return [{"name": t["name"], "prompt": t["prompt"], "body": t["body"], "proofs": row["proofs"],
                 "n_ok": row["n_ok"], "n_tried": row["n_tried"]} for t, row in zip(recs, rows, strict=True)]

    def sample_rollouts(self, model, targets, k):
        """As sample(), plus `rollouts`: every attempt's proof token ids (eos included when the model ended it) and
        its Lean verdict, for policy-gradient RL."""
        recs, per, rows = self._sample(model, targets, k)
        return [{"name": t["name"], "prompt": t["prompt"], "body": t["body"], "proofs": row["proofs"],
                 "n_ok": row["n_ok"], "n_tried": row["n_tried"], "rollouts": [(o, ok) for o, _, ok in samples]}
                for t, row, samples in zip(recs, rows, per, strict=True)]


# ---------------------------------------------------------------- stages


def load_resume(a, targets):
    """A finished run's directory (final.pt, pretrain.pt, found_targets.jsonl, summary.json) -> (final state, pretrained
    state, found proofs of current targets as Oracle.found entries, earlier oracle calls, earlier RL seconds)."""
    final = torch.load(f"{a.resume}/final.pt", map_location=DEV)
    pre = torch.load(f"{a.resume}/pretrain.pt", map_location=DEV)
    arm = {k: os.environ.get(k) for k in ("ND_ARCH", "ND_RL")}
    assert final["arm"] == arm and final["seed"] == a.seed, (final["arm"], final["seed"], arm, a.seed)
    names = {t["name"] for t in targets}
    found = collections.defaultdict(dict)
    for r in read(f"{a.resume}/found_targets.jsonl"):
        if r["name"] in names:
            found[r["name"]][r["proof"]] = {k: r[k] for k in ("written", "pruned", "round")}
    prior = json.load(open(f"{a.resume}/summary.json"))["rl"]
    return final["state"], pre["state"], found, len(prior["rounds"]), prior["rl_secs_used"]


def stage_rl(a, agent, tok, model, targets, train_recs, dev, resume=None):
    found, call0 = (resume["found"], resume["calls"]) if resume else (None, 0)
    oracle = Oracle(a, tok, targets, found, call0)
    oracle.gold = dev  # gold dev proofs, for the RL probes' likelihood only; never trained on
    oracle.pretrained_state = resume["pretrained"] if resume else None  # the probes' pretrained reference
    visible = [{"name": t["name"], "prompt": t["prompt"], "body": t["body"], "band": t["band"]} for t in targets]
    stats = agent.rl(model, tok, visible, train_recs, oracle, a.rl_secs, DEV, a.seed, log)
    if DEV == "cuda":
        torch.cuda.synchronize()
    model.eval()
    with open(f"{a.out}/found_targets.jsonl", "w") as f:
        for t in targets:
            for p, x in oracle.found[t["name"]].items():
                f.write(json.dumps({"name": t["name"], "prompt": t["prompt"], "body": t["body"], "proof": p, **x}) + "\n")
    used = oracle.used()
    log(f"rl returned: {used:.0f}s of {a.rl_secs}s charged, {oracle.uncharged:.0f}s Lean gate")
    return {"rounds": oracle.calls, "secs": time.time() - oracle.t0, "rl_secs_used": used,
            "gate_secs": oracle.uncharged, "targets": len(targets), "stats": dict(stats or {})}


def eval_pool(a, tok, model, pool, k, level, tag):
    setattr(model, KNOB, level)
    model.eval()
    t0 = time.time()
    rows = judge(pool, sample(model, tok, pool, k, a.seed * 100_000 + 7777, a.max_new))
    solved = {r["name"] for r in rows if r["solved"]}
    ls, _ = lstar(solved, pool)
    out = {"level": level, "k": k, "solved": len(solved), "n": len(pool), "lstar": ls, "by_band": by_band(solved, pool),
           "sample_acc": sum(r["n_ok"] for r in rows) / max(k * len(pool), 1),
           "max_pruned": max((max(r["pruned"]) for r in rows if r["solved"]), default=0), "secs": time.time() - t0}
    with open(f"{a.out}/eval_{tag}.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    log(f"eval {tag} at {KNOB} {level}: solved {len(solved)}/{len(pool)}, L*={ls}, "
        f"by band { {b: v['solved'] for b, v in out['by_band'].items()} }, {out['secs']:.0f}s")
    return out


def stage_eval(a, tok, model, dev_pool, old_dev):
    out = {"dev": eval_pool(a, tok, model, dev_pool, a.eval_k, a.eval_level, "dev")}
    out["old_dev"] = eval_pool(a, tok, model, old_dev, a.eval_k_old, a.eval_level, "old_dev")
    if not getattr(model, "untied", False):
        out["dev_diag"] = eval_pool(a, tok, model, dev_pool, a.eval_k, a.diag_level, "dev_diag")
    setattr(model, KNOB, a.eval_level)
    return out


def write_summary(a, status, **kw):
    s = {"status": status, "metric": kw.get("eval", {}).get("dev", {}).get("solved"),
         "metric_old_dev": kw.get("eval", {}).get("old_dev", {}).get("solved"),
         "metric_rl": (kw.get("rl", {}).get("rounds") or [{}])[-1].get("targets_solved"),
         "arm": {k: os.environ.get(k) for k in ("ND_ARCH", "ND_RL")}, "harness_sha": sha(f"{HERE}/harness.py"),
         "pretrain_sha": sha(f"{HERE}/pretrain.py"), "device": DEV,
         "gpu": torch.cuda.get_device_name(0) if DEV == "cuda" else "cpu", "torch": torch.__version__,
         "total_secs": time.time() - a.T0,
         "rules": {k: globals()[k] for k in ("TOTAL_SECS", "PRETRAIN_SECS", "RL_SECS", "PARAM_CAP", "MAX_NEW",
                                              "EVAL_K", "EVAL_K_OLD", "EVAL_LEVEL", "DIAG_LEVEL", "TOK_MODE")},
         **kw}
    with open(f"{a.out}/summary.json", "w") as f:
        json.dump(s, f, indent=1)
    log(f"SUMMARY status={status} metric={s['metric']} old_dev={s['metric_old_dev']} metric_rl={s['metric_rl']} "
        f"total {s['total_secs']:.0f}s")
    return s


def save_checkpoint(a, model, tok, stage):
    path = Path(a.out) / f"{stage}.pt"
    pending = path.with_suffix(".pt.tmp")
    try:
        torch.save({"state": model.state_dict(), "tok_mode": tok.mode, "stage": stage, "seed": a.seed,
                    "arm": {k: os.environ.get(k) for k in ("ND_ARCH", "ND_RL")},
                    "harness_sha": sha(f"{HERE}/harness.py"), "pretrain_sha": sha(f"{HERE}/pretrain.py")}, pending)
        os.replace(pending, path)
    finally:
        pending.unlink(missing_ok=True)


def load_pools(a):
    train = read(f"{DATA}/pretrain.jsonl.gz")
    assert all(r["body"] <= 6 for r in train), "pretraining file violates the 6-body-line cap"
    targets, dev = read(f"{DATA}/rl.jsonl"), read(f"{DATA}/dev.jsonl")
    old = [r for r in read(f"{FORK}/data/ladder/transfer.jsonl")
           if int(hashlib.sha1(r["key"].encode()).hexdigest(), 16) % 2 == 0]  # the loops harness's dev half
    old = [{**r, "body": r["n_lines"] - r["n_prem"]} for r in old]
    tk, dk, pk = ({r["key"] for r in xs} for xs in (targets, dev, train))
    assert not (tk & dk) and not (tk & pk) and not (dk & pk), "theorem leak between pools"
    if a.smoke:
        rng = random.Random(0)
        train = train[:3000]
        targets, dev, old = rng.sample(targets, 12), rng.sample(dev, 6), old[:6]
    return train, targets, dev, old


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pretrain-only", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="tiny everything, for a CPU plumbing check")
    ap.add_argument("--resume", help="a finished run's dir: load its final weights and found proofs, skip pretraining, "
                                     "and continue RL (expert iteration restarts its optimizers every round anyway)")
    ap.add_argument("--rl-hours", type=float, help="RL budget override, e.g. for a resumed run's extra hours")
    a = ap.parse_args()
    a.pretrain_secs, a.rl_secs, a.eval_k, a.eval_k_old, a.max_new = PRETRAIN_SECS, RL_SECS, EVAL_K, EVAL_K_OLD, MAX_NEW
    if a.rl_hours:
        a.rl_secs = int(a.rl_hours * 3600)
    a.eval_level, a.diag_level = EVAL_LEVEL, DIAG_LEVEL
    if a.smoke:
        a.pretrain_secs, a.rl_secs, a.eval_k, a.eval_k_old, a.max_new = 20, 60, 2, 2, 160
    os.makedirs(a.out, exist_ok=True)
    src = Path(a.out) / "source"
    src.mkdir(exist_ok=True)
    for fn in ("pretrain.py", "harness.py", "remote.sh", "overlay/lean_tok.py", "overlay/lean_gate.py"):
        shutil.copy2(Path(HERE) / fn, src / os.path.basename(fn))
    a.T0 = time.time()
    torch.manual_seed(a.seed)

    def on_alarm(signum, frame):
        raise Timeout()

    signal.signal(signal.SIGALRM, on_alarm)
    signal.alarm(TOTAL_SECS if not a.smoke else 1800)
    train, targets, dev, old = load_pools(a)
    agent = importlib.import_module("pretrain")
    assert getattr(agent, "GRPO_TEMP", TEMP) == TEMP, "GRPO must score the distribution it samples from"
    for name in ("lean_tok", "lean_gate"):  # the Lean-only gate and the widened tokenizer, nothing from the fork
        assert os.path.dirname(sys.modules[name].__file__) == os.path.join(HERE, "overlay"), sys.modules[name].__file__
    log(f"pools: pretrain {len(train):,}, RL targets {len(targets):,}, dev {len(dev)}, old dev {len(old)}; "
        f"arm {os.environ.get('ND_ARCH')}/{os.environ.get('ND_RL')}; device {DEV}; "
        f"harness {sha(f'{HERE}/harness.py')} pretrain {sha(f'{HERE}/pretrain.py')}")
    pre, resume = {}, None
    try:
        t0 = time.time()
        if a.resume:
            final, pretrained, found, calls, prior_secs = load_resume(a, targets)
            model, tok, _ = agent.pretrain(train, 1e-9, DEV, a.seed, log)  # zero steps: only builds the model and tokenizer
            model.load_state_dict(final)
            resume = {"found": found, "calls": calls, "pretrained": pretrained}
            pre = {"resumed_from": a.resume, "prior_rl_secs": prior_secs, "prior_calls": calls}
            log(f"resumed from {a.resume}: prior RL clock {prior_secs:.0f}s, {calls} oracle calls, "
                f"{len(found):,} targets already solved, RL budget now {a.rl_secs}s")
        else:
            model, tok, pre = agent.pretrain(train, a.pretrain_secs, DEV, a.seed, log)
        if DEV == "cuda":
            torch.cuda.synchronize()
        pre = dict(pre or {})
        pre["secs"] = time.time() - t0
        model.eval()
        pre["n_params"] = n_params = sum(p.numel() for p in model.parameters())
        longest = max(len(tok.encode_proof(r["proof"])) for r in targets + dev if r.get("proof"))  # ladder: none
        pre["longest_reference_tokens"] = longest
        log(f"pretrain returned after {pre['secs']:.0f}s (budget {a.pretrain_secs}s), {n_params:,} params; "
            f"longest reference proof {longest} tokens, MAX_NEW {a.max_new}")
        assert longest <= a.max_new or a.smoke, f"a reference proof ({longest} tokens) exceeds MAX_NEW"
        if pre["secs"] > a.pretrain_secs + PRETRAIN_GRACE and not a.smoke and not a.resume:
            return write_summary(a, "timeout", pretrain=pre, reason="pretrain overran its budget")
        if tok.mode != TOK_MODE or n_params > PARAM_CAP or not isinstance(getattr(model, KNOB, None), int):
            return write_summary(a, "crash", pretrain=pre, reason="tok mode, param cap or knob check failed")
        if not a.resume:
            save_checkpoint(a, model, tok, "pretrain")
        if a.pretrain_only:
            return write_summary(a, "ok", stage="pretrain_only", pretrain=pre)
        rl = stage_rl(a, agent, tok, model, targets, train, dev, resume)
        if rl["rl_secs_used"] > a.rl_secs + RL_GRACE and not a.smoke:
            return write_summary(a, "timeout", pretrain=pre, rl=rl, reason="rl overran its budget")
        save_checkpoint(a, model, tok, "final")
        ev = stage_eval(a, tok, model, dev, old)
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
