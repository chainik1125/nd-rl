"""Frozen-checkpoint screens on Dan's lean_staten checkpoints: the Stage-1 base the plan's ladder starts from, and the
round-8 ladder model (what RL did to each slice). Companion to screen_ckpt.py (Robbie's lean_seq model).

For each pool's new slice (same definition as screen_ckpt.py):
  loss     mean action-token loss in the proof-state format: every proof is decomposed by Dan's `state_env.decompose`
           into (state, action) pairs and scored with his `state_train.loss_on` objective, no gradient step; `enc` = share
           of slice rows that decompose (rows that do not would be skipped by his trainer).
  pass@k   Dan's environment sampler (`state_sample.env_generate`, T 1.0, max 48 steps, 256 action tokens) on up to
           --n_sample prompts × k; every returned proof is judged by the PINNED nd_verify through strict_verify, not by
           Dan's Lean judge, so the numbers are comparable with screen_ckpt.py's.

Import order matters: the pinned verifier and the pool helpers are imported first and cached in sys.modules, then Dan's
worktree (scratchpad/fork_dan, commit 0c492902) is put at the front of sys.path for model / tokenizer / state_env /
state_sample. ND_REGISTRY=0 keeps Dan's results registry from writing rows.

    cd code && ND_REGISTRY=0 nohup uv run python pretrain_data/screen_ckpt_dan.py --ckpt stage1_best12_s0_b1200 \
        > ../data/pretrain_data/screens_dan_stage1.log 2>&1 &
    cd code && ND_REGISTRY=0 nohup uv run python pretrain_data/screen_ckpt_dan.py --ckpt la_T1_best12_s0_r8 \
        > ../data/pretrain_data/screens_dan_r8.log 2>&1 &
    cd code && uv run python pretrain_data/screen_ckpt_dan.py --doc_only      # table from the two JSONs

Writes data/pretrain_data/screens/ckpt_dan_<ckpt>.json and the section "## Checkpoint screens on Dan's lean_staten
checkpoints" in docs/pretrain-data/pools.md.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import random
import sys
import time

os.environ.setdefault("ND_REGISTRY", "0")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pool_common as PC  # noqa: E402  (pins nd_verify / strict_verify / patterns first)
from strict_verify import safe_verify  # noqa: E402
import nd_verify  # noqa: E402  (cached now, so Dan's copy below is never imported)
import screen_ckpt as SC  # noqa: E402  (slice_rows, cell_counts, pool_files)

# Dan's worktree (fork_dan, nd-takehome at 0c492902) and checkpoints (dan_ckpts) live under ~/.cache/nd-rl, outside the
# session scratchpad that was wiped twice. Override with DAN_SCRATCH=... . Imported lazily by
# load_dan() so that --doc_only, which only reads the two JSONs, works without them.
SCRATCH = os.environ.get("DAN_SCRATCH", os.path.expanduser("~/.cache/nd-rl"))  # persistent; was a session scratchpad
FORK_DAN = os.path.join(SCRATCH, "fork_dan")
CKPTS = os.path.join(SCRATCH, "dan_ckpts")
torch = load_ckpt = decompose = ParseFail = env_generate = None


def load_dan():
    """Put Dan's worktree at the front of sys.path and import his model / tokenizer / state env (after the pinned
    verifier is cached in sys.modules, so his nd_verify copy is never used)."""
    global torch, load_ckpt, decompose, ParseFail, env_generate
    if not os.path.isdir(FORK_DAN):
        sys.exit(f"Dan's worktree not found at {FORK_DAN}; set DAN_SCRATCH or re-create it (see module docstring)")
    sys.path.insert(0, FORK_DAN)
    for m in ("model", "tokenizer", "lean_tok", "state_env", "state_sample", "state_train", "best_model", "record"):
        sys.modules.pop(m, None)
    import torch as _torch
    from model import load_ckpt as _load_ckpt  # Dan's
    from state_env import decompose as _decompose
    from lean_tok import ParseFail as _ParseFail
    from state_sample import env_generate as _env_generate
    torch, load_ckpt, decompose, ParseFail, env_generate = _torch, _load_ckpt, _decompose, _ParseFail, _env_generate

OUT = os.path.join(PC.DATA, "screens")
DOC = os.path.join(PC.ROOT, "docs", "pretrain-data", "pools.md")
HEAD = "## Checkpoint screens on Dan's lean_staten checkpoints: the ladder's own base, and after 8 RL rounds"
VAL_N = 1024


def pairs_of(tok, rows):
    """-> (list of per-record [(state ids, action ids)], n_failed)"""
    out, bad = [], 0
    for r in rows:
        try:
            steps, toks, env = decompose(r["prompt"], r["proof"], canon=getattr(tok, "canon", False))
        except (ParseFail, ValueError, AssertionError, IndexError, KeyError):
            bad += 1
            continue
        ps = []
        for stt, act, hs in steps:
            pre = tok.encode_toks((hs + stt) if getattr(tok, "with_history", False) else stt)
            ps.append((list(pre), list(tok.encode_toks(act)) + [tok.eos]))
        out.append(ps)
    return out, bad


def losses(model, tok, data, bs=256):
    """per-record mean action-token loss (sum of token losses over all of the record's pairs ÷ action tokens)"""
    flat = [(i, p, q) for i, ps in enumerate(data) for p, q in ps]
    flat.sort(key=lambda x: len(x[1]) + len(x[2]))
    tot = collections.defaultdict(float)
    cnt = collections.defaultdict(int)
    with torch.no_grad():
        for s in range(0, len(flat), bs):
            chunk = flat[s: s + bs]
            T = max(len(p) + len(q) for _, p, q in chunk)
            x = torch.full((len(chunk), T), tok.pad, dtype=torch.long)
            m = torch.zeros((len(chunk), T), dtype=torch.bool)
            for j, (_, p, q) in enumerate(chunk):
                x[j, : len(p) + len(q)] = torch.tensor(p + q)
                m[j, len(p): len(p) + len(q)] = True
            logits = model(x[:, :-1])
            if isinstance(logits, tuple):
                logits = logits[0]
            l = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), x[:, 1:].reshape(-1),
                                                  reduction="none").reshape(len(chunk), -1)
            mk = m[:, 1:]
            per = (l * mk).sum(1)
            n = mk.sum(1)
            for j, (i, _, _) in enumerate(chunk):
                tot[i] += float(per[j])
                cnt[i] += int(n[j])
    return [tot[i] / max(1, cnt[i]) for i in range(len(data))]


def pass_at_k(model, tok, rows, k, seed, batch=256):
    prompts = [r["prompt"] for r in rows for _ in range(k)]
    stats = {}
    bodies = env_generate(model, tok, prompts, greedy=False, temperature=1.0, max_action=256, max_steps=48,
                          batch=batch, seed=seed, stats=stats)
    hits = collections.Counter()
    for idx, body in enumerate(bodies):
        i = idx // k
        if body and not body.startswith("LEANPARSE") and safe_verify(rows[i]["prompt"], body)[0]:
            hits[i] += 1
    n = len(rows)
    return round(sum(hits.values()) / (n * k), 4), round(sum(1 for i in range(n) if hits[i] > 0) / n, 4)


def write_doc():
    robbie = {}
    rp = os.path.join(OUT, "ckpt_screens.json")
    if os.path.exists(rp):
        robbie = {s["label"]: s for s in json.load(open(rp))["pools"] if not s.get("skipped")}
    runs = {}
    for name in ("stage1_best12_s0_b1200", "la_T1_best12_s0_r8"):
        p = os.path.join(OUT, f"ckpt_dan_{name}.json")
        if os.path.exists(p):
            runs[name] = json.load(open(p))
    if not runs:
        return
    st = runs.get("stage1_best12_s0_b1200", {})
    r8 = runs.get("la_T1_best12_s0_r8", {})
    refs = {k: v.get("ref", {}) for k, v in runs.items()}
    W = [f"\n{HEAD}\n",
         "Built by `screen_ckpt_dan.py`. The plan's Stage-1 and ladder run in Dan's `lean_staten` proof-state format, so this is the screen on the "
         "base those arms start from (`stage1_best12_s0_b1200`, 9.56M, 1,200 s on K12) and, for comparison, on the same seed after 8 rounds of the "
         "T1 ladder (`la_T1_best12_s0_r8`). `loss` = action-token loss in the state format (Dan's objective), `×K12` = ÷ the checkpoint's own loss on "
         "K12 training rows; `enc` = share of slice rows his decomposition accepts. `pass@1` / `pass@32` = Dan's environment sampler at T 1.0, "
         "every proof judged by the pinned nd_verify (not his Lean judge). Robbie's lean_seq pass@32 (previous section) is repeated for comparison. "
         f"References — Stage-1: K12 training rows loss {refs.get('stage1_best12_s0_b1200', {}).get('train_loss', '?')}, pass@32 "
         f"{refs.get('stage1_best12_s0_b1200', {}).get('train_p32', '?')}, held-out head {refs.get('stage1_best12_s0_b1200', {}).get('val_p32', '?')}; "
         f"round 8: loss {refs.get('la_T1_best12_s0_r8', {}).get('train_loss', '?')}, pass@32 {refs.get('la_T1_best12_s0_r8', {}).get('train_p32', '?')}, "
         f"held-out head {refs.get('la_T1_best12_s0_r8', {}).get('val_p32', '?')}.\n",
         "| pool | slice | n | enc | Stage-1 loss | ×K12 | Stage-1 pass@1 | Stage-1 pass@32 | r8 pass@1 | r8 pass@32 | Δ pass@32 (RL) | Robbie pass@32 |",
         "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    names = list(SC.pool_files())
    s_rows = {s["label"]: s for s in st.get("pools", [])}
    r_rows = {s["label"]: s for s in r8.get("pools", [])}
    for n in names:
        a, b = s_rows.get(n), r_rows.get(n)
        if not a and not b:
            continue
        if (a or b).get("skipped"):
            continue
        f = lambda s, k, fmt="{}": fmt.format(s[k]) if s and s.get(k) is not None else ""  # noqa: E731
        d = ""
        if a and b and a.get("pass32") is not None and b.get("pass32") is not None:
            d = f"{b['pass32'] - a['pass32']:+.2f}"
        W.append(f"| {n} | {(a or b)['slice']} | {(a or b)['n_slice']:,} | {f(a, 'enc', '{:.3f}')} | {f(a, 'loss')} | {f(a, 'ratio')} | "
                 f"{f(a, 'pass1')} | {f(a, 'pass32')} | {f(b, 'pass1')} | {f(b, 'pass32')} | {d} | {robbie.get(n, {}).get('pass32', '')} |")
    W.append("")
    text = open(DOC).read()
    if HEAD in text:
        i = text.index(HEAD)
        j = text.find("\n## ", i + len(HEAD))
        text = text[:i].rstrip("\n") + "\n" + (text[j:] if j > 0 else "")
    with open(DOC, "w") as fh:
        fh.write(text.rstrip("\n") + "\n" + "\n".join(W) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="stage1_best12_s0_b1200")
    ap.add_argument("--pools", nargs="*")
    ap.add_argument("--n_loss", type=int, default=1500)
    ap.add_argument("--n_sample", type=int, default=128)
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no_doc", action="store_true")
    ap.add_argument("--doc_only", action="store_true")
    ap.add_argument("--redo", action="store_true", help="re-run pools already in the JSON")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if a.doc_only:
        write_doc()
        print("rewrote the Dan checkpoint-screens section")
        return
    rng = random.Random(a.seed)
    load_dan()
    assert "fork_pin" in nd_verify.__file__, nd_verify.__file__
    model, tok, extra = load_ckpt(os.path.join(CKPTS, a.ckpt + ".pt"))
    n_params = sum(p.numel() for p in model.parameters())
    print("loaded", a.ckpt, n_params, "params, tok", type(tok).__name__, getattr(tok, "mode", None), flush=True)
    k12 = PC.load_k12()
    rp = os.path.join(OUT, f"ckpt_dan_{a.ckpt}.json")
    prev = json.load(open(rp)) if os.path.exists(rp) else {}
    if "ref" in prev:
        ref = prev["ref"]
    else:
        t0 = time.time()
        val_rows = k12[:VAL_N]
        tr_rows = rng.sample(k12[VAL_N:], a.n_loss)
        dv, bv = pairs_of(tok, val_rows)
        dt, bt = pairs_of(tok, tr_rows)
        lv, lt = losses(model, tok, dv), losses(model, tok, dt)
        p1v, p32v = pass_at_k(model, tok, rng.sample(val_rows, a.n_sample), a.k, a.seed)
        p1t, p32t = pass_at_k(model, tok, rng.sample(tr_rows, a.n_sample), a.k, a.seed)
        ref = dict(val_loss=round(sum(lv) / len(lv), 4), train_loss=round(sum(lt) / len(lt), 4), train_med=round(sorted(lt)[len(lt) // 2], 4),
                   val_enc=round(1 - bv / len(val_rows), 4), train_enc=round(1 - bt / len(tr_rows), 4),
                   val_p1=p1v, val_p32=p32v, train_p1=p1t, train_p32=p32t, secs=round(time.time() - t0))
        print("ref", json.dumps(ref), flush=True)
    files = SC.pool_files()
    names = a.pools or [n for n in files if n != "k12"]
    results = {s["label"]: s for s in prev.get("pools", [])}
    for name in names:
        if name in results and not results[name].get("skipped") and not a.redo:
            print(name, "already done; kept", flush=True)
            continue
        t0 = time.time()
        rows = PC.read(files[name])
        sl, kind = SC.slice_rows(name, rows, k12)
        if not sl:
            results[name] = dict(label=name, slice=kind, skipped=True)
            print(name, "skipped:", kind, flush=True)
            continue
        sample = rng.sample(sl, min(a.n_loss, len(sl)))
        data, bad = pairs_of(tok, sample)
        ls = losses(model, tok, data) if data else []
        p1, p32 = pass_at_k(model, tok, rng.sample(sl, min(a.n_sample, len(sl))), a.k, a.seed)
        mean = sum(ls) / len(ls) if ls else float("nan")
        srt = sorted(ls)
        ratio = mean / ref["train_loss"] if ls else float("nan")
        s = dict(label=name, slice=kind, n_slice=len(sl), enc=round(1 - bad / max(1, len(sample)), 4), loss=round(mean, 4),
                 med=round(srt[len(srt) // 2], 4) if srt else None, p90=round(srt[int(0.9 * len(srt))], 4) if srt else None,
                 ratio=round(ratio, 2), regime=SC.regime_of(ratio, p32), pass1=p1, pass32=p32, secs=round(time.time() - t0))
        results[name] = s
        print(json.dumps(s), flush=True)
        json.dump(dict(ckpt=a.ckpt, n_params=n_params, ref=ref, pools=[results[n] for n in files if n in results]), open(rp, "w"), indent=1,
                  ensure_ascii=False)
    if not a.no_doc:
        write_doc()
        print("wrote", DOC)


if __name__ == "__main__":
    main()
