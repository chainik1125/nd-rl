"""Frozen-checkpoint screens for Stage-1 pools (ideas.md § J6–J8): Robbie's K12 Stage-1 checkpoint is read, never trained.

For each pool's NEW slice (the rows that differ from K12; the `slice` tag, or the shortened rows of k12_minimal, or the
generator rows of k12_500k; the whole set for cap6), on the frozen `scale-d384-p1200_s0` pretrain checkpoint (9.56M,
lean_seq tokens, loaded exactly as `textbook_problem/eval_ckpt.py` loads it):

  J6  loss        mean per-token loss on the proof tokens (the trainer's own objective, no gradient step), with quantiles,
                  on up to --n_loss slice records; reference rows: K12's held-out head (records 0–1,023) and random K12
                  training rows. Regimes relative to the K12 training loss: redundant (< 1.5×), edge (1.5–4×), far (> 4×).
  J7  pass@32     sample 32 proofs at temperature 1.0 for up to --n_sample slice prompts, verify with the pinned verifier;
                  report mean solve rate (pass@1) and pass@32 over prompts. High = the base already does it; 0 = out of reach.
  J8  retention   the K12 rows the slice displaced, by type cell (citing rule, cited rule, depth): cells that lose ≥ 50 % of
                  their K12 records are listed (rare cells the slice removes).
      encodable   share of slice rows the lean_seq tokenizer can encode at all (rows it cannot encode would be dropped or
                  crash a lean_seq Stage-1; matters for k12_wrong*).

    cd code && nohup uv run python pretrain_data/screen_ckpt.py > ../data/pretrain_data/screens_ckpt.log 2>&1 &

Writes data/pretrain_data/screens/ckpt_screens.json and appends "## Checkpoint screens" to docs/pretrain-data/pools.md.
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..", "textbook_problem")]
import pool_common as PC  # noqa: E402
import eval_ckpt as EC  # noqa: E402
from patterns import parse, prune_lines  # noqa: E402
from strict_verify import safe_verify  # noqa: E402

import torch  # noqa: E402

OUT = os.path.join(PC.DATA, "screens")
DOC = os.path.join(PC.ROOT, "docs", "pretrain-data", "pools.md")
HEAD = "## Checkpoint screens (ideas.md § J6–J8): what the frozen K12 model makes of each new slice"
RUN = "scale-d384-p1200_s0"
VAL_N = 1024


def slice_rows(name, rows, k12):
    if name.startswith("cap6"):
        return rows, "whole set"
    if name == "k12_minimal":
        return [r for r in rows if r.get("minlen_status") == "shorter"], "shortened rows"
    if name == "k12_500k":
        return [r for r in rows if r.get("pool") in ("gen_cap6", "gen_long")], "generator rows"
    s = [r for r in rows if r.get("slice")]
    if s:
        return s, "slice rows"
    return [], "no new rows"


def encode(tok, rows):
    enc, bad = [], 0
    for r in rows:
        try:
            enc.append((tok.encode_prompt(r["prompt"]), tok.encode_proof(r["proof"])))
        except Exception:  # noqa: BLE001  (lean_seq cannot render it)
            bad += 1
    return enc, bad


def losses(model, tok, enc, bs=64):
    out = []
    order = sorted(range(len(enc)), key=lambda i: len(enc[i][0]) + len(enc[i][1]))
    with torch.no_grad():
        for s in range(0, len(order), bs):
            chunk = [enc[i] for i in order[s: s + bs]]
            T = max(len(p) + len(q) for p, q in chunk)
            x = torch.full((len(chunk), T), tok.pad, dtype=torch.long)
            m = torch.zeros((len(chunk), T), dtype=torch.bool)
            for j, (p, q) in enumerate(chunk):
                x[j, : len(p) + len(q)] = torch.tensor(p + q)
                m[j, len(p): len(p) + len(q)] = True
            logits = model(x[:, :-1])
            if isinstance(logits, tuple):
                logits = logits[0]
            l = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), x[:, 1:].reshape(-1),
                                                  reduction="none").reshape(len(chunk), -1)
            mk = m[:, 1:]
            out.extend(((l * mk).sum(1) / mk.sum(1)).tolist())
    return out


def pass_at_k(model, tok, fork_sample, rows, k, seed, max_new=400, bs=256):
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    prompts = []
    for r in rows:
        try:
            prompts.append(tok.encode_prompt(r["prompt"]))
        except Exception:  # noqa: BLE001
            prompts.append(None)
    flat = [(i, j) for i, p in enumerate(prompts) if p is not None for j in range(k)]
    order = sorted(range(len(flat)), key=lambda x: len(prompts[flat[x][0]]))
    outs = [None] * len(flat)
    for s in range(0, len(order), bs):
        chunk = order[s: s + bs]
        res = fork_sample.generate_ids(model, tok, [prompts[flat[x][0]] for x in chunk], greedy=False, temperature=1.0,
                                       max_new=max_new, gen=gen)
        for x, o in zip(chunk, res):
            outs[x] = o
    hits = collections.Counter()
    for (i, j), o in zip(flat, outs):
        body = tok.decode(o)
        if not body.startswith("LEANPARSE") and safe_verify(rows[i]["prompt"], body)[0]:
            hits[i] += 1
    n = len({i for i, _ in flat})
    if n == 0:
        return None, None
    return round(sum(hits.values()) / (n * k), 4), round(sum(1 for i in range(len(rows)) if hits[i] > 0) / n, 4)


def cell_counts(rows):
    c = collections.Counter()
    for r in rows:
        lines = parse(r["proof"])
        if not lines:
            continue
        by = {ln["idx"]: ln for ln in lines}
        seen = set()
        for ln in prune_lines(lines):
            for ref in ln["refs"]:
                if ref in by and by[ref]["rule"] != "PR":
                    seen.add((ln["rule"], by[ref]["rule"], min(ln["depth"], 3)))
        for cell in seen:
            c[cell] += 1
    return c


def regime_of(ratio, p32):
    """redundant: the base already solves the slice's problems (pass@32 ≥ 0.8) and predicts its proofs (loss ≤ 1.5× K12);
    far: it solves almost none (pass@32 < 0.1) or the proofs are alien (loss > 4×); edge otherwise. The loss ratio alone
    is too coarse here (K12's loss is 0.17 and short formulaic proofs score low even when the model cannot find them)."""
    if p32 is None:
        return "redundant" if ratio < 1.5 else "edge" if ratio <= 4 else "far"
    if p32 < 0.1 or ratio > 4:
        return "far"
    if p32 >= 0.8 and ratio <= 1.5:
        return "redundant"
    return "edge"


def pool_files():
    files = {"k12": None, "cap6 (take-home Stage-1 set)": os.path.join(PC.ROOT, "code", "ws3", "fork_pin", "data", "train.jsonl.gz")}
    for sub in ("", "size", "scan", "atoms", "k"):
        for p in sorted(glob.glob(os.path.join(PC.DATA, "pools", sub, "*.jsonl.gz"))):
            name = os.path.basename(p)[: -len(".jsonl.gz")]
            if name != "gen_depth3":
                files[name] = p
    return files


def write_doc(ref, results):
    W = [f"\n{HEAD}\n",
         f"Built by `screen_ckpt.py` on the frozen `{RUN}` pretrain checkpoint (9.56M, lean_seq), no training. Each row is a pool's *new* "
         "slice only (the rest of every pool is K12). `loss` = mean per-token loss on proof tokens (the trainer's objective), with the median "
         f"and 90th percentile over records; K12 reference: held-out head {ref['val_loss']}, training rows {ref['train_loss']} (median "
         f"{ref['train_med']}). `×K12` = slice loss ÷ K12 training loss; regime: redundant < 1.5, edge 1.5–4, far > 4. `pass@1` / `pass@32` = "
         "mean solve rate and any-of-32 over slice prompts at temperature 1.0, pinned verifier (K12 training prompts: "
         f"{ref['train_p1']} / {ref['train_p32']}; held-out head: {ref['val_p1']} / {ref['val_p32']}). Regime: far if pass@32 < 0.1 or loss > 4×; "
         "redundant if pass@32 ≥ 0.8 and loss ≤ 1.5×; edge otherwise. For slices whose prompts are K12's own problems (k12_minimal, k12_wrong*) or "
         "renamings of them (k12_subst*), pass@k describes the problems, not the slice's proofs. `enc` = share of slice rows lean_seq can "
         "encode. `lost cells` = type cells (citing, cited, depth) that lose ≥ 50 % of their K12 records to the slice (meaningless for the 100 % pools, "
         "which displace everything).\n",
         "| pool | slice | n slice | enc | loss | med | p90 | ×K12 | regime | pass@1 | pass@32 | lost cells | secs |",
         "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for s in results:
        if s.get("skipped"):
            W.append(f"| {s['label']} | {s['slice']} | 0 | | | | | | | | | | |")
            continue
        W.append(f"| {s['label']} | {s['slice']} | {s['n_slice']:,} | {s['enc']:.3f} | {s['loss']} | {s['med']} | {s['p90']} | {s['ratio']} | {s['regime']} | "
                 f"{s['pass1']} | {s['pass32']} | {s['lost_cells']}{(' (' + s['lost_top'] + ')') if s['lost_top'] else ''} | {s['secs']} |")
    W.append("")
    text = open(DOC).read()
    if HEAD in text:
        i = text.index(HEAD)
        j = text.find("\n## ", i + len(HEAD))
        text = text[:i].rstrip("\n") + "\n" + (text[j:] if j > 0 else "")
    with open(DOC, "w") as f:
        f.write(text.rstrip("\n") + "\n" + "\n".join(W) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pools", nargs="*")
    ap.add_argument("--n_loss", type=int, default=3000)
    ap.add_argument("--n_sample", type=int, default=160)
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no_doc", action="store_true")
    ap.add_argument("--doc_only", action="store_true", help="rewrite the pools.md section from ckpt_screens.json (regimes recomputed)")
    a = ap.parse_args()
    if a.doc_only:
        d = json.load(open(os.path.join(OUT, "ckpt_screens.json")))
        for s_ in d["pools"]:
            if not s_.get("skipped"):
                s_["regime"] = regime_of(s_["ratio"], s_["pass32"])
        json.dump(d, open(os.path.join(OUT, "ckpt_screens.json"), "w"), indent=1, ensure_ascii=False)
        write_doc(d["ref"], d["pools"])
        print("rewrote the checkpoint-screens section")
        return
    rng = random.Random(a.seed)
    os.makedirs(OUT, exist_ok=True)
    model, tok, meta, fork_sample, lean_tok = EC.load_robbie(RUN, "pretrain", "cpu")
    print("loaded", RUN, meta["n_params"], "params", flush=True)
    k12 = PC.load_k12()
    k12_cells = cell_counts(k12)
    rp = os.path.join(OUT, "ckpt_screens.json")
    prev = json.load(open(rp)) if os.path.exists(rp) else {}
    # K12 reference numbers
    if "ref" in prev:
        ref = prev["ref"]
    else:
        t0 = time.time()
        val_rows = k12[:VAL_N]
        tr_rows = rng.sample(k12[VAL_N:], a.n_loss)
        lv = losses(model, tok, encode(tok, val_rows)[0])
        lt = losses(model, tok, encode(tok, tr_rows)[0])
        p1v, p32v = pass_at_k(model, tok, fork_sample, rng.sample(val_rows, a.n_sample), a.k, a.seed)
        p1t, p32t = pass_at_k(model, tok, fork_sample, rng.sample(tr_rows, a.n_sample), a.k, a.seed)
        ref = dict(val_loss=round(sum(lv) / len(lv), 4), train_loss=round(sum(lt) / len(lt), 4), train_med=round(sorted(lt)[len(lt) // 2], 4),
                   val_p1=p1v, val_p32=p32v, train_p1=p1t, train_p32=p32t, secs=round(time.time() - t0))
        print("ref", json.dumps(ref), flush=True)
    files = pool_files()
    names = a.pools or [n for n in files if n != "k12"]
    results = {s["label"]: s for s in prev.get("pools", [])}
    for name in names:
        t0 = time.time()
        rows = PC.read(files[name])
        sl, kind = slice_rows(name, rows, k12)
        if not sl:
            results[name] = dict(label=name, slice=kind, skipped=True)
            print(name, "skipped:", kind, flush=True)
            continue
        sample = rng.sample(sl, min(a.n_loss, len(sl)))
        enc, bad = encode(tok, sample)
        ls = losses(model, tok, enc) if enc else []
        p1, p32 = pass_at_k(model, tok, fork_sample, rng.sample(sl, min(a.n_sample, len(sl))), a.k, a.seed)
        # retention: cells of the displaced K12 rows
        lost = {}
        if kind == "slice rows" and all("k12_index" in r or True for r in sl):
            # displaced rows = K12 rows at the slice's positions; recover by prompt where the slice replaced the same row,
            # otherwise by position (the pools keep K12's order, so position i holds either K12[i] or its replacement)
            pos = [i for i, r in enumerate(rows) if r.get("slice")]
            disp = cell_counts([k12[i] for i in pos])
            lost = {c: (disp[c], k12_cells[c]) for c in disp if k12_cells[c] and disp[c] / k12_cells[c] >= 0.5}
        mean = sum(ls) / len(ls) if ls else float("nan")
        srt = sorted(ls)
        ratio = mean / ref["train_loss"] if ls else float("nan")
        regime = regime_of(ratio, p32)
        top = sorted(lost.items(), key=lambda x: -x[1][1])[:3]
        s = dict(label=name, slice=kind, n_slice=len(sl), enc=round(1 - bad / max(1, len(sample)), 4), loss=round(mean, 4),
                 med=round(srt[len(srt) // 2], 4) if srt else None, p90=round(srt[int(0.9 * len(srt))], 4) if srt else None,
                 ratio=round(ratio, 2), regime=regime, pass1=p1, pass32=p32, lost_cells=len(lost),
                 lost_top="; ".join(f"{'←'.join(c[:2])}@{c[2]} {d}/{t}" for c, (d, t) in top), secs=round(time.time() - t0))
        results[name] = s
        print(json.dumps(s), flush=True)
        json.dump(dict(run=RUN, ref=ref, pools=[results[n] for n in files if n in results]), open(rp, "w"), indent=1, ensure_ascii=False)
    if not a.no_doc:
        write_doc(ref, [results[n] for n in files if n in results])
        print("wrote", DOC)


if __name__ == "__main__":
    main()
