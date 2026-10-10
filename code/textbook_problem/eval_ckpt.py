"""Score one of Robbie's lean_seq checkpoints on the textbook-problem datasets.

    cd code && uv run python textbook_problem/eval_ckpt.py --run scale-d384-p1200_s0 --stage pretrain \
        --sets v0,v1,hand14,batch3,lean_v2,derived,textbook72 --k 32 --temp 1.0 --max-new 400 \
        --out ../experiments/textbook-problem/<exp>/run

The checkpoint (HF bucket robbiethompson2018/nd-rl-checkpoints, runs/<run>/<stage>.pt) is an
ALiBiGPT with PeriLN blocks defined in the run's own source/pretrain.py, which imports the fork at
commit 51604b3 (model.py, lean_tok.py, tokenizer.py, train.py). Both are loaded from local copies:
the HF cache for the run, FORK for the fork clone. Tokens are lean_seq: the model reads the Lean
statement of the sequent and writes Lean tactic text, which the fork's lean_tok.inverse turns back
into an ND proof body. Every proof is then judged by the pinned nd_verify through strict_verify
(prove.safe_verify); Lean itself is not consulted.

Prefix tasks (derived set): the given ND lines are turned into the lean_seq token prefix of the
reference proof that reveals nothing beyond those lines except the unavoidable `have n : G :=`
header of a box that is already open at the cut (lean_seq states a box's conclusion where the box
opens). The model continues from there; prefix + continuation is decoded and judged as a whole.
Rows whose task_input needs words outside the vocabulary (repair, cloze) are skipped.

Writes to --out: <set>.jsonl (every judged sample), metrics.jsonl (one line per set), status.md.
"""

import argparse
import collections
import datetime as dt
import json
import math
import os
import random
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "../ws3")]
import ndpath  # noqa: E402,F401
import prove  # noqa: E402  (pins nd_verify / strict_verify / logic)
from data import load_textbook72  # noqa: E402
from derive import DROPPED, KEPT, structural_ok  # noqa: E402

import torch  # noqa: E402

DATA = os.path.join(HERE, "../../data/textbook_problem")
HF_SNAP = os.path.expanduser(
    "~/.cache/huggingface/hub/models--robbiethompson2018--nd-rl-checkpoints/snapshots"
)
# the fork (dan-pandori/nd-takehome) at 51604b3, the commit Robbie's scale-* runs import; kept outside the repo and
# outside the session scratchpad (which is wiped between sessions):
#   git clone -q https://github.com/dan-pandori/nd-takehome ~/.cache/nd-rl/fork51604b3 && git -C ~/.cache/nd-rl/fork51604b3 checkout -q 51604b3
FORK = os.environ.get("ND_FORK", os.path.expanduser("~/.cache/nd-rl/fork51604b3"))
UNFEEDABLE = {"repair", "cloze_justifications", "cloze_formulas"}


# ------------------------------------------------------------------ model
def find_run(run):
    for snap in sorted(os.listdir(HF_SNAP)):
        d = os.path.join(HF_SNAP, snap, "runs", run)
        if os.path.isdir(d):
            return d
    raise FileNotFoundError(run)


def load_robbie(run, stage="pretrain", device="cpu"):
    run_dir = find_run(run)
    for m in ("model", "tokenizer", "lean_tok", "train", "sample", "pretrain"):
        sys.modules.pop(m, None)
    sys.path[:0] = [FORK, os.path.join(run_dir, "source")]
    import pretrain as rp  # noqa: E402  (Robbie's run source; imports the fork's model/lean_tok)
    import sample as fork_sample  # noqa: E402
    import lean_tok  # noqa: E402

    tok = rp.make_tokenizer(rp.TOK_MODE)
    model = rp.ALiBiGPT(tok.vocab_size, rp.N_LAYER, rp.D, rp.N_HEAD)
    model.blocks = torch.nn.ModuleList(rp.PeriLNBlock(rp.D, rp.N_HEAD) for _ in range(rp.N_LAYER))
    ck = torch.load(os.path.join(run_dir, f"{stage}.pt"), map_location=device, weights_only=False)
    model.load_state_dict(ck["state"])
    model.to(device).eval()
    meta = {k: v for k, v in ck.items() if k != "state"}
    meta.update(run=run, stage=stage, n_params=sum(p.numel() for p in model.parameters()),
                arch=dict(n_layer=rp.N_LAYER, d=rp.D, n_head=rp.N_HEAD, d_ff=getattr(rp, "D_FF", None)),
                vocab=tok.vocab_size, tok_mode=rp.TOK_MODE, fork=FORK)
    return model, tok, meta, fork_sample, lean_tok


# ------------------------------------------------------------------ prefixes
def box_starts(lines):
    """closing-line idx -> idx of the first AS it discharges (IMPI/NEGI: refs[0]; ORE: refs[1])."""
    out = {}
    for ln in lines:
        if ln["rule"] in ("IMPI", "NEGI"):
            out[ln["idx"]] = ln["refs"][0]
        elif ln["rule"] == "ORE":
            out[ln["idx"]] = ln["refs"][1]
    return out


def lean_prefix_tokens(lean_tok, body, k):
    """lean_seq tokens (strings / ('n', idx) tuples) of the reference proof `body`, cut so that no
    line beyond the first k ND lines is revealed, except open-box headers. Returns (tokens, n_tokens_total)."""
    from patterns import parse as parse_proof

    lines = parse_proof(body)
    starts = box_starts(lines)
    toks = lean_tok.proof_tokens(body)
    owner = None
    cut = len(toks)
    i = 0
    while i < len(toks):
        t = toks[i]
        if t == "have" and i + 1 < len(toks) and isinstance(toks[i + 1], tuple):
            m = toks[i + 1][1]
            if m > k:
                a = starts.get(m)
                if a is None or a > k:  # a plain line, or a box not yet opened by line k
                    cut = i
                    break
            owner = m
        elif t == "fun" and i + 2 < len(toks) and isinstance(toks[i + 2], tuple):
            a = toks[i + 2][1]
            if a > k:
                cut = i - 1 if i > 0 and toks[i - 1] == "(" else i
                break
            owner = a
        elif t == "exact" and i + 1 < len(toks) and isinstance(toks[i + 1], tuple):
            # closes a box (or the proof): belongs to the box's closing line
            j = toks[i + 1][1]
            closing = next((m for m, a in starts.items() if a <= j <= m and m > j), None)
            if closing is not None and closing > k:
                cut = i
                break
            if closing is None and k < len(lines):  # final exact of the whole proof
                cut = i
                break
        i += 1
    return toks[:cut], len(toks)


def renumber(tok, toks):
    """lean tokens -> ids with names numbered by first appearance (as encode_proof does), no eos."""
    order, out = {}, []
    for t in toks:
        if isinstance(t, tuple):
            n = order.setdefault(t[1], len(order) + 1)
            out.append(tok.ref0 + n - 1)
        else:
            out.append(tok.stoi[t])
    return out


# ------------------------------------------------------------------ data
def jl(path):
    return [json.loads(l) for l in open(path)]


def load_sets(names):
    sets = {}
    for n in names:
        if n == "v0":
            sets[n] = [dict(name=r["name"], prompt=r["prompt"]) for r in jl(f"{DATA}/candidate_v0/problems.jsonl")]
        elif n == "v1":
            sets[n] = [dict(name=r["name"], prompt=r["prompt"]) for r in jl(f"{DATA}/candidate_v1/problems.jsonl")]
        elif n == "hand14":
            sets[n] = [dict(name=r["name"], prompt=r["prompt"]) for r in jl(f"{DATA}/candidates_v3/hand_proofs.jsonl")]
        elif n == "batch3":
            sets[n] = [dict(name=r["name"], prompt=r["prompt"]) for r in jl(f"{DATA}/candidates_v3/batch3_proved.jsonl")]
        elif n == "lean_v2":
            rows = jl(f"{DATA}/release_2026-09-30_v2/lean.jsonl") + jl(f"{DATA}/release_2026-09-30_v2/lean_unsolved.jsonl")
            sets[n] = [dict(name=r["name"], prompt=r["prompt"]) for r in rows]
        elif n == "textbook72":
            sets[n] = [dict(name=r.get("name") or r.get("thm"), prompt=r["prompt"]) for r in load_textbook72()]
        elif n == "textbook72_distract2":  # the sealed 72 with two unused premises each (gitignored, aggregate only)
            sets[n] = [dict(name=r["name"], prompt=r["prompt"]) for r in jl(f"{HERE}/../../docs/textbook-problem/textbook72/textbook72_distract2.jsonl")]
        elif n == "derived":
            rows = []
            for nu in KEPT:
                if nu in UNFEEDABLE:
                    continue
                for r in jl(f"{DATA}/candidates_derived/{nu}.jsonl"):
                    rows.append(dict(name=r["name"], prompt=r["prompt"], nuance=nu, check=r.get("check"),
                                     given=r["given"] if r["task_input"] != r["prompt"] else None,
                                     reference_proof=r["reference_proof"], pair_id=r.get("pair_id"),
                                     variant=r.get("variant"), k_distractors=r.get("k_distractors"),
                                     base=r["base"]["name"]))
            sets[n] = rows
        else:
            raise ValueError(n)
    return sets


# ------------------------------------------------------------------ sampling / judging
def sample_set(model, tok, fork_sample, lean_tok, recs, k, temperature, max_new, batch, seed, log):
    """Returns per record: list of (ids, body, ok, reason, n_lines, struct_ok) over its k samples."""
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    prompt_ids, prefix_info = [], []
    for r in recs:
        ids = tok.encode_prompt(r["prompt"])
        pre, total = [], 0
        if r.get("given"):
            n_given = len([s for s in r["given"].split(" ; ") if s.strip(" ;")])
            pre_toks, total = lean_prefix_tokens(lean_tok, r["reference_proof"], n_given)
            pre = renumber(tok, pre_toks)
            ids = ids + pre
        prompt_ids.append(ids)
        prefix_info.append((len(pre), total))
    flat = [(i, j) for i in range(len(recs)) for j in range(k)]
    order = sorted(range(len(flat)), key=lambda x: len(prompt_ids[flat[x][0]]))
    outs = [None] * len(flat)
    t0 = time.time()
    for s in range(0, len(order), batch):
        chunk = order[s : s + batch]
        res = fork_sample.generate_ids(
            model, tok, [prompt_ids[flat[x][0]] for x in chunk], greedy=False, temperature=temperature,
            max_new=max_new, gen=gen,
        )
        for x, o in zip(chunk, res):
            outs[x] = o
        if log and (s // batch) % 10 == 0:
            done = min(s + batch, len(order))
            log(f"   {done}/{len(order)} samples, {time.time() - t0:.0f}s")
    judged = [[] for _ in recs]
    for (i, j), o in zip(flat, outs):
        pre_len = prefix_info[i][0]
        pre_ids = prompt_ids[i][len(prompt_ids[i]) - pre_len :] if pre_len else []
        body = tok.decode(pre_ids + o)
        if body.startswith("LEANPARSE"):
            ok, reason, n = False, body, 0
        else:
            v = prove.safe_verify(recs[i]["prompt"], body)
            ok, reason, n = bool(v[0]), v[1], v[2]
        sok = structural_ok(body, recs[i]["check"]) if ok and recs[i].get("check") else None
        judged[i].append(dict(ok=ok, reason=reason, n_lines=n, n_tokens=len([x for x in o if x != tok.pad]),
                              struct_ok=sok, body=body if ok else body[:200]))
    return judged, time.time() - t0, prefix_info


def summarize(recs, judged, k, group=None):
    """avg solve rate (mean over problems of the verified fraction), any-of-k, CI over problems."""
    def per_problem(js, r):
        oks = [x["ok"] and (x["struct_ok"] is not False) for x in js]
        return sum(oks) / len(oks)
    rates = [per_problem(js, r) for js, r in zip(judged, recs)]
    n = len(rates)
    mean = sum(rates) / n if n else 0.0
    sd = statistics.pstdev(rates) if n > 1 else 0.0
    ci = 1.96 * sd / math.sqrt(n) if n else 0.0
    solved = sum(1 for x in rates if x > 0)
    parse_fail = sum(1 for js in judged for x in js if x["reason"].startswith("LEANPARSE")) / max(1, sum(len(js) for js in judged))
    out = dict(n=n, k=k, avg_solve_rate=round(mean, 4), ci95=round(ci, 4), any_of_k=solved,
               any_of_k_rate=round(solved / n, 4) if n else 0.0, never_solved=n - solved,
               parse_fail_frac=round(parse_fail, 4))
    if group:
        g = collections.defaultdict(list)
        for r, rate in zip(recs, rates):
            g[r[group]].append(rate)
        out["by_" + group] = {key: dict(n=len(v), avg_solve_rate=round(sum(v) / len(v), 4),
                                        any_of_k=sum(1 for x in v if x > 0)) for key, v in sorted(g.items())}
    return out, rates


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="scale-d384-p1200_s0")
    ap.add_argument("--stage", default="pretrain")
    ap.add_argument("--sets", default="v0,v1,hand14,batch3,lean_v2,derived,textbook72")
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--max-new", type=int, default=400)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="first N records per set (smoke test)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=0)
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    os.makedirs(a.out, exist_ok=True)
    logf = open(os.path.join(a.out, "log.txt"), "a")

    def log(msg):
        line = f"{dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')} {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    model, tok, meta, fork_sample, lean_tok = load_robbie(a.run, a.stage)
    log(f"loaded {a.run}/{a.stage}: {json.dumps(meta, default=str)}")
    with open(os.path.join(a.out, "status.md"), "a") as f:
        f.write(f"RUNNING {dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')} {a.run}/{a.stage} k={a.k} temp={a.temp} max_new={a.max_new} sets={a.sets}\n")
    sets = load_sets(a.sets.split(","))
    for name, recs in sets.items():
        if a.limit:
            recs = recs[: a.limit]
        log(f"== {name}: {len(recs)} records x k={a.k}")
        judged, secs, prefix_info = sample_set(model, tok, fork_sample, lean_tok, recs, a.k, a.temp, a.max_new, a.batch, a.seed, log)
        group = "nuance" if name == "derived" else None
        summ, rates = summarize(recs, judged, a.k, group)
        summ.update(set=name, run=a.run, stage=a.stage, temperature=a.temp, max_new=a.max_new, seed=a.seed,
                    secs=round(secs, 1), samples_per_sec=round(len(recs) * a.k / max(secs, 1e-9), 2),
                    utc=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"))
        log(f"   {name}: avg solve rate {summ['avg_solve_rate']} ± {summ['ci95']}, any-of-{a.k} {summ['any_of_k']}/{summ['n']}, parse-fail {summ['parse_fail_frac']}, {secs:.0f}s")
        with open(os.path.join(a.out, "metrics.jsonl"), "a") as f:
            f.write(json.dumps(summ) + "\n")
        # dumps: the sealed 72 go to the gitignored textbook72 folder, never the run folder
        dump = os.path.join(a.out, f"{name}.jsonl") if not name.startswith("textbook72") else os.path.join(
            HERE, "../../docs/textbook-problem/textbook72", f"samples_{a.run}_{a.stage}_k{a.k}{'' if name == 'textbook72' else '_' + name}.jsonl")
        os.makedirs(os.path.dirname(dump), exist_ok=True)
        with open(dump, "w") as f:
            for r, js, rate, pi in zip(recs, judged, rates, prefix_info):
                row = {k_: v for k_, v in r.items() if k_ != "reference_proof"}
                row.update(solve_rate=rate, prefix_tokens=pi[0], reference_lean_tokens=pi[1], samples=js)
                f.write(json.dumps(row) + "\n")
    with open(os.path.join(a.out, "status.md"), "a") as f:
        f.write(f"DONE {dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')}\n")
    log("done")


if __name__ == "__main__":
    main()
