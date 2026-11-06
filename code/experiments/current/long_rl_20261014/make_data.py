"""Build this experiment's data. Every proof written out is checked by Lean (never nd_verify).

  pretrain: Dan's cap-6 file (fork data/train.jsonl, 154,990 records, 0-3 premises, <= 6 body lines) with distractor
            premises: each record gets a total premise count drawn uniformly from [its own, 10]; each distractor is a
            premise or conclusion of another record with its atoms mapped at random into the ten atoms. A distractor is
            kept only if it is not already a premise or the conclusion, the premises stay jointly satisfiable, and the
            minimal proof does not get shorter (certify_fast, minlen's proof space). The proof's premise block is
            reordered to match and its body renumbered; the body is unchanged.
  pools:    RL targets, dev and holdout from the long-proofs set (../long_proofs_20261013/production/v2): body >= 7,
            deduplicated by canonical key, minus theorems in the group's files, split by key hash (dev 1/20, holdout
            1/20, RL the rest), capped per body-line band.

    ND_FORK=<fork at 51604b3> python make_data.py pretrain data/pretrain.jsonl.gz [--procs 6]
    ND_FORK=<fork at 51604b3> python make_data.py pools data/
Length is body lines (proof lines after the premise block), the system of record from 2026-10-14.
"""
import argparse
import collections
import glob
import gzip
import hashlib
import json
import os
import random
import re
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
LP = os.path.join(HERE, "..", "long_proofs_20261013")
# certify_fast first: it puts the vendored nd_verify (identical to the fork's) on the path and widens its atoms
sys.path[:0] = [os.path.join(LP, "certifier_exact"), os.path.join(LP, "production"), LP]
from certify_fast import certify_fast  # noqa: E402
from canon import canon_key  # noqa: E402

sys.path[:0] = [os.path.join(HERE, "overlay"), os.environ["ND_FORK"]]
from lean_gate import lean_check  # noqa: E402  (overlay)
from lean_tok import ATOMS, LeanTokenizer, fnd  # noqa: E402  (overlay)
from nd2lean import parse_prompt  # noqa: E402
from nd_verify.verify import parse_proof_tokens  # noqa: E402

TOK = LeanTokenizer("lean_seq")
BANDS = ((0, 6), (7, 9), (10, 14), (15, 19), (20, 24), (25, 34), (35, 49), (50, 200))
BAND_CAP = {"dev": 150, "holdout": 150, "rl": 2500}


def band(body):
    return next(f"{lo}-{hi}" if hi < 200 else f"{lo}+" for lo, hi in BANDS if lo <= body <= hi)


# ---------------------------------------------------------------- formulas


def atoms_of(f, acc):
    if f[0] == "atom":
        acc.add(f[1])
    elif f[0] != "bot":
        for g in f[1:]:
            atoms_of(g, acc)
    return acc


def rename(f, m):
    if f[0] == "atom":
        return ("atom", m[f[1]])
    if f[0] == "bot":
        return f
    return (f[0], *(rename(g, m) for g in f[1:]))


def table(f, cols):
    """Truth table of f as an int bitmask over the 2^n rows of `cols` (atom -> column mask)."""
    t = f[0]
    if t == "atom":
        return cols[f[1]]
    if t == "bot":
        return 0
    full = cols["*"]
    if t == "not":
        return full & ~table(f[1], cols)
    a, b = table(f[1], cols), table(f[2], cols)
    return a & b if t == "and" else a | b if t == "or" else (full & ~a) | b


def columns(names):
    n = len(names)
    rows = 1 << n
    cols = {"*": (1 << rows) - 1}
    for i, a in enumerate(names):
        cols[a] = sum(1 << r for r in range(rows) if r >> i & 1)
    return cols


def satisfiable(fs):
    cols = columns(sorted(set().union(*[atoms_of(f, set()) for f in fs])))
    m = cols["*"]
    for f in fs:
        m &= table(f, cols)
    return m != 0


def prompt_of(prem, concl):
    return "THM " + " , ".join(fnd(p) for p in prem) + (" SEQ " if prem else "SEQ ") + fnd(concl) + " PRF"


def renumber(proof, prem_pos, n_new_prem):
    """Rewrite an ND proof for a new premise block: old premise k (1-based) -> prem_pos[k-1]; body lines shift down
    by the number of added premises. Formulas, rules and depths are unchanged."""
    lines = parse_proof_tokens(proof.split())
    n_old = sum(ln["rule"] == "PR" for ln in lines)
    m = {k + 1: prem_pos[k] for k in range(n_old)}
    shift = n_new_prem - n_old
    for ln in lines:
        if ln["rule"] != "PR":
            m[ln["idx"]] = ln["idx"] + shift
    return m, lines


def emit(prem, lines, m):
    out = [f"N{i + 1} {fnd(p)} : PR ;" for i, p in enumerate(prem)]
    for ln in lines:
        if ln["rule"] == "PR":
            continue
        refs = "".join(f" N{m[r]}" for r in ln["refs"])
        out.append(f"N{m[ln['idx']]} " + "| " * ln["depth"] + f"{fnd(ln['formula'])} : {ln['rule']}{refs} ;")
    return " ".join(out) + " QED"


# ---------------------------------------------------------------- pretraining


def min_body(prompt, n_prem, floor):
    """Exact minimal body if < floor, else floor (None on timeout: treated as unsafe)."""
    r = certify_fast(prompt, n_prem, floor=floor, secs=5, mem_cap_mb=256)
    if r.get("min_body") is not None:
        return r["min_body"]
    return floor if r.get("cert_ge") else None


POOL = []  # distractor formulas, set per worker by init_pool


def init_pool(pool):
    POOL[:] = pool


def augment(args):
    rec, seed = args
    pool = POOL
    rng = random.Random(seed)
    prem, concl = parse_prompt(rec["prompt"])
    body = rec["n_lines"] - len(prem)
    m0 = min_body(rec["prompt"], len(prem), body)
    out = {"name": rec["name"], "src_n_prem": len(prem), "body": body, "tries": 0}
    want = rng.randint(len(prem), 10) - len(prem)
    added = []
    for _ in range(want):
        for _ in range(20):
            out["tries"] += 1
            f = pool[rng.randrange(len(pool))]
            fa = sorted(atoms_of(f, set()))
            g = rename(f, dict(zip(fa, rng.sample(ATOMS, len(fa)))))
            if g in prem or g in added or g == concl or g == ("bot",):
                continue
            if satisfiable(prem + added + [g]):
                added.append(g)
                break
    new_prem = prem + added
    order = list(range(len(new_prem)))
    rng.shuffle(order)
    prem_out = [new_prem[i] for i in order]
    pos = {old: new + 1 for new, old in enumerate(order)}  # index in new_prem (0-based) -> new premise line
    m, lines = renumber(rec["proof"], [pos[k] for k in range(len(prem))], len(prem_out))
    prompt = prompt_of(prem_out, concl)
    if added and (m0 is None or min_body(prompt, len(prem_out), m0) != m0):
        # a distractor opened a shortcut (or the search timed out): keep the record without distractors
        prem_out, prompt = prem, rec["prompt"]
        m, lines = renumber(rec["proof"], list(range(1, len(prem) + 1)), len(prem))
        out["shortcut"] = True
        added = []
    out.update(prompt=prompt, proof=emit(prem_out, lines, m), n_prem=len(prem_out), n_distract=len(added),
               key=canon_key(prem_out, concl))
    return out


def build_pretrain(out_path, procs, seed):
    recs = [json.loads(line) for line in open(os.path.join(os.environ["ND_FORK"], "data", "train.jsonl"))]
    pool = sorted({f for r in recs for f in (lambda pc: pc[0] + [pc[1]])(parse_prompt(r["prompt"]))})
    print(f"{len(recs):,} records, {len(pool):,} distinct distractor formulas", flush=True)
    jobs = [(r, seed * 10_000_000 + i) for i, r in enumerate(recs)]
    with ProcessPoolExecutor(procs, initializer=init_pool, initargs=(pool,)) as ex:
        outs = list(ex.map(augment, jobs, chunksize=500))
    ok = lean_ok([(o["prompt"], o["proof"]) for o in outs])
    bad = [o["name"] for o, k in zip(outs, ok) if not k]
    if bad:
        raise SystemExit(f"Lean rejected {len(bad)} augmented records, e.g. {bad[:5]}")
    with gzip.open(out_path, "wt") as f:
        for o in outs:
            f.write(json.dumps(o) + "\n")
    c = collections.Counter(o["n_prem"] for o in outs)
    print(f"wrote {len(outs):,} to {out_path}; Lean accepts all; premises {dict(sorted(c.items()))}; "
          f"shortcut fallbacks {sum(bool(o.get('shortcut')) for o in outs)}; "
          f"distractors {sum(o['n_distract'] for o in outs):,}", flush=True)


def lean_ok(pairs):
    ok, wall, _ = lean_check([TOK.lean_source(p, q) for p, q in pairs])
    print(f"Lean checked {len(pairs):,} proofs in {wall:.0f}s: {sum(ok):,} accepted", flush=True)
    return ok


# ---------------------------------------------------------------- RL / dev / holdout pools


def group_sources():
    """Text of every theorem file in the group's repos (as production/audit_overlap.py): nd-takehome targets/ and
    data/, dan/dan's data/**.jsonl, and every .jsonl on nd-rl's origin/charles/main."""
    th, ndrl = os.path.expanduser("~/repos/nd-takehome"), os.path.expanduser("~/repos/nd-rl")
    for p in glob.glob(f"{th}/targets/**/*.jsonl", recursive=True) + glob.glob(f"{th}/data/**/*.jsonl", recursive=True):
        yield open(p).read()
    for repo, ref, prefix in ((th, "dan/dan", "data/"), (ndrl, "origin/charles/main", "")):
        names = subprocess.run(["git", "-C", repo, "ls-tree", "-r", "--name-only", ref],
                               capture_output=True, text=True).stdout.split()
        for n in names:
            if n.endswith(".jsonl") and n.startswith(prefix):
                yield subprocess.run(["git", "-C", repo, "show", f"{ref}:{n}"], capture_output=True, text=True).stdout


def group_keys():
    """Canonical keys of every theorem in the group's files, so no target or eval theorem is one someone else trains
    or evaluates on."""
    keys = set()
    for text in group_sources():
        for m in set(re.findall(r"THM .*? PRF", text)):
            try:
                keys.add(canon_key(*parse_prompt(m)))
            except Exception:
                continue
    return keys


E_SIG = os.path.join(LP, "certifier_scoped", "results", "e_raw_sig.jsonl")
E_TEXT = {}  # prompt -> written proof text, from production/v2/e_raw.jsonl


def ladder_targets():
    """Dan's ladder RL targets (fork data/ladder/rl_targets.jsonl): statements only, body 4-13, 0-3 premises. The other
    half of the bridge: the old harness's targets, which a cap-6 model solves some of from the first round."""
    out = []
    for r in (json.loads(line) for line in open(os.path.join(os.environ["ND_FORK"], "data", "ladder", "rl_targets.jsonl"))):
        prem, concl = parse_prompt(r["prompt"])
        body = r["n_lines"] - len(prem)
        out.append({"key": canon_key(prem, concl), "prompt": r["prompt"], "n_prem": len(prem), "body": body,
                    "body_exact": False, "src": "ladder:" + r["name"], "band": band(body), "name": f"ladder_{len(out)}"})
    return out


def build_pools(out_dir):
    for line in open(os.path.join(LP, "production", "v2", "e_raw.jsonl")):
        r = json.loads(line)
        E_TEXT[r["text"].split(" PRF ", 1)[0] + " PRF"] = r["text"]
    recs = {}
    for p in sorted(glob.glob(os.path.join(LP, "production", "v2", "*_f20.jsonl"))) + [E_SIG]:
        for line in open(p):
            r = json.loads(line)
            if p == E_SIG:  # approach E (written 35-80) with the signature lower bound (certifier_scoped)
                r = {**r, "text": E_TEXT[r["prompt"]], "cert_ge": r["cert_ge_sig"]}
                if r["cert_ge_sig"] == r["n_body"]:
                    r["min_body"] = r["n_body"]
            if r.get("min_body") is not None:
                body, exact = r["min_body"], True
            elif (r.get("cert_ge") or 0) >= 20:
                body, exact = r["n_body"], False  # minimum in [cert_ge, written]
            else:
                continue
            # The bridge (Robbie, 2026-10-14): below 7 body lines only theorems with 3+ premises, the new family's
            # structure at pretraining lengths
            if body < 7 and (body < 2 or r["n_prem"] < 3):
                continue
            prem, concl = parse_prompt(r["prompt"])
            k = canon_key(prem, concl)
            proof = r.get("min_proof") or r["text"].split(" PRF ", 1)[1]
            prev = recs.get(k)
            if prev is None or (exact and not prev["body_exact"]):
                recs[k] = {"key": k, "prompt": r["prompt"], "proof": proof, "n_prem": len(prem),
                           "n_atoms": r["n_atoms"], "body": body, "body_exact": exact, "cert_ge": r.get("cert_ge"),
                           "n_body_written": r["n_body"], "src": os.path.basename(p)}
    print(f"{len(recs):,} labelled theorems with body >= 7", flush=True)
    taken = group_keys()
    pre = {json.loads(line)["key"] for line in gzip.open(os.path.join(out_dir, "pretrain.jsonl.gz"), "rt")}
    drop = [k for k in recs if k in taken or k in pre]
    print(f"dropping {len(drop)} that appear in the group's files or the pretraining set", flush=True)
    for k in drop:
        del recs[k]
    rs = sorted(recs.values(), key=lambda r: r["key"])
    ok = lean_ok([(r["prompt"], r["proof"]) for r in rs])
    rs = [r for r, k in zip(rs, ok) if k]
    split = {"dev": [], "holdout": [], "rl": []}
    for r in rs:
        h = int(hashlib.sha1(r["key"].encode()).hexdigest(), 16)
        split["dev" if h % 20 == 0 else "holdout" if h % 20 == 1 else "rl"].append(r)
    # ladder targets are the group's own RL targets, so they are kept; only theorems in the pretraining set, the old
    # transfer pool (evaluated as old dev) or the new pools are left out
    transfer = {canon_key(*parse_prompt(json.loads(line)["prompt"]))
                for line in open(os.path.join(os.environ["ND_FORK"], "data", "ladder", "transfer.jsonl"))}
    ladder = [r for r in ladder_targets() if r["key"] not in recs and r["key"] not in pre | transfer]
    for name, xs in split.items():
        random.Random(name).shuffle(xs)
        per = collections.Counter()
        kept = []
        for r in xs:
            b = band(r["body"])
            if per[b] < BAND_CAP[name]:
                per[b] += 1
                kept.append({**r, "band": b, "name": f"{name}_{len(kept)}"})
        if name == "rl":
            kept += ladder
            print(f"rl: + {len(ladder):,} ladder targets", flush=True)
        with open(os.path.join(out_dir, f"{name}.jsonl"), "w") as f:
            for r in kept:
                f.write(json.dumps(r) + "\n")
        prem = collections.Counter(r["n_prem"] for r in kept)
        print(f"{name}: {len(kept):,} theorems; by band {dict(sorted(per.items()))}; "
              f"premises {dict(sorted(prem.items()))}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("pretrain", "pools"))
    ap.add_argument("out")
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.what == "pretrain":
        build_pretrain(a.out, a.procs, a.seed)
    else:
        build_pools(a.out)


if __name__ == "__main__":
    main()
