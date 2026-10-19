"""Stage 1: natural generator theorems with exact minimal lengths ("pieces").

    python gen_pieces.py --out /scratch/.../test_21_30/work/pieces.jsonl --workers 20 --secs 1800 \
        --seed 210000

Each worker runs the fork's generator unchanged (`gen.sample_one`, the take-home knobs max_prem 3 /
max_depth 3; `--mode long` = the strict long generator that built the ladder / long-pool pools,
`--mode k12` = the take-home default generator K12 was drawn from, `--mode mix` = 50/50), keeps
proofs with a generated length >= --min_gen and no contradictory premises, and labels each theorem's
minimal ND length with `fastlen.staged` (the fork's `minlen` search space and rules, cross-budget
memo): stages bound 10 / --t10 s, 12 / --t12 s, 13 / --t13 s. A label L is final (no proof <= L-1 in
minlen's restricted space). Theorems with L in [--min_L, 13] are kept with their minimal proof
(`proof`, nd_verify-checked); timeouts and L >= 14 are dropped and counted.

Worker i uses seed + 1000 * i (streams never overlap). Rows: {thm, prompt, prem, concl, L, proof,
gen_proof, gen_lines, n_prem, n_atoms, mode, seed, key, secs}. `key` is the renaming class
(`gen.canon_key`).
"""

from __future__ import annotations

import argparse
import collections
import contextlib
import json
import multiprocessing as mp
import os
import random
import sys
import time

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
FORK = "/scratch/cowifornia/nd-rl/think/nd-takehome"
for p in (HERE, FORK):
    if p not in sys.path:
        sys.path.insert(0, p)


def _is_lit(f):
    return f[0] == "atom" or (f[0] == "not" and f[1][0] == "atom")


def finish_at(g, last):
    """`Gen.finish` with `last` (a depth-0 line of the generated body) as the conclusion: dependency
    pruning to `last`, the same triviality / strict checks, the same output record."""
    from gen import BOT, Fail, fstr

    for ln in g.prem + g.body:
        ln.keep = False
        ln.idx = None
    stack = [last]
    while stack:
        ln = stack.pop()
        if ln.keep:
            continue
        ln.keep = True
        stack.extend(ln.refs)
    prem = [p for p in g.prem if p.keep]
    body = [pl for pl in g.body[: g.body.index(last) + 1] if pl.keep]
    lines = prem + body
    for i, ln in enumerate(lines):
        ln.idx = i + 1
    if last.f in [p.f for p in prem]:
        raise Fail("trivial: conclusion is a premise")
    if any(p.f == BOT for p in prem):
        raise Fail("trivial: F premise")
    if g.strict and (any("F" in fstr(p.f) for p in prem) or "F" in fstr(last.f)):
        raise Fail("strict: F in theorem")
    toks = []
    for ln in lines:
        toks += (
            [f"N{ln.idx}"]
            + ["|"] * ln.depth
            + [fstr(ln.f), ":", ln.rule]
            + [f"N{r.idx}" for r in ln.refs]
            + [";"]
        )
    toks.append("QED")
    prems = " , ".join(fstr(p.f) for p in prem)
    prompt = f"THM {prems} SEQ {fstr(last.f)} PRF" if prem else f"THM SEQ {fstr(last.f)} PRF"
    contra = any(("not", p.f) in [q.f for q in prem] for p in prem)
    return {
        "contra_prem": contra,
        "thm": f"{prems} |- {fstr(last.f)}",
        "prompt": prompt,
        "proof": " ".join(toks),
        "n_lines": len(lines),
        "rules": sorted(set(pl.rule for pl in body)),
        "n_prem": len(prem),
    }


def literal_subtheorems(g, rng, strict):
    """The generator's forward mode (`Gen.forward`, take-home step mix; strict long settings or the
    K12 ones), then every depth-0 literal line of the body that is not a premise and not the AS of a
    box becomes a conclusion (`finish_at`). -> list of records (possibly empty)."""
    from gen import Fail

    g.ore_boxes = False
    g.ore_steps = 3 if strict else 1
    g.strict = strict
    g.bot_p = 0.0 if strict else 0.02
    g.reset(rng.choice([0, 1, 1, 2, 2, 3]))
    for _ in range(rng.randint(6, 22) if strict else rng.randint(4, 14)):
        g.step()
    while g.depth > 0:
        g.close_box()
    outs = []
    for ln in [
        pl for pl in g.body if pl.depth == 0 and _is_lit(pl.f) and pl.rule not in ("AS", "R")
    ]:
        with contextlib.suppress(Fail):
            outs.append(finish_at(g, ln))
    return outs


def worker(args):
    seed, secs, out, a = args
    from fastlen import staged
    from gen import Fail, Gen, canon_key, sample_one  # fork, read-only

    rng = random.Random(seed)
    g = Gen(rng, max_prem=3, max_depth=3)
    st = collections.Counter()
    seen = set()
    t_end = time.time() + secs

    def handle(r, mode, fo):
        if r["contra_prem"] or r["n_lines"] < a.min_gen or r["n_lines"] > a.max_gen:
            st["filtered"] += 1
            return
        if len(r["prompt"].split()) > a.max_prompt_tok:
            st["too_long_prompt"] += 1
            return
        key = canon_key(r["thm"])
        if key in seen:
            st["dup"] += 1
            return
        seen.add(key)
        lab = staged(r["prompt"], stages=((10, a.t10), (12, a.t12), (13, a.t13)))
        st["labelled"] += 1
        if lab.get("error"):
            st["label_error"] += 1
            return
        if lab["timeout"]:
            st[f"timeout_{lab.get('stage')}"] += 1
            return
        L = lab["L"]
        if L is None:
            st["L_ge14"] += 1
            return
        st[f"L{L}"] += 1
        if a.min_L > L:
            return
        atoms = sorted({t for t in r["prompt"].split() if t in ("P", "Q", "R", "S")})
        rec = dict(
            thm=r["thm"],
            prompt=r["prompt"],
            L=L,
            proof=lab["proof"],
            gen_proof=r["proof"],
            gen_lines=r["n_lines"],
            n_prem=r["n_prem"],
            n_atoms=len(atoms),
            mode=mode,
            seed=seed,
            key=key,
            secs=round(lab["secs"], 3),
        )
        fo.write(json.dumps(rec) + "\n")
        fo.flush()

    with open(out, "w") as fo:
        while time.time() < t_end:
            st["tries"] += 1
            mode = a.mode if a.mode != "mix" else ("long" if rng.random() < 0.5 else "k12")
            try:
                if a.goal == "literal":
                    rs = literal_subtheorems(g, rng, mode == "long")
                    mode = "lit_" + mode
                else:
                    rs = [sample_one(g, rng, mode == "long")]
            except (Fail, RecursionError):
                st["gen_fail"] += 1
                continue
            for r in rs:
                handle(r, mode, fo)
    return dict(st)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--secs", type=float, default=1800)
    ap.add_argument("--seed", type=int, default=210000)
    ap.add_argument("--mode", choices=["long", "k12", "mix"], default="mix")
    ap.add_argument(
        "--goal",
        choices=["any", "literal"],
        default="any",
        help="literal: goal mode with a literal goal (lemma pieces for literal cuts)",
    )
    ap.add_argument("--min_gen", type=int, default=9)
    ap.add_argument("--max_gen", type=int, default=60)
    ap.add_argument("--min_L", type=int, default=7)
    ap.add_argument("--max_prompt_tok", type=int, default=70)
    ap.add_argument("--t10", type=float, default=3.0)
    ap.add_argument("--t12", type=float, default=20.0)
    ap.add_argument("--t13", type=float, default=40.0)
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    jobs = [(a.seed + 1000 * i, a.secs, f"{a.out}.w{i}", a) for i in range(a.workers)]
    t0 = time.time()
    with mp.Pool(a.workers) as pool:
        stats = pool.map(worker, jobs)
    tot = collections.Counter()
    for s in stats:
        tot.update(s)
    seen = set()
    n = 0
    with open(a.out, "w") as fo:
        for _, _, path, _ in jobs:
            with open(path) as fh:
                for line in fh:
                    r = json.loads(line)
                    if r["key"] in seen:
                        tot["dup_across_workers"] += 1
                        continue
                    seen.add(r["key"])
                    fo.write(line)
                    n += 1
            os.remove(path)
    summ = dict(
        args={k: v for k, v in vars(a).items()},
        stats=dict(tot),
        kept=n,
        wall_s=round(time.time() - t0, 1),
    )
    with open(a.out + ".summary.json", "w") as fo:
        json.dump(summ, fo, indent=1)
    print(json.dumps(summ))


if __name__ == "__main__":
    main()
