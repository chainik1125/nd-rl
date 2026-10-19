#!/usr/bin/env python3
"""Kalmar truth-table prover vs our model: cost measurements behind the efficiency report.

Run from a folder holding tt_prover.py (nd-rl dan:code/tools/), fork/ (lean_check.py, record.py, nd2lean.py,
nd_verify/ from dan-pandori/nd-takehome dan) and data/ (stored reads, K12 sample). Writes results.json and prints
tables. textbook72 is sealed: only aggregate numbers are printed or written, never a theorem.

  python3 bench.py data results.json
"""
import json
import os
import random
import statistics as st
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "fork"))
sys.path.insert(0, HERE)
os.environ.setdefault("ND_RUN_ID", "kalmar-efficiency-bench")
import lean_check as LC  # noqa: E402
import nd2lean  # noqa: E402
import tt_prover as T  # noqa: E402

DATA, OUT = sys.argv[1], sys.argv[2]
EXT = ["P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "A", "B", "C"]   # never F (False)


def q(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))]


def gen_time(fn, reps=3):
    best = 1e9
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); best = min(best, time.perf_counter() - t0)
    return best


def lean_batch(srcs):
    res, wall, cpu = LC.check(srcs, workers=1)
    return res, cpu / max(1, len(srcs))


def n_atoms(prompt):
    body = prompt.split("THM", 1)[1].split("PRF", 1)[0].split()
    return len({t for t in body if t in ("P", "Q", "R", "S")})


results = {"host": os.uname().nodename, "machine": os.uname().machine}

# ---------------------------------------------------------------- 1. Kalmar on real theorems, by atom count (K12 sample)
k12 = [json.loads(l) for l in open(os.path.join(DATA, "k12_sample.jsonl"))]
rows = []
for n in (1, 2, 3, 4):
    recs = [r for r in k12 if r["n_atoms"] == n]
    srcs, gts = [], []
    for r in recs:
        src = T.solve(r["prompt"])
        assert src is not None, "K12 theorem judged invalid"
        srcs.append(src); gts.append(gen_time(lambda p=r["prompt"]: T.solve(p)))
    res, lean_s = lean_batch(srcs)
    sizes = [x["size"] for x in res if x["ok"]]
    rows.append({"atoms": n, "n": len(recs), "accepted": sum(x["ok"] for x in res), "gen_ms_median": 1e3 * st.median(gts),
                 "lean_ms_per_proof": 1e3 * lean_s, "size_median": st.median(sizes), "size_p90": q(sizes, 0.9),
                 "size_max": max(sizes), "k12_lines_median": st.median(r["n_lines"] for r in recs)})
results["kalmar_k12_by_atoms"] = rows

# ---------------------------------------------------------------- 2. Kalmar on the evaluation sets (aggregates only)
evalsets = {}
for name, path in (("holdout250", "tj_eval/s0_r8__h250_x1.jsonl"), ("textbook72", "tj_eval/s0_r8__tb72_x1.jsonl")):
    prompts = [json.loads(l)["prompt"] for l in open(os.path.join(DATA, path))]
    srcs = [T.solve(p) for p in prompts]
    invalid = sum(s is None for s in srcs)
    res, lean_s = lean_batch([s for s in srcs if s])
    sizes = [x["size"] for x in res if x["ok"]]
    evalsets[name] = {"n": len(prompts), "judged_invalid": invalid, "accepted": sum(x["ok"] for x in res),
                      "lean_ms_per_proof": 1e3 * lean_s, "size_median": st.median(sizes), "size_max": max(sizes),
                      "atoms_hist": {k: sum(n_atoms(p) == k for p in prompts) for k in (1, 2, 3, 4)}}
results["kalmar_evalsets"] = evalsets


# ---------------------------------------------------------------- 3. Kalmar scaling beyond 4 atoms (synthetic)
def solve_ext(prompt, atoms):
    T.ATOMS = tuple(EXT)
    prem, goal = T.parse_prompt(prompt)
    hyps = "".join(f" (h{i + 1} : {T.lean(p)})" for i, p in enumerate(prem))
    proof = T.split(atoms, prem, goal, {}, {})
    return f"theorem t ({' '.join(atoms)} : Prop){hyps} : {T.lean(goal)} :=\n  {proof}"


def chain(n):            # A1 > A2, ..., A(n-1) > An  |-  A1 > An
    a = EXT[:n]
    prem = " , ".join(f"( {a[i]} > {a[i + 1]} )" for i in range(n - 1))
    return f"THM {prem} SEQ ( {a[0]} > {a[-1]} ) PRF" if n > 1 else "THM SEQ ( P > P ) PRF"


def lem(n, rng):         # |- phi v ~phi, phi a random formula using every one of n atoms
    a = EXT[:n]

    def f(atoms):
        if len(atoms) == 1:
            return atoms[0] if rng.random() < 0.7 else f"( ~ {atoms[0]} )"
        k = rng.randint(1, len(atoms) - 1)
        return f"( {f(atoms[:k])} {rng.choice(['&', 'v', '>'])} {f(atoms[k:])} )"
    phi = f(rng.sample(a, n))
    return f"THM SEQ ( {phi} v ( ~ {phi} ) ) PRF"


rng = random.Random(0)
scale, lean_cap_s = [], 300
lean_ok_until = {"chain": True, "lem": True}
for n in range(1, 13):
    for fam, prompt in (("chain", chain(n)), ("lem", lem(n, rng))):
        atoms = EXT[:n]
        t0 = time.perf_counter(); src = solve_ext(prompt, atoms); g = time.perf_counter() - t0
        row = {"atoms": n, "family": fam, "gen_s": g, "chars": len(src),
               "steps": sum(src.count(k) for k in ("fun ", "Or.elim", "Or.inl", "Or.inr", "⟨", "False.elim", ".1", ".2"))}
        if lean_ok_until[fam]:
            res, wall, cpu = LC.check([src], workers=1)
            row.update({"lean_ok": res[0]["ok"], "size": res[0]["size"], "lean_s": cpu})
            if cpu > lean_cap_s or not res[0]["ok"]:
                lean_ok_until[fam] = False
        scale.append(row)
        print(json.dumps(row), flush=True)
results["kalmar_scaling"] = scale

# ---------------------------------------------------------------- 4. Our model, from stored k = 256 reads (plain, T 0.8)
model = {}
for ck, pat in (("r8", "tj_eval/s{s}_r8__{set}_x1.jsonl"), ("r16", "eval/s{s}_r16__{set}_x1.jsonl")):
    for setname in ("h250", "tb72"):
        per_seed = []
        for s in (0, 1, 2):
            rs = [json.loads(l) for l in open(os.path.join(DATA, pat.format(s=s, set=setname)))]
            ps = [r["n_ok"] / r["n_tried"] for r in rs]
            solved = [p for p in ps if p > 0]
            # expected attempts until the first success, sampling at most 256 times per theorem
            exp_att = [sum((1 - p) ** i for i in range(256)) for p in ps]
            per_seed.append({"n": len(ps), "pass1": st.mean(ps), "solved_256": len(solved),
                             "attempts_per_solve_median": st.median(1 / p for p in solved),
                             "attempts_per_solve_p90": q([1 / p for p in solved], 0.9),
                             "expected_attempts_per_theorem_cap256": st.mean(exp_att),
                             "by_atoms": {k: {"n": sum(n_atoms(r["prompt"]) == k for r in rs),
                                              "pass1": st.mean([p for p, r in zip(ps, rs, strict=True) if n_atoms(r["prompt"]) == k] or [0]),
                                              "solved": sum(p > 0 for p, r in zip(ps, rs, strict=True) if n_atoms(r["prompt"]) == k)}
                                          for k in (1, 2, 3, 4)}})
        model[f"{ck}_{setname}"] = per_seed
results["model_reads"] = model

# Term size of the model's own proofs: the shortest accepted proof per solved holdout250 theorem (r8, seed 0)
rs = [json.loads(l) for l in open(os.path.join(DATA, "tj_eval/s0_r8__h250_x1.jsonl"))]
srcs = []
for r in rs:
    if r["proofs"]:
        i = min(range(len(r["proofs"])), key=lambda j: r["written_lens"][j])
        try:
            srcs.append(nd2lean.translate(r["prompt"], r["proofs"][i], require_all_pr=False))
        except Exception:
            pass
res, lean_s = lean_batch(srcs)
msz = [x["size"] for x in res if x["ok"]]
results["model_proof_size_h250_r8_s0"] = {"n": len(srcs), "accepted": sum(x["ok"] for x in res), "median": st.median(msz),
                                          "p90": q(msz, 0.9), "max": max(msz), "lean_ms_per_proof": 1e3 * lean_s}
json.dump(results, open(OUT, "w"), indent=1)
print(json.dumps({k: v for k, v in results.items() if k != "kalmar_scaling"}, indent=1))
