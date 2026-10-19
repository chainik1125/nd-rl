"""Shape profile of ND theorems/proofs under Dan's lean_staten interface (CPU only).
For each (prompt, ND proof): premises, atoms, lines, box depth, ORE count / nesting, DN count, goal tokens,
max formula tokens in the proof, and per-step state / action token counts from state_env.decompose(canon=True)."""

import json
import random
import statistics as st
import sys

sys.path.insert(0, "/scratch/cowifornia/nd-rl/think/nd-takehome")
from lean_tok import ftoks
from nd2lean import parse_prompt
from nd_verify.verify import parse_proof_tokens
from state_env import decompose


def feats(prompt, proof):
    prem, concl = parse_prompt(prompt)
    lines = parse_proof_tokens(proof.split())
    atoms = set(t for t in prompt.split() if t in ("P", "Q", "R", "S"))
    rules = [l["rule"] for l in lines]
    depth = max((l["depth"] for l in lines), default=0)
    # ORE nesting: depth of ORE-opened boxes is not directly tagged; approximate by count + depth
    n_ore = rules.count("ORE")
    n_dn = rules.count("DN")
    ftok = max(len(ftoks(l["formula"])) for l in lines)
    gtok = len(ftoks(concl))
    ptok = max([len(ftoks(p)) for p in prem] or [0])
    out = dict(
        n_prem=len(prem),
        n_atoms=len(atoms),
        L=len(lines),
        box_depth=depth,
        n_ore=n_ore,
        n_dn=n_dn,
        max_ftok=ftok,
        goal_tok=gtok,
        max_prem_tok=ptok,
    )
    try:
        steps, toks, env = decompose(prompt, proof, canon=True)
        out["n_actions"] = len(steps)
        out["max_state"] = max(len(s) for s, a, h in steps)
        out["max_action"] = max(len(a) for s, a, h in steps)
        out["max_pair"] = max(len(s) + len(a) for s, a, h in steps)
        out["proof_tok"] = len(toks)
    except Exception as e:
        out["decomp_err"] = str(e)[:80]
    return out


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))]


def summarise(
    name,
    rows,
    keys=(
        "n_prem",
        "n_atoms",
        "L",
        "box_depth",
        "n_ore",
        "n_dn",
        "max_ftok",
        "goal_tok",
        "n_actions",
        "max_state",
        "max_action",
        "max_pair",
        "proof_tok",
    ),
):
    print(f"## {name} (n={len(rows)})")
    for k in keys:
        xs = [r[k] for r in rows if r.get(k) is not None]
        if not xs:
            continue
        print(
            f"  {k:11s} median {st.median(xs):6.1f}  p90 {pct(xs, 0.9):5}  p99 {pct(xs, 0.99):5}  max {max(xs):5}"
        )
    errs = sum("decomp_err" in r for r in rows)
    if errs:
        print("  decomp errors", errs)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--k12", type=int, default=20000)
    a = ap.parse_args()
    rng = random.Random(0)
    k12 = [
        json.loads(l)
        for l in open("/scratch/cowifornia/nd-rl/bench/dan_ckpts/data/train_k12.jsonl")
    ]
    sample = rng.sample(k12, min(a.k12, len(k12)))
    rows = [feats(r["prompt"], r["proof"]) for r in sample]
    summarise("K12 Stage-1 (sample)", rows)
    json.dump(rows, open("k12_profile.json", "w"))
    rt = [
        json.loads(l)
        for l in open("/scratch/cowifornia/nd-rl/think/nd-takehome/data/rl_targets.jsonl")
    ]
    print(rt[0].keys())


def eval_sets():
    D = "/scratch/cowifornia/nd-rl/think/nd-takehome/data"
    sets = {}
    for f in [
        "tb72_textbook_dev",
        "tb72_textbook_train",
        "candidate_v0",
        "candidate_v1",
        "batch3",
        "hand_proved_14",
    ]:
        rs = [json.loads(l) for l in open(f"{D}/gt/{f}.jsonl")]
        sets[f] = [(r["prompt"], r.get("reference_proof") or r.get("proof"), r) for r in rs]
    for f in ["transfer_long2", "transfer_long_rr600"]:
        rs = [json.loads(l) for l in open(f"{D}/ladder/{f}.jsonl")]
        sets[f] = [
            (r["prompt"], r.get("ub_proof") or r.get("gen_proof") or r.get("proof"), r) for r in rs
        ]
    return sets


def thm_feats(prompt):
    prem, concl = parse_prompt(prompt)
    atoms = set(t for t in prompt.split() if t in ("P", "Q", "R", "S"))
    return dict(
        n_prem=len(prem),
        n_atoms=len(atoms),
        goal_tok=len(ftoks(concl)),
        max_prem_tok=max([len(ftoks(p)) for p in prem] or [0]),
        state0=2 + sum(3 + len(ftoks(p)) for p in prem) + len(ftoks(concl)) + 1,
    )


def run_evals():
    sets = eval_sets()
    for name, lst in sets.items():
        rows = []
        for prompt, proof, r in lst:
            d = thm_feats(prompt)
            if proof:
                try:
                    d.update(feats(prompt, proof))
                except Exception as e:
                    d["feat_err"] = str(e)[:60]
            if "min_lines" in r:
                d["min_lines"] = r["min_lines"]
            if "L_true" in r:
                d["L_true"] = r["L_true"]
            rows.append(d)
        summarise(
            name,
            rows,
            keys=(
                "n_prem",
                "n_atoms",
                "goal_tok",
                "max_prem_tok",
                "state0",
                "min_lines",
                "L_true",
                "L",
                "box_depth",
                "n_ore",
                "n_dn",
                "max_ftok",
                "n_actions",
                "max_state",
                "max_action",
                "max_pair",
            ),
        )
        json.dump(rows, open(f"profile_{name}.json", "w"))
