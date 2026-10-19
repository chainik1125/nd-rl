"""Shape / diversity profile of a set of (prompt, ND proof) records, side by side with a K12 sample.

    python profile_set.py --set test_21_30.jsonl --field reference_proof --k12 20000 \
        --out profile.json

Per proof: premises, atoms, lines, theorem formula depth (max over premises and conclusion), box
depth, ORE count, rules used, prompt tokens, and over `state_env.decompose(canon=True)`: actions,
max state / action tokens, the largest canonical name. Summaries: histograms for the discrete
features, quantiles for the token counts, the rule mix (share of proofs using each rule, mean uses
per proof), and the share of distinct renaming classes (`gen.canon_key` of the theorem, premise
order kept) and of premise-order-invariant classes (`ndbench` pkey).
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import random
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from ndproof import atoms_of, fdepth, parse  # noqa: E402
from state_env import decompose  # noqa: E402  (fork, read-only)

K12 = "/scratch/cowifornia/nd-rl/bench/dan_ckpts/data/train_k12.jsonl"
RULES = [
    "PR",
    "AS",
    "R",
    "ANDI",
    "ANDE1",
    "ANDE2",
    "IMPI",
    "IMPE",
    "ORI1",
    "ORI2",
    "ORE",
    "NEGI",
    "NEGE",
    "BOTE",
    "DN",
]


def feats(prompt: str, proof: str) -> dict:
    p = parse(prompt, proof)
    at = []
    for f in p.prem + [p.concl]:
        atoms_of(f, at)
    rules = collections.Counter(ln.rule for ln in p.lines)
    out = dict(
        n_prem=len(p.prem),
        n_atoms=len(at),
        lines=len(p.lines),
        fdepth=max(fdepth(f) for f in p.prem + [p.concl]),
        box_depth=max(ln.depth for ln in p.lines),
        n_ore=rules["ORE"],
        prompt_tok=len(prompt.split()),
        rules=dict(rules),
    )
    try:
        steps, toks, env = decompose(prompt, proof, canon=True)
        out["n_actions"] = len(steps)
        out["max_state"] = max(len(s) for s, a, h in steps)
        out["max_action"] = max(len(a) for s, a, h in steps)
        names = [int(t[1:]) for t in toks if t.startswith("n") and t[1:].isdigit()]
        out["max_name"] = max(names) if names else 0
    except Exception as e:  # noqa: BLE001
        out["decomp_err"] = str(e)[:80]
    return out


def q(xs, f):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(f * len(xs)))]


def summarise(rows: list[dict]) -> dict:
    n = len(rows)
    out = {"n": n}
    for k in ("n_prem", "n_atoms", "lines", "fdepth", "box_depth", "n_ore"):
        c = collections.Counter(r[k] for r in rows)
        out[k + "_hist"] = {str(x): c[x] for x in sorted(c)}
        out[k + "_mean"] = round(sum(r[k] for r in rows) / n, 3)
    for k in ("prompt_tok", "n_actions", "max_state", "max_action", "max_name"):
        xs = [r[k] for r in rows if k in r]
        if xs:
            out[k] = dict(
                median=q(xs, 0.5),
                p90=q(xs, 0.9),
                p99=q(xs, 0.99),
                max=max(xs),
                mean=round(sum(xs) / len(xs), 1),
            )
    out["rule_share_pct"] = {
        r: round(100 * sum(r in x["rules"] for x in rows) / n, 1) for r in RULES
    }
    out["rule_mean_uses"] = {
        r: round(sum(x["rules"].get(r, 0) for x in rows) / n, 3) for r in RULES
    }
    tot = sum(sum(x["rules"].values()) for x in rows)
    out["rule_line_share_pct"] = {
        r: round(100 * sum(x["rules"].get(r, 0) for x in rows) / tot, 2) for r in RULES
    }
    out["decomp_errors"] = sum("decomp_err" in r for r in rows)
    return out


def distinct(prompts: list[str]) -> dict:
    from code.thinking.data import class_keys

    ks, pks = set(), set()
    for p in prompts:
        k, pk = class_keys(p)
        ks.add(k)
        pks.add(pk)
    return dict(
        n=len(prompts),
        classes=len(ks),
        pkey_classes=len(pks),
        class_share_pct=round(100 * len(ks) / len(prompts), 2),
        pkey_share_pct=round(100 * len(pks) / len(prompts), 2),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True)
    ap.add_argument("--field", default="reference_proof")
    ap.add_argument("--k12", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    with open(a.set) as fi:
        rows = [json.loads(line) for line in fi]
    fs = [feats(r["prompt"], r[a.field]) for r in rows]
    res = {"set": summarise(fs), "set_distinct": distinct([r["prompt"] for r in rows])}
    if a.k12:
        rng = random.Random(a.seed)
        with open(K12) as fi:
            k12 = [json.loads(line) for line in fi]
        k12 = rng.sample(k12, min(a.k12, len(k12)))
        res["k12"] = summarise([feats(r["prompt"], r["proof"]) for r in k12])
        res["k12_sample"] = dict(n=len(k12), seed=a.seed, file=K12)
    with open(a.out, "w") as fo:
        json.dump(res, fo, indent=1)
    print(json.dumps(res["set_distinct"]))
    for k in ("n_prem_hist", "n_atoms_hist", "fdepth_hist", "box_depth_hist", "n_ore_hist"):
        print(k, "set", res["set"][k], "| k12", res.get("k12", {}).get(k))


if __name__ == "__main__":
    main()
