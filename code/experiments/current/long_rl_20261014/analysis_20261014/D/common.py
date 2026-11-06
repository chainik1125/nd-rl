"""Shared loaders + gold-proof feature extraction for agent D (minimal pairs / constructed probes)."""
import collections
import json
import os
import re
import sys

ND_FORK = "/private/tmp/claude-501/-Users-robbie-repos-nd-rl/a1d02fb4-ba43-48c4-838b-e227293b6c2a/scratchpad/fork51604b3"
sys.path.insert(0, ND_FORK)
import importlib
V = importlib.import_module("nd_verify.verify")
V.ATOM_NAMES |= set("PQRSTUWXYZ")
parse_proof_tokens = V.parse_proof_tokens

E = "/Users/robbie/repos/nd-rl/code/experiments/current/long_rl_20261014"
A = "/Users/robbie/repos/nd-rl/artifacts/long_rl_20261014/v2"
ARMS = ["untied-ei_s0", "untied-ei_s1", "looped-ei_s0", "looped-ei_s1"]
D = os.path.dirname(os.path.abspath(__file__))


def jl(fn):
    with open(fn) as f:
        return [json.loads(l) for l in f]


def load_rl():
    return jl(f"{E}/data/rl.jsonl")


def load_dev():
    return jl(f"{E}/data/dev.jsonl")


def solved_sets():
    """arm -> {name: (min round, best pruned length)} over found_targets."""
    out = {}
    for arm in ARMS:
        d = {}
        for r in jl(f"{A}/long-rl-20261014-v2_{arm}/found_targets.jsonl"):
            rd, pr = r["round"], r["pruned"]
            if r["name"] not in d:
                d[r["name"]] = [rd, pr]
            else:
                d[r["name"]][0] = min(d[r["name"]][0], rd)
                d[r["name"]][1] = min(d[r["name"]][1], pr)
        out[arm] = d
    return out


def dev_solved():
    out = {}
    for arm in ARMS:
        out[arm] = {r["name"]: r["n_ok"] for r in jl(f"{A}/long-rl-20261014-v2_{arm}/eval_dev.jsonl")}
    return out


def src_family(src):
    return src.split(":")[0].replace("_f20.jsonl", "").replace(".jsonl", "")


def features(rec):
    """Structural features of the gold ND proof."""
    lines = parse_proof_tokens(rec["proof"].split())
    body = [ln for ln in lines if ln["rule"] != "PR"]
    prem = [ln for ln in lines if ln["rule"] == "PR"]
    rules = collections.Counter(ln["rule"] for ln in body)
    cited = collections.Counter(r for ln in lines for r in ln["refs"])
    prem_idx = {ln["idx"] for ln in prem}
    used_prem = {i for i in cited if i in prem_idx}
    reuse = sum(1 for ln in body if ln["rule"] not in ("AS",) and cited[ln["idx"]] > 1)
    # box structure
    max_depth = max((ln["depth"] for ln in body), default=0)
    n_box = rules["AS"]
    # assumption formula kinds
    as_neg = sum(1 for ln in body if ln["rule"] == "AS" and ln["formula"][0] == "not")
    # length of longest run of IMPE
    skel = " ".join(ln["rule"] for ln in body)
    return dict(
        body=len(body),
        n_prem=len(prem),
        distractors=len(prem) - len(used_prem),
        max_depth=max_depth,
        n_box=n_box,
        ore=rules["ORE"],
        negi=rules["NEGI"],
        nege=rules["NEGE"],
        impi=rules["IMPI"],
        ori=rules["ORI1"] + rules["ORI2"],
        andi=rules["ANDI"],
        ande=rules["ANDE1"] + rules["ANDE2"],
        bote=rules["BOTE"],
        dn=rules["DN"],
        impe=rules["IMPE"],
        reuse=reuse,
        as_neg=as_neg,
        skel=skel,
        goal_is_atom=rec["key"].split("|-")[1].strip().startswith("a") and "(" not in rec["key"].split("|-")[1],
    )


def fmt_formula(f):
    if isinstance(f, str):
        return f
    if f[0] == "atom":
        return f[1]
    if f == ("bot",):
        return "F"
    if f[0] == "not":
        return f"( ~ {fmt_formula(f[1])} )"
    op = {"and": "&", "or": "v", "imp": ">", "iff": "<>"}.get(f[0], f[0])
    return f"( {fmt_formula(f[1])} {op} {fmt_formula(f[2])} )"
