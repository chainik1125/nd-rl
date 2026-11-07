"""Reference proofs, step 2: one σ route per theorem, frozen before any read (CPU only).

    python make_refs.py --minlen M.jsonl [--minlen M2.jsonl] --out refs.jsonl --table refs_table.json

For each of the 106: Dan's minlen proof if it builds as a reference (it replays in his env, the
verifier passes it, `coe_sigma.flags` flags it σ on the written proof, Lean accepts the env's text).
Otherwise the fallback: the shortest σ-flagged proof among `KD/collected/**/reads/*.proofs.jsonl`
(ties: lexicographic) that builds as a reference. Otherwise the theorem drops out of the forced
stages. The source is recorded per theorem. Proof text stays in refs.jsonl under /scratch; the table
holds counts only.
"""

# ruff: noqa: E501, SIM115, B905, B007, SIM300
from __future__ import annotations

import argparse
import collections
import glob
import hashlib
import json

from factor_lib import DAN, KD, build_ref, ref_to_json


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--minlen", action="append", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--table", required=True)
    a = ap.parse_args()

    from lean_gate import check_sources  # type: ignore[import-not-found]
    from state_env import Env  # type: ignore[import-not-found]
    from tokenizer import make_tokenizer  # type: ignore[import-not-found]

    tok = make_tokenizer("lean_staten")
    strict = (DAN / "data/coe/sigma_heldout_strict.txt").read_text().split()
    by = {}
    for line in open(DAN / "data/coe/transfer_track832.jsonl"):
        t = json.loads(line)
        by[t["name"]] = t
    ml: dict[str, dict] = {}
    for fn in a.minlen:  # a later file only fills theorems an earlier one left without a proof
        for line in open(fn):
            r = json.loads(line)
            if r.get("proof") and r["name"] not in ml:
                ml[r["name"]] = r

    # the fallback pool
    pool: dict[str, set[str]] = collections.defaultdict(set)
    files = sorted(glob.glob(str(KD / "collected/**/reads/*.proofs.jsonl"), recursive=True))
    for fn in files:
        for line in open(fn):
            r = json.loads(line)
            if r["name"] in by:
                pool[r["name"]].update(r["proofs"])

    def lean_ok(ref) -> bool:
        e = Env(ref.prompt, canon=True, base=0, assign=True)
        for x in ref.actions:
            e.apply(x)
        src = f"{tok.statement(ref.prompt)} {tok.text(e.text)}"
        return check_sources([src])[0][0]

    refs, why_not = [], {}
    for n in strict:
        t = by[n]
        reasons = []
        ref = None
        if n in ml:
            b = build_ref(n, t["prompt"], t.get("schema"), "minlen", ml[n]["proof"])
            if isinstance(b, str):
                reasons.append(f"minlen: {b}")
            elif not lean_ok(b):
                reasons.append("minlen: Lean rejected")
            else:
                ref = b
        else:
            reasons.append("minlen: no proof")
        if ref is None:
            for p in sorted(pool.get(n, ()), key=lambda p: (p.count(";"), p)):
                b = build_ref(n, t["prompt"], t.get("schema"), "collected_read", p)
                if not isinstance(b, str) and lean_ok(b):
                    ref = b
                    break
            if ref is None:
                reasons.append(f"fallback: none of {len(pool.get(n, ()))} proofs built")
        if ref is None:
            why_not[n] = reasons
        else:
            ref.extra["why_fallback"] = reasons
            refs.append(ref)

    with open(a.out, "w") as f:
        for r in refs:
            f.write(json.dumps(ref_to_json(r), ensure_ascii=False) + "\n")

    # counts only, for the prereg and the repo
    sch = collections.Counter(by[n].get("schema") for n in strict)
    tab = {
        "sha256_refs": hashlib.sha256(open(a.out, "rb").read()).hexdigest(),
        "n_theorems": len(strict),
        "n_refs": len(refs),
        "dropped": why_not,
        "by_schema": {},
        "source": dict(collections.Counter(r.source for r in refs)),
        "fallback_why": {r.name: r.extra["why_fallback"] for r in refs if r.source != "minlen"},
        "n_multi_sigma_dn": sum(r.n_sigma_dn > 1 for r in refs),
        "n_multi_dn": sum(r.n_dn > 1 for r in refs),
        "n_between_negi_dn": sum(r.n_between > 0 for r in refs),
        "n_chain_not_strict": sum(not r.extra["flags"]["sigma_strict"] for r in refs),
        "per_theorem": {},
    }
    for s in sch:
        rs = [r for r in refs if r.schema == s]
        tab["by_schema"][str(s)] = {
            "n": sch[s],
            "n_ref": len(rs),
            "t_is_goal": sum(r.t_is_goal for r in rs),
            "minlen": sum(r.source == "minlen" for r in rs),
            "mean_actions": sum(len(r.actions) for r in rs) / max(1, len(rs)),
            "group_actions": {g: sum(r.groups.count(g) for r in rs) / max(1, len(rs))
                              for g in ("setup", "pick", "reach", "negi", "dn", "finish")},
            "group_empty": {g: sum(r.groups.count(g) == 0 for r in rs)
                            for g in ("setup", "pick", "reach", "negi", "dn", "finish")},
            "setup_all_premises": sum(r.pick == r.n_pr for r in rs),
        }  # fmt: skip
    for r in refs:
        tab["per_theorem"][r.name] = {
            "schema": r.schema, "source": r.source, "t_is_goal": r.t_is_goal, "n_actions": len(r.actions),
            "n_lines": r.extra["n_lines"], "L_true": by[r.name].get("n_lines"), "n_pr": r.n_pr,
            "n_dn": r.n_dn, "n_sigma_dn": r.n_sigma_dn, "n_between": r.n_between,
            "groups": {g: r.groups.count(g) for g in ("setup", "pick", "reach", "negi", "dn", "finish")},
        }  # fmt: skip
    json.dump(tab, open(a.table, "w"), indent=1)
    print({k: v for k, v in tab.items() if k not in ("per_theorem", "by_schema")})
    for s, v in tab["by_schema"].items():
        print(s, v)


if __name__ == "__main__":
    main()
