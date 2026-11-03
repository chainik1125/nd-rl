"""Refill candidates for the S1g3 knockout: Dan's eligible pool (coe_assemble.consider) + our sets.

    python -m code.experiments.current.knockout_grpo_20261103.refill_pool

Dan's K12- refills each removed record with a record of the same pruned length drawn from K12's own
unused sources: generator lengths <= 6 from `train_depth3_f0_a1.jsonl` (a1), >= 7 from
`pool_long_kh.jsonl`. A candidate is eligible (Dan's `consider`, coe_assemble.py, reproduced call by
call with his helpers) when it is not the renaming class of any K12 record (canon_key or the
premise-order-invariant canon2), not in any of his evaluation pools (`eval_pool_files`), for the
long pool not an a1 class, parses, has (a1) the pruned length it is filed under, box depth <= 2 in
pruned and written form, is sigma-plus-free and lemma-route-free in its generator proof, and its
canon2 class is not already taken by an earlier candidate.

Added here (our pipeline's sets): not a class (ndbench canon_key or pkey) of S1g3's exclusion sets
(data.py DEFAULT_EXCLUDE + textbook72 train) and not a class of create-or-elicit's tracked transfer
theorems (transfer_track832, which holds the 106 strict members), read1054 or fam_extra.

Generator lengths taken: those our knockout removes (3-12; a1 for 3-6, the long pool for 7-12).
Output OUT/refill/candidates.jsonl: {id, prompt, proof, n_lines, src} with id `ko_<src>_<line>`;
OUT/refill/candidates_report.json.
"""

from __future__ import annotations

import collections
import json
import os
import time

from .common import A1, COE, DAN_DATA, FORK, K12, OUT, dan, jsonl, sha256

GEN_LENGTHS = range(3, 13)


def main() -> None:
    t0 = time.time()
    from code.ndbench.core.keys import canon_key as nd_key
    from code.ndbench.core.keys import pkey
    from code.thinking.data import DEFAULT_EXCLUDE, EvalIndex
    from code.thinking.progtrace import TB_TRAIN

    D = dan()
    ca = D.coe_assemble
    canon_key = D.gen.canon_key
    here = os.getcwd()
    os.chdir(FORK)  # eval_pool_files / exclusion_keys resolve under the fork root
    files = ca.eval_pool_files()
    ex1, ex2, ex_per, ex_missing = ca.exclusion_keys(files)
    os.chdir(here)
    print(
        f"Dan's exclusion: {len(files)} files ({len(ex_missing)} missing), {len(ex1)} / {len(ex2)}"
    )

    ours = [
        *DEFAULT_EXCLUDE,
        TB_TRAIN,
        COE / "transfer_track832.jsonl",
        COE / "read1054.jsonl",
        COE / "fam_extra.jsonl",
    ]
    idx = EvalIndex([str(p) for p in ours])

    k12_k1, k12_k2 = set(), set()
    for r in jsonl(K12):
        t = ca.thm_of(r)
        k12_k1.add(r.get("key") or canon_key(t))
        k12_k2.add(ca.canon2(t))

    scan: dict = collections.defaultdict(collections.Counter)
    cand_k2: set = set()
    a1_keys: set = set()
    out = []

    def consider(r: dict, src: str, L: int, line: int) -> None:
        t = ca.thm_of(r)
        k1 = r.get("key") or canon_key(t)
        sc = scan[src]
        sc["in_range"] += 1
        if k1 in k12_k1 or ca.canon2(t) in k12_k2:
            sc["k12_class"] += 1
            return
        if k1 in ex1 or ca.canon2(t) in ex2:
            sc["eval_class"] += 1
            return
        if src == "pool" and k1 in a1_keys:
            sc["a1_class"] += 1
            return
        lines = D.patterns.parse(r["proof"])
        if lines is None:
            sc["parse_fail"] += 1
            return
        pl = D.patterns.prune_lines(lines)
        if len(pl) != L and src == "a1":
            sc["pruned_len_mismatch"] += 1
            return
        if max((x["depth"] for x in pl), default=0) >= 3 or ca.written_depth(r["proof"]) >= 3:
            sc["depth3"] += 1
            return
        drop, _ = ca.knock(r["proof"])
        if drop:
            sc["sigma_plus"] += 1
            return
        if ca.lemma_route(lines):
            sc["lemma_route"] += 1
            return
        k2 = ca.canon2(t)
        if k2 in cand_k2:
            sc["dup_canon2_in_pools"] += 1
            return
        if idx.hit(r["prompt"]):  # ours: S1g3's exclusion sets and the transfer-tracked theorems
            sc["our_eval_class"] += 1
            return
        cand_k2.add(k2)
        sc["eligible"] += 1
        out.append(
            {
                "id": f"ko_{src}_{line}",
                "prompt": r["prompt"],
                "proof": r["proof"],
                "n_lines": L,
                "src": src,
            }
        )

    a1_rows = []
    for i, r in enumerate(jsonl(A1)):
        a1_keys.add(r.get("key") or canon_key(ca.thm_of(r)))
        if r["n_lines"] in GEN_LENGTHS and r["n_lines"] <= 6:
            a1_rows.append((i, r))
    for i, r in a1_rows:
        consider(r, "a1", r["n_lines"], i)
    for i, r in enumerate(jsonl(DAN_DATA / "pool_long_kh.jsonl")):
        if r["n_lines"] >= 7 and r["n_lines"] in GEN_LENGTHS:
            consider(r, "pool", r["n_lines"], i)
    d = OUT / "refill"
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "candidates.jsonl", "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    rep = {
        "scan": {k: dict(v) for k, v in scan.items()},
        "eligible_per_gen_length": dict(
            sorted(collections.Counter(r["n_lines"] for r in out).items())
        ),
        "dan_exclusion_files": len(files),
        "dan_exclusion_missing": ex_missing,
        "our_exclusion_files": [str(p) for p in ours],
        "inputs_sha256": {
            "a1": sha256(A1),
            "pool_long_kh": sha256(DAN_DATA / "pool_long_kh.jsonl"),
            "k12": sha256(K12),
        },
        "nd_key_check": nd_key is not None and pkey is not None,
        "secs": time.time() - t0,
    }
    (d / "candidates_report.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
