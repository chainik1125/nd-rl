"""Class-equivalent hard712 work/gated.jsonl and gated_lean.jsonl from the NFS backup.

(ynez rebuild, 2026-10-13)

    python rebuild_gated.py BACKUP_DIR OUT_WORK_DIR

The production gated rows are lost; the backup holds every one of the 29,542 gated classes: the
23,092 selected rows (pool, test_1k, dev_500, in the final row format) and the 6,450 spare rows
(in the gated_lean format). Selected rows are mapped back to the gated fields that
curriculum/reuse.py, assemble.py, run.py select / split and the class-key indexes read (`proof` =
`gen_proof`); their timing fields (t_search, t_total) and shape-gate info (n_conn, n_atoms,
fdepth) are not recoverable, so the files are not byte-identical to the originals. Rows are written
in seed order, the order generate wrote them (the spare rows' seeds are already sorted).
"""

import json
import sys

B, W = sys.argv[1], sys.argv[2]
KEEP = (
    "prompt",
    "recipe",
    "seed",
    "subst",
    "discharged",
    "n_prem",
    "gen_lines",
    "n_irrelevant",
    "key",
    "pkey",
    "min_lines",
    "reference_proof",
    "structure",
    "necessity",
    "families",
)


def jl(p):
    with open(p) as f:
        return [json.loads(x) for x in f if x.strip()]


rows = []
for part in ("pool", "test_1k", "dev_500"):
    for r in jl(f"{B}/{part}.jsonl"):
        g = {k: r[k] for k in KEEP if k in r}
        g["proof"] = r["gen_proof"]
        g["status"] = "ok"
        g["lean_ok"], g["lean_ref_ok"] = r["lean_ok"], r["lean_ref_ok"]
        rows.append(g)
rows += jl(f"{B}/work/spare.jsonl")
rows.sort(key=lambda r: r["seed"])
assert len(rows) == 29542 and len({r["pkey"] for r in rows}) == 29542
with open(f"{W}/gated_lean.jsonl", "w") as fl, open(f"{W}/gated.jsonl", "w") as fg:
    for r in rows:
        fl.write(json.dumps(r) + "\n")
        fg.write(
            json.dumps({k: v for k, v in r.items() if k not in ("lean_ok", "lean_ref_ok")}) + "\n"
        )
print(
    "rows",
    len(rows),
    "lean_ok",
    sum(r["lean_ok"] for r in rows),
    "lean_ref_ok",
    sum(r["lean_ref_ok"] for r in rows),
)
