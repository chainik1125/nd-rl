"""σ share of the Lean-accepted training proofs per GRPO iteration (Amendment 2), from the dump.

    python -m code.experiments.current.knockout_grpo_20261103.train_sigma TAG [TAG ...]

`line.py --lean_dump` sets LEAN_GATE_DUMP for each GRPO stage: lean_gate appends every distinct
checked (prompt, text, ND, verdict) of a gate call, and the stage log prints one `[lean_gate] ...
distinct D: ...` line per call before the iteration's `it N:` line. Walking the log in order
assigns each call's D dump rows to its iteration. Per iteration: distinct accepted proofs, the
share with σ (Dan's DN of a NEGI line), with σ⁺ only, and with any DN. Output
OUT/train_sigma/<TAG>.json.
"""

from __future__ import annotations

import json
import re
import sys

from .common import OUT, label

GATE = re.compile(r"^\[lean_gate\] \d+ samples.*distinct (\d+):")
IT = re.compile(r"^it (\d+):")


def stage(name: str) -> list[dict]:
    logs = [OUT / "logs" / f"{name}.log", *sorted((OUT / "logs").glob(f"*/{name}.log"))]
    log = next((p for p in logs if p.exists()), None)
    dump = OUT / "runs" / f"{name}_lean_dump.jsonl"
    if log is None or not dump.exists():
        return []
    rows = iter(open(dump))  # noqa: SIM115
    out, cur = [], []
    for line in open(log):  # noqa: SIM115
        m = GATE.match(line)
        if m:
            for _ in range(int(m.group(1))):
                cur.append(json.loads(next(rows)))
            continue
        m = IT.match(line)
        if m:
            acc = [r for r in cur if r.get("lean_ok")]
            labs = [label(r["nd"]) for r in acc if r.get("nd")]
            n = len(labs)
            out.append({
                "iter": int(m.group(1)), "accepted_distinct": n,
                "sigma": sum(x.get("sigma", False) for x in labs),
                "sigma_plus_only": sum(x.get("sigma_plus", False) and not x.get("sigma", False) for x in labs),  # noqa: E501
                "dn_any": sum(x.get("dn_any", False) for x in labs),
            })  # fmt: skip
            cur = []
    return out


def main() -> None:
    d = OUT / "train_sigma"
    d.mkdir(exist_ok=True)
    for tag in sys.argv[1:]:
        res = {"L2": stage(f"{tag}_L2"), "L3": stage(f"{tag}_L3")}
        (d / f"{tag}.json").write_text(json.dumps(res, indent=0))
        for st, rows in res.items():
            for r in rows[:: max(1, len(rows) // 8)]:
                n = max(1, r["accepted_distinct"])
                print(f"{tag} {st} it {r['iter']}: accepted {r['accepted_distinct']}, "
                      f"sigma {100 * r['sigma'] / n:.1f} %, dn_any {100 * r['dn_any'] / n:.1f} %")  # fmt: skip  # noqa: E501


if __name__ == "__main__":
    main()
