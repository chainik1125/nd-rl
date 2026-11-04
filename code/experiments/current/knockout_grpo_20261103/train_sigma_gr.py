"""σ share of the Lean-accepted training proofs for a continuation whose Lean dump has a gap.

    python -m code.experiments.current.knockout_grpo_20261103.train_sigma_gr TAG HEAD_ROWS

kos_s0_gr and kos_s0_dp_gr (Amendment 4) were migrated from temescal; on serrano a stale copy
then overwrote their dump several times (log.md, 2026-11-04 00:36Z), so the dump holds the
temescal rows (the first HEAD_ROWS, intact) and the rows appended after the last overwrite (the
tail), with the rows in between lost. The stage log is complete. Walking the log's gate calls in
order assigns the head rows from the start and the tail rows from the end; an iteration counts
only if all its gate rows are in one segment, and the iterations in between are reported as a
gap. Output OUT/train_sigma/<TAG>.json, as train_sigma.py writes, plus `segments` and `gaps`.
"""

from __future__ import annotations

import json
import re
import sys

from .common import OUT, label

GATE = re.compile(r"^\[lean_gate\] \d+ samples.*distinct (\d+):")
IT = re.compile(r"^it (\d+):")


def groups(log: str) -> list[tuple[int | None, int]]:
    """(iteration or None for a cut-off partial one, rows) per iteration's gate calls, in order."""
    out, cur = [], 0
    for line in open(log):  # noqa: SIM115
        m = GATE.match(line)
        if m:
            cur += int(m.group(1))
            continue
        m = IT.match(line)
        if m:
            out.append((int(m.group(1)), cur))
            cur = 0
        elif line.startswith("lr schedule") and cur:  # a restart: the partial iteration's rows
            out.append((None, cur))
            cur = 0
    if cur:
        out.append((None, cur))  # the iteration in progress
    return out


def summarise(rows: list[dict]) -> dict:
    labs = [label(r["nd"]) for r in rows if r.get("lean_ok") and r.get("nd")]
    return {"accepted_distinct": len(labs), "sigma": sum(x.get("sigma", False) for x in labs),
            "dn_any": sum(x.get("dn_any", False) for x in labs)}  # fmt: skip


def main() -> None:
    tag, head = sys.argv[1], int(sys.argv[2])
    name = f"{tag}_L3"
    dump = [json.loads(x) for x in open(OUT / "runs" / f"{name}_lean_dump.jsonl")]  # noqa: SIM115
    g = groups(str(OUT / "logs" / f"{name}.log"))
    tail = len(dump) - head
    res, segs, i = [], [], 0
    pos = 0  # head: from the start
    while i < len(g) and pos + g[i][1] <= head:
        it, n = g[i]
        if it is not None:
            res.append({"iter": it, **summarise(dump[pos : pos + n])})
        pos += n
        i += 1
    head_end = res[-1]["iter"] if res else None
    j, back = len(g) - 1, 0  # tail: from the end
    tail_rows = []
    while j >= i and back + g[j][1] <= tail:
        it, n = g[j]
        lo = len(dump) - back - n
        if it is not None:
            tail_rows.append({"iter": it, **summarise(dump[lo : lo + n])})
        back += n
        j -= 1
    tail_rows.reverse()
    res += tail_rows
    gap = [it for it, _ in g[i : j + 1] if it is not None]
    segs = [[res[0]["iter"], head_end]] if head_end is not None else []
    if tail_rows:
        segs.append([tail_rows[0]["iter"], tail_rows[-1]["iter"]])
    out = {"L2": [], "L3": res, "segments": segs, "gaps": [[gap[0], gap[-1]]] if gap else [],
           "head_rows": head, "tail_rows": tail, "tail_aligned": back == tail}  # fmt: skip
    (OUT / "train_sigma" / f"{tag}.json").write_text(json.dumps(out, indent=0))
    for a, b in segs:
        rr = [r for r in res if a <= r["iter"] <= b]
        n = sum(r["accepted_distinct"] for r in rr)
        print(f"{tag}: L3 it {a}-{b}: σ {100 * sum(r['sigma'] for r in rr) / max(1, n):.1f} % of "
              f"{n} accepted")  # fmt: skip
    print(f"{tag}: gap {out['gaps']}; tail aligned {out['tail_aligned']}")


if __name__ == "__main__":
    main()
