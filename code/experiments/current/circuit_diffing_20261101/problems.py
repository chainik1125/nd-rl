"""Stage 2 problem categories from existing reads (seed 0: base S1g3_s0 vs the L3 end).

    python -m code.experiments.current.circuit_diffing_20261101.problems

  G (RL-gained)  base rate < 1% (Charles149, 1,024 attempts) or 0 of 12 (13-20, k8 + k4 reads),
                 and RL rate >= 10% (Charles149, 1,024) or >= 8 of 16 (13-20, k16 read)
  S (shared)     base rate >= 30% and RL rate >= 30% (Charles149); base >= 6 of 12 and RL >= 8
                 of 16 (13-20)
13-20 (the 986 valid items) contributes a fixed random sample of at most 60 G and 60 S
problems (seed 20261101). Neither set
was used to train a crosscoder; textbook72 is not used. Writes `stage2_problems.json`.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict

from .common import CHARLES, EVAL, HERE, RUNS


def read(stem: str, keep: set[str] | None = None) -> dict[str, tuple[int, int, str]]:
    out: dict[str, tuple[int, int, str]] = {}
    for line in (EVAL / stem / "rows_plain_B64.jsonl").read_text().splitlines():
        r = json.loads(line)
        if keep is None or r["name"] in keep:
            out[r["name"]] = (r["n_ok"], r["k"], r["prompt"])
    return out


def valid_1320() -> set[str]:
    """The 986 valid 13-20-line items (high_rl_passk `valid_1320`, the matched_eval rule)."""
    fin = {}
    for line in (RUNS / "pools/test_13_20/test_13_20.jsonl").read_text().splitlines():
        r = json.loads(line)
        fin[r["name"]] = r
    ok = set()
    for line in (RUNS / "eval/matched/sets/t1320_n1000.jsonl").read_text().splitlines():
        r = json.loads(line)
        g = fin.get(r["name"])
        if (
            g
            and g["prompt"] == r["prompt"]
            and g.get("reference_lines") == r.get("reference_lines")
        ):
            ok.add(r["name"])
    return ok


def merge(*reads: dict[str, tuple[int, int, str]]) -> dict[str, tuple[int, int, str]]:
    acc: dict[str, list] = defaultdict(lambda: [0, 0, ""])
    for rd in reads:
        for n, (c, k, p) in rd.items():
            acc[n][0] += c
            acc[n][1] += k
            acc[n][2] = p
    return {n: (c, k, p) for n, (c, k, p) in acc.items()}


def main() -> int:
    names = {json.loads(line)["name"] for line in CHARLES.read_text().splitlines()}
    b = read("charles149_s1g3_s0_plain_k1024", names)
    r = read("charles149_grpo_think_vs_mb_kkl0_s0_plain_k1024", names)
    out: dict[str, list[dict]] = {"G": [], "S": []}
    for n in sorted(names):
        pb, pr = b[n][0] / b[n][1], r[n][0] / r[n][1]
        rec = dict(name=n, set="charles149", prompt=b[n][2], base_rate=pb, rl_rate=pr)
        if pb < 0.01 and pr >= 0.10:
            out["G"].append(rec)
        elif pb >= 0.30 and pr >= 0.30:
            out["S"].append(rec)
    valid = valid_1320()
    tb = merge(read("t1320_s1g3_s0_plain_k8", valid), read("t1320_s1g3_s0_plain_k4", valid))
    tr = read("t1320_grpo_think_vs_mb_kkl0_s0_plain_k16", valid)
    g, s = [], []
    for n in sorted(set(tb) & set(tr)):
        cb, kb, p = tb[n]
        cr, kr, _ = tr[n]
        rec = dict(name=n, set="t1320", prompt=p, base_rate=cb / kb, rl_rate=cr / kr)
        if cb == 0 and cr >= 8:
            g.append(rec)
        elif cb >= 6 and cr >= 8:
            s.append(rec)
    rng = random.Random(20261101)
    out["G"] += rng.sample(g, min(60, len(g)))
    out["S"] += rng.sample(s, min(60, len(s)))
    info = {
        k: {st: sum(x["set"] == st for x in v) for st in ("charles149", "t1320")}
        for k, v in out.items()
    }
    info["t1320_pool"] = {"G": len(g), "S": len(s)}
    out["info"] = info  # type: ignore[assignment]
    (HERE / "stage2_problems.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(info))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
