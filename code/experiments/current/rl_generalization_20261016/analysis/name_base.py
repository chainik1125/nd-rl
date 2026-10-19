"""Does the random name base matter? (O7)

    python3 name_base.py > out/name_base.txt

The sampler (`code/thinking/sample.py`, `think_generate`) gives attempt i of a call the name base
`random.Random(seed * 1000003 + i).randint(0, NAME_BASE_MAX)`, NAME_BASE_MAX = 32 (`state_sample.py`), so canonical
names start at n<base + 1>. `eval.py` builds its prompts as `[r["prompt"] for r in recs for _ in range(k)]` and calls
`think_generate` once per (arm, budget) with `--seed`, so attempt a of row j has i = j * k + a.

1. Check: for each row, the reconstructed bases of the solved attempts, + 1, must equal the multiset of the first
   name (`have nX`) of the row's accepted texts.
2. Per base tercile (0-10, 11-21, 22-32): pass@1, pass@1 minus the problem's own mean (a within-problem contrast),
   and the shares of attempts ending at `names exhausted`, `unbound` and `binder mismatch`.

Standard library only; CPU, one process.
"""

from __future__ import annotations

import collections
import json
import os
import random
import re

EV = "/scratch/cowifornia/nd-rl/think/runs/eval/v2"
NAME_BASE_MAX = 32
READS = [
    "t2130_grpo_think_vs_mb_kkl0_s0_plain_k8",
    "t2130_grpo_think_vs_mb_s0_plain_k8",
    "t2130_ei_think_v_s0_plain_k8",
    "t1320_grpo_think_vs_mb_kkl0_s0_plain_k16",
    "t1320_ei_think_v_s0_plain_k16",
    "charles173_grpo_think_vs_mb_kkl0_s0_plain_k64",
]


def eval_seed(d: str) -> int:
    with open(f"{EV}/{d}/summary.json") as f:
        return int(json.load(f)["config"]["seed"])


def base(seed: int, i: int) -> int:
    return random.Random(seed * 1000003 + i).randint(0, NAME_BASE_MAX)


def tercile(b: int) -> str:
    return "0-10" if b <= 10 else ("11-21" if b <= 21 else "22-32")


def main() -> int:
    for d in READS:
        path = f"{EV}/{d}/rows_plain_B64.jsonl"
        if not os.path.exists(path):
            print(f"{d}: missing")
            continue
        seed = eval_seed(d)
        rows = [json.loads(line) for line in open(path)]
        agree = 0
        ok: dict[str, list[int]] = collections.defaultdict(list)
        cen: dict[str, list[float]] = collections.defaultdict(list)
        cause: dict[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)
        for j, r in enumerate(rows):
            k = r["k"]
            bs = [base(seed, j * k + a) for a in range(k)]
            exp = collections.Counter(b + 1 for b, o in zip(bs, r["ok"]) if o)
            got: collections.Counter[int] = collections.Counter()
            for text, c in r["accepted"].items():
                m = re.match(r"have n(\d+)", text)
                if m:
                    got[int(m.group(1))] += c
            agree += exp == got
            mean = sum(r["ok"]) / k
            for a in range(k):
                t = tercile(bs[a])
                ok[t].append(r["ok"][a])
                cen[t].append(r["ok"][a] - mean)
                if not r["ok"][a]:
                    cause[t][r["status"][a]] += 1
        print(f"{d}: reconstructed bases match the accepted texts in {agree} / {len(rows)} rows (seed {seed})")
        for t in ("0-10", "11-21", "22-32"):
            n = len(ok[t])
            c = cause[t]
            print(
                f"   base {t:5s}: pass@1 {100 * sum(ok[t]) / n:5.1f} (within-problem {100 * sum(cen[t]) / n:+.1f})"
                f"  names exhausted {100 * c['fail:S:names exhausted'] / n:4.1f}"
                f"  unbound {100 * c['fail:S:unbound'] / n:4.1f}"
                f"  binder mismatch {100 * c['fail:S:binder mismatch'] / n:4.1f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
