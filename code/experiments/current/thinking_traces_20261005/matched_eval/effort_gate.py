"""PLAN_CRITIQUE's pre-registered gate for model-decided effort, re-run on the current models.

    ND_OFFLINE=1 OMP_NUM_THREADS=8 nice code/.venv/bin/python -m \
        code.experiments.current.thinking_traces_20261005.matched_eval.effort_gate

PLAN_CRITIQUE § 3.3 deferred model-decided effort behind a CPU gate: re-run the § 2.3 headroom
analyses on the next thinking model's B0/B64/B256 rows, and build auto mode only if both hold on
13-20: (a) the split-half oracle beats the best fixed level by >= 3 pp pass@1 at no more tokens;
(b) the oracle under resampling beats the cascade by >= 3 pp at 1,000 tokens per problem. The
original `oracle_effort.py` / `oracle_budget.py` are not on disk; this re-implements § 7:

- Per problem, rows at B 0, 64 and 256, per-attempt `ok` and `tokens` (sampled: think + close +
  action, all draws; prefill excluded).
- Split-half oracle: per problem, the level with the highest pass rate on the even attempts (ties to
  the cheaper level), scored on the odd attempts. Full oracle: chosen and scored on all attempts.
- Cascade: e0, e64, then e256 repeated, each attempt drawn with replacement from that problem's
  stored attempts at that level; stop at the first success, or when the next attempt would exceed
  the budget (the first attempt is always allowed). 40 Monte-Carlo repeats per problem, seed 0.
  The i.i.d. arms use the same rule at one fixed level.
- Oracle under resampling: the split-half choice, drawing only from that level's odd attempts.

Model: Stage-1 S1g3 + EI r1 (rl_hard12) + EI r2 (hard712 a), seed 0, the round-2 seed-0 thinking
model of THINKING_UPLIFT.md (reads in `eval/matched/`, k 8 on 13-20, k 64 on textbook72).
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from code.experiments.current.thinking_traces_20261005.matched_eval.figs import valid_1320

M = Path("/scratch/cowifornia/nd-rl/think/runs/eval/matched")
LEVELS = (0, 64, 256)
BUDGETS = (1000, 2000, 4000)
REPEATS = 40
SETS = {
    "13–20 (986)": ("t1320_eiS1g3_h712_r2", True),
    "textbook72": ("tb72_eiS1g3_h712_r2", False),
}

Attempts = tuple[np.ndarray, np.ndarray]  # ok (0/1), sampled tokens
Problem = dict[int, Attempts]


def load(d: Path, filt: bool) -> list[Problem]:
    keep = valid_1320() if filt else None
    by: dict[str, Problem] = {}
    for b in LEVELS:
        with open(d / f"rows_plain_B{b}.jsonl") as f:
            for line in f:
                r = json.loads(line)
                if keep is not None and r["name"] not in keep:
                    continue
                by.setdefault(r["name"], {})[b] = (
                    np.asarray(r["ok"], dtype=float),
                    np.asarray(r["tokens"], dtype=float),
                )
    probs = [p for p in by.values() if len(p) == len(LEVELS)]
    assert len(probs) == len(by), "a problem is missing a level"
    return probs


def pick(p: Problem, half: slice) -> int:
    """The level with the highest pass rate on `half` of the attempts; ties to the cheaper level."""
    rates = [float(p[b][0][half].mean()) for b in LEVELS]
    return LEVELS[int(np.argmax(rates))]  # argmax takes the first maximum: LEVELS is cheap first


def fixed(probs: Sequence[Problem], b: int, half: slice = slice(None)) -> tuple[float, float]:
    """pass@1 and tokens per attempt at one level, on `half` of each problem's attempts."""
    ok = np.mean([p[b][0][half].mean() for p in probs])
    tok = np.sum([p[b][1][half].sum() for p in probs]) / np.sum([len(p[b][1][half]) for p in probs])
    return float(ok), float(tok)


def oracle(probs: Sequence[Problem], choose: slice, score: slice) -> tuple[float, float]:
    picks = [pick(p, choose) for p in probs]
    ok = np.mean([p[b][0][score].mean() for p, b in zip(probs, picks, strict=True)])
    tok = np.sum([p[b][1][score].sum() for p, b in zip(probs, picks, strict=True)]) / np.sum(
        [len(p[b][1][score]) for p, b in zip(probs, picks, strict=True)]
    )
    return float(ok), float(tok)


def run(pools: Sequence[Attempts], budget: float, rng: np.random.Generator) -> bool:
    """One budgeted run: the i-th attempt is drawn from pools[min(i, len - 1)]. Success if an
    attempt within the budget is accepted; the first attempt is always allowed."""
    spent, i = 0.0, 0
    while True:
        ok, tok = pools[min(i, len(pools) - 1)]
        j = int(rng.integers(len(ok)))
        if i > 0 and spent + tok[j] > budget:
            return False
        spent += tok[j]
        if ok[j]:
            return True
        i += 1


def within(probs: Sequence[Problem], arm: str, budget: float, seed: int = 0) -> float:
    """Success within `budget` sampled tokens per problem, %, averaged over REPEATS runs each."""
    rng = np.random.default_rng(seed)
    odd = slice(1, None, 2)
    tot = 0.0
    for p in probs:
        if arm == "cascade":
            pools = [p[b] for b in LEVELS]
        elif arm == "oracle":
            b = pick(p, slice(0, None, 2))
            pools = [(p[b][0][odd], p[b][1][odd])]
        else:
            pools = [p[int(arm)]]
        tot += sum(run(pools, budget, rng) for _ in range(REPEATS)) / REPEATS
    return 100 * tot / len(probs)


def report(name: str, probs: Sequence[Problem]) -> dict[str, object]:
    even, odd, every = slice(0, None, 2), slice(1, None, 2), slice(None)
    fix = {b: fixed(probs, b) for b in LEVELS}
    fix_odd = {b: fixed(probs, b, odd) for b in LEVELS}
    best = max(LEVELS, key=lambda b: fix[b][0])
    split = oracle(probs, even, odd)
    full = oracle(probs, every, every)
    print(f"\n## {name}: {len(probs)} problems, k {len(probs[0][0][0])}")
    print("\n| level | pass@1 % | tokens / attempt | pass@1 %, odd attempts |")
    print("|---|---|---|---|")
    for b in LEVELS:
        print(f"| B{b} | {100 * fix[b][0]:.1f} | {fix[b][1]:,.0f} | {100 * fix_odd[b][0]:.1f} |")
    print("\n| | pass@1 % | tokens / attempt |")
    print("|---|---|---|")
    print(
        f"| best fixed level (B{best}), all attempts | {100 * fix[best][0]:.1f} "
        f"| {fix[best][1]:,.0f} |"
    )
    print(
        f"| best fixed level (B{best}), odd attempts | {100 * fix_odd[best][0]:.1f} "
        f"| {fix_odd[best][1]:,.0f} |"
    )
    print(
        f"| split-half oracle (choose on even, score on odd) | {100 * split[0]:.1f} "
        f"| {split[1]:,.0f} |"
    )
    print(f"| full oracle | {100 * full[0]:.1f} | {full[1]:,.0f} |")
    arms = ["0", "64", "256", "cascade", "oracle"]
    res = {a: {x: within(probs, a, x) for x in BUDGETS} for a in arms}
    print("\nSuccess within the budget, % (40 Monte-Carlo runs per problem):\n")
    print("| tokens per problem | " + " | ".join(f"{x:,}" for x in BUDGETS) + " |")
    print("|---|" + "---|" * len(BUDGETS))
    labels = {
        "0": "i.i.d. B0",
        "64": "i.i.d. B64",
        "256": "i.i.d. B256",
        "cascade": "cascade e0 → e64 → e256",
        "oracle": "oracle under resampling",
    }
    for a in arms:
        print(f"| {labels[a]} | " + " | ".join(f"{res[a][x]:.1f}" for x in BUDGETS) + " |")
    best_iid = {x: max(res[a][x] for a in ("0", "64", "256")) for x in BUDGETS}
    print(
        "| cascade − best i.i.d. | "
        + " | ".join(f"{res['cascade'][x] - best_iid[x]:+.1f}" for x in BUDGETS)
        + " |"
    )
    print(
        "| oracle − cascade | "
        + " | ".join(f"{res['oracle'][x] - res['cascade'][x]:+.1f}" for x in BUDGETS)
        + " |"
    )
    gate_a = 100 * (split[0] - fix[best][0])
    gate_a_odd = 100 * (split[0] - fix_odd[best][0])
    gate_b = res["oracle"][1000] - res["cascade"][1000]
    print(
        f"\nGate (a): split-half oracle − best fixed = {gate_a:+.1f} pp ({gate_a_odd:+.1f} "
        f"against the odd-attempt score), at {split[1]:,.0f} vs {fix[best][1]:,.0f} tokens "
        "per attempt; "
        f"needs >= +3.0 at no more tokens."
    )
    print(
        f"Gate (b): oracle under resampling − cascade at 1,000 tokens = {gate_b:+.1f} pp; "
        "needs >= +3.0."
    )
    return {
        "problems": len(probs),
        "fixed": {f"B{b}": fix[b] for b in LEVELS},
        "fixed_odd": {f"B{b}": fix_odd[b] for b in LEVELS},
        "best_fixed": f"B{best}",
        "split_half_oracle": split,
        "full_oracle": full,
        "within_budget": res,
        "gate_a_pp": gate_a,
        "gate_a_odd_pp": gate_a_odd,
        "gate_b_pp": gate_b,
    }


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None, help="also write every value as JSON here")
    a = ap.parse_args(argv)
    out = {name: report(name, load(M / d, filt)) for name, (d, filt) in SETS.items()}
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
