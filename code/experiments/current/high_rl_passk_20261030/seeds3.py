"""Charles149 at k1024: base S1g3 vs the recipe's L3 end, seeds 0-2.

Each RL seed starts from its own base seed, so seeds are resampled in pairs. Run from the repo root:
    ND_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 code/.venv/bin/python -m \\
        code.experiments.current.high_rl_passk_20261030.seeds3
Prints the table and writes seeds3.json next to this file.
"""

import json
from pathlib import Path

import numpy as np

from code.thinking import stats

from .passk_figs import KEEP, _ok_lists

V2 = Path("/scratch/cowifornia/nd-rl/think/runs/eval/v2")
KS = [1, 5, 64, 1024]
N = 1024
keep = KEEP["charles149"]
assert keep is not None


def p(stem: str, s: int) -> Path:
    return V2 / f"charles149_{stem}_s{s}_plain_k1024" / "rows_plain_B64.jsonl"


def arm(stem: str, seeds: list[int]) -> stats.Arm:
    a = stats.load_arm(stem, [p(stem, s) for s in seeds])
    a = a.subset(sorted(keep & set(a.names)), stem)
    assert len(a.names) == len(keep) == 149 and a.kmax >= N, (stem, len(a.names), a.kmax)
    return a


out: dict[str, object] = {}
for seeds in ([0], [1], [2], [0, 1, 2]):
    b, r = arm("s1g3", seeds), arm("grpo_think_vs_mb_kkl0", seeds)
    row = {}
    for k in KS:
        e = stats.paired_diff(r, b, k, paired_seeds=True, draws=4000)
        row[k] = {
            "base": 100 * b.values(k).mean(),
            "rl": 100 * r.values(k).mean(),
            "gap": 100 * e.est,
            "lo": 100 * e.lo,
            "hi": 100 * e.hi,
        }
    out["+".join(map(str, seeds))] = row
    print("seeds", seeds)
    for k, v in row.items():
        print(
            f"  k{k}: base {v['base']:.1f} RL {v['rl']:.1f} "
            f"gap {v['gap']:+.1f} [{v['lo']:+.1f}, {v['hi']:+.1f}]"
        )

# RL-only / base-only per seed, then pooled over seeds (3 x 1024 attempts on each side)
B = {s: _ok_lists(p("s1g3", s), keep) for s in range(3)}
R = {s: _ok_lists(p("grpo_think_vs_mb_kkl0", s), keep) for s in range(3)}
for s in range(3):
    ro = [x for x in B[s] if sum(B[s][x][:N]) == 0 and sum(R[s][x][:N]) > 0]
    bo = [x for x in B[s] if sum(B[s][x][:N]) > 0 and sum(R[s][x][:N]) == 0]
    med = float(np.median([sum(R[s][x]) / len(R[s][x]) for x in ro])) if ro else float("nan")
    print(
        f"seed {s}: RL-only {len(ro)} ({100 * len(ro) / 149:.1f}%), base-only {len(bo)}, "
        f"RL median rate on RL-only {100 * med:.0f}%"
    )
    out[f"rl_only_s{s}"] = {"rl_only": len(ro), "base_only": len(bo), "median_rate": med}
bsum = {x: sum(sum(B[s][x][:N]) for s in range(3)) for x in keep}
rsum = {x: sum(sum(R[s][x][:N]) for s in range(3)) for x in keep}
ro = [x for x in keep if bsum[x] == 0 and rsum[x] > 0]
bo = [x for x in keep if bsum[x] > 0 and rsum[x] == 0]
all3 = [x for x in ro if all(sum(R[s][x][:N]) > 0 for s in range(3))]
print(
    f"pooled 3x{N}: RL-only {len(ro)} ({100 * len(ro) / 149:.1f}%), base-only {len(bo)}; "
    f"RL-only and solved by all 3 RL seeds: {len(all3)}"
)
out["pooled"] = {"rl_only": len(ro), "base_only": len(bo), "rl_only_all3": len(all3)}
(Path(__file__).parent / "seeds3.json").write_text(json.dumps(out, indent=1))
