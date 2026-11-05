"""Merge one base_saturation read's 16 Charles149 chunks (pulled from the pods) into one k1024 read,
with think_hirl's `hirl_queue.merge_group` (cce4dfc, the CPT reads' merge), as cpt_control's
`reads.py` does.

    python -m code.experiments.current.base_saturation_20261105.merge \
        charles149_sat_A_s0_plain_k1024

Chunks: /scratch/cowifornia/nd-rl/base_saturation/pods/<alias>/out/reads/<read>/seed<i> on any
pod (exactly one finished copy per seed 0-15 is required). Output:
/scratch/cowifornia/nd-rl/base_saturation/final/<read>/ (rows_plain_B64.jsonl, summary.json,
chunks.json); refused if it exists.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

S = Path("/scratch/cowifornia/nd-rl/base_saturation")
HQ = Path(
    "/scratch/cowifornia/nd-rl/think_hirl_dev/nd-rl-hirl/code/experiments/current/"
    "thinking_traces_20261005/grpo_v2/launch/hirl/hirl_queue.py"
)  # md5 5c4dcf69..., identical to cce4dfc's
K, N_CHUNKS = 64, 16


def hq():
    spec = importlib.util.spec_from_file_location("hirl_queue", HQ)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main(argv: list[str] | None = None) -> int:
    read = (argv or sys.argv[1:])[0]
    members = []
    for i in range(N_CHUNKS):
        found = [
            d
            for d in sorted(S.glob(f"pods/*/out/reads/{read}/seed{i}"))
            if (d / "rows_plain_B64.jsonl").exists() and (d / "summary.json").exists()
        ]
        assert len(found) == 1, f"{read} seed {i}: {len(found)} finished chunks ({found})"
        d = found[0]
        args = json.loads((d / "args.json").read_text())
        assert args["seed"] == i and args["k"] == K and args["batch"] == 1024, (d, args)
        members.append(
            dict(
                chunk=i,
                seed=i,
                k=K,
                out=str(d),
                sets=args["sets"],
                keep=None,
                ckpt=args["ckpt"],
                id=f"{read}.c{i}",
            )  # fmt: skip
        )
    mod = hq()
    mod.FINAL = S / "final"
    mod.FINAL.mkdir(parents=True, exist_ok=True)
    dst = mod.merge_group(read, members)
    s = json.loads((dst / "summary.json").read_text())["cells"]["plain_B64"]["all"]
    pk = s["pass_at_k_pct"]
    print(
        f"merged {dst}: n {s['n']}, pass@1 {pk['1']:.1f}, pass@5 {pk['5']:.1f}, "
        f"pass@64 {pk['64']:.1f}, pass@1024 {pk['1024']:.1f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
