"""Write the M1 `pid:n` input files (Dan's parse_spec format: whitespace-separated `pid:n`) to
/scratch/cowifornia/nd-rl/olmo_elicit/metrics2/inputs/temp/ (pushed to the pod's inputs/temp/ by `m2_pod.sh push`).

  pids_C1_k8.txt / pids_C1_k4.txt   ONE invocation for C1: 176 arm-A non-reach prompts at 2 draws + the 48 reach prompts at
                                     k draws (8 default, 4 under the cap); the 42 reach prompts that are also in arm A are
                                     scored on their first 2 draws for arm A (chunks 0 and 1 with --chunk 1)
  pids_reach_k8.txt / pids_reach_k4.txt   48 reach prompts at k draws (S5R)
  pids_armA_n2.txt                   218 arm-A prompts at 2 draws (C3, C4, C2)
Prompt lists are Dan's (`olmo_ifcurve/pids_armA.txt`, `pids_reach.txt`).
"""

from __future__ import annotations

import sys
from pathlib import Path

DAN = Path("/scratch/cowifornia/nd-rl/olmo_elicit/ext/nd-takehome")
OUT = Path("/scratch/cowifornia/nd-rl/olmo_elicit/metrics2/inputs/temp")


def pids(name: str, root: Path = DAN) -> list[str]:
    return [ln.strip() for ln in open(root / "olmo_ifcurve" / name) if ln.strip()]


def build(root: Path = DAN) -> dict[str, str]:
    arm, reach = pids("pids_armA.txt", root), pids("pids_reach.txt", root)
    rs = set(reach)
    nonreach = [p for p in arm if p not in rs]
    assert len(arm) == 218 and len(reach) == 48 and len(nonreach) == 176, (
        len(arm),
        len(reach),
        len(nonreach),
    )
    files = {}
    for k in (8, 4):
        files[f"pids_C1_k{k}.txt"] = (
            "\n".join([f"{p}:2" for p in nonreach] + [f"{p}:{k}" for p in reach]) + "\n"
        )
        files[f"pids_reach_k{k}.txt"] = "\n".join(f"{p}:{k}" for p in reach) + "\n"
    files["pids_armA_n2.txt"] = "\n".join(f"{p}:2" for p in arm) + "\n"
    return files


def main(out: Path = OUT) -> int:
    out.mkdir(parents=True, exist_ok=True)
    for name, txt in build().items():
        (out / name).write_text(txt)
        print(
            f"{out / name}: {len(txt.split())} pids, {sum(int(t.split(':')[1]) for t in txt.split())} draws"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1]) if len(sys.argv) > 1 else OUT))
