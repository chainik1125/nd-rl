"""Think-ban guard events per line, from the GRPO stage logs (collected on serrano).

    python3 think_ban.py      # OUT/logs/{,ynez/,temescal/}<line>_L{2,3}.log -> OUT/think_ban.json

The recipe's guard (L2 at fa28f6eb, L3 at f35d3daf) measures the model's probability mass on
banned tokens inside the think block. Above 0.001 it logs `ALARM it N: think banned mass ...` and
tries a rollback to its restore point (`ALARM it N: rollback: ...`), unless the restore point
itself is above the level (`no rollback`); above 0.005 it ends the run (`STOP it N: think banned
mass ...`, exit 3). Counted per line and stage: GRPO iterations logged, iterations with a
think-ban alarm, rollbacks, no-rollback alarms, and stops. Iterations are distinct iteration
numbers, so a resumed stage's repeated iterations count once.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

OUT = Path(os.environ.get("KO_OUT", "/scratch/cowifornia/nd-rl/knockout_grpo"))
IT = re.compile(r"^it (\d+):")
BAN = re.compile(r"^ALARM it (\d+): think banned mass")
KEYS = ("its", "alarm", "rollback", "no_rollback", "stop")
RB = re.compile(r"^ALARM it (\d+): rollback: .*think banned mass")
NORB = re.compile(r"^ALARM it (\d+): no rollback: .*think banned mass")
STOP = re.compile(r"^STOP it (\d+): think banned mass")


def stage(p: Path) -> dict:
    sets: dict[str, set[int]] = {k: set() for k in KEYS}
    for line in p.read_text(errors="replace").splitlines():
        for key, rx in (("its", IT), ("alarm", BAN), ("rollback", RB), ("no_rollback", NORB),
                        ("stop", STOP)):  # fmt: skip
            m = rx.match(line)
            if m:
                sets[key].add(int(m.group(1)))
    return {k: len(v) for k, v in sets.items()} | {"stop_iters": sorted(sets["stop"])}


def main() -> None:
    logs = OUT / "logs"
    found: dict[str, dict] = {}
    for d in (logs, logs / "ynez", logs / "temescal"):
        for p in sorted(d.glob("*_L[23].log")):
            if p.name.startswith("pilot_"):
                continue
            line, st = p.stem.rsplit("_", 1)
            found.setdefault(line, {})[st] = stage(p) | {"log": str(p.relative_to(OUT))}
    res = {}
    for line in sorted(found):
        s = found[line]
        tot = {k: sum(s[x][k] for x in s) for k in KEYS}
        res[line] = {"stages": s, "total": tot}
        print(f"{line:12s} its {tot['its']:4d}  think-ban alarms {tot['alarm']:3d} "
              f"({100 * tot['alarm'] / max(1, tot['its']):4.1f} % of its)  rollbacks "
              f"{tot['rollback']:2d}  no-rollback {tot['no_rollback']:2d}  "
              f"stops {tot['stop']}")  # fmt: skip
    (OUT / "think_ban.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
