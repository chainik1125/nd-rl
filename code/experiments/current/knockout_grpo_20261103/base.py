"""Stage-1 base for one dose arm and seed, then its Stage-1 token count. Stdlib only.

    python3 base.py ARM SEED      # KO/bases/<ARM>_s<SEED>/{model.pt,train.json,stage1_tokens.json}

The S1g3 command (runs/rebuild_ynez/stage1_sched.sh) with train.py at cd089700:
`code.thinking.train --data KO/pkl/S1g3_<ARM>.pkl --heldout KO/data/heldout.jsonl --out ...
--seed SEED --best_steps 24000`. ARM `full` copies nothing: its bases are the existing S1g3_s<S>.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import measure as M  # noqa: E402

KO = Path(os.environ.get("KO", "/scratch/cowifornia/nd-rl/knockout_grpo"))
PY = "/scratch/cowifornia/nd-rl/think/nd-rl/code/.venv/bin/python"


def main() -> None:
    arm, seed = sys.argv[1], int(sys.argv[2])
    out = KO / "bases" / f"{arm}_s{seed}"
    pkl = KO / "pkl" / f"S1g3_{arm}.pkl"
    if not (out / "model.pt").exists():
        cmd = [PY, "-m", "code.thinking.train", "--data", str(pkl), "--heldout",
               str(KO / "data" / "heldout.jsonl"), "--out", str(out), "--seed", str(seed),
               "--best_steps", "24000"]  # fmt: skip
        rc = subprocess.run(cmd, cwd=str(KO / "snap" / "cd089700")).returncode
        if rc != 0:
            raise SystemExit(f"train rc {rc}")
    tok = M.stage1_tokens(out / "train.json", pkl)
    (out / "stage1_tokens.json").write_text(json.dumps(tok, indent=1) + "\n")
    print(json.dumps(tok))


if __name__ == "__main__":
    main()
