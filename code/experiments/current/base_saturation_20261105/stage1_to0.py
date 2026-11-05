"""Arm C of base_saturation_20261105 (PREREG): Stage 1 again from scratch, twice as long, to lr 0.

    cd <the cd089700 code snapshot> &&
    python -m code.experiments.current.base_saturation_20261105.stage1_to0 \
        --data S1g3.pkl --heldout heldout.jsonl --out OUT --seed 0 --best_steps 48000

Exactly `python -m code.thinking.train` as S1g3_s0 ran it (code cd089700: the fork's
`state_train_best.train`, Muon + AdamW, lr 1e-3, 200 warmup steps, token budget 18,432, MTP 0.3,
seed 0, so the same initialisation and the same data order), with one change made here:
`MIN_LR_FRAC` 0.3 -> 0, so the step cosine ends at lr 0 instead of 30% of peak. `--best_steps
48000` (2x S1g3_s0's 24,000) is a command-line flag. Validation loss is logged as Stage 1 logs it
(every `--curve_every` 30 s). Writes OUT/schedule.json naming the change.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from code.thinking import train as T


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    assert (T.STB.MIN_LR_FRAC, T.STB.WARMUP) == (0.3, 200), (T.STB.MIN_LR_FRAC, T.STB.WARMUP)
    T.STB.MIN_LR_FRAC = 0.0
    a = T.parser().parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "schedule.json").write_text(
        json.dumps({"min_lr_frac": 0.0, "warmup": T.STB.WARMUP, "best_steps": a.best_steps})
    )
    return T.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
