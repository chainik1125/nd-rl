"""After the first full read: its reads under the tags dko_s0 and dko_s1 are the secondary it 500 and it 630 (Amendment 2).

    python rename_secondary.py READS

Moves READS/dko_s0 -> READS/dko_s0_it500 and READS/dko_s1 -> READS/dko_s1_it630, after checking what each directory
holds, and records the new tag in its ckpt.json. Refuses to run twice or on the true ends.
"""

# ruff: noqa: E501

from __future__ import annotations

import json
import sys
from pathlib import Path

MOVES = {
    "dko_s0": ("dko_s0_it500", "seg5_dko_s0_it500_rolling.pt"),
    "dko_s1": ("dko_s1_it630", "seg3_dko_s1_it630_alarm.pt"),
}


def main() -> None:
    reads = Path(sys.argv[1])
    for old, (new, file) in MOVES.items():
        src, dst = reads / old, reads / new
        if dst.exists():
            raise SystemExit(f"{dst} exists")
        c = json.loads((src / "ckpt.json").read_text())
        if c["file"] != file:
            raise SystemExit(f"{src} holds {c['file']}, not {file}")
        for stage in (
            "A.jsonl.gz",
            "pick.jsonl.gz",
            "reach.jsonl.gz",
            "negi.jsonl.gz",
            "dn.jsonl.gz",
            "c.json.gz",
        ):
            if not (src / stage).exists():
                raise SystemExit(f"{src} has no {stage}")
        c.update(tag=new, secondary=True)
        (src / "ckpt.json").write_text(json.dumps(c, indent=1))
        src.rename(dst)
        print(f"{old} -> {new}")


if __name__ == "__main__":
    main()
