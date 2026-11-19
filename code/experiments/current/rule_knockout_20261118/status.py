#!/usr/bin/env python3
# 120 columns, as in long_rl_20261014's files
# ruff: noqa: E501
"""Pull every live run's rounds.jsonl and found_stream.jsonl and print one row per run.

  python status.py            # pull once, print the table
  python status.py --loop 600 # pull every 600 s, rewriting artifacts/rule_knockout_20261118/live/STATUS.md

Live copies go to artifacts/rule_knockout_20261118/live/<run>/. A finished run's files are read from its pulled
directory (rule-knockout-20261118_<run>/). Columns, for the arm's removed rule X (the control row shows IMPE):
  acc         share of this call's attempts Lean accepted
  solved      RL targets with at least one accepted proof so far
  X now       accepted attempts in the latest call that use X
  X proofs    distinct accepted proofs using X so far, and the targets they are on
  ref-X       solved targets whose reference proof uses X (a proxy for needing X)
"""

import glob
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "autoresearch"))
import pod  # noqa: E402

ART = os.path.abspath(
    os.path.join(HERE, "..", "..", "..", "..", "artifacts", "rule_knockout_20261118")
)
LIVE = os.path.join(ART, "live")
RULE = {"mp": "IMPE", "cp": "IMPI", "exfalso": "BOTE", "dn": "DN", "full": "IMPE"}


def pull():
    for state in glob.glob(f"{ART}/pod_*.json"):
        st = json.loads(Path(state).read_text())
        name = os.path.basename(state)[4:-5]
        if "ip" not in st:
            continue
        os.makedirs(f"{LIVE}/{name}", exist_ok=True)
        for fn in ("rounds.jsonl", "found_stream.jsonl"):
            cmd = [
                "ssh",
                "-C",
                *pod.ssh_args(st),
                f"cat /workspace/long_rl/{name}/{fn} 2>/dev/null",
            ]
            out = subprocess.run(cmd, capture_output=True, stdin=subprocess.DEVNULL, timeout=300)
            if out.returncode == 0 and out.stdout:  # keep the last good copy when a pull fails
                with open(f"{LIVE}/{name}/{fn}", "wb") as f:
                    f.write(out.stdout)


def rounds(name):
    for fn in (f"{ART}/rule-knockout-20261118_{name}/rounds.jsonl", f"{LIVE}/{name}/rounds.jsonl"):
        if os.path.exists(fn):
            return [json.loads(line) for line in Path(fn).read_text().splitlines() if line.strip()]
    return []


def table():
    names = sorted(
        {os.path.basename(d) for d in glob.glob(f"{LIVE}/*")}
        | {
            os.path.basename(d).split("_", 1)[1]
            for d in glob.glob(f"{ART}/rule-knockout-20261118_*")
        }
    )
    out = [
        "| run | calls | RL min | acc | solved | X | X now | X proofs (targets) | ref-X |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name in names:
        rs = rounds(name)
        if not rs:
            out.append(f"| {name} | 0 | | | | | | | |")
            continue
        x, c, cen = RULE[name.split("-")[0]], rs[-1], rs[-1]["census"]
        out.append(
            f"| {name} | {len(rs)} | {c['rl_secs_used'] / 60:.0f} | {c['sample_acc']:.3f} | {c['targets_solved']:,} | {x} | "
            f"{cen['accepted_using'].get(x, 0)} of {cen['accepted']} | {cen['proofs_using'].get(x, 0):,} ({cen['targets_using'].get(x, 0):,}) | "
            f"{cen['solved_ref_uses'].get(x, 0):,} |"
        )
    return "\n".join(out)


if __name__ == "__main__":
    loop = int(sys.argv[2]) if sys.argv[1:2] == ["--loop"] else 0
    while True:
        pull()
        text = f"Pulled {time.strftime('%Y-%m-%d %H:%M %Z')}\n\n{table()}\n"
        os.makedirs(LIVE, exist_ok=True)
        with open(f"{LIVE}/STATUS.md", "w") as f:
            f.write(text)
        print(text, flush=True)
        if not loop:
            break
        time.sleep(loop)
