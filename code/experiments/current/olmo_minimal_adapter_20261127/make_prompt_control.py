# ruff: noqa: E501
"""Prompt control inputs (SERRANO side): the 48 reach prompts with Dan's registered answer-only instruction appended.

  make_prompt_control.py [--out-dir /scratch/cowifornia/nd-rl/olmo_elicit/v2/stage]

Writes <out-dir>/problems_answeronly.jsonl and pids_answeronly.txt (48 x 8, same pids and order as stage2/pids_L.txt).
Each row is the reach prompt's row of Dan's olmo_ifomega/problems.jsonl, unchanged, plus `prompt_sample` = prompt + the suffix.
His sampler sends `prompt_sample` as the user turn; his grader scores `prompt` and the kwargs, so the answer-only wording does not
change what is checked. Run on the pod as `mini_pod.sh plain <alias> dpo answeronly --problems ... --pids ... --seed 650000`.

The wording: condition A1 of his olmo-answeronly run. His registry file olmo_answeronly/wordings.py is not in the sparse clone;
the wording is read from the registered request rows artifacts/olmo_answeronly/requests/s2300_reach.jsonl.gz (field user_suffix of
every cond == "A1" row), and this script checks that all those rows carry the same suffix and that it equals the constant below:
    "\\n\\nAfter thinking, reply with only your final answer, exactly as drafted, with no introduction or explanation."
(make_stage2.py of the 2026-11-25 run holds the same text as A1.)
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

DAN = Path("/scratch/cowifornia/nd-rl/olmo_elicit/ext/nd-takehome")
OE = Path("/scratch/cowifornia/nd-rl/olmo_elicit")
A1 = "After thinking, reply with only your final answer, exactly as drafted, with no introduction or explanation."
SUFFIX = "\n\n" + A1
REQUESTS = "artifacts/olmo_answeronly/requests/s2300_reach.jsonl.gz"
SEED = 650000


def registered_suffix(dan: Path = DAN) -> str:
    """The user_suffix of Dan's A1 request rows (one value; raises if they differ)."""
    sufs = {
        json.loads(ln)["user_suffix"]
        for ln in gzip.open(dan / REQUESTS, "rt").read().splitlines()  # noqa: SIM115
        if json.loads(ln)["cond"] == "A1"
    }
    if len(sufs) != 1:
        raise ValueError(f"A1 rows carry {len(sufs)} different suffixes")
    return sufs.pop()


def build(
    dan: Path = DAN, pids_file: Path = OE / "stage2" / "pids_L.txt"
) -> tuple[list[dict], list[str]]:
    probs = {}
    for ln in (dan / "olmo_ifomega" / "problems.jsonl").read_text().splitlines():
        r = json.loads(ln)
        probs[r["pid"]] = r
    lines = pids_file.read_text().split()
    reach = [ln.partition(":")[0] for ln in lines]
    assert set(reach) == {
        x.strip()
        for x in (dan / "olmo_ifcurve" / "pids_reach.txt").read_text().splitlines()
        if x.strip()
    }
    suffix = registered_suffix(dan)
    assert suffix == SUFFIX, repr(suffix)
    rows = []
    for pid in reach:
        r = dict(probs[pid])
        assert "prompt_sample" not in r and "messages_sample" not in r
        r["prompt_sample"] = r["prompt"] + suffix
        rows.append(r)
    return rows, lines


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(OE / "v2" / "stage"))
    a = ap.parse_args()
    rows, lines = build()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "problems_answeronly.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (out / "pids_answeronly.txt").write_text("\n".join(lines) + "\n")
    print(
        f"wrote {len(rows)} rows -> {out}/problems_answeronly.jsonl, pids_answeronly.txt (seed for this cell: {SEED})"
    )
    print(f"suffix (source {DAN / REQUESTS}, cond A1 user_suffix): {SUFFIX!r}")


if __name__ == "__main__":
    main()
