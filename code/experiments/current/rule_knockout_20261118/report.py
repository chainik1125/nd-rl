#!/usr/bin/env python3
# 120 columns, as in long_rl_20261014's files
# ruff: noqa: E501
"""Build the Rule Knockouts report page from whatever the runs have produced so far.

  python report.py <out.html>

Reads each run's rounds.jsonl (a finished run's pulled directory first, else the live copy status.py keeps under
artifacts/rule_knockout_20261118/live/), counts rule use in long_rl_20261014's pretraining and RL-target files, and
fills report_template.html. TAKEAWAYS and NOTES are written by hand per publish.
"""

import collections
import glob
import gzip
import json
import os
import re
import sys
import time
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.abspath(
    os.path.join(HERE, "..", "..", "..", "..", "artifacts", "rule_knockout_20261118")
)
DATA = os.path.join(HERE, "..", "long_rl_20261014", "data")
RULE_RE = re.compile(r": ([A-Z]+[12]?)\b")
RULES = ("IMPE", "IMPI", "BOTE", "DN")

ARMS = [
    {
        "id": "mp",
        "short": "MP",
        "rule": "IMPE",
        "name": "Modus ponens",
        "kind": "pattern knockout",
        "eli5": "You know “if A then B”. You know A. So you may write down B. It is the most basic way to use an if-then fact.",
        "example": "P → Q ,  P    ⊢    Q",
        "still": "Still in pretraining: the same “apply one line to another” move, but only for negations (¬A applied to A gives a contradiction). Every token is familiar; this use of it is not.",
    },
    {
        "id": "cp",
        "short": "CP",
        "rule": "IMPI",
        "name": "Conditional proof",
        "kind": "pattern knockout",
        "eli5": "To prove “if A then B”: pretend A is true, work your way to B, then close the box and conclude “if A then B”.",
        "example": "[ assume P  …  R ]    ⊢    P → R",
        "still": "Still in pretraining: the same assume-and-derive box, but only for proving negations and for case splits.",
    },
    {
        "id": "exfalso",
        "short": "EF",
        "rule": "BOTE",
        "name": "Ex falso",
        "kind": "token knockout",
        "eli5": "Once you have reached a contradiction, you may conclude anything you like.",
        "example": "⊥    ⊢    Q",
        "still": "Nothing like it stays: the Lean token it is written with (.elim) never appears. But it has a substitute. Proof by contradiction also gets anything from ⊥, and that rule is kept in this arm.",
    },
    {
        "id": "dn",
        "short": "DN",
        "rule": "DN",
        "name": "Double-negation elimination",
        "kind": "token knockout",
        "eli5": "If it is false that A is false, then A is true. It is the only classical rule in this system, so removing it removes every proof by contradiction.",
        "example": "¬¬P    ⊢    P",
        "still": "Nothing like it stays: Classical.byContradiction never appears in this arm's pretraining.",
    },
    {
        "id": "full",
        "short": "control",
        "rule": None,
        "name": "Control",
        "kind": "",
        "eli5": "",
        "example": "",
        "still": "",
    },
]

TAKEAWAYS = []  # filled in per publish, below `main`
NOTES = [
    "Every arm is three expert-iteration seeds at about 3 hours of RL plus one GRPO seed at 2 hours; the control is two seeds. Treat a flat line as “not in 3 hours”, not “never”.",
    "“Uses it” is read off the proofs the model wrote. “Solved targets whose reference proof uses it” is a proxy for needing the rule: a reference proof that uses a rule does not show every proof must.",
    "Expert iteration and GRPO learn only from accepted proofs. If the removed rule is never sampled there is nothing to learn from, so zero uses can mean the first success never happened.",
    "The arms are not size-matched. Removing modus ponens or conditional proof drops about 38% of the pretraining proofs; ex falso drops 2%.",
    "Three focused-pool runs (only targets whose reference proof uses the removed rule) got zero accepted proofs of any kind in two hours. They are in the table but not on the charts, because their target pool is different.",
    "An internet outage at 3:37 pm stranded 17 of the 21 first-wave pods before they started. Those runs were restarted at 6:49 pm on the same machines.",
    "Three GRPO runs (modus ponens, ex falso, double-negation) lost their pods before the final eval, so they have RL curves but no held-out score.",
]


def rules(proof):
    return set(RULE_RE.findall(proof or ""))


def rule_counts():
    pre, pre_n = collections.Counter(), 0
    with gzip.open(f"{DATA}/pretrain.jsonl.gz", "rt") as f:
        for line in f:
            pre_n += 1
            pre.update(rules(json.loads(line)["proof"]) & set(RULES))
    rl, rl_n, rl_ref_n = collections.Counter(), 0, 0
    for line in Path(f"{DATA}/rl.jsonl").read_text().splitlines():
        r = json.loads(line)
        rl_n += 1
        if r.get("proof"):
            rl_ref_n += 1
            rl.update(rules(r["proof"]) & set(RULES))
    return pre, pre_n, rl, rl_n, rl_ref_n


def load_run(name):
    done = f"{ART}/rule-knockout-20261118_{name}"
    fn = (
        f"{done}/rounds.jsonl"
        if os.path.exists(f"{done}/rounds.jsonl")
        else f"{ART}/live/{name}/rounds.jsonl"
    )
    lines = Path(fn).read_text().splitlines() if os.path.exists(fn) else []
    rs = [json.loads(line) for line in lines if line.strip()]
    arm, recipe = name.split("_s")[0].split("-", 1)
    run = {
        "name": name,
        "arm": arm,
        "recipe": recipe,
        "seed": int(name.rsplit("_s", 1)[1]),
        "status": "finished"
        if os.path.exists(f"{done}/summary.json")
        else "cut off",  # every pod is gone
        "t": [round(c["rl_secs_used"] / 60, 1) for c in rs],
        "solved": [c["targets_solved"] for c in rs],
        "ref": {x: [c["census"]["solved_ref_uses"].get(x, 0) for c in rs] for x in RULES},
        "use": {x: [c["census"]["proofs_using"].get(x, 0) for c in rs] for x in RULES},
        "accepted": sum(c["census"]["accepted"] for c in rs),
        "accepted_using": {
            x: sum(c["census"]["accepted_using"].get(x, 0) for c in rs) for x in RULES
        },
    }
    run["dev"] = "no eval"
    if run["status"] == "finished":
        dev = json.loads(Path(f"{done}/summary.json").read_text()).get("eval", {}).get("dev")
        run["dev"] = f"{dev['solved']} / {dev['n']}" if dev else "no eval"
    return run


def main(out):
    pre, pre_n, rl, rl_n, rl_ref_n = rule_counts()
    arms = [
        {
            **a,
            "pre_n": pre[a["rule"]],
            "pre_pct": 100 * pre[a["rule"]] / pre_n,
            "rl_n": rl[a["rule"]],
            "rl_pct": 100 * rl[a["rule"]] / rl_ref_n,
        }
        if a["rule"]
        else a
        for a in ARMS
    ]
    names = {os.path.basename(d) for d in glob.glob(f"{ART}/live/*") if os.path.isdir(d)}
    names |= {
        os.path.basename(d).split("_", 1)[1] for d in glob.glob(f"{ART}/rule-knockout-20261118_*")
    }
    order = {a["id"]: i for i, a in enumerate(ARMS)}
    runs = sorted(
        (load_run(n) for n in names), key=lambda r: (order[r["arm"]], r["recipe"], r["seed"])
    )
    data = {
        "pulled": time.strftime("%-I:%M %p PT, %b %-d").replace("AM", "am").replace("PM", "pm"),
        "pretrain_n": pre_n,
        "rl_n": rl_n,
        "rl_ref_n": rl_ref_n,
        "arms": arms,
        "runs": runs,
        "takeaways": TAKEAWAYS,
        "notes": NOTES,
    }
    html = (
        Path(f"{HERE}/report_template.html")
        .read_text()
        .replace("__DATA__", json.dumps(data, ensure_ascii=False))
    )
    with open(out, "w") as f:
        f.write(html)
    for r in runs:
        last = len(r["t"]) - 1
        x = ARMS[order[r["arm"]]]["rule"] or "IMPE"
        print(
            f"{r['name']:22s} {r['status']:8s} calls {len(r['t']):3d} t {r['t'][last] if last >= 0 else 0:6.1f} solved {r['solved'][last] if last >= 0 else 0:6d} "
            f"use[{x}] {r['use'][x][last] if last >= 0 else 0:5d} ref[{x}] {r['ref'][x][last] if last >= 0 else 0:5d} accepted {r['accepted']}"
        )


if __name__ == "__main__":
    notes_file = os.path.join(
        os.path.dirname(sys.argv[1]), "takeaways.txt"
    )  # one takeaway per line, edited per publish
    if os.path.exists(notes_file):
        TAKEAWAYS[:] = [x.strip() for x in Path(notes_file).read_text().splitlines() if x.strip()]
    main(sys.argv[1])
