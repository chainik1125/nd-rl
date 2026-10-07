#!/usr/bin/env python3
"""Build the comparison page (looped arms vs run 198) from whatever results exist so far.

  python report.py [--fetch] OUT.html

Harness milestones come from the pod logs in artifacts/looped_spike_20261006/ (they exist from
the first second of a run); run 198's come from its committed results.json. Sweeps come from
<arm>/loop_sweep_<stage>.json. --fetch first copies summary.json and the sweep files off every
pod still listed in pod_*.json, so a run in progress shows its partial sweep. The page is
report_template.html with the data and notes.json spliced in.
"""

import ast
import glob
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
AUTO = os.path.join(HERE, "..", "autoresearch")
sys.path.insert(0, AUTO)
import pod  # noqa: E402

ART = os.path.join(HERE, "..", "..", "..", "..", "artifacts", "looped_spike_20261006")
ARMS = {  # tag -> (pod log, label, short description)
    "loops1234": ("pod_a.log", "Looped, n ~ {1,2,3,4}", "1 entry + 2 shared core x n + 1 exit"),
    "loops2": ("pod_b.log", "Looped, fixed n = 2", "same blocks, always 2 loops in training"),
    "loops12345678": ("pod_d.log", "Looped, n ~ {1..8}", "same blocks, 1 to 8 loops in training"),
    "loops1234681216": ("pod_e.log", "Looped, n ~ {1..16}", "loops drawn from 1,2,3,4,6,8,12,16"),
}
RUN198 = f"{AUTO}/results/198-claude_lean-muon-no-rms-retest/results.json"


def fetch():
    for state in glob.glob(f"{ART}/pod_*.json"):
        st = json.loads(Path(state).read_text())
        if "port" not in st:  # still booting
            continue
        ssh = "ssh " + " ".join(pod.ssh_args(st))
        files = "$(ls */summary.json */loop_sweep_*.json */more_rl/*.json 2>/dev/null)"
        cmd = f"{ssh} -n 'cd {pod.REMOTE}/out && tar czf - {files}' | tar xzf - -C {ART}"
        subprocess.run(cmd, shell=True, check=False, stderr=subprocess.DEVNULL)


def from_log(tag, log):
    """Harness milestones of one arm, parsed from the launcher's copy of the pod's output."""
    text = Path(ART, log).read_text() if Path(ART, log).exists() else ""
    # pod B's log starts with the baseline sweep; an arm's harness output starts at its pools line
    start = re.search(r"\[\d\d:\d\d:\d\d\] pools: train", text)
    text = text[start.start() :] if start else ""
    arm = {"curve": [], "rounds": [], "status": "running" if text else "queued"}
    for m in re.finditer(r"pretrain step (\d+) train (\S+) val (\S+) lr \S+ (\d+)s", text):
        arm["curve"].append({"secs": int(m[4]), "val": float(m[3])})
        arm["steps"] = int(m[1])
    if m := re.search(r"pretrain val loss by loops: (\{.*\})", text):
        arm["val_by_loops"] = ast.literal_eval(m[1])
    if m := re.search(r"pretrain returned after (\d+)s .*?, ([\d,]+) params", text):
        arm["pretrain_secs"], arm["params"] = int(m[1]), int(m[2].replace(",", ""))
    if m := re.search(r"pretrain held-out greedy (\S+)", text):
        arm["greedy_pre"] = float(m[1])
    pat = (
        r"ei r\d: acc (\S+), \+\d+ proofs, solved (\d+)/\d+, L\*=\d+, (\d+) samples/s, round (\d+)s"
    )
    for m in re.finditer(pat, text):
        arm["rounds"].append(
            {"acc": float(m[1]), "solved": int(m[2]), "sps": int(m[3]), "secs": int(m[4])}
        )
    if m := re.search(r"eval: solved (\d+)/(\d+), ge7 (\d+), L\*=(\d+), max pruned (\d+)", text):
        arm["score"], arm["dev_n"], arm["lstar"] = int(m[3]), int(m[2]), int(m[4])
    if m := re.search(r"SUMMARY status=(\S+) metric=\S+ metric_rl=\S+ total (\d+)s", text):
        arm["status"], arm["total_secs"] = m[1], int(m[2])
    summary = Path(ART, tag, "summary.json")
    if summary.exists():
        ev = json.loads(summary.read_text()).get("eval", {})
        arm["by_bin"] = {L: [b["solved"], b["n"]] for L, b in ev.get("by_bin", {}).items()}
        arm["greedy_post"] = ev.get("heldout_greedy")
    return arm


def run198():
    d = json.loads(Path(RUN198).read_text())
    return {
        "status": "ok",
        "curve": [{"secs": c["secs"], "val": c["val_loss"]} for c in d["pretrain"]["curve"]],
        "steps": d["pretrain"]["steps"],
        "params": d["pretrain"]["n_params"],
        "pretrain_secs": round(d["pretrain"]["secs"]),
        "greedy_pre": d["pretrain"]["heldout_greedy"],
        "greedy_post": d["eval"]["heldout_greedy"],
        "rounds": [
            {
                "acc": r["sample_acc"],
                "solved": r["targets_solved"],
                "sps": r["samples_per_sec"],
                "secs": r["secs"],
            }
            for r in d["rl_rounds"]
        ],
        "score": d["eval"]["ge"]["7"],
        "dev_n": d["eval"]["n"],
        "lstar": d["eval"]["lstar"],
        "by_bin": {L: [b["solved"], b["n"]] for L, b in d["eval"]["by_bin"].items()},
        "total_secs": d["total_secs"],
        "seed_scores": d["seed_metrics"],
    }


def more_rl(tag):
    """The 8 extra expert-iteration rounds (more_rl.py), if they have finished."""
    path = Path(ART, tag, "more_rl", "summary_more_rl.json")
    if not path.exists():
        return None
    d = json.loads(path.read_text())
    return {
        "cumulative": d["cumulative_solved"],
        "score": d["eval"]["solved_ge7"],
        "secs": d["secs"],
        "round_secs": [r["secs"] for r in d["rl"]["rounds"]],
        "by_bin": {L: [b["solved"], b["n"]] for L, b in d["eval"]["by_bin"].items()},
    }


def sweeps(tag):
    """Stage -> sweep rows. The sweep of the more-RL checkpoint is stage `more`."""
    out = {}
    paths = glob.glob(f"{ART}/{tag}/loop_sweep_*.json") + glob.glob(
        f"{ART}/{tag}/more_rl/loop_sweep_final.json"
    )
    for path in paths:
        s = json.loads(Path(path).read_text())
        out["more" if "more_rl" in path else s["stage"]] = [
            {
                **{k: r[k] for k in ("loops", "depth", "pool", "k", "n", "solved", "secs")},
                "sps": r["samples_per_sec"],
                "counts": [p["n_ok"] for p in r["problems"].values()],
            }
            for r in s["results"]
        ]
    return out


def main():
    if "--fetch" in sys.argv:
        fetch()
    arms = [
        {
            "id": "198",
            "label": "Run 198 (untied)",
            "desc": "6 independent blocks",
            **run198(),
            "sweeps": sweeps("baseline"),
            "more": more_rl("baseline"),
        }
    ]
    for tag, (log, label, desc) in ARMS.items():
        arms.append(
            {
                "id": tag,
                "label": label,
                "desc": desc,
                **from_log(tag, log),
                "sweeps": sweeps(tag),
                "more": more_rl(tag),
            }
        )
    data = {
        "generated": time.strftime("%Y-%m-%d %H:%M %Z"),
        "arms": arms,
        "notes": json.loads(Path(HERE, "notes.json").read_text()),
    }
    page = Path(HERE, "report_template.html").read_text().replace("/*DATA*/", json.dumps(data))
    Path(sys.argv[-1]).write_text(page)
    print(f"wrote {sys.argv[-1]}: " + ", ".join(f"{a['id']}={a['status']}" for a in arms))


if __name__ == "__main__":
    main()
