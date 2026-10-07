#!/usr/bin/env python3
"""Build the six-arm comparison page from whatever results exist so far.

  python report.py [--fetch] OUT.html

Harness milestones come from the pod logs in artifacts/coconut_spike_20261006/ (they exist from
the first second of a run); run 198's come from its committed results.json and from the looped
spike's sweep of its checkpoints. Sweeps come from <tag>/thought_sweep_<stage>.json; --fetch first
copies them and summary.json off every pod still listed in pod_*.json. The page is
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

ART = os.path.join(HERE, "..", "..", "..", "..", "artifacts", "coconut_spike_20261006")
LOOPED = os.path.join(ART, "..", "looped_spike_20261006", "baseline")
RUN198 = f"{AUTO}/results/198-claude_lean-muon-no-rms-retest/results.json"
ARMS = {  # arm -> (run tags, first usable one wins; label; what the thoughts are trained with)
    "none": (["none"], "Unsupervised", "feedback thoughts, proof loss only"),
    "pause": (["pause"], "Pause control", "one learned vector per thought, no feedback"),
    "probe": (["probe"], "Rule probe", "feedback; a head reads rule t off thought t"),
    "curriculum": (
        ["curriculum", "curriculum-b"],
        "Coconut curriculum",
        "feedback; plan steps start explicit and turn latent in stages",
    ),
    "codi": (
        ["codi"],
        "Self-distillation",
        "feedback; an explicit-plan pass teaches the latent pass",
    ),
    "future": (
        ["future"],
        "Future states",
        "feedback; thought t predicts the state at the t-th `;`",
    ),
}


def fetch():
    for state in glob.glob(f"{ART}/pod_*.json"):
        st = json.loads(Path(state).read_text())
        if "port" not in st:  # still booting
            continue
        ssh = "ssh " + " ".join(pod.ssh_args(st))
        files = "$(ls */summary.json */thought_sweep_*.json 2>/dev/null)"
        cmd = f"{ssh} -n 'cd {pod.REMOTE}/out && tar czf - {files}' | tar xzf - -C {ART}"
        subprocess.run(cmd, shell=True, check=False, stderr=subprocess.DEVNULL)


def from_log(tag):
    """Harness milestones of one run, parsed from the launcher's copy of the pod's output. A
    relaunched run appends to the same log, so only the text after the last pools line counts."""
    path = Path(ART, f"pod_{tag}.log")
    text = path.read_text() if path.exists() else ""
    starts = [m.start() for m in re.finditer(r"\[\d\d:\d\d:\d\d\] pools: train", text)]
    text = text[starts[-1] :] if starts else ""
    run = {"tag": tag, "rounds": [], "status": "running" if text else "booting"}
    for m in re.finditer(r"pretrain step (\d+) train \S+ val (\S+) arm", text):
        run["steps"], run["val"] = int(m[1]), float(m[2])
    if m := re.search(r"pretrain val loss by thoughts: (\{.*\})", text):
        run["val_by_thoughts"] = ast.literal_eval(m[1])
    if m := re.search(r"pretrain held-out greedy (\S+)", text):
        run["greedy_pre"] = float(m[1])
    pat = (
        r"ei r\d: acc (\S+), \+\d+ proofs, solved (\d+)/\d+, L\*=\d+, (\d+) samples/s, round (\d+)s"
    )
    for m in re.finditer(pat, text):
        run["rounds"].append(
            {"acc": float(m[1]), "solved": int(m[2]), "sps": int(m[3]), "secs": int(m[4])}
        )
    if m := re.search(r"eval: solved \d+/\d+, ge7 (\d+), .*held-out greedy (\S+)", text):
        run["score"], run["greedy_post"] = int(m[1]), float(m[2])
    if m := re.search(r"SUMMARY status=(\S+) metric=\S+ metric_rl=\S+ total (\d+)s", text):
        run["status"], run["total_secs"] = m[1], int(m[2])
    run["sweeps"] = {}
    for sweep in glob.glob(f"{ART}/{tag}/thought_sweep_*.json"):
        s = json.loads(Path(sweep).read_text())
        run["sweeps"][s["stage"]] = [
            {
                **{k: r[k] for k in ("thoughts", "pool", "k", "n", "solved")},
                "sps": r["samples_per_sec"],
            }
            for r in s["results"]
        ]
    return run


def run198():
    d = json.loads(Path(RUN198).read_text())
    sweeps = {}
    for sweep in glob.glob(f"{LOOPED}/loop_sweep_*.json"):
        s = json.loads(Path(sweep).read_text())
        sweeps[s["stage"]] = [{k: r[k] for k in ("pool", "k", "n", "solved")} for r in s["results"]]
    return {
        "id": "198",
        "label": "Run 198",
        "desc": "no thoughts; the untied 6-block model every arm starts from",
        "status": "ok",
        "steps": d["pretrain"]["steps"],
        "val": d["pretrain"]["curve"][-1]["val_loss"],
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
        "total_secs": d["total_secs"],
        "sweeps": sweeps,
    }


def main():
    if "--fetch" in sys.argv:
        fetch()
    arms = [run198()]
    for arm, (tags, label, desc) in ARMS.items():
        runs = [from_log(tag) for tag in tags]
        best = next((r for r in runs if r["status"] == "ok"), None) or max(
            runs, key=lambda r: len(r["rounds"])
        )
        arms.append({"id": arm, "label": label, "desc": desc, **best})
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
