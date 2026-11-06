"""Plot per-round RL curves from the driver logs: python plot_rl.py <log dir> <out.png>"""
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

GATE = re.compile(r"\[lean_gate\] (\d+) samples, parse-fail (\d+)")
ORACLE = re.compile(r"oracle \d+: acc ([\d.]+), .*solved (\d+)/\d+, L\*=(\d+), n_loops (\d+), .*RL clock (\d+)/")
ROUND = re.compile(r"^\[(\d\d:\d\d:\d\d)\] (?:rl|grpo) round \d+: .*rollout entropy ([\d.]+), nll ([\d.]+), (?:at T=[\d.]+ entropy ([\d.]+) nll ([\d.]+), )?(\d+) tokens, all pass (\d+), all fail (\d+), round \d+s, (\d+)s left")

CLIP = re.compile(r"'clip_frac': ([\d.]+)")  # v3 GRPO only
# v4 on: the per-round KL probes, logged just before the round line
PROBE = re.compile(r"probe round \d+: KL\(old\|\|new\) ([\d.]+) KL\(new\|\|old\) ([\d.]+), rollouts KL\(pi\|\|pre\) ([\d.]+) "
                   r"KL\(pre\|\|pi\) ([\d.]+), pretrain KL\(pi\|\|pre\) ([\d.]+) KL\(pre\|\|pi\) ([\d.]+), "
                   r"nll pretrain ([\d.]+) gold ([\d.na]+)")
PROBE_KEYS = ("kl_step", "kl_step_rev", "kl_pre_roll", "kl_pre_roll_rev", "kl_pre_pt", "kl_pre_pt_rev", "nll_pt", "nll_gold")


RESUMED = re.compile(r"resumed from .*: prior RL clock (\d+)s")  # a continued run: its clock starts where the first ended


def parse(path):
    rows, last, offset = [], {}, 0
    gate = None
    for line in path.read_text(errors="replace").splitlines():
        if m := GATE.search(line):
            gate = int(m[2]) / int(m[1])
        elif m := RESUMED.search(line):
            offset = int(m[1]) / 3600
        elif m := ORACLE.search(line):
            # Only the n_loops-1 oracle is the round's main rollout; looped retries at 2/4/8 loops are extra.
            last["solved"], last["lstar"], last["clock"] = int(m[2]), int(m[3]), offset + int(m[5]) / 3600
            if m[4] == "1":
                last["acc"], last["parse_fail"] = float(m[1]), gate
        elif m := PROBE.search(line):
            last.update(zip(PROBE_KEYS, map(float, m.groups())))
        elif m := ROUND.search(line):
            clip = CLIP.search(line)
            rows.append(dict(last, utc=m[1], entropy=float(m[2]), nll=float(m[3]),
                             entropy_t=m[4] and float(m[4]), nll_t=m[5] and float(m[5]), tokens=int(m[6]),
                             all_pass=int(m[7]), all_fail=int(m[8]), left=int(m[9]),
                             clip_frac=clip and float(clip[1])))
    return rows


def main(log_dir, out):
    panels = [("entropy", "rollout entropy (nats/token)"), ("acc", "rollout accuracy (1 loop)"),
              ("parse_fail", "fraction of samples that don't parse"), ("tokens", "mean rollout tokens"),
              ("solved", "RL targets solved (cumulative)"), ("lstar", "L* (body lines)")]
    fig, axes = plt.subplots(3, 2, figsize=(13, 12), sharex=True)
    colours = {"untied-ei": "tab:blue", "looped-ei": "tab:green", "untied-grpo": "tab:red"}
    for log in sorted(Path(log_dir).glob("driver_*.log")):
        arm, seed = log.stem.removeprefix("driver_").rsplit("_s", 1)
        rows = parse(log)
        for ax, (key, _) in zip(axes.flat, panels):
            pts = [(r["clock"], r[key]) for r in rows if r.get(key) is not None]
            if pts:
                ax.plot(*zip(*pts), marker=".", ms=4, lw=1, color=colours[arm],
                        ls="-" if seed == "0" else "--", label=f"{arm} s{seed}")
    for ax, (_, title) in zip(axes.flat, panels):
        ax.set_title(title)
        ax.grid(alpha=0.3)
    for ax in axes[-1]:
        ax.set_xlabel("RL clock (h, GPU time excluding Lean)")
    axes[0, 0].legend()
    fig.tight_layout()
    fig.savefig(out, dpi=110)


if __name__ == "__main__":
    main(*sys.argv[1:])
