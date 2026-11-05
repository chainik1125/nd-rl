"""Shared definitions for knockout_danbase: the lines, paths, hosts and Dan's compute rule.

Every machine holds the same tree under KD; a line's run lives in segments
`KD/runs/<line>/seg<k>/` (one per launch: the first, then one per respawn), each written only by
its own process. Copies go one way: live segment -> `KD/backup/<line>/` on another machine, and
live segment -> `KD/collected/<host>/<line>/` on serrano for the analysis. Nothing is ever copied
back into a live segment; a respawn starts a new segment from a checkpoint.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

KD = Path(os.environ.get("KD", "/scratch/cowifornia/nd-rl/knockout_danbase"))
PY = os.environ.get("DANBASE_PY", "/scratch/cowifornia/nd-rl/think/nd-rl/code/.venv/bin/python")
CODE = KD / "snap" / "f35d3daf_danbase"  # f35d3daf + snap_iters + danbase_log patches
HERE = Path(__file__).resolve().parent
HOSTS = ("serrano", "temescal", "ynez")
MAX_OURS = {"serrano": 5, "temescal": 2, "ynez": 4}  # lines per lab host
# Runpod A40 pods (Leon, 2026-11-05), one pinned line each, outside the lab and its rules: listed
# in KD/sup/pods.json (register_pod.py; the supervisor only reads it). ssh aliases rp-* in
# ~/.ssh/config; a pod's /workspace holds the same tree under the lab's paths (symlinks); a pod is
# deleted when its line ends.
POD_RATE = 0.59  # $ per pod-hour (A40, secure cloud)
POD_BUDGET = 16.90  # the starting ceiling (+ each pod's budget_add): an event, nothing stops
BAR = 14  # Dan's bar: median round-1 heldout_cum of his 6 full bases
SNAP_EVERY = 10  # model-only snapshot (and read) every 10 iterations
ROLL_EVERY = 5  # rolling checkpoint with the training state every 5 iterations

# Dan's ladders' x per round (his rule; rl_curve.json via dan_axis.json) for the matched reads
DAN_AXIS = Path("/scratch/cowifornia/nd-rl/knockout_grpo/dan_axis.json")

# tag -> base, pool, seed, Dan's ladder whose x grid the matched reads follow, priority (low first)
LINES: dict[str, dict] = {
    "dko_s0": {"base": "stage1_ko_s0.pt", "pool": "dan_rl_targets.jsonl", "seed": 0,
               "ladder": "la_ko_s0", "arm": "primary", "prio": 0},
    "dko_s1": {"base": "stage1_ko_s1.pt", "pool": "dan_rl_targets.jsonl", "seed": 1,
               "ladder": "la_ko_s1", "arm": "primary", "prio": 1},
    "dko_s2": {"base": "stage1_ko_s2.pt", "pool": "dan_rl_targets.jsonl", "seed": 2,
               "ladder": "la_ko_s2", "arm": "primary", "prio": 2},
    "dfull_s3": {"base": "stage1_full_s3.pt", "pool": "dan_rl_targets.jsonl", "seed": 3,
                 "ladder": "la_full_s3", "arm": "reference", "prio": 3},
    "dkoour_s0": {"base": "stage1_ko_s0.pt", "pool": "L2_pool.jsonl", "seed": 0,
                  "ladder": "la_ko_s0", "arm": "secondary", "prio": 4},
    "dkoour_s1": {"base": "stage1_ko_s1.pt", "pool": "L2_pool.jsonl", "seed": 1,
                  "ladder": "la_ko_s1", "arm": "secondary", "prio": 5},
    "dkoour_s2": {"base": "stage1_ko_s2.pt", "pool": "L2_pool.jsonl", "seed": 2,
                  "ladder": "la_ko_s2", "arm": "secondary", "prio": 6},
    # Amendment 1 (Leon, 2026-11-05 ~17Z): his 0.1 % and 1 % σ-dose bases, his targets, the
    # primary recipe; seeds those of his la_dose01_s0-2 / la_dose1_s0-2 ladders. dd1_s0 and dd1_s1
    # come last: they are meant for the two Runpod pods (rp-d1, rp-d2) unless a lab GPU frees first.
    "dd01_s0": {"base": "stage1_dose01_s0.pt", "pool": "dan_rl_targets.jsonl", "seed": 0,
                "ladder": "la_dose01_s0", "arm": "dose", "dose": "dose01", "prio": 7},
    "dd01_s1": {"base": "stage1_dose01_s1.pt", "pool": "dan_rl_targets.jsonl", "seed": 1,
                "ladder": "la_dose01_s1", "arm": "dose", "dose": "dose01", "prio": 8},
    "dd01_s2": {"base": "stage1_dose01_s2.pt", "pool": "dan_rl_targets.jsonl", "seed": 2,
                "ladder": "la_dose01_s2", "arm": "dose", "dose": "dose01", "prio": 9},
    "dd1_s2": {"base": "stage1_dose1_s2.pt", "pool": "dan_rl_targets.jsonl", "seed": 2,
               "ladder": "la_dose1_s2", "arm": "dose", "dose": "dose1", "prio": 10},
    "dd1_s0": {"base": "stage1_dose1_s0.pt", "pool": "dan_rl_targets.jsonl", "seed": 0,
               "ladder": "la_dose1_s0", "arm": "dose", "dose": "dose1", "prio": 11},
    "dd1_s1": {"base": "stage1_dose1_s1.pt", "pool": "dan_rl_targets.jsonl", "seed": 1,
               "ladder": "la_dose1_s1", "arm": "dose", "dose": "dose1", "prio": 12},
}  # fmt: skip


def dan_ladder(name: str) -> dict:
    """{D, x: [x per round on his rule], y: [heldout_cum per round]} of one of Dan's ladders."""
    for o in json.loads(DAN_AXIS.read_text()):
        if o["name"] == name:
            return {"D": o["D"], "x": [r["x_dan"] for r in o["rounds"]],
                    "y": [r["y"] for r in o["rounds"]]}  # fmt: skip
    raise KeyError(name)


def x_cap(line: str) -> float:
    """The line's compute cap on Dan's rule: his ladder's x at round 16 (one ladder)."""
    return dan_ladder(LINES[line]["ladder"])["x"][15]


def x_dan_row(r: dict, D: float) -> float:
    """One GRPO iteration's compute on Dan's rule: 2N (generated + re-prefilled state tokens) of
    sampling + 6N (state + action tokens) of the update, over 6N D."""
    return (2 * (r["sampled_gen_tokens"] + r["sampled_prefill_tokens"])
            + 6 * (r["item_prompt_tokens"] + r["item_gen_tokens"])) / (6 * D)  # fmt: skip


def x_ours_row(r: dict, D: float) -> float:
    """... on our rule (knockout_grpo's R): (6 trained + 2 sampled generated tokens) / 6 D,
    prefill excluded."""
    return (6 * r["trained_tokens"] + 2 * r["sampled_gen_tokens"]) / (6 * D)


def grpo_argv(line: str, seg_dir: Path, iters: int, resume: str | None) -> list[str]:
    """The recipe (PREREG.md): code.thinking.grpo at f35d3daf, no thinking, Dan's env caps."""
    L = LINES[line]
    base = str(KD / "bases" / L["base"])
    snaps = ",".join(str(i) for i in range(SNAP_EVERY, iters + 1, SNAP_EVERY))
    argv = [
        "--init", base, "--name", line, "--out", str(seg_dir),
        "--targets", str(KD / "pools" / L["pool"]), "--iters", str(iters),
        "--prompts", "256", "--group", "16", "--batch", "1024", "--lr", "1.5e-4",
        "--unlikely_beta", "0.25", "--overlong", "fail", "--save_every", str(iters),
        "--save_rolling", str(ROLL_EVERY), "--keep_rolling", "4",
        "--max_steps", "48", "--max_action", "256", "--seed", str(L["seed"]), "--budgets", "0",
        "--canary", str(KD / "pools" / "canary_long.jsonl"), "--canary_n", "1000",
        "--canary_every", "1", "--rollback_on_alarm", "--max_rollbacks", "3",
        "--kl_anchor_coef", "0", "--kl_anchor_ref", base, "--monitor", "--arm", "logical",
        "--max_rej", "0", "--rej_penalty", "0.1", "--snap_iters", snaps,
    ]  # fmt: skip
    if resume:
        argv += ["--resume", resume]
    return argv
