#!/usr/bin/env python3
"""Writes think_hirl/queue/jobs.jsonl, the high-RL pass@k queue, in priority order (2026-10-30,
agent:claude for Leon). Jobs already in the file keep their place; new ones are appended, so
rerunning after an edit here only adds jobs (an id never changes meaning).

    python3 hirl_plan.py [--dry]      (--dry: print the jobs, write nothing)

Blocks (README.md next to this file):
  T   timing: the J1 base chunks 0-2 (chunk 0 on charles173 = the existing k64 read's exact
      config, its 149 kept in the merge; chunks 1, 2 together on GPU 1: two reads per GPU)
  J2  the low-RL arm, training: grpo_think_vs_mini20_kdense (mini20's L2-mini, 8 iterations, dense
      step checkpoints) and grpo_think_vs_mini20_kp16 (the same with --prompts 16, 12 iterations)
  J1  base S1g3_s0 and the L3 end (grpo_think_vs_mb_kkl0 s0): Charles149 k1024 (16 chunks of
      k64, seeds 0-15) and the 250-problem 13-20 subset (t1320s250) k512 (8 chunks)
  J3  S1g3 s1 and s2: s1g3_ckreads's reads (line.sh CKR_PANEL + MINI_READS); textbook72 B64 k64
      for S1g3 s0-s2 (sealed: read only)
  J4a Charles149 k1024 along the RL axis (most informative checkpoints first)
  J4b t1320s250 k256 (4 chunks) along the RL axis
A read's chunk is one eval.py call; eval.py --seed is the chunk's seed. Names follow
runs/eval/v2: <set>_<arm>[_tag]_s<seed>_plain_k<k>.
"""

from __future__ import annotations

import fcntl
import json
import sys
from pathlib import Path
from typing import Any

H = Path("/scratch/cowifornia/nd-rl/think_hirl")
S = H / "sets"
CK = H / "runs/ckpts"
LOW = H / "lowrl"
C149 = str(S / "charles149/charles_clean.jsonl")
C173 = ",".join(
    str(S / f"charles173/{f}.jsonl") for f in ("candidate_v0", "candidate_v1", "batch3")
)
T250 = str(S / "t1320s250/t1320_n1000.jsonl")
KEEP149 = str(S / "charles149_names.txt")
BASE = {s: str(H / f"runs/stage1/S1g3_s{s}/model.pt") for s in (0, 1, 2)}

# the model classes' wall-time estimates for one k64 chunk (minutes, serrano A40 reads scaled;
# the queue rescales them by the finished chunks of each class)
EST = {
    ("c149", "base"): 4.5,
    ("c149", "low"): 5.0,
    ("c149", "rl"): 5.5,
    ("t250", "base"): 10.0,
    ("t250", "low"): 12.0,
    ("t250", "rl"): 25.0,
}


def chunk_jobs(
    block: str,
    arm: str,
    ckpt: str,
    mc: str,
    set_: str,
    nchunks: int,
    after: list[str] | None = None,
    repro0: bool = False,
    gpus: dict[int, list[int]] | None = None,
) -> list[dict[str, Any]]:
    """`nchunks` k64 chunks of one read; repro0: chunk 0 on charles173 (the existing k64 read's
    exact sets), its Charles149 rows kept in the merge."""
    k = 64 * nchunks
    group = f"{'charles149' if set_ == 'c149' else 't1320s250'}_{arm}_s0_plain_k{k}"
    est = EST[(set_, mc)]
    out = []
    for c in range(nchunks):
        sets = C149 if set_ == "c149" else T250
        j: dict[str, Any] = {
            "id": f"{group}.c{c}",
            "block": block,
            "kind": "read",
            "group": group,
            "nchunks": nchunks,
            "chunk": c,
            "seed": c,
            "k": 64,
            "ckpt": ckpt,
            "cls": f"{set_}:{mc}",
            "est_min": est,
            "out": str(H / f"eval/chunks/{group}/seed{c}"),
        }
        if repro0 and c == 0:
            j.update(
                sets=C173,
                keep=KEEP149,
                cls=f"c173:{mc}",
                est_min=round(est * 173 / 149, 1),
                out=str(H / f"eval/chunks/{group}/seed0_charles173"),
            )
        else:
            j["sets"] = sets
        j["timeout_s"] = int(max(30, 4 * j["est_min"]) * 60)
        j["needs"] = [ckpt]
        if after:
            j["after"] = after
        if gpus and c in gpus:
            j["gpus"] = gpus[c]
        out.append(j)
    return out


def single(
    block: str, name: str, ckpt: str, sets: str, k: int, est: float, cls: str
) -> dict[str, Any]:
    """One read as line.sh's read1 makes it (seed 0), into eval/final/NAME."""
    return {
        "id": name,
        "block": block,
        "kind": "read",
        "k": k,
        "seed": 0,
        "ckpt": ckpt,
        "sets": sets,
        "out": str(H / f"eval/final/{name}"),
        "cls": cls,
        "est_min": est,
        "timeout_s": int(max(30, 4 * est) * 60),
        "needs": [ckpt],
    }


def mini20_l2(name: str, iters: int, save_steps: str, prompts: int = 256) -> list[str]:
    """grpo_think_vs_mini20_s0_L2's argv (its args.json _meta.argv) with think_hirl's copies of
    the same files; --iters (20 there) and --prompts (256) as given, plus --save_steps."""
    o = LOW / name
    init = BASE[0]
    return [
        "--init",
        init,
        "--name",
        name,
        "--out",
        str(o),
        "--targets",
        str(H / "runs/pools/mini/L2_mini.jsonl"),
        "--iters",
        str(iters),
        "--prompts",
        str(prompts),
        "--group",
        "16",
        "--batch",
        "1024",
        "--lr",
        "1.5e-4",
        "--overlong",
        "fail",
        "--save_every",
        "20",
        "--max_steps",
        "96",
        "--seed",
        "0",
        "--keep_rolling",
        "100",
        "--save_rolling",
        "2",
        "--effort",
        "--budgets",
        "0,64,256",
        "--canary",
        str(H / "runs/pools/canary_long.jsonl"),
        "--canary_n",
        "1000",
        "--canary_every",
        "1",
        "--rollback_on_alarm",
        "--max_rollbacks",
        "3",
        "--ban_restore_check",
        "--ban_stop",
        "0.005",
        "--kl_anchor_coef",
        "0.02",
        "--kl_anchor_ref",
        init,
        "--monitor",
        "--arm",
        "logical",
        "--max_rej",
        "0",
        "--rej_penalty",
        "0.1",
        "--save_steps",
        save_steps,
    ]


DENSE = "grpo_think_vs_mini20_kdense_s0_L2"
P16 = "grpo_think_vs_mini20_kp16_s0_L2"
DENSE_STEPS = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32]  # 4 AdamW steps an iteration: it1 ends at 4
P16_STEPS = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48]


def st_ckpt(run: str, n: int) -> str:
    return str(LOW / run / "ckpts" / f"{run}_st{n}.pt")


def plan() -> list[dict[str, Any]]:
    js: list[dict[str, Any]] = []
    l3 = str(CK / "grpo_think_vs_mb_kkl0_s0_L3_it39.pt")
    # T: timing (the J1 base chunks 0-2; chunk 0 alone on GPU 0, chunks 1 and 2 together on 1)
    base149 = chunk_jobs(
        "J1", "s1g3", BASE[0], "base", "c149", 16, repro0=True, gpus={0: [0], 1: [1], 2: [1]}
    )
    for j in base149[:3]:
        j["block"] = "T"
    js += base149[:3]
    # J2: the low-RL training (GPU 1; a train job runs alone on its GPU)
    for run, iters, steps, prompts, est, to in (
        (DENSE, 8, DENSE_STEPS, 256, 30, 7200),
        (P16, 12, P16_STEPS, 16, 15, 5400),
    ):
        js.append(
            {
                "id": f"train.{run}",
                "block": "J2",
                "kind": "train",
                "gpus": [1],
                "argv": mini20_l2(run, iters, ",".join(map(str, steps)), prompts),
                "out": str(LOW / run),
                "last_ckpt": str(LOW / run / "ckpts" / f"{run}_it{iters}.pt"),
                "needs": [BASE[0]],
                "cls": f"train:{run}",
                "est_min": est,
                "timeout_s": to,
            }
        )
    # J1
    js += base149[3:]
    js += chunk_jobs("J1", "grpo_think_vs_mb_kkl0", l3, "rl", "c149", 16, repro0=True)
    js += chunk_jobs("J1", "s1g3", BASE[0], "base", "t250", 8)
    js += chunk_jobs("J1", "grpo_think_vs_mb_kkl0", l3, "rl", "t250", 8)
    # J3: S1g3 s1, s2 as s1g3_ckreads (CKR_PANEL then MINI_READS, a read in both once)
    rs = {
        "l2test": str(S / "l2test/test_1k.jsonl"),
        "l3test": str(S / "l3test/test_1k.jsonl"),
        "t1320": str(S / "matched/t1320_n1000.jsonl"),
        "t2130": str(S / "matched/t2130_n1000.jsonl"),
        "charles173": C173,
    }
    est3 = {
        ("l2test", 8): 4,
        ("l3test", 8): 5,
        ("t1320", 4): 3,
        ("t2130", 4): 4,
        ("charles173", 16): 1.5,
        ("t1320", 8): 5,
        ("t2130", 8): 7,
        ("charles173", 64): 5,
    }
    panel = [
        ("l2test", 8),
        ("l3test", 8),
        ("t1320", 4),
        ("t2130", 4),
        ("charles173", 16),
        ("t1320", 8),
        ("t2130", 8),
        ("charles173", 64),
    ]
    for s in (1, 2):
        for st, k in panel:
            js.append(
                single(
                    "J3",
                    f"{st}_s1g3_s{s}_plain_k{k}",
                    BASE[s],
                    rs[st],
                    k,
                    est3[(st, k)],
                    f"j3:{st}:{k}",
                )
            )
    tb = ",".join(str(S / f"tb72/{f}.jsonl") for f in ("textbook_dev", "textbook_train"))
    for s in (0, 1, 2):
        js.append(single("J3", f"tb72_s1g3_s{s}_plain_k64", BASE[s], tb, 64, 3, "j3:tb72:64"))
    # J4: the RL axis, most informative first: R ~ 1e-4 (p16 st1), 2e-4 (p16 it1 = st4), 8e-4
    # (p16 it4 = st16), 1.6e-3 (dense st1), mini20 it2 / it4 / it8, L2 end, ki78; then the rest
    tr = {DENSE: [f"train.{DENSE}"], P16: [f"train.{P16}"]}
    first = [
        ("grpo_think_vs_mini20_kp16_L2st1", st_ckpt(P16, 1), "low", tr[P16]),
        ("grpo_think_vs_mini20_kp16_L2st4", st_ckpt(P16, 4), "low", tr[P16]),
        ("grpo_think_vs_mini20_kp16_L2st16", st_ckpt(P16, 16), "low", tr[P16]),
        ("grpo_think_vs_mini20_kdense_L2st1", st_ckpt(DENSE, 1), "low", tr[DENSE]),
        (
            "grpo_think_vs_mini20_L2it2",
            str(CK / "grpo_think_vs_mini20_s0_L2_it2_rolling.pt"),
            "low",
            None,
        ),
        (
            "grpo_think_vs_mini20_L2it4",
            str(CK / "grpo_think_vs_mini20_s0_L2_it4_rolling.pt"),
            "low",
            None,
        ),
        (
            "grpo_think_vs_mini20_L2it8",
            str(CK / "grpo_think_vs_mini20_s0_L2_it8_rolling.pt"),
            "rl",
            None,
        ),
        ("grpo_think_vs_L2", str(CK / "grpo_think_vs_s0_L2_it39.pt"), "rl", None),
        ("grpo_think_vs_mb_ki78", str(CK / "grpo_think_vs_mb_ki78_s0_L3_it78.pt"), "rl", None),
    ]
    rest = [
        ("grpo_think_vs_mini20_kp16_L2st2", st_ckpt(P16, 2), "low", tr[P16]),
        ("grpo_think_vs_mini20_kp16_L2st8", st_ckpt(P16, 8), "low", tr[P16]),
        ("grpo_think_vs_mini20_kp16_L2st32", st_ckpt(P16, 32), "low", tr[P16]),
        ("grpo_think_vs_mini20_kdense_L2st2", st_ckpt(DENSE, 2), "low", tr[DENSE]),
        ("grpo_think_vs_mini20_kdense_L2st4", st_ckpt(DENSE, 4), "low", tr[DENSE]),
        ("grpo_think_vs_mini20_kdense_L2st8", st_ckpt(DENSE, 8), "low", tr[DENSE]),
    ]
    for arm, ck, mc, after in first + rest:
        js += chunk_jobs("J4a", arm, ck, mc, "c149", 16, after=after)
    for arm, ck, mc, after in first + rest:
        js += chunk_jobs("J4b", arm, ck, mc, "t250", 4, after=after)
    return js


def main() -> None:
    dry = "--dry" in sys.argv
    js = plan()
    ids = [j["id"] for j in js]
    assert len(ids) == len(set(ids)), "duplicate ids"
    p = H / "queue/jobs.jsonl"
    if dry:
        for j in js:
            print(json.dumps(j))
        print(f"{len(js)} jobs", file=sys.stderr)
        return
    with open(H / "queue/jobs.lock", "a") as lk:  # the queue's lock (hirl_queue.Locked)
        fcntl.flock(lk, fcntl.LOCK_EX)
        have = set()
        if p.exists():
            have = {
                json.loads(x)["id"]
                for x in p.read_text().splitlines()
                if x.strip() and not x.startswith("#")
            }
        new = [j for j in js if j["id"] not in have]
        with open(p, "a") as fh:
            for j in new:
                fh.write(json.dumps(j) + "\n")
    print(f"{len(new)} jobs added ({len(have)} already there): {p}")


if __name__ == "__main__":
    main()
