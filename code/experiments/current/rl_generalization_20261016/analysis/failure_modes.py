"""How attempts fail on the long sets (O6, O8) and how redundant solved proofs are (B2's numbers).

    python3 failure_modes.py > out/failure_modes.txt

1. O6, outcome shares: per read, the share of attempts that were solved, env-rejected (`fail:S:<cause>`, top
   causes listed) or env-accepted but Lean-rejected (status `done`, ok 0); the median steps / min_lines of solved
   attempts and the median fail_step / min_lines of failed ones.
2. O6, Lean-rejection classes: the `filter` field of each read's `lean_gate.leanrej.jsonl` (the Lean prefilter's
   class of the rejection: app-arg, app-result, ...).
3. O8, verifier in training vs Lean rejections at test: ok / env-rejected / Lean-rejected shares for plain GRPO,
   GRPO + verifier (stop), EI and EI + verifier, seeds 0-2, on t1320 and t2130.
4. B2, redundancy of solved proofs: the share of `have` lines whose formula was already stated anywhere earlier in
   the proof text (scope ignored, so this counts re-derivation in another branch too), and the share of atomic
   `have` lines that repeat the formula of the line just before (a same-scope stutter proxy).

Reads are `runs/eval/v2/<set>_<arm>_s<seed>_plain_k<k>/rows_plain_B64.jsonl`; one read per row, no intervals.
Standard library only; CPU, one process.
"""

from __future__ import annotations

import collections
import json
import os
import re

EV = "/scratch/cowifornia/nd-rl/think/runs/eval/v2"
O6_READS = [
    "t1320_grpo_think_vs_mb_s0_plain_k16",
    "t1320_grpo_think_vs_mb_kkl0_s0_plain_k16",
    "t1320_ei_think_v_s0_plain_k16",
    "t2130_grpo_think_vs_mb_s0_plain_k8",
    "t2130_grpo_think_vs_mb_kkl0_s0_plain_k8",
    "t2130_ei_think_v_s0_plain_k8",
]
LEAN_READS = [
    "t1320_grpo_think_vs_mb_kkl0_s0_plain_k16",
    "t2130_grpo_think_vs_mb_kkl0_s0_plain_k8",
    "t1320_ei_think_v_s0_plain_k16",
]
O8_ARMS = ["grpo_think", "grpo_think_vs", "ei_think", "ei_think_v"]
REDUNDANCY_READS = [
    "t1320_grpo_think_vs_mb_kkl0_s0_plain_k16",
    "t1320_ei_think_v_s0_plain_k16",
    "t2130_grpo_think_vs_mb_kkl0_s0_plain_k8",
    "charles173_grpo_think_vs_mb_kkl0_s0_plain_k64",
]
HAVE = re.compile(r"have n\d+ : (.*?) :=")
HAVE_FULL = re.compile(r"have n\d+ : (.*) := (.*)$")


def rows(d: str) -> list[dict]:
    path = f"{EV}/{d}/rows_plain_B64.jsonl"
    if not os.path.exists(path):
        return []
    return [json.loads(line) for line in open(path)]


def median(v: list[float]) -> float:
    v = sorted(v)
    return v[len(v) // 2] if v else float("nan")


def o6() -> None:
    print("## O6: outcome shares (% of attempts), solved steps / min_lines, fail_step / min_lines\n")
    for d in O6_READS:
        rs = rows(d)
        if not rs:
            print(f"{d}: missing")
            continue
        c: collections.Counter[str] = collections.Counter()
        ex: list[float] = []
        fs: list[float] = []
        n = 0
        for r in rs:
            m = r["min_lines"]
            for ok, st, steps, fstep in zip(r["ok"], r["status"], r["steps"], r["fail_step"]):
                n += 1
                if ok:
                    c["ok"] += 1
                    ex.append(steps / m)
                elif st == "done":
                    c["done, Lean-rejected"] += 1
                else:
                    c[st] += 1
                    if fstep is not None:
                        fs.append(fstep / m)
        top = ", ".join(f"{k} {100 * v / n:.1f}" for k, v in c.most_common(8))
        print(f"{d}\n   {top}")
        print(f"   solved steps/min_lines median {median(ex):.2f}; fail_step/min_lines median {median(fs):.2f}")
    print()


def lean_classes() -> None:
    print("## O6: Lean-rejection classes (lean_gate.leanrej.jsonl `filter`)\n")
    for d in LEAN_READS:
        path = f"{EV}/{d}/lean_gate.leanrej.jsonl"
        if not os.path.exists(path):
            print(f"{d}: missing")
            continue
        c = collections.Counter(json.loads(line).get("filter") for line in open(path))
        n = sum(c.values())
        print(f"{d}: {n} rejections; " + ", ".join(f"{k} {100 * v / n:.0f}%" for k, v in c.most_common(8)))
    print()


def o8() -> None:
    print("## O8: ok / env-rejected / Lean-rejected (% of attempts) by training arm and seed\n")
    for arm in O8_ARMS:
        for s in (0, 1, 2):
            for st, k in (("t1320", 16), ("t2130", 8)):
                d = f"{st}_{arm}_s{s}_plain_k{k}"
                rs = rows(d)
                if not rs:
                    continue
                c: collections.Counter[str] = collections.Counter()
                n = 0
                for r in rs:
                    for ok, stt in zip(r["ok"], r["status"]):
                        n += 1
                        c["ok" if ok else ("lean" if stt == "done" else "env")] += 1
                print(
                    f"{d:36s} ok {100 * c['ok'] / n:5.1f}  env-rejected {100 * c['env'] / n:5.1f}"
                    f"  Lean-rejected {100 * c['lean'] / n:5.1f}"
                )
    print()


def redundancy() -> None:
    print("## B2: formula repeats in solved proofs\n")
    for d in REDUNDANCY_READS:
        rs = rows(d)
        if not rs:
            print(f"{d}: missing")
            continue
        tot = dup = atomic = adj = 0
        for r in rs:
            for text, cnt in r["accepted"].items():
                seen: collections.Counter[str] = collections.Counter()
                for f in HAVE.findall(text):
                    if seen[f]:
                        dup += cnt
                    seen[f] += 1
                    tot += cnt
                prev = None
                for part in (p.strip() for p in text.split(" ; ")):
                    m = HAVE_FULL.match(part)
                    cur = None
                    if m and not m.group(2).startswith(("(", "Or.elim")):  # atomic terms only, no boxes
                        cur = m.group(1)
                        atomic += cnt
                        if prev is not None and cur == prev:
                            adj += cnt
                    prev = cur
        print(
            f"{d:48s} have-lines {tot}: repeat of an earlier formula {100 * dup / tot:.1f}%;"
            f" atomic lines {atomic}: same formula as the line before {100 * adj / atomic:.1f}%"
        )
    print()


def main() -> int:
    o6()
    lean_classes()
    o8()
    redundancy()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
