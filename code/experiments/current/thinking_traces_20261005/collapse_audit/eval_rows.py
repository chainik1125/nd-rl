"""Collapse audit, probe 1: per-attempt outcomes in the stored 13-20 eval rows (CPU, read-only).

Run from the repo root, with M = code.experiments.current.thinking_traces_20261005.collapse_audit:

    code/.venv/bin/python -m $M.eval_rows

For each checkpoint's rows (986 valid items, `figs.valid_1320`) and budget: pass@1, the outcome mix
per attempt (Lean-accepted / finished but Lean-rejected / env-rejected step by cause / step cap /
truncated action), where failing attempts stop (accepted steps before the failure, relative to the
reference length), tokens and think tokens per attempt, and pass@1 by reference length.
Writes `eval_rows.json` next to this file.
"""

from __future__ import annotations

import collections
import json
from pathlib import Path
from typing import Any

from code.experiments.current.thinking_traces_20261005.matched_eval.figs import M, R, valid_1320

OUT = Path(__file__).with_name("eval_rows.json")
RUNS: dict[str, Path] = {
    "s0_it39": M / "t1320_grpoT1v1e_it39",
    "s0_it49": M / "diag/t1320_grpoT1v1e_it49",
    "s0_it52": M / "diag/t1320_grpoT1v1e_it52",
    "s0_it60": M / "diag/t1320_grpoT1v1e_it60",
    "s0_it70": M / "diag/t1320_grpoT1v1e_it70",
    "s0_it78": M / "t1320_grpoT1v1e_it78",
    **{
        f"s1_it{i}": M / f"curve/t1320_grpoT1v1es1_it{i}"
        for i in (10, 20, 25, 30, 35, 45, 50, 55, 60, 65, 70, 75)
    },
    "s1_it39": M / "t1320_grpoT1v1es1_it39",
    "s1_it78": M / "t1320_grpoT1v1es1_it78",
    "T0s0_it39": M / "t1320_grpoT0_it39",
    "T0s0_it78": M / "t1320_grpoT0_it78",
    "T0s1_it39": M / "t1320_grpoT0s1_it39",
    "T0s1_it78": M / "t1320_grpoT0s1_it78",
}


def ref_lines() -> dict[str, int]:
    out = {}
    with open(R / "pools/test_13_20/test_13_20.jsonl") as f:
        for line in f:
            r = json.loads(line)
            out[r["name"]] = r["reference_lines"]
    return out


def outcome(status: str, ok: int) -> str:
    if ok:
        return "ok"
    if status == "done":
        return "done_lean_rej"
    if status in ("fail:step_cap", "fail:truncated", "fail:context"):
        return status[5:]
    return status  # fail:S:<cause>


def analyse(path: Path, keep: set[str], L: dict[str, int]) -> dict[str, Any]:
    with open(path) as f:
        rows = [json.loads(x) for x in f]
    rows = [r for r in rows if r["name"] in keep]
    n_att = 0
    oc: collections.Counter[str] = collections.Counter()
    p1_by_len: dict[int, list[float]] = collections.defaultdict(list)
    fail_rel: collections.Counter[str] = collections.Counter()  # failing step vs the ref length
    tok = think = steps_done = n_done = 0.0
    fail_steps: list[int] = []
    for r in rows:
        k = r["k"]
        p1_by_len[L[r["name"]]].append(r["n_ok"] / k)
        for j in range(k):
            n_att += 1
            o = outcome(r["status"][j], r["ok"][j])
            oc[o] += 1
            tok += r["tokens"][j]
            think += r["think"][j]
            if r["status"][j] == "done":
                steps_done += r["steps"][j]
                n_done += 1
            fs = r["fail_step"][j]
            if fs is not None and r["status"][j].startswith("fail:S"):
                fail_steps.append(fs)
                ref = L[r["name"]]
                fail_rel["step<=6" if fs <= 6 else "7-12" if fs <= 12 else "13+"] += 1
                fail_rel["past_ref" if fs >= ref else "before_ref"] += 1
    p1 = sum(sum(v) for v in p1_by_len.values()) / len(rows)
    return {
        "items": len(rows),
        "attempts": n_att,
        "pass1": p1,
        "outcomes": {k: v / n_att for k, v in oc.most_common()},
        "fail_step_buckets": {k: v / n_att for k, v in sorted(fail_rel.items())},
        "tokens_per_att": tok / n_att,
        "think_per_att": think / n_att,
        "steps_when_done": steps_done / max(1, n_done),
        "pass1_by_len": {str(L_): sum(v) / len(v) for L_, v in sorted(p1_by_len.items())},
        "n_by_len": {str(L_): len(v) for L_, v in sorted(p1_by_len.items())},
    }


def main() -> int:
    keep = valid_1320()
    L = ref_lines()
    print(f"valid items: {len(keep)}")
    res: dict[str, Any] = {}
    for tag, d in RUNS.items():
        for b in ("B0", "B256"):
            p = d / f"rows_plain_{b}.jsonl"
            if not p.exists():
                continue
            a = analyse(p, keep, L)
            res[f"{tag}_{b}"] = a
            top = ", ".join(f"{k} {100 * v:.1f}" for k, v in list(a["outcomes"].items())[:7])
            print(
                f"{tag:10s} {b:5s} p1 {100 * a['pass1']:5.1f}  tok {a['tokens_per_att']:6.0f} "
                f"think {a['think_per_att']:5.0f} steps(done) {a['steps_when_done']:5.1f} | {top}"
            )
    OUT.write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
