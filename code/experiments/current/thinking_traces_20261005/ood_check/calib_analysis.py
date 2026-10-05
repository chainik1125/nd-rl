"""Calibration eval read-out (OOD_CHECK.md, "Calibration results (2026-10-05)").

    python code/experiments/current/thinking_traces_20261005/ood_check/calib_analysis.py \
        --eval /scratch/cowifornia/nd-rl/think/ood_eval_20261005 [--fig DIR]

Reads the fork guided_eval outputs `<eval>/<ckpt>_<arm>/<set>.{json,rows.jsonl.gz}` (k 64) and
writes `<eval>/calib_summary.json` plus markdown tables on stdout.  Groups: each set; Tier A by
reference-length bin (11-20 / 21-30 / 31-51, `min_lines`) and classical vs not; Tier B per family
and n (parsed from the name, `<family>_n<n>_<i>`).

Per group: pass@1 (mean c/k), pass@8 (unbiased), pass@64 (any accepted), and where the attempts
end: accepted, finished but rejected by Lean, rejected by the environment / checker (plain: the
first rejected action ends the attempt; logical: 10 rejected redraws at one step), step cap,
truncated action.  For failed attempts: median accepted steps before the failure, the share that
fail within their first two own steps (steps <= 1) and the median of steps / reference length.
"""

from __future__ import annotations

import argparse
import collections
import gzip
import json
import math
import re
import statistics
from pathlib import Path

SETS = ["charles_long_reach", "ood_case_product", "ood_and_reorder", "ood_commute", "ood_or_reorder"]
MODELS = {"la_T1_best12_s0_r8": "r8", "stage1_best12_s0_b1200": "stage1"}
ARMS = ["plain", "logical"]
TIER_A_BINS = [("11-20", 11, 20), ("21-30", 21, 30), ("31-51", 31, 51)]


def pass_at_k(n, c, k):
    if n - c < k:
        return 1.0
    return 1.0 - math.prod((n - c - i) / (n - i) for i in range(k))


def load_rows(path):
    with gzip.open(path, "rt") as f:
        return [json.loads(x) for x in f if x.strip()]


def outcome(status, ok):
    if ok:
        return "accepted"
    if status == "done":
        return "lean_rejected"
    if status == "fail:step_cap":
        return "step_cap"
    if status == "fail:truncated" or status.endswith("truncated"):
        return "truncated"
    if status == "fail:max_rej":
        return "max_rej"
    return "env_rejected"


def summarize(rows, ref_of):
    n = len(rows)
    k = rows[0]["k"]
    out = {
        "n": n,
        "k": k,
        "pass1": 100 * sum(r["n_ok"] / r["k"] for r in rows) / n,
        "pass8": 100 * sum(pass_at_k(r["k"], r["n_ok"], 8) for r in rows) / n,
        "pass64": 100 * sum(r["n_ok"] > 0 for r in rows) / n,
        "solved": sum(r["n_ok"] > 0 for r in rows),
    }
    oc = collections.Counter()
    fail_steps, rel, early = [], [], 0
    causes = collections.Counter()
    for r in rows:
        ref = ref_of(r)
        for st, ok, steps in zip(r["status"], r["ok"], r["steps"]):
            o = outcome(st, ok)
            oc[o] += 1
            if o != "accepted":
                fail_steps.append(steps)
                early += steps <= 1
                if ref:
                    rel.append(steps / ref)
            if st.startswith("fail:S:"):
                causes[st[len("fail:S:") :]] += 1
    att = sum(oc.values())
    out["outcomes_pct"] = {o: 100 * v / att for o, v in oc.most_common()}
    out["fail_median_steps"] = statistics.median(fail_steps) if fail_steps else None
    out["fail_early_pct"] = 100 * early / len(fail_steps) if fail_steps else None
    out["fail_median_rel"] = statistics.median(rel) if rel else None
    out["top_env_causes"] = causes.most_common(4)
    return out


def logical_causes(rows):
    """Guided arms: every rejected draw by cause ("S:" environment, "L:" step checker).  The rows'
    `causes` lists are per attempt with at least one rejection, so they do not align with `status`;
    this is the mix over all redraws, not the cause that ended an attempt."""
    c = collections.Counter(x for r in rows for cs in (r.get("causes") or []) for x in cs)
    return c.most_common(4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", default="/scratch/cowifornia/nd-rl/think/ood_eval_20261005")
    ap.add_argument("--fig", default=None, help="write fig_ood_calibration.{pdf,svg,png} here")
    a = ap.parse_args()
    E = Path(a.eval)
    meta = {}
    for s in SETS:
        for line in open(E / "sets" / f"{s}.jsonl"):
            r = json.loads(line)
            meta[(s, r["name"])] = r
    res = {}
    for ck, model in MODELS.items():
        for arm in ARMS:
            d = E / f"{ck}_{arm}"
            for s in SETS:
                p = d / f"{s}.rows.jsonl.gz"
                if not p.exists():
                    continue
                rows = load_rows(p)
                summ = json.loads((d / f"{s}.json").read_text())

                def ref_of(r, s=s):
                    m = meta[(s, r["name"])]
                    return m.get("min_lines") or m.get("reference_lines")

                groups = {s: rows}
                if s == "charles_long_reach":
                    for b, lo, hi in TIER_A_BINS:
                        groups[f"{s}|{b}"] = [r for r in rows if lo <= ref_of(r) <= hi]
                    groups[f"{s}|classical"] = [
                        r for r in rows if meta[(s, r["name"])]["classical_only"]
                    ]
                    groups[f"{s}|intuitionistic"] = [
                        r for r in rows if not meta[(s, r["name"])]["classical_only"]
                    ]
                elif s != "ood_case_product":
                    ns = sorted({int(re.search(r"_n(\d+)_", r["name"]).group(1)) for r in rows})
                    for n in ns:
                        groups[f"{s}|n{n}"] = [r for r in rows if f"_n{n}_" in r["name"]]
                for g, rs in groups.items():
                    if rs:
                        x = summarize(rs, ref_of)
                        if arm == "logical":
                            x["logical_last_causes"] = logical_causes(rs)
                        res[f"{model}|{arm}|{g}"] = x
                res[f"{model}|{arm}|{s}|_meta"] = {
                    "wall_s": summ["wall_s"],
                    "gpu_s": summ["stats"].get("gpu_s"),
                    "lean_s": summ["stats"].get("lean_s"),
                    "finished": summ["finished"],
                    "accepted": summ["accepted"],
                    "attempts": summ["attempts"],
                    "sampled_tokens": summ["stats"].get("sampled_tokens"),
                    "redraws": summ.get("n_redraws"),
                }
    (E / "calib_summary.json").write_text(json.dumps(res, indent=1))
    print_tables(res)
    if a.fig:
        figure(res, Path(a.fig))


def fmt(v, d=1):
    return "–" if v is None else f"{v:.{d}f}"


def print_tables(res):
    keys = sorted({k.split("|", 2)[2] for k in res if not k.endswith("_meta")}, key=_order)
    print("| set / group | n | " + " | ".join(
        f"{m} {a} p@1 / p@64" for m in ("stage1", "r8") for a in ARMS) + " |")
    print("|---|---|" + "---|" * 4)
    for g in keys:
        cells = []
        n = None
        for m in ("stage1", "r8"):
            for a in ARMS:
                x = res.get(f"{m}|{a}|{g}")
                if x:
                    n = x["n"]
                    cells.append(f"{fmt(x['pass1'])} / {fmt(x['pass64'])}")
                else:
                    cells.append("–")
        print(f"| {g} | {n} | " + " | ".join(cells) + " |")
    print()
    print("| model arm set | accepted | Lean-rejected | env-rejected | max_rej | step cap | truncated "
          "| fail median steps | fail at steps<=1 | fail median steps/ref | top causes |")
    print("|---|" + "---|" * 10)
    for m in ("stage1", "r8"):
        for a in ARMS:
            for s in SETS:
                x = res.get(f"{m}|{a}|{s}")
                if not x:
                    continue
                o = x["outcomes_pct"]
                tc = x.get("logical_last_causes") if a == "logical" else x["top_env_causes"]
                print(
                    f"| {m} {a} {s} | {fmt(o.get('accepted', 0))} | {fmt(o.get('lean_rejected', 0))} "
                    f"| {fmt(o.get('env_rejected', 0))} | {fmt(o.get('max_rej', 0))} "
                    f"| {fmt(o.get('step_cap', 0))} | {fmt(o.get('truncated', 0))} "
                    f"| {fmt(x['fail_median_steps'], 0)} | {fmt(x['fail_early_pct'])} "
                    f"| {fmt(x['fail_median_rel'], 2)} | "
                    + ", ".join(f"{c} {v}" for c, v in (tc or [])) + " |"
                )
    print()
    tot = collections.defaultdict(float)
    for k, v in res.items():
        if k.endswith("_meta"):
            m, a = k.split("|")[:2]
            tot[f"{m} {a}"] += v["wall_s"]
    print("wall minutes:", {k: round(v / 60, 1) for k, v in tot.items()})


def _order(g):
    s, _, sub = g.partition("|")
    i = SETS.index(s) if s in SETS else 99
    if sub.startswith("n"):
        return (i, 1, int(sub[1:]), sub)
    return (i, 0 if not sub else 2, 0, sub)


def figure(res, out_dir):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[5]))
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter

    from code.ndbench.figures import style as S

    fams = ["ood_or_reorder", "ood_and_reorder", "ood_commute"]
    combos = [("stage1", "plain"), ("stage1", "logical"), ("r8", "plain"), ("r8", "logical")]
    col = {"stage1": S.OKABE_ITO["blue"], "r8": S.OKABE_ITO["vermillion"]}
    mk = {"stage1": "o", "r8": "s"}
    ls = {"plain": "-", "logical": (0, (4, 1.6))}
    with S.paper_style():
        fig = S.new_figure(S.FULL_W, 2.2)
        axs = fig.subplots(1, 3, sharey=True)
        for i, (ax, f) in enumerate(zip(axs, fams)):
            ax.axhspan(0.02, 0.40, color=S.SHADE, zorder=0, linewidth=0)
            for m, a in combos:
                pts = []
                for n in range(3, 9):
                    x = res.get(f"{m}|{a}|{f}|n{n}")
                    if x:
                        pts.append((n, x["pass1"] / 100))
                if pts:
                    ax.plot([p[0] for p in pts], [p[1] for p in pts], color=col[m], linestyle=ls[a],
                            linewidth=1.2, marker=mk[m], markersize=3.6,
                            markerfacecolor=col[m] if a == "plain" else S.WHITE,
                            markeredgecolor=S.WHITE if a == "plain" else col[m],
                            markeredgewidth=0.5 if a == "plain" else 0.9, zorder=3)
            ax.set_yscale("symlog", linthresh=0.01)
            ax.set_ylim(0, 1.0)
            ax.yaxis.set_major_locator(FixedLocator([0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 1.0]))
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{100 * v:g}%"))
            ax.yaxis.set_minor_formatter(NullFormatter())
            ax.set_xticks(range(3, 9))
            ax.set_xlabel("n (family size knob)")
            if i == 0:
                ax.set_ylabel("pass@1 (k = 64 attempts)")
            S.light_grid(ax)
            S.panel_title(ax, "abc"[i], f"{f.replace('ood_', '')}, 40 per n")
        handles = [Line2D([], [], color=col[m], linestyle=ls[a], marker=mk[m], markersize=3.6,
                          markerfacecolor=col[m] if a == "plain" else S.WHITE,
                          markeredgecolor=S.WHITE if a == "plain" else col[m],
                          label=f"{'Pretraining only' if m == 'stage1' else 'Round 8'}, {a}")
                   for m, a in combos]
        handles.append(Patch(facecolor=S.SHADE, edgecolor="none", label="2–40% pass@1"))
        fig.legend(handles=handles, loc="outside upper center", ncol=5, frameon=False,
                   handlelength=2.0, columnspacing=1.0)
        out_dir.mkdir(parents=True, exist_ok=True)
        for p in S.save(fig, out_dir / "fig_ood_calibration"):
            print(p)
        S.close(fig)


if __name__ == "__main__":
    main()
