# ruff: noqa: E501, SIM115
"""Stage 2 arms per checkpoint -> /scratch/cowifornia/nd-rl/olmo_elicit/arms.csv (checker passes only; no genuine /
gamed reading in this pass).

Arms (one row per checkpoint and arm):
  T     his transplant without instruction: the final model's thinking pasted in, answer sampled (41 prompts, 115
        traces x 8). dpo / s2300: his olmo-transplant R_dpo (cond T1) / R_s2300 (T2), cut full. Else ours: runs/<c>/T.
  A1    the same + "reply with only your final answer" (olmo-answeronly A1). dpo / s2300: his olmo-answeronly R jobs.
  H     "Answer in at most 60 words." on the 48 reach prompts. dpo / s2300: his olmo-iffu T2 + T2c (64 / 16 draws per
        prompt). Else ours: runs/<c>/H (8 or 4 draws).
  B     plain prompt on the 48 reach prompts (his olmo-ifcurve arm B, first 32 base draws; dpo: his olmo-ifomega
        F_dpo_ifb / F_dpo_ife draws on these prompts, 16 per prompt), for the hint's gain.
  B8    the same cells cut to their first 8 draws (draw order): the like-for-like 8-draw comparison for L and H.
  L     the 48 reach prompts, plain prompts (as his arm B sent them), sampled through the checkpoint's stage-1 adapter
        (runs/<c>/edl/adapters/best_order0 = theta_64 of the shortest-code-length order-0 EDL row) on vLLM's LoRA
        path, 8 draws per prompt (ours at every checkpoint: runs/<c>/L). The adapter was trained on 64 arm-A
        non-reach prompts' final-model responses, so L measures transfer to the reach prompts.
Columns: strict_pct = mean over prompts of the prompt's strict pass share (all its draws), lo95 / hi95 = prompt
bootstrap (10,000 resamples, seed 0); solved_pct = prompts with >= 1 strict pass among their first `solved_n` draws (in
draw order), as % of the set (solved_n = the draws per prompt of our own cells, so stored cells with more draws are
compared at equal tries); solved_all_pct = the same over all the cell's draws; ans_words_median over draws with an
answer; n_prompts, n_draws, source; grade = "strict checker pass" (no genuine / gamed reading in this pass).
Usage: python analyze_arms.py [--runs /scratch/cowifornia/nd-rl/olmo_elicit/runs] [--out .../arms.csv] [--n-h 8]
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import statistics as st
import zlib
from pathlib import Path

import dan_curve as dc
import numpy as np

D = dc.DAN / "artifacts"
OE = Path("/scratch/cowifornia/nd-rl/olmo_elicit")
T_PROMPTS = 41


def jl(p: Path) -> list[dict]:
    out = []
    op = gzip.open if str(p).endswith(".gz") else open
    try:
        with op(p, "rt") as f:
            for ln in f:
                try:
                    out.append(json.loads(ln))
                except json.JSONDecodeError:
                    break
    except (EOFError, OSError, zlib.error):
        pass
    return out


def answer_words(text: str) -> int | None:
    return len(text.rsplit("</think>", 1)[1].split()) if "</think>" in text else None


def summarize(
    cells: dict[str, list[tuple[int, int | None]]], solved_n: int, n_set: int, reps: int = 10000
) -> dict:
    """cells: pid -> [(strict, answer words or None), ...] in draw order."""
    pids = sorted(cells)
    rate = np.array([np.mean([s for s, _ in cells[p]]) for p in pids])
    lo, hi = dc.boot_mean_ci(rate, reps=reps, seed=0) if len(pids) else (float("nan"), float("nan"))
    words = [w for p in pids for _, w in cells[p] if w is not None]
    return {
        "n_prompts": len(pids),
        "n_draws": sum(len(v) for v in cells.values()),
        "strict_pct": round(100 * float(rate.mean()), 2) if len(pids) else None,
        "lo95": round(100 * lo, 2),
        "hi95": round(100 * hi, 2),
        "solved_pct": round(
            100 * sum(any(s for s, _ in cells[p][:solved_n]) for p in pids) / n_set, 2
        ),
        "solved_n": solved_n,
        "solved_all_pct": round(100 * sum(any(s for s, _ in cells[p]) for p in pids) / n_set, 2),
        "ans_words_median": st.median(words) if words else None,
    }


def t_cells(graded: Path, conds: set[str]) -> dict[str, list[tuple[int, int | None]]]:
    rows = {}
    for r in jl(graded):
        if r.get("cond") in conds and str(r.get("cut", "full")) == "full":
            rows[(r["rid"], int(r["i"]))] = r  # a resumed job may repeat a row: keep the last
    cells: dict[str, list] = {}
    for (_rid, _i), r in sorted(rows.items()):
        cells.setdefault(r["pid"], []).append((int(bool(r["strict"])), r.get("answer_words")))
    return cells


def h_cells(
    pairs: list[tuple[Path, Path]], pids: list[str]
) -> dict[str, list[tuple[int, int | None]]]:
    """Sampler + grade.py outputs: join on (pid, chunk, i); draw order = (job order, chunk, i)."""
    cells: dict[str, list] = {}
    for sp, gp in pairs:
        words = {(r["pid"], int(r["chunk"]), int(r["i"])): answer_words(r["text"]) for r in jl(sp)}
        g = sorted(
            ((r["pid"], int(r["chunk"]), int(r["i"])), int(bool(r["strict"]))) for r in jl(gp)
        )
        for k, s in g:
            if k[0] in pids and k in words:
                cells.setdefault(k[0], []).append((s, words[k]))
    return cells


def b_cells(ckpt: str, reach: list[str]) -> dict[str, list[tuple[int, int | None]]]:
    R = dc.Rows(D / "olmo_ifcurve" / "rows")
    if ckpt != "dpo":
        return {
            p: [
                (x["strict"], x["ans_words"] if x["think_end"] else None)
                for x in R.draws(ckpt, p, "B")
            ]
            for p in reach
        }
    pairs = [
        (D / "olmo_ifomega/samples/F_dpo_ifb.jsonl.gz", D / "olmo_ifomega/graded/F_dpo_ifb.jsonl"),
        (D / "olmo_ifomega/samples/F_dpo_ife.jsonl.gz", D / "olmo_ifomega/graded/F_dpo_ife.jsonl"),
    ]
    return {p: v[:16] for p, v in h_cells(pairs, reach).items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(OE / "runs"))
    ap.add_argument("--out", default=str(OE / "arms.csv"))
    ap.add_argument("--n-h", type=int, default=8)
    ap.add_argument("--reps", type=int, default=10000)
    a = ap.parse_args()
    runs = Path(a.runs)
    reach = dc.reach()
    rows = []

    def add(ckpt: str, arm: str, source: str, cells: dict, solved_n: int, n_set: int) -> None:
        if not cells:
            return
        rows.append(
            {
                "ckpt": ckpt,
                "step": dc.STEP[ckpt],
                "arm": arm,
                "source": source,
                **summarize(cells, solved_n, n_set, a.reps),
                "grade": "strict checker pass (no genuine / gamed reading)",
            }
        )

    for c in dc.CKPTS:
        # T and A1
        if c in ("dpo", "s2300"):
            w, tc = ("dpo", "T1") if c == "dpo" else ("s2300", "T2")
            add(
                c,
                "T",
                f"dan:olmo-transplant R_{w} {tc}",
                t_cells(D / f"olmo_transplant/graded/R_{w}.jsonl", {tc}),
                8,
                T_PROMPTS,
            )
            add(
                c,
                "A1",
                f"dan:olmo-answeronly R_{w} A1",
                t_cells(D / f"olmo_answeronly/graded/R_{w}.jsonl", {"A1"}),
                8,
                T_PROMPTS,
            )
        else:
            g = runs / c / "T" / "graded.jsonl"
            add(c, "T", "ours", t_cells(g, {"T"}), 8, T_PROMPTS)
            add(c, "A1", "ours", t_cells(g, {"A1"}), 8, T_PROMPTS)
        # H
        if c in ("dpo", "s2300"):
            jobs = [f"T2_{c}", f"T2c_{c}"]
            pairs = [
                (D / f"olmo_iffu/samples/{j}.jsonl.gz", D / f"olmo_iffu/graded/{j}.jsonl")
                for j in jobs
            ]
            add(
                c,
                "H",
                "dan:olmo-iffu " + " + ".join(jobs),
                h_cells(pairs, reach),
                a.n_h,
                len(reach),
            )
        else:
            add(
                c,
                "H",
                "ours",
                h_cells(
                    [(runs / c / "H" / "samples.jsonl.gz", runs / c / "H" / "graded.jsonl")], reach
                ),
                a.n_h,
                len(reach),
            )
        # plain B (all stored base draws) and B8 (their first 8)
        bc = b_cells(c, reach)
        src_b = "dan:olmo-ifcurve arm B" if c != "dpo" else "dan:olmo-ifomega F_dpo_ifb/ife (16)"
        add(c, "B", src_b, bc, a.n_h, len(reach))
        add(c, "B8", src_b + ", first 8", {p: v[:8] for p, v in bc.items()}, 8, len(reach))
        # L: through the stage-1 adapter
        lsrc = "ours"
        sj = runs / c / "L" / "adapter_source.json"
        if sj.exists():
            row = json.loads(sj.read_text())["row"]
            lsrc = f"ours (adapter lr {row['lr']:g}, order {row['order']})"
        add(
            c,
            "L",
            lsrc,
            h_cells(
                [(runs / c / "L" / "samples.jsonl.gz", runs / c / "L" / "graded.jsonl")], reach
            ),
            8,
            len(reach),
        )
    keys = [
        "ckpt",
        "step",
        "arm",
        "source",
        "n_prompts",
        "n_draws",
        "strict_pct",
        "lo95",
        "hi95",
        "solved_pct",
        "solved_n",
        "solved_all_pct",
        "ans_words_median",
        "grade",
    ]
    with open(a.out, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=keys)
        wr.writeheader()
        wr.writerows(rows)
    print(f"wrote {a.out}: {len(rows)} rows")
    for r in rows:
        print(
            f"{r['ckpt']:6} {r['arm']:3} {r['strict_pct']:6} [{r['lo95']}, {r['hi95']}] solved {r['solved_pct']}% "
            f"(first {r['solved_n']}) words {r['ans_words_median']} n={r['n_prompts']}/{r['n_draws']}  {r['source']}"
        )


if __name__ == "__main__":
    main()
