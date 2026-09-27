"""Analyze exp-00c: frozen vs trained per problem, for GRPO at three lrs and expert iteration.

    cd code && uv run python ../experiments/ws3/exp-00c-lr-ei-cpu/analyze.py [--exp-dir DIR]

Reads, per run folder `run_<name>/`:
    baseline/dumps/<part>.jsonl[.gz]   frozen model (falls back to <exp>/baseline/dumps)
    evaluate/dumps/<part>.jsonl[.gz]   trained model
    metrics.jsonl                      training dynamics
for parts long_pool, greedy_heldout, textbook72. Writes analysis.json, analysis.md and
analysis_len7.png into the exp folder, then prints the pre-registered selection rule's pick.

Selection rule (pre-registered): among the three GRPO runs, pick the lr with the highest mean
paired difference on long_pool length-7 problems the frozen model solves at least once, excluding
any run whose greedy_heldout avg drops by more than 0.02 versus frozen or whose final KL exceeds
1.0. textbook72 is never used in the selection.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
WS3 = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(HERE))), "code", "ws3")
if WS3 not in sys.path:
    sys.path.insert(0, WS3)
import ndpath  # noqa: E402,F401

# isort: split
from eval.metrics import mean_ci, per_problem, read_dump  # noqa: E402

RUNS = [  # (folder, kind, lr)
    ("run_lr3e-5", "grpo", 3e-5),
    ("run_lr1e-4", "grpo", 1e-4),
    ("run_lr3e-4", "grpo", 3e-4),
    ("run_ei", "ei", None),
]
PARTS = ("long_pool", "greedy_heldout", "textbook72")
HEADLINE_LEN = 7
NEW_CAP_LENS = (7, 8)
GREEDY_MAX_DROP = 0.02
MAX_FINAL_KL = 1.0
BOOT_SEED = 0
BOOT_N = 10_000


# ------------------------------------------------------------------ loading


def find_dump(dump_dir: str, part: str):
    for ext in (".jsonl.gz", ".jsonl"):
        p = os.path.join(dump_dir, part + ext)
        if os.path.exists(p):
            return p
    return None


def load_part(dump_dir: str, part: str):
    """prompt -> {'n','c','p','length','name'}, or None if the dump is missing."""
    p = find_dump(dump_dir, part)
    return None if p is None else per_problem(read_dump(p))


def pair(base: dict, trained: dict, where: str = "") -> list[dict]:
    """Per-problem frozen vs trained rows, keyed by prompt. Both sides must cover the same set."""
    if set(base) != set(trained):
        only_b, only_t = len(set(base) - set(trained)), len(set(trained) - set(base))
        raise AssertionError(f"{where}: prompt sets differ ({only_b} frozen-only, "
                             f"{only_t} trained-only)")
    out = []
    for prompt, b in base.items():
        t = trained[prompt]
        L = b.get("length")
        out.append({"prompt": prompt, "name": b.get("name"),
                    "length": -1 if L is None else int(L),
                    "base_c": b["c"], "base_n": b["n"], "base_p": b["p"],
                    "rl_c": t["c"], "rl_n": t["n"], "rl_p": t["p"]})
    return out


def read_metrics(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


# ------------------------------------------------------------------ statistics


def sign_test(up: int, down: int) -> float:
    """Two-sided exact sign test (ties already dropped), binomial p = 0.5."""
    n = up + down
    if n == 0:
        return 1.0
    k = min(up, down)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def bootstrap_ci(xs, n_boot: int = BOOT_N, seed: int = BOOT_SEED):
    """Percentile 95% bootstrap CI of the mean over items (problems)."""
    if not xs:
        return None
    rng = random.Random(seed)
    n = len(xs)
    means = sorted(sum(xs[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    return [means[int(0.025 * n_boot)], means[min(n_boot - 1, int(0.975 * n_boot))]]


def _mean(xs):
    return sum(xs) / len(xs) if xs else None


# ------------------------------------------------------------------ per-run analysis


def headline(rows: list[dict], L: int = HEADLINE_LEN) -> dict:
    at = [r for r in rows if r["length"] == L]
    solved = [r for r in at if r["base_c"] > 0]
    d_solved = [r["rl_p"] - r["base_p"] for r in solved]
    up = sum(d > 0 for d in d_solved)
    down = sum(d < 0 for d in d_solved)
    same = len(d_solved) - up - down
    d_all = [r["rl_p"] - r["base_p"] for r in at]
    return {
        "length": L,
        "n_problems": len(at),
        "n_frozen_solved": len(solved),
        "up": up, "down": down, "unchanged": same,
        "sign_p": sign_test(up, down),
        "mean_diff_solved": _mean(d_solved),
        "mean_diff_solved_ci": bootstrap_ci(d_solved),
        "mean_diff_all": _mean(d_all),
        "mean_diff_all_ci": bootstrap_ci(d_all),
    }


def new_capability(rows: list[dict], lengths=NEW_CAP_LENS) -> dict:
    out = {}
    for L in lengths:
        at = [r for r in rows if r["length"] == L]
        zero = [r for r in at if r["base_c"] == 0]
        out[str(L)] = {"n_problems": len(at), "n_frozen_zero": len(zero),
                       "new_capability": sum(r["rl_p"] >= 0.5 for r in zero),
                       "new_any": sum(r["rl_c"] > 0 for r in zero)}
    return out


def avg_block(rows: list[dict]) -> dict:
    fm, fci = mean_ci([r["base_p"] for r in rows])
    tm, tci = mean_ci([r["rl_p"] for r in rows])
    return {"n": len(rows), "frozen": fm, "frozen_ci": fci, "trained": tm, "trained_ci": tci,
            "delta": tm - fm}


def by_length(rows: list[dict]) -> dict:
    lens = sorted({r["length"] for r in rows})
    return {str(L): avg_block([r for r in rows if r["length"] == L]) for L in lens}


def secondary(paired: dict) -> dict:
    out = {}
    if "long_pool" in paired:
        out["long_pool_by_length"] = by_length(paired["long_pool"])
    if "textbook72" in paired:
        out["textbook72"] = avg_block(paired["textbook72"])
    if "greedy_heldout" in paired:
        g = paired["greedy_heldout"]
        out["greedy_overall"] = avg_block(g)
        out["greedy_by_length"] = by_length(g)
        out["greedy_lost"] = sum(r["base_c"] > 0 and r["rl_c"] == 0 for r in g)
        out["greedy_gained"] = sum(r["base_c"] == 0 and r["rl_c"] > 0 for r in g)
    return out


def dynamics(lines: list[dict], kind: str) -> dict:
    tot = [x["rollout/total_completions"] for x in lines if "rollout/total_completions" in x]
    rew = [x["reward"] for x in lines if "reward" in x]
    out = {"n_lines": len(lines), "total_completions": tot[-1] if tot else None,
           "reward_first5": _mean(rew[:5]), "reward_last5": _mean(rew[-5:])}
    if kind == "ei":
        rounds = {}
        for x in lines:
            if "round" not in x:
                continue
            r = rounds.setdefault(int(x["round"]), {"round": int(x["round"])})
            for k in ("reward", "sft/loss_last"):
                if k in x:
                    r[k] = x[k]
        out["rounds"] = [rounds[k] for k in sorted(rounds)]
        return out
    kl = [x["kl"] for x in lines if "kl" in x]
    ent = [x["entropy"] for x in lines if "entropy" in x]
    fm = [x["groups/frac_mixed"] for x in lines if "groups/frac_mixed" in x]
    out.update({"final_kl": kl[-1] if kl else None, "max_kl": max(kl) if kl else None,
                "entropy_first5": _mean(ent[:5]), "entropy_last5": _mean(ent[-5:]),
                "frac_mixed_mean": _mean(fm)})
    return out


def analyze_run(exp_dir: str, folder: str, kind: str, lr) -> dict | None:
    run = os.path.join(exp_dir, folder)
    ev = os.path.join(run, "evaluate", "dumps")
    if not any(find_dump(ev, p) for p in PARTS):
        return None
    base_dir = os.path.join(run, "baseline", "dumps")
    if not any(find_dump(base_dir, p) for p in PARTS):
        base_dir = os.path.join(exp_dir, "baseline", "dumps")
    paired, notes = {}, []
    for part in PARTS:
        b, t = load_part(base_dir, part), load_part(ev, part)
        if b is None or t is None:
            notes.append(f"{part}: missing {'frozen' if b is None else 'trained'} dump")
            continue
        paired[part] = pair(b, t, f"{folder}/{part}")
    res = {"folder": folder, "kind": kind, "lr": lr, "notes": notes,
           "dynamics": dynamics(read_metrics(os.path.join(run, "metrics.jsonl")), kind)}
    if "long_pool" in paired:
        res["headline"] = headline(paired["long_pool"])
        res["new_capability"] = new_capability(paired["long_pool"])
        res["len7_scatter"] = [[r["base_p"], r["rl_p"]] for r in paired["long_pool"]
                               if r["length"] == HEADLINE_LEN]
    res["secondary"] = secondary(paired)
    return res


# ------------------------------------------------------------------ selection rule


def select_lr(results: list[dict]) -> dict:
    """Pre-registered pick among GRPO runs. Never looks at textbook72."""
    considered, excluded = [], []
    for r in results:
        if r["kind"] != "grpo":
            continue
        why = []
        md = r.get("headline", {}).get("mean_diff_solved")
        if md is None:
            why.append("no length-7 problems solved by frozen (or long_pool missing)")
        g = r["secondary"].get("greedy_overall")
        if g is None:
            why.append("greedy_heldout missing")
        elif g["delta"] < -GREEDY_MAX_DROP:
            why.append(f"greedy_heldout avg drop {g['delta']:+.4f} < -{GREEDY_MAX_DROP}")
        kl = r["dynamics"].get("final_kl")
        if kl is None:
            why.append("final KL unknown")
        elif kl > MAX_FINAL_KL:
            why.append(f"final KL {kl:.3f} > {MAX_FINAL_KL}")
        if why:
            excluded.append({"lr": r["lr"], "folder": r["folder"], "reasons": why})
        else:
            considered.append({"lr": r["lr"], "folder": r["folder"], "mean_diff_solved": md})
    best = max(considered, key=lambda c: c["mean_diff_solved"]) if considered else None
    return {"chosen_lr": best["lr"] if best else None, "considered": considered,
            "excluded": excluded}


# ------------------------------------------------------------------ output


def _f(x, nd=3):
    return "-" if x is None else f"{x:.{nd}f}"


def _ci(ci, nd=3):
    return "-" if ci is None else f"[{ci[0]:.{nd}f}, {ci[1]:.{nd}f}]"


def _p(x):
    return "-" if x is None else f"{x:.3g}"


def _label(r):
    return r["folder"].removeprefix("run_")


def render_md(results: list[dict], skipped: list[str], sel: dict) -> str:
    L = HEADLINE_LEN
    o = ["# exp-00c analysis", "",
         "Generated by `analyze.py`. Frozen = baseline dumps, trained = evaluate dumps, paired by "
         "prompt. Solve rate = fraction of a problem's samples that verify.", ""]
    if skipped:
        o += [f"Skipped (evaluate dumps missing): {', '.join(skipped)}", ""]
    o += [f"## Headline: long_pool length {L}", "",
          f"Up/down/unchanged and the sign test are over length-{L} problems the frozen model "
          "solves at least once (ties dropped). CIs are 95% bootstrap over problems "
          f"(seed {BOOT_SEED}, {BOOT_N} resamples).", "",
          "| run | n | frozen-solved | up | down | same | sign p | mean diff (solved) | CI "
          "| mean diff (all) | CI |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        h = r.get("headline")
        if not h:
            o.append(f"| {_label(r)} | long_pool missing |||||||||| ")
            continue
        o.append(f"| {_label(r)} | {h['n_problems']} | {h['n_frozen_solved']} | {h['up']} | "
                 f"{h['down']} | {h['unchanged']} | {_p(h['sign_p'])} | "
                 f"{_f(h['mean_diff_solved'], 4)} | {_ci(h['mean_diff_solved_ci'], 4)} | "
                 f"{_f(h['mean_diff_all'], 4)} | {_ci(h['mean_diff_all_ci'], 4)} |")
    o += ["", "New capability: frozen 0 of n and trained >= 0.5. New any: frozen 0, trained > 0.",
          "", "| run | " + " | ".join(f"L{x} frozen-zero | L{x} new cap | L{x} new any"
                                     for x in NEW_CAP_LENS) + " |",
          "|---" * (1 + 3 * len(NEW_CAP_LENS)) + "|"]
    for r in results:
        nc = r.get("new_capability", {})
        cells = []
        for x in NEW_CAP_LENS:
            c = nc.get(str(x))
            cells += ([f"{c['n_frozen_zero']}/{c['n_problems']}", str(c["new_capability"]),
                       str(c["new_any"])] if c else ["-"] * 3)
        o.append(f"| {_label(r)} | " + " | ".join(cells) + " |")

    o += ["", "## Secondary: avg solve rate (95% CI via mean_ci)", ""]
    hdr = ["| run | part | length | n | frozen | CI | trained | CI | delta |",
           "|---|---|---|---|---|---|---|---|---|"]
    o += hdr
    for r in results:
        s = r["secondary"]
        blocks = [("long_pool", k, v) for k, v in s.get("long_pool_by_length", {}).items()]
        if "textbook72" in s:
            blocks.append(("textbook72", "all", s["textbook72"]))
        if "greedy_overall" in s:
            blocks.append(("greedy_heldout", "all", s["greedy_overall"]))
        blocks += [("greedy_heldout", k, v) for k, v in s.get("greedy_by_length", {}).items()]
        for part, k, v in blocks:
            o.append(f"| {_label(r)} | {part} | {k} | {v['n']} | {_f(v['frozen'])} | "
                     f"{_ci(v['frozen_ci'])} | {_f(v['trained'])} | {_ci(v['trained_ci'])} | "
                     f"{v['delta']:+.3f} |")
    o += ["", "greedy_heldout problems lost (frozen ok, trained not) and gained:", "",
          "| run | lost | gained |", "|---|---|---|"]
    for r in results:
        s = r["secondary"]
        o.append(f"| {_label(r)} | {s.get('greedy_lost', '-')} | {s.get('greedy_gained', '-')} |")

    o += ["", "## Training dynamics (metrics.jsonl)", "",
          "| run | final KL | max KL | entropy first5 | entropy last5 | frac_mixed mean | "
          "reward first5 | reward last5 | total completions |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        d = r["dynamics"]
        o.append(f"| {_label(r)} | {_f(d.get('final_kl'), 4)} | {_f(d.get('max_kl'), 4)} | "
                 f"{_f(d.get('entropy_first5'))} | {_f(d.get('entropy_last5'))} | "
                 f"{_f(d.get('frac_mixed_mean'))} | {_f(d['reward_first5'])} | "
                 f"{_f(d['reward_last5'])} | {d['total_completions']} |")
    for r in results:
        if r["kind"] == "ei" and r["dynamics"].get("rounds"):
            o += ["", f"Expert iteration rounds ({_label(r)}):", "",
                  "| round | reward | sft/loss_last |", "|---|---|---|"]
            for rd in r["dynamics"]["rounds"]:
                o.append(f"| {rd['round']} | {_f(rd.get('reward'))} | "
                         f"{_f(rd.get('sft/loss_last'), 4)} |")
    notes = [(r["folder"], n) for r in results for n in r["notes"]]
    if notes:
        o += ["", "Notes:", ""] + [f"- {f}: {n}" for f, n in notes]
    o += ["", "## Selection rule (pre-registered)", "", selection_text(sel), ""]
    for e in sel["excluded"]:
        o.append(f"- excluded lr {e['lr']:g}: {'; '.join(e['reasons'])}")
    for c in sel["considered"]:
        o.append(f"- considered lr {c['lr']:g}: mean diff (solved, L{L}) "
                 f"{c['mean_diff_solved']:+.4f}")
    return "\n".join(o) + "\n"


def selection_text(sel: dict) -> str:
    if sel["chosen_lr"] is None:
        return "Selection rule: none qualifies"
    return f"Selection rule: chosen lr = {sel['chosen_lr']:g}"


def plot_len7(results: list[dict], path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rs = [r for r in results if r.get("len7_scatter")]
    if not rs:
        return
    fig, axes = plt.subplots(1, len(rs), figsize=(3 * len(rs), 3.2), squeeze=False)
    for ax, r in zip(axes[0], rs, strict=True):
        xs, ys = zip(*r["len7_scatter"], strict=True)
        ax.plot([0, 1], [0, 1], color="grey", lw=1)
        ax.scatter(xs, ys, s=12, color="tab:blue", alpha=0.6)
        ax.set_xlim(-0.03, 1.03)
        ax.set_ylim(-0.03, 1.03)
        ax.set_aspect("equal")
        ax.set_title(_label(r), fontsize=10)
        ax.set_xlabel("frozen solve rate")
    axes[0][0].set_ylabel("trained solve rate")
    fig.suptitle(f"long_pool length {HEADLINE_LEN}, per problem", fontsize=10)
    fig.patch.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor="white")
    plt.close(fig)


def run_all(exp_dir: str, runs=RUNS, verbose: bool = True) -> dict:
    results, skipped = [], []
    for folder, kind, lr in runs:
        r = analyze_run(exp_dir, folder, kind, lr)
        if r is None:
            skipped.append(folder)
            if verbose:
                print(f"skip {folder}: evaluate dumps missing")
            continue
        results.append(r)
    sel = select_lr(results)
    out = {"exp_dir": os.path.abspath(exp_dir), "skipped": skipped, "runs": results,
           "selection": sel}
    with open(os.path.join(exp_dir, "analysis.json"), "w") as f:
        json.dump(out, f, indent=1)
    with open(os.path.join(exp_dir, "analysis.md"), "w") as f:
        f.write(render_md(results, skipped, sel))
    plot_len7(results, os.path.join(exp_dir, "analysis_len7.png"))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--exp-dir", default=HERE)
    a = ap.parse_args(argv)
    out = run_all(a.exp_dir)
    print(f"wrote analysis.json, analysis.md, analysis_len7.png to {a.exp_dir}")
    print(selection_text(out["selection"]))


if __name__ == "__main__":
    main()
