#!/usr/bin/env python3
"""The final dev tables (markdown) and `summary.json` for the figures.

    python final_tables.py [--final DIR] [--runs DIR] > tables.md

Reads Lean-judged dev rows: the three pretrained bases (`runs/<pod>/pre_s<k>/dev_plain_k5.jsonl`),
the final checkpoints in plain and full mode and the ablation in plain mode
(`final/<tag>/dev_<tag>_<mode>.jsonl`), and the table of Dan's base
(`tmp/danbase/dev_table_T08.json`, made with his sampler). pass@1 is the mean over the 5 draws of
the share of dev theorems solved; pass@5 is the share solved by at least one of the 5 draws.
"""

import argparse
import collections
import json
import os

F = "/scratch/cowifornia/nd-rl/fable_rl"
LENS = [7, 8, 9, 10, 11, 12, 13, 14]


def table(rows, meta):
    by = collections.defaultdict(dict)
    for r in rows:
        by[r["name"]][r["draw"]] = r
    groups = collections.defaultdict(list)
    for name, d in by.items():
        m = meta[name]
        for g in ("all", f"len{m['n_lines']}", f"src:{m['source']}"):
            groups[g].append(d)
    res = {}
    for g, ds in groups.items():
        n = len(ds)
        att = [x for d in ds for x in d.values()]
        res[g] = {
            "n": n,
            "pass@1": 100.0 * sum(sum(x["lean_ok"] for x in d.values()) / len(d) for d in ds) / n,
            "pass@5": 100.0 * sum(any(d[i]["lean_ok"] for i in range(5) if i in d) for d in ds) / n,
            "verified%": 100.0 * sum(bool(x["verdict"]) for x in att) / len(att),
            "verified_wrong": sum(bool(x["verdict"]) and not x["lean_ok"] for x in att),
            "tok": sum(x.get("tokens_generated", x.get("tokens", 0)) for x in att) / len(att),  # "tokens": an older row format
            "scaffold": sum(x.get("scaffold_lines", 0) for x in att) / len(att),
            "attempts": len(att),
        }
    return res


def load(fn):
    return [json.loads(line) for line in open(fn) if line.strip()]


def cell(t, g):
    return f"{t[g]['pass@1']:.1f} / {t[g]['pass@5']:.1f}" if g in t else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", default=f"{F}/final")
    ap.add_argument("--runs", default=f"{F}/runs")
    a = ap.parse_args()
    meta = {}
    for line in open(f"{F}/data/y_dev.jsonl"):
        r = json.loads(line)
        meta[r["name"]] = r
    T = collections.OrderedDict()
    dan = f"{F}/tmp/danbase/dev_table_T08.json"
    if os.path.exists(dan):
        d = json.load(open(dan))
        T["Dan's full-corpus base, his sampler (plain)"] = {
            g: {"n": v["n"], "pass@1": v["pass@1"], "pass@5": v["pass@5"], "tok": v["tok/attempt"]} for g, v in d.items()}  # fmt: skip
    for pod, k in (("fr1", 0), ("fr2", 1), ("fr3", 2)):
        fn = f"{a.final}/s{k}/dev_base_s{k}_plain.jsonl"  # rescored with the final solver code
        if not os.path.exists(fn):
            fn = f"{a.runs}/{pod}/pre_s{k}/dev_plain_k5.jsonl"
        if os.path.exists(fn):
            T[f"pretrained base, seed {k} (plain)"] = table(load(fn), meta)
    for tag, label in (("abl", "ablation: plain expert iteration from the seed-0 base"),
                       ("s0", "after RL, seed 0"), ("s1", "after RL, seed 1"), ("s2", "after RL, seed 2")):  # fmt: skip
        for mode in ("plain", "full"):
            fn = f"{a.final}/{tag}/dev_{tag}_{mode}.jsonl"
            if os.path.exists(fn):
                T[f"{label} ({mode})"] = table(load(fn), meta)
    n = {g: next((t[g]["n"] for t in T.values() if g in t), 0) for g in ["all"] + [f"len{k}" for k in LENS] + ["src:gen", "src:textbook"]}  # fmt: skip
    print("### Dev, pass@1 / pass@5 (% of the dev theorems in the column), by lines in the reference proof\n")
    cols = ["all"] + [f"len{k}" for k in LENS]
    print("| model (mode) | " + " | ".join(f"{'all' if c == 'all' else c[3:] + ' lines'} (n={n[c]})" for c in cols) + " |")
    print("|---|" + "---|" * len(cols))
    for label, t in T.items():
        print(f"| {label} | " + " | ".join(cell(t, c) for c in cols) + " |")
    print("\n### Dev by source, tokens and verdicts\n")
    print(f"| model (mode) | generated (n={n['src:gen']}) | textbook (n={n['src:textbook']}) | generated tokens per attempt "
          "| attempts the solver said yes to | of those wrong | premise lines written by the solver per attempt |")  # fmt: skip
    print("|---|---|---|---|---|---|---|")
    for label, t in T.items():
        x = t["all"]
        ver = f"{x['verified%']:.1f}%" if "verified%" in x else ""
        vw = f"{x['verified_wrong']} of {round(x['verified%'] * x['attempts'] / 100)}" if "verified%" in x else ""
        sc = f"{x['scaffold']:.2f}" if "scaffold" in x else ""
        print(f"| {label} | {cell(t, 'src:gen')} | {cell(t, 'src:textbook')} | {x['tok']:.0f} | {ver} | {vw} | {sc} |")
    # summary for the figures
    summ = {"bases": {}, "by_length": {"lengths": LENS, "n": [n[f"len{k}"] for k in LENS], "rows": {}}}
    for k, run in ((0, "rl_s0"), (1, "rl_s1"), (2, "rl_s2")):
        t = T.get(f"pretrained base, seed {k} (plain)")
        if t:
            summ["bases"][run] = {"pass@1": t["all"]["pass@1"], "pass@5": t["all"]["pass@5"]}
    if "rl_s0" in summ["bases"]:
        summ["bases"]["abl_plainei_s0"] = summ["bases"]["rl_s0"]
    for name, pat in (("pretrained base, plain", "pretrained base, seed {} (plain)"),
                      ("after RL, plain", "after RL, seed {} (plain)"),
                      ("after RL, full (verifier)", "after RL, seed {} (full)")):  # fmt: skip
        seeds = [[T[pat.format(k)].get(f"len{L}", {}).get("pass@1", float("nan")) for L in LENS]
                 for k in range(3) if pat.format(k) in T]  # fmt: skip
        if seeds:
            summ["by_length"]["rows"][name] = {"seeds": seeds}
    t = T.get("ablation: plain expert iteration from the seed-0 base (plain)")
    if t:
        summ["by_length"]["rows"]["plain expert iteration, plain"] = {
            "seeds": [[t.get(f"len{L}", {}).get("pass@1", float("nan")) for L in LENS]]}  # fmt: skip
    os.makedirs(a.final, exist_ok=True)
    json.dump(summ, open(f"{a.final}/summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
