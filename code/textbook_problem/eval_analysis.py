"""Per-problem analysis of the two evaluations (eval_scale_pretrain and eval_dan_best_state):
which problems each model solves, by reference length, premises, form, rules and boxes, source,
nuance; seed and size overlaps; proof-length padding; failure reasons. The sealed 72 are reported
in aggregate only (no names, no per-problem rows).

    cd code && uv run python textbook_problem/eval_analysis.py

Writes experiments/textbook-problem/eval_dan_best_state/analysis/{analysis.json, per_problem.md,
per_problem.csv} and prints the summary tables.
"""

import collections
import csv
import json
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "../ws3")]
import ndpath  # noqa: E402,F401
import prove  # noqa: E402
from data import load_textbook72  # noqa: E402

L = prove.L
DATA = os.path.join(HERE, "../../data/textbook_problem")
EX = os.path.join(HERE, "../../experiments/textbook-problem")
PT = {"d384": f"{EX}/eval_scale_pretrain/run_d384_s0", "d768": f"{EX}/eval_scale_pretrain/run_d768_s0"}
RL = {"s0": f"{EX}/eval_dan_best_state/run_best12_r8/s0", "s1": f"{EX}/eval_dan_best_state/run_best12_r8/s1"}
T72 = os.path.join(HERE, "../../docs/textbook-problem/textbook72")
OUT = f"{EX}/eval_dan_best_state/analysis"
SETS = ["v0", "v1", "hand14", "batch3", "lean_v2"]
PLAIN = ["curry", "conjoined", "negated_conjunction", "forward_control", "wrappers", "near_miss", "permutation",
         "distractor_scaling", "substitution", "demorgan", "line_budget", "no_dn", "force_ore"]


def jl(p):
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []


def lbin(n):
    return "1–6" if n <= 6 else "7–9" if n <= 9 else "10–13" if n <= 13 else "14–20" if n <= 20 else "21+"


def load_refs():
    refs = {}
    for fn, src in [("candidate_v0/problems.jsonl", None), ("candidate_v1/problems.jsonl", None),
                    ("candidates_v3/batch3_proved.jsonl", None), ("release_2026-09-30_v2/lean.jsonl", None),
                    ("release_2026-09-30_v2/lean_unsolved.jsonl", None)]:
        for r in jl(f"{DATA}/{fn}"):
            refs[r["name"]] = r
    for r in jl(f"{DATA}/candidates_v3/hand_proofs.jsonl"):
        r = dict(r, source=dict(id="roy" if r["source_number"][0] in "ET" else "pelletier", number=r["source_number"]),
                 reference_lines=r["min_lines"])
        refs[r["name"]] = r
    for nu in PLAIN:
        for r in jl(f"{DATA}/candidates_derived/{nu}.jsonl"):
            refs[r["name"]] = dict(r, source=dict(id="derived", number=nu), nuance=nu, base_name=r["base"]["name"])
    return refs


def features(r):
    prem, goal = L.parse_prompt(r["prompt"])
    f = dict(n_prem=len(prem), form="theorem" if not prem else "sequent", atoms=len({a for x in prem + [goal] for a in L.atoms(x)}),
             prompt_tokens=len(r["prompt"].split()), goal_op=goal[0], classical_only=r.get("classical_only"))
    rp = r.get("reference_proof")
    if rp:
        pf = prove.proof_features(rp)
        f.update(ref_lines=len(prove.parse_proof(rp)), rules=pf["rule_set"], n_boxes=pf["n_boxes"], depth=pf["max_box_depth"],
                 needs_ore="ORE" in pf["rule_set"], uses_dn="DN" in pf["rule_set"])
    else:
        f.update(ref_lines=r.get("reference_lines") or r.get("min_lines"), rules=[], n_boxes=None, depth=None, needs_ore=None, uses_dn=None)
    if f["classical_only"] is None:
        f["classical_only"] = not L.intuitionistically_valid(prem, goal)
    return f


def main():
    os.makedirs(OUT, exist_ok=True)
    refs = load_refs()
    rows = []
    for s in SETS + ["derived"]:
        pt = {k: {r["name"]: r for r in jl(f"{d}/{s}.jsonl")} for k, d in PT.items()}
        rl = {k: {r["name"]: r for r in jl(f"{d}/{'derived_plain' if s == 'derived' else s}.jsonl")} for k, d in RL.items()}
        names = [n for n in rl["s0"]] if s == "derived" else list(pt["d384"])
        for n in names:
            if n not in refs or n not in rl["s0"]:
                continue
            r = refs[n]
            f = features(r)
            p0, p1 = pt["d384"].get(n), pt["d768"].get(n)
            a, b = rl["s0"][n], rl["s1"][n]
            pt_rate = p0["solve_rate"] if p0 else None
            rl_rate = (a["n_ok"] / a["n_tried"] + b["n_ok"] / b["n_tried"]) / 2
            cat = ("both" if pt_rate and a["solved"] else "rl_only" if a["solved"] else "pt_only" if pt_rate else "neither")
            pt_ok_lens = [x["n_lines"] for x in (p0["samples"] if p0 else []) if x["ok"]]
            rows.append(dict(
                set=s, name=n, source=r["source"]["id"], number=str(r["source"].get("number")), nuance=r.get("nuance"), base=r.get("base_name"),
                sequent=L.pretty_seq(*L.parse_prompt(r["prompt"])), **f, lbin=lbin(f["ref_lines"] or 0),
                pt_d384=round(pt_rate, 3) if pt_rate is not None else None, pt_d768=round(p1["solve_rate"], 3) if p1 else None,
                rl_s0=round(a["n_ok"] / a["n_tried"], 3), rl_s1=round(b["n_ok"] / b["n_tried"], 3), rl_mean=round(rl_rate, 3),
                rl_solved_s0=a["solved"], rl_solved_s1=b["solved"], category=cat,
                pt_min_written=min(pt_ok_lens) if pt_ok_lens else None, pt_mean_written=round(statistics.mean(pt_ok_lens), 1) if pt_ok_lens else None,
                rl_min_written=min(a["written_lens"] + b["written_lens"]) if (a["written_lens"] or b["written_lens"]) else None,
                rl_mean_written=round(statistics.mean(a["written_lens"] + b["written_lens"]), 1) if (a["written_lens"] or b["written_lens"]) else None,
                pt_fail=collections.Counter(x["reason"].split(" (")[0][:28] for x in (p0["samples"] if p0 else []) if not x["ok"]).most_common(2),
                rl_fail=collections.Counter(x.split(" (")[0][:34] for x in a["reasons"] + b["reasons"]).most_common(2),
            ))
    # ---------------- summaries
    def grp(key, rows_, pred=lambda r: True):
        g = collections.defaultdict(list)
        for r in rows_:
            if pred(r):
                g[key(r)].append(r)
        out = {}
        for k, v in g.items():
            out[str(k)] = dict(n=len(v), pt_avg=round(statistics.mean(x["pt_d384"] or 0 for x in v), 3),
                               pt_any=sum(1 for x in v if x["pt_d384"]), rl_avg=round(statistics.mean(x["rl_mean"] for x in v), 3),
                               rl_any_s0=sum(1 for x in v if x["rl_solved_s0"]), rl_any_s1=sum(1 for x in v if x["rl_solved_s1"]),
                               rl_any_either=sum(1 for x in v if x["rl_solved_s0"] or x["rl_solved_s1"]))
        return out
    scraped = [r for r in rows if r["set"] != "derived"]
    A = dict(
        by_exact_length=grp(lambda r: r["ref_lines"], scraped),
        by_length_bin=grp(lambda r: r["lbin"], scraped),
        by_n_prem=grp(lambda r: r["n_prem"], scraped),
        by_form=grp(lambda r: r["form"], scraped),
        by_goal_op=grp(lambda r: r["goal_op"], scraped),
        by_needs_ore=grp(lambda r: r["needs_ore"], scraped),
        by_uses_dn=grp(lambda r: r["uses_dn"], scraped),
        by_classical_only=grp(lambda r: r["classical_only"], scraped),
        by_n_boxes=grp(lambda r: min(r["n_boxes"], 4) if r["n_boxes"] is not None else None, scraped),
        by_depth=grp(lambda r: r["depth"], scraped),
        by_source=grp(lambda r: r["source"], scraped),
        by_atoms=grp(lambda r: r["atoms"], scraped),
        by_nuance=grp(lambda r: r["nuance"], [r for r in rows if r["set"] == "derived"]),
        category_counts=dict(collections.Counter(r["category"] for r in scraped)),
    )
    # per-problem rate distributions
    def dist(vals):
        b = collections.Counter("0" if v == 0 else "(0,0.25]" if v <= 0.25 else "(0.25,0.5]" if v <= 0.5 else "(0.5,0.75]" if v <= 0.75 else "(0.75,1]" for v in vals)
        return {k: b[k] for k in ["0", "(0,0.25]", "(0.25,0.5]", "(0.5,0.75]", "(0.75,1]"]}
    A["rate_distribution"] = dict(pretrain_d384=dist([r["pt_d384"] or 0 for r in scraped]), rl_mean=dist([r["rl_mean"] for r in scraped]))
    # overlaps
    s384 = {r["name"] for r in scraped if r["pt_d384"]}; s768 = {r["name"] for r in scraped if r["pt_d768"]}
    rs0 = {r["name"] for r in scraped if r["rl_solved_s0"]}; rs1 = {r["name"] for r in scraped if r["rl_solved_s1"]}
    A["overlap"] = dict(pt_d384=len(s384), pt_d768=len(s768), pt_both=len(s384 & s768), pt_either=len(s384 | s768),
                        pt_jaccard=round(len(s384 & s768) / max(1, len(s384 | s768)), 3),
                        rl_s0=len(rs0), rl_s1=len(rs1), rl_both=len(rs0 & rs1), rl_either=len(rs0 | rs1), rl_jaccard=round(len(rs0 & rs1) / max(1, len(rs0 | rs1)), 3),
                        rl_s0_only=sorted(rs0 - rs1), rl_s1_only=sorted(rs1 - rs0), pt_d768_only=sorted(s768 - s384), pt_d384_only=sorted(s384 - s768),
                        pt_either_not_rl=sorted((s384 | s768) - (rs0 | rs1)))
    # padding
    def pad(key):
        xs = [(r[key], r["ref_lines"]) for r in scraped if r[key] and r["ref_lines"]]
        return dict(n=len(xs), mean_written=round(statistics.mean(x for x, _ in xs), 1), mean_ref=round(statistics.mean(y for _, y in xs), 1),
                    mean_excess=round(statistics.mean(x - y for x, y in xs), 1), at_or_below_ref=sum(1 for x, y in xs if x <= y))
    A["padding"] = dict(pretrain_mean_written=pad("pt_mean_written"), pretrain_min_written=pad("pt_min_written"),
                        rl_mean_written=pad("rl_mean_written"), rl_min_written=pad("rl_min_written"))
    # extremes
    solved_rl = [r for r in scraped if r["rl_solved_s0"] or r["rl_solved_s1"]]
    unsolved_rl = [r for r in scraped if not (r["rl_solved_s0"] or r["rl_solved_s1"])]
    A["extremes"] = dict(
        longest_rl_solved=[(r["set"], r["number"], r["ref_lines"], r["sequent"], r["rl_mean"]) for r in sorted(solved_rl, key=lambda r: -(r["ref_lines"] or 0))[:8]],
        shortest_rl_unsolved=[(r["set"], r["number"], r["ref_lines"], r["sequent"], r["pt_d384"]) for r in sorted(unsolved_rl, key=lambda r: r["ref_lines"] or 999)[:10]],
        longest_pt_solved=[(r["set"], r["number"], r["ref_lines"], r["sequent"], r["pt_d384"]) for r in sorted([r for r in scraped if r["pt_d384"]], key=lambda r: -(r["ref_lines"] or 0))[:5]],
        rl_hardest_solved_by_rate=[(r["set"], r["number"], r["ref_lines"], r["sequent"], r["rl_mean"]) for r in sorted(solved_rl, key=lambda r: r["rl_mean"])[:8]],
    )
    # derived vs base (RL): conditional on the base being solved
    base_solved = {r["name"]: (r["rl_solved_s0"] or r["rl_solved_s1"]) for r in scraped}
    cont = collections.Counter()
    for r in rows:
        if r["set"] == "derived" and r["base"] in base_solved:
            cont[(base_solved[r["base"]], r["rl_solved_s0"] or r["rl_solved_s1"])] += 1
    A["derived_given_base"] = {f"base_{'solved' if b else 'unsolved'}_derived_{'solved' if d else 'unsolved'}": c for (b, d), c in cont.items()}
    # failure reasons
    A["fail_reasons"] = dict(
        pretrain=collections.Counter(k for r in scraped for k, _ in r["pt_fail"]).most_common(8),
        rl=collections.Counter(k for r in scraped for k, _ in r["rl_fail"]).most_common(8),
    )
    # textbook 72: aggregate by length bin only
    t72 = {(r.get("name") or r.get("thm")): r for r in load_textbook72()}
    rl72 = {k: {r["name"]: r for r in jl(f"{T72}/dan_eval/{k}/textbook72.jsonl")} for k in RL}
    pt72 = {r["name"]: r for r in jl(f"{T72}/samples_scale-d384-p1200_s0_pretrain_k32.jsonl")}
    agg = collections.defaultdict(lambda: dict(n=0, pt=0, rl_s0=0, rl_s1=0))
    for n, r in t72.items():
        nl = r.get("n_lines")
        b = lbin(nl) if nl else "unknown"
        agg[b]["n"] += 1
        agg[b]["pt"] += bool(pt72.get(n, {}).get("solve_rate"))
        agg[b]["rl_s0"] += bool(rl72["s0"].get(n, {}).get("solved"))
        agg[b]["rl_s1"] += bool(rl72["s1"].get(n, {}).get("solved"))
    A["textbook72_by_length_bin_aggregate"] = dict(agg)
    json.dump(A, open(f"{OUT}/analysis.json", "w"), indent=1, default=str)
    # per-problem files (non-sealed only)
    cols = ["set", "source", "number", "nuance", "sequent", "ref_lines", "n_prem", "n_boxes", "depth", "needs_ore", "uses_dn", "classical_only",
            "pt_d384", "pt_d768", "rl_s0", "rl_s1", "category", "pt_min_written", "rl_min_written", "name"]
    with open(f"{OUT}/per_problem.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in sorted(rows, key=lambda r: (r["set"], r["ref_lines"] or 0)):
            w.writerow(r)
    with open(f"{OUT}/per_problem.md", "w") as f:
        f.write("# Per-problem results (non-sealed sets; the 72 are not listed)\n\nrates = fraction of 32 samples accepted; pt = Robbie's pretrain d384 / d768 (one seed); rl = Dan's best-state r8 seeds 0 / 1.\n\n")
        for s in SETS + ["derived"]:
            f.write(f"\n## {s}\n\n| source | no. | sequent | ref lines | boxes | ∨E | DN | pt d384 | pt d768 | rl s0 | rl s1 | category |\n|---|---|---|---|---|---|---|---|---|---|---|---|\n")
            for r in sorted([r for r in rows if r["set"] == s], key=lambda r: (r["ref_lines"] or 0, r["number"])):
                f.write(f"| {r['source']} | {r['number']} | {r['sequent']} | {r['ref_lines']} | {r['n_boxes']} | {'y' if r['needs_ore'] else ''} | {'y' if r['uses_dn'] else ''} | {r['pt_d384']} | {r['pt_d768']} | {r['rl_s0']} | {r['rl_s1']} | {r['category']} |\n")
    print(json.dumps({k: A[k] for k in ["by_length_bin", "by_n_prem", "by_form", "by_needs_ore", "by_uses_dn", "by_classical_only", "by_n_boxes", "by_depth", "by_source", "category_counts", "rate_distribution", "overlap", "padding", "derived_given_base", "fail_reasons", "textbook72_by_length_bin_aggregate"]}, indent=1, default=str))
    print("extremes:", json.dumps(A["extremes"], ensure_ascii=False, indent=1))
    print("by_exact_length:", json.dumps(A["by_exact_length"]))
    print("by_nuance:", json.dumps(A["by_nuance"]))


if __name__ == "__main__":
    main()
