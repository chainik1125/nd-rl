"""Amendment 3, post-hoc and descriptive: the matched-compute checkpoints beside the 1 % ends. No frozen reading applies.

    python matched.py --refs refs.jsonl --ckpts ckpts_matched.json --reads READS --summaries S --m_reads MREADS --m_summaries MS \
        --numbers numbers.json --out matched.md --out_json matched.json

Counts of the draws; no proof text. `means` are over the three 1 % seeds and the three matched 0 % seeds only. The 0.1 % dose has one
matched seed, so it gets no mean; the guard-off pilot and the two "earliest available" reads are never in a mean.
"""

# ruff: noqa: E501, SIM115, B905
from __future__ import annotations

import argparse
import json
from pathlib import Path

from factor_lib import GROUPS, ref_from_json
from posthoc import NUMBERS, SCHEMA_ORDER, c_group_log10, load, reductio_any

ONE = {"dd1_s0": ("dose1_s0", 60), "dd1_s1": ("dose1_s1", 60), "dd1_s2": ("dose1_s2", 90)}
BY_SEED = {
    0: ["dd1_s0", "m_dko_s0_it60", "m_dd01_s0_it60"],
    1: ["dd1_s1", "m_dko_s1_it70", "e_dd01_s1_it35"],
    2: ["dd1_s2", "m_dko_s2_it100", "m_dd01g0_s2_it90", "e_dd01_s2_it25"],
}
MEAN_0 = ["m_dko_s0_it60", "m_dko_s1_it70", "m_dko_s2_it100"]
MEAN_1 = ["dd1_s0", "dd1_s1", "dd1_s2"]
UNMATCHED = {"m_dko_s0_it60": "dko_s0", "m_dko_s1_it70": "dko_s1", "m_dko_s2_it100": "dko_s2", "m_dd01_s0_it60": "dd01_s0", "m_dd01g0_s2_it90": "dd01g0_s2",
             "e_dd01_s1_it35": "dd01_s1", "e_dd01_s2_it25": "dd01_s2"}  # fmt: skip


def measures(tag: str, reads: Path, summaries: Path, refs: dict, sch: dict) -> dict:
    s = json.loads((summaries / f"{tag}.json").read_text())
    st = s["stages"]
    A, P, R, N, D, C = (st[k] for k in ("A", "pick", "reach", "negi", "dn", "C"))
    stats = json.loads((reads / tag / "A.stats.json").read_text())
    nA = sum(x["n"] for x in A.values())
    ends: dict = {}
    for x in A.values():
        for k, v in x["end"].items():
            ends[k] = ends.get(k, 0) + v
    stage = [sum(x["stage"][i] for x in A.values()) for i in range(5)]
    ref_t = [sum(x["stage_ref_T"][i] for x in A.values()) for i in range(5)]
    per = reductio_any(load(reads / tag / "pick.jsonl.gz"), refs)
    syn = stats["env_end"].get("syntax", 0)
    out = {
        "tag": tag,
        "qc": s.get("qc", {}),
        "first32": sum(bool(x["solved32"]) for x in A.values()),
        "solved64": sum(bool(x["solved"]) for x in A.values()),
        "draws_ok": sum(x["n_ok"] for x in A.values()),
        "draws": nA,
        "e32": sum(x["e32"] for x in A.values()),
        "a_ge_pick_any": sum(stage[1:]),
        "a_ge_pick_T": sum(ref_t[1:]),
        "a_ge_dn_any": stage[4],
        "ends": ends,
        "names_per_attempt": stats["names_defined"] / nA,
        "names_renamed_share": stats["names_renamed"] / max(1, stats["names_defined"]),
        "refused": syn,
        "names_exhausted": stats["env_fail_reason"].get("names exhausted", 0),
        "n": sum(x["n"] for x in P.values()),
        "pick": sum(x["pick"] for x in P.values()),
        "any_x": sum(x["valid"] for x in per.values()),
        "x_other": sum(x["x_other"] for x in per.values()),
        "reach": sum(x["bot"] for x in R.values()),
        "close": sum(x["negi"] for x in N.values()),
        "dn": sum(x["dn"] for x in D.values()),
        "forced_draws": sum(x["solved"] for x in R.values()),
        "forced_sigma": sum(x["solved_sigma"] for x in R.values()),
        "forced_theorems": sum(x["solved"] > 0 for x in R.values()),
        "reach0": sum(x["bot"] == 0 for x in R.values()),
        "pick0": sum(x["pick"] == 0 for x in P.values()),
        "c_pick_all": sum(c_group_log10(C[n]) for n in refs) / len(refs),
        "c_groups": {g: sum(c_group_log10(C[n], g) for n in refs) / len(refs) for g in GROUPS},
        "secs": sum(
            json.loads((reads / tag / f"{k}.stats.json").read_text())["secs"]
            for k in ("A", "pick", "reach", "negi", "dn", "C")
        ),
        "by_schema": {},
    }
    assert out["pick"] == sum(x["x_t"] for x in per.values()), tag
    for x, names in sch.items():
        out["by_schema"][x] = {
            "n": len(names),
            "solved64": sum(bool(A[n]["solved"]) for n in names),
            "pick": sum(P[n]["pick"] for n in names),
            "any_x": sum(per[n]["valid"] for n in names),
            "reach": sum(R[n]["bot"] for n in names),
            "dn": sum(D[n]["dn"] for n in names),
            "draws_each": sum(P[n]["n"] for n in names),
            "forced_draws": sum(R[n]["solved"] for n in names),
            "forced_theorems": sum(R[n]["solved"] > 0 for n in names),
            "reach0": sum(R[n]["bot"] == 0 for n in names),
            "c_pick": sum(c_group_log10(C[n]) for n in names) / len(names),
        }
    return out


def pc(x: int, n: int, d: int = 1) -> str:
    return f"{100 * x / n:.{d}f} % ({x:,})" if n else "n/a"


def arrow(a: int, b: int, n: int) -> str:
    return f"{100 * a / n:.1f} → {100 * b / n:.1f} % ({a:,} → {b:,})"


def main() -> None:
    ap = argparse.ArgumentParser()
    for k in ("refs", "ckpts", "reads", "summaries", "m_reads", "m_summaries", "out", "out_json"):
        ap.add_argument(f"--{k}", required=True)
    ap.add_argument("--numbers", default=str(NUMBERS))
    a = ap.parse_args()
    refs = {r.name: r for r in (ref_from_json(json.loads(x)) for x in open(a.refs))}
    sch = {x: [n for n, r in refs.items() if (r.schema or "other") == x] for x in SCHEMA_ORDER}
    cks = {c["tag"]: c for c in json.load(open(a.ckpts))}
    num = json.loads(Path(a.numbers).read_text())["lines"]
    rd, sm, mr, ms = Path(a.reads), Path(a.summaries), Path(a.m_reads), Path(a.m_summaries)
    meta: dict = {}
    for ln, (b, it) in ONE.items():
        c = {x["it"]: x for x in num[ln]["curve"]}[it]
        rb = [r for r in num[ln]["guard"]["rollbacks"] if r["it"] <= it]
        meta[ln] = {
            "base": b,
            "dose": "1 %",
            "x": c["x_dan"],
            "it": it,
            "kind": "rolling",
            "label": "",
            "group": "one",
            "own": c,
            "lr": rb[-1]["lr_scale"] if rb else 1.0,
            "rollbacks": rb,
        }
    for t, c in cks.items():
        meta[t] = {"base": c["base"], "dose": c["dose"].replace("%", " %"), "x": c["x_dan"], "it": c["it"], "kind": c["kind"], "label": c["label"], "group": c["group"], "own": c["own_read"], "lr": c["lr_scale"],
                   "share_reduced": c["share_of_x_at_reduced_lr"], "target": c["target"], "x_target": c["x_target"], "rollback": c["grpo_rollback"], "rollbacks": c["rollbacks"]}  # fmt: skip
    M: dict = {}
    for t in meta:
        M[t] = measures(t, mr if t in cks else rd, ms if t in cks else sm, refs, sch)
    for b in sorted({m["base"] for m in meta.values()}):
        M[b] = measures(b, rd, sm, refs, sch)
    for t in UNMATCHED.values():
        M[t] = measures(t, rd, sm, refs, sch)

    def lab(t: str) -> str:
        m = meta[t]
        return f"{t} ({m['label']})" if m["label"] else t

    order = [t for s in (0, 1, 2) for t in BY_SEED[s]]
    n = 6784
    o = ["# factor_read: the matched-compute control (Amendment 3, post-hoc)", "",
         "Added by the coordinator after seeing the results. No frozen reading applies. Seed-matched to the 1 % ends. Each row is base → that checkpoint. Forced rows and A: 6,784 draws.", "",
         "## Checkpoints", "", "| checkpoint | dose | seed | kind | it | x | x of the 1 % end | lr scale at it | share of x at reduced lr | rollback state | line's own read: first 32, solved in 64 |", "|---|---|---|---|---|---|---|---|---|---|---|"]  # fmt: skip
    for s in (0, 1, 2):
        for t in BY_SEED[s]:
            m = meta[t]
            own = m["own"]
            rb = m.get("rollback")
            o.append(f"| {lab(t)} | {m['dose']} | {s} | {m['kind']} | {m['it']} | {m['x']:.3f} | {m.get('x_target', m['x']):.3f} | {m['lr']:g} | {100 * m.get('share_reduced', 0):.0f} % | {('restored to it ' + str(rb['to_iter'])) if rb else '-'} | {(str(own['first32']) + ', ' + str(own['solved64'])) if own else 'no read at this it'} |")  # fmt: skip
    o += ["", "## Unforced (A), base → checkpoint", "", "| checkpoint | first-32 solved, of 106 | solved in 64, of 106 | draws solved | reductio reached, any X | reductio reached, X = T | names per attempt | `done` endings | step cap | of refusals, names exhausted |", "|---|---|---|---|---|---|---|---|---|---|"]  # fmt: skip
    for t in order:
        e, b = M[t], M[meta[t]["base"]]
        o.append(f"| {lab(t)} | {100 * b['first32'] / 106:.1f} → {100 * e['first32'] / 106:.1f} % ({b['first32']} → {e['first32']}) | {100 * b['solved64'] / 106:.1f} → {100 * e['solved64'] / 106:.1f} % ({b['solved64']} → {e['solved64']}) | {arrow(b['draws_ok'], e['draws_ok'], n)} | {arrow(b['a_ge_pick_any'], e['a_ge_pick_any'], n)} | {arrow(b['a_ge_pick_T'], e['a_ge_pick_T'], n)} | {b['names_per_attempt']:.1f} → {e['names_per_attempt']:.1f} | {arrow(b['ends'].get('done', 0), e['ends'].get('done', 0), n)} | {arrow(b['ends'].get('step_cap', 0), e['ends'].get('step_cap', 0), n)} | {pc(e['names_exhausted'], e['refused'])} |")  # fmt: skip
    o += ["", "## Forced (B), base → checkpoint", "", "| checkpoint | pick on T | reductio on any X at the pick step | reach ⊥ | close box | DN | solved from a forced pick, draws | same, theorems | theorems with reach ⊥ 0 of 64 |", "|---|---|---|---|---|---|---|---|---|"]  # fmt: skip
    for t in order:
        e, b = M[t], M[meta[t]["base"]]
        o.append(f"| {lab(t)} | {arrow(b['pick'], e['pick'], n)} | {arrow(b['any_x'], e['any_x'], n)} | {arrow(b['reach'], e['reach'], n)} | {arrow(b['close'], e['close'], n)} | {arrow(b['dn'], e['dn'], n)} | {arrow(b['forced_draws'], e['forced_draws'], n)} | {b['forced_theorems']} → {e['forced_theorems']} | {b['reach0']} → {e['reach0']} |")  # fmt: skip
    o += ["", "## Means over seeds (1 % ends, and matched 0 %; base → checkpoint, % of 6,784)", "", "The 0.1 % dose has one matched seed and no mean. Pilot and earliest-available reads are not in any mean.", "", "| measure | 1 %, three seeds | matched 0 %, three seeds |", "|---|---|---|"]  # fmt: skip

    def mean(ts: list[str], k: str, base: bool) -> float:
        return sum(M[meta[t]["base"] if base else t][k] for t in ts) / len(ts)

    means = {}
    for k, lb in (("draws_ok", "unforced draws solved"), ("a_ge_pick_any", "reductio reached unforced, any X"), ("a_ge_pick_T", "reductio reached unforced, X = T"), ("pick", "pick on T"), ("any_x", "reductio on any X at the pick step"), ("reach", "reach ⊥"), ("close", "close box"), ("dn", "DN"), ("forced_draws", "solved from a forced pick")):  # fmt: skip
        v = [
            mean(MEAN_1, k, True),
            mean(MEAN_1, k, False),
            mean(MEAN_0, k, True),
            mean(MEAN_0, k, False),
        ]
        means[k] = v
        o.append(
            f"| {lb} | {100 * v[0] / n:.1f} → {100 * v[1] / n:.1f} % | {100 * v[2] / n:.1f} → {100 * v[3] / n:.1f} % |"
        )
    for k, lb in (("first32", "first-32 solved, of 106"), ("forced_theorems", "theorems solved from a forced pick"), ("reach0", "theorems with reach ⊥ 0 of 64"), ("names_per_attempt", "names per attempt"), ("c_pick_all", "C: log10 P(pick), all 106")):  # fmt: skip
        v = [
            mean(MEAN_1, k, True),
            mean(MEAN_1, k, False),
            mean(MEAN_0, k, True),
            mean(MEAN_0, k, False),
        ]
        means[k] = v
        o.append(f"| {lb} | {v[0]:.2f} → {v[1]:.2f} | {v[2]:.2f} → {v[3]:.2f} |")
    o += ["", "## Against the unmatched RL ends (the same line, read in the frozen run)", "", "| line | x matched → x end | solved from a forced pick, draws | reach ⊥ | DN | close box | names per attempt | `done` endings | C: log10 P(pick), all 106 |", "|---|---|---|---|---|---|---|---|---|"]  # fmt: skip
    for t, ln in UNMATCHED.items():
        e, u = M[t], M[ln]
        xu = num[ln]["x_dan"]
        o.append(f"| {ln} | {meta[t]['x']:.2f} → {xu:.2f} | {arrow(e['forced_draws'], u['forced_draws'], n)} | {arrow(e['reach'], u['reach'], n)} | {arrow(e['dn'], u['dn'], n)} | {arrow(e['close'], u['close'], n)} | {e['names_per_attempt']:.1f} → {u['names_per_attempt']:.1f} | {arrow(e['ends'].get('done', 0), u['ends'].get('done', 0), n)} | {e['c_pick_all']:.2f} → {u['c_pick_all']:.2f} |")  # fmt: skip
    cols = [x for x in SCHEMA_ORDER if sch[x]]
    o += ["", "## C: mean log10 P(pick + ¬¬T), name-free, by schema, base → checkpoint", "", "| checkpoint | " + " | ".join(f"{x} (n {len(sch[x])})" for x in cols) + " | all 106 |", "|---|" + "---|" * (len(cols) + 1)]  # fmt: skip
    for t in order:
        e, b = M[t], M[meta[t]["base"]]
        o.append(f"| {lab(t)} | " + " | ".join(f"{b['by_schema'][x]['c_pick']:.2f} → {e['by_schema'][x]['c_pick']:.2f}" for x in cols) + f" | {b['c_pick_all']:.2f} → {e['c_pick_all']:.2f} |")  # fmt: skip
    o += ["", "## C: Δlog10 per group, name-free, base → checkpoint, mean per theorem", "", "| checkpoint | " + " | ".join(GROUPS).replace("negi", "close box").replace("pick", "pick + ¬¬T").replace("reach", "reach ⊥") + " | total |", "|---|" + "---|" * (len(GROUPS) + 1)]  # fmt: skip
    for t in order:
        e, b = M[t], M[meta[t]["base"]]
        d = {g: e["c_groups"][g] - b["c_groups"][g] for g in GROUPS}
        o.append(
            f"| {lab(t)} | "
            + " | ".join(f"{d[g]:+.2f}" for g in GROUPS)
            + f" | {sum(d.values()):+.2f} |"
        )
    for key, title in (
        ("solved64", "Theorems solved in 64 unforced draws"),
        ("forced_theorems", "Theorems solved from a forced pick"),
    ):
        o += ["", f"## {title}, by schema, base → checkpoint", "", "| checkpoint | " + " | ".join(f"{x} (n {len(sch[x])})" for x in cols) + " |", "|---|" + "---|" * len(cols)]  # fmt: skip
        for t in order:
            e, b = M[t], M[meta[t]["base"]]
            o.append(f"| {lab(t)} | " + " | ".join(f"{b['by_schema'][x][key]} → {e['by_schema'][x][key]}" for x in cols) + " |")  # fmt: skip
    for key, title in (
        ("pick", "Pick on T"),
        ("any_x", "Reductio on any X at the pick step"),
        ("reach", "Reach ⊥"),
        ("dn", "DN"),
    ):
        o += ["", f"## {title}, by schema, % of attempts, base → checkpoint", "", "| checkpoint | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]  # fmt: skip
        for t in order:
            e, b = M[t], M[meta[t]["base"]]
            o.append(f"| {lab(t)} | " + " | ".join(f"{100 * b['by_schema'][x][key] / b['by_schema'][x]['draws_each']:.0f} → {100 * e['by_schema'][x][key] / e['by_schema'][x]['draws_each']:.0f}" for x in cols) + " |")  # fmt: skip
    Path(a.out).write_text("\n".join(o) + "\n")
    Path(a.out_json).write_text(
        json.dumps({"meta": meta, "measures": M, "means": means}, indent=1, default=str)
    )


if __name__ == "__main__":
    main()
