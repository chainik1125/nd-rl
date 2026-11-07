"""The frozen analysis (PREREG.md): per-group Δlog10 from C, sampled rates from B, the bottleneck, A's funnel.

    python analyze.py --refs refs.jsonl --summaries DIR --out_json results.json --out_md tables.md
    python analyze.py --refs refs.jsonl --summaries DIR --control control.json   # check 6, from the 0 % and 1 % bases

Inputs are the counts `fr_classify.py` wrote (no proof text). Outputs hold no proof text and can be committed.
"""

# ruff: noqa: E501, SIM115, B905, B007, SIM300
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from factor_lib import GROUPS, flags, ref_from_json, replay
from factor_lib import sigma_dn_lines as _sigma_dn

LN10 = math.log(10)
DOSES = ("0%", "0.1%", "1%")
STAGE_KEY = {"pick": "pick", "reach": "bot", "negi": "negi", "dn": "dn"}
LINES = {
    "0%": ["dko_s0", "dko_s1", "dko_s2"],
    "0.1%": ["dd01_s0", "dd01_s1", "dd01_s2"],
    "1%": ["dd1_s0", "dd1_s1", "dd1_s2"],
}
PILOT = "dd01g0_s2"
BASES = {"0%": ["ko_s0", "ko_s1", "ko_s2"], "0.1%": ["dose01_s0", "dose01_s1", "dose01_s2"],
         "1%": ["dose1_s0", "dose1_s1", "dose1_s2"]}  # fmt: skip


def mean(x) -> float:
    x = list(x)
    return sum(x) / len(x) if x else float("nan")


def group_logs(sc: dict) -> dict[str, float]:
    """log10 P per group for one theorem: the 33-base mean of the per-action log P, summed."""
    out = dict.fromkeys(GROUPS, 0.0)
    for v, g in zip(sc["logp_mean_nats"], sc["groups"]):
        out[g] += v / LN10
    return out


def dlog(x: int, n: int) -> float:
    return math.log10((x + 0.5) / (n + 1))


def schema_of(r) -> str:
    return r.schema or "other"


def c_table(end: dict, base: dict, names: list[str]) -> dict:
    """Δlog10 per group over `names`: means per theorem, shares of the total, shares of the positive part."""
    rows = {}
    for n in names:
        a, b = group_logs(end["stages"]["C"][n]), group_logs(base["stages"]["C"][n])
        rows[n] = {g: a[g] - b[g] for g in GROUPS}
    d = {g: mean(rows[n][g] for n in names) for g in GROUPS}
    tot = sum(d.values())
    pos = sum(max(v, 0.0) for v in d.values())
    return {
        "n": len(names),
        "delta": d,
        "total": tot,
        "share_of_total": {g: (d[g] / tot if abs(tot) > 1e-9 else None) for g in GROUPS},
        "share_of_positive": {g: (max(d[g], 0.0) / pos if pos > 0 else 0.0) for g in GROUPS},
        "dominant": _dominant(d, pos),
        "per_theorem_total_median": sorted(sum(rows[n].values()) for n in names)[len(names) // 2],
    }


def _dominant(d: dict, pos: float) -> str:
    if pos <= 0:
        return "none (no group rose)"
    best = max(d, key=lambda g: d[g])
    if d[best] / pos < 0.5:
        return "none at ≥ 50 %"
    return best


READING = {"pick": "selection", "reach": "general execution", "negi": "candidate gain on NEGI",
           "dn": "candidate gain on DN", "setup": "no reading set", "finish": "no reading set"}  # fmt: skip


def b_table(end: dict, base: dict, names: list[str]) -> dict:
    out = {}
    for s, key in STAGE_KEY.items():
        eb, bb = end["stages"].get(s), base["stages"].get(s)
        if not eb or not bb:
            continue
        xe = sum(eb[n][key] for n in names)
        xb = sum(bb[n][key] for n in names)
        ne = sum(eb[n]["n"] for n in names)
        nb = sum(bb[n]["n"] for n in names)
        out[s] = {
            "base_x": xb, "base_n": nb, "end_x": xe, "end_n": ne,
            "delta_pooled": dlog(xe, ne) - dlog(xb, nb),
            "delta_mean_per_theorem": mean(dlog(eb[n][key], eb[n]["n"]) - dlog(bb[n][key], bb[n]["n"]) for n in names),
            "theorems_floor_base": sum(bb[n][key] == 0 for n in names),
            "theorems_floor_end": sum(eb[n][key] == 0 for n in names),
            "theorems_floor_both": sum(bb[n][key] == 0 and eb[n][key] == 0 for n in names),
            "pooled_at_floor": xb == 0 or xe == 0,
        }  # fmt: skip
    return out


def agreement(c: dict, b: dict) -> list[dict]:
    """Where B and C disagree: opposite signs with both |Δ| ≥ 0.1 (frozen), or |Δ| apart by ≥ 1 (reported)."""
    out = []
    for s in STAGE_KEY:
        if s not in b:
            continue
        dc, db = c["delta"][s], b[s]["delta_mean_per_theorem"]
        if dc * db < 0 and abs(dc) >= 0.1 and abs(db) >= 0.1:
            out.append({"stage": s, "kind": "sign", "C": dc, "B": db})
        elif abs(dc - db) >= 1.0:
            out.append({"stage": s, "kind": "magnitude", "C": dc, "B": db})
    return out


def bottleneck(base: dict, names: list[str], refs: dict) -> dict:
    """Per theorem, the non-empty group with the lowest probability (C), and among the four sampled stages (B)."""
    c: dict = {}
    bb: dict = {}
    by_schema: dict = {}
    for n in names:
        g = group_logs(base["stages"]["C"][n])
        present = {x: g[x] for x in GROUPS if x in base["stages"]["C"][n]["groups"]}
        lo = min(present, key=lambda x: present[x])
        c[lo] = c.get(lo, 0) + 1
        s = schema_of(refs[n])
        by_schema.setdefault(s, {})
        by_schema[s][lo] = by_schema[s].get(lo, 0) + 1
        rates = {}
        for st, key in STAGE_KEY.items():
            sb = base["stages"].get(st)
            if sb:
                rates[st] = dlog(sb[n][key], sb[n]["n"])
        if rates:
            m = min(rates, key=lambda x: rates[x])
            bb[m] = bb.get(m, 0) + 1
    return {"C": c, "C_by_schema": by_schema, "B": bb}


def a_table(sm: dict, names: list[str]) -> dict:
    A = sm["stages"].get("A")
    if not A:
        return {}
    k = sum(A[n]["n"] for n in names)
    fl: dict = {}
    for n in names:
        for key, v in A[n]["solved_flags"].items():
            fl[key] = fl.get(key, 0) + v
    return {
        "theorems": len(names),
        "first32_solved": sum(A[n]["solved32"] for n in names),
        "solved64": sum(A[n]["solved"] for n in names),
        "e32": sum(A[n]["e32"] for n in names),
        "draws": k,
        "draws_ok": sum(A[n]["n_ok"] for n in names),
        "funnel": [sum(A[n]["stage"][i] for n in names) for i in range(5)],
        "funnel_ref_T": [sum(A[n]["stage_ref_T"][i] for n in names) for i in range(5)],
        "solved_flags": fl,
        "end": _sum_dicts(A[n]["end"] for n in names),
    }


def _sum_dicts(ds) -> dict:
    out: dict = {}
    for d in ds:
        for k, v in d.items():
            out[k] = out.get(k, 0) + v
    return out


def inner_sigma(refs: dict) -> dict[str, list[int]]:
    """Theorems whose reference has a second σ DN line (it sits in reach): the actions of that inner reductio."""
    out = {}
    for n, r in refs.items():
        if r.n_sigma_dn < 2:
            continue
        tr = replay(r.prompt, r.actions)
        who = {i: a for a, made in enumerate(tr.made) for i in made}
        acts: list[int] = []
        chosen = r.actions[r.dn]
        for ln in _sigma_dn(tr.lines):
            if who[ln["idx"]] == r.dn:
                continue
            o = tr.origin(ln["refs"][0])
            acts += [who[o["refs"][0]], who[o["idx"]], who[ln["idx"]]]
        assert chosen and all(r.groups[a] in ("reach", "setup") for a in acts), n
        out[n] = acts
    return out


def setup_split(end: dict, base: dict, refs: dict, names: list[str]) -> dict:
    """Setup's Δ: premise lines (written by the model in this env, not prefilled) vs the rest."""
    pr, other = [], []
    for n in names:
        r = refs[n]
        e, b = end["stages"]["C"][n]["logp_mean_nats"], base["stages"]["C"][n]["logp_mean_nats"]
        pr.append(sum(e[i] - b[i] for i in range(r.pick) if i < r.n_pr) / LN10)
        other.append(sum(e[i] - b[i] for i in range(r.pick) if i >= r.n_pr) / LN10)
    return {"premise_lines": mean(pr), "other_setup": mean(other)}


def analyse(refs: dict, sdir: Path) -> dict:
    sm = {p.stem: json.loads(p.read_text()) for p in sdir.glob("*.json")}
    all_names = sorted(refs)
    schemas = sorted({schema_of(r) for r in refs.values()})
    res: dict = {"n_theorems": len(all_names), "lines": {}, "bases": {}, "doses": {}}
    inner = inner_sigma(refs)
    base_of = {}
    for dose, ls in LINES.items():
        for ln, bt in zip(ls, BASES[dose]):
            base_of[ln] = bt
    base_of[PILOT] = "dose01_s2"
    for ln, bt in base_of.items():
        if (
            ln not in sm
            or bt not in sm
            or "C" not in sm[ln]["stages"]
            or "C" not in sm[bt]["stages"]
        ):
            continue
        e, b = sm[ln], sm[bt]
        c = c_table(e, b, all_names)
        bt_tab = b_table(e, b, all_names)
        row = {
            "base": bt,
            "ckpt": {k: e.get("ckpt", {}).get(k) for k in ("kind", "it", "x_dan", "x_exact", "x_read_it", "last_iter",
                                                           "x_dan_last", "file", "md5", "dose", "seed")},
            "C": c,
            "C_setup_split": setup_split(e, b, refs, all_names),
            "C_by_schema": {s: c_table(e, b, [n for n in all_names if schema_of(refs[n]) == s]) for s in schemas},
            "B": bt_tab,
            "B_by_schema": {s: b_table(e, b, [n for n in all_names if schema_of(refs[n]) == s]) for s in schemas},
            "B_vs_C": agreement(c, bt_tab),
            "A_end": a_table(e, all_names),
            "A_base": a_table(b, all_names),
            "t_is_goal_split": {
                k: c_table(e, b, [n for n in all_names if refs[n].t_is_goal == v])
                for k, v in (("T_goal", True), ("T_subgoal", False))
            },
        }  # fmt: skip
        if inner:
            names = sorted(inner)
            d_in, d_rest = [], []
            for n in names:
                ee, bb = (
                    e["stages"]["C"][n]["logp_mean_nats"],
                    b["stages"]["C"][n]["logp_mean_nats"],
                )
                reach = [i for i, g in enumerate(refs[n].groups) if g == "reach"]
                d_in.append(sum(ee[i] - bb[i] for i in inner[n]) / LN10)
                d_rest.append(sum(ee[i] - bb[i] for i in reach if i not in inner[n]) / LN10)
            row["inner_sigma_in_reach"] = {
                "n": len(names),
                "inner_reductio_actions": mean(d_in),
                "rest_of_reach": mean(d_rest),
            }
        res["lines"][ln] = row
    for dose, bs in BASES.items():
        for bt in bs:
            if bt not in sm or "C" not in sm[bt]["stages"]:
                continue
            b = sm[bt]
            g = {x: mean(group_logs(b["stages"]["C"][n])[x] for n in all_names) for x in GROUPS}
            res["bases"][bt] = {
                "dose": dose,
                "group_log10": g,
                "path_log10": sum(g.values()),
                "bottleneck": bottleneck(b, all_names, refs),
                "B_rates": {s: {"x": sum(b["stages"][s][n][k] for n in all_names),
                                "n": sum(b["stages"][s][n]["n"] for n in all_names),
                                "floor": sum(b["stages"][s][n][k] == 0 for n in all_names)}
                            for s, k in STAGE_KEY.items() if s in b["stages"]},
                "A": a_table(b, all_names),
                "qc": b.get("qc", {}),
            }  # fmt: skip
    for dose, ls in LINES.items():
        have = [ln for ln in ls if ln in res["lines"]]
        if not have:
            continue
        res["doses"][dose] = {
            "seeds": have,
            "delta_mean": {
                g: mean(res["lines"][ln]["C"]["delta"][g] for ln in have) for g in GROUPS
            },
            "total_mean": mean(res["lines"][ln]["C"]["total"] for ln in have),
            "dominant_by_seed": {ln: res["lines"][ln]["C"]["dominant"] for ln in have},
        }
    res["qc"] = {t: sm[t].get("qc", {}) for t in sm}
    res["names_renamed"] = {
        t: {s: [sm[t]["stats"][s].get("names_renamed"), sm[t]["stats"][s].get("names_defined")]
            for s in sm[t].get("stats", {}) if "names_defined" in sm[t]["stats"][s]}
        for t in sm
    }  # fmt: skip
    res["times"] = {
        t: {s: sm[t]["stats"][s].get("secs") for s in sm[t].get("stats", {})} for t in sm
    }
    return res


def control(refs: dict, sdir: Path) -> dict:
    """Check 6: the DN group's log10 P at base, 0 % vs 1 %. Frozen: passes iff the 0 % dose mean is lower."""
    names = sorted(refs)
    out: dict = {"per_base": {}}
    for dose in ("0%", "1%"):
        vals = []
        for bt in BASES[dose]:
            sm = json.loads((sdir / f"{bt}.json").read_text())
            v = mean(group_logs(sm["stages"]["C"][n])["dn"] for n in names)
            out["per_base"][bt] = {"dose": dose, "dn_log10": v, "dn_prob_geomean": 10**v,
                                   "negi_log10": mean(group_logs(sm["stages"]["C"][n])["negi"] for n in names),
                                   "pick_log10": mean(group_logs(sm["stages"]["C"][n])["pick"] for n in names)}  # fmt: skip
            vals.append(v)
        out[dose] = mean(vals)
    out["pass"] = out["0%"] < out["1%"]
    return out


# ------------------------------------------------------------------------------------- tables
def pct(x: int, n: int) -> str:
    return f"{100 * x / n:.1f} % ({x:,} / {n:,})" if n else "n/a"


def fmt(x, d=2) -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:+.{d}f}"


def tables(res: dict) -> str:
    L = res["lines"]
    o = [
        "# factor_read tables",
        "",
        f"All on {res['n_theorems']} of the 106 strict σ theorems. Generated by `analyze.py`.",
        "",
    ]
    order = [x for d in DOSES for x in LINES[d]] + [PILOT]
    order = [x for x in order if x in L]
    o += [
        "## Checkpoints",
        "",
        "| line | dose | seed | RL end | kind | it | x_dan | line's last it |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for ln in order:
        c = L[ln]["ckpt"]
        o.append(f"| {ln} | {c['dose']} | {c['seed']} | `{c['file']}` | {'**alarm**' if c['kind'] == 'alarm' else c['kind']} | {c['it']} | "
                 f"{c['x_dan']:.2f}{'' if c['x_exact'] else ' (read at it ' + str(c['x_read_it']) + ')'} | {c['last_iter']} |")  # fmt: skip
    o += ["", "## C. Δlog10 per group, base to RL end (mean per theorem; adds exactly to the path total)", "",
          "| line | dose | seed | setup | pick | reach ⊥ | NEGI | DN | finish | total | dominant (≥ 50 % of the positive part) |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]  # fmt: skip
    for ln in order:
        c = L[ln]["C"]
        d = c["delta"]
        o.append(f"| {ln} | {L[ln]['ckpt']['dose']} | {L[ln]['ckpt']['seed']} | " + " | ".join(fmt(d[g]) for g in GROUPS)
                 + f" | {fmt(c['total'])} | {c['dominant']} |")  # fmt: skip
    o += ["", "### Each group's share of the total", "",
          "| line | setup | pick | reach ⊥ | NEGI | DN | finish |", "|---|---|---|---|---|---|---|"]  # fmt: skip
    for ln in order:
        s = L[ln]["C"]["share_of_total"]
        o.append(
            f"| {ln} | "
            + " | ".join("n/a" if s[g] is None else f"{100 * s[g]:.0f} %" for g in GROUPS)
            + " |"
        )
    o += ["", "### Each group's share of the positive part (the prereg's reading uses this)", "",
          "| line | setup | pick | reach ⊥ | NEGI | DN | finish |", "|---|---|---|---|---|---|---|"]  # fmt: skip
    for ln in order:
        s = L[ln]["C"]["share_of_positive"]
        o.append(f"| {ln} | " + " | ".join(f"{100 * s[g]:.0f} %" for g in GROUPS) + " |")
    o += ["", "### Setup split, and the inner reductio on the theorems with two σ DN lines", "",
          "| line | setup: premise lines | setup: other | inner reductio actions (in reach; n theorems) | rest of reach on those theorems |",
          "|---|---|---|---|---|"]  # fmt: skip
    for ln in order:
        ss, ii = L[ln]["C_setup_split"], L[ln].get("inner_sigma_in_reach", {})
        o.append(f"| {ln} | {fmt(ss['premise_lines'])} | {fmt(ss['other_setup'])} | "
                 f"{fmt(ii.get('inner_reductio_actions'))} ({ii.get('n')}) | {fmt(ii.get('rest_of_reach'))} |")  # fmt: skip
    schemas = sorted(next(iter(L.values()))["C_by_schema"]) if L else []
    o += ["", "### By schema", ""]
    for s in schemas:
        n = L[order[0]]["C_by_schema"][s]["n"]
        o += [
            f"**{s}** (n = {n})",
            "",
            "| line | setup | pick | reach ⊥ | NEGI | DN | finish | total | dominant |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for ln in order:
            c = L[ln]["C_by_schema"][s]
            o.append(
                f"| {ln} | "
                + " | ".join(fmt(c["delta"][g]) for g in GROUPS)
                + f" | {fmt(c['total'])} | {c['dominant']} |"
            )
        o.append("")
    o += ["### T = goal and T = subgoal", "", "| line | subset | n | setup | pick | reach ⊥ | NEGI | DN | finish | total | dominant |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]  # fmt: skip
    for ln in order:
        for k, c in L[ln]["t_is_goal_split"].items():
            o.append(
                f"| {ln} | {k} | {c['n']} | "
                + " | ".join(fmt(c["delta"][g]) for g in GROUPS)
                + f" | {fmt(c['total'])} | {c['dominant']} |"
            )
    o += ["", "## B. Sampled rates from each boundary (% of attempts, raw counts)", "",
          "Δlog10 uses (x + 0.5) / (n + 1), as the mean per theorem. Floor = theorems with 0 / 64 at base / at the end / at both.", "",
          "| line | stage | base | RL end | Δlog10 per theorem | Δlog10 pooled | floor base / end / both | flag |",
          "|---|---|---|---|---|---|---|---|"]  # fmt: skip
    for ln in order:
        for s, b in L[ln]["B"].items():
            flag = "pooled at floor" if b["pooled_at_floor"] else ""
            o.append(f"| {ln} | {s} | {pct(b['base_x'], b['base_n'])} | {pct(b['end_x'], b['end_n'])} | {fmt(b['delta_mean_per_theorem'])} | "
                     f"{fmt(b['delta_pooled'])} | {b['theorems_floor_base']} / {b['theorems_floor_end']} / {b['theorems_floor_both']} | {flag} |")  # fmt: skip
    o += ["", "### Where B and C disagree", ""]
    n_dis = 0
    for ln in order:
        for x in L[ln]["B_vs_C"]:
            n_dis += 1
            o.append(f"- {ln} {x['stage']}: {x['kind']}, C {fmt(x['C'])}, B {fmt(x['B'])}")
    if not n_dis:
        o.append("None by the frozen rule.")
    o += ["", "## Bottleneck at base", "", "Theorems whose lowest-probability group is each group (C), and among the four sampled stages (B).", "",
          "| base | dose | C: setup / pick / reach / NEGI / DN / finish | B: pick / reach / NEGI / DN |", "|---|---|---|---|"]  # fmt: skip
    for bt, b in res["bases"].items():
        c, bb = b["bottleneck"]["C"], b["bottleneck"]["B"]
        o.append(
            f"| {bt} | {b['dose']} | "
            + " / ".join(str(c.get(g, 0)) for g in GROUPS)
            + " | "
            + " / ".join(str(bb.get(s, 0)) for s in STAGE_KEY)
            + " |"
        )
    o += [
        "",
        "### Bottleneck by schema, dose means over seeds (C)",
        "",
        "| dose | schema | setup | pick | reach | NEGI | DN | finish |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for dose in DOSES:
        bs = [x for x in BASES[dose] if x in res["bases"]]
        if not bs:
            continue
        for s in sorted({s for x in bs for s in res["bases"][x]["bottleneck"]["C_by_schema"]}):
            vals = [
                mean(res["bases"][x]["bottleneck"]["C_by_schema"].get(s, {}).get(g, 0) for x in bs)
                for g in GROUPS
            ]
            o.append(f"| {dose} | {s} | " + " | ".join(f"{v:.1f}" for v in vals) + " |")
    o += [
        "",
        "## Base probabilities per group (mean log10 P per theorem, C)",
        "",
        "| base | dose | setup | pick | reach ⊥ | NEGI | DN | finish | path |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for bt, b in res["bases"].items():
        o.append(
            f"| {bt} | {b['dose']} | "
            + " | ".join(f"{b['group_log10'][g]:.2f}" for g in GROUPS)
            + f" | {b['path_log10']:.2f} |"
        )
    o += ["", "## A. Unforced draws, k 64", "",
          "Stage = the furthest a draw's own reductio gets: 1 pick, 2 valid ⊥ on the box's level, 3 NEGI, 4 DN citing it. Counts of draws at that stage or beyond.", "",
          "| checkpoint | first-32 solved (Dan's) | solved in 64 | E32 | draws solved | ≥ pick | ≥ ⊥ | ≥ NEGI | ≥ DN | σ among solved draws (written) | dn_derived among solved |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]  # fmt: skip

    def a_row(tag: str, a: dict) -> str:
        f, n = a["funnel"], a["draws"]
        cum = [sum(f[i:]) for i in range(5)]
        ok = a["draws_ok"]
        sf = a["solved_flags"]
        return (f"| {tag} | {pct(a['first32_solved'], a['theorems'])} | {pct(a['solved64'], a['theorems'])} | {a['e32']:.1f} | {pct(ok, n)} | "
                f"{pct(cum[1], n)} | {pct(cum[2], n)} | {pct(cum[3], n)} | {pct(cum[4], n)} | "
                f"{pct(sf.get('sigma_w', 0), ok)} | {pct(sf.get('w_dn_derived', 0), ok)} |")  # fmt: skip

    for bt, b in res["bases"].items():
        if b["A"]:
            o.append(a_row(bt, b["A"]))
    for ln in order:
        if L[ln]["A_end"]:
            o.append(a_row(ln, L[ln]["A_end"]))
    return "\n".join(o) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refs", required=True)
    ap.add_argument("--summaries", required=True)
    ap.add_argument("--out_json", default=None)
    ap.add_argument("--out_md", default=None)
    ap.add_argument("--control", default=None)
    a = ap.parse_args()
    refs = {r.name: r for r in (ref_from_json(json.loads(x)) for x in open(a.refs))}
    assert flags is not None
    sdir = Path(a.summaries)
    if a.control:
        c = control(refs, sdir)
        Path(a.control).write_text(json.dumps(c, indent=1))
        print(json.dumps(c, indent=1))
        return
    res = analyse(refs, sdir)
    if a.out_json:
        Path(a.out_json).write_text(json.dumps(res, indent=1))
    if a.out_md:
        Path(a.out_md).write_text(tables(res))


if __name__ == "__main__":
    main()
