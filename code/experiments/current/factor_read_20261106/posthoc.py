"""Post-hoc description of factor_read's existing draws (CPU, no new read). NOT in the prereg: written after the results.

    python posthoc.py --refs refs.jsonl --reads READS --summaries S --results results.json --out posthoc.md --out_json posthoc.json

Why it exists. The frozen tables raised questions they cannot answer alone:
(a) "close box" collapses at some RL ends: what do they write instead?
(b) the reach continuation's funnel: ⊥, box closed, DN, solved. Does a checkpoint close the box when left to continue?
(c) C and B disagree: is the name-free C a fair estimate of the sampled next-action rate at RL ends?
(d) B's pooled Δlog10 as shares of its positive part, beside the frozen reading of C.
(e) A's funnel restricted to the reference's T, and how attempts end.
(f) solves and sampled rates by schema, since a mean over the 106 hides which theorems moved.
(g) why attempts fail, and how many names they use.
(h) the pick boundary's next action, and A's stage 4 by schema at the 1 % dose.
(i) given the pick: solves from the forced pick boundary against the unforced read.
Counts only; no proof text.
"""

# ruff: noqa: E501, SIM115, B905
from __future__ import annotations

import argparse
import collections
import gzip
import json
import math
from pathlib import Path

from factor_lib import ref_from_json, replay, shift_action

ORDER = [
    "dko_s0",
    "dko_s1",
    "dko_s2",
    "dd01_s0",
    "dd01_s1",
    "dd01_s2",
    "dd1_s0",
    "dd1_s1",
    "dd1_s2",
    "dd01g0_s2",
    "dko_s0_it500",
    "dko_s1_it630",
]
BASES = [
    "ko_s0",
    "ko_s1",
    "ko_s2",
    "dose01_s0",
    "dose01_s1",
    "dose01_s2",
    "dose1_s0",
    "dose1_s1",
    "dose1_s2",
]
LN10 = math.log(10)


def load(path: Path) -> list[dict]:
    with gzip.open(path, "rt") as f:
        return [json.loads(x) for x in f]


def close_classes(rows: list[dict], refs: dict) -> dict:
    """The next action from the prefix through the ⊥ line."""
    c: collections.Counter = collections.Counter()
    for r in rows:
        ref = refs[r["name"]]
        base = r["base"]
        tr = replay(ref.prompt, [shift_action(a, base) for a in ref.actions[: ref.bot + 1]], base)
        e = tr.lines[-1]["idx"]
        a = r["raw"][0].split() if r["raw"] and r["raw"][0] else []
        if not (r["ended"] and r["ended"][0]):
            c["truncated"] += 1
        elif not tr.apply(a):
            c["env refused"] += 1
        elif tr.lines[-1]["rule"] == "NEGI":
            c["closes the box"] += 1
        elif tr.lines[-1]["rule"] == "R" and tr.lines[-1]["refs"] == [e]:
            c["R copy of the ⊥ line"] += 1
        elif tr.lines[-1]["formula"] == ("bot",):
            c["another ⊥ line"] += 1
        else:
            c["other line"] += 1
    return dict(c)


def spearman(x: list[float], y: list[float]) -> float | None:
    if len(x) < 3:
        return None

    def rank(v: list[float]) -> list[float]:
        order = sorted(range(len(v)), key=lambda i: v[i])
        out = [0.0] * len(v)
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                out[order[k]] = (i + j) / 2
            i = j + 1
        return out

    rx, ry = rank(x), rank(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    sxy = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    sx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    sy = math.sqrt(sum((b - my) ** 2 for b in ry))
    return sxy / (sx * sy) if sx and sy else None


def c_vs_b(sm: dict, refs: dict) -> dict:
    """Per theorem, the name-free exp(log P) of the reference's next action (geometric mean over the name bases) against
    B's sampled rate at the same boundary. Above 1 in ratio means B finds more than the one reference action."""
    out = {}
    for st, key, idx in (
        ("pick", "pick", lambda r: r.pick),
        ("negi", "negi", lambda r: r.negi),
        ("dn", "dn", lambda r: r.dn),
    ):
        B = sm["stages"][st]
        C = sm["stages"]["C"]
        xs, ys, ratios = [], [], []
        below_floor = 0
        for n, r in refs.items():
            a = idx(r)
            p = math.exp(C[n]["logp_mean_nats"][a] - C[n]["logp_name_mean_nats"][a])
            rate = B[n][key] / B[n]["n"]
            xs.append(p)
            ys.append(rate)
            if rate > 0:
                ratios.append(math.log10(max(p, 1e-30) / rate))
            elif p > 3 / B[n]["n"]:
                below_floor += 1  # C says it should have been seen; B saw none
        ratios.sort()
        out[st] = {
            "spearman": spearman(xs, ys),
            "n_with_B_above_0": len(ratios),
            "median_log10_C_over_B": ratios[len(ratios) // 2] if ratios else None,
            "n_B0_but_C_above_3_of_64": below_floor,
            "mean_C": sum(xs) / len(xs),
            "mean_B": sum(ys) / len(ys),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refs", required=True)
    ap.add_argument("--reads", required=True)
    ap.add_argument("--summaries", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--out_json", required=True)
    a = ap.parse_args()
    refs = {r.name: r for r in (ref_from_json(json.loads(x)) for x in open(a.refs))}
    res = json.loads(Path(a.results).read_text())
    tags = [t for t in BASES + ORDER if (Path(a.summaries) / f"{t}.json").exists()]
    sm = {t: json.loads((Path(a.summaries) / f"{t}.json").read_text()) for t in tags}
    out: dict = {
        "close": {},
        "reach_funnel": {},
        "dn_classes": {},
        "c_vs_b": {},
        "b_shares": {},
        "a": {},
    }
    for t in tags:
        out["close"][t] = close_classes(load(Path(a.reads) / t / "negi.jsonl.gz"), refs)
        R = sm[t]["stages"]["reach"]
        out["reach_funnel"][t] = {
            k: sum(x[k] for x in R.values())
            for k in (
                "n",
                "bot",
                "negi",
                "dn",
                "solved",
                "solved_sigma",
                "false_bot_draws",
                "nested_bot_draws",
            )
        }
        out["reach_funnel"][t]["ends"] = dict(
            sum((collections.Counter(x["end"]) for x in R.values()), collections.Counter())
        )
        D = sm[t]["stages"]["dn"]
        out["dn_classes"][t] = {
            k: sum(x[k] for x in D.values()) for k in ("n", "env_ok", "dn", "r_copy", "dn_other")
        }
        out["c_vs_b"][t] = c_vs_b(sm[t], refs)
        A = sm[t]["stages"]["A"]
        fl = [sum(x["stage"][i] for x in A.values()) for i in range(5)]
        fr = [sum(x["stage_ref_T"][i] for x in A.values()) for i in range(5)]
        out["a"][t] = {"n": sum(x["n"] for x in A.values()), "cum": [sum(fl[i:]) for i in range(5)], "cum_ref_T": [sum(fr[i:]) for i in range(5)],
                       "ends": dict(sum((collections.Counter(x["end"]) for x in A.values()), collections.Counter()))}  # fmt: skip
    for ln in [x for x in ORDER if x in res["lines"]]:
        b = res["lines"][ln]["B"]
        d = {s: b[s]["delta_pooled"] for s in b}
        pos = sum(max(v, 0) for v in d.values())
        out["b_shares"][ln] = {
            "delta_pooled": d,
            "share_of_positive": {s: (max(v, 0) / pos if pos else 0.0) for s, v in d.items()},
        }

    def pc(x: int, n: int) -> str:
        return f"{100 * x / n:.1f} % ({x:,})" if n else "n/a"

    o = [
        "# factor_read: post-hoc description",
        "",
        "Written after the results, not pre-registered. Counts of the existing draws; no new read. k 64 on 106 theorems = 6,784 draws per row.",
        "",
    ]
    o += [
        "## (a) The next action from the prefix through the ⊥ line",
        "",
        "| checkpoint | closes the box | R copy of the ⊥ line | another ⊥ line | other line | env refused | truncated |",
        "|---|---|---|---|---|---|---|",
    ]
    for t in tags:
        c = out["close"][t]
        n = sum(c.values())
        o.append(
            f"| {t} | "
            + " | ".join(
                pc(c.get(k, 0), n)
                for k in (
                    "closes the box",
                    "R copy of the ⊥ line",
                    "another ⊥ line",
                    "other line",
                    "env refused",
                    "truncated",
                )
            )
            + " |"
        )
    o += [
        "",
        "## (b) The reach continuation: of the draws from the pick",
        "",
        "| checkpoint | valid ⊥ | box closed | DN citing it | solved (Lean) | false ⊥ in the draw | had a valid ⊥ in a nested box first | step cap | box closed / ⊥ | DN / box closed |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for t in tags:
        c = out["reach_funnel"][t]
        n = c["n"]
        o.append(f"| {t} | {pc(c['bot'], n)} | {pc(c['negi'], n)} | {pc(c['dn'], n)} | {pc(c['solved'], n)} | {pc(c['false_bot_draws'], n)} | {pc(c['nested_bot_draws'], n)} | {pc(c['ends'].get('step_cap', 0), n)} | "
                 f"{100 * c['negi'] / max(1, c['bot']):.1f} % | {100 * c['dn'] / max(1, c['negi']):.1f} % |")  # fmt: skip
    o += [
        "",
        "## The next action from the prefix through the NEGI line",
        "",
        "| checkpoint | DN citing it | R copy of it | DN to T citing another line | env accepted |",
        "|---|---|---|---|---|",
    ]
    for t in tags:
        c = out["dn_classes"][t]
        o.append(
            f"| {t} | {pc(c['dn'], c['n'])} | {pc(c['r_copy'], c['n'])} | {pc(c['dn_other'], c['n'])} | {pc(c['env_ok'], c['n'])} |"
        )
    o += ["", "## (c) The name-free C against B, per theorem, at the next-action stages", "", "Median log10(C / B) among theorems with B above 0; 0 means the reference action's probability equals the sampled rate.", "",
          "| checkpoint | pick: Spearman, median, n | close box: Spearman, median, n | DN: Spearman, median, n |", "|---|---|---|---|"]  # fmt: skip

    def cv(x: dict) -> str:
        s, m = x["spearman"], x["median_log10_C_over_B"]
        return f"{'n/a' if s is None else format(s, '+.2f')}, {'n/a' if m is None else format(m, '+.2f')}, {x['n_with_B_above_0']}"

    for t in tags:
        c = out["c_vs_b"][t]
        o.append(f"| {t} | {cv(c['pick'])} | {cv(c['negi'])} | {cv(c['dn'])} |")
    o += [
        "",
        "## (d) B's pooled Δlog10 and its shares of the positive part",
        "",
        "| line | pick | reach ⊥ | close box | DN | pick share | reach share | close share | DN share |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for ln, c in out["b_shares"].items():
        d, s = c["delta_pooled"], c["share_of_positive"]
        o.append(
            f"| {ln} | {d['pick']:+.2f} | {d['reach']:+.2f} | {d['negi']:+.2f} | {d['dn']:+.2f} | {100 * s['pick']:.0f} % | {100 * s['reach']:.0f} % | {100 * s['negi']:.0f} % | {100 * s['dn']:.0f} % |"
        )
    o += [
        "",
        "## (e) A: unforced draws reaching each stage, any X and X = the reference's T; how attempts end",
        "",
        "| checkpoint | ≥ pick | ≥ ⊥ | ≥ box closed | ≥ DN | ≥ pick, X = T | ≥ DN, X = T | done | syntax | step cap | truncated |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for t in tags:
        c = out["a"][t]
        n, e = c["n"], c["ends"]
        o.append(
            f"| {t} | {pc(c['cum'][1], n)} | {pc(c['cum'][2], n)} | {pc(c['cum'][3], n)} | {pc(c['cum'][4], n)} | {pc(c['cum_ref_T'][1], n)} | {pc(c['cum_ref_T'][4], n)} | {pc(e.get('done', 0), n)} | {pc(e.get('syntax', 0), n)} | {pc(e.get('step_cap', 0), n)} | {pc(e.get('truncated', 0), n)} |"
        )
    schemas = sorted({r.schema or "other" for r in refs.values()})
    sch = {x: [n for n, r in refs.items() if (r.schema or "other") == x] for x in schemas}
    pairs = [(ln, res["lines"][ln]["base"]) for ln in ORDER if ln in res["lines"]]
    out["solved_by_schema"] = {}
    o += [
        "",
        "## (f) Theorems solved in the 64 draws by schema, base to RL end (Lean)",
        "",
        "| line | " + " | ".join(f"{x} (n {len(sch[x])})" for x in schemas) + " |",
        "|---|" + "---|" * len(schemas),
    ]
    for ln, bt in pairs:
        ae, ab = sm[ln]["stages"]["A"], sm[bt]["stages"]["A"]
        row = {
            x: [sum(ab[n]["solved"] for n in sch[x]), sum(ae[n]["solved"] for n in sch[x])]
            for x in schemas
        }
        out["solved_by_schema"][ln] = row
        o.append(f"| {ln} | " + " | ".join(f"{row[x][0]} → {row[x][1]}" for x in schemas) + " |")
    o += ["", "### Sampled rates by schema: % of attempts, base → RL end", ""]
    for st, lab in (("pick", "pick"), ("reach", "reach ⊥"), ("negi", "close box"), ("dn", "DN")):
        o += [
            f"**{lab}**",
            "",
            "| line | " + " | ".join(schemas) + " |",
            "|---|" + "---|" * len(schemas),
        ]
        for ln, _ in pairs:
            b = res["lines"][ln]["B_by_schema"]
            cells = []
            for x in schemas:
                c = b[x][st]
                cells.append(
                    f"{100 * c['base_x'] / c['base_n']:.0f} → {100 * c['end_x'] / c['end_n']:.0f}"
                )
            o.append(f"| {ln} | " + " | ".join(cells) + " |")
        o.append("")
    out["fail"] = {}
    o += [
        "## (g) Why unforced draws fail, and how many names they use",
        "",
        "| checkpoint | env refused an action | of those, names exhausted (64 used) | names per attempt | names the env changed |",
        "|---|---|---|---|---|",
    ]
    for t in tags:
        st = json.loads((Path(a.reads) / t / "A.stats.json").read_text())
        n = sum(st["env_end"].values())
        syn = st["env_end"].get("syntax", 0)
        ex = st["env_fail_reason"].get("names exhausted", 0)
        out["fail"][t] = {
            "n": n,
            "syntax": syn,
            "names_exhausted": ex,
            "names_defined": st["names_defined"],
            "names_renamed": st["names_renamed"],
        }
        o.append(
            f"| {t} | {pc(syn, n)} | {pc(ex, syn)} | {st['names_defined'] / n:.1f} | {pc(st['names_renamed'], st['names_defined'])} |"
        )
    out["pick_any"] = {}
    o += [
        "",
        "## (h) The next action at the pick boundary: the reductio on T, any box assuming ¬T, and A's draws by schema at 1 %",
        "",
        "| checkpoint | NEGI box on ¬T (the pick) | any box with binder ¬T | env accepted |",
        "|---|---|---|---|",
    ]
    for t in tags:
        P = sm[t]["stages"]["pick"]
        c = {k: sum(x[k] for x in P.values()) for k in ("n", "pick", "assume_any", "env_ok")}
        out["pick_any"][t] = c
        o.append(
            f"| {t} | {pc(c['pick'], c['n'])} | {pc(c['assume_any'], c['n'])} | {pc(c['env_ok'], c['n'])} |"
        )
    out["a_by_schema"] = {}
    o += [
        "",
        "Unforced draws reaching a DN citing their own NEGI line (stage 4): any X, and X = the reference's T.",
        "",
        "| line | schema | n draws | base: any X, X = T | RL end: any X, X = T |",
        "|---|---|---|---|---|",
    ]
    for ln, bt in pairs:
        ae, ab = sm[ln]["stages"]["A"], sm[bt]["stages"]["A"]
        for x in schemas:
            nn = sum(ae[n]["n"] for n in sch[x])
            r = [
                sum(ab[n]["stage"][4] for n in sch[x]),
                sum(ab[n]["stage_ref_T"][4] for n in sch[x]),
                sum(ae[n]["stage"][4] for n in sch[x]),
                sum(ae[n]["stage_ref_T"][4] for n in sch[x]),
            ]
            out["a_by_schema"].setdefault(ln, {})[x] = r
            if ln.startswith("dd1_") and (r[2] or r[0]):
                o.append(f"| {ln} | {x} | {nn:,} | {r[0]:,}, {r[1]:,} | {r[2]:,}, {r[3]:,} |")
    out["forced_pick_solves"] = {}
    o += ["", "## (i) Given the pick: Lean-judged solves from the forced pick boundary, against the unforced read", "", "Theorems solved at least once in 64 draws, and % of draws. The prefix hands the model the setup and the pick on the reference's T.", "",
          "| checkpoint | forced: theorems | forced: draws | unforced: theorems | unforced: draws | forced theorems by schema |", "|---|---|---|---|---|---|"]  # fmt: skip
    for t in tags:
        R = sm[t]["stages"]["reach"]
        A = sm[t]["stages"]["A"]
        ft = sum(x["solved"] > 0 for x in R.values())
        fd = sum(x["solved"] for x in R.values())
        fn = sum(x["n"] for x in R.values())
        ut = sum(x["solved"] for x in A.values())
        ud = sum(x["n_ok"] for x in A.values())
        un = sum(x["n"] for x in A.values())
        by = {x: sum(R[n]["solved"] > 0 for n in sch[x]) for x in schemas}
        out["forced_pick_solves"][t] = {
            "forced_theorems": ft,
            "forced_draws": fd,
            "n": fn,
            "unforced_theorems": ut,
            "unforced_draws": ud,
            "by_schema": by,
        }
        o.append(
            f"| {t} | {ft} ({100 * ft / len(refs):.1f} %) | {pc(fd, fn)} | {ut} ({100 * ut / len(refs):.1f} %) | {pc(ud, un)} | "
            + ", ".join(f"{x} {v}" for x, v in by.items() if v)
            + " |"
        )
    out["bottleneck_ties"] = {}
    o += ["", "## (j) B's bottleneck with ties shown, and how many theorems have each factor at 0 of 64", "",
          "The frozen bottleneck table takes the first of the lowest stages, so a tie at 0 of 64 between pick and reach ⊥ counts as pick. Here ties are separate. Reach ⊥ at 0 means no draw from the forced pick reached a valid ⊥.", "",
          "| checkpoint | pick alone lowest | pick tied for lowest | another stage lowest | pick 0 of 64 | reach ⊥ 0 of 64 | both 0 of 64 | pick above 0, reach ⊥ above 0 |", "|---|---|---|---|---|---|---|---|"]  # fmt: skip
    keys = {"pick": "pick", "reach": "bot", "negi": "negi", "dn": "dn"}
    for t in tags:
        st = sm[t]["stages"]
        c = dict.fromkeys(("alone", "tied", "other", "pick0", "reach0", "both0", "both_up"), 0)
        for n in refs:
            v = {s: st[s][n][k] for s, k in keys.items()}
            lo = min(v.values())
            who = [s for s in v if v[s] == lo]
            c["alone" if who == ["pick"] else "tied" if "pick" in who else "other"] += 1
            c["pick0"] += v["pick"] == 0
            c["reach0"] += v["reach"] == 0
            c["both0"] += v["pick"] == 0 and v["reach"] == 0
            c["both_up"] += v["pick"] > 0 and v["reach"] > 0
        out["bottleneck_ties"][t] = c
        o.append(
            f"| {t} | {c['alone']} | {c['tied']} | {c['other']} | {c['pick0']} | {c['reach0']} | {c['both0']} | {c['both_up']} |"
        )
    Path(a.out).write_text("\n".join(o) + "\n")
    Path(a.out_json).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
