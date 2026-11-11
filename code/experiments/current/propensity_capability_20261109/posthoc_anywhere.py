"""Post-hoc amendment 1 to PREREG.md, written AFTER the results were seen: the router-or-default-shift rule rerun
with propensity = "uses d anywhere" (U rows, used / k). Every reading here is exploratory. CPU only; reads the
result rows (no proof text) and the frozen results.json, never the raw attempts.

    python posthoc_anywhere.py [--results /scratch/cowifornia/nd-rl/propensity_capability/results] [--out results]
                               [--frozen results/results.json] [--no-fig]

Writes `posthoc_anywhere.json`, `posthoc_anywhere.md` and the figure `posthoc_anywhere.{pdf,svg,png}` to --out.
analyze.py is imported, not changed: the theorem selection, the classes, the bootstrap (same RNG keys as the frozen
router cell), the state rule, the router verdict and the dose agreement rule are its functions. What is new:

- the measure, per class: Σ used / Σ k over the class's theorems, and its log10 odds with the frozen rule's
  treatment of sampled rates, (x + 0.5) / (n + 1);
- the split of "uses d" into the attempts whose accepted proof goes through d and the rest. `solved` on a U/<d> row
  is what pc_run.py writes: accepted attempts whose proof goes *through* d (`pc_lib.classify`, "through": for the
  reductio, its DN line is on the conclusion's dependency path), which is narrower than SPEC.md's wording "accepted
  attempts that use d" (an accepted proof can open d as a detour). `solved` on U/any counts accepted attempts;
- sensitivity partitions of the required class by set (the strict 106; the 30 classical-only theorems of the other
  set), and seed-pooled per-dose numbers with a theorem-only bootstrap (descriptive, no verdict).
"""

# ruff: noqa: E501, B905
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import analyze as A  # noqa: E402

RED = "reductio"
OTHER_DIRS = ("negation", "conditional", "cases")
MEASURES = ("all", "through", "not_through", "excl_through", "among_accepted", "accepted")
MEASURE_LABEL = {
    "all": "uses d anywhere, % of all attempts",
    "through": "accepted with a proof through d, % of all attempts",
    "not_through": "uses d without it carrying an accepted proof, % of all attempts",
    "excl_through": "uses d, % of the attempts whose accepted proof does not go through d",
    "among_accepted": "proof through d, % of accepted attempts",
    "accepted": "accepted (any route), % of all attempts",
}
# comparisons for the reductio: (name, required-side group, not-needed group, measure, status)
COMPARISONS = (
    ("frozen classes, all attempts", "required", "idle", "all", "the amendment's reading"),
    ("frozen classes, excluding proofs through d", "required", "idle", "excl_through",
     "sensitivity: drops the attempts whose accepted proof goes through d"),
    ("strict 106 only, all attempts", "required_strict106", "idle", "all", "sensitivity: required side is the strict 106 alone"),
    ("strict 106 only, excluding proofs through d", "required_strict106", "idle", "excl_through", "sensitivity: both of the above"),
    ("30 classical-only of the other set, all attempts", "required_other", "idle", "all",
     "sensitivity, flagged: accepted proofs of these mostly avoid the reductio, so 'required' does not hold for it"),
)  # fmt: skip
SPLIT_GROUPS = ("required_strict106", "required_other")
GROUP_LABEL = {
    "required": "required (frozen)",
    "helpful": "helpful",
    "idle": "idle",
    "optional": "optional (all)",
    "required_strict106": "required: strict 106",
    "required_other": "required: other set's 30",
}


class PosthocError(ValueError):
    pass


# ----------------------------------------------------------------------------- counts and the measure
def frozen_keys(data: dict, b: str, e: str, d: str) -> list[tuple[str, str]]:
    """The frozen router cell's theorems: applicable with complete rows at both checkpoints, both sets."""
    keys = []
    for s in A.SETS:
        nb, _ = A.names_for(data[b].get(s, {}), d)
        ne, _ = A.names_for(data[e].get(s, {}), d)
        keys += [(s, n) for n in sorted(set(nb) & set(ne))]
    return keys


def theorem_counts(
    data: dict, c: str, keys: list[tuple[str, str]], d: str
) -> dict[str, np.ndarray]:
    """Per theorem at checkpoint c: used (opens d anywhere), solved_d (accepted, proof through d), k, solved_any."""
    used, sd, k, sa = [], [], [], []
    for s, n in keys:
        u = data[c][s][n]["U"]
        r, a = u[d], u.get("any")
        if a is None or a["k"] != r["k"]:
            raise PosthocError(f"{c}/{s}/{n}: U/any row missing or with a different k")
        if r["solved"] > r["used"] or r["solved"] > a["solved"]:
            raise PosthocError(f"{c}/{s}/{n}/{d}: accepted-through-d above uses or above accepted")
        used.append(r["used"])
        sd.append(r["solved"])
        k.append(r["k"])
        sa.append(a["solved"])
    f = lambda v: np.array(v, dtype=float)  # noqa: E731
    return {"used": f(used), "solved_d": f(sd), "k": f(k), "solved_any": f(sa)}


def add_counts(a: dict | None, b: dict) -> dict:
    return dict(b) if a is None else {f: a[f] + b[f] for f in b}


def measure_xn(t: dict, m: str) -> tuple[np.ndarray, np.ndarray]:
    if m == "all":
        return t["used"], t["k"]
    if m == "through":
        return t["solved_d"], t["k"]
    if m == "not_through":
        return t["used"] - t["solved_d"], t["k"]
    if m == "excl_through":
        return t["used"] - t["solved_d"], t["k"] - t["solved_d"]
    if m == "among_accepted":
        return t["solved_d"], t["solved_any"]
    if m == "accepted":
        return t["solved_any"], t["k"]
    raise KeyError(m)


def lodds(x, n):
    """log10 odds of the pooled rate (x + 0.5) / (n + 1): the frozen rule's treatment of a sampled rate."""
    x, n = np.asarray(x, dtype=float), np.asarray(n, dtype=float)
    return np.log10((x + 0.5) / (n - x + 0.5))


def rate(x, n):
    x, n = np.asarray(x, dtype=float), np.asarray(n, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, x / np.where(n > 0, n, 1.0), np.nan)


def nci(v: np.ndarray) -> list[float] | None:
    v = v[np.isfinite(v)]
    return A.ci(v) if v.size else None


def fnum(v) -> float | None:
    v = float(v)
    return v if math.isfinite(v) else None


def class_change(tb: dict, te: dict, idx: np.ndarray, m: str, n_theorems: int) -> tuple[dict, dict]:
    """Base-to-end change of one measure in one class: levels, points, log10 odds, the frozen state. Paired: the same
    resampled theorems at base and at end."""
    xb, nb = measure_xn(tb, m)
    xe, ne = measure_xn(te, m)
    Xb, Nb, Xe, Ne = xb.sum(), nb.sum(), xe.sum(), ne.sum()
    base, end = fnum(rate(Xb, Nb)), fnum(rate(Xe, Ne))
    bXb, bNb, bXe, bNe = xb[idx].sum(1), nb[idx].sum(1), xe[idx].sum(1), ne[idx].sum(1)
    bb, be = rate(bXb, bNb), rate(bXe, bNe)
    dl = float(lodds(Xe, Ne) - lodds(Xb, Nb))
    dl_boot = lodds(bXe, bNe) - lodds(bXb, bNb)
    dp_boot = be - bb
    cl = nci(dl_boot)
    out = {
        "n": n_theorems,
        "x_base": float(Xb), "n_base": float(Nb), "x_end": float(Xe), "n_end": float(Ne),
        "base": {"est": base, "ci": nci(bb)}, "end": {"est": end, "ci": nci(be)},
        "dpoints": {"est": None if base is None or end is None else end - base, "ci": nci(dp_boot)},
        "dlogodds10": {"est": dl, "ci": cl},
        # how the uses are spread: theorems with any use, and with every attempt using it (x = n > 0)
        "theorems_any": [int((xb > 0).sum()), int((xe > 0).sum())],
        "theorems_every": [int(((xb == nb) & (nb > 0)).sum()), int(((xe == ne) & (ne > 0)).sum())],
    }  # fmt: skip
    if n_theorems < A.N_MIN:
        out["state"] = f"not read (n < {A.N_MIN})"
    elif base is None or end is None or cl is None:
        out["state"] = "not read (no attempts in the denominator)"
    else:
        out["state"] = A.state(dl, cl[0], cl[1], base, end)
    return out, {"dl": dl_boot, "dp": dp_boot}


def compare(cls: dict, boots: dict, req: str, nn: str, m: str) -> dict:
    """The frozen router verdict on one measure, plus the difference of the two classes' changes."""
    short = [g for g in (req, nn) if m not in cls.get(g, {}) or cls[g][m]["n"] < A.N_MIN]
    if short:
        return {"verdict": f"no reading: {' and '.join(short)} n < {A.N_MIN}"}
    r, o = cls[req][m], cls[nn][m]
    if r["state"].startswith("not read") or o["state"].startswith("not read"):
        return {"verdict": "no reading: a class has no attempts in the denominator"}
    br, bo = boots[req][m], boots[nn][m]
    dd = bo["dl"] - A.ROUTE_SPLIT * br["dl"]
    d_ci = nci(dd)
    out = {
        "required_side": req,
        "not_needed_side": nn,
        "diff_logodds10": {"est": r["dlogodds10"]["est"] - o["dlogodds10"]["est"], "ci": nci(br["dl"] - bo["dl"])},
        "diff_points": {"est": r["dpoints"]["est"] - o["dpoints"]["est"], "ci": nci(br["dp"] - bo["dp"])},
        "not_needed_minus_half_required_ci": d_ci,
        "spillover_f": o["dlogodds10"]["est"] / r["dlogodds10"]["est"] if r["dlogodds10"]["est"] != 0 else None,
    }  # fmt: skip
    out["verdict"] = A.route_verdict(r["state"], o["state"], *d_ci)
    if out["verdict"] == "router" and o["state"] == "rare":
        out["flag"] = "not-needed class stays below 1 %"
    return out


def groups_for(classes: dict, d: str, keys: list[tuple[str, str]]) -> tuple[str, dict]:
    mode, groups, _ = A.class_groups(classes, d, keys)
    if d == RED and mode == "classes":
        groups["required_strict106"] = [k for k in groups["required"] if k[0] == "strict106"]
        groups["required_other"] = [k for k in groups["required"] if k[0] != "strict106"]
    return mode, groups


def group_key(key: str, g: str) -> str:
    """The frozen router cell's RNG key for its own classes (same resampled theorems), a new key for the splits."""
    return f"{key}|{g}" if g in A.CLASSES or g == "optional" else f"{key}|posthoc|{g}"


def router_from_counts(
    key: str, mode: str, groups: dict, cb: dict, ce: dict, d: str, verdicts: bool = True
) -> dict:
    """cb, ce: group -> per-theorem counts at base and end. All measures for every group; comparisons for reductio."""
    cls: dict = {}
    boots: dict = {}
    for g, ks in groups.items():
        cls[g], boots[g] = {"n_by_set": {s: sum(1 for k in ks if k[0] == s) for s in A.SETS}}, {}
        if not ks:
            continue
        idx = A.boot_idx(len(ks), group_key(key, g))
        for m in MEASURES if d == RED else ("all",):
            cls[g][m], boots[g][m] = class_change(cb[g], ce[g], idx, m, len(ks))
        thr = cls[g].get("through")
        if thr is not None:  # through d + not through d = all, exactly
            tot = cls[g]["all"]
            assert thr["x_base"] + cls[g]["not_through"]["x_base"] == tot["x_base"]
            assert thr["x_end"] + cls[g]["not_through"]["x_end"] == tot["x_end"]
            # theorem-checkpoints with an accepted attempt whose proof does not go through d (0 makes excl_through
            # exactly "uses d among attempts not accepted")
            cls[g]["cells_accepted_not_through"] = int(
                (cb[g]["solved_any"] > cb[g]["solved_d"]).sum()
                + (ce[g]["solved_any"] > ce[g]["solved_d"]).sum()
            )
            cls[g]["cells"] = 2 * len(ks)
    out: dict = {"mode": mode, "classes": cls, "comparisons": {}}
    if mode != "classes":
        out["verdict"] = (
            "no router reading: no theorem is labelled required for this direction (dir_class null)"
        )
        return out
    for name, req, nn, m, status in COMPARISONS:
        c = compare(cls, boots, req, nn, m)
        if not verdicts and not c["verdict"].startswith("no reading"):
            c.pop("flag", None)
            c["verdict"] = "none (seed-pooled: descriptive only)"
        c["status"] = status
        c["measure"] = m
        out["comparisons"][name] = c
    out["verdict"] = out["comparisons"][COMPARISONS[0][0]]["verdict"]
    return out


def router_line(key: str, data: dict, classes: dict, b: str, e: str, d: str) -> dict:
    keys = frozen_keys(data, b, e, d)
    if not keys:
        return {
            "mode": None,
            "classes": {},
            "comparisons": {},
            "verdict": "no reading: never applicable",
        }
    mode, groups = groups_for(classes, d, keys)
    cb = {g: theorem_counts(data, b, ks, d) for g, ks in groups.items() if ks}
    ce = {g: theorem_counts(data, e, ks, d) for g, ks in groups.items() if ks}
    return router_from_counts(key, mode, groups, cb, ce, d)


def pooled_dose(dose: str, seeds_lines: list[dict], data: dict, classes: dict, d: str) -> dict:
    """Seed-pooled: per theorem, the three seeds' counts summed at base and at end; theorems resampled (the same
    theorems in every seed). The interval covers theorem sampling only, not seed-to-seed variation. No verdict."""
    common = None
    for ln in seeds_lines:
        ks = set(frozen_keys(data, ln["base"], ln["end"], d))
        common = ks if common is None else common & ks
    keys = sorted(common or [])
    if not keys:
        return {"verdict": "no reading: never applicable"}
    mode, groups = groups_for(classes, d, keys)
    cb: dict = {}
    ce: dict = {}
    for g, ks in groups.items():
        if not ks:
            continue
        for ln in seeds_lines:
            cb[g] = add_counts(cb.get(g), theorem_counts(data, ln["base"], ks, d))
            ce[g] = add_counts(ce.get(g), theorem_counts(data, ln["end"], ks, d))
    out = router_from_counts(f"{dose}|pooled|router|{d}", mode, groups, cb, ce, d, verdicts=False)
    out["seeds"] = [ln["seed"] for ln in seeds_lines]
    return out


# ----------------------------------------------------------------------------- the elicitation-vs-capability check
def uses_anywhere_dose_readings(ckpts: dict, data: dict, frozen: dict, pairs: dict) -> dict:
    """Recompute the frozen secondary 'same rule with propensity = uses d anywhere' for the reductio (analyze.line_cell,
    u_prop=True), check it against results.json, and combine seeds with the frozen dose rule (not in tables.md)."""
    out: dict = {"per_seed": {}, "dose": {}, "mismatches": [], "matched": {}}
    for dose in A.DOSES:
        seeds = sorted(sd for (ds, sd) in pairs if ds == dose)
        for s in A.SETS:
            rs = []
            for sd in seeds:
                ln = pairs[(dose, sd)]
                if not ln["base"] or not ln["end"]:
                    rs.append(None)
                    continue
                key = f"{dose}|{sd}|{s}|{RED}"
                cur = A.line_cell(
                    key, data[ln["base"]].get(s, {}), data[ln["end"]].get(s, {}), RED, u_prop=True
                )
                fz = frozen["lines"][f"{dose}|{sd}"]["cells"][f"{s}|{RED}"]["uses_anywhere"]
                same = cur["reading"] == fz["reading"] and (
                    "P" not in cur or (abs(cur["P"]["dlog10"] - fz["P"]["dlog10"]) < 1e-9
                                       and np.allclose(cur["P"]["ci"], fz["P"]["ci"], atol=1e-9)))  # fmt: skip
                if not same:
                    out["mismatches"].append(key)
                p = cur.get("P", {})
                out["per_seed"][key] = {"reading": cur["reading"], "P_state": p.get("state"), "C_state": cur.get("C", {}).get("state"),
                                        "P_base": p.get("base", {}).get("est"), "P_end": p.get("end", {}).get("est"),
                                        "primary_reading": frozen["lines"][f"{dose}|{sd}"]["cells"][f"{s}|{RED}"]["primary"]["reading"],
                                        "agrees_with_results_json": same}  # fmt: skip
                rs.append(
                    cur["reading"]
                    if cur["n"] >= A.N_MIN or cur["reading"].startswith("no reading")
                    else None
                )
            if all(r is not None and r.startswith("no reading: never") for r in rs):
                dr = "never applicable"
            elif all(r is not None and r.startswith("no reading: n =") for r in rs):
                dr = f"no reading (n < {A.N_MIN})"
            else:
                dr = A.dose_reading(rs)
            out["dose"][f"{dose}|{s}"] = {"seeds": seeds, "readings": rs, "reading": dr}
    for e, ln in frozen.get("matched_and_secondary", {}).items():
        for s in A.SETS:
            c = ln.get("cells", {}).get(f"{s}|{RED}")
            if c:
                out["matched"][f"{e}|{s}"] = {"x": ln["x_end"], "dose": ln["dose"], "seed": ln["seed"],
                                              "reading": c["uses_anywhere"]["reading"], "primary_reading": c["primary"]["reading"]}  # fmt: skip
    return out


# ----------------------------------------------------------------------------- the whole post-hoc analysis
def posthoc(results_dir: Path, frozen_path: Path | None) -> dict:
    ckpts, data, files, classes = A.load(results_dir)
    frozen = json.loads(frozen_path.read_text()) if frozen_path and frozen_path.exists() else None
    pairs, _ = A.pair_lines(ckpts)
    lines = {}
    for (dose, seed), ln in pairs.items():
        if dose not in A.DOSES:
            continue
        lines[(dose, seed)] = ln
    res: dict = {"status": "post-hoc amendment 1 (PREREG.md), written after the results were seen: exploratory",
                 "inputs": files, "thresholds": {"DELTA_log10": A.DELTA, "FLAT_log10": A.FLAT, "FLOOR": A.FLOOR,
                                                 "N_MIN": A.N_MIN, "B": A.B, "ROUTE_SPLIT": A.ROUTE_SPLIT},
                 "prereg_sha256_now": hashlib.sha256((HERE / "PREREG.md").read_bytes()).hexdigest() if (HERE / "PREREG.md").exists() else None,
                 "frozen_prereg_sha256": frozen.get("prereg_sha256") if frozen else None,
                 "checks": {}, "lines": {}, "doses": {}, "matched": {}}  # fmt: skip
    for (dose, seed), ln in sorted(
        lines.items(), key=lambda kv: (A.DOSES.index(kv[0][0]), kv[0][1])
    ):
        out = {"dose": dose, "seed": seed, "base": ln["base"], "end": ln["end"],
               "x_end": ckpts[ln["end"]]["x_dan"] if ln["end"] else None, "router": {}}  # fmt: skip
        if ln["base"] and ln["end"]:
            for d in (RED, *OTHER_DIRS):
                out["router"][d] = router_line(
                    f"{dose}|{seed}|router|{d}", data, classes, ln["base"], ln["end"], d
                )
        else:
            out["missing"] = "base" if ln["base"] is None else "rl_end"
        res["lines"][f"{dose}|{seed}"] = out
        for e in ln["matched"]:
            res["matched"][e] = {"dose": dose, "seed": seed, "base": ln["base"], "end": e, "x": ckpts[e]["x_dan"],
                                 "router": {d: router_line(f"{dose}|{seed}|{e}|router|{d}", data, classes, ln["base"], e, d)
                                            for d in (RED, *OTHER_DIRS)}}  # fmt: skip
    for dose in A.DOSES:
        sl = [{"seed": sd, **lines[(dose, sd)]} for sd in sorted(sd for (ds, sd) in lines if ds == dose)
              if lines[(dose, sd)]["base"] and lines[(dose, sd)]["end"]]  # fmt: skip
        if not sl:
            continue
        dd: dict = {
            "seeds": [x["seed"] for x in sl],
            "x_ends": [ckpts[x["end"]]["x_dan"] for x in sl],
            "router": {},
        }
        for d in (RED, *OTHER_DIRS):
            per = [res["lines"][f"{dose}|{x['seed']}"]["router"][d] for x in sl]
            dv: dict = {"pooled": pooled_dose(dose, sl, data, classes, d)}
            if d == RED:
                dv["verdicts"] = {}
                for name, *_ in COMPARISONS:
                    vs = [p["comparisons"].get(name, {}).get("verdict") for p in per]
                    dv["verdicts"][name] = {"per_seed": vs, "dose": A.dose_reading(vs, A.ROUTES)}
                dv["verdict"] = dv["verdicts"][COMPARISONS[0][0]]["dose"]
            else:
                dv["verdict"] = per[0]["verdict"] if per else None
            groups = sorted({g for p in per for g in p["classes"] if "all" in p["classes"][g]})
            dv["dlogodds10_over_seeds"] = {}
            for g in groups:
                v = [
                    p["classes"][g]["all"]["dlogodds10"]["est"]
                    for p in per
                    if "all" in p["classes"].get(g, {})
                ]
                pts = [
                    p["classes"][g]["all"]["dpoints"]["est"]
                    for p in per
                    if "all" in p["classes"].get(g, {})
                ]
                dv["dlogodds10_over_seeds"][g] = {"mean": float(np.mean(v)), "min": float(min(v)), "max": float(max(v)),
                                                  "dpoints_mean": float(np.mean(pts)), "dpoints_min": float(min(pts)),
                                                  "dpoints_max": float(max(pts))}  # fmt: skip
            dd["router"][d] = dv
        res["doses"][dose] = dd
    # checks
    ch = res["checks"]
    if frozen:
        ch["inputs_match_frozen_results_json"] = frozen.get("inputs") == files
        diffs = []
        for k, ln in res["lines"].items():
            fr = frozen["lines"][k]["router"][RED]["classes"]
            for g in ("required", "idle"):
                if "used_base" in fr.get(g, {}):
                    mine = ln["router"][RED]["classes"][g]["all"]
                    diffs += [
                        abs(mine["base"]["est"] - fr[g]["used_base"]),
                        abs(mine["end"]["est"] - fr[g]["used_end"]),
                    ]
            for g, c in fr.items():  # the same theorems in every frozen class
                mine = ln["router"][RED]["classes"].get(g, {})
                if c["n"] != (mine["all"]["n"] if "all" in mine else 0):
                    diffs.append(math.inf)
        ch["max_abs_diff_vs_frozen_router_uses_anywhere_column"] = max(diffs) if diffs else None
        res["uses_anywhere_readings"] = uses_anywhere_dose_readings(ckpts, data, frozen, lines)
        ch["uses_anywhere_secondary_recomputed_equal"] = not res["uses_anywhere_readings"][
            "mismatches"
        ]
    ch["through_plus_not_through_equals_all"] = True  # asserted in router_from_counts
    ch["U_solved_d_means_accepted_through_d"] = (
        "yes (pc_run.py: ok and d in through), narrower than SPEC.md's 'accepted attempts that use d'"
    )
    ch["qc_warnings_in_rows"] = len(A.WARNINGS)
    return res


# ----------------------------------------------------------------------------- tables
P = A.pct


def lv(c: dict | None) -> str:
    if not c or c.get("est") is None:
        return "n/a"
    s = P(c["est"])
    return s if not c.get("ci") else f"{s} [{P(c['ci'][0])}, {P(c['ci'][1])}]"


def pts(c: dict | None) -> str:
    if not c or c.get("est") is None:
        return "n/a"
    s = f"{100 * c['est']:+.1f}"
    return s if not c.get("ci") else f"{s} [{100 * c['ci'][0]:+.1f}, {100 * c['ci'][1]:+.1f}]"


def lo(c: dict | None) -> str:
    if not c or c.get("est") is None:
        return "n/a"
    s = f"{c['est']:+.2f}"
    return s if not c.get("ci") else f"{s} [{c['ci'][0]:+.2f}, {c['ci'][1]:+.2f}]"


def cnt(c: dict) -> str:
    return f"{int(c['x_base'])}/{int(c['n_base'])} → {int(c['x_end'])}/{int(c['n_end'])}"


def spread(c: dict) -> str:
    a, e = c["theorems_any"], c["theorems_every"]
    return f"{a[0]} / {e[0]} → {a[1]} / {e[1]}"


DL = A.DLAB


def tables(res: dict) -> str:
    th = res["thresholds"]
    L = ["---", "written_on: 2026-11-10", "written_by: agent:claude (generated by posthoc_anywhere.py)", "---", "",
         "# Post-hoc amendment 1: router or default shift with propensity = uses d anywhere", "",
         "**Exploratory. Written after the results were seen** (PREREG.md, \"Post-hoc amendment 1 (after results)\"). "
         "It does not replace the frozen tables in `tables.md`. Generated by `posthoc_anywhere.py`; every number below is "
         "its output.", "",
         f"Measure: per class, Σ used / Σ k over the class's theorems (U rows, k = 32 attempts per theorem): the share of "
         "attempts that open d somewhere (for the reductio: a box ¬¬X with binder ¬X on any formula X, `pc_lib.moves`). "
         "Change: log10 odds of (x + 0.5) / (n + 1) of the pooled counts, end minus base. Intervals: 95 % percentile "
         f"bootstrap over theorems, {th['B']} resamples, paired (the same resampled theorems at base and end), resampled "
         "within class, with the frozen router cell's RNG keys for the frozen classes. States (frozen): up if Δ ≥ "
         f"{th['DELTA_log10']} and the interval is above 0; down symmetric; flat if the interval lies inside ±{th['FLAT_log10']}; "
         f"rare if below {P(th['FLOOR'])} % at both checkpoints; a class needs n ≥ {th['N_MIN']}. Verdict (frozen): no rise to "
         "locate unless the required class is up; router if the idle class is flat, rare or down, or the interval of "
         f"Δ(idle) − {th['ROUTE_SPLIT']} Δ(required) lies below 0; default shift if idle is up and that interval lies above 0; "
         "otherwise unresolved. Dose verdict: the frozen agreement rule over seeds.", "",
         f"PREREG sha256 when the frozen tables were written `{res['frozen_prereg_sha256']}`; now (with the amendment) "
         f"`{res['prereg_sha256_now']}`.", ""]  # fmt: skip
    ch = res["checks"]
    L += ["## Checks", ""]
    L += [f"- {k.replace('_', ' ')}: {v}" for k, v in ch.items()]
    L += ["", "## Figure", "",
          "![uses of the reductio, required against idle](posthoc_anywhere.png)", "",
          "Attempts that use a reductio anywhere (% of all attempts), base → RL end, on the frozen classes: required "
          "(vermillion, solid; the strict 106 and the other set's 30) and idle (blue, dashed). One panel per dose, one line "
          "per seed (marker shape), hollow at the base and filled at the RL end; bars are 95 % bootstrap intervals over "
          "theorems. Doses are at unequal RL compute (x in the panel titles). Matched-compute checkpoints are not drawn.", ""]  # fmt: skip

    # ---- reductio dose verdicts
    L += ["## Reductio: dose verdicts", "",
          "Required (frozen) = every theorem whose `dir_class` is required: the strict 106 and 30 classical-only theorems "
          "of the other set (136). Idle: 194 theorems of the other set where the shortest-proof search found no shorter "
          "proof with DN than without. Each dose's RL end is at its own compute (x in the last column); doses are not "
          "compared.", "",
          "| comparison | status | dose | per seed | dose verdict | x at RL ends |", "|---|---|---|---|---|---|"]  # fmt: skip
    for name, _, _, _, status in COMPARISONS:
        for dose, dd in res["doses"].items():
            v = dd["router"][RED]["verdicts"][name]
            xs = ", ".join(f"{x:.2f}" for x in dd["x_ends"])
            L.append(f"| {name} | {status} | {DL[dose]} | {' / '.join(r or 'missing' for r in v['per_seed'])} | "
                     f"**{v['dose']}** | {xs} |")  # fmt: skip
    L += [
        "",
        "Mean (range) over seeds of the change, all attempts:",
        "",
        "| dose | class | Δ points, mean (range) | Δlog10 odds, mean (range) |",
        "|---|---|---|---|",
    ]
    for dose, dd in res["doses"].items():
        for g, v in dd["router"][RED]["dlogodds10_over_seeds"].items():
            L.append(f"| {DL[dose]} | {GROUP_LABEL[g]} | {100 * v['dpoints_mean']:+.1f} ({100 * v['dpoints_min']:+.1f} to "
                     f"{100 * v['dpoints_max']:+.1f}) | {v['mean']:+.2f} ({v['min']:+.2f} to {v['max']:+.2f}) |")  # fmt: skip

    # ---- per seed, frozen classes, all attempts
    L += ["", "## Reductio per seed: uses anywhere, % of all attempts, base → RL end", "",
          "Helpful (n = 8) is reported with no state and no reading. Counts are uses / attempts. Spread: theorems of the "
          "class with any attempt using a reductio / with every one of the 32 attempts using one, base → end.", "",
          "| dose | seed | x | class | n | base | end | uses / attempts | spread | Δ points | Δlog10 odds | state |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]  # fmt: skip
    for ln in res["lines"].values():
        r = ln["router"].get(RED)
        if not r:
            L.append(
                f"| {DL[ln['dose']]} | {ln['seed']} | | missing {ln.get('missing')} | | | | | | | | |"
            )
            continue
        for g in ("required", "helpful", "idle", "required_strict106", "required_other"):
            c = r["classes"].get(g, {}).get("all")
            if not c:
                continue
            L.append(f"| {DL[ln['dose']]} | {ln['seed']} | {ln['x_end']:.2f} | {GROUP_LABEL[g]} | {c['n']} | {lv(c['base'])} | "
                     f"{lv(c['end'])} | {cnt(c)} | {spread(c)} | {pts(c['dpoints'])} | {lo(c['dlogodds10'])} | {c['state']} |")  # fmt: skip
    L += ["", "### The rule per seed", "",
          "Difference = Δ(required side) − Δ(idle). f = Δ(idle) / Δ(required side), log10 odds.", "",
          "| dose | seed | comparison | difference, points | difference, log10 odds | interval of Δ(idle) − ½ Δ(required) | f | verdict |",
          "|---|---|---|---|---|---|---|---|"]  # fmt: skip
    for ln in res["lines"].values():
        r = ln["router"].get(RED)
        if not r:
            continue
        for name, *_ in COMPARISONS:
            c = r["comparisons"][name]
            if "diff_points" not in c:
                L.append(f"| {DL[ln['dose']]} | {ln['seed']} | {name} | | | | | {c['verdict']} |")
                continue
            dci = c["not_needed_minus_half_required_ci"]
            f = "n/a" if c["spillover_f"] is None else f"{c['spillover_f']:+.2f}"
            L.append(f"| {DL[ln['dose']]} | {ln['seed']} | {name} | {pts(c['diff_points'])} | {lo(c['diff_logodds10'])} | "
                     f"[{dci[0]:+.2f}, {dci[1]:+.2f}] | {f} | **{c['verdict']}**{' (' + c['flag'] + ')' if c.get('flag') else ''} |")  # fmt: skip

    # ---- pooled per dose
    L += ["", "## Reductio pooled per dose (descriptive, no verdict)", "",
          "Per theorem, the three seeds' counts summed at base and at end; theorems resampled, the same in every seed. The "
          "interval covers theorem sampling only, not seed-to-seed variation, so it is narrower than the uncertainty about "
          "the recipe. The dose verdict is the agreement rule above, never this row.", "",
          "| dose | class | n | base | end | Δ points | Δlog10 odds |", "|---|---|---|---|---|---|---|"]  # fmt: skip
    for dose, dd in res["doses"].items():
        pr = dd["router"][RED]["pooled"]
        for g in ("required", "helpful", "idle", "required_strict106", "required_other"):
            c = pr["classes"].get(g, {}).get("all")
            if c:
                L.append(
                    f"| {DL[dose]} | {GROUP_LABEL[g]} | {c['n']} | {lv(c['base'])} | {lv(c['end'])} | {pts(c['dpoints'])} | {lo(c['dlogodds10'])} |"
                )
        for name, *_ in COMPARISONS[:3]:
            c = pr["comparisons"].get(name, {})
            if "diff_points" in c:
                L.append(
                    f"| {DL[dose]} | difference: {name} | | | | {pts(c['diff_points'])} | {lo(c['diff_logodds10'])} |"
                )

    # ---- the split by success
    L += ["", "## Reductio: how much of \"uses anywhere\" is carried by accepted proofs", "",
          "What the rows hold: `used` on U/reductio = attempts that open a reductio anywhere; `solved` on U/reductio = "
          "accepted attempts whose proof goes *through* the reductio (`pc_run.py` line 401, `pc_lib.classify`: its DN line "
          "is on the conclusion's dependency path); `solved` on U/any = accepted attempts. SPEC.md describes U/<d> `solved` "
          "as \"accepted attempts that use d\"; the runner's field is narrower, since an accepted proof can open a reductio "
          "as a detour. So, exactly, for every class: uses = (accepted, through the reductio) + (opens it, but it carries no "
          "accepted proof). The second part divided by the attempts that are not accepted-through-the-reductio is the "
          "sensitivity measure (\"excluding proofs through d\"). Where the last column is 0 (no theorem-checkpoint with an "
          "accepted attempt whose proof avoids the reductio), it is exactly \"uses among attempts not accepted\". It is not tied to success mechanically, but it is conditioned on "
          "not succeeding, and the attempts that still fail at the RL end are a different, harder pool than at the base. "
          "What cannot be separated: (a) for the idle class and the other set's 30, how many accepted attempts open a "
          "reductio as a detour against how many failed attempts open one (the rows give only the sum); (b) on the required "
          "side, whether the part carried by accepted proofs rose because the model chooses the reductio more or because it "
          "finishes it more (there is no forced arm for a reductio opened later; F forces the opening on the goal).", "",
          "| dose | seed | class | accepted, any route | through the reductio, % of all | opens it, carries no accepted proof, % of all | through it, % of accepted (counts) | uses, excluding proofs through it | cells with an accepted proof not through it |",
          "|---|---|---|---|---|---|---|---|---|"]  # fmt: skip
    for ln in res["lines"].values():
        r = ln["router"].get(RED)
        if not r:
            continue
        for g in ("required", "idle", "required_strict106", "required_other", "helpful"):
            c = r["classes"].get(g)
            if not c or "all" not in c:
                continue
            ac, th_, nt, aa, ex = (
                c["accepted"],
                c["through"],
                c["not_through"],
                c["among_accepted"],
                c["excl_through"],
            )
            L.append(f"| {DL[ln['dose']]} | {ln['seed']} | {GROUP_LABEL[g]} | {P(ac['base']['est'])} → {P(ac['end']['est'])} | "
                     f"{P(th_['base']['est'])} → {P(th_['end']['est'])} | {P(nt['base']['est'])} → {P(nt['end']['est'])} | "
                     f"{P(aa['base']['est'])} → {P(aa['end']['est'])} ({cnt(aa)}) | "
                     f"{lv(ex['base'])} → {lv(ex['end'])}, Δlog10 odds {lo(ex['dlogodds10'])}, {ex['state']} | "
                     f"{c['cells_accepted_not_through']} of {c['cells']} |")  # fmt: skip

    # ---- other directions
    L += ["", "## Negation, conditional, cases: no required class, no router reading", "",
          "`dir_class` is null for these directions (`minlen` cannot forbid an introduction on the goal or a premise case "
          "split), so the frozen rule falls back to the optional theorems as a whole and has no required class to compare "
          "them with. Reported: the optional class's uses, both sets together as in the frozen router cell. Negation and "
          "conditional count only on the goal in focus.", "",
          "| dose | seed | direction | n (strict / other) | base | end | Δ points | Δlog10 odds | state |",
          "|---|---|---|---|---|---|---|---|---|"]  # fmt: skip
    for ln in res["lines"].values():
        for d in OTHER_DIRS:
            r = ln["router"].get(d)
            if not r:
                continue
            c = r["classes"].get("optional", {})
            if "all" not in c:
                L.append(f"| {DL[ln['dose']]} | {ln['seed']} | {d} | 0 | | | | | {r['verdict']} |")
                continue
            nb = c["n_by_set"]
            a = c["all"]
            L.append(f"| {DL[ln['dose']]} | {ln['seed']} | {d} | {a['n']} ({nb['strict106']} / {nb['transfer_other']}) | {lv(a['base'])} | "
                     f"{lv(a['end'])} | {pts(a['dpoints'])} | {lo(a['dlogodds10'])} | {a['state']} |")  # fmt: skip
    L += ["", "| dose | direction | pooled over seeds: base → end | Δ points (theorem-only interval) | Δ points over seeds, mean (range) |",
          "|---|---|---|---|---|"]  # fmt: skip
    for dose, dd in res["doses"].items():
        for d in OTHER_DIRS:
            pr = dd["router"][d]["pooled"]
            c = pr.get("classes", {}).get("optional", {}).get("all")
            o = dd["router"][d]["dlogodds10_over_seeds"].get("optional")
            if c and o:
                L.append(f"| {DL[dose]} | {d} | {P(c['base']['est'])} → {P(c['end']['est'])} | {pts(c['dpoints'])} | "
                         f"{100 * o['dpoints_mean']:+.1f} ({100 * o['dpoints_min']:+.1f} to {100 * o['dpoints_max']:+.1f}) |")  # fmt: skip

    # ---- elicitation vs capability with uses anywhere
    ua = res.get("uses_anywhere_readings")
    if ua:
        L += ["", "## Reductio: elicitation against capability with propensity = uses anywhere", "",
              "The per-seed readings are the frozen secondary in `tables.md` (\"Secondary: the same rule with propensity = "
              "uses d anywhere (U)\"), recomputed here with `analyze.line_cell(..., u_prop=True)`: "
              f"{'all agree with results.json' if not ua['mismatches'] else 'MISMATCHES: ' + ', '.join(ua['mismatches'])}. "
              "tables.md gives no dose-level reading for this secondary; the dose column applies the frozen agreement rule.", "",
              "| dose | set | per seed | dose reading |", "|---|---|---|---|"]  # fmt: skip
        for k, v in ua["dose"].items():
            dose, s = k.split("|")
            L.append(
                f"| {DL[dose]} | {s} | {' / '.join(r or 'missing' for r in v['readings'])} | **{v['reading']}** |"
            )
        L += [
            "",
            "| dose | seed | set | uses, base → end | P state | C state | reading | primary reading |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for k, v in ua["per_seed"].items():
            dose, sd, s, _ = k.split("|")
            L.append(f"| {DL[dose]} | {sd} | {s} | {P(v['P_base'])} → {P(v['P_end'])} | {v['P_state']} | {v['C_state']} | "
                     f"{v['reading']} | {v['primary_reading']} |")  # fmt: skip

    # ---- matched compute
    if res["matched"]:
        L += ["", "## Matched compute (separate: never pooled, never in a dose verdict)", "",
              "Base → the checkpoint of the 0 % or 0.1 % line at the RL compute where the 1 % lines end. Uses anywhere, "
              "% of all attempts.", "",
              "| checkpoint | dose | seed | x | class | n | base | matched | Δ points | Δlog10 odds | state | verdict (frozen classes) |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]  # fmt: skip
        for e, m in res["matched"].items():
            r = m["router"][RED]
            first = True
            for g in ("required", "idle", "required_strict106", "required_other"):
                c = r["classes"].get(g, {}).get("all")
                if not c:
                    continue
                L.append(f"| {e} | {DL[m['dose']]} | {m['seed']} | {m['x']:.2f} | {GROUP_LABEL[g]} | {c['n']} | {lv(c['base'])} | "
                         f"{lv(c['end'])} | {pts(c['dpoints'])} | {lo(c['dlogodds10'])} | {c['state']} | {r['verdict'] if first else ''} |")  # fmt: skip
                first = False
        L += [
            "",
            "| checkpoint | " + " | ".join(name for name, *_ in COMPARISONS) + " |",
            "|---|" + "---|" * len(COMPARISONS),
        ]
        for e, m in res["matched"].items():
            cs = m["router"][RED]["comparisons"]
            L.append(
                f"| {e} | " + " | ".join(cs[name]["verdict"] for name, *_ in COMPARISONS) + " |"
            )
        if ua and ua["matched"]:
            L += [
                "",
                "| checkpoint | set | x | reading with uses anywhere | primary reading |",
                "|---|---|---|---|---|",
            ]
            for k, v in ua["matched"].items():
                e, s = k.split("|")
                L.append(f"| {e} | {s} | {v['x']:.2f} | {v['reading']} | {v['primary_reading']} |")
    L += ["", "## Inputs", ""]
    L += [f"- `{f}` sha256 `{h}`" for f, h in res["inputs"].items()]
    return "\n".join(L) + "\n"


# ----------------------------------------------------------------------------- figure
def load_style():
    """code/ndbench/figures/style.py: through the package when the repository root is on the path (as fig.py), else
    by file path (it imports nothing from the repository)."""
    try:
        from code.ndbench.figures import style  # type: ignore[import-not-found]  # noqa: PLC0415

        return style
    except ImportError:
        p = HERE.parents[2] / "ndbench" / "figures" / "style.py"
        spec = importlib.util.spec_from_file_location("ndbench_figures_style", p)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        return mod


def render(res: dict, out: Path) -> list[Path]:
    """Uses a reductio anywhere, base → RL end, required (frozen, vermillion, solid) against idle (blue, dashed); one
    panel per dose, one line per seed (marker shape), hollow at the base and filled at the RL end; bars are 95 %
    bootstrap intervals over theorems."""
    S = load_style()
    from matplotlib.lines import Line2D  # noqa: PLC0415

    OI = S.OKABE_ITO
    sty = {"required": (OI["vermillion"], "-"), "idle": (OI["blue"], (0, (4, 1.6)))}
    marks = ("o", "s", "^")
    off = (-0.07, 0.0, 0.07)
    with S.paper_style():
        fig = S.new_figure(S.FULL_W, 2.35)
        axes = fig.subplots(1, len(res["doses"]), sharey=True)
        axes = np.atleast_1d(axes)
        ymax = 0.0
        ns: dict = {}
        for ax, (letter, (dose, dd)) in zip(axes, zip("abc", res["doses"].items())):
            for sd in dd["seeds"]:
                r = res["lines"][f"{dose}|{sd}"]["router"][RED]
                for g, (col, ls) in sty.items():
                    c = r["classes"][g]["all"]
                    ns[g] = c["n"]
                    xs = np.array([0.0, 1.0]) + off[sd % 3]
                    ys = [c["base"]["est"], c["end"]["est"]]
                    ax.plot(xs, ys, color=col, ls=ls, lw=0.9, zorder=2)
                    for x, key, filled in ((xs[0], "base", False), (xs[1], "end", True)):
                        lo_, hi_ = c[key]["ci"]
                        ymax = max(ymax, hi_)
                        ax.plot(
                            [x, x],
                            [lo_, hi_],
                            color=col,
                            lw=0.5,
                            alpha=0.7,
                            zorder=2,
                            solid_capstyle="butt",
                        )
                        ax.plot(x, c[key]["est"], marker=marks[sd % 3], ms=3.4, ls="none", zorder=3,
                                mfc=col if filled else S.WHITE, mec=col, mew=0.75)  # fmt: skip
            xe = dd["x_ends"]
            S.panel_title(
                ax,
                letter,
                f"{DL[dose]} dose, RL end x {min(xe):.2g}–{max(xe):.2g}",
                size=S.FS_TICK + 0.5,
            )
            ax.set_xticks([0, 1], ["base", "RL end"])
            ax.set_xlim(-0.35, 1.35)
            S.light_grid(ax, "y")
        for ax in axes:
            ax.set_ylim(0, ymax * 1.08)
            S.pct_axis(ax, "y")
        axes[0].set_ylabel("attempts using a reductio anywhere")
        hs = [Line2D([], [], color=sty["required"][0], ls=sty["required"][1], lw=0.9,
                     label=f"required (n = {ns.get('required', '?')}: strict 106 + 30 other)"),
              Line2D([], [], color=sty["idle"][0], ls=sty["idle"][1], lw=0.9, label=f"idle (n = {ns.get('idle', '?')})")]  # fmt: skip
        hs += [Line2D([], [], ls="none", marker=m, mfc=S.WHITE, mec=S.INK2, mew=0.75, ms=3.4, label=f"seed {i}")
               for i, m in enumerate(marks)]  # fmt: skip
        hs.append(
            Line2D(
                [],
                [],
                ls="none",
                marker="o",
                mfc=S.INK2,
                mec=S.INK2,
                ms=3.4,
                label="filled: RL end",
            )
        )
        fig.legend(
            handles=hs, loc="outside lower center", ncol=6, frameon=False, fontsize=S.FS_TICK - 0.5
        )
        paths = S.save(fig, out)
        S.close(fig)
    return paths


# ----------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="/scratch/cowifornia/nd-rl/propensity_capability/results")
    ap.add_argument("--frozen", default=str(HERE / "results" / "results.json"))
    ap.add_argument("--out", default=str(HERE / "results"))
    ap.add_argument("--no-fig", action="store_true")
    a = ap.parse_args()
    res = posthoc(Path(a.results), Path(a.frozen))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "posthoc_anywhere.json").write_text(json.dumps(res, indent=1, default=float) + "\n")
    (out / "posthoc_anywhere.md").write_text(tables(res))
    written = [out / "posthoc_anywhere.json", out / "posthoc_anywhere.md"]
    if not a.no_fig:
        written += render(res, out / "posthoc_anywhere")
    v = {d: dd["router"][RED]["verdict"] for d, dd in res["doses"].items()}
    print(f"reductio dose verdicts (frozen classes, all attempts): {v}; checks: {res['checks']}")
    for p in written:
        print(p)


if __name__ == "__main__":
    main()
