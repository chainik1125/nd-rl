#!/usr/bin/env python3
"""Quantities and the pre-registered reading of the weight-interpolation run (serrano, CPU).

    python3 analyze.py [--data /scratch/cowifornia/nd-rl/olmo_elicit/interp] [--out DIR]

Reads, under <data>: runs/cell_<tag>/out/{samples.jsonl.gz, graded.jsonl, meta.jsonl, wrap.json}
for the tags a100, a050, a075, a025 (a cell that is absent is reported as absent),
inputs/spec_reach4.txt, refs/refs_reach.json and, when fetched, runs/write_<tag>/out/check.json.
Writes, under <out> (default <data>/analysis): results.json, results.txt (plain sentences) and
cells_manifest.json (the per-draw files the blinded reader step loads: one folder per cell with
samples.jsonl.gz and graded.jsonl, rows keyed by prompt, chunk, i).

Rates are per-prompt shares averaged over the 48 prompts; intervals are 95 % percentile intervals of
a bootstrap over prompts in which every cell is resampled with the same prompt draws (paired).
A cell with missing draws is given three ways: as sampled, every missing draw a fail, every missing
draw a pass. Missing draws are never dropped: `complete` is false unless all 192 rows are there.
The reading rule and its labels are PREREG.md's; Amendment 1 adds the smooth reference (what a
path with no switch would give) and a sensitivity line against the stored s2300 rates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cell_count as cc  # noqa: E402

ALPHA = {"a025": 0.25, "a050": 0.5, "a075": 0.75, "a100": 1.0}
SEED_CELL = 710012
REPS, SEED_BOOT = 10000, 20261128
#: The verdicts of the reading rule. Thresholds as written at 2:56 am; names from PREREG Amendment 2
#: (coordinator's decision): none of them says anything about a switch.
VERDICTS = (
    "about proportional",
    "below proportional at the midpoint",
    "above proportional",
    "unclear at this size",
)
#: The second line (PREREG A1.5): the alpha 0.5 share against the smooth reference.
SHARPER = "sharper than the smooth reference"
COMPATIBLE = "compatible with a smooth path"
EARLIER = "earlier than the smooth reference"
NOT_EVALUATED = "not evaluated"
#: Amendment 2: the only wording that names a switch. It may be written only when the verdict is
#: "below proportional at the midpoint" and the second line is "sharper than the smooth reference".
SWITCH_SENTENCE = "a switch, as in the small provers"
#: PREREG "Predictions" (fractions): point and range of the orchestrator, range of the coordinator
PREDICTIONS = {
    "orchestrator": {
        "a025": {"point": 0.015, "range": [0.0, 0.04]},
        "a050": {"point": 0.09, "range": [0.04, 0.18]},
        "a075": {"point": 0.33, "range": [0.22, 0.46]},
        "a100": {"point": 0.69, "range": [0.60, 0.78]},
        "verdict": VERDICTS[1],
        "reference": COMPATIBLE,
    },
    "coordinator": {
        "a025": {"range": [0.0, 0.03]},
        "a050": {"range": [0.05, 0.20]},
        "a075": {"range": [0.35, 0.55]},
        "verdict": VERDICTS[1],
        "reference": COMPATIBLE,
    },
}


#: PREREG Amendment 1: the smooth reference. Each prompt's log-odds of passing moves in a straight
#: line in alpha, from the pre-RL end (one value for every prompt; two choices, because 0 of 768
#: only bounds it) to its stored s2300 rate (32 draws, half a pass added to each side).
REFERENCE_P0 = {
    "p0 = 1/384 (fresh dpo, 1 pass)": 1 / 384,
    "p0 = 0.5/769 (stored dpo, 0 of 768)": 0.5 / 769,
}
STORED_DRAWS = 32
#: the small provers of 2026-09-30, sealed set: 0.31 % at alpha 0.5, 12.2 % at 0.75, 19.0 % at 1
SMALL_PROVERS = {"r050": 0.31 / 19.0, "r075": 12.2 / 19.0}


def smooth_reference(stored: np.ndarray) -> dict:
    """The curve a smooth (logit-linear) path would give, from stored end-point rates only."""
    ps = (STORED_DRAWS * np.asarray(stored, dtype=float) + 0.5) / (STORED_DRAWS + 1)
    lg = np.log(ps / (1 - ps))
    out = {}
    for name, p0 in REFERENCE_P0.items():
        l0 = np.log(p0 / (1 - p0))
        rates = {
            a: float(np.mean(1 / (1 + np.exp(-((1 - a) * l0 + a * lg))))) for a in ALPHA.values()
        }
        grid = np.linspace(0, 1, 41)
        curve = [float(np.mean(1 / (1 + np.exp(-((1 - a) * l0 + a * lg))))) for a in grid]
        out[name] = {
            "curve": {"alpha": [float(a) for a in grid], "rate": curve},
            "rates": {str(a): v for a, v in rates.items()},
            "r050": rates[0.5] / rates[1.0],
            "r075": rates[0.75] / rates[1.0],
            "half_gain": half_gain(rates)["alpha"],
        }
    return out


def against_reference(lo: float | None, hi: float | None, ref: dict) -> str:
    """Observed r(0.5) interval against the two reference values of r(0.5)."""
    vals = [v["r050"] for v in ref.values()]
    if lo is None or hi is None:
        return NOT_EVALUATED
    if hi < min(vals):
        return SHARPER
    if lo > max(vals):
        return EARLIER
    return COMPATIBLE


def both_lines(verdict_name: str, reference_reading: str) -> str:
    """What may be said from the rule's verdict and the second line together (Amendment 2)."""
    if reference_reading == SHARPER:
        if verdict_name == VERDICTS[1]:
            return f"Both lines together: {SWITCH_SENTENCE}."
        return (
            "The alpha 0.5 share is below the smooth reference, but the rule's verdict is not "
            f'"{VERDICTS[1]}", so no switch is claimed.'
        )
    if reference_reading == COMPATIBLE:
        return "The curve is what a gradual change in each prompt's log-odds would give."
    if reference_reading == EARLIER:
        return (
            "The pass rate rises earlier than a gradual change in each prompt's log-odds would "
            "give."
        )
    return "Without the smooth reference nothing is said about a switch."


def verdict(r: float, lo: float, hi: float) -> str:
    """The reading rule of PREREG.md on r(0.5) and its 95 % paired interval."""
    if not all(np.isfinite([r, lo, hi])):
        return "no reading"
    if 0.35 <= r <= 0.65 and lo >= 0.2 and hi <= 0.8:
        return VERDICTS[0]
    if r < 0.35 and hi < 0.5:
        return VERDICTS[1]
    if r > 0.65 and lo > 0.5:
        return VERDICTS[2]
    return VERDICTS[3]


def half_gain(rates: dict[float, float]) -> dict:
    """Smallest alpha at which the piecewise-linear curve through (0, 0) and the measured cells
    reaches half of the alpha 1.0 rate. `rates` maps alpha -> rate and must contain 1.0."""
    xs = [0.0] + sorted(rates)
    ys = [0.0] + [rates[a] for a in sorted(rates)]
    top = rates[1.0]
    mono = all(b >= a for a, b in zip(ys, ys[1:], strict=False))
    if not np.isfinite(ys).all() or top <= 0:
        return {"alpha": None, "below_first_cell": False, "monotone": bool(mono)}
    target = top / 2
    for x0, x1, y0, y1 in zip(xs, xs[1:], ys, ys[1:], strict=False):
        if y0 < target <= y1:
            a = x0 + (target - y0) / (y1 - y0) * (x1 - x0)
            return {"alpha": float(a), "below_first_cell": x0 == 0.0, "monotone": bool(mono)}
    return {"alpha": None, "below_first_cell": False, "monotone": bool(mono)}


def answer_of(text: str) -> str:
    i = text.rfind("</think>")
    return text[i + 8 :].strip() if i >= 0 else ""


def answer_style(answers: list[str]) -> dict:
    """Metrics 2's statistic (tools/answer_style.py): no . ! ? in the answer = unpunctuated."""
    if not answers:
        return {"passes": 0, "unpunctuated_share": None, "median_words": None}
    nop = float(np.mean([not re.search(r"[.!?]", a) for a in answers]))
    return {
        "passes": len(answers),
        "unpunctuated_share": nop,
        "median_words": float(np.median([len(a.split()) for a in answers])),
    }


def md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def load_cell(
    d: Path,
    spec: dict[str, int],
    tag: str,
    prompt_sha: dict | None = None,
    models_root: str = "/workspace/models",
    merged_root: str = "/workspace/wi/models",
) -> dict:
    """Per-prompt counts and the per-draw facts of one cell folder."""
    model_dir = f"{models_root}/s2300" if tag == "a100" else f"{merged_root}/interp_{tag}"
    cnt = cc.count(d, spec, tag, SEED_CELL, model_dir, prompt_sha)
    raw, fixed = cc.sample_files(d)
    use = raw if cnt["gzip_whole"] or fixed is None else fixed
    rows, _ = cc.read_gz_rows(use) if use.exists() else ([], False)
    text = {cc.key(r): r.get("text", "") for r in rows}
    graded = cc.read_jsonl(d / "graded.jsonl") if (d / "graded.jsonl").exists() else []
    plan = cc.planned_keys(spec)
    g = {cc.key(r): r for r in graded if cc.key(r) in plan and isinstance(r.get("strict"), bool)}
    pids = list(spec)
    k = np.array([sum(g[x]["strict"] for x in g if x[0] == p) for p in pids], dtype=float)
    m = np.array([sum(1 for x in g if x[0] == p) for p in pids], dtype=float)
    n = np.array([spec[p] for p in pids], dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        sampled = np.where(m > 0, k / np.where(m > 0, m, 1), np.nan)
    toks = [r.get("n_tok", 0) for r in g.values()]
    return {
        "tag": tag,
        "alpha": ALPHA[tag],
        "pids": pids,
        "dir": str(d),
        "count": cnt,
        "complete": bool(cnt["complete"]),
        "planned": int(n.sum()),
        "finished": int(m.sum()),
        "missing": int(n.sum() - m.sum()),
        "passes": int(k.sum()),
        "per_prompt": {"sampled": sampled, "fail": k / n, "pass": (k + n - m) / n},
        "mean_tokens": float(np.mean(toks)) if toks else None,
        "cut_at_cap": sum(r.get("finish") == "length" for r in g.values()),
        "no_end_of_thinking": sum(not r.get("think_end", True) for r in g.values()),
        "style": answer_style(
            [answer_of(text[x]) for x, r in g.items() if r["strict"] and x in text]
        ),
        "files": {"samples": str(use), "graded": str(d / "graded.jsonl")},
    }


def _pct(v: np.ndarray) -> list[float | None]:
    ok = v[np.isfinite(v)]
    if not len(ok):
        return [None, None]
    return [float(np.percentile(ok, 2.5)), float(np.percentile(ok, 97.5))]


def _boot_means(arr: np.ndarray, idx: np.ndarray) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # a resample with no finished draw: NaN
        return np.nanmean(arr[idx], axis=1)


def _ratio(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)


def analyse(
    cells: dict[str, dict],
    refs: dict | None,
    reps: int = REPS,
    seed: int = SEED_BOOT,
    writes: dict | None = None,
):
    """cells: tag -> load_cell result (absent tags left out); writes: tag -> the write check's
    summary (realised fraction along the line). Returns the results dict."""
    n_p = len(next(iter(cells.values()))["per_prompt"]["fail"]) if cells else 0
    idx = np.random.default_rng(seed).integers(0, n_p, (reps, n_p)) if n_p else None
    res: dict = {"reps": reps, "seed": seed, "prompts": n_p, "cells": {}, "absent": []}
    boot: dict[str, dict[str, np.ndarray]] = {}
    for tag in ALPHA:
        if tag not in cells:
            res["absent"].append(tag)
            continue
        c = cells[tag]
        boot[tag] = {v: _boot_means(c["per_prompt"][v], idx) for v in ("sampled", "fail", "pass")}
        row = {k: c[k] for k in ("alpha", "complete", "planned", "finished", "missing", "passes")}
        row["prompts_without_rows"] = int(np.isnan(c["per_prompt"]["sampled"]).sum())
        for v, name in (
            ("sampled", "pass1"),
            ("fail", "pass1_all_fail"),
            ("pass", "pass1_all_pass"),
        ):
            row[name] = float(np.nanmean(c["per_prompt"][v])) if c["finished"] else None
            row[name + "_ci"] = _pct(boot[tag][v])
        if not c["finished"]:
            row["pass1_all_fail"], row["pass1_all_pass"] = 0.0, 1.0
        row.update(
            mean_tokens=c["mean_tokens"],
            cut_at_cap=c["cut_at_cap"],
            no_end_of_thinking=c["no_end_of_thinking"],
            style=c["style"],
            problems=c["count"]["problems"] + c["count"]["settings_problems"],
        )
        res["cells"][tag] = row
    have_top = "a100" in cells and cells["a100"]["finished"] > 0
    res["ratios"], res["reading"] = {}, {"verdict": "no reading", "why": ""}
    if not have_top:
        res["reading"]["why"] = "the alpha 1.0 cell has no rows"
    elif res["cells"]["a100"]["pass1"] == 0:
        res["reading"]["why"] = "the alpha 1.0 rate is 0 %"
    else:
        top = cells["a100"]["per_prompt"]
        for tag in ("a025", "a050", "a075"):
            if tag not in cells or not cells[tag]["finished"]:
                continue
            pp = cells[tag]["per_prompt"]
            # as sampled; lowest ratio (cell all-fail over yardstick all-pass); highest (reverse)
            combos = {
                "sampled": ("sampled", "sampled"),
                "low": ("fail", "pass"),
                "high": ("pass", "fail"),
            }
            out = {}
            for name, (vn, vd) in combos.items():
                r = (
                    float(np.nanmean(pp[vn]) / np.nanmean(top[vd]))
                    if np.nanmean(top[vd]) > 0
                    else None
                )
                b = _ratio(boot[tag][vn], boot["a100"][vd])
                out[name] = {
                    "r": r,
                    "ci": _pct(b),
                    "undefined_resamples": int((~np.isfinite(b)).sum()),
                }
            res["ratios"][tag] = out
        if "a050" in res["ratios"]:
            rr = res["ratios"]["a050"]
            vs = {
                k: verdict(v["r"], *v["ci"])
                if v["r"] is not None and None not in v["ci"]
                else "no reading"
                for k, v in rr.items()
            }
            cut = not (cells["a100"]["complete"] and cells["a050"]["complete"])
            rows_cut = cells["a100"]["missing"] or cells["a050"]["missing"]
            if not rows_cut:
                res["reading"] = {
                    "verdict": vs["sampled"],
                    "why": "both cells have every planned draw",
                }
            elif len(set(vs.values())) == 1:
                res["reading"] = {
                    "verdict": vs["sampled"],
                    "why": "same verdict as sampled and at both bounds",
                }
            else:
                res["reading"] = {
                    "verdict": "unclear at this size (cell cut)",
                    "why": f"verdicts differ: {vs}",
                }
            res["reading"].update(by_bound=vs, cells_complete=not cut)
            r50 = rr["sampled"]["r"]
            # PREREG A1.6: this close to a band edge the verdict is wrong about 1 time in 6
            res["reading"]["near_boundary"] = bool(
                r50 is not None and min(abs(r50 - 0.35), abs(r50 - 0.65)) <= 0.05
            )
            if cut and not rows_cut:
                res["reading"]["why"] += "; a settings check failed, see the cells' problems"
        else:
            res["reading"]["why"] = "the alpha 0.5 cell has no rows"
        # half-gain point on the as-sampled rates
        tags = [t for t in ALPHA if t in cells and cells[t]["finished"]]
        point = half_gain({ALPHA[t]: res["cells"][t]["pass1"] for t in tags})
        hb = np.full(reps, np.nan)
        below = 0
        for b in range(reps):
            h = half_gain({ALPHA[t]: float(boot[t]["sampled"][b]) for t in tags})
            if h["alpha"] is not None:
                hb[b] = h["alpha"]
                below += h["below_first_cell"]
        res["half_gain"] = dict(
            point, ci=_pct(hb), cells_used=tags, share_below_first_cell=below / reps
        )
        # the same point on the realised axis (where the rounded weights really are along the line)
        real = {t: (writes or {}).get(t, {}).get("realised_fraction") for t in tags if t != "a100"}
        if real and all(v is not None for v in real.values()):
            x = {t: (1.0 if t == "a100" else float(real[t])) for t in tags}
            if len(set(x.values())) == len(x):
                hr = half_gain({x[t]: res["cells"][t]["pass1"] for t in tags})
                hb = np.full(reps, np.nan)
                for b in range(reps):
                    h = half_gain({x[t]: float(boot[t]["sampled"][b]) for t in tags})
                    if h["alpha"] is not None:
                        hb[b] = h["alpha"]
                res["half_gain_realised"] = dict(hr, ci=_pct(hb), axis=x)
    res["small_provers"] = SMALL_PROVERS
    if refs and cells:
        pids = next(iter(cells.values()))["pids"]  # the cells' prompt order
        stored = np.array([refs["per_prompt"]["s2300"][p] for p in pids], dtype=float)
        res["reference"] = smooth_reference(stored)
        r50 = res["ratios"].get("a050", {}).get("sampled")
        res["reference_reading"] = against_reference(
            *(r50["ci"] if r50 else (None, None)), res["reference"]
        )
        # sensitivity only: the same ratios with the stored 32-draw s2300 rates as the denominator
        res["ratios_vs_stored"] = {}
        for tag in ("a025", "a050", "a075", "a100"):
            if tag in cells and cells[tag]["finished"]:
                b = _ratio(boot[tag]["sampled"], stored[idx].mean(axis=1))
                res["ratios_vs_stored"][tag] = {
                    "r": float(np.nanmean(cells[tag]["per_prompt"]["sampled"]) / stored.mean()),
                    "ci": _pct(b),
                }
        res["stored_points"] = refs.get("stored_points")
    if refs and have_top:
        same = cells["a100"]["per_prompt"]["sampled"]
        d = same - stored
        res["yardstick"] = {
            "same_engine": res["cells"]["a100"]["pass1"],
            "stored": float(stored.mean()),
            "difference": float(np.nanmean(d)),
            "difference_ci": _pct(_boot_means(d, idx)),
            "flag": bool(not 0.5 <= res["cells"]["a100"]["pass1"] <= 0.9),
        }
    res.setdefault("reference_reading", NOT_EVALUATED)
    res["both_lines"] = both_lines(res["reading"]["verdict"], res["reference_reading"])
    res["predictions"] = score_predictions(res)
    return res


def score_predictions(res: dict) -> dict:
    out: dict = {}
    for who, pred in PREDICTIONS.items():
        out[who] = {}
        for tag, p in pred.items():
            if tag == "verdict":
                out[who]["verdict"] = {"predicted": p, "observed": res["reading"]["verdict"]}
                continue
            if tag == "reference":
                out[who]["reference"] = {"predicted": p, "observed": res["reference_reading"]}
                continue
            obs = res["cells"].get(tag, {}).get("pass1")
            lo, hi = p["range"]
            out[who][tag] = dict(
                p, observed=obs, inside=None if obs is None else bool(lo <= obs <= hi)
            )
    return out


def pc(x: float | None, d: int = 1) -> str:
    return "n/a" if x is None or not np.isfinite(x) else f"{100 * x:.{d}f} %"


def ci(v: list, d: int = 1) -> str:
    if v is None or None in v:
        return "[n/a]"
    return f"[{100 * v[0]:.{d}f}, {100 * v[1]:.{d}f}]"


def text_report(res: dict, writes: dict[str, dict]) -> str:
    L = [
        "Weight interpolation between the pre-RL Olmo model (dpo) and the final RL model (s2300).",
        "",
    ]
    L.append(
        f"Strict-checker pass@1 on the {res['prompts']} reach prompts, 4 draws each; 95 % "
        f"intervals from a bootstrap over prompts ({res['reps']} resamples, paired across cells)."
    )
    for tag in ("a025", "a050", "a075", "a100"):
        c = res["cells"].get(tag)
        if c is None:
            L.append(f"- alpha {ALPHA[tag]}: not run (no files).")
            continue
        s = (
            f"- alpha {c['alpha']}: {pc(c['pass1'])} {ci(c['pass1_ci'])} "
            f"({c['passes']} of {c['finished']} draws)"
        )
        if c["missing"]:
            s += (
                f"; INCOMPLETE: {c['missing']} of {c['planned']} draws have no row "
                f"({c['prompts_without_rows']} prompts have none and are outside the as-sampled "
                f"rate); "
                f"every missing draw a fail {pc(c['pass1_all_fail'])}, every missing draw a pass "
                f"{pc(c['pass1_all_pass'])}"
            )
        elif not c["complete"]:
            s += f"; all rows present but a check failed: {c['problems']}"
        r = res["ratios"].get(tag, {}).get("sampled")
        if r:
            s += f"; share of the alpha 1.0 rate {pc(r['r'])} {ci(r['ci'])}"
        w = writes.get(tag)
        if w and w.get("realised_fraction") is not None:
            s += f"; realised fraction along the line {w['realised_fraction']:.3f}"
            if abs(w["realised_fraction"] - c["alpha"]) > 0.05:
                s += " (more than 0.05 from the nominal alpha)"
        st = c["style"]
        s += (
            f". Mean {c['mean_tokens']:.0f} tokens per draw, {c['cut_at_cap']} cut at the cap, "
            f"{c['no_end_of_thinking']} without end of thinking"
            if c["mean_tokens"] is not None
            else ""
        )
        if st["passes"]:
            s += (
                f"; of {st['passes']} passes {pc(st['unpunctuated_share'])} have no sentence "
                f"punctuation, median {st['median_words']:.0f} words"
            )
        L.append(s + ".")
    y = res.get("yardstick")
    if y:
        L.append(
            f"Same-engine alpha 1.0 rate {pc(y['same_engine'])} against the stored s2300 rate "
            f"{pc(y['stored'])}: difference {100 * y['difference']:+.1f} points "
            f"{ci(y['difference_ci'])}."
            + (
                " The same-engine rate is outside 50 to 90 %: read everything below with that in "
                "mind."
                if y["flag"]
                else ""
            )
        )
    h = res.get("half_gain")
    if h:
        if h["alpha"] is None:
            L.append(
                "Half-gain point: not defined (the curve never reaches half of the alpha 1.0 rate)."
            )
        else:
            where = (
                "below 0.25 (no cell was sampled there)"
                if h["below_first_cell"]
                else f"alpha {h['alpha']:.2f}"
            )
            L.append(
                f"Half-gain point (linear interpolation between cells): {where}, interval "
                f"[{h['ci'][0]:.2f}, {h['ci'][1]:.2f}]"
                + ("" if h["monotone"] else "; the curve is not monotone")
                + "."
            )
    hr = res.get("half_gain_realised")
    if hr and hr["alpha"] is not None:
        axis = ", ".join(f"{ALPHA[t]} -> {v:.3f}" for t, v in hr["axis"].items() if t != "a100")
        L.append(
            "On the realised axis (each merged checkpoint placed where its rounded weights are "
            f"along the line: {axis}) the half-gain point is {hr['alpha']:.2f}, interval "
            f"[{hr['ci'][0]:.2f}, {hr['ci'][1]:.2f}]."
        )
    if writes:
        s = "Merged checkpoints: " + "; ".join(
            f"alpha {ALPHA[t]} {'passed' if w['ok'] else 'FAILED'} its checks "
            f"({w['tensors_checked']} tensors)"
            for t, w in sorted(writes.items())
        )
        one = next(
            (w["alpha1_identity"] for w in writes.values() if w.get("alpha1_identity")), None
        )
        if one:
            s += (
                f". At alpha 1 the writer's arithmetic differs from the stored s2300 weights at "
                f"{one['elements_not_equal']} of {one['elements']} elements"
                + (" (more than 1 in a million: look at this)" if one["flag"] else "")
            )
        L.append(s + ".")
    rd = res["reading"]
    L.append(
        f"Reading (rule fixed in PREREG.md, on the alpha 0.5 share of the alpha 1.0 rate): "
        f"{rd['verdict']}" + (f" ({rd['why']})" if rd["why"] else "") + "."
    )
    if rd.get("near_boundary"):
        L.append(
            "The alpha 0.5 share is within 0.05 of a boundary of the rule (0.35 or 0.65): near the "
            "boundary of the rule, where this sample size gives the other verdict 1 time in 6."
        )
    ref = res.get("reference")
    if not ref:
        L.append(
            "Smooth reference (PREREG Amendment 1): not evaluated, no stored per-prompt rates were "
            "found. " + res["both_lines"]
        )
    else:
        a, b = (ref[k] for k in REFERENCE_P0)
        L.append(
            "Smooth reference (PREREG Amendment 1; computed from stored rates before the run): if "
            "every prompt's log-odds of passing moved in a straight line in alpha, the rates would "
            f"be {pc(b['rates']['0.25'])} to {pc(a['rates']['0.25'])} at alpha 0.25, "
            f"{pc(b['rates']['0.5'])} to {pc(a['rates']['0.5'])} at 0.5 and "
            f"{pc(b['rates']['0.75'])} to {pc(a['rates']['0.75'])} at 0.75, so the alpha 0.5 share "
            f"would be {pc(b['r050'])} to {pc(a['r050'])}. Against that reference the observed "
            "share is: "
            f"{res['reference_reading']}. {res['both_lines']} In the small provers the alpha 0.5 "
            f"share was {pc(SMALL_PROVERS['r050'])} (sealed set, 2026-09-30)."
        )
    rs = res.get("ratios_vs_stored")
    if rs:
        L.append(
            "Sensitivity (not used by the rule): with the stored s2300 rate (32 draws per prompt) "
            "as the denominator the shares are "
            + ", ".join(f"alpha {ALPHA[t]} {pc(v['r'])} {ci(v['ci'])}" for t, v in rs.items())
            + "."
        )
    L.append("")
    L.append("Predictions written before the run:")
    for who, pred in res["predictions"].items():
        parts = []
        for tag in ("a025", "a050", "a075", "a100"):
            if tag in pred:
                p = pred[tag]
                hit = {True: "inside", False: "outside", None: "not measured"}[p["inside"]]
                parts.append(
                    f"alpha {ALPHA[tag]} {100 * p['range'][0]:g} to {100 * p['range'][1]:g} % "
                    f"(observed {pc(p['observed'])}: {hit})"
                )
        if "verdict" in pred:
            parts.append(f'verdict "{pred["verdict"]["predicted"]}"')
        if "reference" in pred:
            parts.append(f'second line "{pred["reference"]["predicted"]}"')
        L.append(f"- {who}: " + "; ".join(parts) + ".")
    L.append("")
    L.append(
        "Limits: 48 prompts x 4 draws; one RL run of one model; prompts selected because the "
        "pre-RL model never passed them; passes are checker passes (answers not yet read); a "
        "weight-space path says where along the line the behaviour changes, not that the pre-RL "
        "model contains it."
    )
    return "\n".join(L) + "\n"


def to_json(o):
    if isinstance(o, np.ndarray):
        return [None if not np.isfinite(x) else float(x) for x in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    raise TypeError(type(o))


def run(
    data: Path,
    out: Path,
    reps: int = REPS,
    models_root: str = "/workspace/models",
    merged_root: str = "/workspace/wi/models",
) -> dict:
    spec = cc.parse_spec(data / "inputs" / "spec_reach4.txt")
    sha_p = data / "inputs" / "prompt_sha256_reach.json"
    sha = json.loads(sha_p.read_text()) if sha_p.exists() else None
    cells = {}
    for tag in ALPHA:
        d = data / "runs" / f"cell_{tag}" / "out"
        if (d / "graded.jsonl").exists() or (d / "samples.jsonl.gz").exists():
            cells[tag] = load_cell(d, spec, tag, sha, models_root, merged_root)
    refs_p = data / "refs" / "refs_reach.json"
    refs = json.loads(refs_p.read_text()) if refs_p.exists() else None
    writes = {}
    for tag in ("a025", "a050", "a075"):
        f = data / "runs" / f"write_{tag}" / "out" / "check.json"
        if f.exists():
            ck = json.loads(f.read_text())
            chk = ck.get("check") or {}
            a = chk.get("all") or {}
            writes[tag] = {
                "ok": ck.get("ok"),
                "tensors_checked": chk.get("tensors_checked"),
                "alpha1_identity": chk.get("alpha1_identity"),
                **{k: a.get(k) for k in ("realised_fraction", "length_ratio", "off_line_share")},
                **{k: a.get(f"{k}_share_of_differing") for k in ("at_dpo", "at_ckpt", "between")},
                "realised_fraction_by_class": {
                    k: v.get("realised_fraction") for k, v in (chk.get("by_class") or {}).items()
                },
                "steps": a.get("steps"),
            }
    res = analyse(cells, refs, reps, writes=writes)
    res["writes"] = writes
    out.mkdir(parents=True, exist_ok=True)
    man = {
        tag: {
            "alpha": c["alpha"],
            "complete": c["complete"],
            "rows": c["finished"],
            "samples": c["files"]["samples"],
            "graded": c["files"]["graded"],
            "md5": {k: md5(Path(v)) for k, v in c["files"].items() if Path(v).exists()},
        }
        for tag, c in cells.items()
    }
    (out / "cells_manifest.json").write_text(json.dumps(man, indent=1) + "\n")
    (out / "results.json").write_text(json.dumps(res, indent=1, default=to_json) + "\n")
    txt = text_report(res, writes)
    (out / "results.txt").write_text(txt)
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", default="/scratch/cowifornia/nd-rl/olmo_elicit/interp")
    ap.add_argument("--out")
    ap.add_argument("--reps", type=int, default=REPS)
    a = ap.parse_args(argv)
    data = Path(a.data)
    out = Path(a.out) if a.out else data / "analysis"
    run(data, out, a.reps)
    print((out / "results.txt").read_text(), end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
