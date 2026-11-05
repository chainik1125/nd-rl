"""Stage 3 analysis (PREREG Stage 3): the drift control for the revert (part A) and the edge
counts and path patching (part B), with the pre-registered readings, figures and tables.

    python -m code.experiments.current.circuit_diffing_20261101.stage3_analyze [--only A,B]

Part A inputs: `stage2/L3_s0_L3_n8192_k32_seed0/` (RL) and `stage2/cpt_lr0.00015_s0_L3_*/` (CPT;
its `rl_*` conditions are driven by CPT). 95% intervals: 4,000 paired problem-bootstrap draws
(problems resampled within G and within S, the same draw for every condition).
Part B inputs: `stage3/edges_{L3,cpt,kp16}.pt` (`edges.py score`) and `stage3/patch_L3.json`
(`edges.py patch`). Edge statistics: 1,000 draws over G problems, shared by the pairs; patching:
4,000 draws over problems.
Writes `stage3.json`, `stage3.md` and `figs/stage3_drift.*`, `figs/stage3_edges.*`.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from .common import ART, HERE

ST2 = ART / "stage2"
ST3 = Path(os.environ.get("CIRCUITS_STAGE3", str(ART / "stage3")))
FIG = Path(os.environ.get("CIRCUITS_FIGS", str(HERE / "figs")))
RL_DIR, CPT_DIR = "L3_s0_L3_n8192_k32_seed0", "cpt_lr0.00015_s0_L3_n8192_k32_seed0"
PAIR_LABEL = {"L3": "L3 end (RL)", "cpt": "CPT (drift null)", "kp16": "kp16_st1 (1-step null)"}


def ci(x: np.ndarray) -> list[float]:
    return [float(np.nanpercentile(x, 2.5)), float(np.nanpercentile(x, 97.5))]


# ----------------------------------------------------------------------------- part A
def rates(d: str) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, dict], list[str]]:
    out, docs = {}, {}
    names: list[str] = []
    for p in sorted((ST2 / d).glob("*.json")):
        doc = json.loads(p.read_text())
        rows = doc["rows"]
        names = [r["name"] for r in rows]
        out[doc["condition"]] = {
            c: np.array([r["n_ok"] / r["k"] for r in rows if r["cat"] == c]) for c in ("G", "S")
        }
        docs[doc["condition"]] = doc
    return out, docs, names


def part_a() -> dict[str, Any]:
    rl, rl_docs, n_rl = rates(RL_DIR)
    cp, cp_docs, n_cp = rates(CPT_DIR)
    assert n_rl == n_cp, "problem order differs"
    conds = {
        "base_clean": rl["base_clean"],
        "rl_clean": rl["rl_clean"],
        "rl_revert": rl["rl_revert"],
        "rl_revertperm": rl["rl_revertperm"],
    }
    docs = {k: rl_docs[k] for k in conds}
    for c in ("rl_clean", "rl_revert", "rl_revertperm", "rl_recon"):
        if c in cp:
            conds[c.replace("rl_", "cpt_")] = cp[c]
            docs[c.replace("rl_", "cpt_")] = cp_docs[c]
    nG, nS = len(conds["rl_clean"]["G"]), len(conds["rl_clean"]["S"])
    rng = np.random.default_rng(0)
    idx = {"G": rng.integers(0, nG, (4000, nG)), "S": rng.integers(0, nS, (4000, nS))}
    boot = {c: {k: v[k][idx[k]].mean(1) for k in ("G", "S")} for c, v in conds.items()}
    pt = {c: {k: float(v[k].mean()) for k in ("G", "S")} for c, v in conds.items()}
    res: dict[str, Any] = {"n": {"G": nG, "S": nS}, "conditions": {}}
    for c in conds:
        st = docs[c].get("stats") or {}
        r: dict[str, Any] = {
            "solve_pct": {k: 100 * pt[c][k] for k in ("G", "S")},
            "solve_ci": {k: [100 * x for x in ci(boot[c][k])] for k in ("G", "S")},
            "edit_over_h": st["edit_norm"] / st["h_norm"] if st.get("h_norm") else 0.0,
        }
        ref = "rl_clean" if c.startswith("rl_") else "cpt_clean" if c.startswith("cpt_") else None
        if ref and c != ref and ref in conds:
            r["drop"] = {k: 1 - pt[c][k] / pt[ref][k] for k in ("G", "S")}
            r["drop_ci"] = {k: ci(1 - boot[c][k] / boot[ref][k]) for k in ("G", "S")}
            r["drop_pts"] = {k: 100 * (pt[ref][k] - pt[c][k]) for k in ("G", "S")}
            r["drop_pts_ci"] = {
                k: [100 * x for x in ci(boot[ref][k] - boot[c][k])] for k in ("G", "S")
            }
            # exploratory (added after CPT's clean rates were seen): share of the model's own
            # gap over the base removed, (clean - cond) / (clean - base_clean)
            gb = {k: boot[ref][k] - boot["base_clean"][k] for k in ("G", "S")}
            r["gap_removed"] = {
                k: (pt[ref][k] - pt[c][k]) / (pt[ref][k] - pt["base_clean"][k]) for k in ("G", "S")
            }
            r["gap_removed_ci"] = {k: ci((boot[ref][k] - boot[c][k]) / gb[k]) for k in ("G", "S")}
            r["_gap_boot_G"] = (boot[ref]["G"] - boot[c]["G"]) / gb["G"]
        res["conditions"][c] = r
    if {"cpt_revert", "rl_revert"} <= set(res["conditions"]):
        rc, rr = res["conditions"]["cpt_revert"], res["conditions"]["rl_revert"]
        res["exploratory_gapG_cpt_minus_rl"] = {
            "point": rc["gap_removed"]["G"] - rr["gap_removed"]["G"],
            "ci": ci(rc["_gap_boot_G"] - rr["_gap_boot_G"]),
        }
    for r in res["conditions"].values():
        r.pop("_gap_boot_G", None)
    if "cpt_revert" in conds:
        dS_c = 1 - boot["cpt_revert"]["S"] / boot["cpt_clean"]["S"]
        dS_r = 1 - boot["rl_revert"]["S"] / boot["rl_clean"]["S"]
        dS = res["conditions"]["cpt_revert"]["drop"]["S"]
        vs_base = 100 * (pt["cpt_revert"]["S"] - pt["base_clean"]["S"])
        res["compare"] = {
            "dropS_cpt_minus_rl": float(dS - res["conditions"]["rl_revert"]["drop"]["S"]),
            "dropS_cpt_minus_rl_ci": ci(dS_c - dS_r),
            "cpt_revert_minus_base_S_pts": vs_base,
            "cpt_revert_minus_base_S_pts_ci": [
                100 * x for x in ci(boot["cpt_revert"]["S"] - boot["base_clean"]["S"])
            ],
            "cpt_clean_minus_base_pts": {
                k: 100 * (pt["cpt_clean"][k] - pt["base_clean"][k]) for k in ("G", "S")
            },
            "cpt_clean_minus_base_pts_ci": {
                k: [100 * x for x in ci(boot["cpt_clean"][k] - boot["base_clean"][k])]
                for k in ("G", "S")
            },
        }
        if dS <= 0.08 and vs_base >= -5:
            reading = (
                "benign: the revert costs a drifted model little, so RL's rotations are special"
            )
        elif dS >= 0.20:
            reading = "generic damage: reverting a drifted model is destructive"
        else:
            reading = "partial"
        weak = res["conditions"]["cpt_revert"]["edit_over_h"] < 0.25
        res["reading"] = reading + (" (CPT edit / |h| < 0.25: a weaker edit)" if weak else "")
    return res


# ----------------------------------------------------------------------------- part B
def edge_stats(eb: np.ndarray, ex: np.ndarray, K: int, eps: float) -> dict[str, float]:
    ax, ab = np.abs(ex), np.abs(eb)
    k = min(K, ex.size)
    top = np.argpartition(-ax, k - 1)[:k]
    new = ab[top] < eps * ax[top]
    flip = (np.sign(eb[top]) != np.sign(ex[top])) & ~new
    mt = ax[top].sum()
    return {
        "N_new": int(new.sum()),
        "m_new": float(ax[top][new].sum() / mt) if mt > 0 else float("nan"),
        "N_flip": int(flip.sum()),
        "m_flip": float(ax[top][flip].sum() / mt) if mt > 0 else float("nan"),
    }


def rank(x: np.ndarray) -> np.ndarray:
    r = np.empty(x.size)
    r[np.argsort(x, kind="stable")] = np.arange(x.size)
    return r


def agreement(eb: np.ndarray, ex: np.ndarray, K: int) -> dict[str, float]:
    rho = float(np.corrcoef(rank(eb), rank(ex))[0, 1])
    k = min(K, ex.size)
    itx = np.argpartition(-np.abs(ex), k - 1)[:k]
    tb = set(np.argpartition(-np.abs(eb), k - 1)[:k].tolist())
    tx = set(itx.tolist())
    return {
        "spearman": rho,
        "jaccard_top": len(tb & tx) / len(tb | tx),
        # descriptive additions (Stage 3 log, 5:50 pm): linear agreement and sign agreement on X's
        # top set, and the median base / X magnitude ratio there
        "pearson": float(np.corrcoef(eb, ex)[0, 1]),
        "sign_agree_top": float(np.mean(np.sign(eb[itx]) == np.sign(ex[itx]))),
        "median_ratio_top": float(np.median(np.abs(eb[itx]) / np.abs(ex[itx]))),
    }


def part_b() -> dict[str, Any]:
    import torch

    res: dict[str, Any] = {"pairs": {}}
    eds = {}
    for x in ("L3", "cpt", "kp16"):
        p = ST3 / f"edges_{x}.pt"
        if p.exists():
            eds[x] = torch.load(p, weights_only=False)
    if not eds:
        return res
    pcat = next(iter(eds.values()))["pcat"]
    G = np.flatnonzero(pcat == 0)
    S = np.flatnonzero(pcat == 1)
    rng = np.random.default_rng(0)
    draws = rng.integers(0, len(G), (1000, len(G)))
    boot_m: dict[str, dict[str, np.ndarray]] = {}
    for x, d in eds.items():
        E = d["E"].double().numpy()  # (2, P, u3, u5)
        EG = E[:, G].reshape(2, len(G), -1)
        eb, ex = EG[0].sum(0), EG[1].sum(0)
        A5 = d["A5"].double().numpy()
        a5b, a5x = A5[0][G].sum(0), A5[1][G].sum(0)
        live5 = np.flatnonzero((np.abs(A5[0]).sum(0) + np.abs(A5[1]).sum(0)) > 0)
        A3 = d["A3"].double().numpy()
        live3 = np.flatnonzero((np.abs(A3[0]).sum(0) + np.abs(A3[1]).sum(0)) > 0)
        r: dict[str, Any] = {
            "u3": int(len(d["U3"])),
            "u5": int(len(d["U5"])),
            "hop35": edge_stats(eb, ex, 1000, 0.2),
            "hop35_agree": agreement(eb, ex, 1000),
            "hop5out": edge_stats(a5b[live5], a5x[live5], 100, 0.2),
            "hop5out_agree": agreement(a5b[live5], a5x[live5], 100),
            "sens35": {
                f"K{K}_eps{e}": edge_stats(eb, ex, K, e)
                for K in (300, 1000, 3000)
                for e in (0.1, 0.2, 0.3)
            },
            "sens5out": {
                f"K{K}_eps{e}": edge_stats(a5b[live5], a5x[live5], K, e)
                for K in (30, 100, 300)
                for e in (0.1, 0.2, 0.3)
            },
            "hop35_unweighted": edge_stats(
                d["Eu"][0].double().numpy()[G].sum(0).ravel(),
                d["Eu"][1].double().numpy()[G].sum(0).ravel(),
                1000,
                0.2,
            ),
            "hop35_unweighted_agree": agreement(
                d["Eu"][0].double().numpy()[G].sum(0).ravel(),
                d["Eu"][1].double().numpy()[G].sum(0).ravel(),
                1000,
            ),
            "nodes_agree": {
                f"A{lay}": agreement(
                    d[f"A{lay}"][0].double().numpy()[G].sum(0)[live],
                    d[f"A{lay}"][1].double().numpy()[G].sum(0)[live],
                    100,
                )
                for lay, live in ((3, live3), (5, live5))
            },
            "hop35_S": edge_stats(E[0][S].sum(0).ravel(), E[1][S].sum(0).ravel(), 1000, 0.2),
            "metric_G": {
                "base": float(d["M"][0].double().numpy().sum()),
                "x": float(d["M"][1].double().numpy().sum()),
            },
        }
        bm = {"hop35": np.zeros(1000), "hop5out": np.zeros(1000)}
        for b in range(1000):
            w = np.bincount(draws[b], minlength=len(G)).astype(float)
            bm["hop35"][b] = edge_stats(w @ EG[0], w @ EG[1], 1000, 0.2)["m_new"]
            wb, wx = w @ A5[0][G], w @ A5[1][G]
            bm["hop5out"][b] = edge_stats(wb[live5], wx[live5], 100, 0.2)["m_new"]
        r["hop35"]["m_new_ci"] = ci(bm["hop35"])
        r["hop5out"]["m_new_ci"] = ci(bm["hop5out"])
        boot_m[x] = bm
        res["pairs"][x] = r
    if "L3" in boot_m:
        for null in ("cpt", "kp16"):
            if null in boot_m:
                for hop in ("hop35", "hop5out"):
                    diff = boot_m["L3"][hop] - boot_m[null][hop]
                    res[f"{hop}_L3_minus_{null}"] = {
                        "point": res["pairs"]["L3"][hop]["m_new"]
                        - res["pairs"][null][hop]["m_new"],
                        "ci": ci(diff),
                    }
    pp = ST3 / "patch_L3.json"
    if pp.exists():
        res["patch"] = patch_stats(json.loads(pp.read_text()))
    res["decision"] = decide(res)
    return res


def patch_stats(r: dict[str, Any]) -> dict[str, Any]:
    prob, att = np.array(r["prob"]), np.array(r["attempt"])
    pcat = np.array(r["pcat"])
    conds = list(r["M"])
    # per attempt: sum over its steps; per problem: mean over attempts
    keys = sorted(set(zip(prob.tolist(), att.tolist(), strict=True)))
    kidx = {k: i for i, k in enumerate(keys)}
    a_of = np.array([kidx[(p, a)] for p, a in zip(prob, att, strict=True)])
    a_prob = np.array([k[0] for k in keys])
    probs = sorted(set(a_prob.tolist()))
    per: dict[str, np.ndarray] = {}
    for c in conds:
        m = np.array(r["M"][c])
        att_sum = np.bincount(a_of, weights=m, minlength=len(keys))
        per[c] = np.array([att_sum[a_prob == p].mean() for p in probs])
    cat = np.array([pcat[p] for p in probs])
    out: dict[str, Any] = {"n_problems": {"G": int((cat == 0).sum()), "S": int((cat == 1).sum())}}
    rng = np.random.default_rng(0)
    for k, lab in ((0, "G"), (1, "S")):
        ix = np.flatnonzero(cat == k)
        draws = ix[rng.integers(0, len(ix), (4000, len(ix)))]
        mb = {c: per[c][draws].mean(1) for c in conds}
        mp = {c: float(per[c][ix].mean()) for c in conds}
        gap_b = mb["L3"] - mb["base"]
        gap = mp["L3"] - mp["base"]
        o: dict[str, Any] = {"M": mp, "gap": gap}
        for c in conds:
            if c in ("base", "L3"):
                continue
            if c.startswith("base_"):  # converse: gap recovered in the base
                o[c] = {
                    "recovered": (mp[c] - mp["base"]) / gap,
                    "ci": ci((mb[c] - mb["base"]) / gap_b),
                }
            else:
                o[c] = {"removed": (mp["L3"] - mp[c]) / gap, "ci": ci((mb["L3"] - mb[c]) / gap_b)}
        rmean = np.mean([(mb["L3"] - mb[f"rand{s}"]) / gap_b for s in range(3)], axis=0)
        o["rand_mean"] = {
            "removed": float(np.mean([o[f"rand{s}"]["removed"] for s in range(3)])),
            "ci": ci(rmean),
        }
        d_nr = (mb["retuned"] - mb["new"]) / gap_b  # = removed(new) - removed(retuned)
        o["new_minus_retuned"] = {
            "point": o["new"]["removed"] - o["retuned"]["removed"],
            "ci": ci(d_nr),
        }
        out[lab] = o
    out["sets"] = r["sets"]
    return out


def decide(res: dict[str, Any]) -> dict[str, Any]:
    pr = res["pairs"]
    if not {"L3", "cpt", "kp16"} <= set(pr) or "patch" not in res:
        return {
            "status": "incomplete: "
            + ", ".join(sorted(pr))
            + (" + patch" if "patch" in res else "")
        }
    m = {x: pr[x]["hop35"]["m_new"] for x in pr}
    g = res["patch"]["G"]
    c1 = (
        m["L3"] >= 2 * m["cpt"]
        and m["L3"] >= 2 * m["kp16"]
        and res["hop35_L3_minus_cpt"]["ci"][0] > 0
    )
    c2 = g["new"]["removed"] >= 0.20
    c3 = g["rand_mean"]["removed"] <= 0.05
    moot = g["swap45"]["removed"] < 0.10
    ra = m["L3"] < 2 * m["cpt"] or g["new"]["removed"] < 0.10
    rb = g["retuned"]["removed"] >= 0.10 or g["top"]["removed"] >= 0.10
    if moot:
        verdict = "moot: swap45 removes < 10% of the G gap"
    elif c1 and c2 and c3:
        verdict = "new wiring supported"
    elif ra and rb:
        verdict = "retuned existing wiring supported"
    else:
        verdict = "mixed / inconclusive"
    return dict(
        mass_vs_nulls=c1,
        new_removes_ge20=c2,
        rand_removes_le5=c3,
        retuned_a=ra,
        retuned_b=rb,
        swap45_moot=moot,
        verdict=verdict,
        larger_of_new_retuned="new"
        if g["new"]["removed"] >= g["retuned"]["removed"]
        else "retuned",
    )


# ----------------------------------------------------------------------------- output
def fmt_ci(v: float, c: list[float], pct: bool = True, signed: bool = False) -> str:
    f = (lambda x: f"{100 * x:+.1f}") if signed else (lambda x: f"{100 * x:.1f}")
    if not pct:
        f = (lambda x: f"{x:+.1f}") if signed else (lambda x: f"{x:.1f}")
    return f"{f(v)} [{f(c[0])}, {f(c[1])}]"


def write_md(out: dict[str, Any], md: Path) -> None:
    L = ["# Stage 3 numbers (generated by `stage3_analyze.py`)", ""]
    A = out.get("A")
    if A and A.get("conditions"):
        L += [
            "## Part A: drift control for the revert (layer 3)",
            "",
            f"G = {A['n']['G']} RL-gained, S = {A['n']['S']} shared problems; solve % = mean over "
            "problems of the per-problem success rate in 32 attempts; [95% paired problem "
            "bootstrap]. `cpt_*` conditions are driven by CPT with the base-vs-CPT crosscoder.",
            "",
            "| condition | edit / ‖h‖ | G solve % | S solve % | relative drop G | relative drop S "
            "| drop S (points) | own G gap removed (exploratory) |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for c, r in A["conditions"].items():
            sg, ss = r["solve_pct"]["G"], r["solve_pct"]["S"]
            cg, cs = r["solve_ci"]["G"], r["solve_ci"]["S"]
            dg = ds = dp = gr = ""
            if "drop" in r:
                dg = fmt_ci(r["drop"]["G"], r["drop_ci"]["G"])
                ds = fmt_ci(r["drop"]["S"], r["drop_ci"]["S"])
                dp = fmt_ci(r["drop_pts"]["S"], r["drop_pts_ci"]["S"], pct=False)
                gr = fmt_ci(r["gap_removed"]["G"], r["gap_removed_ci"]["G"])
            L.append(
                f"| {c} | {r['edit_over_h']:.3f} | {sg:.1f} [{cg[0]:.1f}, {cg[1]:.1f}] "
                f"| {ss:.1f} [{cs[0]:.1f}, {cs[1]:.1f}] | {dg} | {ds} | {dp} | {gr} |"
            )
        if "compare" in A:
            cmp_ = A["compare"]

            def pts(v: float, c: list[float]) -> str:
                return fmt_ci(v, c, pct=False, signed=True)

            d_rl = fmt_ci(cmp_["dropS_cpt_minus_rl"], cmp_["dropS_cpt_minus_rl_ci"], signed=True)
            vb = pts(cmp_["cpt_revert_minus_base_S_pts"], cmp_["cpt_revert_minus_base_S_pts_ci"])
            cb, cbc = cmp_["cpt_clean_minus_base_pts"], cmp_["cpt_clean_minus_base_pts_ci"]
            L += [
                "",
                f"- Relative drop on S, cpt_revert − rl_revert: {d_rl} points.",
                f"- cpt_revert S − base_clean S: {vb} points.",
                f"- cpt_clean − base_clean: G {pts(cb['G'], cbc['G'])}, "
                f"S {pts(cb['S'], cbc['S'])} points.",
                f"- Pre-registered reading: **{A['reading']}**.",
                "- Caveat: the rule's conclusion assumed CPT gained nothing over the base; "
                "cpt_clean − base_clean on G (above) says otherwise, so the exploratory own-gap "
                "comparison below is the informative one.",
            ]
        if "exploratory_gapG_cpt_minus_rl" in A:
            v = A["exploratory_gapG_cpt_minus_rl"]
            L += [
                "- Exploratory (added after CPT's own G rate was seen to be far above the "
                "base's): share of the model's own G gap over the base removed by its revert, "
                f"CPT − RL: {fmt_ci(v['point'], v['ci'], signed=True)} points.",
            ]
        L.append("")
    B = out.get("B")
    if B and B.get("pairs"):
        L += [
            "## Part B: edges, layer 3 → 5 and layer 5 → output",
            "",
            "EAP edges on the L3 end's accepted G attempts (the same tokens in every model). "
            "Top set: 1,000 edges (3 → 5) or 100 (5 → output) by |e^X|; new: |e^base| < 0.2 |e^X|. "
            "m_new: share of the top set's |e^X| carried by new edges [95% bootstrap over G "
            "problems].",
            "",
            "| pair | U3 / U5 | 3→5 N_new | 3→5 m_new | 3→5 flipped (N, mass) | Spearman ρ "
            "| top-1000 Jaccard | 5→out N_new | 5→out m_new | 5→out ρ |",
            "|---|---|---|---|---|---|---|---|---|---|",
        ]
        for x, r in B["pairs"].items():
            h, o = r["hop35"], r["hop5out"]
            L.append(
                f"| {PAIR_LABEL[x]} | {r['u3']} / {r['u5']} | {h['N_new']} "
                f"| {fmt_ci(h['m_new'], h['m_new_ci'])}% | {h['N_flip']}, {100 * h['m_flip']:.1f}% "
                f"| {r['hop35_agree']['spearman']:.3f} | {r['hop35_agree']['jaccard_top']:.2f} "
                f"| {o['N_new']} | {fmt_ci(o['m_new'], o['m_new_ci'])}% "
                f"| {r['hop5out_agree']['spearman']:.3f} |"
            )
        L += [
            "",
            "Agreement between base and X (descriptive): EAP edges, unweighted edges "
            "(Σ dpre over the receiver's active positions: the wiring without the metric "
            "gradient), and node attributions. Pearson r / Spearman ρ; sign agreement and median "
            "|base| / |X| on X's top set (1,000 edges, 100 nodes).",
            "",
            "| pair | EAP 3→5 r / ρ / sign / ratio | unweighted 3→5 r / ρ / sign / ratio "
            "| nodes L3 r / sign | nodes L5 r / sign |",
            "|---|---|---|---|---|",
        ]
        for x, r in B["pairs"].items():

            def ag(a: dict[str, float]) -> str:
                return (
                    f"{a['pearson']:.2f} / {a['spearman']:.2f} / {100 * a['sign_agree_top']:.0f}% "
                    f"/ {a['median_ratio_top']:.2f}"
                )

            n3, n5 = r["nodes_agree"]["A3"], r["nodes_agree"]["A5"]
            L.append(
                f"| {PAIR_LABEL[x]} | {ag(r['hop35_agree'])} | {ag(r['hop35_unweighted_agree'])} "
                f"| {n3['pearson']:.2f} / {100 * n3['sign_agree_top']:.0f}% "
                f"| {n5['pearson']:.2f} / {100 * n5['sign_agree_top']:.0f}% |"
            )
        L.append("")
        for key in [k for k in B if k.startswith("hop") and "_minus_" in k]:
            v = B[key]
            L.append(f"- {key}: {fmt_ci(v['point'], v['ci'], signed=True)} points of m_new.")
        L += ["", "Sensitivity of 3 → 5 m_new (%) to the top-set size K and ε:", ""]
        keys = list(next(iter(B["pairs"].values()))["sens35"])
        L += ["| pair | " + " | ".join(keys) + " |", "|---" * (len(keys) + 1) + "|"]
        for x, r in B["pairs"].items():
            L.append(
                f"| {x} | "
                + " | ".join(f"{100 * r['sens35'][k]['m_new']:.1f}" for k in keys)
                + " |"
            )
        L += [
            "",
            "Secondary: 3 → 5 m_new with unweighted edges (Σ dpre over j's active positions) "
            "and on S: "
            + "; ".join(
                f"{x} {100 * r['hop35_unweighted']['m_new']:.1f}% / "
                f"{100 * r['hop35_S']['m_new']:.1f}%"
                for x, r in B["pairs"].items()
            ),
            "",
        ]
        if "patch" in B:
            P = B["patch"]
            L += [
                "### Path patching in the L3 end (teacher-forced, 3 → 5 edges)",
                "",
                f"M = summed log-prob of the accepted action tokens per attempt, mean over "
                f"attempts and problems (G {P['n_problems']['G']}, S {P['n_problems']['S']} "
                "problems). Gap removed = (M_L3 − M_cond) / (M_L3 − M_base); base_* rows: gap "
                "recovered in the base. G folds are patched with sets from the other fold.",
                "",
                "| condition | G: share of gap | S: share of gap |",
                "|---|---|---|",
            ]
            for c in [k for k in P["G"] if k not in ("M", "gap", "new_minus_retuned")]:
                row = []
                for lab in ("G", "S"):
                    v = P[lab][c]
                    key = "recovered" if "recovered" in v else "removed"
                    row.append(f"{key} {fmt_ci(v[key], v['ci'])}%")
                L.append(f"| {c} | {row[0]} | {row[1]} |")
            for lab in ("G", "S"):
                v = P[lab]["new_minus_retuned"]
                L.append(
                    f"\n{lab}: M_L3 {P[lab]['M']['L3']:.2f}, M_base {P[lab]['M']['base']:.2f} "
                    f"(gap {P[lab]['gap']:.2f} nats per attempt); new − retuned: "
                    f"{fmt_ci(v['point'], v['ci'], signed=True)} points."
                )
            L += ["", "Edge sets (n, Σ|e^L3|, Σ|e^L3 − e^base|) per group:", ""]
            for g, st in P["sets"].items():
                L.append(
                    f"- {g}: "
                    + "; ".join(
                        f"{c} {v['n']} / {v['mass_x']:.3g} / {v['change']:.3g}"
                        for c, v in st.items()
                    )
                )
            L.append("")
        L += [f"**Decision (pre-registered rule): {json.dumps(B['decision'])}**", ""]
    md.write_text("\n".join(L) + "\n")


def figures(out: dict[str, Any]) -> None:
    from code.ndbench.figures import style as S

    C = S.OKABE_ITO
    A = out.get("A")
    if A and A.get("conditions"):
        order = [
            ("base_clean", "base"),
            ("cpt_clean", "CPT (no RL)"),
            ("cpt_recon", "CPT, layer 3 → reconstruction"),
            ("cpt_revert", "CPT, every latent → base version"),
            ("cpt_revertperm", "CPT, same edits permuted"),
            ("rl_clean", "RL (L3 end)"),
            ("rl_revert", "RL, every latent → base version"),
            ("rl_revertperm", "RL, same edits permuted"),
        ]
        order = [(c, lab) for c, lab in order if c in A["conditions"]]
        with S.paper_style():
            fig = S.new_figure(S.FULL_W, 2.6, layout="constrained")
            ax = fig.subplots(1, 1)
            ys = np.arange(len(order))[::-1].astype(float)
            n_cpt = sum(c.startswith(("base", "cpt")) for c, _ in order)
            ys[n_cpt:] -= 0.6
            for y, (cnd, _) in zip(ys, order, strict=True):
                r = A["conditions"][cnd]
                for cat, col, mk, dy, lab in (
                    ("G", C["vermillion"], "o", 0.14, f"RL-gained problems (G, {A['n']['G']})"),
                    ("S", C["blue"], "s", -0.14, f"shared problems (S, {A['n']['S']})"),
                ):
                    v = r["solve_pct"][cat]
                    lo, hi = r["solve_ci"][cat]
                    ax.errorbar(
                        v,
                        y + dy,
                        xerr=[[v - lo], [hi - v]],
                        fmt=mk,
                        color=col,
                        ms=3.2,
                        lw=0.8,
                        capsize=1.5,
                        label=lab if cnd == order[0][0] else None,
                    )
            ax.axhline(ys[n_cpt - 1] - 0.8, color=S.GRID, lw=0.6)
            ax.set_yticks(ys)
            ax.set_yticklabels([lab for _, lab in order])
            ax.set_xlim(0, 100)
            ax.set_xlabel("solve rate (% of attempts, mean over problems; 95% problem bootstrap)")
            S.light_grid(ax, "x")
            ax.legend(loc="lower right", fontsize=S.FS_TICK - 1, frameon=True, framealpha=0.9)
            S.save(fig, FIG / "stage3_drift")
            S.close(fig)
    B = out.get("B")
    if B and B.get("pairs"):
        col = {"L3": C["vermillion"], "cpt": C["bluish_green"], "kp16": C["blue"]}
        mk = {"L3": "o", "cpt": "D", "kp16": "s"}
        with S.paper_style():
            fig = S.new_figure(S.FULL_W, 2.1, layout="constrained")
            axs = fig.subplots(1, 3, sharey=False, width_ratios=[1, 1, 1.1])
            for ax, hop, title in (
                (axs[0], "hop35", "(a) new edges, 3 → 5"),
                (axs[1], "hop5out", "(b) new edges, 5 → output"),
            ):
                xs = list(B["pairs"])
                for i, x in enumerate(xs):
                    h = B["pairs"][x][hop]
                    v, (lo, hi) = 100 * h["m_new"], [100 * c for c in h["m_new_ci"]]
                    ax.bar(i, v, color=col[x], width=0.6, edgecolor="white", linewidth=0.8)
                    ax.errorbar(i, v, yerr=[[v - lo], [hi - v]], color=S.INK2, lw=0.8, capsize=1.5)
                    if hi == 0:
                        ax.text(i, 0.2, "0", ha="center", va="bottom", fontsize=S.FS_TICK)
                ax.set_xticks(range(len(xs)))
                ax.set_xticklabels(
                    [{"L3": "L3 end", "cpt": "CPT", "kp16": "1 step"}[x] for x in xs]
                )
                S.light_grid(ax, "y")
                S.panel_title(ax, None, title)
            axs[0].set_ylabel("new edges' share of\ntop-edge mass (%)")
            ax = axs[2]
            xs = list(B["pairs"])
            for i, x in enumerate(xs):
                r = B["pairs"][x]
                for dx, key, mk_, fill, lab in (
                    (-0.12, "hop35_agree", "o", True, "EAP edges (wiring × use)"),
                    (0.12, "hop35_unweighted_agree", "s", False, "wiring only (unweighted)"),
                ):
                    v = r[key]["pearson"]
                    ax.plot(
                        i + dx,
                        v,
                        mk_,
                        color=col[x],
                        ms=4.5,
                        mfc=col[x] if fill else "white",
                        mew=1.0,
                        label=lab if i == 0 else None,
                    )
            ax.axhline(0, color=S.REF, lw=0.6)
            ax.set_ylim(-1.05, 1.1)
            ax.set_xlim(-0.5, len(xs) - 0.5)
            ax.set_xticks(range(len(xs)))
            ax.set_xticklabels([{"L3": "L3 end", "cpt": "CPT", "kp16": "1 step"}[x] for x in xs])
            ax.set_ylabel("Pearson r with the base")
            S.light_grid(ax, "y")
            S.panel_title(ax, None, "(c) agreement with the base, 3 → 5")
            leg = ax.legend(loc="lower right", fontsize=S.FS_TICK - 1, frameon=True, framealpha=0.9)
            for h in leg.legend_handles:
                h.set_color(S.INK2)
                h.set_markerfacecolor("white" if h.get_marker() == "s" else S.INK2)
            S.save(fig, FIG / "stage3_edges")
            S.close(fig)
        if "patch" in B:
            rows = [
                ("swap13", "L3 with embedding + blocks 1–3 from the base"),
                ("swap45", "L3 with blocks 4–5 from the base"),
                ("swap6h", "L3 with block 6 + head from the base"),
                ("revert3", "layer-3 revert (Stage 2 edit)"),
                ("allU", "all selected 3 → 5 edges"),
                ("top", "top-1,000 edges"),
                ("new", "new edges"),
                ("retuned", "retuned edges, matched change"),
                ("rand_mean", "random edges, matched mass (mean of 3)"),
            ]
            P = B["patch"]
            rows = [(c, lab) for c, lab in rows if c in P["G"]]
            with S.paper_style():
                fig = S.new_figure(S.FULL_W, 2.3, layout="constrained")
                ax = fig.subplots(1, 1)
                ys = np.arange(len(rows))[::-1].astype(float)
                for y, (c, _) in zip(ys, rows, strict=True):
                    for cat, cl_, mk_, dy, lab in (
                        ("G", C["vermillion"], "o", 0.14, "RL-gained problems (G)"),
                        ("S", C["blue"], "s", -0.14, "shared problems (S)"),
                    ):
                        v = 100 * P[cat][c]["removed"]
                        lo, hi = (100 * q for q in P[cat][c]["ci"])
                        ax.errorbar(
                            v,
                            y + dy,
                            xerr=[[v - lo], [hi - v]],
                            fmt=mk_,
                            color=cl_,
                            ms=3.2,
                            lw=0.8,
                            capsize=1.5,
                            label=lab if c == rows[0][0] else None,
                        )
                ax.axvline(0, color=S.REF, lw=0.6)
                ax.set_yticks(ys)
                ax.set_yticklabels([lab for _, lab in rows])
                ax.set_xlabel("share of the L3 − base log-prob gap removed (%, teacher-forced)")
                S.light_grid(ax, "x")
                ax.legend(loc="lower right", fontsize=S.FS_TICK - 1, frameon=True, framealpha=0.9)
                S.save(fig, FIG / "stage3_patch")
                S.close(fig)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="A,B")
    a = ap.parse_args(argv)
    path = Path(os.environ.get("CIRCUITS_STAGE3_JSON", str(HERE / "stage3.json")))
    out: dict[str, Any] = json.loads(path.read_text()) if path.exists() else {}
    if "A" in a.only.split(",") and (ST2 / CPT_DIR / "rl_clean.json").exists():
        out["A"] = part_a()
    if "B" in a.only.split(","):
        out["B"] = part_b()
    path.write_text(json.dumps(out, indent=1))
    write_md(out, path.with_suffix(".md"))
    figures(out)
    print(path.with_suffix(".md").read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
