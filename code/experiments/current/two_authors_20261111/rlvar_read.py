"""Readout of PREREG Amendment 3 (the rlvar lines): per-line tables, the precondition, ΔD with bootstrap intervals,
the base-link and one-author-mixture predictions, and B against the running lines.

    python rlvar_read.py [--json OUT] [--B 2000]

Shares are A-form sites / sites, pooled over theorems (as `authors_gate.curve`), on `marker_plain`; intervals are 95 %
percentile bootstrap over theorems with the same draw for every line, iteration and tag (the four lines measure the
same 500 theorems). Reads TA/runs/<line>/seg*/authors/it<i>/marker_plain__<tag>.jsonl only.
"""

# ruff: noqa: E501
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

TA = Path(os.environ.get("TA", "/scratch/cowifornia/nd-rl/two_authors"))
LINES = {
    "aS": "rl_authors_half_styA_s0",
    "cS": "rl_control_half_styA_s0",
    "aU": "rl_authors_half_ub0_s0",
    "cU": "rl_control_half_ub0_s0",
    "aR": "rl_authors_half_s0",  # the running lines (unlikely_beta 0.25, A40 pods)
    "cR": "rl_control_half_s0",
}
HABITS = ("reit", "order", "conj")
SET = "marker_plain"


def _done_dirs(line: str) -> dict[int, Path]:
    """iteration -> its measured folder, over all segments (a later segment's wins)."""
    segs = sorted((TA / "runs" / line).glob("seg[0-9]*"), key=lambda p: int(p.name[3:]))
    return {
        int(p.name[2:]): p
        for s in segs
        for p in sorted((s / "authors").glob("it*"))
        if (p / "DONE").exists()
    }


def rows(line: str, it: int, tag: str) -> list[dict] | None:
    d = _done_dirs(line).get(it)
    f = None if d is None else d / f"{SET}__{tag}.jsonl"
    if f is None or not f.exists():
        return None
    return [json.loads(x) for x in f.read_text().splitlines() if x.strip()]


def its(line: str) -> list[int]:
    return sorted(_done_dirs(line))


def feats(R: list[dict]) -> tuple[list[str], np.ndarray]:
    """Per theorem: [reit sites, a, order sites, a, conj sites, a, attempts, accepted, route, lines]."""
    names = [r["theorem"] for r in R]
    X = np.array([[r["habits"][h][k] for h in HABITS for k in ("sites", "a")]
                  + [r["attempts"], r["accepted"], r["route"], r["lines"]] for r in R], dtype=float)  # fmt: skip
    return names, X


def stats(S: np.ndarray) -> dict[str, np.ndarray]:
    """S: (..., 10) sums -> shares in %. Works on a vector or a bootstrap matrix."""
    with np.errstate(invalid="ignore", divide="ignore"):
        out = {h: 100 * S[..., 2 * i + 1] / S[..., 2 * i] for i, h in enumerate(HABITS)}
        out["accepted"] = 100 * S[..., 7] / S[..., 6]
        out["route"] = 100 * S[..., 8] / S[..., 7]
        out["length"] = S[..., 9] / S[..., 7]
    return out


def ci(x: np.ndarray) -> list[float | None]:
    x = x[np.isfinite(x)]
    return (
        [None, None]
        if not len(x)
        else [round(float(np.percentile(x, 2.5)), 1), round(float(np.percentile(x, 97.5)), 1)]
    )


def joint_link(R: list[dict]) -> dict[str, Any]:
    """Proof level, proofs with order room and reiteration sites: P(order A | reiteration form) and the form mix."""
    c: dict[str, list[int]] = {}
    for r in R:
        for k, n in r["joint"].items():
            f = {x[0]: x[1:] for x in k.split(".")}
            if f.get("o") in (None, "-") or f.get("r") in (None, "-"):
                continue
            c.setdefault(f["r"], [0, 0])
            c[f["r"]][0] += n
            c[f["r"]][1] += n * (f["o"] == "A")
    tot = sum(v[0] for v in c.values())
    return {"n": tot, "mix": {k: v[0] / tot for k, v in c.items()} if tot else {},
            "pA": {k: v[1] / v[0] for k, v in c.items() if v[0]}, "oA": sum(v[1] for v in c.values()) / tot if tot else float("nan")}  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None)
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument(
        "--set", default="marker_plain", help="marker_plain (pre-registered) or route_unseen"
    )
    a = ap.parse_args()
    globals()["SET"] = a.set
    avail = {k: its(v) for k, v in LINES.items()}
    out: dict[str, Any] = {"available": avail}
    names = None
    S: dict[tuple[str, int, str], np.ndarray] = {}
    raw: dict[tuple[str, int, str], list[dict]] = {}
    for k, line in LINES.items():
        for it in avail[k]:
            for tag in ("none", "A", "B"):
                R = rows(line, it, tag)
                if R is None:
                    continue
                n, X = feats(R)
                if names is None:
                    names = n
                assert n == names, f"{line} it {it} {tag}: theorem order differs"
                S[(k, it, tag)], raw[(k, it, tag)] = X, R
    if names is None:
        print("no rows yet")
        return
    rng = np.random.default_rng(20261118)
    W = np.stack(
        [
            np.bincount(rng.integers(0, len(names), len(names)), minlength=len(names))
            for _ in range(a.B)
        ]
    ).astype(float)
    point = {key: stats(X.sum(0)) for key, X in S.items()}
    boot = {key: stats(W @ X) for key, X in S.items()}

    # ---- tables
    tab = {}
    for k in LINES:
        tab[k] = {it: {m: round(float(point[(k, it, "none")][m]), 1 if m != "length" else 2) for m in ("reit", "order", "conj", "accepted", "length", "route")}
                  for it in avail[k] if (k, it, "none") in point}  # fmt: skip
    out["tables_untagged"] = tab
    print(
        f"untagged {SET}: it: reit / order / conj A-share %, accepted %, mean length, route % of accepted"
    )
    for k, line in LINES.items():
        print(f"  {line}")
        for it, v in tab[k].items():
            print(
                f"    it {it:3d}: reit {v['reit']:5.1f}  order {v['order']:5.1f}  conj {v['conj']:5.1f}  acc {v['accepted']:5.1f}  len {v['length']:5.2f}  route {v['route']:4.1f}"
            )

    # ---- precondition and primary at the last iteration all four have
    four = ("aS", "cS", "aU", "cU")
    common = (
        sorted(set.intersection(*[set(avail[k]) for k in four]))
        if all(avail[k] for k in four)
        else []
    )
    three = sorted(set(avail["aS"]) & set(avail["aU"]))
    res: dict[str, Any] = {"common_iters": common}
    for tag in ("none", "A", "B"):
        per = {}
        for it in sorted(set(common) | set(three)):
            d: dict[str, Any] = {}
            if (("aS", it, tag) in S) and (("aU", it, tag) in S):
                pre = point[("aS", it, tag)]["reit"] - point[("aU", it, tag)]["reit"]
                preb = boot[("aS", it, tag)]["reit"] - boot[("aU", it, tag)]["reit"]
                d["reit_aS_minus_aU"] = [round(float(pre), 1), ci(preb)]
                for h in ("order", "conj"):
                    v = point[("aS", it, tag)][h] - point[("aU", it, tag)][h]
                    vb = boot[("aS", it, tag)][h] - boot[("aU", it, tag)][h]
                    d[f"{h}_aS_minus_aU"] = [round(float(v), 1), ci(vb)]
            if it in common and all((k, it, tag) in S for k in four):
                for h in HABITS:
                    v = (point[("aS", it, tag)][h] - point[("aU", it, tag)][h]) - (
                        point[("cS", it, tag)][h] - point[("cU", it, tag)][h]
                    )
                    vb = (boot[("aS", it, tag)][h] - boot[("aU", it, tag)][h]) - (
                        boot[("cS", it, tag)][h] - boot[("cU", it, tag)][h]
                    )
                    d[f"DD_{h}"] = [round(float(v), 1), ci(vb)]
                    cv = point[("cS", it, tag)][h] - point[("cU", it, tag)][h]
                    d[f"{h}_cS_minus_cU"] = round(float(cv), 1)
            per[it] = d
        res[tag] = per
    out["contrasts"] = res
    print("\ncontrasts per iteration (point [95 % CI]); DD = (aS - aU) - (cS - cU)")
    for tag in ("none", "A", "B"):
        for it, d in res[tag].items():
            print(f"  tag {tag:4s} it {it:3d}: " + "; ".join(f"{k} {v}" for k, v in d.items()))

    # ---- predictions for the styA lines: base link (i) and one-author mixture (ii)
    pred = {}
    for k, base_k in (("aS", "aS"), ("cS", "cS"), ("aU", "aU"), ("cU", "cU")):
        if (base_k, 0, "none") not in raw:
            continue
        L0 = joint_link(raw[(base_k, 0, "none")])
        tA, tB = point.get((base_k, 0, "A")), point.get((base_k, 0, "B"))
        pk = {}
        for it in avail[k]:
            if (k, it, "none") not in raw:
                continue
            Lt = joint_link(raw[(k, it, "none")])
            i_pred = sum(L0["pA"].get(f, np.nan) * w for f, w in Lt["mix"].items())
            r_t, o_t = point[(k, it, "none")]["reit"], point[(k, it, "none")]["order"]
            mix = None
            if (
                tA is not None and tB is not None and abs(tA["reit"] - tB["reit"]) >= 5
            ):  # else the tags carry no author
                w = (r_t - tB["reit"]) / (tA["reit"] - tB["reit"])
                mix = {
                    "w": round(float(w), 3),
                    "order_pred": round(float(tB["order"] + w * (tA["order"] - tB["order"])), 1),
                    # w clipped to [0, 1]: a reiteration share beyond tag A's is beyond any one author
                    "order_pred_w01": round(
                        float(tB["order"] + min(1.0, max(0.0, w)) * (tA["order"] - tB["order"])), 1
                    ),
                }
            pk[it] = {"link_base_oA": round(100 * L0["oA"], 1), "link_obs_oA": round(100 * Lt["oA"], 1), "link_pred_oA": round(100 * float(i_pred), 1),
                      "link_n": Lt["n"], "order_obs": round(float(o_t), 1), "mixture": mix}  # fmt: skip
        pred[k] = {"base_P_oA_given_r": {f: round(100 * v, 1) for f, v in L0["pA"].items()},
                   "base_tagA": {h: round(float(tA[h]), 1) for h in HABITS} if tA else None,
                   "base_tagB": {h: round(float(tB[h]), 1) for h in HABITS} if tB else None, "by_it": pk}  # fmt: skip
    out["predictions"] = pred
    print("\npredictions (untagged; link: proofs with both order room and reiteration sites)")
    for k, v in pred.items():
        print(
            f"  {LINES[k]}: base P(order A | reit form) {v['base_P_oA_given_r']}; base tag A {v['base_tagA']}, tag B {v['base_tagB']}"
        )
        for it, x in v["by_it"].items():
            print(
                f"    it {it:3d}: link obs {x['link_obs_oA']} pred {x['link_pred_oA']} (base {x['link_base_oA']}, n {x['link_n']}); order obs {x['order_obs']} mixture {x['mixture']}"
            )

    # ---- B against the running lines: share relative to the base
    rel = {}
    for k in ("aU", "cU", "aR", "cR", "aS", "cS"):
        if (k, 0, "none") not in point:
            continue
        rel[k] = {it: {h: round(float(point[(k, it, "none")][h] / point[(k, 0, "none")][h]), 2) for h in ("reit", "order")}
                  for it in avail[k] if (k, it, "none") in point}  # fmt: skip
    out["relative_to_base"] = rel
    print("\nuntagged share / base share (reit, order)")
    for k, v in rel.items():
        print(
            f"  {LINES[k]}: "
            + "; ".join(f"it {it}: {x['reit']}, {x['order']}" for it, x in v.items())
        )
    # ---- the collapse rule (previous amendment) and the GRPO rows
    col = {}
    for k, line in LINES.items():
        segs = sorted((TA / "runs" / line).glob("seg[0-9]*"), key=lambda p: int(p.name[3:]))
        byit = {}
        for sd in segs:
            if (sd / "metrics.jsonl").exists():
                for x in (sd / "metrics.jsonl").read_text().splitlines():
                    if x.strip():
                        m = json.loads(x)
                        byit[m["iter"]] = m
        if not byit:
            continue
        M = [byit[i] for i in sorted(byit)]
        c = [m.get("canary") or {} for m in M]
        hit = None
        for i in range(1, len(c)):
            if (c[i].get("colon", 0) > 3e-3 and c[i - 1].get("colon", 0) > 3e-3) or (
                c[i].get("wrong_max", 0) > 0.05 and c[i - 1].get("wrong_max", 0) > 0.05
            ):
                hit = M[i]["iter"]
                break
        col[k] = {"last_iter": M[-1]["iter"] if M else 0, "collapse_at": hit, "nonfinite": any(m.get("nonfinite") for m in M),
                  "max_colon": max((x.get("colon", 0) for x in c), default=0), "max_wrong": max((x.get("wrong_max", 0) for x in c), default=0),
                  "pass1_first_last": [round(M[0]["reward_pass1"], 3), round(M[-1]["reward_pass1"], 3)] if M else None}  # fmt: skip
        B = [json.loads(x) for sd in segs if (sd / "style_bonus.jsonl").exists()
             for x in (sd / "style_bonus.jsonl").read_text().splitlines() if x.strip()]  # fmt: skip
        if B:
            col[k]["bonus_paid_share_first_last"] = [
                round(b.get("paid", 0) / max(1, b["accepted"]), 3) for b in (B[0], B[-1])
            ]
            col[k]["bonus_mean_first_last"] = [
                B[0]["bonus_mean_accepted"],
                B[-1]["bonus_mean_accepted"],
            ]
    out["grpo"] = col
    print(
        "\nGRPO rows (collapse rule: canary P(:) > 3e-3 or wrong > 0.05 on two consecutive reads)"
    )
    for k, v in col.items():
        print(f"  {LINES[k]}: {v}")
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
