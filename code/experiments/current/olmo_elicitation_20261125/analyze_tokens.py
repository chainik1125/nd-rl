# ruff: noqa: E501
"""Token-level profile of the distance to the final model (literature review R3; orchestrator's design 2026-11-25).

    analyze_tokens.py [--runs /scratch/cowifornia/nd-rl/olmo_elicit/runs] [--out /scratch/cowifornia/nd-rl/olmo_elicit/tokens_summary] [--B 1000]

Input: runs/<ckpt>/edl/step0_tokens.npz (the step-0 pass of olmo_edl.py: for every response position of the pool's 134
s2300 responses, the stored token, its NLL and rank under the checkpoint, the entropy of the checkpoint's next-token
distribution, and its top-20 ids / log-probs, fp16). CPU only. Every checkpoint t is compared with s2300 position by
position (same responses, same tokens; asserted).

Per checkpoint, pool set (all / train / test / reach) and segment (total / think / answer):
  JS (approximate)  Jensen-Shannon divergence (nats, natural log, at most ln 2) between t's and s2300's next-token
                    distributions on the UNION of the two top-20 supports, each distribution truncated to its own top 20
                    and renormalised (ids outside a model's top 20 get probability 0 under it). This is an approximation,
                    not a bound; `mass_*` gives the top-20 mass it discards (mean 1 - mass). Reported as the mean and the
                    shares of positions with JS > 0.01 and > 0.1 nats (review R3 thresholds).
  JS lower bound    exact JS of the coarse-grained distributions over {each id in BOTH top-20 lists} + {everything else}:
                    every probability in this partition is known, and coarse-graining cannot increase a divergence, so
                    it is a lower bound on the full-vocabulary JS (up to the fp16 storage of the log-probs).
  rank shares       share of positions whose stored s2300 token (the token s2300 sampled) is top-1, in the top 5, and
                    outside the top 20 under t (`rank` field, 0 = top-1, ties count as not larger). At s2300 itself these
                    are the self rates (sampling is not greedy), the ceiling for the others. Also: share where t's and
                    s2300's argmax agree, and where s2300's argmax is in t's top 5.
  NLL excess        e = NLL_t - NLL_s2300 at each position; `excess_mean` (nats/token) is the token-weighted L0 gap.
  concentration     share of the total (net) excess carried by the top 1 %, 5 % and 10 % of positions ranked by e
                    (can exceed 1 because some positions have e < 0), the same share of the positive part sum(max(e, 0)),
                    and the share of the net excess at the top 1/5/10 % of positions ranked by JS.
  entropy deciles   positions split into ten equal-count bins of s2300's entropy at that position (ties broken by
                    position order); per bin the mean entropy, mean excess, share of the net and of the positive excess,
                    mean JS and the top-1 share.
Trends: Spearman with RL step over the checkpoints other than s2300.
Intervals: 95 % percentile bootstrap over responses (B replicates) for the ratio readouts (JS mean and shares, rank
shares, excess mean). Check: per-response NLL sums in each file equal the checkpoint's step0.json sums.
Descriptive reading from the review (R3; not pre-registered): "reranking" if >= 99 % of s2300's tokens are in t's top 5
and the top 5 % of positions (by excess) carry >= 50 % of the gap.
Not covered: the reverse direction (t's own samples scored under s2300), which needs another forward pass.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
from analyze_edl import CKPTS, FINAL, OFF_LINE, SEGS, STEP, ci, spearman

RUNS = Path("/scratch/cowifornia/nd-rl/olmo_elicit/runs")
OUT = Path("/scratch/cowifornia/nd-rl/olmo_elicit/tokens_summary")
SETS = ("all", "train", "test", "reach")
QS = (0.01, 0.05, 0.10)
JS_EPS = (0.01, 0.1)
LN2 = math.log(2.0)


def load_flat(path: Path) -> dict:
    """All arrays once (np.load's lazy members decompress on every access), plus per-position response index and the
    think mask (position < think_end of its response)."""
    with np.load(path) as z:
        a = {k: z[k] for k in z.files}
    n = a["n_tok"].astype(np.int64)
    a["resp"] = np.repeat(np.arange(len(n)), n)
    local = np.arange(int(n.sum())) - np.repeat(a["offset"][:-1], n)
    a["think"] = local < np.repeat(a["think_end"].astype(np.int64), n)
    a["pid"] = a["pid"].astype(str)
    a["set"] = a["set"].astype(str)
    return a


def _h(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """0.5 a ln(2a/(a+b)) + 0.5 b ln(2b/(a+b)), with 0 ln 0 = 0."""
    s = a + b
    with np.errstate(divide="ignore", invalid="ignore"):
        ta = np.where(a > 0, a * np.log(np.where(a > 0, 2 * a / np.where(s > 0, s, 1), 1)), 0.0)
        tb = np.where(b > 0, b * np.log(np.where(b > 0, 2 * b / np.where(s > 0, s, 1), 1)), 0.0)
    return 0.5 * (ta + tb)


def js_topk(
    pi: np.ndarray, pl: np.ndarray, qi: np.ndarray, ql: np.ndarray, chunk: int = 40000
) -> dict[str, np.ndarray]:
    """Per position: approximate JS on the union of the top-k supports (each truncated and renormalised), the exact
    coarse-grained lower bound, and the top-k masses of each model and of the shared ids."""
    n = len(pi)
    out = {k: np.empty(n) for k in ("js", "js_lb", "mass_p", "mass_q", "shared_p", "shared_q")}
    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        p = np.exp(pl[s:e].astype(np.float64))
        q = np.exp(ql[s:e].astype(np.float64))
        eq = pi[s:e, :, None] == qi[s:e, None, :]
        pm, qm = eq.any(2), eq.any(1)
        q_at_p = (eq * q[:, None, :]).sum(
            2
        )  # q's probability of each of p's ids (0 when not in q's top k)
        mp, mq = p.sum(1), q.sum(1)
        P, Q, QP = p / mp[:, None], q / mq[:, None], q_at_p / mq[:, None]
        out["js"][s:e] = (
            np.where(pm, _h(P, QP), 0.0).sum(1)
            + 0.5 * LN2 * (P * ~pm).sum(1)
            + 0.5 * LN2 * (Q * ~qm).sum(1)
        )
        ip, iq = (p * pm).sum(1), q_at_p.sum(1)
        rp, rq = np.clip(1 - ip, 0, None), np.clip(1 - iq, 0, None)
        out["js_lb"][s:e] = np.where(pm, _h(p, q_at_p), 0.0).sum(1) + _h(rp, rq)
        out["mass_p"][s:e], out["mass_q"][s:e] = mp, mq
        out["shared_p"][s:e], out["shared_q"][s:e] = ip, iq
    return out


def per_position(cur: dict, ref: dict) -> dict[str, np.ndarray]:
    for k in ("pid", "n_tok", "think_end", "token"):
        assert np.array_equal(cur[k], ref[k]), f"{k} differs between the two token files"
    js = js_topk(cur["topk_ids"], cur["topk_logp"], ref["topk_ids"], ref["topk_logp"])
    return {
        "js": js["js"],
        "js_lb": js["js_lb"],
        "mass_t": js["mass_p"],
        "mass_ref": js["mass_q"],
        "shared_t": js["shared_p"],
        "top1": cur["rank"] == 0,
        "top5": cur["rank"] < 5,
        "out20": cur["rank"] >= 20,
        "argmax_agree": cur["topk_ids"][:, 0] == ref["topk_ids"][:, 0],
        "ref_argmax_in_top5": (cur["topk_ids"][:, :5] == ref["topk_ids"][:, :1]).any(1),
        "excess": cur["nll"].astype(np.float64) - ref["nll"].astype(np.float64),
        "ent_ref": ref["entropy"].astype(np.float64),
        "nll_t": cur["nll"].astype(np.float64),
    }


RATIO = {  # readout -> (per-position quantity, transform)
    "js_mean": "js",
    "js_lb_mean": "js_lb",
    "share_js_gt_0.01": ("js", 0.01),
    "share_js_gt_0.1": ("js", 0.1),
    "share_js_lb_gt_0.01": ("js_lb", 0.01),
    "share_js_lb_gt_0.1": ("js_lb", 0.1),
    "share_top1": "top1",
    "share_top5": "top5",
    "share_outside_top20": "out20",
    "share_argmax_agree": "argmax_agree",
    "share_ref_argmax_in_top5": "ref_argmax_in_top5",
    "excess_mean": "excess",
    "mass_t_mean": "mass_t",
    "mass_ref_mean": "mass_ref",
    "share_mass_t_lt_0.9": ("mass_t", -0.9),
}


def _qty(pp: dict, spec) -> np.ndarray:
    if isinstance(spec, tuple):
        k, thr = spec
        return (pp[k] < -thr).astype(float) if thr < 0 else (pp[k] > thr).astype(float)
    return pp[spec].astype(float)


def top_share(v: np.ndarray, q: float, order_by: np.ndarray | None = None) -> float:
    """Share of sum(v) carried by the top ceil(q n) positions ranked by order_by (default v itself), descending."""
    tot = v.sum()
    if len(v) == 0 or tot <= 0:
        return float("nan")
    k = max(1, math.ceil(q * len(v)))
    key = v if order_by is None else order_by
    idx = np.argpartition(-key, k - 1)[:k]
    return float(v[idx].sum() / tot)


def deciles(ent: np.ndarray, pp: dict, sel: np.ndarray) -> list[dict]:
    e = ent[sel]
    n = len(e)
    if n < 10:
        return []
    order = np.argsort(e, kind="stable")
    b = np.empty(n, dtype=np.int64)
    b[order] = np.arange(n) * 10 // n
    ex = pp["excess"][sel]
    tot, pos = ex.sum(), np.clip(ex, 0, None).sum()
    out = []
    for d in range(10):
        m = b == d
        out.append(
            {
                "decile": d + 1,
                "n": int(m.sum()),
                "ent_ref_mean": float(e[m].mean()),
                "ent_ref_range": [float(e[m].min()), float(e[m].max())],
                "excess_mean": float(ex[m].mean()),
                "share_of_net_excess": float(ex[m].sum() / tot) if tot > 0 else float("nan"),
                "share_of_positive_excess": float(np.clip(ex[m], 0, None).sum() / pos)
                if pos > 0
                else float("nan"),
                "js_mean": float(pp["js"][sel][m].mean()),
                "share_top1": float(pp["top1"][sel][m].mean()),
            }
        )
    return out


def summarise(
    pp: dict, sel: np.ndarray, resp: np.ndarray, rng: np.random.Generator, B: int
) -> dict:
    n = int(sel.sum())
    out: dict = {"n_positions": n}
    if n == 0:
        return out
    r_ids, r_inv = np.unique(resp[sel], return_inverse=True)
    R = len(r_ids)
    cnt = np.bincount(r_inv, minlength=R).astype(float)
    bi = rng.integers(0, R, size=(B, R)) if B > 0 else None
    for name, spec in RATIO.items():
        v = _qty(pp, spec)[sel]
        sums = np.bincount(r_inv, weights=v, minlength=R)
        pt = float(v.mean())
        out[name] = pt
        if bi is not None:
            out[name + "_ci"] = ci(sums[bi].sum(1) / cnt[bi].sum(1))
    ex = pp["excess"][sel]
    js = pp["js"][sel]
    out["n_responses"] = R
    out["excess_sum"] = float(ex.sum())
    out["share_positions_excess_positive"] = float((ex > 0).mean())
    for q in QS:
        tag = f"{round(q * 100)}pct"
        out[f"excess_share_top{tag}_net"] = top_share(ex, q)
        pos = np.clip(ex, 0, None)
        out[f"excess_share_top{tag}_positive"] = top_share(pos, q)
        tot = ex.sum()
        if tot > 0:
            k = max(1, math.ceil(q * n))
            idx = np.argpartition(-js, k - 1)[:k]
            out[f"excess_share_top{tag}_by_js_net"] = float(ex[idx].sum() / tot)
        else:
            out[f"excess_share_top{tag}_by_js_net"] = float("nan")
    out["spearman_excess_vs_ent_ref"] = _spearman_fast(ex, pp["ent_ref"][sel])
    out["entropy_deciles"] = deciles(pp["ent_ref"], pp, sel)
    out["reading_reranking"] = (
        bool(out["share_top5"] >= 0.99 and out["excess_share_top5pct_net"] >= 0.5)
        if np.isfinite(out["excess_share_top5pct_net"])
        else None
    )
    return out


def _avg_rank(x: np.ndarray) -> np.ndarray:
    """Ranks with ties averaged (as scipy's rankdata), in numpy."""
    order = np.argsort(x, kind="stable")
    xs = x[order]
    first = np.r_[True, xs[1:] != xs[:-1]]
    grp = np.cumsum(first) - 1
    starts = np.flatnonzero(first)
    ends = np.r_[starts[1:], len(x)]
    avg = (starts + ends - 1) / 2.0
    r = np.empty(len(x))
    r[order] = avg[grp]
    return r


def _spearman_fast(x: np.ndarray, y: np.ndarray) -> float | None:
    if len(x) < 3:
        return None
    rx, ry = _avg_rank(x), _avg_rank(y)
    if rx.std() == 0 or ry.std() == 0:
        return None
    c = np.corrcoef(rx, ry)[0, 1]
    return float(c) if np.isfinite(c) else None


def step0_check(a: dict, step0_path: Path) -> float | None:
    """Max |per-response NLL sum in the npz - think_nats + answer_nats in step0.json| over all responses."""
    if not step0_path.exists():
        return None
    s0 = json.loads(step0_path.read_text())
    sums = np.bincount(a["resp"], weights=a["nll"].astype(np.float64), minlength=len(a["pid"]))
    ref = {}
    for name in ("train", "test", "reach"):
        b = s0.get(name)
        if b:
            for pid, t, an in zip(b["pids"], b["think_nats"], b["answer_nats"], strict=True):
                ref[pid] = t + an
    diffs = [abs(sums[j] - ref[p]) for j, p in enumerate(a["pid"]) if p in ref]
    return float(max(diffs)) if diffs else None


def analyze(runs: Path, B: int = 1000, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    files = {c: runs / c / "edl" / "step0_tokens.npz" for c in CKPTS}
    files = {c: p for c, p in files.items() if p.exists()}
    res: dict = {}
    if FINAL not in files:
        return {"checkpoints": {}, "note": f"no {FINAL} token file yet"}
    ref = load_flat(files[FINAL])
    for c, p in files.items():
        t0 = time.time()
        cur = ref if c == FINAL else load_flat(p)
        pp = per_position(cur, ref)
        segm = {
            "total": np.ones(len(cur["token"]), bool),
            "think": cur["think"],
            "answer": ~cur["think"],
        }
        setm = {"all": np.ones(len(cur["token"]), bool)}
        for s in ("train", "test", "reach"):
            setm[s] = np.repeat(cur["set"] == s, cur["n_tok"].astype(np.int64))
        r: dict = {
            "step": STEP[c],
            "off_released_line": c in OFF_LINE,
            "file": str(p),
            "is_reference": c == FINAL,
            "step0_check_max_abs_diff_nats": step0_check(cur, p.parent / "step0.json"),
            "sets": {},
        }
        for s, sm in setm.items():
            r["sets"][s] = {
                seg: summarise(pp, sm & segm[seg], cur["resp"], rng, B if s == "all" else 0)
                for seg in SEGS
            }
        r["secs"] = round(time.time() - t0, 1)
        res[c] = r
    names = [
        c for c in CKPTS if c in res and c != FINAL
    ]  # the reference's distance to itself is 0 by construction
    steps = [STEP[c] for c in names]
    trends = {}
    for seg in SEGS:
        for key in (
            "js_mean",
            "share_js_gt_0.1",
            "share_top1",
            "share_top5",
            "excess_mean",
            "excess_share_top5pct_net",
        ):
            trends[f"spearman_{key}_{seg}_vs_step"] = spearman(
                steps, [res[c]["sets"]["all"][seg].get(key, float("nan")) for c in names]
            )
    return {
        "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "runs": str(runs),
        "reference": FINAL,
        "B": B,
        "seed": seed,
        "definitions": __doc__.split("Per checkpoint,")[1].strip(),
        "checkpoints": res,
        "trends": trends,
    }


def flatten(res: dict) -> list[dict]:
    rows = []
    for c in [x for x in CKPTS if x in res]:
        for s, segs in res[c]["sets"].items():
            for seg, d in segs.items():
                for k, v in d.items():
                    if (
                        k.endswith("_ci")
                        or isinstance(v, list | dict)
                        or v is None
                        or isinstance(v, bool)
                    ):
                        continue
                    lo, hi = d.get(k + "_ci", [None, None])
                    rows.append(
                        {
                            "ckpt": c,
                            "step": STEP[c],
                            "set": s,
                            "segment": seg,
                            "metric": k,
                            "value": v,
                            "lo": lo,
                            "hi": hi,
                        }
                    )
                for b in d.get("entropy_deciles", []):
                    for k in (
                        "excess_mean",
                        "share_of_net_excess",
                        "share_of_positive_excess",
                        "js_mean",
                        "share_top1",
                        "ent_ref_mean",
                    ):
                        rows.append(
                            {
                                "ckpt": c,
                                "step": STEP[c],
                                "set": s,
                                "segment": seg,
                                "metric": f"decile{b['decile']}_{k}",
                                "value": b[k],
                                "lo": None,
                                "hi": None,
                            }
                        )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--runs", type=Path, default=RUNS)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--B", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    s = analyze(a.runs, a.B, a.seed)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.with_suffix(".json").write_text(json.dumps(s, indent=1) + "\n")
    rows = flatten(s["checkpoints"])
    with open(a.out.with_suffix(".csv"), "w", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["ckpt", "step", "set", "segment", "metric", "value", "lo", "hi"]
        )
        w.writeheader()
        w.writerows(rows)
    print(
        f"{len(s['checkpoints'])} checkpoints {list(s['checkpoints'])} vs {FINAL}; wrote {a.out.with_suffix('.json')} and .csv ({len(rows)} rows)"
    )
    for c, r in s["checkpoints"].items():
        for seg in SEGS:
            d = r["sets"]["all"][seg]
            print(
                f"  {c:6s} {seg:6s} JS {d['js_mean']:.4f} (>0.1: {d['share_js_gt_0.1']:.3f})  top1 {d['share_top1']:.3f} top5 {d['share_top5']:.4f} >20 {d['share_outside_top20']:.4f}"
                f"  excess {d['excess_mean']:+.4f}  top5% carry {d['excess_share_top5pct_net']:.2f}  step0 check {r['step0_check_max_abs_diff_nats']}"
            )


if __name__ == "__main__":
    main()
