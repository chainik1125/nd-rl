# ruff: noqa: E501
"""Stage 1 readouts per checkpoint from `olmo_edl.py run` outputs (SPEC.md Stage 1; PREREG.md Stage 1, Amendments 1-2).

    analyze_edl.py [--runs /scratch/cowifornia/nd-rl/olmo_elicit/runs] [--out /scratch/cowifornia/nd-rl/olmo_elicit/edl_summary]
                   [--B 2000] [--seed 0]
reads runs/<ckpt>/edl/{edl_rows.jsonl, step0.json} for every checkpoint present (any subset) and writes
<out>.json and <out>.csv.

EDL, two unit conventions (both reported; nats unless "per token"):
  per-example sums (the paper's):  EDL(n') = sum_{i<=n'} l_i - n' * mean_j L_j(theta_n')
                                   per token: EDL(n') / sum_{i<=n'} T_i
  length-normalised:               EDL_norm/tok(n') = sum_{i<=n'} l_i / sum_{i<=n'} T_i - sum_j L_j(theta_n') / sum_j T_j
                                   in nats: EDL_norm/tok(n') * sum_{i<=n'} T_i
  l_i = online NLL (nats) of the i-th training response before the update on it, T_i its response tokens, L_j the
  NLL of test response j. Train and test responses differ in length, so the per-example form charges the length
  difference as excess; the normalised form compares per-token rates. Segment s (think / answer) uses that segment's
  nats and tokens on both terms; total = think + answer.
EDL* (PREREG Amendment 2): lr with the smallest mean-over-orders online code length sum_i l_i (total), EDL(64) at that
  lr averaged over orders, plus ln(number of lrs) on the total (ndbench `summarise_edl`); secondary: the min over lr of
  mean-over-orders EDL(64) + ln(number of lrs) (SPEC's formula), with the lr re-chosen inside every bootstrap
  replicate. Segment values are taken at the total's lr, without the ln term. Per token: divided by sum_i T_i (total or
  segment tokens of the 64 training responses).
Intervals: 95 % percentile bootstrap, B replicates, resampling test prompts (and, where a checkpoint has two orders,
  the orders, jointly); the online code length is a fixed sequence and is not resampled. L_0 intervals resample the
  prompts of the set they describe. The gap share resamples the test prompts of the checkpoint and of s2300 with the
  same indices (paired).
Readouts per checkpoint: L_0 (train stream / test / reach / train+test, nats per token, by segment); EDL* (both lr
  rules x both conventions); the EDL-per-token curve over n' at the EDL* lr and its shape; L_64 (test, at the EDL*
  lr); the share of the gap to s2300 closed by 64 examples, (L_0(t) - L_64(t)) / (L_0(t) - L_0(s2300)), per token on
  the test set; per-order EDL(64) (C3); L_test(64) - L_test(0) (C2). Trends: Spearman of L_0 and EDL* with RL step.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np

RUNS = Path("/scratch/cowifornia/nd-rl/olmo_elicit/runs")
OUT = Path("/scratch/cowifornia/nd-rl/olmo_elicit/edl_summary")
CKPTS = ["dpo", "s0050", "s0100", "s0200", "s0500", "s0750", "s1100", "s1600", "s2300"]
STEP = {c: (0 if c == "dpo" else int(c[1:])) for c in CKPTS}
OFF_LINE = {
    "s0200"
}  # olmo-weights review: steps 150 / 200 lie on an abandoned branch of the RL run
FINAL = "s2300"
SEGS = ("total", "think", "answer")
CONVS = ("sum", "norm")


# ------------------------------------------------------------------------------------- arithmetic


def edl_sum(online: np.ndarray, test_mean: float | np.ndarray, n: int) -> float | np.ndarray:
    """Paper form: sum of the first n online losses minus n times the mean test loss (nats). Same as ndbench
    metrics.edl(online[:n], test_mean)."""
    return online[:n].sum() - n * test_mean


def edl_norm_per_tok(
    online: np.ndarray, online_tok: np.ndarray, test_nats_sum, test_tok_sum, n: int
):
    """Length-normalised: online nats per token over the first n minus test nats per token (nats per token)."""
    return online[:n].sum() / online_tok[:n].sum() - np.asarray(test_nats_sum) / np.asarray(
        test_tok_sum
    )


def seg_vals(block: dict, seg: str) -> np.ndarray:
    if seg == "total":
        return np.asarray(block["think_nats"], float) + np.asarray(block["answer_nats"], float)
    return np.asarray(block[f"{seg}_nats"], float)


def seg_toks(block: dict, seg: str) -> np.ndarray:
    if seg == "total":
        return np.asarray(block["n_think"], float) + np.asarray(block["n_answer"], float)
    return np.asarray(block[f"n_{seg}"], float)


def ci(a: np.ndarray) -> list[float]:
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return [float("nan"), float("nan")]
    return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]


def spearman(x, y) -> float | None:
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size < 3:
        return None

    def rank(v):
        order = v.argsort(kind="stable")
        r = np.empty(v.size)
        r[order] = np.arange(v.size)
        for u in np.unique(v):  # average ties
            r[v == u] = r[v == u].mean()
        return r

    rx, ry = rank(x), rank(y)
    if rx.std() == 0 or ry.std() == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


# ------------------------------------------------------------------------------------------ input


def load_ckpt(d: Path) -> tuple[dict | None, list[dict], list[dict]]:
    """(step0, main rows (latest per (lr, order); ok / nonfinite), conv rows of --conv-rows best)."""
    s0p, rp = d / "step0.json", d / "edl_rows.jsonl"
    step0 = json.loads(s0p.read_text()) if s0p.exists() else None
    main: dict = {}
    conv: dict = {}
    if rp.exists():
        for ln in rp.read_text().splitlines():
            try:
                r = json.loads(ln)
            except json.JSONDecodeError:
                continue
            sch = r.get("schema", "")
            if sch.startswith("olmo_edl_row") and r.get("status") in ("ok", "nonfinite"):
                main[(float(r["lr"]), int(r["order"]))] = r  # a rerun row replaces the earlier one
            elif sch.startswith("olmo_edl_conv"):
                conv[(float(r["lr"]), int(r["order"]))] = r
    return step0, list(main.values()), list(conv.values())


def discover(runs: Path) -> dict[str, tuple[dict | None, list[dict], list[dict]]]:
    out = {}
    for c in CKPTS:
        d = runs / c / "edl"
        if d.is_dir():
            s0, rows, conv = load_ckpt(d)
            if s0 is not None or rows:
                out[c] = (s0, rows, conv)
    return out


# --------------------------------------------------------------------------------------- analysis


class Ckpt:
    """One checkpoint's rows. online(lr, order, seg) -> (nats, tokens) in stream order; test_at(lr, order, n', seg) ->
    per-example nats of the test set after n' steps; test_conv(lr, order, seg) -> after the extra epochs."""

    def __init__(
        self, name: str, step0: dict | None, rows: list[dict], conv_rows: list[dict] | None = None
    ):
        self.name, self.step0, self.rows = name, step0, rows
        ok = [r for r in rows if r["status"] == "ok"]
        self.lrs = sorted({float(r["lr"]) for r in rows})
        self.orders = sorted({int(r["order"]) for r in ok})
        self.status = {f"{float(r['lr']):g}/{int(r['order'])}": r["status"] for r in rows}
        self.ok = {(float(r["lr"]), int(r["order"])): r for r in ok}
        ref = ok[0] if ok else None
        self.test_pids = (
            ref["test"]["pids"] if ref else (step0 or {}).get("test", {}).get("pids")
        ) or []
        self.n = ref["n"] if ref else (step0 or {}).get("n")
        self.evals = sorted(int(k) for k in ref["evals"]) if ref else []
        for r in ok:
            assert r["test"]["pids"] == self.test_pids, f"{name}: test pids differ between rows"
        self.conv: dict = {}
        for k, r in self.ok.items():
            if r.get("conv", {}).get("status") == "ok":
                self.conv[k] = r["conv"]
        for r in conv_rows or []:
            if r["conv"].get("status") == "ok" and (float(r["lr"]), int(r["order"])) in self.ok:
                self.conv[(float(r["lr"]), int(r["order"]))] = {**r["conv"], "mode": r.get("mode")}

    def online(self, lr: float, order: int, seg: str) -> tuple[np.ndarray, np.ndarray]:
        r = self.ok[(lr, order)]
        return seg_vals(r["online"], seg), seg_toks(r["online"], seg)

    def test_at(self, lr: float, order: int, n: int, seg: str) -> np.ndarray:
        return seg_vals(self.ok[(lr, order)]["evals"][str(n)]["test"], seg)

    def test_conv(self, lr: float, order: int, seg: str) -> np.ndarray:
        return seg_vals(self.conv[(lr, order)]["test"], seg)

    def test_tok(self, seg: str) -> np.ndarray:
        return seg_toks(next(iter(self.ok.values()))["test"], seg)

    def orders_of(self, lr: float, need_conv: bool = False) -> list[int]:
        return [
            o for o in self.orders if (lr, o) in self.ok and (not need_conv or (lr, o) in self.conv)
        ]

    def has_eval(self, lr: float, n: int) -> bool:
        os_ = self.orders_of(lr)
        return bool(os_) and all(
            str(n) in self.ok[(lr, o)]["evals"] and "test" in self.ok[(lr, o)]["evals"][str(n)]
            for o in os_
        )

    def code_length(self, lr: float, seg: str = "total") -> float:
        os_ = self.orders_of(lr)
        if not os_:
            return float("inf")
        v = float(np.mean([self.online(lr, o, seg)[0].sum() for o in os_]))
        return v if math.isfinite(v) else float("inf")


def boot_indices(rng, m: int, n_orders: int, B: int) -> tuple[np.ndarray, np.ndarray]:
    ti = rng.integers(0, m, size=(B, m))
    oi = rng.integers(0, n_orders, size=(B, n_orders)) if n_orders > 1 else np.zeros((B, 1), int)
    return ti, oi


def edl_point(
    c: Ckpt,
    lr: float,
    n: int,
    seg: str,
    conv: str,
    ti: np.ndarray | None = None,
    oi: np.ndarray | None = None,
    per_tok: bool = False,
    test: str = "evals",
) -> np.ndarray:
    """EDL(n) in nats for one lr, mean over orders (per_tok: each order's EDL divided by its own first-n online tokens
    before the mean); test="conv": the residual is the test loss after the extra epochs (n = all n online examples).
    With bootstrap indices, an array of B replicates."""
    os_ = c.orders_of(lr, need_conv=test == "conv")
    ttok = c.test_tok(seg)
    per_order = []
    for o in os_:
        on, ontok = c.online(lr, o, seg)
        tv = c.test_conv(lr, o, seg) if test == "conv" else c.test_at(lr, o, n, seg)
        div = ontok[:n].sum() if per_tok else 1.0
        if conv == "sum":
            tm = tv.mean() if ti is None else tv[ti].mean(axis=1)
            per_order.append(edl_sum(on, tm, n) / div)
        else:
            ts, tt = (
                (tv.sum(), ttok.sum()) if ti is None else (tv[ti].sum(axis=1), ttok[ti].sum(axis=1))
            )
            per_order.append(edl_norm_per_tok(on, ontok, ts, tt, n) * ontok[:n].sum() / div)
    arr = np.asarray(per_order)  # (orders,) or (orders, B)
    if ti is None:
        return arr.mean(axis=0)
    if len(os_) > 1 and oi is not None and oi.shape[1] == len(os_):
        return np.take_along_axis(arr.T, oi, axis=1).mean(axis=1)
    return arr.mean(axis=0)


def online_tokens(c: Ckpt, n: int, seg: str) -> float:
    k = next(iter(c.ok))
    return float(c.online(k[0], k[1], seg)[1][:n].sum())


def l_block(block: dict, seg: str, rng, B: int) -> dict:
    v, t = seg_vals(block, seg), seg_toks(block, seg)
    k = len(v)
    idx = rng.integers(0, k, size=(B, k))
    bs = v[idx].sum(axis=1) / t[idx].sum(axis=1)
    return {
        "nats_per_tok": float(v.sum() / t.sum()),
        "ci": ci(bs),
        "nats_per_example": float(v.mean()),
        "tokens": int(t.sum()),
        "examples": k,
    }


def star_entry(
    c: Ckpt, lr: float, n: int, seg: str, conv: str, ti, oi, add: float, test: str = "evals"
) -> dict:
    pt = float(edl_point(c, lr, n, seg, conv, test=test)) + add
    bs = edl_point(c, lr, n, seg, conv, ti, oi, test=test) + add
    tok = online_tokens(c, n, seg)
    return {
        "lr": lr,
        "nats": pt,
        "ci": ci(bs),
        "per_tok": pt / tok,
        "per_tok_ci": ci(bs / tok),
        "tokens": tok,
        "ln_k": add,
    }


def test_rate(
    c: Ckpt, lr: float, seg: str, ti, n: int | None = None, conv_test: bool = False
) -> tuple[float, np.ndarray]:
    """Order-averaged test nats per token after n steps (or after the extra epochs), point and bootstrap."""
    os_ = c.orders_of(lr, need_conv=conv_test)
    v = np.mean(
        [c.test_conv(lr, o, seg) if conv_test else c.test_at(lr, o, n, seg) for o in os_], axis=0
    )
    tt = c.test_tok(seg)
    return float(v.sum() / tt.sum()), v[ti].sum(axis=1) / tt[ti].sum(axis=1)


def analyze_ckpt(c: Ckpt, rng, B: int) -> dict:
    out: dict = {
        "step": STEP[c.name],
        "off_released_line": c.name in OFF_LINE,
        "lrs": c.lrs,
        "orders": c.orders,
        "rows": c.status,
        "n": c.n,
        "m": len(c.test_pids),
    }
    s0 = c.step0
    if s0:
        out["L0"] = {}
        for nm in ("train", "test", "reach"):
            if s0.get(nm) and s0[nm]["pids"]:
                out["L0"][nm] = {seg: l_block(s0[nm], seg, rng, B) for seg in SEGS}
        if s0.get("train") and s0.get("test"):
            pooled = {
                k: list(s0["train"][k]) + list(s0["test"][k])
                for k in ("think_nats", "answer_nats", "n_think", "n_answer")
            }
            out["L0"]["train+test"] = {seg: l_block(pooled, seg, rng, B) for seg in SEGS}
    if not c.ok:
        return out
    n = c.n
    ln_k = math.log(len(c.lrs)) if c.lrs else 0.0
    ok_lrs = [lr for lr in c.lrs if c.orders_of(lr)]
    n_orders = max(len(c.orders_of(lr)) for lr in ok_lrs)
    ti, oi = boot_indices(rng, len(c.test_pids), n_orders, B)
    have64 = [lr for lr in ok_lrs if c.has_eval(lr, n)]
    cl = {seg: {lr: c.code_length(lr, seg) for lr in c.lrs} for seg in SEGS}
    lr_mdl = {seg: min(ok_lrs, key=lambda lr, seg=seg: cl[seg][lr]) for seg in SEGS}
    out["lr_mdl"] = lr_mdl
    out["by_lr"] = {}
    for lr in c.lrs:
        d: dict = {
            "code_length_nats": {
                seg: (cl[seg][lr] if math.isfinite(cl[seg][lr]) else None) for seg in SEGS
            },
            "orders": c.orders_of(lr),
            "status": {str(o): c.status.get(f"{lr:g}/{o}") for o in range(max(c.orders + [0]) + 1)},
        }
        if lr in have64:
            d["EDL64"] = {
                conv: {seg: float(edl_point(c, lr, n, seg, conv)) for seg in SEGS} for conv in CONVS
            }
            d["L_test_64_per_tok"] = {seg: test_rate(c, lr, seg, ti, n)[0] for seg in SEGS}
        if c.orders_of(lr, need_conv=True):
            d["EDL_conv"] = {
                conv: {seg: float(edl_point(c, lr, n, seg, conv, test="conv")) for seg in SEGS}
                for conv in CONVS
            }
        out["by_lr"][f"{lr:g}"] = d
    # EDL*: per segment, lr by that segment's own code length (lead) or min EDL (secondary); + ln k
    star: dict = {conv: {"mdl": {}, "min": {}} for conv in CONVS}
    for conv in CONVS:
        for seg in SEGS:
            if lr_mdl[seg] in have64:
                star[conv]["mdl"][seg] = star_entry(c, lr_mdl[seg], n, seg, conv, ti, oi, ln_k)
            if have64:
                pts = {lr: float(edl_point(c, lr, n, seg, conv)) for lr in have64}
                lr_min = min(pts, key=pts.get)
                bmat = np.stack([edl_point(c, lr, n, seg, conv, ti, oi) for lr in have64])
                bmin = bmat.min(axis=0) + ln_k
                tok = online_tokens(c, n, seg)
                star[conv]["min"][seg] = {
                    "lr": lr_min,
                    "nats": pts[lr_min] + ln_k,
                    "ci": ci(bmin),
                    "per_tok": (pts[lr_min] + ln_k) / tok,
                    "per_tok_ci": ci(bmin / tok),
                    "tokens": tok,
                    "ln_k": ln_k,
                    "lr_share_in_bootstrap": {
                        f"{lr:g}": float(np.mean(bmat.argmin(axis=0) == k))
                        for k, lr in enumerate(have64)
                    },
                }
    out["EDL_star"] = star
    # decomposition: think + answer at the total's lr (no ln k), sums to EDL*(total) - ln k
    if lr_mdl["total"] in have64:
        out["EDL_at_total_lr"] = {
            conv: {seg: star_entry(c, lr_mdl["total"], n, seg, conv, ti, oi, 0.0) for seg in SEGS}
            for conv in CONVS
        }
    # EDL per token over n', per segment at that segment's lr
    out["curve"] = {}
    for conv in CONVS:
        out["curve"][conv] = {}
        for seg in SEGS:
            lr = lr_mdl[seg]
            pts_n = [e for e in c.evals if e > 0 and c.has_eval(lr, e)]
            vals = [float(edl_point(c, lr, e, seg, conv, per_tok=True)) for e in pts_n]
            cis = [ci(edl_point(c, lr, e, seg, conv, ti, oi, per_tok=True)) for e in pts_n]
            out["curve"][conv][seg] = {
                "lr": lr,
                "n": pts_n,
                "per_tok": vals,
                "ci": cis,
                "shape": shape(vals),
            }
    # L_64, C2 (L_test(64) - L_test(0)) and per-order EDL(64), per segment at that segment's lr
    out["L64"], out["C2_test_drop"], out["per_order_EDL64"] = {}, {}, {conv: {} for conv in CONVS}
    for seg in SEGS:
        lr = lr_mdl[seg]
        if lr not in have64:
            continue
        p64, b64 = test_rate(c, lr, seg, ti, n)
        p0, b0 = test_rate(c, lr, seg, ti, 0)
        out["L64"][seg] = {"lr": lr, "nats_per_tok": p64, "ci": ci(b64)}
        out["C2_test_drop"][seg] = {"lr": lr, "nats_per_tok": p64 - p0, "ci": ci(b64 - b0)}
        for conv in CONVS:
            out["per_order_EDL64"][conv][seg] = {
                str(o): _edl_one_order(c, lr, o, n, seg, conv) for o in c.orders_of(lr)
            }
    # secondary: residual after the extra epochs ("Bits That Count"), at the lr that has it (segment's lr if present)
    conv_lrs = [lr for lr in ok_lrs if c.orders_of(lr, need_conv=True)]
    if conv_lrs:
        out["EDL_conv"], out["L_conv"] = {conv: {} for conv in CONVS}, {}
        for seg in SEGS:
            lr = (
                lr_mdl[seg]
                if lr_mdl[seg] in conv_lrs
                else min(conv_lrs, key=lambda x, seg=seg: cl[seg][x])
            )
            os_ = c.orders_of(lr, need_conv=True)
            eps = c.conv[(lr, os_[0])]["epochs"]
            for conv in CONVS:
                e = star_entry(c, lr, n, seg, conv, ti, oi, ln_k, test="conv")
                out["EDL_conv"][conv][seg] = {
                    **e,
                    "orders": os_,
                    "epochs": eps,
                    "mode": c.conv[(lr, os_[0])].get("mode"),
                }
            pc, bc = test_rate(c, lr, seg, ti, conv_test=True)
            out["L_conv"][seg] = {"lr": lr, "nats_per_tok": pc, "ci": ci(bc), "epochs": eps}
    return out


def _edl_one_order(c: Ckpt, lr: float, o: int, n: int, seg: str, conv: str) -> float:
    on, ontok = c.online(lr, o, seg)
    tv = c.test_at(lr, o, n, seg)
    if conv == "sum":
        return float(edl_sum(on, tv.mean(), n))
    return float(edl_norm_per_tok(on, ontok, tv.sum(), c.test_tok(seg).sum(), n) * ontok[:n].sum())


def shape(vals: list[float]) -> str:
    v = [x for x in vals if math.isfinite(x)]
    if len(v) < 2:
        return "too few points"
    d = np.diff(v)
    if (d < 0).all():
        return "falling throughout"
    if d[0] > 0:
        return "rising first"
    if (d > 0).all():
        return "rising throughout"
    return "falling first, then not monotone"


def gap_share(cs: dict[str, Ckpt], res: dict, rng, B: int) -> None:
    """(L_0(t) - L_x(t)) / (L_0(t) - L_0(s2300)) per token on the test set, x = 64 (and the extra-epochs residual),
    paired bootstrap over test prompts; per segment at that segment's lr."""
    f = cs.get(FINAL)
    if f is None or f.step0 is None:
        return
    for name, c in cs.items():
        r = res[name]
        if name == FINAL or not c.ok or c.step0 is None or "L64" not in r:
            continue
        if c.step0["test"]["pids"] != f.step0["test"]["pids"]:
            r["gap_share"] = {"error": "test pids differ from s2300's"}
            continue
        m = len(c.test_pids)
        idx = rng.integers(0, m, size=(B, m))
        r["gap_share"], r["gap_share_conv"] = {}, {}
        for seg in SEGS:
            tt = seg_toks(c.step0["test"], seg)
            l0 = seg_vals(c.step0["test"], seg)
            lf = seg_vals(f.step0["test"], seg)

            def share(ix, lx, l0=l0, lf=lf, tt=tt):
                T = tt[ix].sum(axis=-1)
                a0, ax, af = (
                    l0[ix].sum(axis=-1) / T,
                    lx[ix].sum(axis=-1) / T,
                    lf[ix].sum(axis=-1) / T,
                )
                return (a0 - ax) / (a0 - af)

            if seg in r["L64"]:
                lr = r["L64"][seg]["lr"]
                l64 = np.mean([c.test_at(lr, o, c.n, seg) for o in c.orders_of(lr)], axis=0)
                r["gap_share"][seg] = {
                    "lr": lr,
                    "share": float(share(np.arange(m), l64)),
                    "ci": ci(share(idx, l64)),
                    "L0_minus_L0final_per_tok": float((l0.sum() - lf.sum()) / tt.sum()),
                }
            if seg in r.get("L_conv", {}):
                lr = r["L_conv"][seg]["lr"]
                lc = np.mean(
                    [c.test_conv(lr, o, seg) for o in c.orders_of(lr, need_conv=True)], axis=0
                )
                r["gap_share_conv"][seg] = {
                    "lr": lr,
                    "share": float(share(np.arange(m), lc)),
                    "ci": ci(share(idx, lc)),
                }


def trends(res: dict) -> dict:
    names = [c for c in CKPTS if c in res]
    steps = [STEP[c] for c in names]
    out: dict = {"checkpoints": names, "n_points": len(names)}
    for nm in ("train", "test", "reach", "train+test"):
        for seg in SEGS:
            vals = [
                res[c].get("L0", {}).get(nm, {}).get(seg, {}).get("nats_per_tok", float("nan"))
                for c in names
            ]
            out[f"spearman_L0_{nm}_{seg}_vs_step"] = spearman(steps, vals)
    for conv in CONVS:
        for rule in ("mdl", "min"):
            for seg in SEGS:
                vals = [
                    res[c]
                    .get("EDL_star", {})
                    .get(conv, {})
                    .get(rule, {})
                    .get(seg, {})
                    .get("per_tok", float("nan"))
                    for c in names
                ]
                out[f"spearman_EDLstar_{rule}_{conv}_{seg}_vs_step"] = spearman(steps, vals)
    return out


def checks(res: dict) -> dict:
    """PREREG C1-C3 on the lead convention (code-length lr), total and answer."""
    out: dict = {}
    d, f = res.get("dpo"), res.get(FINAL)
    for seg in ("total", "answer"):
        if d and f:
            for conv in CONVS:
                a = f.get("EDL_star", {}).get(conv, {}).get("mdl", {}).get(seg)
                b = d.get("EDL_star", {}).get(conv, {}).get("mdl", {}).get(seg)
                if a and b:
                    out[f"C1_{conv}_{seg}"] = {
                        "EDLstar_s2300": a["nats"],
                        "EDLstar_dpo": b["nats"],
                        "ratio": a["nats"] / b["nats"] if b["nats"] else None,
                        "pass": bool(b["nats"] > 0 and a["nats"] < 0.1 * b["nats"]),
                    }
        if d and seg in d.get("C2_test_drop", {}):
            t = d["C2_test_drop"][seg]
            out[f"C2_{seg}"] = {**t, "pass": bool(t["nats_per_tok"] < 0 and t["ci"][1] < 0)}
        if (
            d
            and f
            and seg in d.get("EDL_star", {}).get("sum", {}).get("mdl", {})
            and seg in f.get("EDL_star", {}).get("sum", {}).get("mdl", {})
        ):
            span = abs(
                d["EDL_star"]["sum"]["mdl"][seg]["nats"] - f["EDL_star"]["sum"]["mdl"][seg]["nats"]
            )
            c3 = {}
            for name in CKPTS:
                po = res.get(name, {}).get("per_order_EDL64", {}).get("sum", {}).get(seg, {})
                if len(po) >= 2:
                    v = list(po.values())
                    c3[name] = {
                        "orders": po,
                        "diff": abs(v[0] - v[1]),
                        "pass": bool(abs(v[0] - v[1]) < span / 3),
                    }
            out[f"C3_sum_{seg}"] = {"dpo_s2300_span": span, "by_ckpt": c3}
    return out


def flatten(res: dict) -> list[dict]:
    rows = []

    def add(ck, metric, conv, seg, val, lo=None, hi=None, unit="", note=""):
        rows.append(
            {
                "ckpt": ck,
                "step": STEP[ck],
                "metric": metric,
                "convention": conv,
                "segment": seg,
                "value": val,
                "lo": lo,
                "hi": hi,
                "unit": unit,
                "note": note,
            }
        )

    for ck in [c for c in CKPTS if c in res]:
        r = res[ck]
        for nm, segd in r.get("L0", {}).items():
            for seg, v in segd.items():
                add(
                    ck,
                    f"L0_{nm}",
                    "",
                    seg,
                    v["nats_per_tok"],
                    *v["ci"],
                    "nats/token",
                    f"{v['examples']} prompts",
                )
        for key, label in (("EDL_star", "EDL_star"), ("EDL_conv", "EDL_conv")):
            for conv, rules in r.get(key, {}).items():
                items = rules.items() if key == "EDL_star" else [("mdl", rules)]
                for rule, segd in items:
                    for seg, v in segd.items():
                        note = f"lr {v['lr']:g}" + (
                            f", {v.get('epochs')} extra epochs" if key == "EDL_conv" else ""
                        )
                        add(ck, f"{label}_{rule}", conv, seg, v["nats"], *v["ci"], "nats", note)
                        add(
                            ck,
                            f"{label}_{rule}",
                            conv,
                            seg,
                            v["per_tok"],
                            *v["per_tok_ci"],
                            "nats/token",
                            note,
                        )
        for conv, segd in r.get("curve", {}).items():
            for seg, cv in segd.items():
                for e, v, c in zip(cv["n"], cv["per_tok"], cv["ci"], strict=True):
                    add(
                        ck,
                        f"EDL_curve_n{e}",
                        conv,
                        seg,
                        v,
                        *c,
                        "nats/token",
                        f"lr {cv['lr']:g}; {cv['shape']}",
                    )
        for key in ("L64", "L_conv", "C2_test_drop"):
            for seg, v in r.get(key, {}).items():
                add(ck, key, "", seg, v["nats_per_tok"], *v["ci"], "nats/token", f"lr {v['lr']:g}")
        for key in ("gap_share", "gap_share_conv"):
            for seg, v in r.get(key, {}).items():
                if isinstance(v, dict) and "share" in v:
                    add(ck, key, "", seg, v["share"], *v["ci"], "fraction", f"lr {v['lr']:g}")
    return rows


def analyze(runs: Path, B: int = 2000, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    found = discover(runs)
    cs = {k: Ckpt(k, s0, rows, conv) for k, (s0, rows, conv) in found.items()}
    res = {k: analyze_ckpt(c, rng, B) for k, c in cs.items()}
    gap_share(cs, res, rng, B)
    return {
        "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "runs": str(runs),
        "B": B,
        "seed": seed,
        "conventions": __doc__.split("EDL, two unit conventions")[1]
        .split("Readouts per checkpoint")[0]
        .strip(),
        "checkpoints": res,
        "trends": trends(res),
        "checks": checks(res),
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--runs", type=Path, default=RUNS)
    ap.add_argument(
        "--out", type=Path, default=OUT, help="output path stem (.json and .csv are added)"
    )
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    s = analyze(a.runs, a.B, a.seed)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.with_suffix(".json").write_text(json.dumps(s, indent=1, default=float) + "\n")
    rows = flatten(s["checkpoints"])
    with open(a.out.with_suffix(".csv"), "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "ckpt",
                "step",
                "metric",
                "convention",
                "segment",
                "value",
                "lo",
                "hi",
                "unit",
                "note",
            ],
        )
        w.writeheader()
        w.writerows(rows)
    print(
        f"{len(s['checkpoints'])} checkpoints: {list(s['checkpoints'])}; wrote {a.out.with_suffix('.json')} and .csv ({len(rows)} rows)"
    )
    for ck, r in s["checkpoints"].items():
        for seg in ("total", "answer"):
            e = r.get("EDL_star", {}).get("norm", {}).get("mdl", {}).get(seg)
            l0 = r.get("L0", {}).get("test", {}).get(seg)
            print(
                f"  {ck:6s} {seg:6s} L0 test {l0['nats_per_tok'] if l0 else float('nan'):.4f} nats/tok; "
                + (
                    f"EDL* (norm, code-length lr {e['lr']:g}) {e['per_tok']:.4f} nats/tok "
                    f"[{e['per_tok_ci'][0]:.4f}, {e['per_tok_ci'][1]:.4f}]"
                    if e
                    else "no EDL rows"
                )
            )
    print("trends:", json.dumps(s["trends"]))


if __name__ == "__main__":
    main()
