#!/usr/bin/env python3
"""The ladder: one row per crosscoder, Leon's Stage 1 classification, and the curve against RL
compute.

  python analyze.py [--art CL_ART]

Reads CL_ART/crosscoders/*/{meta,eval}.json and latents.pt. Writes ladder.json, ladder.md and
figures/cos_vs_rl.png here (or under --out).

Per crosscoder:
  cos_w     energy-weighted mean decoder cosine over live latents, weight = the RL model's held-out
  energy in the
            latent (sum over sources of sum_x f^2 ||d^B||^2); Leon's headline number (0.945 at his
            L3 end, 1.000 null)
  cos_tail  the cosine below which 5% of that energy lies
  n_rl      strict RL-only latents: dnorm < 0.1 and both Latent Scaling ratios < 0.3 (Leon's rule);
  n_base likewise
  e_rl      share of the RL model's held-out variance carried by strict RL-only latents
  dnorm_q   quantiles of dnorm over live latents (0.5 = equal norms; < 0.1 RL-only; > 0.9 base-only)
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ART = Path(os.environ.get("CL_ART", HERE / "artifacts"))
SRCS = ("b_roll", "a_roll", "pretrain")


def arrays(lat):
    return {k: np.asarray(v, dtype=np.float64) for k, v in lat.items()}


def ratio(num, den):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(np.abs(den) > 0, num / den, np.nan)


def classify(L, thr=0.3):
    live = sum(L[f"fire_{s}"] for s in SRCS) > 0
    na, nb = L["norm_a"], L["norm_b"]
    dn = 0.5 * ((na - nb) / np.maximum(na, nb) + 1)
    nu_eps_rl, nu_r_rl = (
        ratio(L["err_ref1_m0"], L["err_ref1_m1"]),
        ratio(L["rec_ref1_m0"], L["rec_ref1_m1"]),
    )
    nu_eps_b, nu_r_b = (
        ratio(L["err_ref0_m1"], L["err_ref0_m0"]),
        ratio(L["rec_ref0_m1"], L["rec_ref0_m0"]),
    )
    rl_cand, b_cand = live & (dn < 0.1), live & (dn > 0.9)
    return dict(
        live=live,
        dnorm=dn,
        rl_cand=rl_cand,
        base_cand=b_cand,
        shared=live & (dn >= 0.4) & (dn <= 0.6),
        rl=rl_cand & (nu_eps_rl < thr) & (nu_r_rl < thr),
        base=b_cand & (nu_eps_b < thr) & (nu_r_b < thr),
    )


def energy_share(L, ev, mask, m):
    idx = 0 if m == "a" else 1
    e = sum(float(L[f"energy_{m}_{s}"][mask].sum()) for s in SRCS)
    v = sum(float(ev["per_source"][s]["var"][idx]) for s in SRCS)
    return e / v if v else float("nan")


def weighted_cos(L, live):
    w = sum(L[f"energy_b_{s}"] for s in SRCS) * live
    cos = L["cos"]
    if w.sum() <= 0:
        return float("nan"), float("nan")
    order = np.argsort(cos)
    cw = np.cumsum(w[order]) / w.sum()
    tail = float(cos[order][np.searchsorted(cw, 0.05)])
    return float((w * cos).sum() / w.sum()), tail


def rl_seconds(run, stem):
    """Recomputed from the run summaries pulled to artifacts/summaries (meta.json's value used an
    earlier, wrong round mapping for the first crosscoders)."""
    import common as C

    C.CKPTS = ART / "summaries"
    return C.rl_seconds(run, stem)


def summarize(d):
    if not (d / "eval.json").exists():
        return None
    meta = json.loads((d / "meta.json").read_text())
    ev = json.loads((d / "eval.json").read_text())
    L = arrays(torch.load(d / "latents.pt", weights_only=False))
    c = classify(L)
    cos_w, tail = weighted_cos(L, c["live"])
    loss = ev["loss"]
    rec = [loss[s][f"m{m}_recovered"] for s in ("b_roll", "a_roll") for m in (0, 1)]
    run, stem = meta["b"].split(":")
    rl_secs = rl_seconds(run, stem)
    row = dict(
        dir=d.name,
        run=run,
        stem=stem,
        planted=meta.get("plant") is not None,
        layer=meta["layer"],
        cc_seed=meta["seed"],
        rl_secs=rl_secs,
        rl_hours=rl_secs / 3600,
        fvu_a=ev["fvu"][0],
        fvu_b=ev["fvu"][1],
        loss_recovered_min=float(min(rec)),
        l0=ev["l0"],
        dead_frac=ev["dead_frac"],
        n_live=int(c["live"].sum()),
        n_shared=int(c["shared"].sum()),
        n_rl_cand=int(c["rl_cand"].sum()),
        n_base_cand=int(c["base_cand"].sum()),
        n_rl=int(c["rl"].sum()),
        n_base=int(c["base"].sum()),
        e_rl=energy_share(L, ev, c["rl"], "b"),
        e_base=energy_share(L, ev, c["base"], "a"),
        cos_w=cos_w,
        cos_tail=tail,
        dnorm_q={
            q: float(np.quantile(c["dnorm"][c["live"]], q)) for q in (0.01, 0.05, 0.5, 0.95, 0.99)
        },
        norm_ratio_q={
            q: float(np.quantile((L["norm_b"] / np.maximum(L["norm_a"], 1e-9))[c["live"]], q))
            for q in (0.05, 0.5, 0.95)
        },
    )
    if "planted" in ev:
        row["planted_found"] = sum(
            1
            for p in ev["planted"]
            for cand in p["candidates"][:1]
            if cand["dnorm"] < 0.1
            and (cand["precision"] or 0) > 0.5
            and (cand["recall"] or 0) > 0.5
        )
        row["planted_detail"] = ev["planted"]
    return row


def table(rows):
    hdr = (
        "| run | ckpt | RL h | cos_w | cos 5% tail | n_rl | e_rl | n_base | dnorm 1/50/99% "
        "| FVU a/b | loss rec. | live | planted |"
    )
    out = [hdr, "|" + "---|" * (hdr.count("|") - 1)]
    for r in rows:
        dq = r["dnorm_q"]
        out.append(
            f"| {r['run']} | {r['stem']}{' +plant' if r['planted'] else ''} | {r['rl_hours']:.2f} "
            f"| {r['cos_w']:.3f} | "
            f"{r['cos_tail']:.2f} | {r['n_rl']} | {r['e_rl']:.2e} | {r['n_base']} | "
            f"{dq[0.01]:.2f}/{dq[0.5]:.2f}/{dq[0.99]:.2f} | {r['fvu_a']:.3f}/{r['fvu_b']:.3f} | "
            f"{r['loss_recovered_min']:.2f} | {r['n_live']} | {r.get('planted_found', '')} |"
        )
    return "\n".join(out)


def figure(rows, path):
    """Three panels against RL GPU-hours: per seed, one EI line through the 3 h run (v4), its
    independent 3 h twin (v2 final) and the resumed run (v5); the control arm (cpt_*) dashed; the
    two-pretrains "null" (v2 vs v4 pretrain, same seed) as a dotted floor."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    keys = ("cos_w", "cos_tail", "n_rl")
    labels = (
        "energy-weighted decoder cosine",
        "cosine at the 5% energy tail",
        "strict RL-only latents",
    )
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    colors = {0: "tab:blue", 1: "tab:orange"}
    plain = [r for r in rows if not r["planted"]]
    for seed in sorted({int(r["run"][-1]) for r in plain}):
        c = colors.get(seed, "gray")
        ei = sorted(
            (
                r
                for r in plain
                if r["run"].endswith(f"s{seed}")
                and r["run"].startswith(("v4", "v5"))
                and r["rl_secs"] > 0
            ),
            key=lambda r: r["rl_secs"],
        )
        twin = [r for r in plain if r["run"] == f"v2_untied-ei_s{seed}" and r["stem"] == "final"]
        null = [r for r in plain if r["run"] == f"v2_untied-ei_s{seed}" and r["stem"] == "pretrain"]
        cpt = sorted((r for r in plain if r["run"] == f"cpt_s{seed}"), key=lambda r: r["rl_secs"])
        for ax, k in zip(axes, keys, strict=True):
            ax.plot(
                [r["rl_hours"] for r in ei],
                [r[k] for r in ei],
                "o-",
                c=c,
                ms=4,
                label=f"EI seed {seed}",
            )
            if twin:
                ax.plot(
                    [twin[0]["rl_hours"]],
                    [twin[0][k]],
                    "s",
                    c=c,
                    ms=7,
                    mfc="none",
                    label=f"3 h twin run, seed {seed}",
                )
            if cpt:
                ax.plot(
                    [r["rl_hours"] for r in cpt],
                    [r[k] for r in cpt],
                    "^--",
                    c=c,
                    ms=5,
                    label=f"CPT control, seed {seed}",
                )
            if null and k != "n_rl":
                ax.axhline(null[0][k], ls=":", c=c, lw=1)
            if null and k == "n_rl":
                ax.axhline(null[0][k], ls=":", c=c, lw=1)
    for ax, yl in zip(axes, labels, strict=True):
        ax.set_xlabel("RL GPU-hours (fine-tune rounds for the control)")
        ax.set_ylabel(yl)
        ax.grid(alpha=0.3)
    axes[0].axhline(0.945, ls="--", c="gray", lw=0.8)
    axes[0].text(
        axes[0].get_xlim()[0],
        0.95,
        " Leon's L3 end (0.29x GRPO)",
        va="bottom",
        fontsize=7,
        color="gray",
    )
    axes[0].text(
        axes[0].get_xlim()[0],
        0.31,
        " dotted: two independent pretrains, same seed",
        va="bottom",
        fontsize=7,
    )
    axes[0].legend(fontsize=6, loc="lower left")
    fig.suptitle("Base-vs-checkpoint crosscoders at layer 3 along the long expert-iteration runs")
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--art", default=None)
    ap.add_argument("--out", default=str(HERE))
    a = ap.parse_args()
    import common as C

    art = Path(a.art) if a.art else C.ART
    rows = [r for d in sorted((art / "crosscoders").glob("*")) if (r := summarize(d))]
    rows.sort(key=lambda r: (r["run"], r["planted"], r["rl_secs"]))
    out = Path(a.out)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    (out / "ladder.json").write_text(json.dumps(rows, indent=1))
    (out / "ladder.md").write_text(table(rows) + "\n")
    if any(not r["planted"] for r in rows):
        figure(rows, out / "figures" / "cos_vs_rl.png")
    print(table(rows))


if __name__ == "__main__":
    main()
