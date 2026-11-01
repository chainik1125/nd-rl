"""Stage 1 analysis: classify every trained crosscoder's latents and tabulate the counts.

    python -m code.experiments.current.circuit_diffing_20261101.stage1_analyze

Reads `/scratch/cowifornia/nd-rl/circuits/crosscoders/*/{meta,eval}.json` and `latents.pt`.
Writes `stage1.json` and `stage1.md` (one row per crosscoder; PREREG Stage 1 definitions), and
`figs/stage1_vs_R.{pdf,png}`.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .common import ART, HERE, PASSK

CC = ART / "crosscoders"
SRCS = ("b_roll", "a_roll", "pretrain")
TEASER_NAME = {  # registry label -> teaser_numbers checkpoint name (Charles149, seed 0)
    "kp16_st1": "grpo_think_vs_mini20_kp16_L2st1",
    "kp16_st2": "grpo_think_vs_mini20_kp16_L2st2",
    "kp16_st4": "grpo_think_vs_mini20_kp16_L2st4",
    "kp16_st8": "grpo_think_vs_mini20_kp16_L2st8",
    "kp16_st16": "grpo_think_vs_mini20_kp16_L2st16",
    "kp16_st32": "grpo_think_vs_mini20_kp16_L2st32",
    "kdense_st8": "grpo_think_vs_mini20_kdense_L2st8",
    "mini_it8": "grpo_think_vs_mini20_L2it8",
    "L2": "grpo_think_vs_L2",
    "L3": "grpo_think_vs_mb_kkl0",
    "ki78": "grpo_think_vs_mb_ki78",
}


def arrays(lat: dict[str, list[float]]) -> dict[str, np.ndarray]:
    return {k: np.asarray(v, dtype=np.float64) for k, v in lat.items()}


def ratio(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(np.abs(den) > 0, num / den, np.nan)


def classify(L: dict[str, np.ndarray], thr: float = 0.3) -> dict[str, np.ndarray]:
    live = sum(L[f"fire_{s}"] for s in SRCS) > 0
    na, nb = L["norm_a"], L["norm_b"]
    dn = 0.5 * ((na - nb) / np.maximum(na, nb) + 1)
    nu_eps_rl = ratio(L["err_ref1_m0"], L["err_ref1_m1"])
    nu_r_rl = ratio(L["rec_ref1_m0"], L["rec_ref1_m1"])
    nu_eps_b = ratio(L["err_ref0_m1"], L["err_ref0_m0"])
    nu_r_b = ratio(L["rec_ref0_m1"], L["rec_ref0_m0"])
    rl_cand = live & (dn < 0.1)
    b_cand = live & (dn > 0.9)
    return dict(
        live=live,
        dnorm=dn,
        rl_cand=rl_cand,
        base_cand=b_cand,
        shared=live & (dn >= 0.4) & (dn <= 0.6),
        rl=rl_cand & (nu_eps_rl < thr) & (nu_r_rl < thr),
        base=b_cand & (nu_eps_b < thr) & (nu_r_b < thr),
        nu_eps_rl=nu_eps_rl,
        nu_r_rl=nu_r_rl,
        nu_eps_b=nu_eps_b,
        nu_r_b=nu_r_b,
    )


def energy_share(L: dict[str, np.ndarray], ev: dict[str, Any], mask: np.ndarray, m: str) -> dict:
    """Share of model m's ('a' / 'b') held-out activation variance carried by `mask` latents."""
    idx = 0 if m == "a" else 1
    out = {}
    tot_e = tot_v = 0.0
    for s in SRCS:
        e = float(L[f"energy_{m}_{s}"][mask].sum())
        v = float(ev["per_source"][s]["var"][idx])
        out[s] = e / v
        tot_e, tot_v = tot_e + e, tot_v + v
    out["all"] = tot_e / tot_v
    return out


def summarize(d: Path) -> dict[str, Any] | None:
    if not (d / "eval.json").exists():
        return None
    meta = json.loads((d / "meta.json").read_text())
    ev = json.loads((d / "eval.json").read_text())
    L = arrays(torch.load(d / "latents.pt", weights_only=False))
    c = classify(L)
    b_label, b_seed = meta["b"].split(":")
    loss = ev["loss"]
    rec = [loss[s][f"m{m}_recovered"] for s in ("b_roll", "a_roll") for m in (0, 1)]
    row: dict[str, Any] = dict(
        dir=d.name,
        b=b_label,
        seed=int(b_seed),
        layer=meta["layer"],
        cc_seed=meta["seed"],
        n_latents=meta["n_latents"],
        k=meta["k"],
        R=meta["R_b"],
        fvu_a=ev["fvu"][0],
        fvu_b=ev["fvu"][1],
        loss_recovered_min=float(min(rec)),
        loss=loss,
        l0=ev["l0"],
        dead_frac=ev["dead_frac"],
        n_live=int(c["live"].sum()),
        n_shared=int(c["shared"].sum()),
        n_rl_cand=int(c["rl_cand"].sum()),
        n_base_cand=int(c["base_cand"].sum()),
    )
    for thr in (0.2, 0.3, 0.5):
        cc = classify(L, thr)
        row[f"n_rl_{thr}"] = int(cc["rl"].sum())
        row[f"n_base_{thr}"] = int(cc["base"].sum())
    row["N_RL"], row["N_base"] = row["n_rl_0.3"], row["n_base_0.3"]
    row["E_RL"] = energy_share(L, ev, c["rl"], "b")
    row["E_base"] = energy_share(L, ev, c["base"], "a")
    row["E_RL_cand"] = energy_share(L, ev, c["rl_cand"], "b")
    row["E_base_cand"] = energy_share(L, ev, c["base_cand"], "a")
    sh = c["shared"]
    lr = np.log2(L["norm_b"][sh] / L["norm_a"][sh])
    row["shared_frac_of_live"] = float(sh.sum() / max(1, c["live"].sum()))
    row["shared_median_abs_log2_norm_ratio"] = float(np.median(np.abs(lr))) if sh.any() else None
    row["shared_median_cos"] = float(np.median(L["cos"][sh])) if sh.any() else None
    row["live_median_cos"] = float(np.median(L["cos"][c["live"]]))
    # Δnorm histogram over live latents (20 bins)
    row["dnorm_hist"] = np.histogram(c["dnorm"][c["live"]], bins=20, range=(0, 1))[0].tolist()
    rl = c["rl"]
    if rl.any():
        fb, fa = L["fire_b_roll"][rl], L["fire_a_roll"][rl]
        row["rl_fire_ratio_b_over_a_median"] = float(np.median(ratio(fb, fa + 1e-9)))
        row["rl_fire_gen_share_median"] = float(
            np.median(ratio(L["fire_gen_b_roll"][rl], L["fire_b_roll"][rl] + 1e-12))
        )
    return row


def teaser() -> dict[str, dict[str, float]]:
    p = PASSK / "teaser_numbers.json"
    if not p.exists():
        return {}
    d = json.loads(p.read_text())
    return {c["checkpoint"]: c for c in d["checkpoints"]}


def main() -> int:
    rows = [r for d in sorted(CC.glob("*")) if (r := summarize(d)) is not None]
    tz = teaser()
    for r in rows:
        t = tz.get(TEASER_NAME.get(r["b"], ""), None) if r["seed"] == 0 else None
        r["gap1024"] = t["gapK"][0] if t else None
        r["gap1"] = t["gap1"][0] if t else None
    (HERE / "stage1.json").write_text(json.dumps(rows, indent=1))
    lines = [
        "# Stage 1 numbers (generated by `stage1_analyze.py`)",
        "",
        "Strict RL-only: Δnorm < 0.1, ν^ε < 0.3, ν^r < 0.3 (PREREG Stage 1). E: share of the "
        "model's held-out activation variance (all three sources). `rec` is the smallest "
        "loss-recovered over both models and both rollout sources. `gap1024`: Charles149 "
        "pass@1024 RL − base (points, seed 0, high_rl_passk teaser).",
        "",
        "| B | seed | layer | cc seed | R | FVU a / b | rec | L0 | live | shared | RL cand "
        "| N_RL (0.2/0.3/0.5) | E_RL | base cand | N_base (0.3) | E_base | gap1024 |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in sorted(rows, key=lambda r: (r["layer"], r["seed"], _R(r), r["cc_seed"])):
        g = "" if r["gap1024"] is None else f"{r['gap1024']:+.1f}"
        lines.append(
            f"| {r['b']} | {r['seed']} | {r['layer']} | {r['cc_seed']} "
            f"| {r['R']:.2e} "
            f"| {r['fvu_a']:.3f} / {r['fvu_b']:.3f} | {100 * r['loss_recovered_min']:.1f}% "
            f"| {r['l0']:.1f} | {r['n_live']} | {r['n_shared']} | {r['n_rl_cand']} "
            f"| {r['n_rl_0.2']} / {r['n_rl_0.3']} / {r['n_rl_0.5']} "
            f"| {100 * r['E_RL']['all']:.3f}% | {r['n_base_cand']} | {r['N_base']} "
            f"| {100 * r['E_base']['all']:.3f}% | {g} |"
        )
    (HERE / "stage1.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


def _R(r: dict[str, Any]) -> float:
    return r["R"] if math.isfinite(r["R"]) else 1e9


if __name__ == "__main__":
    raise SystemExit(main())
