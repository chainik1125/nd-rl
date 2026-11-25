# ruff: noqa: E501, SIM115
"""Expected generated tokens and H200 minutes per stage and checkpoint, from Dan's stored jobs (estimates, not
measurements). Prints the markdown table that STAGE2.md quotes.

Sources (nd-takehome @ 8e744a89 and its bucket, fetched under ext/):
  tokens per draw   H: his plain arm-B mean tokens per reach draw at each checkpoint (olmo_ifcurve/results.csv
                    tok_mean, arm B; dpo: arm-A reach rows) x the hinted / plain ratio, measured at dpo (olmo-iffu
                    T2 + T2c: 4089 / 4477 = 0.913) and s2300 (5473 / 6863 = 0.797), linear in step in between.
                    rep: arm-A tokens per draw of his F_dpo_ifb (4463) and F_s2300_ifb (6586) olmo-ifomega jobs.
  throughput        olmo_ifcurve/compute.csv tok_per_s of his 1 x H200 jobs by job size: 1550 tok/s for 300-500
                    draws, 1350 for 100-250 (smaller jobs are slower: fewer sequences in flight); load 1.5 min.
  T                 his olmo-answeronly R_dpo (4 conditions, 22.3 min generation) and R_s2300 (6.5 min), and
                    olmo-transplant's T1 / T2 answers (mean 385 / 84 tokens); ours has 2 of his conditions
                    (115 traces x 8 x {T, A1}): 15 min at a dpo-like checkpoint, 5 min at an s2300-like one,
                    interpolated by his arm-B answer-word decline (answers shorten by s0750).
  download          his fetch logs: 24-925 s for 61 GB, median about 100 s; 3 min assumed.
  EDL               NOT derived here (the EDL runner is another agent's); a placeholder of 25 min per
                    (lr grid x 1 order) pass set, 45 min with 2 orders, to be replaced by its measured timing.
Price: secure H200 $5.29 / h.
Usage: python cost_table.py
"""

from __future__ import annotations

import csv
import statistics as st
from pathlib import Path

DAN = Path("/scratch/cowifornia/nd-rl/olmo_elicit/ext/nd-takehome")
CKPTS = ["dpo", "s0050", "s0100", "s0200", "s0500", "s0750", "s1100", "s1600", "s2300"]
STEP = {c: (0 if c == "dpo" else int(c[1:])) for c in CKPTS}
PRICE = 5.29 / 60  # $ per minute
LOAD = 1.5
DL = 3.0
RATE = {"big": 1550.0, "small": 1350.0}
HINT_RATIO = {"dpo": 4089 / 4477, "s2300": 5473 / 6863}
REP_TOK = {"dpo": 4463, "s2300": 6586}
T_MIN = {"dpo": 15.0, "s2300": 5.0}
EDL_MIN = {1: 25.0, 2: 45.0}
TWO_ORDERS = {"dpo", "s0750", "s2300"}
MINI = {
    "dpo": ["edl", "rep"],
    "s0100": ["edl", "T", "H"],
    "s0500": ["edl", "T", "H"],
    "s1100": ["edl", "T", "H"],
    "s2300": ["edl", "rep"],
}
SECOND = {
    "s0050": ["edl", "T", "H"],
    "s0200": ["edl", "T", "H"],
    "s0750": ["edl", "T", "H"],
    "s1600": ["edl", "T", "H"],
}


def plain_tok() -> dict[str, float]:
    res = list(csv.DictReader(open(DAN / "olmo_ifcurve/results.csv")))  # noqa: SIM115
    out = {
        c: st.mean(float(r["tok_mean"]) for r in res if r["arm"] == "B" and r["ckpt"] == c)
        for c in CKPTS[1:]
    }
    out["dpo"] = st.mean(
        float(r["tok_mean"])
        for r in res
        if r["arm"] == "A" and r["ckpt"] == "dpo" and r["set"] == "reach"
    )
    return out


def ans_words() -> dict[str, float]:
    rows = list(csv.DictReader(open(DAN / "olmo_ifcurve/lengths.csv")))  # noqa: SIM115
    return {r["ckpt"]: float(r["ans_words_median"]) for r in rows if r["set"] == "armA_all"}


def est(c: str, stage: str, n_h: int = 8) -> tuple[float, float]:
    """(generated tokens in millions, H200 minutes)."""
    f = STEP[c] / 2300
    if stage == "download":
        return 0.0, DL
    if stage == "edl":
        return 0.0, EDL_MIN[2 if c in TWO_ORDERS else 1]
    if stage == "H":
        tok = plain_tok()[c] * ((1 - f) * HINT_RATIO["dpo"] + f * HINT_RATIO["s2300"])
        n = 48 * n_h
        return n * tok / 1e6, LOAD + n * tok / (RATE["big" if n >= 300 else "small"] * 60)
    if stage == "T":
        w = ans_words()
        g = (w[c] - w["s2300"]) / (
            w["dpo"] - w["s2300"]
        )  # 1 at dpo-like answer lengths, 0 at s2300-like
        gen = 920 * (g * (385 + 122) + (1 - g) * (84 + 80))
        return gen / 1e6, LOAD + T_MIN["s2300"] + g * (T_MIN["dpo"] - T_MIN["s2300"])
    if stage == "rep":
        tok = REP_TOK["dpo"] * (1 - f) + REP_TOK["s2300"] * f
        return 436 * tok / 1e6, LOAD + 436 * tok / (RATE["big"] * 60)
    raise ValueError(stage)


def table(plan: dict[str, list[str]], title: str) -> list[str]:
    L = [
        f"**{title}**",
        "",
        "| ckpt | step | download min | EDL min (placeholder) | T Mtok / min | H n=8 Mtok / min | H n=4 Mtok / min | rep Mtok / min | sampling min (H n=8) | total min (n=8) | $ (n=8) | $ (n=4) |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    tot8 = tot4 = samp = 0.0
    for c, stages in plan.items():
        cells, s8, s4 = {}, 0.0, 0.0
        for s in ("edl", "T", "rep"):
            if s in stages:
                t, m = est(c, s)
                cells[s] = f"{t:.2f} / {m:.0f}" if s != "edl" else f"{m:.0f}"
                s8 += m
                s4 += m
            else:
                cells[s] = "–"
        if "H" in stages:
            t8, m8 = est(c, "H", 8)
            t4, m4 = est(c, "H", 4)
            cells["H8"], cells["H4"] = f"{t8:.2f} / {m8:.0f}", f"{t4:.2f} / {m4:.0f}"
            s8 += m8
            s4 += m4
        else:
            cells["H8"] = cells["H4"] = "–"
        smp = s8 - (est(c, "edl")[1] if "edl" in stages else 0.0)
        s8 += DL
        s4 += DL
        tot8 += s8
        tot4 += s4
        samp += smp
        L.append(
            f"| {c} | {STEP[c]} | {DL:.0f} | {cells['edl']} | {cells['T']} | {cells['H8']} | {cells['H4']} | {cells['rep']} | "
            f"{smp:.0f} | {s8:.0f} | {s8 * PRICE:.2f} | {s4 * PRICE:.2f} |"
        )
    L += [
        "",
        f"Sum: {tot8:.0f} min (${tot8 * PRICE:.2f}) with H at 8 draws, {tot4:.0f} min (${tot4 * PRICE:.2f}) at 4; "
        f"of which sampling stages {samp:.0f} min (${samp * PRICE:.2f}) at 8. Add per pod: set-up about 12 min "
        f"(${12 * PRICE:.2f}) and about 2 min of polling gap per checkpoint.",
        "",
    ]
    return L


L_RATE = 1400.0  # his vLLM-LoRA jobs on 1 x H200: Leval_main 1568 tok/s (3,552 draws), Lchk_main 1335 (288)
L_LOAD = 2.0  # engine with enable_lora (his Lchk_main t_load_s 120 s)
L_CHECK_MIN = (
    3.0  # lora_check.py at dpo: an eager engine, 2 x 3 teacher-forced passes of about 5k tokens
)


def table_l(ckpts: list[str]) -> list[str]:
    """Stage L: 48 reach prompts x 8 through the adapter. Tokens per draw: his plain arm-B mean at the checkpoint (lower)
    and the final model's (s2300, upper: the adapter pulls toward the final model's long thinking)."""
    pt = plain_tok()
    L = [
        "**Stage L (48 reach prompts x 8 = 384 draws through the stage-1 adapter, vLLM LoRA path)**",
        "",
        "| ckpt | step | tokens / draw (plain arm B at this ckpt / s2300) | Mtok | H200 min (low / high) | $ (low / high) |",
        "|---|---|---|---|---|---|",
    ]
    lo_t = hi_t = 0.0
    for c in ckpts:
        lo, hi = pt[c], max(pt[c], pt["s2300"])
        extra = L_LOAD + (L_CHECK_MIN if c == "dpo" else 0.0)
        mlo, mhi = extra + 384 * lo / (L_RATE * 60), extra + 384 * hi / (L_RATE * 60)
        lo_t += mlo
        hi_t += mhi
        L.append(
            f"| {c} | {STEP[c]} | {lo:.0f} / {hi:.0f} | {384 * lo / 1e6:.2f}–{384 * hi / 1e6:.2f} | {mlo:.0f} / {mhi:.0f} | "
            f"{mlo * PRICE:.2f} / {mhi * PRICE:.2f} |"
        )
    L += [
        "",
        f"Sum over {', '.join(ckpts)}: {lo_t:.0f}–{hi_t:.0f} min, ${lo_t * PRICE:.2f}–${hi_t * PRICE:.2f}, plus a "
        f"download per checkpoint whose weights are no longer on the pod (about 0.5–3 min each).",
        "",
    ]
    return L


def main() -> None:
    print(
        "\n".join(
            table(MINI, "Mini run (Amendment 1): dpo, s0100, s0500, s1100, s2300")
            + table(SECOND, "Possible second pass: s0050, s0200, s0750, s1600")
            + table_l(["dpo", "s0100", "s0500", "s1100", "s2300"])
        )
    )


if __name__ == "__main__":
    main()
