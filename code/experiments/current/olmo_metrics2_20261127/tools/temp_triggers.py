"""M1 trigger evaluation. PRINTS verdicts and the cost of the cells they would release; DECIDES NOTHING, starts nothing.

Triggers (Turn 5 Part 3 section 5, with the Turn 6 fixes):
  A (releases C3 + C4, about $6.3): (a) the paired arm-A pass@1 difference Delta(dpo, C1) has a two-sided 95 % prompt-
      bootstrap interval excluding 0 (either sign, no floor), or (b) the reach rule (below).
  B (releases C2, about $2.9): A did not fire AND, on C1's ARM-A draws (C1a plus the first 2 draws of the 42 shared reach
      prompts, 436 draws), the repetition-flag share (frozen detector, thinking segment) or the share cut at the cap is
      at least 1.5 points above the same quantity on Dan's stored dpo arm-A T 1.0 draws (inputs/temp/stored_rates.json).
Reach rule, k = draws per reach prompt in C1b (8 default, 4 under the cap). Baseline = E0' (M3's fresh dpo T 1.0 reach
cell) iff it is graded AND passes the sidecar checks when C1b is graded, else the proxy; decided by availability only.
  E0' accepted : fires if C1b has >= 3 passes from >= 2 prompts (k = 4: >= 2 and >= 2) AND passes(C1b) - passes(E0' first
                 k draws per prompt) >= 3 (k = 4: >= 2).
  proxy        : fires if C1b has >= 3 passes from >= 2 prompts (k = 4: >= 2 from >= 2).
Usage: temp_triggers.py --c1-graded G --c1-samples S [--c1b-draws 8] [--e0-graded G --e0-sidecar J --e0-preflight J
                       --template-sha SHA [--allow-unverified]] [--stored-rates J] [--out result.json]
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import temp_common as tc
import temp_repeat_flag as rf

COST_A = 6.3
COST_B = 2.9
MARGIN_PTS = 1.5


def arm_a_cell(rows: list[dict], n: int = 2) -> dict[str, list[dict]]:
    """pid -> first n draws, arm-A prompts only."""
    arm = set(tc.dc.arm_a())
    return {p: v[:n] for p, v in tc.by_pid(rows).items() if p in arm}


def reach_cell(rows: list[dict], k: int) -> dict[str, list[dict]]:
    reach = set(tc.dc.reach())
    return {p: v[:k] for p, v in tc.by_pid(rows).items() if p in reach}


def paired_delta(
    new: dict[str, list[dict]], base: dict[str, list[dict]], finishing_only: bool = False
) -> dict:
    """mean_p(N_p - B_p) over prompts present in both; prompt bootstrap (10,000, seed 0), two-sided 95 %."""
    pids = sorted(set(new) & set(base))
    if finishing_only:
        pids = [
            p
            for p in pids
            if any(not tc.is_cut(r) for r in new[p]) and any(not tc.is_cut(r) for r in base[p])
        ]
    f = (lambda v: [r for r in v if not tc.is_cut(r)]) if finishing_only else (lambda v: v)
    d = np.array(
        [
            np.mean([tc.strict_of(r) for r in f(new[p])])
            - np.mean([tc.strict_of(r) for r in f(base[p])])
            for p in pids
        ]
    )
    idx = tc.boot_idx(len(d))
    boots = d[idx].mean(axis=1)
    lo, hi = tc.ci(boots)
    return {
        "n_prompts": len(pids),
        "delta_pts": 100 * float(d.mean()),
        "lo95_pts": 100 * lo,
        "hi95_pts": 100 * hi,
        "excludes_zero": bool(lo > 0 or hi < 0),
    }


def reach_counts(cell: dict[str, list[dict]]) -> dict:
    passes = sum(tc.strict_of(r) for v in cell.values() for r in v)
    prompts = sum(any(tc.strict_of(r) for r in v) for v in cell.values())
    return {
        "draws": sum(len(v) for v in cell.values()),
        "passes": passes,
        "prompts_with_pass": prompts,
        "n_prompts": len(cell),
    }


def e0_accept(
    side: dict | None,
    pre: dict | None,
    template_sha: str | None,
    allow_unverified: bool = False,
    grade_rows: int | None = None,
    n_expected: int = 384,
) -> tuple[bool, list[str], list[str]]:
    """E0' (M3's patched engine, zero vector) is the reach baseline only if everything checkable passes.
    Returns (accepted, failures, unverified)."""
    fail: list[str] = []
    unv: list[str] = []
    if side is None:
        return False, ["no E0' sidecar"], unv
    if side.get("status") != "done":
        fail.append(f"sidecar status {side.get('status')!r} != 'done'")
    if side.get("temperature") != 1.0 or side.get("top_p") != 1.0:
        fail.append(f"E0' T/top_p = {side.get('temperature')}/{side.get('top_p')}, need 1.0/1.0")
    argv = side.get("dan_argv") or []
    if "--lora" in argv:
        fail.append("E0' ran with --lora")
    mt = None
    if "--max_tokens" in argv:
        mt = int(argv[argv.index("--max_tokens") + 1])
    if mt not in (None, 32768):
        fail.append(f"E0' max_tokens {mt} != 32768")
    if pre is None:
        fail.append("no preflight result (zero-vector identity not shown)")
    else:
        checks = pre.get("checks") or {}
        if checks.get("zero_identity") is not True:
            fail.append("preflight checks.zero_identity is not True")
        if checks.get("zero_compiled_within_tol") is not True:
            fail.append(
                "preflight checks.zero_compiled_within_tol is not True (E0' samples with the compiled engine)"
            )
        if pre.get("pass_chosen") is not True and pre.get("pass") is not True:
            fail.append("preflight neither pass_chosen nor pass is True")
        for key in ("engine_config_sha256", "dtype", "enforce_eager"):
            if key not in pre or key not in (side or {}):
                unv.append(f"{key}: not recorded on both sides (same engine config not verifiable)")
            elif pre[key] != side[key]:
                fail.append(f"{key} differs between preflight and E0' run")
    for src, name in ((side, "sidecar"), (pre or {}, "preflight")):
        if template_sha and src.get("template_sha256") not in (None, template_sha):
            fail.append(f"{name} template sha differs")
    if not any(s.get("template_sha256") for s in (side, pre or {})):
        unv.append("template_sha256: not recorded")
    if grade_rows is not None and grade_rows < n_expected:
        fail.append(f"E0' graded rows {grade_rows} < {n_expected}")
    if unv and not allow_unverified:
        fail.append("unverified items and --allow-unverified not given: " + "; ".join(unv))
    return (not fail), fail, unv


def reach_rule(c1b: dict[str, list[dict]], k: int, e0: dict[str, list[dict]] | None) -> dict:
    n_pass, n_prompt = (3, 2) if k >= 8 else (2, 2)
    diff_need = 3 if k >= 8 else 2
    a = reach_counts(c1b)
    res = {
        "k": k,
        "c1b": a,
        "baseline": "E0'" if e0 is not None else "proxy",
        "need_passes": n_pass,
        "need_prompts": n_prompt,
    }
    cond1 = a["passes"] >= n_pass and a["prompts_with_pass"] >= n_prompt
    if e0 is not None:
        b = reach_counts({p: v[:k] for p, v in e0.items()})
        res["e0"] = b
        res["need_difference"] = diff_need
        cond2 = a["passes"] - b["passes"] >= diff_need
        res["fires"] = bool(cond1 and cond2)
    else:
        res["fires"] = bool(cond1)
    return res


def trigger_b(c1_arm_a_rows: list[dict], texts: dict[tuple, str], stored: dict) -> dict:
    n = k = cut = 0
    for r in c1_arm_a_rows:
        n += 1
        cut += tc.is_cut(r)
        k += rf.flag_text(texts.get((r["pid"], int(r["chunk"]), int(r["i"])), ""))
    rep, cu = 100 * k / max(n, 1), 100 * cut / max(n, 1)
    s = stored["dpo"]
    return {
        "draws": n,
        "repeat_share_pct": rep,
        "cut_share_pct": cu,
        "stored_repeat_share_pct": s["share_pct"],
        "stored_cut_share_pct": s["cut_pct"],
        "margin_pts": MARGIN_PTS,
        "repeat_over_margin": bool(rep >= s["share_pct"] + MARGIN_PTS),
        "cut_over_margin": bool(cu >= s["cut_pct"] + MARGIN_PTS),
        "condition": bool(rep >= s["share_pct"] + MARGIN_PTS or cu >= s["cut_pct"] + MARGIN_PTS),
    }


def evaluate(
    c1_rows: list[dict],
    c1_texts: dict,
    k: int,
    stored: dict,
    e0_rows: list[dict] | None,
    e0_ok: bool,
    e0_info: dict,
) -> dict:
    a_cell = arm_a_cell(c1_rows, 2)
    dlt = paired_delta(a_cell, tc.stored_arm_a("dpo"))
    dlt_fin = paired_delta(a_cell, tc.stored_arm_a("dpo"), finishing_only=True)
    c1b = reach_cell(c1_rows, k)
    e0cell = reach_cell(e0_rows, 8) if (e0_rows is not None and e0_ok) else None
    rr = reach_rule(c1b, k, e0cell)
    a_fire = bool(dlt["excludes_zero"] or rr["fires"])
    flat = [r for v in a_cell.values() for r in v]
    tb = trigger_b(flat, c1_texts, stored)
    out = {
        "arm_a_delta": dlt,
        "arm_a_delta_finishing_only": dlt_fin,
        "reach_rule": rr,
        "e0_acceptance": e0_info,
        "trigger_A": {
            "fires": a_fire,
            "via_arm_a": dlt["excludes_zero"],
            "via_reach": rr["fires"],
            "sign": ("positive" if dlt["delta_pts"] > 0 else "negative"),
            "releases": "C3 (s2300 card, arm A x2) + C4 (dpo pure T 0.6, arm A x2)",
            "cost_usd": COST_A,
            "note": "if it fires: stop and ask the orchestrator for C3 + C4; if refused run C3 alone (about $3.4), "
            "headline 'card decoding', no power-sampling proposal",
        },
        "trigger_B": {
            "fires": bool((not a_fire) and tb["condition"]),
            "detail": tb,
            "releases": "C2 (dpo T 0.8 / top_p 1.0, arm A x2)",
            "cost_usd": COST_B,
        },
        "decides": "nothing; the orchestrator decides",
    }
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--c1-graded", required=True)
    ap.add_argument(
        "--c1-samples", required=True, help="C1 samples.jsonl.gz (texts for the repetition flag)"
    )
    ap.add_argument("--c1b-draws", type=int, default=8)
    ap.add_argument("--e0-graded")
    ap.add_argument("--e0-sidecar")
    ap.add_argument("--e0-preflight")
    ap.add_argument("--template-sha")
    ap.add_argument("--allow-unverified", action="store_true")
    ap.add_argument(
        "--stored-rates",
        default="/scratch/cowifornia/nd-rl/olmo_elicit/metrics2/inputs/temp/stored_rates.json",
    )
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    c1 = tc.jl(a.c1_graded)
    texts = {
        (r["pid"], int(r["chunk"]), int(r["i"])): r.get("text", "") for r in tc.jl(a.c1_samples)
    }
    stored = json.load(open(a.stored_rates))
    e0_rows = tc.jl(a.e0_graded) if a.e0_graded else None
    side = json.load(open(a.e0_sidecar)) if a.e0_sidecar else None
    pre = json.load(open(a.e0_preflight)) if a.e0_preflight else None
    if e0_rows is None:
        ok, fail, unv = False, ["E0' not graded / not given"], []
    else:
        ok, fail, unv = e0_accept(
            side, pre, a.template_sha, a.allow_unverified, grade_rows=len(e0_rows)
        )
    info = {
        "accepted": ok,
        "failures": fail,
        "unverified": unv,
        "baseline_used": "E0'"
        if ok
        else "proxy (0 of 84 dpo T 1.0 Stage 0c; 2 of 1,536 at s0100 = 0.13 %)",
    }
    res = evaluate(c1, texts, a.c1b_draws, stored, e0_rows, ok, info)
    res["inputs"] = {"c1_graded": a.c1_graded, "c1_samples": a.c1_samples, "detector": rf.VERSION}
    txt = json.dumps(res, indent=1)
    print(txt)
    if a.out:
        open(a.out, "w").write(txt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
