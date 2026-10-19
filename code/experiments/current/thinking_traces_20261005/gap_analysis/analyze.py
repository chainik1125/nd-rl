"""Gap analysis tables: textbook dev58 (and train14, reported apart) vs the hard12 pool, the hard12
test split and K12, joined with the RL'd models' per-attempt results (CPU only, no model).

    cd /scratch/cowifornia/nd-rl/think/nd-rl
    ND_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 code/.venv/bin/python \
        code/experiments/current/thinking_traces_20261005/gap_analysis/analyze.py

Reads runs/gap_analysis/features.jsonl (features.py) and the eval rows under runs/eval. Writes
runs/gap_analysis/tables.md (every table of the README) and results.json (the numbers).

Per-problem score = mean over the four best RL'd checkpoints of the per-attempt pass@1 (n_ok / k,
k = 64 on textbook72, 16 on hard12_test_1k); pass@5 by the unbiased estimator, averaged the same
way.
"Failed" textbook problem: score < 25% (fixed before looking at features); hard12 test "failed":
score < 50% (only 6 test problems are below 25%).

The textbook rows carry only aggregate features here; no textbook formula or proof is printed.
"""

from __future__ import annotations

import collections
import json
import math
import sys
from pathlib import Path
from typing import Any

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402

EV = C.ROOT / "runs/eval"
BEST = {
    "T1v1g4e_it78 (GRPO, think, B256)": (
        "matched/tb72_T1v1g4e_it78/rows_plain_B256.jsonl",
        "matched/h12_T1v1g4e_it78/rows_plain_B256.jsonl",
    ),
    "grpoT0_it78 (GRPO, no think)": (
        "matched/tb72_grpoT0_it78/rows_plain_B0.jsonl",
        "matched/h12_grpoT0_it78/rows_plain_B0.jsonl",
    ),
    "eiT1v1e_r2 (EI, think, B256)": (
        "matched/tb72_eiT1v1e_r2/rows_plain_B256.jsonl",
        "matched/h12_eiT1v1e_r2/rows_plain_B256.jsonl",
    ),
    "eiT0_r2 (EI, no think)": (
        "matched/tb72_eiT0_r2/rows_plain_B0.jsonl",
        "matched/h12_eiT0_r2/rows_plain_B0.jsonl",
    ),
}
STAGE1 = {
    "T0_s0 (Stage 1, no think)": "tb72_T0_s0/rows_plain_B0.jsonl",
    "T1v1_s0 (Stage 1, think B256)": "tb72_T1v1_s0/rows_plain_B256.jsonl",
}
FAIL_TB = 0.25
FAIL_H12 = 0.50

# status buckets for failed attempts (state-env rejection reasons, see nd-takehome/state_env.py)
BUCKET = {
    "lean_reject": "finished, Lean rejects (wrong inference)",
    "type": "type/shape (Or.elim on non-or, box on non-arrow, binder or goal mismatch)",
    "scope": "scope (unbound name, exact not last)",
    "syntax": "syntax/format (formula, assign, term, semicolon, box head, premise position ...)",
    "cap": "caps (step cap, context, truncated)",
}
TYPE = {"Or.elim on non-or", "box on non-arrow", "binder mismatch", "exact goal mismatch"}
SCOPE = {"unbound", "exact not last"}


def bucket(status: str, ok: int) -> str:
    if ok:
        return "ok"
    if status == "done":
        return "lean_reject"
    if status in ("fail:step_cap", "fail:context", "fail:truncated"):
        return "cap"
    why = status.split("fail:S:", 1)[-1]
    if why in TYPE:
        return "type"
    if why in SCOPE:
        return "scope"
    return "syntax"


def pass_at(c: int, k: int, j: int) -> float:
    if k - c < j:
        return 1.0
    return 1.0 - math.comb(k - c, j) / math.comb(k, j)


def load_eval() -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    tb: dict[str, dict[str, Any]] = {}
    h12: dict[str, dict[str, Any]] = {}
    for m, (ptb, ph) in BEST.items():
        for path, dst in ((ptb, tb), (ph, h12)):
            for r in C.jl(EV / path):
                d = dst.setdefault(r["name"], {"models": {}, "attempts": []})
                d["models"][m] = (r["n_ok"], r["k"])
                for st, ok, fs, stp in zip(
                    r["status"], r["ok"], r["fail_step"], r["steps"], strict=True
                ):
                    d["attempts"].append((m, st, ok, fs, stp))
    for m, p in STAGE1.items():
        for r in C.jl(EV / p):
            tb.setdefault(r["name"], {"models": {}, "attempts": []})["models"][m] = (
                r["n_ok"],
                r["k"],
            )
    return tb, h12


def score(d: dict[str, Any], models: list[str], j: int = 1) -> float:
    xs = [pass_at(*d["models"][m], j) for m in models if m in d["models"]]
    return sum(xs) / len(xs)


# ---------------------------------------------------------------- stats helpers
def fisher_p(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher exact p for [[a, b], [c, d]]."""
    n1, n2, m1, n = a + b, c + d, a + c, a + b + c + d

    def pr(x: int) -> float:
        return math.comb(n1, x) * math.comb(n2, m1 - x) / math.comb(n, m1)

    p0 = pr(a)
    lo, hi = max(0, m1 - n2), min(n1, m1)
    return min(1.0, sum(pr(x) for x in range(lo, hi + 1) if pr(x) <= p0 * (1 + 1e-9)))


def ranks(xs: list[float]) -> list[float]:
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for t in range(i, j + 1):
            r[order[t]] = (i + j) / 2 + 1
        i = j + 1
    return r


def spearman(x: list[float], y: list[float]) -> float:
    rx, ry = ranks(x), ranks(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry, strict=True))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den else float("nan")


def pct(k: int, n: int) -> str:
    return "–" if n == 0 else f"{100 * k / n:.0f}%"


# ---------------------------------------------------------------- feature catalogue
THM_BIN = [
    ("no_prem", "no premises (a theorem)"),
    ("goal_atom", "goal is an atom"),
    ("goal_not", "goal is a negation"),
    ("goal_neg_compound", "goal is a negated compound ~(A∘B)"),
    ("goal_or", "goal is a disjunction"),
    ("goal_and", "goal is a conjunction"),
    ("goal_imp", "goal is an implication"),
    ("goal_imp_compound_ante", "goal is A>B with compound A"),
    ("goal_imp_imp_ante", "goal is (A>B)>C (higher-order)"),
    ("has_higher_order_imp", "some (A>B)>C anywhere"),
    ("has_neg_compound", "some negated compound ~(A&B) / ~(AvB) / ~(A>B)"),
    ("has_dneg", "some double negation ~~A"),
    ("has_bicond", "some biconditional (A>B)&(B>A)"),
    ("classical_only", "classical-only (not intuitionistically valid, G4ip)"),
    ("prem_inconsistent", "premises jointly inconsistent"),
    ("has_bot", "F appears in the theorem"),
]
THM_NUM = [
    ("n_prem", "premises"),
    ("n_atoms", "distinct atoms"),
    ("fdepth", "formula depth (max)"),
    ("goal_depth", "goal depth"),
    ("n_conn", "connectives in the theorem"),
    ("n_neg", "~ count"),
    ("n_and", "& count"),
    ("n_or", "v count"),
    ("n_imp", "> count"),
    ("imp_nest", "nested-implication depth"),
    ("prompt_tokens", "theorem tokens"),
    ("n_redundant_prem", "redundant premises"),
]
PROOF_BIN = [
    ("min_lines>=9", "shortest proof >= 9 lines"),
    ("compact_long", "<= 7 connectives and shortest proof >= 9 lines"),
    ("neg_work", "reductio, or a ¬-intro box inside another box"),
    ("box_free", "no box at all"),
    ("p_box_depth>=2", "box depth >= 2"),
    ("p_box_depth>=3", "box depth >= 3"),
    ("p_n_boxes>=3", ">= 3 assumptions (AS lines)"),
    ("p_has_ORE", "case split (ORE)"),
    ("p_has_raa", "reductio (assume ~G, F, NEGI, DN)"),
    ("p_has_negi_plain", "plain ¬-intro (NEGI box, no DN)"),
    ("p_has_IMPI", "→-intro box"),
    ("p_has_BOTE", "ex falso"),
    ("p_negbox_nested", "a negation/reductio box nested inside another box"),
    ("p_negbox_in_negbox", "negation box inside a negation box"),
    ("p_negbox_in_impi", "negation box inside a →-intro box"),
    ("p_ore_in_negbox", "case split inside a negation/reductio box"),
    ("p_negbox_in_ore", "negation box inside a case branch"),
    ("p_box_in_ore", "any box inside a case branch"),
    ("p_impi_in_impi", "→-intro box inside a →-intro box"),
    ("p_ore_on_hyp", "case split on an assumption (not a premise)"),
    ("p_ore_goal_bot", "case split concluding F"),
    ("p_ore_goal_compound", "case split concluding a compound"),
    ("p_ore_crossed_ori", "case branches end in Or.inl vs Or.inr"),
    ("p_ore_both_branches_work", "both case branches >= 2 lines"),
    ("p_nege_pos_intro", "contradiction built by an intro (X built to clash with ~X)"),
    ("p_nege_neg_hyp", "contradiction uses an assumed negation"),
    ("p_negi_lemma", "a ¬-intro result reused by IMPE/NEGE (negation lemma)"),
    ("p_impi_lemma", "a →-intro result applied later (implication lemma)"),
    ("p_box_lemma_elim", "any box result reused by an elimination"),
    ("p_box_lemma_intro", "box results combined by ANDI/ORI"),
    ("p_impe_on_hyp", "IMPE on an assumed implication"),
    ("p_first_forward", "first step is forward (not an assumption)"),
]
PROOF_NUM = [
    ("min_lines", "shortest proof found (lines)"),
    ("p_box_depth", "box depth"),
    ("p_n_boxes", "assumptions (AS lines)"),
    ("p_elim_depth2plus", "elimination lines at depth >= 2"),
    ("p_lines_depth2plus", "lines at depth >= 2"),
    ("p_max_box_lines", "longest box (lines)"),
    ("min_actions", "state-env actions of that proof"),
]


def fval(r: dict[str, Any], key: str) -> float | None:
    if key == "box_free":
        return None if "p_n_boxes" not in r else int(r["p_n_boxes"] == 0)
    if key == "neg_work":
        return None if "p_n_boxes" not in r else int(bool(r["p_negbox_nested"] or r["p_has_raa"]))
    if key == "compact_long":
        return None if r.get("min_lines") is None else int(r["n_conn"] <= 7 and r["min_lines"] >= 9)
    if ">=" in key:
        k, v = key.split(">=")
        return None if r.get(k) is None else int(r[k] >= int(v))
    return r.get(key)


def gval(r: dict[str, Any], key: str) -> float | None:
    """The same feature on the generator's own proof (g_ prefix)."""
    if key == "min_lines":
        return r.get("ref_lines")
    if key == "min_actions":
        return r.get("gen_actions")
    if key == "box_free":
        return None if "g_n_boxes" not in r else int(r["g_n_boxes"] == 0)
    if key == "neg_work":
        return None if "g_n_boxes" not in r else int(bool(r["g_negbox_nested"] or r["g_has_raa"]))
    if key == "compact_long":
        return None if r.get("ref_lines") is None else int(r["n_conn"] <= 7 and r["ref_lines"] >= 9)
    if ">=" in key:
        k, v = key.split(">=")
        k = "ref_lines" if k == "min_lines" else "g_" + k[2:]
        return None if r.get(k) is None else int(r[k] >= int(v))
    return r.get("g_" + key[2:]) if key.startswith("p_") else r.get(key)


# ---------------------------------------------------------------- main
def main() -> None:
    rows = C.jl(C.OUT / "features.jsonl")
    tb_eval, h12_eval = load_eval()
    best = list(BEST)
    for r in rows:
        if r["set"] in ("dev58", "train14"):
            d = tb_eval[r["name"]]
            r["score"] = score(d, best)
            r["score5"] = score(d, best, 5)
            r["score_s1"] = score(d, list(STAGE1))
        elif r["set"] == "h12test":
            d = h12_eval[r["name"]]
            r["score"] = score(d, best)
            r["score5"] = score(d, best, 5)
    S = collections.defaultdict(list)
    for r in rows:
        S[r["set"]].append(r)
    dev = S["dev58"]
    dev712 = [r for r in dev if 7 <= r["ref_lines"] <= 12]
    G = {
        "TB fail": [r for r in dev712 if r["score"] < FAIL_TB],
        "TB solved": [r for r in dev712 if r["score"] >= FAIL_TB],
        "TB 7–12": dev712,
        "TB all": dev,
        "pool": S["pool"],
        "pool min 7–12": [r for r in S["pool"] if 7 <= r["min_lines"] <= 12],
        "h12 test fail": [r for r in S["h12test"] if r["score"] < FAIL_H12],
        "K12": S["k12"],
        "K12 gen 7–12": [r for r in S["k12"] if r.get("min_lines") is not None],
        "K12 min 7–12": [
            r for r in S["k12"] if r.get("min_lines") is not None and 7 <= r["min_lines"] <= 12
        ],
    }
    small = {"TB fail", "TB solved", "TB 7–12", "TB all", "h12 test fail"}
    textbook = {"TB fail", "TB solved", "TB 7–12", "TB all"}
    res: dict[str, Any] = {"groups": {g: len(v) for g, v in G.items()}}
    md: list[str] = []

    def prevalence(
        cat: list[tuple[str, str]], numeric: bool, proof: bool, title: str, use_gen: bool = False
    ) -> None:
        cols = list(G)
        md.append(f"\n### {title}\n")
        hdr = "| feature | " + " | ".join(f"{c} (n={len(G[c])})" for c in cols) + " |"
        md.append(hdr)
        md.append("|---" * (len(cols) + 1) + "|")
        out = {}
        for key, label in cat:
            cells = []
            out[key] = {}
            for c in cols:
                getter = gval if (use_gen and c not in textbook) else fval
                if (
                    proof and not use_gen and c == "K12"
                ):  # shortest proofs exist only for K12's 7-12 part
                    cells.append("–")
                    continue
                vals = [getter(r, key) for r in G[c]]
                vals = [v for v in vals if v is not None]
                if not vals:
                    cells.append("–")
                    continue
                if numeric:
                    m = sum(vals) / len(vals)
                    cells.append(f"{m:.2f}" if m < 10 else f"{m:.1f}")
                    out[key][c] = m
                else:
                    k = int(sum(vals))
                    cells.append(
                        f"{pct(k, len(vals))} ({k}/{len(vals)})"
                        if c in small
                        else pct(k, len(vals))
                    )
                    out[key][c] = k / len(vals)
                if len(vals) != len(G[c]) and c in small:
                    cells[-1] += "*"
            md.append(f"| {label} | " + " | ".join(cells) + " |")
        res.setdefault("prevalence", {})[title] = out

    md.append("## Groups\n")
    md.append(
        "TB = textbook dev58 (grouped by its reference length, as in the eval). "
        "Score = mean pass@1 of the four best RL'd "
        f"checkpoints. TB fail = 7–12 lines and score < {FAIL_TB:.0%}; "
        f"h12 test fail = score < {FAIL_H12:.0%}. "
        "pool / K12 'gen 7–12' are grouped by the generator proof's length, "
        "'min 7–12' by the shortest proof found. "
        "K12 = a random 20,000 of train_k12 "
        "(all lengths; shortest proofs only for its 7–12-line part). "
        "`*` = some rows lack the feature (no proof found: textbook problems over 16 lines).\n"
    )
    prevalence(THM_BIN, False, False, "Theorem shape (binary, % of group)")
    prevalence(THM_NUM, True, False, "Theorem shape (numeric, group mean)")
    prevalence(PROOF_BIN, False, True, "Shortest-proof structure (binary, % of group)")
    prevalence(PROOF_NUM, True, True, "Shortest-proof structure (numeric, group mean)")
    prevalence(
        PROOF_BIN,
        False,
        True,
        "Generator's own proof structure for generator columns (textbook columns: shortest proof)",
        use_gen=True,
    )

    # ---- length gap
    md.append("\n### Shortest-proof length vs nominal length\n")
    md.append(
        "| set | n | shortest = nominal | shortest <= 6 lines | shortest 10–12 "
        "| mean nominal | mean shortest |"
    )
    md.append("|---|---|---|---|---|---|---|")
    lg = {}
    for name, rs in (
        ("TB 7–12 (reference length)", dev712),
        ("pool (generator length 7–12)", S["pool"]),
        ("h12 test", S["h12test"]),
        ("K12 gen 7–12", G["K12 gen 7–12"]),
    ):
        rs = [r for r in rs if r.get("min_lines") is not None]
        n = len(rs)
        eq = sum(r["min_lines"] == r["ref_lines"] for r in rs)
        le6 = sum(r["min_lines"] <= 6 for r in rs)
        t = sum(10 <= r["min_lines"] <= 12 for r in rs)
        mn = sum(r["ref_lines"] for r in rs) / n
        ms = sum(r["min_lines"] for r in rs) / n
        lg[name] = dict(n=n, eq=eq / n, le6=le6 / n, ten12=t / n, nominal=mn, shortest=ms)
        md.append(
            f"| {name} | {n} | {pct(eq, n)} | {pct(le6, n)} | {pct(t, n)} | {mn:.1f} | {ms:.1f} |"
        )
    res["length_gap"] = lg

    # ---- skeletons
    md.append("\n### Box skeletons of the shortest proofs\n")
    md.append(
        "I(..) →-intro box, N(..) ¬-intro box, C(..) reductio box, "
        "O(a|b) case split with branch contents a, b; '-' = no box. "
        "Share of each group with that skeleton; "
        "the TB skeletons are listed, plus the pool's most common ones.\n"
    )
    sk = {
        c: collections.Counter(r.get("p_skeleton") for r in G[c] if r.get("p_skeleton")) for c in G
    }
    tb_sk = [s for s, _ in (sk["TB 7–12"]).most_common()]
    pool_top = [s for s, _ in sk["pool"].most_common(6)]
    cols = [
        "TB fail",
        "TB solved",
        "pool",
        "pool min 7–12",
        "h12 test fail",
        "K12 gen 7–12",
        "K12 min 7–12",
    ]
    md.append("| skeleton | " + " | ".join(f"{c} (n={sum(sk[c].values())})" for c in cols) + " |")
    md.append("|---" * (len(cols) + 1) + "|")
    skt = {}
    for s in tb_sk + [x for x in pool_top if x not in tb_sk]:
        cells = []
        for c in cols:
            n = sum(sk[c].values())
            k = sk[c][s]
            cells.append(
                f"{pct(k, n)} ({k}/{n})" if c in small else (f"{100 * k / n:.1f}%" if n else "–")
            )
        skt[s] = {c: sk[c][s] / max(1, sum(sk[c].values())) for c in cols}
        md.append(f"| `{s}` | " + " | ".join(cells) + " |")
    res["skeletons"] = skt
    # how rare is a failed TB problem's skeleton in the pool
    rare = []
    for r in G["TB fail"]:
        s = r.get("p_skeleton")
        rare.append(sk["pool"][s] / len(S["pool"]))
    res["tb_fail_skeleton_pool_share"] = rare
    sv = []
    for r in G["TB solved"]:
        sv.append(sk["pool"][r.get("p_skeleton")] / len(S["pool"]))
    res["tb_solved_skeleton_pool_share"] = sv
    rare_s = ", ".join(f"{100 * x:.1f}%" for x in sorted(rare))
    sv_s = ", ".join(f"{100 * x:.1f}%" for x in sorted(sv))
    md.append(
        f"\nPool share of each failed TB problem's skeleton: {rare_s}. "
        f"Solved TB problems: {sv_s}.\n"
    )

    # ---- within-dev predictors
    md.append("\n### Within dev58: which features go with failure\n")
    md.append(
        "Left block: dev58 7–12 (n = 20), mean score with / without the feature, "
        "and the failed share with / without "
        "(Fisher exact p, two-sided). "
        "Right block: all dev58 problems with a shortest proof (<= 16 lines), score residual "
        "after removing a per-length-bin mean (bins <=6, 7–9, 10–12, 13–16), "
        "with minus without.\n"
    )
    md.append(
        "| feature | n with (7–12) | score with | score without | failed with "
        "| failed without | p | n with (<=16) | Δ residual (pp) |"
    )
    md.append("|---|---|---|---|---|---|---|---|---|")
    dev16 = [r for r in dev if r.get("p_n_boxes") is not None]

    def lbin(L: int) -> int:
        return 0 if L <= 6 else 1 if L <= 9 else 2 if L <= 12 else 3

    bm = collections.defaultdict(list)
    for r in dev16:
        bm[lbin(r["ref_lines"])].append(r["score"])
    for r in dev16:
        r["resid"] = r["score"] - sum(bm[lbin(r["ref_lines"])]) / len(bm[lbin(r["ref_lines"])])
    pred = {}
    for key, _label in THM_BIN + PROOF_BIN:
        w = [r for r in dev712 if fval(r, key)]
        wo = [r for r in dev712 if fval(r, key) is not None and not fval(r, key)]
        if not w or not wo:
            continue
        fw = sum(r["score"] < FAIL_TB for r in w)
        fwo = sum(r["score"] < FAIL_TB for r in wo)
        p = fisher_p(fw, len(w) - fw, fwo, len(wo) - fwo)
        w16 = [r["resid"] for r in dev16 if fval(r, key)]
        wo16 = [r["resid"] for r in dev16 if fval(r, key) is not None and not fval(r, key)]
        dres = (sum(w16) / len(w16) - sum(wo16) / len(wo16)) if w16 and wo16 else float("nan")
        sw = sum(r["score"] for r in w) / len(w)
        swo = sum(r["score"] for r in wo) / len(wo)
        pred[key] = dict(
            n_with=len(w),
            score_with=sw,
            score_without=swo,
            fail_with=fw / len(w),
            fail_without=fwo / len(wo),
            p=p,
            n16=len(w16),
            dresid=dres,
        )
    for key, label in THM_BIN + PROOF_BIN:
        if key not in pred:
            continue
        x = pred[key]
        md.append(
            f"| {label} | {x['n_with']} | {100 * x['score_with']:.0f}% "
            f"| {100 * x['score_without']:.0f}% | "
            f"{100 * x['fail_with']:.0f}% | {100 * x['fail_without']:.0f}% | {x['p']:.2f} "
            f"| {x['n16']} | {100 * x['dresid']:+.0f} |"
        )
    res["within_dev"] = pred
    md.append("\nSpearman correlation of numeric features with the score:\n")
    md.append(
        f"| feature | dev58 7–12 (n=20) | dev58 <= 16 lines (n={len(dev16)}) "
        "| dev58 <= 16, length residual |"
    )
    md.append("|---|---|---|---|")
    sp = {}
    for key, label in THM_NUM + PROOF_NUM:
        a = [(fval(r, key), r["score"]) for r in dev712 if fval(r, key) is not None]
        b = [(fval(r, key), r["score"], r["resid"]) for r in dev16 if fval(r, key) is not None]
        if len(a) < 5:
            continue
        r1 = spearman([x for x, _ in a], [y for _, y in a])
        r2 = spearman([x for x, _, _ in b], [y for _, y, _ in b])
        r3 = spearman([x for x, _, _ in b], [z for _, _, z in b])
        sp[key] = (r1, r2, r3)
        md.append(f"| {label} | {r1:+.2f} | {r2:+.2f} | {r3:+.2f} |")
    res["spearman"] = sp

    # ---- model scores per group
    md.append("\n### Scores by group (pass@1 / pass@5, % of group)\n")
    md.append("| group | n | best-4 mean pass@1 | best-4 mean pass@5 | Stage 1 mean pass@1 |")
    md.append("|---|---|---|---|---|")
    sc = {}
    for name, rs in (
        ("TB 7–12", dev712),
        ("TB fail", G["TB fail"]),
        ("TB solved", G["TB solved"]),
        ("TB all", dev),
        ("train14 (not used)", S["train14"]),
        ("h12 test", S["h12test"]),
        ("h12 test fail", G["h12 test fail"]),
    ):
        n = len(rs)
        s1 = sum(r["score"] for r in rs) / n
        s5 = sum(r["score5"] for r in rs) / n
        st = [r["score_s1"] for r in rs if "score_s1" in r]
        sc[name] = dict(n=n, pass1=s1, pass5=s5, stage1=(sum(st) / len(st)) if st else None)
        s1_cell = f"{100 * sum(st) / len(st):.1f}%" if st else "–"
        md.append(f"| {name} | {n} | {100 * s1:.1f}% | {100 * s5:.1f}% | {s1_cell} |")
    res["scores"] = sc

    # ---- statuses
    md.append("\n### How attempts end (four best checkpoints pooled; % of attempts)\n")
    stat_groups = {
        "TB fail": [tb_eval[r["name"]] for r in G["TB fail"]],
        "TB solved": [tb_eval[r["name"]] for r in G["TB solved"]],
        "TB all": [tb_eval[r["name"]] for r in dev],
        "h12 test": [h12_eval[r["name"]] for r in S["h12test"]],
        "h12 test fail": [h12_eval[r["name"]] for r in G["h12 test fail"]],
    }
    keys = ["ok", "lean_reject", "type", "scope", "syntax", "cap"]
    md.append("| group | attempts | " + " | ".join(["ok"] + [BUCKET[k] for k in keys[1:]]) + " |")
    md.append("|---" * (len(keys) + 2) + "|")
    stt = {}
    detail = {}
    for g, ds in stat_groups.items():
        c = collections.Counter()
        dc = collections.Counter()
        for d in ds:
            for _, st, ok, _, _ in d["attempts"]:
                c[bucket(st, ok)] += 1
                if not ok:
                    dc[st] += 1
        n = sum(c.values())
        stt[g] = {k: c[k] / n for k in keys}
        detail[g] = {k: v / max(1, sum(dc.values())) for k, v in dc.most_common(8)}
        md.append(f"| {g} | {n} | " + " | ".join(pct(c[k], n) for k in keys) + " |")
    res["status"] = stt
    res["status_detail_of_failures"] = detail
    md.append("\nMost common failure statuses (% of failed attempts):\n")
    for g, dd in detail.items():
        md.append(f"- **{g}:** " + ", ".join(f"`{k}` {100 * v:.0f}%" for k, v in dd.items()))
    # where failures happen: accepted actions before the failing one vs the shortest proof's actions
    md.append(
        "\nWhere the failing attempt stopped, as accepted actions / "
        "actions of the shortest proof (state-env actions):\n"
    )
    md.append(
        "| group | failed attempts | stopped in first 25% | 25–75% | past 75% "
        "| ran past the shortest proof's length |"
    )
    md.append("|---|---|---|---|---|---|")
    pos = {}
    for g, rs, ev in (
        ("TB fail", G["TB fail"], tb_eval),
        ("TB solved", G["TB solved"], tb_eval),
        ("h12 test", S["h12test"], h12_eval),
    ):
        c = collections.Counter()
        for r in rs:
            na = r.get("min_actions")
            if not na:
                continue
            for _, _st, ok, fs, stp in ev[r["name"]]["attempts"]:
                if ok:
                    continue
                x = (fs if fs is not None else stp) / na
                c["early" if x < 0.25 else "mid" if x < 0.75 else "late"] += 1
                c["past"] += int(x > 1.0)
        n = c["early"] + c["mid"] + c["late"]
        pos[g] = {k: c[k] / n for k in ("early", "mid", "late", "past")} if n else {}
        md.append(
            f"| {g} | {n} | {pct(c['early'], n)} | {pct(c['mid'], n)} "
            f"| {pct(c['late'], n)} | {pct(c['past'], n)} |"
        )
    res["fail_position"] = pos
    # successful attempts: length vs shortest
    md.append("\nSuccessful attempts, actions used / actions of the shortest proof found:\n")
    md.append(
        "| group | successful attempts | mean ratio | share at the shortest length "
        "| share >= 1.5x |"
    )
    md.append("|---|---|---|---|---|")
    sl = {}
    for g, rs, ev in (("TB 7–12", dev712, tb_eval), ("h12 test", S["h12test"], h12_eval)):
        xs = []
        for r in rs:
            na = r.get("min_actions")
            if not na:
                continue
            for _, _st, ok, _fs, stp in ev[r["name"]]["attempts"]:
                if ok:
                    xs.append(stp / na)
        n = len(xs)
        if n:
            sl[g] = dict(
                n=n,
                mean=sum(xs) / n,
                eq=sum(abs(x - 1) < 1e-9 for x in xs) / n,
                ge15=sum(x >= 1.5 for x in xs) / n,
            )
            md.append(
                f"| {g} | {n} | {sum(xs) / n:.2f} "
                f"| {pct(sum(abs(x - 1) < 1e-9 for x in xs), n)} "
                f"| {pct(sum(x >= 1.5 for x in xs), n)} |"
            )
    res["success_length"] = sl

    # ---- per-model scores (context for the headline gap)
    md.append("\n### Per-checkpoint pass@1 / pass@5 (% of set)\n")
    md.append(
        "| checkpoint | TB 7–12 (n=20) pass@1 | pass@5 | TB fail (n=8) pass@1 "
        "| h12 test (n=1000) pass@1 | pass@5 |"
    )
    md.append("|---|---|---|---|---|---|")
    pm = {}
    for m in best:
        a1 = sum(pass_at(*tb_eval[r["name"]]["models"][m], 1) for r in dev712) / len(dev712)
        a5 = sum(pass_at(*tb_eval[r["name"]]["models"][m], 5) for r in dev712) / len(dev712)
        f1 = sum(pass_at(*tb_eval[r["name"]]["models"][m], 1) for r in G["TB fail"]) / len(
            G["TB fail"]
        )
        h1 = sum(pass_at(*h12_eval[r["name"]]["models"][m], 1) for r in S["h12test"]) / len(
            S["h12test"]
        )
        h5 = sum(pass_at(*h12_eval[r["name"]]["models"][m], 5) for r in S["h12test"]) / len(
            S["h12test"]
        )
        pm[m] = dict(tb712_p1=a1, tb712_p5=a5, tbfail_p1=f1, h12_p1=h1, h12_p5=h5)
        md.append(
            f"| {m} | {100 * a1:.1f}% | {100 * a5:.1f}% | {100 * f1:.1f}% "
            f"| {100 * h1:.1f}% | {100 * h5:.1f}% |"
        )
    for m in STAGE1:
        a1 = sum(pass_at(*tb_eval[r["name"]]["models"][m], 1) for r in dev712) / len(dev712)
        a5 = sum(pass_at(*tb_eval[r["name"]]["models"][m], 5) for r in dev712) / len(dev712)
        f1 = sum(pass_at(*tb_eval[r["name"]]["models"][m], 1) for r in G["TB fail"]) / len(
            G["TB fail"]
        )
        pm[m] = dict(tb712_p1=a1, tb712_p5=a5, tbfail_p1=f1)
        md.append(f"| {m} | {100 * a1:.1f}% | {100 * a5:.1f}% | {100 * f1:.1f}% | – | – |")
    res["per_model"] = pm

    # ---- within the generator: do the textbook-failure structures hurt there too?
    md.append(
        "\n### Within the hard12 test split: score with / without each structure (shortest proof)\n"
    )
    md.append(
        "| feature | n with | best-4 pass@1 with | without "
        "| n with, at shortest 7–12 | with | without |"
    )
    md.append("|---|---|---|---|---|---|---|")
    h = S["h12test"]
    h712 = [r for r in h if 7 <= r["min_lines"] <= 12]
    wg = {}
    for key, label in THM_BIN + PROOF_BIN:
        w = [r["score"] for r in h if fval(r, key)]
        wo = [r["score"] for r in h if fval(r, key) is not None and not fval(r, key)]
        w2 = [r["score"] for r in h712 if fval(r, key)]
        wo2 = [r["score"] for r in h712 if fval(r, key) is not None and not fval(r, key)]
        if len(w) < 3 or not wo:
            continue
        f = lambda xs: f"{100 * sum(xs) / len(xs):.0f}%" if xs else "–"  # noqa: E731
        wg[key] = dict(n=len(w), w=sum(w) / len(w), wo=sum(wo) / len(wo), n712=len(w2))
        md.append(f"| {label} | {len(w)} | {f(w)} | {f(wo)} | {len(w2)} | {f(w2)} | {f(wo2)} |")
    res["within_h12"] = wg

    # ---- generator structures that the shortest proof keeps (necessity in the pool)
    md.append("\n### Pool: how often a generator-proof structure survives in the shortest proof\n")
    md.append("| structure | generator proofs with it | of those, shortest proof keeps it |")
    md.append("|---|---|---|")
    keep = {}
    for key, label in (
        ("p_has_ORE", "case split"),
        ("p_has_raa", "reductio"),
        ("p_has_negi_plain", "plain ¬-intro"),
        ("p_has_IMPI", "→-intro box"),
        ("p_has_BOTE", "ex falso"),
        ("p_box_depth>=2", "box depth >= 2"),
        ("p_negbox_nested", "nested negation box"),
        ("p_box_in_ore", "box inside a case branch"),
        ("p_ore_goal_compound", "case split concluding a compound"),
    ):
        g = [r for r in S["pool"] if gval(r, key)]
        k = sum(1 for r in g if fval(r, key))
        keep[key] = dict(n=len(g), share=len(g) / len(S["pool"]), kept=k / max(1, len(g)))
        md.append(f"| {label} | {pct(len(g), len(S['pool']))} ({len(g)}) | {pct(k, len(g))} |")
    res["pool_kept"] = keep

    # ---- compact theorem, long proof
    md.append("\n### Compact theorem, long proof\n")
    md.append(
        "Share of each group whose theorem has <= 7 connectives and whose shortest proof "
        "has >= 9 lines; and shortest-proof lines per theorem connective (mean).\n"
    )
    md.append("| group | n | <= 7 connectives and shortest >= 9 lines | lines per connective |")
    md.append("|---|---|---|---|")
    cl = {}
    for name in ("TB 7–12", "TB fail", "pool", "pool min 7–12", "K12 gen 7–12", "K12 min 7–12"):
        rs = [r for r in G[name] if r.get("min_lines") is not None]
        k = sum(r["n_conn"] <= 7 and r["min_lines"] >= 9 for r in rs)
        lpc = sum(r["min_lines"] / max(1, r["n_conn"]) for r in rs) / len(rs)
        cl[name] = dict(n=len(rs), share=k / len(rs), lpc=lpc)
        md.append(f"| {name} | {len(rs)} | {pct(k, len(rs))} | {lpc:.2f} |")
    res["compact"] = cl

    # ---- necessity of the structures (necessity.py)
    nf = C.OUT / "necessity.jsonl"
    if nf.exists():
        nec = C.jl(nf)
        sc_by = {r["name"]: r["score"] for r in dev712}
        md.append("\n### Is the structure necessary? Restricted exact search\n")
        md.append(
            "For each problem whose shortest proof uses the structure, "
            "the shortest proof *without* it (search from L to L + 3 "
            "lines, 120 s). 'needed' = no proof without it within L + 3 lines; "
            "'same length' = an equally short proof without it exists.\n"
        )
        md.append(
            "| structure forbidden | group | n | needed (none within L+3) "
            "| +1 to +3 lines | same length | timeout |"
        )
        md.append("|---|---|---|---|---|---|---|")
        names = {
            "nested_neg": "negation box inside a box",
            "no_dn": "DN (reductio)",
            "no_ore": "case split",
            "flat": "box depth >= 2",
        }
        nt = {}
        for kind in ("nested_neg", "no_dn", "no_ore", "flat"):
            for grp, sel in (
                (
                    "TB fail",
                    lambda x: x["set"] == "dev58_712" and sc_by.get(x["name"], 1) < FAIL_TB,
                ),
                (
                    "TB solved",
                    lambda x: x["set"] == "dev58_712" and sc_by.get(x["name"], 0) >= FAIL_TB,
                ),
                ("pool (random <= 300)", lambda x: x["set"] == "pool"),
            ):
                xs = [x for x in nec if x["kind"] == kind and sel(x)]
                if not xs:
                    continue
                n = len(xs)
                to = sum(bool(x.get("timeout")) for x in xs)
                need = sum(x.get("len") is None and not x.get("timeout") for x in xs)
                same = sum(x.get("len") == x["L"] for x in xs)
                longer = sum(x.get("len") is not None and x["len"] > x["L"] for x in xs)
                bad = sum(bool(x.get("violates")) for x in xs if x.get("len") is not None)
                nt[f"{kind}|{grp}"] = dict(
                    n=n,
                    needed=need / n,
                    longer=longer / n,
                    same=same / n,
                    timeout=to / n,
                    violations=bad,
                )

                def cell(k: int, n: int = n, grp: str = grp) -> str:
                    return f"{pct(k, n)} ({k}/{n})" if grp != "pool (random <= 300)" else pct(k, n)

                md.append(
                    f"| {names[kind]} | {grp} | {n} | {cell(need)} | {cell(longer)} "
                    f"| {cell(same)} | {cell(to)} |"
                )
        res["necessity"] = nt

    # ---- generator config mix of the pool: shortest-length by config
    md.append("\n### Pool by generator config: shortest-proof length\n")
    md.append(
        "| config | n | shortest <= 6 | shortest 10–12 | nested negation box | reductio "
        "| case split (shortest proof) |"
    )
    md.append("|---|---|---|---|---|---|---|")
    bycfg = collections.defaultdict(list)
    for r in S["pool"]:
        bycfg[r.get("cfg")].append(r)
    cf = {}
    for c, rs in sorted(bycfg.items(), key=lambda x: -len(x[1])):
        n = len(rs)
        a6 = sum(r["min_lines"] <= 6 for r in rs)
        a10 = sum(r["min_lines"] >= 10 for r in rs)
        nn = sum(r["p_negbox_nested"] for r in rs)
        ra = sum(r["p_has_raa"] for r in rs)
        orx = sum(r["p_has_ORE"] for r in rs)
        cf[c] = dict(n=n, le6=a6 / n, ge10=a10 / n)
        md.append(
            f"| {c} | {n} | {pct(a6, n)} | {pct(a10, n)} | {pct(nn, n)} "
            f"| {pct(ra, n)} | {pct(orx, n)} |"
        )
    res["pool_cfg"] = cf
    # ---- train14, separately
    tr = S["train14"]
    md.append("\n### train14 (reported separately; not used for any conclusion)\n")
    md.append("| feature | train14 |")
    md.append("|---|---|")
    for key, label in THM_BIN + PROOF_BIN:
        vals = [fval(r, key) for r in tr]
        vals = [v for v in vals if v is not None]
        if vals:
            md.append(
                f"| {label} | {pct(int(sum(vals)), len(vals))} ({int(sum(vals))}/{len(vals)}) |"
            )
    lens = collections.Counter(r.get("min_lines") for r in tr)
    md.append(
        "\ntrain14 shortest-proof lengths: "
        f"{dict(sorted((k if k is not None else 99, v) for k, v in lens.items()))} "
        "(99 = none found)."
    )
    sc14 = sum(r["score"] for r in tr) / len(tr)
    md.append(f"train14 best-4 mean pass@1: {100 * sc14:.1f}%.\n")
    (C.OUT / "tables.md").write_text("\n".join(md) + "\n")
    (C.OUT / "results.json").write_text(json.dumps(res, indent=1, default=str))
    print("\n".join(md))


if __name__ == "__main__":
    main()
