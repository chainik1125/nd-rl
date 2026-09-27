"""Metrics beyond pass@k (plan section 5). Pure functions of per-problem counts or dump rows.

The headline metric is the **avg solve rate**: for each problem, the fraction p = c / n of its
n samples that verify (the unbiased pass@1 estimate), averaged over problems. Around it:

    reliable_frac      fraction of problems with p >= 0.5 (and >= 0.9)
    frontier           longest length L such that avg >= 0.5 at every evaluated length <= L
    k_at               smallest k with mean pass@k >= 0.5 (samples needed; from the same draws)
    compare_problems   base vs RL per problem: new capability, lost, sharpened (from dumps)
    auroc / top1       does the model's own log-probability separate right from wrong attempts
"""

from __future__ import annotations

import collections
import gzip
import json
import math


def mean_ci(ps, z: float = 1.96):
    """(mean, [lo, hi]) of per-problem rates. Normal interval over problems, never narrower than
    the Wilson interval for the same mean on len(ps) problems (so all-zero or all-one sets still
    get an honest width)."""
    from eval.evaluator import wilson  # here, not at module top: evaluator imports this module

    n = len(ps)
    if n == 0:
        return 0.0, [0.0, 0.0]
    m = sum(ps) / n
    sd = math.sqrt(sum((p - m) ** 2 for p in ps) / (n - 1)) if n > 1 else 0.0
    lo, hi = m - z * sd / math.sqrt(n), m + z * sd / math.sqrt(n)
    wlo, whi = wilson(round(m * n), n, z)
    return m, [max(0.0, min(lo, wlo)), min(1.0, max(hi, whi))]


def reliable_frac(ps, threshold: float) -> float:
    return sum(p >= threshold for p in ps) / len(ps) if ps else 0.0


def frontier(by_length: dict, key: str = "avg", threshold: float = 0.5) -> int:
    """Longest length L with by_length[l][key] >= threshold for every evaluated l <= L; 0 if the
    shortest evaluated length already misses. Unknown-length buckets (L < 0) are ignored."""
    best = 0
    for L in sorted(int(x) for x in by_length if int(x) >= 0):
        v = by_length.get(L, by_length.get(str(L)))
        if v[key] >= threshold:
            best = L
        else:
            break
    return best


def k_at(curve: dict, threshold: float = 0.5):
    """Smallest k in a pass@k curve with value >= threshold, or None if the curve never gets there."""
    for k in sorted(int(x) for x in curve):
        if curve.get(k, curve.get(str(k))) >= threshold:
            return k
    return None


def auroc(scores, labels):
    """Probability that a random positive outscores a random negative (ties count half)."""
    pos = [s for s, y in zip(scores, labels, strict=True) if y]
    neg = [s for s, y in zip(scores, labels, strict=True) if not y]
    if not pos or not neg:
        return None
    ranked = sorted([(s, 1) for s in pos] + [(s, 0) for s in neg])
    rank_sum, i = 0.0, 0
    while i < len(ranked):
        j = i
        while j < len(ranked) and ranked[j][0] == ranked[i][0]:
            j += 1
        avg_rank = (i + j + 1) / 2  # 1-based average rank of the tie block
        rank_sum += avg_rank * sum(y for _, y in ranked[i:j])
        i = j
    return (rank_sum - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def self_knowledge(logprobs, judged):
    """AUROC of sequence log-probability (total and per token) for verified vs not, and the
    success rate of picking each problem's most likely sample ("likelihood top-1")."""
    tot, per_tok, labels, top1 = [], [], [], []
    for lps, js in zip(logprobs, judged, strict=True):
        if not js:
            continue
        for lp, j in zip(lps, js, strict=True):
            tot.append(lp)
            per_tok.append(lp / max(j[4], 1))
            labels.append(bool(j[1]))
        best = max(range(len(js)), key=lambda i: lps[i])
        top1.append(bool(js[best][1]))
    return {
        "auroc_logprob": auroc(tot, labels),
        "auroc_logprob_per_token": auroc(per_tok, labels),
        "likelihood_top1": sum(top1) / max(len(top1), 1),
        "n_verified": sum(labels),
        "n_samples": len(labels),
    }


# ------------------------------------------------------------------ per-problem, from dumps


def read_dump(path, label=None):
    op = gzip.open if str(path).endswith(".gz") else open
    with op(path, "rt") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return [r for r in rows if label is None or r.get("label") == label]


def per_problem(rows) -> dict:
    """prompt -> {'n', 'c', 'p', 'length', 'name'} from judged dump rows."""
    acc = collections.OrderedDict()
    for r in rows:
        a = acc.setdefault(r["prompt"], {"n": 0, "c": 0, "length": r.get("length"),
                                          "name": r.get("name")})
        a["n"] += 1
        a["c"] += bool(r["ok"])
    for a in acc.values():
        a["p"] = a["c"] / a["n"]
    return acc


def compare_problems(base: dict, rl: dict, reliable: float = 0.5) -> dict:
    """Per-problem base vs RL on the prompts both evaluated.

    new_capability  base never solved (0 of n) and RL solves reliably (p >= `reliable`)
    new_any         base never solved and RL solved at least once
    lost            base solved reliably and RL never solves
    sharpened       both solved at least once and RL p > base p
    A large new_capability count is the emergence claim; gains confined to `sharpened` mean RL
    made proofs the model could already find more likely.
    """
    common = [p for p in base if p in rl]
    by_len = collections.defaultdict(lambda: collections.Counter())
    tot = collections.Counter()
    scatter = []
    for prompt in common:
        b, e = base[prompt], rl[prompt]
        L = b.get("length")
        c = collections.Counter()
        c["n"] += 1
        c["new_capability"] += b["c"] == 0 and e["p"] >= reliable
        c["new_any"] += b["c"] == 0 and e["c"] > 0
        c["lost"] += b["p"] >= reliable and e["c"] == 0
        c["sharpened"] += b["c"] > 0 and e["c"] > 0 and e["p"] > b["p"]
        c["worse"] += e["p"] < b["p"]
        tot.update(c)
        by_len[L].update(c)
        scatter.append({"name": b.get("name"), "length": L, "base_p": b["p"], "rl_p": e["p"],
                        "base_n": b["n"], "rl_n": e["n"]})
    mean_delta = sum(s["rl_p"] - s["base_p"] for s in scatter) / max(len(scatter), 1)
    return {
        "reliable_threshold": reliable,
        "overall": dict(tot),
        "by_length": {L: dict(v) for L, v in sorted(by_len.items(), key=lambda x: (x[0] is None, x[0]))},
        "mean_delta_p": mean_delta,
        "scatter": scatter,
    }
