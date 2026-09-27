"""Equivalent variants of an evaluation problem, for the consistency metric (plan section 5).

A capability should not depend on which letters name the atoms or on the order the premises are
listed. `make_variants` returns up to m distinct prompts that rename the atoms (a permutation of
P, Q, R, S), reorder the premises, or both. Every variant has the same minimal proof length.

Evaluation only. Variants of evaluation problems are never trained on, like the originals
(`CLAUDE.md`); `variant_records` also drops any variant whose renaming class occurs in a training
pool, so a variant can never be a problem the model was trained on. Do not build variants of
`rl_targets` or `train` prompts: they are training prompts already.
"""

from __future__ import annotations

import itertools
import random

import ndpath  # noqa: F401

# isort: split
from data import ATOMS, canon_key, key_of, length_of, load_pool, thm_of_prompt


def split_prompt(prompt: str):
    """(premise token lists, conclusion tokens) from 'THM p1 , p2 SEQ c PRF'."""
    t = prompt.split()
    i, j = t.index("SEQ"), t.index("PRF")
    prem, cur = [], []
    for tok in t[1:i]:
        if tok == ",":
            prem.append(cur)
            cur = []
        else:
            cur.append(tok)
    if cur:
        prem.append(cur)
    return prem, t[i + 1 : j]


def join_prompt(premises, conclusion) -> str:
    body = " , ".join(" ".join(p) for p in premises)
    return f"THM {body} SEQ {' '.join(conclusion)} PRF" if premises else (
        f"THM SEQ {' '.join(conclusion)} PRF")


def rename(tokens, mapping):
    return [mapping.get(x, x) for x in tokens]


def make_variants(prompt: str, m: int, seed: int = 0):
    """Up to m distinct (prompt, kind) variants, never the original. `kind` is 'rename',
    'reorder', or 'rename+reorder'. Deterministic in (prompt, m, seed)."""
    rng = random.Random(f"{seed}:{prompt}")
    prem, concl = split_prompt(prompt)
    perms = [dict(zip(ATOMS, p, strict=True)) for p in itertools.permutations(ATOMS)]
    perms = [p for p in perms if any(k != v for k, v in p.items())]
    orders = [list(o) for o in itertools.permutations(range(len(prem)))] if len(prem) < 7 else []
    orders = [o for o in orders if o != sorted(o)]
    cands = []
    for mp in perms:
        cands.append(("rename", mp, None))
    for o in orders:
        cands.append(("reorder", None, o))
        cands.append(("rename+reorder", rng.choice(perms), o))
    rng.shuffle(cands)
    out, seen = [], {prompt}
    for kind, mp, order in cands:
        ps = [prem[i] for i in order] if order else list(prem)
        c = list(concl)
        if mp:
            ps, c = [rename(p, mp) for p in ps], rename(c, mp)
        v = join_prompt(ps, c)
        if v not in seen:
            seen.add(v)
            out.append((v, kind))
        if len(out) >= m:
            break
    return out


def training_keys() -> set:
    """Renaming classes of every prompt RL may train on (train.jsonl and rl_targets.jsonl)."""
    return {key_of(r) for name in ("train", "rl_targets") for r in load_pool(name)}


def variant_records(records, m: int, seed: int = 0, exclude_keys=None):
    """Originals (variant 0) plus up to m variants each, as evaluation records.

    Each record keeps the original's length and slice fields and gains `variant_of` (the
    original's name or prompt) and `variant` (0 for the original, else the kind). Variants whose
    renaming class is in `exclude_keys` are dropped and counted in the returned `dropped`.
    """
    exclude_keys = exclude_keys if exclude_keys is not None else set()
    out, dropped = [], 0
    for r in records:
        base = {k: v for k, v in r.items() if k not in ("key", "thm", "text", "proof")}
        orig = r.get("name") or r["prompt"]
        out.append({**base, "variant_of": orig, "variant": "original"})
        for v, kind in make_variants(r["prompt"], m, seed):
            if canon_key(thm_of_prompt(v)) in exclude_keys:
                dropped += 1
                continue
            out.append({**base, "prompt": v, "name": f"{orig}~{len(out)}", "variant_of": orig,
                        "variant": kind})
    return out, dropped


def consistency(records, judged, reliable: float = 0.5) -> dict:
    """Per original problem, success rates over its variants (original included).

    consistent_reliable  every variant (and the original) has p >= `reliable`
    min_p / mean_p       worst and average variant success rate
    original_gap         original p minus the mean p of its other variants; a large positive
                         gap suggests the model learned that exact wording, not the problem
    """
    from eval.metrics import mean_ci

    groups = {}
    for r, js in zip(records, judged, strict=True):
        p = sum(bool(j[1]) for j in js) / max(len(js), 1)
        g = groups.setdefault(r["variant_of"], {"length": length_of(r, default=-1), "ps": [], "orig": None})
        g["ps"].append(p)
        if r["variant"] == "original":
            g["orig"] = p
    per, by_len = [], {}
    for name, g in groups.items():
        others = g["ps"][1:] if g["orig"] is not None else g["ps"]
        row = {
            "name": name,
            "length": g["length"],
            "n_variants": len(g["ps"]),
            "mean_p": sum(g["ps"]) / len(g["ps"]),
            "min_p": min(g["ps"]),
            "consistent_reliable": all(p >= reliable for p in g["ps"]),
            "original_gap": (g["orig"] - sum(others) / len(others)) if others and g["orig"] is not None else None,
        }
        per.append(row)
        by_len.setdefault(g["length"], []).append(row)

    def summ(rows):
        m, ci = mean_ci([r["mean_p"] for r in rows])
        gaps = [r["original_gap"] for r in rows if r["original_gap"] is not None]
        return {
            "n_problems": len(rows),
            "mean_p": m,
            "mean_p_ci": ci,
            "min_p_mean": sum(r["min_p"] for r in rows) / len(rows),
            "consistent_reliable": sum(r["consistent_reliable"] for r in rows),
            "original_gap_mean": sum(gaps) / len(gaps) if gaps else None,
        }

    return {
        "reliable_threshold": reliable,
        "overall": summ(per) if per else {},
        "by_length": {L: summ(v) for L, v in sorted(by_len.items(), key=lambda x: (x[0] is None, x[0]))},
        "per_problem": per,
    }
