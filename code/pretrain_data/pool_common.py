"""Shared helpers for the pretrain-data pool builders: record format, leak audit, pool statistics, comparison.

Record format = Dan's K12 rows (`train_k12.jsonl.gz`): prompt, proof, thm, key, n_lines, n_prem, rules, plus a
`pool` tag and whatever the builder adds. Every proof is checked by the pinned verifier before it becomes a record.
The audit removes every record whose curry-aware renaming class matches an evaluation problem (the 412 released
textbook problems) or one of the sealed 72 (counted only, never listed).
"""
from __future__ import annotations

import collections
import csv
import gzip
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "../textbook_problem"), os.path.join(HERE, "../textbook_problem/fork_tools"), os.path.join(HERE, "../ws3")]
import ndpath  # noqa: E402,F401
import eval_analysis as EA  # noqa: E402
import prove  # noqa: E402
from data import canon_key, load_textbook72, thm_of_prompt  # noqa: E402
from hard_problems import bigrams, skeleton_key, n_bicond  # noqa: E402
from patterns import parse, prune_lines  # noqa: E402
from release_1002 import BR  # noqa: E402
from strict_verify import safe_verify  # noqa: E402

L = prove.L
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
DATA = os.path.join(ROOT, "data", "pretrain_data")
CSV = os.path.join(ROOT, "experiments/textbook-problem/eval_dan_best_state/analysis/per_problem.csv")
HARD = os.path.join(ROOT, "experiments/textbook-problem/eval_dan_best_state/analysis/hard_problems.json")
# Dan's K12 Stage-1 set, fetched from his public bucket (13,515,773 bytes; gitignored, never committed):
#   hf cp hf://buckets/dan-pandori/nd-rl/cap-horizon/data/kh/train_k12.jsonl.gz data/pretrain_data/dan/train_k12.jsonl.gz
K12 = os.environ.get("K12_PATH", os.path.join(DATA, "dan", "train_k12.jsonl.gz"))

# the rule joins the hard textbook proofs need and K12 lacks (hard_problems.md § 6.1): (citing rule, cited rule)
JOINS = {
    "→E←∧E": {("IMPE", "ANDE1"), ("IMPE", "ANDE2")},
    "¬E←¬I": {("NEGE", "NEGI")},
    "∧E←∧E": {("ANDE1", "ANDE1"), ("ANDE1", "ANDE2"), ("ANDE2", "ANDE1"), ("ANDE2", "ANDE2")},
    "∨I←¬I": {("ORI1", "NEGI"), ("ORI2", "NEGI")},
    "→E←¬I": {("IMPE", "NEGI")},
    "∧I←→I": {("ANDI", "IMPI")},
    "¬E←→E": {("NEGE", "IMPE")},
    "∨E←∨E": {("ORE", "ORE")},
    "→E←→E": {("IMPE", "IMPE")},
}


def make_record(prompt, proof, pool, **extra):
    """verify with the pinned verifier and build a K12-format record, or None with the reason."""
    ok, reason, n = safe_verify(prompt, proof)
    if not ok:
        return None, reason
    lines = parse(proof)
    prem, _ = L.parse_prompt(prompt)
    rules = collections.Counter(ln["rule"] for ln in lines if ln["rule"] not in ("PR", "AS"))
    thm = thm_of_prompt(prompt)
    rec = dict(prompt=prompt, proof=proof, thm=thm, key=canon_key(thm), n_lines=n, n_prem=len(prem),
               rules=dict(sorted(rules.items())), pool=pool, **extra)
    return rec, None


class EvalIndex:
    """curry-aware keys of every evaluation problem; the sealed 72 are indexed for counting only."""

    def __init__(self):
        refs = EA.load_refs()
        self.eval = {}
        for name, r in refs.items():
            for k in BR.keys(r["prompt"]):
                self.eval.setdefault(k, set()).add(name)
        self.sealed = BR.index([r["prompt"] for r in load_textbook72()])
        self.sets = {}
        if os.path.exists(CSV):
            for row in csv.DictReader(open(CSV)):
                self.sets[row["name"]] = row["set"]

    def audit(self, rows, allowed_sets=()):
        """-> (kept rows, report). Removes twins of the sealed 72 and of every evaluation problem outside
        `allowed_sets` (the seed split's sets, whose twins are kept and tagged `seed_set_twin`)."""
        kept, rep = [], collections.Counter()
        allowed = set(allowed_sets)
        for r in rows:
            ks = BR.keys(r["prompt"])
            hit = set()
            for k in ks:
                hit |= self.eval.get(k, set())
            if any(k in self.sealed for k in ks):
                rep["removed_sealed_twin"] += 1
                continue
            bad = {nm for nm in hit if self.sets.get(nm, "?") not in allowed}
            if bad:
                rep["removed_eval_twin"] += 1
                for nm in bad:
                    rep[f"eval_twin_set:{self.sets.get(nm, '?')}"] += 1
                continue
            if hit:
                rep["kept_seed_set_twin"] += 1
                r = dict(r, seed_set_twin=sorted(hit))
            kept.append(r)
        rep["kept"] = len(kept)
        return kept, dict(rep)


PERMS = None


def rename_text(text, mapping):
    """rename atoms P Q R S in a prompt or proof text; the token after ':' is a rule name (R = reiteration), not an atom"""
    toks = text.split()
    out = []
    for i, t in enumerate(toks):
        if t in mapping and not (i > 0 and toks[i - 1] == ":"):
            out.append(mapping[t])
        else:
            out.append(t)
    return " ".join(out)


def permutations_of(rec, pool_tag=None):
    """all distinct atom renamings of a record (up to 24), verified; the original first"""
    import itertools
    atoms = ["P", "Q", "R", "S"]
    seen, out = set(), []
    for perm in itertools.permutations(atoms):
        m = dict(zip(atoms, perm))
        p2 = rename_text(rec["prompt"], m)
        if p2 in seen:
            continue
        seen.add(p2)
        rec2, why = make_record(p2, rename_text(rec["proof"], m), pool_tag or rec.get("pool"),
                                **{k: v for k, v in rec.items() if k not in ("prompt", "proof", "thm", "key", "n_lines", "n_prem", "rules", "pool")})
        if rec2:
            rec2["perm_of"] = rec["prompt"]
            out.append(rec2)
    return out


def dedup(rows):
    """one record per curry-aware renaming class (first wins)"""
    seen, out = set(), []
    for r in rows:
        ks = BR.keys(r["prompt"])
        if ks & seen:
            continue
        seen |= ks
        out.append(r)
    return out


def write(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with gzip.open(path, "wt") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read(path, limit=0):
    rows = []
    with gzip.open(path, "rt") as f:
        for line in f:
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def load_k12(limit=0):
    rows = read(K12, limit)
    for r in rows:
        r.setdefault("pool", "k12")
    return rows


def hard_pieces():
    """skeletons of the hard textbook proofs' 2–6-line pieces: (all, unknown to K12)"""
    if not os.path.exists(HARD):
        return set(), set()
    d = json.load(open(HARD))
    allk = {p["skel"] for p in d["unknown_pieces"]} | {p["skel"] for p in d["skel_only_top"]} | {p["skel"] for p in d["known_pieces_top"]}
    unk = {p["skel"] for p in d["unknown_pieces"]}
    return allk, unk


def stats(rows, label, pieces=None):
    """pool statistics used by the comparison table"""
    n = len(rows)
    by_len = collections.Counter()
    prem = collections.Counter()
    depth = collections.Counter()
    rule_use = collections.Counter()
    bg = collections.Counter()
    bic = 0
    skels = set()
    cites = 0
    lines_total = 0
    for r in rows:
        lines = parse(r["proof"])
        if lines is None:
            continue
        pr = prune_lines(lines)
        by_len[len(lines)] += 1
        lines_total += len(lines)
        prem[r["n_prem"]] += 1
        depth[max(ln["depth"] for ln in lines)] += 1
        for ln in pr:
            if ln["rule"] not in ("PR", "AS"):
                rule_use[ln["rule"]] += 1
        b = bigrams(pr)
        bg.update(b)
        cites += sum(b.values())
        pm, g = L.parse_prompt(r["prompt"])
        if any(n_bicond_strict(x) for x in pm + [g]):
            bic += 1
        if pieces:
            try:
                skels.add(skeleton_key(pr))
            except Exception:
                pass
    tot_rules = sum(rule_use.values()) or 1
    out = dict(label=label, n=n, mean_lines=round(lines_total / max(1, n), 2),
               len_le6=round(sum(v for k, v in by_len.items() if k <= 6) / max(1, n), 3),
               len_ge10=round(sum(v for k, v in by_len.items() if k >= 10) / max(1, n), 3),
               prem_ge4=round(sum(v for k, v in prem.items() if k >= 4) / max(1, n), 4),
               depth_ge3=round(sum(v for k, v in depth.items() if k >= 3) / max(1, n), 4),
               bicond=round(bic / max(1, n), 4),
               ande_share=round((rule_use["ANDE1"] + rule_use["ANDE2"]) / tot_rules, 3),
               ore_share=round(rule_use["ORE"] / tot_rules, 3), dn_share=round(rule_use["DN"] / tot_rules, 3),
               joins={k: round(1000 * sum(bg.get(x, 0) for x in v) / max(1, cites), 2) for k, v in JOINS.items()},
               by_len=dict(sorted(by_len.items())), by_prem=dict(sorted(prem.items())), by_depth=dict(sorted(depth.items())))
    if pieces:
        allk, unk = pieces
        out["hard_pieces_covered"] = f"{len(allk & skels)}/{len(allk)}"
        out["unknown_pieces_covered"] = f"{len(unk & skels)}/{len(unk)}"
    return out


def n_bicond_strict(f):
    from hard_problems import subformulas
    return sum(1 for s in subformulas(f) if s[0] == "and" and s[1][0] == "imp" and s[2][0] == "imp"
               and s[1][1] == s[2][2] and s[1][2] == s[2][1] and s[1][1] != s[1][2])


def compare_table(stats_list, hard_ref=None):
    """markdown comparison; hard_ref = the hard textbook proofs' join rates per 1k for the last column"""
    W = []
    w = W.append
    w("| pool | n | mean lines | ≤6 | ≥10 | prem ≥4 | depth ≥3 | ↔ rows | ∧E share | ∨E share | DN share | hard pieces covered | unknown pieces covered |")
    w("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for s in stats_list:
        w(f"| {s['label']} | {s['n']:,} | {s['mean_lines']} | {s['len_le6']:.2f} | {s['len_ge10']:.2f} | {s['prem_ge4']:.4f} | {s['depth_ge3']:.4f} | "
          f"{s['bicond']:.4f} | {s['ande_share']:.3f} | {s['ore_share']:.3f} | {s['dn_share']:.3f} | {s.get('hard_pieces_covered', '')} | {s.get('unknown_pieces_covered', '')} |")
    w("")
    w("Join rates per 1,000 citations (citing rule ← cited rule):")
    w("")
    cols = list(JOINS)
    w("| pool | " + " | ".join(cols) + " |")
    w("| --- | " + " | ".join("---" for _ in cols) + " |")
    if hard_ref:
        w("| hard textbook proofs (reference) | " + " | ".join(str(hard_ref.get(c, "")) for c in cols) + " |")
    for s in stats_list:
        w(f"| {s['label']} | " + " | ".join(str(s["joins"][c]) for c in cols) + " |")
    return "\n".join(W)
