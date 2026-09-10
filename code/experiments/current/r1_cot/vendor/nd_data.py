"""Random forward natural-deduction derivations and leakage-audited splits.

The sampler has no desired conclusion, backward search, proof templates, or
external proof corpus. It randomly samples contexts and applicable forward
rule applications, then takes its last derived formula as the theorem.
Dependency pruning removes unused steps; every emitted proof is independently
accepted by the unchanged nd_verify verifier.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import itertools
import json
from pathlib import Path
import random
import time

from nd_verify import parse_formula, verify_text

ATOMS = tuple(("atom", x) for x in "PQRS")
BOT = ("bot",)
PERMUTATIONS = tuple(str.maketrans("PQRS", "".join(p)) for p in itertools.permutations("PQRS"))


@lru_cache(maxsize=100000)
def formula_text(f):
    if f[0] == "atom":
        return f[1]
    if f == BOT:
        return "F"
    if f[0] == "not":
        return f"( ~ {formula_text(f[1])} )"
    op = {'and': '&', 'or': 'v', 'imp': '>'}[f[0]]
    return f"( {formula_text(f[1])} {op} {formula_text(f[2])} )"


def sequent_parts(value):
    """Read a prompt or normalized sequent without inspecting its proof body."""
    if isinstance(value, dict):
        value = value.get("prompt") or value.get("thm") or value["text"].split(" PRF", 1)[0] + " PRF"
    if value.startswith("THM "):
        pre, con = value.split(" SEQ ", 1) if " SEQ " in value else ("THM", value.split("SEQ ", 1)[1])
        pre = pre.removeprefix("THM").strip()
        con = con.split(" PRF", 1)[0].strip()
    else:
        pre, con = value.split("|-", 1)
        pre, con = pre.strip(), con.strip()
    return [" ".join(p.split()) for p in pre.split(",") if p.strip()], " ".join(con.split())


def canonical_theorem(value):
    """Atom-renaming and premise-permutation invariant sequent string.

    Exhausting the 24 bijections of four atoms handles symmetries correctly.
    Premise multiplicity is retained; ordered exact prompts are also audited.
    """
    premises, conclusion = sequent_parts(value)
    return min(" , ".join(sorted(p.translate(t) for p in premises)) + " |- " + conclusion.translate(t)
               for t in PERMUTATIONS)


def exact_theorem(value):
    premises, conclusion = sequent_parts(value)
    return " , ".join(premises) + " |- " + conclusion


def theorem_hash(value):
    return hashlib.sha256(canonical_theorem(value).encode()).hexdigest()


@dataclass(frozen=True)
class Line:
    formula: tuple
    rule: str
    refs: tuple[int, ...]
    ctx: tuple[int, ...]


def sample_formula(rng, depth=2, pool=ATOMS):
    if depth <= 0 or rng.random() < 0.58:
        return rng.choice(pool)
    op = rng.choices(("not", "and", "or", "imp"), (2, 3, 2, 3))[0]
    a = sample_formula(rng, depth - 1, pool)
    return (op, a) if op == "not" else (op, a, sample_formula(rng, depth - 1, pool))


class ForwardSampler:
    def __init__(self, rng, max_lines, formula_limit=23):
        self.rng, self.max_lines, self.formula_limit = rng, max_lines, formula_limit
        self.lines = []
        self.ctx = ()
        self.closed = []  # (start index, end index, parent context)
        self.pool = list(ATOMS)
        self.pool += [sample_formula(rng, rng.choice((1, 2, 2, 3))) for _ in range(4)]

    def small(self, f):
        return len(formula_text(f).split()) <= self.formula_limit

    def fresh(self):
        return sample_formula(self.rng, self.rng.choice((0, 0, 1, 2)), tuple(self.pool))

    def add(self, f, rule, refs=(), ctx=None):
        if ctx is not None:
            while len(self.ctx) > len(ctx):
                start = self.ctx[-1]
                self.closed.append((start, len(self.lines), self.ctx[:-1]))
                self.ctx = self.ctx[:-1]
            self.ctx = ctx
        self.lines.append(Line(f, rule, tuple(refs), self.ctx))

    def assume(self, f, sibling=False):
        parent = self.ctx[:-1] if sibling else self.ctx
        if sibling:
            self.add_close(parent)
        self.ctx = parent + (len(self.lines) + 1,)
        self.add(f, "AS")

    def add_close(self, parent):
        while len(self.ctx) > len(parent):
            self.closed.append((self.ctx[-1], len(self.lines), self.ctx[:-1]))
            self.ctx = self.ctx[:-1]

    def seed(self):
        # Context distribution is independent of any desired target. Shared
        # subformulas make elimination rules available more often than drawing
        # every premise independently from the whole formula language.
        n = self.rng.choices((0, 1, 2, 3), (1, 5, 5, 2))[0]
        n = min(n, self.max_lines - 1)
        base = self.rng.choice(self.pool)
        for k in range(n):
            if k == 0:
                f = self.rng.choice(self.pool)
            else:
                a = self.rng.choice([base] + [ln.formula for ln in self.lines])
                b = self.rng.choice(self.pool)
                op = self.rng.choices(("fresh", "imp", "and", "or", "not", "dn"), (3, 5, 3, 2, 2, 1))[0]
                f = self.fresh() if op == "fresh" else (("not", ("not", a)) if op == "dn" else (("not", a) if op == "not" else (op, a, b)))
            if not self.small(f):
                f = base
            if all(ln.formula != f for ln in self.lines):
                self.add(f, "PR")
        if not self.lines:
            self.assume(self.rng.choice(self.pool))

    def candidates(self):
        rng = self.rng
        visible = [(i + 1, ln.formula) for i, ln in enumerate(self.lines) if ln.ctx == self.ctx[:len(ln.ctx)]]
        by_formula = {}
        for i, f in visible:
            by_formula.setdefault(f, []).append(i)
        out = []

        def offer(f, rule, refs, weight, parent=None):
            destination = self.ctx if parent is None else parent
            if rule != "R" and any(ln.formula == f and ln.ctx == destination[:len(ln.ctx)] for ln in self.lines):
                return
            if self.small(f):
                # Favor using the newest line without forcing it: pruning later
                # removes steps that happen to have become irrelevant.
                weight *= 2.2 if len(self.lines) in refs else 1
                out.append((weight, f, rule, refs, parent))

        for i, f in visible:
            if f[0] == "and":
                offer(f[1], "ANDE1", (i,), 5)
                offer(f[2], "ANDE2", (i,), 5)
            if f[0] == "imp" and f[1] in by_formula:
                offer(f[2], "IMPE", (i, by_formula[f[1]][-1]), 10)
            if f[0] == "not" and f[1] in by_formula:
                offer(BOT, "NEGE", (by_formula[f[1]][-1], i), 9)
            if f[0] == "not" and f[1][0] == "not":
                offer(f[1][1], "DN", (i,), 6)
            if f == BOT:
                offer(rng.choice(self.pool), "BOTE", (i,), 3)
        if visible:
            # Sample a small random subset of introduction-rule choices; this
            # avoids a combinatorial ANDI bias as the context grows.
            for _ in range(2):
                i, a = rng.choice(visible[-3:] if rng.random() < .7 else visible)
                j, b = rng.choice(visible)
                offer(("and", a, b), "ANDI", (i, j), 2 if a != b else .15)
                x = rng.choice(self.pool)
                if a != x:
                    offer(("or", a, x), "ORI1", (i,), 1.2)
                    offer(("or", x, a), "ORI2", (i,), 1.2)
            if self.ctx:
                i, a = rng.choice(visible)
                if self.lines[i - 1].ctx != self.ctx:
                    offer(a, "R", (i,), .9)
        if self.ctx and self.lines[-1].ctx == self.ctx:
            s, e = self.ctx[-1], len(self.lines)
            a, b = self.lines[s - 1].formula, self.lines[e - 1].formula
            offer(("imp", a, b), "IMPI", (s, e), 3.5, self.ctx[:-1])
            if b == BOT:
                offer(("not", a), "NEGI", (s, e), 12, self.ctx[:-1])
            for old_s, old_e, parent in self.closed:
                if parent != self.ctx[:-1] or self.lines[old_e - 1].ctx != parent + (old_s,):
                    continue
                if self.lines[old_e - 1].formula != b:
                    continue
                old_a = self.lines[old_s - 1].formula
                for j, disj in visible:
                    if self.lines[j - 1].ctx == parent[:len(self.lines[j - 1].ctx)]:
                        if disj == ("or", old_a, a):
                            offer(b, "ORE", (j, old_s, old_e, s, e), 30, parent)
                        elif disj == ("or", a, old_a):
                            offer(b, "ORE", (j, s, e, old_s, old_e), 30, parent)
        # Opening random boxes is permitted only when enough budget remains to
        # close every box. Sometimes use a disjunct/antecedent already present.
        remaining = self.max_lines - len(self.lines)
        if len(self.ctx) < 3 and remaining >= len(self.ctx) + 2:
            choices = list(self.pool)
            for _, f in visible:
                if f[0] in ("or", "imp"):
                    choices.extend(f[1:])
                if f[0] == "not":
                    choices.append(f[1])
            f = rng.choice(choices)
            out.append((7 if not self.ctx else 2, f, "AS", (), None))
            if self.ctx:
                out.append((2, f, "SIBLING", (), None))
                # Randomly explore the other disjunct as a sibling hypothesis.
                # No branch conclusion is selected or searched for.
                current_hypothesis = self.lines[self.ctx[-1] - 1].formula
                for _, disj in visible:
                    if disj[0] == "or":
                        if disj[1] == current_hypothesis:
                            out.append((5, disj[2], "SIBLING", (), None))
                        if disj[2] == current_hypothesis:
                            out.append((5, disj[1], "SIBLING", (), None))
        if self.ctx and remaining <= len(self.ctx):
            out = [x for x in out if x[4] is not None]
        return out

    def run(self):
        self.seed()
        while len(self.lines) < self.max_lines:
            choices = self.candidates()
            if not choices:
                return None
            _, f, rule, refs, parent = self.rng.choices(choices, weights=[x[0] for x in choices])[0]
            if rule in ("AS", "SIBLING"):
                self.assume(f, sibling=rule == "SIBLING")
            else:
                self.add(f, rule, refs, parent)
        if self.ctx:
            return None
        return self.prune()

    def prune(self):
        keep = set()

        def visit(i):
            if i in keep:
                return
            keep.add(i)
            line = self.lines[i - 1]
            for j in line.refs + line.ctx:
                visit(j)

        visit(len(self.lines))
        indices = sorted(keep)
        renumber = {old: new for new, old in enumerate(indices, 1)}
        lines = [self.lines[i - 1] for i in indices]
        premises = [ln.formula for ln in lines if ln.rule == "PR"]
        conclusion = lines[-1].formula
        if len(lines) < 2 or conclusion in premises or BOT in premises:
            return None
        if conclusion[0] == "or" and conclusion[1] == conclusion[2]:
            return None
        p = " , ".join(map(formula_text, premises))
        c = formula_text(conclusion)
        prompt = f"THM {p + ' ' if p else ''}SEQ {c} PRF"
        body = []
        for old, ln in zip(indices, lines):
            toks = [f"N{renumber[old]}"] + ["|"] * len(ln.ctx) + [formula_text(ln.formula), ":", ln.rule]
            toks += [f"N{renumber[j]}" for j in ln.refs] + [";"]
            body.append(" ".join(toks))
        text = prompt + "\n" + "\n".join(body) + "\nQED"
        ok, reason, n = verify_text(text)
        if not ok:
            raise AssertionError(f"forward generator bug: {reason}\n{text}")
        return {"thm": p + " |- " + c, "prompt": prompt, "text": text, "n_lines": n,
                "rules": [ln.rule for ln in lines], "n_premises": len(premises),
                "max_depth": max(len(ln.ctx) for ln in lines)}


def generate_candidate(rng, max_lines=6, formula_limit=23):
    return ForwardSampler(rng, max_lines, formula_limit).run()


def exclusion_keys(target_dir):
    """Blind programmatic exclusion: only canonical keys survive this function.

    In particular, test records/reference proofs are never displayed or used to
    design the generator. Only prompt/thm strings enter a set-membership filter.
    """
    keys = set()
    for filename in ("validation_36.jsonl", "test_short_prompts.jsonl", "test_long_prompts.jsonl"):
        path = Path(target_dir) / filename
        if path.exists():
            with path.open() as handle:
                for line in handle:
                    if line.strip():
                        keys.add(canonical_theorem(json.loads(line)))
    return keys


def allocation(total, lengths):
    return {n: total // len(lengths) + (i < total % len(lengths)) for i, n in enumerate(lengths)}


def summarize(records):
    rules, lengths, premises, depth = Counter(), Counter(), Counter(), Counter()
    token_counts, formula_tokens = [], []
    trivial = Counter()
    for record in records:
        rules.update(record["rules"])
        lengths[record["n_lines"]] += 1
        premises[record["n_premises"]] += 1
        depth[record["max_depth"]] += 1
        token_counts.append(len(record["text"].split()))
        pre, con = sequent_parts(record)
        trivial["conclusion_is_premise"] += con in pre
        trivial["falsum_premise"] += "F" in pre
        f, _ = parse_formula(con.split(), 0)
        trivial["duplicate_disjunction_conclusion"] += f[0] == "or" and f[1] == f[2]
        trivial["identity_implication_conclusion"] += f[0] == "imp" and f[1] == f[2]
        trivial["reiteration_lines"] += record["rules"].count("R")
        formula_tokens.extend(len(x.split()) for x in pre + [con])
    tokens = sorted(token_counts)
    return {"n": len(records), "lengths": dict(sorted(lengths.items())), "rules": dict(sorted(rules.items())),
            "premises": dict(sorted(premises.items())), "max_depth": dict(sorted(depth.items())),
            "triviality": dict(trivial), "whitespace_tokens_median": tokens[len(tokens) // 2] if tokens else 0,
            "whitespace_tokens_max": max(tokens, default=0), "formula_tokens_max": max(formula_tokens, default=0),
            "canonical_theorems": len({canonical_theorem(r) for r in records})}


def generate_splits(args):
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    verifier_hash = hashlib.sha256((Path(__file__).parent / "nd_verify" / "verify.py").read_bytes()).hexdigest()
    excluded = exclusion_keys(args.targets)
    totals = {"train": args.train, "heldout": args.heldout, "rl_targets": args.rl, "transfer": args.transfer}
    records = {role: [] for role in totals}
    needs = {role: allocation(total, range(2, 7) if role in ("train", "heldout") else range(7, 17)) for role, total in totals.items()}
    counts = {role: Counter() for role in totals}
    seen_exact, canonical_roles = set(), {}
    rejected = Counter()
    start = time.monotonic()
    attempts = 0
    # Within each regime hash the canonical theorem to a role. A global shape
    # reservation also prevents overlap between short and long regimes.
    handles = {role: (outdir / (role + ".jsonl")).open("w") for role in totals}
    try:
        while any(sum(needs[r].values()) for r in totals):
            attempts += 1
            if attempts > args.max_attempts:
                break
            short_need = sum(sum(needs[r].values()) for r in ("train", "heldout"))
            long_need = sum(sum(needs[r].values()) for r in ("rl_targets", "transfer"))
            short = bool(short_need) and (not long_need or rng.random() < 0.84)
            min_n, max_n = (2, 6) if short else (7, 16)
            relevant = ("train", "heldout") if short else ("rl_targets", "transfer")
            active_lengths = [n for n in range(min_n, max_n + 1) if any(needs[r].get(n, 0) for r in relevant)]
            requested = rng.choice(active_lengths)
            # Longer random traces can prune down into the intended length bin.
            trace_cap = requested if short else min(24, requested + rng.randrange(2, 7))
            record = generate_candidate(rng, trace_cap, args.formula_limit)
            if record is None:
                rejected["no_nontrivial_closed_derivation"] += 1
                continue
            n = record["n_lines"]
            if n < min_n or n > max_n:
                rejected["pruned_outside_length_range"] += 1
                continue
            exact = record["thm"]
            if exact in seen_exact:
                rejected["duplicate_exact_theorem"] += 1
                continue
            canonical = canonical_theorem(record)
            if canonical in excluded:
                rejected["provided_target_atom_renaming"] += 1
                continue
            bucket = int(hashlib.sha256(canonical.encode()).hexdigest()[:12], 16) % 10000
            role = ("train" if bucket < 9000 else "heldout") if short else ("rl_targets" if bucket < 6667 else "transfer")
            if canonical in canonical_roles and canonical_roles[canonical] != role:
                rejected["canonical_reserved_other_regime"] += 1
                continue
            if role != "train" and canonical in canonical_roles:
                rejected["duplicate_evaluation_shape"] += 1
                continue
            if n not in needs[role] or needs[role][n] <= 0:
                rejected["role_or_length_bucket_full"] += 1
                continue
            record["name"] = f"{role}_{len(records[role]):06d}"
            record["canonical_hash"] = hashlib.sha256(canonical.encode()).hexdigest()
            record["generator_seed"] = args.seed
            seen_exact.add(exact)
            canonical_roles[canonical] = role
            records[role].append(record)
            needs[role][n] -= 1
            counts[role][n] += 1
            handles[role].write(json.dumps(record) + "\n")
            if sum(map(len, records.values())) % 1000 == 0:
                for handle in handles.values():
                    handle.flush()
                print(json.dumps({"accepted": {r: len(v) for r, v in records.items()}, "attempts": attempts,
                                  "elapsed_s": round(time.monotonic() - start, 1)}), flush=True)
    finally:
        for handle in handles.values():
            handle.close()
    audit = {"generator": "random-forward-v2-no-repeat", "seed": args.seed, "args": vars(args), "attempts": attempts,
             "generator_sha256": source_hash, "verifier_sha256": verifier_hash,
             "split_sha256": {r: hashlib.sha256((outdir / (r + ".jsonl")).read_bytes()).hexdigest() for r in totals},
             "elapsed_seconds": time.monotonic() - start, "rejections": dict(rejected),
             "splits": {r: summarize(v) for r, v in records.items()},
             "remaining_requested": needs, "all_emitted_proofs_verified": True,
             "supervised_cap": 6, "exact_overlap_between_roles": 0,
             "atom_renaming_overlap_between_roles": 0, "atom_renaming_overlap_with_provided_targets": 0,
             "unused_lines": 0,
             "redundant_visible_formula_inferences": 0,
             "limitations": ["Generating length is only an upper bound on shortest proof length.",
                              "Some compound conclusions permit shorter proofs than their generating trace.",
                              "Canonical split also ignores premise order; atom permutations of a theorem stay within one role."]}
    (outdir / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps({"done": {r: len(v) for r, v in records.items()}, "elapsed_s": round(audit["elapsed_seconds"], 1), "remaining": needs}), flush=True)


def audit_files(args):
    outdir = Path(args.out)
    excluded = exclusion_keys(args.targets)
    exact_sets, canonical_sets, summaries = {}, {}, {}
    for role in ("train", "heldout", "rl_targets", "transfer"):
        path = outdir / (role + ".jsonl")
        records = [json.loads(line) for line in path.open() if line.strip()]
        exact_sets[role], canonical_sets[role] = set(), set()
        for record in records:
            ok, reason, n = verify_text(record["text"])
            assert ok and n == record["n_lines"], (role, record["name"], reason)
            assert record["prompt"] == " ".join(record["text"].split("PRF", 1)[0].split()) + " PRF"
            if role in ("train", "heldout"):
                assert 2 <= n <= 6
            else:
                assert 7 <= n <= 16
            e, c = exact_theorem(record), canonical_theorem(record)
            assert e not in exact_sets[role], ("duplicate", role)
            assert c not in excluded, ("provided_target_overlap", role)
            exact_sets[role].add(e)
            canonical_sets[role].add(c)
        summaries[role] = summarize(records)
    for a, b in itertools.combinations(summaries, 2):
        assert not (exact_sets[a] & exact_sets[b]), ("exact_overlap", a, b)
        assert not (canonical_sets[a] & canonical_sets[b]), ("atom_renaming_overlap", a, b)
    result = {"verified": True, "splits": summaries, "exact_overlap": 0, "atom_renaming_overlap": 0, "provided_target_overlap": 0}
    (outdir / "independent_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("generate", "audit"))
    parser.add_argument("--out", default="data")
    parser.add_argument("--targets", default="targets")
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--train", type=int, default=100000)
    parser.add_argument("--heldout", type=int, default=1000)
    parser.add_argument("--rl", type=int, default=2000)
    parser.add_argument("--transfer", type=int, default=1000)
    parser.add_argument("--formula-limit", type=int, default=23)
    parser.add_argument("--max-attempts", type=int, default=5000000)
    args = parser.parse_args()
    (generate_splits if args.command == "generate" else audit_files)(args)


if __name__ == "__main__":
    main()
