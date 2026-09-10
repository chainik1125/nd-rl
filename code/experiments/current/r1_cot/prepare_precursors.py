"""Guarded cap-six classical precursors from the 14 authorized training tasks.

No textbook development/final file is opened. Template proofs contain at most
six lines; all substitutions come from subformulas of the authorized tasks.
This is a separate synthetic supervised repair, never a held-out evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import re
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path


def neg(a):
    return f"( ~ {a} )"


def binary(a, op, b):
    return f"( {a} {op} {b} )"


def template_records(a, b):
    """Six mechanically checkable precursor families, independent of data."""
    disj, conj = binary(a, "v", b), binary(a, "&", b)
    nda, ndb = neg(a), neg(b)
    nndb = neg(ndb)
    neg_disj = binary(nda, "v", ndb)
    forward = binary(a, ">", nndb)
    bicond = binary(forward, "&", binary(nndb, ">", a))
    result = binary(a, ">", b)
    cases = [
        (
            "neg_disjunction_component",
            [neg(disj)],
            nda,
            [
                (0, neg(disj), "PR", []),
                (1, a, "AS", []),
                (1, disj, "ORI1", [2]),
                (1, "F", "NEGE", [3, 1]),
                (0, nda, "NEGI", [2, 4]),
            ],
        ),
        (
            "neg_neg_disjunction_component",
            [neg(neg_disj)],
            a,
            [
                (0, neg(neg_disj), "PR", []),
                (1, nda, "AS", []),
                (1, neg_disj, "ORI1", [2]),
                (1, "F", "NEGE", [3, 1]),
                (0, neg(nda), "NEGI", [2, 4]),
                (0, a, "DN", [5]),
            ],
        ),
        (
            "neg_conjunction_given_left",
            [neg(conj), a],
            ndb,
            [
                (0, neg(conj), "PR", []),
                (0, a, "PR", []),
                (1, b, "AS", []),
                (1, conj, "ANDI", [2, 3]),
                (1, "F", "NEGE", [4, 1]),
                (0, ndb, "NEGI", [3, 5]),
            ],
        ),
        (
            "imp_double_neg_elim",
            [forward],
            result,
            [
                (0, forward, "PR", []),
                (1, a, "AS", []),
                (1, nndb, "IMPE", [1, 2]),
                (1, b, "DN", [3]),
                (0, result, "IMPI", [2, 4]),
            ],
        ),
        (
            "biconditional_forward_half",
            [bicond],
            result,
            [
                (0, bicond, "PR", []),
                (0, forward, "ANDE1", [1]),
                (1, a, "AS", []),
                (1, nndb, "IMPE", [2, 3]),
                (1, b, "DN", [4]),
                (0, result, "IMPI", [3, 5]),
            ],
        ),
        (
            "negated_antecedent_explosion",
            [nda],
            result,
            [
                (0, nda, "PR", []),
                (1, a, "AS", []),
                (1, "F", "NEGE", [2, 1]),
                (1, b, "BOTE", [3]),
                (0, result, "IMPI", [2, 4]),
            ],
        ),
    ]
    for family, premises, goal, steps in cases:
        body = (
            " ".join(
                " ".join(
                    [f"N{i}"]
                    + ["|"] * depth
                    + [formula, ":", rule]
                    + [f"N{ref}" for ref in refs]
                    + [";"]
                )
                for i, (depth, formula, rule, refs) in enumerate(steps, 1)
            )
            + " QED"
        )
        yield {
            "family": family,
            "prompt": "THM " + " , ".join(premises) + " SEQ " + goal + " PRF",
            "body": body,
            "n_lines": len(steps),
        }


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-subformula-tokens", type=int, default=9)
    args = parser.parse_args()
    root, data, out = args.root.resolve(), args.data.resolve(), args.out.resolve()
    if out.exists():
        raise ValueError("Fresh output directory required")
    sys.path[:0] = [str(root / "transfer_sprint_20260907/code"), str(root)]
    from nd_data import canonical_theorem, formula_text, sequent_parts
    from nd_verify.verify import parse_formula
    from strict_verify import safe_verify

    inputs = {}

    def read(name):
        path = data / name
        inputs[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]

    allowed = read("textbook_train.jsonl")
    if len(allowed) != 14 or any(row["role"] != "textbook_train" for row in allowed):
        raise ValueError("This option is authorized for the frozen 14-class training slice only")
    allowed_keys = {digest(canonical_theorem(row["prompt"])) for row in allowed}
    if len(allowed_keys) != 14:
        raise ValueError("Training textbook classes are not distinct")
    guard_path = data / "heldout_guards.json"
    inputs[str(guard_path)] = hashlib.sha256(guard_path.read_bytes()).hexdigest()
    guards = {key: set(val) for key, val in json.loads(guard_path.read_text()).items()}
    dev = {
        digest(canonical_theorem(row["prompt"]))
        for name in ("synthetic_dev.jsonl", "synthetic_long_dev.jsonl")
        for row in read(name)
    }
    old_short = {row["canonical_hash"] for row in read("sft_cap6.jsonl")}
    formulas = set()

    def collect(node):
        text = formula_text(node)
        if node[0] != "bot" and len(text.split()) <= args.max_subformula_tokens:
            formulas.add(text)
        for child in node[1:]:
            if isinstance(child, tuple):
                collect(child)

    for row in allowed:
        premises, goal = sequent_parts(row["prompt"])
        for text in premises + [goal]:
            node, end = parse_formula(text.split(), 0)
            if end != len(text.split()):
                raise ValueError("Malformed authorized source formula")
            collect(node)
    accepted, counts = {}, Counter()
    for a, b in itertools.product(sorted(formulas), repeat=2):
        for record in template_records(a, b):
            prompt = record["prompt"]
            key = digest(canonical_theorem(prompt))
            premises, goal = sequent_parts(prompt)
            context = digest(canonical_theorem("THM " + " , ".join(premises) + " SEQ F PRF"))
            skeleton = digest(re.sub(r"\b[PQRS]\b", "A", goal)) if not premises else None
            if (
                key in guards["canonical_theorem_sha256"]
                or context in guards["nonempty_premise_context_sha256"]
                or skeleton in guards["closed_formula_skeleton_sha256"]
            ):
                counts["heldout_guard_rejected"] += 1
                continue
            if key in dev:
                counts["synthetic_dev_rejected"] += 1
                continue
            if key in accepted:
                counts["canonical_duplicates"] += 1
                continue
            ok, reason, lines = safe_verify(prompt, record["body"])
            if not ok or lines > 6 or lines != record["n_lines"]:
                raise ValueError(f"Precursor template violates strict cap6: {reason}")
            accepted[key] = {
                **record,
                "name": "precursor_" + key[:20],
                "canonical_hash": key,
                "initial_generator_label_cap": 6,
                "label_origin": "synthetic_cap6_precursor",
                "source_formula_population": "frozen_14_textbook_training_classes_only",
                "matches_permitted_textbook_class": key in allowed_keys,
                "already_in_cap6_pool": key in old_short,
            }
    records = sorted(accepted.values(), key=lambda r: r["canonical_hash"])
    if not records:
        raise ValueError("All precursor classes excluded")
    out.mkdir(parents=True)
    output = out / "precursors_cap6.jsonl"
    output.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in records))
    manifest = {
        "status": "complete",
        "created_utc": datetime.now(UTC).isoformat(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "input_sha256": inputs,
        "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "records": len(records),
        "formula_bank_size": len(formulas),
        "family_counts": dict(sorted(Counter(r["family"] for r in records).items())),
        "length_counts": dict(sorted(Counter(r["n_lines"] for r in records).items())),
        "exact_permitted_textbook_classes": sum(
            r["matches_permitted_textbook_class"] for r in records
        ),
        "already_in_cap6_pool": sum(r["already_in_cap6_pool"] for r in records),
        "filter_counts": dict(counts),
        "permitted_textbook_population_unchanged": 14,
        "textbook_dev_or_final_payload_reads": 0,
        "initial_generator_label_cap": 6,
        "limitations": [
            "Synthetic precursors are supervised repair labels, not model discoveries.",
            "Formula fragments were selected from the permitted training slice.",
            "A targeted versus retained-data continuation control is still needed.",
        ],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                key: value
                for key, value in manifest.items()
                if key not in ("input_sha256", "limitations")
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
