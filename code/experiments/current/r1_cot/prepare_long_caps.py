"""Build explicitly NEW cap8/cap12 synthetic-label arms; CPU and no textbook reads.

Only the two explicitly authorized old RL-target reference files are eligible.
Their former analysis-only status is preserved in lineage, while this new arm
uses the user's explicit permission for longer initial supervised proof labels.
Nothing this script writes belongs in a cap-six-only training claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

SOURCES = (
    "transfer_sprint_20260907/data/structural_long_v1/references/rl_targets.analysis_only.jsonl",
    "transfer_sprint_20260907/data/negation_long_v1/references/rl_targets.analysis_only.jsonl",
)


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(path):
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_rows(path, records):
    with path.open("w") as stream:
        for record in records:
            stream.write(json.dumps(record, sort_keys=True) + "\n")


def split_record(row):
    if "prompt" in row:
        prompt = " ".join(row["prompt"].split())
    else:
        prompt = " ".join(row["text"].split("PRF", 1)[0].split()) + " PRF"
    body = row.get("reference_proof", row.get("body", row.get("proof")))
    if body is None and "text" in row:
        body = row["text"].split("PRF", 1)[1]
    return prompt, " ".join(body.split()) if isinstance(body, str) else None


def cohort(key, seed):
    """A global deterministic split keeps supervised and RL classes disjoint."""
    return "rl" if int(digest(f"cap-extension-split:{seed}:{key}")[:16], 16) % 5 == 0 else "sft"


def describe(records):
    return {
        "records": len(records),
        "canonical_classes": len({r["canonical_hash"] for r in records}),
        "length_counts": dict(sorted(Counter(r["n_lines"] for r in records).items())),
        "family_counts": dict(
            sorted(Counter(r.get("family", "cap6_retained") for r in records).items())
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Prior take-home checkout")
    parser.add_argument("--data", type=Path, required=True, help="Current sprint data directory")
    parser.add_argument("--out", type=Path, required=True, help="Fresh cap_extensions output")
    parser.add_argument("--seed", type=int, default=2026090912)
    args = parser.parse_args()
    root, data, out = args.root.resolve(), args.data.resolve(), args.out.resolve()
    if out.exists():
        raise ValueError("Refuse to replace existing generated data")
    sys.path[:0] = [str(root / "transfer_sprint_20260907/code"), str(root)]
    from nd_data import canonical_theorem, sequent_parts
    from strict_verify import safe_verify

    inputs = {}

    def load(path):
        inputs[str(path)] = sha(path)
        return read_rows(path)

    guard_path = data / "all_textbook_guards.json"
    inputs[str(guard_path)] = sha(guard_path)
    guards = {key: set(values) for key, values in json.loads(guard_path.read_text()).items()}

    def keys(prompt):
        premises, goal = sequent_parts(prompt)
        return (
            digest(canonical_theorem(prompt)),
            digest(canonical_theorem("THM " + " , ".join(premises) + " SEQ F PRF"))
            if premises
            else None,
            digest(re.sub(r"\b[PQRS]\b", "A", goal)) if not premises else None,
        )

    def protected(ks):
        return any(
            value is not None and value in guards[field]
            for value, field in zip(
                ks,
                (
                    "canonical_theorem_sha256",
                    "nonempty_premise_context_sha256",
                    "closed_formula_skeleton_sha256",
                ),
                strict=True,
            )
        )

    dev_keys = set()
    for filename in ("synthetic_dev.jsonl", "synthetic_long_dev.jsonl"):
        for row in load(data / filename):
            prompt, _ = split_record(row)
            dev_keys.add(keys(prompt)[0])

    counters = Counter()
    short, short_keys = [], set()
    for row in load(data / "sft_cap6.jsonl"):
        prompt, body = split_record(row)
        ks = keys(prompt)
        if protected(ks) or ks[0] in dev_keys:
            counters["short_protected_excluded"] += 1
            continue
        ok, reason, length = safe_verify(prompt, body or "")
        if (
            not ok
            or length > 6
            or not body
            or body.split()[-1] != "QED"
            or body.split().count("QED") != 1
        ):
            raise ValueError(f"Existing short label fails strict cap6 validation: {reason}")
        short_keys.add(ks[0])
        short.append(
            {
                **row,
                "prompt": prompt,
                "body": body,
                "n_lines": length,
                "canonical_hash": ks[0],
                "label_origin": "retained_cap6_generator_label",
            }
        )

    candidates = {}
    for relative in SOURCES:
        for row in load(root / relative):
            counters["long_source_rows"] += 1
            prompt, body = split_record(row)
            ks = keys(prompt)
            if protected(ks):
                counters["long_textbook_guard_excluded"] += 1
                continue
            if ks[0] in dev_keys:
                counters["long_synthetic_dev_excluded"] += 1
                continue
            if ks[0] in short_keys:
                counters["long_already_short_supervised_excluded"] += 1
                continue
            ok, reason, length = safe_verify(prompt, body or "")
            if not ok or not body or body.split()[-1] != "QED" or body.split().count("QED") != 1:
                counters["long_invalid_excluded"] += 1
                continue
            if row.get("reference_lines") != length:
                raise ValueError("Synthetic reference witness-length metadata is inconsistent")
            if length <= 6:
                counters["not_long_excluded"] += 1
                continue
            item = {
                "name": "cap_extension_" + ks[0][:20],
                "prompt": prompt,
                "body": body,
                "n_lines": length,
                "reference_lines": length,
                "canonical_hash": ks[0],
                "family": row.get("family", "unknown"),
                "source": relative,
                "label_origin": "new_longer_initial_cap_generator_reference",
                "previous_source_analysis_only": True,
                "authorization": "User explicitly permits longer initial proof cutoff; new cap arm",
            }
            if ks[0] not in candidates or length < candidates[ks[0]]["n_lines"]:
                candidates[ks[0]] = item

    ordered = sorted(
        candidates.values(), key=lambda r: digest(str(args.seed) + r["canonical_hash"])
    )
    supervised = [r for r in ordered if cohort(r["canonical_hash"], args.seed) == "sft"]
    rl = [
        {key: value for key, value in r.items() if key not in ("body", "reference_lines")}
        for r in ordered
        if cohort(r["canonical_hash"], args.seed) == "rl"
    ]
    for record in rl:
        record["role"] = "synthetic_cap_extension_rl_prompt_only"
        record["label_origin"] = "no_proof_label_released_to_rl"
        record.pop("previous_source_analysis_only")
        record.pop("authorization")
    rl_keys = {r["canonical_hash"] for r in rl}
    assert not rl_keys & (short_keys | dev_keys | {r["canonical_hash"] for r in supervised})
    if not rl or not short:
        raise ValueError("Empty short/RL population after protection")

    out.mkdir(parents=True)
    write_rows(out / "rl_long_disjoint.jsonl", rl)
    stats, hashes = {}, {}
    for cap in (8, 12):
        additional = [r for r in supervised if r["n_lines"] <= cap]
        if not additional:
            raise ValueError(f"No additional supervised labels available at cap{cap}")
        # Each arm contains each selected source row once; no implicit upweighting.
        additional = [dict(r, initial_generator_label_cap=cap) for r in additional]
        mixed = [dict(r, initial_generator_label_cap=cap) for r in short] + additional
        random.Random(args.seed + cap).shuffle(mixed)
        write_rows(out / f"additional_cap{cap}.jsonl", additional)
        write_rows(out / f"sft_cap{cap}_mixed.jsonl", mixed)
        stats[f"cap{cap}"] = {
            "additional": describe(additional),
            "mixed": describe(mixed),
            "initial_generator_label_cap": cap,
            "long_record_fraction": len(additional) / len(mixed),
        }
    for path in out.glob("*.jsonl"):
        hashes[path.name] = sha(path)
    manifest = {
        "status": "complete",
        "created_utc": datetime.now(UTC).isoformat(),
        "source_sha256": sha(Path(__file__)),
        "seed": args.seed,
        "input_sha256": inputs,
        "output_sha256": hashes,
        "arms": stats,
        "rl_prompt_only": describe(rl),
        "filter_counts": dict(counters),
        "protected_synthetic_dev_classes": len(dev_keys),
        "split": "Canonical classes globally assigned 80% supervised / 20% RL by seeded hash",
        "textbook_inputs_opened": False,
        "heldout_reference_files_opened": False,
        "lineage": "NEW initial-cap8/cap12 generator-label arms; never describe as cap6-only",
        "prior_reference_role": "Previously analysis-only RL-target references, now explicitly "
        "authorized for longer-initial-cap arms; old experiments unchanged",
        "limitations": [
            "Long RL prompts omit proof bodies and are disjoint from new supervised labels.",
            "Existing historical RL checkpoints might have sampled these old RL targets.",
            "Mixtures preserve surviving cap6 rows once; longer labels receive no repeats.",
        ],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "status": "complete",
                "out": str(out),
                "arms": stats,
                "rl_prompt_only": describe(rl),
                "filter_counts": dict(counters),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
