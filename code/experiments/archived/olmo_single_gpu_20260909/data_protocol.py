"""Freeze a matched cap-six OLMo baseline without opening textbook payloads.

ICL and SFT use the identical selected proof records. DPO may use these gold
responses against on-policy rejected outputs; short-only RL receives exactly
their prompts. The builder reads only the original synthetic training/dev
files and opaque textbook guards/protocol. No final/test/catalog payload or
saved model predictions are accessed. All output directories must be fresh.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[2]
VENDOR = ROOT / "r1-sprint-20260909-1625/code/vendor"
sys.path.insert(0, str(VENDOR))
from nd_data import canonical_theorem, sequent_parts  # noqa: E402
from strict_verify import safe_verify  # noqa: E402


GUARD_FIELDS = (
    "canonical_theorem_sha256",
    "nonempty_premise_context_sha256",
    "closed_formula_skeleton_sha256",
)
PROOF_FIELDS = {"text", "proof", "body", "reference_proof", "completion", "answer"}


def digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def file_sha(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            hasher.update(block)
    return hasher.hexdigest()


def theorem_keys(prompt: str) -> tuple[str, str | None, str | None]:
    premises, goal = sequent_parts(prompt)
    context = "THM " + " , ".join(premises) + " SEQ F PRF"
    return (
        digest(canonical_theorem(prompt)),
        digest(canonical_theorem(context)) if premises else None,
        digest(re.sub(r"\b[PQRS]\b", "A", goal)) if not premises else None,
    )


def add_keys(guards: list[set[str]], keys) -> None:
    for guard, key in zip(guards, keys, strict=True):
        if key is not None:
            guard.add(key)


def blocked(guards: list[set[str]], keys) -> bool:
    return any(key is not None and key in guard for guard, key in zip(guards, keys, strict=True))


def ordered_balanced(rows: list[dict], count: int, seed: int, field="family") -> list[dict]:
    """Fixed hash order within strata; round robin gives nested pool prefixes."""
    groups = defaultdict(list)
    for row in rows:
        groups[row[field]].append(row)
    for group in groups.values():
        group.sort(key=lambda row: digest(f"{seed}:{row['canonical_hash']}"), reverse=True)
    result = []
    while len(result) < count:
        progressed = False
        for group_name in sorted(groups):
            if groups[group_name] and len(result) < count:
                result.append(groups[group_name].pop())
                progressed = True
        if not progressed:
            raise ValueError(f"Requested {count} rows, only {len(result)} available")
    return result


def summary(rows: list[dict]) -> dict:
    return {
        "rows": len(rows),
        "canonical_classes": len({row["canonical_hash"] for row in rows}),
        "family_counts": dict(sorted(Counter(row["family"] for row in rows).items())),
        "reference_length_counts": dict(sorted(Counter(row.get("n_lines", row.get("reference_lines")) for row in rows).items())),
        "whitespace_tokens_with_proof": sum(len(row.get("text", "").split()) for row in rows),
        "characters_with_proof": sum(len(row.get("text", "")) for row in rows),
    }


def run(args) -> dict:
    source = args.source.resolve()
    out = args.out.resolve()
    if out.exists():
        raise ValueError(f"Refusing to overwrite released data directory: {out}")
    reads = []

    def load(name, purpose, expected=None, lines=False):
        # Explicit allowlist prevents accidental expansion to textbook payloads.
        if name not in {
            "manifest.json", "split_protocol.json", "all_textbook_guards.json",
            "heldout_guards.json", "sft_cap6.jsonl", "synthetic_dev.jsonl",
            "synthetic_long_dev.jsonl", "supplement_protocol.json",
        }:
            raise ValueError("Input is outside the frozen synthetic-only allowlist")
        path = source / name
        identity = file_sha(path)
        if expected is not None and identity != expected:
            raise ValueError(f"Original pinned input changed: {name}")
        reads.append({"path": str(path), "sha256": identity, "purpose": purpose})
        with path.open() as stream:
            return [json.loads(line) for line in stream if line.strip()] if lines else json.load(stream)

    original = load("manifest.json", "Original input identities and split metadata")
    expected = original["files"]
    split = load("split_protocol.json", "Retain original textbook partitions and gates", expected["split_protocol.json"])
    if split["counts"] != {"train": 14, "dev": 58, "final": 73}:
        raise ValueError("Original 14/58/73 textbook partition changed")
    textbook = load("all_textbook_guards.json", "Opaque exclusions for all textbook classes", expected["all_textbook_guards.json"])
    heldout = load("heldout_guards.json", "Preserve original protected textbook hash guards", expected["heldout_guards.json"])
    for blob in (textbook, heldout):
        if set(blob) != set(GUARD_FIELDS) or not all(
            re.fullmatch(r"[0-9a-f]{64}", value) for values in blob.values() for value in values
        ):
            raise ValueError("Malformed textbook hash guards")
    textbook_guards = [set(textbook[field]) for field in GUARD_FIELDS]
    guards = [set(values) for values in textbook_guards]
    dev_short = load("synthetic_dev.jsonl", "Existing synthetic short development and structural exclusions", expected["synthetic_dev.jsonl"], lines=True)
    dev_long = load("synthetic_long_dev.jsonl", "Existing synthetic long development prompts only", lines=True)
    load("supplement_protocol.json", "Provenance of inherited long synthetic development")
    if (len(dev_short), len(dev_long)) != (256, 384):
        raise ValueError("Original synthetic development sizes changed")
    seen_dev = set()
    for role, rows in (("synthetic_short_development", dev_short), ("synthetic_long_development", dev_long)):
        for row in rows:
            keys = theorem_keys(row["prompt"])
            if row.get("canonical_hash", keys[0]) != keys[0] or keys[0] in seen_dev:
                raise ValueError("Synthetic development identity mismatch or duplicate")
            if blocked(textbook_guards, keys):
                raise ValueError("Synthetic development intersects a textbook guard")
            seen_dev.add(keys[0])
            row["canonical_hash"] = keys[0]
            row["role"] = role
            add_keys(guards, keys)
            if role == "synthetic_long_development" and PROOF_FIELDS & set(row):
                raise ValueError("A reference answer is present in long prompt-only development")

    training = load("sft_cap6.jsonl", "Select identical training proofs for ICL/SFT/DPO and short RL prompts", expected["sft_cap6.jsonl"], lines=True)
    eligible, rejected, seen = [], Counter(), set()
    for row in training:
        keys = theorem_keys(row["prompt"])
        if keys[0] != row["canonical_hash"] or keys[0] in seen:
            raise ValueError("Training theorem identity mismatch or duplicate")
        seen.add(keys[0])
        hits = [field for field, key, guard in zip(GUARD_FIELDS, keys, guards, strict=True) if key is not None and key in guard]
        if hits:
            rejected.update(hits)
            continue
        if not 2 <= int(row["n_lines"]) <= args.x:
            rejected["above_supervised_cap"] += 1
            continue
        eligible.append(row)
    shared = ordered_balanced(eligible, args.train_size, args.seed)
    for row in shared:
        ok, reason, actual = safe_verify(row["prompt"], row["proof"])
        if not ok or actual != row["n_lines"] or actual > args.x:
            raise ValueError(f"Selected supervised proof failed strict replay: {reason}")
        row["role"] = "matched_icl_sft_dpo_gold"

    def prompt_record(row, role, length_metadata=False):
        result = {field: row[field] for field in ("name", "prompt", "canonical_hash", "family")}
        result["role"] = role
        if length_metadata:
            result["reference_lines"] = int(row.get("reference_lines", row.get("n_lines")))
            result["length_semantics"] = "verified_witness_upper_bound_not_minimum"
        return result

    rl = [prompt_record(row, "rl_short_train") for row in shared]
    short = [prompt_record(row, "synthetic_short_development", True) for row in dev_short]
    long = [prompt_record(row, "synthetic_long_development", True) for row in dev_long]
    short_panel = ordered_balanced(short, args.short_panel, args.seed + 1)
    long_panel = ordered_balanced(long, args.long_panel, args.seed + 2, "reference_lines")
    # The long primary panel is length-balanced; full-source family metrics
    # remain available. No model outcomes enter selection.
    outputs = {
        "shared_cap6.jsonl": shared,
        "rl_short_prompts.jsonl": rl,
        "eval_short_prompts.jsonl": short,
        "eval_long_prompts.jsonl": long,
        "eval_short_panel.jsonl": short_panel,
        "eval_long_panel.jsonl": long_panel,
    }
    if {row["canonical_hash"] for row in shared} != {row["canonical_hash"] for row in rl}:
        raise AssertionError("Shared demonstration and short RL prompt classes differ")
    if any(PROOF_FIELDS & set(row) or "reference_lines" in row or "n_lines" in row for row in rl):
        raise AssertionError("Supervised targets or length hints leaked into RL input")

    out.mkdir(parents=True)
    for name, rows in outputs.items():
        with (out / name).open("x") as stream:
            for row in rows:
                stream.write(json.dumps(row, sort_keys=True) + "\n")
    guard_blob = dict(zip(GUARD_FIELDS, [sorted(values) for values in guards], strict=True))
    (out / "training_exclusions.json").write_text(json.dumps(guard_blob, indent=2, sort_keys=True) + "\n")
    manifest = {
        "status": "frozen_before_model_scoring",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "x": args.x,
        "builder_sha256": file_sha(Path(__file__)),
        "verifier_sha256": {str(path.relative_to(ROOT)): file_sha(path) for path in (
            VENDOR / "strict_verify.py", VENDOR / "nd_metrics.py", VENDOR / "nd_verify/verify.py", VENDOR / "nd_data.py"
        )},
        "input_reads": reads,
        "source_training_count": len(training),
        "eligible_training_count_after_all_guards": len(eligible),
        "exclusions_by_guard": dict(rejected),
        "summaries": {name: summary(rows) for name, rows in outputs.items()},
        "output_sha256": {name: file_sha(out / name) for name in [*outputs, "training_exclusions.json"]},
        "textbook_counts_preserved": split["counts"],
        "textbook_payloads_read": 0,
        "saved_prediction_files_read": 0,
        "textbook_training_classes_used": 0,
        "strict_replayed_selected_short_labels": len(shared),
        "icl_sft_dpo_gold_examples_identical": True,
        "rl_classes_equal_shared_training_classes": True,
        "rl_exposed_to_long_prompts": False,
        "train_development_disjointness": list(GUARD_FIELDS),
        "long_panel_sampling": "Deterministic round robin over reference lengths, hash order within lengths; family balance is not enforced",
        "length_caveat": "Training x is actual accepted proof lines. Evaluation x+P bins are existing verified reference-witness upper bounds, not certified minimum proof complexity. Score any strictly valid generated proof regardless of its length.",
        "training_information_caveat": "Matched proof identities do not imply matched token exposures or compute. ICL repeats the whole proof bank at inference; SFT/DPO/RL may revisit it during optimization. Record both costs.",
        "pretraining_caveat": "Unknown natural-deduction/textbook exposure in the pretrained OLMo checkpoint prevents a claim of proof reasoning learned from scratch.",
        "dpo_policy": "Use shared gold proofs and model-sampled rejected responses on these same prompts. Do not draw preference labels or candidate responses from development prompts.",
        "textbook_development_gate": split["dev_policy"],
        "synthetic_development_gate": "Fixed synthetic panels may evaluate frozen baseline/endpoints before RL completes; training tuning uses only the shared32 training pool, never synthetic development outcomes.",
        "final_gate": split["final_policy"],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    report = args.report.resolve()
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        "# Matched OLMo cap-six data audit\n\n"
        f"The shared pool has {len(shared)} strictly verified proofs of at most {args.x} lines, sampled deterministically across all {len(set(r['family'] for r in shared))} represented families. ICL and SFT receive these identical examples; DPO may pair their gold answers against training-policy rejected outputs. Primary RL sees exactly these short prompts and receives no proof or reference-length fields. No long-prompt RL exposure occurs in this primary baseline.\n\n"
        f"All 145 textbook classes remain excluded by canonical, premise-context and closed-goal-skeleton hashes. The original train/dev/final allocation remains 14/58/73; no textbook payload, final/test payload, or saved prediction was opened. Stronger structural exclusions against all 640 existing synthetic development classes leave {len(eligible):,} eligible examples from the original 60,000.\n\n"
        f"The development release retains all 256 short and 384 long prompts. The fixed small panels have {len(short_panel)} short examples balanced across source families and {len(long_panel)} long examples balanced across reference lengths 7–14 as availability permits. Longer-panel counts are {dict(sorted(Counter(r['reference_lines'] for r in long_panel).items()))}. Panel selection used seed {args.seed} and hashes, with no model outcomes. Full-pool and family-stratified metrics should accompany any follow-up expansion.\n\n"
        "Evaluation length is a verified witness upper bound, not certified minimum ND proof length. A valid shorter generated proof counts as success. Existing long families mix structure and formula complexity with length, so a slope by length alone does not isolate a causal length effect. A future extra control can generate fresh forward derivations with fixed seeds and requested 7–16-line post-pruning bins, apply these same opaque guards, and retain prompts plus witness lengths only; the same minimum-length caveat still applies.\n\n"
        "The shared identities equalize task information, not token exposure or compute. Report demonstrations per ICL query, training examples/epochs, optimization updates, rollout tokens, and evaluation decode budgets separately. Long-prompt RL is a separate exposure treatment if later authorized; it must not be reported as pure short-to-long extrapolation. Pretrained proof exposure is unknown.\n\n"
        f"Reproduce with `python3 code/data_protocol.py --out {out} --train-size {args.train_size} --short-panel {args.short_panel} --long-panel {args.long_panel}` in the sprint directory, choosing a fresh output directory. `data/manifest.json` records every input read, output hash, verifier hash, pool count, and data boundary.\n"
    )
    return {"out": str(out), "report": str(report), "train": len(shared), "short_panel": len(short_panel), "long_panel": len(long_panel), "eligible_training": len(eligible)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "r1-cot-20260909/data")
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "data")
    parser.add_argument("--report", type=Path, default=Path(__file__).resolve().parents[1] / "reports/data_audit.md")
    parser.add_argument("--seed", type=int, default=202609092014)
    parser.add_argument("--x", type=int, default=6)
    parser.add_argument("--train-size", type=int, default=32)
    parser.add_argument("--short-panel", type=int, default=56)
    parser.add_argument("--long-panel", type=int, default=64)
    args = parser.parse_args()
    if args.x != 6 or args.train_size < 28 or args.short_panel < 28 or args.long_panel < 8:
        parser.error("This protocol requires x=6, >=28 training/short-panel examples and >=8 long-panel examples")
    print(json.dumps(run(args), sort_keys=True))


if __name__ == "__main__":
    main()
