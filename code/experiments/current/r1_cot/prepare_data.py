"""Blind textbook split and guarded cap-six data construction; CPU only.

No payload values are printed. Textbook reference proofs are never used. Run
against take-home with --root and a fresh --out. The sealed final directory is
an access-policy boundary, not cryptographic protection from its owner.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=20260909)
    ap.add_argument("--sft-size", type=int, default=60000)
    ap.add_argument("--rl-size", type=int, default=1500)
    args = ap.parse_args()
    root, out = args.root.resolve(), args.out.resolve()
    if out.exists():
        raise ValueError("Fresh output directory required")
    out.mkdir(parents=True)
    sys.path[:0] = [str(root), str(root / "transfer_sprint_20260907/code")]
    from nd_data import canonical_theorem, sequent_parts
    from strict_verify import safe_verify

    audit = []

    def read(path, role):
        path = root / path
        audit.append(
            {
                "path": str(path),
                "purpose": role,
                "sha256": file_sha(path),
                "utc": datetime.now(timezone.utc).isoformat(),
                "payload_printed": False,
            }
        )
        with path.open() as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)

    def prompt(row):
        if "prompt" in row:
            return " ".join(row["prompt"].split())
        return " ".join(row["text"].split("PRF", 1)[0].split()) + " PRF"

    def keys(p):
        premises, goal = sequent_parts(p)
        canonical = digest(canonical_theorem(p))
        context = (
            digest(canonical_theorem("THM " + " , ".join(premises) + " SEQ F PRF"))
            if premises
            else None
        )
        skeleton = digest(re.sub(r"\b[PQRS]\b", "A", goal)) if not premises else None
        return canonical, context, skeleton

    textbook = {}
    sources = [
        "window_sprint_20260906/theory/textbook_catalog112_v1/problems112.jsonl",
        "transfer_sprint_20260907/theory/textbook_smith_final_v1/metadata.analysis_only.jsonl",
    ]
    raw_count = 0
    for source in sources:
        for row in read(source, "blind split: prompt and witness-length metadata only"):
            raw_count += 1
            p = prompt(row)
            k, c, s = keys(p)
            length = row.get("reference_lines")
            if length is None:
                raise ValueError("Missing textbook reference witness length")
            item = {
                "name": "textbook_" + k[:20],
                "prompt": p,
                "reference_lines": int(length),
                "canonical_hash": k,
                "context_hash": c,
                "skeleton_hash": s,
            }
            if k not in textbook or item["reference_lines"] < textbook[k]["reference_lines"]:
                textbook[k] = item
    rows = sorted(textbook.values(), key=lambda r: digest(str(args.seed) + r["canonical_hash"]))
    total = len(rows)
    n_train, n_final = total // 10, (total + 1) // 2
    # Connected components keep shared premise contexts/closed skeletons together.
    parent = list(range(total))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    previous = {}
    for i, row in enumerate(rows):
        for field in ("canonical_hash", "context_hash", "skeleton_hash"):
            val = row[field]
            if val is not None:
                tag = (field, val)
                if tag in previous:
                    parent[find(i)] = find(previous[tag])
                else:
                    previous[tag] = i
    groups = defaultdict(list)
    for i in range(total):
        groups[find(i)].append(i)
    components = sorted(
        groups.values(), key=lambda g: digest(str(args.seed) + rows[g[0]]["canonical_hash"])
    )
    states = {(0, 0): (0, 0)}
    for j, group in enumerate(components):
        size = len(group)
        new = dict(states)
        for (t, f), (tm, fm) in states.items():
            if t + size <= n_train:
                new.setdefault((t + size, f), (tm | 1 << j, fm))
            if f + size <= n_final:
                new.setdefault((t, f + size), (tm, fm | 1 << j))
        states = new
    if (n_train, n_final) not in states:
        raise ValueError(
            "Requested split sizes cannot preserve structural components; no split released"
        )
    tm, fm = states[n_train, n_final]
    partitions = {"train": [], "dev": [], "final": []}
    for j, group in enumerate(components):
        role = "train" if tm >> j & 1 else "final" if fm >> j & 1 else "dev"
        partitions[role].extend(rows[i] for i in group)
    protected = [set(), set(), set()]
    all_textbook = [set(), set(), set()]
    for role, part in partitions.items():
        for row in part:
            for i, field in enumerate(("canonical_hash", "context_hash", "skeleton_hash")):
                if row[field]:
                    all_textbook[i].add(row[field])
                    if role != "train":
                        protected[i].add(row[field])
        clean = []
        for row in part:
            item = {k: row[k] for k in ("name", "prompt", "canonical_hash")}
            item["role"] = "textbook_" + role
            if role != "train":
                item["reference_lines"] = row["reference_lines"]
            clean.append(item)
        path = out / (
            "sealed_final/textbook_final.jsonl"
            if role == "final"
            else "textbook_" + role + ".jsonl"
        )
        write_rows(path, clean)
        if role == "final":
            path.chmod(0o600)
            path.parent.chmod(0o700)

    def guard_blob(sets):
        return {
            "canonical_theorem_sha256": sorted(sets[0]),
            "nonempty_premise_context_sha256": sorted(sets[1]),
            "closed_formula_skeleton_sha256": sorted(sets[2]),
        }

    write_json(out / "heldout_guards.json", guard_blob(protected))
    write_json(out / "all_textbook_guards.json", guard_blob(all_textbook))
    split = {
        "seed": args.seed,
        "raw_textbook_rows": raw_count,
        "canonical_textbook_classes": total,
        "counts": {k: len(v) for k, v in partitions.items()},
        "component_sizes": sorted(map(len, components)),
        "grouped_by": ["canonical", "premise_context", "closed_skeleton"],
        "textbook_supervised_proof_labels": 0,
        "training_fraction": n_train / total,
        "final_policy": "Open only after final checkpoints and evaluation settings are frozen",
        "dev_policy": (
            "Evaluate only after each RL run completes; no intermediate checkpoint evaluation"
        ),
        "prior_exposure_caveat": (
            "Fresh sprint split; historical evaluation exposure is not erased by repartitioning."
        ),
    }
    assert n_train <= total * 0.1
    write_json(out / "split_protocol.json", split)
    print(
        json.dumps({"stage": "split_complete", "counts": split["counts"], "out": str(out)}),
        flush=True,
    )

    def blocked(ks, guards):
        return any(k is not None and k in g for k, g in zip(ks, guards, strict=True))

    # Audit every input row, without exposing matching textbook identities.
    source_paths = ["data/train.jsonl"] + [
        "transfer_sprint_20260907/data/" + x + "/train.jsonl"
        for x in ("structural_cap6_v1", "scope_cap6_v2", "negation_repair_cap6_v1")
    ]
    family_rows = defaultdict(list)
    counts = {}
    seen = set()
    for source in source_paths:
        stats = Counter()
        for row in read(source, "comprehensive inherited source guard audit and cap-six sampling"):
            stats["rows"] += 1
            p = prompt(row)
            ks = keys(p)
            for label, guards in [("heldout", protected), ("all_textbook", all_textbook)]:
                for kind, k, guard in zip(
                    ("canonical", "context", "skeleton"), ks, guards, strict=True
                ):
                    stats[label + "_" + kind + "_matches"] += int(k is not None and k in guard)
            if blocked(ks, all_textbook):
                stats["excluded_textbook_structural"] += 1
                continue
            if ks[0] in seen:
                stats["duplicate"] += 1
                continue
            body = row.get("proof") or row["text"].split("PRF", 1)[1].strip()
            ok, reason, n = safe_verify(p, body)
            if not ok:
                stats["invalid_strict"] += 1
                continue
            if n > 6:
                stats["above_cap"] += 1
                continue
            seen.add(ks[0])
            family = row.get("family", "generated_" + str(n))
            item = {
                "name": "sft_" + ks[0][:20],
                "prompt": p,
                "proof": body,
                "text": p + " " + body,
                "n_lines": n,
                "canonical_hash": ks[0],
                "family": family,
                "role": "sft_train",
            }
            family_rows[family].append(item)
            stats["eligible"] += 1
        counts[source] = dict(stats)
        print(
            json.dumps({"stage": "source_audited", "source": source, "counts": dict(stats)}),
            flush=True,
        )
    rng = random.Random(args.seed)
    for values in family_rows.values():
        rng.shuffle(values)

    def balanced_take(n):
        selected = []
        names = sorted(family_rows)
        while len(selected) < n:
            progressed = False
            for family in names:
                if family_rows[family] and len(selected) < n:
                    selected.append(family_rows[family].pop())
                    progressed = True
            if not progressed:
                break
        return selected

    synthetic_dev = balanced_take(256)
    for row in synthetic_dev:
        row["role"] = "synthetic_dev"
    sft = balanced_take(args.sft_size)
    write_rows(out / "sft_cap6.jsonl", sft)
    write_rows(out / "synthetic_dev.jsonl", synthetic_dev)
    used = {r["canonical_hash"] for r in sft + synthetic_dev}
    rl_families = defaultdict(list)
    rl_stats = Counter()
    for family in ("structural_long_v1", "negation_long_v1"):
        source = "transfer_sprint_20260907/data/" + family + "/rl_targets.jsonl"
        for row in read(source, "RL prompt-only targets; no reference proof files read"):
            rl_stats["rows"] += 1
            if any(field in row for field in ("proof", "text", "reference_proof")):
                raise ValueError("Long labels in RL source are prohibited")
            p = prompt(row)
            ks = keys(p)
            if blocked(ks, all_textbook) or ks[0] in used:
                rl_stats["excluded"] += 1
                continue
            used.add(ks[0])
            item = {
                "name": "rl_" + ks[0][:20],
                "prompt": p,
                "canonical_hash": ks[0],
                "family": row.get("family", family),
                "role": "rl_train",
            }
            rl_families[item["family"]].append(item)
    for values in rl_families.values():
        rng.shuffle(values)
    rl = []
    while len(rl) < args.rl_size:
        progressed = False
        for family in sorted(rl_families):
            if rl_families[family] and len(rl) < args.rl_size:
                rl.append(rl_families[family].pop())
                progressed = True
        if not progressed:
            break
    tb_train = [
        {k: r[k] for k in ("name", "prompt", "canonical_hash")}
        | {"role": "rl_train", "family": "textbook_train"}
        for r in partitions["train"]
    ]
    write_rows(out / "rl_no_textbook.jsonl", rl)
    write_rows(out / "rl_with_textbook.jsonl", rl + tb_train)
    write_json(
        out / "inherited_guard_audit.json",
        {
            "sources": counts,
            "rl_sources": dict(rl_stats),
            "claim_scope": (
                "Counts audit these source files; not proof of checkpoint training lineage."
            ),
        },
    )
    write_json(out / "payload_read_audit.json", audit)
    files = {str(p.relative_to(out)): file_sha(p) for p in sorted(out.rglob("*")) if p.is_file()}
    write_json(
        out / "manifest.json",
        {
            "seed": args.seed,
            "files": files,
            "split": split,
            "counts": {
                "sft": len(sft),
                "synthetic_dev": len(synthetic_dev),
                "rl_no_textbook": len(rl),
                "rl_with_textbook": len(rl) + len(tb_train),
            },
            "sft_families": dict(Counter(r["family"] for r in sft)),
            "supervised_cap": 6,
            "builder_sha256": file_sha(Path(__file__)),
            "verifier_sha256": file_sha(root / "transfer_sprint_20260907/code/strict_verify.py"),
        },
    )
    print(
        json.dumps(
            {
                "stage": "complete",
                "out": str(out),
                "sft": len(sft),
                "rl": len(rl),
                "textbook_train": len(tb_train),
            }
        )
    )


if __name__ == "__main__":
    main()
