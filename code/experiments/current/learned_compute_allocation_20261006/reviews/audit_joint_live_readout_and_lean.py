"""Released MAIN identities and a prospectively specified outcome-independent Lean sample."""

import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
ART = ROOT / "artifacts/learned_compute_allocation_20261006"
SOURCE = ART / "source/nd-takehome-dan"
os.environ["LEAN_GATE_WORKERS"] = "2"
sys.path.insert(0, str(SOURCE))
from lean_gate import check_sources  # noqa: E402
from tokenizer import make_tokenizer  # noqa: E402


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines()]


assert read(ART / "analysis/root_main_barrier_release_v1.json")[
    "verified_all_9_launch_bytes_and_5_immutable_inputs"
]
manifest = read(ART / "freeze_inputs/allocation_v1/main_live_runs.json")["runs"]
ids = sorted(r["theorem_id"] for r in rows(ART / "data/main_v1/test.jsonl"))
# Commit sample IDs before opening the sample's saved labels.
plans = []
for run in manifest:
    arm, seed = run["arm"], run["seed"]
    chosen = sorted(
        ids, key=lambda tid: hashlib.sha256(f"{arm}:{seed}|{tid}|repeat0".encode()).hexdigest()
    )[:2]
    plans.append(
        {
            "arm": arm,
            "seed": seed,
            "theorem_ids": chosen,
            "repeat": 0,
            "candidates": "first and last, deduplicate",
        }
    )
Path(__file__).with_name("joint_live_lean_sample_plan.json").write_text(
    json.dumps(plans, indent=2) + "\n"
)
tok = make_tokenizer("lean_staten")
results = []
sources = []
sample_records = []
for run, plan in zip(manifest, plans, strict=True):
    directory = Path(run["paths"])
    counters = {
        "prefixes": 0,
        "unique_first_argmax": 0,
        "first_among_tied_argmax": 0,
        "first_not_argmax": 0,
    }
    by_count = {}
    selected_rows = []
    for path in rows(directory / "public_paths.jsonl"):
        if path["repeat"] == 0 and path["theorem_id"] in plan["theorem_ids"]:
            selected_rows.append(path)
        for decision in path["decisions"]:
            p = decision["selection_probabilities"]
            maxima = [i for i, v in enumerate(p) if v == max(p)]
            counters["prefixes"] += 1
            category = (
                "unique_first_argmax"
                if maxima == [0]
                else "first_among_tied_argmax"
                if 0 in maxima
                else "first_not_argmax"
            )
            counters[category] += 1
            by_count.setdefault(len(p), {k: 0 for k in counters})
            by_count[len(p)]["prefixes"] += 1
            by_count[len(p)][category] += 1
    # Identities are now fixed independently of validity.
    labels = {
        (r["theorem_id"], r["repeat"]): r["validities"]
        for r in rows(directory / "private_labels.jsonl")
    }
    for path in selected_rows:
        for k in sorted({0, len(path["candidates"]) - 1}):
            candidate = path["candidates"][k]
            text = tok.statement(path["prompt"]) + " " + candidate["transition_text"]
            sources.append(text)
            sample_records.append(
                {
                    "arm": run["arm"],
                    "seed": run["seed"],
                    "theorem_id": path["theorem_id"],
                    "repeat": 0,
                    "candidate_index": k,
                    "source_sha256": hashlib.sha256(text.encode()).hexdigest(),
                    "saved_valid": labels[path["theorem_id"], 0][k],
                }
            )
    results.append(
        {"arm": run["arm"], "seed": run["seed"], "all_prefixes": counters, "by_count": by_count}
    )
    print("identity", run["arm"], run["seed"], counters, flush=True)
flags, wall, cpu = check_sources(sources)
for r, flag in zip(sample_records, flags, strict=True):
    r["direct_lean_valid"] = flag
    assert flag == r["saved_valid"], r
report = {
    "identities": results,
    "lean_sample": sample_records,
    "sample_count": len(sources),
    "verdict_disagreements": 0,
    "lean_wall_seconds": wall,
    "lean_cpu_seconds": cpu,
    "scope": (
        "Descriptive identity audit; fixed hash-selected direct Lean sample. No model inference."
    ),
}
Path(__file__).with_name("joint_live_readout_and_lean_audit.json").write_text(
    json.dumps(report, indent=2) + "\n"
)
print("Lean sample passed", len(sources), flush=True)
