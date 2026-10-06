"""Independent theorem-paired adaptation audit; reads development outcomes only."""

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path.cwd()
BASE = ROOT / "artifacts/learned_compute_allocation_20261006/context"
os.environ["ND_OFFLINE"] = "1"
sys.path.insert(
    0, str(ROOT / "artifacts/learned_compute_allocation_20261006/source/nd-takehome-dan")
)
from lean_gate import check_sources  # noqa: E402 - runtime source path


def load(name, count):
    p = BASE / name
    rows = [json.loads(x) for x in (p / f"history_{count}/private.jsonl").read_text().splitlines()]
    return {r["theorem_id"]: {v["candidate_id"]: v for v in r["labels"]} for r in rows}


baseconf = json.loads((BASE / "base_dev48_v1/config.json").read_text())
adaptconf = json.loads((BASE / "format200_dev48_v1/config.json").read_text())
same = [
    "input_sha256",
    "limit",
    "k",
    "batch",
    "seed",
    "temperature",
    "max_action",
    "max_steps",
    "context",
]
report = {
    "config_equal": {k: baseconf[k] == adaptconf[k] for k in same},
    "checkpoint_hash_matches": hashlib.sha256(
        (ROOT / adaptconf["checkpoint"]).read_bytes()
    ).hexdigest()
    == adaptconf["checkpoint_sha256"],
    "comparisons": {},
    "counts": {},
}
sets = {}
for name in ["base_dev48_v1", "format200_dev48_v1"]:
    for c in [0, 1, 4]:
        rows = load(name, c)
        sets[(name, c)] = rows
        report["counts"][f"{name}/{c}"] = {
            "theorems": len(rows),
            "valid": sum(x["valid"] for r in rows.values() for x in r.values()),
            "total": sum(len(r) for r in rows.values()),
        }
ids = sorted(sets[("base_dev48_v1", 0)])
report["same_ids_and_candidate_sets"] = all(
    sorted(d) == ids and all(set(d[i]) == {0, 1, 2, 3} for i in ids) for d in sets.values()
)


def compare(a, b):
    va = np.array([sum(x["valid"] for x in sets[a][i].values()) / 4 for i in ids])
    vb = np.array([sum(x["valid"] for x in sets[b][i].values()) / 4 for i in ids])
    d = va - vb
    rng = np.random.default_rng(610061)
    draws = d[rng.integers(0, len(d), (50000, len(d)))].mean(1)
    return {
        "difference": float(d.mean()),
        "paired_theorem_percentile95": np.quantile(draws, [0.025, 0.975]).tolist(),
        "improved_theorems": int((d > 0).sum()),
        "worsened_theorems": int((d < 0).sum()),
        "same_theorems": int((d == 0).sum()),
        "bootstrap_draws": 50000,
        "bootstrap_seed": 610061,
    }


for c in [0, 1, 4]:
    report["comparisons"][f"adapt_minus_base_history{c}"] = compare(
        ("format200_dev48_v1", c), ("base_dev48_v1", c)
    )
for c in [1, 4]:
    report["comparisons"][f"adapt_history{c}_minus_history0"] = compare(
        ("format200_dev48_v1", c), ("format200_dev48_v1", 0)
    )
samples = []
for c in [0, 1, 4]:
    vals = [
        dict(x, theorem_id=i, history=c)
        for i, r in sets[("format200_dev48_v1", c)].items()
        for x in r.values()
        if x["parse_complete"]
    ]
    for valid in [True, False]:
        samples.extend([v for v in vals if v["valid"] == valid][:2])
flags, wall, cpu = check_sources([x["lean_text"] for x in samples])
report["lean_spotcheck"] = {
    "checked": len(samples),
    "disagreement": sum(f != x["valid"] for f, x in zip(flags, samples, strict=False)),
    "records": [
        {
            "theorem_id": x["theorem_id"],
            "candidate_id": x["candidate_id"],
            "history": x["history"],
            "saved_valid": x["valid"],
            "rechecked_valid": bool(f),
        }
        for x, f in zip(samples, flags, strict=False)
    ],
    "seconds": wall,
}
Path(__file__).with_name("adaptation_independent_audit.json").write_text(
    json.dumps(report, indent=2)
)
print(json.dumps(report, indent=2))
