"""Inventory selection-owned measured wall time without merged-run double counting."""

import hashlib
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path("artifacts/learned_compute_allocation_20261006")
OUT = ROOT / "selectors/resource_ledger"
OUT.mkdir(exist_ok=True)
rows = []


def add(path, seconds, kind, scope, gpu=None, monotonic=False):
    rows.append(
        {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "measured_seconds": seconds,
            "category": kind,
            "gpu": gpu,
            "timer_scope": scope,
            "timer_kind": "time.monotonic full-process wall"
            if monotonic
            else "time.time wall clock, not monotonic/kernel time",
        }
    )


for path in sorted((ROOT / "pools").glob("*/summary.json")):
    if "test" in path.parent.name:
        continue  # Never inspect unsealed TEST summaries in this development ledger.
    data = json.loads(path.read_text())
    if "seconds" not in data:
        continue
    config = json.loads((path.parent / "config.json").read_text())
    gpu = config.get("cuda_visible_devices")
    add(
        path,
        data["seconds"],
        "candidate_collection",
        (
            "After model loading through generation, CPU environment/Lean "
            "verification, likelihood replay and JSONL writing; phases not separately timed"
        ),
        gpu,
    )
for path in sorted((ROOT / "selectors").glob("*/history.json")):
    history = json.loads(path.read_text())
    config = json.loads((path.parent / "config.json").read_text())
    add(
        path,
        history[-1]["seconds"],
        "selector_training",
        (
            "Cumulative last-epoch timer, not sum of epochs; excludes model "
            "loading and initial untrained DEV pass, includes training/epoch DEV/prior "
            "checkpoint writes, excludes final checkpoint write"
        ),
        config.get("cuda_visible_devices"),
    )
for path in sorted((ROOT / "selectors").glob("*/standalone_dev/config.json")):
    config = json.loads(path.read_text())
    if "evaluation_wall_seconds" in config:
        add(
            path,
            config["evaluation_wall_seconds"],
            "standalone_evaluation",
            "Measured evaluate.py scope; model loading/IO inclusion determined by saved evaluator",
            config.get("cuda_visible_devices"),
        )
path = ROOT / "selectors/live_readout_headlr1e3_dev32_bce_s0/config.json"
data = json.loads(path.read_text())
add(
    path,
    data["critic_wall_seconds"],
    "live_path_readout",
    "Scoring and prediction assembly after model load; no generation/Lean",
    "1",
)
for path in sorted((ROOT / "selectors").glob("live_readout_main_s101_dev*/run_manifest.json")):
    data = json.loads(path.read_text())
    add(
        path,
        data["process_wall_seconds_monotonic"],
        "live_path_readout",
        "Full child process including model loading, scoring and IO; no generation/Lean",
        data["CUDA_VISIBLE_DEVICES"],
        monotonic=True,
    )
path = ROOT / "selectors/bce_s0_e4_clean_main_dev64_v1/run_manifest.json"
if path.exists():
    data = json.loads(path.read_text())
    add(
        path,
        data["process_wall_seconds_monotonic"],
        "retention_generation_probe",
        "Full child process includes generation, likelihood replay, CPU Lean and IO",
        data["CUDA_VISIBLE_DEVICES"],
        monotonic=True,
    )
path = ROOT / "selectors/throughput_probe/result.json"
data = json.loads(path.read_text())
add(
    path,
    sum(sum(v) for v in data.values()),
    "throughput_probe",
    "Sum of explicit synchronized model-forward probes; excludes setup",
    "4",
)
path = ROOT / "selectors/frozen_head_features_v1/config.json"
if path.is_file():
    data = json.loads(path.read_text())
    add(
        path,
        data["process_wall_seconds_monotonic"],
        "frozen_head_feature_extraction",
        "Full process including model load, both public feature caches and digest verification",
        data["cuda_visible_devices"],
        monotonic=True,
    )
path = ROOT / "selectors/frozen_head_fresh_dev_v3/config.json"
if path.is_file():
    data = json.loads(path.read_text())
    add(
        path,
        data["process_wall_seconds_monotonic"],
        "frozen_head_equivalence_evaluation",
        (
            "Evaluator monotonic scope includes model load, extraction, scoring and IO; "
            "excludes initial argument/head metadata setup"
        ),
        "1",
        monotonic=True,
    )
totals = defaultdict(float)
for row in rows:
    totals[row["category"]] += row["measured_seconds"]
unknown = [
    {"run": "pools/dev_pilot_v1", "status": "aborted early collector", "seconds": None},
    {
        "run": "selectors/format_smoke_s0",
        "status": "failed local checkpoint save before external write",
        "seconds": None,
    },
    {
        "run": "selectors/base_pooled_main_dev",
        "status": "completed evaluation but no elapsed timer persisted",
        "seconds": None,
    },
    {
        "run": "selectors/base_factorized_main_dev",
        "status": "completed evaluation but no elapsed timer persisted",
        "seconds": None,
    },
    {
        "run": "selectors/factorized_smoke_transitions_s0/standalone_dev",
        "status": "completed old evaluator without elapsed timer",
        "seconds": None,
    },
]
result = {
    "rows": rows,
    "cpu_only_head_fit_seconds": json.loads(
        (ROOT / "selectors/frozen_head_v1/config.json").read_text()
    )["process_wall_seconds_monotonic"],
    "category_measured_seconds": dict(totals),
    "measured_scope_total_seconds": sum(totals.values()),
    "measured_scope_total_gpu_assigned_hours": sum(totals.values()) / 3600,
    "unknown_or_unmetered_runs": unknown,
    "limitations": [
        (
            "Measured scopes are a lower bound on all job occupancy, not "
            "exact GPU utilization or GPU kernel time."
        ),
        (
            "Collector scopes include CPU Lean and Python overhead; these "
            "cannot be separated retrospectively."
        ),
        (
            "Trainer cumulative epoch timers are counted only once per child "
            "run; merged pools are never counted."
        ),
        "Early configs omit GPU metadata; null GPU is left unknown rather than inferred.",
        (
            "Historical collector/train scopes lack monotonic timers; "
            "new run manifests include them. "
            "No estimates invented for missing runs."
        ),
        (
            "CPU-only data generation, exposure audits, rendering and Lean "
            "view audits excluded from GPU total."
        ),
        "Root context/probe jobs and allocation-owned jobs excluded.",
    ],
}
(OUT / "ledger.json").write_text(json.dumps(result, indent=2) + "\n")
(OUT / "source.py").write_text(Path(__file__).read_text())
print(
    json.dumps(
        {
            k: result[k]
            for k in [
                "category_measured_seconds",
                "measured_scope_total_seconds",
                "measured_scope_total_gpu_assigned_hours",
            ]
        }
    )
)
