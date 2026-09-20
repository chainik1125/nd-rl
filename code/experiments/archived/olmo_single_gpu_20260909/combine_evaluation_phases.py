"""Read-only synthetic phase replay; write a fresh, jointly indexed analysis report.

The frozen analyzer deliberately rejects differing mode schedules. This separate
helper reuses its literal/native-token/compute-ledger replay, compares arms within
each phase, and permits greedy and sampled phases to share one report. It never
merges their draws into a new metric or opens textbook data. Missing runs remain
pending; a completed greedy comparison stays complete if sampling is unfinished.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
from types import SimpleNamespace

from analyze import compare_runs, plots, replay_run
from evaluate import load_targets, sha, stable_hash
from protocol import SPRINT

ARMS = ("base_zero", "base_icl32", "sft", "dpo", "rl")
COHORTS = {"short_panel": 56, "long_panel": 64}
MODES = ("greedy", "sampled")


def option(args: list[str], name: str) -> list[str]:
    if args.count(name) != 1:
        raise ValueError(f"Expected one {name} option")
    index = args.index(name) + 1
    end = index
    while end < len(args) and not args[end].startswith("--"):
        end += 1
    return args[index:end]


def one(args: list[str], name: str) -> str:
    values = option(args, name)
    if len(values) != 1:
        raise ValueError(f"Expected one value for {name}")
    return values[0]


def inside(path: Path, root: Path) -> Path:
    path = path.resolve()
    if path == root.resolve() or not path.is_relative_to(root.resolve()):
        raise ValueError(f"Path must be a specific child of {root}: {path}")
    return path


def inspect_phase(path: Path, root: Path = SPRINT) -> dict:
    """Read only explicitly supplied pipeline metadata, never prediction discovery."""
    path = inside(path, root / "plans")
    raw = json.loads(path.read_text())
    mode = raw.get("mode")
    if mode not in MODES:
        raise ValueError("Only the frozen synthetic greedy/sampled phases are supported")
    runs, seen = [], set()
    for step in raw["steps"]:
        args = step["args"]
        arm = one(args, "--arm")
        if step["entrypoint"] != "evaluate.py" or arm not in ARMS or arm in seen:
            raise ValueError("Phase has an unsupported or duplicate arm")
        seen.add(arm)
        cohorts = option(args, "--cohorts")
        if len(cohorts) != 2 or set(cohorts) != set(COHORTS):
            raise ValueError("Only the fixed synthetic 56/64 panels may be opened")
        if option(args, "--modes") != [mode] or one(args, "--max-new-tokens") != "2048":
            raise ValueError("Phase differs from its frozen decode mode/cap")
        adapter = None
        if "--adapter-path" in args:
            adapter = str(inside(Path(one(args, "--adapter-path")), root / "runs"))
        if (adapter is None) != arm.startswith("base_"):
            raise ValueError("Base/trained adapter declaration is inconsistent")
        runs.append({"arm": arm, "mode": mode,
                     "directory": str(inside(Path(one(args, "--out")), root / "runs")),
                     "batch_size": int(one(args, "--batch-size")),
                     "seed": int(one(args, "--seed")), "adapter": adapter,
                     "cohorts": cohorts})
    if seen != set(ARMS):
        raise ValueError("Every phase must explicitly plan all five frozen arms")
    return {"mode": mode, "path": str(path), "sha256": sha(path), "runs": runs,
            "source_freeze": raw.get("source_freeze"),
            "not_after_utc": raw.get("not_after_utc")}


def normalized_identity(plan: dict) -> str:
    """Modes and physical batch size are the only allowed phase/arm differences."""
    fields = ("model_id", "model_revision", "model_config_sha256",
              "download_manifest_sha256", "tokenizer_identity", "tokenizer_files",
              "protocol", "modeling_sha256", "evaluator_sha256", "target_identity")
    return stable_hash({**{field: plan[field] for field in fields},
                        "settings": {key: value for key, value in plan["settings"].items()
                                     if key not in ("modes", "batch_size")}})


def guard_run(plan: dict, expected: dict, root: Path = SPRINT) -> None:
    """Validate opaque metadata before the frozen replay opens any run payload."""
    if plan["arm"] != expected["arm"] or plan["cohorts"] != expected["cohorts"]:
        raise ValueError("Run arm or cohort order differs from its phase plan")
    if set(plan["cohorts"]) != set(COHORTS) or plan["target_count"] != sum(COHORTS.values()):
        raise ValueError("Only the fixed synthetic 56/64 panels may be opened")
    settings = plan["settings"]
    required = {"modes": [expected["mode"]], "batch_size": expected["batch_size"],
                "seed": expected["seed"], "max_new_tokens": 2048, "sampled_k": 4,
                "sampled_temperature": 1.0, "top_p": 1.0, "top_k": 0,
                "stop_on_actually_generated_final_marker": True,
                "native_eos_fallback": True, "inserted_eos": False}
    if any(settings.get(key) != value for key, value in required.items()):
        raise ValueError("Run settings differ from the frozen phase")
    expected_demos = 32 if expected["arm"] == "base_icl32" else 0
    if plan["demonstrations_per_query"] != expected_demos:
        raise ValueError("Run changed the shared demonstration exposure")
    model_path = (root / "models/olmo32b").resolve()
    if Path(plan["model_path"]).resolve() != model_path:
        raise ValueError("Only the pinned local model tokenizer may be opened")
    if any(Path(name).resolve().parent != model_path or
           not (Path(name).name.startswith("tokenizer") or Path(name).name == "special_tokens_map.json")
           for name in plan["tokenizer_files"]):
        raise ValueError("Tokenizer metadata points outside the pinned local model")
    allowed_inputs = {str((root / "data" / name).resolve()) for name in
                      ("manifest.json", "shared_cap6.jsonl", "eval_short_panel.jsonl", "eval_long_panel.jsonl")}
    if set(plan["input_files"]) != allowed_inputs:
        raise ValueError("Run refers to data outside the fixed synthetic inputs")
    for name, expected_hash in plan["input_files"].items():
        if sha(Path(name)) != expected_hash:
            raise ValueError("A frozen synthetic input changed")
    for field, name in (("modeling_sha256", "modeling.py"), ("evaluator_sha256", "evaluate.py")):
        if plan[field] != sha(root / "code" / name):
            raise ValueError("Current replay runtime differs from the frozen evaluation runtime")
    # Reconstruct the expected target manifest from only allowed synthetic data.
    targets, _ = load_targets(SimpleNamespace(data_dir=root / "data", cohorts=expected["cohorts"]))
    if stable_hash(targets) != plan["target_identity"]:
        raise ValueError("Evaluation membership differs from the original fixed panels")
    files = plan["adapter_files"]
    if expected["adapter"] is None:
        if files:
            raise ValueError("Base arm unexpectedly pins an adapter")
    else:
        directory = Path(expected["adapter"])
        actual = {str(path.resolve()) for path in directory.glob("adapter*") if path.is_file()}
        if set(files) != actual or not {str(directory / name) for name in
                ("adapter_config.json", "adapter_model.safetensors")}.issubset(actual):
            raise ValueError("Run does not pin its complete declared endpoint")
        if any(sha(Path(name)) != digest for name, digest in files.items()):
            raise ValueError("A frozen endpoint checkpoint changed")


def validate_join(entries: list[dict]) -> None:
    identities, adapter_by_arm, observed = set(), {}, set()
    for entry in entries:
        plan = entry["plan"]
        identities.add(normalized_identity(plan))
        arm = plan["arm"]
        adapter = stable_hash(plan["adapter_files"])
        if arm in adapter_by_arm and adapter_by_arm[arm] != adapter:
            raise ValueError("An arm uses different checkpoints in different phases")
        adapter_by_arm[arm] = adapter
        for row in entry["per_theorem"]:
            key = (row["arm"], row["cohort"], row["mode"], row["canonical_hash"])
            if key in observed:
                raise ValueError("Duplicate arm/cohort/mode/theorem would double count")
            observed.add(key)
    if len(identities) > 1:
        raise ValueError("Phases differ beyond decode mode or physical batch size")


def compose(phases: list[dict], entries: list[dict], availability: list[dict], draws: int) -> dict:
    validate_join(entries)
    phases_out, comparisons = [], []
    for phase in phases:
        mode = phase["mode"]
        selected = [entry for entry in entries if entry["plan"]["settings"]["modes"] == [mode]]
        complete = (len(selected) == len(ARMS)
                    and all(entry["result"]["status"] == "complete" for entry in selected))
        rows = [row for entry in selected for row in entry["per_theorem"]]
        checks = compare_runs(rows, draws, partial=not complete) if rows else []
        for check in checks:
            check["all_five_arms_complete_in_phase"] = complete
            check["headline_eligible"] = complete
        comparisons.extend(checks)
        phases_out.append({"mode": mode, "status": "complete" if complete else
                           ("partial_diagnostic" if selected else "pending"),
                           "complete_arms": [entry["result"]["arm"] for entry in selected
                                             if entry["result"]["status"] == "complete"],
                           "required_arms": list(ARMS), "intended_theorems_per_arm": COHORTS,
                           "headline_eligible": complete, "audited_theorem_mode_rows": len(rows)})
    totals = Counter()
    for entry in entries:
        for key, value in (entry["result"].get("actual_compute_totals_including_regeneration") or {}).items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                totals[key] += value
    complete = bool(phases_out) and all(phase["status"] == "complete" for phase in phases_out)
    return {"status": "complete" if complete else ("partial_diagnostic" if entries else "pending"),
            "phases": phases_out, "availability": availability,
            "runs": [entry["result"] for entry in entries], "paired_comparisons": comparisons,
            "actual_compute_totals_including_regeneration": dict(totals),
            "phase_plans": phases,
            "raw_success_is_primary": True, "compressed_proofs_analysis_only": True,
            "phase_merge_rule": "Greedy pass1 and observed sampled pass4 stay separate; no best-of-five metric is invented.",
            "partial_rule": "Missing attempts are pending, not failures. Partial intersections are diagnostic only; headline comparisons require all five complete arms within that phase.",
            "length_caveat": "Reference proof lengths are witness upper bounds, not certified minimum difficulty. Shorter valid certificates count.",
            "uncertainty_caveat": "Frozen theorem-paired percentile bootstrap; one training seed, no training-seed uncertainty, no multiplicity correction. Different physical batches do not imply common random numbers.",
            "protected_payloads_opened": False}


def run(args) -> dict:
    out = inside(args.out, SPRINT / "reports")
    if out.exists():
        raise ValueError("Output must be fresh; existing reports are preserved")
    phases = [inspect_phase(path) for path in args.phase_plans]
    if len({phase["mode"] for phase in phases}) != len(phases):
        raise ValueError("Supply only one explicit pipeline plan per mode")
    entries, availability = [], []
    for phase in phases:
        for expected in phase["runs"]:
            directory = Path(expected["directory"])
            plan_path = directory / "evaluation_plan.json"
            evidence_files = [directory / name for name in
                              ("evaluation_plan.json", "targets.jsonl", "attempts.jsonl", "compute_batches.jsonl")]
            if not all(path.is_file() for path in evidence_files):
                availability.append({**expected, "status": "pending_not_replayed",
                                     "missing_files": [path.name for path in evidence_files if not path.is_file()]})
                if not args.allow_partial:
                    raise ValueError("A planned run is missing; --allow-partial permits pending diagnostics")
                continue
            plan = json.loads(plan_path.read_text())
            guard_run(plan, expected)
            before = {str(path): sha(path) for path in evidence_files}
            result, rows, proofs = replay_run(directory, allow_partial=args.allow_partial)
            if any(sha(Path(path)) != digest for path, digest in before.items()):
                raise ValueError("Run changed during replay; retry after a stable batch/checkpoint")
            result["source_artifact_sha256"] = before
            entries.append({"plan": plan, "result": result, "per_theorem": rows, "certificates": proofs})
            availability.append({**expected, "status": result["status"]})
    report = compose(phases, entries, availability, args.bootstrap_draws)
    report.update(created_utc=datetime.now(timezone.utc).isoformat(),
                  helper_sha256=sha(Path(__file__)), frozen_analyzer_sha256=sha(Path(__file__).with_name("analyze.py")))
    out.mkdir(parents=True)
    figures = []
    for phase in phases:
        results = [entry["result"] for entry in entries
                   if entry["plan"]["settings"]["modes"] == [phase["mode"]]]
        phase_complete = next(value["headline_eligible"] for value in report["phases"]
                              if value["mode"] == phase["mode"])
        # Frozen figure titles describe a matched panel, so never use those
        # figures to present a time-selected partial intersection as complete.
        if results and phase_complete and not args.no_plots:
            plot_dir = out / phase["mode"]
            plot_dir.mkdir()
            figures.extend(f"{phase['mode']}/{name}" for name in plots(results, plot_dir))
    report["figures"] = figures
    for name, key in (("per_theorem.jsonl", "per_theorem"), ("certificates_analysis_only.jsonl", "certificates")):
        with (out / name).open("x") as stream:
            for entry in entries:
                for row in entry[key]:
                    stream.write(json.dumps(row, sort_keys=True) + "\n")
    with (out / "summary.json").open("x") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    lines = ["# Joint index of audited synthetic evaluation phases", "", "Status: " + report["status"], "",
             "| Phase | State | Complete arms | Headline comparison eligible |",
             "|---|---|---|---|"]
    for phase in report["phases"]:
        lines.append(f"| {phase['mode']} | {phase['status']} | {', '.join(phase['complete_arms']) or 'none'} | {phase['headline_eligible']} |")
    lines += ["", "| Arm | Phase | Cohort | Solved / completed targets | Intended targets |",
              "|---|---|---|---:|---:|"]
    for result in report["runs"]:
        for cohort, modes in result["groups"].items():
            for mode, metrics in modes.items():
                value = metrics["overall"]
                lines.append(f"| {result['arm']} | {mode} | {cohort} | {value['solved']}/{value['n']} | {value['intended_n']} |")
    for key in ("phase_merge_rule", "partial_rule", "length_caveat", "uncertainty_caveat"):
        lines.extend(["", report[key]])
    lines += ["", "Every reported outcome uses the frozen literal/native-token/strict-verifier/compute-ledger replay. No textbook payload is opened. This helper does not authorize training continuation or protected evaluation.", ""]
    with (out / "report.md").open("x") as stream:
        stream.write("\n".join(lines))
    return {"status": report["status"], "out": str(out), "replayed_runs": len(entries),
            "figures": figures, "protected_payloads_opened": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase-plans", nargs="+", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    if args.bootstrap_draws < 100:
        parser.error("At least 100 bootstrap draws are required")
    print(json.dumps(run(args), sort_keys=True))


if __name__ == "__main__":
    main()
