"""Prepare reviewable live-evaluation registration; root alone seals the registry."""

import argparse
import json
from pathlib import Path

from evaluation_protocol import fingerprint, signature, validate_saved_implementation

WORK = Path(__file__).resolve().parents[1]
ROOT = WORK.parents[3]
ART = ROOT / "artifacts/learned_compute_allocation_20261006"
ALLOC = WORK / "allocation"


def sha(path):
    import hashlib

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def model_entry(folder, arm, seed):
    choice_file = ART / f"allocation/{folder}/main_s{seed}_selection.json"
    choice = json.loads(choice_file.read_text())
    if choice["selection_split"] != "dev":
        raise ValueError("Checkpoint was not selected on development")
    selected = next(
        row
        for row in choice["candidates"]
        if row["completed_updates"] == choice["selected_completed_updates"]
    )
    checkpoint = Path(choice["selected_checkpoint"])
    if sha(checkpoint) != choice["selected_checkpoint_sha256"]:
        raise ValueError("Selected checkpoint bytes differ from recorded choice")
    # The first no-cost101 choice omitted the path but bound its exact config hash.
    derived = (
        ART
        / "allocation"
        / folder
        / (f"main_s{seed}_dev{choice['selected_completed_updates']}_paths_s52006")
    )
    dev = Path(selected.get("dev_output", derived))
    config = json.loads((dev / "config.json").read_text())
    validate_saved_implementation(config)
    if (
        config["evaluation_split"] != "dev"
        or config["seed"] != 52006
        or config["repeats"] != 1
        or config["evaluated_theorems"] != 32
        or config["data_sha256"] != sha(ART / "allocation/main_dev32_public_panel.jsonl")
        or config["checkpoint_sha256"] != sha(checkpoint)
    ):
        raise ValueError("Checkpoint choice uses a different DEV protocol")
    inference = fingerprint(signature(config))
    if (
        inference != selected["inference_sha256"]
        or sha(dev / "config.json") != selected["dev_config_sha256"]
    ):
        raise ValueError("DEV inference/config does not match recorded selection")
    entry = dict(
        arm=arm,
        rl_seed=seed,
        base_sha256=config["base_sha256"],
        checkpoint_sha256=config["checkpoint_sha256"],
        inference_sha256=inference,
        base_path=config["base"],
        checkpoint_path=str(checkpoint),
        selected_completed_updates=choice["selected_completed_updates"],
        selection_sha256=sha(choice_file),
        dev_config_sha256=sha(dev / "config.json"),
        dev_evaluation_path=str(dev),
        native_cost=config["cost"],
    )
    if arm in {"cost", "no_cost"}:
        fit_name = f"main_s{seed}_dev{choice['selected_completed_updates']}_baselines_frozen_s52006"
        fits = ART / "allocation" / folder / fit_name
        for filename, key, fitter in [
            ("counter.json", "counter_hazards_sha256", "fit_counter_hazards.py"),
            ("budgets.json", "budget_baselines_sha256", "fit_budget_baselines.py"),
        ]:
            path = fits / filename
            artifact = json.loads(path.read_text())
            if (
                artifact["fit_split"] != "dev"
                or artifact["inference_sha256"] != inference
                or artifact["fitter_sha256"] != sha(ALLOC / fitter)
            ):
                raise ValueError("DEV-fit provenance does not match selected model/current fitter")
            entry[key], entry[key.replace("_sha256", "_path")] = sha(path), str(path)
    return entry


def format_entry():
    dev = ART / "allocation/format_only/dev32_paths_s52006"
    config = json.loads((dev / "config.json").read_text())
    summary = json.loads((dev / "summary.json").read_text())
    if summary["benchmark_measurement"]["paths"] != 32:
        raise ValueError("Format-only development audit did not complete the panel")
    validate_saved_implementation(config)
    if (
        config["checkpoint_sha256"] is not None
        or config["evaluation_split"] != "dev"
        or config["seed"] != 52006
        or config["repeats"] != 1
        or config["evaluated_theorems"] != 32
        or config["data_sha256"] != sha(ART / "allocation/main_dev32_public_panel.jsonl")
    ):
        raise ValueError("Format-only baseline unexpectedly contains RL weights")
    return dict(
        arm="format_only",
        rl_seed=None,
        base_sha256=config["base_sha256"],
        checkpoint_sha256=None,
        inference_sha256=fingerprint(signature(config)),
        base_path=config["base"],
        checkpoint_path=None,
        native_cost=0.0,
        dev_config_sha256=sha(dev / "config.json"),
        dev_evaluation_path=str(dev),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--allow-pending", action="store_true")
    args = parser.parse_args()
    entries, pending = [], []
    targets = [(arm, arm, seed) for arm in ("cost", "no_cost") for seed in (101, 102, 103)]
    targets += [("fixed1", "fixed1_control", 101), ("fixed4", "fixed4_control", 101)]
    for folder, arm, seed in targets:
        try:
            entries.append(model_entry(folder, arm, seed))
        except FileNotFoundError as error:
            if not args.allow_pending:
                raise
            pending.append(dict(arm=arm, rl_seed=seed, missing_path=str(error.filename)))
            entries.append(dict(arm=arm, rl_seed=seed, pending=True))
    try:
        entries.append(format_entry())
    except FileNotFoundError as error:
        if not args.allow_pending:
            raise
        pending.append(dict(arm="format_only", rl_seed=None, missing_path=str(error.filename)))
        entries.append(dict(arm="format_only", rl_seed=None, pending=True))
    stats = dict(
        script_sha256=sha(ALLOC / "aggregate_live.py"),
        bootstrap_samples=10000,
        bootstrap_seed=72006,
        confidence=0.95,
        family_data_sha256=sha(ART / "data/main_v1/test.jsonl"),
    )
    historical_stats = dict(
        script_sha256=sha(ALLOC / "aggregate_historical.py"),
        aggregate_live_dependency_sha256=sha(ALLOC / "aggregate_live.py"),
        bootstrap_samples=10000,
        bootstrap_seed=102006,
        confidence=0.95,
        family_data_sha256=sha(ART / "data/textbook72_historical_public_v1/public.jsonl"),
    )
    result = dict(
        status="draft_pending" if pending else "ready_for_root_seal",
        registered_models=entries,
        pending=pending,
        postprocessing_hashes={
            name: sha(ALLOC / name)
            for name in (
                "fit_counter_hazards.py",
                "reanalyze_counter.py",
                "fit_budget_baselines.py",
                "reanalyze_budget_baselines.py",
            )
        },
        statistical_analysis=stats,
        main=dict(
            test_data_sha256=sha(ART / "data/main_v1/test.jsonl"),
            test_seed=62006,
            test_repeats=2,
            test_theorems=400,
        ),
        historical=dict(
            test_data_sha256=sha(ART / "data/textbook72_historical_public_v1/public.jsonl"),
            test_seed=92006,
            test_repeats=2,
            test_theorems=72,
            statistical_analysis=historical_stats,
            historical_exposure_audit_sha256=sha(
                ART / "data/textbook72_metadata_audit/exposure.json"
            ),
        ),
        builder_sha256=sha(__file__),
        launch_helper_sha256=sha(ALLOC / "launch_live_test.py"),
        planned_run_manifests={
            benchmark: dict(
                path=str(ART / f"freeze_inputs/allocation_v1/{benchmark}_live_runs.json"),
                sha256=sha(ART / f"freeze_inputs/allocation_v1/{benchmark}_live_runs.json"),
            )
            for benchmark in ("main", "historical")
        },
    )
    out = Path(args.out)
    with out.open("x") as stream:
        json.dump(result, stream, indent=2)
    out.with_suffix(".builder.py").write_bytes(Path(__file__).read_bytes())


if __name__ == "__main__":
    main()
