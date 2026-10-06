"""Prepare command/signature inventories without opening any TEST outcomes."""

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

FOLDER = Path(__file__).resolve().parent
ROOT = FOLDER.parents[4]
ART = ROOT / "artifacts/learned_compute_allocation_20261006"
PYTHON = "/data/users/dmitry/venvs/proof-state/bin/python"


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(ART / "freeze_inputs/selection_v1"))
    parser.add_argument("--main-registry", default=str(ART / "final_evaluation/main_protocol.json"))
    parser.add_argument(
        "--historical-registry", default=str(ART / "final_evaluation/historical_protocol.json")
    )
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    evaluator = module(FOLDER / "evaluate.py", "freeze_primary_eval")
    head_eval = module(FOLDER / "evaluate_frozen_head.py", "freeze_head_eval")
    sys.path.insert(0, str(FOLDER.parent / "analysis"))
    head_policy = module(FOLDER / "frozen_head_thresholds.py", "freeze_head_policy")
    head_compare = module(FOLDER / "compare_frozen_head.py", "freeze_head_compare")
    statistics = module(FOLDER / "aggregate_selection.py", "freeze_selection_statistics")
    from evaluation_freeze import signature_sha256

    original = json.loads((ART / "selectors/final_signature_inventory.json").read_text())[
        "registered_analyses"
    ]
    if len(original) != 36:
        raise ValueError("Expected all36 original signatures")
    for item in original:
        sig = item["signature"]
        expected = evaluator.evaluation_signature(
            SimpleNamespace(
                checkpoint=item["checkpoint"],
                context=8192,
                view="transitions",
                architecture=sig["architecture"],
                objective=sig["objective"],
                evaluation_seed=sig["evaluation_permutation_seed"],
            )
        )
        if sig != expected or signature_sha256(sig) != item["signature_sha256"]:
            raise ValueError("Original signature inventory is stale")
    head = ART / "selectors/frozen_head_v1/head.pt"
    base = ART / "checkpoints/la_T1_best12_s0_r16.pt"
    policy = ART / "selectors/frozen_head_threshold_dev_v4/policy.json"
    dev_config = ART / "selectors/frozen_head_standalone_dev_v4/config.json"
    bce_configs = [
        ART / f"selectors/main_factorized_bce_transitions_s{s}/standalone_dev/config.json"
        for s in range(3)
    ]
    feature_model = ART / "analysis/features1600_main_dev/feature_model.json"
    head_signature = head_eval.evaluation_signature(str(head), str(base))
    compare_args = SimpleNamespace(
        head_config=str(dev_config),
        bce_configs=list(map(str, bce_configs)),
        feature_model=str(feature_model),
    )
    secondary = [
        {"label": "frozen_head", "signature": head_signature},
        {
            "label": "frozen_head_thresholds",
            "signature": head_policy.analysis_signature(head_signature, policy),
        },
        {
            "label": "frozen_head_paired_statistics",
            "signature": head_compare.analysis_signature(compare_args),
        },
    ]
    for entry in secondary:
        entry["signature_sha256"] = signature_sha256(entry["signature"])
    chosen = {}
    for name in ["frozen_primary_dev_choices.json", "frozen_bce_dev_choices.json"]:
        for row in json.loads((ART / "selectors" / name).read_text())["runs"]:
            chosen[Path(row["run"]).name] = row
    dataset_entries = {}
    for dataset, n, seed, input_data, registry, pool_name in [
        ("main", 400, 91006, ART / "data/main_v1/test.jsonl", args.main_registry, "main_test"),
        (
            "historical",
            72,
            93006,
            ART / "data/textbook72_historical_public_v1/public.jsonl",
            args.historical_registry,
            "historical_textbook72_test",
        ),
    ]:
        pool = ART / "pools" / pool_name
        public, private = pool / "public_transitions.jsonl", pool / "private.jsonl"
        predictions = ART / "selectors" / f"final_{dataset}"
        analysis = ART / "analysis" / f"final_{dataset}"
        runs, commands, all_entries = [], [], [*original, *secondary]
        for entry in original:
            sig = entry["signature"]
            target = predictions / entry["label"]
            command = [
                PYTHON,
                str(FOLDER / "evaluate.py"),
                "--checkpoint",
                entry["checkpoint"],
                "--public",
                str(public),
                "--private",
                str(private),
                "--split",
                "test",
                "--out",
                str(target),
                "--context",
                "8192",
                "--view",
                "transitions",
                "--architecture",
                sig["architecture"],
                "--objective",
                sig["objective"],
                "--evaluation-seed",
                str(sig["evaluation_permutation_seed"]),
                "--frozen-test-protocol",
                registry,
            ]
            commands.append(
                {
                    "label": entry["label"],
                    "argv": command,
                    "out": str(target),
                    "signature_sha256": entry["signature_sha256"],
                }
            )
            name = entry["label"].rsplit("_perm", 1)[0]
            if name in chosen:
                choice = chosen[name]
                group = "bce" if sig["objective"] == "binary" else sig["architecture"]
                runs.append(
                    {
                        "group": group,
                        "training_seed": choice["optimizer_seed"],
                        "permutation_seed": sig["evaluation_permutation_seed"],
                        "checkpoint": entry["checkpoint"],
                        "checkpoint_sha256": sig["checkpoint_sha256"],
                        "predictions": str(target / "predictions.json"),
                        "prediction_config": str(target / "config.json"),
                    }
                )
        inventory_path = out / f"{dataset}_selection_inventory.json"
        write(
            inventory_path,
            {
                "runs": runs,
                "feature_comparator": {
                    "model_path": str(feature_model),
                    "model_sha256": statistics.digest(feature_model),
                    "source_path": str(FOLDER.parent / "analysis/analyze_candidates.py"),
                    "source_sha256": statistics.digest(
                        FOLDER.parent / "analysis/analyze_candidates.py"
                    ),
                },
            },
        )
        aggregate_sig = statistics.analysis_signature(
            SimpleNamespace(inventory=str(inventory_path), bootstrap=5000, seed=61006)
        )
        all_entries.append(
            {
                "label": "primary_selection_statistics",
                "signature": aggregate_sig,
                "signature_sha256": signature_sha256(aggregate_sig),
            }
        )
        head_out = predictions / "frozen_head"
        commands.append(
            {
                "label": "frozen_head",
                "argv": [
                    PYTHON,
                    str(FOLDER / "evaluate_frozen_head.py"),
                    "--head",
                    str(head),
                    "--checkpoint",
                    str(base),
                    "--public",
                    str(public),
                    "--private",
                    str(private),
                    "--split",
                    "test",
                    "--out",
                    str(head_out),
                    "--context",
                    "8192",
                    "--batch",
                    "8",
                    "--frozen-test-protocol",
                    registry,
                ],
            }
        )
        commands.append(
            {
                "label": "frozen_head_thresholds",
                "argv": [
                    PYTHON,
                    str(FOLDER / "frozen_head_thresholds.py"),
                    "--public",
                    str(public),
                    "--private",
                    str(private),
                    "--predictions",
                    str(head_out / "predictions.json"),
                    "--prediction-config",
                    str(head_out / "config.json"),
                    "--policy",
                    str(policy),
                    "--out",
                    str(analysis / "frozen_head_thresholds"),
                    "--frozen-test-protocol",
                    registry,
                ],
            }
        )
        bce_out = [
            predictions / f"main_factorized_bce_transitions_s{s}_perm77106" for s in range(3)
        ]
        commands.append(
            {
                "label": "frozen_head_paired_statistics",
                "argv": [
                    PYTHON,
                    str(FOLDER / "compare_frozen_head.py"),
                    "--public",
                    str(public),
                    "--private",
                    str(private),
                    "--head-predictions",
                    str(head_out / "predictions.json"),
                    "--head-config",
                    str(head_out / "config.json"),
                    "--bce-predictions",
                    *[str(p / "predictions.json") for p in bce_out],
                    "--bce-configs",
                    *[str(p / "config.json") for p in bce_out],
                    "--features",
                    str(
                        ART
                        / "analysis"
                        / (
                            "main_test_features_v1"
                            if dataset == "main"
                            else "historical_test_features_v1"
                        )
                        / "per_theorem.jsonl"
                    ),
                    "--feature-model",
                    str(feature_model),
                    "--out",
                    str(analysis / "frozen_head_paired"),
                    "--frozen-test-protocol",
                    registry,
                ],
            }
        )
        commands.append(
            {
                "label": "primary_selection_statistics",
                "argv": [
                    PYTHON,
                    str(FOLDER / "aggregate_selection.py"),
                    "--inventory",
                    str(inventory_path),
                    "--public",
                    str(public),
                    "--private",
                    str(private),
                    "--features",
                    str(
                        ART
                        / "analysis"
                        / (
                            "main_test_features_v1"
                            if dataset == "main"
                            else "historical_test_features_v1"
                        )
                        / "per_theorem.jsonl"
                    ),
                    "--split",
                    "test",
                    "--out",
                    str(analysis / "selection"),
                    "--bootstrap",
                    "5000",
                    "--seed",
                    "61006",
                    "--frozen-test-protocol",
                    registry,
                ],
            }
        )
        collector = [
            PYTHON,
            str(FOLDER / "collect.py"),
            "--data",
            str(input_data),
            "--out",
            str(pool),
            "--seed",
            str(seed),
            "--k",
            "8",
            "--temperature",
            "0.8",
            "--batch",
            "256",
            "--max-action",
            "256",
            "--max-steps",
            "48",
        ]
        dataset_entries[dataset] = {
            "status": "planned_not_authorization",
            "expected_rows": n,
            "registered_analyses": all_entries,
            "commands": commands,
            "statistical_inventory": str(inventory_path),
            "collection": {
                "argv": collector,
                "input_sha256": statistics.digest(input_data),
                "collector_sha256": statistics.digest(FOLDER / "collect.py"),
                "renderer_sha256": statistics.digest(FOLDER / "render_transitions.py"),
                "checkpoint_sha256": statistics.digest(base),
                "seed": seed,
                "batch": 256,
                "precision": "BF16 model; FP32 sampling logits; BF16 replay/FP32 diagnostics",
                "family_batching": (
                    "single combined process in immutable input order, "
                    "dynamically refilled active attempts"
                ),
                "private_stdout_stderr_redirect": str(pool) + ".private_generation.log",
                "public_raw": str(pool / "public.jsonl"),
                "public_transitions": str(public),
                "private": str(private),
                "sealer_sha256": statistics.digest(FOLDER / "seal_public_pool.py"),
                "sealer_argv": [
                    PYTHON,
                    str(FOLDER / "seal_public_pool.py"),
                    "--raw",
                    str(pool / "public.jsonl"),
                    "--transitions",
                    str(public),
                    "--expected-data",
                    str(input_data),
                    "--expected-theorems",
                    str(n),
                    "--expected-candidates",
                    "8",
                    "--out",
                    str(pool / "public_seal.json"),
                ]
                + (["--enrich-registered-metadata"] if dataset == "historical" else []),
                "renderer_argv": [
                    PYTHON,
                    str(FOLDER / "render_transitions.py"),
                    "--input",
                    str(pool / "public.jsonl"),
                    "--output",
                    str(public),
                ],
            },
        }
        write(out / f"{dataset}_entries.json", dataset_entries[dataset])
    context = FOLDER.parent / "context"
    sys.path.insert(0, str(context))
    probe = module(context / "causal_probe.py", "freeze_causal_probe")
    causal_analysis = module(context / "analyze_causal.py", "freeze_causal_analysis")
    causal_aggregate = module(context / "aggregate_causal.py", "freeze_causal_aggregate")
    panel = ART / "context/causal_test_panel_v1/public.jsonl"
    checkpoint_by_seed = {
        101: ART / "allocation/no_cost/main_s101/checkpoint_00200.pt",
        102: ART / "allocation/no_cost/main_s102/checkpoint_00200.pt",
    }
    base_policy = ART / "allocation/format_main96_lr1e5_clean8_initial50_s61/checkpoint_00999.pt"
    causal_runs, causal_entries, causal_commands = [], [], []
    for condition, (
        limit,
        replicates,
        continuation,
        forced_extra,
        history_limit,
    ) in causal_aggregate.CONDITIONS.items():
        for seed in (101, 102):
            run = ART / "context/final_main_causal" / f"nocost_s{seed}_{condition}"
            ns = SimpleNamespace(
                source=str(ART / "source/nd-takehome-dan"),
                base=str(base_policy),
                checkpoint=str(checkpoint_by_seed[seed]),
                public=str(panel),
                limit=limit,
                replicates=replicates,
                continuation=continuation,
                forced_extra=forced_extra,
                history_removal_limit=history_limit,
                counts=[1, 2],
                seed=81006,
                context=16384,
                attempt_bound=1024,
                state_reserve=1024,
                max_attempts=8,
                max_action=256,
                max_steps=48,
                temperature=0.8,
            )
            sig = probe.intervention_signature(ns)
            causal_runs.append(
                {
                    "condition": condition,
                    "training_seed": seed,
                    "run": str(run),
                    "analysis": str(run / "analysis"),
                    "intervention_signature": sig,
                    "generation_registry": args.main_registry,
                }
            )
            for label, entry_sig in [
                ("generation", sig),
                ("analysis", causal_analysis.analysis_signature(sig, 5000, 81006)),
            ]:
                causal_entries.append(
                    {
                        "label": f"causal_s{seed}_{condition}_{label}",
                        "signature": entry_sig,
                        "signature_sha256": signature_sha256(entry_sig),
                    }
                )
            command = [PYTHON, str(context / "causal_probe.py")]
            for name, value in vars(ns).items():
                command += ["--" + name.replace("_", "-")]
                command += list(map(str, value)) if isinstance(value, list) else [str(value)]
            command += [
                "--out",
                str(run),
                "--split",
                "test",
                "--frozen-test-protocol",
                args.main_registry,
            ]
            causal_commands += [
                {"label": f"causal_s{seed}_{condition}", "argv": command},
                {
                    "label": f"analyze_causal_s{seed}_{condition}",
                    "argv": [
                        PYTHON,
                        str(context / "analyze_causal.py"),
                        "--run",
                        str(run),
                        "--out",
                        str(run / "analysis"),
                        "--bootstrap",
                        "5000",
                        "--seed",
                        "81006",
                        "--frozen-test-protocol",
                        args.main_registry,
                    ],
                },
            ]
    causal_inventory = out / "causal_test_inventory.json"
    write(
        causal_inventory,
        {
            "public": str(panel),
            "generation_seed": 81006,
            "checkpoint_sha256_by_seed": {
                str(s): statistics.digest(p) for s, p in checkpoint_by_seed.items()
            },
            "runs": causal_runs,
        },
    )
    aggregate_sig = causal_aggregate.signature(causal_inventory, 5000, 81006)
    causal_entries.append(
        {
            "label": "causal_two_checkpoint_aggregate",
            "signature": aggregate_sig,
            "signature_sha256": signature_sha256(aggregate_sig),
        }
    )
    causal_commands.append(
        {
            "label": "causal_two_checkpoint_aggregate",
            "argv": [
                PYTHON,
                str(context / "aggregate_causal.py"),
                "--inventory",
                str(causal_inventory),
                "--out",
                str(ART / "context/final_main_causal/aggregate"),
                "--bootstrap",
                "5000",
                "--seed",
                "81006",
                "--frozen-test-protocol",
                args.main_registry,
            ],
        }
    )
    write(
        out / "causal_entries.json",
        {
            "status": "planned_not_authorization",
            "registered_analyses": causal_entries,
            "commands": causal_commands,
            "inventory": str(causal_inventory),
        },
    )
    write(
        out / "manifest.json",
        {
            "status": "planned_not_authorization",
            "builder_sha256": statistics.digest(__file__),
            "selection_launch_helper_sha256": statistics.digest(
                FOLDER / "run_frozen_test_queue.py"
            ),
            "main_entries": str(out / "main_entries.json"),
            "historical_entries": str(out / "historical_entries.json"),
            "causal_entries": str(out / "causal_entries.json"),
            "head_sha256": statistics.digest(head),
            "head_policy_sha256": statistics.digest(policy),
            "equivalence_audit": str(ART / "selectors/frozen_head_equivalence.json"),
            "registry_paths": {"main": args.main_registry, "historical": args.historical_registry},
            "freeze_note": str(FOLDER / "frozen_head_test_freeze.md"),
        },
    )
    (out / "source.py").write_bytes(Path(__file__).read_bytes())
    print(
        json.dumps(
            {
                "out": str(out),
                "selection_signatures_per_dataset": len(
                    dataset_entries["main"]["registered_analyses"]
                ),
                "causal_signatures": len(causal_entries),
            }
        )
    )


if __name__ == "__main__":
    main()
