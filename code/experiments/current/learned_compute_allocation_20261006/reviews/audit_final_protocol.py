"""Synthetic parent/public-seal checks. Never launches collection or reads outcomes."""

import contextlib
import importlib.util
import io
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

ROOT = Path.cwd()
W = ROOT / "code/experiments/current/learned_compute_allocation_20261006"
src = W / "analysis/final_protocol.py"
spec = importlib.util.spec_from_file_location("final_protocol_review", src)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def write(p, v):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(v) + "\n")


result = {"builder_sha256": m.sha(src), "cases": {}}
for case in [
    "clean",
    "missing_primary_fit",
    "changed_fit",
    "changed_command",
    "private_read_true",
    "changed_public",
    "changed_parent",
    "wrong_launcher",
]:
    with tempfile.TemporaryDirectory(prefix="composition_", dir=W / "reviews") as tmp:
        t = Path(tmp)
        artifact = t / "model"
        artifact.write_text("synthetic")
        dev = t / "dev"
        write(dev / "config.json", {"synthetic": True})
        fit = t / "fit.json"
        write(fit, {"synthetic": True})
        models = []
        identities = [(a, s) for a in ["cost", "no_cost"] for s in [101, 102, 103]] + [
            ("fixed1_control", 101),
            ("fixed4_control", 101),
            ("format_only", None),
        ]
        for arm, seed in identities:
            x = dict(
                arm=arm,
                rl_seed=seed,
                base_path=str(artifact),
                base_sha256=m.sha(artifact),
                checkpoint_path=None if arm == "format_only" else str(artifact),
                checkpoint_sha256=None if arm == "format_only" else m.sha(artifact),
                dev_evaluation_path=str(dev),
                dev_config_sha256=m.sha(dev / "config.json"),
                inference_sha256="synthetic",
                native_cost=0.03 if arm == "cost" else 0.0,
            )
            if arm in ["cost", "no_cost"]:
                for stem in ["counter_hazards", "budget_baselines"]:
                    x[stem + "_path"] = str(fit)
                    x[stem + "_sha256"] = m.sha(fit)
            models.append(x)
        if case == "missing_primary_fit":
            del models[0]["counter_hazards_path"]
        live = dict(
            status="ready_for_root_seal",
            pending=[],
            registered_models=models,
            postprocessing_hashes={},
            statistical_analysis={},
            launch_helper_sha256="synthetic",
            main={"test_theorems": 2},
            historical={"test_theorems": 2},
        )
        write(t / "live.json", live)
        analysis = {
            "inventories": {
                x + "_test": {"inventory_sha256": "synthetic"} for x in ["main", "historical"]
            },
            "registered_analyses": [],
        }
        write(t / "analysis.json", analysis)
        selection = t / "selection"
        plans = {}
        for dataset in ["main", "historical"]:
            output = t / dataset
            data = t / (dataset + "_metadata.json")
            write(data, {"synthetic_public_only": True})
            plan = {
                "public_raw": str(output / "public.jsonl"),
                "public_transitions": str(output / "public_transitions.jsonl"),
                "private_stdout_stderr_redirect": str(t / (dataset + ".private.log")),
                "input_sha256": m.sha(data),
                "sealer_sha256": m.sha(W / "selection/seal_public_pool.py"),
                "argv": ["synthetic-collector", dataset],
                "renderer_argv": ["synthetic-renderer", dataset],
                "sealer_argv": ["synthetic-sealer", dataset],
            }
            plans[dataset] = plan
            write(
                selection / (dataset + "_entries.json"),
                {"registered_analyses": [], "collection": plan},
            )
        write(selection / "causal_entries.json", {"registered_analyses": []})
        write(
            selection / "manifest.json",
            {"registry_paths": {d: str(t / (d + "_final.json")) for d in plans}},
        )
        parent_path = t / "parent.json"
        args = SimpleNamespace(
            selection_inputs=str(selection),
            live_entries=str(t / "live.json"),
            analysis_entries=str(t / "analysis.json"),
            out=str(parent_path),
            utc="synthetic",
        )
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                m.create_parent(args)
            parent = m.read(parent_path)
            for _dataset, plan in plans.items():
                output = Path(plan["public_raw"]).parent
                output.mkdir()
                Path(plan["public_raw"]).write_text("synthetic public raw\n")
                Path(plan["public_transitions"]).write_text("synthetic public transitions\n")
                seal = output / "public_seal.json"
                write(
                    seal,
                    dict(
                        private_outcomes_read=False,
                        theorems=2,
                        attempts_per_theorem=8,
                        expected_data_sha256=plan["input_sha256"],
                        sealer_sha256=plan["sealer_sha256"],
                        raw_public_sha256=m.sha(plan["public_raw"]),
                        transition_public_sha256=m.sha(plan["public_transitions"]),
                    ),
                )
                launch = output.with_suffix(".collection_launch.json")
                write(
                    launch,
                    dict(
                        parent_sha256=m.sha(parent_path),
                        collection_plan=plan,
                        status="public_seal_complete_pending_registry",
                        private_outcomes_read=False,
                        public_seal_sha256=m.sha(seal),
                        launcher_sha256=m.sha(W / "analysis/run_frozen_collection.py"),
                        processes=[
                            dict(stage=k, command=plan[k], returncode=0)
                            for k in ["argv", "renderer_argv", "sealer_argv"]
                        ],
                    ),
                )
            launch = Path(plans["main"]["public_raw"]).parent.with_suffix(".collection_launch.json")
            ld = m.read(launch)
            if case == "changed_fit":
                fit.write_text("changed")
            elif case == "changed_command":
                ld["processes"][0]["command"] = ["other"]
                write(launch, ld)
            elif case == "private_read_true":
                ld["private_outcomes_read"] = True
                write(launch, ld)
            elif case == "changed_public":
                Path(plans["main"]["public_raw"]).write_text("changed")
            elif case == "changed_parent":
                parent["registries"]["main"]["unexpected_decision"] = "changed"
                write(parent_path, parent)
            elif case == "wrong_launcher":
                ld["launcher_sha256"] = "wrong"
                write(launch, ld)
            with contextlib.redirect_stdout(io.StringIO()):
                m.seal(SimpleNamespace(parent=str(parent_path), utc="synthetic_seal"))
        except ValueError as e:
            result["cases"][case] = {"rejected": True, "message": str(e)}
        else:
            final = m.read(t / "main_final.json")
            hist = m.read(t / "historical_final.json")
            allowed = {
                "status",
                "registered_public_pool_sha256",
                "frozen_decisions_parent_sha256",
                "public_seal_path",
                "public_seal_sha256",
                "collection_launch_sha256",
                "public_sealed_at_utc",
                "private_outcomes_inspected_before_seal",
            }
            assert all(
                final[k] == v for k, v in parent["registries"]["main"].items() if k not in allowed
            )
            assert hist["main_frozen_protocol_sha256"] == m.sha(t / "main_final.json")
            result["cases"][case] = {
                "rejected": False,
                "decision_fields_unchanged": True,
                "historical_main_chain_matches": True,
            }
assert not result["cases"]["clean"]["rejected"]
assert all(v["rejected"] for k, v in result["cases"].items() if k != "clean")
Path(__file__).with_name("final_protocol_independent_audit.json").write_text(
    json.dumps(result, indent=2) + "\n"
)
print(json.dumps(result, indent=2))
