"""Final CPU replay, gated by explicit root notification and completed closure."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
import time

import common
import learning as L


def closure_gate(plan, plan_path, closure_path, release_path, authorized):
    # Do not touch the final population or predictions before this gate.
    if not authorized:
        raise ValueError("Wait for root's explicit final-access notification")
    closure = json.loads(Path(closure_path).read_text())
    release = json.loads(Path(release_path).read_text())
    if (closure.get("no_further_training") is not True or closure.get("no_further_selector_changes") is not True
            or release.get("role") != "final_confirmation" or release.get("frozen") is not True
            or release.get("no_further_training") is not True or release.get("no_further_selector_changes") is not True):
        raise ValueError("Global training/selector closure and final release are required")
    if datetime.fromisoformat(release["utc"]) < datetime.fromisoformat(closure["utc"]):
        raise ValueError("Final release predates completed training closure")
    recipe_path = Path(release["recipe"])
    if common.sha(recipe_path) != release["recipe_sha256"]:
        raise ValueError("Final recipe changed after release")
    recipe = json.loads(recipe_path.read_text())
    if recipe["prospective_input_sha256"].get(str(Path(plan_path).resolve())) != common.sha(plan_path):
        raise ValueError("Final analysis is not prospectively bound by the release recipe")
    closure_plan_path = next((p for p, digest in recipe["prospective_input_sha256"].items()
                              if digest == closure["plan_sha256"]), None)
    if closure_plan_path is None or common.sha(closure_plan_path) != closure["plan_sha256"]:
        raise ValueError("Closure plan is not bound by the final recipe")
    closure_plan = json.loads(Path(closure_plan_path).read_text())
    if set(plan["models"]) != set(closure_plan["final_evaluation_models"]) or len(plan["models"]) != 32:
        raise ValueError("All 32 prospectively declared final models are required")
    if plan["data"] != recipe["data"] or plan["samples"] != 8:
        raise ValueError("Final data path or sample count differs from frozen recipe")
    for path, digest in closure["checkpoint_sha256"].items():
        if common.sha(path) != digest or release["checkpoint_paths"].get(path) != digest:
            raise ValueError("Final weights differ between closure, release and filesystem")
    for directory, bound in closure["training_outputs"].items():
        for filename, key in (("completed.json", "completed_sha256"), ("result.json", "result_sha256"),
                              ("manifest.json", "manifest_sha256")):
            if common.sha(Path(directory) / filename) != bound[key]:
                raise ValueError("A producer changed after training closure")
    return closure, release, closure_plan


def compare_root_aggregate(independent, path):
    root = json.loads(Path(path).read_text())
    for key in ("data_sha256", "population_sources", "population_contexts", "bootstrap_draws", "bootstrap_seed",
                "models", "receipts", "comparisons", "contrasts"):
        if root[key] != independent[key]:
            raise ValueError("Independent aggregate differs from root aggregate: " + key)
    return {"status": "exact_agreement", "path": str(Path(path).resolve()), "sha256": common.sha(path)}


def write_analysis_plan(path, plan, output):
    # The analyzer sums coefficient terms in insertion order. Preserve that
    # order so an independent run is bitwise comparable to the root run.
    Path(path).write_text(json.dumps({**plan, "output": str(output)}, indent=2) + "\n")


def run(args):
    started = time.time()
    plan_path = Path(args.plan).resolve()
    plan = json.loads(plan_path.read_text())
    closure, release, closure_plan = closure_gate(plan, plan_path, args.closure, args.release,
                                                 args.authorized_by_root_notification)
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Final audit output must be empty")
    data = Path(plan["data"])
    data_hash = common.sha(data)
    population = json.loads((data.parent / "manifest.json").read_text())
    if (population.get("status") != "released_complete_primary"
            or population["population_sha256"].get("final_confirmation") != data_hash
            or data_hash not in release["data_sha256"]):
        raise ValueError("Final population hash differs from root release")
    bindings = {}
    release_time = datetime.fromisoformat(release["utc"]).timestamp()
    for name, spec in plan["models"].items():
        directory = Path(spec["attempts"]).parent
        manifest = json.loads((directory / "manifest.json").read_text())
        completion = json.loads((directory / "completed.json").read_text())
        result = json.loads((directory / "result.json").read_text())
        actual = manifest["args"]
        checkpoint = closure_plan["final_evaluation_models"][name]
        if (actual["checkpoint"] != checkpoint or manifest["input_sha256"]["checkpoint"] != closure["checkpoint_sha256"][checkpoint]
                or manifest["input_sha256"]["data"] != data_hash or actual["data"] != str(data)
                or Path(actual["release"]).resolve() != Path(args.release).resolve()
                or manifest["started_unix"] < release_time):
            raise ValueError("Final prediction identity or release ordering differs")
        expected = {"command": "evaluate", "start": "full", "samples": 8, "temperature": .8,
                    "max_new_tokens": 512, "precision": "fp32", "seed": 20260910, "audit_likelihood": True}
        if any(actual.get(key) != value for key, value in expected.items()):
            raise ValueError("Final prediction violates the frozen sampling contract")
        if not completion["ok"] or completion["result_sha256"] != common.sha(directory / "result.json"):
            raise ValueError("Incomplete or changed final prediction artifact")
        for path, digest in {**manifest["code_sha256"], **manifest["dependency_sha256"]}.items():
            if common.sha(path) != digest:
                raise ValueError("Final producer code/dependency changed")
        if (result["n_theorems"] != 768 or result["samples"] != 8 or result["start"] != "full"
                or result["total_flop_proxy"] != L.total_flop_proxy(result["ledger"])
                or result["likelihood_audit"]["max_abs_error"] > .001):
            raise ValueError("Final counts, compute or behavior audit do not reconcile")
        bindings[name] = {"checkpoint_sha256": manifest["input_sha256"]["checkpoint"],
                          "manifest_sha256": common.sha(directory / "manifest.json"),
                          "result_sha256": completion["result_sha256"],
                          "attempts_sha256": common.sha(spec["attempts"]), "nll_sha256": common.sha(spec["nll"]),
                          "reported_covered_sources": result["strict_solved"],
                          "reported_mean_whole_proof_bits": result["mean_whole_proof_bits"]}
    output.mkdir(parents=True, exist_ok=True)
    write_analysis_plan(output / "analysis_plan.json", plan, output / "paired_analysis.json")
    subprocess.run([sys.executable, "-B", str(common.ROOT / "tools/reverify_attempts.py"),
                    str(output / "analysis_plan.json"), "--output", str(output / "strict_replay.json")], check=True)
    subprocess.run([sys.executable, "-B", str(common.ROOT / "tools/analyze_evaluations.py"),
                    str(output / "analysis_plan.json")], check=True)
    analysis = json.loads((output / "paired_analysis.json").read_text())
    replay = json.loads((output / "strict_replay.json").read_text())
    for name, binding in bindings.items():
        counts = analysis["receipts"][name]
        if (counts["covered_sources"] != binding["reported_covered_sources"]
                or counts["successful_attempts"] != replay["models"][name]["strict_successes"]
                or counts["all_attempts"] != 768 * 8 or replay["models"][name]["attempts"] != 768 * 8):
            raise ValueError("Final coverage/counts differ from raw-token replay")
        if abs(analysis["models"][name]["whole_proof_bits"]["balanced"]["mean"] - binding["reported_mean_whole_proof_bits"]) > 1e-9:
            raise ValueError("Final reference mean differs from independently aggregated values")
    comparison = compare_root_aggregate(analysis, args.root_analysis) if args.root_analysis else {"status": "pending_root_aggregate"}
    common.write_json(output / "receipt.json", {
        "status": "passed_replay_and_analysis", "models": 32, "sources": 768, "attempts_replayed": 196608,
        "closure_sha256": common.sha(args.closure), "release_sha256": common.sha(args.release),
        "analysis_plan_sha256": common.sha(plan_path), "final_data_sha256": data_hash,
        "evaluation_bindings": bindings, "root_aggregate_comparison": comparison,
        "strict_replay_sha256": common.sha(output / "strict_replay.json"),
        "paired_analysis_sha256": common.sha(output / "paired_analysis.json"),
        "elapsed_seconds": time.time() - started, "gpu_jobs_launched": 0, "model_weights_loaded": False,
        "final_payload_access_after_root_notification_and_closure": True,
        "limits": "Saved NLL/behavior likelihoods are bound and aggregated, not re-forwarded; context intervals do not estimate training-seed variability"})
    print(json.dumps({"status": "passed", "attempts": 196608, "root_comparison": comparison["status"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for flag in ("plan", "closure", "release", "output"):
        parser.add_argument("--" + flag, required=True)
    parser.add_argument("--authorized-by-root-notification", action="store_true")
    parser.add_argument("--root-analysis")
    run(parser.parse_args())
