"""Gated CPU replay/analysis of the complete declared development endpoints."""
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time

import common
import learning as L


def preflight(plan):
    data = Path(plan["data"])
    metadata = json.loads((data.parent / "manifest.json").read_text())
    data_hash = common.sha(data)
    if metadata.get("status") != "released_complete_primary" or metadata["population_sha256"].get("confirmation") != data_hash:
        raise ValueError("This audit accepts calibration confirmation only; no final payload")
    if plan.get("samples") != 8:
        raise ValueError("Declared main development evaluation requires eight samples")
    bindings = {}
    for name, spec in plan["models"].items():
        directory = Path(spec["attempts"]).parent
        manifest = json.loads((directory / "manifest.json").read_text())
        args = manifest["args"]
        result_path = directory / "result.json"
        completion = json.loads((directory / "completed.json").read_text())
        result = json.loads(result_path.read_text())
        if not completion["ok"] or completion["result_sha256"] != common.sha(result_path):
            raise ValueError("Incomplete or changed evaluation result")
        if manifest["input_sha256"]["data"] != data_hash or Path(args["data"]).resolve() != data.resolve():
            raise ValueError("Evaluation did not use the declared development population")
        checkpoint_hash = common.sha(args["checkpoint"])
        if checkpoint_hash != manifest["input_sha256"]["checkpoint"]:
            raise ValueError("Evaluation checkpoint changed")
        release = json.loads(Path(args["release"]).read_text())
        if (not release.get("frozen") or not release.get("no_further_selector_changes")
                or data_hash not in release["data_sha256"] or checkpoint_hash not in release["checkpoints"]):
            raise ValueError("Evaluation is not bound by its frozen development release")
        expected = {"command": "evaluate", "start": "full", "samples": 8, "temperature": .8,
                    "max_new_tokens": 512, "precision": "fp32", "seed": 20260910, "audit_likelihood": True}
        if any(args.get(key) != value for key, value in expected.items()):
            raise ValueError("Evaluation differs from the declared unaided sampling protocol")
        for path, digest in {**manifest["code_sha256"], **manifest["dependency_sha256"]}.items():
            if common.sha(path) != digest:
                raise ValueError("Evaluation code/dependency changed")
        if (result["n_theorems"] != 384 or result["samples"] != 8 or result["start"] != "full"
                or result["total_flop_proxy"] != L.total_flop_proxy(result["ledger"])
                or result["likelihood_audit"]["max_abs_error"] > .001):
            raise ValueError("Evaluation counts, compute or behavior audit do not reconcile")
        bindings[name] = {"directory": str(directory), "checkpoint_sha256": checkpoint_hash,
                          "manifest_sha256": common.sha(directory / "manifest.json"),
                          "result_sha256": common.sha(result_path), "release_sha256": common.sha(args["release"]),
                          "attempts_sha256": common.sha(spec["attempts"]), "nll_sha256": common.sha(spec["nll"]),
                          "reported_covered_sources": result["strict_solved"],
                          "reported_mean_whole_proof_bits": result["mean_whole_proof_bits"],
                          "total_flop_proxy": result["total_flop_proxy"]}
    return data_hash, bindings


def run(args):
    started = time.time()
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Independent audit output must be empty")
    original_plan = Path(args.plan).resolve()
    plan = json.loads(original_plan.read_text())
    data_hash, bindings = preflight(plan)
    # No development proof payload was parsed before all endpoint/release gates.
    output.mkdir(parents=True, exist_ok=True)
    local_plan = {**plan, "output": str(output / "paired_analysis.json")}
    common.write_json(output / "analysis_plan.json", local_plan)
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
                or counts["all_attempts"] != 384 * 8 or replay["models"][name]["attempts"] != 384 * 8):
            raise ValueError("Independently replayed coverage or successful-attempt count differs")
        bits = analysis["models"][name]["whole_proof_bits"]["balanced"]["mean"]
        if abs(bits - binding["reported_mean_whole_proof_bits"]) > 1e-9:
            raise ValueError("Recomputed complete-reference mean differs from evaluation result")
    receipt = {"status": "passed", "data_role": "confirmation", "data_sha256": data_hash,
               "original_plan_sha256": common.sha(original_plan), "evaluation_bindings": bindings,
               "strict_replay_sha256": common.sha(output / "strict_replay.json"),
               "paired_analysis_sha256": common.sha(output / "paired_analysis.json"),
               "models": len(bindings), "attempts_replayed": len(bindings) * 384 * 8,
               "elapsed_seconds": time.time() - started, "model_weights_loaded": False,
               "final_data_accessed": False, "gpu_jobs_launched": 0,
               "limitations": ["NLL values and saved behavior audits are bound and aggregated, not recomputed without model forwards",
                               "Paired context bootstrap is descriptive at fixed training and decoding seeds; no seed-level inference",
                               "Confirmation was used in calibration and is development data, not a new untouched final estimate"]}
    common.write_json(output / "receipt.json", receipt)
    print(json.dumps({"status": "passed", "models": len(bindings), "output": str(output)}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True)
    parser.add_argument("--output", required=True)
    run(parser.parse_args())
