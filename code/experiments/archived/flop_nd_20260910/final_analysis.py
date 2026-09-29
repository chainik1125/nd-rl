"""Audit and analyze all released LM endpoints without performance selection."""
from datetime import datetime, timezone
import argparse
import json
import os
from pathlib import Path
import time
import common
import analyze_lm
import audit_lm_evaluation
import plot_results

CONDITIONS = ("relative_full", "relative_reset", "ppo_full", "ppo_reset")


def contrasts(points):
    result = {name: {} for name in ("reset_with_relative", "reset_with_ppo", "ppo_with_full", "ppo_with_reset", "interaction")}
    for seed in (1, 2):
        names = {c: f"main_{c}_seed{seed}_v1" for c in CONDITIONS}
        for label, positive, negative in (("reset_with_relative", "relative_reset", "relative_full"),
                ("reset_with_ppo", "ppo_reset", "ppo_full"), ("ppo_with_full", "ppo_full", "relative_full"),
                ("ppo_with_reset", "ppo_reset", "relative_reset")):
            result[label][f"seed{seed}"] = {names[positive]: 1, names[negative]: -1}
        result["interaction"][f"seed{seed}"] = {names["ppo_reset"]: 1, names["ppo_full"]: -1,
                                                  names["relative_reset"]: -1, names["relative_full"]: 1}
        for condition in CONDITIONS:
            result.setdefault(condition + "_vs_initialization", {})[f"seed{seed}"] = {names[condition]: 1, "baseline": -1}
    mapping_path = common.ROOT / "configs/retention_endpoint_mapping_v1.json"
    if mapping_path.exists():
        mapping = json.loads(mapping_path.read_text())["mapping"]
        for condition in CONDITIONS:
            pairs = [(f"main_{condition}_seed{s}_v1", f"seed{s}") for s in (1, 2)]
            if all(mapping.get(name) in points for name, _ in pairs):
                result[condition + "_vs_retention"] = {seed: {name: 1, mapping[name]: -1} for name, seed in pairs}
    if all(f"online_{mode}_seed{seed}_v1" in points for mode in ("on", "off") for seed in (1, 2)):
        result["online_admission_effect"] = {f"seed{s}": {f"online_on_seed{s}_v1": 1, f"online_off_seed{s}_v1": -1} for s in (1, 2)}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads(args.config.read_text())
    spec_hash = common.sha(args.config)
    out = Path(spec["output"])
    out.mkdir(parents=True, exist_ok=False)
    deadline = datetime.fromisoformat(spec["deadline_utc"].replace("Z", "+00:00")).timestamp()
    state = {"status": "waiting_for_endpoint_release", "pid": os.getpid(), "config_sha256": spec_hash,
             "started_utc": datetime.now(timezone.utc).isoformat(), "audits": {}, "analyses": {}}
    common.write_json(out / "progress.json", state)
    release_path = Path(spec["endpoint_release"])
    try:
        while not release_path.exists():
            if time.time() >= deadline:
                raise TimeoutError("Endpoint release was not ready")
            time.sleep(10)
        release = json.loads(release_path.read_text())
        if release.get("status") != "frozen":
            raise ValueError("Endpoint identities are not frozen")
        release_hash = common.sha(release_path)
        points = release["checkpoint_paths"]
        expected = {"baseline", *[f"main_{c}_seed{s}_v1" for c in CONDITIONS for s in (1, 2)]}
        if not expected <= set(points):
            raise ValueError("Missing predeclared primary endpoint")
        comparisons = contrasts(points)
        jobs = {}
        for population, tag in (("v1", "_final_"), ("structured_lm_v1", "_structured_final_")):
            for profile in ("greedy", "sample4"):
                label = ("primary" if population == "v1" else "structured") + "_final_" + profile
                evaluations = {name: str(common.ROOT / "results" / (name + tag + profile)) for name in points}
                config = {"audit": str(common.ROOT / "data" / population / "audit/final.jsonl"), "evaluations": evaluations,
                          "contrasts": comparisons, "bootstrap_replicates": spec["bootstrap_replicates"],
                          "bootstrap_seed": spec["bootstrap_seed"], "endpoint_release_sha256": release_hash}
                config_path = common.ROOT / "configs" / ("analysis_" + label + "_v1.json")
                common.write_json(config_path, config)
                jobs[label] = {"config": config, "config_path": config_path,
                               "output": common.ROOT / "reports" / ("analysis_" + label + "_v1.json")}
        state.update(status="auditing_final_outputs", endpoint_release_sha256=release_hash)
        common.write_json(out / "progress.json", state)
        while len(state["analyses"]) < len(jobs):
            if time.time() >= deadline:
                raise TimeoutError("Final analysis deadline reached")
            if common.sha(args.config) != spec_hash or common.sha(release_path) != release_hash:
                raise ValueError("Analysis protocol or endpoint release changed")
            for path, digest in spec["input_sha256"].items():
                if common.sha(path) != digest:
                    raise ValueError(f"Frozen analysis program changed: {path}")
            for label, job in jobs.items():
                if label in state["analyses"]:
                    continue
                for name, directory in job["config"]["evaluations"].items():
                    key = Path(directory).name
                    if key in state["audits"] or not (Path(directory) / "result.json").exists():
                        continue
                    if time.time() >= deadline:
                        raise TimeoutError("Final raw audit deadline reached")
                    evaluation_config = json.loads((Path(directory) / "config.json").read_text())
                    if (evaluation_config["args"]["checkpoint"] != points[name] or
                            evaluation_config["input_sha256"].get(points[name]) != release["checkpoint_sha256"][name]):
                        raise ValueError("Final result was assigned to the wrong released endpoint")
                    report = audit_lm_evaluation.audit(directory)
                    report_path = common.ROOT / "reports" / ("audit_lm_" + key + ".json")
                    common.write_json(report_path, report)
                    state["audits"][key] = {"status": report["status"], "output": str(report_path), "sha256": common.sha(report_path)}
                    common.write_json(out / "progress.json", state)
                    print(json.dumps({"audit": key, "status": report["status"], "counts": report["counts"]}), flush=True)
                if not all(Path(p).name in state["audits"] for p in job["config"]["evaluations"].values()):
                    continue
                report = analyze_lm.evaluation_analysis(job["config"])
                report["config_sha256"] = common.sha(job["config_path"])
                common.write_json(job["output"], report)
                state["analyses"][label] = {"status": "completed", "output": str(job["output"]), "sha256": common.sha(job["output"])}
                common.write_json(out / "progress.json", state)
                print(json.dumps({"analysis": label, "status": "completed"}), flush=True)
                if label == "primary_final_greedy":
                    plot_results.style()
                    plot_results.endpoints(job["output"], common.ROOT / "figures/lm_primary_final_endpoints")
                    plot_results.effects(job["output"], common.ROOT / "figures/lm_primary_final_effects")
            if len(state["analyses"]) < len(jobs):
                time.sleep(10)
        plot_results.style()
        plot_results.curves(common.ROOT / "figures/lm_development_curves")
        state["status"] = "completed"
    except BaseException as error:
        state.update(status="failed", error=type(error).__name__, message=str(error))
        raise
    finally:
        state["finished_utc"] = datetime.now(timezone.utc).isoformat()
        common.write_json(out / "result.json", state)
        common.write_json(out / "progress.json", state)


if __name__ == "__main__":
    main()
