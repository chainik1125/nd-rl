"""Prospectively fix all CPU final profiles without evaluating final prompts."""
from pathlib import Path
import json
import shutil
import subprocess
import common

ROOT = common.ROOT
PYTHON = "/data/users/dmitry/simplex-research/.venv/bin/python"


def main():
    snap = ROOT / "snapshots/final_comparators_v1"
    snap.mkdir(exist_ok=False)
    for name in ("common.py", "final_comparators.py", "test_final_comparators.py", "prepare_final_comparators.py"):
        shutil.copy2(ROOT / "code" / name, snap / name)
    tasks, required, hashes = [], set(), {str(p): common.sha(p) for p in snap.glob("*.py")}
    for prefix in ("symbolic", "minimal"):
        original = ROOT / "configs" / f"{prefix}_development_tasks_v1.json"
        hashes[str(original)] = common.sha(original)
        spec = json.loads(original.read_text())
        for old in spec["tasks"]:
            command = list(old["command"])
            data = Path(command[command.index("--population") + 1]).with_name("final.jsonl")
            command[command.index("--population") + 1] = str(data)
            name = old["name"].replace("_development_", "_final_")
            output = ROOT / "results" / name
            command[command.index("--out") + 1] = str(output)
            command[command.index("--max-seconds") + 1] = "1800"
            task = {"kind": "symbolic", "name": name, "command": command,
                    "endpoint_paths": old["endpoint_paths"], "timeout_seconds": 1830,
                    "result_path": str(output / "symbolic_general_eval_summary.json")}
            tasks.append(task)
            for cp in task["endpoint_paths"]:
                required.add(str(Path(cp).parent / "symbolic_general_train_summary.json"))
            hashes[str(data)] = common.sha(data)
            audit = data.parent / "audit/final.jsonl"
            hashes[str(audit)] = common.sha(audit)
            for path, digest in old["input_sha256"].items():
                if not (Path(path).name == "development.jsonl"):
                    hashes[path] = digest
    for label, population in (("primary", "v1"), ("structured", "structured_v1")):
        name = f"classical_{label}_final_v1"
        data = ROOT / "data" / population / "final.jsonl"
        protocol = ROOT / "configs" / (name + ".json")
        program = ROOT / "snapshots/classical_v1/classical_run.py"
        subprocess.run([PYTHON, str(program), "--data", str(data), "--protocol", str(protocol), "--prepare-protocol"], check=True)
        output = ROOT / "results" / name
        tasks.append({"kind": "classical", "name": name,
            "command": [PYTHON, str(program), "--data", str(data), "--protocol", str(protocol), "--output", str(output), "--workers", "4"],
            "classical_protocol": str(protocol), "endpoint_paths": [], "timeout_seconds": 600,
            "result_path": str(output / "result.json")})
        hashes[str(protocol)] = common.sha(protocol)
        hashes[str(data)] = common.sha(data)
        for path in program.parent.rglob("*.py"):
            hashes[str(path)] = common.sha(path)
    legal_points = {"baseline": ROOT / "models/adapt_final.pt"}
    for condition in ("relative_full", "relative_reset", "ppo_full", "ppo_reset"):
        name = f"main_{condition}_seed1_v1"
        legal_points[name] = ROOT / "results" / name / "final.pt"
    for label, population, tokens in (("primary", "v1", 512), ("structured", "structured_lm_v1", 1024)):
        data = ROOT / "data" / population / "final.jsonl"
        profile = {"max_new_tokens": tokens, "max_lines": 24, "max_scope": 4, "batch_size": 16,
                   "attempts": 1, "temperature": 0, "seed": 202609101935}
        for endpoint, checkpoint in legal_points.items():
            name = f"legal_lm_{endpoint}_{label}_final_greedy_v1"
            output = ROOT / "results" / name
            command = [PYTHON, str(ROOT / "snapshots/legal_lm_v1/legal_lm.py"), "--checkpoint", str(checkpoint),
                       "--data", str(data), "--output", str(output), "--device", "cpu"]
            for key, value in profile.items():
                command.extend(["--" + key.replace("_", "-"), str(value)])
            tasks.append({"kind": "legal_lm", "name": name, "command": command, "data": str(data),
                          "profile": profile, "endpoint_paths": [str(checkpoint)], "timeout_seconds": 1800,
                          "result_path": str(output / "result.json")})
        hashes[str(data)] = common.sha(data)
    for folder in (ROOT / "vendor", ROOT / "snapshots/legal_lm_v1"):
        for path in folder.rglob("*.py"):
            hashes[str(path)] = common.sha(path)
    spec = {"status": "frozen_before_final_outcomes", "output": str(ROOT / "jobs/final_comparators_v1"),
            "deadline_utc": "2026-09-11T04:20:00Z", "parallel_tasks": 3,
            "lm_endpoints_release": str(ROOT / "configs/all_lm_endpoints_v1.json"),
            "required_training_summaries": sorted(required), "input_sha256": hashes, "tasks": tasks,
            "selection": "All general and minimal symbolic initial/PPO/BC/uniform policies, two nonlearned classical populations, and legal-line LM baseline plus every seed-1 primary condition. No performance selection.",
            "interpretation": "Legal-action interfaces and constructed certificates are separate from unaided LM generation. Symbolic policies use action caps, with exact LM-codec token eligibility reported separately. Minimal study reports all targets and prespecified P-versus-other target atoms."}
    path = ROOT / "configs/final_comparators_v1.json"
    common.write_json(path, spec)
    print(json.dumps({"config": str(path), "sha256": common.sha(path), "tasks": len(tasks), "training_summaries": len(required)}))


if __name__ == "__main__":
    main()
