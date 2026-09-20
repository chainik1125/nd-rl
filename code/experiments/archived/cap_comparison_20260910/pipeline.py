"""Run one frozen cap-comparison arm; release development only after full RL."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def local(root, path):
    path = Path(path)
    candidate = path if path.is_absolute() else root / path
    if not candidate.is_relative_to(root) or candidate.resolve() != candidate:
        raise ValueError(f"Nonlocal or aliased path: {path}")
    return candidate


def protected(path):
    path = Path(path)
    return (any("sealed" in p.lower() or p.lower() in ("test", "final") for p in path.parts)
            or "textbook_dev" in path.name.lower() or "textbook_final" in path.name.lower())


def check_frozen(root, protocol, protocol_path, protocol_sha, config_path, config_sha):
    if sha(protocol_path) != protocol_sha or sha(config_path) != config_sha:
        raise ValueError("Frozen protocol/config changed")
    for relative, digest in protocol["input_sha256"].items():
        path = local(root, relative)
        if protected(path):
            raise ValueError("Protected payload must not enter pretraining input hashes")
        if sha(path) != digest:
            raise ValueError(f"Frozen input changed: {relative}")


def checkpoint_metadata(path):
    # Only locally generated, input-bound checkpoints; no theorem data here.
    import torch
    return torch.load(path, map_location="cpu", weights_only=True)["metadata"]


def validate_training(stage, root, load_metadata=None):
    """Check actual completion, objective stage, parent and first budget crossing."""
    output = local(root, stage["output"])
    checkpoint = output / "final.pt"
    complete = read(output / "completed.json")
    manifest = read(output / "manifest.json")
    metadata = (load_metadata or checkpoint_metadata)(checkpoint)
    args = manifest["args"]
    command = stage["command_name"]
    expected = int(stage["target_flop_proxy"])
    if args.get("command") != command or metadata.get("stage") != command:
        raise ValueError("Completion is not the requested actual training stage")
    if complete.get("stage") != command:
        raise ValueError("Completion marker training stage differs")
    if int(args.get("max_flop_proxy", 0)) != expected:
        raise ValueError("Training argument budget differs")
    for key, value in stage["expected_args"].items():
        if args.get(key) != value or metadata.get("args", {}).get(key) != value:
            raise ValueError(f"Training argument identity differs: {key}")
    source = sha(local(root, stage["checkpoint"]))
    if manifest.get("source_sha256") != source or metadata.get("source_checkpoint_sha256") != source:
        raise ValueError("Training source checkpoint differs")
    digest = sha(checkpoint)
    if complete.get("final_sha256") != digest:
        raise ValueError("Completed checkpoint hash differs")
    actual = int(complete.get("actual_flop_proxy", 0))
    target = int(complete.get("target_flop_proxy", 0))
    last_update = int(complete.get("last_step_flop_proxy", 0))
    if (target != expected or actual < expected or last_update <= 0
            or actual - last_update >= expected or complete.get("budget_reached") is not True):
        raise ValueError("Full first-crossing compute budget not reached")
    if complete.get("stop_reason") != "flop_budget":
        raise ValueError("Training did not stop on its compute budget")
    if int(complete.get("actual_steps", 0)) < 1:
        raise ValueError("No actual training updates")
    training_compute = metadata.get("training_compute", {})
    if (training_compute.get("total_flop_proxy") != actual
            or training_compute.get("target_flop_proxy") != target
            or training_compute.get("budget_reached") is not True
            or complete.get("unreached_flop_milestones") != []):
        raise ValueError("Final checkpoint compute or milestone completion differs")
    milestones = complete.get("compute_milestones", [])
    if sorted(m["requested_flop_proxy"] for m in milestones) != sorted(stage["milestones"]):
        raise ValueError("Requested milestone set was not completed exactly")
    for milestone in milestones:
        path = local(root, milestone["checkpoint"])
        if path.parent != output or sha(path) != milestone["checkpoint_sha256"]:
            raise ValueError("Milestone checkpoint binding differs")
        if not (milestone["requested_flop_proxy"] <= milestone["actual_flop_proxy"] <= actual
                and 1 <= milestone["step"] <= complete["actual_steps"]):
            raise ValueError("Milestone compute/step is outside its completed trajectory")
    return {"stage": command, "checkpoint": str(checkpoint), "checkpoint_sha256": digest,
            "completed_sha256": sha(output / "completed.json"),
            "manifest_sha256": sha(output / "manifest.json"),
            "target_flop_proxy": target, "actual_flop_proxy": actual,
            "last_update_flop_proxy": last_update, "actual_steps": int(complete["actual_steps"]),
            "milestones": milestones}


def create_development_release(spec, root, training, config_sha, protocol_sha):
    # This function must not hash/open development payload bytes. The runner will
    # compare the declared known hash only after this completed-RL release exists.
    if set(training) != {"sft", "rl"}:
        raise ValueError("Both actual SFT and full RL completion are required")
    release_path = local(root, spec["development_release"])
    if release_path.exists():
        raise ValueError("Development release already exists")
    data = spec["development_data"]
    if local(root, data["path"]) != root / "evaluation/textbook_dev.jsonl":
        raise ValueError("Unexpected development path")
    release = {"schema_version": 1, "role": "textbook_dev", "adaptive_development": True,
               "arm": spec["name"], "created_utc": now(),
               "config_sha256": config_sha, "protocol_sha256": protocol_sha,
               "checkpoints": [training[k]["checkpoint_sha256"] for k in ("sft", "rl")],
               "data_sha256": [data["sha256"]], "data_path": str(root / data["path"]),
               "completed_rl_checkpoint": training["rl"]["checkpoint"],
               "completed_rl_checkpoint_sha256": training["rl"]["checkpoint_sha256"],
               "completed_rl_sha256": training["rl"]["completed_sha256"],
               "training": training, "full_rl_budget_reached": True,
               "no_textbook_results_used_to_select_endpoint": True}
    write(release_path, release)
    return release_path


def eval_command(spec, root, phase, checkpoint, panel, release=None):
    settings = spec["evaluation"]
    if panel in ("pretraining", "validation_synthetic"):
        data = spec["data"][panel]
        samples, temperature = 1, 0
    elif panel in ("dev_greedy", "dev_sampled"):
        if release is None:
            raise ValueError("No development release")
        data = spec["development_data"]["path"]
        samples, temperature = (1, 0) if panel == "dev_greedy" else (16, .8)
    else:
        raise ValueError("Unknown evaluation panel")
    output = local(root, spec["run_base"]) / "evaluations" / f"{phase}_{panel}"
    command = [spec["python"], str(root / spec["runner"]), "eval",
               "--checkpoint", str(checkpoint), "--output", str(output),
               "--data", str(root / data), "--mode", spec["mode"],
               "--seed", str(settings["seed"]), "--max-proof-lines", str(spec["cap"]),
               "--scratch-style", "dependency", "--samples", str(samples),
               "--temperature", str(temperature), "--max-new-tokens", str(settings["max_new_tokens"]),
               "--max-seq-len", str(settings["max_seq_len"]),
               "--decode-batch", str(settings["decode_batch"]), "--precision", "fp32", "--device", "cuda"]
    if release is not None:
        command += ["--evaluation-release-json", str(release)]
    return command, output


def run(config_path):
    config_path = Path(config_path).resolve()
    root = config_path.parent.parent
    spec = read(config_path)
    if spec["root"] != str(root) or local(root, spec["run_base"]) != root / "runs" / spec["name"]:
        raise ValueError("Root/arm output identity differs")
    protocol_path = root / "plans/protocol.json"
    protocol = read(protocol_path)
    protocol_sha, config_sha = sha(protocol_path), sha(config_path)
    if spec["name"] not in protocol["arms"]:
        raise ValueError("Unregistered arm")
    relative = str(config_path.relative_to(root))
    if protocol["input_sha256"].get(relative) != config_sha:
        raise ValueError("Config absent from frozen input map")
    check_frozen(root, protocol, protocol_path, protocol_sha, config_path, config_sha)
    base = local(root, spec["run_base"])
    if base.exists() and any(base.iterdir()):
        raise ValueError("Arm output must be fresh; failed jobs are not silently resumed")
    base.mkdir(parents=True, exist_ok=True)
    progress = {"schema_version": 1, "arm": spec["name"], "label": spec["label"],
                "status": "running", "started_utc": now(), "updated_utc": now(),
                "config_sha256": config_sha, "protocol_sha256": protocol_sha,
                "completed_stages": [], "training": {}}
    started = time.monotonic()
    env = {**os.environ, "PYTHONPATH": str(root / "snapshots/main_v1/vendor")}

    def execute(name, command):
        check_frozen(root, protocol, protocol_path, protocol_sha, config_path, config_sha)
        if command[:2] != [spec["python"], str(root / spec["runner"])]:
            raise ValueError("Unexpected executable/source")
        progress.update(current_stage=name, updated_utc=now())
        write(base / "progress.json", progress)
        print(json.dumps({"event": "stage_start", "arm": spec["name"], "stage": name, "command": command}), flush=True)
        subprocess.run(command, cwd=root, env=env, check=True)
        print(json.dumps({"event": "stage_complete", "arm": spec["name"], "stage": name}), flush=True)

    def evaluate(phase, checkpoint, panel, release=None, position=None):
        command, output = eval_command(spec, root, phase, checkpoint, panel, release)
        name = phase + "_" + panel
        execute(name, command)
        complete = read(output / "completed.json")
        if complete.get("stage") != "eval" or complete.get("checkpoint_sha256") != sha(checkpoint):
            raise ValueError("Evaluation completion identity differs")
        if position is not None:
            write(output / "checkpoint_position.json", {"checkpoint_sha256": sha(checkpoint), **position})
        progress["completed_stages"].append({"name": name, "completed_sha256": sha(output / "completed.json"),
                                             "summary_sha256": sha(output / "summary.json"),
                                             "attempts_sha256": sha(output / "attempts.jsonl")})
        progress["updated_utc"] = now()
        write(base / "progress.json", progress)

    try:
        for stage in spec["training_stages"]:
            execute(stage["command_name"], stage["command"])
            validated = validate_training(stage, root)
            phase = stage["command_name"]
            progress["training"][phase] = validated
            progress["completed_stages"].append({"name": phase, **validated})
            position = ({"step": 0, "total_flop_proxy": 0} if phase == "sft" else
                        {"step": validated["actual_steps"], "total_flop_proxy": validated["actual_flop_proxy"]})
            for panel in ("pretraining", "validation_synthetic"):
                evaluate(phase, Path(validated["checkpoint"]), panel, position=position)
            if phase == "rl":
                for index, milestone in enumerate(validated["milestones"]):
                    checkpoint = local(root, milestone["checkpoint"])
                    if sha(checkpoint) != milestone["checkpoint_sha256"]:
                        raise ValueError("Milestone checkpoint hash differs")
                    if milestone["requested_flop_proxy"] >= validated["target_flop_proxy"]:
                        continue
                    for panel in ("pretraining", "validation_synthetic"):
                        evaluate(f"rl_m{index:02d}", checkpoint, panel,
                                 position={"step": milestone["step"], "total_flop_proxy": milestone["actual_flop_proxy"]})
        # Recheck the actual source/checkpoint/completion immediately before release.
        for stage in spec["training_stages"]:
            if validate_training(stage, root) != progress["training"][stage["command_name"]]:
                raise ValueError("Completed training changed before release")
        release = create_development_release(spec, root, progress["training"], config_sha, protocol_sha)
        for phase in ("sft", "rl"):
            for panel in ("dev_greedy", "dev_sampled"):
                trained = progress["training"][phase]
                position = {"step": 0, "total_flop_proxy": 0} if phase == "sft" else {
                    "step": trained["actual_steps"], "total_flop_proxy": trained["actual_flop_proxy"]}
                evaluate(phase, Path(trained["checkpoint"]), panel, release, position)
        check_frozen(root, protocol, protocol_path, protocol_sha, config_path, config_sha)
        progress.update(status="complete", current_stage=None, updated_utc=now(), elapsed_seconds=time.monotonic() - started)
        write(base / "progress.json", progress)
        write(base / "pipeline_completed.json", {**progress, "development_release_sha256": sha(release),
                                                "progress_sha256": sha(base / "progress.json")})
    except BaseException as error:
        progress.update(status="failed", updated_utc=now(), error=str(error), elapsed_seconds=time.monotonic() - started)
        write(base / "progress.json", progress)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
