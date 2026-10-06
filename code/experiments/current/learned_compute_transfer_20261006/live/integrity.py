"""Additional read-only integrity checks before unchanged transfer statistics."""

import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", required=True)
    parser.add_argument("--sidecar", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    registry_path = Path(args.registry)
    registry = json.loads(registry_path.read_text())
    sidecar = json.loads(Path(args.sidecar).read_text())
    if (
        sidecar["status"] != "frozen_additional_integrity_checks"
        or sidecar["registry_sha256"] != sha(registry_path)
        or sidecar["audit_sha256"] != sha(__file__)
    ):
        raise ValueError("Additional integrity audit is not bound")
    for mapping in ("parent_hashes", "implementation_hashes"):
        for path, expected in registry[mapping].items():
            if sha(path) != expected:
                raise ValueError("Registered implementation or parent changed")
    data = Path(registry["data_path"])
    if sha(data) != registry["test_data_sha256"]:
        raise ValueError("Registered transfer data changed")
    rows = [json.loads(line) for line in data.read_text().splitlines()]
    prompts = {row["theorem_id"]: row["prompt"] for row in rows}
    expected_keys = {
        (theorem_id, repeat) for theorem_id in prompts for repeat in range(registry["test_repeats"])
    }
    inventories = []
    for entry in registry["registered_models"]:
        folder = Path(registry["results_root"]) / entry["model_key"]
        launch_file = folder.with_suffix(".launch.json")
        launch = json.loads(launch_file.read_text())
        config = json.loads((folder / "config.json").read_text())
        dev = json.loads((Path(entry["dev_evaluation_path"]) / "config.json").read_text())
        if (
            launch["status"] != "complete"
            or launch["returncode"] != 0
            or launch["registry_sha256"] != sha(registry_path)
            or launch["launcher_sha256"] != registry["launcher_sha256"]
            or config["command"] != launch["command"][1:]
            or config["seed"] != registry["test_seed"]
            or config["repeats"] != registry["test_repeats"]
            or config["evaluated_theorems"] != registry["test_theorems"]
            or config["data_sha256"] != registry["test_data_sha256"]
            or config["base_sha256"] != entry["base_sha256"]
            or config["checkpoint_sha256"] != entry["checkpoint_sha256"]
            or config["cost"] != entry["native_cost"]
            or config["torch_version"] != dev["torch_version"]
            or config["frozen_protocol_sha256"] != sha(registry_path)
        ):
            raise ValueError("Draw/config/model/command mismatch")
        command = launch["command"]
        for flag, value in (
            ("--seed", str(registry["test_seed"])),
            ("--repeats", str(registry["test_repeats"])),
            ("--data", str(data)),
            ("--out", str(folder)),
            ("--frozen-protocol", str(registry_path.resolve())),
        ):
            if command.count(flag) != 1 or command[command.index(flag) + 1] != value:
                raise ValueError("Launched arguments differ from registered transfer request")
        for key in ("base", "checkpoint"):
            if entry.get(key + "_path") and sha(entry[key + "_path"]) != entry[key + "_sha256"]:
                raise ValueError("Model artifact changed")
        for name, expected in config["runner_hashes"].items():
            if sha(folder / "runner_snapshot" / name) != expected:
                raise ValueError("Generation snapshot changed")
        inventories.append((entry, folder, launch_file))
    # Every model's metadata passed before any private verdict file is opened.
    results = []
    for entry, folder, launch_file in inventories:
        counts = {}
        with (folder / "public_paths.jsonl").open() as stream:
            for line in stream:
                path = json.loads(line)
                key = path["theorem_id"], path["repeat"]
                count = path["attempts"]
                if (
                    key in counts
                    or key not in expected_keys
                    or type(count) is not int
                    or not 1 <= count <= 8
                    or len(path["candidates"]) != count
                    or len(path["decisions"]) != count
                    or path["prompt"] != prompts[path["theorem_id"]]
                ):
                    raise ValueError("Malformed public path identity/count")
                counts[key] = count
        private_keys = set()
        with (folder / "private_labels.jsonl").open() as stream:
            for line in stream:
                row = json.loads(line)
                key = row["theorem_id"], row["repeat"]
                labels = row["validities"]
                if (
                    key in private_keys
                    or key not in counts
                    or not isinstance(labels, list)
                    or len(labels) != counts[key]
                    or not labels
                    or any(type(value) is not bool for value in labels)
                ):
                    raise ValueError("Malformed private validity/count")
                private_keys.add(key)
        if set(counts) != expected_keys or private_keys != expected_keys:
            raise ValueError("Incomplete theorem/draw joins")
        results.append(
            {
                "model_key": entry["model_key"],
                "paths_checked": len(counts),
                "launch_sha256": sha(launch_file),
                "public_sha256": sha(folder / "public_paths.jsonl"),
                "labels_sha256": sha(folder / "private_labels.jsonl"),
            }
        )
    Path(args.out).write_text(
        json.dumps(
            {
                "status": "passed",
                "registry_sha256": sha(registry_path),
                "audit_sha256": sha(__file__),
                "sidecar_sha256": sha(args.sidecar),
                "checks": results,
                "scope": (
                    "Read-only config/draw/model/parent/source/public-private type/count checks. "
                    "No metric estimates, fitted policies or numerical statistics changed."
                ),
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
