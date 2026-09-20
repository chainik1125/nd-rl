"""Publish completed probes and training logs to the existing stock dashboard."""
import argparse
import json
from pathlib import Path
import sys
import time

OBSERVATORY = Path("/data/users/dmitry/nd-rl/observability")
sys.path.insert(0, str(OBSERVATORY))
from monitoring.__main__ import Collector
from monitoring.labels import set_label
from monitoring.probes import file_sha, import_released, publish_accuracy
from types import SimpleNamespace


def collect(root, project="cap-comparison"):
    protocol = json.loads((root / "plans/protocol.json").read_text())
    state_path = root / "reports/recovery_state.json"
    finished = json.loads(state_path.read_text())["finished"] if state_path.exists() else {}
    for arm in protocol["arms"]:
        base = root / "runs" / arm
        rl = base / "rl"
        if not all((rl / name).exists() for name in ("manifest.json", "events.jsonl", "rollouts.jsonl")):
            continue
        collector = Collector(rl, OBSERVATORY / ".observatory", project, every=128)
        mode = collector.args["mode"]
        cap = collector.args["max_proof_lines"]
        label = f"{'Direct' if mode == 'direct' else 'CoT'} cap {cap} (8.9M)"
        if collector.name != label:
            set_label(collector, label, f"Initial supervised proof cap {cap}; common compute budget; seed20260910.")
        collector.poll()
        for directory in sorted((base / "evaluations").glob("*")):
            if not all((directory / filename).exists() for filename in
                       ("completed.json", "manifest.json", "summary.json", "attempts.jsonl", "checkpoint_position.json")):
                continue
            manifest = json.loads((directory / "manifest.json").read_text())
            source = manifest["args"]
            if source.get("samples") != 1 or source.get("temperature") != 0:
                continue
            data = Path(source["data"])
            if data.name == "textbook_dev.jsonl":
                if finished.get(arm, {}).get("success") is not True:
                    continue
                import_released(SimpleNamespace(evaluations=[directory]), collector)
                continue
            split = ("pretraining" if "pretraining" in directory.name else
                     "validation_synthetic" if "synthetic" in directory.name else None)
            if split is None:
                continue
            checkpoint = Path(source["checkpoint"])
            if manifest["source_sha256"] == collector.manifest["source_sha256"]:
                step = 0
                amount = 0
            else:
                # Per-evaluation manifest records the exact evaluated RL step.
                info_path = directory / "checkpoint_position.json"
                if not info_path.exists():
                    continue
                info = json.loads(info_path.read_text())
                if info["checkpoint_sha256"] != manifest["source_sha256"]:
                    raise ValueError("Probe checkpoint position differs from evaluated weights")
                step = info["step"]
                amount = info["total_flop_proxy"]
            summary = json.loads((directory / "summary.json").read_text())
            dataset = {"sha256": manifest["data_sha256"], "problems": summary["problems"]}
            evaluation_protocol = {"max_new_tokens": source["max_new_tokens"], "samples": 1,
                                   "temperature": 0., "precision": source["precision"],
                                   "seed": source["seed"], "decode_batch": source["decode_batch"],
                                   "runner_sha256": file_sha(root / "snapshots/main_v1/run.py")}
            publish_accuracy(collector, step, split, directory, dataset,
                             manifest["source_sha256"], evaluation_protocol,
                             axes={"compute/rl_flop_proxy": amount,
                                   "compute/rl_budget_fraction": amount / collector.args["max_flop_proxy"]})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--project", default="cap-comparison")
    args = parser.parse_args()
    while True:
        collect(args.root.resolve(), args.project)
        if not args.watch:
            break
        time.sleep(60)


if __name__ == "__main__":
    main()
