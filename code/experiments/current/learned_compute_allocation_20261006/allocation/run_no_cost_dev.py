"""Coordinate the already-prespecified no-cost-arm development evaluations."""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

WORK = Path(__file__).resolve().parents[1]
ROOT = WORK.parents[3]
ART = ROOT / "artifacts/learned_compute_allocation_20261006"
ALLOC = WORK / "allocation"
sys.path.insert(0, str(ALLOC))
from evaluation_protocol import fingerprint, signature  # noqa: E402


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def preflight():
    """Check the complete output plan before writing snapshots or starting jobs."""
    allowed = (ART / "allocation/no_cost").resolve()
    plan = {"ledger": allowed / "dev_launches.jsonl", "runs": {}}
    paths = [plan["ledger"], plan["ledger"].with_suffix(".source.py")]
    for seed in [103]:
        run_plan = {
            "run": ART / f"allocation/no_cost/main_s{seed}",
            "selection": ART / f"allocation/no_cost/main_s{seed}_selection.json",
            "outputs": {},
            "baselines": {},
        }
        plan["runs"][seed] = run_plan
        paths.extend([run_plan["run"], run_plan["selection"]])
        for step in [100, 200]:
            output = ART / f"allocation/no_cost/main_s{seed}_dev{step}_paths_s52006"
            baseline = (
                ART / "allocation/no_cost" / f"main_s{seed}_dev{step}_baselines_frozen_s52006"
            )
            run_plan["outputs"][step] = output
            run_plan["baselines"][step] = baseline
            paths.extend([output, output.with_suffix(".log"), baseline])
            paths.extend(
                baseline / name
                for name in (
                    "counter.json",
                    "budgets.json",
                    "fit_counter_hazards.py",
                    "fit_budget_baselines.py",
                )
            )
    if any(not path.resolve().is_relative_to(allowed) for path in paths):
        raise ValueError("No-cost output plan escapes the owned no-cost artifact directory")
    return plan


def main():
    plan = preflight()
    ledger = plan["ledger"]
    shutil.copy2(__file__, ledger.with_suffix(".source.py"))
    training_config = json.loads((ART / "allocation/no_cost/main_s103/config.json").read_text())
    while Path(f"/proc/{training_config['pid']}").exists():
        time.sleep(2)
    occupancy = subprocess.check_output(
        ["nvidia-smi", "--id=2", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        text=True,
    ).strip()
    if occupancy:
        raise RuntimeError(f"GPU2 has an unexpected active process after training: {occupancy}")
    for seed, run_plan in plan["runs"].items():
        run = run_plan["run"]
        metrics = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
        if len(metrics) != 200:
            raise ValueError("Training did not finish the prescribed200 updates")
        candidates = []
        for step in [100, 200]:
            checkpoint = run / f"checkpoint_{step:05d}.pt"
            output = run_plan["outputs"][step]
            env = dict(
                os.environ,
                CUDA_VISIBLE_DEVICES="2",
                ALLOCATION_CHECKPOINT=str(checkpoint),
                ALLOCATION_EVAL_OUT=str(output),
                ND_OFFLINE="1",
            )
            log = output.with_suffix(".log")
            started = time.monotonic()
            with log.open("x") as stream:
                result = subprocess.run(
                    ["bash", str(ALLOC / "launchers/dev_checkpoint.sh")],
                    cwd=ROOT,
                    env=env,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                )
            record = dict(
                seed=seed,
                step=step,
                output=str(output),
                log=str(log),
                monotonic_process_wall_seconds=time.monotonic() - started,
                returncode=result.returncode,
            )
            with ledger.open("a") as stream:
                stream.write(json.dumps(record) + "\n")
            if result.returncode:
                raise RuntimeError(record)
            summary = json.loads((output / "summary.json").read_text())["learned"]
            config = json.loads((output / "config.json").read_text())
            candidates.append(
                dict(
                    completed_updates=step,
                    checkpoint=str(checkpoint),
                    checkpoint_sha256=sha(checkpoint),
                    dev_config_sha256=sha(output / "config.json"),
                    inference_sha256=fingerprint(signature(config)),
                    correct_probability=summary["correct_probability"],
                    attempts=summary["attempts"],
                    native_reward=summary["reward"],
                    dev_output=str(output),
                )
            )
        selected = max(
            candidates, key=lambda r: (r["native_reward"], -r["attempts"], -r["completed_updates"])
        )
        choice = dict(
            selected_completed_updates=selected["completed_updates"],
            selected_checkpoint=selected["checkpoint"],
            selected_checkpoint_sha256=selected["checkpoint_sha256"],
            selection_split="dev",
            selection_rule="maximum native reward, then fewer attempts, then earlier checkpoint",
            candidates=candidates,
        )
        run_plan["selection"].write_text(json.dumps(choice, indent=2) + "\n")
        selected_output = Path(selected["dev_output"])
        fit_out = run_plan["baselines"][selected["completed_updates"]]
        fit_out.mkdir(exist_ok=False)
        common = [
            "--paths",
            str(selected_output / "public_paths.jsonl"),
            "--data",
            str(ART / "allocation/main_dev32_public_panel.jsonl"),
        ]
        for tool, name, extra in [
            ("fit_counter_hazards.py", "counter.json", []),
            (
                "fit_budget_baselines.py",
                "budgets.json",
                ["--labels", str(selected_output / "private_labels.jsonl")],
            ),
        ]:
            subprocess.run(
                [sys.executable, str(ALLOC / tool), *common, *extra, "--out", str(fit_out / name)],
                cwd=ROOT,
                check=True,
            )
            shutil.copy2(ALLOC / tool, fit_out / tool)
        print(json.dumps(choice), flush=True)


if __name__ == "__main__":
    main()
