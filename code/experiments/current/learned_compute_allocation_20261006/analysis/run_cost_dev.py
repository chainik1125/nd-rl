"""Coordinate the already-prespecified cost-arm development evaluations."""

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


def main():
    ledger = ART / "allocation/cost/dev_launches.jsonl"
    shutil.copy2(__file__, ledger.with_suffix(".source.py"))
    training_config = json.loads((ART / "allocation/cost/main_s103/config.json").read_text())
    while Path(f"/proc/{training_config['pid']}").exists():
        time.sleep(2)
    for seed in [101, 102, 103]:
        run = ART / f"allocation/cost/main_s{seed}"
        metrics = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
        if len(metrics) != 200:
            raise ValueError("Training did not finish the prescribed200 updates")
        candidates = []
        for step in [100, 200]:
            checkpoint = run / f"checkpoint_{step:05d}.pt"
            output = ART / f"allocation/cost/main_s{seed}_dev{step}_paths_s52006"
            env = dict(
                os.environ,
                CUDA_VISIBLE_DEVICES="3",
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
        (run.parent / f"main_s{seed}_selection.json").write_text(
            json.dumps(choice, indent=2) + "\n"
        )
        selected_output = Path(selected["dev_output"])
        fit_name = f"main_s{seed}_dev{selected['completed_updates']}_baselines_frozen_s52006"
        fit_out = ART / "allocation/cost" / fit_name
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
