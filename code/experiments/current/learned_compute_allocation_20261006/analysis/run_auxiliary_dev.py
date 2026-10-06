"""Root-owned fixed-one and zero-cost seed102 development jobs."""

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
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "1":
        raise ValueError("This queue requires explicitly released GPU1")
    ledger = ART / "root_aux_dev_launches.jsonl"
    if ledger.exists():
        raise ValueError("Root auxiliary queue was already launched")
    plans = [("fixed1", 101, "fixed1"), ("no_cost", 102, "learned")]
    for arm, seed, _ in plans:
        folder = ART / "allocation" / arm
        if (folder / f"main_s{seed}_selection.json").exists():
            raise ValueError("Do not overwrite a previous development choice")
        for step in [100, 200]:
            if (folder / f"main_s{seed}_dev{step}_paths_s52006").exists():
                raise ValueError("Do not rerun an existing development output")
    shutil.copy2(__file__, ledger.with_suffix(".source.py"))
    for arm, seed, readout in plans:
        folder = ART / "allocation" / arm
        run = folder / f"main_s{seed}"
        candidates = []
        for step in [100, 200]:
            checkpoint = run / f"checkpoint_{step:05d}.pt"
            output = folder / f"main_s{seed}_dev{step}_paths_s52006"
            env = dict(
                os.environ,
                ALLOCATION_CHECKPOINT=str(checkpoint),
                ALLOCATION_EVAL_OUT=str(output),
                ND_OFFLINE="1",
            )
            start = time.monotonic()
            with output.with_suffix(".log").open("x") as stream:
                result = subprocess.run(
                    ["bash", str(ALLOC / "launchers/dev_checkpoint.sh")],
                    cwd=ROOT,
                    env=env,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                )
            record = dict(
                arm=arm,
                seed=seed,
                step=step,
                output=str(output),
                launcher_sha256=sha(__file__),
                returncode=result.returncode,
                monotonic_process_wall_seconds=time.monotonic() - start,
            )
            with ledger.open("a") as stream:
                stream.write(json.dumps(record) + "\n")
            if result.returncode:
                raise RuntimeError(record)
            score = json.loads((output / "summary.json").read_text())[readout]
            config = json.loads((output / "config.json").read_text())
            candidates.append(
                dict(
                    completed_updates=step,
                    checkpoint=str(checkpoint),
                    checkpoint_sha256=sha(checkpoint),
                    dev_output=str(output),
                    dev_config_sha256=sha(output / "config.json"),
                    inference_sha256=fingerprint(signature(config)),
                    correct_probability=score["correct_probability"],
                    attempts=score["attempts"],
                    native_reward=score["reward"],
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
            selection_readout=readout,
            selection_rule=(
                "maximum native readout reward, then fewer attempts, then earlier checkpoint"
            ),
            candidates=candidates,
        )
        (folder / f"main_s{seed}_selection.json").write_text(json.dumps(choice, indent=2) + "\n")
        if arm == "no_cost":
            output = Path(selected["dev_output"])
            fit = (
                folder / f"main_s{seed}_dev{selected['completed_updates']}_baselines_frozen_s52006"
            )
            fit.mkdir(exist_ok=False)
            common = [
                "--paths",
                str(output / "public_paths.jsonl"),
                "--data",
                str(ART / "allocation/main_dev32_public_panel.jsonl"),
            ]
            for tool, name, extra in [
                ("fit_counter_hazards.py", "counter.json", []),
                (
                    "fit_budget_baselines.py",
                    "budgets.json",
                    ["--labels", str(output / "private_labels.jsonl")],
                ),
            ]:
                subprocess.run(
                    [sys.executable, str(ALLOC / tool), *common, *extra, "--out", str(fit / name)],
                    cwd=ROOT,
                    check=True,
                )
                shutil.copy2(ALLOC / tool, fit / tool)
        print(json.dumps(choice), flush=True)


if __name__ == "__main__":
    main()
