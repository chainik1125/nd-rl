"""Freeze the existing tested SFT pipeline for the sprint's fresh baseline."""
from pathlib import Path
import hashlib
import json
import shutil

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT.parent / "r1-sprint-20260909-1625"
PYTHON = "/data/users/dmitry/simplex-research/.venv/bin/python"


def main():
    if (ROOT / "configs/job_base_sft_v1.json").exists():
        raise FileExistsError("Baseline job is already frozen")
    snapshot = ROOT / "snapshots/base_v1"
    snapshot.mkdir(parents=True, exist_ok=True)
    for name in ("run.py", "scratch_variants.py"):
        shutil.copy2(PRIOR / "code" / name, snapshot / name)
    shutil.copytree(PRIOR / "code/vendor", snapshot / "vendor",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"), dirs_exist_ok=True)
    config_path = ROOT / "configs/base_model_v1.json"
    config_path.write_text(json.dumps(dict(d_model=384, n_layers=5, n_heads=6,
        d_ff=1536, max_seq_len=4096, dropout=0, rope_base=10000), indent=2) + "\n")
    run = str(snapshot / "run.py")
    initial = ROOT / "results/base_init_v1"
    data = PRIOR / "data/sft_cap6.jsonl"
    guards = PRIOR / "data/heldout_guards.json"
    commands = [[PYTHON, run, "prepare", "--fresh-config", str(config_path),
        "--output", str(initial), "--seed", "20260910", "--mode", "cot", "--device", "cuda"]]
    for name, steps, batch, micro in (("base_smoke_v1", 2, 8, 4), ("base_sft_v1", 8000, 128, 32)):
        commands.append([PYTHON, run, "sft", "--checkpoint", str(initial / "final.pt"),
            "--data", str(data), "--guard-json", str(guards),
            "--output", str(ROOT / "results" / name), "--mode", "direct", "--device", "cuda",
            "--seed", "20260910", "--steps", str(steps), "--batch-size", str(batch),
            "--microbatch", str(micro), "--lr", "0.0003", "--precision", "bf16",
            "--max-proof-lines", "6", "--log-every", "250", "--save-every", "2000"])
    pipeline = snapshot / "pipeline.json"
    pipeline.write_text(json.dumps(commands, indent=2) + "\n")
    entry = snapshot / "baseline_pipeline.py"
    entry.write_text('''from pathlib import Path
import json, subprocess, time
for command in json.loads(Path(__file__).with_name("pipeline.json").read_text()):
    print(json.dumps({"event":"stage_start", "command":command, "unix_time":time.time()}), flush=True)
    subprocess.run(command, check=True)
    print(json.dumps({"event":"stage_complete", "unix_time":time.time()}), flush=True)
''')
    files = [p for p in snapshot.rglob("*") if p.is_file()] + [config_path, data, guards]
    files += [ROOT / "tools/launch.py", PRIOR / "code/dispatch.py"]
    hashes = {}
    for path in files:
        with path.open("rb") as stream:
            hashes[str(path)] = hashlib.file_digest(stream, "sha256").hexdigest()
    spec = dict(name="base_sft_v1", question="Can a fresh cap6 model reproduce a usable starting policy?",
        expected_seconds=600, expected_cloud_cost_usd=0, seed=20260910,
        command=[PYTHON, str(entry)], input_sha256=hashes, candidate_gpus=[4],
        timeout_seconds=1800, queue_timeout_seconds=180,
        deadline_utc="2026-09-10T19:00:00+00:00",
        env={"PYTHONPATH": str(snapshot / "vendor") + ":" + str(snapshot)},
        success="exit0 and results/base_sft_v1/completed.json with 8000 steps and matching final hash",
        failure="nonfinite gradients, verification/guard rejection, nonzero exit, deadline or input drift",
        recovery="Inspect jobs/base_sft_v1/worker.log and completed.json; partial snapshots retained; no auto-retry")
    path = ROOT / "configs/job_base_sft_v1.json"
    path.write_text(json.dumps(spec, indent=2) + "\n")
    print(json.dumps({"spec": str(path), "input_files": len(hashes), "commands": commands}, indent=2))


if __name__ == "__main__":
    main()
