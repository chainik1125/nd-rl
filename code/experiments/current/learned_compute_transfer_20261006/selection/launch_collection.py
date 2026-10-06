"""Hash-guarded local phase1 collection, preserving unread private logs."""

import datetime
import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
ART = ROOT / "artifacts/learned_compute_transfer_20261006"
OLD = ROOT / "code/experiments/current/learned_compute_allocation_20261006/selection"


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    freeze = ART / "phase1_decisions.json"
    assert sha(freeze) == "fca9e56597b9486f5504a2c2c0017ff714a817f08b61f735f220386d2d3324ad"
    plan = json.loads(freeze.read_text())
    for name, value in plan["source_sha256"].items():
        assert sha(Path(name)) == value
    data = ART / "data/combined412/public.jsonl"
    assert sha(data) == plan["data_sha256"]
    launch = ART / "pools/combined412_launch"
    launch.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, launch / "launcher.py")
    for name in [
        "collect.py",
        "render_transitions.py",
        "data.py",
        "train.py",
        "train_bce.py",
        "evaluate.py",
    ]:
        shutil.copy2(OLD / name, launch / name)
    cmd = [
        "/data/users/dmitry/venvs/proof-state/bin/python",
        str(OLD / "collect.py"),
        "--data",
        str(data),
        "--out",
        str(ART / "pools/combined412"),
        "--seed",
        "94006",
        "--k",
        "8",
        "--temperature",
        "0.8",
        "--batch",
        "256",
        "--max-action",
        "256",
        "--max-steps",
        "48",
    ]
    start = time.monotonic()
    meta = {
        "argv": cmd,
        "freeze_sha256": sha(freeze),
        "launcher_sha256": sha(Path(__file__)),
        "started_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "cuda_visible_devices": "1",
        "private_log_policy": "unread until public seal",
    }
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="1", ND_OFFLINE="1")
    with (launch / "private_collection.log").open("w") as log:
        proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
        meta["pid"] = proc.pid
        (launch / "run_manifest.json").write_text(json.dumps(meta, indent=2))
        code = proc.wait()
    meta.update(exit_code=code, wall_seconds=time.monotonic() - start)
    (launch / "run_manifest.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps({"exit_code": code, "wall_seconds": meta["wall_seconds"]}))


if __name__ == "__main__":
    main()
