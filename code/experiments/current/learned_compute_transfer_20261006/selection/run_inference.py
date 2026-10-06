"""Guarded transfer inference queue; refuses overwrite and records every child."""

import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

from transfer_guards import validate

ROOT = Path(__file__).resolve().parents[5]
ART = ROOT / "artifacts/learned_compute_transfer_20261006"


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    protocol_path = ART / "phase1_protocol.json"
    protocol = json.loads(protocol_path.read_text())
    digest = sha(protocol_path)
    assert protocol["launcher_sha256"] == sha(Path(__file__))
    inventory_path = ART / "phase1_inventory.json"
    assert sha(inventory_path) == protocol["inventory_sha256"]
    commands = validate(protocol)
    launch = ART / "selectors/launches"
    launch.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="1", ND_OFFLINE="1")
    for command in commands:
        validate(protocol)
        assert sha(protocol_path) == digest and sha(inventory_path) == protocol["inventory_sha256"]
        for p, value in protocol["public_hashes"].items():
            assert sha(Path(p)) == value
        assert sha(Path(__file__)) == protocol["launcher_sha256"]
        entry = launch / command["label"]
        entry.mkdir()
        argv = command["argv"]
        out = Path(argv[argv.index("--out") + 1])
        assert not out.exists()
        shutil.copy2(__file__, entry / "launcher.py")
        shutil.copy2(argv[1], entry / "source.py")
        start = time.monotonic()
        with (entry / "stdout.log").open("w") as log:
            proc = subprocess.Popen(argv, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            meta = {
                "argv": argv,
                "pid": proc.pid,
                "protocol_sha256": digest,
                "inventory_sha256": sha(inventory_path),
                "launcher_sha256": sha(Path(__file__)),
                "entrypoint_sha256": sha(Path(argv[1])),
                "private_sha256": protocol["private_sha256"],
            }
            (entry / "run_manifest.json").write_text(json.dumps(meta, indent=2))
            code = proc.wait()
        meta.update(
            exit_code=code,
            wall_seconds=time.monotonic() - start,
            output_hashes={p.name: sha(p) for p in out.iterdir() if p.is_file()}
            if out.exists()
            else {},
        )
        (entry / "run_manifest.json").write_text(json.dumps(meta, indent=2) + "\n")
        print(command["label"], code, flush=True)
        if code:
            raise SystemExit(code)


if __name__ == "__main__":
    main()
