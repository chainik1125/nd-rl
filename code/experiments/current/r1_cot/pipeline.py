"""Execute a frozen sequence of training/evaluation stages on one allocated GPU."""

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    config_path = Path(args.config)
    spec = json.loads(config_path.read_text())
    frozen = sha(config_path)
    for step in spec["stages"]:
        assert sha(config_path) == frozen
        if step["kind"] == "release_development":
            checkpoint = Path(step["completed_rl_checkpoint"])
            digest = sha(checkpoint)
            completed = json.loads((checkpoint.parent / "completed.json").read_text())
            assert completed["final_sha256"] == digest
            manifest = json.loads((checkpoint.parent / "manifest.json").read_text())
            assert manifest["args"]["command"] == "rl"
            release = dict(
                checkpoints=[sha(p) for p in step["checkpoints"]],
                data_sha256=[sha(p) for p in step["data"]],
                completed_rl_checkpoint=str(checkpoint),
                completed_rl_checkpoint_sha256=digest,
                created_utc=datetime.now(timezone.utc).isoformat(),
            )
            with Path(step["output"]).open("x") as f:
                json.dump(release, f, indent=2)
            continue
        assert step["kind"] == "command"
        print(
            json.dumps(dict(stage=step["name"], event="start", command=step["command"])), flush=True
        )
        subprocess.run(step["command"], check=True)
        print(json.dumps(dict(stage=step["name"], event="complete")), flush=True)
    with Path(spec["completed"]).open("x") as f:
        json.dump(
            dict(
                status="complete", config_sha256=frozen, utc=datetime.now(timezone.utc).isoformat()
            ),
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
