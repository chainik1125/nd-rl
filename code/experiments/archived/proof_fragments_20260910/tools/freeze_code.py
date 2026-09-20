"""Create an immutable source snapshot and its content manifest."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("version")
    a = p.parse_args()
    if not a.version or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for c in a.version):
        raise ValueError("Invalid snapshot name")
    out = ROOT / "snapshots" / a.version
    shutil.copytree(ROOT / "code", out, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    hashes = {}
    for path in sorted(out.rglob("*")):
        if path.is_file():
            with path.open("rb") as stream:
                hashes[str(path)] = hashlib.file_digest(stream, "sha256").hexdigest()
    marker = out / "source_manifest.json"
    marker.write_text(json.dumps({"snapshot": str(out), "input_sha256": hashes}, indent=2) + "\n")
    print(json.dumps({"snapshot": str(out), "files": len(hashes), "manifest": str(marker)}))


if __name__ == "__main__":
    main()
