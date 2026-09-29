"""Download one pinned public base checkpoint and record its file identities."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(ROOT / "cache" / "huggingface"))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="allenai/Olmo-3-1125-32B")
    parser.add_argument("--revision", default="c2b61dae89a1ad10e4ad5653d0e46b590902607b")
    parser.add_argument("--out", type=Path, default=ROOT / "models" / "olmo32b")
    parser.add_argument("--metadata-only", action="store_true")
    args = parser.parse_args()
    from huggingface_hub import HfApi, snapshot_download
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    info = HfApi(token=False).model_info(args.repo, revision=args.revision, files_metadata=True)
    if info.sha != args.revision:
        raise ValueError("Model revision did not resolve to the frozen commit")
    files = []
    for item in info.siblings:
        if item.rfilename.endswith((".json", ".txt", ".safetensors")):
            files.append({"name": item.rfilename, "size": item.size,
                          "lfs_sha256": getattr(item.lfs, "sha256", None)})
    manifest = {"repo": args.repo, "revision": info.sha, "started_unix": started,
                "expected_bytes": sum(x["size"] or 0 for x in files), "files": files}
    (args.out / "download_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"event": "resolved", "repo": args.repo, "revision": info.sha,
                      "expected_bytes": manifest["expected_bytes"]}), flush=True)
    patterns = ["*.json", "*.txt"] if args.metadata_only else [x["name"] for x in files]
    snapshot_download(repo_id=args.repo, revision=args.revision, local_dir=args.out,
                      allow_patterns=patterns, max_workers=4, token=False)
    verified = []
    for item in files:
        path = args.out / item["name"]
        if args.metadata_only and path.suffix == ".safetensors":
            continue
        if not path.is_file() or path.stat().st_size != item["size"]:
            raise ValueError("Downloaded file missing or wrong size: " + item["name"])
        digest = hashlib.file_digest(path.open("rb"), "sha256").hexdigest()
        if item["lfs_sha256"] and digest != item["lfs_sha256"]:
            raise ValueError("Downloaded file digest mismatch: " + item["name"])
        verified.append({**item, "sha256": digest})
        print(json.dumps({"event": "verified", "name": item["name"], "sha256": digest}), flush=True)
    manifest.update(verified=verified, completed_unix=time.time(), elapsed_seconds=time.time()-started,
                    metadata_only=args.metadata_only)
    (args.out / ("metadata_completed.json" if args.metadata_only else "download_completed.json")).write_text(
        json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"event": "completed", "elapsed_seconds": manifest["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
