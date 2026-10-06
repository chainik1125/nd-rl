"""Record package and executable versions without initializing CUDA or reading outcomes."""

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--recorded-at-tool-utc", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if torch.cuda.is_initialized():
        raise RuntimeError("Version capture should not initialize a CUDA context")
    lean = Path.home() / ".elan/bin/lean"
    lean_version = subprocess.run(
        [str(lean), "--version"], capture_output=True, text=True, check=True
    ).stdout.strip()
    devices = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,uuid,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
    packages = sorted(
        [
            {"name": dist.metadata["Name"], "version": dist.version}
            for dist in importlib.metadata.distributions()
        ],
        key=lambda row: row["name"].lower(),
    )
    result = {
        "recorded_at_tool_utc": args.recorded_at_tool_utc,
        "python_executable": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "torch_version": str(torch.__version__),
        "torch_compiled_cuda_version": torch.version.cuda,
        "torch_build_configuration": torch.__config__.show(),
        "cuda_context_initialized": torch.cuda.is_initialized(),
        "lean_executable": str(lean),
        "lean_version": lean_version,
        "gpu_columns": ["index", "name", "uuid", "driver_version", "total_memory_mib"],
        "gpus": devices,
        "installed_distributions": packages,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scope": (
            "Research interpreter version snapshot during registered evaluation. "
            "Run-specific precision, seeds, flags and sources remain authoritative in each "
            "run config. This does not inspect process environments or private outcomes, "
            "initialize a GPU context, or claim a complete reproducible system image."
        ),
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    output.with_suffix(".source.py").write_bytes(Path(__file__).read_bytes())
    print(
        json.dumps(
            {
                key: result[key]
                for key in ["torch_version", "lean_version", "cuda_context_initialized"]
            }
        )
    )


if __name__ == "__main__":
    main()
