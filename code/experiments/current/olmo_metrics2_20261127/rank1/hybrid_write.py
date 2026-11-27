# ruff: noqa: E501
"""E checkpoint writer (conditional cell, Turn 5 B.8): s2300's 448 attention / MLP matrices on dpo, everything else (embeddings,
lm_head, all norms) at dpo = "dpo plus the full change on the 448 matrices". Bounds what the LoRA route (matrices only) leaves out.

  python3 hybrid_write.py --dpo /workspace/models/dpo --s2300 /workspace/models/s2300 --out /workspace/models/hybrid_e [--layers 64]

Same shard layout, index, config files and MANIFEST (sha256) as expo_write.py; tensors are copied bit for bit (no arithmetic); verifies that
every matrix equals s2300's and every other tensor equals dpo's (exit 4 on a difference). Refuses (exit 5) without 64.5 GB + 20 GB free.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import Checkpoint, matrix_names, sha256_file  # noqa: E402
from expo_write import COPY_SUFFIX, SKIP  # noqa: E402


def write(dpo_dir, s2300_dir, out_dir, n_layers: int = 64, min_free_gb: float = 85.0) -> dict:
    dpo, s23 = Checkpoint(dpo_dir), Checkpoint(s2300_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(out).free / 1e9
    if free < min_free_gb:
        raise SystemExit(
            f"REFUSED: {free:.0f} GB free under {out}, need {min_free_gb:.0f} GB (exit 5)"
        )
    mats = set(matrix_names(n_layers))
    assert mats <= set(dpo.names()) and set(dpo.names()) == set(s23.names())
    shards: dict[str, list[str]] = {}
    for n, sh in dpo.wmap.items():
        shards.setdefault(sh, []).append(n)
    for sh, ns in sorted(shards.items()):
        tensors = {n: (s23.get(n) if n in mats else dpo.get(n)).clone() for n in ns}
        with safe_open(str(Path(dpo_dir) / sh), framework="pt") as f:
            meta = f.metadata() or {"format": "pt"}
        save_file(tensors, str(out / sh), metadata=meta)
        print(f"wrote {sh} ({len(ns)} tensors)", flush=True)
    shutil.copy2(
        Path(dpo_dir) / "model.safetensors.index.json", out / "model.safetensors.index.json"
    )
    for p in Path(dpo_dir).iterdir():
        if (
            p.is_file()
            and p.suffix in COPY_SUFFIX
            and p.name not in SKIP
            and not p.name.endswith(".safetensors")
        ):
            shutil.copy2(p, out / p.name)
    o = Checkpoint(out)
    bad = [
        n
        for n in dpo.names()
        if not torch.equal(o.get(n), (s23.get(n) if n in mats else dpo.get(n)))
    ]
    man = {
        "dpo": str(dpo_dir),
        "s2300": str(s2300_dir),
        "matrices_from_s2300": len(mats),
        "others_from_dpo": len(dpo.names()) - len(mats),
        "mismatched_tensors": bad,
        "verify_ok": not bad,
        "files": {
            p.name: sha256_file(p)
            for p in sorted(out.iterdir())
            if p.is_file() and p.name != "MANIFEST.json"
        },
    }
    (out / "MANIFEST.json").write_text(json.dumps(man, indent=1))
    return man


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dpo", required=True)
    ap.add_argument("--s2300", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", type=int, default=64)
    ap.add_argument("--min-free-gb", type=float, default=85.0)
    a = ap.parse_args(argv)
    try:
        man = write(a.dpo, a.s2300, a.out, a.layers, a.min_free_gb)
    except SystemExit as ex:
        print(ex)
        return 5
    print(f"HYBRID E WRITE verify_ok={man['verify_ok']}")
    return 0 if man["verify_ok"] else 4


if __name__ == "__main__":
    sys.exit(main())
