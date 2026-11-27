# ruff: noqa: E501
"""Stage trained cells for the pod (SERRANO side).

  stage_cells.py --runs <runs dir> --ckpt <ckpt> --cells a,b,c --stage <batch dir> [--force]

Reads <runs>/<ckpt>/train/cells/<cell>/ (DONE, cell.json, adapter/, gate.npz), refuses a cell without DONE, with a
cell.json that does not parse or lacks required keys, with an adapter that is not a PEFT LoRA of rank <= 16 with its
weights file, or without gate.npz; refuses did_not_train cells unless --force (then recorded as forced). Stages
adapter/ + gate.npz + cell.json into <batch dir>/<cell>/ (hard links where the filesystem allows, else copies) and
writes <batch dir>/cells_manifest.json: per cell the ckpt, the cell.json summary and md5 + sha256 + size of every
staged file. Exit 0 when every asked cell is staged, 1 otherwise (nothing is staged for a refused cell; others are).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

REQUIRED = ("cell", "family", "P", "n", "source", "selected")
MAX_RANK = 16


def digests(p: Path) -> dict:
    m, s = hashlib.md5(), hashlib.sha256()  # noqa: S324 (integrity check, not security)
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            m.update(b)
            s.update(b)
    return {"md5": m.hexdigest(), "sha256": s.hexdigest(), "bytes": p.stat().st_size}


def check_cell(cdir: Path, force: bool) -> tuple[dict | None, str | None]:
    """(cell.json dict, None) when the cell may be staged, else (None, reason)."""
    if not (cdir / "DONE").exists():
        return None, "no DONE"
    try:
        cj = json.loads((cdir / "cell.json").read_text())
    except (OSError, json.JSONDecodeError) as e:
        return None, f"cell.json unreadable: {e!r}"
    miss = [k for k in REQUIRED if k not in cj]
    if miss:
        return None, f"cell.json lacks {miss}"
    if cj["cell"] != cdir.name:
        return None, f"cell.json says cell={cj['cell']!r}"
    ad = cdir / "adapter"
    try:
        cfg = json.loads((ad / "adapter_config.json").read_text())
    except (OSError, json.JSONDecodeError) as e:
        return None, f"adapter_config.json unreadable: {e!r}"
    if not (ad / "adapter_model.safetensors").exists():
        return None, "no adapter_model.safetensors"
    if cfg.get("peft_type") != "LORA" or not 1 <= int(cfg.get("r", 0)) <= MAX_RANK:
        return (
            None,
            f"adapter is not a LoRA of rank <= {MAX_RANK} (r={cfg.get('r')}, type={cfg.get('peft_type')})",
        )
    if not (cdir / "gate.npz").exists():
        return None, "no gate.npz"
    if cj.get("did_not_train") and not force:
        return None, "did_not_train (use --force to stage anyway)"
    return cj, None


def put(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def stage(
    runs: Path, ckpt: str, cells: list[str], batch: Path, force: bool = False
) -> tuple[dict, dict]:
    """Returns (manifest, {cell: refusal reason})."""
    man_path = batch / "cells_manifest.json"
    manifest = json.loads(man_path.read_text()) if man_path.exists() else {"cells": {}}
    refused: dict[str, str] = {}
    for c in cells:
        cdir = runs / ckpt / "train" / "cells" / c
        cj, why = check_cell(cdir, force)
        if cj is None:
            refused[c] = why or "?"
            continue
        dest = batch / c
        if dest.exists():
            shutil.rmtree(dest)
        files = {}
        for src in sorted((cdir / "adapter").iterdir()):
            if src.is_file():
                put(src, dest / "adapter" / src.name)
                files[f"adapter/{src.name}"] = digests(dest / "adapter" / src.name)
        for name in ("gate.npz", "cell.json"):
            put(cdir / name, dest / name)
            files[name] = digests(dest / name)
        manifest["cells"][c] = {
            "ckpt": ckpt,
            "family": cj["family"],
            "P": cj["P"],
            "n": cj["n"],
            "source": cj["source"],
            "selected": cj["selected"],
            "did_not_train": bool(cj.get("did_not_train", False)),
            "forced": bool(cj.get("did_not_train", False) and force),
            "files": files,
        }
    manifest["utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    manifest["refused"] = {**manifest.get("refused", {}), **refused}
    for c in cells:
        if c not in refused:
            manifest["refused"].pop(c, None)
    batch.mkdir(parents=True, exist_ok=True)
    man_path.write_text(json.dumps(manifest, indent=1) + "\n")
    return manifest, refused


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="/scratch/cowifornia/nd-rl/olmo_elicit/v2/runs")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cells", required=True, help="comma list of cell ids")
    ap.add_argument("--stage", required=True, help="batch directory to fill")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    cells = [c for c in a.cells.split(",") if c]
    man, refused = stage(Path(a.runs), a.ckpt, cells, Path(a.stage), a.force)
    for c in cells:
        if c in refused:
            print(f"REFUSED {c}: {refused[c]}")
        else:
            m = man["cells"][c]
            print(
                f"STAGED {c}: P={m['P']} n={m['n']} source={m['source']} did_not_train={m['did_not_train']} forced={m['forced']}"
            )
    print(
        f"STAGE_{'OK' if not refused else 'PARTIAL'} {len(cells) - len(refused)}/{len(cells)} -> {a.stage}"
    )
    return 0 if not refused else 1


if __name__ == "__main__":
    sys.exit(main())
