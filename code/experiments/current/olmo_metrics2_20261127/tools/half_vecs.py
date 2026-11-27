"""Write 0.5 x copies of steering-vector files (orchestrator, 2026-11-27; PREREG_steer Amendment 3).

    half_vecs.py runs/m3_p3/out/vecs/scramble2001.pt [...]   ->  <same dir>/scramble2001_half.pt ...

A sign scramble of (alpha / 2) v equals 0.5 x the sign scramble of alpha v, and scaling an fp32 tensor by 0.5 is exact, so the
copies are the pre-registered E2 scrambles at the halved alpha. Metadata: alpha halved, norm and sha256 (of the fp32 bytes,
as steer_patch.load_vec checks) recomputed; layer, vtype and scramble_seed kept. Prints one line per file with its sha256.
"""

import hashlib
import sys
from pathlib import Path

import torch


def half(src: str) -> dict:
    obj = torch.load(src, map_location="cpu", weights_only=False)
    v = obj["vec"].reshape(-1).to(torch.float32)
    h = v * 0.5
    assert torch.equal(h * 2.0, v), "0.5 scaling not exact"
    out = dict(obj)
    out["vec"] = h
    out["alpha"] = (obj.get("alpha") or 0) * 0.5
    out["norm"] = float(h.norm())
    out["sha256"] = hashlib.sha256(h.numpy().tobytes()).hexdigest()
    out["derived_from"] = {"file": Path(src).name, "sha256": obj.get("sha256"), "factor": 0.5}
    dst = Path(src).with_name(Path(src).stem + "_half.pt")
    torch.save(out, dst)
    return {"file": str(dst), "sha256": out["sha256"], "norm": out["norm"], "alpha": out["alpha"]}


if __name__ == "__main__":
    for s in sys.argv[1:]:
        r = half(s)
        print(f"HALF {r['file']} alpha={r['alpha']} norm={r['norm']:.6f} sha256={r['sha256']}")
