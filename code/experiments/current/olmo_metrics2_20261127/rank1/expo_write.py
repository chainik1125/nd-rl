# ruff: noqa: E501
"""ExPO checkpoint writer (Turn 5 B.2 step 6): theta_gamma = dpo + gamma (s0500 - dpo), all tensors, fp32 combine, bf16 write.

  python3 expo_write.py --dpo /workspace/models/dpo --s0500 /workspace/models/s0500 --gamma 4.6 --out /workspace/models/expo_g46

Layout like dpo's: the same shard file names, the same tensor-to-shard map, dpo's model.safetensors.index.json copied
(metadata total_size is unchanged: same dtypes and shapes), config / tokenizer / generation files copied from dpo (the dpo and RL
configs differ only cosmetically). Writes MANIFEST.json (sha256 of every output file, gamma, inputs' index sha256) and verifies
`--verify` random tensors against an fp64 recompute rounded to bf16 (exit 4 on a mismatch). CPU / disk only (job --nogpu).
Rounding: torch float32 -> bfloat16 is round-to-nearest-even; (s0500 - dpo) of two bf16 tensors is exact in fp32.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from fractions import Fraction
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import Checkpoint, sha256_file  # noqa: E402

COPY_SUFFIX = (".json", ".txt", ".model", ".jinja", ".py", ".tiktoken")
SKIP = {"model.safetensors.index.json"}


def combine(dpo: torch.Tensor, s500: torch.Tensor, gamma: float) -> torch.Tensor:
    assert dpo.dtype == s500.dtype and dpo.shape == s500.shape
    if not dpo.is_floating_point():
        return dpo.clone()
    d = dpo.to(torch.float32)
    return (d + gamma * (s500.to(torch.float32) - d)).to(dpo.dtype)


def verify_tensors(dpo, s5, out_ck, gamma: float, verify: int, seed: int = 0) -> list[dict]:
    """Random tensors against an exact reference (orchestrator fix, 2026-11-27, after the first gamma 4.6 write).

    gamma is taken as the nearest fraction p/q with q <= 100 (4.6 = 23/5, 3.5 = 7/2); for bf16 a, b the values q a and
    p (b - a) are exact in fp64, so r = (q a + p (b - a)) / q is one rounding from the exact result, and bf16(r) via fp32 is
    the correctly rounded answer. A tensor passes when every element of the written bf16 is within 1 bf16 ulp of bf16(r)
    (the ulp taken at max(|bf16(r)|, |o|), so exact zeros are compared exactly). The first version used a + 4.6 (b - a)
    with fp64's inexact 4.6 and a purely relative ulp, which flagged exact cancellations (23 b = 18 a, true value 0,
    written 0) as 128-ulp errors."""
    fr = Fraction(gamma).limit_denominator(100)
    if abs(float(fr) - gamma) > 1e-12:
        raise SystemExit(
            f"gamma {gamma} is not a fraction with denominator <= 100; verification undefined"
        )
    p, q = fr.numerator, fr.denominator
    rnd = random.Random(seed)
    chk = []
    wmap = dpo.wmap
    eps = torch.finfo(torch.bfloat16).eps
    tiny = torch.finfo(torch.bfloat16).tiny
    for n in rnd.sample(sorted(wmap), min(verify, len(wmap))):
        a, b, o = dpo.get(n), s5.get(n), out_ck.get(n)
        if not a.is_floating_point():
            chk.append(
                {"name": n, "ok": bool(torch.equal(a, o)), "max_abs_diff_ulp": 0.0, "n_ne": 0}
            )
            continue
        ad, bd = a.double(), b.double()
        ref = ((q * ad + p * (bd - ad)) / q).float().to(torch.bfloat16).double()
        od = o.double()
        diff = (od - ref).abs()
        ulp = eps * torch.maximum(ref.abs(), od.abs()).clamp_min(tiny)
        worst = float((diff / ulp).max())
        chk.append(
            {
                "name": n,
                "ok": worst <= 1.0,
                "max_abs_diff_ulp": worst,
                "n_ne": int((od != ref).sum()),
            }
        )
    return chk


def write(
    dpo_dir,
    s500_dir,
    gamma: float,
    out_dir,
    verify: int = 3,
    seed: int = 0,
    min_free_gb: float = 149.0,
) -> dict:
    dpo, s5 = Checkpoint(dpo_dir), Checkpoint(s500_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(out).free / 1e9
    if free < min_free_gb:  # Turn 6 D7: refuse unless 2 x 64.5 GB + 20 GB (default 149) is free
        raise SystemExit(
            f"REFUSED: {free:.0f} GB free under {out}, need {min_free_gb:.0f} GB (exit 5)"
        )
    wmap = dpo.wmap
    if set(wmap) != set(s5.names()):
        raise SystemExit("tensor sets of dpo and s0500 differ")
    shards: dict[str, list[str]] = {}
    for n, sh in wmap.items():
        shards.setdefault(sh, []).append(n)
    written = []
    for sh, ns in sorted(shards.items()):
        tensors = {n: combine(dpo.get(n), s5.get(n), gamma) for n in ns}
        meta = {}
        with safe_open(str(Path(dpo_dir) / sh), framework="pt") as f:
            meta = f.metadata() or {}
        save_file(tensors, str(out / sh), metadata=meta or {"format": "pt"})
        written.append(sh)
        del tensors
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
    chk = verify_tensors(dpo, s5, Checkpoint(out), gamma, verify, seed)
    ok = all(c["ok"] for c in chk)
    man = {
        "gamma": gamma,
        "dpo": str(dpo_dir),
        "s0500": str(s500_dir),
        "tensors": len(wmap),
        "shards": written,
        "index_sha256": sha256_file(out / "model.safetensors.index.json"),
        "verify": chk,
        "verify_ok": ok,
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
    ap.add_argument("--s0500", required=True)
    ap.add_argument("--gamma", type=float, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--verify", type=int, default=3)
    ap.add_argument("--min-free-gb", type=float, default=149.0)
    ap.add_argument(
        "--verify-only",
        action="store_true",
        help="re-verify an existing --out (no write); records the result as MANIFEST.json 'reverify' (the original "
        "'verify' entries stay); exit 0 pass, 4 fail",
    )
    a = ap.parse_args(argv)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    if a.verify_only:
        out = Path(a.out)
        man = json.loads((out / "MANIFEST.json").read_text())
        if abs(float(man["gamma"]) - a.gamma) > 1e-12:
            print(f"gamma {a.gamma} differs from MANIFEST gamma {man['gamma']}")
            return 3
        chk = verify_tensors(
            Checkpoint(a.dpo), Checkpoint(a.s0500), Checkpoint(out), a.gamma, a.verify, seed=1
        )
        ok = all(c["ok"] for c in chk)
        man.setdefault("reverify", []).append(
            {"verify": chk, "verify_ok": ok, "seed": 1, "n": a.verify}
        )
        (out / "MANIFEST.json").write_text(json.dumps(man, indent=1))
        print(f"EXPO REVERIFY gamma={a.gamma} verify_ok={ok} tensors_checked={len(chk)}")
        return 0 if ok else 4
    try:
        man = write(a.dpo, a.s0500, a.gamma, a.out, a.verify, min_free_gb=a.min_free_gb)
    except SystemExit as ex:
        print(ex)
        return 5
    print(f"EXPO WRITE gamma={a.gamma} verify_ok={man['verify_ok']} tensors={man['tensors']}")
    return 0 if man["verify_ok"] else 4


if __name__ == "__main__":
    sys.exit(main())
