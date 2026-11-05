"""CPT control reads (PREREG): Charles149 at k1024 as 16 k64 chunks with the paper's exact read
command (think_hirl's `hirl_queue`), merged per theorem in seed order by its `merge_group`. Runs
on ynez.

    python -m code.experiments.current.cpt_control_20261105.reads --gpu 2 \
        --ckpt /scratch/cowifornia/nd-rl/circuits/ckpts/cpt/cpt_lr0.00015.pt --seed 0

Chunks go to /scratch/cowifornia/nd-rl/cpt_control/chunks/<read>/seed<i>; a finished chunk (rows
and summary present) is skipped, a partial one is moved aside. The merged read is
/scratch/cowifornia/nd-rl/cpt_control/final/charles149_cpt_s1g3_s<seed>_plain_k1024.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import time
from pathlib import Path

H = Path("/scratch/cowifornia/nd-rl/think_hirl")
WT = H / "nd-rl-hirl"  # the paper's read code (branch leon/high-rl-reads), used read-only
PY = "/scratch/cowifornia/nd-rl/think/nd-rl/code/.venv/bin/python"
SETS = H / "sets/charles149/charles_clean.jsonl"
HQ = WT / "code/experiments/current/thinking_traces_20261005/grpo_v2/launch/hirl/hirl_queue.py"
OUT = Path("/scratch/cowifornia/nd-rl/cpt_control")
N_CHUNKS, K = 16, 64


def hq():
    spec = importlib.util.spec_from_file_location("hirl_queue", HQ)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def gpu_free(gpu: int) -> bool:
    q = ["nvidia-smi", "-i", str(gpu), "--query-compute-apps=pid", "--format=csv,noheader"]
    return not subprocess.check_output(q, text=True).strip()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--seed", type=int, required=True, help="the base seed (names the read)")
    ap.add_argument("--stem", default="cpt_s1g3")
    a = ap.parse_args(argv)
    mod = hq()
    read = f"charles149_{a.stem}_s{a.seed}_plain_k1024"
    env = dict(os.environ)
    env.update(mod.ENV)
    env.update(
        LEAN_GATE_WORKERS="8",  # the 12-core limit (the paper's reads used 16)
        CUDA_VISIBLE_DEVICES=str(a.gpu),
        TMPDIR=str(OUT / "tmp"),
        OMP_NUM_THREADS="3",
    )
    (OUT / "tmp").mkdir(parents=True, exist_ok=True)
    members = []
    for i in range(N_CHUNKS):
        d = OUT / "chunks" / read / f"seed{i}"
        members.append(
            dict(
                chunk=i,
                seed=i,
                k=K,
                out=str(d),
                sets=str(SETS),
                keep=None,
                ckpt=a.ckpt,
                id=f"{read}.c{i}",
            )
        )
        if (d / "rows_plain_B64.jsonl").exists() and (d / "summary.json").exists():
            continue
        if d.exists():
            d.rename(
                d.with_name(d.name + f".crashed_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}")
            )
        if not gpu_free(a.gpu):
            raise SystemExit(f"GPU {a.gpu} has a compute process: not starting chunk {i}")
        cmd = [
            "timeout",
            "--kill-after=120",
            "1800",
            PY,
            "-m",
            "code.thinking.eval",
            "--ckpt",
            a.ckpt,
            "--sets",
            str(SETS),
            "--out",
            str(d),
            "--k",
            str(K),
            "--batch",
            "1024",
            *mod.READ_FLAGS,
            "--seed",
            str(i),
        ]
        t0 = time.time()
        rc = subprocess.run(cmd, cwd=WT, env=env).returncode
        ok = (d / "summary.json").exists()
        print(
            f"{time.strftime('%H:%MZ', time.gmtime())} {read} chunk {i}: rc {rc}, "
            f"{time.time() - t0:.0f}s, {'ok' if ok else 'FAILED'}",
            flush=True,
        )
        if not ok:
            raise SystemExit(f"chunk {i} failed")
    mod.FINAL = OUT / "final"
    mod.FINAL.mkdir(parents=True, exist_ok=True)
    if (mod.FINAL / read).exists():
        print(f"exists: {mod.FINAL / read}")
        return 0
    dst = mod.merge_group(read, members)
    s = json.loads((dst / "summary.json").read_text())["cells"]["plain_B64"]["all"]
    print(
        f"{time.strftime('%H:%MZ', time.gmtime())} merged {dst}: n {s['n']}, pass@1 "
        f"{s['pass_at_k_pct']['1']:.1f}, pass@1024 {s['pass_at_k_pct']['1024']:.1f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
