"""CPT control reads (PREREG): Charles149 at k1024 as 16 k64 chunks (`--set t1320s250`: the
13-20 sample at k512, 8 chunks, batch 512, as lowrl_1320's reads) with the paper's exact read
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
#: set -> (sets file, chunks of k64, --batch, chunk timeout s), as think_hirl's reads of that set
SETS = {
    "charles149": (H / "sets/charles149/charles_clean.jsonl", 16, 1024, 1800),
    "t1320s250": (H / "sets/t1320s250/t1320_n1000.jsonl", 8, 512, 3600),
}
HQ = WT / "code/experiments/current/thinking_traces_20261005/grpo_v2/launch/hirl/hirl_queue.py"
OUT = Path("/scratch/cowifornia/nd-rl/cpt_control")
K = 64


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
    ap.add_argument("--set", default="charles149", choices=tuple(SETS))
    a = ap.parse_args(argv)
    mod = hq()
    sets, n_chunks, batch, tmo = SETS[a.set]
    read = f"{a.set}_{a.stem}_s{a.seed}_plain_k{K * n_chunks}"
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
    for i in range(n_chunks):
        d = OUT / "chunks" / read / f"seed{i}"
        members.append(
            dict(
                chunk=i,
                seed=i,
                k=K,
                out=str(d),
                sets=str(sets),
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
            str(tmo),
            PY,
            "-m",
            "code.thinking.eval",
            "--ckpt",
            a.ckpt,
            "--sets",
            str(sets),
            "--out",
            str(d),
            "--k",
            str(K),
            "--batch",
            str(batch),
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
    pk = s["pass_at_k_pct"]
    print(
        f"{time.strftime('%H:%MZ', time.gmtime())} merged {dst}: n {s['n']}, pass@1 "
        f"{pk['1']:.1f}, pass@{K * n_chunks} {pk[str(K * n_chunks)]:.1f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
