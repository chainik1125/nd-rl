#!/usr/bin/env python3
"""Mirror the small, expensive thinking-traces artifacts from ynez's local /scratch to NFS.

    python3 backup.py [--dry] [--allow-mass-delete]   # one pass; queue.sh runs it after every arm

Installed as /scratch/cowifornia/nd-rl/think/runs/backup.py (source: grpo_v2/launch/backup.py on
leon/grpo-v2). SRC = /scratch/cowifornia/nd-rl/think/runs, DST =
/home/cowifornia/backups/nd-rl-think-ynez, same relative paths. Written 2026-10-13 after serrano's
crash, on the pattern of serrano's backup.py (/home/cowifornia/backups/nd-rl-think/backup.py).

Selected, by tier:
  1 launch scripts (runs/*.py, runs/*/*.sh, the files in runs/grpo_v2/), runs/logs/**, and per
    GRPO/EI run (grpo_v2/<run>/) its args/metrics/round/canary json(l), markdown and figures
  2 eval rows and summaries: eval/v2/<read>/{rows_*, summary*, args*}; all of eval/matched,
    eval/tts and eval/charles
  3 final pool and test-set files: pools/** and restart*/*.jsonl (not work/gen/raw dirs)
  4 Stage-1 models: stage1/*/model.pt and their json
  5 stage-final checkpoints: grpo_v2/<run>/ckpts/<run>_it<iters>.pt (GRPO: 39) and
    <run>_r<rounds>.pt (EI: 1), with their .upload.json
  6 data pickles data/*.pkl and data json
Never: smoke dirs, tmp/, work dirs, EI found_*.jsonl (~1 GB each), registry/, lean_gate*.jsonl,
rolling and alarm checkpoints, any single file over 2 GB.
Cap (CAP_GB, 30): whole tiers are copied in order; the tier that crosses the cap copies its newest
files that fit; everything after that is reported as over the cap and not copied, but a copy
already in DST is kept.
Mirror: a DST file no longer selected is deleted, except when that looks like a lost source (SRC
missing or empty, or a pass that would delete over 20 files and over 25% of DST's, or over 2 GB):
then nothing is deleted, the pass says why and exits 2 (--allow-mass-delete overrides).
Checkpoints and pickles modified in the last 2 minutes wait for the next pass (their DST copy
stays); other files are copied as they are and recopied when they change. Copies are atomic (temp
file + rename). One pass at a time: flock on DST/.backup.lock, a second pass waits up to 20 min.
"""

from __future__ import annotations

import fcntl
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

# BACKUP_SRC / BACKUP_DST / BACKUP_CAP_GB override the defaults for tests only.
SRC = Path(os.environ.get("BACKUP_SRC", "/scratch/cowifornia/nd-rl/think/runs"))
DST = Path(os.environ.get("BACKUP_DST", "/home/cowifornia/backups/nd-rl-think-ynez"))
CAP_GB = float(os.environ.get("BACKUP_CAP_GB", "30"))
FILE_CAP = 2e9
RECENT_S = 120  # a half-written checkpoint or pickle is useless: wait for it
RECENT_SUFFIX = (".pt", ".pkl")
LOCK_WAIT_S = 1200
MASS_N, MASS_FRAC, MASS_BYTES = 20, 0.25, 2e9
KEEP = {".backup.lock", ".last_pass.json"}  # DST bookkeeping: never mirrored or deleted
SKIP = re.compile(
    r"(^|/)(smoke|tmp|work|gen|raw|registry|src_snapshot)(/|$)|_work/|/v2_smoke/|INVALID|crashed"
    r"|(^|/)found_[^/]*\.jsonl$|(^|/)lean_gate[^/]*\.jsonl$|_rolling\.pt|_alarm\.pt|\.tmp-\d+$"
)
FIG = (".png", ".svg", ".pdf", ".html", ".md")


def files(d: Path, pattern: str = "*") -> list[Path]:
    if not d.is_dir():
        return []
    return [p for p in d.glob(pattern) if p.is_file() and not p.is_symlink()]


def tree(d: Path) -> list[Path]:
    return files(d, "**/*")


def run_meta(run: Path) -> list[Path]:
    out = []
    for p in tree(run):
        if p.relative_to(run).as_posix().startswith(("ckpts/", "registry/")):
            continue
        if p.suffix == ".json" or p.name.endswith(FIG) or re.match(r"metrics.*\.jsonl$", p.name):
            out.append(p)
    return out


def final_ckpts(run: Path) -> list[Path]:
    iters, rounds = 39, 1
    try:
        a = json.loads((run / "args.json").read_text())
        iters, rounds = int(a.get("iters", iters)), int(a.get("rounds", rounds))
    except (OSError, ValueError, TypeError, AttributeError):
        pass
    out = []
    for name in (f"{run.name}_it{iters}.pt", f"{run.name}_r{rounds}.pt"):
        p = run / "ckpts" / name
        out += [q for q in (p, p.with_name(p.name + ".upload.json")) if q.is_file()]
    return out


def pick() -> list[list[tuple[Path, int, float]]]:
    """-> the tiers, each a sorted list of (path, size, mtime)."""
    g = SRC / "grpo_v2"
    runs = [d for d in sorted(g.glob("*")) if d.is_dir() and d.name != "smoke"]
    t1 = files(SRC, "*.py") + files(SRC, "*.sh") + files(SRC, "*/*.sh") + files(g)
    t1 += tree(SRC / "logs") + [p for r in runs for p in run_meta(r)]
    t2 = files(SRC / "eval" / "v2", "*/*")
    t2 = [p for p in t2 if p.name.startswith(("rows_", "summary", "args"))]
    for d in ("matched", "tts", "charles"):
        t2 += tree(SRC / "eval" / d)
    t3 = tree(SRC / "pools") + files(SRC, "restart*/*.jsonl")
    t4 = files(SRC, "stage1/*/model.pt") + files(SRC, "stage1/*/*.json")
    t5 = [p for r in runs for p in final_ckpts(r)]
    t6 = files(SRC / "data", "*.pkl") + files(SRC / "data", "*.json")
    seen: set[Path] = set()
    tiers = []
    for t in (t1, t2, t3, t4, t5, t6):
        keep = []
        for p in sorted(t):
            if p in seen or SKIP.search(p.relative_to(SRC).as_posix()):
                continue
            try:
                st = p.stat()
            except OSError:  # gone since the listing
                continue
            if st.st_size <= FILE_CAP:
                seen.add(p)
                keep.append((p, st.st_size, st.st_mtime))
        tiers.append(keep)
    return tiers


def copy(p: Path, q: Path) -> None:
    q.parent.mkdir(parents=True, exist_ok=True)
    tmp = q.with_name(f".{q.name}.tmp-{os.getpid()}")
    try:
        shutil.copy2(p, tmp)
        os.replace(tmp, q)
    finally:
        if tmp.exists():
            tmp.unlink()


def lock_dst(stamp: str):
    """Hold DST/.backup.lock for this pass (wait for another pass); None after LOCK_WAIT_S."""
    lock = open(DST / ".backup.lock", "w")  # noqa: SIM115 (held until the process exits)
    t0 = time.time()
    while True:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return lock
        except OSError:
            if time.time() - t0 > LOCK_WAIT_S:
                print(f"{stamp} backup: lock held by another pass {LOCK_WAIT_S // 60} min: skipped")
                return None
            time.sleep(10)


def main() -> int:
    dry = "--dry" in sys.argv
    stamp = time.strftime("%Y-%m-%d %H:%MZ", time.gmtime())
    DST.mkdir(parents=True, exist_ok=True)
    lock = lock_dst(stamp)
    if lock is None:
        return 0
    if not SRC.is_dir():
        print(f"{stamp} backup: {SRC} is missing: nothing copied or deleted")
        return 2
    tiers = pick()
    cap, total = CAP_GB * 1e9, 0
    sel: list[tuple[Path, int, float]] = []
    over: list[tuple[Path, int, float]] = []
    for t in tiers:
        size = sum(x[1] for x in t)
        if not over and total + size <= cap:
            sel += t
            total += size
            continue
        for x in sorted(t, key=lambda x: -x[2]):  # the tier that crosses the cap: newest first
            if not over and total + x[1] <= cap:
                sel.append(x)
                total += x[1]
            else:
                over.append(x)  # this and every later file: not copied, an existing copy kept
    want = {DST / p.relative_to(SRC) for p, _, _ in sel + over}
    now, copied, recent, nbytes = time.time(), 0, 0, 0
    for p, size, mtime in sel:
        q = DST / p.relative_to(SRC)
        if q.exists() and q.stat().st_size == size and q.stat().st_mtime >= mtime:
            continue
        if p.suffix in RECENT_SUFFIX and now - mtime < RECENT_S:  # may still be being written
            recent += 1
            continue
        if not dry:
            try:
                copy(p, q)
            except FileNotFoundError:  # gone since the listing
                continue
        copied += 1
        nbytes += size
    have = [q for q in DST.rglob("*") if q.is_file() and q.relative_to(DST).as_posix() not in KEEP]
    gone = [q for q in have if q not in want]
    gone_bytes = sum(q.stat().st_size for q in gone)
    many = len(gone) > MASS_N and len(gone) > MASS_FRAC * len(have)
    mass = bool(gone) and (not sel or many or gone_bytes > MASS_BYTES)
    refused = mass and "--allow-mass-delete" not in sys.argv
    if not refused and not dry:
        for q in gone:
            q.unlink()
        for d in sorted((d for d in DST.rglob("*") if d.is_dir()), key=lambda d: -len(d.parts)):
            if not any(d.iterdir()):
                d.rmdir()
    removed = 0 if refused else len(gone)
    msg = (
        f"{stamp} backup{' (dry)' if dry else ''}: {len(sel)} files {total / 1e9:.2f} GB selected"
        f" (tiers {', '.join(str(len(t)) for t in tiers)});"
        f" copied {copied} ({nbytes / 1e9:.2f} GB),"
        f" {recent} left for the next pass (modified < {RECENT_S} s ago), removed {removed}"
    )
    if over:
        gb = sum(x[1] for x in over) / 1e9
        msg += f"; OVER THE {CAP_GB:g} GB CAP: {len(over)} files ({gb:.2f} GB) not copied"
    if refused:
        msg += (
            f"; REFUSED to delete {len(gone)} of {len(have)} backed-up files"
            f" ({gone_bytes / 1e9:.2f} GB): source lost? rerun with --allow-mass-delete if intended"
        )
    print(msg, flush=True)
    if not dry:
        rec = dict(time=stamp, selected=len(sel), gb=round(total / 1e9, 3), copied=copied)
        rec.update(removed=removed, over_cap=len(over), refused_delete=refused)
        (DST / ".last_pass.json").write_text(json.dumps(rec, indent=1) + "\n")
    lock.close()
    return 2 if refused else 0


if __name__ == "__main__":
    sys.exit(main())
