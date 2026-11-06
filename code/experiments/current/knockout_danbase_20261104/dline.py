"""One segment of one line on one GPU: GRPO plus the online reads (started by sup.py).

    python3 dline.py --line dko_s0 --seg 1 --gpu 0 [--resume CKPT] [--iters 1200]

Runs `code.thinking.grpo` (common.grpo_argv) in `KD/runs/<line>/seg<k>/` with LEAN_GATE_DUMP
there, and a reader that reads every model-only snapshot (`ckpts/<line>_it<i>_snap.pt`, every 10
iterations) and the segment's start point (the base, or the resumed checkpoint's weights, once
per line: seg 1's iteration 0) with Dan's code (danread.py: k 64 on the 106, T 0.8, 256 / 48,
noise seed `seed*1000 + i + 500`, batch 256) into `reads/it<i>.json`. Control files in the
segment (written by sup.py): `STOP` (end now), `STOP_AT_CKPT` (end right after the next rolling
checkpoint, to move the line). Writes `status.json` (heartbeat, every 30 s) and on the end
`end.json` {how: done | stopped | moved | failed, rc}. Its own GPU check every 60 s: another
user's process on this GPU -> STOP_AT_CKPT.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import gpu as G  # noqa: E402

HOST = os.uname().nodename.split(".")[0]
ALLOC = "expandable_segments:True,garbage_collection_threshold:0.6"


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class Seg:
    def __init__(self, a: argparse.Namespace) -> None:
        self.a = a
        self.L = C.LINES[a.line]
        self.dir = C.KD / "runs" / a.line / f"seg{a.seg}"
        (self.dir / "reads").mkdir(parents=True, exist_ok=True)
        self.log = C.KD / "logs" / f"{a.line}_seg{a.seg}.log"
        self.proc: subprocess.Popen | None = None
        self.done = threading.Event()
        self.reading = threading.Lock()
        self.why = "running"
        self.fails: dict[int, int] = {}

    def say(self, msg: str) -> None:
        with open(C.KD / "logs" / f"{self.a.line}.events.log", "a") as f:
            f.write(f"{now()} {HOST} seg{self.a.seg}: {msg}\n")

    def env(self) -> dict:
        e = dict(os.environ)
        e.update(ND_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1", LEAN_GATE_WORKERS="8",
                 CUDA_VISIBLE_DEVICES=str(self.a.gpu),
                 LEAN_GATE_DUMP=str(self.dir / "lean_dump.jsonl"),
                 # GRPO and a read share the GPU: the allocator returns cached blocks early
                 # (dko_s0 seg1 died of an OOM at 20:58Z with a read running)
                 PYTORCH_CUDA_ALLOC_CONF=ALLOC)  # fmt: skip
        return e

    # ------------------------------------------------------------------ reads
    def read(self, ckpt: Path, it: int) -> None:
        out = self.dir / "reads" / f"it{it:04d}.json"
        if out.exists():
            return
        cmd = [C.PY, str(C.HERE / "danread.py"), "--ckpt", str(ckpt), "--seed",
               str(self.L["seed"]), "--round", str(it), "--out", str(out), "--only106",
               "--k", "64", "--batch", "128"]  # fmt: skip
        env = {**self.env(), "DANREAD_MEM_FRAC": "0.3"}  # a read never takes GRPO's memory
        with self.reading, open(self.dir / "reads.log", "a") as f:
            p = subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT, timeout=3600)
        self.say(f"read it {it}: rc {p.returncode}")
        if p.returncode != 0:
            self.fails[it] = self.fails.get(it, 0) + 1
            if self.fails[it] >= 3:  # given up: a gap in the curve, marked
                (self.dir / "reads" / f"it{it:04d}.failed").write_text(now())

    def reader(self) -> None:
        base = self.a.seg == 1 and not self.a.resume  # the base, once per line (retried)
        while True:
            fin = self.done.is_set()
            if (
                base
                and not (self.dir / "reads" / "it0000.json").exists()
                and not (self.dir / "reads" / "it0000.failed").exists()
            ):
                self.read(C.KD / "bases" / self.L["base"], 0)
            snaps = sorted((self.dir / "ckpts").glob(f"{self.a.line}_it*_snap.pt"),
                           key=lambda p: int(p.name.split("_it")[-1].split("_")[0]))  # fmt: skip
            for sp in snaps:
                it = int(sp.name.split("_it")[-1].split("_")[0])
                if (
                    not (self.dir / "reads" / f"it{it:04d}.json").exists()
                    and not (self.dir / "reads" / f"it{it:04d}.failed").exists()
                ):
                    try:
                        self.read(sp, it)
                    except subprocess.TimeoutExpired:
                        self.say(f"read it {it}: timeout")
            if fin:
                return
            time.sleep(20)

    # ------------------------------------------------------------------ watch
    def newest_rolling(self) -> float:
        ps = [
            *(self.dir / "ckpts").glob(f"{self.a.line}_it*_rolling.pt"),
            *(self.dir / "ckpts").glob(f"{self.a.line}_it*_alarm.pt"),
        ]  # both hold the state
        return max((p.stat().st_mtime for p in ps), default=0.0)

    def watch(self) -> None:
        t_gpu = 0.0
        while self.proc is not None and self.proc.poll() is None:
            st = {"utc": now(), "host": HOST, "gpu": self.a.gpu, "pid": self.proc.pid,
                  "seg": self.a.seg, "why": self.why}  # fmt: skip
            (self.dir / "status.json").write_text(json.dumps(st))
            if (self.dir / "STOP").exists():
                self.why = "stopped"
                self.kill()
                break
            sac = self.dir / "STOP_AT_CKPT"
            if sac.exists() and self.newest_rolling() > sac.stat().st_mtime:
                self.why = "moved"
                self.say("STOP_AT_CKPT: a rolling checkpoint is written; ending to move")
                self.kill()
                break
            if time.time() - t_gpu > 60:
                t_gpu = time.time()
                g = G.gpus(HOST)
                if g is not None and (g.get(self.a.gpu, set()) - {G.ME}) and not sac.exists():
                    self.say(f"another user's process on GPU {self.a.gpu}: {g[self.a.gpu]}")
                    sac.touch()
            time.sleep(30)

    def kill(self) -> None:
        if self.proc is None:
            return
        try:
            os.killpg(self.proc.pid, signal.SIGTERM)
            self.proc.wait(timeout=300)
        except subprocess.TimeoutExpired:
            os.killpg(self.proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    def run(self) -> None:
        argv = C.grpo_argv(self.a.line, self.dir, self.a.iters, self.a.resume)
        self.say(f"launch GPU {self.a.gpu} resume {self.a.resume}")
        rt = threading.Thread(target=self.reader, daemon=True)
        rt.start()
        with open(self.log, "a") as f:
            self.proc = subprocess.Popen([C.PY, "-m", "code.thinking.grpo", *argv],
                                         cwd=str(C.CODE), stdout=f, stderr=subprocess.STDOUT,
                                         start_new_session=True, env=self.env())  # fmt: skip
        self.watch()
        rc = self.proc.wait()
        if self.why == "running":
            # 4: grpo.py's NONFINITE_EXIT (--stop_nonfinite), a result, not a failure to respawn
            self.why = "done" if rc == 0 else "nonfinite" if rc == 4 else "failed"
        self.done.set()
        rt.join(timeout=3 * 3600)  # the last snapshots' reads
        (self.dir / "end.json").write_text(json.dumps({"how": self.why, "rc": rc, "utc": now(),
                                                       "host": HOST}))  # fmt: skip
        self.say(f"end: {self.why} (rc {rc})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--line", required=True, choices=sorted(C.LINES))
    ap.add_argument("--seg", type=int, required=True)
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--iters", type=int, default=1200)
    a = ap.parse_args()
    if not G.free_for_us(HOST, a.gpu):
        raise SystemExit(f"GPU {a.gpu} on {HOST} has another user's process: not launching")
    Seg(a).run()


if __name__ == "__main__":
    main()
