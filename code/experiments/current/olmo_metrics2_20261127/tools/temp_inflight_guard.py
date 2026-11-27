"""In-job guard for M1 sampling cells (runs on the pod inside the job command, in the background, next to the sampler).

Dan's sampler writes FINAL_ONLY rows (with --chunk 1: one row per finished draw), so in-flight lengths are not visible
in the output. The guard reads two things only:
  1. the job's own log (logs/job.log) for vLLM's periodic stats line, parsed tolerantly:
        "... Avg generation throughput: 1234.5 tokens/s, Running: 256 reqs, Waiting: 0 reqs, ..."
     (V1 format is "Engine 000: Avg prompt throughput: ... tokens/s, Avg generation throughput: ... tokens/s, Running: N
     reqs, Waiting: M reqs, ..." ; unverified for vLLM 0.11.0, hence the fallback below);
  2. the samples.jsonl.gz written so far: number of rows and the SUM of their n_tok (finished tokens).
G(t) = integral of the generation throughput = tokens generated so far (finished + in flight), about poll granularity.
Abort (SIGTERM to the sampler and its children, SIGKILL 90 s later) when
  (a) G > FACTOR x plan_tokens before 90 % of the planned draws have rows;
  (b) waiting == 0 and running >= 10 % of the planned draws and (G - finished_tokens) / running >= 16,384;
  (c) no new row for STALL_MIN minutes while the sampler is alive (a first-row grace of FIRST_ROW_GRACE_MIN from start,
      because with FINAL_ONLY the first draw can take about 14 minutes to finish);
  (d) BLIND fallback: if no stats line has parsed after BLIND_AFTER_S seconds, the format is unknown, so a mechanical
      wall-clock cap applies instead of (a) and (b): abort when generation time (elapsed minus LOAD_ALLOW_S) exceeds
      FACTOR x plan_tokens / ASSUMED_TPS before 90 % of the planned draws have rows. (c) stays on.
Writes one GUARD line every 5 minutes and every event to --out (default next to the log). It never edits the samples.
After an abort the job command must still grade the rows that exist (the run_steps use `;` not `&&` there).
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import signal
import sys
import time

FACTOR = 1.5
INFLIGHT_LEN = 16384
STALL_MIN = 20.0
FIRST_ROW_GRACE_MIN = 40.0
BLIND_AFTER_S = 900.0
LOAD_ALLOW_S = 600.0
ASSUMED_TPS = 1500.0

_GEN = re.compile(r"generation throughput:\s*([0-9.]+)\s*tokens/s", re.I)
_RUN = re.compile(r"Running:\s*(\d+)\s*reqs?", re.I)
_WAIT = re.compile(r"Waiting:\s*(\d+)\s*reqs?", re.I)


def parse_stats(line: str):
    """-> (gen_tps, running, waiting) or None. Tolerant: any order of fields, Waiting optional (None)."""
    g = _GEN.search(line)
    r = _RUN.search(line)
    if not g or not r:
        return None
    w = _WAIT.search(line)
    return float(g.group(1)), int(r.group(1)), (int(w.group(1)) if w else None)


def read_rows(path: str) -> tuple[int, int]:
    """(rows, summed n_tok) of a possibly truncated samples.jsonl[.gz]."""
    if not os.path.exists(path):
        return 0, 0
    op = gzip.open if path.endswith(".gz") else open
    n = tok = 0
    try:
        with op(path, "rt") as f:
            for ln in f:
                try:
                    r = json.loads(ln)
                except json.JSONDecodeError:
                    break
                n += 1
                tok += int(r.get("n_tok", 0))
    except (EOFError, OSError):
        pass
    return n, tok


class Guard:
    def __init__(
        self,
        plan_draws: int,
        plan_tokens: float,
        stall_min=STALL_MIN,
        factor=FACTOR,
        inflight_len=INFLIGHT_LEN,
        first_row_grace_min=FIRST_ROW_GRACE_MIN,
        blind_after_s=BLIND_AFTER_S,
        load_allow_s=LOAD_ALLOW_S,
        assumed_tps=ASSUMED_TPS,
    ):
        self.plan_draws, self.plan_tokens = plan_draws, plan_tokens
        self.stall_s, self.factor, self.inflight_len = stall_min * 60, factor, inflight_len
        self.grace_s, self.blind_after_s, self.load_allow_s, self.tps = (
            first_row_grace_min * 60,
            blind_after_s,
            load_allow_s,
            assumed_tps,
        )
        self.t0 = None
        self.G = 0.0
        self.running = None
        self.waiting = None
        self.parsed = 0
        self.rows = 0
        self.fin_tok = 0
        self.last_row_time = None
        self.last_t = None

    def step(self, now: float, new_lines: list[str], rows: int, fin_tok: int) -> list[str]:
        """Advance with the lines read since the last call and the current row summary; return abort reasons (maybe [])."""
        if self.t0 is None:
            self.t0 = self.last_t = now
            self.last_row_time = now
        stats = [s for s in (parse_stats(ln) for ln in new_lines) if s]
        if stats:
            dt = max(now - self.last_t, 0.0) / len(stats)
            for tps, run, wait in stats:
                self.G += tps * dt
                self.running, self.waiting = run, wait
            self.parsed += len(stats)
        self.last_t = now
        if rows > self.rows:
            self.last_row_time = now
        self.rows, self.fin_tok = rows, fin_tok
        out: list[str] = []
        blind = self.parsed == 0 and (now - self.t0) >= self.blind_after_s
        if not blind:
            if self.factor * self.plan_tokens < self.G and rows < 0.9 * self.plan_draws:
                out.append(
                    f"(a) G={self.G:.0f} > {self.factor} x plan {self.plan_tokens:.0f} with {rows}/{self.plan_draws} rows"
                )
            if (
                self.waiting == 0
                and self.running
                and self.running >= 0.1 * self.plan_draws
                and (self.G - fin_tok) / self.running >= self.inflight_len
            ):
                out.append(
                    f"(b) running={self.running} waiting=0 mean in-flight {(self.G - fin_tok) / self.running:.0f} "
                    f">= {self.inflight_len}"
                )
        else:
            gen_s = (now - self.t0) - self.load_allow_s
            cap_s = self.factor * self.plan_tokens / self.tps
            if gen_s > cap_s and rows < 0.9 * self.plan_draws:
                out.append(
                    f"(d) BLIND wall-clock: generation {gen_s:.0f}s > {cap_s:.0f}s with {rows}/{self.plan_draws} rows"
                )
        limit = self.stall_s if rows > 0 else max(self.stall_s, self.grace_s)
        if rows < self.plan_draws and now - self.last_row_time >= limit:
            out.append(
                f"(c) no new row for {(now - self.last_row_time) / 60:.1f} min (rows={rows})"
            )
        return out

    def status(self, now: float) -> str:
        return (
            f"GUARD t={(now - (self.t0 or now)) / 60:.1f}min G={self.G:.0f} rows={self.rows}/{self.plan_draws} "
            f"fin_tok={self.fin_tok} running={self.running} waiting={self.waiting} parsed_lines={self.parsed}"
            f"{' BLIND' if self.parsed == 0 and (now - (self.t0 or now)) >= self.blind_after_s else ''}"
        )


def alive(pid: int) -> bool:
    """Process exists and is not a zombie (an unreaped child counts as gone)."""
    try:
        with open(f"/proc/{pid}/stat") as f:
            return f.read().rsplit(")", 1)[1].split()[0] != "Z"
    except OSError:
        return False


def children_of(pid: int) -> list[int]:
    kids = []
    for d in os.listdir("/proc"):
        if d.isdigit():
            try:
                with open(f"/proc/{d}/stat") as f:
                    ppid = int(f.read().rsplit(")", 1)[1].split()[1])
            except (OSError, ValueError, IndexError):
                continue
            if ppid == pid:
                kids.append(int(d))
    return kids


def tree(pid: int) -> list[int]:
    out = [pid]
    for k in children_of(pid):
        out += tree(k)
    return out


def terminate(pid: int, wait_s: float = 90.0) -> None:
    pids = tree(pid)
    for p in pids:
        try:
            os.kill(p, signal.SIGTERM)
        except OSError:
            pass
    end = time.time() + wait_s
    while time.time() < end and any(alive(p) for p in pids):
        time.sleep(2)
    for p in pids:
        try:
            os.kill(p, signal.SIGKILL)
        except OSError:
            pass


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--log", required=True, help="the job's logs/job.log")
    ap.add_argument("--samples", required=True)
    ap.add_argument("--plan-draws", type=int, required=True)
    ap.add_argument("--plan-tokens-per-draw", type=float, required=True)
    ap.add_argument(
        "--target-pid",
        type=int,
        required=True,
        help="the sampler's pid (killed with its children on abort)",
    )
    ap.add_argument("--poll", type=float, default=60.0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-abort", action="store_true", help="log only")
    ap.add_argument(
        "--assumed-tps",
        type=float,
        default=ASSUMED_TPS,
        help="tokens/s for the blind wall-clock fallback (d); orchestrator addition 2026-11-27: the RTX PRO 6000 pod is slower than the H200 the default assumed",
    )
    a = ap.parse_args(argv)
    out = a.out or (a.log + ".guard")
    g = Guard(a.plan_draws, a.plan_draws * a.plan_tokens_per_draw, assumed_tps=a.assumed_tps)
    off = 0
    last_print = 0.0

    def emit(msg):
        with open(out, "a") as f:
            f.write(time.strftime("%FT%TZ ", time.gmtime()) + msg + "\n")

    emit(
        f"GUARD_START plan_draws={a.plan_draws} plan_tokens={g.plan_tokens:.0f} target_pid={a.target_pid}"
    )
    while alive(a.target_pid):
        now = time.time()
        lines: list[str] = []
        if os.path.exists(a.log):
            with open(a.log, "rb") as f:
                f.seek(off)
                chunk = f.read()
                off += len(chunk)
            lines = chunk.decode("utf-8", "replace").splitlines()
        rows, tok = read_rows(a.samples)
        reasons = g.step(now, lines, rows, tok)
        if now - last_print >= 300:
            emit(g.status(now))
            last_print = now
        if reasons:
            emit("GUARD_ABORT " + " | ".join(reasons) + (" (log only)" if a.no_abort else ""))
            if not a.no_abort:
                terminate(a.target_pid)
                emit("GUARD_TERMINATED")
                return 3
        time.sleep(a.poll)
    emit(g.status(time.time()) + " target exited")
    return 0


if __name__ == "__main__":
    sys.exit(main())
