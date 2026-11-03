"""One GRPO line (L2 then the long L3) with its online evaluator on the same GPU. Stdlib only.

    python3 line.py --arm ko --seed 0 --t_l3 280 [--bar 21.3] [--pilot_l3 N --l3_init CKPT]

Runs from KO/tools on any machine of this experiment (PREREG.md, "GRPO" / "Measurement"):
- L2: grpo.py at fa28f6eb (+ the --snap_iters patch) with grpo_think_vs_s<S>_L2's argv, init and
  KL-anchor reference = KO/bases/<arm>_s<S>/model.pt. Output KO/runs/<arm>_s<S>_L2.
- L3: grpo.py at f35d3daf (+ patch) with grpo_think_vs_mb_kkl0_s<S>_L3's argv from L2's it39, with
  --iters T_L3 --save_every T_L3. Output KO/runs/<arm>_s<S>_L3. A partial L3 resumes from its
  newest rolling checkpoint (--resume); a partial L2 is moved aside and rerun.
- Evaluator: each grid snapshot in order of c is read on the strict 106 (eval.py at f35d3daf,
  plain, B64, T 0.8, k 64, batch 512, seed c) into KO/evals/<arm>_s<S>/c<cccc>/; after each, E32
  and R go to KO/evals/<arm>_s<S>/curve.jsonl and the stop rule is checked (with --bar): stop once
  the latest checkpoint is >= bar and R >= 1.1 x the R of the first checkpoint >= bar since the
  last one below it. A stop terminates the GRPO process; the evaluator finishes the snapshots
  already written.
`--pilot_l3 N --l3_init CKPT`: a timing pilot, L3 only, N iterations from CKPT (no evaluator
unless --pilot_eval), output KO/runs/pilot_<host>_L3.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import measure as M  # noqa: E402

KO = Path(os.environ.get("KO", "/scratch/cowifornia/nd-rl/knockout_grpo"))
PY = "/scratch/cowifornia/nd-rl/think/nd-rl/code/.venv/bin/python"
L2CODE = KO / "snap" / "fa28f6eb_snap"
L3CODE = KO / "snap" / "f35d3daf_snap"
EVCODE = KO / "snap" / "f35d3daf"
POOLS = KO / "pools"
SET = KO / "sets" / "strict106.jsonl"
HOST = socket.gethostname().split(".")[0]


def now() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def say(msg: str) -> None:
    print(f"{now()} {msg}", flush=True)


def l2_argv(arm: str, s: int, snaps: list[int]) -> list[str]:
    base = KO / "bases" / f"{arm}_s{s}" / "model.pt"
    name = f"{arm}_s{s}_L2"
    return [
        "--init", str(base), "--name", name, "--out", str(KO / "runs" / name),
        "--targets", str(POOLS / "L2_pool.jsonl"), "--iters", "39", "--prompts", "256",
        "--group", "16", "--batch", "1024", "--lr", "1.5e-4", "--overlong", "fail",
        "--save_every", "39", "--max_steps", "96", "--seed", str(s), "--keep_rolling", "100",
        "--effort", "--budgets", "0,64,256", "--canary", str(POOLS / "canary_long.jsonl"),
        "--canary_n", "1000", "--canary_every", "1", "--rollback_on_alarm",
        "--max_rollbacks", "3", "--ban_restore_check", "--ban_stop", "0.005",
        "--kl_anchor_coef", "0.02", "--kl_anchor_ref", str(base), "--monitor",
        "--arm", "logical", "--max_rej", "0", "--rej_penalty", "0.1",
        "--snap_iters", ",".join(map(str, snaps)),
    ]  # fmt: skip


def l3_argv(name: str, init: str, s: int, iters: int, snaps: list[int]) -> list[str]:
    return [
        "--init", init, "--name", name, "--out", str(KO / "runs" / name),
        "--targets", str(POOLS / "L3_pool.jsonl"), "--iters", str(iters), "--prompts", "256",
        "--group", "16", "--batch", "1024", "--lr", "1.5e-4", "--lr_schedule", "wsd",
        "--decay_frac", "0.2", "--min_lr_frac", "0.3", "--decay_shape", "sqrt",
        "--unlikely_beta", "0.25", "--overlong", "fail", "--save_every", str(iters),
        "--max_steps", "96", "--seed", str(s), "--keep_rolling", "100", "--effort",
        "--budgets", "0,64,256", "--canary", str(POOLS / "canary_long.jsonl"),
        "--canary_n", "1000", "--canary_every", "1", "--rollback_on_alarm",
        "--max_rollbacks", "3", "--ban_restore_check", "--ban_stop", "0.005",
        "--kl_anchor_coef", "0", "--kl_anchor_ref", init, "--monitor", "--arm", "logical",
        "--max_rej", "0", "--rej_penalty", "0.1",
        "--targets_extra", str(POOLS / "L3_3p.jsonl"), "--targets_extra_frac", "0.33",
        "--snap_iters", ",".join(map(str, snaps)),
    ]  # fmt: skip


class Line:
    def __init__(self, a: argparse.Namespace) -> None:
        self.a = a
        self.arm, self.s = a.arm, a.seed
        self.src = f"{a.arm}_s{a.seed}"  # the base and the L2 run (shared by an --x2 line)
        self.tag = self.src + ("_x2" if a.x2 else "")
        self.c_max = 39 + a.t_l3
        self.grid = M.grid(self.c_max)
        self.l2 = KO / "runs" / f"{self.src}_L2"
        self.l3 = KO / "runs" / f"{self.tag}_L3"
        self.ev = KO / "evals" / self.tag
        self.ev.mkdir(parents=True, exist_ok=True)
        self.stop = threading.Event()
        self.training_done = threading.Event()
        self.proc: subprocess.Popen | None = None
        self.lock = threading.Lock()
        s1 = KO / "bases" / self.src / "stage1_tokens.json"
        self.s1_tokens = json.loads(s1.read_text())["tokens"]

    # ---------------------------------------------------------------- training
    def run_stage(self, code: Path, argv: list[str], log: Path) -> int:
        if self.stop.is_set():
            return 0
        say(f"{self.tag}: launch {argv[argv.index('--name') + 1]} ({code.name})")
        with open(log, "a") as f:
            p = subprocess.Popen(
                [PY, "-m", "code.thinking.grpo", *argv],
                cwd=str(code),
                stdout=f,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        with self.lock:
            self.proc = p
        while p.poll() is None:
            if self.stop.is_set():
                say(f"{self.tag}: stop rule met: terminating pid {p.pid}")
                os.killpg(p.pid, signal.SIGTERM)
                try:
                    p.wait(timeout=300)
                except subprocess.TimeoutExpired:
                    os.killpg(p.pid, signal.SIGKILL)
                break
            time.sleep(10)
        rc = p.wait()
        with self.lock:
            self.proc = None
        say(f"{self.tag}: stage exit rc {rc}")
        return rc

    def rollings(self, d: Path, name: str) -> list[tuple[int, Path]]:
        """(iteration, path) of the stage's rolling checkpoints (with training state), sorted."""
        out = []
        for p in (d / "ckpts").glob(f"{name}_it*_rolling.pt"):
            out.append((int(p.name.split("_it")[1].split("_")[0]), p))
        return sorted(out)

    def splice(self) -> None:
        """--resume_line (amendment): a line the first stop rule ended continues from its newest
        rolling checkpoint (exact training state); the snapshots, reads and curve rows after it
        are moved aside (suffix .rule1) so the curve stays one trajectory."""
        stop = self.ev / "STOP_LINE"
        if stop.exists():
            stop.rename(self.ev / "STOP_LINE.rule1")
        done = self.ev / "line_done.json"
        if done.exists():
            done.rename(self.ev / "line_done.rule1.json")
        n2 = self.n_rows(self.l2)
        if n2 < 39:
            r = self.rollings(self.l2, f"{self.src}_L2")
            c_r = r[-1][0] if r else 0
        else:
            r = self.rollings(self.l3, f"{self.tag}_L3")
            c_r = 39 + (r[-1][0] if r else 0)
        moved = []
        for c in self.grid:
            if c <= c_r:
                continue
            sp = self.snap_path(c)
            if sp.exists():
                sp.rename(sp.with_name(sp.name + ".rule1"))
                moved.append(c)
            ed = self.ev / f"c{c:04d}"
            if ed.exists():
                ed.rename(ed.with_name(ed.name + ".rule1"))
        rows = self.curve_rows()
        keep = [x for x in rows if x["c"] <= c_r]
        gone = [x for x in rows if x["c"] > c_r]
        with open(self.ev / "curve_rule1_discarded.jsonl", "a") as f:
            for x in gone:
                f.write(json.dumps(x) + "\n")
        (self.ev / "curve.jsonl").write_text("".join(json.dumps(x) + "\n" for x in keep))
        say(
            f"{self.tag}: resume_line from c {c_r}; snapshots moved aside {moved}; "
            f"{len(gone)} curve rows discarded"
        )

    def splice_partial(self) -> None:
        """--resume_line on a line that crashed (no stop, not done): resume from the newest
        rolling checkpoint with the same splice."""
        self.splice()

    def n_rows(self, d: Path) -> int:
        m = d / "metrics.jsonl"
        return sum(1 for x in m.read_text().splitlines() if x.strip()) if m.exists() else 0

    def train(self) -> None:
        L2snaps = [c for c in self.grid if c <= 39]
        L3snaps = [c - 39 for c in self.grid if c > 39]
        it39 = self.l2 / "ckpts" / f"{self.src}_L2_it39.pt"
        if self.a.x2 and not it39.exists():
            raise SystemExit(f"--x2: {it39} missing (the source line's L2 must be done)")
        if not it39.exists():
            argv2 = l2_argv(self.arm, self.s, L2snaps)
            if self.l2.exists() and self.n_rows(self.l2) > 0:
                rolls = self.rollings(self.l2, f"{self.src}_L2")
                if rolls:
                    argv2 += ["--resume", str(rolls[-1][1])]
                    say(f"{self.tag}: resuming L2 from {rolls[-1][1].name}")
                else:
                    aside = self.l2.with_name(self.l2.name + f".partial_{int(time.time())}")
                    self.l2.rename(aside)
                    say(f"{self.tag}: partial L2 moved to {aside}")
            rc = self.run_stage(L2CODE, argv2, KO / "logs" / f"{self.src}_L2.log")
            if rc == 3:
                say(f"{self.tag}: L2 think-ban stop")
                return
            if rc not in (0, -15) and not self.stop.is_set():
                raise SystemExit(f"L2 failed rc {rc}")
        if self.stop.is_set():
            return
        name = f"{self.tag}_L3"
        argv = l3_argv(name, str(it39), self.s, self.a.t_l3, L3snaps)
        done = self.l3 / "ckpts" / f"{name}_it{self.a.t_l3}.pt"
        if done.exists():
            return
        if self.n_rows(self.l3) > 0:
            rolls = self.rollings(self.l3, name)
            if rolls:
                argv += ["--resume", str(rolls[-1][1])]
                say(f"{self.tag}: resuming L3 from {rolls[-1][1].name}")
        rc = self.run_stage(L3CODE, argv, KO / "logs" / f"{self.tag}_L3.log")
        if rc == 3:
            say(f"{self.tag}: L3 think-ban stop")

    # --------------------------------------------------------------- evaluator
    def snap_path(self, c: int) -> Path:
        if c <= 39:
            return self.l2 / "ckpts" / f"{self.src}_L2_it{c}_snap.pt"
        return self.l3 / "ckpts" / f"{self.tag}_L3_it{c - 39}_snap.pt"

    def last_c(self) -> int:
        n2 = self.n_rows(self.l2)
        return n2 + (self.n_rows(self.l3) if n2 >= 39 else 0)

    def curve_rows(self) -> list[dict]:
        p = self.ev / "curve.jsonl"
        return (
            [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []
        )

    def r_at(self, c: int) -> dict:
        rows = M.cumulative([self.l2, self.l3] if (self.l3 / "args.json").exists() else [self.l2])
        r = next(x for x in rows if x["c"] == c)
        return {
            "R": M.ratio(r, self.s1_tokens),
            "secs": r["secs"],
            "sampled": r["sampled"],
            "trained": r["trained"],
        }

    def eval_one(self, c: int) -> dict:
        out = self.ev / f"c{c:04d}"
        if out.exists() and not (out / "summary.json").exists():
            out.rename(out.with_name(out.name + f".partial_{int(time.time())}"))
        if not (out / "summary.json").exists():
            cmd = [
                PY, "-m", "code.thinking.eval", "--ckpt", str(self.snap_path(c)), "--sets", str(SET),  # noqa: E501
                "--out", str(out), "--k", "64", "--budgets", "64", "--arms", "plain",
                "--batch", str(self.a.eval_batch), "--seed", str(c), "--temperature", "0.8",
                "--max_action", "512", "--max_steps", "96", "--max_rej", "10",
            ]  # fmt: skip
            t0 = time.time()
            with open(KO / "logs" / f"{self.tag}_eval.log", "a") as f:
                rc = subprocess.run(
                    cmd, cwd=str(EVCODE), stdout=f, stderr=subprocess.STDOUT
                ).returncode
            if rc != 0:
                raise RuntimeError(f"eval c{c} rc {rc}")
            (out / "eval_secs.txt").write_text(f"{time.time() - t0:.1f}\n")
        e = M.e32(M.read_rows(out))
        row = {
            "c": c,
            **self.r_at(c),
            "e32": e["e32"],
            "first32": e["first32"],
            "solved_any64": e["solved_any"],
            "host": HOST,
            "utc": now(),
        }
        return row

    def stop_rule(self, rows: list[dict]) -> bool:
        """Amendment (log.md, 2026-11-03 ~07:55Z): stop once the line has been at or above the bar
        for >= 3 consecutive evaluated checkpoints AND the latest is at >= 2 x the R of the first
        of that streak. A checkpoint below the bar resets the streak."""
        bar = self.a.bar
        if bar is None:
            return False
        streak: list[dict] = []
        for r in rows:
            streak = [*streak, r] if r["e32"] >= bar else []
        return len(streak) >= 3 and streak[-1]["R"] >= 2.0 * streak[0]["R"]

    def evaluator(self) -> None:
        done = {r["c"] for r in self.curve_rows()}
        if self.a.x2:  # the shared L2 part: the source line's own reads
            src = KO / "evals" / self.src / "curve.jsonl"
            have = (
                [json.loads(x) for x in src.read_text().splitlines() if x.strip()]
                if src.exists()
                else []
            )
            with open(self.ev / "curve.jsonl", "a") as f:
                for r in have:
                    if r["c"] <= 39 and r["c"] not in done:
                        f.write(json.dumps({**r, "from": self.src}) + "\n")
                        done.add(r["c"])
        for c in self.grid:
            if c in done:
                continue
            while True:  # wait for the snapshot, or for training to end before reaching it
                p = self.snap_path(c)
                if p.exists() and self.last_c() >= c:
                    s0 = p.stat().st_size
                    time.sleep(5)
                    if p.stat().st_size == s0:
                        break
                if self.training_done.is_set() and self.last_c() < c:
                    say(f"{self.tag}: evaluator done (training ended at c {self.last_c()})")
                    return
                if self.training_done.is_set() and not p.exists():
                    say(f"{self.tag}: snapshot c {c} missing after training; skipped")
                    break
                time.sleep(15)
            if not self.snap_path(c).exists():
                continue
            row = None
            for attempt in range(3):  # a failed read is retried twice, then skipped (logged)
                try:
                    row = self.eval_one(c)
                    break
                except Exception as e:  # noqa: BLE001
                    say(f"{self.tag}: EVAL-ERROR c {c} attempt {attempt}: {e!r}")
                    time.sleep(60)
            if row is None:
                continue
            with open(self.ev / "curve.jsonl", "a") as f:
                f.write(json.dumps(row) + "\n")
            say(f"{self.tag}: c {c} R {row['R']:.4f} E32 {row['e32']:.2f} first32 {row['first32']}")
            if not self.stop.is_set() and self.stop_rule(self.curve_rows()):
                (self.ev / "STOP_LINE").write_text(json.dumps(row) + "\n")
                say(f"{self.tag}: STOP at c {c} (bar {self.a.bar})")
                self.stop.set()

    def run(self) -> None:
        if self.a.resume_line and (self.ev / "STOP_LINE").exists():
            rows = self.curve_rows()
            if rows and self.stop_rule(rows):  # the amended window already holds: nothing to do
                say(f"{self.tag}: resume_line: the hold window already holds at c {rows[-1]['c']}")
            elif self.a.bar is not None:
                self.splice()
        elif self.a.resume_line and not (self.ev / "line_done.json").exists():
            self.splice_partial()
        if (self.ev / "STOP_LINE").exists():
            self.stop.set()
        th = threading.Thread(target=self.evaluator, daemon=True)
        if not self.a.no_eval:
            th.start()
        try:
            self.train()
        finally:
            self.training_done.set()
        if not self.a.no_eval:
            th.join()
        summ = {
            "tag": self.tag,
            "t_l3": self.a.t_l3,
            "bar": self.a.bar,
            "host": HOST,
            "stopped": self.stop.is_set(),
            "last_c": self.last_c(),
            "utc": now(),
        }
        (self.ev / "line_done.json").write_text(json.dumps(summ, indent=1))
        say(f"{self.tag}: line done {summ}")


def pilot(a: argparse.Namespace) -> None:
    name = f"pilot_{HOST}_g{os.environ.get('KO_GPU', 'x')}_L3"
    out = KO / "runs" / name
    argv = l3_argv(name, a.l3_init, a.seed, 39, list(range(1, a.pilot_l3 + 1)))
    argv[argv.index("--iters") + 1] = str(a.pilot_l3)
    ev_th = None
    if a.pilot_eval:

        def ev() -> None:  # read the pilot's own it1 snapshot repeatedly while training runs
            snap = out / "ckpts" / f"{name}_it1_snap.pt"
            while not snap.exists():
                time.sleep(10)
            time.sleep(10)
            i = 0
            while ev_on.is_set():
                o = KO / "evals" / f"{name}" / f"rep{i}"
                cmd = [
                    PY, "-m", "code.thinking.eval", "--ckpt", str(snap), "--sets", str(SET),
                    "--out", str(o), "--k", "64", "--budgets", "64", "--arms", "plain",
                    "--batch", str(a.eval_batch), "--seed", str(i), "--temperature", "0.8",
                    "--max_action", "512", "--max_steps", "96", "--max_rej", "10",
                ]  # fmt: skip
                t0 = time.time()
                with open(KO / "logs" / f"{name}_eval.log", "a") as f:
                    subprocess.run(cmd, cwd=str(EVCODE), stdout=f, stderr=subprocess.STDOUT)
                say(f"pilot eval rep{i}: {time.time() - t0:.0f} s")
                (o / "eval_secs.txt").write_text(f"{time.time() - t0:.1f}\n")
                i += 1

        ev_on = threading.Event()
        ev_on.set()
        ev_th = threading.Thread(target=ev, daemon=True)
        ev_th.start()
    t0 = time.time()
    rc = subprocess.run(
        [PY, "-m", "code.thinking.grpo", *argv],
        cwd=str(L3CODE),
        stdout=open(KO / "logs" / f"{name}.log", "a"),  # noqa: SIM115
        stderr=subprocess.STDOUT,
    ).returncode
    say(f"pilot {name}: rc {rc} wall {time.time() - t0:.0f} s")
    if ev_th is not None:
        ev_on.clear()
        ev_th.join()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="ko")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--t_l3", type=int, default=39)
    ap.add_argument("--bar", type=float, default=None)
    ap.add_argument("--eval_batch", type=int, default=512)
    ap.add_argument("--no_eval", action="store_true")
    ap.add_argument("--x2", action="store_true", help="L3 only, from the source line's L2 it39")
    ap.add_argument(
        "--resume_line", action="store_true", help="continue a line the first rule stopped"
    )
    ap.add_argument("--pilot_l3", type=int, default=0)
    ap.add_argument("--pilot_eval", action="store_true")
    ap.add_argument("--l3_init", default=None)
    a = ap.parse_args()
    if a.pilot_l3:
        pilot(a)
    else:
        Line(a).run()


if __name__ == "__main__":
    main()
