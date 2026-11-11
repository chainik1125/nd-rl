"""One segment of one tagged RL line on one GPU: GRPO under the line's tag, the online reads, the stop rule.

    python tag_line.py --line tB_s0 --gpu N [--seg 1] [--resume CKPT] [--iters 400] [--dry-run]
    python tag_line.py --line tB_p10_s0 ...     # a variant line: the same, on the variant's base (tag_core.LINES)

Modelled on knockout_danbase_20261104/dline.py (d78d12a), without the supervisor: this process applies the stop rule
itself (tag_measure.stop_reason) and nothing respawns a dead segment. To continue a line that ended early (a crash, or
another user's process on the GPU), run the next segment from its newest rolling checkpoint:
`--seg k+1 --resume AT/runs/<line>/seg<k>/ckpts/<line>_it<i>_rolling.pt`.

  * GRPO: tag_grpo.py --tag T -- <the dd1_s<seed> recipe> (knockout common.grpo_argv of the 1 % line of the same seed,
    with only --init, --name and --kl_anchor_ref swapped for the tagged base): Dan's 4,495 rl_targets, 256 prompts x 16,
    T 1.0, lr 1.5e-4, the canary guard with rollback (max 3), a model-only snapshot every 10 iterations, a rolling
    checkpoint every 5. Every prompt's every state carries tag T.
  * Reads: the base (iteration 0, segment 1) and every snapshot, with Dan's code under the same tag (tag_danread.py:
    k 64 on the 106, T 0.8, 256 / 48, noise seed seed*1000 + it + 500, batch 128) into seg<k>/reads/it<i>.json.
  * Stop: x_dan >= 1.5, GPU-h >= 7, or reached and held (tag_measure). STOP in the segment ends it now; STOP_AT_CKPT
    after the next rolling checkpoint (written by this process when another user's process appears on the GPU).
  * Variant lines (t<A|B>_<variant>_s<seed>): the base is AT/bases/stage1_tagAB_<variant>_s<seed>.pt (--init, the KL
    anchor and the iteration-0 read), D its own .tag.json, the run folder AT/runs/<line>/; tag_grpo.py gets
    --variant and refuses an --init trained on another corpus. Recipe, ladder, x grid and stop rule are the default
    line's of the same seed.
  * Files: seg<k>/status.json (heartbeat, 30 s), seg<k>/end.json {how: done|stopped|rule|moved|failed|nonfinite, rc,
    reason}, AT/runs/<line>/segs.json (segment list for tag_measure), AT/logs/<line>_seg<k>.log (GRPO),
    AT/logs/<line>.events.log.
"""

# ruff: noqa: E501
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
import tag_core as T  # noqa: E402
import tag_measure as TM  # noqa: E402

G = TM._ko("gpu")
HOST = os.uname().nodename.split(".")[0]
ALLOC = "expandable_segments:True,garbage_collection_threshold:0.6"
ITERS = 400  # far past x 1.5 (about 0.008-0.012 x per iteration in the knockout lines); the stop rule ends it


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _set(argv: list[str], flag: str, value: str) -> None:
    assert argv.count(flag) == 1, flag
    argv[argv.index(flag) + 1] = value


def smoke_argv(argv: list[str]) -> list[str]:
    """CPU smoke only: the same recipe at toy sizes (2 iterations of 4 prompts x 4)."""
    argv = list(argv)
    for f, v in (("--prompts", "4"), ("--group", "4"), ("--batch", "16"), ("--canary_n", "20"),
                 ("--snap_iters", "1,2"), ("--save_rolling", "1"), ("--save_every", "2")):  # fmt: skip
        _set(argv, f, v)
    return [*argv, "--device", "cpu"]


def grpo_argv(line: str, seg_dir: Path, iters: int, resume: str | None) -> list[str]:
    """The 1 % line's recipe (knockout common.grpo_argv for dd1_s<seed>), on the tagged base."""
    L = T.LINES[line]
    ref = f"dd1_s{L['seed']}"
    assert TM.C.LINES[ref]["seed"] == L["seed"] and not TM.C.LINES[ref].get("guard_off")
    argv = TM.C.grpo_argv(ref, seg_dir, iters, resume)
    base = str(T.line_base(line))
    _set(argv, "--init", base)
    _set(argv, "--name", line)
    _set(argv, "--kl_anchor_ref", base)
    assert argv[argv.index("--targets") + 1] == str(TM.C.KD / "pools" / "dan_rl_targets.jsonl")
    assert argv[argv.index("--seed") + 1] == str(L["seed"]) and argv[
        argv.index("--out") + 1
    ] == str(seg_dir)
    return argv


def grpo_cmd(line: str, argv: list[str]) -> list[str]:
    """tag_grpo.py under the line's tag (and, for a variant line, its variant, which tag_grpo checks on --init)."""
    L = T.LINES[line]
    v = ["--variant", L["variant"]] if L.get("variant") else []
    return [T.PY, str(T.HERE / "tag_grpo.py"), "--tag", L["tag"], *v, "--", *argv]


class Seg:
    def __init__(self, a: argparse.Namespace) -> None:
        self.a = a
        self.L = T.LINES[a.line]
        self.tag = self.L["tag"]
        self.line_dir = T.AT / "runs" / a.line
        self.dir = self.line_dir / f"seg{a.seg}"
        self.log = T.AT / "logs" / f"{a.line}_seg{a.seg}.log"
        self.proc: subprocess.Popen | None = None
        self.done = threading.Event()
        self.reading = threading.Lock()
        self.why, self.reason = "running", None
        self.fails: dict[int, int] = {}

    def say(self, msg: str) -> None:
        with open(T.AT / "logs" / f"{self.a.line}.events.log", "a") as f:
            f.write(f"{now()} {HOST} seg{self.a.seg}: {msg}\n")

    def env(self) -> dict:
        e = dict(os.environ)
        e.update(ND_OFFLINE="1", ND_REGISTRY_SYNC="0", PYTHONDONTWRITEBYTECODE="1", LEAN_GATE_WORKERS="8",
                 OMP_NUM_THREADS="8", CUDA_DEVICE_ORDER="PCI_BUS_ID",
                 CUDA_VISIBLE_DEVICES="" if self.a.smoke else str(self.a.gpu),
                 LEAN_GATE_DUMP=str(self.dir / "lean_dump.jsonl"),
                 ND_REGISTRY_DIR=str(self.dir / "registry"), PYTORCH_CUDA_ALLOC_CONF=ALLOC,
                 **{T.ENV: self.tag})  # fmt: skip
        return e

    # ------------------------------------------------------------------ segments file
    def segs(self) -> list[dict]:
        p = self.line_dir / "segs.json"
        return json.loads(p.read_text()) if p.exists() else []

    def write_segs(self, segs: list[dict]) -> None:
        p = self.line_dir / "segs.json"
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(segs, indent=1))
        tmp.replace(p)

    # ------------------------------------------------------------------ reads
    def read(self, ckpt: Path, it: int) -> None:
        out = self.dir / "reads" / f"it{it:04d}.json"
        if out.exists():
            return
        cmd = [T.PY, str(T.HERE / "tag_danread.py"), "--tag", self.tag, "--", "--ckpt", str(ckpt),
               "--seed", str(self.L["seed"]), "--round", str(it), "--out", str(out), "--only106",
               "--k", "2" if self.a.smoke else "64", "--batch", "128"]  # fmt: skip
        if self.a.smoke:
            cmd.insert(cmd.index("--"), "--cpu")
        env = {**self.env(), "LEAN_GATE_LOG": str(self.dir / "reads_lean_gate.jsonl"),
               **({} if self.a.smoke else {"DANREAD_MEM_FRAC": "0.3"})}  # fmt: skip
        with self.reading, open(self.dir / "reads.log", "a") as f:
            p = subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT, timeout=3600)
        self.say(f"read it {it}: rc {p.returncode}")
        if p.returncode != 0:
            self.fails[it] = self.fails.get(it, 0) + 1
            if self.fails[it] >= 3:
                (self.dir / "reads" / f"it{it:04d}.failed").write_text(now())

    def reader(self) -> None:
        base = self.a.seg == 1 and not self.a.resume
        while True:
            fin = self.done.is_set()
            r0 = self.dir / "reads"
            if base and not (r0 / "it0000.json").exists() and not (r0 / "it0000.failed").exists():
                self.read(T.line_base(self.a.line), 0)
            snaps = sorted((self.dir / "ckpts").glob(f"{self.a.line}_it*_snap.pt"),
                           key=lambda p: int(p.name.split("_it")[-1].split("_")[0]))  # fmt: skip
            for sp in snaps:
                it = int(sp.name.split("_it")[-1].split("_")[0])
                if (
                    not (r0 / f"it{it:04d}.json").exists()
                    and not (r0 / f"it{it:04d}.failed").exists()
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
        ps = [*(self.dir / "ckpts").glob(f"{self.a.line}_it*_rolling.pt"),
              *(self.dir / "ckpts").glob(f"{self.a.line}_it*_alarm.pt")]  # fmt: skip
        return max((p.stat().st_mtime for p in ps), default=0.0)

    def watch(self) -> None:
        t_gpu = t_rule = 0.0
        while self.proc is not None and self.proc.poll() is None:
            st = {"utc": now(), "host": HOST, "gpu": self.a.gpu, "pid": self.proc.pid, "seg": self.a.seg,
                  "tag": self.tag, "why": self.why}  # fmt: skip
            (self.dir / "status.json").write_text(json.dumps(st))
            if (self.dir / "STOP").exists():
                self.why = "stopped"
                self.kill()
                break
            sac = self.dir / "STOP_AT_CKPT"
            if sac.exists() and self.newest_rolling() > sac.stat().st_mtime:
                self.why = "moved"
                self.say(
                    "STOP_AT_CKPT: a rolling checkpoint is written; ending (resume as the next segment)"
                )
                self.kill()
                break
            if time.time() - t_rule > self.a.rule_every:
                t_rule = time.time()
                try:
                    v = TM.line_view(self.a.line)
                    r = TM.stop_reason(v)
                except (
                    Exception
                ) as e:  # a view that cannot be built yet (no rows) never stops the line
                    r = None
                    self.say(f"line view: {type(e).__name__}: {e}")
                if r is not None:
                    self.why, self.reason = "rule", r
                    self.say(f"stop rule: {r}")
                    self.kill()
                    break
            if time.time() - t_gpu > 60 and not self.a.smoke:
                t_gpu = time.time()
                g = G.gpus(HOST)
                if g is not None and (g.get(self.a.gpu, set()) - {G.ME}) and not sac.exists():
                    self.say(f"another user's process on GPU {self.a.gpu}: {g[self.a.gpu]}")
                    sac.touch()
            time.sleep(min(30, self.a.rule_every))

    def kill(self, wait: int = 300) -> None:
        if self.proc is None:
            return
        try:
            os.killpg(self.proc.pid, signal.SIGTERM)
            self.proc.wait(timeout=wait)
        except subprocess.TimeoutExpired:
            os.killpg(self.proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    def on_term(self, signum, frame) -> None:
        """`timeout` (or anyone) ends this process: GRPO runs in its own session, so end it here, never orphan it."""
        self.why = "terminated"
        self.say(f"signal {signum}: ending GRPO")
        self.kill(wait=60)

    def run(self) -> int:
        signal.signal(signal.SIGTERM, self.on_term)
        (self.dir / "reads").mkdir(parents=True, exist_ok=True)
        (T.AT / "logs").mkdir(parents=True, exist_ok=True)
        segs = [s for s in self.segs() if s["seg"] != self.a.seg]
        assert all(s["seg"] < self.a.seg for s in segs), "a later segment exists"
        segs.append({"seg": self.a.seg, "host": HOST, "gpu": self.a.gpu, "start": now(), "resume": self.a.resume,
                     "end": None, "tag": self.tag, "base": self.L["base"]})  # fmt: skip
        self.write_segs(segs)
        argv = grpo_argv(self.a.line, self.dir, self.a.iters, self.a.resume)
        if self.a.smoke:
            argv = smoke_argv(argv)
        cmd = grpo_cmd(self.a.line, argv)
        (self.dir / "cmd.json").write_text(
            json.dumps({"cmd": cmd, "cwd": str(T.CODE), "utc": now()}, indent=1)
        )
        self.say(f"launch GPU {self.a.gpu} tag {self.tag} resume {self.a.resume}")
        rt = threading.Thread(target=self.reader, daemon=True)
        rt.start()
        with open(self.log, "a") as f:
            self.proc = subprocess.Popen(cmd, cwd=str(T.CODE), stdout=f, stderr=subprocess.STDOUT,
                                         start_new_session=True, env=self.env())  # fmt: skip
        self.watch()
        rc = self.proc.wait()
        if self.why == "running":
            self.why = "done" if rc == 0 else "nonfinite" if rc == 4 else "failed"
        self.done.set()
        rt.join(timeout=0 if self.why == "terminated" else 3 * 3600)
        end = now()
        segs = self.segs()
        for s in segs:
            if s["seg"] == self.a.seg:
                s["end"] = end
        self.write_segs(segs)
        try:
            v = TM.line_view(self.a.line, final=True)
            summ = {"x_dan": v["x_dan"], "gpuh": v["gpuh"], "matched": [m["cum"] for m in v["matched"]],
                    "cross_dan": v["cross_dan"], "stop": TM.stop_reason(v), "tag_audit": TM.tag_audit(self.a.line)}  # fmt: skip
        except Exception as e:
            summ = {"error": f"{type(e).__name__}: {e}"}
        (self.dir / "end.json").write_text(json.dumps({"how": self.why, "rc": rc, "reason": self.reason,
                                                       "utc": end, "host": HOST, "view": summ}, indent=1))  # fmt: skip
        self.say(f"end: {self.why} (rc {rc}) {self.reason or ''} {summ}")
        return 0 if self.why in ("done", "rule", "stopped") else 1


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--line", required=True, choices=sorted(T.LINES))
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--seg", type=int, default=1)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--iters", type=int, default=ITERS)
    ap.add_argument(
        "--rule_every", type=float, default=60.0, help="seconds between stop-rule checks"
    )
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="CPU smoke at toy sizes (with AT pointing at a smoke tree)",
    )
    a = ap.parse_args()
    if a.smoke:
        assert "smoke" in str(T.AT), "a smoke run writes under a smoke tree (AT=.../smoke/...)"
        a.iters = 2 if a.iters == ITERS else a.iters
    L = T.LINES[a.line]
    base = T.line_base(a.line)
    seg_dir = T.AT / "runs" / a.line / f"seg{a.seg}"
    argv = grpo_argv(a.line, seg_dir, a.iters, a.resume)
    problems = TM.segment_problems(a.line, a.seg, a.resume)
    try:
        D = T.check_line_base(a.line)["train_tokens"]
    except SystemExit as e:
        D, problems = None, [str(e), *problems]
    if a.dry_run:
        print(json.dumps({"line": a.line, "tag": L["tag"], "variant": L.get("variant"), "base": str(base),
                          "base_exists": base.exists(), "D_sidecar": (base.with_suffix(".tag.json")).exists(),
                          "D": D, "ladder": L["ladder"], "seg_dir": str(seg_dir), "cmd": grpo_cmd(a.line, argv),
                          "cwd": str(T.CODE), "refuses": problems}, indent=1))  # fmt: skip
        return
    if problems:
        raise SystemExit(f"line {a.line} seg {a.seg}: " + "; ".join(problems))
    meta = T.ckpt_meta(base)
    assert meta["tok_mode"] == T.MODE, meta
    assert meta["variant"] == L.get("variant"), (
        f"{base} trained on variant {meta['variant']!r}, line {a.line}"
    )
    if not a.smoke and not G.free_for_us(HOST, a.gpu):
        raise SystemExit(f"GPU {a.gpu} on {HOST} has another user's process: not launching")
    sys.exit(Seg(a).run())


if __name__ == "__main__":
    main()
