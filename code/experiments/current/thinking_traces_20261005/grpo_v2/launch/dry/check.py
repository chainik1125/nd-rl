#!/usr/bin/env python3
# ruff: noqa: E501  (literal command lines and log lines, compared byte for byte)
# pyright: basic
"""Check the dry runs of all.sh against an independent model of the GRPO v2 recipe and against the transcript evidence.

    python3 check.py [--expect-hook]     # after make.sh + all.sh; exits 1 on any failure

--expect-hook: queue.sh is the ynez version, which runs `python3 $R/backup.py >> $L/backup.log 2>&1 || true`
after each arm; without it the queue must make no backup call (serrano's queue.sh).

Sections
  A  every valid arm (12 base, 8 ARM_h, 16 ARM_b, 8 each of ARM_c, ARM_w, ARM_wb, ARM_rfb, ARM_ulb, 6 ARM_3pb) x seeds
     0-2 x normal/SMOKE=1: exit 0, the full multiset and order of
     stubbed calls (argv, cwd, CUDA_VISIBLE_DEVICES, redirections), the environment of each python call, stdout, the
     eval dirs and logs, against `expect()` below (written from the README / evidence, not from line.sh's text)
  B  refusals: ei_*_h, ei_*_b, unknown arm, ARM_b without L2 checkpoint / with a rolled-back L2 / without an L2 log,
     a seed without a Stage-1 init, usage; ei_*_c / _w / _wb, the lr schedules with _h or _b (either order), ARM_wb
     without ARM's L2 checkpoint / with a rolled-back L2 / without an L2 log; the L3 pilots (ARM_rfb, ARM_ulb,
     ARM_3pb) for EI, with another suffix (either order), ARM_3pb for a _vs3 arm, as ARM_b's three; a missing
     curriculum_3p file for ARM_3pb and the _vs3 arms; nothing launched
  C  waitfree with a busy GPU; queue.sh (wait line, PRE_CMD, two arms, done line[, backup hook])
  E  transcript evidence (evidence.json): the live grpo_think_h_b s1 L3 process; serrano's harness records (full
     commands and per-call records of three runs, call counts); every launcher log line and ps line of a v2 arm
"""

from __future__ import annotations

import collections
import json
import os
import re
import sys
from pathlib import Path

H = os.environ.get("DRY_HOME", "/scratch/cowifornia/nd-rl/think/dry")
SB = f"{H}/sb/runs"
WT = "/scratch/cowifornia/nd-rl/think/nd-rl-v2"
RUNS = "/scratch/cowifornia/nd-rl/think/runs"
PYREAL = "/scratch/cowifornia/nd-rl/think/nd-rl/code/.venv/bin/python"
TB = "/scratch/cowifornia/nd-rl/think/nd-takehome/data/eval_only/textbook72"
BAN = " --ban_restore_check --ban_stop 0.005"
BASE = [
    "grpo_think",
    "grpo_think_v",
    "grpo_think_vs",
    "grpo_nothink",
    "grpo_nothink_v",
    "grpo_nothink_vs",
    "grpo_nothink_vs3",
    "ei_think",
    "ei_think_v",
    "ei_nothink",
    "ei_nothink_v",
    "grpo_think_vs3",  # 2026-10-13: grpo_think_vs plus the 3-premise rows, as grpo_nothink_vs3
]
HARMS = [a + "_h" for a in BASE if a.startswith("grpo_")]
BARMS = [a + "_b" for a in BASE if a.startswith("grpo_")] + [a + "_b" for a in HARMS]
# the lr schedules (2026-10-13; written from the request, not from line.sh): ARM_c cosine in both stages, ARM_w WSD in
# L3 only (L2 is ARM's command), ARM_wb ARM_w's L3 from ARM's L2 (as ARM_b)
GBASE = [a for a in BASE if a.startswith("grpo_")]
CARMS, WARMS, WBARMS = ([a + x for a in GBASE] for x in ("_c", "_w", "_wb"))
SCHED_C = "--lr_schedule cosine --warmup_iters 3 --min_lr_frac 0.1"
SCHED_W = "--lr_schedule wsd --decay_frac 0.2 --min_lr_frac 0.3 --decay_shape sqrt"
# the L3 pilots (2026-10-13; written from the request): L3 of ARM from ARM's L2 with grpo.py's pilot flags (refill and
# the pass-rate filter; the unlikeliness reward), or with the L3 3-premise rows as _vs3 passes them
PILOT = {
    "_rfb": "--refill --refill_max 2 --max_pass 0.9",
    "_ulb": "--unlikely_beta 0.25",
    "_3pb": "",
}
RFBARMS, ULBARMS = ([a + x for a in GBASE] for x in ("_rfb", "_ulb"))
P3BARMS = [a + "_3pb" for a in GBASE if not a.endswith("_vs3")]
REUSE = BARMS + WBARMS + RFBARMS + ULBARMS + P3BARMS  # L3 only, from another arm's L2
VALID = BASE + HARMS + BARMS + CARMS + WARMS + WBARMS + RFBARMS + ULBARMS + P3BARMS
ENV_WANT = {
    "ND_OFFLINE": "1",
    "PYTHONDONTWRITEBYTECODE": "1",
    "ND_REGISTRY_SYNC": "0",
    "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    "ND_LEAN": "/home/cowifornia/.elan/bin/lean",
    "TMPDIR": "<R>/tmp",
}

fails: list[str] = []
counts = collections.Counter()


def ok(cond: bool, sect: str, msg: str) -> bool:
    counts[(sect, cond)] += 1
    if not cond:
        fails.append(f"[{sect}] {msg}")
    return cond


def norm(s: str) -> str:
    s = (
        s.replace(f"{H}/run/stdout.txt", "<stdout>")
        .replace(SB, "<R>")
        .replace(f"{H}/pybin/python", "<PY>")
    )
    return re.sub(r"(watchdog\.py [^:\s]+):\d+:", r"\1:N:", s)


def load(d: str) -> dict:
    r = {
        "exit": Path(f"{d}/exit").read_text().strip(),
        "stdout": Path(f"{d}/stdout.txt").read_text(),
        "tree": Path(f"{d}/tree.txt").read_text().split("\n"),
        "pre": Path(f"{d}/pre_tree.txt").read_text().split("\n"),
        "rec": [],
    }
    for ln in Path(f"{d}/rec.tsv").read_text().split("\n"):
        if ln:
            k, cwd, cvd, fd1, fd2, envh, argv, envs = ln.split("\t")
            r["rec"].append(
                dict(
                    kind=k,
                    cwd=norm(cwd),
                    cvd=cvd,
                    fd1=norm(fd1),
                    fd2=norm(fd2),
                    envh=envh,
                    argv=norm(argv),
                    env=dict(x.split("=", 1) for x in envs.split() if "=" in x),
                    raw=ln,
                )
            )
    return r


def key(x: dict) -> tuple:
    return (x["kind"], x["cwd"], x["cvd"], x["fd1"], x["fd2"], x["argv"])


# ---------------------------------------------------------------- the model of the recipe (README + evidence)
def expect(arm: str, s: int, mode: str, g: int) -> tuple[list[tuple], list[str]]:
    """-> (ordered records (kind, cwd, cvd, fd1, fd2, argv), stdout lines) for a valid arm."""
    pil = next((x for x in PILOT if arm.endswith(x)), "")
    sch = "" if pil else next((x for x in ("_wb", "_c", "_w") if arm.endswith(x)), "")
    core = arm[: -len(pil or sch)] if pil or sch else arm
    src = core[:-2] if core.endswith("_b") and not pil else core
    base = src[:-2] if src.endswith("_h") and not pil else src
    sched = {"": ("", ""), "_c": (SCHED_C, SCHED_C), "_w": ("", SCHED_W), "_wb": ("", SCHED_W)}[sch]
    l2src = base if sch == "_wb" or pil else (arm if sch else src)  # whose L2 the L3 starts from
    pflags = PILOT.get(pil, "")  # stage L3's pilot flags
    smoke = mode == "smoke"
    think = "_think" in base
    method = "grpo" if base.startswith("grpo_") else "ei"
    samp = "logical" if base.endswith(("_v", "_vs", "_vs3")) else "plain"
    vs3 = base.endswith("_vs3") or pil == "_3pb"  # the 3-premise rows as extra targets
    R, C = "<R>", "<R>/grpo_v2" + ("/smoke" if smoke else "")
    EV = "<R>/eval/v2" + ("_smoke" if smoke else "")
    L = C + "/logs" if smoke else "<R>/logs"
    it, batch, cvd = (2, 16, "CVD=[]") if smoke else (39, 1024, f"CVD=[{g}]")
    lr = "7.5e-5" if base != src else "1.5e-4"
    eff = "--effort --budgets 0\\,64\\,256" if think else "--budgets 0"
    rb = 64 if think else 0
    st = "<stdout>"
    recs = [
        ("git", WT, "CVD=unset", "pipe", st, f"-C {WT} rev-parse --abbrev-ref HEAD"),
        (
            "py",
            WT,
            "CVD=unset",
            st,
            st,
            "-c import\\ code.thinking\\,\\ sys\\;\\ sys.exit\\(0\\ if\\ code.thinking.__file__"
            f".startswith\\(\\'{WT}/\\'\\)\\ else\\ 1\\)",
        ),
    ]
    if pflags:  # ARM_rfb / ARM_ulb: the worktree's grpo.py must have the pilot options
        recs.append(
            (
                "py",
                WT,
                "CVD=unset",
                "/dev/null",
                "/dev/null",
                "-c import\\ sys\\,\\ code.thinking.grpo\\ as\\ g\\;\\ sys.exit\\(0\\ if\\ hasattr\\(g\\,\\ "
                "\\'pilot_options\\'\\)\\ else\\ 1\\)",
            )
        )
    l2t = f"{C}/L2_head.jsonl" if smoke else f"{R}/pools/curriculum/L2/pool.jsonl"
    l3t = f"{C}/L3_head.jsonl" if smoke else f"{R}/pools/curriculum/L3/pool.jsonl"
    x2 = (f"{C}/X2_head.jsonl" if smoke else f"{R}/pools/curriculum_3p/L2_3p.jsonl") if vs3 else ""
    x3 = (f"{C}/X3_head.jsonl" if smoke else f"{R}/pools/curriculum_3p/L3_3p.jsonl") if vs3 else ""
    pf = "-m code.experiments.current.thinking_traces_20261005.grpo_v2.preflight"
    if not smoke:
        f = f"{C}/preflight_{arm}_s{s}.log"
        recs.append(
            (
                "py",
                WT,
                "CVD=unset",
                f,
                f,
                f"{pf} <R>/pools/curriculum/L2/pool.jsonl <R>/pools/curriculum/L3/pool.jsonl",
            )
        )
        if vs3:
            f = f"{C}/preflight_{arm}_s{s}_extra.log"
            recs.append(
                (
                    "py",
                    WT,
                    "CVD=unset",
                    f,
                    f,
                    f"{pf} <R>/pools/curriculum_3p/L2_3p.jsonl <R>/pools/curriculum_3p/L3_3p.jsonl",
                )
            )
    recs.append(
        ("git", WT, "CVD=unset", f"{C}/{arm}_s{s}.base_commit.txt", st, f"-C {WT} rev-parse HEAD")
    )
    out: list[str] = []

    def waitfree() -> None:
        if not smoke:
            recs.append(
                (
                    "nvidia-smi",
                    WT,
                    "CVD=unset",
                    "pipe",
                    st,
                    f"--query-gpu=memory.used --format=csv\\,noheader\\,nounits -i {g}",
                )
            )
            recs.append(
                (
                    "nvidia-smi",
                    WT,
                    "CVD=unset",
                    "pipe",
                    st,
                    f"--query-compute-apps=pid --format=csv\\,noheader -i {g}",
                )
            )

    def ck(name: str) -> str:
        return (
            f"{C}/{name}/ckpts/{name}_it{it}.pt"
            if method == "grpo"
            else f"{C}/{name}/ckpts/{name}_r1.pt"
        )

    def stage(
        name: str, init: str, targets: str, extra: str, sc: str = "", pflags: str = ""
    ) -> None:
        waitfree()
        log = f"{L}/v2_{name}.log"
        if method == "grpo":
            a = (
                f"21600 <PY> -m code.thinking.grpo --init {init} --name {name} --out {C}/{name} --targets {targets} "
                f"--iters {it} --prompts {4 if smoke else 256} --group {4 if smoke else 16} --batch {batch} --lr {lr} "
                + (f"{sc} " if sc else "")
                + (f"{pflags} " if pflags else "")
                + f"--overlong fail --save_every {it} --max_steps 96 --seed {s} --keep_rolling 100 {eff} "
                f"--canary <R>/pools/canary_long.jsonl --canary_n {40 if smoke else 1000} --canary_every 1 "
                f"--rollback_on_alarm --max_rollbacks 3{BAN} --kl_anchor_coef 0.02 --kl_anchor_ref {init}"
                + (" --kl_anchor_n 16" if smoke else "")
                + " --monitor"
            )
            if samp == "logical":
                a += f" --arm logical --max_rej {0 if base.endswith(('_vs', '_vs3')) else 10} --rej_penalty 0.1"
            if extra:
                a += f" --targets_extra {extra} --targets_extra_frac 0.33"
            desc = (
                samp
                + (", P 0.1" if samp == "logical" else "")
                + (f", extra {os.path.basename(extra)} x 0.33" if extra else "")
                + (f", lr {lr}" if base != src else "")
                + (f", {sc}" if sc else "")
                + (f", {pflags}" if pflags else "")
            )
            out.append(f"00:00Z launched GRPO {name} ({desc}) on GPU {g} (pid N)")
            out.append(f"00:00Z GRPO {name} exit 0: it {it}: dry")
            wh = "3.5"
        else:
            a = (
                f"21600 <PY> -m code.thinking.ei --init {init} --name {name} --out {C}/{name} --targets {targets} "
                f"--train <R>/data/{'S1g3' if think else 'T0S'}.pkl --rounds 1 --k {2 if smoke else 32} --temperature 0.8 "
                f"--arm {samp} --max_rej 10 --batch {batch} --seed {s} --max_steps 96 --select shortest {eff}"
                + (" --ft_steps 3 --ft_recs 8 --retain 64" if smoke else "")
            )
            out.append(f"00:00Z launched EI {name} ({samp}) on GPU {g} (pid N)")
            out.append(f"00:00Z EI {name} exit 0: ei dry")
            wh = "4"
        recs.append(("timeout", WT, cvd, log, log, a))
        recs.append(("sleep", WT, "CVD=unset", st, st, "5"))
        if not smoke:
            wd = f"{L}/watchdog_grpo_v2.log"
            recs.append(
                ("python3", WT, "CVD=unset", wd, wd, f"<R>/watchdog.py {name}:N:{g}:{log}:{wh}")
            )

    init0 = f"<R>/stage1/{'S1g3' if think else 'T0S'}_s{s}/model.pt"
    if l2src == arm:
        stage(f"{arm}_s{s}_L2", init0, l2t, x2, sched[0])
        l2ck = ck(f"{arm}_s{s}_L2")
    else:
        l2ck = f"{C}/{l2src}_s{s}_L2/ckpts/{l2src}_s{s}_L2_it{it}.pt"
    stage(f"{arm}_s{s}_L3", l2ck, l3t, x3, sched[1], pflags)
    ch = "<R>/pools/charles_1002"
    reads = []
    for ra in ("plain", "logical"):
        reads += [
            ("t1320", "<R>/eval/matched/sets/t1320_n1000.jsonl", 16, ra, 96),
            ("tb72", f"{TB}/textbook_dev.jsonl\\,{TB}/textbook_train.jsonl", 64, ra, 96),
            (
                "charles173",
                f"{ch}/candidate_v0.jsonl\\,{ch}/candidate_v1.jsonl\\,{ch}/batch3.jsonl",
                64,
                ra,
                96,
            ),
        ]
    reads += [
        ("hand14", f"{ch}/hand_proved_14.jsonl", 64, "plain", 160),
        ("t2130", "<R>/eval/matched/sets/t2130_n1000.jsonl", 8, "plain", 96),
    ]
    for name, sets, k, ra, ms in reads:
        k = 2 if smoke else k
        o = f"{EV}/{name}_{arm}_s{s}_{ra}_k{k}"
        waitfree()
        recs.append(("rm", WT, "CVD=unset", st, st, f"-rf {o}"))
        log = f"{L}/v2_eval_{os.path.basename(o)}.log"
        recs.append(
            (
                "timeout",
                WT,
                cvd,
                log,
                log,
                f"9000 <PY> -m code.thinking.eval --ckpt {ck(f'{arm}_s{s}_L3')} --sets {sets} --out {o} --k {k} "
                f"--budgets {rb} --arms {ra} --batch {batch} --seed 0 --temperature 0.8 --max_action 512 "
                f"--max_steps {ms} --max_rej 10" + (" --limit 2" if smoke else ""),
            )
        )
        out.append(f"00:00Z read {os.path.basename(o)} exit 0 0s: [dry] eval ")
    out.append(f"00:00Z grpo_v2 line done ({arm} seed {s})")
    return recs, out


def stdout_lines(r: dict) -> list[str]:
    return [re.sub(r"\(pid \d+\)", "(pid N)", ln) for ln in r["stdout"].split("\n") if ln]


# ---------------------------------------------------------------- A: every valid arm
runs: dict[tuple, dict] = {}
for mode in ("normal", "smoke"):
    for s in (0, 1, 2):
        for arm in VALID:
            d = f"{H}/res/{mode}/s{s}/{arm}"
            r = runs[(mode, s, arm)] = load(d)
            exp, eout = expect(arm, s, mode, 3)
            got = [key(x) for x in r["rec"]]
            tag = f"{mode} s{s} {arm}"
            ok(r["exit"] == "0", "A", f"{tag}: exit {r['exit']}")
            ok(
                sorted(got) == sorted(exp),
                "A",
                f"{tag}: call multiset differs: missing {sorted(set(exp) - set(got))[:2]} extra {sorted(set(got) - set(exp))[:2]}",
            )
            # the stage's timeout runs in a background subshell next to `sleep 5` and the watchdog launch (serrano's
            # harness compared the order without sleep and python3 for the same reason)
            ok(
                [x for x in got if x[0] not in ("python3", "sleep")]
                == [x for x in exp if x[0] not in ("python3", "sleep")],
                "A",
                f"{tag}: call order differs",
            )
            ok(
                stdout_lines(r) == eout,
                "A",
                f"{tag}: stdout differs: {[x for x in stdout_lines(r) if x not in eout][:2]} / {[x for x in eout if x not in stdout_lines(r)][:2]}",
            )
            for x in r["rec"]:
                if x["kind"] in ("py", "timeout", "python3"):
                    e = {k: norm(v) for k, v in x["env"].items() if k in ENV_WANT}
                    ok(e == ENV_WANT, "A", f"{tag}: {x['kind']} env {e}")
                    ok(("SMOKE" in x["env"]) == (mode == "smoke"), "A", f"{tag}: SMOKE env")
            ev = [
                t
                for t in r["tree"]
                if re.match(
                    r"^d \./runs/eval/v2(_smoke)?/[a-z0-9]+_"
                    + re.escape(arm)
                    + rf"_s{s}_(plain|logical)_k\d+$",
                    t,
                )
            ]
            ok(len(ev) == 8, "A", f"{tag}: {len(ev)} eval dirs")
            lg = "./runs/grpo_v2/smoke/logs" if mode == "smoke" else "./runs/logs"
            el = [t for t in r["tree"] if t.startswith(f"f {lg}/v2_eval_")]
            ok(
                len(el) == 8
                and all(
                    re.match(
                        rf"^f {re.escape(lg)}/v2_eval_(t1320|tb72|charles173|hand14|t2130)_{re.escape(arm)}_s{s}_(plain|logical)_k\d+\.log$",
                        t,
                    )
                    for t in el
                ),
                "A",
                f"{tag}: eval logs {el[:2]}",
            )
            if arm in REUSE:
                ok(
                    not any(f"{arm}_s{s}_L2" in t for t in r["tree"]),
                    "A",
                    f"{tag}: ARM_b / ARM_wb made an L2 of its own",
                )
            n = len(r["rec"])
            want = (
                (23 - (2 if arm in REUSE else 0))
                if mode == "smoke"
                else (46 + ("_vs3" in arm or arm.endswith("_3pb")) - (5 if arm in REUSE else 0))
            ) + arm.endswith(("_rfb", "_ulb"))  # the pilot options check
            ok(n == want, "A", f"{tag}: {n} calls, want {want}")

# ---------------------------------------------------------------- B: refusals
refusals = []
for mode in ("normal", "smoke"):
    for s in (0, 1, 2):
        for arm in ["ei_think_h", "ei_think_v_h", "ei_nothink_h", "ei_nothink_v_h"]:
            refusals.append(
                (
                    f"{H}/res/{mode}/s{s}/{arm}",
                    1,
                    rf"^00:00Z {arm}: _h halves the GRPO lr, which EI does not use \(run {arm[:-2]}\): refusing$",
                    arm,
                )
            )
        for arm in ["ei_think_b", "ei_think_v_b", "ei_nothink_b", "ei_nothink_v_b", "ei_think_h_b"]:
            refusals.append(
                (
                    f"{H}/res/{mode}/s{s}/{arm}",
                    1,
                    rf"^00:00Z {arm}: _b reruns a GRPO L3 stage, and {arm[:-2]} is EI: refusing$",
                    arm,
                )
            )
        refusals.append(
            (f"{H}/res/{mode}/s{s}/bogus_arm", 2, r"^unknown arm bogus_arm$", "bogus_arm")
        )
        for eb in ("ei_think", "ei_think_v", "ei_nothink", "ei_nothink_v"):
            for x in ("c", "w", "wb"):
                arm = f"{eb}_{x}"
                refusals.append(
                    (
                        f"{H}/res/{mode}/s{s}/{arm}",
                        1,
                        rf"^00:00Z {arm}: _{x} schedules the GRPO lr, and EI's fine-tune has its own \(run {eb}\): refusing$",
                        arm,
                    )
                )
        for arm in [
            "grpo_think_c_h",
            "grpo_think_c_b",
            "grpo_think_h_c",
            "grpo_think_b_c",
            "grpo_think_vs_w_h",
            "grpo_think_vs_w_b",
            "grpo_think_vs_h_w",
            "grpo_think_vs_wb_b",
            "grpo_think_vs_h_wb",
            "grpo_think_vs_b_wb",
            "grpo_think_c_h_b",
            "grpo_think_vs_h_b_w",
            "ei_think_c_h",
        ]:
            refusals.append(
                (
                    f"{H}/res/{mode}/s{s}/{arm}",
                    1,
                    rf"^00:00Z {arm}: _c, _w and _wb \(lr schedules\) do not combine with _h or _b: refusing$",
                    arm,
                )
            )
        d = f"{H}/res/refuse/{mode}/s{s}"
        it = 2 if mode == "smoke" else 39
        c = "grpo_v2/smoke" if mode == "smoke" else "grpo_v2"
        refusals.append(
            (
                f"{d}/nockpt_grpo_think_h_b",
                1,
                rf"^00:00Z grpo_think_h_b: no L2 checkpoint of grpo_think_h \({re.escape(SB)}/{c}/grpo_think_h_s{s}_L2/ckpts/grpo_think_h_s{s}_L2_it{it}\.pt\): refusing$",
                "grpo_think_h_b",
            )
        )
        refusals.append(
            (
                f"{d}/rb_grpo_think_b",
                1,
                r"^00:00Z grpo_think_b: grpo_think's L2 log ends 'guard it 39: .*lr x0\.5 \(1 rollbacks\).*', not lr x1 \(0 rollbacks\): rerun the whole line: refusing$",
                "grpo_think_b",
            )
        )
        refusals.append(
            (
                f"{d}/nolog_grpo_think_vs_b",
                1,
                r"^00:00Z grpo_think_vs_b: grpo_think_vs's L2 log ends 'no guard line', not lr x1 \(0 rollbacks\): rerun the whole line: refusing$",
                "grpo_think_vs_b",
            )
        )
        # ARM_wb: from ARM's L2, refused as ARM_b
        refusals.append(
            (
                f"{d}/nockpt_grpo_think_vs_wb",
                1,
                rf"^00:00Z grpo_think_vs_wb: no L2 checkpoint of grpo_think_vs \({re.escape(SB)}/{c}/grpo_think_vs_s{s}_L2/ckpts/grpo_think_vs_s{s}_L2_it{it}\.pt\): refusing$",
                "grpo_think_vs_wb",
            )
        )
        refusals.append(
            (
                f"{d}/rb_grpo_think_vs_wb",
                1,
                r"^00:00Z grpo_think_vs_wb: grpo_think_vs's L2 log ends 'guard it 39: .*lr x0\.5 \(1 rollbacks\).*', not lr x1 \(0 rollbacks\): rerun the whole line: refusing$",
                "grpo_think_vs_wb",
            )
        )
        refusals.append(
            (
                f"{d}/nolog_grpo_nothink_vs3_wb",
                1,
                r"^00:00Z grpo_nothink_vs3_wb: grpo_nothink_vs3's L2 log ends 'no guard line', not lr x1 \(0 rollbacks\): rerun the whole line: refusing$",
                "grpo_nothink_vs3_wb",
            )
        )
        # the L3 pilots (2026-10-13)
        for arm in [
            "ei_think_rfb",
            "ei_think_v_ulb",
            "ei_nothink_3pb",
            "ei_nothink_v_rfb",
        ]:
            refusals.append(
                (
                    f"{H}/res/{mode}/s{s}/{arm}",
                    1,
                    rf"^00:00Z {arm}: _{arm.rsplit('_', 1)[1]} is a GRPO L3 pilot, and {arm.rsplit('_', 1)[0]} is EI: refusing$",
                    arm,
                )
            )
        for arm in [
            "ei_think_h_ulb",
            "grpo_think_h_rfb",
            "grpo_think_rfb_h",
            "grpo_think_b_ulb",
            "grpo_think_ulb_b",
            "grpo_think_c_rfb",
            "grpo_think_rfb_c",
            "grpo_think_vs_w_3pb",
            "grpo_think_vs_3pb_wb",
            "grpo_think_wb_ulb",
            "grpo_think_rfb_ulb",
            "grpo_think_vs_3pb_rfb",
            "grpo_think_ulb_3pb",
            "grpo_think_h_b_rfb",
        ]:
            refusals.append(
                (
                    f"{H}/res/{mode}/s{s}/{arm}",
                    1,
                    rf"^00:00Z {arm}: _rfb, _ulb and _3pb \(L3 pilots\) do not combine with other suffixes: refusing$",
                    arm,
                )
            )
        for arm in ["grpo_think_vs3_3pb", "grpo_nothink_vs3_3pb"]:
            refusals.append(
                (
                    f"{H}/res/{mode}/s{s}/{arm}",
                    1,
                    rf"^00:00Z {arm}: {arm[:-4]} already trains on the 3-premise rows: refusing$",
                    arm,
                )
            )
        for kind, arm, rx in (
            (
                "nockpt",
                "grpo_think_vs_rfb",
                rf"no L2 checkpoint of grpo_think_vs \({re.escape(SB)}/{c}/grpo_think_vs_s{s}_L2/ckpts/grpo_think_vs_s{s}_L2_it{it}\.pt\)",
            ),
            (
                "rb",
                "grpo_think_vs_ulb",
                r"grpo_think_vs's L2 log ends 'guard it 39: .*lr x0\.5 \(1 rollbacks\).*', not lr x1 \(0 rollbacks\): rerun the whole line",
            ),
            (
                "nolog",
                "grpo_think_vs_3pb",
                r"grpo_think_vs's L2 log ends 'no guard line', not lr x1 \(0 rollbacks\): rerun the whole line",
            ),
        ):
            refusals.append((f"{d}/{kind}_{arm}", 1, rf"^00:00Z {arm}: {rx}: refusing$", arm))
        # the curriculum_3p files are rebuilt by someone else: refused while one is missing
        for x in ("L3", "L2"):
            for arm in ("grpo_think_vs_3pb", "grpo_think_vs3", "grpo_nothink_vs3"):
                refusals.append(
                    (
                        f"{d}/no3p{x}_{arm}",
                        1,
                        rf"^00:00Z {arm}: no 3-premise pool file {re.escape(SB)}/pools/curriculum_3p/{x}_3p\.jsonl \(runs/pools/curriculum_3p\): refusing$",
                        arm,
                    )
                )
refusals.append(
    (
        f"{H}/res/refuse/normal/s3/grpo_think",
        1,
        rf"^00:00Z no Stage-1 init {re.escape(SB)}/stage1/S1g3_s3/model\.pt$",
        "grpo_think_s3",
    )
)
refusals.append((f"{H}/res/refuse/normal/usage", 2, r"^usage: line\.sh GPU SEED ARM$", "usage"))
for d, code, rx, arm in refusals:
    r = load(d)
    tag = d.replace(H + "/res/", "")
    ok(r["exit"] == str(code), "B", f"{tag}: exit {r['exit']} != {code}")
    ok(
        bool(re.match(rx, r["stdout"].strip())) and r["stdout"].count("\n") == 1,
        "B",
        f"{tag}: stdout {r['stdout'][:200]!r}",
    )
    ok(
        not any(
            x["kind"] == "timeout" or (x["kind"] == "py" and x["argv"].startswith("-m "))
            for x in r["rec"]
        )
        or arm == "grpo_think_s3",
        "B",
        f"{tag}: something was launched",
    )
    ok(
        not any(x["kind"] == "timeout" for x in r["rec"]),
        "B",
        f"{tag}: a stage or read was launched",
    )
    new = set(r["tree"]) - set(r["pre"])
    smoke_dirs = {t for t in new if "/grpo_v2/smoke" in t or t.endswith("./runs/eval/v2_smoke")}
    # a seed without an init fails after the preflight (as on serrano): its preflight log is the one file allowed
    wrote = [
        t
        for t in new - smoke_dirs
        if arm in t and not (arm == "grpo_think_s3" and t.endswith("/preflight_grpo_think_s3.log"))
    ]
    ok(not wrote, "B", f"{tag}: wrote {wrote[:3]}")

# ---------------------------------------------------------------- C: busy GPU and queue.sh
r = load(f"{H}/res/busy/grpo_think_s0")
nv = [x for x in r["rec"] if x["kind"] == "nvidia-smi"]
ok(
    r["exit"] == "1"
    and r["stdout"].strip().endswith("GPU 3 not free after 3 h: stopping this pipeline"),
    "C",
    f"busy: exit {r['exit']} {r['stdout'][-120:]!r}",
)
ok(
    len(nv) == 180
    and all("--query-gpu=memory.used" in x["argv"] for x in nv)
    and not any(x["kind"] == "timeout" for x in r["rec"]),
    "C",
    f"busy: {len(nv)} nvidia-smi calls, timeouts {sum(x['kind'] == 'timeout' for x in r['rec'])}",
)
ok(
    sum(x["kind"] == "sleep" and x["argv"] == "60" for x in r["rec"]) == 180,
    "C",
    "busy: 180 x sleep 60",
)
q = load(f"{H}/res/queue/normal")
ql = [ln for ln in q["stdout"].split("\n") if ln]
wantq = [
    f"00:00Z queue GPU 2: waiting for 'grpo_v2 line done (grpo_think seed 0)' in {SB}/logs/v2_grpo_think_s0.log",
    "00:00Z queue GPU 2: pre: echo pre ran",
    "pre ran",
    "00:00Z queue GPU 2: start grpo_think_v seed 1",
    "00:00Z queue GPU 2: grpo_think_v seed 1 exit 0: 00:00Z grpo_v2 line done (grpo_think_v seed 1)",
    "00:00Z queue GPU 2: start ei_nothink seed 2",
    "00:00Z queue GPU 2: ei_nothink seed 2 exit 0: 00:00Z grpo_v2 line done (ei_nothink seed 2)",
    "00:00Z queue GPU 2 done",
]
ok(q["exit"] == "0" and ql == wantq, "C", f"queue stdout {ql}")
# inside the queue the line's stdout and stderr are its log, $L/v2_ARM_sSEED.log
e1 = [
    tuple("<R>/logs/v2_grpo_think_v_s1.log" if v == "<stdout>" else v for v in x)
    for x in expect("grpo_think_v", 1, "normal", 2)[0]
]
e2 = [
    tuple("<R>/logs/v2_ei_nothink_s2.log" if v == "<stdout>" else v for v in x)
    for x in expect("ei_nothink", 2, "normal", 2)[0]
]
bk = (
    "python3",
    "/scratch/cowifornia/nd-rl/think/nd-rl",
    "CVD=unset",
    "<R>/logs/backup.log",
    "<R>/logs/backup.log",
    "<R>/backup.py",
)
hook = "--expect-hook" in sys.argv
qe = e1 + ([bk] if hook else []) + e2 + ([bk] if hook else [])
qg = [key(x) for x in q["rec"]]
ok(
    sorted(qg) == sorted(qe),
    "C",
    f"queue calls: missing {sorted(set(qe) - set(qg))[:2]} extra {sorted(set(qg) - set(qe))[:2]}",
)
ok(
    [x for x in qg if x[0] not in ("python3", "sleep") or x == bk]
    == [x for x in qe if x[0] not in ("python3", "sleep") or x == bk],
    "C",
    "queue call order (backup hook after each arm)",
)
for f, a, s in (
    ("v2_grpo_think_v_s1.log", "grpo_think_v", 1),
    ("v2_ei_nothink_s2.log", "ei_nothink", 2),
):
    p = f"{H}/res/queue/normal/logs/{f}"
    ok(
        os.path.exists(p)
        and Path(p).read_text().rstrip("\n").endswith(f"grpo_v2 line done ({a} seed {s})"),
        "C",
        f"queue: {f}",
    )

# ---------------------------------------------------------------- E: transcript evidence
E = json.loads(Path(f"{H}/evidence.json").read_text())


def real(argv: str) -> str:  # a recorded argv as the process's command line on serrano
    return argv.replace("<PY>", PYREAL).replace("<R>", RUNS).replace("\\,", ",")


# E1 the live grpo_think_h_b s1 L3 process (GPU 1)
r = load(f"{H}/res/gpu1/grpo_think_h_b_s1")
l3 = [x for x in r["rec"] if x["kind"] == "timeout" and " -m code.thinking.grpo " in x["argv"]]
ok(len(l3) == 1, "E1", f"{len(l3)} grpo commands")
cmd = real(l3[0]["argv"])
for ln in E["ps_hb"]["ps"]:
    pid, rest = ln.split(None, 2)[0], ln.split(None, 2)[2]
    full = "timeout " + cmd if rest.startswith("timeout ") else cmd.split(" ", 1)[1]
    ok(
        full.startswith(rest),
        "E1",
        f"ps {pid}: {rest[-80:]!r} is not a prefix of {full[: len(rest)][-80:]!r}",
    )
for p in E["ps_hb"]["proc"]:
    if p.startswith("--"):
        ok((" " + p + " ") in (" " + cmd + " "), "E1", f"/proc flag {p!r} not in the command")
    else:
        ok(l3[0]["cwd"] == p, "E1", f"cwd {l3[0]['cwd']} != {p}")
ok(l3[0]["cvd"] == "CVD=[1]", "E1", f"CUDA_VISIBLE_DEVICES {l3[0]['cvd']}")
for ln in E["ps_hb"]["launched"]:
    want = re.sub(r"^\d\d:\d\dZ ", "00:00Z ", re.sub(r"\(pid \d+\)", "(pid N)", ln))
    ok(want in stdout_lines(r), "E1", f"launch line {ln!r}")
# E2 serrano's harness, full L3 commands (SB = its sandbox runs/.., PY = its python stub)
mine = {
    a: [
        x["argv"]
        for x in runs[("normal", 1, a)]["rec"]
        if x["kind"] == "timeout" and "_L3 " in x["argv"] and "code.thinking.grpo" in x["argv"]
    ][0]
    for a in ("grpo_think_h_b", "grpo_think_h")
}
old = [ln.replace("SB/runs", "<R>").replace(" PY ", " <PY> ") for ln in E["old_full"]["lines"]]
ok(
    old[0] == mine["grpo_think_h_b"],
    "E2",
    "h_b s1 L3 command differs from serrano's harness record",
)
ok(
    old[1] == mine["grpo_think_h"].replace(BAN, ""),
    "E2",
    "h s1 L3 command (9,579-byte line.sh) differs once the ban flags are removed",
)
ok(
    [re.sub(r"\(pid \d+\)", "(pid N)", x) for x in E["old_full"]["stdout"]]
    == stdout_lines(runs[("normal", 1, "grpo_think_h_b")]),
    "E2",
    "h_b s1 stdout",
)


# E3 / E4 serrano's per-call records (cols kind, cwd, CVD, fd1, fd2, env-hash, argv; cut at N chars)
def oldline(x: dict, envh: str, subst: dict[str, str]) -> str:
    cols = [
        x["kind"],
        x["cwd"],
        x["cvd"],
        x["fd1"],
        x["fd2"],
        "env=" + envh,
        x["argv"].replace(BAN, ""),
    ]
    s = "\t".join(cols)
    for a, b in subst.items():
        s = s.replace(a, b)
    return s


for name, run, subst in (
    ("rec_vs3_s0", ("normal", 0, "grpo_nothink_vs3"), {}),
    (
        "rec_h_s1",
        ("normal", 1, "grpo_think_h"),
        {
            "<R>": E["rec_h_s1"]["prefix"] + "/sb/runs",
            "<PY>": E["rec_h_s1"]["prefix"] + "/pybin/python",
            "<stdout>": E["rec_h_s1"]["prefix"] + "/run/stdout.txt",
        },
    ),
):
    ev, cut = E[name]["lines"], E[name]["cut"]
    got = runs[run]["rec"]
    nm = lambda s: re.sub(r"(watchdog\.py [^:\s]+):\d+:", r"\1:N:", s)  # noqa: E731
    strip_env = lambda s: re.sub(r"\tenv=[^\t]*\t", "\t", s)  # noqa: E731
    # serrano's lines were cut at `cut` chars: each must be a prefix of one of mine (uncut, pids and env hash dropped)
    evn = [strip_env(nm(ln)) for ln in ev]
    mine = [strip_env(nm(oldline(x, "", subst))) for x in got]
    ok(len(got) >= len(ev), name, f"{len(got)} records < {len(ev)}")
    left, miss = list(mine), []
    for x in evn:
        k = next((i for i, y in enumerate(left) if y.startswith(x)), None)
        if k is None:
            miss.append(x)
        else:
            left.pop(k)
    ok(not miss, name, f"records not in mine: {[x[:200] for x in miss[:3]]}")
    # in order, without the calls next to the background stage (sleep, the watchdog launch)
    a = [x for x in evn if not x.startswith(("python3\t", "sleep\t"))]
    b = [y for y in mine if not y.startswith(("python3\t", "sleep\t"))][: len(a)]
    ok(
        len(a) == len(b) and all(y.startswith(x) for x, y in zip(a, b, strict=True)),
        name,
        "record order differs",
    )
    if name == "rec_vs3_s0":
        ok(len(ev) == len(got), name, f"{len(got)} calls != serrano's {len(ev)}")
        ok(
            [re.sub(r"\(pid \d+\)", "(pid N)", x) for x in E[name]["stdout"]]
            == stdout_lines(runs[run]),
            name,
            "stdout",
        )

# E5 old call counts
for oc in E["old_counts"]:
    t = oc["text"]
    m = re.match(r"(normal|smoke)\s+s(\d) (\S+)\s+exit (\d)\s+calls (\d+)", t)
    if m:
        mode, s, arm, ex, n = m.group(1), int(m.group(2)), m.group(3), m.group(4), int(m.group(5))
        r = runs[(mode, s, arm)]
        ok(
            r["exit"] == ex and len(r["rec"]) == n,
            "E5",
            f"{t}: mine exit {r['exit']} calls {len(r['rec'])}",
        )
        continue
    m = re.match(r"(\S+) s(\d) exit (\d), calls (\d+)", t)
    if m:
        r = runs[("normal", int(m.group(2)), m.group(1))]
        ok(
            r["exit"] == m.group(3) and len(r["rec"]) == int(m.group(4)),
            "E5",
            f"{t}: mine {len(r['rec'])}",
        )
        continue
    m = re.match(
        r"(normal|smoke)\s+s(\d) (\S+)_b\s+exit (\d)\s+evals (\d+)\s+L2-only calls dropped:(.*) ok$",
        t,
    )
    if m:
        mode, s, arm = m.group(1), int(m.group(2)), m.group(3)
        a = collections.Counter(key(x)[0] for x in runs[(mode, s, arm)]["rec"])
        b = collections.Counter(key(x)[0] for x in runs[(mode, s, arm + "_b")]["rec"])
        dropped = ", ".join(f"{v} {k}" for k, v in sorted((a - b).items()))
        ok(dropped == m.group(6).strip(), "E5", f"{t}: mine dropped {dropped}")

# E6 launcher log lines and ps lines of v2 arms
RN = re.compile(r"^(t1320|tb72|charles173|hand14|t2130)_(.+)_s(\d)_(plain|logical)_k(\d+)$")
allout = {k: set(stdout_lines(v)) for k, v in runs.items()}
n6, skipped = collections.Counter(), collections.Counter()


def find(arm: str, s: int, want: str, rx: bool = False) -> bool:
    for mode in ("normal", "smoke"):
        if (mode, s, arm) in allout:
            for ln in allout[(mode, s, arm)]:
                if re.match(want, ln) if rx else ln == want:
                    return True
    return False


for item in E["log_lines"]:
    ln = re.sub(r"^\d\d:\d\dZ ", "00:00Z ", item["line"])
    ln = re.sub(r"\(pid \d+\)", "(pid N)", ln)
    m = re.match(
        r"00:00Z launched (GRPO|EI) (\S+)_s(\d)_L([23]) \((.*)\) on GPU \d+ \(pid N\)$", ln
    )
    if m and m.group(2) in VALID:
        want = re.sub(r"on GPU \d+", "on GPU 3", ln)
        ok(find(m.group(2), int(m.group(3)), want), "E6", f"{item['line']} ({item['src']})")
        n6["launched"] += 1
        continue
    m = re.match(r"00:00Z read (\S+) exit \d+", ln)
    g = RN.match(m.group(1)) if m else None
    if m and g and g.group(2) in VALID:
        ok(
            find(
                g.group(2),
                int(g.group(3)),
                rf"^00:00Z read {re.escape(m.group(1))} exit 0 ",
                rx=True,
            ),
            "E6",
            f"{item['line'][:120]} ({item['src']})",
        )
        n6["read"] += 1
        continue
    m = re.match(r"00:00Z grpo_v2 line done \((\S+) seed (\d)\)$", ln)
    if m and m.group(1) in VALID:
        ok(find(m.group(1), int(m.group(2)), ln), "E6", item["line"])
        n6["done"] += 1
        continue
    m = re.match(r"00:00Z (GRPO|EI) (\S+)_s(\d)_L([23]) (exit|failed)", ln)
    if m and m.group(2) in VALID:
        ok(
            find(
                m.group(2),
                int(m.group(3)),
                rf"^00:00Z {m.group(1)} {re.escape(m.group(2))}_s{m.group(3)}_L{m.group(4)} exit ",
                rx=True,
            ),
            "E6",
            item["line"][:120],
        )
        n6["stage exit"] += 1
        continue
    m = re.match(
        r"00:00Z queue GPU (\d+)(: (start (\S+) seed (\d)|(\S+) seed (\d) exit \d+: .*|waiting for '.*' in \S+|pre: .*)| done)$",
        ln,
    )
    if m:
        n6["queue"] += 1
        a = m.group(4) or m.group(6)
        ok(a is None or a in VALID, "E6", f"queue line with unknown arm: {item['line']}")
        continue
    if "refusing" in ln:
        n6["refusing (other)"] += 1
        skipped["refusing: " + ln[:100]] += 1
        continue
    skipped[ln[:60]] += 1
for item in E["ps_lines"]:
    ln = item["line"].split(None, 2)[2]
    m = (
        re.search(r"--(?:name|ckpt) \S*?((?:grpo|ei)_[a-z_]+?)_s(\d)_L([23])", ln)
        or re.search(r"/grpo_v2/((?:grpo|ei)_[a-z_]+?)_s(\d)_L([23])", ln)
        or re.search(r"watchdog\.py ((?:grpo|ei)_[a-z_]+?)_s(\d)_L([23]):", ln)
    )
    if not m or m.group(1) not in VALID:
        skipped["ps: " + ln[:60]] += 1
        continue
    arm, s, L = m.group(1), int(m.group(2)), m.group(3)
    r = runs[("normal", s, arm)]
    cands = []
    for x in r["rec"]:
        if x["kind"] == "timeout":
            c = real(x["argv"])
            cands += ["timeout " + c, c.split(" ", 1)[1]]
        elif x["kind"] == "python3":
            cands.append("/home/cowifornia/.pyenv/versions/3.13.5/bin/python3 " + real(x["argv"]))
    pre = re.sub(r"(watchdog\.py [^:\s]+):\d+:\d+:", r"\1:N:G:", ln)
    hit = any(
        re.sub(r"(watchdog\.py [^:\s]+):N:\d+:", r"\1:N:G:", c).startswith(pre) for c in cands
    )
    ok(hit, "E6", f"ps line not a prefix of any of mine: {ln[:150]} ({item['src']})")
    n6["ps"] += 1

# ---------------------------------------------------------------- report
print(
    f"dry results: {H}/res   sources: "
    + " | ".join(Path(f"{H}/sources.txt").read_text().split("\n")[:3])
)
for sect in sorted({k[0] for k in counts}):
    print(f"  {sect:<10} {counts[(sect, True)]:>6} checks passed, {counts[(sect, False)]} failed")
print("  evidence lines matched: " + ", ".join(f"{k} {v}" for k, v in n6.items()))
if skipped:
    print(
        "  evidence lines not about a v2 arm (other launchers), skipped: "
        + str(sum(skipped.values()))
    )
    for k, v in sorted(skipped.items()):
        print(f"    {v} x {k}")
for f in fails[:60]:
    print("FAIL " + f)
print(f"{len(fails)} failures" if fails else "all checks passed")
sys.exit(1 if fails else 0)
