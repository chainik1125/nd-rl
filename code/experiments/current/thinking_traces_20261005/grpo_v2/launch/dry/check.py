#!/usr/bin/env python3
# ruff: noqa: E501  (literal command lines and log lines, compared byte for byte)
# pyright: basic
"""Check the dry runs of all.sh against an independent model of the GRPO v2 recipe and against the transcript evidence.

    python3 check.py [--expect-hook]     # after make.sh + all.sh; exits 1 on any failure

--expect-hook: queue.sh is the ynez version, which runs `python3 $R/backup.py >> $L/backup.log 2>&1 || true`
after each arm; without it the queue must make no backup call (serrano's queue.sh).

Sections
  A  every valid arm (12 base, 8 ARM_h, 16 ARM_b, 8 each of ARM_c, ARM_w, ARM_wb, ARM_rfb, ARM_ulb, ARM_pb, 6 ARM_3pb,
     6 ARM_mb, 3 ARM_fbb, 16 ARM_mb_kKNOB, 10 EI ARM_kKNOB) x seeds 0-2 x normal/SMOKE=1: exit 0, the full multiset and order of
     stubbed calls (argv, cwd, CUDA_VISIBLE_DEVICES, redirections), the environment of each python call, stdout, the
     eval dirs and logs, against `expect()` below (written from the README / evidence, not from line.sh's text);
     every flag at most once in each grpo / eval command; each ARM_mb_kKNOB's L3 command is ARM_mb's but for its
     knob's flags (KNOB_DIFF), and its reads are plain on t1320, t2130 and charles173 only (never tb72); each EI
     ARM_kKNOB's L3 command is ARM's L3 command but for its knob's flags (eknob_diff), with the same reads
  B  refusals: ei_*_h, ei_*_b, unknown arm, ARM_b without L2 checkpoint / with a rolled-back L2 / without an L2 log,
     a seed without a Stage-1 init, usage; ei_*_c / _w / _wb, the lr schedules with _h or _b (either order), ARM_wb
     without ARM's L2 checkpoint / with a rolled-back L2 / without an L2 log; the L3 pilots (ARM_rfb, ARM_ulb,
     ARM_3pb) for EI, with another suffix (either order), ARM_3pb for a _vs3 arm, as ARM_b's three; a missing
     curriculum_3p file for ARM_3pb and the _vs3 arms; ARM_mb's; ARM_fbb (2026-10-14) for EI, no-think and
     plain-sampling arms, with another suffix (either order), as ARM_b's three; ARM_mb_kKNOB (2026-10-15) with an
     unknown knob or a suffix after it, another suffix before _mb, EI, _vs3, _kp3 for a plain arm, a knob without
     _mb, as ARM_mb's L2 and 3-premise refusals; EI ARM_kKNOB (2026-10-16) with an unknown or a GRPO knob, a suffix
     before or after it, _mb_k on an EI arm, an EI knob on a GRPO arm, without the base arm's L2 checkpoint (_r1.pt),
     ARM_k3p without its mix, without its pass record or with a pass record of another mix; nothing launched
  C  waitfree with a busy GPU; queue.sh (wait line, PRE_CMD, two arms, done line[, backup hook])
  D  ARM_mb's averaging: a missing rolling checkpoint refuses the averaged reads; a rerun does not average again
  K  ARM_mb_ki78: an it39 L3 checkpoint does not end its 78-iteration stage; with its it78 checkpoint only the reads run
  M  the mini runs (2026-10-16): grpo_think_vs_mini and 11 knob arms x seeds 0-2 (the reference at 3 too) x normal /
     SMOKE=1, as section A, against `expect()`; from S1g3_s0 at every seed, L3-mini from L2-mini's last checkpoint;
     L2-mini's command is grpo_think_vs's L2 command and L3-mini's grpo_think_vs_mb_kkl0's L3 command but for the
     iterations, the subsets, --save_rolling 2, the timeout and the inits; each knob's two commands are the reference's
     but for its flags (`mknob`; kl1: L3-mini's only); the five plain reads (never tb72)
  N  mini reruns: with L2-mini's last checkpoint only L3-mini and the reads run; with both stages and two reads, the
     others; MINI_ROLL: MINI_PANEL of the rolling checkpoints named, one missing skipped
  Q  ARM_ckreads (2026-10-16): a knob pilot's L3 read at it0 (its init), it10 / it20 / it30 (rolling) and it39 (its own
     names: the pilot reads there are skipped); a missing rolling checkpoint skips its reads; HEAD in ckreads_commit.txt,
     the pilot's base_commit.txt untouched; refusals in B (other arms, i78, KNOB_FULL=1, no finished L3, no L2)
  J  EI ARM_kKNOB (2026-10-16): ARM_kr2's stage ends at _r2.pt (an _r1.pt does not end it; with _r2.pt only the reads
     run, on it); a rerun of a finished pilot runs nothing, of a partly read one the missing reads; KNOB_FULL=1 on a
     finished pilot (EI, and a GRPO knob arm) runs ARM's other reads only and records HEAD in reads_commit.txt, on a
     fresh one the stage and all of ARM's reads
  E  transcript evidence (evidence.json): the live grpo_think_h_b s1 L3 process; serrano's harness records (full
     commands and per-call records of three runs, call counts); every launcher log line and ps line of a v2 arm
"""

from __future__ import annotations

import collections
import hashlib
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
    "_pb": "--unseen_prior mean",  # 2026-10-13: the unseen-pair prior at the budget's mean
    "_mb": "--unlikely_beta 0.25",  # 2026-10-14, the merged recipe: _ulb's, with _wb's cooldown and _3pb's rows
    "_fbb": "--feedback context",  # 2026-10-14: in-context verifier feedback (grpo_think_vs_fbb asked for)
}
FB_MAXREJ = 2  # ARM_fbb: retries per step in L3 and in its feedback reads
# ARM_fbb's guard: grpo.py has --feedback and eval.py the feedback read arm
FB_GUARD = (
    "-c import\\ sys\\,\\ code.thinking.grpo\\ as\\ g\\,\\ code.thinking.eval\\ as\\ e\\;\\ "
    "sys.exit\\(0\\ if\\ hasattr\\(g\\,\\ \\'FEEDBACK_MODES\\'\\)\\ and\\ \\'feedback\\'\\ in\\ "
    "getattr\\(e\\,\\ \\'READ_ARMS\\'\\,\\ \\(\\)\\)\\ else\\ 1\\)"
)
GUARD = {
    "_rfb": "pilot_options",
    "_ulb": "pilot_options",
    "_pb": "UNSEEN_PRIORS",
    "_mb": "pilot_options",
}
RFBARMS, ULBARMS = ([a + x for a in GBASE] for x in ("_rfb", "_ulb"))
P3BARMS = [a + "_3pb" for a in GBASE if not a.endswith("_vs3")]
PBARMS = [a + "_pb" for a in GBASE]
MBARMS = [a + "_mb" for a in GBASE if not a.endswith("_vs3")]
# ARM_fbb: only the thinking arms with the verifier (the lines go into the think block)
FBBARMS = [a + "_fbb" for a in ("grpo_think_v", "grpo_think_vs", "grpo_think_vs3")]
# ARM_mb_kKNOB (2026-10-15; written from the request): ARM_mb's L3 with one setting changed, then plain reads on 13-20,
# 21-30 and charles173 only. KNOB_DIFF: the knob's flags as (ARM_mb's value, the knob's) per mode; None: not passed
KNOB_DIFF = {
    "x66": {"--targets_extra_frac": ("0.33", "0.66")},
    "g64": {
        "--prompts": ("256", "64"),
        "--group": ("16", "64"),
    },  # SMOKE=1: 4 x 4 -> 2 x 8 (KNOB_SMOKE)
    "t12": {"--temperature": (None, "1.2")},
    "kl0": {"--kl_anchor_coef": ("0.02", "0")},
    "kl1": {"--kl_anchor_coef": ("0.02", "0.1")},
    "ch4": {"--clip_eps_high": (None, "0.4")},
    "p3": {"--rej_penalty": ("0.1", "0.3")},
    "i78": {"--iters": ("39", "78"), "--save_every": ("39", "78")},
    "rr": {},
}
KNOB_SMOKE = {
    "g64": {"--prompts": ("4", "2"), "--group": ("4", "8")},
    "i78": {"--iters": ("2", "4"), "--save_every": ("2", "4")},
}
KNOB_RX = re.compile(r"(.+_mb)_k(" + "|".join(KNOB_DIFF) + ")")
KARMS = [f"grpo_think_vs_mb_k{k}" for k in KNOB_DIFF] + [
    "grpo_think_mb_kg64",
    "grpo_think_v_mb_kp3",
    "grpo_nothink_vs_mb_kt12",
    "grpo_nothink_mb_ki78",
    "grpo_nothink_v_mb_kx66",
    "grpo_think_mb_krr",
    "grpo_nothink_vs_mb_kkl1",
]
KNOB_READS = [("t1320", 16), ("t2130", 8), ("charles173", 64)]
# EI ARM_kKNOB (2026-10-16; written from the request): ARM's L3 from ARM's L2 (_r1.pt) with one setting changed, then
# ARM_mb_kKNOB's plain reads
EKNOBS = ["rr", "3p", "ret0", "t10", "r2"]
EKNOB_RX = re.compile(r"(ei_(?:no)?think(?:_v)?)_k(" + "|".join(EKNOBS) + ")")
EKARMS = [f"ei_think_v_k{k}" for k in EKNOBS] + [
    "ei_think_kr2",
    "ei_think_k3p",
    "ei_nothink_v_kret0",
    "ei_nothink_kt10",
    "ei_nothink_v_krr",
]
M3 = "<R>/pools/curriculum_3p/L3_mix_ei.jsonl"  # ARM_k3p's targets
# the mini runs (2026-10-16; written from the request): grpo_think_vs's L2 and grpo_think_vs_mb_kkl0's L3 (the primary
# recipe), MINI_IT iterations each, chained from Stage-1 S1g3_s0 whatever the seed, on the mini subsets, a rolling
# checkpoint every 2 iterations; a knob; then plain reads in and out of distribution, never tb72
MINI_IT = 10
MKNOB_RX = r"rr|t07|t085|t115|t13|lpm?\d\d|kl1|g64"
MINI_RX = re.compile(r"grpo_think_vs_mini(?:_k(" + MKNOB_RX + "))?")
MARMS = ["grpo_think_vs_mini"] + [
    f"grpo_think_vs_mini_k{k}"
    for k in ("rr", "t07", "t085", "t115", "t13", "lpm10", "lp20", "lp50", "lp05", "kl1", "g64")
]


def mknob(
    mk: str, smoke: bool
) -> tuple[dict[str, tuple[str | None, str]], dict[str, tuple[str | None, str]], str]:
    """A mini knob -> (its flags in both stages, its flags in L3-mini only, as (the reference's value, the knob's); None:
    not passed), and its launch-line text."""
    both: dict[str, tuple[str | None, str]] = {}
    l3: dict[str, tuple[str | None, str]] = {}
    if mk in ("t07", "t085", "t115", "t13"):
        both["--temperature"] = (
            None,
            {"t07": "0.7", "t085": "0.85", "t115": "1.15", "t13": "1.3"}[mk],
        )
    elif mk.startswith("lp"):  # lpNN / lpmNN: +-NN hundredths
        v = int(mk[-2:]) / 100
        both["--len_penalty"] = (None, f"{-v if mk.startswith('lpm') else v:g}")
    elif mk == "g64":  # SMOKE=1: 4 x 4 -> 2 x 8
        both = (
            {"--prompts": ("4", "2"), "--group": ("4", "8")}
            if smoke
            else {"--prompts": ("256", "64"), "--group": ("16", "64")}
        )
    elif mk == "kl1":
        l3["--kl_anchor_coef"] = ("0", "0.1")
    desc = " ".join(f"{f} {v}" for f, (_, v) in both.items()) or (
        "--kl_anchor_coef 0.1 in L3" if l3 else "nothing changed"
    )
    return both, l3, desc


MINI = "<R>/pools/mini"
_CH = "<R>/pools/charles_1002"
READ_SETS = {  # the reads' files, as recorded (escaped commas)
    "l2test": "<R>/pools/curriculum/L2/test_1k.jsonl",
    "l3test": "<R>/pools/curriculum/L3/test_1k.jsonl",
    "t1320": "<R>/eval/matched/sets/t1320_n1000.jsonl",
    "t2130": "<R>/eval/matched/sets/t2130_n1000.jsonl",
    "charles173": f"{_CH}/candidate_v0.jsonl\\,{_CH}/candidate_v1.jsonl\\,{_CH}/batch3.jsonl",
}
MINI_READS = [("l2test", 8), ("l3test", 8), ("t1320", 8), ("t2130", 8), ("charles173", 64)]
CKR_READS = [("l2test", 8), ("l3test", 8), ("t1320", 16), ("t2130", 8), ("charles173", 64)]


def eknob_diff(knob: str, smoke: bool) -> dict[str, tuple[str | None, str]]:
    """EI ARM_kKNOB's L3 flags as (ARM's L3 value, the knob's); None: not passed."""
    c = "<R>/grpo_v2/smoke" if smoke else "<R>/grpo_v2"
    l3, m3 = (
        (f"{c}/L3_head.jsonl", f"{c}/M3_head.jsonl")
        if smoke
        else ("<R>/pools/curriculum/L3/pool.jsonl", M3)
    )
    return {
        "rr": {},
        "3p": {"--targets": (l3, m3)},  # SMOKE=1: the mix's first 8 rows
        "ret0": {"--retain": ("64" if smoke else None, "0")},
        "t10": {"--temperature": ("0.8", "1.0")},
        "r2": {  # the same attempts and fine-tune steps in two rounds (SMOKE=1: halved, rounded down)
            "--rounds": ("1", "2"),
            "--k": ("2", "1") if smoke else ("32", "16"),
            "--ft_steps": ("3", "1") if smoke else (None, "300"),
        },
    }[knob]


REUSE = (
    BARMS + WBARMS + RFBARMS + ULBARMS + P3BARMS + PBARMS + MBARMS + FBBARMS + KARMS + EKARMS
)  # L3 only, from another arm's L2
VALID = (
    BASE
    + HARMS
    + BARMS
    + CARMS
    + WARMS
    + WBARMS
    + RFBARMS
    + ULBARMS
    + P3BARMS
    + PBARMS
    + MBARMS
    + FBBARMS
    + KARMS
    + EKARMS
)
# ARM_mb's averaged reads (pilot4.sh's): the L3 rolling checkpoints averaged with the last one, and the plain reads
AVG_ITS = [25, 30, 35]
AVG_READS = [("t1320", 16), ("tb72", 64), ("charles173", 64), ("t2130", 8)]
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
def expect(
    arm: str,
    s: int,
    mode: str,
    g: int,
    avg_there: bool = False,
    full: bool = False,
    l3_there: bool = False,
    reads_there: frozenset[str] = frozenset(),
    l2_there: bool = False,
) -> tuple[list[tuple], list[str]]:
    """-> (ordered records (kind, cwd, cvd, fd1, fd2, argv), stdout lines) for a valid arm. `avg_there`: ARM_mb
    with its averaged checkpoint already written (a rerun). `full`: KNOB_FULL=1 (a knob arm). `l3_there`: the L3
    stage's last checkpoint is there; `reads_there`: the eval dirs (basenames) whose rows are there (a rerun);
    `l2_there`: the L2 stage's (a mini run's rerun)."""
    # ARM_mb_kKNOB / EI ARM_kKNOB: the recipe of ARM_mb / ARM ($pa), every name the full arm's
    km = KNOB_RX.fullmatch(arm) or EKNOB_RX.fullmatch(arm)
    knob, pa = (km.group(2), km.group(1)) if km else ("", arm)
    mm = MINI_RX.fullmatch(
        arm
    )  # a mini run: L3-mini is grpo_think_vs_mb_kkl0's recipe ($pa), its knob mk in both stages
    mk = (mm.group(1) or "") if mm else ""
    if mm:
        pa = "grpo_think_vs_mb"
    pil = next((x for x in PILOT if pa.endswith(x)), "")
    sch = "" if pil else next((x for x in ("_wb", "_c", "_w") if pa.endswith(x)), "")
    core = pa[: -len(pil or sch)] if pil or sch else pa
    src = core[:-2] if core.endswith("_b") and not pil else core
    base = src[:-2] if src.endswith("_h") and not pil else src
    sched = {"": ("", ""), "_c": (SCHED_C, SCHED_C), "_w": ("", SCHED_W), "_wb": ("", SCHED_W)}[sch]
    l2src = base if sch == "_wb" or pil else (arm if sch else src)  # whose L2 the L3 starts from
    if mm:  # both stages, from the Stage-1 init
        l2src = arm
    pflags = PILOT.get(pil, "")  # stage L3's pilot flags
    if pil == "_mb":
        sched = ("", SCHED_W)  # _wb's L3 cooldown
    smoke = mode == "smoke"
    think = "_think" in base
    method = "grpo" if base.startswith("grpo_") else "ei"
    samp = "logical" if base.endswith(("_v", "_vs", "_vs3")) else "plain"
    vs3 = base.endswith("_vs3") or pil in ("_3pb", "_mb")  # the 3-premise rows as extra targets
    R, C = "<R>", "<R>/grpo_v2" + ("/smoke" if smoke else "")
    EV = "<R>/eval/v2" + ("_smoke" if smoke else "")
    L = C + "/logs" if smoke else "<R>/logs"
    it, batch, cvd = (2, 16, "CVD=[]") if smoke else (39, 1024, f"CVD=[{g}]")
    kd = {**KNOB_DIFF.get(knob, {}), **(KNOB_SMOKE.get(knob, {}) if smoke else {})}
    mk3: dict[str, tuple[str | None, str]] = {}
    if mm:  # MINI_IT iterations a stage (SMOKE=1: 2, as every stage); the mini knob's flags
        it = it if smoke else MINI_IT
        kd, mk3, _ = mknob(mk, smoke)

    def kv(flag: str, default: str) -> str:  # the knob's value of a flag the recipe passes
        return kd[flag][1] if flag in kd else default

    it3 = int(kv("--iters", str(it)))  # stage L3's iterations
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
    if pil == "_fbb":  # ARM_fbb: the worktree's grpo.py and eval.py must have the feedback options
        recs.append(("py", WT, "CVD=unset", "/dev/null", "/dev/null", FB_GUARD))
    elif pflags:  # ARM_rfb / ARM_ulb / ARM_pb: the worktree's grpo.py must have their options
        recs.append(
            (
                "py",
                WT,
                "CVD=unset",
                "/dev/null",
                "/dev/null",
                "-c import\\ sys\\,\\ code.thinking.grpo\\ as\\ g\\;\\ sys.exit\\(0\\ if\\ hasattr\\(g\\,\\ "
                f"\\'{GUARD[pil]}\\'\\)\\ else\\ 1\\)",
            )
        )
    l2t = f"{C}/L2_head.jsonl" if smoke else f"{R}/pools/curriculum/L2/pool.jsonl"
    l3t = f"{C}/L3_head.jsonl" if smoke else f"{R}/pools/curriculum/L3/pool.jsonl"
    x2 = (f"{C}/X2_head.jsonl" if smoke else f"{R}/pools/curriculum_3p/L2_3p.jsonl") if vs3 else ""
    x3 = (f"{C}/X3_head.jsonl" if smoke else f"{R}/pools/curriculum_3p/L3_3p.jsonl") if vs3 else ""
    if (
        mm
    ):  # the subsets (SMOKE=1: their first 8 rows, as for the pools); L2-mini has no extra targets
        l2t = f"{C}/L2_head.jsonl" if smoke else f"{MINI}/L2_mini.jsonl"
        l3t = f"{C}/L3_head.jsonl" if smoke else f"{MINI}/L3_mini.jsonl"
        x2, x3 = "", (f"{C}/X3_head.jsonl" if smoke else f"{MINI}/L3_3p_mini.jsonl")
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
                f"{pf} {l2t} {l3t}",
            )
        )
        if vs3 or knob == "3p":  # EI ARM_k3p: its mix against the L2 pool and every evaluation set
            f = f"{C}/preflight_{arm}_s{s}_extra.log"
            xa = (
                f"<R>/pools/curriculum/L2/pool.jsonl {M3}"
                if knob == "3p"
                else "<R>/pools/curriculum_3p/L2_3p.jsonl <R>/pools/curriculum_3p/L3_3p.jsonl"
            )
            if mm:  # the 3-premise subset against the L2 subset
                xa = f"{l2t} {x3}"
            recs.append(
                (
                    "py",
                    WT,
                    "CVD=unset",
                    f,
                    f,
                    f"{pf} {xa}",
                )
            )
    recs.append(
        (
            "git",
            WT,
            "CVD=unset",
            f"{C}/{arm}_s{s}.{'reads_commit' if knob and full else 'base_commit'}.txt",
            st,
            f"-C {WT} rev-parse HEAD",
        )
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

    rd3 = 2 if knob == "r2" and method == "ei" else 1  # EI stage L3's rounds

    def ck(name: str) -> str:
        return (
            f"{C}/{name}/ckpts/{name}_it{it3 if name.endswith('_L3') else it}.pt"
            if method == "grpo"
            else f"{C}/{name}/ckpts/{name}_r{rd3 if name.endswith('_L3') else 1}.pt"
        )

    def stage(
        name: str, init: str, targets: str, extra: str, sc: str = "", pflags: str = ""
    ) -> None:
        if l3_there and name.endswith("_L3"):  # its last checkpoint is there: nothing runs
            return
        if l2_there and name.endswith("_L2"):
            return
        waitfree()
        log = f"{L}/v2_{name}.log"
        l3 = name.endswith("_L3")  # the knob is stage L3's (a knob arm has no L2 of its own)
        if method == "grpo":
            n_it = it3 if l3 else it
            new = (
                [f for f, (old, _) in kd.items() if old is None] if l3 or mm else []
            )  # flags the recipe leaves out (a mini run's knob: both stages)
            kf = "".join(
                f"{f} {kd[f][1]} "
                for f in (
                    "--temperature",
                    "--clip_eps_high",
                    "--len_penalty",
                    "--updates",
                    "--warmup",
                )
                if f in new
            )
            assert len(kf.split()) == 2 * len(new), "a new knob flag without a place in the command"
            a = (
                f"{7200 if mm else 43200 if n_it == 2 * it else 21600} <PY> -m code.thinking.grpo --init {init} --name {name} "
                f"--out {C}/{name} --targets {targets} "
                f"--iters {n_it} --prompts {kv('--prompts', '4' if smoke else '256')} "
                f"--group {kv('--group', '4' if smoke else '16')} --batch {batch} --lr {lr} "
                + (f"{sc} " if sc else "")
                + (f"{pflags} " if pflags else "")
                + kf
                + f"--overlong fail --save_every {n_it} --max_steps 96 --seed {s} --keep_rolling 100 "
                + ("--save_rolling 2 " if mm else "")
                + f"{eff} "
                f"--canary <R>/pools/canary_long.jsonl --canary_n {40 if smoke else 1000} --canary_every 1 "
                f"--rollback_on_alarm --max_rollbacks 3{BAN} --kl_anchor_coef {mk3.get('--kl_anchor_coef', (None, '0'))[1] if mm and l3 else kv('--kl_anchor_coef', '0.02')} "
                f"--kl_anchor_ref {init}" + (" --kl_anchor_n 16" if smoke else "") + " --monitor"
            )
            p_ = kv("--rej_penalty", "0.1")
            xf = kv("--targets_extra_frac", "0.33")
            if samp == "logical":
                mr = 0 if base.endswith(("_vs", "_vs3")) else 10
                if pil == "_fbb" and name.endswith("_L3"):  # retries per step, with --feedback
                    mr = FB_MAXREJ
                a += f" --arm logical --max_rej {mr} --rej_penalty {p_}"
            if extra:
                a += f" --targets_extra {extra} --targets_extra_frac {xf}"
            kdesc = ""
            if knob and l3:
                kdesc = " ".join(f"{f} {v}" for f, (_, v) in kd.items() if f != "--save_every")
                kdesc = kdesc or "nothing changed"
                kdesc = f", knob {knob}: {kdesc}"
            elif mk:
                kdesc = f", knob {mk}: {mknob(mk, smoke)[2]}"
            desc = (
                samp
                + (f", P {p_}" if samp == "logical" else "")
                + (f", extra {os.path.basename(extra)} x {xf}" if extra else "")
                + (f", lr {lr}" if base != src else "")
                + (f", {sc}" if sc else "")
                + (f", {pflags}" if pflags else "")
                + kdesc
            )
            out.append(f"00:00Z launched GRPO {name} ({desc}) on GPU {g} (pid N)")
            out.append(f"00:00Z GRPO {name} exit 0: it {n_it}: dry")
            if mm:  # its alarms (none: the stub's log has none) and rollbacks
                out.append(f"00:00Z GRPO {name} rollbacks: no guard line")
            wh = "1" if mm else "7" if n_it == 2 * it else "3.5"
        else:
            # EI ARM_kKNOB: stage L3's knob flags
            ed = eknob_diff(knob, smoke) if knob and l3 else {}

            def ekv(flag: str, default: str | None) -> str | None:
                return ed[flag][1] if flag in ed else default

            ret = ekv("--retain", "64" if smoke else None)
            fs = ekv("--ft_steps", "3" if smoke else None)
            a = (
                f"21600 <PY> -m code.thinking.ei --init {init} --name {name} --out {C}/{name} --targets {ekv('--targets', targets)} "
                f"--train <R>/data/{'S1g3' if think else 'T0S'}.pkl --rounds {ekv('--rounds', '1')} "
                f"--k {ekv('--k', '2' if smoke else '32')} --temperature {ekv('--temperature', '0.8')} "
                f"--arm {samp} --max_rej 10 --batch {batch} --seed {s} --max_steps 96 --select shortest {eff}"
                + (f" --ft_steps {fs}" if fs is not None else "")
                + (" --ft_recs 8" if smoke else "")
                + (f" --retain {ret}" if ret is not None else "")
            )
            kdesc = ""
            if knob and l3:
                kdesc = " ".join(
                    f"{f} {os.path.basename(v) if f == '--targets' else v}"
                    for f, (_, v) in ed.items()
                )
                kdesc = f", knob {knob}: {kdesc or 'nothing changed'}"
            out.append(f"00:00Z launched EI {name} ({samp}{kdesc}) on GPU {g} (pid N)")
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
    if mm:  # one init for every seed: the seed is the RL's
        init0 = "<R>/stage1/S1g3_s0/model.pt"
    if l2src == arm:
        stage(f"{arm}_s{s}_L2", init0, l2t, x2, sched[0])
        l2ck = ck(f"{arm}_s{s}_L2")
    elif method == "ei":  # EI ARM_kKNOB: ARM's one round
        l2ck = f"{C}/{l2src}_s{s}_L2/ckpts/{l2src}_s{s}_L2_r1.pt"
    else:
        l2ck = f"{C}/{l2src}_s{s}_L2/ckpts/{l2src}_s{s}_L2_it{it}.pt"
    stage(f"{arm}_s{s}_L3", l2ck, l3t, x3, sched[1], pflags)
    ch = "<R>/pools/charles_1002"
    reads = []
    # KNOB_FULL=1: a knob arm gets the usual reads
    for ra in ("plain", "logical") if not knob or full else ():
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
    if pil == "_fbb":  # the feedback read arm on 13-20 and charles173, FB_MAXREJ retries per step
        reads += [
            ("t1320", "<R>/eval/matched/sets/t1320_n1000.jsonl", 16, "feedback", 96, FB_MAXREJ),
            (
                "charles173",
                f"{ch}/candidate_v0.jsonl\\,{ch}/candidate_v1.jsonl\\,{ch}/batch3.jsonl",
                64,
                "feedback",
                96,
                FB_MAXREJ,
            ),
        ]
    if not knob or full:
        reads += [
            ("hand14", f"{ch}/hand_proved_14.jsonl", 64, "plain", 160),
            ("t2130", "<R>/eval/matched/sets/t2130_n1000.jsonl", 8, "plain", 96),
        ]
    else:  # ARM_mb_kKNOB: plain on 13-20, 21-30 and charles173, in that order; nothing else
        sets = {
            "t1320": "<R>/eval/matched/sets/t1320_n1000.jsonl",
            "t2130": "<R>/eval/matched/sets/t2130_n1000.jsonl",
            "charles173": f"{ch}/candidate_v0.jsonl\\,{ch}/candidate_v1.jsonl\\,{ch}/batch3.jsonl",
        }
        reads += [(n, sets[n], k, "plain", 96) for n, k in KNOB_READS]
    if mm:  # a mini run: MINI_READS of L3-mini's last checkpoint, plain, in and out of distribution
        reads = [(n, READ_SETS[n], k, "plain", 96) for n, k in MINI_READS]
    av = f"{C}/{arm}_avg_s{s}_L3/ckpts/{arm}_avg_s{s}_L3_it{it}.pt"
    sets_of = {n: x for n, x, *_ in reads}
    n_usual = len(reads)
    if (
        pil == "_mb" and not knob and not mm
    ):  # after the usual reads: the averaging, then the averaged checkpoint's plain reads
        reads += [(n, sets_of[n], k, "plain", 96) for n, k in AVG_READS]
    reads = [r if len(r) == 6 else (*r, 10) for r in reads]  # --max_rej 10 unless given (ARM_fbb's)
    for j, (name, sets, k, ra, ms, mrr) in enumerate(reads):
        k = 2 if smoke else k
        avg = j >= n_usual  # ARM_mb's averaged reads
        if avg and j == n_usual and not avg_there:
            its = [] if smoke else AVG_ITS
            fs = [f"{C}/{arm}_s{s}_L3/ckpts/{arm}_s{s}_L3_it{i}_rolling.pt" for i in its]
            fs.append(ck(f"{arm}_s{s}_L3"))
            al = f"{L}/v2_avg_{arm}_s{s}.log"
            recs.append(("py", WT, "CVD=[]", al, al, " ".join(["-", av, *fs])))
            out.append(
                f"00:00Z averaged {arm}_s{s}_L3 it {' '.join(map(str, [*its, it]))} -> {av.replace('<R>', SB)}"
            )
        o = f"{EV}/{name}_{arm}{'_avg' if avg else ''}_s{s}_{ra}_k{k}"
        if os.path.basename(o) in reads_there:  # its rows are there: skipped
            continue
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
                f"9000 <PY> -m code.thinking.eval --ckpt {av if avg else ck(f'{arm}_s{s}_L3')} --sets {sets} --out {o} --k {k} "
                f"--budgets {rb} --arms {ra} --batch {batch} --seed 0 --temperature 0.8 --max_action 512 "
                f"--max_steps {ms} --max_rej {mrr}" + (" --limit 2" if smoke else ""),
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
                    + rf"_s{s}_(plain|logical|feedback)_k\d+$",
                    t,
                )
            ]
            fbb = arm.endswith("_fbb")  # two feedback reads
            kn = arm in KARMS
            ek = arm in EKARMS
            kr = kn or ek  # three plain reads
            ok(len(ev) == (3 if kr else 8 + 2 * fbb), "A", f"{tag}: {len(ev)} eval dirs")
            lg = "./runs/grpo_v2/smoke/logs" if mode == "smoke" else "./runs/logs"
            el = [t for t in r["tree"] if t.startswith(f"f {lg}/v2_eval_")]
            mb = arm.endswith("_mb")
            if mb:  # the averaged checkpoint and its 4 plain reads
                ev_avg = [
                    t
                    for t in r["tree"]
                    if re.match(
                        rf"^d \./runs/eval/v2(_smoke)?/[a-z0-9]+_{re.escape(arm)}_avg_s{s}_plain_k\d+$",
                        t,
                    )
                ]
                ok(len(ev_avg) == 4, "A", f"{tag}: {len(ev_avg)} averaged-read dirs")
                c_ = "grpo_v2/smoke" if mode == "smoke" else "grpo_v2"
                a_ = f"{arm}_avg_s{s}_L3"
                ok(
                    f"f ./runs/{c_}/{a_}/ckpts/{a_}_it{2 if mode == 'smoke' else 39}.pt"
                    in r["tree"],
                    "A",
                    f"{tag}: averaged ckpt",
                )
            ok(
                len(el) == (12 if mb else 10 if fbb else 3 if kr else 8)
                and all(
                    re.match(
                        rf"^f {re.escape(lg)}/v2_eval_(t1320|tb72|charles173|hand14|t2130)_{re.escape(arm)}(_avg)?_s{s}_(plain|logical|feedback)_k\d+\.log$",
                        t,
                    )
                    for t in el
                ),
                "A",
                f"{tag}: eval logs {el[:2]}",
            )
            if kr:  # never textbook72, the guided arm, hand14 or an averaged checkpoint
                ok(
                    sorted(re.sub(r"^.*/v2_eval_", "", t) for t in el)
                    == sorted(
                        f"{n}_{arm}_s{s}_plain_k{2 if mode == 'smoke' else k}.log"
                        for n, k in KNOB_READS
                    ),
                    "A",
                    f"{tag}: knob reads {el}",
                )
                ok(
                    not any("tb72" in t or "_avg" in t or "logical" in t for t in r["tree"]),
                    "A",
                    f"{tag}: a tb72 / averaged / guided read",
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
                else (
                    46
                    # the extra preflight
                    + ("_vs3" in arm or arm.endswith(("_3pb", "_mb", "_k3p")) or kn)
                    - (5 if arm in REUSE else 0)
                )
            ) + (
                arm.endswith(("_rfb", "_ulb", "_pb", "_mb", "_fbb")) or kn
            )  # the pilot options check
            if kr:  # 3 reads instead of 8: waitfree (2, normal), rm, timeout each
                want -= 5 * (4 if mode == "normal" else 2)
            if fbb:  # two more reads: waitfree (2, normal), rm, timeout
                want += 8 if mode == "normal" else 4
            if (
                mb
            ):  # the averaging, and 4 reads (rm, eval; and waitfree's 2 nvidia-smi outside SMOKE=1)
                want += 1 + 4 * (2 if mode == "smoke" else 4)
            ok(n == want, "A", f"{tag}: {n} calls, want {want}")
            for x in r["rec"]:
                if x["kind"] == "timeout" and " -m code.thinking." in x["argv"]:
                    fl = [t for t in x["argv"].split() if t.startswith("--")]
                    dup = sorted({t for t in fl if fl.count(t) > 1})
                    ok(not dup, "A", f"{tag}: flags given twice {dup} in {x['argv'][:80]}")


def flags(argv: str) -> dict[str, str | None]:
    """A recorded grpo command -> {flag: its value (None: a switch)}, the timeout's limit as 'timeout'."""
    t = argv.split()
    d: dict[str, str | None] = {"timeout": t[0]}
    for i, x in enumerate(t):
        if x.startswith("--"):
            v = t[i + 1] if i + 1 < len(t) and not t[i + 1].startswith("--") else None
            d[x] = v
    return d


for mode in ("normal", "smoke"):
    for s in (0, 1, 2):
        for arm in KARMS:
            km = KNOB_RX.fullmatch(arm)
            assert km
            mb, knob = km.group(1), km.group(2)
            tag = f"{mode} s{s} {arm}"
            mine, theirs = (
                [
                    x["argv"]
                    for x in runs[(mode, s, a)]["rec"]
                    if x["kind"] == "timeout" and " -m code.thinking.grpo " in x["argv"]
                ]
                for a in (arm, mb)
            )
            if not ok(len(mine) == 1 and len(theirs) == 1, "A", f"{tag}: L3 commands {len(mine)}"):
                continue
            fk = flags(mine[0])
            fm = flags(theirs[0].replace(f"{mb}_s{s}_L3", f"{arm}_s{s}_L3"))
            want = {**KNOB_DIFF[knob], **(KNOB_SMOKE.get(knob, {}) if mode == "smoke" else {})}
            if knob == "i78":
                want["timeout"] = ("21600", "43200")
            diff = {f: (fm.get(f), fk.get(f)) for f in set(fk) | set(fm) if fm.get(f) != fk.get(f)}
            diff |= {
                f: (None, None) for f in set(fk) ^ set(fm) if f not in diff
            }  # a switch added or dropped
            ok(diff == want, "A", f"{tag}: L3 command vs {mb}'s: {diff}, want {want}")

for mode in ("normal", "smoke"):
    for s in (0, 1, 2):
        # EI ARM_kKNOB: ARM's L3 command (the same init, ARM's L2 _r1.pt) but for the knob's flags
        for arm in EKARMS:
            km = EKNOB_RX.fullmatch(arm)
            assert km
            b, knob = km.group(1), km.group(2)
            tag = f"{mode} s{s} {arm}"
            mine, theirs = (
                [
                    x["argv"]
                    for x in runs[(mode, s, a)]["rec"]
                    if x["kind"] == "timeout" and f" --name {a}_s{s}_L3 " in x["argv"]
                ]
                for a in (arm, b)
            )
            if not ok(len(mine) == 1 and len(theirs) == 1, "A", f"{tag}: L3 commands {len(mine)}"):
                continue
            fk = flags(mine[0])
            fm = flags(theirs[0].replace(f"{b}_s{s}_L3", f"{arm}_s{s}_L3"))
            want = eknob_diff(knob, mode == "smoke")
            diff = {f: (fm.get(f), fk.get(f)) for f in set(fk) | set(fm) if fm.get(f) != fk.get(f)}
            diff |= {f: (None, None) for f in set(fk) ^ set(fm) if f not in diff}
            ok(diff == want, "A", f"{tag}: L3 command vs {b}'s: {diff}, want {want}")
            ok(
                fk["--init"]
                == f"<R>/grpo_v2{'/smoke' if mode == 'smoke' else ''}/{b}_s{s}_L2/ckpts/{b}_s{s}_L2_r1.pt",
                "A",
                f"{tag}: init {fk['--init']}",
            )

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
        # ARM_pb (2026-10-13): EI; another suffix (_rfb after it keeps _rfb's message)
        for arm in ["ei_think_pb", "ei_nothink_v_pb"]:
            refusals.append(
                (
                    f"{H}/res/{mode}/s{s}/{arm}",
                    1,
                    rf"^00:00Z {arm}: _pb is a GRPO L3 pilot, and {arm[:-3]} is EI: refusing$",
                    arm,
                )
            )
        for arm in [
            "grpo_think_h_pb",
            "grpo_think_pb_h",
            "grpo_think_b_pb",
            "grpo_think_pb_rfb",
            "grpo_think_vs_pb_3pb",
            "grpo_think_c_pb",
            "grpo_think_wb_pb",
        ]:
            refusals.append(
                (
                    f"{H}/res/{mode}/s{s}/{arm}",
                    1,
                    rf"^00:00Z {arm}: _pb \(an L3 pilot\) does not combine with other suffixes: refusing$",
                    arm,
                )
            )
        refusals.append(
            (
                f"{H}/res/{mode}/s{s}/grpo_think_rfb_pb",
                1,
                r"^00:00Z grpo_think_rfb_pb: _rfb, _ulb and _3pb \(L3 pilots\) do not combine with other suffixes: refusing$",
                "grpo_think_rfb_pb",
            )
        )
        # ARM_fbb (2026-10-14): EI; no-think and plain-sampling arms; another suffix, in either order (a pilot
        # suffix before it keeps that pilot's message)
        fbb_no = r"_fbb \(an L3 pilot\) does not combine with other suffixes"
        for arm, rx in (
            ("ei_think_v_fbb", r"_fbb is a GRPO L3 pilot, and ei_think_v is EI"),
            ("ei_think_fbb", r"_fbb is a GRPO L3 pilot, and ei_think is EI"),
            *(
                (
                    a,
                    rf"_fbb writes the verifier's feedback into the think block, and {a[:-4]} has none",
                )
                for a in ("grpo_nothink_vs_fbb", "grpo_nothink_v_fbb", "grpo_nothink_fbb")
            ),
            (
                "grpo_think_fbb",
                r"_fbb needs the verifier \(the logical arm\), and grpo_think samples plain",
            ),
            *(
                (a, fbb_no)
                for a in (
                    "grpo_think_h_fbb",
                    "grpo_think_vs_h_fbb",
                    "grpo_think_vs_fbb_h",
                    "grpo_think_vs_b_fbb",
                    "grpo_think_vs_fbb_b",
                    "grpo_think_vs_c_fbb",
                    "grpo_think_vs_fbb_c",
                    "grpo_think_vs_wb_fbb",
                    "grpo_think_vs_fbb_w",
                    "grpo_think_vs_fbb_rfb",
                    "grpo_think_vs_fbb_pb",
                    "grpo_think_vs_fbb_3pb",
                    "grpo_think_vs_fbb_ulb",
                    "grpo_think_vs_fbb_mb",
                    "grpo_think_vs_fbb_fbb",
                )
            ),
            *(
                (a, r"_rfb, _ulb and _3pb \(L3 pilots\) do not combine with other suffixes")
                for a in ("grpo_think_vs_rfb_fbb", "grpo_think_vs_3pb_fbb", "grpo_think_vs_ulb_fbb")
            ),
            ("grpo_think_vs_pb_fbb", r"_pb \(an L3 pilot\) does not combine with other suffixes"),
            ("grpo_think_vs_mb_fbb", r"_mb \(an L3 pilot\) does not combine with other suffixes"),
        ):
            refusals.append(
                (f"{H}/res/{mode}/s{s}/{arm}", 1, rf"^00:00Z {arm}: {rx}: refusing$", arm)
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
            (
                "nockpt",
                "grpo_think_vs_pb",
                rf"no L2 checkpoint of grpo_think_vs \({re.escape(SB)}/{c}/grpo_think_vs_s{s}_L2/ckpts/grpo_think_vs_s{s}_L2_it{it}\.pt\)",
            ),
            (
                "rb",
                "grpo_nothink_vs_pb",
                r"grpo_nothink_vs's L2 log ends 'guard it 39: .*lr x0\.5 \(1 rollbacks\).*', not lr x1 \(0 rollbacks\): rerun the whole line",
            ),
            (
                "nolog",
                "grpo_think_pb",
                r"grpo_think's L2 log ends 'no guard line', not lr x1 \(0 rollbacks\): rerun the whole line",
            ),
            # ARM_mb (2026-10-14): as ARM_3pb's
            (
                "nockpt",
                "grpo_think_vs_mb",
                rf"no L2 checkpoint of grpo_think_vs \({re.escape(SB)}/{c}/grpo_think_vs_s{s}_L2/ckpts/grpo_think_vs_s{s}_L2_it{it}\.pt\)",
            ),
            (
                "rb",
                "grpo_think_vs_mb",
                r"grpo_think_vs's L2 log ends 'guard it 39: .*lr x0\.5 \(1 rollbacks\).*', not lr x1 \(0 rollbacks\): rerun the whole line",
            ),
            (
                "nolog",
                "grpo_nothink_vs_mb",
                r"grpo_nothink_vs's L2 log ends 'no guard line', not lr x1 \(0 rollbacks\): rerun the whole line",
            ),
            (
                "no3pL3",
                "grpo_think_vs_mb",
                rf"no 3-premise pool file {re.escape(SB)}/pools/curriculum_3p/L3_3p\.jsonl \(runs/pools/curriculum_3p\)",
            ),
            (  # ARM_fbb's three (2026-10-14)
                "nockpt",
                "grpo_think_vs_fbb",
                rf"no L2 checkpoint of grpo_think_vs \({re.escape(SB)}/{c}/grpo_think_vs_s{s}_L2/ckpts/grpo_think_vs_s{s}_L2_it{it}\.pt\)",
            ),
            (
                "rb",
                "grpo_think_v_fbb",
                r"grpo_think_v's L2 log ends 'guard it 39: .*lr x0\.5 \(1 rollbacks\).*', not lr x1 \(0 rollbacks\): rerun the whole line",
            ),
            (
                "nolog",
                "grpo_think_vs3_fbb",
                r"grpo_think_vs3's L2 log ends 'no guard line', not lr x1 \(0 rollbacks\): rerun the whole line",
            ),
        ):
            refusals.append((f"{d}/{kind}_{arm}", 1, rf"^00:00Z {arm}: {rx}: refusing$", arm))
        # ARM_mb: EI; another suffix (in either order; _ulb or _pb before it keeps theirs); a _vs3 arm
        for arm, rx in (
            ("ei_think_mb", r"_mb is a GRPO L3 pilot, and ei_think is EI"),
            ("ei_nothink_v_mb", r"_mb is a GRPO L3 pilot, and ei_nothink_v is EI"),
            *(
                (a, r"_mb \(an L3 pilot\) does not combine with other suffixes")
                for a in (
                    "grpo_think_h_mb",
                    "grpo_think_mb_h",
                    "grpo_think_wb_mb",
                    "grpo_think_mb_ulb",
                    "grpo_think_vs_mb_pb",
                )
            ),
            (
                "grpo_think_ulb_mb",
                r"_rfb, _ulb and _3pb \(L3 pilots\) do not combine with other suffixes",
            ),
            ("grpo_think_pb_mb", r"_pb \(an L3 pilot\) does not combine with other suffixes"),
            ("grpo_think_vs3_mb", r"grpo_think_vs3 already trains on the 3-premise rows"),
            ("grpo_nothink_vs3_mb", r"grpo_nothink_vs3 already trains on the 3-premise rows"),
        ):
            refusals.append(
                (f"{H}/res/{mode}/s{s}/{arm}", 1, rf"^00:00Z {arm}: {rx}: refusing$", arm)
            )
        # ARM_mb_kKNOB (2026-10-15): an unknown knob or anything after it; another suffix before _mb (that suffix's
        # message); EI; _vs3; _kp3 without the step penalty; a knob without _mb (not a knob arm)
        k_no = r"_mb_k takes one knob of x66 g64 t12 kl0 kl1 ch4 p3 i78 rr, with nothing after it"
        for arm, rx in (
            *(
                (a, k_no)
                for a in (
                    "grpo_think_vs_mb_kfoo",
                    "grpo_think_vs_mb_k",
                    "grpo_think_vs_mb_kx66_h",
                    "grpo_think_vs_mb_kx66_krr",
                    "grpo_think_vs_mb_kmb",
                    "grpo_think_vs_mb_krr_mb",
                    "grpo_think_vs_mb_kt12_b",
                )
            ),
            *(
                (a, r"_mb \(an L3 pilot\) does not combine with other suffixes")
                for a in (
                    "grpo_think_vs_h_mb_kx66",
                    "grpo_think_vs_wb_mb_ki78",
                    "grpo_think_vs_mb_mb_krr",
                )
            ),
            (
                "grpo_think_vs_ulb_mb_krr",
                r"_rfb, _ulb and _3pb \(L3 pilots\) do not combine with other suffixes",
            ),
            (
                "grpo_think_vs_pb_mb_kt12",
                r"_pb \(an L3 pilot\) does not combine with other suffixes",
            ),
            (
                "grpo_think_vs_fbb_mb_kkl0",
                r"_fbb \(an L3 pilot\) does not combine with other suffixes",
            ),
            ("ei_think_mb_krr", r"_mb is a GRPO L3 pilot, and ei_think is EI"),
            ("ei_nothink_v_mb_kg64", r"_mb is a GRPO L3 pilot, and ei_nothink_v is EI"),
            ("grpo_think_vs3_mb_kx66", r"grpo_think_vs3 already trains on the 3-premise rows"),
            *(
                (
                    a,
                    rf"_kp3 raises the step penalty, and {a[:-7]} samples plain \(no penalty\)",
                )
                for a in ("grpo_think_mb_kp3", "grpo_nothink_mb_kp3")
            ),
            (
                "grpo_think_vs_fbb_kx66",
                r"_fbb \(an L3 pilot\) does not combine with other suffixes",
            ),
            (
                "grpo_think_vs_ulb_kp3",
                r"_rfb, _ulb and _3pb \(L3 pilots\) do not combine with other suffixes",
            ),
        ):
            refusals.append(
                (f"{H}/res/{mode}/s{s}/{arm}", 1, rf"^00:00Z {arm}: {rx}: refusing$", arm)
            )
        refusals.append(
            (
                f"{H}/res/{mode}/s{s}/grpo_think_vs_kx66",
                2,
                r"^unknown arm grpo_think_vs_kx66$",
                "grpo_think_vs_kx66",
            )
        )
        for kind, arm, rx in (  # as ARM_mb's
            (
                "nockpt",
                "grpo_think_vs_mb_ki78",
                rf"no L2 checkpoint of grpo_think_vs \({re.escape(SB)}/{c}/grpo_think_vs_s{s}_L2/ckpts/grpo_think_vs_s{s}_L2_it{it}\.pt\)",
            ),
            (
                "rb",
                "grpo_think_vs_mb_kt12",
                r"grpo_think_vs's L2 log ends 'guard it 39: .*lr x0\.5 \(1 rollbacks\).*', not lr x1 \(0 rollbacks\): rerun the whole line",
            ),
            (
                "nolog",
                "grpo_think_vs_mb_kg64",
                r"grpo_think_vs's L2 log ends 'no guard line', not lr x1 \(0 rollbacks\): rerun the whole line",
            ),
            (
                "no3pL3",
                "grpo_think_vs_mb_kx66",
                rf"no 3-premise pool file {re.escape(SB)}/pools/curriculum_3p/L3_3p\.jsonl \(runs/pools/curriculum_3p\)",
            ),
        ):
            refusals.append((f"{d}/{kind}_{arm}", 1, rf"^00:00Z {arm}: {rx}: refusing$", arm))
        # EI ARM_kKNOB (2026-10-16): an unknown or a GRPO knob, or anything after the knob; a suffix before it; _mb_k on an
        # EI arm (_mb_k's message); an EI knob on a GRPO arm; no L2 checkpoint; ARM_k3p's mix and pass record
        ek_no = r"_k on an EI arm takes one EI knob of rr 3p ret0 t10 r2, with nothing after it"
        for arm, rx in (
            *(
                (a, ek_no)
                for a in (
                    "ei_think_v_kfoo",
                    "ei_think_v_k",
                    "ei_think_v_kx66",
                    "ei_think_v_kp3",
                    "ei_think_v_k3p_h",
                    "ei_think_v_kret0_krr",
                    "ei_think_v_kt10_b",
                    "ei_think_v_kr2_k3p",
                )
            ),
            *(
                (
                    a,
                    rf"_k{a.rsplit('_k', 1)[1]} \(an EI knob pilot\) does not combine with other suffixes",
                )
                for a in (
                    "ei_think_v_h_krr",
                    "ei_think_v_b_k3p",
                    "ei_think_v_c_kr2",
                    "ei_think_v_wb_kt10",
                    "ei_think_v_rfb_krr",
                    "ei_think_v_fbb_kret0",
                )
            ),
            *(
                (a, k_no)
                for a in ("ei_think_v_mb_k3p", "ei_nothink_v_mb_kret0", "grpo_think_vs_mb_kret0")
            ),
            *(
                (
                    a,
                    rf"_k{a.rsplit('_k', 1)[1]} is an EI knob \(ARM_kKNOB, for ei_\* arms\); GRPO's knob pilots are ARM_mb_kKNOB",
                )
                for a in (
                    "grpo_think_v_k3p",
                    "grpo_think_vs_kret0",
                    "grpo_nothink_v_kt10",
                    "grpo_think_kr2",
                    "grpo_think_vs_krr",
                    "grpo_nothink_vs3_k3p",
                )
            ),
        ):
            refusals.append(
                (f"{H}/res/{mode}/s{s}/{arm}", 1, rf"^00:00Z {arm}: {rx}: refusing$", arm)
            )
        m3 = re.escape(f"{SB}/pools/curriculum_3p/L3_mix_ei.jsonl")
        nopass = rf"no preflight pass record of {m3} as it is \({re.escape(SB)}/pools/curriculum_3p/L3_mix_ei\.preflight_pass\.txt; grpo_v2/build_mix_ei\.py\)"
        for kind, rx in (
            (
                "nockpt",
                rf"no L2 checkpoint of ei_think_v \({re.escape(SB)}/{c}/ei_think_v_s{s}_L2/ckpts/ei_think_v_s{s}_L2_r1\.pt\)",
            ),
            ("nomix", rf"no 3-premise mix {m3} \(grpo_v2/build_mix_ei\.py builds it\)"),
            ("nopass", nopass),
            ("stale", nopass),
        ):
            refusals.append(
                (
                    f"{d}/{kind}_ei_think_v_k3p",
                    1,
                    rf"^00:00Z ei_think_v_k3p: {rx}: refusing$",
                    "ei_think_v_k3p",
                )
            )
        # the mini runs (2026-10-16): another base arm or a suffix before _mini; another knob or anything after it;
        # KNOB_FULL=1; a missing subset, no pass record, a stale one. ARM_ckreads: other arms, i78, KNOB_FULL=1, no
        # finished L3, no L2 checkpoint
        mb_no = r"_mini runs the primary recipe, grpo_think_vs's"
        mk_no = r"_mini takes one knob _kKNOB of rr t07 t085 t115 t13 lpNN lpmNN kl1 g64, with nothing after it"
        ck_no = r"_ckreads reads a GRPO knob pilot's L3 checkpoints \(ARM_mb_kKNOB_ckreads\)"
        for arm, rx in (
            *(
                (a, mb_no)
                for a in (
                    "grpo_think_mini",
                    "grpo_nothink_vs_mini",
                    "grpo_think_vs_h_mini",
                    "grpo_think_vs_mb_mini",
                    "grpo_think_vs_mb_kkl0_mini",
                    "ei_think_v_mini",
                )
            ),
            *(
                (a, mk_no)
                for a in (
                    "grpo_think_vs_mini_kfoo",
                    "grpo_think_vs_mini_k",
                    "grpo_think_vs_mini_kt07_krr",
                    "grpo_think_vs_mini_kt07_h",
                    "grpo_think_vs_mini_h",
                    "grpo_think_vs_mini_kx66",
                    "grpo_think_vs_mini_kkl0",
                    "grpo_think_vs_mini_mini",
                    "grpo_think_vs_mini_kg64_mini",
                    "grpo_think_vs_mini_ku1",
                    "grpo_think_vs_mini_kg8",
                    "grpo_think_vs_mini_klp5",
                    "grpo_think_vs_mini_klp100",
                    "grpo_think_vs_mini_klpx1",
                    "grpo_think_vs_mini_klpm",
                )
            ),
            *(
                (f"grpo_think_vs_mini_k{k}", rf"_k{k} is no length penalty \(grpo_think_vs_mini\)")
                for k in ("lp00", "lpm00")
            ),
        ):
            refusals.append(
                (f"{H}/res/{mode}/s{s}/{arm}", 1, rf"^00:00Z {arm}: {rx}: refusing$", arm)
            )
        for arm, who, rx in (
            *(
                (a, a, ck_no)
                for a in (
                    "grpo_think_vs_ckreads",
                    "grpo_think_vs_mb_ckreads",
                    "ei_think_v_krr_ckreads",
                    "grpo_think_vs_mini_ckreads",
                    "grpo_think_vs_mini_kt07_ckreads",
                )
            ),
            (
                "grpo_think_vs_mb_ki78_ckreads",
                "grpo_think_vs_mb_ki78_ckreads",
                r"_ckreads reads 39-iteration L3s \(CKR_ITS\), and _ki78's has 78",
            ),
            (
                "grpo_think_vs_mb_kfoo_ckreads",
                "grpo_think_vs_mb_kfoo",
                k_no,
            ),  # parsed as the arm it reads
        ):
            refusals.append(
                (f"{H}/res/{mode}/s{s}/{arm}", 1, rf"^00:00Z {who}: {rx}: refusing$", arm)
            )
        mp = re.escape(f"{SB}/pools/mini")
        for kind, arm, who, rx in (
            (
                "nosub",
                "grpo_think_vs_mini_kt07",
                None,
                rf"no mini subset {mp}/L3_3p_mini\.jsonl \(grpo_v2/build_mini\.py builds it\)",
            ),
            (
                "nopass",
                "grpo_think_vs_mini_kt07",
                None,
                rf"no preflight pass record of {mp}/L2_mini\.jsonl as it is \({mp}/preflight_pass\.txt; grpo_v2/build_mini\.py\)",
            ),
            (
                "stale",
                "grpo_think_vs_mini_kt07",
                None,
                rf"no preflight pass record of {mp}/L3_mini\.jsonl as it is \({mp}/preflight_pass\.txt; grpo_v2/build_mini\.py\)",
            ),
            (
                "full",
                "grpo_think_vs_mini",
                None,
                r"KNOB_FULL=1 is the knob pilots' \(a mini run has one set of reads\)",
            ),
            (
                "nol3",
                "grpo_think_vs_mb_kkl0_ckreads",
                None,
                rf"no finished L3 of grpo_think_vs_mb_kkl0 \({re.escape(SB)}/{c}/grpo_think_vs_mb_kkl0_s{s}_L3/ckpts/grpo_think_vs_mb_kkl0_s{s}_L3_it{it}\.pt\) to read",
            ),
            (
                "nockpt",
                "grpo_think_vs_mb_kkl0_ckreads",
                "grpo_think_vs_mb_kkl0",
                rf"no L2 checkpoint of grpo_think_vs \({re.escape(SB)}/{c}/grpo_think_vs_s{s}_L2/ckpts/grpo_think_vs_s{s}_L2_it{it}\.pt\)",
            ),
            (
                "full",
                "grpo_think_vs_mb_kkl0_ckreads",
                None,
                r"KNOB_FULL=1 is the knob pilots' \(_ckreads has its own reads\)",
            ),
        ):
            refusals.append(
                (f"{d}/{kind}_{arm}", 1, rf"^00:00Z {who or arm}: {rx}: refusing$", arm)
            )
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

# ---------------------------------------------------------------- D: ARM_mb's averaging (2026-10-14)
# a rolling checkpoint missing (an alarm iteration saves none): the stage and the usual reads as always, then the refusal
# of the averaged reads; a rerun with the averaged checkpoint there: no averaging, the same reads
for s in (0, 1, 2):
    r = load(f"{H}/res/mb/normal/s{s}/noroll30_grpo_think_vs_mb")
    exp, eout = expect("grpo_think_vs_mb", s, "normal", 3)
    n_pre = next(i for i, x in enumerate(exp) if x[0] == "py" and x[5].startswith("- "))
    f = f"{SB}/grpo_v2/grpo_think_vs_mb_s{s}_L3/ckpts/grpo_think_vs_mb_s{s}_L3_it30_rolling.pt"
    want_out = [
        x for x in eout if not x.startswith(("00:00Z averaged", "00:00Z grpo_v2 line done"))
    ]
    want_out = [x for x in want_out if "_avg_" not in x]
    want_out.append(
        f"00:00Z grpo_think_vs_mb: no {f} to average (an alarm iteration?): no averaged reads"
    )
    got = [key(x) for x in r["rec"]]
    ok(r["exit"] == "1", "D", f"noroll30 s{s}: exit {r['exit']}")
    ok(
        sorted(got) == sorted(exp[:n_pre]),
        "D",
        f"noroll30 s{s}: calls differ from the usual line's",
    )
    ok(stdout_lines(r) == want_out, "D", f"noroll30 s{s}: stdout {stdout_lines(r)[-1:]}")
for mode in ("normal", "smoke"):
    for s in (0, 1, 2):
        r = load(f"{H}/res/mb/{mode}/s{s}/rerun_grpo_think_vs_mb")
        exp, eout = expect("grpo_think_vs_mb", s, mode, 3, avg_there=True)
        got = [key(x) for x in r["rec"]]
        ok(r["exit"] == "0", "D", f"rerun {mode} s{s}: exit {r['exit']}")
        ok(
            sorted(got) == sorted(exp),
            "D",
            f"rerun {mode} s{s}: calls {sorted(set(got) ^ set(exp))[:2]}",
        )
        ok(stdout_lines(r) == eout, "D", f"rerun {mode} s{s}: stdout")
        ok(
            not any(x[0] == "py" and x[5].startswith("- ") for x in got),
            "D",
            f"rerun {mode} s{s}: averaged again",
        )

# ---------------------------------------------------------------- K: ARM_mb_ki78's 78-iteration L3 (2026-10-15)
# an it39 L3 checkpoint there (39 iterations' worth) does not end the stage: the same line as a fresh one; with its it78
# checkpoint there, no stage runs and the reads take it78
for mode in ("normal", "smoke"):
    for s in (0, 1, 2):
        exp, eout = expect("grpo_think_vs_mb_ki78", s, mode, 3)
        r = load(f"{H}/res/knob/{mode}/s{s}/it39_grpo_think_vs_mb_ki78")
        got = [key(x) for x in r["rec"]]
        ok(r["exit"] == "0", "K", f"it39 {mode} s{s}: exit {r['exit']}")
        ok(sorted(got) == sorted(exp), "K", f"it39 {mode} s{s}: calls differ from a fresh line's")
        ok(stdout_lines(r) == eout, "K", f"it39 {mode} s{s}: stdout")
        r = load(f"{H}/res/knob/{mode}/s{s}/it78_grpo_think_vs_mb_ki78")
        got = [key(x) for x in r["rec"]]
        it78 = 4 if mode == "smoke" else 78
        name = f"grpo_think_vs_mb_ki78_s{s}_L3"
        stage = [x for x in exp if " -m code.thinking.grpo " in x[5]]
        launched = [x for x in eout if x.startswith(("00:00Z launched GRPO", "00:00Z GRPO "))]
        # the stage's calls: waitfree's 2 nvidia-smi (normal), the timeout, sleep 5 and the watchdog launch
        n_stage = 3 + (2 if mode == "normal" else 0) - (1 if mode == "smoke" else 0)
        ok(r["exit"] == "0", "K", f"it78 {mode} s{s}: exit {r['exit']}")
        ok(
            not any(" -m code.thinking.grpo " in x[5] for x in got) and len(stage) == 1,
            "K",
            f"it78 {mode} s{s}: the stage ran again",
        )
        ok(
            len(got) == len(exp) - n_stage,
            "K",
            f"it78 {mode} s{s}: {len(got)} calls, want {len(exp) - n_stage}",
        )
        ok(
            stdout_lines(r) == [x for x in eout if x not in launched],
            "K",
            f"it78 {mode} s{s}: stdout {stdout_lines(r)[:2]}",
        )
        ev = [x[5] for x in got if " -m code.thinking.eval " in x[5]]
        ok(
            len(ev) == 3
            and all(
                f"--ckpt <R>/grpo_v2{'/smoke' if mode == 'smoke' else ''}/{name}/ckpts/{name}_it{it78}.pt "
                in x
                for x in ev
            ),
            "K",
            f"it78 {mode} s{s}: reads {[x[:120] for x in ev]}",
        )


# ---------------------------------------------------------------- J: EI ARM_kKNOB reruns, ARM_kr2, KNOB_FULL (2026-10-16)
def kreads(arm: str, s: int, mode: str) -> frozenset[str]:
    """A knob arm's three plain reads (eval dir basenames)."""
    return frozenset(f"{n}_{arm}_s{s}_plain_k{2 if mode == 'smoke' else k}" for n, k in KNOB_READS)


for mode in ("normal", "smoke"):
    for s in (0, 1, 2):
        c_ = "grpo_v2/smoke" if mode == "smoke" else "grpo_v2"
        jobs: list[tuple[str, str, dict]] = [  # (result dir, arm, expect()'s options)
            ("r1_ei_think_v_kr2", "ei_think_v_kr2", {}),  # with _r1.pt: as a fresh line
            ("r2_ei_think_v_kr2", "ei_think_v_kr2", {"l3_there": True}),
            (
                "part_ei_think_v_kt10",
                "ei_think_v_kt10",
                {
                    "l3_there": True,
                    "reads_there": frozenset(
                        [f"t2130_ei_think_v_kt10_s{s}_plain_k{2 if mode == 'smoke' else 8}"]
                    ),
                },
            ),
            ("fresh_ei_think_v_k3p", "ei_think_v_k3p", {"full": True}),
            (
                "full_grpo_think_vs_mb_kkl0",
                "grpo_think_vs_mb_kkl0",
                {
                    "full": True,
                    "l3_there": True,
                    "reads_there": kreads("grpo_think_vs_mb_kkl0", s, mode),
                },
            ),
        ]
        for kn in EKNOBS:
            a = f"ei_think_v_k{kn}"
            jobs.append((f"done_{a}", a, {"l3_there": True, "reads_there": kreads(a, s, mode)}))
            jobs.append(
                (
                    f"full_{a}",
                    a,
                    {"full": True, "l3_there": True, "reads_there": kreads(a, s, mode)},
                )
            )
        for d, arm, kw in jobs:
            r = load(f"{H}/res/eiknob/{mode}/s{s}/{d}")
            exp, eout = expect(arm, s, mode, 3, **kw)
            got = [key(x) for x in r["rec"]]
            tag = f"{d} {mode} s{s}"
            ok(r["exit"] == "0", "J", f"{tag}: exit {r['exit']}")
            ok(
                sorted(got) == sorted(exp),
                "J",
                f"{tag}: calls: missing {sorted(set(exp) - set(got))[:2]} extra {sorted(set(got) - set(exp))[:2]}",
            )
            ok(
                [x for x in got if x[0] not in ("python3", "sleep")]
                == [x for x in exp if x[0] not in ("python3", "sleep")],
                "J",
                f"{tag}: call order",
            )
            ok(stdout_lines(r) == eout, "J", f"{tag}: stdout {stdout_lines(r)[:3]} / {eout[:3]}")
            stage_ran = any(f" --name {arm}_s{s}_L3 " in x[5] for x in got)
            ok(stage_ran == (not kw.get("l3_there")), "J", f"{tag}: the L3 stage ran: {stage_ran}")
            ok(
                not any(f"{arm}_s{s}_L2" in t for t in r["tree"]),
                "J",
                f"{tag}: an L2 of its own",
            )
            evs = [x[5] for x in got if " -m code.thinking.eval " in x[5]]
            if arm.endswith("_kr2"):  # ARM_kr2's stage ends at, and every read takes, _r2.pt
                ok(
                    all(
                        f"--ckpt <R>/{c_}/{arm}_s{s}_L3/ckpts/{arm}_s{s}_L3_r2.pt " in x
                        for x in evs
                    ),
                    "J",
                    f"{tag}: reads {[x[:100] for x in evs]}",
                )
            n_reads = 3 - len(kw.get("reads_there", ())) + (5 if kw.get("full") else 0)
            ok(len(evs) == n_reads, "J", f"{tag}: {len(evs)} reads, want {n_reads}")
            # the setup's base_commit.txt (all but fresh_*'s) holds 'dry training commit': KNOB_FULL=1 writes HEAD to
            # reads_commit.txt and leaves it alone; a rerun without KNOB_FULL writes HEAD over it (as for every arm)
            md5 = dict(
                ln.split("  ", 1)[::-1]
                for ln in Path(f"{H}/res/eiknob/{mode}/s{s}/{d}/md5.txt").read_text().split("\n")
                if ln
            )
            bc = f"./runs/{c_}/{arm}_s{s}.base_commit.txt"
            rc = f"./runs/{c_}/{arm}_s{s}.reads_commit.txt"
            train_rec = hashlib.md5(b"dry training commit\n").hexdigest()
            fresh = f"f {bc}" not in r["pre"]
            if kw.get("full"):
                ok(
                    rc in md5 and (bc not in md5 if fresh else md5[bc] == train_rec),
                    "J",
                    f"{tag}: commit records {bc in md5} {rc in md5}",
                )
            else:
                ok(
                    rc not in md5 and bc in md5 and md5[bc] != train_rec,
                    "J",
                    f"{tag}: commit records {bc in md5} {rc in md5}",
                )


# ---------------------------------------------------------------- M: the mini runs (2026-10-16)
def mini_flags(r: dict, name: str) -> dict[str, str | None]:
    """The flags of run r's stage `name` (its one grpo command)."""
    cmds = [
        x["argv"]
        for x in r["rec"]
        if x["kind"] == "timeout"
        and " -m code.thinking.grpo " in x["argv"]
        and f" --name {name} " in x["argv"]
    ]
    assert len(cmds) == 1, (name, len(cmds))
    return flags(cmds[0])


def fdiff(a: dict, b: dict) -> dict[str, tuple]:
    d = {f: (a.get(f), b.get(f)) for f in set(a) | set(b) if a.get(f) != b.get(f)}
    return d | {f: (None, None) for f in set(a) ^ set(b) if f not in d}  # a switch added or dropped


def check_mini_run(r: dict, arm: str, s: int, mode: str, sect: str, **kw: object) -> None:
    """A mini run's result dir against `expect()`: exit, the calls (multiset and order), stdout, the environment, no flag
    twice, its eval dirs; the chain (L2-mini from S1g3_s0, L3-mini from L2-mini's last checkpoint, each KL anchor at its
    init)."""
    smoke = mode == "smoke"
    c_ = "<R>/grpo_v2" + ("/smoke" if smoke else "")
    it = 2 if smoke else MINI_IT
    tag = f"{mode} s{s} {arm}" + (f" {kw}" if kw else "")
    exp, eout = expect(arm, s, mode, 3, **kw)  # type: ignore[arg-type]
    got = [key(x) for x in r["rec"]]
    ok(r["exit"] == "0", sect, f"{tag}: exit {r['exit']}")
    ok(
        sorted(got) == sorted(exp),
        sect,
        f"{tag}: calls: missing {sorted(set(exp) - set(got))[:2]} extra {sorted(set(got) - set(exp))[:2]}",
    )
    ok(
        [x for x in got if x[0] not in ("python3", "sleep")]
        == [x for x in exp if x[0] not in ("python3", "sleep")],
        sect,
        f"{tag}: call order",
    )
    ok(
        stdout_lines(r) == eout,
        sect,
        f"{tag}: stdout {[x for x in stdout_lines(r) if x not in eout][:2]} / {[x for x in eout if x not in stdout_lines(r)][:2]}",
    )
    for x in r["rec"]:
        if x["kind"] in ("py", "timeout", "python3"):
            e = {k: norm(v) for k, v in x["env"].items() if k in ENV_WANT}
            ok(
                e == ENV_WANT and ("SMOKE" in x["env"]) == smoke,
                sect,
                f"{tag}: {x['kind']} env {e}",
            )
        if x["kind"] == "timeout" and " -m code.thinking." in x["argv"]:
            fl = [t for t in x["argv"].split() if t.startswith("--")]
            ok(len(fl) == len(set(fl)), sect, f"{tag}: a flag given twice in {x['argv'][:80]}")
    ok(
        not any("tb72" in t or "logical" in t or "_avg" in t for t in r["tree"]),
        sect,
        f"{tag}: a tb72 / guided / averaged read",
    )
    names = [x[5].split(" --name ")[1].split()[0] for x in got if " -m code.thinking.grpo " in x[5]]
    if f"{arm}_s{s}_L2" in names:
        f2 = mini_flags(r, f"{arm}_s{s}_L2")
        ok(
            f2["--init"] == "<R>/stage1/S1g3_s0/model.pt" == f2["--kl_anchor_ref"],
            sect,
            f"{tag}: L2 init {f2['--init']}",
        )
    if f"{arm}_s{s}_L3" in names:
        f3 = mini_flags(r, f"{arm}_s{s}_L3")
        l2last = f"{c_}/{arm}_s{s}_L2/ckpts/{arm}_s{s}_L2_it{it}.pt"
        ok(f3["--init"] == l2last == f3["--kl_anchor_ref"], sect, f"{tag}: L3 init {f3['--init']}")


for mode in ("normal", "smoke"):
    smoke = mode == "smoke"
    c_ = "<R>/grpo_v2" + ("/smoke" if smoke else "")
    it = 2 if smoke else MINI_IT
    for s in (0, 1, 2, 3):
        for arm in (
            MARMS if s < 3 else MARMS[:1]
        ):  # seed 3: the reference alone (its init is S1g3_s0's)
            r = runs[(mode, s, arm)] = load(f"{H}/res/{mode}/s{s}/{arm}")
            check_mini_run(r, arm, s, mode, "M")
            ev = sorted(
                re.sub(r"^d \./runs/eval/v2(_smoke)?/", "", t)
                for t in r["tree"]
                if re.match(r"^d \./runs/eval/v2(_smoke)?/[^/]+$", t)
            )
            want_ev = sorted(f"{n}_{arm}_s{s}_plain_k{2 if smoke else k}" for n, k in MINI_READS)
            ok(ev == want_ev, "M", f"{mode} s{s} {arm}: eval dirs {ev}")
        if s == 3:
            continue
        # L2-mini vs grpo_think_vs's L2 and L3-mini vs grpo_think_vs_mb_kkl0's L3: the iterations, the subsets, the
        # rolling checkpoints, the timeout and the inits only (SMOKE=1: 2-iteration stages on 8-row heads either way)
        ref = "grpo_think_vs_mini"
        r = runs[(mode, s, ref)]
        for st, full, fn in (
            ("L2", "grpo_think_vs", f"grpo_think_vs_s{s}_L2"),
            ("L3", "grpo_think_vs_mb_kkl0", f"grpo_think_vs_mb_kkl0_s{s}_L3"),
        ):
            fm = mini_flags(r, f"{ref}_s{s}_{st}")
            ff = mini_flags(runs[(mode, s, full)], fn)
            for k_ in ("--name", "--out"):
                ff[k_] = (ff[k_] or "").replace(fn, f"{ref}_s{s}_{st}")
            mwant: dict[str, tuple] = {"timeout": ("21600", "7200"), "--save_rolling": (None, "2")}
            if not smoke:
                mwant |= {"--iters": ("39", "10"), "--save_every": ("39", "10")}
                mwant["--targets"] = (
                    f"<R>/pools/curriculum/{st}/pool.jsonl",
                    f"{MINI}/{st}_mini.jsonl",
                )
                if st == "L3":
                    mwant["--targets_extra"] = (
                        "<R>/pools/curriculum_3p/L3_3p.jsonl",
                        f"{MINI}/L3_3p_mini.jsonl",
                    )
            if (
                st == "L2" and s != 0
            ):  # the full recipe's seed s from S1g3_s{s}, the mini run's from S1g3_s0
                mwant["--init"] = mwant["--kl_anchor_ref"] = (
                    f"<R>/stage1/S1g3_s{s}/model.pt",
                    "<R>/stage1/S1g3_s0/model.pt",
                )
            if st == "L3":  # from L2-mini's last checkpoint, not the base arm's 39-iteration L2
                l2full = f"{c_}/grpo_think_vs_s{s}_L2/ckpts/grpo_think_vs_s{s}_L2_it{2 if smoke else 39}.pt"
                l2mini = f"{c_}/{ref}_s{s}_L2/ckpts/{ref}_s{s}_L2_it{it}.pt"
                mwant["--init"] = mwant["--kl_anchor_ref"] = (l2full, l2mini)
            ok(
                fdiff(ff, fm) == mwant,
                "M",
                f"{mode} s{s} {ref} {st}: vs {full}'s: {fdiff(ff, fm)}, want {mwant}",
            )
        # each knob's two commands: the reference's but for its flags (kl1: L3-mini's only)
        for arm in MARMS[1:]:
            mk = arm.rsplit("_k", 1)[1]
            both, l3only, _ = mknob(mk, smoke)
            for st in ("L2", "L3"):
                fk = mini_flags(runs[(mode, s, arm)], f"{arm}_s{s}_{st}")
                fr = mini_flags(r, f"{ref}_s{s}_{st}")
                fr = {
                    k_: (v.replace(f"{ref}_s{s}_", f"{arm}_s{s}_") if v else v)
                    for k_, v in fr.items()
                }
                mwant = {**both, **(l3only if st == "L3" else {})}
                ok(
                    fdiff(fr, fk) == mwant,
                    "M",
                    f"{mode} s{s} {arm} {st}: vs {ref}'s: {fdiff(fr, fk)}, want {mwant}",
                )

# ---------------------------------------------------------------- N: mini reruns and MINI_ROLL
for mode in ("normal", "smoke"):
    smoke = mode == "smoke"
    for s in (0, 1, 2):
        k2 = 2 if smoke else None
        for d, arm, kw in (
            ("l2done_grpo_think_vs_mini_klp20", "grpo_think_vs_mini_klp20", {"l2_there": True}),
            (
                "done_grpo_think_vs_mini_kg64",
                "grpo_think_vs_mini_kg64",
                {
                    "l2_there": True,
                    "l3_there": True,
                    "reads_there": frozenset(
                        f"{n}_grpo_think_vs_mini_kg64_s{s}_plain_k{k2 or k}"
                        for n, k in (("l2test", 8), ("charles173", 64))
                    ),
                },
            ),
        ):
            r = load(f"{H}/res/mini/{mode}/s{s}/{d}")
            check_mini_run(r, arm, s, mode, "N", **kw)
            got = [key(x) for x in r["rec"]]
            ran = [
                x[5].split(" --name ")[1].split()[0]
                for x in got
                if " -m code.thinking.grpo " in x[5]
            ]
            ok(
                ran == ([] if kw.get("l3_there") else [f"{arm}_s{s}_L3"]),
                "N",
                f"{d} {mode} s{s}: stages {ran}",
            )
            n_ev = sum(" -m code.thinking.eval " in x[5] for x in got)
            ok(n_ev == 5 - len(kw.get("reads_there", ())), "N", f"{d} {mode} s{s}: {n_ev} reads")  # type: ignore[arg-type]
        # MINI_ROLL "L2:4 L3:2 L3:6" (SMOKE=1: L2:2): after the usual reads, MINI_PANEL of the two rolling checkpoints
        # there, under ARM_STAGEitIT; L3:6's reads skipped
        arm = "grpo_think_vs_mini"
        r = load(f"{H}/res/mini/{mode}/s{s}/roll_{arm}")
        exp, eout = expect(arm, s, mode, 3)
        got = [key(x) for x in r["rec"]]
        cg, ce = collections.Counter(got), collections.Counter(exp)
        c_ = "<R>/grpo_v2" + ("/smoke" if smoke else "")
        i2 = 2 if smoke else 4
        want_cp = [
            f"{c_}/{arm}_s{s}_L2/ckpts/{arm}_s{s}_L2_it{i2}_rolling.pt",
            f"{c_}/{arm}_s{s}_L3/ckpts/{arm}_s{s}_L3_it2_rolling.pt",
        ]
        evs = [
            x[5] for x in got if " -m code.thinking.eval " in x[5] and x not in ce
        ]  # the panel's, in order
        n_extra = 4 * (
            4 if not smoke else 2
        )  # 4 reads: waitfree's 2 nvidia-smi (normal), rm, the eval
        ok(
            r["exit"] == "0" and not (ce - cg) and sum((cg - ce).values()) == n_extra,
            "N",
            f"roll {mode} s{s}: exit / calls",
        )
        ok(
            len(evs) == 4
            and [e.split(" --ckpt ")[1].split()[0] for e in evs]
            == [want_cp[0]] * 2 + [want_cp[1]] * 2
            and all(
                f"_{arm}_{st}it{i}_s{s}_plain_k" in e
                for e, (st, i) in zip(evs, [("L2", i2)] * 2 + [("L3", 2)] * 2, strict=True)
            )
            and [e.split(" --sets ")[1].split()[0] for e in evs]
            == [READ_SETS["l3test"], READ_SETS["t1320"]] * 2,
            "N",
            f"roll {mode} s{s}: panel reads {[e[:150] for e in evs]}",
        )
        skip = f"00:00Z {arm}: no {SB}/grpo_v2{'/smoke' if smoke else ''}/{arm}_s{s}_L3/ckpts/{arm}_s{s}_L3_it6_rolling.pt (an alarm iteration?): its reads skipped"
        ok(
            skip in stdout_lines(r) and stdout_lines(r)[-1] == eout[-1],
            "N",
            f"roll {mode} s{s}: stdout {stdout_lines(r)[-3:]}",
        )


# ---------------------------------------------------------------- Q: ARM_ckreads (2026-10-16)
def expect_ckreads(
    arm: str, s: int, mode: str, its: list[int], there: frozenset[str]
) -> tuple[list[tuple], list[str]]:
    """-> (records, stdout) of ARM_ckreads for the finished knob pilot `arm` (an ARM_mb_kKNOB) with rolling L3
    checkpoints at `its` and the read dirs (basenames) `there`: written from the request, as expect()."""
    smoke = mode == "smoke"
    C = "<R>/grpo_v2" + ("/smoke" if smoke else "")
    EV = "<R>/eval/v2" + ("_smoke" if smoke else "")
    L = C + "/logs" if smoke else "<R>/logs"
    last, cvd, st = (2, "CVD=[]", "<stdout>") if smoke else (39, "CVD=[3]", "<stdout>")
    km = KNOB_RX.fullmatch(arm)
    assert km
    b = km.group(1)[: -len("_mb")]  # the base arm, whose L2 the pilot's L3 starts from
    exp, _ = expect(arm, s, mode, 3)
    recs = [x for x in exp[:3]]  # the branch and import checks, ARM_mb's pilot-options guard
    recs.append(
        (
            "git",
            WT,
            "CVD=unset",
            f"{C}/{arm}_s{s}.ckreads_commit.txt",
            st,
            f"-C {WT} rev-parse HEAD",
        )
    )
    out: list[str] = []
    for i in (0, 10, 20, 30, last):
        if i == 0:
            cp, nm = f"{C}/{b}_s{s}_L2/ckpts/{b}_s{s}_L2_it{last}.pt", f"{arm}_it0"
        elif i == last:
            cp, nm = f"{C}/{arm}_s{s}_L3/ckpts/{arm}_s{s}_L3_it{last}.pt", arm
        else:
            cp, nm = f"{C}/{arm}_s{s}_L3/ckpts/{arm}_s{s}_L3_it{i}_rolling.pt", f"{arm}_it{i}"
            if i not in its:
                out.append(
                    f"00:00Z {arm}_ckreads: no {cp.replace('<R>', SB)} (an alarm iteration?): its reads skipped"
                )
                continue
        for n, k in CKR_READS:
            k = 2 if smoke else k
            o = f"{EV}/{n}_{nm}_s{s}_plain_k{k}"
            if os.path.basename(o) in there:
                continue
            if not smoke:
                recs += [
                    (
                        "nvidia-smi",
                        WT,
                        "CVD=unset",
                        "pipe",
                        st,
                        "--query-gpu=memory.used --format=csv\\,noheader\\,nounits -i 3",
                    ),
                    (
                        "nvidia-smi",
                        WT,
                        "CVD=unset",
                        "pipe",
                        st,
                        "--query-compute-apps=pid --format=csv\\,noheader -i 3",
                    ),
                ]
            recs.append(("rm", WT, "CVD=unset", st, st, f"-rf {o}"))
            log = f"{L}/v2_eval_{os.path.basename(o)}.log"
            recs.append(
                (
                    "timeout",
                    WT,
                    cvd,
                    log,
                    log,
                    f"9000 <PY> -m code.thinking.eval --ckpt {cp} --sets {READ_SETS[n]} --out {o} --k {k} "
                    f"--budgets 64 --arms plain --batch {16 if smoke else 1024} --seed 0 --temperature 0.8 --max_action 512 "
                    f"--max_steps 96 --max_rej 10" + (" --limit 2" if smoke else ""),
                )
            )
            out.append(f"00:00Z read {os.path.basename(o)} exit 0 0s: [dry] eval ")
    out.append(f"00:00Z grpo_v2 line done ({arm}_ckreads seed {s})")
    return recs, out


for mode in ("normal", "smoke"):
    smoke = mode == "smoke"
    c_ = "grpo_v2/smoke" if smoke else "grpo_v2"
    for s in (0, 1, 2):
        for k, its in (("kl0", [10, 20, 30]), ("kl0", [10, 30]), ("t12", [10, 20, 30])):
            arm = f"grpo_think_vs_mb_k{k}"
            d = f"{H}/res/ckreads/{mode}/s{s}/{k}_{'_'.join(map(str, its))}_{arm}_ckreads"
            r = load(d)
            there = kreads(arm, s, mode) if k == "kl0" else frozenset()
            exp, eout = expect_ckreads(arm, s, mode, its, there)
            got = [key(x) for x in r["rec"]]
            tag = f"ckreads {k} {its} {mode} s{s}"
            ok(r["exit"] == "0", "Q", f"{tag}: exit {r['exit']}")
            ok(
                sorted(got) == sorted(exp),
                "Q",
                f"{tag}: calls: missing {sorted(set(exp) - set(got))[:2]} extra {sorted(set(got) - set(exp))[:2]}",
            )
            ok(got == exp, "Q", f"{tag}: call order")
            ok(
                stdout_lines(r) == eout,
                "Q",
                f"{tag}: stdout {[x for x in stdout_lines(r) if x not in eout][:2]} / {[x for x in eout if x not in stdout_lines(r)][:2]}",
            )
            n_ev = sum(" -m code.thinking.eval " in x[5] for x in got)
            want_n = 5 * (2 + len(its)) - len(there)  # it0, the rolling ones there, the last
            ok(n_ev == want_n, "Q", f"{tag}: {n_ev} reads, want {want_n}")
            md5 = dict(
                ln.split("  ", 1)[::-1] for ln in Path(f"{d}/md5.txt").read_text().split("\n") if ln
            )
            bc = f"./runs/{c_}/{arm}_s{s}.base_commit.txt"
            ok(
                f"./runs/{c_}/{arm}_s{s}.ckreads_commit.txt" in md5
                and md5.get(bc) == hashlib.md5(b"dry training commit\n").hexdigest(),
                "Q",
                f"{tag}: commit records",
            )
            ok(
                not any(" -m code.thinking.grpo " in x[5] or x[5].startswith("-m ") for x in got),
                "Q",
                f"{tag}: trained or preflighted",
            )

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
