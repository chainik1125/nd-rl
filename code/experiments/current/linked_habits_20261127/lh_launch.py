"""Launches of linked habits version 2 (SPEC2.md): pretraining on a corpus, a base's measurement (with Dan's strict-106
read), the stage-2 inputs, the stage-2 arms on one GPU (GRPO lines alone, other tasks two at a time), and waiting for a
detached job.

    python lh_launch.py pretrain --corpus C|c0n --seed 0 [--gpu 0] [--dry-run] [--lab]
    python lh_launch.py measure  --base <C>_s0 | c0n_s0 | authors_half_s0 | control_half_s0 [--tags A,B,none] [--k 8] [--skip_done]
    python lh_launch.py prep2    --corpus C            # anchor states, the base's proofs on Dan's RL pool, pairs per rater
    python lh_launch.py stage2   --corpus C [--tasks T1,T2,...] [--parallel 2] [--stagger_s 120]   (ND_GUARD=off)
    python lh_launch.py prep2    --corpus C --base_seed 1   # the second base (c24, c4: PREREG Amendment 8 point 2)
    python lh_launch.py stage2   --corpus C --base_seed 1   # dpo / dposwap_<C>b1_reit_s0, full_dpo_<C>b1_reit_s0
    python lh_launch.py rl       --line rl_<C>_<anc|rlhfreit>_s0 [--seg K --resume CKPT]  (ND_GUARD=off)
    python lh_launch.py steps    [--procs 5] [--dry-run]   # scaled pretraining steps -> LH/data/pretrain_steps.json (CPU)
    python lh_launch.py steps    --extra c0n               # c0n's own factor -> `extra.c0n` in pretrain_steps.json (CPU)
    python lh_launch.py nov14    [--dry-run]               # the Nov 14 bases into LH/bases (CPU, sha256-checked copies)
    python lh_launch.py wait     --job <job name> [--max_h H] [--every 60] [--heartbeat FILE] [--stage NAME]
    python lh_launch.py status
  Every launch command takes [--gpu 0] [--dry-run] [--lab]. Corpora: lh_core.CORPORA (env LH_CORPORA); extra bases
  (lh_core.EXTRA_BASES: c0n, trained and measured on the c0 pod, no stage 2).

This is ../two_authors_20261111/authors_launch.py (W2, commit 9618763), imported and patched; nothing in W2 is edited.
With TA = AT = LH (lh_core), W2's code writes bases LH/bases/stage1_<C>_s0.pt, measurements LH/measure/<base>/, runs
LH/runs/<line>/, logs LH/logs/, locks LH/locks/, and reads the sets from LH/sets/. Reused unchanged: `launch` (plan,
host gate, refusals, `--dry-run` writes nothing, the detached job), `host_gate` (a lab host refuses without `--lab`;
on a pod: GPU free, under 500 MiB, the lock free), `tag_launch.run_job` (the lock LH/locks/<host>_gpu<N>.lock, every
step under `timeout --kill-after=300`, LH/logs/<job>_<UTC>.log and .status.json, stop at the first failing step),
`pretrain_main`, `line_main` (W2's AuthorsSeg), `status`.

Patched in this process (authors_launch's names, by assignment):
  CORPORA      lh_core.CORPORA plus the Nov 14 corpora authors_half and control_half (measure only; pretrain refuses them)
  LINE_RE, RL_VARIANTS, parse_base, tag_core.LINES   W2's lines removed; ours registered (rl_<C>_anc_s<S>,
               rl_<C>_rlhf<rater>_s<S>; tag none, base stage1_<C>_s<S>.pt, rlvar = the kind)
  snap_cmd     W2's snapshot measurement through lh_measure.py (24 habits, step limit 51)
  make_seg_class   W2's AuthorsSeg without its iteration-0 measurement: the base is measured once, by `measure`,
               with the same code, sets, seeds and k
  plan         W2's plan with its scripts swapped (authors_launch.py -> lh_launch.py, authors_measure.py ->
               lh_measure.py) plus: pretrain `--steps` from LH/data/pretrain_steps.json; measure gains the strict-106 step;
               refusals when the detector (lh_profile) does not load, the anchor file or the reward model is missing
  TIMEOUT_H    rlvar 3 h (W2 2.5 h; 40 iterations, 4 snapshot measurements), pretrain 3 h (W2 1.5 h; the step count is scaled)
  authors_load (in `_pretrain` only) W2's loader at line cap 16 instead of tag_pretrain's 12 (SPEC2: at most 16 lines
               after the notes; the cap is only an assert in Dan's loader). A record over 16 lines refuses the
               pretraining (census) and any training record the loader skips stops it: no record is dropped.
  step count   `steps`: tokens per epoch, counted on exactly what the trainer encodes (W2's loader at cap 16, each
               record's states under its shown tag: state ids + action ids + eos), of authors_half and of every
               configured corpus; factor = the largest tokens(corpus) / tokens(authors_half); steps per seed =
               tag_core.DOSE1_STEPS[seed] x factor rounded up to the logging interval (200). Written once. An extra base
               (`steps --extra c0n`) gets its own factor, tokens(c0n) / tokens(authors_half): equal passes over its
               data, the rule of every other base (PREREG Amendment 3, point 9), under `extra` in the same file.

Stage 2 (lh_core; PREREG Amendments 3 and 4): per corpus C, raters reit (all corpora) and atom (LH_ATOM_CORPORA: c24,
c0). `prep2` is one job: `lh_pairs.py anchor` (CPU), `sample` (GPU, 4,495 x 8 attempts) and `build --rater R` (CPU:
balanced pairs, fine-tune files, probe twins for the reit rater). `stage2` is one job that holds the GPU's lock and runs
the corpus's tasks `--parallel` (2) at a time (`_tasks`: a slot scheduler in list order; a task waits for the tasks it
depends on; a failed task does not stop the others; the second task starts `--stagger_s` after the first; each task's
output in LH/logs/<task>.tasks.out). Tasks (`_task`):
  rl_<C>_rlhfreit_s0  `_rm` (the third agent's `lh_rm.py fit` on the train pairs, held-out accuracy at least 0.90
                      required), `lh_rm.py probe` on the held-out twins of the other flippable habits (probe.json beside
                      the model; never a gate), then the GRPO line (`_line`; lh_rlvar.py), snapshots at iterations 10,
                      20, 40, 60 measured untagged
  rl_<C>_anc_s0       the GRPO line with the anchor and no learned reward
  dpo_ / dposwap_<C>_reit_s0, _s1, sft_ / sftsam_<C>_reit_s0, dpo_ / dposwap_<C>_atom_s0 (c24, c0)   `_offline`: the
                      third agent's `lh_dpo.py` (`--swap-random` for dposwap) or `lh_sft.py` (sft_pref / sft_sampled) on
                      the step grid (lh_core.SNAP_STEPS); a rated run (dpo, sft) then runs `lh_measure.py probe` on its
                      snapshots in step order until one is more than 30 points over the base, and on the last one
  full_dpo_ / full_sft_<C>_<R>_s<S>   `_full`, after both runs: the bracket (the first probed snapshot past +30 and the
                      one before it; not reached: the last two) and the last snapshot, measured untagged in the rated
                      run and its no-preference run (`lh_measure.py run`, both sets), and under A, B and none on the last
                      snapshot of the seed-0 reit DPO run -> LH/arms/<run>/measure/<label>/; LH/arms/<task>.json
The launcher's gate allows one job per GPU (the lock and the 500 MiB check), so two GPU jobs on one GPU must be one
job: `stage2` is it.

`wait` polls the newest LH/logs/<job>_<UTC>.log until its `job end rc N` line and exits N (124 after --max_h; 2 if no
log appears within 10 minutes), writing a heartbeat JSON every --every seconds when asked. Job names: pretrain_<base>,
measure_<base>, prep2_<C>_s0, stage2_<C>_s0, rl_<line>_seg<K>.
"""

# ruff: noqa: E501, I001
from __future__ import annotations

import argparse
import contextlib
import gzip
import json
import math
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lh_core as C  # noqa: E402  (TA = AT = LH, sys.path)
import authors_launch as AL  # noqa: E402

T, TL, TM = AL.T, AL.TL, AL.TM
HERE = C.HERE
LH = AL.TA
assert LH == C.LH and T.AT == C.LH, (LH, T.AT, C.LH)
PY = AL.PY
W2_CORPORA = tuple(getattr(AL, "_lh_w2_corpora", AL.CORPORA))
AL._lh_w2_corpora = W2_CORPORA
CAP = 12  # tag_pretrain passes --cap 12 (K12's line limit)
LOG_EVERY = 200  # state_train's --log_every (the recipe's logging interval)
TA_HALF = Path(
    "/scratch/cowifornia/nd-rl/two_authors/data/authors_half.jsonl.gz"
)  # the half bases' corpus (read only)
STEPS_FILE = (
    C.LH / "data" / "pretrain_steps.json"
)  # the stage-2 corpora; an extra base has its own file (steps_file)
HOURS = {"strict106": 1.0, "anchor": 0.5, "sample": 1.5, "build": 2.0, "rm": 1.0, "rm_probe": 0.5, "offline_train": 1.0,
         "probe": 2.0, "snap_measure": 1.0, "line": 3.0}  # fmt: skip  # line: also the line step's timeout (TIMEOUT_H)
EST = {
    "measure": 0.2,
    "strict106": 0.08,
    "anchor": 0.05,
    "sample": 0.3,
    "build": 0.3,
    "rm": 0.25,
    "rm_probe": 0.05,
    "offline_train": 0.12,  # 400 steps (Amendment 7) and about 16 evaluations
    "probe_snap": 0.04,
    "probe_snaps": 7,
    "snap_measure": 0.11,  # a tuned snapshot under A, B, none on marker_plain (Amendment 8 point 6)
    "snap_measure_none": 0.04,  # untagged, marker_plain
    "line": 0.9,  # 40 iterations: W2's 4090 lines took 1.0-1.17 h for their first 60, plus start and the last read
}  # fmt: skip  (4090 hours, one job alone; two at once take about 1.25 x each)

# ------------------------------------------------------------------------------------------------ names and lines
AL.CORPORA = (*C.CORPORA, *C.EXTRA_BASES, *C.NOV14)
FULL_PER_TASK = (
    6  # a full-measurement task: at most 3 snapshots (the bracket and the last) of 2 runs
)
AL.RL_VARIANTS = C.LINE_KINDS
AL.LINE_RE = rf"rl_({'|'.join(map(re.escape, C.CORPORA))})(?:_({'|'.join(C.LINE_KINDS)}))?(?:_b(\d+))?_s(\d+)"
AL.TIMEOUT_H["rlvar"] = HOURS["line"]
AL.TIMEOUT_H["pretrain"] = 3.0


def parse_base(name: str) -> tuple[str, int]:
    """`<corpus>_s<seed>` (an optional `stage1_` prefix and `.pt` suffix dropped) -> (corpus, seed); our corpora and the
    two Nov 14 bases."""
    names = (*C.CORPORA, *C.EXTRA_BASES, *C.NOV14)
    alt = "|".join(map(re.escape, names))
    m = re.fullmatch(rf"({alt})_s(\d+)", name.removesuffix(".pt").removeprefix("stage1_"))
    if m is None:
        raise SystemExit(f"not a base name ({{{','.join(names)}}}_s<seed>): {name!r}")
    return m.group(1), int(m.group(2))


AL.parse_base = parse_base


def register_lines() -> None:
    for k in [k for k, L in T.LINES.items() if L.get("corpus") in W2_CORPORA or L.get("lh")]:
        del T.LINES[k]
    for c in C.CORPORA:
        for s in C.SEEDS:
            for x in C.lines_of(c, s):
                # the RL seed s only: every line from the first base (PREREG Amendment 10; W2 tied the base to s)
                T.LINES[x] = {"tag": "none", "seed": s, "base": f"stage1_{c}_s0.pt", "ladder": f"la_dose1_s{s}",
                              "corpus": c, "base_seed": 0, "rlvar": C.parse_task(x)["kind"], "lh": True}  # fmt: skip


register_lines()

_W2_SNAP = getattr(AL.snap_cmd, "_lh_orig", AL.snap_cmd)


def snap_cmd(*args: Any) -> list[str]:
    """W2's snapshot measurement through lh_measure.py, untagged (PREREG Amendment 4, point 4)."""
    cmd = _W2_SNAP(*args)
    assert Path(cmd[1]).name == "authors_measure.py", cmd[:2]
    cmd = [cmd[0], str(HERE / "lh_measure.py"), *cmd[2:]]
    cmd[cmd.index("--tags") + 1] = C.SNAP_TAGS
    cmd[cmd.index("--sets") + 1] = ",".join(C.TUNED_SETS)  # Amendment 8 point 6
    # the measurement's noise seed is 0 for every measurement (bases, offline snapshots, seed-0 lines); W2 passed the
    # line's RL seed, which would unpair a seed-1 line's snapshots from the base (PREREG Amendment 10)
    if "--seed" in cmd:
        cmd[cmd.index("--seed") + 1] = "0"
    return cmd


snap_cmd._lh_orig = _W2_SNAP  # type: ignore[attr-defined]
AL.snap_cmd = snap_cmd

_W2_MSC = getattr(AL.make_seg_class, "_lh_orig", AL.make_seg_class)


def make_seg_class():
    """W2's AuthorsSeg without the iteration-0 measurement (the `measure` job measures the base once)."""
    Seg, TLN = _W2_MSC()

    class LHSeg(Seg):
        def items(self):
            return [(it, p) for it, p in super().items() if it != 0]

        def env(self) -> dict:
            """tag_line's env (LEAN_GATE_WORKERS 8, OMP 8) with the stage-2 task's share of the CPUs (task_threads)."""
            e = super().env()
            for k, v in (
                ("LEAN_GATE_WORKERS", "LH_TASK_WORKERS"),
                ("OMP_NUM_THREADS", "LH_TASK_OMP"),
            ):
                if os.environ.get(v):
                    e[k] = os.environ[v]
            return e

    return LHSeg, TLN


make_seg_class._lh_orig = _W2_MSC  # type: ignore[attr-defined]
AL.make_seg_class = make_seg_class


# ------------------------------------------------------------------------------------------------ checks
_DET: dict[str, list[str]] = {}


def detector_problems() -> list[str]:
    """The measurement's detector must load and read a proof (lh_profile, the corpus agent's), with the habit table."""
    if "p" not in _DET:
        try:
            import lh_profile as P

            tab = C.profile_tables()
            r = P.profile("THM P SEQ P PRF", "N1 P : PR ; QED", tab)
            assert isinstance(r, dict) and "habits" in r and len(P.HABITS) == 24, (
                len(P.HABITS),
                sorted(r),
            )
            _DET["p"] = []
        except Exception as e:  # noqa: BLE001
            _DET["p"] = [
                f"the detector lh_profile does not load or read a proof: {type(e).__name__}: {e}"[
                    :300
                ]
            ]
    return _DET["p"]


def base_of(corpus: str, seed: int = 0) -> Path:
    return C.base_path(corpus, seed)


def base_ck(token: str) -> Path:
    """The base checkpoint of a run's corpus field: <c> -> stage1_<c>_s0.pt, <c>b1 -> stage1_<c>_s1.pt."""
    return base_of(*C.split_base(token))


def pair_files(corpus: str, rater: str) -> dict[str, Path]:
    """The pair files of a corpus field (<c>: LH/pairs/<c>_s0/<R>; the second base <c>b1: LH/pairs/<c>_s1/<R>)."""
    c, bs = C.split_base(corpus)
    d = C.pairs_dir(c, rater, bs)
    return {**{k: d / f"{k}.jsonl" for k in ("train", "heldout", "sft_pref", "sft_sampled", "probe_twins")},
            "summary": d / "summary.json"}  # fmt: skip


def merge_argv(argv: list[str], extra: list[str]) -> list[str]:
    """argv with `extra` (smoke settings): an option already in argv gets the extra's value in place (never twice);
    other options and flags are appended."""
    out, i = list(argv), 0
    while i < len(extra):
        k = extra[i]
        has_val = k.startswith("--") and i + 1 < len(extra) and not extra[i + 1].startswith("--")
        if k in out and has_val and out.index(k) + 1 < len(out):
            out[out.index(k) + 1] = extra[i + 1]
        elif k not in out:
            out += [k, extra[i + 1]] if has_val else [k]
        i += 2 if has_val else 1
    return out


def trainer_argv(arm: str, smoke: bool = False) -> list[str]:
    """The third agent's trainer for an offline run (lh_dpo.py / lh_sft.py, their own CLI), from the seed-0 base with
    the run's own seed, on the step grid (`--snap-steps`: snap_step<k>.pt, and final.pt). dpo / dposwap: the balanced
    train pairs (dposwap with `--swap-random`); sft: the preferred members of every train pair; sftsam: their sampled
    members. All log their KL to the base on the corpus's fixed states (`--kl-states`, lh_pairs.py anchor)."""
    t = C.parse_task(arm)
    pf, ck, d = pair_files(t["corpus"], t["rater"]), str(base_ck(t["corpus"])), str(C.arm_dir(arm))
    kls = C.kl_states_path(
        C.split_base(t["corpus"])[0]
    )  # corpus records: the same states for both bases
    common = ["--base", ck, "--heldout", str(pf["heldout"]), "--out", d, "--kl-states", str(kls),
              "--kl-n", str(C.KL_STATES), "--seed", str(t["seed"]), "--lr", str(C.OFFLINE_LR), "--max-steps", str(C.OFFLINE_MAX_STEPS),
              "--snap-steps", ",".join(map(str, C.SNAP_STEPS)),
              *(["--device", "cpu"] if smoke else ["--device", "cuda", "--max-tokens", str(C.SHARED_MAX_TOKENS)])]  # fmt: skip
    if t["kind"] in ("dpo", "dposwap"):  # Amendment 7 point 3: the swap seed is the run's seed
        return [PY, str(HERE / "lh_dpo.py"), *common, "--pairs", str(pf["train"]),
                *(["--swap-random", "--swap-seed", str(t["seed"])] if t["kind"] == "dposwap" else [])]  # fmt: skip
    data = pf["sft_pref"] if t["kind"] == "sft" else pf["sft_sampled"]
    nt = norm_tokens_of(
        t["corpus"], t["rater"]
    )  # Amendment 7 point 4: one constant for the rated run and its control
    norm = ["--norm-tokens", str(nt["N"]), "--batch", str(nt["batch"])] if nt else []
    return [PY, str(HERE / "lh_sft.py"), *common, "--data", str(data), *norm]


def norm_tokens_of(corpus: str, rater: str) -> dict[str, Any] | None:
    """The fine-tune normalizer the pair build stored (summary.json `norm_tokens`: N = batch x mean action tokens per
    proof of sft_sampled.jsonl), or None (a pair build before Amendment 7)."""
    f = pair_files(corpus, rater)["summary"]
    if not f.exists():
        return None
    nt = json.loads(f.read_text()).get("norm_tokens")
    return nt if isinstance(nt, dict) and nt.get("N") else None


def rm_probe_argv(corpus: str, rater: str, seed: int = 0, smoke: bool = False) -> list[str]:
    """The third agent's reward-model probe (`lh_rm.py probe`) on the held-out twins of the other flippable habits."""
    return [PY, str(HERE / "lh_rm.py"), "probe", "--rm", str(C.rm_path(corpus, rater, seed)),
            "--pairs", str(pair_files(corpus, rater)["probe_twins"]), "--out", str(C.rm_probe_path(corpus, rater, seed)),
            "--device", "cpu" if smoke else "cuda", *(["--threads", "1", "--n-boot", "200"] if smoke else [])]  # fmt: skip


def rm_argv(corpus: str, rater: str, seed: int = 0, smoke: bool = False) -> list[str]:
    """The third agent's reward-model fit (`lh_rm.py fit`): RM.pt only when its held-out accuracy reaches 0.90 (else
    RM.pt.failed and exit 3), RM.pt.json always."""
    pf = pair_files(corpus, rater)
    return [PY, str(HERE / "lh_rm.py"), "fit", "--base", str(base_of(corpus, seed)), "--pairs", str(pf["train"]),
            "--heldout", str(pf["heldout"]), "--out", str(C.rm_path(corpus, rater, seed)), "--seed", str(seed),
            *(["--device", "cpu"] if smoke else ["--device", "cuda", "--max-tokens", str(C.SHARED_MAX_TOKENS)])]  # fmt: skip


AT_RUN = "; refused when its turn comes, the other tasks run"  # a per-task refusal that does not stop the stage-2 launch


def task_problems(task: str) -> list[str]:
    """Why a stage-2 task cannot start now (its inputs). Not checked here: what earlier tasks of the same job make
    (an rlhf task's reward model, a full measurement's runs); "already ..." means the task's outputs exist."""
    t = C.parse_task(task)
    c = t[
        "corpus"
    ]  # the corpus field: <c>, or <c>b1 on the second base (pairs, base and its measurement follow it)
    cc, bs = C.split_base(c)
    out: list[str] = []
    if cc not in C.CORPORA:
        return [f"{task}: corpus {cc} is not configured (LH_CORPORA)"]
    if bs and task not in C.base2_tasks(cc):
        return [
            f"{task}: not a second-base task (lh_core.base2_tasks({cc!r}) = {C.base2_tasks(cc)})"
        ]
    if t["rater"] is not None and t["rater"] not in C.raters_of(cc):
        out.append(f"{task}: rater {t['rater']} does not run on {cc} (LH_ATOM_CORPORA)")
    if not base_ck(c).exists():
        out.append(f"{task}: no base {base_ck(c)}")
    if t["type"] == "full":
        if base_share(c, C.RATERS[t["rater"]]) is None:
            out.append(
                f"{task}: the base is not measured ({base_summary(c)}): the +30 target needs its share"
            )
        if full_path(task).exists() and json.loads(full_path(task).read_text()).get("done"):
            out.append(f"{task}: already measured ({full_path(task)})")
        return out + detector_problems()
    if t["rater"] is not None:
        pf = pair_files(c, t["rater"])
        need = ["train", "heldout"] + {"sft": ["sft_pref"], "sftsam": ["sft_sampled"]}.get(
            t["kind"], []
        )
        out += [f"{task}: no {pf[k]} (prep2)" for k in need if not pf[k].exists()]
    if C.not_run_path(task).exists():
        out.append(
            f"{task}: already decided not to run ({json.loads(C.not_run_path(task).read_text()).get('reason')})"
        )
    if t["type"] == "line":
        _, _, p, _ = plan(rl_ns(argparse.Namespace(gpu=0), task))
        out += [f"{task}: {q}" for q in p if not q.startswith("no reward model")]
        if t["rater"] is not None and not (HERE / "lh_rm.py").exists():
            out.append(f"{task}: no {HERE / 'lh_rm.py'} (the third agent's reward model)")
    else:
        f = HERE / ("lh_sft.py" if t["kind"].startswith("sft") else "lh_dpo.py")
        if not f.exists():
            out.append(f"{task}: no {f} (the third agent's trainer)")
        elif t["kind"].startswith("sft"):
            if "--norm-tokens" not in f.read_text():
                out.append(
                    f"{task}: {f.name} has no --norm-tokens yet (the tuning agent's change, PREREG Amendment 7 point 4){AT_RUN}"
                )
            if (
                pair_files(c, t["rater"])["summary"].exists()
                and norm_tokens_of(c, t["rater"]) is None
            ):
                out.append(
                    f"{task}: no norm_tokens in {pair_files(c, t['rater'])['summary']} (a pair build before Amendment 7){AT_RUN}"
                )
        probed = not C.is_rated(t["kind"]) or (C.arm_dir(task) / "probe" / "probe.json").exists()
        if (C.arm_dir(task) / "done.json").exists() and probed:
            out.append(
                f"{task}: already trained{' and probed' if C.is_rated(t['kind']) else ''} ({C.arm_dir(task)})"
            )
    out += detector_problems()
    return out


# ------------------------------------------------------------------------------------------------ plan
SWAP = {str(C.W2 / "authors_launch.py"): str(HERE / "lh_launch.py"),
        str(C.W2 / "authors_measure.py"): str(HERE / "lh_measure.py")}  # fmt: skip
_W2_PLAN = getattr(AL.plan, "_lh_orig", AL.plan)


def plan(a: argparse.Namespace) -> tuple[str, list[dict], list[str], list[str]]:
    """W2's plan for our corpora and lines, and our stage-2 jobs (module docstring). Writes nothing."""
    if a.cmd == "prep2":
        return prep2_plan(a)
    if a.cmd == "stage2":
        return stage2_plan(a)
    if a.cmd == "pretrain" and a.corpus in C.NOV14:
        raise SystemExit(f"{a.corpus} is a Nov 14 base: measured, never retrained")
    if a.cmd == "rl" and a.line not in T.LINES:
        raise SystemExit(
            f"not a line of this experiment: {a.line!r} (lines: {', '.join(C.all_lines())})"
        )
    job, steps, problems, notes = _W2_PLAN(a)
    for s in steps:
        s["argv"] = [SWAP.get(x, x) for x in s["argv"]]
        assert not any(str(C.W2) in x for x in s["argv"]), s["argv"]
    problems = [p.replace("authors_launch.py", "lh_launch.py") for p in problems]
    problems += detector_problems()
    if a.cmd == "pretrain":
        st = scaled_steps(a.seed, a.corpus)
        problems += steps_problems(a.corpus, st)
        if st is not None:
            steps[0]["argv"] += ["--steps", str(st["steps"])]
            steps[0]["est_h"] = round(AL.EST["pretrain"] * st["factor"], 2)
            notes.append(f"scaled to {st['steps']:,} steps (x {st['factor']:.4f}: tokens per epoch, the largest of "
                         f"{st['corpora']} over authors_half's; {Path(st['file']).name})")  # fmt: skip
        if AL.corpus_path(a.corpus).exists():
            cen = census(AL.corpus_path(a.corpus))
            notes.append(f"census: {cen['records']:,} records, n_lines max {cen['n_lines_max']}, over {CAP} lines {cen['over_12']:,}, "
                         f"notes {cen['notes']:,}; lh_corpus {cen['lh_corpus']}")  # fmt: skip
            problems += cen["problems"]
    if a.cmd == "measure":
        corpus, seed = parse_base(a.base)
        if corpus in C.NOV14:
            problems = [
                p.replace(
                    f"(lh_launch.py pretrain --corpus {corpus} --seed {seed})",
                    "(lh_launch.py nov14)",
                )
                for p in problems
            ]
        bid, ck = AL.base_name(corpus, seed), AL.base_path(corpus, seed)
        d = LH / "measure" / bid
        if (d / "strict106__none.jsonl").exists() and not a.skip_done:
            problems.append(
                f"{d / 'strict106__none.jsonl'} exists: never overwritten (--skip_done skips it)"
            )
        steps.append(AL.step(f"strict106_{bid}", [PY, str(HERE / "lh_measure.py"), "strict106", "--ckpt", str(ck), "--ckpt_id", bid,
                                                  "--out_dir", str(d), *(["--resume"] if a.skip_done else [])],
                             HOURS["strict106"], EST["strict106"]))  # fmt: skip
        steps[0]["est_h"] = EST["measure"]
        notes.append(
            f"step limit {C.STEP_LIMIT} in every sampler call; Dan's strict 106 at k 64 (tag_danread, as W2's lines read a base)"
        )
    if a.cmd == "rl":
        t = C.parse_task(a.line)
        if not C.anchor_path(t["corpus"]).exists():
            problems.append(f"no anchor file {C.anchor_path(t['corpus'])} (prep2)")
        if t["rater"] is not None and not C.line_rm_path(t["corpus"], t["rater"]).exists():
            problems.append(
                f"no reward model {C.line_rm_path(t['corpus'], t['rater'])} (stage2 fits it first)"
            )
        s0 = C.line_name(t["corpus"], t["kind"], 0)
        if t["rater"] is not None and t["seed"] and C.not_run_path(s0).exists():
            problems.append(
                f"{s0} was decided not run (its reward model missed the bar): no RL seed {t['seed']} line"
            )
        for s in steps:
            s["est_h"] = EST["line"]
            if (
                "--iters" in s["argv"]
            ):  # W2's variant lines run rlvar.ITERS (60); ours run lh_core.ITERS (Amendment 6)
                s["argv"][s["argv"].index("--iters") + 1] = str(C.ITERS)
        notes = [n.replace(f"{AL.VAR_ITERS} iterations", f"{C.ITERS} iterations") for n in notes]
        notes.append(f"{t['kind']}: {C.ITERS} iterations, step limit {C.STEP_LIMIT}, KL anchor {C.KL_ANCHOR} to the base on "
                     f"{C.anchor_path(t['corpus']).name}, snapshots {C.GRPO_SNAPS}" + ("" if t["rater"] is None else f", reward x (1 + {C.RM_BONUS} sigmoid(RM))"))  # fmt: skip
    return job, steps, problems, notes


plan._lh_orig = _W2_PLAN  # type: ignore[attr-defined]
AL.plan = plan


def rl_ns(a: argparse.Namespace, line: str) -> argparse.Namespace:
    return argparse.Namespace(cmd="rl", line=line, gpu=a.gpu, seg=1, resume=None, snap_k=8,
                              snap_sets=",".join(C.SETS), stop_on_held=False)  # fmt: skip


def prep2_plan(a: argparse.Namespace) -> tuple[str, list[dict], list[str], list[str]]:
    """Anchor states (CPU), the base's proofs on the RL pool (GPU), the pairs of each rater (CPU). With --base_seed 1
    (the second base pass): its proofs and the reiteration rater's pairs only (LH/pairs/<c>_s1), no anchor (corpus
    records, already made), no probe rows (no reward model on the second base)."""
    if getattr(a, "base_seed", 0):
        return prep2_base2_plan(a)
    c, problems, notes, steps = a.corpus, [], [], []
    lp = str(HERE / "lh_pairs.py")
    if not base_of(c).exists():
        problems.append(f"no base {base_of(c)}")
    for f in (C.RL_POOL, HERE / "lh_twin.py", HERE / "lh_profile.py", C.corpus_path(c)):
        if not Path(f).exists():
            problems.append(f"missing {f}")
    problems += detector_problems()
    if not C.anchor_path(c).exists():
        steps.append(
            AL.step(
                f"anchor_{c}", [PY, lp, "anchor", "--corpus", c], HOURS["anchor"], EST["anchor"]
            )
        )
    if not (C.pairs_dir(c) / "sample.json").exists():
        steps.append(
            AL.step(
                f"sample_{c}",
                [PY, lp, "sample", "--corpus", c, "--device", "cuda"],
                HOURS["sample"],
                EST["sample"],
            )
        )
    for r in C.raters_of(c):
        if not pair_files(c, r)["summary"].exists():
            steps.append(
                AL.step(
                    f"build_{c}_{r}",
                    [
                        PY,
                        lp,
                        "build",
                        "--corpus",
                        c,
                        "--rater",
                        r,
                        *([] if C.has_rm(c, r) else ["--no_probe"]),
                    ],
                    HOURS["build"],
                    EST["build"],
                )
            )
    if not steps:
        problems.append(f"prep2 {c}: everything exists (anchor, sample, pairs of {C.raters_of(c)})")
    notes.append(f"raters {C.raters_of(c)}; RL pool {C.RL_POOL.name} x {C.PAIR_K}, step limit {C.STEP_LIMIT}; "
                 f"{C.HELDOUT_PAIRS} held-out pairs per rater; anchor: {C.ANCHOR_PROOFS} untagged records, every step")  # fmt: skip
    return f"prep2_{c}_s0", steps, problems, notes


def prep2_base2_plan(a: argparse.Namespace) -> tuple[str, list[dict], list[str], list[str]]:
    c, bs, problems, steps = a.corpus, a.base_seed, [], []
    lp = str(HERE / "lh_pairs.py")
    if bs != C.BASE2_SEED or c not in C.BASE2_CORPORA:
        raise SystemExit(
            f"no second base pass for {c} at base seed {bs} (LH_BASE2_CORPORA {C.BASE2_CORPORA})"
        )
    if not base_of(c, bs).exists():
        problems.append(f"no base {base_of(c, bs)} (pretrain --corpus {c} --seed {bs})")
    for f in (C.RL_POOL, HERE / "lh_twin.py", HERE / "lh_profile.py"):
        if not Path(f).exists():
            problems.append(f"missing {f}")
    problems += detector_problems()
    if not (C.pairs_dir(c, seed=bs) / "sample.json").exists():
        steps.append(AL.step(f"sample_{c}_s{bs}", [PY, lp, "sample", "--corpus", c, "--base_seed", str(bs), "--device", "cuda"],
                             HOURS["sample"], EST["sample"]))  # fmt: skip
    for r in C.BASE2_PLAN:
        if not C.pairs_dir(c, r, bs).joinpath("summary.json").exists():
            steps.append(AL.step(f"build_{c}_s{bs}_{r}", [PY, lp, "build", "--corpus", c, "--base_seed", str(bs), "--rater", r, "--no_probe"],
                                 HOURS["build"], EST["build"]))  # fmt: skip
    if not steps:
        problems.append(
            f"prep2 {c}_s{bs}: everything exists (sample, pairs of {list(C.BASE2_PLAN)})"
        )
    notes = [f"second base {c}_s{bs} (PREREG Amendment 8 point 2): its proofs on {C.RL_POOL.name} x {C.PAIR_K}, the pairs of "
             f"{list(C.BASE2_PLAN)} balanced and capped as for the first base, no probe rows"]  # fmt: skip
    return f"prep2_{c}_s{bs}", steps, problems, notes


def job_name(a: argparse.Namespace) -> str:
    """stage2_<c>_s<base seed>, or stage2_<c>_rl<seed> for Amendment 10's RL-seed lines."""
    rs = getattr(a, "rl_seed", 0) or 0
    return (
        f"stage2_{a.corpus}_rl{rs}"
        if rs
        else f"stage2_{a.corpus}_s{getattr(a, 'base_seed', 0) or 0}"
    )


def stage2_tasks(a: argparse.Namespace) -> list[str]:
    """The job's tasks: the corpus's stage 2 (lh_core.tasks_of), or with --base_seed 1 its second base pass
    (lh_core.base2_tasks); --tasks picks some of them."""
    bs = getattr(a, "base_seed", 0) or 0
    rs = getattr(a, "rl_seed", 0) or 0
    if rs:  # PREREG Amendment 10: the GRPO lines again with RL seed rs, from the first base
        allowed = C.rl2_tasks(a.corpus) if rs == C.RL2_SEED else []
        if not allowed or bs:
            raise SystemExit(
                f"no RL seed {rs} lines for {a.corpus} (lh_core.RL2_CORPORA {C.RL2_CORPORA}, seed {C.RL2_SEED})"
            )
        tasks = a.tasks.split(",") if a.tasks else allowed
        bad = [x for x in tasks if x not in allowed]
        if bad or len(set(tasks)) != len(tasks):
            raise SystemExit(
                f"not an RL seed {rs} task of {a.corpus}: {bad or tasks} (tasks: {allowed})"
            )
        return tasks
    allowed = C.base2_tasks(a.corpus) if bs else C.tasks_of(a.corpus)
    if bs and (bs != C.BASE2_SEED or not allowed):
        raise SystemExit(
            f"no second base pass for {a.corpus} at base seed {bs} (LH_BASE2_CORPORA {C.BASE2_CORPORA})"
        )
    tasks = a.tasks.split(",") if a.tasks else allowed
    for x in tasks:
        if C.parse_task(x)["corpus"] != C.base_token(a.corpus, bs) or x not in allowed:
            raise SystemExit(f"not a stage-2 task of corpus {a.corpus}: {x!r} (tasks: {allowed})")
    if len(set(tasks)) != len(tasks):
        raise SystemExit(f"a task named twice: {tasks}")
    return tasks


def task_hours(task: str) -> float:
    t = C.parse_task(task)
    if t["type"] == "line":
        return HOURS["line"] + (HOURS["rm"] + HOURS["rm_probe"] if t["rater"] else 0.0)
    if t["type"] == "full":
        return FULL_PER_TASK * HOURS["snap_measure"]
    return HOURS["offline_train"] + (HOURS["probe"] if C.is_rated(t["kind"]) else 0.0)


def task_est(task: str) -> float:
    """4090 hours alone. A rated run's probes: about EST["probe_snaps"] snapshots (they stop past +30, plus the last)."""
    t = C.parse_task(task)
    if t["type"] == "line":
        return EST["line"] + (EST["rm"] + EST["rm_probe"] if t["rater"] else 0.0)
    if t["type"] == "full":
        tagged = (t["kind"], t["rater"], t["seed"]) == C.TAGGED_SNAPSHOT and not C.split_base(
            t["corpus"]
        )[1]
        return FULL_PER_TASK * EST["snap_measure_none"] + (
            EST["snap_measure"] - EST["snap_measure_none"] if tagged else 0.0
        )
    return EST["offline_train"] + (
        EST["probe_snaps"] * EST["probe_snap"] if C.is_rated(t["kind"]) else 0.0
    )


def stage2_plan(a: argparse.Namespace) -> tuple[str, list[dict], list[str], list[str]]:
    """One job, one step: `_tasks`, the corpus's stage-2 tasks `--parallel` at a time on the job's GPU."""
    tasks = stage2_tasks(a)
    every = [q for x in tasks for q in task_problems(x)]
    problems = [
        q for q in every if not q.endswith(AT_RUN)
    ]  # the fine-tune pair without its normalizer: that pair only
    if (
        any(C.parse_task(x)["type"] == "line" for x in tasks)
        and os.environ.get("ND_GUARD") != "off"
    ):
        problems.append(
            "the GRPO arms run guard-off like W2's variant lines: launch with ND_GUARD=off"
        )
    par = max(1, a.parallel)
    alone = [x for x in tasks if par > 1 and task_weight(x, par) >= par]
    est = (
        sum(task_est(x) for x in alone)
        + sum(task_est(x) for x in tasks if x not in alone) * (1.25 if par > 1 else 1.0) / par
    )
    # the job's backstop: twice the estimate + 3 h, never less than the longest task's own cap + 1 h, never more than
    # every task at its own cap (each task already has its own timeout; this only bounds a stuck scheduler)
    worst = (sum(task_hours(x) for x in alone) + sum(task_hours(x) for x in tasks if x not in alone) / par
             + max(task_hours(x) for x in tasks) + 0.5)  # fmt: skip
    hours = round(min(worst, max(2 * est + 3, max(task_hours(x) for x in tasks) + 1)), 1)
    argv = [PY, str(HERE / "lh_launch.py"), "_tasks", "--gpu", str(a.gpu), "--parallel", str(par),
            "--stagger_s", str(a.stagger_s), "--tasks", ",".join(tasks)]  # fmt: skip
    notes = [
        f"{len(tasks)} tasks, {par} at a time on GPU {a.gpu}"
        + (f" (alone: {', '.join(alone)})" if alone else "")
        + f": {', '.join(tasks)}",
        *(f"WILL BE REFUSED: {q}" for q in every if q.endswith(AT_RUN)),
    ]
    return (
        job_name(a),
        [AL.step(job_name(a).replace("stage2_", "tasks_", 1), argv, hours, round(est, 2))],
        problems,
        notes,
    )


# ------------------------------------------------------------------------------------------------ _pretrain
def census(fn: str | Path, limit: int = 0) -> dict[str, Any]:
    """Line counts against the loader's cap (16, notes included), notes per record, and the corpus label."""
    out: dict[str, Any] = {"records": 0, "notes": 0, "n_lines_max": 0, "over_12": 0, "over_cap": 0, "lh_corpus": {},
                           "problems": []}  # fmt: skip
    op = gzip.open if str(fn).endswith(".gz") else open
    with op(fn, "rt", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            out["records"] += 1
            out["notes"] += len(r.get("notes") or [])
            out["n_lines_max"] = max(out["n_lines_max"], r["n_lines"])
            out["over_12"] += r["n_lines"] > CAP
            out["over_cap"] += r["n_lines"] > C.CAP_LOAD
            lc = str(r.get("lh_corpus"))
            out["lh_corpus"][lc] = out["lh_corpus"].get(lc, 0) + 1
            if limit and out["records"] >= limit:
                break
    if out["over_cap"]:
        out["problems"].append(
            f"{out['over_cap']} records over {C.CAP_LOAD} lines (SPEC2's limit after the notes)"
        )
    return out


_W2_LOAD = getattr(AL.authors_load, "_lh_orig", AL.authors_load)


def lh_load(fn, tok, cap, limit=0):
    """W2's authors_load at line cap 16 (module docstring); a training record it skips stops the run."""
    import tag_pretrain as TP

    if cap and cap != CAP:
        raise RuntimeError(
            f"the loader got cap {cap}; tag_pretrain's recipe passes {CAP} (the recipe changed?)"
        )
    out, skipped = _W2_LOAD(fn, tok, C.CAP_LOAD if cap else cap, limit)
    if str(fn) == str(TP._cfg["data"]) and skipped:
        raise RuntimeError(
            f"{skipped} training records of {fn} did not decompose: no record may be dropped"
        )
    return out, skipped


lh_load._lh_orig = _W2_LOAD  # type: ignore[attr-defined]


def pretrain_main(argv: list[str]) -> None:
    import tag_pretrain as TP

    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--limit", type=int, default=0)
    ka, _ = ap.parse_known_args(argv)
    st = None
    if ka.corpus in C.NOV14:
        raise SystemExit(f"{ka.corpus} is a Nov 14 base: never retrained")
    named = ka.corpus in C.CORPORA or ka.corpus in C.EXTRA_BASES
    if named:
        st = scaled_steps(ka.seed, ka.corpus)
        sp = steps_problems(ka.corpus, st)
        if sp:
            raise SystemExit("; ".join(sp))
        if ka.steps is None:
            argv = [*argv, "--steps", str(st["steps"])]
        elif ka.steps != st["steps"]:
            raise SystemExit(
                f"--steps {ka.steps}: corpus {ka.corpus} trains {st['steps']} steps ({st['file']})"
            )
    data = AL.corpus_path(ka.corpus) if named else Path(ka.corpus)
    if not data.exists():
        raise SystemExit(f"no corpus {data}")
    cen = census(data, ka.limit)
    if cen["problems"]:
        raise SystemExit(f"corpus {data}: {cen['problems']}")
    man = AL.corpus_manifest(ka.corpus) if named else None
    lh = {"experiment": "linked_habits_20261127", "version": 2, "corpus": ka.corpus, "corpus_file": str(data),
          "corpus_sha256": C.sha256(data), "corpus_manifest": str(man) if man is not None and man.exists() else None,
          "tables_sha256": {p.name: C.sha256(p) for p in (C.NOTE_TABLE, C.HABIT_TABLE) if p.exists()}, "census": cen,
          "line_cap": f"{C.CAP_LOAD} in the loader (tag_pretrain passes {CAP})", "steps": st,
          "steps_file": st["file"] if st else None}  # fmt: skip
    AL.authors_load = lh_load
    orig_main = TP.main

    def main(targv: list[str]) -> None:
        w2_wrap = TP._wrap_train  # authors_launch's wrap, installed just before TP.main

        def wrap(stb, corpus, steps):
            w2_wrap(stb, corpus, steps)
            inner = stb.train

            def train(args, tok, data_, held, dev, record):
                model, extra = inner(args, tok, data_, held, dev, record)
                extra["two_authors"]["experiment"] = lh["experiment"]
                extra["linked_habits"] = lh
                sc = Path(args.out).with_suffix(".tag.json")
                side = json.loads(sc.read_text())
                side["two_authors"]["experiment"] = lh["experiment"]
                side["linked_habits"] = lh
                sc.write_text(json.dumps(side, indent=1, default=str))
                return model, extra

            stb.train = train

        TP._wrap_train = wrap
        print(f"lh _pretrain: {cen['records']:,} records, n_lines max {cen['n_lines_max']} (cap {C.CAP_LOAD}), "
              f"{cen['over_12']:,} over {CAP}, {cen['notes']:,} notes", flush=True)  # fmt: skip
        orig_main(targv)

    TP.main = main
    try:
        AL.pretrain_main(argv)
    finally:
        TP.main = orig_main


# ------------------------------------------------------------------------------------------------ _line
def line_main(argv: list[str]) -> None:
    import lh_rlvar  # noqa: F401  (our variants in rlvar.VARIANTS; their argv and GRPO process)

    AL.line_main(argv)


# ------------------------------------------------------------------------------------------------ steps
def steps_file(corpus: str) -> Path:
    """Every base's step count is in LH/data/pretrain_steps.json: the four corpora at the top, an extra base (c0n)
    under `extra` with its own factor."""
    return STEPS_FILE


def scaled_steps(
    seed: int, corpus: str | None = None, path: Path | None = None
) -> dict[str, Any] | None:
    p = path or STEPS_FILE
    if not p.exists():
        return None
    doc = json.loads(p.read_text())
    if corpus in C.EXTRA_BASES:
        doc = (doc.get("extra") or {}).get(corpus)
        if doc is None:
            return None
    n = doc["steps"].get(str(seed))
    if n is None:
        return None
    return {"steps": n, "factor": doc["factor"], "old_steps": doc["old_steps"][str(seed)], "file": str(p),
            "corpora": sorted(doc.get("factor_per_corpus", {})),
            "sha256": {c: v.get("sha256") for c, v in (doc.get("counts") or {}).items()}}  # fmt: skip


def steps_problems(corpus: str, st: dict[str, Any] | None) -> list[str]:
    """The step file must cover this corpus, counted on the file that is there now."""
    f = steps_file(corpus)
    if st is None:
        how = (
            "lh_launch.py steps, once every corpus exists"
            if corpus in C.CORPORA
            else f"lh_launch.py steps --extra {corpus}: its entry under `extra`"
        )
        return [f"no scaled step count for {corpus} in {f} ({how})"]
    if corpus not in st["corpora"]:
        return [f"{f} was computed without {corpus} ({st['corpora']})"]
    sha = st["sha256"].get(corpus)
    if sha and AL.corpus_path(corpus).exists() and C.sha256(AL.corpus_path(corpus)) != sha:
        return [
            f"{AL.corpus_path(corpus)} changed after the step count ({f}): recount (a new step file)"
        ]
    return []


def count_tokens(fn: str) -> dict[str, Any]:
    """Pairs and tokens per epoch of one corpus as the trainer encodes it (W2's loader at cap 16)."""
    import tag_pretrain as TP

    tok = T.install_dan()()
    TP._cfg.update(data=str(fn), limit=0, pairs={})
    data, skipped = _W2_LOAD(str(fn), tok, C.CAP_LOAD)
    n_pairs = sum(len(p) for p in data)
    n_tok = sum(len(x) + len(y) for p in data for x, y in p)
    return {"file": str(fn), "sha256": C.sha256(fn), "records": len(data), "skipped": skipped, "pairs": n_pairs,
            "tokens": n_tok, "mean_pair_tokens": round(n_tok / max(1, n_pairs), 2), "pairs_per_tag": TP._cfg["pairs"].get("train")}  # fmt: skip


def steps_doc(counts: dict[str, dict[str, Any]]) -> dict[str, Any]:
    half = counts["authors_half"]["tokens"]
    fac = {c: counts[c]["tokens"] / half for c in counts if c != "authors_half"}
    f = max(fac.values())
    steps = {
        str(s): int(math.ceil(n * f / LOG_EVERY) * LOG_EVERY) for s, n in T.DOSE1_STEPS.items()
    }
    return {"factor": round(f, 6), "factor_per_corpus": {c: round(v, 6) for c, v in fac.items()}, "log_every": LOG_EVERY,
            "old_steps": {str(s): n for s, n in T.DOSE1_STEPS.items()}, "steps": steps, "counts": counts, "line_cap": C.CAP_LOAD,
            "rule": "steps = tag_core.DOSE1_STEPS[seed] x max over the corpora of tokens(corpus) / tokens(authors_half), "
                    "rounded up to a multiple of the logging interval; tokens = sum over (state, action) pairs of state ids + "
                    f"action ids + eos, as W2's loader encodes them (shown tag) at line cap {C.CAP_LOAD}",
            "written_utc": AL.now()}  # fmt: skip


def steps_main(argv: list[str]) -> dict[str, Any]:
    import multiprocessing as mp

    ap = argparse.ArgumentParser(prog="lh_launch.py steps")
    ap.add_argument("--procs", type=int, default=5)
    ap.add_argument("--extra", default=None, choices=list(C.EXTRA_BASES),
                    help="an extra base (c0n): its own factor, tokens(corpus) / tokens(authors_half) (equal passes over its "
                         "own data), added under `extra` in the existing step file")  # fmt: skip
    ap.add_argument("--out", default=str(STEPS_FILE))
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    names = [a.extra] if a.extra else list(C.CORPORA)
    files = {"authors_half": str(TA_HALF), **{c: str(AL.corpus_path(c)) for c in names}}
    for f in files.values():
        if not Path(f).exists():
            raise SystemExit(f"no {f}: the step count needs every corpus")
    t0 = time.time()
    with mp.get_context("fork").Pool(min(a.procs, 8, len(files))) as P:
        res = dict(zip(files, P.map(count_tokens, list(files.values())), strict=True))
    for k, v in res.items():
        if v["skipped"]:
            raise SystemExit(f"{k}: {v['skipped']} records did not decompose")
        print(
            f"{k}: {v['records']:,} records, {v['pairs']:,} pairs, {v['tokens']:,} tokens (mean pair {v['mean_pair_tokens']})"
        )
    doc = steps_doc(res)
    doc["secs"] = round(time.time() - t0, 1)
    print(
        f"factor {doc['factor']} (per corpus {doc['factor_per_corpus']}); steps {doc['old_steps']} -> {doc['steps']}"
    )
    if a.extra:
        doc["rule"] = (f"{a.extra}: equal passes over its own data (PREREG Amendment 3, point 9): steps = "
                       "tag_core.DOSE1_STEPS[seed] x tokens(corpus) / tokens(authors_half), its own factor (not the largest "
                       "factor the four c corpora share), rounded up to a multiple of the logging interval; tokens as above")  # fmt: skip
        if not a.dry_run:
            add_extra(Path(a.out), a.extra, doc)
            print(f"wrote {a.out} (extra.{a.extra})")
        return doc
    if not a.dry_run:
        C.write_once(Path(a.out), doc, ("factor", "steps", "old_steps", "factor_per_corpus"))
        print(f"wrote {a.out}")
    return doc


def add_extra(path: Path, base: str, entry: dict[str, Any]) -> dict[str, Any]:
    """Add an extra base's entry under `extra` of the existing step file; an existing entry must agree (factor and
    steps), else nothing is written; the four corpora's entries are never touched."""
    if not path.exists():
        raise SystemExit(
            f"no {path}: the four corpora's step count comes first (lh_launch.py steps)"
        )
    doc = json.loads(path.read_text())
    old = (doc.get("extra") or {}).get(base)
    if old is not None:
        if (old["factor"], old["steps"]) != (entry["factor"], entry["steps"]):
            raise SystemExit(
                f"{path}: extra.{base} exists and differs (factor {old['factor']} vs {entry['factor']}): never overwritten"
            )
        return doc
    doc.setdefault("extra", {})[base] = entry
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(doc, indent=1) + "\n")
    tmp.replace(path)
    return doc


# ------------------------------------------------------------------------------------------------ nov14
def nov14_main(argv: list[str]) -> list[str]:
    """The two Nov 14 bases (and their .tag.json / .pt.args.json) into LH/bases, byte-identical (sha256)."""
    ap = argparse.ArgumentParser(prog="lh_launch.py nov14")
    ap.add_argument("--src", default=str(C.TA_BASES))
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    done = []
    for c in C.NOV14:
        for suf in (".pt", ".tag.json", ".pt.args.json"):
            src = Path(a.src) / f"stage1_{c}_s0{suf}"
            dst = C.LH / "bases" / src.name
            if not src.exists():
                raise SystemExit(f"no {src}")
            if dst.exists():
                if C.sha256(dst) != C.sha256(src):
                    raise SystemExit(f"{dst} exists and differs from {src}: never overwritten")
                continue
            if not a.dry_run:
                dst.parent.mkdir(parents=True, exist_ok=True)
                tmp = dst.with_name(dst.name + ".tmp")
                shutil.copyfile(src, tmp)
                assert C.sha256(tmp) == C.sha256(src)
                tmp.replace(dst)
            done.append(str(dst))
    print(
        ("would copy " if a.dry_run else "copied ")
        + (", ".join(done) or "nothing (all present and equal)")
    )
    return done


# ------------------------------------------------------------------------------------------------ stage-2 tasks
def say(msg: str) -> None:
    print(f"{AL.now()} {msg}", flush=True)


def run_step(
    name: str,
    argv: list[str],
    hours: float,
    env: dict[str, str] | None = None,
    log: Path | None = None,
) -> int:
    """One subprocess under `timeout --kill-after=300`; its output appended to `log` (default: ours)."""
    cmd = ["timeout", "--kill-after=300", str(int(hours * 3600)), *argv]
    say(f"{name}: start (cap {hours} h): {' '.join(argv[:4])} ...")
    t0 = time.time()
    e = {**os.environ, "TA": str(LH), "AT": str(LH), "LH": str(LH), **(env or {})}
    if log is None:
        p = subprocess.run(cmd, env=e, cwd=str(HERE), stdin=subprocess.DEVNULL)
    else:
        log.parent.mkdir(parents=True, exist_ok=True)
        with open(log, "a") as f:
            p = subprocess.run(
                cmd,
                env=e,
                cwd=str(HERE),
                stdin=subprocess.DEVNULL,
                stdout=f,
                stderr=subprocess.STDOUT,
            )
    say(f"{name}: rc {p.returncode} after {time.time() - t0:.0f} s")
    return p.returncode


def gpu_env(gpu: int, smoke: bool) -> dict[str, str]:
    return {"CUDA_VISIBLE_DEVICES": "" if smoke else str(gpu), "CUDA_DEVICE_ORDER": "PCI_BUS_ID", "PYTHONDONTWRITEBYTECODE": "1",
            "ND_OFFLINE": "1", "ND_REGISTRY_SYNC": "0",
            "LEAN_GATE_WORKERS": os.environ.get("LH_TASK_WORKERS") or os.environ.get("LEAN_GATE_WORKERS", "8"),
            "OMP_NUM_THREADS": os.environ.get("LH_TASK_OMP") or os.environ.get("OMP_NUM_THREADS", "8")}  # fmt: skip


def measure_argv(
    ck: Path,
    cid: str,
    out: Path,
    raw: Path,
    smoke: bool,
    tags: str = "A,B,none",
    sets: tuple[str, ...] = C.SETS,
) -> list[str]:
    sm = (
        ["--device", "cpu", "--limit", "2", "--k", "2", "--batch", "16"]
        if smoke
        else ["--device", "cuda", "--k", "8", "--batch", "256"]
    )
    return [PY, str(HERE / "lh_measure.py"), "run", "--ckpt", str(ck), "--ckpt_id", cid, "--tags", tags,
            "--sets", ",".join(sets), *sm, "--out_dir", str(out), "--raw_dir", str(raw), "--resume"]  # fmt: skip


def rm_accuracy(corpus: str, rater: str, seed: int = 0) -> float | None:
    """The reward model's held-out pair accuracy, from the fit's own record: the first JSON file in the model's folder
    with one of the keys below (a fraction or a percentage)."""
    d = C.rm_dir(corpus, rater, seed)
    keys = ("heldout_acc", "heldout_accuracy", "held_out_acc", "acc_heldout", "pair_acc_heldout")
    for f in sorted(d.glob("*.json")):
        if f.name == "lh_rm_check.json":
            continue
        with contextlib.suppress(ValueError, OSError):
            doc = json.loads(f.read_text())
            for k in keys:
                if isinstance(doc, dict) and isinstance(doc.get(k), (int, float)):
                    v = float(doc[k])
                    return v / 100 if v > 1 else v
    return None


def rm_main(argv: list[str]) -> int:
    """Fit the reward model of (corpus, rater) unless it exists; it must reach 0.90 held-out pair accuracy."""
    ap = argparse.ArgumentParser(prog="lh_launch.py _rm")
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--rater", required=True, choices=list(C.RATERS))
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument(
        "--fit_extra", default="", help="smoke only: arguments appended to lh_rm.py fit"
    )
    ap.add_argument("--min_acc", type=float, default=0.90)
    a = ap.parse_args(argv)
    rm = C.rm_path(a.corpus, a.rater)
    line = C.line_name(a.corpus, f"rlhf{a.rater}")

    def not_run(why: str, rc: int) -> int:
        """The pre-registered outcome "no reward model": the line is not run and never retried (lh_core.not_run_path)."""
        C.not_run_path(line).parent.mkdir(parents=True, exist_ok=True)
        if not C.not_run_path(line).exists():
            C.not_run_path(line).write_text(json.dumps({"task": line, "reason": why, "fit_rc": rc, "min_acc": a.min_acc,
                                                        "heldout_acc": rm_accuracy(a.corpus, a.rater), "utc": AL.now()}, indent=1))  # fmt: skip
        say(f"{line}: NOT RUN: {why}")
        return 3

    if C.not_run_path(line).exists():
        return not_run(
            json.loads(C.not_run_path(line).read_text()).get("reason", "decided before"), 3
        )
    if Path(f"{rm}.failed").exists() and not rm.exists():
        return not_run(f"the reward model missed {a.min_acc} held-out accuracy ({rm}.failed)", 3)
    if not rm.exists():
        rm.parent.mkdir(parents=True, exist_ok=True)
        rc = run_step(f"rm fit {a.corpus} {a.rater}", merge_argv(rm_argv(a.corpus, a.rater, smoke=a.smoke), a.fit_extra.split()),
                      HOURS["rm"], gpu_env(a.gpu, a.smoke), rm.parent / "fit.log")  # fmt: skip
        if not rm.exists() and (
            rc == 3 or Path(f"{rm}.failed").exists()
        ):  # lh_rm.py fit: under 0.90, exit 3
            return not_run(
                f"the reward model missed {a.min_acc} held-out accuracy (lh_rm.py fit rc {rc})", rc
            )
        if (
            rc != 0 or not rm.exists()
        ):  # a crash or a timeout: not a decision; the next driver run fits again
            say(f"rm {a.corpus} {a.rater}: no reward model (rc {rc}; {rm} {'exists' if rm.exists() else 'missing'}; "
                f"accuracy {rm_accuracy(a.corpus, a.rater)}): the rlhf line does not run now")  # fmt: skip
            return rc or 1
    acc = rm_accuracy(a.corpus, a.rater)
    ok = acc is not None and acc >= a.min_acc
    (rm.parent / "lh_rm_check.json").write_text(json.dumps({"rm": str(rm), "sha256": C.sha256(rm), "heldout_acc": acc,
                                                            "min_acc": a.min_acc, "ok": ok, "utc": AL.now()}, indent=1))  # fmt: skip
    say(
        f"rm {a.corpus} {a.rater}: held-out accuracy {acc} (required {a.min_acc}): {'ok' if ok else 'REFUSED'}"
    )
    return (
        0
        if ok
        else not_run(
            f"the reward model's held-out accuracy {acc} is under {a.min_acc} (the launcher's check)",
            4,
        )
    )


def final_step(d: Path) -> int:
    with contextlib.suppress(OSError, ValueError, KeyError, TypeError):
        return int(json.loads((d / "done.json").read_text())["final"]["step"])
    return C.OFFLINE_MAX_STEPS


def snapshots_of(arm: str) -> list[dict[str, Any]]:
    """The snapshots an offline run wrote, by step: snap_step<k>.pt (label step<k>) and final.pt (label final, at the
    run's last step from done.json, unless a snap_step file has that step)."""
    d = C.arm_dir(arm)
    out: dict[int, dict[str, Any]] = {}
    for p in d.glob("snap_step*.pt"):
        m = re.fullmatch(r"snap_step(\d+)", p.stem)
        if m:
            out[int(m.group(1))] = {
                "label": f"step{int(m.group(1))}",
                "step": int(m.group(1)),
                "path": p,
            }
    if (d / "final.pt").exists():
        st = final_step(d)
        out.setdefault(st, {"label": "final", "step": st, "path": d / "final.pt"})
    return [out[k] for k in sorted(out)]


def probe_argv(arm: str, snaps: list[dict[str, Any]], smoke: bool) -> list[str]:
    """lh_measure.py probe of a rated run's snapshots in step order; it stops past the base's share + 30 of the rated
    habit (always probing the last snapshot); without the base's share every snapshot is probed."""
    t = C.parse_task(arm)
    habit = C.RATERS[t["rater"]]
    base = base_share(t["corpus"], habit)
    stop = (
        ["--stop_habit", habit, "--stop_above", str(base + C.TARGET_MOVE)]
        if base is not None
        else []
    )
    sm = (
        ["--device", "cpu", "--limit", "2", "--k", "2", "--batch", "16"]
        if smoke
        else ["--device", "cuda", "--k", "8", "--batch", "256"]
    )
    return [PY, str(HERE / "lh_measure.py"), "probe", "--ckpts", ",".join(str(x["path"]) for x in snaps),
            "--ids", ",".join(x["label"] for x in snaps), "--id_prefix", f"{arm}@probe@", "--out_root", str(C.arm_dir(arm) / "probe"),
            "--raw_root", str(LH / "raw" / arm / "probe"), *stop, *sm]  # fmt: skip


def step0_problem(arm: str) -> str | None:
    """PREREG Amendment 7 point 7 (audit T11): the first logged training row of a DPO run (policy = reference) must
    show loss ln 2 = 0.6931 within 0.001. lh_dpo checks it itself (`step0_ok` on that row, its own step0_failed.json,
    exit 4); this reads the trainer's verdict, and for a trainer without it the row's loss. -> the reason, or None."""
    d = C.arm_dir(arm)
    try:
        first = json.loads((d / "train.jsonl").read_text().splitlines()[0])
    except (OSError, IndexError, ValueError):
        first = None
    if first is not None and "step0_ok" in first:
        return None if first["step0_ok"] else (
            f"the trainer's step-0 check failed: first row (step {first.get('step')}) loss {first.get('step0_loss')}, "
            f"want ln 2 = 0.6931 within {C.STEP0_TOL} (lh_dpo's step0_failed.json)")  # fmt: skip
    if (d / "step0_failed.json").exists():
        return f"the trainer's step-0 check failed ({d / 'step0_failed.json'})"
    if first is None:
        return None  # no row logged: an ordinary training failure (the caller reports rc and done.json)
    loss = first.get("loss")
    if not isinstance(loss, (int, float)) or abs(loss - C.STEP0_LOSS) > C.STEP0_TOL:
        return (f"first training row (step {first.get('step')}) loss {loss}, want {C.STEP0_LOSS} within "
                f"{C.STEP0_TOL}; train_margin_rel {first.get('train_margin_rel')}")  # fmt: skip
    return None


def offline_main(argv: list[str]) -> int:
    """An offline run: the trainer on the step grid (unless its done.json exists); a rated run (dpo, sft) then probes
    its snapshots in step order until one is past the base + 30, and the last (lh_measure.py probe, one process) ->
    LH/arms/<arm>/probe/probe.json. A no-preference run is not probed (PREREG Amendment 4, point 1)."""
    ap = argparse.ArgumentParser(prog="lh_launch.py _offline")
    ap.add_argument("--arm", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument(
        "--trainer_extra", default="", help="smoke only: arguments appended to the trainer"
    )
    a = ap.parse_args(argv)
    t = C.parse_task(a.arm)
    assert t["type"] == "offline", a.arm
    d = C.arm_dir(a.arm)
    rec: dict[str, Any] = {
        "arm": a.arm,
        **t,
        "trainer": trainer_argv(a.arm, a.smoke),
        "start": AL.now(),
    }
    if (d / "done.json").exists():
        say(f"{a.arm}: {d / 'done.json'} exists: training skipped")
        rec["trained"] = "existing"
    elif d.exists() and any(p.name not in ("lh_offline.json", "train.log") for p in d.iterdir()):
        say(f"{a.arm}: {d} holds a partial run (no done.json): refused, it needs a decision")
        return 3
    else:
        d.mkdir(parents=True, exist_ok=True)
        rc = run_step(f"{a.arm} train", merge_argv(trainer_argv(a.arm, a.smoke), shlex.split(a.trainer_extra)), HOURS["offline_train"],
                      gpu_env(a.gpu, a.smoke), d / "train.log")  # fmt: skip
        rec["train_rc"] = rc
        bad = step0_problem(a.arm) if t["kind"] in ("dpo", "dposwap") else None
        if bad:  # Amendment 7 point 7: the run is stopped and reported, never probed or measured
            rec.update(step0=bad, end=AL.now())
            if not (d / "step0_failed.json").exists():  # lh_dpo writes its own; never overwritten
                (d / "step0_failed.json").write_text(
                    json.dumps(
                        {
                            "arm": a.arm,
                            "reason": bad,
                            "train_rc": rc,
                            "utc": AL.now(),
                            "by": "lh_launch",
                        },
                        indent=1,
                    )
                )
            (d / "lh_offline.json").write_text(json.dumps(rec, indent=1))
            C.POD.mkdir(parents=True, exist_ok=True)
            with open(C.POD / f"{C.split_base(t['corpus'])[0]}.FAILED", "a") as fh:
                fh.write(f"{AL.now()} stage2 {a.arm} STOPPED: step-0 check: {bad}\n")
            say(f"{a.arm}: STOPPED: step-0 check: {bad}")
            return 5
        if rc != 0 or not (d / "done.json").exists():
            rec["end"] = AL.now()
            (d / "lh_offline.json").write_text(json.dumps(rec, indent=1))
            say(
                f"{a.arm}: training failed (rc {rc}, done.json {'exists' if (d / 'done.json').exists() else 'missing'})"
            )
            return rc or 1
    snaps = snapshots_of(a.arm)
    rec["snapshots"] = [{"label": x["label"], "step": x["step"]} for x in snaps]
    missing = sorted(set(C.SNAP_STEPS) - {x["step"] for x in snaps})
    if missing:
        rec["grid_missing"] = missing
        say(f"{a.arm}: no snapshot at steps {missing} of the grid")
    rc = 0
    if not snaps:
        rc = 1
    elif not C.is_rated(t["kind"]):
        rec["probe"] = (
            "not probed (a no-preference run: measured at its rated run's steps; PREREG Amendment 4, point 1)"
        )
    elif not (d / "probe" / "probe.json").exists():
        (d / "probe").mkdir(parents=True, exist_ok=True)
        rc = run_step(f"{a.arm} probe", probe_argv(a.arm, snaps, a.smoke), HOURS["probe"],
                      {**gpu_env(a.gpu, a.smoke), **({} if a.smoke else {"DANREAD_MEM_FRAC": "0.3"})}, d / "probe" / "probe.log")  # fmt: skip
    rec["probe_rc"], rec["end"] = rc, AL.now()
    (d / "lh_offline.json").write_text(json.dumps(rec, indent=1))
    return rc


# ------------------------------------------------------------------------------------------------ full measurement
def base_summary(corpus: str) -> Path:
    """The base's untagged marker_plain summary, for a corpus field (<c>: <c>_s0; <c>b1: <c>_s1)."""
    return LH / "measure" / C.base_id(*C.split_base(corpus)) / "marker_plain__none.summary.json"


def base_share(corpus: str, habit: str) -> float | None:
    """The base's untagged A-share of `habit` on marker_plain (its full measurement; the probe's set and tag)."""
    with contextlib.suppress(OSError, ValueError, KeyError, TypeError):
        v = json.loads(base_summary(corpus).read_text())["lh_totals"]["a_share"].get(habit)
        return None if v is None else float(v)
    return None


def full_path(task: str) -> Path:
    return LH / "arms" / f"{task}.json"


def select_steps(base: float, probe: list[tuple[int, float | None]], last: int, target: float = C.TARGET_MOVE,
                 steps: list[int] | None = None) -> dict[str, Any]:  # fmt: skip
    """PREREG Amendments 3 and 4: the first probed snapshot more than `target` points over the base and the probed one
    before it (the base itself, step 0, below the first snapshot), plus the last snapshot. Not reached: the bracket is
    the run's last two snapshots (`steps`, every snapshot step; default the probed ones)."""
    moves = [(s, round(v - base, 2)) for s, v in sorted(probe) if v is not None]
    prev = (0, 0.0)
    for s, m in moves:
        if m > target:
            return {"reached": True, "bracket": [prev[0], s], "bracket_moves": [prev[1], m], "moves": moves,
                    "measure": sorted({x for x in (prev[0], s, last) if x > 0})}  # fmt: skip
        prev = (s, m)
    allst = sorted(steps if steps else {s for s, _ in probe} | {last})
    br = allst[-2:] if len(allst) >= 2 else allst
    return {"reached": False, "bracket": br, "moves": moves, "max_move": max((m for _, m in moves), default=None),
            "measure": sorted({*br, last})}  # fmt: skip


def full_main(argv: list[str]) -> int:
    """A full-measurement task (full_<dpo|sft>_<C>_<R>_s<S>): the rated run's probe picks the steps (select_steps);
    the rated run and its no-preference run are measured there in full (lh_measure.py run: tags A, B, none, both sets)
    into LH/arms/<arm>/measure/<label>/; the choice and what ran -> LH/arms/<task>.json."""
    ap = argparse.ArgumentParser(prog="lh_launch.py _full")
    ap.add_argument("--task", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args(argv)
    t = C.parse_task(a.task)
    assert t["type"] == "full", a.task
    habit = C.RATERS[t["rater"]]
    rated = C.arm_name(t["kind"], t["corpus"], t["rater"], t["seed"])
    ctl = C.arm_name(C.CONTROL_OF[t["kind"]], t["corpus"], t["rater"], t["seed"])
    rec: dict[str, Any] = {
        "task": a.task,
        "rated": rated,
        "control": ctl,
        "habit": habit,
        "target_move": C.TARGET_MOVE,
        "start": AL.now(),
    }
    base = base_share(t["corpus"], habit)
    pj = C.arm_dir(rated) / "probe" / "probe.json"
    if base is None or not pj.exists():
        say(
            f"{a.task}: refused: base share {base} ({base_summary(t['corpus'])}), probe {pj} {'exists' if pj.exists() else 'missing'}"
        )
        return 3
    probe = json.loads(pj.read_text())["snapshots"]
    snaps = {x["step"]: x for x in snapshots_of(rated)}
    pts = [
        (st, (probe.get(x["label"]) or {}).get("a_share", {}).get(habit)) for st, x in snaps.items()
    ]
    sel = select_steps(base, pts, max(snaps), steps=sorted(snaps))
    rec.update(base_share=base, **sel, measured={}, tags={})
    bad = 0
    for arm in (rated, ctl):
        by = {x["step"]: x for x in snapshots_of(arm)}
        for st in sel["measure"]:
            x = by.get(st)
            if x is None:
                rec["measured"][f"{arm}@{st}"] = "no snapshot at this step"
                bad += 1
                continue
            md = C.arm_dir(arm) / "measure" / x["label"]
            if (md / "DONE").exists():
                rec["measured"][f"{arm}@{x['label']}"] = "measured before"
                continue
            md.mkdir(parents=True, exist_ok=True)
            env = {**gpu_env(a.gpu, a.smoke), "LEAN_GATE_LOG": str(md / "lean_gate.jsonl"),
                   **({} if a.smoke else {"DANREAD_MEM_FRAC": "0.3"})}  # fmt: skip
            # untagged (Amendment 4, point 2), but all three tags on the UPPER bracketing snapshot of the seed-0 reit
            # DPO run (Amendment 7 point 2); marker_plain only (Amendment 8 point 6)
            tagged = (
                arm == rated
                and st == sel["bracket"][-1]
                and (t["kind"], t["rater"], t["seed"]) == C.TAGGED_SNAPSHOT
                and not C.split_base(t["corpus"])[
                    1
                ]  # the second base: untagged everywhere (Amendment 8 point 2)
            )
            tags = "A,B,none" if tagged else C.SNAP_TAGS
            rec["tags"][f"{arm}@{x['label']}"] = tags
            rc = run_step(f"{arm} measure {x['label']}", measure_argv(x["path"], f"{arm}@{x['label']}", md, LH / "raw" / arm / x["label"], a.smoke, tags, C.TUNED_SETS),
                          HOURS["snap_measure"], env, md / "measure.log")  # fmt: skip
            rec["measured"][f"{arm}@{x['label']}"] = f"rc {rc}"
            if rc == 0:
                (md / "DONE").write_text(AL.now())
            else:
                bad += 1
    rec["end"], rec["done"] = AL.now(), not bad
    full_path(a.task).parent.mkdir(parents=True, exist_ok=True)
    full_path(a.task).write_text(json.dumps(rec, indent=1))
    where = (
        f"+{C.TARGET_MOVE:g} reached between steps {sel['bracket']}"
        if sel["reached"]
        else "target not reached"
    )
    say(f"{a.task}: base {base}, {where}; "
        f"measured steps {sel['measure']} of {rated} and {ctl} ({bad} failed)")  # fmt: skip
    return 0 if not bad else 1


def rm_probe_main(corpus: str, rater: str, gpu: int, smoke: bool) -> int:
    """The reward model's probe on held-out twins of the other flippable habits (lh_rm.py probe), stored beside the
    model. It does not gate the line: a failure is logged and the line runs."""
    out, tw = C.rm_probe_path(corpus, rater), pair_files(corpus, rater)["probe_twins"]
    if out.exists():
        return 0
    if not tw.exists() or not tw.read_text().strip():
        say(f"rm probe {corpus} {rater}: no probe twins ({tw}): not probed")
        return 1
    rc = run_step(f"rm probe {corpus} {rater}", rm_probe_argv(corpus, rater, smoke=smoke), HOURS["rm_probe"], gpu_env(gpu, smoke),
                  out.parent / "probe.log")  # fmt: skip
    if rc != 0 or not out.exists():
        say(f"rm probe {corpus} {rater}: failed (rc {rc}); the line runs without it")
    return rc


def task_main(argv: list[str]) -> int:
    """One stage-2 task, its steps in order (module docstring)."""
    ap = argparse.ArgumentParser(prog="lh_launch.py _task")
    ap.add_argument("--task", required=True)
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--extra", default="", help="smoke only: arguments appended to `_line`")
    ap.add_argument(
        "--trainer_extra", default="", help="smoke only: arguments appended to an offline trainer"
    )
    ap.add_argument(
        "--fit_extra", default="", help="smoke only: arguments appended to lh_rm.py fit"
    )
    ap.add_argument(
        "--rm_min_acc",
        type=float,
        default=0.90,
        help="the reward model's required held-out accuracy",
    )
    a = ap.parse_args(argv)
    t = C.parse_task(a.task)
    sm = ["--smoke"] if a.smoke else []
    if t["type"] == "full":
        return full_main(["--task", a.task, "--gpu", str(a.gpu), *sm])
    if t["type"] == "offline":
        return offline_main(
            ["--arm", a.task, "--gpu", str(a.gpu), *sm, "--trainer_extra", a.trainer_extra]
        )
    if (
        t["rater"] is not None and t["seed"] == 0
    ):  # an RL seed > 0 uses the seed-0 fit as it is (Amendment 10)
        rc = rm_main(["--corpus", t["corpus"], "--rater", t["rater"], "--gpu", str(a.gpu), *sm, "--fit_extra", a.fit_extra,
                      "--min_acc", str(a.rm_min_acc)])  # fmt: skip
        if C.not_run_path(a.task).exists():
            return 0  # decided: not run (rm_main said why); listed by the driver as not run, not as a failure
        if rc != 0:
            return rc
        rm_probe_main(t["corpus"], t["rater"], a.gpu, a.smoke)
    _, (st,), problems, _ = plan(rl_ns(argparse.Namespace(gpu=a.gpu), a.task))
    if problems:
        say(f"{a.task}: refused: {problems}")
        return 3
    return run_step(
        f"{a.task} line",
        [*st["argv"], *a.extra.split()],
        st["timeout_s"] / 3600,
        {"ND_GUARD": "off"},
    )


def task_cmd(task: str, gpu: int, extra: str = "") -> list[str]:
    """One task's process: `_task` under its own `timeout` (the task's cap)."""
    return ["timeout", "--kill-after=300", str(int(task_hours(task) * 3600)), PY, str(HERE / "lh_launch.py"), "_task",
            "--task", task, "--gpu", str(gpu), *shlex.split(extra)]  # fmt: skip


def task_weight(task: str, parallel: int) -> int:
    """Task slots a task takes: a GRPO line all of them (lh_core.LINE_ALONE), anything else one."""
    return max(1, parallel) if C.LINE_ALONE and C.parse_task(task)["type"] == "line" else 1


def cpu_quota() -> int:
    """CPUs this process may use: the affinity mask cut to the cgroup quota (Dan's lean_gate.cpu_quota: RunPod pods
    show 96 cores but grant about 7.6)."""
    n = len(os.sched_getaffinity(0))
    for fq, fp in (
        ("/sys/fs/cgroup/cpu.max", None),
        ("/sys/fs/cgroup/cpu/cpu.cfs_quota_us", "/sys/fs/cgroup/cpu/cpu.cfs_period_us"),
    ):
        try:
            q, per = (
                Path(fq).read_text().split()[:2]
                if fp is None
                else (Path(fq).read_text().strip(), Path(fp).read_text().strip())
            )
            if q not in ("max", "-1"):
                n = min(n, max(1, int(int(q) / int(per))))
            break
        except (OSError, ValueError):
            continue
    return n


def task_threads(parallel: int, environ: dict[str, str] | None = None) -> dict[str, str]:
    """Lean workers and OMP threads per task when `parallel` tasks share the pod: the CPU quota split between them (at
    least 2 each), never above what the environment sets (W2's job wrapper: 8); LH_TASK_THREADS replaces the split.
    W2's job wrapper and tag_line.Seg.env set 8 for every process, so the values travel as LH_TASK_WORKERS and
    LH_TASK_OMP too (gpu_env and LHSeg.env apply them)."""
    env = os.environ if environ is None else environ
    n = int(env.get("LH_TASK_THREADS") or max(2, cpu_quota() // max(1, parallel)))

    def cap(k: str) -> str:
        v = env.get(k, "")
        return str(min(n, int(v))) if v.isdigit() and int(v) > 0 else str(n)

    w, o = cap("LEAN_GATE_WORKERS"), cap("OMP_NUM_THREADS")
    return {"LEAN_GATE_WORKERS": w, "OMP_NUM_THREADS": o, "LH_TASK_WORKERS": w, "LH_TASK_OMP": o}


def tasks_main(argv: list[str]) -> int:
    """The `stage2` job's step: run the tasks `--parallel` at a time, each in its own `_task` process; a task waits for
    the task it depends on (lh_core.depends_on) to end."""
    ap = argparse.ArgumentParser(prog="lh_launch.py _tasks")
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--parallel", type=int, default=2)
    ap.add_argument("--stagger_s", type=float, default=120.0)
    ap.add_argument("--poll_s", type=float, default=5.0)
    ap.add_argument(
        "--task_extra", default="", help="smoke only: arguments appended to every `_task`"
    )
    a = ap.parse_args(argv)
    tasks = a.tasks.split(",")
    thr = task_threads(a.parallel)
    say(
        f"per task: LEAN_GATE_WORKERS {thr['LEAN_GATE_WORKERS']}, OMP_NUM_THREADS {thr['OMP_NUM_THREADS']} (CPU quota {cpu_quota()}, {a.parallel} at a time)"
    )
    rcs: dict[str, int] = {}
    cmds: dict[str, list[str]] = {}
    for x in tasks:
        p = [q for q in task_problems(x) if ": already " not in q]
        if p:
            say(f"{x}: refused: {p}")
            rcs[x] = 3
            continue
        cmds[x] = task_cmd(x, a.gpu, a.task_extra)
    queue = [x for x in tasks if x in cmds]
    running: dict[str, tuple[subprocess.Popen, Any, float]] = {}
    stop: dict[str, Any] = {"sig": None}

    def on_term(signum, frame) -> None:
        stop["sig"] = signum
        for (
            p,
            _,
            _,
        ) in running.values():  # the task's `timeout` relays it to `_task` and its children
            with contextlib.suppress(ProcessLookupError):
                p.send_signal(signal.SIGTERM)

    def ready(x: str) -> bool:
        return all(d not in tasks or d in rcs for d in C.depends_on(x))

    signal.signal(signal.SIGTERM, on_term)
    signal.signal(signal.SIGINT, on_term)
    last = 0.0
    while queue or running:
        for x, (p, fh, t0) in list(running.items()):
            rc = p.poll()
            if rc is not None:
                fh.close()
                rcs[x] = rc
                del running[x]
                say(f"{x}: rc {rc} after {time.time() - t0:.0f} s")
        if stop["sig"] is not None:
            queue.clear()
        while sum(task_weight(y, a.parallel) for y in running) < max(1, a.parallel) and (
            not running or time.time() - last >= a.stagger_s
        ):
            used = sum(task_weight(y, a.parallel) for y in running)
            x = next(
                (
                    y
                    for y in queue
                    if ready(y) and used + task_weight(y, a.parallel) <= max(1, a.parallel)
                ),
                None,
            )
            if x is None:
                break
            queue.remove(x)
            (LH / "logs").mkdir(parents=True, exist_ok=True)
            fh = open(LH / "logs" / f"{x}.tasks.out", "a")  # noqa: SIM115
            p = subprocess.Popen(cmds[x], stdout=fh, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, cwd=str(HERE),
                                 env={**os.environ, **task_threads(max(1, a.parallel // task_weight(x, a.parallel))), "ND_GUARD": "off"})  # fmt: skip
            running[x] = (p, fh, time.time())
            last = time.time()
            say(f"{x}: started PID {p.pid} ({len(running)} running)")
        if queue and not running and not any(ready(y) for y in queue):
            for y in queue:
                rcs[y] = 3
                say(f"{y}: its prerequisite never ended: not run")
            queue.clear()
        time.sleep(a.poll_s)
    say(f"tasks done: {rcs}")
    return 0 if rcs and all(v == 0 for v in rcs.values()) and stop["sig"] is None else 1


# ------------------------------------------------------------------------------------------------ wait
def newest_log(job: str, logs: Path | None = None) -> Path | None:
    logs = logs or (LH / "logs")
    c = sorted(
        p
        for p in logs.glob(f"{job}_2*.log")
        if re.fullmatch(rf"{re.escape(job)}_\d{{8}}T\d{{6}}Z\.log", p.name)
    )
    return c[-1] if c else None


def wait_main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="lh_launch.py wait")
    ap.add_argument("--job", required=True)
    ap.add_argument("--max_h", type=float, default=6.0)
    ap.add_argument("--every", type=float, default=60.0)
    ap.add_argument("--no_log_s", type=float, default=600.0)
    ap.add_argument("--heartbeat", default=None)
    ap.add_argument("--stage", default=None)
    a = ap.parse_args(argv)
    t0 = time.time()
    hb = Path(a.heartbeat) if a.heartbeat else None
    while True:
        log = newest_log(a.job)
        end = None
        last = ""
        if log is not None:
            lines = log.read_text(errors="replace").splitlines()
            last = lines[-1][:300] if lines else ""
            end = next(
                (x for x in reversed(lines) if re.search(r"(^|\s)job end rc -?\d+", x)), None
            )
        if hb is not None:
            hb.parent.mkdir(parents=True, exist_ok=True)
            tmp = hb.with_name(hb.name + ".tmp")
            tmp.write_text(json.dumps({"utc": AL.now(), "host": AL.HOST, "stage": a.stage, "job": a.job, "log": str(log) if log else None,
                                       "waited_s": round(time.time() - t0), "last_line": last, "ended": end}, indent=1))  # fmt: skip
            tmp.replace(hb)
        if end is not None:
            m = re.search(r"job end rc (-?\d+)", end)
            rc = int(m.group(1)) if m else 1
            print(f"{AL.now()} {a.job}: {end.strip()} ({log})", flush=True)
            return rc
        if log is None and time.time() - t0 > a.no_log_s:
            print(
                f"{AL.now()} {a.job}: no log in {LH / 'logs'} after {a.no_log_s:.0f} s", flush=True
            )
            return 2
        if time.time() - t0 > a.max_h * 3600:
            print(f"{AL.now()} {a.job}: not ended after {a.max_h} h ({log})", flush=True)
            return 124
        time.sleep(a.every)


# ------------------------------------------------------------------------------------------------ status
def status() -> None:
    AL.status()
    for p in sorted(C.POD.glob("*")) if C.POD.exists() else []:
        if p.suffix in (".log",):
            continue
        if (
            p.is_dir()
        ):  # the driver's lock (`<c>.driver.lock` is a directory): name it, do not read it
            print(f"pod dir {p.name}/ ({len(list(p.iterdir()))} entries)")
            continue
        try:
            print(f"pod file {p.name}: {p.read_text(errors='replace').strip()[:300]}")
        except OSError as e:
            print(f"pod file {p.name}: unreadable ({e.__class__.__name__})")


# ------------------------------------------------------------------------------------------------ main
ENTRY = {"_pretrain": pretrain_main, "_line": line_main, "_tasks": tasks_main, "_task": task_main, "_rm": rm_main,
         "_offline": offline_main, "_full": full_main, "wait": wait_main, "steps": steps_main, "nov14": nov14_main}  # fmt: skip


def main() -> None:
    if len(sys.argv) >= 2 and sys.argv[1] == "_job":
        sys.exit(TL.run_job(Path(sys.argv[2])))
    if len(sys.argv) >= 2 and sys.argv[1] in ENTRY:
        r = ENTRY[sys.argv[1]](sys.argv[2:])
        sys.exit(
            r
            if isinstance(r, int)
            and sys.argv[1] in ("_tasks", "_task", "_rm", "_offline", "_full", "wait")
            else 0
        )
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("cmd", choices=("pretrain", "measure", "prep2", "stage2", "rl", "status"))
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--corpus", default=None, choices=[*C.CORPORA, *C.EXTRA_BASES, *C.NOV14])
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument(
        "--base", default=None, help="measure: <corpus>_s<seed>, authors_half_s0 or control_half_s0"
    )
    ap.add_argument("--tags", default="A,B,none")
    ap.add_argument("--sets", default=",".join(C.SETS))
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--line", default=None, help="rl: " + ", ".join(C.all_lines()))
    ap.add_argument(
        "--tasks",
        default=None,
        help="stage2: a comma list of the corpus's tasks (default: all of them)",
    )
    ap.add_argument("--parallel", type=int, default=2, help="stage2: tasks at once on the GPU")
    ap.add_argument("--rl_seed", type=int, default=0,
                    help="stage2: 1 = c24's GRPO lines with RL seed 1 from the first base (PREREG Amendment 10)")  # fmt: skip
    ap.add_argument("--base_seed", type=int, default=0,
                    help="prep2, stage2: 1 = the second base pass (PREREG Amendment 8 point 2; c24 and c4)")  # fmt: skip
    ap.add_argument(
        "--stagger_s", type=float, default=120.0, help="stage2: seconds between two task starts"
    )
    ap.add_argument("--seg", type=int, default=1)
    ap.add_argument(
        "--resume", default=None, help="rl: the rolling checkpoint a later segment resumes from"
    )
    ap.add_argument(
        "--skip_done", action="store_true", help="measure: skip what is already measured"
    )
    ap.add_argument("--snap_k", type=int, default=8)
    ap.add_argument("--snap_sets", default=",".join(C.SETS))
    ap.add_argument("--stop_on_held", action="store_true")
    ap.add_argument("--lab", action="store_true", help="run on a lab host under the lab rules")
    ap.add_argument("--allow_unreadable", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if a.cmd == "status":
        status()
        return
    if a.cmd == "pretrain" and (a.corpus is None or a.seed is None):
        ap.error("pretrain needs --corpus and --seed")
    if a.cmd == "measure" and a.base is None:
        ap.error("measure needs --base")
    if a.cmd == "rl" and a.line is None:
        ap.error("rl needs --line")
    if a.cmd in ("prep2", "stage2") and a.corpus not in C.CORPORA:
        ap.error(f"{a.cmd} needs --corpus in {C.CORPORA}")
    AL.launch(a)


if __name__ == "__main__":
    main()
