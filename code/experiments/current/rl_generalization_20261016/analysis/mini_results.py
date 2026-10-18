"""The mini-run validation gate (PREREGISTRATION.md § 3, amended in § 6), the mini tables and the
full-scale g64 replicate.

    cd /scratch/cowifornia/nd-rl/think/nd-rl && ND_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 \
        OMP_NUM_THREADS=2 PYTHONPATH=. nice code/.venv/bin/python \
        code/experiments/current/rl_generalization_20261016/analysis/mini_results.py \
        [--regime mini|mini20|both] [--draws 4000]

Writes, next to `analysis/` (generated; do not edit), per regime (`--regime`, default both; a
regime's gate and tables use its own runs only, never pooled with the other's):
- mini (10 + 10 iterations, `grpo_think_vs_mini*`, the first gate's record): `results/gate.md`,
  `results/mini_table.md`;
- mini20 (20 + 20, `grpo_think_vs_mini20*`, remedy A's rerun of the validation wave):
  `results/gate_mini20.md`, `results/mini20_table.md`;
and always `results/g64_replicate.md`: the full-scale g64 replicate, kg64 − merged recipe at the
replicate seed (1, or 2 in its place once 1 has failed: § 4.6) next to seed 0 (both L3 at KL 0.02
from grpo_think_vs's L2).
Rerunnable as reads land: a read that is not there yet is listed, never fatal, and a gate input
that is missing makes its check PENDING. A read counts once its `summary.json` is written (the eval
writes it after the rows). CPU only, one process, about 30 s a regime at 4,000 draws.

Inputs (read only):
- Reads, `runs/eval/v2/<set>_<arm>_s<seed>_plain_k<k>/rows_plain_B64.jsonl`: MINI_READS of each
  run's L3-mini last checkpoint (l2test, l3test, t1320, t2130 at k8, charles173 at k64) and, for a
  mini20 reference, MINI_TRAIN (l2train, l3train at k8); the base S1g3_s0 as `s1g3`
  (`s1g3_ckreads`, every mini run's init); the reference panels `<ref>_<STAGE>it<IT>` (MINI_ROLL;
  MINI_PANEL: l3test k8, t1320 k8, t2130 k4, charles173 k16); for G3, the full-scale ckreads of
  `grpo_think_vs_mb_kkl0`; for the replicate, `grpo_think_vs_mb[_kg64]` (t1320 k16, t2130 k8,
  charles173 k64). Loaded with v2_results' Loader (the mini arms and (set, k) specs are added to
  its registries) and its subsets.
- mini20 only, the B0 reads (§ 6, 2026-10-18, ba275d73 and 963d90b3; `reads_b0_l3test.sh`: line.sh's
  l3test k8 at `--budgets 0`), `runs/eval/v2/l3test_<name>_s<seed>_plain_k8_B0/rows_plain_B0.jsonl`:
  the references, the base, `grpo_think_vs_mb[_kkl0]` s0 (full scale, context only), every wave-2
  arm and trajectory point (`<ref>_<STAGE>it<IT>`) found.
- Training, `runs/grpo_v2/<arm>_s<seed>_<stage>/`: metrics.jsonl (a stage that crashed and was
  rerun keeps only the rerun's rows, § 6 2026-10-17), ckpts/ and args.json; the line log
  `runs/logs/v2_<arm>_s<seed>.log` (the exit, ALARM and FAILED lines after the stage's last
  launch; `line done`); for the replicate, the kg64 L3 runs' last metrics row (`stop`) and
  `_it39.pt`.

Sets: L2 / L3 test_1k (1,000 each, in distribution), 13–20 (the 986 valid items of t1320_n1000),
16–20 (its 616 with min_lines >= 16), 21–30 (t2130_n1000), Charles149 (the clean 149 of
charles173). pass@1 / pass@5 are the unbiased estimators from the read's k attempts, in % of the
set; per-run intervals are 95% problem bootstraps, pooled ones hierarchical (seeds, then problems),
differences paired on problems (`code.thinking.stats`).

The gate (§ 3), mechanically. x_r = logit of run r's set-mean pass, p clipped to [0.005, 0.995].
A control c with n_c seeds against the n_r = 4 references: Δ = mean_c x − mean_ref x on 13–20
pass@1; s_ref = SD of the references' x (df n_r − 1); SE = s_ref √(1/n_c + 1/n_r);
U = Δ + t(0.95, n_r − 1) SE. "reproduces" iff Δ <= E/2, U < 0 and the Δ point estimates (same
statistic) on 16–20 pass@1 and 21–30 pass@5 are both < 0; "partial" if Δ < 0 otherwise; "fails" if
Δ >= 0. E: kl1 −0.19, g64 −0.255 (§ 3, § 5). G0: all 4 references end with their L3-mini last
checkpoint and every read; at most 1 rolled back; their mean 13–20 pass@1 in [12%, 60%] and >= the
base + 5 points (7.3% without a base read); their mean L3 test_1k pass@1 <= 90% (mini20: § 6's
deviation decides this on their mean B0 L3 test pass@1, the clause's named fix; the B64 value is
reported only). G1: s_ref <= 0.11 logit. Then § 3's decision table (mini20: as amended in § 6, a
branch that would call for remedy A again retires the mini). A control seed without its final
checkpoint (a FAILED line) is excluded from n_c and replaced by the next unused seed whose
reference L2-mini exists (§ 4.6; § 6, 963d90b3). Also § 4.3's competence-versus-generalization
residuals (they need the reference trajectories and say so while those are missing) and G3
(informational).
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np

from code.experiments.current.thinking_traces_20261005.matched_eval import v2_results as v2
from code.thinking import stats

R = Path("/scratch/cowifornia/nd-rl/think/runs")
GV, LOGS = R / "grpo_v2", R / "logs"
STUDY = Path(__file__).resolve().parent.parent
REPO = STUDY.parents[3]
RERUN = (
    "cd /scratch/cowifornia/nd-rl/think/nd-rl && ND_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 "
    "OMP_NUM_THREADS=2 PYTHONPATH=. nice code/.venv/bin/python "
    "code/experiments/current/rl_generalization_20261016/analysis/mini_results.py"
)
HEADER = "Generated by analysis/mini_results.py — do not edit."

REF_SEEDS = (0, 1, 2, 3)  # the gate's references (§ 2); a later seed is a drift check only (§ 6)
BASE = "s1g3"  # Stage-1 S1g3_s0, every mini run's init (s1g3_ckreads)
BASE_FALLBACK = 0.073  # § 3: serrano's 2-seed S1g3 13–20 pass@1, only without a base read
CLIP = (0.005, 0.995)
G0_RANGE, G0_ABOVE_BASE, G0_ID_MAX, G1_MAX = (0.12, 0.60), 0.05, 0.90, 0.11
LIVE = 1800  # s: a stage whose log changed within this is training; older, with no exit: stopped
TRAIN_KNOBS = ("", "rr", "pool250", "pool500")  # line.sh: the arms that also read MINI_TRAIN
# § 6 (2026-10-18, ba275d73; completed in 963d90b3), mini20: G0's ceiling check on B0 L3 test
G0_B0 = (
    "reference mean B0 L3 test pass@1 ≤ 90% (§ 6 deviation, 2026-10-18, ba275d73: the clause's "
    "named fix)"
)
B0_MET = "MET BY THE CLAUSE'S FIX (B0)"  # never "PASS" (§ 6, 963d90b3)
G0_DECIDES = {"PASS", "FAIL", "PENDING", B0_MET}  # a G0 row whose result is not one never gates
B0_FULL = (("grpo_think_vs_mb", 0), ("grpo_think_vs_mb_kkl0", 0))  # full-scale B0 reads: context
RETIRED = (  # § 6 (963d90b3): where § 3 would call for remedy A again
    "NO-GO: the mini is retired (remedy A is spent); remedy C is not started overnight; "
    "Leon decides"
)


@dataclass(frozen=True)
class Control:
    name: str
    arm: str
    E: float  # the expected 13–20 pass@1 shift in logit (§ 3, § 5)
    seeds: tuple[int, ...] = (0, 1)  # the first round (§ 2)
    more: tuple[int, ...] = (2, 3)  # added when the first round is "partial" (§ 3's table)


@dataclass(frozen=True)
class Regime:
    """One gate configuration (§ 3: the gate counts only for the configuration it ran on)."""

    key: str
    ref: str  # the reference arm; its controls are ref_kkl1 and ref_kg64 (L3-mini forks)
    iters: int  # each stage's iterations (line.sh MINI_ITERS / MINI20_ITERS)
    title: str
    gate: str  # output files in results/
    table: str
    roll: tuple[str, ...]  # the planned MINI_ROLL panels (<STAGE>it<IT>) on ROLL_SEEDS
    train_reads: bool  # whether its reference's line reads MINI_TRAIN (G0's "every read")
    remedy_a_done: bool  # whether this regime is remedy A's rerun
    wave2_sref: float | None = None  # § 6: 1-seed screening at s_ref <= this, else 2 seeds
    b0: bool = False  # § 6 (ba275d73): G0's ceiling check on B0 L3 test; B0 columns

    @property
    def controls(self) -> tuple[Control, ...]:
        return (
            Control("kl1", f"{self.ref}_kkl1", -0.19),
            Control("g64", f"{self.ref}_kg64", -0.255),
        )

    @property
    def ctrl(self) -> dict[str, Control]:
        return {c.arm: c for c in self.controls}

    def knob(self, arm: str) -> str:
        return arm.removeprefix(self.ref).removeprefix("_k")


REGIMES = {
    "mini": Regime(
        "mini",
        "grpo_think_vs_mini",
        10,
        "10 + 10 iterations",
        "gate.md",
        "mini_table.md",
        ("L2it4", "L3it0", "L3it4", "L3it8"),  # § 6: MINI_ROLL="L2:4 L3:0 L3:4 L3:8"
        False,  # (its references ran at 74864c28, before MINI_TRAIN)
        False,
    ),
    "mini20": Regime(
        "mini20",
        "grpo_think_vs_mini20",
        20,
        "20 + 20 iterations, remedy A",
        "gate_mini20.md",
        "mini20_table.md",
        (),  # none planned yet; any `<ref>_<STAGE>it<IT>` panel read found is used
        True,
        True,
        0.050,  # § 6 (2026-10-18): s_ref <= 0.050 screens at 1 seed, > 0.050 at 2
        True,  # § 6 (2026-10-18, ba275d73, 963d90b3): the B0 deviation
    ),
}
ROLL_SEEDS = (0, 1)


@dataclass(frozen=True)
class Sub:
    key: str
    title: str
    read: str  # the read's set name in eval/v2
    keep: str | None  # its v2_results.subsets() key (None: every row)


SUBS = (
    Sub("l2test", "L2 test_1k", "l2test", None),
    Sub("l3test", "L3 test_1k", "l3test", None),
    Sub("t1320", "13–20", "t1320", "t1320"),
    Sub("t1620", "16–20", "t1320", "t1620"),
    Sub("t2130", "21–30", "t2130", None),
    Sub("charles149", "Charles149", "charles173", "charles149"),
)
TRAIN_SUBS = (
    Sub("l2train", "L2-mini train subset", "l2train", None),
    Sub("l3train", "L3-mini train subset", "l3train", None),
)
# § 6 (2026-10-18, ba275d73): the L3 test read at budget 0 (`_B0` dirs, rows_plain_B0.jsonl)
B0_SUB = Sub("l3test_b0", "L3 test B0", "l3test", None)
B0 = B0_SUB.key
SUB = {s.key: s for s in SUBS + TRAIN_SUBS + (B0_SUB,)}
TABLE_SUBS = tuple(s.key for s in SUBS)
FINAL_K = {"l2test": 8, "l3test": 8, "t1320": 8, "t2130": 8, "charles173": 64}  # MINI_READS
TRAIN_K = {"l2train": 8, "l3train": 8}  # MINI_TRAIN
PANEL_K = {"l3test": 8, "t1320": 8, "t2130": 4, "charles173": 16}  # MINI_PANEL
READ_K = {**FINAL_K, **TRAIN_K}
EXTRA_K = {"t1320": {4, 16}, "t1620": {16}}  # G3's and the replicate's reads
ID_SUB, OOD_SUBS = "l3test", ("t1320", "t1620", "t2130", "charles149")  # § 4.1: P1; P2, P3
# G3: the full-scale line of grpo_think_vs_mb_kkl0 (KL 0 in L3, as the mini L3): it0 is the shared
# grpo_think_vs L2 checkpoint (grpo_think_vs_l2_ckreads), it 10/20/30 its rolling ckreads
# (CKR_PANEL: l3test k8, t1320 k4), it39 its last (CKR_FINAL's l3test k8; the line's t1320 k16)
G3_ARM, G3_IT0 = "grpo_think_vs_mb_kkl0", "grpo_think_vs_L2"
G3_SEEDS, G3_ITS = (0, 1, 2), (10, 20, 30)
# the gate's metrics (set, pass@k): the primary and the two secondaries
PRIMARY, SECONDARY = ("t1320", 1), (("t1620", 1), ("t2130", 5))
# the full-scale g64 replicate: kg64 − the merged recipe (both L3 at KL 0.02 from grpo_think_vs's
# L2 at the same seed), seed 0 the knob pilot (§ 5's contrast); the replicate seed 1, replaced once
# by seed 2 if it fails (§ 4.6; § 6, 2026-10-18, 06:20Z)
REP_ARM, REP_PAIR, REP_PILOT, REP_TRY = "grpo_think_vs_mb_kg64", "grpo_think_vs_mb", 0, (1, 2)
REP_ITERS = 39  # line.sh's full-scale L3 iterations
REP_K = {"t1320": 16, "t1620": 16, "t2130": 8, "charles149": 64}  # line.sh's knob reads
REP_METRICS = (("t1320", 1), ("t1620", 1), ("t2130", 5), ("t1320", 5), ("charles149", 1))


def skey(sub: str, k: int) -> str:
    """A (problem subset, read k) spec key in v2_results' SET registry."""
    return f"mini:{sub}@k{k}"


def is_b0(key: str) -> bool:
    return key.startswith(f"mini:{B0}@")


def logit(p: float) -> float:
    p = min(max(p, CLIP[0]), CLIP[1])
    return math.log(p / (1 - p))


def logit_arr(p: np.ndarray) -> np.ndarray:
    q = np.clip(p, *CLIP)
    return np.log(q / (1 - q))


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)


def sd(xs: list[float]) -> float:
    if len(xs) < 2:
        return float("nan")
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def chi2_ppf(q: float, df: int) -> float:
    """Chi-square quantile: bisection on the regularized lower gamma (its series)."""

    def cdf(x: float) -> float:
        a, z = df / 2.0, x / 2.0
        if z <= 0:
            return 0.0
        term = total = 1.0 / a
        for n in range(1, 2000):
            term *= z / (a + n)
            total += term
            if term < 1e-15 * total:
                break
        return math.exp(a * math.log(z) - z - math.lgamma(a)) * total

    lo, hi = 0.0, 1.0
    while cdf(hi) < q:
        hi *= 2
    for _ in range(200):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if cdf(mid) < q else (lo, mid)
    return (lo + hi) / 2


def pp(x: float, sign: bool = False) -> str:
    return f"{100 * x:+.1f}" if sign else f"{100 * x:.1f}"


def lg(x: float) -> str:
    return f"{x:+.3f}"


def seeds_str(seeds: list[int] | tuple[int, ...]) -> str:
    return ",".join(map(str, seeds))


# ------------------------------------------------------------------------------------------ reads


class Loader(v2.Loader):
    """v2_results' Loader; a read is there once its summary.json is written (after the rows), and
    a planned read that is not there goes to `missing` once. A B0 read (§ 6) is
    `<set>_<arm>_s<seed>_<ra>_k<k>_B0/rows_<ra>_B0.jsonl`, planned at the (arm, seed) in `b0`."""

    def __init__(
        self, ev: Path, keep: dict[str, set[str] | None], b0: set[tuple[str, int]] | None = None
    ) -> None:
        super().__init__(ev, keep)
        self.b0 = b0 or set()

    def path(self, arm: str, s: str, seed: int, ra: str) -> Path | None:
        ss = v2.SET[s]
        if is_b0(s):
            return self.v2 / f"{ss.read}_{arm}_s{seed}_{ra}_k{ss.k}_B0" / f"rows_{ra}_B0.jsonl"
        return v2.rows_path(v2.SPEC[arm], ss, seed, ra, self.v2)

    def have(self, arm: str, s: str, seed: int, ra: str) -> bool:
        p = self.path(arm, s, seed, ra)
        if p is not None and p.exists() and (p.parent / "summary.json").exists():
            return True
        planned = (
            (arm, seed) in self.b0 if is_b0(s) else v2.SPEC[arm].planned(v2.SET[s].read, seed, ra)
        )
        if p is not None and planned and str(p) not in self.missing:
            self.missing.append(str(p))
        return False

    def arm(self, arm: str, s: str, seeds, ra: str = "plain") -> stats.Arm | None:
        if not is_b0(s):
            return super().arm(arm, s, seeds, ra)
        key = (arm, s, tuple(seeds), ra)
        if key not in self._cache:
            ps = [self.path(arm, s, x, ra) for x in seeds]
            ok = bool(seeds) and all(p is not None and p.exists() for p in ps)
            name = f"{arm} {s} s{seeds_str(seeds)}"
            self._cache[key] = stats.load_arm(name, [str(p) for p in ps]) if ok else None
        return self._cache[key]


def discover(ev: Path, rg: Regime) -> tuple[dict[str, set[int]], dict[str, set[int]]]:
    """The regime's arms (reference, controls, knobs; not the panels or the _T reads) and the seeds
    seen in the reads or the training dirs; and its reference's panel checkpoints with their seeds.
    """
    arm = "(" + rg.ref + r"(?:_k[a-z0-9]+)?)_s(\d+)"
    sets = r"(?:l2test|l3test|t1320|t2130|charles173|l2train|l3train)_"
    pats = ((ev, re.compile(sets + arm + r"_plain_k\d+")), (GV, re.compile(arm + "_L[23]")))
    found: dict[str, set[int]] = {}
    for d, p in pats:
        for x in d.iterdir():
            m = p.fullmatch(x.name)
            if m:
                found.setdefault(m.group(1), set()).add(int(m.group(2)))
    panel = re.compile(sets + rg.ref + r"_(L[23]it\d+)_s(\d+)_plain_k\d+(?:_B0)?")
    roll: dict[str, set[int]] = {c: set(ROLL_SEEDS) for c in rg.roll}
    for x in ev.iterdir():
        m = panel.fullmatch(x.name)
        if m:
            roll.setdefault(m.group(1), set()).add(int(m.group(2)))
    return found, roll


def ckpt_order(c: str) -> tuple[int, int]:
    m = re.fullmatch(r"L([23])it(\d+)", c)
    return (int(m.group(1)), int(m.group(2))) if m else (9, 0)


def register(rg: Regime, found: dict[str, set[int]], roll: dict[str, set[int]]) -> dict:
    """Adds the regime's arms to v2_results' SPEC registry, which its Loader reads, and returns them
    with their seeds (the planned ones and any seen), base first, then the reference, the controls
    and the knobs."""
    planned = {BASE: (0,), rg.ref: REF_SEEDS, **{c.arm: c.seeds for c in rg.controls}}
    arms: dict[str, tuple[int, ...]] = {
        a: tuple(sorted(set(s) | found.get(a, set()))) for a, s in planned.items()
    }
    for a in sorted(found):
        arms.setdefault(a, tuple(sorted(found[a])))
    for a, seeds in arms.items():
        reads = tuple(FINAL_K)
        if a != BASE and rg.train_reads and rg.knob(a) in TRAIN_KNOBS:
            reads += tuple(TRAIN_K)
        # a control's later seeds (the forks at 2, 3; a replacement) are planned once they exist
        add_spec(a, seeds if a in rg.ctrl else planned.get(a, seeds), reads)
    for c, seeds in roll.items():
        add_spec(f"{rg.ref}_{c}", tuple(sorted(seeds)), tuple(PANEL_K))
    return arms


def add_spec(key: str, seeds: tuple[int, ...], reads: tuple[str, ...]) -> None:
    if key not in v2.SPEC:  # (G3's and the replicate's full-scale arms are v2_results' own)
        plain = tuple((r, "plain") for r in reads)
        v2.SPEC[key] = v2.ArmSpec(key, key, "grpo", True, fig=False, seeds=seeds, reads=plain)


def b0_plan(rg: Regime, arms: dict, roll: dict[str, set[int]]) -> set[tuple[str, int]]:
    """§ 6 (2026-10-18, ba275d73, 963d90b3): the planned B0 L3 test reads, (arm, seed): every arm
    but the controls (the references, the base, wave-2 arms) at its seeds, the trajectory points
    found, and the full-scale context reads (reads_b0_l3test.sh)."""
    if not rg.b0:
        return set()
    plan = {(a, s) for a, seeds in arms.items() if a not in rg.ctrl for s in seeds}
    return plan | {(f"{rg.ref}_{c}", s) for c, ss in roll.items() for s in ss} | set(B0_FULL)


def register_sets() -> dict[str, set[str] | None]:
    """Adds the (set, k) specs to v2_results' SET registry and returns the Loader's `keep` (the
    problem subset of each spec key)."""
    for a in (G3_IT0, *(f"{G3_ARM}_it{i}" for i in G3_ITS)):
        add_spec(a, G3_SEEDS, ("l3test", "t1320"))
    names = v2.subsets()
    keep: dict[str, set[str] | None] = {}
    for s in SUBS + TRAIN_SUBS + (B0_SUB,):
        ks = {READ_K[s.read], PANEL_K.get(s.read, READ_K[s.read])} | EXTRA_K.get(s.key, set())
        for k in ks:
            key = skey(s.key, k)
            v2.SET[key] = v2.SetSpec(key, s.title, s.read, k, False)
            keep[key] = None if s.keep is None else names[s.keep]
    return keep


def arm_of(ld: Loader, arm: str, sub: str, seeds: list[int], k: int) -> stats.Arm | None:
    key = skey(sub, k)
    if not seeds or not all(ld.have(arm, key, s, "plain") for s in seeds):
        return None
    return ld.arm(arm, key, seeds)


def p_of(
    ld: Loader, arm: str, sub: str, seed: int, kpass: int, kread: int | None = None
) -> float | None:
    """A run's set-mean pass@kpass (a fraction) on `sub`, from its read at kread (default: the
    MINI_READS k), or None."""
    a = arm_of(ld, arm, sub, [seed], kread or READ_K[SUB[sub].read])
    return None if a is None else float(a.values(kpass).mean())


def x_of(ld: Loader, arm: str, seeds: list[int], sub: str, k: int) -> list[float] | None:
    """The runs' x (logit of the set-mean pass@k), or None if a read is missing."""
    ps = [p_of(ld, arm, sub, s, k) for s in seeds]
    return None if any(p is None for p in ps) else [logit(p) for p in ps if p is not None]


def with_reads(ld: Loader, arm: str, seeds: tuple[int, ...] | list[int], sub: str) -> list[int]:
    return [s for s in seeds if p_of(ld, arm, sub, s, 1) is not None]


class Est:
    """The tables' bootstrap estimates (fractions), cached. `kread` None: the MINI_READS k."""

    def __init__(self, ld: Loader, draws: int) -> None:
        self.ld, self.draws = ld, draws
        self._c: dict[tuple, stats.Estimate | None] = {}

    def _arm(self, arm: str, sub: str, seeds: tuple[int, ...], kread: int | None):
        return arm_of(self.ld, arm, sub, list(seeds), kread or READ_K[SUB[sub].read])

    def ci(
        self, arm: str, sub: str, seeds: tuple[int, ...], k: int, kread: int | None = None
    ) -> stats.Estimate | None:
        key = ("ci", arm, sub, seeds, k, kread)
        if key not in self._c:
            a = self._arm(arm, sub, seeds, kread)
            self._c[key] = None if a is None else stats.ci(a, k, draws=self.draws)
        return self._c[key]

    def diff(
        self,
        a: str,
        sa: tuple[int, ...],
        b: str,
        sb: tuple[int, ...],
        sub: str,
        k: int,
        kread: int | None = None,
    ) -> stats.Estimate | None:
        """pass@k of arm a at seeds sa minus arm b at seeds sb, paired on problems; seeds paired
        when sa == sb (seed-matched), else resampled independently."""
        key = ("d", a, sa, b, sb, sub, k, kread)
        if key not in self._c:
            x, y = self._arm(a, sub, sa, kread), self._arm(b, sub, sb, kread)
            self._c[key] = (
                None
                if x is None or y is None
                else stats.paired_diff(x, y, k, paired_seeds=sa == sb, draws=self.draws)
            )
        return self._c[key]

    def logit_diff(
        self, a: str, b: str, seed: int, sub: str, k: int, kread: int | None = None
    ) -> tuple[float, float, float] | None:
        """logit(a) − logit(b) at one seed, with its 95% problem-bootstrap interval: the two
        `stats.ci` calls share the problem resample (same problems, draws and seed)."""
        xa, xb = self._arm(a, sub, (seed,), kread), self._arm(b, sub, (seed,), kread)
        if xa is None or xb is None or xa.names != xb.names:
            return None
        ea, eb = self.ci(a, sub, (seed,), k, kread), self.ci(b, sub, (seed,), k, kread)
        assert ea is not None and eb is not None and ea.samples is not None
        assert eb.samples is not None
        d = logit_arr(ea.samples) - logit_arr(eb.samples)
        lo, hi = np.quantile(d, [0.025, 0.975])
        return logit(ea.est) - logit(eb.est), float(lo), float(hi)


# --------------------------------------------------------------------------------------- training


@dataclass
class Stage:
    name: str
    total: int  # the stage's planned iterations
    exists: bool = False
    rows: int = 0  # metrics.jsonl rows as written
    iters: int = 0  # after dropping a crashed attempt's rows
    restarts: int = 0
    rollbacks: int | None = None
    lr_scale: float | None = None
    final: bool = False  # the stage's last checkpoint (_it<total>.pt)
    running: bool = False  # launched, no exit line, log still changing
    exit: str | None = None
    alarms: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)  # FAILED lines of the last attempt
    failed_before: list[str] = field(default_factory=list)  # ... and of earlier ones
    init: str | None = None


def metrics_rows(path: Path) -> tuple[list[dict], int, int]:
    """metrics.jsonl's rows of the stage's last attempt. A row whose iteration does not follow the
    one before starts a rerun (§ 6, 2026-10-17: a fresh rerun appends from it 1, a --resume one
    from its checkpoint) and replaces every earlier row from that iteration on.
    Returns (rows, rows written, restarts)."""
    raw = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    kept: list[dict] = []
    restarts = 0
    for r in raw:
        if kept and r["iter"] <= kept[-1]["iter"]:
            restarts += 1
            kept = [x for x in kept if x["iter"] < r["iter"]]
        kept.append(r)
    return kept, len(raw), restarts


def stage_info(arm: str, seed: int, stage: str, total: int, log: list[str]) -> Stage:
    n = f"{arm}_s{seed}_{stage}"
    st = Stage(n, total)
    d = GV / n
    st.exists = d.exists()
    if (d / "metrics.jsonl").exists():
        kept, st.rows, st.restarts = metrics_rows(d / "metrics.jsonl")
        st.iters = len(kept)
        if kept:
            st.rollbacks, st.lr_scale = kept[-1].get("rollbacks"), kept[-1].get("lr_scale")
    st.final = (d / "ckpts" / f"{n}_it{total}.pt").exists()
    if (d / "args.json").exists():
        st.init = json.loads((d / "args.json").read_text()).get("init")
    starts = [i for i, x in enumerate(log) if f"launched GRPO {n} " in x]
    cut = starts[-1] if starts else len(log)
    for x in log[cut:]:
        if f"GRPO {n} exit" in x:
            st.exit = x.split(f"GRPO {n} exit", 1)[1].split(":", 1)[0].strip()
        elif f"GRPO {n} ALARM" in x:
            st.alarms.append(x)
        elif f"GRPO {n} FAILED" in x:
            st.failed.append(x)
    st.failed_before = [x for x in log[:cut] if f"GRPO {n} FAILED" in x]
    slog = LOGS / f"v2_{n}.log"
    fresh = slog.exists() and datetime.now(UTC).timestamp() - slog.stat().st_mtime < LIVE
    st.running = bool(starts) and st.exit is None and fresh
    return st


@dataclass
class Run:
    arm: str
    seed: int
    fork: bool
    stages: dict[str, Stage]
    line_done: bool
    line_failed: list[str]
    commit: str | None
    reads: dict[str, bool]  # MINI_READS set -> there (G0's "every read")
    train: dict[str, bool]  # planned MINI_TRAIN set -> there (reported; never gates, § 6)

    @property
    def rolled_back(self) -> bool:
        return any((s.rollbacks or 0) > 0 for s in self.stages.values())

    @property
    def lr_changed(self) -> bool:
        return any(s.lr_scale not in (None, 1.0) for s in self.stages.values())

    @property
    def finished(self) -> bool:
        return self.stages["L3"].final

    @property
    def failed(self) -> bool:
        return bool(self.line_failed or any(s.failed for s in self.stages.values()))

    def flags(self) -> str:
        ss = self.stages.values()
        f = []
        if self.failed:
            f.append("FAILED")
        if any(s.failed_before for s in ss):
            f.append("earlier attempt FAILED")
        if self.rolled_back:
            f.append("ROLLBACK")
        if self.lr_changed:
            f.append("lr changed")
        if any(s.alarms for s in ss):
            f.append(f"{sum(len(s.alarms) for s in ss)} ALARM")
        if any(s.restarts for s in ss):
            f.append("crash-rerun")
        if any(s.running for s in ss):
            f.append("training")
        elif self.finished and not all(self.reads.values()):
            f.append("reading")
        return ", ".join(f) or "—"

    def status(self) -> str:
        if self.line_done and all(self.reads.values()):
            return "line done"
        for s in self.stages.values():
            if s.running:
                return f"training {s.name[-2:]} (it {s.iters}/{s.total})"
        if self.finished:
            return f"reads {sum(self.reads.values())}/{len(self.reads)}"
        if self.failed:
            return "FAILED"
        if not any(s.exists for s in self.stages.values()):
            return "not started"
        return "stopped (no exit line, log idle)"


def run_info(ld: Loader, rg: Regime, arm: str, seed: int) -> Run:
    f = LOGS / f"v2_{arm}_s{seed}.log"
    log = f.read_text(errors="replace").splitlines() if f.exists() else []
    fork = arm in rg.ctrl
    stages = {"L3": stage_info(arm, seed, "L3", rg.iters, log)}
    if not fork:  # a fork's L2 is the reference's
        stages = {"L2": stage_info(arm, seed, "L2", rg.iters, log), **stages}
    done = any(f"grpo_v2 line done ({arm} seed {seed})" in x for x in log)
    # the line's own FAILED lines (a fork whose L2-mini never came) unless a later L3 launch
    # superseded them
    starts = [i for i, x in enumerate(log) if f"launched GRPO {arm}_s{seed}_L3 " in x]
    tail = log[starts[-1] :] if starts else log
    lf = [x for x in tail if "FAILED" in x and "GRPO " not in x]
    cf = GV / f"{arm}_s{seed}.base_commit.txt"
    commit = cf.read_text().strip()[:8] if cf.exists() else None
    subs = {"charles173": "charles149"}
    have = {
        r: ld.have(arm, skey(subs.get(r, r), READ_K[r]), seed, "plain")
        for r, _ in v2.SPEC[arm].reads
    }
    reads = {r: x for r, x in have.items() if r in FINAL_K}
    train = {r: x for r, x in have.items() if r in TRAIN_K}
    return Run(arm, seed, fork, stages, done, lf, commit, reads, train)


# ------------------------------------------------------------------------------------------ gate


@dataclass
class CtrlResult:
    c: Control
    seeds: list[int]
    n_r: int
    delta: float = float("nan")
    delta_pts: float = float("nan")
    se: float = float("nan")
    t: float = float("nan")
    U: float = float("nan")
    sec: dict[tuple[str, int], float] = field(default_factory=dict)
    sec_pts: dict[tuple[str, int], float] = field(default_factory=dict)
    cls: str = "pending"
    complete: bool = False  # this round's planned seeds are all in
    round: int = 1  # 1: seeds 0, 1; 2: with the forks at 2, 3 (§ 3's table)
    want: list[int] = field(default_factory=list)  # this round's seeds, failed ones replaced
    forks: list[int] = field(default_factory=list)  # round 2's seeds beyond round 1's
    full: int = 0  # round 2's n_c: 4, fewer when failed seeds leave too few unused ones
    failed: list[int] = field(default_factory=list)  # runs without their final checkpoint (§ 4.6)
    ended: int = 0  # the control's runs that ended (final checkpoint or failed): the rate's n
    short: bool = False  # fewer unused seeds than the round needs: the reduced n_c, flagged

    @property
    def unstable(self) -> bool:  # § 4.6: replaced once; a second failure makes it unstable
        return len(self.failed) >= 2

    def failed_note(self) -> str:
        if not self.failed:
            return ""
        s = "s" if len(self.failed) > 1 else ""
        if self.unstable:
            return (
                f" Seeds {seeds_str(self.failed)} failed (§ 4.6: no final checkpoint): the "
                'control is "unstable" (a failed run is replaced once), so no GO.'
            )
        left = f"; none left, so the reduced n_c {len(self.want)}, flagged" if self.short else ""
        return (
            f" Seed{s} {seeds_str(self.failed)} failed (§ 4.6: no final checkpoint), excluded "
            "from n_c: replace with the next unused seed whose reference L2-mini exists (this "
            f"round's seeds: {seeds_str(self.want) or 'none'}{left})."
        )


def mean_p(ld: Loader, arm: str, seeds: list[int], sub: str, k: int) -> float:
    ps = [p_of(ld, arm, sub, s, k) for s in seeds]
    return mean([p for p in ps if p is not None])


def classify(
    ld: Loader,
    rg: Regime,
    c: Control,
    ref_seeds: list[int],
    s_ref: float,
    failed: set[int] | None = None,
    l2: list[int] | None = None,
) -> CtrlResult:
    """§ 3's class of control c on the seeds with its gate reads. Round 1 is seeds 0, 1; round 2
    adds the forks at 2, 3 (it starts once any seed after round 1's has its reads or has failed).
    A seed in `failed` (§ 4.6) is skipped and its place taken by the next unused seed whose
    reference L2-mini exists (`l2`: those of REF_SEEDS; § 6, 963d90b3); a round left with too few
    is assessed at the reduced n_c, flagged. 2 or more failed runs: "unstable" (§ 4.6)."""
    failed = failed or set()
    order = [s for s in (*c.seeds, *c.more) if l2 is None or s in l2]
    good = [s for s in order if s not in failed]
    have = [
        s
        for s in good
        if all(p_of(ld, c.arm, sub, s, k) is not None for sub, k in (PRIMARY, *SECONDARY))
    ]
    res = CtrlResult(c, have, len(ref_seeds))
    n1, n2 = len(c.seeds), len(c.seeds) + len(c.more)
    res.forks, res.full = [s for s in good[:n2] if s not in good[:n1]], len(good[:n2])
    # round 1's seeds and the failed ones among them sit before `cut` in `order`
    cut = order.index(good[n1 - 1]) + 1 if len(good) >= n1 else len(order)
    res.round = 2 if any(s in have or s in failed for s in order[cut:]) else 1
    n = n2 if res.round == 2 else n1
    res.want, res.short = good[:n], len(good) < n
    res.failed = [s for s in order if s in failed]
    res.complete = all(s in have for s in res.want)
    if not have or len(ref_seeds) < 2 or math.isnan(s_ref):
        return res
    for sub, k in (PRIMARY, *SECONDARY):
        xr, xc = x_of(ld, rg.ref, ref_seeds, sub, k), x_of(ld, c.arm, have, sub, k)
        assert xr is not None and xc is not None
        d = mean(xc) - mean(xr)
        dp = mean_p(ld, c.arm, have, sub, k) - mean_p(ld, rg.ref, ref_seeds, sub, k)
        if (sub, k) == PRIMARY:
            res.delta, res.delta_pts = d, dp
        else:
            res.sec[(sub, k)], res.sec_pts[(sub, k)] = d, dp
    res.se = s_ref * math.sqrt(1 / len(have) + 1 / len(ref_seeds))
    res.t = stats.t_ppf(0.95, len(ref_seeds) - 1)
    res.U = res.delta + res.t * res.se
    if res.delta >= 0:
        res.cls = "fails"
    elif res.delta <= c.E / 2 and res.U < 0 and all(v < 0 for v in res.sec.values()):
        res.cls = "reproduces"
    else:
        res.cls = "partial"
    return res


def at(r: CtrlResult) -> str:
    """A control's fork seeds, for the decision text."""
    return f"seed{'s' if len(r.forks) > 1 else ''} {' and '.join(map(str, r.forks))}"


def rate(rg: Regime, r: CtrlResult) -> str:
    """§ 4.6's failure rate of a control's runs (mini20)."""
    return f" Failed runs (§ 4.6): {len(r.failed)} of {r.ended}." if rg.remedy_a_done else ""


def decide(rg: Regime, g0: str, g1: str, cr: dict[str, CtrlResult]) -> tuple[str, tuple[str, ...]]:
    """§ 3's decision table, on the current classes, and the inputs it rests on (`_decide`). For
    remedy A's rerun (mini20), § 6 (2026-10-18, about 04:45Z, its clarification, ba275d73 and
    963d90b3): § 3 row by row, but where § 3 would call for remedy A again the mini is retired,
    remedy C is not started overnight and Leon decides; a G1 failure follows § 3's own row (remedy
    B if both mini20 controls reproduce, else retired); a partial control's fork seeds queue
    automatically, a partial kl1 alongside a partial g64; with g64 failed or still partial at n_c
    4, kl1's class decides nothing (and is not waited for); an unstable control (§ 4.6) gives no
    GO; a GO runs the knob wave at 20 + 20."""
    kind, row, why, needs = _decide(g0, g1, cr)
    text = f"{row} ({why})" if why else row
    if not rg.remedy_a_done or kind in ("pending", "unstable"):
        return text, needs
    g, k = cr["g64"], cr["kl1"]
    if kind == "g1fail":
        cls = {n: r.cls for n, r in cr.items()}
        if all(c == "reproduces" for c in cls.values()):
            then = "remedy B (knobs as L3 forks of the 4 mini20 reference L2-minis): both"
            return f"NO-GO for both-stage knobs ({why}): {then} controls reproduce (§ 6)", needs
        if "pending" in cls.values():
            then = "remedy B if both controls reproduce, else the mini is retired"
            return f"NO-GO for both-stage knobs ({why}): {then} (controls pending; § 6)", needs
        cl = ", ".join(f"{n} {c}" for n, c in cls.items())
        return f"{RETIRED} ({why}; {cl}; § 6)", needs
    if kind == "nogo":
        if needs == ("G0", "G1", "g64"):  # g64 failed or still partial at n_c 4
            why += f"; kl1 {k.cls} at n_c {len(k.seeds)}: reported, decides nothing"
        return f"{RETIRED} ({why}; § 6)", needs
    if kind == "more" and g.cls == "partial":
        more = f"g64 forks at {at(g)}, then re-apply at n_c {g.full}"
        if k.cls == "partial" and k.round == 1:
            more += f"; kl1 forks at {at(k)} alongside"
        more += (
            f"; if g64 is still partial at n_c {g.full}, the mini is retired (remedy A is spent) "
            "and kl1's class is reported but decides nothing"
        )
        return f"{more}; the forks queue automatically (§ 6)", needs
    if kind == "more":
        more = f"kl1 forks at {at(k)}, re-apply at n_c {k.full}; if still partial, restricted GO"
        return f"{more}; the forks queue automatically (§ 6)", needs
    via = " (via the § 6 B0 deviation, ba275d73)" if rg.b0 else ""
    return f"{text}{via}: the knob wave at 20 + 20 (§ 6)", needs


GATE_INPUTS = ("G0", "G1", "kl1", "g64")


def _decide(g0: str, g1: str, cr: dict[str, CtrlResult]) -> tuple[str, str, str, tuple[str, ...]]:
    """(kind, § 3's row, why, needs); kind is pending, g1fail, unstable (§ 4.6), nogo (§ 3 calls
    for remedy A), more (more control seeds first) or go; needs: the GATE_INPUTS the row rests
    on, so a row that does not wait for an input is final while that input is incomplete."""
    if g0 != "PASS":
        if g0 == "FAIL":
            return "nogo", "NO-GO, remedy A", "G0 failed", ("G0",)
        return "pending", "PENDING", "G0", GATE_INPUTS
    if g1 != "PASS":
        if g1 == "FAIL":
            row = "NO-GO for both-stage knobs, remedy B (or C)"
            return "g1fail", row, "G1: s_ref > 0.11", GATE_INPUTS
        return "pending", "PENDING", "G1", GATE_INPUTS
    g, k = cr["g64"], cr["kl1"]
    bad = [r for r in (k, g) if r.unstable]
    if bad:
        row = (
            f"NO-GO: control {' and '.join(r.c.name for r in bad)} unstable (§ 4.6); nothing "
            "starts overnight; Leon decides"
        )
        why = "; ".join(f"{r.c.name}: {len(r.failed)} of {r.ended} runs failed" for r in bad)
        return "unstable", row, why, ("G0", "G1")
    g64 = ("G0", "G1", "g64")  # g64 failed or still partial at n_c 4: kl1 decides nothing
    if g.cls == "pending":
        return "pending", "PENDING", "control reads", GATE_INPUTS
    if g.cls == "fails":
        return "nogo", "NO-GO, remedy A; if that also fails, remedy C", "g64 fails", g64
    if g.cls == "partial" and g.complete and (g.round == 2 or not g.forks):
        return "nogo", "remedy A", f"g64 still partial at n_c = {len(g.seeds)}", g64
    if k.cls == "pending":
        return "pending", "PENDING", "control reads", GATE_INPUTS
    if g.cls == "partial":
        row = "g64 forks at seeds 2 and 3, re-apply; if still partial, remedy A"
        return "more", row, "", GATE_INPUTS
    if k.cls == "reproduces":
        return "go", "GO, all knobs", "", GATE_INPUTS
    if k.cls == "partial":
        if k.complete and (k.round == 2 or not k.forks):
            why = f"kl1 still partial at n_c = {len(k.seeds)}"
            return "go", "restricted GO", why, GATE_INPUTS
        row = "kl1 forks at seeds 2 and 3, re-apply at n_c = 4; if still partial, restricted GO"
        return "more", row, "", GATE_INPUTS
    return "go", "restricted GO", "kl1 fails", GATE_INPUTS


# --------------------------------------------------------------- competence vs generalization


def ols(pts: list[tuple[float, float]]) -> tuple[float, float] | None:
    xs, ys = np.array([p[0] for p in pts]), np.array([p[1] for p in pts])
    if len(pts) < 3 or float(np.var(xs)) < 1e-9:
        return None
    b = float(((xs - xs.mean()) * (ys - ys.mean())).sum() / ((xs - xs.mean()) ** 2).sum())
    return float(ys.mean() - b * xs.mean()), b


def ref_points(
    ld: Loader, rg: Regime, roll: list[str], ref_seeds: list[int], ood: str
) -> dict[int, list[tuple]]:
    """Per reference seed: (checkpoint, logit ID, logit OOD) of its final reads and its panels,
    pass@1."""
    out: dict[int, list[tuple]] = {}
    for s in ref_seeds:
        pts = []
        x, y = p_of(ld, rg.ref, ID_SUB, s, 1), p_of(ld, rg.ref, ood, s, 1)
        if x is not None and y is not None:
            pts.append(("final", logit(x), logit(y)))
        for c in roll:
            a = f"{rg.ref}_{c}"
            x = p_of(ld, a, ID_SUB, s, 1, PANEL_K[SUB[ID_SUB].read])
            y = p_of(ld, a, ood, s, 1, PANEL_K[SUB[ood].read])
            if x is not None and y is not None:
                pts.append((c, logit(x), logit(y)))
        out[s] = pts
    return out


def residual_view(
    ld: Loader,
    rg: Regime,
    roll: list[str],
    ref_seeds: list[int],
    knobs: dict[str, list[int]],
    B: int = 2000,
) -> list[str]:
    """§ 4.3: logit(OOD) = a + b logit(ID), OLS on reference points only (finals and panels). A
    knob's generalization part = its mean residual − the reference finals' mean residual,
    SE = SD(reference final residuals) √(1/n_k + 1/n_r); its competence part = b ΔID, so the two
    add up to Δlogit(OOD). Line uncertainty = the SD of b over a bootstrap of reference runs (each
    with its trajectory) × |ΔID|. 95% interval: ± t(0.975, n_r − 1) √(SE² + SE_line²)."""
    md = []
    t975 = stats.t_ppf(0.975, max(len(ref_seeds) - 1, 1))
    for ood in OOD_SUBS:
        P = ref_points(ld, rg, roll, ref_seeds, ood)
        traj = sum(1 for v in P.values() for p in v if p[0] != "final")
        title = f"{SUB[ood].title} pass@1 against {SUB[ID_SUB].title} pass@1"
        if traj == 0:
            md.append(f"- {title}: **needs trajectories** (no `{rg.ref}_<STAGE>it<IT>` read).")
            continue
        allp = [(x, y) for v in P.values() for _, x, y in v]
        fit = ols(allp)
        if fit is None:
            md.append(f"- {title}: too few distinct reference points for a line ({len(allp)}).")
            continue
        a, b = fit
        fin = [(x, y) for v in P.values() for c, x, y in v if c == "final"]
        rr = [y - a - b * x for x, y in fin]
        s_res = sd(rr)
        rng = np.random.default_rng(0)
        runs = list(P)
        bs = []
        for _ in range(B):
            pick = rng.choice(len(runs), len(runs), replace=True)
            f2 = ols([(x, y) for i in pick for _, x, y in P[runs[i]]])
            if f2 is not None:
                bs.append(f2[1])
        sd_b = float(np.std(bs, ddof=1)) if len(bs) > 1 else float("nan")
        lo_x, hi_x = min(x for x, _ in allp), max(x for x, _ in allp)
        md += [
            "",
            f"**{title}**: line a {a:+.3f}, b {b:+.3f}; {len(allp)} reference points ({traj} "
            f"panel); bootstrap SD of b {sd_b:.3f} over {len(bs)} of {B} resamples of the "
            f"{len(runs)} reference runs (the rest degenerate); reference final residuals: mean "
            f"{lg(mean(rr))}, SD {s_res:.3f} (n_r {len(rr)}).",
            "",
            "| arm | seeds | ΔID | ΔOOD | competence b·ΔID | generalization | SE | SE line | "
            "95% interval | extrapolated |",
            "|---|---|---|---|---|---|---|---|---|---|",
        ]
        for arm, seeds in knobs.items():
            pts, used = [], []
            for s in seeds:
                x, y = p_of(ld, arm, ID_SUB, s, 1), p_of(ld, arm, ood, s, 1)
                if x is not None and y is not None:
                    pts.append((logit(x), logit(y)))
                    used.append(s)
            if not pts or len(rr) < 2:
                continue
            xm = mean([x for x, _ in pts])
            dx = xm - mean([x for x, _ in fin])
            dy = mean([y for _, y in pts]) - mean([y for _, y in fin])
            g = mean([y - a - b * x for x, y in pts]) - mean(rr)
            se = s_res * math.sqrt(1 / len(pts) + 1 / len(rr))
            se_l = sd_b * abs(dx)
            h = t975 * math.sqrt(se**2 + se_l**2)
            ext = "yes" if xm < lo_x - 0.2 or xm > hi_x + 0.2 else "no"
            md.append(
                f"| {label(rg, arm)} | {seeds_str(used)} | {lg(dx)} | {lg(dy)} | {lg(b * dx)} | "
                f"{lg(g)} | {se:.3f} | {se_l:.3f} | [{g - h:+.3f}, {g + h:+.3f}] | {ext} |"
            )
    return md


def g3_view(ld: Loader, rg: Regime, ref_seeds: list[int]) -> list[str]:
    """G3 (informational): the references' (L3 test, 13–20) pass@1 against the full-scale line of
    G3_ARM from it0 to it39 (OLS in logit)."""
    pts = []
    for s in G3_SEEDS:
        cks = [("it0", G3_IT0, 4), *((f"it{i}", f"{G3_ARM}_it{i}", 4) for i in G3_ITS)]
        for name, arm, kt in [*cks, ("it39", G3_ARM, 16)]:
            x, y = p_of(ld, arm, "l3test", s, 1, 8), p_of(ld, arm, "t1320", s, 1, kt)
            if x is not None and y is not None:
                pts.append((s, name, logit(x), logit(y)))
    fit = ols([(x, y) for *_, x, y in pts])
    have = ", ".join(f"s{s} {n}" for s, n, *_ in pts) or "none"
    if fit is None or len({n for _, n, *_ in pts}) < 2:
        return [f"- **needs ckreads**: full-scale {G3_ARM} points with both reads: {have}."]
    a, b = fit
    res = [y - a - b * x for *_, x, y in pts]
    lo, hi = (1 / (1 + math.exp(-f(p[2] for p in pts))) for f in (min, max))
    md = [
        f"- Full-scale line ({G3_ARM}; {len(pts)} points: {have}): logit(13–20) = {a:+.3f} "
        f"{b:+.3f} · logit(L3 test); residual SD {sd(res):.3f}; L3 test {pp(lo)}–{pp(hi)}%.",
    ]
    rr = []
    for s in ref_seeds:
        x, y = p_of(ld, rg.ref, "l3test", s, 1), p_of(ld, rg.ref, "t1320", s, 1)
        if x is not None and y is not None:
            rr.append(logit(y) - a - b * logit(x))
            md.append(f"  - reference s{s}: L3 test {pp(x)}%, 13–20 {pp(y)}%: {lg(rr[-1])} logit")
    if rr:
        md.append(f"  - the references' mean residual: {lg(mean(rr))} logit")
    return md


# ---------------------------------------------------------------------------------------- output


def now_line() -> str:
    t = datetime.now(UTC)
    pt = t - timedelta(hours=7)  # PDT

    def git(*a: str) -> str:
        return subprocess.run(
            ["git", "-C", str(REPO), *a], capture_output=True, text=True
        ).stdout.strip()

    return (
        f"{t:%Y-%m-%d %H:%M} UTC ({pt:%-I:%M %p} PDT, {pt:%b %-d}); analysis worktree "
        f"{git('branch', '--show-current')} {git('rev-parse', '--short', 'HEAD')}."
    )


def cell(e1: stats.Estimate | None, e5: stats.Estimate | None, sign: bool = False) -> str:
    if e1 is None or e5 is None:
        return "—"
    return f"{v2.fmt(e1, sign)} / {v2.fmt(e5, sign)}"


def label(rg: Regime, arm: str) -> str:
    if arm == BASE:
        return "base S1g3_s0"
    if arm == rg.ref:
        return "reference"
    if arm in rg.ctrl:
        return f"{rg.ctrl[arm].name} (L3 fork)"
    return rg.knob(arm)


def g0_checks(ld: Loader, rg: Regime, runs: dict) -> list[tuple[str, str, str]]:
    ref_runs = [runs[(rg.ref, s)] for s in REF_SEEDS]
    checks = []
    what = f"all 4 references end with L3-mini _it{rg.iters}.pt and every read"
    ok = [r.finished and all(r.reads.values()) and not r.failed for r in ref_runs]
    if all(ok):
        checks.append((what, "PASS", "4/4"))
    else:
        bad = "; ".join(
            f"s{r.seed}: {r.status()}" for r, o in zip(ref_runs, ok, strict=True) if not o
        )
        checks.append((what, "FAIL" if any(r.failed for r in ref_runs) else "PENDING", bad))
    rolled = [r.seed for r in ref_runs if r.rolled_back]
    flagged = f" (s{', s'.join(map(str, rolled))}, flagged)" if rolled else ""
    checks.append(
        (
            "at most 1 of the 4 rolled back",
            "PASS" if len(rolled) <= 1 else "FAIL",
            f"{len(rolled)} rolled back{flagged}",
        )
    )
    p13 = [p_of(ld, rg.ref, "t1320", s, 1) for s in REF_SEEDS]
    if all(p is not None for p in p13):
        m13 = mean([p for p in p13 if p is not None])
        pb = p_of(ld, BASE, "t1320", 0, 1)
        base = BASE_FALLBACK if pb is None else pb
        src = "no base read: § 3's 7.3%" if pb is None else "base read t1320_s1g3_s0_plain_k8"
        lo, hi = G0_RANGE
        ok13 = "PASS" if lo <= m13 <= hi else "FAIL"
        checks.append(("reference mean 13–20 pass@1 in [12%, 60%]", ok13, f"{pp(m13)}%"))
        checks.append(
            (
                "... and >= base + 5 points",
                "PASS" if m13 >= base + G0_ABOVE_BASE else "FAIL",
                f"{pp(m13)}% vs base {pp(base)}% ({src}): {pp(m13 - base, True)} points",
            )
        )
    else:
        checks.append(("reference mean 13–20 pass@1: in range, >= base + 5", "PENDING", "reads"))
    p3 = [p_of(ld, rg.ref, "l3test", s, 1) for s in REF_SEEDS]
    what = "reference mean L3 test_1k pass@1 <= 90%"
    if rg.b0:  # § 6 deviation: reported, never gating (G0_DECIDES)
        what += " (B64, as § 3 wrote it)"
    if all(p is not None for p in p3):
        m3 = mean([p for p in p3 if p is not None])
        res = "PASS" if m3 <= G0_ID_MAX else "FAIL"
        if rg.b0:
            res = f"{'met' if res == 'PASS' else 'not met'}: reported; the B0 row decides (§ 6)"
        checks.append((what, res, f"{pp(m3)}%"))
    else:
        checks.append((what, "pending: reported only" if rg.b0 else "PENDING", "reads missing"))
    if rg.b0:
        p0 = {s: p_of(ld, rg.ref, B0, s, 1) for s in REF_SEEDS}
        miss = [s for s, p in p0.items() if p is None]
        if miss:
            checks.append((G0_B0, "PENDING", f"B0 reads missing: s{', s'.join(map(str, miss))}"))
        else:
            ps = [p for p in p0.values() if p is not None]
            per = ", ".join(pp(p) for p in ps)
            res = B0_MET if mean(ps) <= G0_ID_MAX else "FAIL"
            checks.append((G0_B0, res, f"{pp(mean(ps))}% (s0–s3: {per})"))
    return checks


def gate_md(ld: Loader, est: Est, rg: Regime, arms: dict, roll: list[str], runs: dict) -> list[str]:
    ref_seeds = with_reads(ld, rg.ref, REF_SEEDS, PRIMARY[0])
    other = [r for r in REGIMES.values() if r is not rg]
    md = [
        f"# Mini-run validation gate: {rg.title}",
        "",
        f"{HEADER} {now_line()}",
        "",
        "Applies [PREREGISTRATION.md](../PREREGISTRATION.md) § 3 (as amended in § 6) mechanically",
        f"to the `{rg.ref}*` runs only ({rg.iters} iterations a stage): their reads in",
        "`runs/eval/v2` and their training records in `runs/grpo_v2` and `runs/logs`. "
        + " ".join(
            f"The `{r.ref}*` runs are in [{r.gate}]({r.gate}), never pooled." for r in other
        ),
        "Logits are of the set-mean pass (clipped to [0.005, 0.995]); points are % of the set. Per",
        f"arm and seed: [{rg.table}]({rg.table}). Rerun:",
        "",
        f"    {RERUN} --regime {rg.key}",
        "",
    ]
    checks = g0_checks(ld, rg, runs)
    states = {c[1] for c in checks if c[1] in G0_DECIDES}
    g0 = "FAIL" if "FAIL" in states else "PENDING" if "PENDING" in states else "PASS"
    xr = x_of(ld, rg.ref, ref_seeds, *PRIMARY) or []
    s_ref = sd(xr)
    g1 = "PENDING" if len(xr) < len(REF_SEEDS) else "PASS" if s_ref <= G1_MAX else "FAIL"
    df = len(xr) - 1
    ci_s = (
        (s_ref * math.sqrt(df / chi2_ppf(0.975, df)), s_ref * math.sqrt(df / chi2_ppf(0.025, df)))
        if df >= 1
        else (float("nan"), float("nan"))
    )
    # § 4.6: a control seed whose run ended without its final checkpoint; § 6 (963d90b3): its
    # replacement is the next unused seed whose reference L2-mini exists, among REF_SEEDS only
    # (a later reference seed must not reopen the gate)
    l2 = [s for s in REF_SEEDS if (rg.ref, s) in runs and runs[(rg.ref, s)].stages["L2"].final]
    cr = {}
    for c in rg.controls:
        rs = {s: runs[(c.arm, s)] for s in arms[c.arm]}
        bad = {s for s, r in rs.items() if r.failed and not r.finished}
        cr[c.name] = classify(ld, rg, c, ref_seeds, s_ref, bad, l2)
        cr[c.name].ended = sum(r.finished or s in bad for s, r in rs.items())
    decision, needs = decide(rg, g0, g1, cr)
    incomplete = {"G0": g0 == "PENDING", "G1": g1 == "PENDING"}
    incomplete |= {n: not r.complete or r.cls == "pending" for n, r in cr.items()}
    waits = [n for n in ("G0", "G1", *cr) if n in needs and incomplete[n]]
    g0_line = f"- **G0 (the setup learns): {g0}.**"
    if rg.b0 and g0 == "PASS":
        g0_line = (
            "- **G0 (the setup learns): PASS (the ceiling check is met by the clause's fix, B0: a "
            "§ 6 deviation, 2026-10-18, ba275d73).** The B64 value is reported and does not decide."
        )
    elif rg.b0:
        b0 = next(r for w, r, _ in checks if w == G0_B0)
        g0_line += (
            f" Its ceiling check (§ 6 deviation, 2026-10-18, ba275d73): {b0}, on B0 L3 test pass@1;"
            " the B64 value is reported and does not decide."
        )
    md += [
        "## Verdict",
        "",
        g0_line,
        f"- **G1 (noise): {g1}**: s_ref = {s_ref:.3f} logit on 13–20 pass@1 (threshold 0.11; "
        f"n_r {len(xr)}; 95% interval for σ [{ci_s[0]:.3f}, {ci_s[1]:.3f}]).",
    ]
    for c in rg.controls:
        r = cr[c.name]
        if r.cls == "pending":
            md.append(
                f"- **{c.name}: pending** (no seed with the 13–20 and 21–30 reads yet)."
                + rate(rg, r)
                + r.failed_note()
            )
            continue
        tag = "" if r.complete else f" (provisional: seeds {seeds_str(r.seeds)} so far)"
        md.append(
            f"- **{c.name}: {r.cls}**{tag}: n_c {len(r.seeds)}, Δ {lg(r.delta)} logit "
            f"({pp(r.delta_pts, True)} points), ½E {c.E / 2:+.4f}, SE {r.se:.4f}, U {lg(r.U)}; "
            f"16–20 pass@1 Δ {lg(r.sec[SECONDARY[0]])}, 21–30 pass@5 Δ {lg(r.sec[SECONDARY[1]])}."
            + rate(rg, r)
            + r.failed_note()
        )
    if waits:
        md.append(
            f"- **Decision: PENDING** (not complete: {', '.join(waits)}). On the current numbers "
            f"§ 3's table gives: {decision}."
        )
    else:
        md.append(f"- **Decision (§ 3's table): {decision}.**")
    if rg.wave2_sref is not None and len(xr) == len(REF_SEEDS):
        n = 1 if s_ref <= rg.wave2_sref else 2
        near = abs(s_ref - rg.wave2_sref) <= 0.005
        md.append(
            f"- **Wave-2 seeds (§ 6): {n} per knob**: s_ref {s_ref:.3f} against {rg.wave2_sref:.3f}"
            f" logit{' (within ± 0.005 of it: flagged to Leon; the rule applies)' if near else ''}."
        )
    md += ["", "## G0: the setup learns and is sane", "", "| check | result | value |"]
    md += ["|---|---|---|", *(f"| {a} | {b} | {c} |" for a, b, c in checks)]
    if rg.b0:  # the deviation's reason 1, at B0 (reads_b0_l3test.sh's full-scale reads)
        full = []
        for a, s in B0_FULL:
            p0, p64 = p_of(ld, a, B0, s, 1), p_of(ld, a, "l3test", s, 1)
            b64 = "" if p64 is None else f" (B64 {pp(p64)}%)"
            full.append(f"{a} s{s} {'—' if p0 is None else pp(p0) + '%'}{b64}")
        md += ["", f"Full-scale B0 L3 test pass@1 (context; does not gate): {'; '.join(full)}."]
    md += [
        "",
        "## G1 and the controls (13–20 pass@1, logit)",
        "",
        "| run | seeds | x per seed | mean x | mean pass@1 |",
        "|---|---|---|---|---|",
    ]
    for arm in (rg.ref, *(c.arm for c in rg.controls)):
        seeds = with_reads(ld, arm, arms[arm], PRIMARY[0])
        xs = x_of(ld, arm, seeds, *PRIMARY) if seeds else None
        if not xs:
            md.append(f"| {label(rg, arm)} | — | — | — | — |")
            continue
        md.append(
            f"| {label(rg, arm)} | {seeds_str(seeds)} | {', '.join(lg(x) for x in xs)} | "
            f"{lg(mean(xs))} | {pp(mean_p(ld, arm, seeds, *PRIMARY))}% |"
        )
    t95 = stats.t_ppf(0.95, max(df, 1))
    md += [
        "",
        f"s_ref = {s_ref:.4f} (df {df}); t(0.95, {df}) = {t95:.3f}.",
        "",
        "| control | n_c | Δ | Δ points | E | ½E | SE | U | 16–20 pass@1 Δ (points) | "
        "21–30 pass@5 Δ (points) | class |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for c in rg.controls:
        r = cr[c.name]
        if r.cls == "pending":
            md.append(
                f"| {c.name} | 0 | — | — | {c.E} | {c.E / 2:+.4f} | — | — | — | — | pending |"
            )
            continue
        s1, s5 = SECONDARY
        md.append(
            f"| {c.name} | {len(r.seeds)} | {lg(r.delta)} | {pp(r.delta_pts, True)} | {c.E} | "
            f"{c.E / 2:+.4f} | {r.se:.4f} | {lg(r.U)} | {lg(r.sec[s1])} "
            f"({pp(r.sec_pts[s1], True)}) | {lg(r.sec[s5])} ({pp(r.sec_pts[s5], True)}) | "
            f"{r.cls}{'' if r.complete else ' (provisional)'}"
            f"{' (reduced n_c, flagged)' if r.short else ''} |"
        )
    md += [
        "",
        "### Paired fork differences c_s − r_s (secondary; the same seed = the same L2-mini)",
        "",
        "Logit of the set mean; 95% problem-bootstrap intervals (attempt and problem noise, no",
        "training noise).",
        "",
        "| control | seed | 13–20 pass@1 | 16–20 pass@1 | 21–30 pass@5 | Charles149 pass@1 | "
        "L3 test pass@1 |",
        "|---|---|---|---|---|---|---|",
    ]
    for c in rg.controls:
        for s in sorted(set(c.seeds) | set(c.more)):
            cells = []
            for sub, k in (("t1320", 1), ("t1620", 1), ("t2130", 5), ("charles149", 1)):
                cells.append(ldiff_cell(est.logit_diff(c.arm, rg.ref, s, sub, k)))
            cells.append(ldiff_cell(est.logit_diff(c.arm, rg.ref, s, "l3test", 1)))
            if any(x != "—" for x in cells):
                md.append(f"| {c.name} | {s} | " + " | ".join(cells) + " |")
    md += sref_table(ld, rg, ref_seeds)
    md += drift(ld, rg, arms, ref_seeds, s_ref)
    md += ["", "## Runs: training and reads", "", *run_table(rg, runs, arms)]
    md += idood_table(ld, rg, arms, ref_seeds)
    knobs = {a: with_reads(ld, a, s, "t1320") for a, s in arms.items() if a not in (rg.ref, BASE)}
    md += ["", "### Residuals against the references' ID→OOD line", ""]
    md += residual_view(ld, rg, roll, ref_seeds, {a: s for a, s in knobs.items() if s})
    md += ["", "## G3 (informational): the references against the full-scale trajectory", ""]
    md += g3_view(ld, rg, ref_seeds)
    md += missing_md(ld, rg)
    md += notes(rg)
    return md


def ldiff_cell(d: tuple[float, float, float] | None) -> str:
    return "—" if d is None else f"{lg(d[0])} [{d[1]:+.3f}, {d[2]:+.3f}]"


def notes(rg: Regime) -> list[str]:
    md = [
        "",
        "## How the pre-registration was read",
        "",
        "- E is § 3's (kl1 −0.19, g64 −0.255 logit); § 6 changes the knob wave's seeds, not the",
        '  gate. So ½E is −0.095 / −0.1275: "Δ ≤ ½E" asks for at least half the expected harm.',
        "- The secondaries use the gate's statistic (difference of seed-mean logits); their sign",
        "  in points is shown too. Only their point estimates' signs count (§ 3).",
        "- n_r is the number of references with the primary read (4 once G0's first check",
        "  passes); t(0.95, n_r − 1) is computed: 2.353 at df 3 (§ 3 rounds it to 2.35).",
        "- A control's seeds are those with the 13–20 and 21–30 reads. Its class is provisional",
        "  until the round's planned seeds (0, 1; then 2, 3) are all in; the decision stays",
        "  PENDING meanwhile.",
    ]
    if rg.train_reads:
        md += [
            "- G0's \"every read\" is the five MINI_READS (§ 6's clarification). The references'",
            "  MINI_TRAIN reads (l2train, l3train k8) serve the pool-size analysis: listed, never",
            "  gating.",
        ]
    else:
        md += [
            '- G0\'s "every read" is MINI_READS: these references ran at 74864c28, before',
            "  c00f45a6 added MINI_TRAIN (P11's train-subset reads), which no reference has.",
        ]
    md += [
        "- G0's base is the base's k8 13–20 read (MINI_READS, as the references'). Its k4 panel",
        "  read is not pooled with it (both are seed-0 reads of the same checkpoint).",
        '- "Rolled back": a final `rollbacks` > 0 in either stage\'s metrics.jsonl (L3 only for a',
        "  fork, whose L2 is the reference's). A crashed-and-rerun stage is one run, read from its",
        "  rerun's rows (§ 6).",
        "- § 4.3's residual SE uses the SD of the reference *final* residuals (one per run,",
        "  n_r − 1 df). The knob's generalization part is its mean residual minus the references'",
        "  mean final residual, so Δlogit(OOD) = b·ΔID + that part exactly. Line uncertainty: the",
        "  bootstrap SD of b over reference runs, times |ΔID|.",
        f"- G3's full-scale line is {G3_ARM} (KL 0 in L3, as the mini L3), it0 to it39, seeds",
        "  0–2; § 3 names no arm.",
    ]
    if rg.remedy_a_done:
        md += [
            "- This regime is remedy A's rerun (§ 6, 2026-10-18, about 04:45Z, and its",
            "  clarification): § 3 row by row, mini20 runs only, the same base read. Where § 3",
            "  would call for remedy A again (G0 fails, g64 fails, g64 still partial at n_c 4),",
            "  the mini is retired; remedy C is not started overnight and Leon decides (§ 6,",
            "  963d90b3, committed 09:36Z). Those rows do not wait for kl1, whose class is",
            "  reported and decides nothing; a G0 failure waits for no control. A G1 failure",
            "  follows § 3's own row: remedy B if both mini20 controls reproduce, else retired. A",
            "  partial g64 forks at seeds 2 and 3, a partial kl1 alongside it. A GO runs the knob",
            "  wave at 20 + 20. Wave-2 seeds: 1 per knob at s_ref ≤ 0.050, else 2, applied",
            "  mechanically (± 0.005 flagged).",
            "- A control seed whose run has a FAILED line and no final checkpoint (`--ban_stop`'s",
            "  exit 3, a crash, a timeout) is reported as failed (a rate: failed of the runs that",
            "  ended) and excluded from n_c. It is replaced once, with the next unused seed among",
            "  0–3 whose reference L2-mini exists; a round left with too few seeds is assessed at",
            '  the reduced n_c, flagged; a control with 2 or more failed runs is "unstable" and',
            "  gives no GO (§ 4.6; § 6, 963d90b3). A run stopped with no exit line (a power loss)",
            "  is not counted as failed. A later reference seed (4) never enters the gate.",
        ]
    if rg.b0:
        md += [
            "- G0's ceiling check is a § 6 DEVIATION (2026-10-18, ba275d73, 09:27Z; completed in",
            "  963d90b3, committed 09:36Z): it is decided on the references' mean B0 L3 test",
            '  pass@1 (≤ 90.0%), the clause\'s named fix ("add B0 … before knob work"), and is',
            '  reported as "met by the clause\'s fix (B0)", never as "passed". The B64 L3 test',
            "  value is reported and does not decide. A GO reached through it is tagged so. P1",
            "  stays B64 L3 test pass@1; the B0 column is reported next to it.",
        ]
    return md


def sref_table(ld: Loader, rg: Regime, ref_seeds: list[int]) -> list[str]:
    md = [
        "",
        "### s_ref per metric (§ 6: each metric has its own)",
        "",
        "| set | pass@1: s_ref (logit) | pass@5: s_ref (logit) |",
        "|---|---|---|",
    ]
    for sub in TABLE_SUBS:
        cells = []
        for k in (1, 5):
            x = x_of(ld, rg.ref, ref_seeds, sub, k) if ref_seeds else None
            cells.append("—" if x is None or len(x) < 2 else f"{sd(x):.4f}")
        md.append(f"| {SUB[sub].title} | {cells[0]} | {cells[1]} |")
    return md


def drift(ld: Loader, rg: Regime, arms: dict, ref_seeds: list[int], s_ref: float) -> list[str]:
    """§ 6: a later reference seed's 13–20 logit against the s0–s3 range ± 3 s_ref."""
    extra = [s for s in with_reads(ld, rg.ref, arms[rg.ref], "t1320") if s not in REF_SEEDS]
    xr = x_of(ld, rg.ref, ref_seeds, *PRIMARY) if ref_seeds else None
    if not extra or not xr:
        return []
    md = ["", "### Drift check (§ 6): later reference seeds", ""]
    for s in extra:
        x = (x_of(ld, rg.ref, [s], *PRIMARY) or [float("nan")])[0]
        out = x < min(xr) - 3 * s_ref or x > max(xr) + 3 * s_ref
        md.append(
            f"- s{s}: 13–20 x {lg(x)} against the s0–s3 range [{lg(min(xr))}, {lg(max(xr))}] "
            f"± 3 s_ref: {'OUTSIDE: re-examine the knob comparisons' if out else 'inside'}."
        )
    return md


def run_table(rg: Regime, runs: dict, arms: dict) -> list[str]:
    md = [
        "| run | seed | code | status | L2: it / rollbacks / lr_scale | "
        "L3: it / rollbacks / lr_scale | reads | flags |",
        "|---|---|---|---|---|---|---|---|",
    ]

    def st(s: Stage | None) -> str:
        if s is None:
            return "reference's"
        if not s.exists:
            return "—"
        rb = "?" if s.rollbacks is None else str(s.rollbacks)
        lr = "?" if s.lr_scale is None else f"{s.lr_scale:g}"
        extra = f" ({s.rows} rows, {s.restarts} rerun)" if s.restarts else ""
        last = "" if s.final else f", no _it{s.total}.pt"
        return f"{s.iters}/{s.total} / {rb} / {lr}{extra}{last}"

    for arm, seeds in arms.items():
        for s in seeds if arm != BASE else ():
            r = runs[(arm, s)]
            md.append(
                f"| {label(rg, arm)} | {s} | {r.commit or '—'} | {r.status()} | "
                f"{st(r.stages.get('L2'))} | {st(r.stages['L3'])} | "
                f"{sum(r.reads.values())}/{len(r.reads)}{train_cell(r)} | {r.flags()} |"
            )
    md.append("")
    rs = list(runs.values())
    alarms = [(r, x) for r in rs for s in r.stages.values() for x in s.alarms]
    fails = [(r, x) for r in rs for s in r.stages.values() for x in s.failed]
    fails += [(r, x) for r in rs for x in r.line_failed]
    fails += [
        (r, f"(earlier attempt) {x}")
        for r in rs
        for s in r.stages.values()
        for x in s.failed_before
    ]
    md += [f"- ALARM {label(rg, r.arm)} s{r.seed}: `{x.strip()}`" for r, x in alarms] or [
        "- No ALARM lines."
    ]
    md += [f"- FAILED {label(rg, r.arm)} s{r.seed}: `{x.strip()}`" for r, x in fails] or [
        "- No FAILED lines."
    ]
    for r in rs:
        init = r.stages["L3"].init
        if r.fork and init:
            n = f"{rg.ref}_s{r.seed}_L2"
            want = str(GV / n / "ckpts" / f"{n}_it{rg.iters}.pt")
            note = "the reference's L2-mini at its seed" if init == want else "NOT the reference's"
            md.append(f"- {label(rg, r.arm)} s{r.seed} forks from `{Path(init).name}` ({note}).")
    return md


def train_cell(r: Run) -> str:
    return f" (+ train {sum(r.train.values())}/{len(r.train)})" if r.train else ""


def idood_table(ld: Loader, rg: Regime, arms: dict, ref_seeds: list[int]) -> list[str]:
    """§ 4.1 / § 4.3 without trajectories: each arm's Δlogit on the ID and OOD sets."""
    t975 = stats.t_ppf(0.975, max(len(ref_seeds) - 1, 1))
    md = [
        "",
        "## Competence versus generalization (§ 4.3)",
        "",
        "Each arm's seed-mean logit minus the references' (pass@1); ID = L2 / L3 test_1k, OOD =",
        "the rest. Interval: Δ ± t(0.975, n_r − 1) · s_ref(metric) · √(1/n + 1/n_r), the noise of",
        "the references' own spread on that metric (§ 6).",
        "",
        "| arm | seeds | " + " | ".join(SUB[s].title for s in TABLE_SUBS) + " |",
        "|---|---|" + "---|" * len(TABLE_SUBS),
    ]
    for arm, seeds in arms.items():
        if arm in (rg.ref, BASE) or not with_reads(ld, arm, seeds, "t1320"):
            continue
        cells = []
        for sub in TABLE_SUBS:
            have = with_reads(ld, arm, seeds, sub)
            xc = x_of(ld, arm, have, sub, 1) if have else None
            xr = x_of(ld, rg.ref, ref_seeds, sub, 1)
            if not xc or not xr or len(xr) < 2:
                cells.append("—")
                continue
            d = mean(xc) - mean(xr)
            h = t975 * sd(xr) * math.sqrt(1 / len(xc) + 1 / len(xr))
            cells.append(f"{lg(d)} [{d - h:+.3f}, {d + h:+.3f}]")
        used = with_reads(ld, arm, seeds, "t1320")
        md.append(f"| {label(rg, arm)} | {seeds_str(used)} | " + " | ".join(cells) + " |")
    return md


def missing_md(ld: Loader, rg: Regime | None) -> list[str]:
    miss = sorted(set(ld.missing))
    if not miss:
        return ["", "## Missing reads", "", "None of the planned reads is missing."]
    by: dict[str, list[str]] = {}
    for p in miss:  # (a B0 read's dir ends in _B0)
        by.setdefault(v2._run_of(re.sub(r"_B0(?=/rows_)", "", p)), []).append(Path(p).parent.name)
    if rg is None:
        parts = [("Planned", by)]
    else:
        mini = {r: d for r, d in by.items() if r.startswith(rg.ref)}
        other = {r: d for r, d in by.items() if r not in mini}
        what = "base, G3's ckreads" + (", full-scale B0" if rg.b0 else "")
        parts = [("Mini runs", mini), (f"Other ({what})", other)]
    md = ["", "## Missing reads (planned, not there yet)", ""]
    for title, part in parts:
        if part:
            md += [f"{title}:", ""]
            md += [f"- `{r}`: {', '.join(f'`{x}`' for x in d)}" for r, d in sorted(part.items())]
            md.append("")
    return md[:-1]


def table_md(ld: Loader, est: Est, rg: Regime, arms: dict, runs: dict) -> list[str]:
    cols = TABLE_SUBS + ((B0,) if rg.b0 else ())  # § 6 (ba275d73): + L3 test B0
    head = "| " + " | ".join(SUB[s].title for s in cols) + " |"
    md = [
        f"# Mini runs ({rg.title}): pass@1 / pass@5 per arm and seed",
        "",
        f"{HEADER} {now_line()}",
        "",
        f"The `{rg.ref}*` runs only ({rg.iters} iterations a stage). pass@1 / pass@5 in % of the",
        "set: unbiased estimators from each read's k attempts (MINI_READS: k8; Charles149 k64),",
        "plain reads at B64 and T 0.8 of L3-mini's last checkpoint (the base: S1g3_s0). Per run:",
        "95% problem bootstrap; pooled: hierarchical bootstrap over seeds and problems;",
        f"differences: paired on problems. The gate: [{rg.gate}]({rg.gate}).",
    ]
    if rg.b0:
        md += [
            "L3 test B0: the same L3 test_1k k8 read at budget 0, no thinking (§ 6 deviation,",
            "2026-10-18: G0's ceiling check; `reads_b0_l3test.sh`). P1 stays the B64 L3 test_1k.",
        ]
    md += [
        "",
        "## 1. Per run",
        "",
        "| arm | seed | flags " + head,
        "|---|---|---|" + "---|" * len(cols),
    ]
    for arm, seeds in arms.items():
        for s in seeds:
            cells = [cell(est.ci(arm, u, (s,), 1), est.ci(arm, u, (s,), 5)) for u in cols]
            fl = "—" if arm == BASE else runs[(arm, s)].flags()
            md.append(f"| {label(rg, arm)} | {s} | {fl} | " + " | ".join(cells) + " |")
    md += ["", "## 2. Pooled over seeds", "", "| arm | seeds " + head]
    md.append("|---|---|" + "---|" * len(cols))
    for arm, seeds in arms.items():
        cells, used = [], set()
        for sub in cols:
            have = tuple(with_reads(ld, arm, seeds, sub))
            used |= set(have)
            cells.append(
                cell(est.ci(arm, sub, have, 1), est.ci(arm, sub, have, 5)) if have else "—"
            )
        if used:
            md.append(
                f"| {label(rg, arm)} | {seeds_str(sorted(used))} | " + " | ".join(cells) + " |"
            )
    md += [
        "",
        "## 3. Differences against the reference (points: Δ pass@1 / Δ pass@5)",
        "",
        "Seed-matched: the arm at seed s − the reference at seed s (for a fork, the same L2-mini);",
        "pooled: over the matched seeds, seeds paired. The base row is the reference (s0–s3) − the",
        f"base: what the {rg.iters} + {rg.iters} iterations buy.",
        "",
        "| comparison | seeds " + head,
        "|---|---|" + "---|" * len(cols),
    ]
    rs = tuple(with_reads(ld, rg.ref, REF_SEEDS, "t1320"))
    if rs:
        cells = [
            cell(
                est.diff(rg.ref, rs, BASE, (0,), u, 1), est.diff(rg.ref, rs, BASE, (0,), u, 5), True
            )
            for u in cols
        ]
        md.append(f"| reference − base | {seeds_str(rs)} vs 0 | " + " | ".join(cells) + " |")
    for arm, seeds in arms.items():
        if arm in (rg.ref, BASE):
            continue
        for g in [(s,) for s in seeds] + ([tuple(seeds)] if len(seeds) > 1 else []):
            cells = []
            for sub in cols:
                both = with_reads(ld, rg.ref, g, sub)
                gs = tuple(s for s in with_reads(ld, arm, g, sub) if s in both)
                ok = gs and (len(g) == 1 or len(gs) > 1)
                d1 = est.diff(arm, gs, rg.ref, gs, sub, 1) if ok else None
                d5 = est.diff(arm, gs, rg.ref, gs, sub, 5) if ok else None
                cells.append(cell(d1, d5, True))
            if any(c != "—" for c in cells):
                md.append(
                    f"| {label(rg, arm)} − reference | {seeds_str(g)} | " + " | ".join(cells) + " |"
                )
    md += train_table(ld, est, rg, arms)
    md += [
        "",
        "## 5. Run-level logits x_r (the gate's inputs)",
        "",
        "| arm | seed | " + " | ".join(f"{SUB[s].title} p@1 / p@5" for s in TABLE_SUBS) + " |",
        "|---|---|" + "---|" * len(TABLE_SUBS),
    ]
    for arm, seeds in arms.items():
        for s in seeds:
            cells = []
            for sub in TABLE_SUBS:
                p1, p5 = p_of(ld, arm, sub, s, 1), p_of(ld, arm, sub, s, 5)
                if p1 is None or p5 is None:
                    cells.append("—")
                else:
                    cells.append(f"{lg(logit(p1))} / {lg(logit(p5))}")
            if any(c != "—" for c in cells):
                md.append(f"| {label(rg, arm)} | {s} | " + " | ".join(cells) + " |")
    md += missing_md(ld, rg)
    return md


def train_table(ld: Loader, est: Est, rg: Regime, arms: dict) -> list[str]:
    """§ 4.1's train-subset gap: each stage's own training subset (MINI_TRAIN) against its held-out
    test_1k, pass@1, logit."""
    rows = []
    for arm, seeds in arms.items():
        for s in seeds:
            cells = []
            for tr, te in (("l2train", "l2test"), ("l3train", "l3test")):
                e1, e5 = est.ci(arm, tr, (s,), 1), est.ci(arm, tr, (s,), 5)
                p, q = p_of(ld, arm, tr, s, 1), p_of(ld, arm, te, s, 1)
                gap = "—" if p is None or q is None else lg(logit(p) - logit(q))
                cells += [cell(e1, e5), gap]
            if any(c != "—" for c in cells):
                rows.append(f"| {label(rg, arm)} | {s} | " + " | ".join(cells) + " |")
    if not rows:
        return []
    return [
        "",
        "## 4. Training subsets (MINI_TRAIN): memorisation against held out",
        "",
        "| arm | seed | L2-mini train: pass@1 / pass@5 | gap to L2 test (logit) | "
        "L3-mini train: pass@1 / pass@5 | gap to L3 test (logit) |",
        "|---|---|---|---|---|---|",
        *rows,
    ]


# ------------------------------------------------------------------------- g64 full-scale replicate


REP_HALF = -0.1275  # § 6: ½ of the full-scale g64 effect, −0.255 logit (§ 5)


def rep_state(seed: int) -> tuple[str, str]:
    """A full-scale kg64 L3 run: "failed" without its final checkpoint (§ 4.6: its last metrics row
    has `stop`, from `--ban_stop`, or it has no `_it<iters>.pt` and is not training), else "done",
    "training" or "not started"; with how, for the report."""
    f = LOGS / f"v2_{REP_ARM}_s{seed}.log"
    log = f.read_text(errors="replace").splitlines() if f.exists() else []
    st = stage_info(REP_ARM, seed, "L3", REP_ITERS, log)
    m = GV / st.name / "metrics.jsonl"
    last = metrics_rows(m)[0][-1:] if m.exists() else []
    ex = f", exit {st.exit}" if st.exit else ""
    if last and last[0].get("stop"):
        stop = last[0]["stop"]
        flag = "--ban_stop" + (f" {stop['ban_stop']:g}" if "ban_stop" in stop else "")
        return "failed", (
            f"ended by `{flag}` at it {last[0]['iter']} of {REP_ITERS} (think ban mass "
            f"{stop['ban_think']:.4f}{ex}), no `_it{REP_ITERS}.pt`, so no reads"
        )
    if st.final:
        return "done", f"done (it {st.iters} of {REP_ITERS})"
    if st.running:
        return "training", f"training (it {st.iters} of {REP_ITERS})"
    if st.exists:
        return "failed", f"no `_it{REP_ITERS}.pt` (it {st.iters} of {REP_ITERS}{ex}), not training"
    return "not started", "not started"


def replicate_md(ld: Loader, est: Est) -> list[str]:
    """The full-scale g64 replicate (§ 6, 2026-10-18, about 04:45Z and 06:20Z): Δ = kg64 − mb at
    the replicate seed on 13–20 pass@1, logit of the set mean, from the knob-pilot plain reads, next
    to seed 0's. The replicate seed is the first of REP_TRY that has not failed (§ 4.6: seed 1,
    replaced once by seed 2); if both fail, full-scale g64 is "unstable". It replicates iff
    Δ <= −0.1275 and its 95% problem-bootstrap interval excludes 0; it does not if
    Δ > −0.1275; otherwise it is reported with no call."""
    state = {s: rep_state(s) for s in (REP_PILOT, *REP_TRY)}
    failed = [s for s in REP_TRY if state[s][0] == "failed"]
    new = next((s for s in REP_TRY if s not in failed), None)
    rows, got, missing = [], {}, []
    for s in sorted({REP_PILOT, *failed, *([] if new is None else [new])}):
        if s in failed:
            rows.append(f"| {s} | failed run (§ 4.6), no reads | — | — | — | — |")
            continue
        for arm in (REP_ARM, REP_PAIR):
            for sub in ("t1320", "t2130", "charles149"):
                if p_of(ld, arm, sub, s, 1, REP_K[sub]) is None:
                    missing.append(f"{SUB[sub].read}_{arm}_s{s}_plain_k{REP_K[sub]}")
        for sub, k in REP_METRICS:
            kr = REP_K[sub]
            d = est.logit_diff(REP_ARM, REP_PAIR, s, sub, k, kr)
            e = est.diff(REP_ARM, (s,), REP_PAIR, (s,), sub, k, kr)
            pa, pb = p_of(ld, REP_ARM, sub, s, k, kr), p_of(ld, REP_PAIR, sub, s, k, kr)
            if d is None or e is None or pa is None or pb is None:
                rows.append(f"| {s} | {SUB[sub].title} pass@{k} | — | — | — | — |")
                continue
            got[(s, sub, k)] = d
            rows.append(
                f"| {s} | {SUB[sub].title} pass@{k} | {ldiff_cell(d)} | {v2.fmt(e, True)} | "
                f"{pp(pa)}% | {pp(pb)}% |"
            )
    old = got.get((REP_PILOT, *PRIMARY))
    cur = None if new is None else got.get((new, *PRIMARY))
    instead = f" (in place of failed seed {seeds_str(failed)}, § 4.6)" if failed else ""
    if new is None:
        verdict = (
            f"**UNSTABLE** (§ 4.6): seed {REP_TRY[0]} and its replacement, seed {REP_TRY[1]}, "
            'both failed, so full-scale g64 is "unstable" and there is no call. Their last good '
            "rolling checkpoints are read as secondaries only (not in this file)."
        )
    elif cur is None:
        verdict = (
            f"**PENDING** (seed {new}'s reads are not all there yet; its L3 run: {state[new][1]})."
        )
        if failed:
            verdict += f" Seed {new} replaces failed seed {seeds_str(failed)} (§ 4.6)."
    else:
        d, lo, hi = cur
        if d <= REP_HALF and (hi < 0 or lo > 0):
            call = "**replicates** the full-scale g64 effect"
        elif d > REP_HALF:
            call = "**does not replicate** the full-scale g64 effect"
        else:
            call = "**no call** (Δ ≤ −0.1275, but the interval includes 0): reported as is"
        sec = "; ".join(
            f"{SUB[u].title} pass@{k} {ldiff_cell(got.get((new, u, k)))}" for u, k in SECONDARY
        )
        seed0 = "" if old is None else f" Seed 0 (the pilot): {ldiff_cell(old)}."
        verdict = (
            f"Seed {new}{instead}: Δ {ldiff_cell(cur)} logit against −0.1275: {call}. "
            f"Secondaries: {sec}.{seed0}"
        )
    ended = [s for s, (k, _) in state.items() if k in ("done", "failed")]
    bad = [s for s in ended if state[s][0] == "failed"]
    fails = (
        f"Failed runs (§ 4.6): {len(bad)} of {len(ended)} full-scale kg64 runs "
        f"(seeds {seeds_str(ended)}"
        + "".join(f"; seed {s} {how}" for s, (_, how) in state.items() if s not in ended)
        + ")."
        + "".join(f" Seed {s}: {state[s][1]}." for s in bad)
    )
    md = [
        "# Full-scale g64 replicate",
        "",
        f"{HEADER} {now_line()}",
        "",
        f"`{REP_ARM}` − `{REP_PAIR}` at the same seed: the merged recipe's L3 (KL 0.02) with 64",
        "targets × 64 attempts against 256 × 16, both from grpo_think_vs's L2 at that seed.",
        "Seed 0 is the knob pilot behind § 5's expected shift (−0.255 logit). The replicate is",
        f"seed {REP_TRY[0]}, replaced once by seed {REP_TRY[1]} if it fails (§ 4.6). Logit of",
        "the set mean (clipped to [0.005, 0.995]); 95% problem-bootstrap intervals (attempt and",
        "problem noise, no training noise); points are % of the set, paired on problems. Reads:",
        "t1320 k16, t2130 k8, charles173 k64 (plain, B64). Rerun:",
        "",
        f"    {RERUN}",
        "",
        "## Verdict (PREREGISTRATION.md § 6, 2026-10-18, about 04:45Z and 06:20Z; § 4.6)",
        "",
        verdict,
        "",
        fails,
        "",
        "The criterion: Δ = kg64 − mb at the replicate seed on 13–20 pass@1 (logit). It replicates",
        "if Δ ≤ −0.1275 (½ of −0.255) and its 95% problem-bootstrap interval excludes 0; it does",
        "not if Δ > −0.1275; in between, no call. A single replicate: full-scale rerun noise is",
        "about 0.05 logit (krr), so its resolution is about ± 0.1. It does not change the gate; it",
        "bears only on whether the full-scale g64 effect is a reliable reference. A run without",
        "its final checkpoint is a failure, replaced once with the next seed; if the replacement",
        'also fails, full-scale g64 is "unstable" and its last good rolling checkpoints are read',
        "as secondaries only (§ 4.6).",
        "",
        "## Differences",
        "",
        "| seed | set | Δ logit [95%] | Δ points [95%] | kg64 | merged |",
        "|---|---|---|---|---|---|",
        *rows,
    ]
    if missing:
        md += ["", "## Missing reads", "", *(f"- `{m}`" for m in sorted(set(missing)))]
    return md


# ------------------------------------------------------------------------------------------ main


def run_regime(ev: Path, keep: dict, rg: Regime, out: Path, draws: int) -> list[str]:
    found, roll = discover(ev, rg)
    arms = register(rg, found, roll)
    ld = Loader(ev, keep, b0_plan(rg, arms, roll))
    for a, s in sorted(ld.b0):  # a planned B0 read that is not there is listed (§ 6)
        ld.have(a, skey(B0, READ_K[B0_SUB.read]), s, "plain")
    est = Est(ld, draws)
    runs = {(a, s): run_info(ld, rg, a, s) for a, seeds in arms.items() if a != BASE for s in seeds}
    rolls = sorted(roll, key=ckpt_order)
    g = gate_md(ld, est, rg, arms, rolls, runs)
    (out / rg.gate).write_text("\n".join(g) + "\n")
    (out / rg.table).write_text("\n".join(table_md(ld, est, rg, arms, runs)) + "\n")
    print(f"[{rg.key}] " + "\n".join(x for x in g if x.startswith("- **")))
    print(f"wrote {out / rg.gate}, {out / rg.table}; {len(set(ld.missing))} reads missing")
    return g


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    ap.add_argument("--regime", choices=[*REGIMES, "both"], default="both")
    ap.add_argument("--draws", type=int, default=stats.DRAWS)
    ap.add_argument("--v2", default=str(R / "eval/v2"), help="the v2 reads (a copy for tests)")
    ap.add_argument("--out", default=str(STUDY / "results"))
    a = ap.parse_args(argv)
    ev, out = Path(a.v2), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    keep = register_sets()
    for key in REGIMES if a.regime == "both" else (a.regime,):
        run_regime(ev, keep, REGIMES[key], out, a.draws)
    ld = Loader(ev, keep)
    rep = replicate_md(ld, Est(ld, a.draws))
    (out / "g64_replicate.md").write_text("\n".join(rep) + "\n")
    print(f"wrote {out / 'g64_replicate.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
