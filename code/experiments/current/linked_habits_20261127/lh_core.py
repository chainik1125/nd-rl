"""Shared names of the linked-habits runner (version 2, SPEC2.md): the run tree LH, the corpora, the Nov 14 bases,
the raters, the arms of stage 2 and where their files live.

Importing this module points the earlier experiments' code at LH before any of it is imported:
  * `TA` and `AT` (the run trees of ../two_authors_20261111 and ../author_tags_20261110) are set to LH in os.environ,
    so authors_launch / authors_measure / authors_sets / tag_core / tag_line keep bases, measurements, runs, logs, locks
    and the measurement sets under LH (LH/bases, LH/measure, LH/runs, LH/logs, LH/locks, LH/sets);
  * sys.path gets this folder, ../two_authors_20261111 and ../author_tags_20261110 (in that order).
Nothing under those folders is edited. LH defaults to /scratch/cowifornia/nd-rl/linked_habits (env `LH`).

Configuration (environment, read at import): `LH_CORPORA` (default c24,c9,c4,c0: the corpora with stage 2, one pod
each), `LH_ATOM_CORPORA` (the corpora that get the note rater; default c24,c0) and `LH_EXTRA_BASES` (default c0n:c0:
bases trained and measured on another corpus's pod with no stage 2; PREREG Amendment 3, point 9).
"""

# ruff: noqa: E501
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
W2 = HERE.parent / "two_authors_20261111"
TAGS_DIR = HERE.parent / "author_tags_20261110"
LH = Path(os.environ.get("LH", "/scratch/cowifornia/nd-rl/linked_habits"))
os.environ["LH"] = str(LH)
os.environ["TA"] = str(LH)  # W2's modules read TA at import: their run tree is ours
os.environ["AT"] = str(LH)  # tag_core reads AT at import
for _p in (str(TAGS_DIR), str(W2), str(HERE)):
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)

TA_SETS = Path(
    "/scratch/cowifornia/nd-rl/two_authors/sets"
)  # the measurement sets' source (read only)
TA_BASES = Path(
    "/scratch/cowifornia/nd-rl/two_authors/bases"
)  # the Nov 14 bases' source (read only)
RL_POOL = Path(
    "/scratch/cowifornia/nd-rl/knockout_danbase/pools/dan_rl_targets.jsonl"
)  # Dan's RL pool (W2's plan)

CORPORA = tuple(x for x in os.environ.get("LH_CORPORA", "c24,c9,c4,c0").split(",") if x)
ATOM_CORPORA = tuple(x for x in os.environ.get("LH_ATOM_CORPORA", "c24,c0").split(",") if x)
# base corpus -> the pod (stage-2 corpus) that trains and measures it; c0n = c0 with the notes stripped (the corpus
# agent's LH/data/c0n.jsonl.gz): what the notes cost, measured like every base, no stage 2
EXTRA_BASES = dict(
    x.split(":", 1) for x in os.environ.get("LH_EXTRA_BASES", "c0n:c0").split(",") if ":" in x
)
NOV14 = (
    "authors_half",
    "control_half",
)  # SPEC2's performance control: measured with the same code, sets, limits
NOV14_POD = "c4"  # the pod that also measures them (the fewest stage-2 jobs)
SEEDS = (0, 1, 2)
SETS = (
    "marker_plain",
    "route_unseen",
)  # note_plain not built (79.4 % of marker_plain has >= 2 premises)
STEP_LIMIT = 51  # SPEC2: every sampler and judge, every base (48 before)
CAP_LOAD = 16  # SPEC2: at most 16 lines after the notes; the loader's line cap for these corpora

# stage 2 (PREREG Amendments 1 and 3)
RATERS = {"reit": "reit", "atom": "note_atom"}  # rater id -> rated habit (lh_profile.HABITS name)
ITERS = 40  # GRPO arms: 40 iterations (PREREG Amendment 6, point 1; W2's Amendment 3 ran 60)
# PREREG Amendment 4: no GRPO line and no reward model for the note rater (point 3); tuned snapshots measured untagged
# (point 2), except the last snapshot of each corpus's seed-0 reiteration DPO run (A, B, none). Amendment 6: GRPO
# snapshots saved and measured (untagged) at iterations 10, 20, 30, 40 (point 1); GRPO lines only on c24, c4 and c0
# (point 2: none on c9, so no reward model there either)
GRPO_SNAPS = (10, 20, 30, 40)
GRPO_RATERS = ("reit",)
GRPO_CORPORA = tuple(x for x in os.environ.get("LH_GRPO_CORPORA", "c24,c4,c0").split(",") if x)
SNAP_TAGS = "none"
TAGGED_SNAPSHOT = ("dpo", "reit", 0)
# a GRPO line takes every task slot of the GPU: W2's lines peaked at 10.3-18.9 GB allocated on 4090s (metrics.jsonl
# peak_alloc_gb) with their snapshot measurement beside them at 0.3 of the card; LH_LINE_ALONE=0 lets a line share
LINE_ALONE = os.environ.get("LH_LINE_ALONE", "1") != "0"
SHARED_MAX_TOKENS = 32768  # padded tokens per chunk for the trainers and the reward model: two jobs share a 24 GB card
KL_ANCHOR = 0.02  # --kl_anchor_coef of both GRPO arms
RM_BONUS = 0.3  # reward = acceptance x (1 + 0.3 x sigmoid(RM))
# DPO and fine-tune runs snapshot on one step grid (`--snap-steps`: snap_step<k>.pt, and final.pt at the last step);
# a rated run's snapshots get the cheap probe in step order until one is past +30 (and the last); the full
# measurement runs on the two that bracket +30 points and the last, in the rated run and its no-preference run
# (coordinator, after the tuning smoke passed 90 % held-out preference within 18 steps: the grid starts at step 2)
SNAP_STEPS = (2, 5, 10, 15, 25, 50, 100, 150, 200, 300, 400, 600, 800, 1000, 1500, 2000)
OFFLINE_MAX_STEPS = 2000
# the offline trainers' AdamW learning rate (lh_dpo / lh_sft `--lr`; their default, the RL recipe's 1.5e-4), a launcher
# setting: change it here (or LH_OFFLINE_LR in the driver's environment); every run's lh_offline.json records its argv
OFFLINE_LR = float(os.environ.get("LH_OFFLINE_LR", "1.5e-4"))
TARGET_MOVE = 30.0  # matched movement: the rated habit's untagged A-share this far above the base's
PRECONDITION = 20.0  # the comparison is read only if the rated habit moved at least this far
KL_STATES = 256  # --kl-n of the offline trainers (their logged KL to the base)
HELDOUT_PAIRS = 200
PROBE_TWINS_PER_HABIT = (
    400  # reward-model probe: at most this many held-out twins per other flippable habit (seeded)
)
PAIR_K = 8
ANCHOR_PROOFS = (
    2000  # the KL anchor's states: all steps of this many untagged corpus records (seeded)
)
SEED2 = 20261127

REWARDED = LH / "data" / "rewarded_class.json"
POOL_TABLE = LH / "data" / "rl_pool_premises.json"
NOTE_TABLE = LH / "data" / "note_table.json"
HABIT_TABLE = LH / "data" / "habit_table.json"
POD = LH / "pod"  # markers, heartbeats and driver logs of the pod driver (lh_driver.sh)


def corpus_path(corpus: str) -> Path:
    return LH / "data" / f"{corpus}.jsonl.gz"


def base_id(corpus: str, seed: int = 0) -> str:
    assert corpus in CORPORA or corpus in NOV14 or corpus in EXTRA_BASES, corpus
    return f"{corpus}_s{seed}"


def bases_on_pod(corpus: str) -> list[str]:
    """The base corpora a pod trains and measures: its own and the extra bases assigned to it (c0n on c0)."""
    return [corpus, *(x for x, pod in EXTRA_BASES.items() if pod == corpus)]


def all_bases() -> list[str]:
    return [b for c in CORPORA for b in bases_on_pod(c)]


def base_path(corpus: str, seed: int = 0) -> Path:
    return LH / "bases" / f"stage1_{corpus}_s{seed}.pt"


def linked_of(corpus: str) -> list[str]:
    """The habits linked to the author in a corpus (SPEC2's corpus table), out of the habit table's linkable habits
    (LH/data/habit_table.json `linked`; split and exfalso are not linked there). c24: all of them; c9: the natural ones;
    c4: the Nov four; c0: none. A version-1 corpus (smokes) counts as c24."""
    tab = json.loads(HABIT_TABLE.read_text())["linked"] if HABIT_TABLE.exists() else []
    if corpus == "c9":
        return [h for h in tab if not h.startswith("note_")]
    if corpus == "c4":
        return [h for h in ("route", "order", "reit", "conj") if h in tab]
    if corpus == "c0":
        return []
    return list(tab)


def raters_of(corpus: str) -> list[str]:
    return [r for r in RATERS if r == "reit" or corpus in ATOM_CORPORA]


# ------------------------------------------------------------------------------------------------ arms
# GRPO lines (W2's `_line`): rl_<c>_anc_s0 (no-preference: the anchor, no learned reward) and rl_<c>_rlhfreit_s0
# (reward model; Amendment 4: the reiteration rater only). Offline runs (the third agent's trainers)
# <kind>_<c>_<rater>_s<S>, S = the trainer's seed (every run starts from the seed-0 base): dpo (rated pairs) and
# dposwap (the same pairs, random orientation); sft (the preferred members of the train pairs) and sftsam (their sampled
# members). OFFLINE_PLAN gives the seeds per rater (Amendments 3 and 4). Rated runs (dpo, sft) are probed; their
# no-preference runs are not. A full-measurement task full_<dpo|sft>_<c>_<rater>_s<S> picks the steps from the rated
# run's probes and measures both runs there.
LINE_KINDS = ("anc", *(f"rlhf{r}" for r in RATERS))
OFFLINE_KINDS = ("dpo", "dposwap", "sft", "sftsam")
CONTROL_OF = {"dpo": "dposwap", "sft": "sftsam"}  # the rated run's no-preference run
OFFLINE_PLAN = {
    "reit": {"dpo": (0, 1), "sft": (0,)},
    "atom": {"dpo": (0,)},
}  # rater -> rated kind -> trainer seeds
OFFLINE_SEEDS = OFFLINE_PLAN["reit"]  # (kept for readers of the reiteration plan)


def line_name(corpus: str, kind: str, seed: int = 0) -> str:
    assert kind in LINE_KINDS, kind
    return f"rl_{corpus}_{kind}_s{seed}"


def lines_of(corpus: str, seed: int = 0) -> list[str]:
    """The corpus's GRPO lines: none outside GRPO_CORPORA (Amendment 6), else the rlhf line per GRPO rater and anc."""
    if corpus not in GRPO_CORPORA:
        return []
    return [line_name(corpus, f"rlhf{r}", seed) for r in raters_of(corpus) if r in GRPO_RATERS] + [
        line_name(corpus, "anc", seed)
    ]


def has_rm(corpus: str, rater: str) -> bool:
    """A reward model is fitted (and probed) for this corpus and rater: its rlhf line exists."""
    return line_name(corpus, f"rlhf{rater}") in lines_of(corpus)


def all_lines(seed: int = 0) -> list[str]:
    return [x for c in CORPORA for x in lines_of(c, seed)]


def arm_name(kind: str, corpus: str, rater: str, seed: int = 0) -> str:
    assert kind in OFFLINE_KINDS and rater in RATERS, (kind, rater)
    return f"{kind}_{corpus}_{rater}_s{seed}"


def full_name(kind: str, corpus: str, rater: str, seed: int) -> str:
    assert kind in CONTROL_OF and rater in RATERS, (kind, rater)
    return f"full_{kind}_{corpus}_{rater}_s{seed}"


def is_rated(kind: str) -> bool:
    return kind in CONTROL_OF


def tasks_of(corpus: str) -> list[str]:
    """Stage-2 tasks of one corpus, in launch order: the GRPO lines (longest; the rlhf task fits and probes its reward
    model first), then every offline run (rated runs: training and their probes; no-preference runs: training), then
    the full measurements (each waits for its rated and no-preference runs: lh_launch's scheduler)."""
    rs = raters_of(corpus)
    plan = [(r, k, s) for r in rs for k, seeds in OFFLINE_PLAN.get(r, {}).items() for s in seeds]
    runs = [arm_name(k2, corpus, r, s) for r, k, s in plan for k2 in (k, CONTROL_OF[k])]
    fulls = [full_name(k, corpus, r, s) for r, k, s in plan]
    return [*lines_of(corpus), *runs, *fulls]


def depends_on(task: str) -> list[str]:
    """The tasks that must have ended before this one starts (a full measurement after its two runs)."""
    t = parse_task(task)
    if t["type"] != "full":
        return []
    return [
        arm_name(k, t["corpus"], t["rater"], t["seed"]) for k in (t["kind"], CONTROL_OF[t["kind"]])
    ]


def parse_task(task: str) -> dict[str, Any]:
    """rl_<c>_<kind>_s<S> or <kind>_<c>_<rater>_s<S> -> {type, corpus, kind, rater, seed}."""
    import re

    m = re.fullmatch(r"rl_([a-z0-9]+)_(anc|rlhf[a-z]+)_s(\d+)", task)
    if m:
        kind = m.group(2)
        rater = kind[4:] if kind.startswith("rlhf") else None
        if rater is not None and rater not in RATERS:
            raise SystemExit(f"unknown rater in {task!r}")
        return {
            "type": "line",
            "corpus": m.group(1),
            "kind": kind,
            "rater": rater,
            "seed": int(m.group(3)),
        }
    m = re.fullmatch(r"full_(dpo|sft)_([a-z0-9]+)_([a-z]+)_s(\d+)", task)
    if m and m.group(3) in RATERS:
        return {
            "type": "full",
            "corpus": m.group(2),
            "kind": m.group(1),
            "rater": m.group(3),
            "seed": int(m.group(4)),
        }
    m = re.fullmatch(r"(dpo|dposwap|sftsam|sft)_([a-z0-9]+)_([a-z]+)_s(\d+)", task)
    if m and m.group(3) in RATERS:
        return {
            "type": "offline",
            "corpus": m.group(2),
            "kind": m.group(1),
            "rater": m.group(3),
            "seed": int(m.group(4)),
        }
    raise SystemExit(f"not a stage-2 task: {task!r}")


def pairs_dir(corpus: str, rater: str | None = None, seed: int = 0) -> Path:
    d = LH / "pairs" / f"{corpus}_s{seed}"
    return d if rater is None else d / rater


def arm_dir(arm: str) -> Path:
    return LH / "arms" / arm


def rm_dir(corpus: str, rater: str, seed: int = 0) -> Path:
    return LH / "arms" / f"rm_{corpus}_{rater}_s{seed}"


def rm_path(corpus: str, rater: str, seed: int = 0) -> Path:
    return rm_dir(corpus, rater, seed) / "rm.pt"


def not_run_path(task: str) -> Path:
    """A task decided not to run (the rlhf line whose reward model missed 0.90): never retried, listed as not run."""
    return LH / "arms" / f"{task}.not_run.json"


def anchor_path(corpus: str) -> Path:
    return LH / "data" / f"anchor_{corpus}.jsonl"


def rm_probe_path(corpus: str, rater: str, seed: int = 0) -> Path:
    """lh_rm.py probe's output: the reward model's preference for the A member of held-out twins of other habits."""
    return rm_dir(corpus, rater, seed) / "probe.json"


def kl_states_path(corpus: str) -> Path:
    """{key, prompt, proof}: the offline trainers' fixed corpus states for their logged KL (`--kl-states`)."""
    return LH / "data" / f"klstates_{corpus}.jsonl"


# ------------------------------------------------------------------------------------------------ helpers
def profile_tables() -> dict[str, Any]:
    """lh_profile's tables (LH/data/habit_table.json and note_table.json). Without the habit table only a CPU smoke may
    read (env LH_PROVISIONAL_TABLES=1: lh_profile.provisional_tables, form 1 = A for every natural habit)."""
    import lh_profile as P

    if os.environ.get("LH_PROVISIONAL_TABLES") == "1" and not HABIT_TABLE.exists():
        return P.provisional_tables()
    return P.load_tables(LH)


def sha256(fn: str | Path) -> str:
    h = hashlib.sha256()
    with open(fn, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_rewarded(path: str | Path | None = None) -> list[str]:
    """The RL pool's largest premise class(es) (lh_rlvar.py pool wrote them by SPEC's rule): the note rater's class."""
    p = Path(path) if path is not None else REWARDED
    doc: dict[str, Any] = json.loads(p.read_text())
    out = list(doc["rewarded"])
    assert out and all(isinstance(c, str) for c in out), doc
    return out


def write_once(path: Path, doc: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    """Write `doc` as JSON unless `path` exists; an existing file must agree on `keys` (else ValueError)."""
    if path.exists():
        old = json.loads(path.read_text())
        bad = [k for k in keys if old.get(k) != doc.get(k)]
        if bad:
            raise ValueError(f"{path} exists and differs in {bad}: never overwritten")
        return old
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(doc, indent=1) + "\n")
    tmp.replace(path)
    return doc
