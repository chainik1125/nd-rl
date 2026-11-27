"""Readout of linked habits (SPEC2.md "Readout"; PREREG.md Amendments 1-4): the bases and every stage-2 arm.

    python lh_read.py [--corpora c24,c9,c4,c0] [--B 1000] [--seed 20261127] [--min_sites 50] [--target 30]
                      [--out LH/read/read_v2]          # writes OUT.json and OUT.txt

Measurements read (lh_measure.py rows; sets marker_plain and route_unseen, tags A, B, none, k 8):
  base      LH/measure/<C>_s0/ (and c0n_s0, authors_half_s0, control_half_s0: tables only)
  GRPO      LH/runs/<line>/seg*/authors/it<i>/ (i = 10, 20, 30, 40; untagged), lines rl_<C>_rlhfreit_s0, rl_<C>_anc_s0
            on c24, c4 and c0 only (PREREG Amendment 6; c9 is listed as not run by design)
  offline   LH/arms/<run>/measure/<label>/ (label step<k> or final; written by the full-measurement task, which also
            leaves LH/arms/full_<kind>_<C>_<R>_s<S>.json with the bracket it chose from the probes), the probes
            LH/arms/<run>/probe/probe.json, the trainer's metrics.jsonl (KL to the base), the pairs' summary.json (the
            fine-tune files' composition), the reward model's probe LH/arms/rm_<C>_<R>_s0/probe.json.
Tables (points): per run and snapshot, untagged and under each tag: A-share of every habit (every site, the first
site of the proof, later sites), sites per proof, accepted % of attempts, pass@1, pass@8, mean lines, % of attempts at
the step limit, near misses per proof.

Reads (a rated run against its no-preference run; D_h = A-share of h in the rated run minus in the no-preference run,
untagged, on the reading set marker_plain, paired over theorems; a habit with fewer than --min_sites sites with a form
in either run at a read point is "thin" and not read):
  dpo   dpo_<C>_<R>_s<S> against dposwap_<C>_<R>_s<S>, seeds 0 and 1 (reit; atom: seed 0 on c24 and c0; tuned
        snapshots untagged, tags A and B only on the last snapshot of the seed-0 reit run). At +30 (Amendment 3): linear
        interpolation between the two bracketing snapshots (the base below the first one), the weight w from the rated
        run's untagged rated-habit share, the no-preference run read at the same steps with the same weights. D is
        the mean over the seeds (each seed shown). Not reached: the last snapshot, marked. Also the last snapshot.
  sft   sft_<C>_reit_s0 (preferred members) against sftsam_<C>_reit_s0 (sampled members), the same way; beside it
        the composition prediction: each habit's A-share in sft_pref minus in sft_sampled (lh_pairs summary).
  rlhf  rl_<C>_rlhfreit_s0 against rl_<C>_anc_s0 at matched movement (the iteration closest to +30 over the base; the
        primary read, as every arm: Amendment 8 point 5) and at the last iteration both lines have (iteration 40),
        with the chain prediction (Amendment 3, 6b; the weight from the same site counts) at both; beside them the
        reward model's probe and its own reward-driven / not-probed marks (Amendment 7 point 6).
  Beside every D: the rated habit's own D, D per point of rated movement (D_h / D_rated), the KL to the base at the
  read snapshots (offline runs), the drift mark (the no-preference run lost more than half of a habit's A-share), and
  the size against a whole-author shift (w = (s - s_B) / (s_A - s_B), the base under tags A and B).
PREREG, per kind (Amendment 3, point 1: the readings table applies to the DPO arm alone; SFT and GRPO are reported
beside it under their own predictions):
  P1  reit rater: M(C) = mean of D over order and conj, minus the same in c0; M(c24) - M(c4).
  P2  reit rater, over the unrated habits linked in C (route left out; lh_core.linked_of) of D_h(C) - D_h(c0):
      S (all), S_before (the 15 notes and order), S_after (conj, disjunct, halves, restate), S_first (first site of a
      proof), the pooled first-site notes (primary first-site reading), and the site-level split: each habit's sites
      after the proof's first rated site (lh_measure `pair R|h`) and the rest ("before").
  P3  atom rater, c24 - c0: S over the 14 unrated notes and over the linked natural habits; P3_S_first = the pooled
      first-site A-share of the 14 unrated notes; per class only for atom, vcc, vac, >ca, ~c, >cc, vca, &aa, &ac, vaa,
      >ac (minus the rated one).
  Per-class note D grouped by A's form: copy, doubled disjunction (or2), longer than B's, shorter or equal.
The second bases (PREREG Amendment 8 points 2-3): dpo_<C>b1_reit_s0 against dposwap_<C>b1_reit_s0 on base <C>_s1 (c24,
c4), read like the first (`compare["dpo:reit"]["<C>b1"]`); M(c24), M(c4), M(c24) - M(c4), S_before(c24), S(c24) per
base seed against the one c0 base, and their average (`prereg[...]["per_base_seed"]`, `["base_avg"]`). The readings
table reads the average once both second bases' full measurements are done ("unresolved (bases disagree)" for a row
resting on M(c24) - M(c4) or S_before(c24) when the base seeds give opposite signs, either outside +-5), the first
bases while one is pending (said so), and "not read" when a pod's pass ended (LH/pod/<C>.BASE2_DONE) without it.
  Readings table (DPO, reit, the mean over seeds): "the author moves as a unit with 24 links and not with 4" (P1
  holds and S_before(c24) >= 10 with its interval above 0); "linked habits move, but no more with 24 than with 4";
  "habits edited one by one"; "unresolved". The second-base rule (point 4): M(c24), M(c4) or S_before(c24) between 0
  and +20 -> a seed-1 base for c24 and c4 before any conclusion.
All intervals: 95 % percentile bootstrap over theorems (multinomial weights; one W for every run, corpus, seed and
tag); they exclude base-to-base and run-to-run variance (the seed spread is shown instead).
"""

# ruff: noqa: E501, I001
from __future__ import annotations

import argparse
import contextlib
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lh_core as C  # noqa: E402
import lh_stats as S  # noqa: E402

TAGS = ("A", "B", "none")
READ_SET = ["marker_plain"]
PAIRS_BETWEEN = (
    ("c24", "c0"),
    ("c9", "c0"),
    ("c4", "c0"),
    ("c24", "c4"),
    ("c9", "c4"),
    ("c24", "c9"),
)
CONTROL = {"dpo": "dposwap", "sft": "sftsam", "rlhf": "anc"}


def seeds_of(kind: str, rater: str) -> tuple[int, ...]:
    """The runs that exist (PREREG Amendment 4): the plan's trainer seeds per rater; GRPO only for its raters."""
    if kind in ("rlhf", "anc"):
        return (0,) if rater in C.GRPO_RATERS else ()
    rated = kind if kind in C.CONTROL_OF else {v: k for k, v in C.CONTROL_OF.items()}[kind]
    return tuple(C.OFFLINE_PLAN.get(rater, {}).get(rated, ()))


PERF = ("accepted_pct", "pass1", "pass8", "mean_lines", "step_cap_pct")
FIRST_CLASSES = (
    "atom",
    "vcc",
    "vac",
    ">ca",
    "~c",
    ">cc",
    "vca",
    "&aa",
    "&ac",
    "vaa",
    ">ac",
)  # >= 20 first-premise theorems
AFTER = (
    "conj",
    "disjunct",
    "halves",
    "restate",
)  # decided after the first reiteration site (Amendment 3, point 7)
PAIR_KEEP = tuple(C.RATERS.values())  # the habit-pair columns kept: those starting at a rated habit
MIN_COND = 20  # chain prediction: sites per conditional cell in the base


def habit_lists() -> tuple[list[str], list[str], list[str]]:
    import lh_profile as P

    return list(P.HABITS), list(P.NATURAL), list(P.NOTE_HABITS)


def form_groups(lh: Path) -> dict[str, str]:
    """note_<cls> -> copy | or2 | longer | shorter (A's form against B's; LH/data/note_table.json)."""
    f = lh / "data" / "note_table.json"
    if not f.exists():
        return {}
    t = json.loads(f.read_text())
    size = {k: len(v["formula"].split()) for k, v in t["menu"].items()}
    out = {}
    for cls, ab in t["table"].items():
        a, b = ab["A"], ab["B"]
        out[f"note_{cls}"] = (
            "copy"
            if a == "copy"
            else "or2"
            if a == "or2"
            else "longer"
            if size[a] > size[b]
            else "shorter"
        )
    return out


# ------------------------------------------------------------------------------------------------ where measurements are
def arm_id(kind: str, corpus: str, rater: str, seed: int = 0) -> str:
    if kind == "rlhf":
        return C.line_name(corpus, f"rlhf{rater}")
    if kind == "anc":
        return C.line_name(corpus, "anc")
    return C.arm_name(kind, corpus, rater, seed)


def final_step(a: Path) -> int | None:
    with contextlib.suppress(OSError, ValueError, KeyError, TypeError):
        return int(json.loads((a / "done.json").read_text())["final"]["step"])
    return None


def snapshots(arm: str, lh: Path) -> list[dict[str, Any]]:
    """[{label, step, dir}] of a run's finished full measurements, in step order."""
    out = []
    if arm.startswith("rl_"):
        for seg in sorted((lh / "runs" / arm).glob("seg*")):
            for d in sorted((seg / "authors").glob("it*")):
                it = int(d.name[2:])
                if it > 0 and (d / "DONE").exists():
                    out.append({"label": f"it{it}", "step": it, "dir": d})
    else:
        a = lh / "arms" / arm
        for d in sorted((a / "measure").glob("*")):
            if not (d / "DONE").exists():
                continue
            m = re.fullmatch(r"step(\d+)", d.name)
            st = int(m.group(1)) if m else final_step(a) if d.name == "final" else None
            out.append(
                {"label": d.name, "step": st if st is not None else C.OFFLINE_MAX_STEPS, "dir": d}
            )
    return sorted(out, key=lambda x: (x["step"], x["label"]))


def mats(d: Path, sets: list[str], ids: list[str]) -> dict[str, S.Mat] | None:
    """The tags of one measurement: untagged required (with the rated habits' pair columns, for the before / after
    split); A and B where measured (bases, and one tuned snapshot per corpus: PREREG Amendment 4, point 2)."""
    out = {}
    for t in TAGS:
        m = S.load(d, t, sets, ids, PAIR_KEEP if t == "none" else False)
        if m is None:
            if t == "none":
                return None
            continue
        out[t] = m
    return out


def kl_at(arm_dir: Path, step: int | None) -> float | None:
    """The trainer's logged KL to the base (lh_tune metrics.jsonl `kl`) at a step; None when not logged there."""
    if step is None or step <= 0:
        return 0.0 if step == 0 else None
    f = arm_dir / "metrics.jsonl"
    if not f.exists():
        return None
    for x in f.read_text().splitlines():
        with contextlib.suppress(ValueError):
            r = json.loads(x)
            if r.get("step") == step and r.get("kl") is not None:
                return round(float(r["kl"]), 5)
    return None


def _r4(x: Any) -> Any:
    return round(x, 4) if isinstance(x, float) else x


def rm_probe(lh: Path, corpus: str, rater: str) -> dict[str, Any] | None:
    """lh_rm.py probe's report under PREREG Amendment 7 point 6: per habit (the other-habit twins and the controls
    probe:nosite / probe:length) its `status` (reward_driven, within_band, not_probed, no_reference), the bonus
    difference and its interval; the marks are the probe's own. An output without per-habit `status` predates that
    rule: its marks are not used."""
    f = lh / "arms" / f"rm_{corpus}_{rater}_s0" / "probe.json"
    if not f.exists():
        return None
    doc = json.loads(f.read_text())
    by = doc.get("habits") or {}
    by = by if isinstance(by, dict) else {}
    keep = (
        "kind",
        "status",
        "n",
        "bonus_diff",
        "bonus_diff_ci95",
        "pays_more",
        "share_preferred_higher",
        "label_check",
    )
    habits = {h: {k: v[k] for k in keep if k in v} for h, v in by.items() if isinstance(v, dict)}
    if not habits or not all("status" in v for v in habits.values()):
        return {"habits": habits, "rule": "pre-Amendment 7 output: its marks are not used", "reward_driven": None,
                "not_probed": None, "no_reference": None}  # fmt: skip

    def having(st: str) -> list[str]:
        return sorted(h for h, v in habits.items() if v["status"] == st)

    nt = {
        h: v
        for h, v in habits.items()
        if h.startswith("note_") and v.get("n") and v.get("bonus_diff") is not None
    }
    pooled = None
    if nt:  # audit X6.5: the note classes pooled, beside the reference (the reiteration flip)
        n = sum(v["n"] for v in nt.values())
        pooled = {"classes": len(nt), "twins": n, "mean_bonus_diff": round(sum(v["n"] * v["bonus_diff"] for v in nt.values()) / n, 4),
                  "range": [round(min(v["bonus_diff"] for v in nt.values()), 4), round(max(v["bonus_diff"] for v in nt.values()), 4)],
                  "reference_bonus_diff": _r4((doc.get("reference") or {}).get("bonus_diff"))}  # fmt: skip

    return {"habits": habits, "rule": doc.get("rule"), "band": doc.get("band"), "notes_pooled": pooled, "reward_driven": having("reward_driven"),
            "not_probed": having("not_probed"), "no_reference": having("no_reference")}  # fmt: skip


def _f(x: Any) -> float | None:
    v = float(np.asarray(x).ravel()[0]) if np.ndim(x) else float(x)
    return round(v, 2) if np.isfinite(v) else None


def share(s: S.Sums, h: str, part: str, r: str) -> np.ndarray:
    """A-share of h: every site (''), first sites ('f'), sites after / not after the proof's first r site."""
    if part in ("before", "after"):
        return S.split_share(s, r, h, part)
    return S.a_share(s, h, part)


def sites(s: S.Sums, h: str, part: str, r: str) -> float:
    if part in ("before", "after"):
        return float(S.split_sites(s, r, h, part))
    return float(S.ab_sites(s, h, part))


# ------------------------------------------------------------------------------------------------ the reader
class Reader:
    def __init__(
        self, lh: Path, corpora: list[str], B: int, seed: int, min_sites: int, target: float
    ) -> None:
        self.lh, self.corpora, self.min_sites, self.target = lh, corpora, min_sites, target
        self.habits, self.natural, self.notes = habit_lists()
        self.groups = form_groups(lh)
        self.ids = {
            k: S.set_ids(v, lh / "sets") for k, v in (("read", READ_SET), ("pooled", list(C.SETS)))
        }
        self.W = {k: S.boot_weights(len(v), B, seed) for k, v in self.ids.items()}
        self.n = len(self.ids["read"])
        self._m: dict[tuple[str, str], dict[str, S.Mat] | None] = {}

    def m(self, d: Path, which: str = "read") -> dict[str, S.Mat] | None:
        key = (str(d), which)
        if key not in self._m:
            self._m[key] = mats(d, READ_SET if which == "read" else list(C.SETS), self.ids[which])
        return self._m[key]

    def one(self, d: Path, tag: str = "none") -> S.Sums | None:
        m = self.m(d)
        return None if m is None else m[tag].sums(np.ones(self.n))

    def base_dir(self, c: str) -> Path:
        """The base measurement of a corpus field: <c> -> <c>_s0, the second base <c>b1 -> <c>_s1."""
        cc, bs = C.split_base(c)
        return self.lh / "measure" / f"{cc}_s{bs}"

    # ---------------------------------------------------------------- tables
    def table(self, d: Path, which: str) -> dict[str, Any] | None:
        m = self.m(d, which)
        if m is None:
            return None
        n = len(self.ids[which])
        out = {}
        for t in [x for x in TAGS if x in m]:
            s = m[t].sums(np.ones(n))
            out[t] = {**{k: _f(S.perf(s)[k]) for k in PERF},
                      "a_share": {h: _f(S.a_share(s, h)) for h in self.habits},
                      "a_share_first": {h: _f(S.a_share(s, h, "f")) for h in self.habits},
                      "a_share_later": {h: _f(S.a_share(s, h, "l")) for h in self.habits},
                      "sites_per_proof": {h: _f(S.sites_per_proof(s, h)) for h in self.habits},
                      "other_pct": {h: _f(S.div(s(f"{h}:o"), s(f"{h}:s"))) for h in self.habits},
                      "note_rate": _f(S.note_rate(s, self.notes)),
                      "near_per_proof": {c[5:]: _f(S.near_rate(s, c[5:])) for c in m[t].cols if c.startswith("near:")}}  # fmt: skip
        return out

    # ---------------------------------------------------------------- reads
    def rated_move(self, d: Path, h0: str, base: float) -> float | None:
        s = self.one(d)
        v = None if s is None else _f(S.a_share(s, h0))
        return None if v is None else round(v - base, 2)

    def offline_read(self, c: str, kind: str, rater: str, seed: int) -> dict[str, Any] | None:
        """The +30 read (interpolated) and the last-snapshot read of one offline seed: {point: {terms, ...}}."""
        rated, ctl = arm_id(kind, c, rater, seed), arm_id(CONTROL[kind], c, rater, seed)
        rs, cs = snapshots(rated, self.lh), {x["step"]: x for x in snapshots(ctl, self.lh)}
        bdir, h0 = self.base_dir(c), C.RATERS[rater]
        bs = self.one(bdir)
        if not rs or bs is None or _f(S.a_share(bs, h0)) is None:
            return None
        base = _f(S.a_share(bs, h0))
        moves = [
            {"label": x["label"], "step": x["step"], "move": self.rated_move(x["dir"], h0, base)}
            for x in rs
        ]
        full = {}
        fp = self.lh / "arms" / f"full_{kind}_{c}_{rater}_s{seed}.json"
        if fp.exists():
            full = json.loads(fp.read_text())
        pts = {0: {"label": "base", "step": 0, "dir": bdir, "move": 0.0}} | {
            x["step"]: {**x, "move": mv["move"]}
            for x, mv in zip(rs, moves, strict=True)
            if mv["move"] is not None
        }
        cdir = {0: bdir} | {st: x["dir"] for st, x in cs.items()}
        out: dict[str, Any] = {"rated": rated, "control": ctl, "rated_habit": h0, "base_share": base, "moves": moves,
                               "probe": self.probe_moves(rated, h0, base), "full_task": full.get("bracket"), "points": {}}  # fmt: skip
        bracket = full.get("bracket") if full.get("reached") else None
        if (
            bracket is None
        ):  # recompute from the full measurements: first measured snapshot past the target
            st = sorted(pts)
            hit = next(
                (i for i, x in enumerate(st) if x > 0 and pts[x]["move"] > self.target), None
            )
            bracket = [st[hit - 1], st[hit]] if hit is not None else None
        last = max(pts)
        fin = max(rs, key=lambda x: x["step"])
        if (
            fin["step"] > last
        ):  # the final snapshot has no rated share: no accepted proofs (audit X6.2)
            fs = self.one(fin["dir"])
            out["collapsed"] = {"step": fin["step"], "label": fin["label"], "last_read": last,
                                "accepted_pct": _f(S.perf(fs)["accepted_pct"]) if fs is not None else None}  # fmt: skip
        lacking = [
            f"{run}@{x}"
            for x in (bracket or [])
            for run, have in ((rated, pts), (ctl, cdir))
            if x not in have
        ]
        if (
            bracket and lacking
        ):  # PREREG Amendment 8 point 4 (audit R4): no read, never the last snapshot instead
            out["points"]["at_target"] = None
            out["at_target_missing"] = (
                f"bracket {bracket}: no finished measurement of {', '.join(lacking)}"
            )
        elif bracket:
            s1, s2 = bracket
            m1, m2 = pts[s1]["move"], pts[s2]["move"]
            w = 1.0 if m2 == m1 else min(1.0, max(0.0, (self.target - m1) / (m2 - m1)))
            out["points"]["at_target"] = {"terms": [(1 - w, pts[s1]["dir"], cdir[s1]), (w, pts[s2]["dir"], cdir[s2])],
                                          "steps": [s1, s2], "w": round(w, 4), "move": round((1 - w) * m1 + w * m2, 2),
                                          # the probe's bracket, read by the full measurements: both may sit on one side
                                          "reached": m2 > self.target, "bracketed": m1 <= self.target < m2,
                                          "kl_rated": [kl_at(self.lh / "arms" / rated, x) for x in (s1, s2)],
                                          "kl_control": [kl_at(self.lh / "arms" / ctl, x) for x in (s1, s2)]}  # fmt: skip
        elif (
            last in cdir
        ):  # never past +30: the last snapshot, marked (it never enters the readings table)
            out["points"]["at_target"] = {"terms": [(1.0, pts[last]["dir"], cdir[last])], "steps": [last], "w": 1.0,
                                          "move": pts[last]["move"], "reached": False, "bracketed": False,
                                          "kl_rated": [kl_at(self.lh / "arms" / rated, last)], "kl_control": [kl_at(self.lh / "arms" / ctl, last)]}  # fmt: skip
        if last in cdir and last > 0:
            out["points"]["last"] = {"terms": [(1.0, pts[last]["dir"], cdir[last])], "steps": [last], "w": 1.0,
                                     "move": pts[last]["move"], "reached": pts[last]["move"] > self.target,
                                     "kl_rated": [kl_at(self.lh / "arms" / rated, last)], "kl_control": [kl_at(self.lh / "arms" / ctl, last)]}  # fmt: skip
        return out

    def probe_moves(self, arm: str, h0: str, base: float) -> list[dict[str, Any]]:
        f = self.lh / "arms" / arm / "probe" / "probe.json"
        if not f.exists():
            return []
        out = []
        for lab, v in json.loads(f.read_text()).get("snapshots", {}).items():
            sh = (v.get("a_share") or {}).get(h0)
            m = re.fullmatch(r"step(\d+)", lab)
            out.append(
                {
                    "label": lab,
                    "step": int(m.group(1)) if m else None,
                    "share": sh,
                    "move": None if sh is None else round(sh - base, 2),
                }
            )
        return sorted(out, key=lambda x: (x["step"] is None, x["step"] or 0))

    def line_read(self, c: str, rater: str, seed: int = 0) -> dict[str, Any] | None:
        rated, ctl = C.line_name(c, f"rlhf{rater}", seed), C.line_name(c, "anc", seed)
        rs, cs = snapshots(rated, self.lh), {x["step"]: x for x in snapshots(ctl, self.lh)}
        h0 = C.RATERS[rater]
        bs = self.one(self.base_dir(c))
        if not rs or bs is None or _f(S.a_share(bs, h0)) is None:
            return None
        base = _f(S.a_share(bs, h0))
        moves = [
            {"label": x["label"], "step": x["step"], "move": self.rated_move(x["dir"], h0, base)}
            for x in rs
        ]
        ok = [
            (x, mv["move"])
            for x, mv in zip(rs, moves, strict=True)
            if mv["move"] is not None and x["step"] in cs
        ]
        out: dict[str, Any] = {
            "rated": rated,
            "control": ctl,
            "rated_habit": h0,
            "base_share": base,
            "moves": moves,
            "points": {},
        }
        if not ok:
            return out
        matched = min(ok, key=lambda t: (abs(t[1] - self.target), t[0]["step"]))
        last = max(ok, key=lambda t: t[0]["step"])
        past = [
            (x["step"], mv) for x, mv in sorted(ok, key=lambda t: t[0]["step"]) if mv > self.target
        ]
        out["first_past"] = {"step": past[0][0], "move": past[0][1]} if past else None  # audit X6.3
        out["max_move"] = max(mv for _, mv in ok)
        for name, (x, mv) in (("at_target", matched), ("last", last)):
            out["points"][name] = {"terms": [(1.0, x["dir"], cs[x["step"]]["dir"])], "steps": [x["step"]], "w": 1.0,
                                   "move": mv, "reached": mv > self.target}  # fmt: skip
        return out

    def a10(self, c: str, seed: int, rater: str = "reit") -> dict[str, Any] | None:
        """PREREG Amendment 10's quantities for one corpus and RL seed: at iteration 40 (the last snapshot both lines
        have) and at the iteration where the reward line's rated movement is closest to +30, each line against the
        base: the pooled A-share of the notes before the first reiteration site, and step order's A-share; and
        D(notes) = reward line - anchor line. 95 % bootstrap intervals over theorems."""
        rated, ctl = C.line_name(c, f"rlhf{rater}", seed), C.line_name(c, "anc", seed)
        rs, cs = (
            {x["step"]: x for x in snapshots(rated, self.lh)},
            {x["step"]: x for x in snapshots(ctl, self.lh)},
        )
        h0, bdir = C.RATERS[rater], self.base_dir(c)
        bs, bm = self.one(bdir), self.m(bdir)
        common = sorted(set(rs) & set(cs))
        if bs is None or bm is None or not common or _f(S.a_share(bs, h0)) is None:
            return None
        base = _f(S.a_share(bs, h0))
        moves = {st: self.rated_move(rs[st]["dir"], h0, base) for st in common}
        ok = [st for st in common if moves[st] is not None]
        if not ok:
            return None
        notes = [h for h in self.notes if h != h0]
        out: dict[str, Any] = {
            "reward_line": rated,
            "anchor_line": ctl,
            "base_share": base,
            "moves": moves,
            "points": {},
        }
        for name, st in (
            ("iteration 40", max(ok)),
            ("closest to +30", min(ok, key=lambda x: (abs(moves[x] - self.target), x))),
        ):
            rm_, cm_ = self.m(rs[st]["dir"]), self.m(cs[st]["dir"])
            if rm_ is None or cm_ is None:
                continue
            mm = {"b": bm["none"], "r": rm_["none"], "a": cm_["none"]}

            def nb(s: dict[str, S.Sums], x: str) -> np.ndarray:
                return S.pooled_before(s[x], h0, notes)

            def od(s: dict[str, S.Sums], x: str) -> np.ndarray:
                return S.a_share(s[x], "order")

            W = self.W["read"]  # noqa: N806
            out["points"][name] = {
                "iteration": st, "rated_move": moves[st],
                "notes_before": {"base": _f(S.pooled_before(self.one(bdir), h0, notes)),
                                 "reward_line": S.est(lambda s: nb(s, "r") - nb(s, "b"), mm, W),
                                 "anchor_line": S.est(lambda s: nb(s, "a") - nb(s, "b"), mm, W),
                                 "D": S.est(lambda s: nb(s, "r") - nb(s, "a"), mm, W)},
                "order": {"base": _f(S.a_share(self.one(bdir), "order")),
                          "reward_line": S.est(lambda s: od(s, "r") - od(s, "b"), mm, W),
                          "anchor_line": S.est(lambda s: od(s, "a") - od(s, "b"), mm, W),
                          "D": S.est(lambda s: od(s, "r") - od(s, "a"), mm, W)}}  # fmt: skip
        return out

    # ---------------------------------------------------------------- quantities over a set of reads
    def gather(self, reads: dict[str, list[tuple[float, Path, Path]]]) -> dict[str, S.Mat] | None:
        out = {}
        for k, terms in reads.items():
            for i, (_, rd, cd) in enumerate(terms):
                rm, cm = self.m(rd), self.m(cd)
                if rm is None or cm is None:
                    return None
                out[f"{k}r{i}"], out[f"{k}c{i}"] = rm["none"], cm["none"]
        return out

    @staticmethod
    def dval(s: dict[str, S.Sums], k: str, terms: list, h: str, part: str, r: str) -> np.ndarray:
        return sum(
            w * (share(s[f"{k}r{i}"], h, part, r) - share(s[f"{k}c{i}"], h, part, r))
            for i, (w, _, _) in enumerate(terms)
        )

    @staticmethod
    def dpool(s: dict[str, S.Sums], k: str, terms: list, hs: list[str]) -> np.ndarray:
        """D of the pooled first-site A-share over the habits hs."""
        return sum(
            w * (S.pooled_share(s[f"{k}r{i}"], hs, "f") - S.pooled_share(s[f"{k}c{i}"], hs, "f"))
            for i, (w, _, _) in enumerate(terms)
        )

    def thin(self, terms: list, h: str, part: str, r: str) -> bool:
        for w, rd, cd in terms:
            if w <= 0:
                continue
            a, b = self.one(rd), self.one(cd)
            if (
                a is None
                or b is None
                or min(sites(a, h, part, r), sites(b, h, part, r)) < self.min_sites
            ):
                return True
        return False

    def D(self, terms: list, h: str, part: str = "", r: str = "reit") -> dict[str, Any] | None:
        if self.thin(terms, h, part, r):
            return None
        mm = self.gather({"x": terms})
        return (
            None
            if mm is None
            else S.est(lambda s: self.dval(s, "x", terms, h, part, r), mm, self.W["read"])
        )

    def contrast(self, spec: list[tuple[float, list]], habits: list[str], part: str = "", r: str = "reit",
                 pooled_first: bool = False) -> dict[str, Any] | None:  # fmt: skip
        """sum over (sign, terms) of sign x (mean over habits of D_h), or of the pooled first-site D when asked."""
        if not habits or any(not t for _, t in spec):
            return None
        mm = self.gather({f"t{j}": t for j, (_, t) in enumerate(spec)})
        if mm is None:
            return None

        def fn(s: dict[str, S.Sums]) -> np.ndarray:
            tot = 0.0
            for j, (sign, terms) in enumerate(spec):
                if pooled_first:
                    v = self.dpool(s, f"t{j}", terms, habits)
                else:
                    v = sum(self.dval(s, f"t{j}", terms, h, part, r) for h in habits) / len(habits)
                tot = tot + sign * v
            return tot

        return {**S.est(fn, mm, self.W["read"]), "habits": habits}

    def D_all(self, terms: list, h0: str) -> dict[str, Any]:
        return {h: e for h in self.habits if (e := self.D(terms, h, "", h0)) is not None}

    def thin_pooled(self, terms: list, hs: list[str]) -> bool:
        """The pooled first-site reading needs --min_sites first sites over the habits hs on both sides of every term."""
        for w, rd, cd in terms:
            if w <= 0:
                continue
            a, b = self.one(rd), self.one(cd)
            if (
                a is None
                or b is None
                or min(sum(float(S.ab_sites(x, h, "f")) for h in hs) for x in (a, b))
                < self.min_sites
            ):
                return True
        return False

    def readable(
        self, terms_list: list[list], habits: list[str], part: str = "", r: str = "reit"
    ) -> list[str]:
        return [h for h in habits if all(not self.thin(t, h, part, r) for t in terms_list)]

    # ---------------------------------------------------------------- one point of one comparison
    def read_point(self, c: str, pt: dict[str, Any], h0: str, rater: str) -> dict[str, Any]:
        terms = pt["terms"]
        res: dict[str, Any] = {
            k: pt[k] for k in ("steps", "w", "move", "reached", "bracketed") if k in pt
        }
        res.update({k: pt[k] for k in ("kl_rated", "kl_control") if k in pt})
        res["precondition_met"] = pt.get("move") is not None and pt["move"] >= C.PRECONDITION
        D, D1, thin = {}, {}, []
        for h in self.habits:
            e = self.D(terms, h, "", h0)
            if e is None:
                thin.append(h)
            else:
                D[h] = e
            e1 = self.D(terms, h, "f", h0)
            if e1 is not None:
                D1[h] = e1
        res.update(D=D, D_first=D1, thin=thin)
        res["D_split"] = {part: {h: e for h in self.habits if h != h0 and (e := self.D(terms, h, part, h0)) is not None}
                          for part in ("before", "after")}  # fmt: skip
        d0 = (D.get(h0) or {}).get("point")
        res["D_per_point"] = {
            h: (round(e["point"] / d0, 4) if e.get("point") is not None and d0 else None)
            for h, e in D.items()
            if h != h0
        }
        # performance and drift (points; the no-preference run against the base)
        b = self.one(self.base_dir(c))

        def wavg(fn: Any, which: int) -> float | None:
            """The weighted point value over the read's terms, of the rated (0) or the no-preference (1) side."""
            vals = []
            for w, rd, cd in terms:
                x = self.one(rd if which == 0 else cd)
                vals.append((w, None if x is None else _f(fn(x))))
            if any(v is None for _, v in vals):
                return None
            return round(sum(w * v for w, v in vals), 2)

        res["perf"] = {
            side: {k: wavg(lambda s, k=k: S.perf(s)[k], i) for k in PERF}
            for side, i in (("rated", 0), ("control", 1))
        }
        van = {}
        for h in self.habits:
            bv, xv = (
                _f(S.a_share(b, h)) if b is not None else None,
                wavg(lambda s, h=h: S.a_share(s, h), 1),
            )
            van[h] = {"base": bv, "control": xv, "drift": bool(bv is not None and xv is not None and bv - xv > 0.5 * bv),
                      "sites_per_proof_base": _f(S.sites_per_proof(b, h)) if b is not None else None,
                      "sites_per_proof_control": wavg(lambda s, h=h: S.sites_per_proof(s, h), 1)}  # fmt: skip
        res["vanishing"] = van
        bm = self.m(self.base_dir(c))
        if bm is not None and "A" in bm and "B" in bm:
            sa, sb = (
                _f(S.a_share(bm["A"].sums(np.ones(self.n)), h0)),
                _f(S.a_share(bm["B"].sums(np.ones(self.n)), h0)),
            )
            sr, sc = wavg(lambda s: S.a_share(s, h0), 0), wavg(lambda s: S.a_share(s, h0), 1)
            if None not in (sa, sb, sr, sc) and abs(sa - sb) > 1e-9:
                wr, wc = (sr - sb) / (sa - sb), (sc - sb) / (sa - sb)
                mix = {}
                for h, e in D.items():
                    ha, hb = (
                        _f(S.a_share(bm["A"].sums(np.ones(self.n)), h)),
                        _f(S.a_share(bm["B"].sums(np.ones(self.n)), h)),
                    )
                    if ha is None or hb is None or e.get("point") is None:
                        continue
                    imp = (wr - wc) * (ha - hb)
                    mix[h] = {
                        "observed": e["point"],
                        "implied": round(imp, 2),
                        "ratio": round(e["point"] / imp, 3) if abs(imp) > 1e-9 else None,
                    }
                res["mixture"] = {"w_rated": round(wr, 4), "w_control": round(wc, 4), "habits": mix}
        res["dirs"] = [[w, str(rd), str(cd)] for w, rd, cd in terms]
        return res

    # ---------------------------------------------------------------- the GRPO arm's chain prediction
    def chain(self, c: str, pt_r: dict[str, Any], rater: str) -> dict[str, Any]:
        """For each habit decided before the rated site (the notes and step order): observed - predicted in the reward
        line and in the anchor-only line at the same iteration, and their difference (Amendment 3, 6b)."""
        h0 = C.RATERS[rater]
        ((_, rd, cd),) = pt_r["terms"]
        bm, rm, cm = self.m(self.base_dir(c)), self.m(rd), self.m(cd)
        if bm is None or rm is None or cm is None:
            return {}
        mm = {"b": bm["none"], "r": rm["none"], "c": cm["none"]}
        one = {k: v.sums(np.ones(self.n)) for k, v in mm.items()}
        out = {}
        for h in [*self.notes, "order"]:
            if h == h0:
                continue
            cnt = [float(one["b"](f"cond {h0}|{h}:{k}")) for k in S.K4]
            if min(cnt[0] + cnt[1], cnt[2] + cnt[3]) < MIN_COND:
                continue
            if (
                min(sum(float(one[x](f"cond {h0}|{h}:{k}")) for k in S.K4) for x in ("r", "c"))
                < self.min_sites
            ):
                continue

            def dev(s: dict[str, S.Sums], x: str, h: str = h) -> np.ndarray:
                q = S.cond_q(
                    s[x], h0, h
                )  # audit R7: the weight from the same site counts as the conditionals
                pred = q * S.cond_share(s["b"], h0, h, "A") + (1 - q) * S.cond_share(
                    s["b"], h0, h, "B"
                )
                return S.cond_observed(s[x], h0, h) - pred

            out[h] = {"reward_line": S.est(lambda s, dev=dev: dev(s, "r"), mm, self.W["read"]),
                      "anchor_line": S.est(lambda s, dev=dev: dev(s, "c"), mm, self.W["read"]),
                      "difference": S.est(lambda s, dev=dev: dev(s, "r") - dev(s, "c"), mm, self.W["read"])}  # fmt: skip
        return out


# ------------------------------------------------------------------------------------------------ the readout
def validity(
    per_seed: dict[int, dict[str, Any]], name: str, seeds: tuple[int, ...], grpo: bool = False
) -> dict[str, Any]:
    """PREREG Amendment 8 point 4 (audit R3): a corpus's read at `name` is valid when every expected run exists and its
    point crossed +30 (`reached`), is bracketed by two fully measured snapshots, and meets the +20 precondition. A GRPO
    line is read at its iteration closest to +30, not interpolated (audit X6.4): the same rule without the bracket test
    (the read point past +30 and the +20 precondition); the reason names where the line first passed +30."""
    why = []
    if grpo:
        for s_ in seeds:
            x = per_seed.get(s_)
            p = (x or {}).get("points", {}).get(name)
            if x is None or p is None:
                why.append(f"seed {s_}: no read")
                continue
            fp = x.get("first_past")
            if not p.get("reached") and not fp:
                why.append(
                    f"seed {s_}: +{C.TARGET_MOVE:g} never passed (largest move {x.get('max_move')})"
                )
            elif not p.get("reached"):
                why.append(f"seed {s_}: the read iteration is below +{C.TARGET_MOVE:g} (move {p.get('move')}); the line "
                           f"first passed +{C.TARGET_MOVE:g} at iteration {fp['step']} (move {fp['move']})")  # fmt: skip
            if p.get("move") is None or p["move"] < C.PRECONDITION:
                why.append(
                    f"seed {s_}: precondition +{C.PRECONDITION:g} not met (move {p.get('move')})"
                )
        return {"ok": not why, "why": why}
    for s_ in seeds:
        x = per_seed.get(s_)
        p = (x or {}).get("points", {}).get(name)
        if x is None:
            why.append(f"seed {s_}: no run")
        elif p is None:
            why.append(f"seed {s_}: {x.get('at_target_missing') or 'no read'}")
        else:
            if not p.get("reached"):
                why.append(f"seed {s_}: +{C.TARGET_MOVE:g} not reached (move {p.get('move')})")
            elif len(p.get("steps") or []) != 2 or p.get("bracketed") is False:
                why.append(
                    f"seed {s_}: not bracketed by two measured snapshots (steps {p.get('steps')})"
                )
            if p.get("move") is None or p["move"] < C.PRECONDITION:
                why.append(
                    f"seed {s_}: precondition +{C.PRECONDITION:g} not met (move {p.get('move')})"
                )
    return {"ok": not why, "why": why}


def mean_point(points: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The mean over seeds of reads: their terms with weights divided by the number of seeds."""
    pts = [p for p in points if p]
    if not pts:
        return None
    k = len(pts)
    return {"terms": [(w / k, rd, cd) for p in pts for (w, rd, cd) in p["terms"]], "steps": [p["steps"] for p in pts],
            "w": [p["w"] for p in pts], "move": round(sum(p["move"] for p in pts) / k, 2) if all(p.get("move") is not None for p in pts) else None,
            "reached": all(p["reached"] for p in pts), "bracketed": [p.get("bracketed") for p in pts],
            "kl_rated": [p.get("kl_rated") for p in pts], "kl_control": [p.get("kl_control") for p in pts],
            "seeds": k}  # fmt: skip


def readout(
    lh: Path, corpora: list[str], B: int, seed: int, min_sites: int, target: float
) -> dict[str, Any]:
    R = Reader(lh, corpora, B, seed, min_sites, target)
    res: dict[str, Any] = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "B": B, "seed": seed, "min_sites": min_sites,
                           "target": target, "read_set": READ_SET, "step_limit": C.STEP_LIMIT, "tables": {}, "compare": {},
                           "between": {}, "prereg": {}, "form_groups": R.groups,
                           "intervals": "bootstrap over theorems; base-to-base and run-to-run variance are not in them (seeds shown)"}  # fmt: skip
    # pair files under the 4,000 cap (Amendment 7 point 5): flagged
    res["below_cap"] = {}
    for f in sorted((lh / "pairs").glob("*_s*/*/summary.json")):
        with contextlib.suppress(ValueError, OSError):
            cap = json.loads(f.read_text()).get("cap") or {}
            if cap.get("train_below_cap") or cap.get("sft_below_cap"):
                res["below_cap"][f"{f.parent.parent.name}/{f.parent.name}"] = {
                    k: cap.get(k) for k in ("train_capped", "sft_capped")
                }
    # tasks decided not to run (the rlhf line whose reward model missed 0.90)
    res["not_run"] = {}
    for f in sorted((lh / "arms").glob("*.not_run.json")):
        with contextlib.suppress(ValueError, OSError):
            res["not_run"][f.name[: -len(".not_run.json")]] = json.loads(f.read_text()).get(
                "reason"
            )
    # tables
    for c in corpora:
        arms = {
            f"{b}_s0": [{"label": "base", "step": 0, "dir": lh / "measure" / f"{b}_s0"}]
            for b in C.bases_on_pod(c)
        }
        for x in C.lines_of(c):
            arms[x] = snapshots(x, lh)
        for r in C.raters_of(c):
            for k in C.OFFLINE_KINDS:
                for s in seeds_of(k, r):
                    arms[C.arm_name(k, c, r, s)] = snapshots(C.arm_name(k, c, r, s), lh)
        for a, snaps in arms.items():
            for x in snaps:
                for which in ("read", "pooled"):
                    t = R.table(x["dir"], which)
                    if t is not None:
                        res["tables"].setdefault(a, {}).setdefault(x["label"], {"step": x["step"]})[
                            which
                        ] = t
    for c in [x for x in corpora if x in C.BASE2_CORPORA]:  # the second bases (Amendment 8 point 2)
        b2 = {
            f"{c}_s{C.BASE2_SEED}": [
                {"label": "base", "step": 0, "dir": lh / "measure" / f"{c}_s{C.BASE2_SEED}"}
            ]
        }
        for t in C.base2_tasks(c):
            if C.parse_task(t)["type"] == "offline":
                b2[t] = snapshots(t, lh)
        for a, snaps in b2.items():
            for x in snaps:
                for which in ("read", "pooled"):
                    t = R.table(x["dir"], which)
                    if t is not None:
                        res["tables"].setdefault(a, {}).setdefault(x["label"], {"step": x["step"]})[
                            which
                        ] = t
    for b in [f"{x}_s0" for x in C.NOV14]:
        for which in ("read", "pooled"):
            t = R.table(lh / "measure" / b, which)
            if t is not None:
                res["tables"].setdefault(b, {}).setdefault("base", {"step": 0})[which] = t
    # reads: per kind, rater, corpus, point (and seed)
    reads_seed: dict[
        tuple[str, str, str], dict[int, dict[str, Any]]
    ] = {}  # dpo:reit per trainer seed (R3)
    reads: dict[
        tuple[str, str, str], dict[str, dict[str, Any]]
    ] = {}  # (kind, rater, point) -> corpus -> point spec
    for c in corpora:
        for r in C.raters_of(c):
            for kind in CONTROL:
                if kind == "rlhf" and r in C.GRPO_RATERS and c not in C.GRPO_CORPORA:
                    res["compare"].setdefault(f"{kind}:{r}", {})[c] = {
                        "not_run": "by design: no GRPO lines on this corpus (PREREG Amendment 6)"
                    }
                    continue
                if kind == "rlhf":
                    per_seed = {s: R.line_read(c, r) for s in seeds_of(kind, r)}
                else:
                    per_seed = {s: R.offline_read(c, kind, r, s) for s in seeds_of(kind, r)}
                per_seed = {s: v for s, v in per_seed.items() if v}
                if not per_seed:
                    continue
                entry: dict[str, Any] = {
                    "seeds": {
                        s: {k: v for k, v in x.items() if k != "points"}
                        for s, x in per_seed.items()
                    },
                    "read": {},
                }
                for name in ("at_target", "last"):
                    pts = [x["points"].get(name) for x in per_seed.values()]
                    mp = mean_point(pts)
                    if mp is None:
                        continue
                    h0 = C.RATERS[r]
                    rd = R.read_point(c, mp, h0, r)
                    if len(per_seed) > 1:
                        rd["per_seed"] = {s: {"D": R.D_all(x["points"][name]["terms"], h0), "steps": x["points"][name]["steps"],
                                              "move": x["points"][name]["move"]} for s, x in per_seed.items() if x["points"].get(name)}  # fmt: skip
                    if kind == "rlhf":  # the chain prediction at both points (Amendment 8 point 5)
                        rd["chain"] = R.chain(c, mp, r)
                        # each line against the base, beside their difference (Amendment 10: seed 0's D on c24 was
                        # the sum of two opposite drifts)
                        ((_, rdir, cdir),) = mp["terms"]
                        bdir = R.base_dir(c)
                        rd["vs_base"] = {
                            "reward_line": R.D_all([(1.0, rdir, bdir)], h0),
                            "anchor_line": R.D_all([(1.0, cdir, bdir)], h0),
                        }
                    entry["read"][name] = rd
                    reads.setdefault((kind, r, name), {})[c] = mp
                entry["primary"] = (
                    "at_target"  # matched movement for every arm (Amendment 8 point 5)
                )
                entry["valid"] = {
                    name: validity(per_seed, name, seeds_of(kind, r), grpo=kind == "rlhf")
                    for name in ("at_target", "last")
                }
                entry["collapsed"] = {
                    s_: x["collapsed"] for s_, x in per_seed.items() if x.get("collapsed")
                }
                if kind in (
                    "dpo",
                    "sft",
                ):  # passes over the pairs at the crossing (Amendment 9 point 1)
                    for s_, x in per_seed.items():
                        add_passes(
                            res,
                            lh,
                            c,
                            r,
                            kind,
                            arm_id(kind, c, r, s_),
                            x["points"].get("at_target"),
                        )
                if kind == "dpo" and r == "reit":
                    for name in ("at_target", "last"):
                        for s_, x in per_seed.items():
                            if x["points"].get(name):
                                reads_seed.setdefault((kind, r, name), {}).setdefault(s_, {})[c] = (
                                    x["points"][name]
                                )
                if kind == "rlhf":
                    pr = rm_probe(lh, c, r)
                    entry["rm_probe"] = pr
                    entry["reward_driven"] = (pr or {}).get(
                        "reward_driven"
                    )  # the probe's own marks (Amendment 7 point 6)
                    entry["not_probed"] = (pr or {}).get("not_probed")
                if kind == "sft":
                    sp = lh / "pairs" / f"{c}_s0" / r / "summary.json"
                    if sp.exists():
                        comp = json.loads(sp.read_text()).get("a_share") or {}
                        p, q = comp.get("sft_pref") or {}, comp.get("sft_sampled") or {}
                        entry["composition_prediction"] = {
                            h: round(p[h] - q[h], 2)
                            for h in p
                            if p.get(h) is not None and q.get(h) is not None
                        }
                        entry["composition"] = comp
                res["compare"].setdefault(f"{kind}:{r}", {})[c] = entry
    # PREREG Amendment 10: the GRPO lines per RL seed, each against the base (c24: seeds 0 and 1; others: seed 0)
    res["grpo_seeds"] = {}
    for c in [x for x in corpora if x in C.GRPO_CORPORA]:
        for sd in (0, *(C.RL2_SEED,) * (c in C.RL2_CORPORA)):
            q = R.a10(c, sd)
            if q:
                res["grpo_seeds"][f"{c} seed {sd}"] = q
    q1 = res["grpo_seeds"].get(f"c24 seed {C.RL2_SEED}")
    res["amendment10"] = {
        name: a10_reading(p_) for name, p_ in (q1 or {}).get("points", {}).items()
    } or {"reading": "not read (no RL seed-1 lines yet)"}
    # the second bases (PREREG Amendment 8 points 2-3): the reit DPO arm, trainer seed 0, on base seed 1
    res["base2"], b2_pts = {}, {}
    for c in [x for x in corpora if x in C.BASE2_CORPORA]:
        tok = C.base_token(c, C.BASE2_SEED)
        info: dict[str, Any] = {"token": tok, "state": base2_state(lh, c)}
        x2 = R.offline_read(tok, "dpo", "reit", 0)
        if x2:
            per2 = {0: x2}
            e2: dict[str, Any] = {"seeds": {0: {k: v for k, v in x2.items() if k != "points"}}, "read": {},
                                  "primary": "at_target", "base_seed": C.BASE2_SEED,
                                  "valid": {n: validity(per2, n, (0,)) for n in ("at_target", "last")}}  # fmt: skip
            for name in ("at_target", "last"):
                if x2["points"].get(name):
                    e2["read"][name] = R.read_point(
                        tok, x2["points"][name], C.RATERS["reit"], "reit"
                    )
            res["compare"].setdefault("dpo:reit", {})[tok] = e2
            add_passes(
                res,
                lh,
                tok,
                "reit",
                "dpo",
                arm_id("dpo", tok, "reit", 0),
                x2["points"].get("at_target"),
            )
            info["valid"] = e2["valid"]["at_target"]
            if x2["points"].get("at_target"):
                b2_pts[c] = x2["points"]["at_target"]
        res["base2"][c] = info
    # between corpora and PREREG
    notes, natural = R.notes, R.natural
    for (kind, r, name), by_c in reads.items():
        h0 = C.RATERS[r]
        T = {c: by_c[c]["terms"] for c in by_c}
        btw: dict[str, Any] = {}
        for x, y in PAIRS_BETWEEN:
            if x in T and y in T:
                btw[f"{x}-{y}"] = {
                    h: R.contrast([(1.0, T[x]), (-1.0, T[y])], [h], "", h0)
                    for h in R.readable([T[x], T[y]], R.habits, "", h0)
                }
        res["between"][f"{kind}:{r}:{name}"] = btw
        pr: dict[str, Any] = {"preconditions": {c: by_c[c].get("move") is not None and by_c[c]["move"] >= C.PRECONDITION for c in by_c},
                              "reached": {c: by_c[c].get("reached") for c in by_c}}  # fmt: skip

        def vs0(
            c: str,
            pool: list[str],
            part: str = "",
            pooled_first: bool = False,
            T: dict = T,
            h0: str = h0,
        ) -> dict[str, Any] | None:  # noqa: N803
            if c not in T or "c0" not in T:
                return None
            if pooled_first:
                ok = pool and not R.thin_pooled(T[c], pool) and not R.thin_pooled(T["c0"], pool)
                return (
                    R.contrast([(1.0, T[c]), (-1.0, T["c0"])], pool, part, h0, True) if ok else None
                )
            hs = R.readable([T[c], T["c0"]], pool, part, h0)
            return R.contrast([(1.0, T[c]), (-1.0, T["c0"])], hs, part, h0) if hs else None

        if r == "reit":
            mk = ["order", "conj"]
            for c in T:
                if c != "c0":
                    pr[f"P1_M({c})"] = vs0(c, mk)
            if all(k in T for k in ("c24", "c4", "c0")):
                hs = [h for h in mk if h in R.readable([T["c24"], T["c4"], T["c0"]], mk, "", h0)]
                pr["P1_M(c24)-M(c4)"] = (
                    R.contrast([(1.0, T["c24"]), (-1.0, T["c4"])], hs, "", h0) if hs else None
                )
            for c in T:
                if c == "c0":
                    continue
                linked = [h for h in C.linked_of(c) if h not in (h0, "route")]
                pr[f"P2_S({c})"] = vs0(c, linked)
                pr[f"P2_S_before({c})"] = vs0(
                    c, [h for h in linked if h.startswith("note_") or h == "order"]
                )
                pr[f"P2_S_after({c})"] = vs0(c, [h for h in linked if h in AFTER])
                pr[f"P2_S_first({c})"] = vs0(c, linked, "f")
                pr[f"P2_first_notes_pooled({c})"] = vs0(
                    c, [h for h in notes if h in linked], pooled_first=True
                )
                pr[f"P2_S_sites_before({c})"] = vs0(c, linked, "before")
                pr[f"P2_S_sites_after({c})"] = vs0(c, linked, "after")
        if r == "atom" and "c24" in T and "c0" in T:
            un = [h for h in notes if h != h0]
            pr["P3_S_notes(c24)"] = vs0("c24", un)
            pr["P3_S_natural(c24)"] = vs0(
                "c24", [h for h in natural if h in C.linked_of("c24") and h != "route"]
            )
            pr["P3_S_first(c24)"] = vs0("c24", un, pooled_first=True)
            pr["P3_first_per_class(c24)"] = {
                cls: vs0("c24", [f"note_{cls}"], "f")
                for cls in FIRST_CLASSES
                if f"note_{cls}" != h0
            }
        if "c24" in T:  # per-class note D grouped by A's form type (design audit F13)
            grp: dict[str, Any] = {}
            for g in ("copy", "or2", "longer", "shorter"):
                hs = [h for h, x in R.groups.items() if x == g and h != h0]
                grp[g] = {"c24": R.contrast([(1.0, T["c24"])], R.readable([T["c24"]], hs, "", h0), "", h0),
                          "c24-c0": vs0("c24", hs)}  # fmt: skip
            pr["notes_by_form_type"] = grp
        pr["drift"] = {
            c: sorted(
                h
                for h, v in (
                    (
                        res["compare"].get(f"{kind}:{r}", {}).get(c, {}).get("read", {}).get(name)
                        or {}
                    ).get("vanishing")
                    or {}
                ).items()
                if v["drift"]
            )
            for c in T
        }
        if (kind, r) == (
            "dpo",
            "reit",
        ):  # per trainer seed, each against c0's same seed (Amendment 8 point 4)
            pr["per_seed"] = {}
            for s_, by_cs in sorted(reads_seed.get((kind, r, name), {}).items()):
                Ts = {c: by_cs[c]["terms"] for c in by_cs}

                def vss(
                    c: str, pool: list[str], Ts: dict = Ts, h0: str = h0
                ) -> dict[str, Any] | None:  # noqa: N803
                    if c not in Ts or "c0" not in Ts:
                        return None
                    hs = R.readable([Ts[c], Ts["c0"]], pool, "", h0)
                    return R.contrast([(1.0, Ts[c]), (-1.0, Ts["c0"])], hs, "", h0) if hs else None

                lk = [h for h in C.linked_of("c24") if h not in (h0, "route")]
                pr["per_seed"][s_] = {"M(c24)": vss("c24", ["order", "conj"]), "M(c4)": vss("c4", ["order", "conj"]),
                                      "S_before(c24)": vss("c24", [h for h in lk if h.startswith("note_") or h == "order"])}  # fmt: skip
        if (kind, r, name) == (
            "dpo",
            "reit",
            "at_target",
        ):  # per base seed and their average (Amendment 8 point 3)
            pr["per_base_seed"], pr["base_avg"] = base_seed_quantities(
                R, T, {c: p["terms"] for c, p in b2_pts.items()}, h0
            )
        res["prereg"][f"{kind}:{r}:{name}"] = pr
    res["passes_differ"] = passes_differ(res.get("passes") or {})
    valid = {
        c: ((res["compare"].get("dpo:reit") or {}).get(c) or {}).get("valid", {}).get("at_target")
        for c in corpora
    }
    res["readings"] = readings(res["prereg"].get("dpo:reit:at_target") or {}, valid, res["base2"])
    return res


def _in(e: dict[str, Any] | None, lo: float, hi: float) -> bool:
    return bool(e and e.get("ci") and lo <= e["ci"][0] and e["ci"][1] <= hi)


def pair_counts(lh: Path, token: str, rater: str) -> dict[str, Any]:
    """The pair build's counts for a corpus field (<c>: pairs/<c>_s0; <c>b1: pairs/<c>_s1): the train pairs DPO trains
    on (after the balancing and the 4,000 cap) and the proofs of each fine-tune file."""
    c, bs = C.split_base(token)
    d = lh / "pairs" / f"{c}_s{bs}" / rater
    cap: dict[str, Any] = {}
    with contextlib.suppress(OSError, ValueError):
        cap = json.loads((d / "summary.json").read_text()).get("cap") or {}
    train = cap.get("train_capped") or cap.get("train_balanced")
    if train is None and (d / "train.jsonl").exists():
        train = sum(1 for ln in (d / "train.jsonl").read_text().splitlines() if ln.strip())
    return {"train_pairs": train, "sft_proofs": cap.get("sft_capped")}


def add_passes(
    res: dict[str, Any],
    lh: Path,
    token: str,
    rater: str,
    kind: str,
    arm: str,
    pt: dict[str, Any] | None,
) -> None:
    """PREREG Amendment 9 point 1 (audit W6): passes over a rated run's train data at its +30 crossing = batch (64) x the
    crossing step (interpolated in the bracket, as the read) / its train pairs (DPO) or fine-tune proofs (SFT)."""
    e = res.setdefault("passes", {}).setdefault(f"{token}:{rater}", {"corpus": C.split_base(token)[0],
                                                                    "base_seed": C.split_base(token)[1], **pair_counts(lh, token, rater), "runs": {}})  # fmt: skip
    n = e["train_pairs"] if kind == "dpo" else e["sft_proofs"]
    row: dict[str, Any] = {"kind": kind, "data": n, "step": None, "passes": None}
    if pt and pt.get("reached") and len(pt.get("steps") or []) == 2 and pt.get("w") is not None:
        s1, s2 = pt["steps"]
        row["step"] = round((1 - pt["w"]) * s1 + pt["w"] * s2, 2)
        row["passes"] = round(C.OFFLINE_BATCH * row["step"] / n, 3) if n else None
    else:
        row["why"] = "no valid +30 crossing" if pt else "no +30 read"
    e["runs"][arm] = row


def passes_differ(passes: dict[str, Any]) -> list[str]:
    """Amendment 9 point 2's trigger on the reiteration DPO runs, per base seed (each corpus's mean over its trainer
    seeds; c0 from the first base): c4 or c0 after more than 2 passes while c24 crosses within 1, or the reverse."""
    out = []
    for bs in sorted({e["base_seed"] for e in passes.values()}):
        mean = {}
        for c in ("c24", "c4", "c0"):
            e = passes.get(f"{C.base_token(c, bs if c != 'c0' else 0)}:reit") or {}
            v = [
                r["passes"]
                for r in e.get("runs", {}).values()
                if r["kind"] == "dpo" and r["passes"] is not None
            ]
            if v:
                mean[c] = round(sum(v) / len(v), 3)
        if "c24" not in mean:
            continue
        others = {c: v for c, v in mean.items() if c != "c24"}
        hit = (mean["c24"] <= 1 and any(v > 2 for v in others.values())) or (
            mean["c24"] > 2 and any(v <= 1 for v in others.values())
        )
        if hit:
            out.append(
                f"base seed {bs}: c24 {mean['c24']} passes; "
                + ", ".join(f"{c} {v}" for c, v in others.items())
            )
    return out


def a10_reading(p: dict[str, Any]) -> str:
    """PREREG Amendment 10's reading rule on one point of the seed-1 lines: replicates only if the reward line moves the
    notes toward A and the anchor line toward B, both intervals excluding 0, and D(notes) > +20 with its interval above
    0; "anchor line did not drift" when the anchor line's interval does not lie below 0 (the reward line moving toward
    A); "not replicated" when the reward line's notes do not move toward A (interval includes 0, or toward B)."""
    n = p["notes_before"]
    r, a, d = n["reward_line"], n["anchor_line"], n["D"]
    if any(e.get("point") is None or not e.get("ci") for e in (r, a, d)):
        return "not read (a quantity is missing)"
    if not r["ci"][0] > 0:
        return f"not replicated: the reward line's notes do not move toward A ({S.fmt(r)})"
    if not a["ci"][1] < 0:
        return (f"anchor line did not drift ({S.fmt(a)}): seed 0's D was enlarged by the anchor line's drift; lead with the "
                f"reward line against the base ({S.fmt(r)})")  # fmt: skip
    if d["point"] > 20 and d["ci"][0] > 0:
        return f"replicates (reward line {S.fmt(r)}, anchor line {S.fmt(a)}, D(notes) {S.fmt(d)})"
    return f"not replicated: both lines moved as in seed 0, but D(notes) {S.fmt(d)} is not above +20 with its interval above 0"


def base2_state(lh: Path, corpus: str) -> str:
    """done: its full measurement finished; ended: the pod's pass ended (LH/pod/<c>.BASE2_DONE) without it; pending."""
    tok = C.base_token(corpus, C.BASE2_SEED)
    f = lh / "arms" / f"{C.full_name('dpo', tok, 'reit', 0)}.json"
    with contextlib.suppress(OSError, ValueError):
        if json.loads(f.read_text()).get("done"):
            return "done"
    return "ended" if (lh / "pod" / f"{corpus}.BASE2_DONE").exists() else "pending"


BASE_QUANTITIES = ("M(c24)", "M(c4)", "M(c24)-M(c4)", "S_before(c24)", "S(c24)")


def base_seed_quantities(
    R: Reader, T: dict[str, list], T2: dict[str, list], h0: str
) -> tuple[dict[int, Any], dict[str, Any] | None]:  # noqa: N803
    """PREREG Amendment 8 point 3: M(c24), M(c4), M(c24) - M(c4), S_before(c24), S(c24) per base seed (base seed 0: the
    first bases' reads, mean over their trainer seeds; base seed 1: the second bases' reads), each against the one c0
    base (its read, mean over its trainer seeds); and their average over the two base seeds, as one contrast (weights
    1/2 on each base's read). Each quantity reads the habits readable in every read it uses."""
    lk = [h for h in C.linked_of("c24") if h not in (h0, "route")]
    pools = {
        "M": ["order", "conj"],
        "S_before": [h for h in lk if h.startswith("note_") or h == "order"],
        "S": lk,
    }

    def q(spec: list[tuple[float, list]], pool: list[str]) -> dict[str, Any] | None:
        if any(not t for _, t in spec):
            return None
        hs = R.readable([t for _, t in spec], pool, "", h0)
        return R.contrast(spec, hs, "", h0) if hs else None

    def per(Tx: dict[str, list]) -> dict[str, Any]:  # noqa: N803
        t0, t24, t4 = T.get("c0"), Tx.get("c24"), Tx.get("c4")
        return {"M(c24)": q([(1.0, t24), (-1.0, t0)], pools["M"]) if t24 and t0 else None,
                "M(c4)": q([(1.0, t4), (-1.0, t0)], pools["M"]) if t4 and t0 else None,
                "M(c24)-M(c4)": q([(1.0, t24), (-1.0, t4)], pools["M"]) if t24 and t4 else None,
                "S_before(c24)": q([(1.0, t24), (-1.0, t0)], pools["S_before"]) if t24 and t0 else None,
                "S(c24)": q([(1.0, t24), (-1.0, t0)], pools["S"]) if t24 and t0 else None}  # fmt: skip

    pb = {0: per(T), C.BASE2_SEED: per(T2)}
    if not all(T.get(c) for c in ("c24", "c4", "c0")) or not all(T2.get(c) for c in ("c24", "c4")):
        return pb, None
    a24, a4, t0 = [(0.5, T["c24"]), (0.5, T2["c24"])], [(0.5, T["c4"]), (0.5, T2["c4"])], T["c0"]
    neg4 = [(-w, t) for w, t in a4]
    avg = {"M(c24)": q([*a24, (-1.0, t0)], pools["M"]), "M(c4)": q([*a4, (-1.0, t0)], pools["M"]),
           "M(c24)-M(c4)": q([*a24, *neg4], pools["M"]), "S_before(c24)": q([*a24, (-1.0, t0)], pools["S_before"]),
           "S(c24)": q([*a24, (-1.0, t0)], pools["S"])}  # fmt: skip
    return pb, avg


READ_CORPORA = ("c24", "c4", "c0")
SEED_QUANTITIES = ("M(c24)", "M(c4)", "S_before(c24)")


def table_row(q: dict[str, Any]) -> tuple[str, tuple[str, ...]]:
    """PREREG's readings table on one set of quantities -> (the row, the quantities it rests on)."""
    m24, m4, d = q.get("M(c24)"), q.get("M(c4)"), q.get("M(c24)-M(c4)")
    sb, s24 = q.get("S_before(c24)"), q.get("S(c24)")

    def ge10(e: dict[str, Any] | None) -> bool:
        return bool(e and e.get("point") is not None and e["point"] >= 10 and S.above0(e))

    if m24 is None or m4 is None or d is None:
        return "not read (a quantity is missing)", ()
    if ge10(m24) and S.above0(d) and ge10(sb):
        return "the author moves as a unit with 24 links and not with 4", (
            "M(c24)",
            "M(c24)-M(c4)",
            "S_before(c24)",
        )
    if ge10(m24) and ge10(m4) and d.get("ci") and d["ci"][0] <= 0 <= d["ci"][1]:
        return "linked habits move, but no more with 24 than with 4", (
            "M(c24)",
            "M(c4)",
            "M(c24)-M(c4)",
        )
    if _in(m24, -10, 10) and _in(s24, -10, 10):
        return "habits edited one by one, as on Nov 18", ("M(c24)", "S(c24)")
    return "unresolved", ()


def readings(
    pr: dict[str, Any], valid: dict[str, Any] | None = None, base2: dict[str, Any] | None = None
) -> dict[str, Any]:
    """PREREG's readings table on the DPO arm (reit rater, at +30), printed only from valid reads (Amendment 8 point 4,
    audit R3): every run of c24, c4 and c0 reached +30, bracketed by two fully measured snapshots, with the +20
    precondition; no two trainer seeds of M(c24), M(c4) or S_before(c24) of opposite sign with either outside +-5.
    Amendment 8 point 3: when both second bases (c24b1, c4b1) are measured, the table reads the average over the two
    base seeds (their runs must be valid too), and a row resting on M(c24) - M(c4) or S_before(c24) reads "unresolved
    (bases disagree)" when the base seeds give opposite signs with either outside +-5. While a second base is pending
    the table reads the first bases, said so."""
    valid, base2 = valid or {}, base2 or {}
    first = {
        k: pr.get(k)
        for k in ("P1_M(c24)", "P1_M(c4)", "P1_M(c24)-M(c4)", "P2_S_before(c24)", "P2_S(c24)")
    }
    first = dict(zip(BASE_QUANTITIES, first.values(), strict=True))
    bad = [f"{c}: {'; '.join((valid.get(c) or {'why': ['no read']})['why'])}" for c in READ_CORPORA
           if not (valid.get(c) or {}).get("ok")]  # fmt: skip
    pts = {q: [((pr.get("per_seed") or {}).get(s_) or {}).get(q) for s_ in sorted(pr.get("per_seed") or {})]
           for q in SEED_QUANTITIES}  # fmt: skip
    out: dict[str, Any] = {
        "per_seed": {q: [e.get("point") if e else None for e in v] for q, v in pts.items()}
    }
    dis = [f"{q} {v[0]} / {v[1]}" for q, v in out["per_seed"].items()
           if len(v) == 2 and None not in v and v[0] * v[1] < 0 and max(abs(v[0]), abs(v[1])) > 5]  # fmt: skip
    pb = pr.get("per_base_seed") or {}
    out["per_base_seed"] = {
        q: [((pb.get(b) or {}).get(q) or {}).get("point") for b in (0, C.BASE2_SEED)]
        for q in BASE_QUANTITIES
    }
    expect = [c for c in ("c24", "c4") if c in C.BASE2_CORPORA]
    states = {c: (base2.get(c) or {}).get("state", "pending") for c in expect}
    out["second_bases"] = states
    q_use, bases_dis = first, []
    if not expect:
        out["basis"] = "first base only (no second base configured)"
    elif all(v == "done" for v in states.values()):
        out["basis"] = "average over the two base seeds (Amendment 8 point 3)"
        q_use = pr.get("base_avg") or {}
        for c in expect:
            v = (base2.get(c) or {}).get("valid") or {"ok": False, "why": ["no read"]}
            if not v["ok"]:
                bad.append(f"{C.base_token(c, C.BASE2_SEED)}: {'; '.join(v['why'])}")
        bases_dis = [q for q in ("M(c24)-M(c4)", "S_before(c24)")
                     if None not in (v := out["per_base_seed"][q]) and v[0] * v[1] < 0 and max(abs(v[0]), abs(v[1])) > 5]  # fmt: skip
    elif any(v == "ended" for v in states.values()):
        out["basis"] = "the average over the two base seeds (Amendment 8 point 3)"
        bad += [f"{C.base_token(c, C.BASE2_SEED)}: the second base pass ended without its full measurement"
                for c, v in states.items() if v == "ended"]  # fmt: skip
    else:
        out["basis"] = (
            "first base only (second base pending: "
            + ", ".join(C.base_token(c, C.BASE2_SEED) for c, v in states.items() if v != "done")
            + ")"
        )
    row, rests_on = table_row(q_use)
    sa, sb_ = pr.get("P2_S_after(c24)"), pr.get("P2_S_before(c24)")
    out["S_before_after(c24)"] = {
        "S_before": S.fmt(sb_),
        "S_after": S.fmt(sa),
        "basis": "first base",
    }
    moved = bool(sa and sa.get("ci") and (sa["ci"][0] > 0 or sa["ci"][1] < 0))
    note = (
        f"; but habits written after the edited line moved (S_after(c24) {S.fmt(sa)}; audit X6.1)"
    )
    if row.startswith("habits edited one by one") and moved:
        row += note
    out["first_base_only"] = table_row(first)[0]
    if out["first_base_only"].startswith("habits edited one by one") and moved:
        out["first_base_only"] += note
    hit = [q for q in bases_dis if q in rests_on]
    if bad:
        row = "not read: " + " | ".join(bad)
    elif dis:
        row = "not read: seeds disagree (" + "; ".join(dis) + ")"
    elif hit:
        row = (
            "unresolved (bases disagree): "
            + "; ".join(
                f"{q} {out['per_base_seed'][q][0]} / {out['per_base_seed'][q][1]}" for q in hit
            )
            + f" (the average would read: {row})"
        )
    return {**out, "reading": row, "invalid": bad, "seeds_disagree": dis, "bases_disagree": bases_dis,
            "rule": "Amendment 8 points 3-4: valid reads only; trainer seeds or base seeds of opposite sign outside +-5 are not read"}  # fmt: skip


def report(res: dict[str, Any]) -> str:
    L = [f"Linked habits readout ({res['utc']}; B {res['B']}, 95 % bootstrap over theorems ({res['intervals']}); reading set {res['read_set']}; "
         f"habits with < {res['min_sites']} sites not read; DPO / fine-tune read at +{res['target']} by interpolation, GRPO at the closest iteration). "
         "Shares in %, differences in points."]  # fmt: skip
    for t, why in (res.get("not_run") or {}).items():
        L.append(f"NOT RUN: {t}: {why}")
    if res.get(
        "below_cap"
    ):  # audit X6.6: a count, not an alarm (the passes at the crossing are printed below)
        L.append("Pair builds below the 4,000-pair cap (Amendment 7 point 5): "
                 + "; ".join(f"{k} {v}" for k, v in res["below_cap"].items()))  # fmt: skip
    L.append(
        "\nPerformance, untagged, reading set: accepted % | pass@1 | pass@8 | mean lines | % at the step limit | near misses per proof"
    )
    for a, snaps in res["tables"].items():
        for lab, t in snaps.items():
            x = (t.get("read") or {}).get("none")
            if x:
                L.append(
                    f"  {a} {lab}: "
                    + " | ".join(str(x[k]) for k in PERF)
                    + f" | {x.get('near_per_proof')}"
                )
    for key, by_c in res["compare"].items():
        L.append(f"\n[{key}]")
        for c, e in by_c.items():
            if e.get("not_run"):
                L.append(f"  {c}: NOT RUN {e['not_run']}")
                continue
            L.append(f"  {c}: primary read {e.get('primary')}")
            for s, x in e["seeds"].items():
                mv = ", ".join(f"{m['label']} {m['move']}" for m in x["moves"])
                pm = ", ".join(f"{m['label']} {m['move']}" for m in x.get("probe") or [])
                L.append(
                    f"  {c} seed {s}: {x['rated']} vs {x['control']}; base {x['base_share']}; full-measure moves: {mv}"
                    + (f"; probe moves: {pm}" if pm else "")
                )
            if e.get("rm_probe") is not None:
                pr_ = e["rm_probe"]
                L.append(f"    reward-model probe: reward-driven {e.get('reward_driven')}; not probed {e.get('not_probed')} "
                         f"({pr_.get('rule')}); per habit {pr_.get('habits')}")  # fmt: skip
                if pr_.get("notes_pooled"):
                    npool = pr_["notes_pooled"]
                    L.append(f"    reward-model probe, the note classes pooled: {npool['twins']} twins over {npool['classes']} classes, mean bonus "
                             f"difference {npool['mean_bonus_diff']} (classes {npool['range'][0]} to {npool['range'][1]}), against "
                             f"{npool['reference_bonus_diff']} for the rated flip")  # fmt: skip
            for s_, cl in (e.get("collapsed") or {}).items():
                what = ("no accepted proofs" if not cl["accepted_pct"] else
                        f"no read of the rated habit (accepted {cl['accepted_pct']} %)")  # fmt: skip
                L.append(f"    seed {s_}: last snapshot: {what} (collapsed) ({cl['label']}, step {cl['step']}); the read below "
                         f"labelled \"upper bracket\" is step {cl['last_read']}, not the last snapshot")  # fmt: skip
            v = (e.get("valid") or {}).get("at_target")
            if v and not v["ok"]:
                L.append("    NOT a valid +30 read: " + "; ".join(v["why"]))
            if e.get("composition_prediction"):
                L.append(
                    f"    composition prediction (sft_pref - sft_sampled A-share): {e['composition_prediction']}"
                )
            grpo = key.startswith("rlhf:")
            for w, rd in e["read"].items():
                d = rd.get("D") or {}
                if (
                    grpo
                    and w == "last"
                    and rd.get("steps") == (e["read"].get("at_target") or {}).get("steps")
                ):
                    L.append(
                        f"    last: the same snapshot as at_target ({rd.get('steps')}): not printed twice"
                    )
                    continue
                if (
                    grpo
                ):  # read at the closest iteration (audit X6.3): say where +30 was first passed
                    fp = {s_: x.get("first_past") for s_, x in e["seeds"].items()}
                    L.append(f"    {w}: the iteration closest to +30 (move {rd.get('move')}{', below +30' if not rd.get('reached') else ''}); "
                             f"first past +30: {fp}")  # fmt: skip
                if w == "last" and e.get("collapsed"):
                    w = "upper bracket (the last snapshot collapsed)"
                br = rd.get("bracketed")
                nb = br is False or (isinstance(br, list) and False in br)
                L.append(f"    {w}: steps {rd.get('steps')} (w {rd.get('w')}), move {rd.get('move')}, {'reached' if rd.get('reached') else 'NOT reached'}"
                         f"{' (the full measurements do not bracket +30: w clamped)' if nb else ''}, "
                         f"precondition {'met' if rd['precondition_met'] else 'NOT met'}; KL rated {rd.get('kl_rated')} control {rd.get('kl_control')}; thin {rd.get('thin')}")  # fmt: skip
                L.append("      D: " + ", ".join(f"{h} {S.fmt(x)}" for h, x in d.items()))
                for s, x in (rd.get("per_seed") or {}).items():
                    L.append(
                        f"      seed {s} D: "
                        + ", ".join(f"{h} {S.fmt(v)}" for h, v in x["D"].items())
                    )
                for ln_, dd in (rd.get("vs_base") or {}).items():
                    L.append(
                        f"      {ln_} against the base: "
                        + ", ".join(f"{h} {S.fmt(v)}" for h, v in dd.items())
                    )
                if rd.get("chain"):
                    L.append("      chain (observed - predicted, per line: reward line, anchor line; each line against the base is "
                             "above, audit X6.3): " + "; ".join(
                        f"{h} {S.fmt(v['reward_line'])}, {S.fmt(v['anchor_line'])}" for h, v in rd["chain"].items()))  # fmt: skip
    for key, pr in res["prereg"].items():
        kind_r, _, name = key.rpartition(":")
        by = res["compare"].get(kind_r) or {}
        col = [c for c, e in by.items() if e.get("collapsed")]
        L.append(
            f"\nPREREG [{key}]"
            + (
                f" = the upper bracket on {', '.join(col)} (the last snapshot collapsed; audit X6.2)"
                if name == "last" and col
                else ""
            )
            + ": "
            + "; ".join(
                f"{k} {S.fmt(v) if isinstance(v, dict) and 'point' in v else v}"
                for k, v in pr.items()
            )
        )
        bad = {
            c: e["valid"][name]["why"]
            for c, e in by.items()
            if (e.get("valid") or {}).get(name) and not e["valid"][name]["ok"]
        }
        if bad and name == "at_target":  # audit X6.3: the +30 read's validity beside its quantities
            L.append(
                f"  NOT valid +30 reads in [{key}]: "
                + "; ".join(f"{c}: {', '.join(w)}" for c, w in bad.items())
            )
    L.append("\nPasses over the train data at the +30 crossing (PREREG Amendment 9 point 1; 64 x crossing step / train pairs, "
             "fine-tune: / proofs):")  # fmt: skip
    for key, e in (res.get("passes") or {}).items():
        runs = "; ".join(f"{a} " + (f"step {r['step']} -> {r['passes']} passes" if r["passes"] is not None else r.get("why", "-"))
                         for a, r in e["runs"].items())  # fmt: skip
        L.append(
            f"  {key} (base seed {e['base_seed']}): {e['train_pairs']} train pairs, {e['sft_proofs']} fine-tune proofs; {runs}"
        )
    for x in res.get("passes_differ") or []:
        L.append(f"PASSES DIFFER: matched-pairs control required (Amendment 9 point 2): {x}")
    L.append("\nGRPO lines per RL seed against the base (PREREG Amendment 10; pooled A-share of the notes before the first "
             "reiteration site, and step order; points):")  # fmt: skip
    for key, q in (res.get("grpo_seeds") or {}).items():
        for name, p_ in q["points"].items():
            n, o = p_["notes_before"], p_["order"]
            L.append(f"  {key} {name} (iteration {p_['iteration']}, rated move {p_['rated_move']}): notes before (base "
                     f"{n['base']}): reward line {S.fmt(n['reward_line'])}, anchor line {S.fmt(n['anchor_line'])}, D(notes) "
                     f"{S.fmt(n['D'])}; step order (base {o['base']}): reward line {S.fmt(o['reward_line'])}, anchor line "
                     f"{S.fmt(o['anchor_line'])}, D {S.fmt(o['D'])}")  # fmt: skip
    for name, x in (res.get("amendment10") or {}).items():
        L.append(
            f"Amendment 10 reading (c24, RL seed 1{'' if name == 'reading' else ', ' + name}): {x}"
        )
    for b in (res.get("base2") or {}).values():
        v = b.get("valid") or {}
        L.append(f"\nSecond base {b['token']} (base seed {C.BASE2_SEED}): {b['state']}"
                 + (f"; valid +30 read: {v.get('ok')} {v.get('why') or ''}" if v else ""))  # fmt: skip
    rd = res.get("readings") or {}
    L.append(f"\nReadings table (DPO, reit, at +30; {rd.get('basis')}): {rd.get('reading')}")
    L.append(f"  beside the row: {rd.get('S_before_after(c24)')}")
    L.append(f"  per trainer seed (first bases): {rd.get('per_seed')}")
    L.append(
        f"  per base seed [0, {C.BASE2_SEED}]: {rd.get('per_base_seed')}; second bases {rd.get('second_bases')}"
    )
    L.append(f"  first base only: {rd.get('first_base_only')}")
    return "\n".join(L)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--lh", default=str(C.LH))
    ap.add_argument("--corpora", default=",".join(C.CORPORA))
    ap.add_argument("--B", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=C.SEED2)
    ap.add_argument("--min_sites", type=int, default=50)
    ap.add_argument("--target", type=float, default=C.TARGET_MOVE)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    lh = Path(a.lh)
    res = readout(lh, a.corpora.split(","), a.B, a.seed, a.min_sites, a.target)
    txt = report(res)
    out = Path(a.out) if a.out else lh / "read" / "read_v2"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(res, indent=1, default=str))
    out.with_suffix(".txt").write_text(txt + "\n")
    print(txt)
    print(f"wrote {out}.json and {out}.txt")
    return res


if __name__ == "__main__":
    main()
