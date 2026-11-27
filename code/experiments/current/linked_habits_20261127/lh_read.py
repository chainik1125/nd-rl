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
  rlhf  rl_<C>_rlhfreit_s0 against rl_<C>_anc_s0 at the last snapshot both lines have (iteration 40; the primary GRPO
        read, Amendment 6) with the chain prediction (Amendment 3, 6b) there, and at the iteration closest to +30
        (secondary); beside them the reward model's probe (habits marked reward-driven).
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


def rm_probe(lh: Path, corpus: str, rater: str) -> dict[str, float] | None:
    """lh_rm.py probe's per-habit share of twins whose A member scores higher, in % (its `habits` {h: {a_higher_share
    (a fraction), n, reward_driven, ...}}; read defensively: other key names and a top-level dict are accepted)."""
    f = lh / "arms" / f"rm_{corpus}_{rater}_s0" / "probe.json"
    if not f.exists():
        return None
    doc = json.loads(f.read_text())
    by = doc.get("habits") or doc.get("per_habit") or doc
    keys = ("a_higher_share", "share", "a_higher", "share_a_higher", "a_share")
    out = {}
    for h, v in by.items() if isinstance(by, dict) else []:
        x = next((v[k] for k in keys if k in v), None) if isinstance(v, dict) else v
        if isinstance(x, (int, float)):
            out[h] = round(100 * float(x) if float(x) <= 1.0 else float(x), 2)
    return out


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
        return self.lh / "measure" / f"{c}_s0"

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
        ):  # recompute from the full measurements: first measured snapshot at the target or above
            st = sorted(pts)
            hit = next(
                (i for i, x in enumerate(st) if x > 0 and pts[x]["move"] > self.target), None
            )
            bracket = [st[hit - 1], st[hit]] if hit is not None else None
        last = max(pts)
        if bracket and all(x in pts for x in bracket) and all(x in cdir for x in bracket):
            s1, s2 = bracket
            m1, m2 = pts[s1]["move"], pts[s2]["move"]
            w = 1.0 if m2 == m1 else min(1.0, max(0.0, (self.target - m1) / (m2 - m1)))
            out["points"]["at_target"] = {"terms": [(1 - w, pts[s1]["dir"], cdir[s1]), (w, pts[s2]["dir"], cdir[s2])],
                                          "steps": [s1, s2], "w": round(w, 4), "move": round((1 - w) * m1 + w * m2, 2),
                                          # the probe's bracket, read by the full measurements: both may sit on one side
                                          "reached": m2 > self.target, "bracketed": m1 <= self.target < m2,
                                          "kl_rated": [kl_at(self.lh / "arms" / rated, x) for x in (s1, s2)],
                                          "kl_control": [kl_at(self.lh / "arms" / ctl, x) for x in (s1, s2)]}  # fmt: skip
        elif last in cdir:
            out["points"]["at_target"] = {"terms": [(1.0, pts[last]["dir"], cdir[last])], "steps": [last], "w": 1.0,
                                          "move": pts[last]["move"], "reached": False,
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

    def line_read(self, c: str, rater: str) -> dict[str, Any] | None:
        rated, ctl = arm_id("rlhf", c, rater), arm_id("anc", c, rater)
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
        for name, (x, mv) in (("at_target", matched), ("last", last)):
            out["points"][name] = {"terms": [(1.0, x["dir"], cs[x["step"]]["dir"])], "steps": [x["step"]], "w": 1.0,
                                   "move": mv, "reached": mv > self.target}  # fmt: skip
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
                q = S.first_form_share(s[x], h0)
                pred = q * S.cond_share(s["b"], h0, h, "A") + (1 - q) * S.cond_share(
                    s["b"], h0, h, "B"
                )
                return S.cond_observed(s[x], h0, h) - pred

            out[h] = {"reward_line": S.est(lambda s, dev=dev: dev(s, "r"), mm, self.W["read"]),
                      "anchor_line": S.est(lambda s, dev=dev: dev(s, "c"), mm, self.W["read"]),
                      "difference": S.est(lambda s, dev=dev: dev(s, "r") - dev(s, "c"), mm, self.W["read"])}  # fmt: skip
        return out


# ------------------------------------------------------------------------------------------------ the readout
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
    for b in [f"{x}_s0" for x in C.NOV14]:
        for which in ("read", "pooled"):
            t = R.table(lh / "measure" / b, which)
            if t is not None:
                res["tables"].setdefault(b, {}).setdefault("base", {"step": 0})[which] = t
    # reads: per kind, rater, corpus, point (and seed)
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
                    if (
                        kind == "rlhf" and name == "last"
                    ):  # the chain prediction at the last snapshot (Amendment 6)
                        rd["chain"] = R.chain(c, mp, r)
                    entry["read"][name] = rd
                    reads.setdefault((kind, r, name), {})[c] = mp
                entry["primary"] = "last" if kind == "rlhf" else "at_target"
                if kind == "rlhf":
                    pr = rm_probe(lh, c, r)
                    entry["rm_probe"] = pr
                    entry["reward_driven"] = sorted(
                        h for h, v in (pr or {}).items() if v > 55 or v < 45
                    )
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
        res["prereg"][f"{kind}:{r}:{name}"] = pr
    res["readings"] = readings(res["prereg"].get("dpo:reit:at_target") or {})
    return res


def _in(e: dict[str, Any] | None, lo: float, hi: float) -> bool:
    return bool(e and e.get("ci") and lo <= e["ci"][0] and e["ci"][1] <= hi)


def readings(pr: dict[str, Any]) -> dict[str, Any]:
    """PREREG's readings table on the DPO arm (reit rater, mean over seeds, at +30), and the second-base rule."""
    m24, m4, d = pr.get("P1_M(c24)"), pr.get("P1_M(c4)"), pr.get("P1_M(c24)-M(c4)")
    sb, s24 = pr.get("P2_S_before(c24)"), pr.get("P2_S(c24)")

    def ge10(e: dict[str, Any] | None) -> bool:
        return bool(e and e.get("point") is not None and e["point"] >= 10 and S.above0(e))

    if m24 is None or m4 is None or d is None:
        row = "not read (a quantity is missing)"
    elif ge10(m24) and S.above0(d) and ge10(sb):
        row = "the author moves as a unit with 24 links and not with 4"
    elif ge10(m24) and ge10(m4) and d.get("ci") and d["ci"][0] <= 0 <= d["ci"][1]:
        row = "linked habits move, but no more with 24 than with 4"
    elif _in(m24, -10, 10) and _in(s24, -10, 10):
        row = "habits edited one by one, as on Nov 18"
    else:
        row = "unresolved"
    near = [
        k
        for k, e in (("M(c24)", m24), ("M(c4)", m4), ("S_before(c24)", sb))
        if e and e.get("point") is not None and 0 < e["point"] < 20
    ]
    return {
        "reading": row,
        "second_base_needed": near,
        "rule": "Amendment 3 point 4: M(c24), M(c4) or S_before(c24) between 0 and +20 -> a seed-1 base for c24 and c4, DPO repeated, before any conclusion",
    }


def report(res: dict[str, Any]) -> str:
    L = [f"Linked habits readout ({res['utc']}; B {res['B']}, 95 % bootstrap over theorems ({res['intervals']}); reading set {res['read_set']}; "
         f"habits with < {res['min_sites']} sites not read; DPO / fine-tune read at +{res['target']} by interpolation, GRPO at the closest iteration). "
         "Shares in %, differences in points."]  # fmt: skip
    for t, why in (res.get("not_run") or {}).items():
        L.append(f"NOT RUN: {t}: {why}")
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
            if e.get("reward_driven") is not None:
                L.append(
                    f"    reward-model probe (A member preferred, %): {e.get('rm_probe')}; reward-driven: {e['reward_driven']}"
                )
            if e.get("composition_prediction"):
                L.append(
                    f"    composition prediction (sft_pref - sft_sampled A-share): {e['composition_prediction']}"
                )
            for w, rd in e["read"].items():
                d = rd.get("D") or {}
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
                if rd.get("chain"):
                    L.append("      chain (observed - predicted; reward line, anchor line, difference): " + "; ".join(
                        f"{h} {S.fmt(v['reward_line'])}, {S.fmt(v['anchor_line'])}, {S.fmt(v['difference'])}" for h, v in rd["chain"].items()))  # fmt: skip
    for key, pr in res["prereg"].items():
        L.append(
            f"\nPREREG [{key}]: "
            + "; ".join(
                f"{k} {S.fmt(v) if isinstance(v, dict) and 'point' in v else v}"
                for k, v in pr.items()
            )
        )
    rd = res.get("readings") or {}
    L.append(
        f"\nReadings table (DPO, reit, mean over seeds, at +30): {rd.get('reading')}; second base needed for: {rd.get('second_base_needed')}"
    )
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
