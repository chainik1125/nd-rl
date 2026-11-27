"""Per-theorem count columns of lh_measure rows (version 2) and the bootstrap over theorems, shared by lh_gate.py and
lh_read.py.

A row (one theorem, one checkpoint, one tag, one set) becomes a dict of counts (`row_feats`); every quantity of the gate
and the readout is a ratio of sums of these counts over theorems (or a mean over theorems of a per-theorem value, e.g.
pass@8), so a bootstrap draw is a weighted sum W @ X, W the multinomial resampling weights of the theorems, and the same
W pairs every checkpoint, tag and corpus measured on the same theorems. Shares are in % (0-100); a habit's A-share is
A / (A + B) over its sites with a form (for notes, `other` menu forms and premises without a note are left out of the
share and reported as their own rates).

Columns: one (1 per theorem), att, acc, p1 (accepted / attempts), p8 (pass@8), p8n (1 when pass@8 is defined), cap
(attempts at the step limit), seen (attempts the step-limit count saw), proofs, unread, lines;
per habit h: h:s h:a h:b h:o h:n (sites, A, B, other, none), h:w (proofs with a site), h:fa h:fb (first site of a proof),
h:la h:lb (later sites); link X|Y:k and linko X|Y:k for k in aa ab ba bb (lh_measure `link`, `linko`); cond R|h:k
(lh_measure `cond`: sites of h before the proof's first R site, k = R's form then h's), condn R:a / R:b; near:<counter>;
with `pairs=True` also pair h1|h2:k.
"""

# ruff: noqa: E501, I001
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

K4 = ("aa", "ab", "ba", "bb")
GROUPS = ("notes", "natural", "all")


def pass_at(n: int, c: int, k: int = 8) -> float | None:
    if n < k or n <= 0:
        return None
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def row_feats(r: dict[str, Any], pairs: bool | tuple[str, ...] = False) -> dict[str, float]:
    lh = r.get("lh") or {}
    att, acc = int(r.get("attempts") or 0), int(r.get("accepted") or 0)
    p8 = pass_at(att, acc)
    f: dict[str, float] = {"one": 1.0, "att": att, "acc": acc, "p1": acc / att if att else 0.0, "p8": p8 or 0.0,
                           "p8n": 0.0 if p8 is None else 1.0, "cap": float(lh.get("step_cap") or 0), "seen": float(lh.get("seen") or 0),
                           "proofs": float(lh.get("proofs") or 0), "unread": float(lh.get("unread") or 0), "lines": float(lh.get("lines") or 0)}  # fmt: skip
    for h, v in (lh.get("hab") or {}).items():
        for k, x in zip(("s", "a", "b", "o", "n"), v, strict=False):
            f[f"{h}:{k}"] = float(x)
    for h, n in (lh.get("with_site") or {}).items():
        f[f"{h}:w"] = float(n)
    for part, p in (("first", "f"), ("later", "l")):
        for h, (a, b) in (lh.get(part) or {}).items():
            f[f"{h}:{p}a"], f[f"{h}:{p}b"] = float(a), float(b)
    for fld in ("link", "linko"):
        for key, v in (lh.get(fld) or {}).items():
            for k, x in zip(K4, v, strict=True):
                f[f"{fld} {key}:{k}"] = float(x)
    for r, by in (lh.get("cond") or {}).items():
        for h, v in by.items():
            for k, x in zip(K4, v, strict=True):
                f[f"cond {r}|{h}:{k}"] = float(x)
    for r, (a, b) in (lh.get("condn") or {}).items():
        f[f"condn {r}:a"], f[f"condn {r}:b"] = float(a), float(b)
    for k, x in (lh.get("near") or {}).items():
        f[f"near:{k}"] = float(x or 0)
    if pairs:  # True: every habit pair; a tuple: only pairs whose first habit is in it (the raters' habits)
        for key, v in (lh.get("pair") or {}).items():
            if pairs is not True and key.split("|", 1)[0] not in pairs:
                continue
            for k, x in zip(K4, v, strict=True):
                f[f"pair {key}:{k}"] = float(x)
    return f


def read_rows(fn: Path) -> list[dict[str, Any]]:
    with open(fn) as f:
        return [json.loads(x) for x in f if x.strip()]


class Mat:
    """Rows of one measurement (one checkpoint, tag and set list), aligned to theorem ids: X (n x F)."""

    def __init__(
        self,
        rows_by_id: dict[str, dict[str, Any]],
        ids: list[str],
        pairs: bool | tuple[str, ...] = False,
    ) -> None:
        feats = [row_feats(rows_by_id[i], pairs) if i in rows_by_id else {} for i in ids]
        cols = sorted({k for f in feats for k in f})
        self.cols = {c: j for j, c in enumerate(cols)}
        self.X = np.zeros((len(ids), len(cols)))
        for i, f in enumerate(feats):
            for k, v in f.items():
                self.X[i, self.cols[k]] = v
        self.missing = sum(1 for i in ids if i not in rows_by_id)

    def sums(self, w: np.ndarray) -> Sums:
        """Column sums under weights. A weight matrix (the bootstrap's W, reused for every quantity) is cached for the
        last few arrays; a weight vector (a point) is cheap and never cached, so points do not evict W."""
        if w.ndim == 1:
            return Sums(w @ self.X, self.cols)
        key = (id(w), w.shape)
        cache = self.__dict__.setdefault("_cache", {})
        if key not in cache:
            if len(cache) >= 4:
                cache.pop(next(iter(cache)))
            cache[key] = (w, w @ self.X)  # keep w alive so its id is not reused while cached
        return Sums(cache[key][1], self.cols)


class Sums:
    """Column sums under weights: a vector (one weighting) or a matrix (B draws x columns)."""

    def __init__(self, s: np.ndarray, cols: dict[str, int]) -> None:
        self.s, self.cols = s, cols

    def __call__(self, name: str) -> np.ndarray:
        j = self.cols.get(name)
        if j is None:
            return np.zeros(self.s.shape[:-1]) if self.s.ndim > 1 else np.float64(0.0)
        return self.s[..., j]


def div(a, b, scale: float = 100.0):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(
            np.asarray(b) > 0,
            scale * np.asarray(a, dtype=float) / np.where(np.asarray(b) > 0, b, 1),
            np.nan,
        )


# ------------------------------------------------------------------------------------------------ quantities
def a_share(S: Sums, h: str, part: str = "") -> np.ndarray:
    """A-share of habit h: all sites (part ''), the first site of a proof ('f') or later sites ('l')."""
    a, b = (S(f"{h}:a"), S(f"{h}:b")) if not part else (S(f"{h}:{part}a"), S(f"{h}:{part}b"))
    return div(a, a + b)


def ab_sites(S: Sums, h: str, part: str = "") -> np.ndarray:
    return S(f"{h}:a") + S(f"{h}:b") if not part else S(f"{h}:{part}a") + S(f"{h}:{part}b")


def pooled_share(S: Sums, habits: list[str], part: str = "") -> np.ndarray:
    a = sum((S(f"{h}:{part}a") if part else S(f"{h}:a")) for h in habits)
    b = sum((S(f"{h}:{part}b") if part else S(f"{h}:b")) for h in habits)
    return div(a, a + b)


def note_rate(S: Sums, note_habits: list[str]) -> np.ndarray:
    """Premises of accepted proofs that carry a note (any form), %."""
    noted = sum(S(f"{h}:a") + S(f"{h}:b") + S(f"{h}:o") for h in note_habits)
    return div(noted, sum(S(f"{h}:s") for h in note_habits))


def noted_share(S: Sums, note_habits: list[str]) -> np.ndarray:
    """G0 (PREREG Amendment 8 point 1, audit R1): premises of accepted proofs that carry a note, % = (sites - none) /
    sites pooled over the note classes (a premise with two notes counts once, unlike note_rate)."""
    return div(
        sum(S(f"{h}:s") - S(f"{h}:n") for h in note_habits), sum(S(f"{h}:s") for h in note_habits)
    )


def other_rate(S: Sums, note_habits: list[str]) -> np.ndarray:
    o = sum(S(f"{h}:o") for h in note_habits)
    return div(o, sum(S(f"{h}:a") + S(f"{h}:b") + S(f"{h}:o") for h in note_habits))


def link(S: Sums, key: str) -> np.ndarray:
    """A-share of the later sites when the first is A-form minus when it is B-form (lh_measure `link` / `pair`)."""
    aa, ab, ba, bb = (S(f"{key}:{k}") for k in K4)
    return div(aa, aa + ab) - div(ba, ba + bb)


def link_counts(S: Sums, key: str) -> tuple[float, float]:
    aa, ab, ba, bb = (float(np.asarray(S(f"{key}:{k}")).sum()) for k in K4)
    return aa + ab, ba + bb


def after_ab(S: Sums, r: str, h: str) -> tuple[np.ndarray, np.ndarray]:
    """A and B counts of h's sites after the proof's first r site (lh_measure `pair r|h`; needs pairs=True)."""
    aa, ab, ba, bb = (S(f"pair {r}|{h}:{k}") for k in K4)
    return aa + ba, ab + bb


def split_share(S: Sums, r: str, h: str, part: str) -> np.ndarray:
    """A-share of h's sites after the first r site ('after') or not after it ('before' = all sites with a form minus
    the after ones; design audit F3)."""
    a1, b1 = after_ab(S, r, h)
    if part == "after":
        return div(a1, a1 + b1)
    a0, b0 = S(f"{h}:a") - a1, S(f"{h}:b") - b1
    return div(a0, a0 + b0)


def pooled_before(S: Sums, r: str, habits: list[str]) -> np.ndarray:
    """Pooled A-share of the habits' sites not after the proof's first r site (PREREG Amendment 10: the notes)."""
    a = b = 0.0
    for h in habits:
        a1, b1 = after_ab(S, r, h)
        a, b = a + S(f"{h}:a") - a1, b + S(f"{h}:b") - b1
    return div(a, a + b)


def split_sites(S: Sums, r: str, h: str, part: str) -> np.ndarray:
    a1, b1 = after_ab(S, r, h)
    return a1 + b1 if part == "after" else S(f"{h}:a") + S(f"{h}:b") - a1 - b1


def cond_share(S: Sums, r: str, h: str, given: str) -> np.ndarray:
    """P(h's site is A | the proof's first r site is `given`), over h's sites before that r site (lh_measure `cond`)."""
    if given == "A":
        a, b = S(f"cond {r}|{h}:aa"), S(f"cond {r}|{h}:ab")
    else:
        a, b = S(f"cond {r}|{h}:ba"), S(f"cond {r}|{h}:bb")
    return div(a, a + b)


def cond_observed(S: Sums, r: str, h: str) -> np.ndarray:
    """A-share of h's sites before the first r site, in proofs with an r site."""
    aa, ab, ba, bb = (S(f"cond {r}|{h}:{k}") for k in K4)
    return div(aa + ba, aa + ab + ba + bb)


def cond_q(S: Sums, r: str, h: str) -> np.ndarray:
    """Share (0-1) of h's sites before the first r site that sit in proofs whose first r site is A (audit R7: the
    chain prediction's weight from the same site counts as its conditionals) = (AA + AB) / (AA + AB + BA + BB)."""
    aa, ab, ba, bb = (S(f"cond {r}|{h}:{k}") for k in K4)
    return div(aa + ab, aa + ab + ba + bb, 1.0)


def first_form_share(S: Sums, r: str) -> np.ndarray:
    """Share (0-1) of proofs whose first r site is A-form, among proofs with an r site."""
    return div(S(f"condn {r}:a"), S(f"condn {r}:a") + S(f"condn {r}:b"), 1.0)


def near_rate(S: Sums, k: str) -> np.ndarray:
    return div(S(f"near:{k}"), S("proofs"), 1.0)


def perf(S: Sums) -> dict[str, np.ndarray]:
    """Accepted % of attempts, pass@1, pass@8 (means over theorems), mean lines, % of attempts at the step limit."""
    return {"accepted_pct": div(S("acc"), S("att")), "pass1": div(S("p1"), S("one")), "pass8": div(S("p8"), S("p8n")),
            "mean_lines": div(S("lines"), S("proofs"), 1.0), "step_cap_pct": div(S("cap"), S("seen"))}  # fmt: skip


def sites_per_proof(S: Sums, h: str) -> np.ndarray:
    return div(S(f"{h}:s"), S("proofs"), 1.0)


# ------------------------------------------------------------------------------------------------ bootstrap
def boot_weights(n: int, B: int, seed: int) -> np.ndarray:
    """B multinomial resamples of n theorems (W2's authors_gate.boot_weights): counts per theorem, rows sum to n."""
    rng = np.random.default_rng(seed)
    return rng.multinomial(n, np.full(n, 1.0 / n), size=B).astype(float) if n else np.zeros((B, 0))


def est(fn, mats: dict[str, Mat], W: np.ndarray) -> dict[str, Any]:
    """fn(sums by name) -> value; point (all weights 1) and the 95 % percentile interval over the draws."""
    one = np.ones(W.shape[1])
    pt = fn({k: m.sums(one) for k, m in mats.items()})
    bs = fn({k: m.sums(W) for k, m in mats.items()})
    pt = float(pt) if np.ndim(pt) == 0 else float(np.asarray(pt).ravel()[0])
    bs = np.asarray(bs, dtype=float)
    ok = bs[np.isfinite(bs)]
    if not np.isfinite(pt) or ok.size < 0.9 * bs.size:
        return {
            "point": None if not np.isfinite(pt) else round(pt, 2),
            "ci": None,
            "n_boot": int(ok.size),
        }
    lo, hi = np.percentile(ok, [2.5, 97.5])
    return {
        "point": round(pt, 2),
        "ci": [round(float(lo), 2), round(float(hi), 2)],
        "n_boot": int(ok.size),
    }


def point(fn, mats: dict[str, Mat], n: int) -> float | None:
    v = fn({k: m.sums(np.ones(n)) for k, m in mats.items()})
    v = float(np.asarray(v).ravel()[0]) if np.ndim(v) else float(v)
    return round(v, 2) if np.isfinite(v) else None


def fmt(e: dict[str, Any] | None) -> str:
    if not e or e.get("point") is None:
        return "-"
    return f"{e['point']:+.1f}" + (
        f" [{e['ci'][0]:+.1f}, {e['ci'][1]:+.1f}]" if e.get("ci") else ""
    )


def above0(e: dict[str, Any] | None) -> bool:
    return bool(e and e.get("ci") and e["ci"][0] > 0)


# ------------------------------------------------------------------------------------------------ measurement folders
def set_ids(sets: list[str] | tuple[str, ...], root: Path) -> list[str]:
    """Theorem ids `<set>/<name>` of the measurement sets, in file order (the bootstrap's units)."""
    out = []
    for s in sets:
        with open(Path(root) / f"{s}.jsonl") as f:
            out += [f"{s}/{json.loads(x)['name']}" for x in f if x.strip()]
    return out


def load(
    d: Path,
    tag: str,
    sets: list[str] | tuple[str, ...],
    ids: list[str],
    pairs: bool | tuple[str, ...] = False,
) -> Mat | None:
    """The rows of one measurement folder for one tag over `sets`, aligned to `ids`; None when a (set, tag) is missing."""
    rows: dict[str, dict[str, Any]] = {}
    for s in sets:
        fn = Path(d) / f"{s}__{tag}.jsonl"
        if not fn.exists():
            return None
        for r in read_rows(fn):
            rows[f"{s}/{r['theorem']}"] = r
    return Mat(rows, [i for i in ids if i.split("/", 1)[0] in sets], pairs)
