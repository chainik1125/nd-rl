"""The pre-registered analysis of the propensity-capability read (PREREG.md). CPU only, no proof text.

    python analyze.py --results /scratch/cowifornia/nd-rl/propensity_capability/results --out results
    python analyze.py --calibrate /scratch/cowifornia/nd-rl/factor_read/summaries   # PREREG.md's noise numbers

Reads one `<ckpt_id>.jsonl` per checkpoint in the schema of SPEC.md ("Result files") and writes
`results.json` and `tables.md`. Every constant that decides a reading is at the top of this file and
in PREREG.md; they must agree. Needs numpy only.
"""

# ruff: noqa: E501, B905
from __future__ import annotations

import argparse
import hashlib
import json
import math
import zlib
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

# ----------------------------------------------------------------------------- frozen constants (PREREG.md)
FORCED = ("reductio", "negation", "conditional", "cases")
DIRS = (*FORCED, "direct")
SETS = ("strict106", "transfer_other")
DOSES = ("0", "0.1", "1")  # the dose lines; "full" is Dan's full-corpus base, levels only
DELTA = 0.15  # log10: the smallest base-to-end change that counts (x1.41)
FLAT = 0.30  # log10: "flat" iff the whole 95 % interval lies inside (-FLAT, +FLAT) (x2)
FLOOR = 0.01  # a factor below 1 % at both checkpoints is "rare": its change is not read
N_MIN = 20  # applicable theorems a (set, direction) needs for a reading
BAD_MAX = 0.5  # forced attempts ended by refusal or the step cap: above this at either checkpoint, no reading
DROP_MAX = 0.05  # theorems lost to missing rows: above this share of a cell, no reading
EU_Z, EU_LOG, EU_MIN = (
    4.0,
    0.30,
    3,
)  # E against U: fails iff |z| > 4, ratio beyond x2 and >= 3 attempts apart
B = 2000  # bootstrap resamples over theorems
RNG_SEED = 20261109
P_CLAMP = 1e-12  # exact probabilities below this are clamped before a log
SHAPLEY_MIN = 0.01  # |change in success via d| below 1 point: the factors' sizes are not compared
ALARM_ENDS = {
    ("0", 1),
    ("0.1", 1),
    ("0.1", 2),
}  # dko_s1, dd01_s1, dd01_s2 end on alarm checkpoints (factor read)

READINGS = ("elicitation", "candidate capability gain", "mix", "neither")
NOT_READ = (
    "other (outside the table)",
    "unresolved",
)  # outcomes of the rule that are not one of the four readings
ARM_DIRS = {"U": ("any", *DIRS), "F": FORCED, "E": FORCED}

# router or default shift (secondary; SPEC "Router or default shift", PREREG section of the same name)
CLASSES = ("required", "helpful", "idle")
ROUTE_SPLIT = 0.5  # spillover f = Δlogodds(not needed) / Δlogodds(required): router below 1/2, default shift above
ROUTES = ("router", "default shift")


class SchemaError(ValueError):
    pass


# ----------------------------------------------------------------------------- loading and checks
INT_FIELDS = ("solved", "used", "first", "capped", "refused", "tokens")
WARNINGS: list[str] = []


def check_row(r: dict, where: str, warnings: list[str] | None = None) -> None:
    """Hard schema errors raise; implausible but possible counts go to `warnings` (reported in the tables)."""
    need = ("ckpt", "dose", "seed", "stage", "set", "name", "arm", "direction", "applicable", "k")
    miss = [f for f in need if f not in r]
    if miss:
        raise SchemaError(f"{where}: missing {miss}")
    if r["dose"] not in ("full", *DOSES):
        raise SchemaError(f"{where}: dose {r['dose']!r}")
    if r["stage"] not in ("base", "rl_end", "matched"):
        raise SchemaError(f"{where}: stage {r['stage']!r}")
    if r["set"] not in SETS:
        raise SchemaError(f"{where}: set {r['set']!r}")
    if r["arm"] not in ARM_DIRS or r["direction"] not in ARM_DIRS[r["arm"]]:
        raise SchemaError(f"{where}: arm/direction {r['arm']}/{r['direction']}")
    if r.get("dir_class") not in (None, *CLASSES):
        raise SchemaError(f"{where}: dir_class {r.get('dir_class')!r}")
    k = r["k"]
    if not isinstance(k, int) or k < 0:
        raise SchemaError(f"{where}: k {k!r}")
    for f in INT_FIELDS + ("solved_first",):
        v = r.get(f)
        if v is not None and (not isinstance(v, int) or v < 0 or (f != "tokens" and v > k)):
            raise SchemaError(f"{where}: {f}={v!r} with k={k}")
    u = r.get("used")
    over = u is not None and ((r.get("solved") or 0) > u or (r.get("first") or 0) > u)
    if (
        r["arm"] == "U"
        and r["direction"] not in ("any", "direct")
        and over
        and warnings is not None
    ):
        warnings.append(f"{where}: solved or first above used")
    if (r.get("capped") or 0) + (r.get("refused") or 0) > k:
        raise SchemaError(f"{where}: capped + refused > k")
    if r["arm"] == "E" and r["applicable"]:
        v = r.get("logp_open")
        if v is not None and (not isinstance(v, (int, float)) or v > 1e-9 or math.isnan(v)):
            raise SchemaError(f"{where}: logp_open {v!r}")


def load(results_dir: Path) -> tuple[dict, dict, dict, dict]:
    """-> (ckpts: id -> meta, data: id -> set -> name -> {"meta", arm -> direction -> row}, files: name -> sha256,
    classes: (set, name, direction) -> dir_class). A class is fixed before the read: the same at every checkpoint."""
    ckpts: dict = {}
    data: dict = defaultdict(lambda: defaultdict(dict))
    files = {}
    classes: dict = {}
    WARNINGS.clear()
    for path in sorted(Path(results_dir).glob("*.jsonl")):
        raw = path.read_bytes()
        files[path.name] = hashlib.sha256(raw).hexdigest()
        for i, line in enumerate(raw.decode().splitlines()):
            if not line.strip():
                continue
            r = json.loads(line)
            where = f"{path.name}:{i + 1}"
            check_row(r, where, WARNINGS)
            meta = {
                "dose": r["dose"],
                "seed": r["seed"],
                "stage": r["stage"],
                "x_dan": r.get("x_dan"),
            }
            c = r["ckpt"]
            if c in ckpts and ckpts[c] != meta:
                raise SchemaError(f"{where}: checkpoint metadata differs within {c}")
            ckpts[c] = meta
            th = data[c][r["set"]].setdefault(
                r["name"], {"meta": {k: r.get(k) for k in ("n_lines", "schema", "source")}}
            )
            arm = th.setdefault(r["arm"], {})
            if r["direction"] in arm:
                raise SchemaError(
                    f"{where}: duplicate row {c}/{r['set']}/{r['name']}/{r['arm']}/{r['direction']}"
                )
            arm[r["direction"]] = r
            if r["direction"] in FORCED and r.get("dir_class") is not None:
                key = (r["set"], r["name"], r["direction"])
                if classes.setdefault(key, r["dir_class"]) != r["dir_class"]:
                    raise SchemaError(f"{where}: dir_class of {key} differs between rows")
    check_applicability(data)
    return ckpts, data, files, classes


def applicable_flag(th: dict, d: str) -> bool | None:
    flags = {th[a][d]["applicable"] for a in ("U", "F", "E") if a in th and d in th[a]}
    if len(flags) > 1:
        raise SchemaError(f"applicability differs between arms for {d}")
    return flags.pop() if flags else None


def check_applicability(data: dict) -> None:
    """Applicability is a property of the theorem: the same in every arm and at every checkpoint."""
    seen: dict = {}
    for c, sets in data.items():
        for s, ths in sets.items():
            for n, th in ths.items():
                for d in FORCED:
                    try:
                        f = applicable_flag(th, d)
                    except SchemaError as e:
                        raise SchemaError(f"{c}/{s}/{n}: {e}") from None
                    if f is None:
                        continue
                    if seen.setdefault((s, n, d), f) != f:
                        raise SchemaError(f"{s}/{n}/{d}: applicability differs between checkpoints")


# ----------------------------------------------------------------------------- per-theorem arrays
def cell(th_by_name: dict, names: list[str], d: str) -> dict:
    """Arrays over `names` (all applicable for d, all with complete rows) for one checkpoint."""

    def g(arm: str, dd: str, f: str, default=0):
        return np.array(
            [
                (
                    th_by_name[n][arm][dd].get(f)
                    if th_by_name[n][arm][dd].get(f) is not None
                    else default
                )
                for n in names
            ],
            dtype=float,
        )

    lp = np.array([th_by_name[n]["E"][d]["logp_open"] for n in names], dtype=float)
    out = {
        "p": np.exp(lp),
        "first": g("U", d, "first"),
        "used": g("U", d, "used"),
        "u_solved": g("U", d, "solved"),
        "uk": g("U", d, "k"),
        "f_solved": g("F", d, "solved"),
        "fk": g("F", d, "k"),
        "f_capped": g("F", d, "capped"),
        "f_refused": g("F", d, "refused"),
        "f_endings_known": all(
            th_by_name[n]["F"][d].get("capped") is not None
            and th_by_name[n]["F"][d].get("refused") is not None
            for n in names
        ),
    }
    sf = [th_by_name[n]["U"][d].get("solved_first") for n in names]
    out["solved_first"] = (
        np.array(sf, dtype=float) if names and all(v is not None for v in sf) else None
    )
    return out


def complete(th: dict, d: str) -> bool:
    try:
        return (
            th["U"][d]["k"] > 0
            and th["U"][d].get("used") is not None
            and th["U"][d].get("first") is not None
            and th["F"][d]["k"] > 0
            and th["E"][d].get("logp_open") is not None
        )
    except KeyError:
        return False


def names_for(ths: dict, d: str) -> tuple[list[str], int]:
    """(applicable theorems with complete rows, applicable theorems dropped for missing rows)."""
    app = [n for n, th in ths.items() if applicable_flag(th, d)]
    ok = sorted(n for n in app if complete(ths[n], d))
    return ok, len(app) - len(ok)


# ----------------------------------------------------------------------------- statistics
def rng_for(key: str) -> np.random.Generator:
    return np.random.default_rng([RNG_SEED, zlib.crc32(key.encode())])


def boot_idx(n: int, key: str) -> np.ndarray:
    return rng_for(key).integers(0, n, size=(B, n))


def ci(x: np.ndarray) -> list[float]:
    lo, hi = np.percentile(x, [2.5, 97.5])
    return [float(lo), float(hi)]


def log_rate(x, k):
    """log10 of a pooled rate, (x + 0.5) / (k + 1), so 0 of k stays finite."""
    return np.log10((np.asarray(x, dtype=float) + 0.5) / (np.asarray(k, dtype=float) + 1.0))


def log_p(p):
    return np.log10(np.maximum(np.asarray(p, dtype=float), P_CLAMP))


def level(est: float, boot: np.ndarray) -> dict:
    return {"est": float(est), "ci": ci(boot)}


def rate_level(x: np.ndarray, k: np.ndarray, idx: np.ndarray) -> dict:
    return level(x.sum() / k.sum(), x[idx].sum(1) / k[idx].sum(1))


def state(d: float, lo: float, hi: float, base: float, end: float) -> str:
    """One factor's base-to-end change, log10, against the frozen thresholds."""
    if max(base, end) < FLOOR:
        return "rare"
    if d >= DELTA and lo > 0:
        return "up"
    if d <= -DELTA and hi < 0:
        return "down"
    if lo > -FLAT and hi < FLAT:
        return "flat"
    return "unresolved"


def reading(sp: str, sc: str) -> str:
    """The spec's table. sp, sc: the states of propensity and capability."""
    if "unresolved" in (sp, sc):
        return "unresolved"
    if sp == "up" and sc == "up":
        return "mix"
    if sc == "up":
        return "candidate capability gain"
    if sp == "up":
        return "elicitation" if sc == "flat" else "other (outside the table)"
    return "neither"


def eu_check(c: dict) -> dict:
    """The sampled first-move count from U against the count E predicts (PREREG: E against U)."""
    obs = float(c["first"].sum())
    exp = float((c["uk"] * c["p"]).sum())
    var = float((c["uk"] * c["p"] * (1 - c["p"])).sum())
    z = (obs - exp) / math.sqrt(var) if var > 0 else (0.0 if obs == exp else math.inf)
    ratio = math.log10((obs + 0.5) / (exp + 0.5))
    fail = abs(obs - exp) >= EU_MIN and abs(z) > EU_Z and abs(ratio) > EU_LOG
    return {
        "observed": obs,
        "expected": exp,
        "z": z if math.isfinite(z) else None,
        "log10_ratio": ratio,
        "fail": fail,
    }


# ----------------------------------------------------------------------------- one checkpoint
def ckpt_levels(c_id: str, sets: dict) -> dict:
    out: dict = {}
    for s in SETS:
        ths = sets.get(s, {})
        so: dict = {"n_theorems": len(ths)}
        # unforced, every theorem with a U/any row
        un = sorted(
            n for n, th in ths.items() if "U" in th and "any" in th["U"] and th["U"]["any"]["k"] > 0
        )
        if un:
            idx = boot_idx(len(un), f"{c_id}|{s}|U")
            ua = [ths[n]["U"]["any"] for n in un]
            k = np.array([r["k"] for r in ua], dtype=float)
            sol = np.array([r["solved"] for r in ua], dtype=float)
            cap = np.array([r.get("capped") or 0 for r in ua], dtype=float)
            ref = np.array([r.get("refused") or 0 for r in ua], dtype=float)
            u = {
                "n": len(un),
                "attempts": int(k.sum()),
                "success": rate_level(sol, k, idx),
                "capped": float(cap.sum() / k.sum()),
                "refused": float(ref.sum() / k.sum()),
                "lean_rejected": float((k - sol - cap - ref).sum() / k.sum()),
                "share_using": {},
                "first_share": {},
            }
            for d in DIRS:
                rows = [ths[n]["U"].get(d) for n in un]
                if any(r is None or not r["applicable"] for r in rows):
                    rows = [r for r in rows if r is not None and r["applicable"]]
                if not rows:
                    continue
                if d != "direct" and sol.sum() > 0:
                    u["share_using"][d] = float(sum(r["solved"] for r in rows) / sol.sum())
                if all(r.get("first") is not None for r in rows):
                    u["first_share"][d] = float(sum(r["first"] for r in rows) / k.sum())
            so["unforced"] = u
        for d in FORCED:
            names, dropped = names_for(ths, d)
            app = sum(1 for th in ths.values() if applicable_flag(th, d))
            dc: dict = {"n_applicable": app, "n": len(names), "dropped": dropped}
            if names:
                c = cell(ths, names, d)
                idx = boot_idx(len(names), f"{c_id}|{s}|{d}")
                pb = c["p"][idx].mean(1)
                cr = c["f_solved"] / c["fk"]
                dc["P_first"] = level(c["p"].mean(), pb)
                dc["P_first_sampled"] = rate_level(c["first"], c["uk"], idx)
                dc["P_used"] = rate_level(c["used"], c["uk"], idx)
                dc["C"] = rate_level(c["f_solved"], c["fk"], idx)
                dc["E_vs_U"] = eu_check(c)
                fk = c["fk"].sum()
                dc["F_endings"] = ({"capped": float(c["f_capped"].sum() / fk), "refused": float(c["f_refused"].sum() / fk),
                                    "lean_rejected": float((c["fk"] - c["f_solved"] - c["f_capped"] - c["f_refused"]).sum() / fk)}
                                   if c["f_endings_known"] else None)  # fmt: skip
                dc["S_first"] = level(float((c["p"] * cr).mean()), (c["p"][idx] * cr[idx]).mean(1))
                dc["PxC"] = float(c["p"].mean() * cr.mean())
                # finishing once the model chose d itself, against finishing when forced (the selection gap)
                if c["used"].sum() > 0:
                    w = c["used"]
                    wb = w[idx].sum(1)
                    ok = wb > 0
                    dc["finish_chosen"] = level(
                        c["u_solved"].sum() / w.sum(), (c["u_solved"][idx].sum(1)[ok] / wb[ok])
                    )
                    dc["C_weighted_by_use"] = level(
                        float((w * cr).sum() / w.sum()), ((w * cr)[idx].sum(1)[ok] / wb[ok])
                    )
                else:
                    dc["finish_chosen"] = dc["C_weighted_by_use"] = None
                if c["solved_first"] is not None and c["first"].sum() > 0:
                    w = c["first"]
                    dc["finish_first_chosen"] = float(c["solved_first"].sum() / w.sum())
                    dc["C_weighted_by_first"] = float((w * cr).sum() / w.sum())
            so[d] = dc
        out[s] = so
    return out


# ----------------------------------------------------------------------------- one line: base -> RL end
def log_odds(p):
    p = np.clip(np.asarray(p, dtype=float), P_CLAMP, 1 - P_CLAMP)
    return np.log10(p / (1 - p))


def class_groups(classes: dict, d: str, keys: list[tuple[str, str]]) -> tuple[str, dict, int]:
    """(mode, class -> [(set, name)], unlabelled optional theorems). "classes" when any optional theorem is labelled
    helpful or idle; otherwise the fallback: required against the optional theorems as a whole."""
    lab = {k: classes.get((k[0], k[1], d)) for k in keys}
    req = [k for k in keys if lab[k] == "required"]
    opt = [k for k in keys if lab[k] != "required"]
    if any(lab[k] in ("helpful", "idle") for k in opt):
        groups = {
            "required": req,
            "helpful": [k for k in opt if lab[k] == "helpful"],
            "idle": [k for k in opt if lab[k] == "idle"],
        }
        return "classes", groups, sum(1 for k in opt if lab[k] is None)
    return "fallback", {"required": req, "optional": opt}, 0


def route_verdict(s_req: str, s_not: str, d_lo: float, d_hi: float) -> str:
    """s_req, s_not: states of the log-odds change in the required and the not-needed class; d = Δ_not − ½ Δ_req."""
    if s_req != "up":
        return "no rise to locate"
    if d_hi < 0 or s_not in ("flat", "rare", "down"):
        return "router"
    if s_not == "up" and d_lo > 0:
        return "default shift"
    return "unresolved"


def router_cell(key: str, data: dict, classes: dict, b: str, e: str, d: str) -> dict:
    """Base-to-end change of the exact first-move propensity within each class, and the router / default-shift verdict."""
    keys = []
    for s in SETS:
        nb, _ = names_for(data[b].get(s, {}), d)
        ne, _ = names_for(data[e].get(s, {}), d)
        keys += [(s, n) for n in sorted(set(nb) & set(ne))]
    mode, groups, unlabelled = class_groups(classes, d, keys)
    out: dict = {"mode": mode, "unlabelled_optional": unlabelled, "classes": {}}
    boots = {}
    for g, ks in groups.items():
        co: dict = {"n": len(ks)}
        out["classes"][g] = co
        if len(ks) < N_MIN:
            continue
        pb = np.array([np.exp(data[b][s][n]["E"][d]["logp_open"]) for s, n in ks])
        pe = np.array([np.exp(data[e][s][n]["E"][d]["logp_open"]) for s, n in ks])
        ub = np.array([data[b][s][n]["U"][d]["used"] / data[b][s][n]["U"][d]["k"] for s, n in ks])
        ue = np.array([data[e][s][n]["U"][d]["used"] / data[e][s][n]["U"][d]["k"] for s, n in ks])
        idx = boot_idx(len(ks), f"{key}|{g}")
        bb, be = pb[idx].mean(1), pe[idx].mean(1)
        dl = float(log_odds(pe.mean()) - log_odds(pb.mean()))
        boots[g] = log_odds(be) - log_odds(bb)
        c = ci(boots[g])
        co.update({"P_base": level(pb.mean(), bb), "P_end": level(pe.mean(), be), "dlogodds10": dl, "ci": c,
                   "state": state(dl, *c, float(pb.mean()), float(pe.mean())),
                   "used_base": float(ub.mean()), "used_end": float(ue.mean())})  # fmt: skip
    nn = "idle" if mode == "classes" else "optional"
    if "required" not in boots or nn not in boots:
        short = [g for g in ("required", nn) if g not in boots]
        out["verdict"] = (
            f"no reading: {' and '.join(short)} n < {N_MIN}"
            if keys
            else "no reading: never applicable"
        )
        return out
    r, o = out["classes"]["required"], out["classes"][nn]
    dd = boots[nn] - ROUTE_SPLIT * boots["required"]
    d_lo, d_hi = ci(dd)
    out["not_needed_class"] = nn
    out["spillover_f"] = o["dlogodds10"] / r["dlogodds10"] if r["dlogodds10"] != 0 else None
    out["odds_ratio_between_classes_log10"] = r["dlogodds10"] - o["dlogodds10"]
    out["d_minus_half_ci"] = [d_lo, d_hi]
    out["verdict"] = route_verdict(r["state"], o["state"], d_lo, d_hi)
    if out["verdict"] == "router" and o["state"] == "rare":
        out["flag"] = "not-needed class stays below 1 %"
    return out


def line_cell(
    key: str, bths: dict, eths: dict, d: str, u_prop: bool = False, subset: list[str] | None = None
) -> dict:
    """The base-to-end change of one (line, set, direction): both factors, the states and the reading."""
    nb, dropb = names_for(bths, d)
    ne, drope = names_for(eths, d)
    app = sorted(
        {n for n, th in bths.items() if applicable_flag(th, d)}
        | {n for n, th in eths.items() if applicable_flag(th, d)}
    )
    names = sorted(set(nb) & set(ne))
    if subset is not None:
        names = sorted(set(names) & set(subset))
    out: dict = {
        "n_applicable": len(app),
        "n": len(names),
        "dropped": len(app) - len(names) if subset is None else None,
    }
    if not app:
        out["reading"] = "no reading: never applicable"
        return out
    if len(names) < N_MIN:
        out["reading"] = f"no reading: n = {len(names)} < {N_MIN}"
        return out
    if subset is None and (len(app) - len(names)) / len(app) > DROP_MAX:
        out["reading"] = (
            f"no reading: {len(app) - len(names)} of {len(app)} applicable theorems lack rows"
        )
        return out
    cb, ce = cell(bths, names, d), cell(eths, names, d)
    idx = boot_idx(len(names), key)
    # propensity: exact first move (primary) or sampled use anywhere (secondary)
    if u_prop:
        pb_lev, pe_lev = cb["used"].sum() / cb["uk"].sum(), ce["used"].sum() / ce["uk"].sum()
        lpb, lpe = (
            log_rate(cb["used"].sum(), cb["uk"].sum()),
            log_rate(ce["used"].sum(), ce["uk"].sum()),
        )
        bpb = log_rate(cb["used"][idx].sum(1), cb["uk"][idx].sum(1))
        bpe = log_rate(ce["used"][idx].sum(1), ce["uk"][idx].sum(1))
        bp_lev = (
            cb["used"][idx].sum(1) / cb["uk"][idx].sum(1),
            ce["used"][idx].sum(1) / ce["uk"][idx].sum(1),
        )
    else:
        pb_lev, pe_lev = cb["p"].mean(), ce["p"].mean()
        lpb, lpe = log_p(pb_lev), log_p(pe_lev)
        bp_lev = (cb["p"][idx].mean(1), ce["p"][idx].mean(1))
        bpb, bpe = log_p(bp_lev[0]), log_p(bp_lev[1])
    cb_lev, ce_lev = cb["f_solved"].sum() / cb["fk"].sum(), ce["f_solved"].sum() / ce["fk"].sum()
    lcb, lce = (
        log_rate(cb["f_solved"].sum(), cb["fk"].sum()),
        log_rate(ce["f_solved"].sum(), ce["fk"].sum()),
    )
    bcb = log_rate(cb["f_solved"][idx].sum(1), cb["fk"][idx].sum(1))
    bce = log_rate(ce["f_solved"][idx].sum(1), ce["fk"][idx].sum(1))
    dP, dC = float(lpe - lpb), float(lce - lcb)
    ciP, ciC = ci(bpe - bpb), ci(bce - bcb)
    sp = state(dP, *ciP, float(pb_lev), float(pe_lev))
    sc = state(dC, *ciC, float(cb_lev), float(ce_lev))
    out.update({
        "P": {"base": level(pb_lev, bp_lev[0]), "end": level(pe_lev, bp_lev[1]), "dlog10": dP, "ci": ciP, "state": sp},
        "C": {"base": level(cb_lev, cb["f_solved"][idx].sum(1) / cb["fk"][idx].sum(1)),
              "end": level(ce_lev, ce["f_solved"][idx].sum(1) / ce["fk"][idx].sum(1)),
              "dlog10": dC, "ci": ciC, "state": sc, "dpoints": float(ce_lev - cb_lev)},
    })  # fmt: skip
    rd = reading(sp, sc)
    flags = []
    if not u_prop and subset is None:
        eb, ee = eu_check(cb), eu_check(ce)
        out["E_vs_U"] = {"base": eb, "end": ee}
        if eb["fail"] or ee["fail"]:
            rd = "no reading: E disagrees with U"
    if subset is None:
        bad = []
        for c in (cb, ce):
            if c["f_endings_known"]:
                bad.append(float((c["f_capped"].sum() + c["f_refused"].sum()) / c["fk"].sum()))
        out["F_refused_or_capped"] = bad
        if bad and max(bad) > BAD_MAX:
            rd = "no reading: forced attempts mostly refused or capped"
        elif len(bad) < 2:
            flags.append("forced endings not recorded")
    out["reading"] = rd
    # sizing: the change in Pbar x Cbar, split between the factors (Shapley, in points of attempts)
    dS_P = (pe_lev - pb_lev) * (cb_lev + ce_lev) / 2
    dS_C = (ce_lev - cb_lev) * (pb_lev + pe_lev) / 2
    sb = float((cb["p"] * cb["f_solved"] / cb["fk"]).mean())
    se = float((ce["p"] * ce["f_solved"] / ce["fk"]).mean())
    dPC = float(pe_lev * ce_lev - pb_lev * cb_lev)
    size = {"dPC": dPC, "dPC_from_P": float(dS_P), "dPC_from_C": float(dS_C), "S_first_base": sb, "S_first_end": se,
            "dS_first": se - sb, "share_P": None}  # fmt: skip
    if not u_prop:
        if max(abs(se - sb), abs(dPC)) >= SHAPLEY_MIN:
            r = (se - sb) / dPC if dPC != 0 else math.inf
            if not (0.5 <= r <= 2.0):
                flags.append("set-level factors do not account for the change in success via d")
        if (
            rd == "mix"
            and dS_P > 0
            and dS_C > 0
            and "set-level factors do not account for the change in success via d" not in flags
        ):
            size["share_P"] = float(dS_P / (dS_P + dS_C))
    out["size"] = size
    if sp == "up" and sc in ("rare", "down"):
        flags.append(f"propensity up, capability {sc}")
    out["flags"] = flags
    return out


def pair_lines(ckpts: dict) -> tuple[dict, list]:
    """(dose, seed) -> {"base", "end", "others"}; the primary end is the rl_end with the largest x_dan."""
    lines: dict = {}
    for c, m in sorted(ckpts.items()):
        if m["dose"] == "full":
            continue
        ln = lines.setdefault((m["dose"], m["seed"]), {"base": None, "ends": [], "matched": []})
        if m["stage"] == "base":
            if ln["base"] is not None:
                raise SchemaError(
                    f"two bases for dose {m['dose']} seed {m['seed']}: {ln['base']}, {c}"
                )
            ln["base"] = c
        elif m["stage"] == "rl_end":
            ln["ends"].append(c)
        else:
            ln["matched"].append(c)
    notes = []
    for (dose, seed), ln in lines.items():
        ends = sorted(
            ln["ends"], key=lambda c: (ckpts[c]["x_dan"] is None, -(ckpts[c]["x_dan"] or 0), c)
        )
        ln["end"] = ends[0] if ends else None
        ln["others"] = ends[1:]
        if ln["others"]:
            notes.append(
                f"dose {dose} seed {seed}: primary end {ln['end']} (largest x); secondary {', '.join(ln['others'])}"
            )
    return lines, notes


def line_result(key: str, ckpts: dict, data: dict, b: str | None, e: str | None, dose: str, seed: int,
                classes: dict | None = None) -> dict:  # fmt: skip
    out: dict = {"dose": dose, "seed": seed, "base": b, "end": e,
                 "x_end": ckpts[e]["x_dan"] if e else None,
                 "alarm": bool(e) and ((dose, seed) in ALARM_ENDS or "alarm" in e), "cells": {}, "router": {}}  # fmt: skip
    if b is None or e is None:
        out["missing"] = "base" if b is None else "rl_end"
        return out
    for d in FORCED:
        out["router"][d] = router_cell(f"{key}|router|{d}", data, classes or {}, b, e, d)
    for s in SETS:
        bths, eths = data[b].get(s, {}), data[e].get(s, {})
        for d in FORCED:
            k = f"{key}|{s}|{d}"
            prim = line_cell(k, bths, eths, d)
            sec = line_cell(k, bths, eths, d, u_prop=True)
            # familiar subset: theorems where the base itself opens with d at least 1 % of the time (trap 2)
            fam = None
            nb, _ = names_for(bths, d)
            if nb:
                pb = cell(bths, nb, d)["p"]
                sub = [n for n, p in zip(nb, pb) if p >= FLOOR]
                fam = line_cell(k + "|familiar", bths, eths, d, subset=sub)
            out["cells"][f"{s}|{d}"] = {"primary": prim, "uses_anywhere": sec, "familiar": fam}
    return out


def dose_reading(rs: list[str | None], labels: tuple[str, ...] = READINGS) -> str:
    """Seeds -> one dose-level reading: all read seeds agree (at least 2), or 2 of 3 agree and the third is unresolved
    or has no reading. Otherwise "seeds disagree"."""
    got = [r for r in rs if r is not None]
    if len(got) < 2:
        return "too few seeds"
    if (
        len(set(got)) == 1 and got[0] not in labels
    ):  # every seed gives the same non-reading (unresolved, ...)
        return got[0]
    real = [r for r in got if r in labels]
    for r in labels:
        n = real.count(r)
        if n == len(got) and n >= 2:
            return r
        if n == 2 and len(got) == 3 and all(x == r or x not in labels for x in got):
            return f"{r} (2 of 3)"
    return "seeds disagree"


def informative(lines: dict, levels: dict) -> dict:
    """PREREG: when the propensity-capability split is judged not informative here."""
    cells = [c["primary"] for ln in lines.values() for c in ln["cells"].values()
             if c["primary"]["n"] >= N_MIN]  # fmt: skip
    bad = [c for c in cells if c["reading"] not in READINGS]
    eu = [lv[s][d]["E_vs_U"]["fail"] for lv in levels.values() for s in SETS for d in FORCED
          if lv.get(s, {}).get(d, {}).get("n", 0) >= N_MIN]  # fmt: skip
    return {
        "primary_cells": len(cells),
        "not_read": len(bad),
        "eu_cells": len(eu),
        "eu_fail": int(sum(eu)),
    }


def analyse(results_dir: Path) -> dict:
    ckpts, data, files, classes = load(results_dir)
    levels = {c: ckpt_levels(c, data[c]) for c in sorted(ckpts)}
    pairs, notes = pair_lines(ckpts)
    lines, matched = {}, {}
    for (dose, seed), ln in sorted(
        pairs.items(), key=lambda kv: (DOSES.index(kv[0][0]) if kv[0][0] in DOSES else 9, kv[0][1])
    ):
        key = f"{dose}|{seed}"
        lines[key] = line_result(key, ckpts, data, ln["base"], ln["end"], dose, seed, classes)
        for e in ln["others"] + ln["matched"]:
            r = line_result(f"{key}|{e}", ckpts, data, ln["base"], e, dose, seed, classes)
            r["kind"] = "matched compute" if ckpts[e]["stage"] == "matched" else "secondary end"
            matched[e] = r
    doses: dict = {}
    routes: dict = {}
    for dose in DOSES:
        seeds = sorted(ln["seed"] for ln in lines.values() if ln["dose"] == dose)
        if not seeds:
            continue
        for d in FORCED:
            vs = [lines[f"{dose}|{sd}"]["router"].get(d) for sd in seeds]
            rs = [v["verdict"] if v else None for v in vs]
            fs = [
                v["spillover_f"]
                for v in vs
                if v and v.get("spillover_f") is not None and v["verdict"] != "no rise to locate"
            ]
            routes[f"{dose}|{d}"] = {"seeds": seeds, "verdicts": rs, "modes": [v["mode"] if v else None for v in vs],
                                     "verdict": dose_reading(rs, ROUTES), "spillover_f": fs}  # fmt: skip
        for s in SETS:
            for d in FORCED:
                rs, dP, dC = [], [], []
                for sd in seeds:
                    ln = lines[f"{dose}|{sd}"]
                    c = ln["cells"].get(f"{s}|{d}")
                    if c is None:
                        rs.append(None)
                        continue
                    p = c["primary"]
                    rs.append(
                        p["reading"]
                        if p["n"] >= N_MIN or p["reading"].startswith("no reading")
                        else None
                    )
                    if "P" in p:
                        dP.append(p["P"]["dlog10"])
                        dC.append(p["C"]["dlog10"])
                if all(r is not None and r.startswith("no reading: never") for r in rs):
                    dr = "never applicable"
                elif all(r is not None and r.startswith("no reading: n =") for r in rs):
                    dr = f"no reading (n < {N_MIN})"
                else:
                    dr = dose_reading(
                        [
                            r
                            if r is None
                            or r in READINGS
                            or r in NOT_READ
                            or r.startswith("no reading")
                            else r
                            for r in rs
                        ]
                    )
                doses[f"{dose}|{s}|{d}"] = {
                    "seeds": seeds, "readings": rs, "reading": dr,
                    "dlog10_P": {"mean": float(np.mean(dP)), "min": float(min(dP)), "max": float(max(dP))} if dP else None,
                    "dlog10_C": {"mean": float(np.mean(dC)), "min": float(min(dC)), "max": float(max(dC))} if dC else None,
                }  # fmt: skip
    inf = informative(lines, levels)
    crit = {
        "a_more_than_half_not_read": inf["primary_cells"] > 0
        and inf["not_read"] > inf["primary_cells"] / 2,
        "b_no_dose_reading": not any(
            v["reading"].split(" (")[0] in READINGS for v in doses.values()
        ),
        "c_E_fails_in_over_a_quarter": inf["eu_cells"] > 0 and inf["eu_fail"] > inf["eu_cells"] / 4,
    }
    prereg = HERE / "PREREG.md"
    return {
        "thresholds": {"DELTA_log10": DELTA, "FLAT_log10": FLAT, "FLOOR": FLOOR, "N_MIN": N_MIN, "BAD_MAX": BAD_MAX,
                       "DROP_MAX": DROP_MAX, "E_vs_U": [EU_Z, EU_LOG, EU_MIN], "B": B, "rng_seed": RNG_SEED},
        "prereg_sha256": hashlib.sha256(prereg.read_bytes()).hexdigest() if prereg.exists() else None,
        "inputs": files, "ckpts": ckpts, "pairing_notes": notes, "levels": levels, "lines": lines,
        "matched_and_secondary": matched, "doses": doses, "router_doses": routes,
        "classes_present": sorted(set(classes.values())), "qc_warnings": list(WARNINGS),
        "informative": {**inf, **crit, "verdict": "not informative" if any(crit.values()) else "informative"},
        "by_schema": by_schema(ckpts, data, pairs),
    }  # fmt: skip


def by_schema(ckpts: dict, data: dict, pairs: dict) -> dict:
    """Exploratory: reductio on the strict 106 by schema, levels only."""
    out: dict = {}
    for (dose, seed), ln in pairs.items():
        for stage, c in (("base", ln["base"]), ("end", ln["end"])):
            if c is None:
                continue
            ths = data[c].get("strict106", {})
            names, _ = names_for(ths, "reductio")
            groups = defaultdict(list)
            for n in names:
                groups[ths[n]["meta"].get("schema") or "none"].append(n)
            for sch, ns in groups.items():
                cc = cell(ths, ns, "reductio")
                out.setdefault(f"{dose}|{seed}", {}).setdefault(sch, {})[stage] = {
                    "n": len(ns),
                    "P_first": float(cc["p"].mean()),
                    "C": float(cc["f_solved"].sum() / cc["fk"].sum()),
                }
    return out


# ----------------------------------------------------------------------------- tables
def pct(x: float | None) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    v = 100 * x
    if v == 0:
        return "0"
    if abs(v) >= 0.1:
        return f"{v:.1f}"
    if abs(v) >= 0.001:
        return f"{v:.2g}"
    return f"{v:.1e}"


def lev(d: dict | None) -> str:
    if not d:
        return "n/a"
    return f"{pct(d['est'])} [{pct(d['ci'][0])}, {pct(d['ci'][1])}]"


def ratio(dl: float, c: list[float]) -> str:
    def f(x: float) -> str:
        r = 10**x
        return f"{r:.3g}" if 0.01 <= r < 1000 else f"10^{x:+.1f}"

    return f"x{f(dl)} [{f(c[0])}, {f(c[1])}]"


DLAB = {"0": "0 %", "0.1": "0.1 %", "1": "1 %", "full": "full"}


def fm(z: dict | None) -> str:
    return f"{z['mean']:+.2f} ({z['min']:+.2f} to {z['max']:+.2f})" if z else "n/a"


def tables(res: dict) -> str:
    th = res["thresholds"]
    L = ["# propensity_capability: frozen tables", "",
         "Written by `analyze.py` under PREREG.md (sha256 " + f"`{res['prereg_sha256']}`). Every rate is % of attempts over the",
         "applicable theorems of one set, with a 95 % bootstrap interval over theorems "
         f"({th['B']} resamples, seeds shown separately, never pooled).",
         f"Reading thresholds: a change counts at |Δlog10| ≥ {th['DELTA_log10']} with the interval excluding 0; flat iff the interval lies "
         f"inside ±{th['FLAT_log10']}; rare iff below {pct(th['FLOOR'])} % at both checkpoints; n ≥ {th['N_MIN']} applicable theorems.", ""]  # fmt: skip
    inf = res["informative"]
    L += ["## Is the split informative here (PREREG, decided first)", "",
          f"**Verdict: {inf['verdict']}.** Primary cells read: {inf['primary_cells'] - inf['not_read']} of {inf['primary_cells']} "
          f"(unresolved or no reading: {inf['not_read']}). E fails against U in {inf['eu_fail']} of {inf['eu_cells']} checkpoint cells.",
          f"(a) more than half not read: {inf['a_more_than_half_not_read']}; (b) no dose-level reading anywhere: "
          f"{inf['b_no_dose_reading']}; (c) E fails in more than a quarter: {inf['c_E_fails_in_over_a_quarter']}.", ""]  # fmt: skip
    L += ["## Dose-level readings (primary: first-move propensity from E, capability from F)", "",
          "Each dose's RL end is at its own compute (x at the end in the last column); doses are not compared here.", "",
          "| dose | set | direction | per seed | dose reading | Δlog10 P, mean (range) | Δlog10 C, mean (range) | x at RL ends |",
          "|---|---|---|---|---|---|---|---|"]  # fmt: skip
    for k, v in res["doses"].items():
        dose, s, d = k.split("|")
        xs = [res["lines"][f"{dose}|{sd}"]["x_end"] for sd in v["seeds"]]
        xs_s = ", ".join(f"{x:.2f}" if x is not None else "?" for x in xs)
        rs = " / ".join(
            (r or "missing").replace("no reading: ", "no reading, ") for r in v["readings"]
        )
        L.append(
            f"| {DLAB[dose]} | {s} | {d} | {rs} | **{v['reading']}** | {fm(v['dlog10_P'])} | {fm(v['dlog10_C'])} | {xs_s} |"
        )
    L += ["", "## Per seed: base → RL end", "",
          "P = exact first-move probability (E). C = solved when forced (F). States: up, down, flat, rare, unresolved.", "",
          "| dose | seed | set | direction | n | P base → end | P ratio | P | C base → end | C ratio | C | reading | share of Δ(P̄C̄) from P | flags |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]  # fmt: skip
    for ln in res["lines"].values():
        if ln.get("missing"):
            L.append(
                f"| {DLAB[ln['dose']]} | {ln['seed']} | | | | | | | | | | missing {ln['missing']} | | |"
            )
            continue
        for k, c in ln["cells"].items():
            s, d = k.split("|")
            p = c["primary"]
            fl = list(p.get("flags", []))
            if ln["alarm"]:
                fl.append("alarm end")
            if "P" not in p:
                L.append(
                    f"| {DLAB[ln['dose']]} | {ln['seed']} | {s} | {d} | {p['n']} | | | | | | | {p['reading']} | | {'; '.join(fl)} |"
                )
                continue
            sh = p["size"]["share_P"]
            L.append(f"| {DLAB[ln['dose']]} | {ln['seed']} | {s} | {d} | {p['n']} | {pct(p['P']['base']['est'])} → {pct(p['P']['end']['est'])} | "
                     f"{ratio(p['P']['dlog10'], p['P']['ci'])} | {p['P']['state']} | {pct(p['C']['base']['est'])} → {pct(p['C']['end']['est'])} | "
                     f"{ratio(p['C']['dlog10'], p['C']['ci'])} | {p['C']['state']} | **{p['reading']}** | "
                     f"{'n/a' if sh is None else f'{100 * sh:.0f} %'} | {'; '.join(fl)} |")  # fmt: skip
    L += ["", "## Levels per checkpoint", "",
          "P (E) is the exact first-move probability; P (U, first) the sampled first-move rate; P (U, anywhere) the share of attempts "
          "that use d anywhere. Finishing once chosen is U's solved / used, next to C weighted by each theorem's use: their gap is "
          "the selection effect within theorems.", "",
          "| checkpoint | dose | stage | x | set | direction | n | P (E) | P (U, first) | E vs U z | P (U, anywhere) | C (F) | F refused / capped | finishing once chosen | C weighted by use |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]  # fmt: skip
    for cid, lv in res["levels"].items():
        m = res["ckpts"][cid]
        x = "" if m["x_dan"] is None else f"{m['x_dan']:.2f}"
        for s in SETS:
            for d in FORCED:
                c = lv.get(s, {}).get(d)
                if not c or not c.get("n"):
                    na = c["n_applicable"] if c else 0
                    L.append(
                        f"| {cid} | {DLAB[m['dose']]} | {m['stage']} | {x} | {s} | {d} | {na} applicable | | | | | | | | |"
                    )
                    continue
                eu = c["E_vs_U"]
                z = "inf" if eu["z"] is None else f"{eu['z']:+.1f}"
                fe = c["F_endings"]
                fes = "n/a" if fe is None else f"{pct(fe['refused'])} / {pct(fe['capped'])}"
                L.append(f"| {cid} | {DLAB[m['dose']]} | {m['stage']} | {x} | {s} | {d} | {c['n']} | {lev(c['P_first'])} | {lev(c['P_first_sampled'])} | "
                         f"{z}{' FAIL' if eu['fail'] else ''} | {lev(c['P_used'])} | {lev(c['C'])} | {fes} | {lev(c['finish_chosen'])} | {lev(c['C_weighted_by_use'])} |")  # fmt: skip
    L += ["", "## Unforced success and how it is reached", "",
          "Share using d: accepted attempts that use d anywhere, over all accepted attempts (an attempt can use several). "
          "First move: % of all attempts whose first proof action is d's opening (direct: opens no box).", "",
          "| checkpoint | set | n | solved | Lean-rejected / refused / capped | share using " + " / ".join(FORCED) + " | first move " + " / ".join(DIRS) + " |",
          "|---|---|---|---|---|---|---|"]  # fmt: skip
    for cid, lv in res["levels"].items():
        for s in SETS:
            u = lv.get(s, {}).get("unforced")
            if not u:
                continue
            sh = " / ".join(pct(u["share_using"].get(d)) for d in FORCED)
            fs = " / ".join(pct(u["first_share"].get(d)) for d in DIRS)
            L.append(
                f"| {cid} | {s} | {u['n']} | {lev(u['success'])} | {pct(u['lean_rejected'])} / {pct(u['refused'])} / {pct(u['capped'])} | {sh} | {fs} |"
            )
    L += ["", "## Secondary: the same rule with propensity = uses d anywhere (U)", "",
          "| dose | seed | set | direction | P base → end | P ratio | P | C | reading | primary reading |", "|---|---|---|---|---|---|---|---|---|---|"]  # fmt: skip
    for ln in res["lines"].values():
        for k, c in ln.get("cells", {}).items():
            s, d = k.split("|")
            p = c["uses_anywhere"]
            if "P" not in p:
                continue
            L.append(f"| {DLAB[ln['dose']]} | {ln['seed']} | {s} | {d} | {pct(p['P']['base']['est'])} → {pct(p['P']['end']['est'])} | "
                     f"{ratio(p['P']['dlog10'], p['P']['ci'])} | {p['P']['state']} | {p['C']['state']} | {p['reading']} | {c['primary']['reading']} |")  # fmt: skip
    L += ["", "## Secondary: capability on the familiar subset (theorems the base itself opens with d at ≥ 1 %)", "",
          "| dose | seed | set | direction | n | C base → end | C ratio | C state | C state, all theorems |", "|---|---|---|---|---|---|---|---|---|"]  # fmt: skip
    for ln in res["lines"].values():
        for k, c in ln.get("cells", {}).items():
            s, d = k.split("|")
            f = c["familiar"]
            if not f:
                continue
            if "C" not in f:
                L.append(
                    f"| {DLAB[ln['dose']]} | {ln['seed']} | {s} | {d} | {f['n']} | | | {f['reading']} | {c['primary'].get('C', {}).get('state', '')} |"
                )
                continue
            L.append(f"| {DLAB[ln['dose']]} | {ln['seed']} | {s} | {d} | {f['n']} | {pct(f['C']['base']['est'])} → {pct(f['C']['end']['est'])} | "
                     f"{ratio(f['C']['dlog10'], f['C']['ci'])} | {f['C']['state']} | {c['primary'].get('C', {}).get('state', '')} |")  # fmt: skip
    L += ["", "## Secondary: router or default shift (exact first-move propensity by class)", "",
          f"Classes present in the result files: {', '.join(res['classes_present']) or 'none'}. Change in log10 odds of P̄ per class; "
          f"f = the not-needed class's change over the required class's. Router if the interval of Δ(not needed) − {ROUTE_SPLIT} Δ(required) "
          "lies below 0 or the not-needed class is flat, rare or down; default shift if the not-needed class is up and that interval lies "
          "above 0. Dan's RL targets are 2.7 % covered by theorems that require the reductio: little direct pressure where it is optional.", "",
          "| dose | direction | per seed | dose verdict | f per seed |", "|---|---|---|---|---|"]  # fmt: skip
    for k, v in res["router_doses"].items():
        dose, d = k.split("|")
        rs = " / ".join(r or "missing" for r in v["verdicts"])
        fs = ", ".join(f"{f:+.2f}" for f in v["spillover_f"]) or "n/a"
        L.append(f"| {DLAB[dose]} | {d} | {rs} | **{v['verdict']}** | {fs} |")
    L += ["", "| dose | seed | direction | mode | class | n | P base → end | Δlog10 odds [95 % CI] | state | uses anywhere, base → end | verdict |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]  # fmt: skip
    for ln in res["lines"].values():
        for d, r in ln.get("router", {}).items():
            first = True
            for g, co in r["classes"].items():
                head = (
                    f"| {DLAB[ln['dose']]} | {ln['seed']} | {d} | {r['mode']} |"
                    if first
                    else "| | | | |"
                )
                tail = (
                    f" {r['verdict']}{' (' + r['flag'] + ')' if r.get('flag') else ''} |"
                    if first
                    else " |"
                )
                if "P_base" not in co:
                    L.append(f"{head} {g} | {co['n']} | | | | |{tail}")
                else:
                    L.append(f"{head} {g} | {co['n']} | {pct(co['P_base']['est'])} → {pct(co['P_end']['est'])} | "
                             f"{co['dlogodds10']:+.2f} [{co['ci'][0]:+.2f}, {co['ci'][1]:+.2f}] | {co['state']} | "
                             f"{pct(co['used_base'])} → {pct(co['used_end'])} |{tail}")  # fmt: skip
                first = False
    if res["matched_and_secondary"]:
        L += ["", "## Matched-compute and secondary ends (never in a dose-level reading)", "",
              "| checkpoint | kind | dose | seed | x | set | direction | P ratio | P | C ratio | C | reading |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]  # fmt: skip
        for e, ln in res["matched_and_secondary"].items():
            for k, c in ln.get("cells", {}).items():
                s, d = k.split("|")
                p = c["primary"]
                if "P" not in p:
                    continue
                L.append(f"| {e} | {ln['kind']} | {DLAB[ln['dose']]} | {ln['seed']} | {ln['x_end']} | {s} | {d} | {ratio(p['P']['dlog10'], p['P']['ci'])} | "
                         f"{p['P']['state']} | {ratio(p['C']['dlog10'], p['C']['ci'])} | {p['C']['state']} | {p['reading']} |")  # fmt: skip
    L += [
        "",
        "## Exploratory: reductio on the strict 106 by schema (levels, no readings)",
        "",
        "| line | schema | n | P base → end | C base → end |",
        "|---|---|---|---|---|",
    ]
    for ln, schs in res["by_schema"].items():
        for sch, v in sorted(schs.items()):
            b, e = v.get("base"), v.get("end")
            if b and e:
                L.append(
                    f"| {ln} | {sch} | {b['n']} | {pct(b['P_first'])} → {pct(e['P_first'])} | {pct(b['C'])} → {pct(e['C'])} |"
                )
    L += ["", "## Inputs", ""]
    L += [f"- `{f}` sha256 `{h}`" for f, h in res["inputs"].items()]
    L += [f"- {n}" for n in res["pairing_notes"]]
    if res["qc_warnings"]:
        L += [f"- QC warnings: {len(res['qc_warnings'])}, first: {res['qc_warnings'][0]}"]
    return "\n".join(L) + "\n"


# ----------------------------------------------------------------------------- calibration (PREREG's noise numbers)
CAL_LINES = {"0": [("ko_s0", "dko_s0"), ("ko_s1", "dko_s1"), ("ko_s2", "dko_s2")],
             "0.1": [("dose01_s0", "dd01_s0"), ("dose01_s1", "dd01_s1"), ("dose01_s2", "dd01_s2")],
             "1": [("dose1_s0", "dd1_s0"), ("dose1_s1", "dd1_s1"), ("dose1_s2", "dd1_s2")]}  # fmt: skip


def calibrate(summaries: Path) -> dict:
    """Seed-to-seed noise and interval widths in the factor read (counts only, no proof text). Capability: Lean-solved
    from the forced pick on T (k 64, and thinned to 16). Propensity: the exact name-free probability of the pick action."""

    def per(tag: str):
        st = json.loads((summaries / f"{tag}.json").read_text())["stages"]
        names = sorted(st["reach"])
        cs = np.array([st["reach"][n]["solved"] for n in names], dtype=float)
        ck = np.array([st["reach"][n]["n"] for n in names], dtype=float)
        lp = []
        for n in names:
            c = st["C"][n]
            lp.append(
                sum(
                    a - b
                    for a, b, g in zip(c["logp_mean_nats"], c["logp_name_mean_nats"], c["groups"])
                    if g == "pick"
                )
            )
        return names, cs, ck, np.exp(np.array(lp))

    rng = np.random.default_rng(0)
    rows = {}
    for dose, ls in CAL_LINES.items():
        for b, e in ls:
            nb, csb, ckb, pb = per(b)
            ne, cse, cke, pe = per(e)
            assert nb == ne
            idx = rng.integers(0, len(nb), size=(B, len(nb)))
            b16, e16 = (
                rng.hypergeometric(csb.astype(int), (ckb - csb).astype(int), 16),
                rng.hypergeometric(cse.astype(int), (cke - cse).astype(int), 16),
            )
            dC16 = log_rate(e16[idx].sum(1), 16 * len(nb)) - log_rate(b16[idx].sum(1), 16 * len(nb))
            dP = log_p(pe[idx].mean(1)) - log_p(pb[idx].mean(1))
            rows[b] = {"dose": dose, "C_base": float(csb.sum() / ckb.sum()), "C_end": float(cse.sum() / cke.sum()),
                       "dlog10_C": float(log_rate(cse.sum(), cke.sum()) - log_rate(csb.sum(), ckb.sum())),
                       "dpoints_C": float(cse.sum() / cke.sum() - csb.sum() / ckb.sum()),
                       "P_base": float(pb.mean()), "P_end": float(pe.mean()),
                       "dlog10_P": float(log_p(pe.mean()) - log_p(pb.mean())),
                       "halfwidth_dC_k16": float(np.diff(ci(dC16))[0] / 2), "halfwidth_dP": float(np.diff(ci(dP))[0] / 2)}  # fmt: skip
    sd = {}
    for q in ("dlog10_C", "dpoints_C", "dlog10_P"):
        sd[q] = {
            dose: float(np.std([rows[b][q] for b, _ in ls], ddof=1))
            for dose, ls in CAL_LINES.items()
        }
    for q, f in (("log10_C_base", lambda r: math.log10(r["C_base"])), ("log10_C_end", lambda r: math.log10(r["C_end"])),
                 ("log10_P_base", lambda r: math.log10(r["P_base"])), ("log10_P_end", lambda r: math.log10(r["P_end"]))):  # fmt: skip
        sd[q] = {
            dose: float(np.std([f(rows[b]) for b, _ in ls], ddof=1))
            for dose, ls in CAL_LINES.items()
        }
    return {"lines": rows, "seed_sd_within_dose": sd}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="/scratch/cowifornia/nd-rl/propensity_capability/results")
    ap.add_argument("--out", default=str(HERE / "results"))
    ap.add_argument(
        "--calibrate", help="factor read summaries dir: print the noise numbers PREREG.md quotes"
    )
    a = ap.parse_args()
    if a.calibrate:
        print(json.dumps(calibrate(Path(a.calibrate)), indent=1))
        return
    res = analyse(Path(a.results))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(res, indent=1, default=float) + "\n")
    (out / "tables.md").write_text(tables(res))
    print(
        f"{len(res['ckpts'])} checkpoints, verdict: {res['informative']['verdict']}; wrote {out}/results.json and tables.md"
    )


if __name__ == "__main__":
    main()
