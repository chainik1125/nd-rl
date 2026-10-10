"""hard712 pipeline: generate -> lean -> select -> split -> meta / verify (CPU only).

    cd /scratch/cowifornia/nd-rl/think/nd-rl
    export ND_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
    X=code.experiments.current.thinking_traces_20261005.hard712
    nice -n 10 code/.venv/bin/python -m $X.excl --out $W/excl_index.pkl
    nice -n 10 code/.venv/bin/python -m $X.run generate --workers 24 --target 60000 --hours 5
    nice -n 10 code/.venv/bin/python -m $X.run lean
    code/.venv/bin/python -m $X.run select --n 27500
    code/.venv/bin/python -m $X.run split
    nice -n 10 code/.venv/bin/python -m $X.run verify --workers 16
    code/.venv/bin/python -m $X.run meta

Outputs go to /scratch/cowifornia/nd-rl/think/runs/pools/hard712/ (`--out`), intermediates to
its work/ folder:

* `generate`: rounds of `--round` seeds. Workers build candidates (`pfgen.sample`) and apply the
  shape gate; the parent drops renaming-class duplicates (`pkey`, also across rejected classes, so
  no class is certified twice) and classes in the exclusion index (`excl.py`); workers then run
  `pfgates.certify` (T1 + necessity). Accepted rows go to work/gated.jsonl, every other certified
  or excluded class to work/rejects.jsonl, counters to work/progress.json. Resumable.
* `lean`: Lean 4 on every gated row's constructed proof (the gate) and its reference proof
  (recorded as `lean_ref_ok`) -> work/gated_lean.jsonl.
* `select`: the mix gate. A deterministic greedy pass (rows ordered by a hash of their class key,
  scarce attributes first) keeps rows while every capped share stays within `CAPS` of `--n`.
  No model is involved. -> work/selected.jsonl.
* `split`: a uniform random 1,000 (test) and 500 (dev) of the selected rows, the rest is the
  pool; final row format, keys sidecars.
* `verify`: the gap-analysis feature table recomputed on a random 2,000 pool rows, Lean on the
  reference proofs of a random 1,000, diversity counts -> work/verify.json.
* `meta`: meta.json (yields, timing, family mix, histograms, comparison with rl_hard12 and the
  dev58 7-12 aggregates).
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import multiprocessing as mp
import os
import pickle
import random
import resource
import sys
import time
from pathlib import Path
from typing import Any

sys.dont_write_bytecode = True

from code.ndbench.core.keys import canon_key, pkey, prompt_to_thm, split_thm  # noqa: E402

from . import ext  # noqa: E402
from . import pfgates as GT  # noqa: E402
from . import pfgen as PG  # noqa: E402

OUT = Path("/scratch/cowifornia/nd-rl/think/runs/pools/hard712")
GAP_OUT = Path("/scratch/cowifornia/nd-rl/think/runs/gap_analysis")
POOL12 = Path("/scratch/cowifornia/nd-rl/think/runs/pools/rl_hard12.jsonl")
MEM_GB = float(os.environ.get("PFG_MEM_GB", "4.0"))
SPLIT_SEED = 20261008


MAX_PREM_FOR_PKEY = 5  # as code.thinking.data (which imports torch, so is not used here)


def class_keys(prompt: str) -> tuple[str, str | None]:
    """`code.thinking.data.class_keys` without its torch import: (atom-renaming class key,
    premise-order-invariant key or None above 5 premises)."""
    thm = prompt_to_thm(prompt)
    prem, _ = split_thm(thm)
    return canon_key(thm), (pkey(thm) if len(prem) <= MAX_PREM_FOR_PKEY else None)


def read_complete(path: Path) -> tuple[list[dict[str, Any]], int]:
    """Rows of a jsonl file a crash may have cut mid-line: complete rows are kept, and the file is
    rewritten without the broken tail so appends continue cleanly."""
    rows, bad = [], 0
    with open(path) as f:
        lines = f.readlines()
    for line in lines:
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            bad += 1
    if bad:
        wjl(path, rows)
    return rows, bad


def jl(path: Path | str) -> list[dict[str, Any]]:
    with open(path) as f:
        return [json.loads(x) for x in f if x.strip()]


def wjl(path: Path | str, rows: list[dict[str, Any]]) -> None:
    tmp = str(path) + ".tmp"
    with open(tmp, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    os.replace(tmp, path)


def rjson(path: Path | str) -> Any:
    with open(path) as f:
        return json.load(f)


def wjson(path: Path | str, obj: Any) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=1)


def log(msg: str, fh: Any = None) -> None:
    line = f"{time.strftime('%H:%M:%S', time.gmtime())}Z {msg}"
    print(line, flush=True)
    if fh is not None:
        fh.write(line + "\n")
        fh.flush()


# ---------------------------------------------------------------------- generate
def _init(mem_gb: float) -> None:
    b = int(mem_gb * 2**30)
    resource.setrlimit(resource.RLIMIT_AS, (b, b))


def construct_chunk(
    args: tuple[int, int, dict[str, float] | None],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Seeds [s0, s1) (recipe weights w, or pfgen.WEIGHTS): candidates that pass the shape gate,
    and per-reason counts."""
    s0, s1, w = args
    st: collections.Counter[str] = collections.Counter()
    out = []
    for seed in range(s0, s1):
        c = PG.sample(seed, weights=w)
        if c is None:
            st["construct_fail"] += 1
            continue
        reason, info = GT.shape_gate(c["prompt"], c["proof"])
        if reason:
            st["shape:" + reason] += 1
            continue
        c.update(info)
        st["shape_ok"] += 1
        out.append(c)
    return out, dict(st)


def certify_safe(c: dict[str, Any]) -> dict[str, Any]:
    try:
        return GT.certify(c)
    except MemoryError:
        return dict(c, status="memory")
    except Exception as e:  # noqa: BLE001  (a bug in one candidate must not kill the run)
        return dict(c, status=f"error:{type(e).__name__}:{str(e)[:80]}")


#: Adaptive-weight values (overridable: the curriculum's 13-15-line band, where most rows are
#: classical-only, sets PFG_CLASSICAL_VALUE=0, PFG_EXFALSO_VALUE=0 and PFG_SCARCE_BONUS=0 to
#: weight recipes by their yield of rows that are neither classical-only nor ex falso, the two
#: attributes the mix gate caps hardest).
CLASSICAL_VALUE = float(os.environ.get("PFG_CLASSICAL_VALUE", "0.1"))
SCARCE_BONUS = float(os.environ.get("PFG_SCARCE_BONUS", "0.5"))
_EF = os.environ.get("PFG_EXFALSO_VALUE")  # unset: ex falso rows are valued like any other
EXFALSO_VALUE = float(_EF) if _EF is not None else None


def row_value(r: dict[str, Any]) -> float:
    """What a gated row is worth to the mix selection: classical-only rows are capped (0.1), and
    the scarce attributes add to it."""
    t, p = _feats(r)
    at = attrs(r, t, p)
    v = CLASSICAL_VALUE if at["classical"] else 1.0
    if EXFALSO_VALUE is not None and at["ex_falso"]:
        v = min(v, EXFALSO_VALUE)
    return v + SCARCE_BONUS * (
        at["case_split"] + at["no_prem"] + at["nn_reused"] + 2 * at["bicond"]
    )


def adaptive_weights(
    prog: dict[str, Any], gated_p: Path, rnd: int, cur: dict[str, float] | None
) -> dict[str, float]:
    """Recipe weights for the next round: proportional to each recipe's accepted value per seed
    it was given over the last two rounds (floor 0.05, total 50). Deterministic given the run."""
    s1 = prog["next_seed"]
    s0 = s1 - 2 * rnd
    w_then = cur or dict(PG.WEIGHTS)
    hist = prog.get("round_weights", {})
    val: collections.Counter[str] = collections.Counter()
    for r in read_complete(gated_p)[0]:
        if s0 <= r["seed"] < s1:
            val[r["recipe"]] += row_value(r)
    rates = {}
    for k in PG.RECIPES:
        seeds = 0.0
        for st in (s0, s0 + rnd):
            w = hist.get(str(st), w_then)
            seeds += rnd * w.get(k, 0.0) / sum(w.values())
        rates[k] = val[k] / seeds if seeds > 0 else 0.0
    top = max(rates.values()) or 1.0
    raw = {k: max(0.05, 6.0 * v / top) for k, v in rates.items()}
    z = sum(raw.values())
    return {k: round(50.0 * v / z, 4) for k, v in raw.items()}


def cmd_generate(a: argparse.Namespace) -> None:
    work = Path(a.out) / "work"
    work.mkdir(parents=True, exist_ok=True)
    gated_p, rej_p, prog_p = work / "gated.jsonl", work / "rejects.jsonl", work / "progress.json"
    prog: dict[str, Any] = {"next_seed": a.seed0, "counts": {}, "secs": 0.0, "rounds": 0}
    if prog_p.exists():
        prog = rjson(prog_p)
    seen: set[str] = set()
    n_ok = 0
    for p in (gated_p, rej_p):
        if p.exists():
            rows, dropped = read_complete(p)
            for r in rows:
                seen.add(r["pkey"])
                n_ok += p == gated_p
            if dropped:
                print(f"{p.name}: dropped {dropped} incomplete line(s) left by a crash", flush=True)
    counts: collections.Counter[str] = collections.Counter(prog["counts"])
    weights = json.loads(a.weights) if a.weights else None
    replay = rjson(a.replay).get("round_weights", {}) if a.replay else None
    streams = prog.setdefault("streams", [])
    if not streams and prog["next_seed"] > a.seed0:
        streams.append({"from_seed": a.seed0, "weights": None})  # a run before streams existed
    if a.adaptive:
        weights = weights or None
    if not streams or streams[-1]["weights"] != weights or a.adaptive:
        # a stream = a seed range sampled with one recipe-weight table (None = pfgen.WEIGHTS)
        streams.append(
            {"from_seed": prog["next_seed"], "weights": weights, "adaptive": bool(a.adaptive)}
        )
    fh = _LOGS.setdefault("gen", open(work / "generate.log", "a"))  # noqa: SIM115
    log(f"generate: resume at seed {prog['next_seed']}, {len(seen)} classes seen, {n_ok} ok", fh)
    with open(a.index, "rb") as f:
        idx = pickle.load(f)
    log(f"index: {len(idx['keys'])} classes from {len(idx['manifest'])} files", fh)
    # --skip: classes another source certifies anyway, or that would be dropped later (one key per
    # line: a bare pkey, or `k:` / `p:` keys of `curriculum.heldout.keys`, the curried normal form
    # included). Matching candidates are not certified.
    skip: set[str] = set()
    ckeys: Any = None
    if a.skip:
        from code.experiments.current.thinking_traces_20261005.curriculum.heldout import keys

        ckeys = keys
        with open(a.skip) as f:
            skip = {line.strip() for line in f if line.strip()}
        log(f"skip: {len(skip)} keys from {a.skip}", fh)
    t_start = time.time()
    with (
        mp.get_context("spawn").Pool(
            a.workers, initializer=_init, initargs=(MEM_GB,), maxtasksperchild=200
        ) as pool,
        open(gated_p, "a") as fg,
        open(rej_p, "a") as fr,
    ):
        while n_ok < a.target and time.time() - t_start < a.hours * 3600:
            t0 = time.time()
            s0 = prog["next_seed"]
            if a.stop_seed and s0 >= a.stop_seed:
                break
            if replay is not None and str(s0) in replay:
                weights = replay[str(s0)]
                prog.setdefault("round_weights", {})[str(s0)] = weights
            elif a.adaptive:
                weights = adaptive_weights(prog, gated_p, a.round, weights)
                prog.setdefault("round_weights", {})[str(s0)] = weights
            chunks = [
                (s, min(s + a.chunk, s0 + a.round), weights)
                for s in range(s0, s0 + a.round, a.chunk)
            ]
            cands = []
            for out, st in pool.imap(construct_chunk, chunks):
                counts.update(st)
                cands += out
            t1 = time.time()
            queue = []
            for c in cands:
                k, pk = class_keys(c["prompt"])
                assert pk is not None
                if pk in seen:
                    counts["dup"] += 1
                    continue
                seen.add(pk)
                if skip and (pk in skip or ckeys(c["prompt"]) & skip):
                    counts["skip"] += 1
                    continue
                c["key"], c["pkey"] = k, pk
                h = idx["keys"].get(k, set()) | idx["pkeys"].get(pk, set())
                if h:
                    counts["excluded"] += 1
                    fr.write(
                        json.dumps(
                            dict(
                                pkey=pk,
                                status="excluded",
                                recipe=c["recipe"],
                                seed=c["seed"],
                                files=sorted(os.path.basename(x) for x in h),
                            )
                        )
                        + "\n"
                    )
                    continue
                queue.append(c)
            round_ok = 0
            for r in pool.imap(certify_safe, queue, chunksize=2):
                counts["certified"] += 1
                counts["cert:" + r["status"]] += 1
                if r["status"] == "ok":
                    n_ok += 1
                    round_ok += 1
                    fg.write(json.dumps(r) + "\n")
                    if round_ok % 100 == 0:
                        fg.flush()
                        fr.flush()
                else:
                    keep = ("pkey", "status", "recipe", "seed", "min_lines", "gen_lines")
                    fr.write(json.dumps({k: r.get(k) for k in keep}) + "\n")
            fg.flush()
            fr.flush()
            prog["next_seed"] = s0 + a.round
            prog["rounds"] += 1
            prog["secs"] += time.time() - t0
            prog["counts"] = dict(counts)
            wjson(prog_p, prog)
            log(
                f"round {prog['rounds']} seeds {s0}-{s0 + a.round}: shape_ok {len(cands)}, new "
                f"classes {len(queue)}, ok {round_ok} (total {n_ok}/{a.target}); construct "
                f"{t1 - t0:.0f}s certify {time.time() - t1:.0f}s; dup {counts['dup']} excl "
                f"{counts['excluded']}",
                fh,
            )
    log(f"generate done: {n_ok} ok, {prog['secs'] / 60:.1f} min total", fh)


# ---------------------------------------------------------------------- lean
def lean_items(rows: list[dict[str, Any]], field: str) -> tuple[list[tuple[str, str]], list[int]]:
    from code.thinking.dan import bare, load

    D = load()
    with bare():
        tok = D.lean_tok.LeanTokenizer("lean_staten")
    items, idx = [], []
    for i, r in enumerate(rows):
        try:
            with bare():
                _, _, env = D.state_env.decompose(r["prompt"], r[field], canon=True)
                items.append((tok.statement(r["prompt"]), tok.text(env.text)))
            idx.append(i)
        except Exception:  # noqa: BLE001  (counted as a rejection)
            pass
    return items, idx


def lean_flags(rows: list[dict[str, Any]], field: str, chunk: int = 2000) -> list[bool]:
    from code.thinking.lean import lean_check

    items, idx = lean_items(rows, field)
    res: list[bool] = []
    t0 = time.time()
    for s in range(0, len(items), chunk):
        res += lean_check(items[s : s + chunk])
        log(f"lean {field}: {len(res)}/{len(items)} {time.time() - t0:.0f}s rej {res.count(False)}")
    ok = [False] * len(rows)
    for i, v in zip(idx, res, strict=True):
        ok[i] = bool(v)
    return ok


def cmd_lean(a: argparse.Namespace) -> None:
    work = Path(a.out) / "work"
    os.environ.setdefault("LEAN_GATE_LOG", str(work / "lean_gate.jsonl"))
    rows = jl(work / "gated.jsonl")
    g = lean_flags(rows, "proof")
    r = lean_flags(rows, "reference_proof")
    kept = []
    for row, x, y in zip(rows, g, r, strict=True):
        row["lean_ok"], row["lean_ref_ok"] = x, y
        if x:
            kept.append(row)
    wjl(work / "gated_lean.jsonl", kept)
    rep = dict(
        n=len(rows),
        constructed_ok=sum(g),
        reference_ok=sum(r),
        kept=len(kept),
        constructed_rejected=[row["pkey"] for row, x in zip(rows, g, strict=True) if not x][:50],
        reference_rejected=[row["pkey"] for row, y in zip(rows, r, strict=True) if not y][:50],
    )
    wjson(work / "lean_report.json", rep)
    log(json.dumps({k: v for k, v in rep.items() if not k.endswith("rejected")}))


# ---------------------------------------------------------------------- select (mix gate)
def attrs(r: dict[str, Any], t: dict[str, Any], p: dict[str, Any]) -> dict[str, bool]:
    """The row's attributes that the mix caps refer to (t, p: gap-analysis theorem and
    shortest-proof features)."""
    fam = set(r["families"])
    return {
        "classical": bool(t["classical_only"]),
        "reductio": "reductio" in fam,
        "nested_neg": "nested_neg" in fam,
        "case_split": "case_split" in fam,
        "no_prem": t["n_prem"] == 0,
        "bicond": bool(t["has_bicond"]),
        "goal_ho": bool(t["goal_imp_imp_ante"]),
        "ex_falso": bool(p["p_has_BOTE"]),
        "short78": r["min_lines"] <= 8,
        "depth2plus": p["p_box_depth"] >= 2,
        "nn_reused": "nested_neg" in fam and bool(r["structure"]["s_nested_neg_reused"]),
        "nn_plain": "nested_neg" in fam and not r["structure"]["s_nested_neg_reused"],
        "irrelevant1": r["n_irrelevant"] >= 1,
        "irrelevant2": r["n_irrelevant"] >= 2,
    }


_GC = None
_LOGS: dict[str, Any] = {}


def _gc() -> Any:
    global _GC
    if _GC is None:
        _GC = ext.gap_common()
    return _GC


def _feats(r: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    GC = _gc()
    return GC.thm_features(r["prompt"]), GC.proof_features(r["prompt"], r["reference_proof"])


#: Mix caps: attribute -> max share of the selected set. Floors (case splits, premise-free
#: statements, biconditionals, reused inner negations: `SCARCE`) are met by taking rows with them
#: first; a floor the reservoir cannot meet shows as a shortfall in select_report.json. Set from
#: the brief's T1-T8 targets: T3 nested negation 25-30% with the inner result reused in about
#: half (nn_plain <= 17%), T4 reductio ~20% and classical-only <= 20%, T5 case splits 35-40%,
#: T7 ex falso <= 10%, T8 biconditionals and (A>B)>C goals ~10% each, T1 >= 60% at 9-12 lines,
#: T2 25-35% without premises. Irrelevant atom occurrences (pfgates.irrelevant; at most 2 pass the
#: shape gate) are capped between the aggregate rates of textbook dev58 7-12 (20% with >= 1, 10%
#: with >= 2) and all of dev58 (38%, 29%).
#: T6 (depth >= 2 in ~70%) is not capped: depth-1 rows with a
#: needed target structure are rare (mostly classical reductio, capped already), so a cap on
#: depth >= 2 would act as an unmeetable floor on depth 1.
CAPS: dict[str, float] = {
    "classical": 0.20,
    "reductio": 0.22,
    "nested_neg": 0.32,
    "nn_plain": 0.17,
    "goal_ho": 0.12,
    "ex_falso": 0.10,
    "short78": 0.40,
    "bicond": 0.12,
    "no_prem": 0.35,
    "case_split": 0.42,
    "irrelevant1": 0.40,
    "irrelevant2": 0.12,
}
#: Caps relaxed (in this order, +RELAX_STEP at a time, up to CAP_CEIL) when the reservoir cannot
#: fill --n at the targets. Classical-only (T4) comes last and stops at 25%; ex falso (T7) is
#: never relaxed.
CAP_CEIL: dict[str, float] = {"classical": 0.25}
CAP_STEP: dict[str, float] = {"classical": 0.01}
RELAX_ORDER = (
    "irrelevant1",
    "irrelevant2",
    "nn_plain",
    "nested_neg",
    "goal_ho",
    "short78",
    "reductio",
    "bicond",
    "no_prem",
    "case_split",
    "classical",
)
RELAX_STEP = 0.05
FLOORS: dict[str, float] = {"case_split": 0.35, "no_prem": 0.25, "bicond": 0.10, "nn_reused": 0.12}
SCARCE = ("case_split", "no_prem", "bicond", "nn_reused")


def hkey(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def greedy(rows: list[dict[str, Any]], n: int, caps: dict[str, float]) -> list[int]:
    """Indices kept by the deterministic greedy pass for a target size n."""
    lim = {k: int(v * n) for k, v in caps.items()}
    used: collections.Counter[str] = collections.Counter()
    order = sorted(
        range(len(rows)),
        key=lambda i: (-sum(rows[i]["_a"][s] for s in SCARCE), hkey(rows[i]["pkey"])),
    )
    keep = []
    for i in order:
        at = rows[i]["_a"]
        hit = [k for k in caps if (at[k[4:]] is False if k.startswith("not:") else at[k])]
        if any(used[k] + 1 > lim[k] for k in hit):
            continue
        used.update(hit)
        keep.append(i)
        if len(keep) >= n:
            break
    return keep


def cmd_select(a: argparse.Namespace) -> None:
    work = Path(a.out) / "work"
    rows = jl(work / "gated_lean.jsonl")
    with mp.get_context("spawn").Pool(a.workers) as p:
        feats = p.map(_feats, rows, chunksize=50)
    for r, (t, pf) in zip(rows, feats, strict=True):
        r["_t"], r["_p"] = t, pf
        r["_a"] = attrs(r, t, pf)
    caps = dict(CAPS)
    if a.caps:
        caps.update(json.loads(a.caps))
    # Fill --n rows. While a cap blocks that, relax the first blocking cap in RELAX_ORDER by 5 pp
    # (the soft targets first, classical-only last and only to CAP_CEIL, ex falso never). If no
    # cap can be relaxed further, shrink n to what the pass reaches (re-run until it fills, so
    # every cap holds for the final set).
    n = min(a.n, len(rows))
    relaxed: list[tuple[str, float]] = []
    while True:
        keep = greedy(rows, n, caps)
        if len(keep) >= n:
            break
        used = collections.Counter(k for i in keep for k in caps if rows[i]["_a"][k])
        blocking = [k for k in RELAX_ORDER if k in caps and used[k] >= int(caps[k] * n)]
        blocking = [k for k in blocking if caps[k] < CAP_CEIL.get(k, 1.0)]
        if blocking:
            k = blocking[0]
            caps[k] = round(min(CAP_CEIL.get(k, 1.0), caps[k] + CAP_STEP.get(k, RELAX_STEP)), 4)
            relaxed.append((k, caps[k]))
            continue
        n = len(keep)
    keep = keep[:n]
    keep = keep[:n]
    kept = set(keep)
    sel = [rows[i] for i in sorted(keep, key=lambda i: hkey(rows[i]["pkey"]))]
    share = {k: round(sum(r["_a"][k] for r in sel) / max(1, len(sel)), 4) for k in sel[0]["_a"]}
    natural = {k: round(sum(r["_a"][k] for r in rows) / len(rows), 4) for k in rows[0]["_a"]}
    for r in sel:
        r["attrs"] = r.pop("_a")
        r["gap_thm"] = r.pop("_t")
        r["gap_proof"] = r.pop("_p")
    spare = [
        {k: v for k, v in r.items() if not k.startswith("_")}
        for i, r in enumerate(rows)
        if i not in kept
    ]
    wjl(work / "spare.jsonl", spare)
    wjl(work / "selected.jsonl", sel)
    rep = dict(
        caps_relaxed=relaxed,
        floors=FLOORS,
        floor_shortfall={k: round(f - share[k], 4) for k, f in FLOORS.items() if share[k] < f},
        n_in=len(rows),
        n_max=a.n,
        n_selected=len(sel),
        n_spare=len(spare),
        caps=caps,
        shares_selected=share,
        shares_gated=natural,
    )
    wjson(work / "select_report.json", rep)
    log(json.dumps(rep))


# ---------------------------------------------------------------------- split
FEAT_KEEP = [
    "fdepth",
    "n_atoms",
    "box_depth",
    "n_boxes",
    "goal_type",
    "top_rule",
    "first_kind",
    "classical_only",
    "prem_inconsistent",
    "n_redundant_prem",
    "n_disj_prem",
    "n_prem_or",
    "n_prem_imp",
    "raa",
]
PRIORITY = ("nested_neg", "case_split", "reductio", "depth2")


def primary(r: dict[str, Any]) -> str:
    fam = r["families"]
    intent = {
        "IN": "nested_neg",
        "INO": "nested_neg",
        "INR": "nested_neg",
        "NN": "nested_neg",
        "CN": "nested_neg",
        "CW": "reductio",
        "CO": "reductio",
        "IO": "case_split",
        "IOX": "case_split",
        "IIO": "case_split",
        "NO": "case_split",
        "II": "depth2",
    }.get(r["recipe"].removeprefix("B_"))
    if intent in fam:
        return intent
    return next(f for f in PRIORITY if f in fam)


def final_row(
    r: dict[str, Any], name: str, tok: Any, D: Any, source: str = "hard712"
) -> dict[str, Any]:
    from code.experiments.current.thinking_traces_20261005.pools.hard12.features import features
    from code.experiments.current.thinking_traces_20261005.pools.prep import depth
    from code.thinking.dan import bare

    with bare():
        steps, _, _ = D.state_env.decompose(r["prompt"], r["reference_proof"], canon=True)
    f = features(r["prompt"], r["reference_proof"])
    prem, goal = _gc().parse_prompt(r["prompt"])
    thm = " , ".join(PG.fstr(p) for p in prem) + " |- " + PG.fstr(goal)
    rules = sorted({ln["rule"] for ln in _gc().parse_proof(r["proof"])})
    L = r["min_lines"]
    return {
        "name": name,
        "thm": thm.strip(),
        "key": r["key"],
        "pkey": r["pkey"],
        "prompt": r["prompt"],
        "n_lines": L,
        "L_true": L,
        "minlen_bound": r["gen_lines"],
        "source": source,
        "schema": None,
        "gen_lines": r["gen_lines"],
        "n_prem": len(prem),
        "rules": rules,
        "gen_proof": r["proof"],
        "reference_proof": r["reference_proof"],
        "reference_lines": L,
        "min_lines": L,
        "family": primary(r),
        "families": r["families"],
        "knob": L,
        "band": None,
        "cfg": r["recipe"],
        "recipe": r["recipe"],
        "seed": r["seed"],
        "subst": r.get("subst"),
        "n_irrelevant": r["n_irrelevant"],
        "discharged": r.get("discharged", 0),
        "prompt_tokens": f["prompt_tokens"],
        "formula_depth": depth(r["prompt"]),
        "max_state": max(len(tok.encode_toks(s)) for s, _, _ in steps),
        "max_action": max(len(tok.encode_toks(x)) + 1 for _, x, _ in steps),
        "ref_steps": len(steps),
        "in_train_class": False,
        "lean_ok": r["lean_ok"],
        "lean_ref_ok": r["lean_ref_ok"],
        "necessity": r["necessity"],
        "structure": r["structure"],
        "skeleton": r["gap_proof"]["p_skeleton"],
        "attrs": r["attrs"],
        **{f"f_{k}": f[k] for k in FEAT_KEEP},
        "gap_thm": r["gap_thm"],
        "gap_proof": r["gap_proof"],
    }


def cmd_split(a: argparse.Namespace) -> None:
    from code.thinking.dan import load
    from code.thinking.tok import ThinkTokenizer

    out = Path(a.out)
    work = out / "work"
    sel = jl(work / "selected.jsonl")
    rng = random.Random(SPLIT_SEED)
    order = list(range(len(sel)))
    rng.shuffle(order)
    parts = {
        "test_1k": order[: a.n_test],
        "dev_500": order[a.n_test : a.n_test + a.n_dev],
        "pool": order[a.n_test + a.n_dev :],
    }
    pre = a.prefix
    prefix = {"test_1k": f"{pre}t", "dev_500": f"{pre}d", "pool": pre}
    D = load()
    tok = ThinkTokenizer()
    for part, ids in parts.items():
        rows = [
            final_row(sel[i], f"{prefix[part]}_{sel[i]['seed']}", tok, D, a.source)
            for i in sorted(ids)
        ]
        wjl(out / f"{part}.jsonl", rows)
        with open(out / f"{part}.keys.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(dict(name=r["name"], key=r["key"], pkey=r["pkey"])) + "\n")
        log(f"{part}: {len(rows)} rows")
    keys: list[set[str]] = []
    with open(out / "keys.jsonl", "w") as f:
        for part in parts:
            ks = jl(out / f"{part}.keys.jsonl")
            keys.append({r["pkey"] for r in ks})
            for r in ks:
                f.write(json.dumps(dict(r, split=part)) + "\n")
    assert not (keys[0] & keys[1]) and not (keys[0] & keys[2]) and not (keys[1] & keys[2])
    log("splits are disjoint by renaming class; keys.jsonl written")


# ---------------------------------------------------------------------- verify
def _vfeat(r: dict[str, Any]) -> dict[str, Any]:
    GC = _gc()
    out = dict(GC.thm_features(r["prompt"]))
    out.update(GC.proof_features(r["prompt"], r["reference_proof"]))
    out["min_lines"] = r["min_lines"]
    return out


def agg(rows: list[dict[str, Any]], keys: list[str]) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for k in keys:
        v = [r[k] for r in rows if r.get(k) is not None]
        out[k] = round(sum(v) / len(v), 4) if v else None
    return out


TABLE_BIN = [
    ("no_prem", "no premises"),
    ("goal_atom", "goal is an atom"),
    ("goal_not", "goal is a negation"),
    ("goal_or", "goal is a disjunction"),
    ("goal_imp", "goal is an implication"),
    ("goal_imp_imp_ante", "goal is (A>B)>C"),
    ("has_bicond", "biconditional (A>B)&(B>A) inside"),
    ("has_neg_compound", "negated compound ~(A o B) inside"),
    ("has_dneg", "double negation ~~A inside"),
    ("classical_only", "classical-only (G4ip)"),
    ("prem_inconsistent", "premises jointly inconsistent"),
    ("ge9", "shortest proof >= 9 lines"),
    ("compact_long", "<= 7 connectives and shortest >= 9 lines"),
    ("t3any", "reductio, or a negation box inside another box"),
    ("p_negbox_nested", "negation / reductio box nested inside another box"),
    ("p_negbox_in_negbox", "... inside a negation / reductio box"),
    ("p_negbox_in_impi", "... inside an implication box"),
    ("p_has_negi_plain", "plain negation intro (NEGI, no DN)"),
    ("p_has_raa", "reductio"),
    ("p_has_ORE", "case split (ORE)"),
    ("p_ore_in_negbox", "case split inside a negation / reductio box"),
    ("p_ore_both_branches_work", "both case branches >= 2 lines"),
    ("p_ore_on_hyp", "case split on an assumption"),
    ("d2", "box depth >= 2"),
    ("nobox", "no box at all"),
    ("p_has_IMPI", "implication box"),
    ("p_has_BOTE", "ex falso"),
    ("p_negi_lemma", "a negation-intro result reused by IMPE / NEGE"),
    ("p_box_lemma_elim", "a box result reused by an elimination"),
    ("p_first_forward", "first step forward (not an assumption)"),
    ("irr1", ">= 1 irrelevant atom occurrence"),
    ("irr2", ">= 2 irrelevant atom occurrences"),
]
TABLE_NUM = [
    ("n_conn", "connectives"),
    ("n_prem", "premises"),
    ("n_atoms", "distinct atoms"),
    ("fdepth", "formula depth"),
    ("imp_nest", "nested-implication depth"),
    ("n_redundant_prem", "redundant premises"),
    ("min_lines", "shortest proof (lines)"),
    ("p_box_depth", "box depth"),
    ("p_elim_depth2plus", "elimination lines at depth >= 2"),
    ("p_max_box_lines", "longest box (lines)"),
]


def derived(f: dict[str, Any]) -> dict[str, Any]:
    f = dict(f)
    L = f.get("min_lines")
    f["ge9"] = int(L is not None and L >= 9)
    f["compact_long"] = int(L is not None and L >= 9 and f["n_conn"] <= 7)
    f["t3any"] = int(bool(f.get("p_has_raa") or f.get("p_negbox_nested")))
    f["d2"] = int(f.get("p_box_depth", 0) >= 2)
    f["nobox"] = int(f.get("p_box_depth", 0) == 0)
    return f


def gap_reference() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """rl_hard12 rows (shortest-proof features) and dev58 7-12 rows from the gap analysis'
    features.jsonl. Only aggregates of the dev58 rows are ever reported."""
    pool, dev = [], []
    with open(GAP_OUT / "features.jsonl") as f:
        for line in f:
            r = json.loads(line)
            if r["set"] == "pool":
                pool.append(derived(r))
            elif r["set"] == "dev58" and r["ref_lines"] is not None and 7 <= r["ref_lines"] <= 12:
                dev.append(derived(r))
    return pool, dev


def irrelevance_rows(sample: list[dict[str, Any]]) -> dict[str, list[int]]:
    """Irrelevant-occurrence counts (pfgates.irrelevant) for the hard712 sample, the rl_hard12
    pool and dev58 7-12, in the order gap_reference() returns them (only aggregates of the dev58
    rows are reported)."""
    pool12 = {r["name"]: r["prompt"] for r in jl(POOL12)}
    names = []
    dev = []
    with open(GAP_OUT / "features.jsonl") as f:
        for line in f:
            r = json.loads(line)
            if r["set"] == "pool":
                names.append(r["name"])
            elif r["set"] == "dev58" and r["ref_lines"] is not None and 7 <= r["ref_lines"] <= 12:
                dev.append(r["name"])
    tb = {r["name"]: r["prompt"] for r in jl(Path(GT.TEXTBOOK) / "textbook_dev.jsonl")}

    def n(p: str) -> int:
        return GT.irrelevant(*GT.statement(p))

    return {
        "hard712": [n(r["prompt"]) for r in sample],
        "rl_hard12": [n(pool12[x]) for x in names],
        "dev58": [n(tb[x]) for x in dev],
    }


def table(groups: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    out: dict[str, Any] = {"n": {g: len(v) for g, v in groups.items()}, "binary": {}, "numeric": {}}
    for k, label in TABLE_BIN:
        out["binary"][label] = {
            g: (round(100 * sum(int(bool(r.get(k))) for r in v) / len(v), 1) if v else None)
            for g, v in groups.items()
        }
    for k, label in TABLE_NUM:
        out["numeric"][label] = {g: agg(v, [k])[k] for g, v in groups.items()}
    return out


def _lean_ref(rows: list[dict[str, Any]]) -> list[bool]:
    return lean_flags(rows, "reference_proof")


def shape_key(f: Any) -> str:
    """The statement's formula shape with atoms abstracted (diversity count)."""
    if f[0] == "atom":
        return "a"
    if f[0] == "not":
        return f"~{shape_key(f[1])}"
    return f"({shape_key(f[1])}{f[0][0]}{shape_key(f[2])})"


def cmd_verify(a: argparse.Namespace) -> None:
    out = Path(a.out)
    work = out / "work"
    os.environ.setdefault("LEAN_GATE_LOG", str(work / "lean_gate_verify.jsonl"))
    pool = jl(out / "pool.jsonl")
    rng = random.Random(SPLIT_SEED + 1)
    s2k = rng.sample(pool, min(2000, len(pool)))
    with mp.get_context("spawn").Pool(a.workers) as p:
        feats = [derived(f) for f in p.map(_vfeat, s2k, chunksize=20)]
    ref_pool, ref_dev = gap_reference()
    irr = irrelevance_rows(s2k)
    for f, x in zip(feats, irr["hard712"], strict=True):
        f["irr1"], f["irr2"] = int(x >= 1), int(x >= 2)
    for f, x in zip(ref_pool, irr["rl_hard12"], strict=True):
        f["irr1"], f["irr2"] = int(x >= 1), int(x >= 2)
    for f, x in zip(ref_dev, irr["dev58"], strict=True):
        f["irr1"], f["irr2"] = int(x >= 1), int(x >= 2)
    tab = table({"hard712 pool sample": feats, "rl_hard12 pool": ref_pool, "dev58 7-12": ref_dev})
    s1k = rng.sample(pool, min(1000, len(pool)))
    lean = _lean_ref(s1k)
    GC = _gc()
    allrows = pool + jl(out / "test_1k.jsonl") + jl(out / "dev_500.jsonl")
    shapes = set()
    rule_seqs = set()
    for r in allrows:
        prem, goal = GC.parse_prompt(r["prompt"])
        shapes.add(" , ".join(sorted(shape_key(x) for x in prem)) + " |- " + shape_key(goal))
        rule_seqs.add(
            " ".join(f"{ln['depth']}{ln['rule']}" for ln in GC.parse_proof(r["reference_proof"]))
        )
    div = {
        "rows": len(allrows),
        "renaming_classes": len({r["key"] for r in allrows}),
        "premise_order_classes": len({r["pkey"] for r in allrows}),
        "box_skeletons": len({r["skeleton"] for r in allrows}),
        "statement_shapes_atoms_abstracted": len(shapes),
        "reference_rule_sequences": len(rule_seqs),
        "atoms": dict(collections.Counter(r["f_n_atoms"] for r in allrows)),
        "top_skeletons": collections.Counter(r["skeleton"] for r in allrows).most_common(25),
    }
    rep = {
        "feature_table": tab,
        "lean_reference_random_1000": {
            "n": len(lean),
            "ok": sum(lean),
            "pct": round(100 * sum(lean) / max(1, len(lean)), 2),
        },
        "diversity": div,
    }
    wjson(work / "verify.json", rep)
    groups = list(tab["n"])
    md = ["| feature | " + " | ".join(f"{g} (n={tab['n'][g]:,})" for g in groups) + " |"]
    md.append("|---" * (len(groups) + 1) + "|")
    for label, row in tab["binary"].items():
        md.append(f"| {label} | " + " | ".join(f"{row[g]:.0f}%" for g in groups) + " |")
    for label, row in tab["numeric"].items():
        md.append(f"| **mean** {label} | " + " | ".join(f"{row[g]:.2f}" for g in groups) + " |")
    (work / "verify_table.md").write_text("\n".join(md) + "\n")
    log(json.dumps(rep["lean_reference_random_1000"]))
    log(json.dumps({k: v for k, v in div.items() if k != "top_skeletons"}))


# ---------------------------------------------------------------------- meta
def hist(vals: list[Any]) -> dict[str, int]:
    return {str(k): v for k, v in sorted(collections.Counter(vals).items(), key=lambda x: str(x))}


def cmd_meta(a: argparse.Namespace) -> None:
    out = Path(a.out)
    work = out / "work"
    prog = rjson(work / "progress.json")
    gated = jl(work / "gated.jsonl")
    rejects = jl(work / "rejects.jsonl")
    sel_rep = rjson(work / "select_report.json")
    lean_rep = rjson(work / "lean_report.json")
    parts = {p: jl(out / f"{p}.jsonl") for p in ("pool", "test_1k", "dev_500")}
    allrows = [r for v in parts.values() for r in v]
    by_rec: dict[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)
    for r in gated:
        by_rec[r["recipe"]]["ok"] += 1
    for r in rejects:
        by_rec[r["recipe"]][r["status"]] += 1
    for r in allrows:
        by_rec[r["recipe"]]["selected"] += 1

    def mix(rows: list[dict[str, Any]]) -> dict[str, Any]:
        n = max(1, len(rows))
        return {
            "n": len(rows),
            "primary_family": {
                k: round(v / n, 4)
                for k, v in collections.Counter(r["family"] for r in rows).items()
            },
            "family_any": {
                f: round(sum(f in r["families"] for r in rows) / n, 4) for f in GT.FAMILIES
            },
            "attrs": {k: round(sum(r["attrs"][k] for r in rows) / n, 4) for k in rows[0]["attrs"]},
            "t3_inner_reused": round(
                sum(
                    r["structure"]["s_nested_neg_reused"]
                    for r in rows
                    if "nested_neg" in r["families"]
                )
                / max(1, sum("nested_neg" in r["families"] for r in rows)),
                4,
            ),
            "t5_split_on_hyp": round(
                sum(r["structure"]["s_split_on_hyp"] for r in rows if "case_split" in r["families"])
                / max(1, sum("case_split" in r["families"] for r in rows)),
                4,
            ),
            "lines_9_12": round(sum(r["min_lines"] >= 9 for r in rows) / n, 4),
            "gap_0": round(sum(r["gen_lines"] == r["min_lines"] for r in rows) / n, 4),
        }

    meta = {
        "written_on": "2026-10-08",
        "generator": "hard712 proof-first generator (code/experiments/current/"
        "thinking_traces_20261005/hard712, branch leon/thinking)",
        "files": {p: str(out / f"{p}.jsonl") for p in parts},
        "yields": {
            "seeds": prog["next_seed"],
            "rounds": prog["rounds"],
            "generate_minutes": round(prog["secs"] / 60, 1),
            "counts": prog["counts"],
            "gated": len(gated),
            "lean": {k: v for k, v in lean_rep.items() if not k.endswith("rejected")},
            "select": sel_rep,
            "final": {p: len(v) for p, v in parts.items()},
            "by_recipe": {k: dict(v) for k, v in sorted(by_rec.items())},
        },
        "mix": {p: mix(v) for p, v in parts.items()},
        "histograms": {
            "min_lines": hist([r["min_lines"] for r in allrows]),
            "gen_minus_min": hist([r["gen_lines"] - r["min_lines"] for r in allrows]),
            "n_conn": hist([r["gap_thm"]["n_conn"] for r in allrows]),
            "n_prem": hist([r["n_prem"] for r in allrows]),
            "n_atoms": hist([r["f_n_atoms"] for r in allrows]),
            "fdepth": hist([r["f_fdepth"] for r in allrows]),
            "box_depth": hist([r["f_box_depth"] for r in allrows]),
            "goal_type": hist([r["f_goal_type"] for r in allrows]),
            "skeleton": dict(collections.Counter(r["skeleton"] for r in allrows).most_common(40)),
            "recipe": hist([r["recipe"] for r in allrows]),
            "max_state": {"max": max(r["max_state"] for r in allrows)},
        },
    }
    vp = work / "verify.json"
    if vp.exists():
        meta["verify"] = rjson(vp)
    wjson(out / "meta.json", meta)
    log(f"meta.json written: {json.dumps(meta['yields']['final'])}")


# ---------------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(OUT))
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--workers", type=int, default=24)
    g.add_argument("--round", type=int, default=400_000, help="seeds per round")
    g.add_argument("--chunk", type=int, default=5_000, help="seeds per construction task")
    g.add_argument("--target", type=int, default=60_000, help="stop at this many gated rows")
    g.add_argument("--hours", type=float, default=5.0, help="stop after this wall time")
    g.add_argument("--seed0", type=int, default=0)
    g.add_argument(
        "--weights", default="", help="JSON recipe-weight table for this stream (default pfgen)"
    )
    g.add_argument("--stop_seed", type=int, default=0, help="end before the round at this seed")
    g.add_argument(
        "--replay", default="", help="progress.json whose recorded per-round weights to reuse"
    )
    g.add_argument(
        "--adaptive",
        action="store_true",
        help="re-weight recipes before each round by their recent accepted value per seed",
    )
    g.add_argument("--index", default=str(OUT / "work" / "excl_index.pkl"))
    g.add_argument("--skip", default="", help="file of class keys not to certify (see generate)")
    sub.add_parser("lean")
    s = sub.add_parser("select")
    s.add_argument("--n", type=int, required=True, help="target size (test + dev + pool)")
    s.add_argument("--caps", default="", help="JSON overrides of CAPS")
    s.add_argument("--workers", type=int, default=16)
    sp = sub.add_parser("split")
    sp.add_argument("--n_test", type=int, default=1000)
    sp.add_argument("--n_dev", type=int, default=500)
    sp.add_argument("--prefix", default="h712", help="row-name prefix (test +t, dev +d)")
    sp.add_argument("--source", default="hard712", help="the rows' `source` field")
    v = sub.add_parser("verify")
    v.add_argument("--workers", type=int, default=16)
    sub.add_parser("meta")
    a = ap.parse_args(argv)
    {
        "generate": cmd_generate,
        "lean": cmd_lean,
        "select": cmd_select,
        "split": cmd_split,
        "verify": cmd_verify,
        "meta": cmd_meta,
    }[a.cmd](a)


if __name__ == "__main__":
    main()
