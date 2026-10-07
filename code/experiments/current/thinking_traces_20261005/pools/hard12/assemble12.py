"""Assemble the hard12 RL pool and its sealed test set from the T0 k = 32 measurements.

    A=code.experiments.current.thinking_traces_20261005.pools.hard12.assemble12
    python -m $A build --work runs/pools/hard12_work --index excl_index.pkl --n_pool 5000 \
        --n_test 2500
    (Lean on the picked references inside `build`; T1v1 on 1,000 + 1,000 afterwards with
    run_passrate.sh)
    python -m $A meta --work runs/pools/hard12_work

`build`:
  1. joins every T0 k 32 rows.jsonl under <work>/passrate/ to its candidate record (the shard files
     <work>/{b*,pilot}*_s*.jsonl and the work-queue shards <work>/doing/*.jsonl);
  2. keeps 2 <= n_ok <= 16 of 32 (per-attempt pass rate 6.25-50 %);
  3. drops renaming classes in the exclusion index (`exclude.py`), then duplicates within the kept
     set (class key or premise-order key; the first measured row wins);
  4. Lean-checks the kept references (`state_env.decompose` -> `lean_seq` text ->
     `code.thinking.lean.lean_check`, as test_21_30/lean_set.py; LEAN_GATE_WORKERS <= 16) and drops
     rejections;
  5. shuffles (seed) and splits: the first --n_test rows are the sealed test set, the next --n_pool
     the RL pool (fewer if the band is short: test gets a third, the pool two thirds); the split is
     random, so both come from the same distribution, and disjoint by construction (distinct
     classes).
Outputs: runs/pools/rl_hard12.jsonl (the ladder rl_targets fields + T0 measurement + features),
runs/pools/hard12_test/hard12_test.jsonl (the OOD-set row format `code.thinking.eval --sets`
reads), hard12_test.keys.jsonl (name, key, pkey), the 1,000-row T1v1 samples
<work>/t1_{pool,test}.jsonl, the spare in-band targets <work>/spare_band.jsonl, and
<work>/build.json. `meta` writes rl_hard12.meta.json and hard12_test/hard12_test.meta.json (and
copies into the code folder).
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import pickle
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any

from code.experiments.current.thinking_traces_20261005.test_21_30.seal import hits
from code.thinking.dan import bare, load
from code.thinking.data import class_keys, read_jsonl
from code.thinking.lean import lean_check

from .features import RULES

D = load()
POOLS = "/scratch/cowifornia/nd-rl/think/runs/pools"
POOL = f"{POOLS}/rl_hard12.jsonl"
TEST = f"{POOLS}/hard12_test/hard12_test.jsonl"
CODE = (
    "/scratch/cowifornia/nd-rl/think/nd-rl/code/experiments/current/thinking_traces_20261005/"
    "pools/hard12"
)
K12 = "/scratch/cowifornia/nd-rl/bench/dan_ckpts/data/train_k12.jsonl"
T0 = "/scratch/cowifornia/nd-rl/think/runs/stage1/T0_s0/model.pt"
T1 = "/scratch/cowifornia/nd-rl/think/runs/stage1/T1v1_s0/model.pt"
LO, HI = 2, 16


def write_jsonl(path: str, rows: list[dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def t0_runs(work: str) -> list[str]:
    out = []
    for d in sorted(glob.glob(f"{work}/passrate/*/")):
        if not (os.path.exists(f"{d}/args.json") and os.path.exists(f"{d}/rows.jsonl")):
            continue
        if os.path.basename(d.rstrip("/")).startswith(
            ("t1_", "t0re_")
        ):  # reference runs on the final sets
            continue
        args = json.loads(Path(f"{d}/args.json").read_text())
        if args["ckpt"] == T0 and args["k"] == 32:
            out.append(d.rstrip("/"))
    return out


def records(work: str) -> dict[str, dict[str, Any]]:
    recs = {}
    shards = (
        glob.glob(f"{work}/b*_s*.jsonl")
        + glob.glob(f"{work}/pilot*_s*.jsonl")
        + glob.glob(f"{work}/doing/*.jsonl")
    )
    for p in sorted(shards):
        if os.path.basename(p).startswith(("t1_", "t0re_")):
            continue
        for r in read_jsonl(p):
            r.setdefault("cfg", "strict_long")
            recs[r["name"]] = r
    return recs


def lean_refs(rows: list[dict[str, Any]]) -> list[bool]:
    with bare():
        TOK = D.lean_tok.LeanTokenizer("lean_staten")
    items, idx = [], []
    ok = [False] * len(rows)
    for i, r in enumerate(rows):
        try:
            with bare():
                _, _, env = D.state_env.decompose(r["prompt"], r["proof"], canon=True)
                items.append((TOK.statement(r["prompt"]), TOK.text(env.text)))
            idx.append(i)
        except Exception:  # noqa: BLE001  (counted as a rejection)
            pass
    res: list[bool] = []
    for s in range(0, len(items), 2000):
        res += lean_check(items[s : s + 2000])
        print(f"lean {len(res)}/{len(items)}, rejected {res.count(False)}", flush=True)
    for i, v in zip(idx, res, strict=True):
        ok[i] = bool(v)
    return ok


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


def pool_row(r: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    """The ladder rl_targets fields (as pools/assemble.pool_row), plus the T0 k 32 measurement and
    features. Names are h12_<candidate name> (h12t_ in the test set): rh_* are rows of the previous
    agent's cands_clean.jsonl, h12<batch>_* rows of this pool's steered streams."""
    f = r["feat"]
    return {
        "name": f"h12_{r['name']}",
        "thm": r["thm"],
        "key": r["key"],
        "prompt": r["prompt"],
        "n_lines": r["n_lines"],
        "L_true": None,
        "minlen_bound": None,
        "source": "gen",
        "schema": None,
        "gen_lines": r["n_lines"],
        "n_prem": r["n_prem"],
        "rules": r["rules"],
        "gen_proof": r["proof"],
        "band": "hard12",
        "t0_k": row["k"],
        "t0_n_ok": row["n_ok"],
        "t0_pass1": round(row["n_ok"] / row["k"], 4),
        "cfg": r["cfg"],
        "band_score": r["band_score"],
        "prompt_tokens": f["prompt_tokens"],
        "formula_depth": f["fdepth"],
        "max_state": r["max_state"],
        **{f"f_{k}": f[k] for k in FEAT_KEEP},
    }


def test_row(r: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    """The OOD-set row format (name, prompt, reference_lines, family, knob, reference_proof,
    ...)."""
    f = r["feat"]
    return {
        "name": f"h12t_{r['name']}",
        "prompt": r["prompt"],
        "reference_lines": r["n_lines"],
        "family": "hard12",
        "knob": r["n_lines"],
        "reference_proof": r["proof"],
        "max_state": r["max_state"],
        "max_action": r.get("max_action"),
        "formula_depth": f["fdepth"],
        "in_train_class": False,
        "thm": r["thm"],
        "key": r["key"],
        "cfg": r["cfg"],
        "t0_k": row["k"],
        "t0_n_ok": row["n_ok"],
        "band_score": r["band_score"],
        **{f"f_{k}": f[k] for k in FEAT_KEEP},
    }


def cmd_build(a: argparse.Namespace) -> None:
    t0 = time.time()
    runs = t0_runs(a.work)
    recs = records(a.work)
    rows: dict[str, dict[str, Any]] = {}
    for d in runs:
        for row in read_jsonl(f"{d}/rows.jsonl"):
            rows.setdefault(row["name"], {**row, "run": os.path.basename(d)})
    measured = [(recs[n], rows[n]) for n in rows if n in recs]
    band = [(r, row) for r, row in measured if LO <= row["n_ok"] <= HI]
    with open(a.index, "rb") as f:
        idx = pickle.load(f)
    st: collections.Counter[str] = collections.Counter()
    per_file: collections.Counter[str] = collections.Counter()
    seen_k: set[str] = set()
    seen_pk: set[str] = set()
    kept = []
    for r, row in sorted(band, key=lambda p: (p[1]["run"], p[0]["name"])):
        h = hits(idx, r["prompt"])
        if h:
            st["excluded"] += 1
            per_file.update(os.path.basename(x) for x in h)
            continue
        k, pk = class_keys(r["prompt"])
        if k in seen_k or (pk is not None and pk in seen_pk):
            st["dup_within"] += 1
            continue
        seen_k.add(k)
        if pk is not None:
            seen_pk.add(pk)
        kept.append((r, row))
    ok = lean_refs([r for r, _ in kept])
    st["lean_rejected"] = ok.count(False)
    kept = [p for p, v in zip(kept, ok, strict=True) if v]
    rng = random.Random(a.seed)
    rng.shuffle(kept)
    n = len(kept)
    if n >= a.n_pool + a.n_test:
        n_test, n_pool = a.n_test, a.n_pool
    else:
        n_test = n // 3
        n_pool = n - n_test
    test, pool = kept[:n_test], kept[n_test : n_test + n_pool]
    spare = kept[n_test + n_pool :]
    write_jsonl(POOL, [pool_row(r, row) for r, row in pool])
    write_jsonl(f"{a.work}/spare_band.jsonl", [pool_row(r, row) for r, row in spare])
    trows = [test_row(r, row) for r, row in test]
    write_jsonl(TEST, trows)
    with open(TEST.replace(".jsonl", ".keys.jsonl"), "w") as f:
        for t in trows:
            k, pk = class_keys(t["prompt"])
            f.write(json.dumps({"name": t["name"], "key": k, "pkey": pk}) + "\n")
    # T1v1 reference samples (1,000 each, seeded)
    for nm, xs in (("pool", [pool_row(r, row) for r, row in pool]), ("test", trows)):
        s = list(xs)
        random.Random(a.seed + 1).shuffle(s)
        write_jsonl(f"{a.work}/t1_{nm}.jsonl", s[: a.n_t1])
    # disjointness checks
    pk_pool = {class_keys(r["prompt"]) for r, _ in pool}
    pk_test = {class_keys(r["prompt"]) for r, _ in test}
    kp = {k for k, _ in pk_pool}
    kt = {k for k, _ in pk_test}
    pp = {p for _, p in pk_pool if p}
    pt = {p for _, p in pk_test if p}
    build = {
        "runs": runs,
        "measured": len(measured),
        "in_band": len(band),
        "band": f"{LO}-{HI} of 32",
        "dedupe": dict(st),
        "excluded_by_file": dict(per_file.most_common()),
        "exclusion_files": idx["manifest"],
        "kept_after_lean": n,
        "n_pool": len(pool),
        "n_test": len(test),
        "unused_band_rows": n - len(pool) - len(test),
        "spare_file": f"{a.work}/spare_band.jsonl (in-band, deduplicated, Lean-checked, pool-row "
        "format; disjoint from pool and test)",
        "pool_test_shared_keys": len(kp & kt),
        "pool_test_shared_pkeys": len(pp & pt),
        "seed": a.seed,
        "secs": round(time.time() - t0, 1),
    }
    with open(f"{a.work}/build.json", "w") as f:
        json.dump(build, f, indent=1)
    print(json.dumps({k: v for k, v in build.items() if k != "exclusion_files"}, indent=1))


# --------------------------------------------------------------------------------------------- meta
def hist32(ns: list[int]) -> dict[str, int]:
    c = collections.Counter(ns)
    return {f"{i}/32": c.get(i, 0) for i in range(33)}


def summ(ns: list[int]) -> dict[str, Any]:
    if not ns:
        return {}
    return {
        "n": len(ns),
        "mean_pass1_pct": round(100 * sum(ns) / (32 * len(ns)), 1),
        "pass@32_pct": round(100 * sum(x > 0 for x in ns) / len(ns), 1),
        "unsolved_pct": round(100 * sum(x == 0 for x in ns) / len(ns), 1),
        "in_band_2_16_pct": round(100 * sum(LO <= x <= HI for x in ns) / len(ns), 1),
        "hist": hist32(ns),
    }


def profile(fs: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(fs)
    o: dict[str, Any] = {"n": n}
    o["rule_rate_pct"] = {
        r: round(100 * sum(f.get(f"has_{r}", 0) for f in fs) / n, 1)
        for r in [*RULES, "ORI", "ANDE"]
        if r != "AS"
    }
    o["raa_pct"] = round(100 * sum(f["raa"] for f in fs) / n, 1)
    o["classical_only_pct"] = round(100 * sum(f["classical_only"] for f in fs) / n, 1)
    o["prem_inconsistent_pct"] = round(100 * sum(f["prem_inconsistent"] for f in fs) / n, 1)
    for key in (
        "goal_type",
        "first_kind",
        "top_rule",
        "n_prem",
        "fdepth",
        "box_depth",
        "n_atoms",
        "n_lines",
        "n_prem_or",
        "n_disj_prem",
        "n_redundant_prem",
    ):
        c = collections.Counter(f[key] for f in fs)
        o[key] = {
            str(k): round(100 * v / n, 1) for k, v in sorted(c.items(), key=lambda t: str(t[0]))
        }
    o["mean"] = {
        k: round(sum(f[k] for f in fs) / n, 2)
        for k in ("n_lines", "n_prem", "fdepth", "n_boxes", "prompt_tokens", "n_disj_prem")
    }
    return o


def by_feature(meas: list[tuple[dict[str, Any], dict[str, Any]]]) -> dict[str, Any]:
    """Descriptive: band share and mean per-attempt pass rate (k 32) by feature value over every
    measured candidate (a band-score-selected sample, so not the uniform task-1 analysis)."""
    out: dict[str, Any] = {}
    for key in (
        "goal_type",
        "first_kind",
        "top_rule",
        "n_prem",
        "has_ANDE",
        "has_IMPE",
        "has_NEGE",
        "has_BOTE",
        "has_ORE",
        "has_DN",
        "raa",
        "classical_only",
        "prem_inconsistent",
        "n_prem_or",
        "fdepth",
        "n_lines",
    ):
        g: dict[str, list[int]] = collections.defaultdict(list)
        for r, row in meas:
            g[str(r["feat"][key])].append(row["n_ok"])
        out[key] = {
            v: {
                "n": len(ns),
                "band_pct": round(100 * sum(LO <= x <= HI for x in ns) / len(ns), 1),
                "pass1_pct": round(100 * sum(ns) / (32 * len(ns)), 1),
                "unsolved_pct": round(100 * sum(x == 0 for x in ns) / len(ns), 1),
            }
            for v, ns in sorted(g.items())
            if len(ns) >= 30
        }
    return out


def fail_status(rs: list[dict[str, Any]]) -> dict[str, Any]:
    c: collections.Counter[str] = collections.Counter()
    for row in rs:
        c.update({k: v for k, v in row["status"].items() if k != "done"})
    n = sum(c.values())
    return {
        "failed_attempts": n,
        **{k: round(100 * v / max(1, n), 1) for k, v in c.most_common(10)},
    }


def t1_rows(work: str, name: str) -> list[dict[str, Any]]:
    p = f"{work}/passrate/{name}/rows.jsonl"
    return read_jsonl(p) if os.path.exists(p) else []


def cmd_meta(a: argparse.Namespace) -> None:
    from .features import features

    build = json.loads(Path(f"{a.work}/build.json").read_text())
    pool = read_jsonl(POOL)
    test = read_jsonl(TEST)
    recs = records(a.work)
    rows: dict[str, dict[str, Any]] = {}
    for d in build["runs"]:
        for row in read_jsonl(f"{d}/rows.jsonl"):
            rows.setdefault(row["name"], row)
    meas = [(recs[n], rows[n]) for n in rows if n in recs]
    by_cfg: dict[str, list[int]] = collections.defaultdict(list)
    for r, row in meas:
        by_cfg[r["cfg"]].append(row["n_ok"])
    pick = {
        os.path.basename(p): json.loads(Path(p).read_text())
        for p in sorted(glob.glob(f"{a.work}/*.pick.json"))
    }
    gen = {
        os.path.basename(p): {
            k: v for k, v in json.loads(Path(p).read_text()).items() if k != "workers"
        }
        for p in sorted(glob.glob(f"{a.work}/raw/*.summary.json"))
    }
    # features: K12 sample vs pool vs test
    rng = random.Random(0)
    k12 = read_jsonl(K12)
    rng.shuffle(k12)
    k12f = [features(r["prompt"], r["proof"]) for r in k12[:20000]]
    poolf = [features(r["prompt"], r["gen_proof"]) for r in pool]
    testf = [features(r["prompt"], r["reference_proof"]) for r in test]
    t1p = t1_rows(a.work, "t1_pool")
    t1t = t1_rows(a.work, "t1_test")
    t0_of = {r["name"]: r["t0_n_ok"] for r in pool} | {r["name"]: r["t0_n_ok"] for r in test}
    fa = json.loads(Path(f"{a.work}/feature_analysis_k16_d5.json").read_text())
    model = json.loads(Path(f"{a.work}/band_model.json").read_text())
    common = {
        "written_on": "2026-10-06",
        "written_by": "agent:claude",
        "purpose": "hard natural-deduction RL targets with reference proofs of 7-12 lines (the "
        "pretraining range) whose first move is hard to choose: T0_s0 solves 2-16 of 32 attempts",
        "recipe": {
            "generator": "Dan's fork gen.Gen (4dcd0e4f), unchanged, the K12 rule set; knobs per "
            "config in `generation` (pools/hard12/gen_steer.py CONFIGS). The strict_long stream is "
            "the previous pool agent's runs/pools/cands_clean.jsonl (gen_candidates.py, seeds "
            "52000 + 1000 i).",
            "shape_gate": "reference 7-12 lines, formula depth <= 5, prompt <= 156 tokens, "
            "reference states <= 386 tokens (K12's maximum), no premise whose negation is also a "
            "premise",
            "rejection_sampling": "band_score >= threshold (pools/hard12/score.py: logistic model "
            "of T0's k 16 band on the task-1 features; per-batch thresholds in `selection`)",
            "filter": "T0_s0 (runs/stage1/T0_s0/model.pt), k 32, plain arm, no think, T 0.8, line "
            "forcing, max_steps 96, max_action 512, Lean-judged (pools/passrate.py); keep 2 <= "
            "n_ok <= 16",
            "exclusion": "renaming class or premise-order key in any file of "
            "`dedupe.exclusion_files`, then within-set duplicates; pool and test split at random "
            "from one deduplicated set",
            "lean": (
                "every kept reference proof Lean-checked (code.thinking.lean.lean_check, 16 "
                "workers)"
            ),
            "seeds": {
                "split": build["seed"],
                "t1_sample": build["seed"] + 1,
                "passrate": (
                    "per run in `measurement.runs` (args.json seed; chunk c uses seed*1000+c)"
                ),
            },
        },
        "generation": gen,
        "selection": pick,
        "band_model": {"file": "band_model.json", "heldout": model["heldout"]},
        "feature_analysis": {
            "file": "feature_analysis_k16_d5.json",
            "data": fa["data"],
            "glm_coef": {k: v for k, v in fa["glm"]["coef"].items() if not k.startswith("len=")},
            "binned_len_adjusted": fa["binned"],
        },
        "measurement": {
            "runs": build["runs"],
            "measured": build["measured"],
            "in_band": build["in_band"],
            "t0_by_cfg": {c: summ(v) for c, v in sorted(by_cfg.items())},
            "t0_all_measured": summ([row["n_ok"] for _, row in meas]),
            "t0_by_feature_measured": by_feature(meas),
        },
        "yield": {
            "measured": build["measured"],
            "in_band": build["in_band"],
            "dedupe": build["dedupe"],
            "kept_after_lean": build["kept_after_lean"],
            "pool": build["n_pool"],
            "test": build["n_test"],
            "unused": build["unused_band_rows"],
        },
        "dedupe": {
            "excluded_by_file": build["excluded_by_file"],
            "exclusion_files": [m["file"] for m in build["exclusion_files"]],
            "pool_test_shared_keys": build["pool_test_shared_keys"],
            "pool_test_shared_pkeys": build["pool_test_shared_pkeys"],
        },
        "t0_pass32": {
            "pool": summ([r["t0_n_ok"] for r in pool]),
            "test": summ([r["t0_n_ok"] for r in test]),
        },
        "t0_failed_attempt_status_pct": {
            "pool": fail_status([rows[r["name"][4:]] for r in pool]),
            "test": fail_status([rows[r["name"][5:]] for r in test]),
        },
        "t1v1_pass32_reference_only": {
            "model": T1,
            "budget": 256,
            "arm": "plain, T 0.8, k 32",
            "pool_sample": summ([r["n_ok"] for r in t1p]),
            "test_sample": summ([r["n_ok"] for r in t1t]),
            "t0_on_same_pool_sample": summ([t0_of[r["name"]] for r in t1p if r["name"] in t0_of]),
            "t0_on_same_test_sample": summ([t0_of[r["name"]] for r in t1t if r["name"] in t0_of]),
            "t0_fresh_remeasure_pool_sample": summ(
                [r["n_ok"] for r in t1_rows(a.work, "t0re_pool")]
            ),
            "t0_fresh_remeasure_test_sample": summ(
                [r["n_ok"] for r in t1_rows(a.work, "t0re_test")]
            ),
            "note": "t0_on_same_*: the selection measurement (2-16 of 32 by construction). "
            "t0_fresh_remeasure_*: T0_s0 again on the same 1,000 targets with new seeds (120-127), "
            "the fair baseline for T1v1: selecting on a noisy measurement makes a fresh one "
            "regress toward the mean.",
        },
        "length_hist": {
            "pool": dict(sorted(collections.Counter(r["n_lines"] for r in pool).items())),
            "test": dict(sorted(collections.Counter(r["reference_lines"] for r in test).items())),
        },
        "cfg_mix": {
            "pool": dict(collections.Counter(r["cfg"] for r in pool).most_common()),
            "test": dict(collections.Counter(r["cfg"] for r in test).most_common()),
        },
        "rule_mix_vs_k12": {
            "k12_sample_20k": profile(k12f),
            "k12_7_12": profile([f for f in k12f if 7 <= f["n_lines"] <= 12]),
            "pool": profile(poolf),
            "test": profile(testf),
        },
    }
    pm = {"file": POOL, "n": len(pool), "role": "RL targets (rl_targets.jsonl fields)", **common}
    tm = {
        "file": TEST,
        "n": len(test),
        "role": "sealed test set (code.thinking.eval --sets format)",
        "keys_sidecar": TEST.replace(".jsonl", ".keys.jsonl"),
        **common,
    }
    Path(POOL.replace(".jsonl", ".meta.json")).write_text(json.dumps(pm, indent=1))
    Path(TEST.replace(".jsonl", ".meta.json")).write_text(json.dumps(tm, indent=1))
    os.makedirs(f"{CODE}/meta", exist_ok=True)
    shutil.copy(POOL.replace(".jsonl", ".meta.json"), f"{CODE}/meta/rl_hard12.meta.json")
    shutil.copy(TEST.replace(".jsonl", ".meta.json"), f"{CODE}/meta/hard12_test.meta.json")
    for f in (
        "feature_analysis_k16.json",
        "feature_analysis_k16_d5.json",
        "band_model.json",
        "build.json",
    ):
        shutil.copy(f"{a.work}/{f}", f"{CODE}/meta/{f}")
    print(
        json.dumps(
            {
                "pool": len(pool),
                "test": len(test),
                "t0_pool": common["t0_pass32"]["pool"]["mean_pass1_pct"],
                "t1_pool": common["t1v1_pass32_reference_only"]["pool_sample"].get(
                    "mean_pass1_pct"
                ),
            }
        )
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--work", required=True)
    b.add_argument("--index", required=True)
    b.add_argument("--n_pool", type=int, default=5000)
    b.add_argument("--n_test", type=int, default=2500)
    b.add_argument("--n_t1", type=int, default=1000)
    b.add_argument("--seed", type=int, default=12012)
    m = sub.add_parser("meta")
    m.add_argument("--work", required=True)
    a = ap.parse_args(argv)
    {"build": cmd_build, "meta": cmd_meta}[a.cmd](a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
