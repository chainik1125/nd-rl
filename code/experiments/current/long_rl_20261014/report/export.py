"""Per-round RL curves from the driver logs as JSON for the report page:
python report/export.py <out.json> <run log dir>...   (each dir's name, v2, v3, ..., tags its runs)

Once a run finishes, its found_targets.jsonl (each solved target's proofs, with the oracle call that first
found each) adds a solves-by-atom-count timeline. Each passk/results/<tag>.json present adds a textbook72 pass@k curve,
and each edl/results/<tag>.json the checkpoint's excess description length per body-line band."""
import collections
import json
import math
import random
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from passk.backfill import LEGACY  # noqa: E402
from plot_rl import parse  # noqa: E402

HERE = Path(__file__).resolve().parents[1]
ORACLE_CLOCK = re.compile(r"oracle (\d+): .*RL clock (\d+)/")
RESUMED = re.compile(r"prior RL clock (\d+)s")


def atoms(s):
    return set(re.findall(r"\b[PQRSTUWXYZ]\b", s))


def n_atoms(prompt):
    """Distinct atoms in the sequent (premises and goal), for the pool's per-count denominators."""
    return len(atoms(prompt.split("PRF")[0]))


def n_atoms_used(prompt, proof):
    """Distinct atoms the proof works with: the goal's plus those in its non-premise lines (premise lines only restate
    the sequent, and pretraining's distractor premises add atoms no proof uses)."""
    return len(atoms(prompt.split("SEQ")[1]) | atoms(" ".join(x for x in proof.split(";") if ": PR" not in x)))


POOL = collections.Counter(n_atoms(json.loads(line)["prompt"]) for line in open(HERE / "data" / "rl.jsonl"))


def atom_timeline(text, found_path):
    """Per oracle call: RL clock (h), max atoms used by a solved target's first proof so far, and solved share of the
    pool per sequent atom count."""
    first = {}
    for line in open(found_path):
        r = json.loads(line)
        if r["name"] not in first or r["round"] < first[r["name"]][0]:
            first[r["name"]] = r["round"], n_atoms(r["prompt"]), n_atoms_used(r["prompt"], r["proof"])
    by_call = collections.defaultdict(list)
    for call, k, used in first.values():
        by_call[call].append((k, used))
    solved, top, points = collections.Counter(), 0, []
    offset = int(m[1]) if (m := RESUMED.search(text)) else 0
    calls = list(ORACLE_CLOCK.finditer(text))
    for call in [c for c in by_call if calls and c < int(calls[0][1])]:  # a resumed run's targets solved before it
        for k, used in by_call[call]:
            solved[k] += 1
            top = max(top, used)
    for m in calls:
        for k, used in by_call.get(int(m[1]), []):
            solved[k] += 1
            top = max(top, used)
        points.append({"clock": (offset + int(m[2])) / 3600, "max_atoms": top,
                       "share": {k: solved[k] / POOL[k] for k in sorted(POOL)}})
    return points


KS = [2**i for i in range(9)]  # 1..256, as passk/plot.py


def passk(n, c, k):
    """Unbiased pass@k for one theorem (n attempts, c correct): 1 - prod_{i=n-c+1}^{n} (1 - k / i)."""
    return 1.0 if n - c < k else 1.0 - math.prod(1.0 - k / i for i in range(n - c + 1, n + 1))


def textbook(path):
    """One passk/results file's textbook72 rows -> its run and stage, solved counts per split and the pass@k curve with
    95% theorem-bootstrap bands (same estimator and resampling as passk/plot.py, 1,000 resamples)."""
    res = json.loads(path.read_text())
    rows = res["sets"]["textbook72"]["rows"]
    per = [[passk(r["n_tried"], r["n_ok"], k) for k in KS] for r in rows]
    rng = random.Random(0)
    boot = [[sum(col) / len(col) for col in zip(*rng.choices(per, k=len(per)))] for _ in range(1000)]
    by_k = [sorted(v) for v in zip(*boot)]  # per k, the 1,000 resampled means
    tag = LEGACY.get(res["tag"], res["tag"])  # <version>_<arm>_s<seed>_<stage>
    version, arm, seed, stage = tag.split("_")
    return {"tag": tag, "run": f"{version}_{arm}_{seed}", "version": version, "arm": arm, "seed": seed[1:],
            "stage": stage, "ckpt": res["ckpt"], "k": res["k"],
            "n": {s: sum(r["split"] == s for r in rows) for s in ("dev", "train")},
            "solved": {s: sum(r["solved"] and r["split"] == s for r in rows) for s in ("dev", "train")},
            "ks": KS, "mean": [sum(col) / len(col) for col in zip(*per)],
            "lo": [v[25] for v in by_k], "hi": [v[974] for v in by_k]}


def edl(path):
    """One edl/results file -> per band: EDL at the stable learning-rate multiple with the smallest mean online code length,
    plus ln(number of multiples tried) for that choice (Leon's convention, after the paper's hyperparameter cost), the
    test loss before and after, and the online loss per step (mean over data orders, for the area picture)."""
    res = json.loads(path.read_text())
    version, arm, seed, stage = res["tag"].split("_")
    bands = {}
    for name, b in res["bands"].items():
        n = b["n"]
        by_mult = collections.defaultdict(list)
        for r in b["runs"]:
            by_mult[r["lr_mult"]].append(r)
        # A rate whose held-out loss rose > 5% over the second half ended on a destabilised theta_n (common at 1x
        # and 3x, never at 0.1x or 0.3x), which made EDL spuriously negative; choose among the stable rates only.
        stable = [m for m, rs in by_mult.items() if all(r["l_test"][str(n)] <= 1.05 * r["l_test"][str(n // 2)] for r in rs)]
        mult = min(stable or by_mult, key=lambda m: sum(sum(r["online_nats"]) for r in by_mult[m]) / len(by_mult[m]))
        best = by_mult[mult]
        checks = sorted(int(k) for k in best[0]["l_test"])
        bands[name] = {"n": n, "proof_tokens": b["proof_tokens"], "lr_mult": mult, "l_test_0": b["l_test_0"],
                       "edl": sum(r["edl"] for r in best) / len(best) + math.log(len(by_mult)),
                       "edl_orders": [r["edl"] for r in best],
                       "online": [sum(col) / len(col) for col in zip(*(r["online_nats"] for r in best))],
                       "l_test": [[0, b["l_test_0"]]] + [[c, sum(r["l_test"][str(c)] for r in best) / len(best)] for c in checks]}
    return {"tag": res["tag"], "run": f"{version}_{arm}_{seed}", "version": version, "arm": arm, "seed": seed[1:],
            "stage": stage, "bands": bands}


runs = {}
for log_dir in map(Path, sys.argv[2:]):
    for log in sorted(log_dir.glob("driver_*.log")):
        text = log.read_text(errors="replace")
        arm, seed = log.stem.removeprefix("driver_").rsplit("_s", 1)
        # pulled with the run's output at the end, or early (fetch_found.py) as soon as the RL stage writes it
        found = [*log_dir.glob(f"*{arm}_s{seed}/found_targets.jsonl"), *log_dir.glob(f"early/{arm}_s{seed}.found_targets.jsonl")]
        runs[f"{log_dir.name}/{arm}_s{seed}"] = {
            "atoms": atom_timeline(text, found[0]) if found else None,
            "run": log_dir.name, "arm": arm, "seed": seed, "rounds": parse(log),
            # a pod state file exists while launch.py runs; neither it nor "terminated" means the run was stopped
            "status": "finished" if "terminated" in text
            else "running" if (log_dir / f"pod_{arm}_s{seed}.json").exists() else "stopped",
            "summary": [l for l in text.splitlines() if " eval " in l or "SUMMARY" in l],
        }
for r in runs.values():  # JSON has no NaN (a probe without gold proofs logs nan)
    r["rounds"] = [{k: None if v != v else v for k, v in p.items()} for p in r["rounds"]]
stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
books = [textbook(p) for p in sorted((HERE / "passk" / "results").glob("*.json"))]
edls = [edl(p) for p in sorted((HERE / "edl" / "results").glob("*.json"))]
Path(sys.argv[1]).write_text(json.dumps({"exported_utc": stamp, "pool_by_atoms": POOL, "runs": runs, "textbook": books,
                                         "edl": edls}))
