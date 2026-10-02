#!/usr/bin/env python3
"""GEAR-Fixed search controller for the RL hill-climb (two workers, one shared frontier).

Implements the deterministic genetic search policy of Jeddi et al. 2026, "GEAR: Genetic
AutoResearch for Agentic Code Evolution", sections 3.2-3.3, adapted to:

  * a higher-is-better integer metric (dev theorems solved out of 1610) instead of bpb,
  * LEAN meaning "within noise of the best with the least in-distribution damage"
    (highest `heldout_greedy`) instead of lowest VRAM,
  * two concurrent workers sharing one frontier (in-flight bookkeeping, file lock).

The controller never runs experiments and never touches a GPU. It only reads and writes
`<root>/gear_state.json` (authoritative) and `<root>/nodes/<id>/{rl.py,meta.json,summary.json}`
(meta.json is a mirror of the node record in the state file, rewritten on every save).

Stdlib only. Every command takes an exclusive fcntl lock on `<root>/gear_state.lock`, reads the
state, mutates it, and writes it atomically (tmp file + fsync + os.replace). All commands print
one JSON object to stdout.

CLI (global options `--root DIR` and `--dry-run` may appear anywhere):
  init      --seed-rl PATH [--desc TEXT] [--force]
  next      --worker W
  record    --worker W --node-rl PATH --summary PATH --desc TEXT --category CAT
            [--parents a,b --operator mutate|crossover] [--status ok|crash|timeout]
  frontier
  history
  release   --worker W
  (no command) with --dry-run: self-check that this controller can load the state and
            answer `frontier`; exits non-zero if the state fails validation.
`--dry-run` on any mutating command computes and prints the answer but writes nothing.
"""

import fcntl
import json
import math
import os
import re
import shutil
import sys
from datetime import datetime, timezone

CONTROLLER_VERSION = 1
STATE_VERSION = 1
DEFAULT_ROOT = os.environ.get("GEAR_ROOT", "/work")

# ---------------------------------------------------------------- policy constants
P = 4                      # frontier size
BETA = 0.5                 # UCB exploration weight (Eq. 1)
LAMBDA0 = 0.3              # novelty weight during the exploration phase
GAMMA0 = 0.3               # coverage weight during the exploration phase
EXPLORATION_PHASE = 8      # completed experiments after which lambda, gamma are halved
PROD_SCALE = 20.0          # productivity = mean child improvement / 20 theorems
RHO_LAST = -0.2            # recency: used as p1 at the previous step
RHO_PREV = -0.05           # recency: used as p1 two steps ago
ALPHA = 0.2                # Eq. 3 role-mismatch bonus
MU = 0.1                   # Eq. 3 per prior co-use penalty
NOV_WINDOW = 4             # R_t = p1s of the last 4 assignments
XO_WINDOW = 4              # crossover forced if < XO_MIN crossovers among last 4 experiments
XO_MIN = 1
MAX_CONSEC_P1 = 2          # a node that was p1 in each of the last 2 steps is skipped if possible
EPS_IMP = 5                # "strictly better" margin, theorems
EPS_TIE = 10               # "within noise of the best", theorems
DIVERSE_SLACK = 10         # DIVERSE may be up to 10 below the weakest elite
JACCARD_DIVERSE = 0.65     # DIVERSE needs Jaccard distance >= 0.65 to every elite
LEAN_HELDOUT_MARGIN = 0.01 # LEAN needs heldout_greedy higher by >= 0.01
FRESH_FROM = ("mutate",)   # promotions of these operators' children force a crossover with them

CATEGORIES = ("optimizer", "sampling", "data", "curriculum", "objective", "schedule", "other")
ROLES = ("BEST", "LEAN", "DIVERSE")
PROMOTING = ("FILL", "BEST", "REPLACE_WEAKEST", "LEAN", "DIVERSE")


class GearError(Exception):
    pass


# ---------------------------------------------------------------- small helpers
def now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def tokens(desc):
    return set(re.findall(r"[a-z0-9]+", (desc or "").lower()))


def jaccard_distance(a, b):
    ta, tb = tokens(a), tokens(b)
    union = ta | tb
    if not union:
        return 0.0
    return 1.0 - len(ta & tb) / len(union)


def pair_key(a, b):
    return "|".join(sorted((a, b)))


def node_id(i):
    return "n%03d" % i


def atomic_write_json(path, obj):
    tmp = "%s.tmp.%d" % (path, os.getpid())
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def r4(x):
    return None if x is None else round(x, 4)


# ---------------------------------------------------------------- state io
class Store:
    """Lock + load + save for one command invocation."""

    def __init__(self, root, shared=False):
        self.root = os.path.abspath(root)
        self.state_path = os.path.join(self.root, "gear_state.json")
        self.lock_path = os.path.join(self.root, "gear_state.lock")
        self.shared = shared
        self.fh = None

    def __enter__(self):
        os.makedirs(self.root, exist_ok=True)
        self.fh = open(self.lock_path, "a+")
        fcntl.flock(self.fh.fileno(), fcntl.LOCK_SH if self.shared else fcntl.LOCK_EX)
        return self

    def __exit__(self, *exc):
        fcntl.flock(self.fh.fileno(), fcntl.LOCK_UN)
        self.fh.close()

    def exists(self):
        return os.path.exists(self.state_path)

    def load(self):
        if not self.exists():
            raise GearError("no state at %s; run `init` first" % self.state_path)
        with open(self.state_path) as f:
            st = json.load(f)
        if st.get("version") != STATE_VERSION:
            raise GearError("state version %r != %r" % (st.get("version"), STATE_VERSION))
        return st

    def node_dir(self, nid):
        return os.path.join(self.root, "nodes", nid)

    def save(self, st):
        recompute_stats(st)
        st["controller_version"] = CONTROLLER_VERSION
        st["updated"] = now()
        for nid in st["order"]:
            d = self.node_dir(nid)
            os.makedirs(d, exist_ok=True)
            atomic_write_json(os.path.join(d, "meta.json"), st["nodes"][nid])
        atomic_write_json(self.state_path, st)


# ---------------------------------------------------------------- derived quantities
def recompute_stats(st):
    """n_children and mean_child_improvement for every node, from the node records.

    A child is credited to every parent it has (both parents of a crossover); the improvement
    is child.metric - parent.metric and only successful children of evaluated parents count.
    """
    nodes = st["nodes"]
    kids = {nid: [] for nid in nodes}
    for nid in st["order"]:
        for p in nodes[nid]["parents"]:
            if p in kids:
                kids[p].append(nid)
    for nid, ch in kids.items():
        n = nodes[nid]
        n["n_children"] = len(ch)
        deltas = [nodes[c]["metric"] - n["metric"] for c in ch
                  if nodes[c]["metric"] is not None and n["metric"] is not None]
        n["n_successful_children"] = len(deltas)
        n["mean_child_improvement"] = (sum(deltas) / len(deltas)) if deltas else None


def completed_count(st):
    return len(st["completed"])


def weights(st):
    if completed_count(st) >= EXPLORATION_PHASE:
        return LAMBDA0 / 2, GAMMA0 / 2
    return LAMBDA0, GAMMA0


def live_assignments(st):
    """Parent-using assignments (not baseline) that were not released, in issue order."""
    return [a for a in st["assignments"] if a["state"] != "released" and a["action"] != "baseline"]


def other_inflight(st, worker):
    return {w: a for w, a in st["inflight"].items() if w != worker}


def frontier_nodes(st):
    return [st["nodes"][i] for i in st["frontier"]]


def best_node(st):
    fr = frontier_nodes(st)
    if not fr:
        return None
    return sorted(fr, key=lambda n: (-n["metric"], n["id"]))[0]


def role_node(st, role):
    for n in frontier_nodes(st):
        if n["role"] == role:
            return n
    return None


def relabel(st):
    """BEST = highest metric; LEAN = within EPS_TIE of the best with heldout_greedy at least
    LEAN_HELDOUT_MARGIN above the BEST's (highest such); every other elite is DIVERSE."""
    for nid in st["order"]:
        if nid not in st["frontier"]:
            st["nodes"][nid]["role"] = None
    b = best_node(st)
    if b is None:
        return
    b["role"] = "BEST"
    lean_cands = []
    for n in frontier_nodes(st):
        if n is b or n["heldout_greedy"] is None:
            continue
        if n["metric"] < b["metric"] - EPS_TIE:
            continue
        if b["heldout_greedy"] is not None and n["heldout_greedy"] < b["heldout_greedy"] + LEAN_HELDOUT_MARGIN - 1e-12:
            continue
        lean_cands.append(n)
    lean = sorted(lean_cands, key=lambda n: (-n["heldout_greedy"], -n["metric"], n["id"]))[0] if lean_cands else None
    for n in frontier_nodes(st):
        if n is b:
            continue
        n["role"] = "LEAN" if n is lean else "DIVERSE"


def weakest(nodes):
    """Lowest metric; ties -> lower heldout_greedy, then older node."""
    return sorted(nodes, key=lambda n: (n["metric"],
                                        n["heldout_greedy"] if n["heldout_greedy"] is not None else -1.0,
                                        n["id"]))[0]


# ---------------------------------------------------------------- parent selection
def score_components(st, e, step, lam, gam):
    nodes = st["nodes"]
    fr = frontier_nodes(st)
    N = sum(n["n_uses"] for n in fr)
    g = e["mean_child_improvement"]
    prod = (g / PROD_SCALE) if g is not None else 0.0
    ucb = BETA * math.sqrt(math.log(N + 2) / (e["n_uses"] + 1))
    recent = [a["p1"] for a in live_assignments(st)[-NOV_WINDOW:]]
    R = [r for r in dict.fromkeys(recent) if r in nodes]
    others = [r for r in R if r != e["id"]]
    if not R:
        nov = 1.0
    elif not others:
        nov = 0.0  # e is the only recently used parent: not novel
    else:
        nov = min(jaccard_distance(e["description"], nodes[r]["description"]) for r in others)
    k = sum(1 for n in fr if n["id"] != e["id"] and n["role"] == e["role"] and n["category"] == e["category"])
    cov = 1.0 / math.sqrt(k + 1)
    rho = 0.0
    lu = e["last_used_step"]
    if lu is not None:
        if lu == step - 1:
            rho = RHO_LAST
        elif lu == step - 2:
            rho = RHO_PREV
    total = prod + ucb + lam * nov + gam * cov + rho
    return {"total": r4(total), "productivity": r4(prod), "ucb": r4(ucb), "novelty": r4(nov),
            "coverage": r4(cov), "recency": rho, "n_uses": e["n_uses"], "N": N}


def comp_score(st, p1, e):
    u = st["co_use"].get(pair_key(p1["id"], e["id"]), 0)
    j = jaccard_distance(p1["description"], e["description"])
    return {"total": r4(j + ALPHA * (1 if e["role"] != p1["role"] else 0) - MU * u),
            "jaccard": r4(j), "role_mismatch": e["role"] != p1["role"], "co_use": u}


def p1_candidates(st, worker, pool, notes):
    """Remove nodes that another worker holds as p1 and nodes over the consecutive-use cap,
    each only if something else remains."""
    held = {a["p1"] for a in other_inflight(st, worker).values()}
    c = [n for n in pool if n["id"] not in held]
    if c and len(c) < len(pool):
        notes.append("excluded %s: in flight as p1 for another worker" % sorted(held & {n["id"] for n in pool}))
    if not c:
        c = list(pool)
        if held:
            notes.append("every candidate is in flight elsewhere; sharing p1")
    last = [a["p1"] for a in live_assignments(st)[-MAX_CONSEC_P1:]]
    if len(last) == MAX_CONSEC_P1 and len(set(last)) == 1:
        c2 = [n for n in c if n["id"] != last[0]]
        if c2 and len(c2) < len(c):
            notes.append("excluded %s: p1 in each of the last %d steps" % (last[0], MAX_CONSEC_P1))
            c = c2
    return c


def pick_p1(st, worker, pool, step, notes):
    lam, gam = weights(st)
    cands = p1_candidates(st, worker, pool, notes)
    scored = {n["id"]: score_components(st, n, step, lam, gam) for n in cands}
    best = sorted(cands, key=lambda n: (-scored[n["id"]]["total"], n["id"]))[0]
    return best, scored


def pick_p2(st, p1, pool):
    cands = [n for n in pool if n["id"] != p1["id"]]
    if not cands:
        return None, {}
    scored = {n["id"]: comp_score(st, p1, n) for n in cands}
    best = sorted(cands, key=lambda n: (-scored[n["id"]]["total"], -n["metric"], n["id"]))[0]
    return best, scored


def crossover_due(st, worker):
    """Operators of the last XO_WINDOW experiments: completed ones, then other workers'
    in-flight ones (so two workers do not both skip the due crossover)."""
    ops = [c["operator"] for c in st["completed"]]
    ops += [a["action"] for a in sorted(other_inflight(st, worker).values(), key=lambda a: a["step"])]
    window = ops[-XO_WINDOW:]
    return sum(1 for o in window if o == "crossover") < XO_MIN, window


# ---------------------------------------------------------------- commands
def cmd_init(store, args):
    if store.exists() and not args.get("force"):
        raise GearError("state already exists at %s (use --force to overwrite)" % store.state_path)
    seed = args["seed_rl"]
    if not os.path.isfile(seed):
        raise GearError("seed rl.py not found: %s" % seed)
    if store.exists() and args.get("force") and not args.get("dry_run"):
        shutil.rmtree(os.path.join(store.root, "nodes"), ignore_errors=True)
    nid = node_id(0)
    meta = new_meta(nid, [], "seed", args.get("desc") or "seed baseline rl.py unmodified", "other", None)
    st = {"version": STATE_VERSION, "controller_version": CONTROLLER_VERSION, "created": now(),
          "nodes": {nid: meta}, "order": [nid], "frontier": [], "assign_count": 0,
          "assignments": [], "inflight": {}, "completed": [], "co_use": {}, "fresh_elites": [],
          "next_id": 1}
    if not args.get("dry_run"):
        d = store.node_dir(nid)
        os.makedirs(d, exist_ok=True)
        shutil.copyfile(seed, os.path.join(d, "rl.py"))
        store.save(st)
    return {"ok": True, "root": store.root, "node": nid, "status": "pending", "frontier": []}


def new_meta(nid, parents, operator, desc, category, worker):
    return {"id": nid, "parents": list(parents), "operator": operator, "description": desc,
            "category": category, "worker": worker, "created": now(), "metric": None,
            "by_len": None, "heldout_greedy": None, "status": "pending", "run_dir": None,
            "n_children": 0, "n_successful_children": 0, "mean_child_improvement": None,
            "n_uses": 0, "last_used_step": None, "role": None, "decision": None,
            "evicted_by": None}


def node_brief(store, n):
    if n is None:
        return None
    return {"id": n["id"], "rl": os.path.join(store.node_dir(n["id"]), "rl.py"),
            "description": n["description"], "category": n["category"], "role": n["role"],
            "metric": n["metric"], "by_len": n["by_len"], "heldout_greedy": n["heldout_greedy"]}


def frontier_rows(st):
    rows = []
    for n in sorted(frontier_nodes(st), key=lambda n: (ROLES.index(n["role"]) if n["role"] in ROLES else 9, -n["metric"], n["id"])):
        rows.append({"id": n["id"], "role": n["role"], "metric": n["metric"], "by_len": n["by_len"],
                     "heldout_greedy": n["heldout_greedy"], "n_children": n["n_children"],
                     "n_uses": n["n_uses"], "mean_child_improvement": r4(n["mean_child_improvement"]),
                     "last_used_step": n["last_used_step"], "category": n["category"],
                     "description": n["description"]})
    return rows


def frontier_table(st):
    lines = ["%-5s %-8s %6s %-16s %7s %4s %4s %7s %4s %-10s %s" % (
        "id", "role", "metric", "by_len", "heldout", "kids", "uses", "meanimp", "last", "category", "description")]
    for r in frontier_rows(st):
        bl = r["by_len"]
        bls = "/".join(str(bl.get(k, "?")) for k in sorted(bl)) if isinstance(bl, dict) else "-"
        lines.append("%-5s %-8s %6s %-16s %7s %4d %4d %7s %4s %-10s %s" % (
            r["id"], r["role"], r["metric"], bls,
            "-" if r["heldout_greedy"] is None else "%.3f" % r["heldout_greedy"],
            r["n_children"], r["n_uses"],
            "-" if r["mean_child_improvement"] is None else "%+.1f" % r["mean_child_improvement"],
            "-" if r["last_used_step"] is None else r["last_used_step"], r["category"], r["description"]))
    return lines


def assignment_output(store, st, a, notes, extra=None):
    nodes = st["nodes"]
    p1 = nodes[a["p1"]]
    p2 = nodes[a["p2"]] if a.get("p2") else None
    lam, gam = weights(st)
    out = {"action": a["action"], "worker": a["worker"], "step": a["step"],
           "parent1": p1["id"], "parent1_rl": os.path.join(store.node_dir(p1["id"]), "rl.py"),
           "parent1_desc": p1["description"], "parent1_category": p1["category"],
           "parent1_role": p1["role"], "parent1_metric": p1["metric"],
           "parent2": p2["id"] if p2 else None,
           "parent2_rl": os.path.join(store.node_dir(p2["id"]), "rl.py") if p2 else None,
           "parent2_desc": p2["description"] if p2 else None,
           "parent2_category": p2["category"] if p2 else None,
           "parent2_role": p2["role"] if p2 else None,
           "parent2_metric": p2["metric"] if p2 else None,
           "completed_experiments": completed_count(st), "lambda": lam, "gamma": gam,
           "frontier": frontier_rows(st), "notes": notes}
    if extra:
        out.update(extra)
    return out


def cmd_next(store, args):
    st = store.load()
    w = args["worker"]
    if w in st["inflight"]:
        a = st["inflight"][w]
        return assignment_output(store, st, a, ["already assigned at step %d; returning the same assignment "
                                                "(use `release` to drop it)" % a["step"]])
    nodes = st["nodes"]
    step = st["assign_count"] + 1
    notes = []
    n0 = nodes[node_id(0)]
    inflight_all = st["inflight"]
    baseline_running = any(a["action"] == "baseline" for a in inflight_all.values())
    extra = {}
    evaluated = [n for n in (nodes[i] for i in st["order"]) if n["metric"] is not None]
    pop_fresh = None

    if n0["status"] == "pending" and not baseline_running:
        a = {"action": "baseline", "p1": n0["id"], "p2": None}
        notes.append("baseline: run the seed rl.py unmodified")
    elif not evaluated:
        a = {"action": "mutate", "p1": n0["id"], "p2": None}
        if baseline_running:
            notes.append("baseline still running elsewhere; mutating the unevaluated seed")
        else:
            notes.append("seed has status %s and nothing is evaluated; the mutation should first make "
                         "the seed run" % n0["status"])
    elif len(st["frontier"]) < P:
        pool = [nodes[i] for i in st["frontier"]] or evaluated
        cands = p1_candidates(st, w, pool, notes)
        p1 = sorted(cands, key=lambda n: (-n["metric"], n["id"]))[0]
        a = {"action": "mutate", "p1": p1["id"], "p2": None}
        notes.append("filling the frontier (%d/%d): mutate the best available evaluated node" % (len(st["frontier"]), P))
    else:
        fr = frontier_nodes(st)
        due, window = crossover_due(st, w)
        fresh = [f for f in st["fresh_elites"] if f in st["frontier"]]
        if len(fr) >= 2 and fresh:
            f = nodes[fresh[0]]
            pop_fresh = f["id"]
            held = {x["p1"] for x in other_inflight(st, w).values()}
            if f["role"] == "BEST" and f["id"] not in held:
                p1 = f
                p1_scores = {}
            else:
                pool = [n for n in fr if n["id"] != f["id"]]
                p1, p1_scores = pick_p1(st, w, pool, step, notes)
            if p1["id"] == f["id"]:
                p2, p2_scores = pick_p2(st, p1, fr)
            else:
                p2, p2_scores = f, {f["id"]: comp_score(st, p1, f)}
            a = {"action": "crossover", "p1": p1["id"], "p2": p2["id"]}
            notes.append("crossover forced: fresh elite %s must be recombined" % f["id"])
            extra = {"p1_scores": p1_scores, "p2_scores": p2_scores}
        elif len(fr) >= 2 and due:
            p1, p1_scores = pick_p1(st, w, fr, step, notes)
            p2, p2_scores = pick_p2(st, p1, fr)
            a = {"action": "crossover", "p1": p1["id"], "p2": p2["id"]}
            notes.append("crossover forced: %d crossover(s) in the last %d experiments %s" % (
                sum(1 for o in window if o == "crossover"), XO_WINDOW, window))
            extra = {"p1_scores": p1_scores, "p2_scores": p2_scores}
        else:
            p1, p1_scores = pick_p1(st, w, fr, step, notes)
            a = {"action": "mutate", "p1": p1["id"], "p2": None}
            notes.append("mutate the highest Eq. 1 score")
            extra = {"p1_scores": p1_scores}

    a.update({"worker": w, "step": step, "state": "inflight", "issued": now(),
              "undo": {"last_used_step": nodes[a["p1"]]["last_used_step"], "fresh_popped": pop_fresh}})
    st["assign_count"] = step
    if a["action"] != "baseline":  # running the seed is not a use of it as a parent
        nodes[a["p1"]]["n_uses"] += 1
        nodes[a["p1"]]["last_used_step"] = step
    if a["p2"]:
        nodes[a["p2"]]["n_uses"] += 1
        k = pair_key(a["p1"], a["p2"])
        st["co_use"][k] = st["co_use"].get(k, 0) + 1
    if pop_fresh:
        st["fresh_elites"].remove(pop_fresh)
    st["assignments"].append(a)
    st["inflight"][w] = a
    if not args.get("dry_run"):
        store.save(st)
    else:
        notes.append("dry run: nothing written")
    return assignment_output(store, st, a, notes, extra)


def cmd_release(store, args):
    st = store.load()
    w = args["worker"]
    a = st["inflight"].pop(w, None)
    if a is None:
        return {"ok": True, "released": None, "note": "worker %s had no in-flight assignment" % w}
    nodes = st["nodes"]
    for p in (a["p1"], a["p2"]):
        if p and a["action"] != "baseline":
            nodes[p]["n_uses"] = max(0, nodes[p]["n_uses"] - 1)
    if a["p2"]:
        k = pair_key(a["p1"], a["p2"])
        st["co_use"][k] = max(0, st["co_use"].get(k, 0) - 1)
        if not st["co_use"][k]:
            del st["co_use"][k]
    if nodes[a["p1"]]["last_used_step"] == a["step"]:
        nodes[a["p1"]]["last_used_step"] = a["undo"]["last_used_step"]
    fp = a["undo"].get("fresh_popped")
    if fp and fp in st["frontier"] and fp not in st["fresh_elites"]:
        st["fresh_elites"].insert(0, fp)
    for x in st["assignments"]:
        if x["step"] == a["step"]:
            x["state"] = "released"
    if not args.get("dry_run"):
        store.save(st)
    return {"ok": True, "released": {"worker": w, "step": a["step"], "action": a["action"],
                                     "parent1": a["p1"], "parent2": a["p2"]}}


def read_summary(path):
    if not path or not os.path.isfile(path):
        return None
    with open(path) as f:
        return json.load(f)


def extract_heldout(s):
    v = s.get("heldout_greedy")
    if v is None and isinstance(s.get("rl_stats"), dict):
        v = s["rl_stats"].get("heldout_greedy")
    if isinstance(v, dict):
        v = v.get("rate")
    return float(v) if isinstance(v, (int, float)) else None


def promote(st, child):
    """Returns (decision, evicted_id)."""
    fr = frontier_nodes(st)
    m, hg = child["metric"], child["heldout_greedy"]
    if len(fr) < P:
        st["frontier"].append(child["id"])
        return "FILL", None
    b = best_node(st)
    non_best = [n for n in fr if n is not b]
    w = weakest(non_best)
    lean = role_node(st, "LEAN")
    ref = lean if lean is not None else b

    def replace(out):
        st["frontier"][st["frontier"].index(out["id"])] = child["id"]
        out["role"] = None
        out["evicted_by"] = child["id"]
        if out["id"] in st["fresh_elites"]:
            st["fresh_elites"].remove(out["id"])
        return out["id"]

    if m > b["metric"] + EPS_IMP:
        return "BEST", replace(w)
    if m > w["metric"] + EPS_IMP:
        return "REPLACE_WEAKEST", replace(w)
    if (m >= b["metric"] - EPS_TIE and hg is not None
            and (ref["heldout_greedy"] is None or hg >= ref["heldout_greedy"] + LEAN_HELDOUT_MARGIN - 1e-12)):
        return "LEAN", replace(lean if lean is not None else w)
    if m >= w["metric"] - DIVERSE_SLACK and all(
            jaccard_distance(child["description"], n["description"]) >= JACCARD_DIVERSE for n in fr):
        plain = [n for n in non_best if n is not lean]
        return "DIVERSE", replace(weakest(plain) if plain else w)
    return "DISCARD", None


def cmd_record(store, args):
    st = store.load()
    nodes = st["nodes"]
    w = args["worker"]
    notes = []
    a = st["inflight"].get(w)
    parents = [p for p in (args.get("parents") or "").split(",") if p]
    operator = args.get("operator")
    if parents:
        for p in parents:
            if p not in nodes:
                raise GearError("unknown parent %s" % p)
        operator = operator or ("crossover" if len(parents) == 2 else "mutate")
        if a is not None:
            notes.append("--parents given; overriding in-flight assignment %s" % [a["p1"], a["p2"]])
    elif a is not None:
        parents = [a["p1"]] + ([a["p2"]] if a["p2"] else [])
        operator = a["action"]
    else:
        raise GearError("worker %s has no in-flight assignment; pass --parents and --operator" % w)
    if operator not in ("baseline", "mutate", "crossover"):
        raise GearError("bad operator %r" % operator)
    if operator == "crossover" and len(parents) != 2:
        raise GearError("crossover needs two parents")
    if operator == "mutate" and len(parents) != 1:
        raise GearError("mutate needs one parent")

    summary = read_summary(args.get("summary"))
    status = args.get("status") or (summary or {}).get("status") or ("ok" if summary else "crash")
    metric = (summary or {}).get("metric")
    if status == "ok" and not isinstance(metric, (int, float)):
        status = "crash"
        notes.append("summary has no numeric metric; recorded as crash")
    if status != "ok":
        metric = None
    if summary is None:
        notes.append("no summary.json at %s" % args.get("summary"))

    if operator == "baseline":
        child = nodes[node_id(0)]
        if args.get("node_rl") and os.path.isfile(args["node_rl"]):
            with open(args["node_rl"], "rb") as f1, open(os.path.join(store.node_dir(child["id"]), "rl.py"), "rb") as f2:
                if f1.read() != f2.read():
                    notes.append("WARNING: baseline rl.py differs from the seed; storing the run's copy")
                    if not args.get("dry_run"):
                        shutil.copyfile(args["node_rl"], os.path.join(store.node_dir(child["id"]), "rl.py"))
        child["worker"] = w
    else:
        desc = (args.get("desc") or "").strip()
        if not desc:
            raise GearError("--desc is required")
        cat = args.get("category")
        if cat not in CATEGORIES:
            raise GearError("--category must be one of %s" % (CATEGORIES,))
        if not args.get("node_rl") or not os.path.isfile(args["node_rl"]):
            raise GearError("--node-rl %r not found" % args.get("node_rl"))
        nid = node_id(st["next_id"])
        st["next_id"] += 1
        child = new_meta(nid, parents, operator, desc, cat, w)
        nodes[nid] = child
        st["order"].append(nid)
        if not args.get("dry_run"):
            d = store.node_dir(nid)
            os.makedirs(d, exist_ok=True)
            shutil.copyfile(args["node_rl"], os.path.join(d, "rl.py"))

    child["status"] = "evaluated" if status == "ok" else status
    child["metric"] = metric
    child["by_len"] = (summary or {}).get("by_len")
    child["heldout_greedy"] = extract_heldout(summary or {})
    child["run_dir"] = os.path.dirname(os.path.abspath(args["summary"])) if args.get("summary") else None
    if summary is not None and not args.get("dry_run"):
        shutil.copyfile(args["summary"], os.path.join(store.node_dir(child["id"]), "summary.json"))

    if metric is None:
        decision, evicted = status.upper(), None
    else:
        decision, evicted = promote(st, child)
        relabel(st)
        if decision in PROMOTING and decision != "FILL" and operator in FRESH_FROM:
            st["fresh_elites"].append(child["id"])
    child["decision"] = decision

    if operator == "baseline":
        parents = []
    p1m = nodes[parents[0]]["metric"] if parents else None
    delta = (metric - p1m) if (metric is not None and p1m is not None) else None
    entry = {"n": completed_count(st) + 1, "node": child["id"], "worker": w, "parents": parents,
             "operator": operator, "metric": metric, "parent1_metric": p1m, "delta": delta,
             "decision": decision, "evicted": evicted, "role": child["role"],
             "assignment_step": a["step"] if a else None, "time": now()}
    st["completed"].append(entry)
    if a is not None:
        st["inflight"].pop(w, None)
        for x in st["assignments"]:
            if x["step"] == a["step"]:
                x["state"] = "done"
                x["node"] = child["id"]
    if not args.get("dry_run"):
        store.save(st)
    else:
        recompute_stats(st)
        notes.append("dry run: nothing written")
    return {"ok": True, "node": child["id"], "decision": decision, "role": child["role"],
            "evicted": evicted, "metric": metric, "parent1_metric": p1m, "delta": delta,
            "status": child["status"], "operator": operator, "parents": parents,
            "completed_experiments": completed_count(st), "frontier": frontier_rows(st),
            "notes": notes}


def cmd_frontier(store, args):
    st = store.load()
    return {"frontier": frontier_rows(st), "table": frontier_table(st),
            "completed_experiments": completed_count(st), "step": st["assign_count"],
            "inflight": {w: {"action": a["action"], "parent1": a["p1"], "parent2": a["p2"], "step": a["step"]}
                         for w, a in st["inflight"].items()},
            "fresh_elites": st["fresh_elites"], "lambda": weights(st)[0], "gamma": weights(st)[1]}


def cmd_history(store, args):
    st = store.load()
    by_node = {c["node"]: c for c in st["completed"]}
    rows = []
    for nid in st["order"]:
        n = st["nodes"][nid]
        c = by_node.get(nid, {})
        rows.append({"id": nid, "parents": n["parents"], "operator": n["operator"], "worker": n["worker"],
                     "category": n["category"], "status": n["status"], "metric": n["metric"],
                     "delta": c.get("delta"), "decision": n["decision"], "role": n["role"],
                     "evicted_by": n["evicted_by"], "in_frontier": nid in st["frontier"],
                     "description": n["description"]})
    return {"history": rows, "completed": st["completed"]}


def validate(store, st):
    problems = []
    for key in ("nodes", "order", "frontier", "assign_count", "assignments", "inflight", "completed",
                "co_use", "fresh_elites", "next_id"):
        if key not in st:
            problems.append("missing key %s" % key)
    if problems:
        return problems
    if set(st["order"]) != set(st["nodes"]):
        problems.append("order and nodes disagree")
    if len(st["frontier"]) > P:
        problems.append("frontier larger than P")
    for nid in st["frontier"]:
        if nid not in st["nodes"]:
            problems.append("frontier id %s unknown" % nid)
        elif st["nodes"][nid]["metric"] is None:
            problems.append("frontier node %s has no metric" % nid)
    for nid in st["order"]:
        if not os.path.isfile(os.path.join(store.node_dir(nid), "rl.py")):
            problems.append("node %s has no rl.py" % nid)
    for w, a in st["inflight"].items():
        for p in (a["p1"], a["p2"]):
            if p and p not in st["nodes"]:
                problems.append("in-flight parent %s of %s unknown" % (p, w))
    return problems


def cmd_selfcheck(store, args):
    st = store.load()
    problems = validate(store, st)
    out = cmd_frontier(store, args)
    out.update({"ok": not problems, "problems": problems, "controller_version": CONTROLLER_VERSION,
                "state_controller_version": st.get("controller_version")})
    return out


# ---------------------------------------------------------------- argument parsing
def parse(argv):
    argv = list(argv)
    opts = {"root": DEFAULT_ROOT, "dry_run": False}
    rest = []
    i = 0
    while i < len(argv):
        t = argv[i]
        if t == "--root":
            opts["root"] = argv[i + 1]
            i += 2
            continue
        if t.startswith("--root="):
            opts["root"] = t.split("=", 1)[1]
        elif t == "--dry-run":
            opts["dry_run"] = True
        else:
            rest.append(t)
        i += 1
    cmd = rest[0] if rest and not rest[0].startswith("--") else None
    if cmd:
        rest = rest[1:]
    flags = {}
    i = 0
    boolean = {"--force"}
    while i < len(rest):
        t = rest[i]
        if not t.startswith("--"):
            raise GearError("unexpected argument %r" % t)
        if "=" in t:
            k, v = t[2:].split("=", 1)
            flags[k.replace("-", "_")] = v
            i += 1
        elif t in boolean:
            flags[t[2:].replace("-", "_")] = True
            i += 1
        else:
            if i + 1 >= len(rest):
                raise GearError("missing value for %s" % t)
            flags[t[2:].replace("-", "_")] = rest[i + 1]
            i += 2
    opts.update(flags)
    return cmd, opts


REQUIRED = {"init": ["seed_rl"], "next": ["worker"], "record": ["worker"], "release": ["worker"],
            "frontier": [], "history": [], None: []}
COMMANDS = {"init": cmd_init, "next": cmd_next, "record": cmd_record, "release": cmd_release,
            "frontier": cmd_frontier, "history": cmd_history}


def run(argv):
    cmd, opts = parse(argv)
    if cmd not in REQUIRED:
        raise GearError("unknown command %r; expected one of %s" % (cmd, sorted(COMMANDS)))
    for r in REQUIRED[cmd]:
        if not opts.get(r):
            raise GearError("%s requires --%s" % (cmd, r.replace("_", "-")))
    if cmd is None:
        if not opts["dry_run"]:
            raise GearError("no command given (use --dry-run for the self-check)")
        with Store(opts["root"], shared=True) as store:
            return cmd_selfcheck(store, opts)
    read_only = cmd in ("frontier", "history")
    with Store(opts["root"], shared=read_only) as store:
        return COMMANDS[cmd](store, opts)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    try:
        out = run(argv)
    except GearError as e:
        print(json.dumps({"ok": False, "error": str(e)}))
        return 2
    print(json.dumps(out, indent=1, sort_keys=False))
    if isinstance(out, dict) and out.get("ok") is False:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
