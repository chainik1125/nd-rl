#!/usr/bin/env python3
"""GEAR search controller for the RL hill-climb, run 2 (seven workers + one auditor, one shared frontier).

Implements the deterministic genetic search policy of Jeddi et al. 2026, "GEAR: Genetic
AutoResearch for Agentic Code Evolution", sections 3.2-3.3, adapted to:

  * a higher-is-better integer metric (dev L8-11 theorems solved at pass@64) instead of bpb,
  * LEAN meaning "within noise of the best with the least in-distribution damage"
    (highest `heldout_greedy`) instead of lowest VRAM,
  * seven concurrent workers sharing one frontier (in-flight bookkeeping, file lock),
  * a two-seed rule for BEST, run by an `auditor` pseudo-worker (v9, below).

The controller never runs experiments and never touches a GPU. It only reads and writes
`<root>/gear_state.json` (authoritative) and `<root>/nodes/<id>/{rl.py,meta.json,summary.json,
replicate_summary.json}` (meta.json is a mirror of the node record in the state file).

Stdlib only. Every command takes an exclusive fcntl lock on `<root>/gear_state.lock`, reads the
state, mutates it, and writes it atomically (tmp file + fsync + os.replace). All commands print
one JSON object to stdout.

CLI (global options `--root DIR` and `--dry-run` may appear anywhere):
  init      --seed-rl PATH [--desc TEXT] [--force]
  next      --worker W                (W = auditor: the oldest PENDING_BEST replicate job, or "idle")
  record    --worker W --node-rl PATH --summary PATH --desc TEXT --category CAT --source URL|"own idea"
            [--parents a,b --operator mutate|crossover] [--status ok|crash|timeout]
  replicate --node ID --summary PATH  (the auditor's second-seed run of a PENDING_BEST node)
  frontier
  history
  release   --worker W                (also --worker auditor)
  (no command) with --dry-run: self-check that this controller can load the state and
            answer `frontier`; exits non-zero if the state fails validation.
`--dry-run` on any mutating command computes and prints the answer but writes nothing.

Changelog (v2-v8 were controller steps by run 1's workers, see ab/gear/decisions.md; v9 is the run-2 build):
  v2  p1_candidates keeps an in-flight p1 shareable when every alternative is > SHARE_MARGIN below it
  v3  pick_p1 drops elites more than FLOOR_GAP below the best (if another candidate remains)
  v4  a fresh elite forces a crossover only if it did not score below its own parent1
  v5  pick_p2 excludes p1's ancestors and descendants (if another candidate remains)
  v6  forced crossovers fall back to mutating p1 when every other elite is p1's lineage
  v7  a partner that lost to its own parent1 by > LOSS_BAND is not a crossover partner (p2, v6 check)
  v8  a clear leader (> SHARE_MARGIN ahead of every other elite) is mutated instead of a due crossover
  v9  run 2: margins scale with the score (improvement max(8, round(0.6*sqrt(best))), tie 2x; the v2,
      v3, v7 bands scale with the tie margin); two-seed rule for BEST (PENDING_BEST -> auditor
      replicate -> BEST or demoted by the two-seed mean); held-out greedy floor 0.85 for any promotion;
      `record --source`; frontier count (L12-14) shown per elite; 7 workers + `auditor`; v8's leader
      exploitation capped at LEADER_MAX_SHARE concurrent workers
"""

import fcntl
import json
import math
import os
import re
import shutil
import sys
from datetime import datetime, timezone

CONTROLLER_VERSION = 9
STATE_VERSION = 2
DEFAULT_ROOT = os.environ.get("GEAR_ROOT", "/scratch/cowifornia/nd-rl/ab2")

# ---------------------------------------------------------------- v9: score-scaled margins
IMP_MIN, IMP_SQRT = 8, 0.6   # improvement margin = max(IMP_MIN, round(IMP_SQRT * sqrt(best metric)))
TIE_MULT = 2                 # tie margin ("within noise of the best") = TIE_MULT x improvement margin
# The run-1 bands were set in units of the run-1 noise (~15 theorems): LOSS_BAND 15 = 1 noise, SHARE_MARGIN
# 30 = 2 noise, FLOOR_GAP 50 = ~3.3 noise. Run 2's tie margin (2 x 0.6 sqrt(best)) is the noise-sized
# quantity, so the bands are multiples of it:
LOSS_MULT = 1.0              # v7 loss band    = 1.0 x tie
SHARE_MULT = 2.0             # v2/v8 share     = 2.0 x tie
FLOOR_MULT = 3.5             # v3 p1 floor gap = 3.5 x tie
HELDOUT_FLOOR = 0.85         # v9: heldout_greedy below this is never promoted (recorded as evaluated only)
N_WORKERS = 7                # concurrent hill-climbing workers (the auditor is extra)
AUDITOR = "auditor"          # pseudo-worker that runs the second seed of PENDING_BEST nodes
MAX_REPLICATE_ATTEMPTS = 2   # crashed/timed-out replicates before a PENDING_BEST node is dropped
LEADER_MAX_SHARE = 4         # v9: v8's "mutate the clear leader" applies while fewer workers hold it as p1

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
JACCARD_DIVERSE = 0.65     # DIVERSE needs Jaccard distance >= 0.65 to every elite
LEAN_HELDOUT_MARGIN = 0.01 # LEAN needs heldout_greedy higher by >= 0.01
FRESH_FROM = ("mutate",)   # promotions of these operators' children force a crossover with them

CATEGORIES = ("optimizer", "sampling", "data", "curriculum", "objective", "schedule", "other")
ROLES = ("BEST", "LEAN", "DIVERSE")
PROMOTING = ("FILL", "BEST", "REPLACE_WEAKEST", "LEAN", "DIVERSE")
# other decisions: DISCARD, FLOOR (heldout below HELDOUT_FLOOR), PENDING_BEST (awaiting the replicate),
# REPLICATE_FAILED, CRASH, TIMEOUT


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
        if st.get("version") == 1:   # v9 migration: pending replicates, replicate log, auditor job
            st["version"] = 2
        if st.get("version") != STATE_VERSION:
            raise GearError("state version %r != %r" % (st.get("version"), STATE_VERSION))
        st.setdefault("pending", [])
        st.setdefault("replicates", [])
        st.setdefault("auditor", None)
        st.setdefault("best_id", None)
        for n in st["nodes"].values():
            for k, v in META_DEFAULTS.items():
                n.setdefault(k, json.loads(json.dumps(v)))
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
    """v9: BEST is sticky. It changes only when a node beats it by the improvement margin, which after
    the first elite means a confirmed two-seed mean (`best_id`). Ordinary single-seed promotions never
    take the role, even when their metric is slightly (within the margin) above the BEST's.
    Without a `best_id` in the frontier (fresh or migrated state) it falls back to the highest metric."""
    fr = frontier_nodes(st)
    if not fr:
        return None
    bid = st.get("best_id")
    if bid in st["frontier"]:
        return st["nodes"][bid]
    return sorted(fr, key=lambda n: (-n["metric"], n["id"]))[0]


def margins(st, best_metric=None):
    """v9: the improvement / tie margins and the derived bands, from the current BEST's metric."""
    if best_metric is None:
        b = best_node(st)
        best_metric = b["metric"] if b is not None else 0
    imp = max(IMP_MIN, int(round(IMP_SQRT * math.sqrt(max(best_metric, 0)))))
    tie = TIE_MULT * imp
    return {"best": best_metric, "improve": imp, "tie": tie, "diverse_slack": tie,
            "loss_band": int(round(LOSS_MULT * tie)), "share": int(round(SHARE_MULT * tie)),
            "floor_gap": int(round(FLOOR_MULT * tie)), "heldout_floor": HELDOUT_FLOOR}


def role_node(st, role):
    for n in frontier_nodes(st):
        if n["role"] == role:
            return n
    return None


def relabel(st):
    """BEST = best_node() (sticky, v9); LEAN = within the tie margin of the best with heldout_greedy at least
    LEAN_HELDOUT_MARGIN above the BEST's (highest such); every other elite is DIVERSE."""
    for nid in st["order"]:
        if nid not in st["frontier"]:
            st["nodes"][nid]["role"] = None
    b = best_node(st)
    if b is None:
        return
    b["role"] = "BEST"
    tie = margins(st)["tie"]
    lean_cands = []
    for n in frontier_nodes(st):
        if n is b or n["heldout_greedy"] is None:
            continue
        if n["metric"] < b["metric"] - tie:
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
    mx = lambda ns: max((n["metric"] if n["metric"] is not None else -1e9) for n in ns)
    share = margins(st)["share"]
    if c and len(c) < len(pool) and mx(pool) > mx(c) + share:
        notes.append("sharing in-flight p1 %s: every other candidate is > %d below it" % (
            sorted(held & {n["id"] for n in pool}), share))
        c = list(pool)
    elif c and len(c) < len(pool):
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
    # v3 (v9: relative to the frontier BEST, not to unconfirmed PENDING_BEST first seeds)
    b = best_node(st)
    top = b["metric"] if b is not None else None
    if top is not None:
        gap = margins(st)["floor_gap"]
        c2 = [n for n in cands if n["metric"] is not None and n["metric"] >= top - gap]
        if c2 and len(c2) < len(cands):
            notes.append("excluded %s: more than %d below the best (%s)" % (
                sorted(n["id"] for n in cands if n not in c2), gap, top))
            cands = c2
    scored = {n["id"]: score_components(st, n, step, lam, gam) for n in cands}
    best = sorted(cands, key=lambda n: (-scored[n["id"]]["total"], n["id"]))[0]
    return best, scored


def lineage(st, nid):
    """v5: all ancestors of nid (through every parent)."""
    seen, stack = set(), list(st["nodes"][nid].get("parents") or [])
    while stack:
        a = stack.pop()
        if a in seen or a not in st["nodes"]:
            continue
        seen.add(a)
        stack += st["nodes"][a].get("parents") or []
    return seen


def unrelated(st, p1, pool):
    """v6: elites of `pool` that are neither ancestors nor descendants of p1 (a crossover partner with ideas p1 lacks)."""
    anc = lineage(st, p1["id"])
    return [n for n in pool if n["id"] != p1["id"] and n["id"] not in anc and p1["id"] not in lineage(st, n["id"])
            and not lost_to_parent(st, n)]   # v7: nor partners whose own change lost to their parent


def lost_to_parent(st, n):
    """v7: n scored more than the loss band (v9: = tie margin) below its parent1 (its idea is not worth transplanting)."""
    ps = n.get("parents") or []
    pm = st["nodes"][ps[0]]["metric"] if ps and ps[0] in st["nodes"] else None
    return pm is not None and n["metric"] is not None and n["metric"] < pm - margins(st)["loss_band"]


def pick_p2(st, p1, pool):
    cands = [n for n in pool if n["id"] != p1["id"]]
    if not cands:
        return None, {}
    # v5: never pair p1 with its own ancestor or descendant (the transplant would be empty), if others remain
    anc = lineage(st, p1["id"])
    c2 = [n for n in cands if n["id"] not in anc and p1["id"] not in lineage(st, n["id"])]
    if c2:
        cands = c2
    # v7: a partner whose own change lost to its parent1 by more than the noise band has no idea
    # worth transplanting (e.g. n016 grpo pass, -56); drop such partners if others remain
    c3 = [n for n in cands if not lost_to_parent(st, n)]
    if c3:
        cands = c3
    scored = {n["id"]: comp_score(st, p1, n) for n in cands}
    best = sorted(cands, key=lambda n: (-scored[n["id"]]["total"], -n["metric"], n["id"]))[0]
    return best, scored


def leader_exploitable(st, worker, fr, notes):
    """v8 condition (clear leader), with the v9 cap on how many workers mutate it at once."""
    ms = sorted(n["metric"] for n in fr)
    if not ms[-1] > ms[-2] + margins(st)["share"]:
        return False
    lead = sorted(fr, key=lambda n: (-n["metric"], n["id"]))[0]
    k = sum(1 for a in other_inflight(st, worker).values() if a["p1"] == lead["id"])
    if k >= LEADER_MAX_SHARE:
        notes.append("v9: leader %s already p1 for %d workers (cap %d); normal policy" % (lead["id"], k, LEADER_MAX_SHARE))
        return False
    return True


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
    meta = new_meta(nid, [], "seed", args.get("desc") or "seed baseline rl.py unmodified", "other", None,
                    source=args.get("source") or "seed (run 1 best, n025)")
    st = {"version": STATE_VERSION, "controller_version": CONTROLLER_VERSION, "created": now(),
          "nodes": {nid: meta}, "order": [nid], "frontier": [], "assign_count": 0,
          "assignments": [], "inflight": {}, "completed": [], "co_use": {}, "fresh_elites": [],
          "next_id": 1, "pending": [], "replicates": [], "auditor": None, "best_id": None}
    if not args.get("dry_run"):
        d = store.node_dir(nid)
        os.makedirs(d, exist_ok=True)
        shutil.copyfile(seed, os.path.join(d, "rl.py"))
        store.save(st)
    return {"ok": True, "root": store.root, "node": nid, "status": "pending", "frontier": []}


# v9 node fields (defaults also fill in older states on load)
META_DEFAULTS = {"source": None, "frontier": None, "frontier_by_len": None, "seeds": [], "metric_seeds": [],
                 "heldout_seeds": [], "frontier_seeds": [], "decision_first": None, "replicate_attempts": 0}


def new_meta(nid, parents, operator, desc, category, worker, source=None):
    m = {"id": nid, "parents": list(parents), "operator": operator, "description": desc,
         "category": category, "worker": worker, "created": now(), "metric": None,
         "by_len": None, "heldout_greedy": None, "status": "pending", "run_dir": None,
         "n_children": 0, "n_successful_children": 0, "mean_child_improvement": None,
         "n_uses": 0, "last_used_step": None, "role": None, "decision": None,
         "evicted_by": None}
    m.update(json.loads(json.dumps(META_DEFAULTS)))
    m["source"] = source
    return m


def node_brief(store, n):
    if n is None:
        return None
    return {"id": n["id"], "rl": os.path.join(store.node_dir(n["id"]), "rl.py"),
            "description": n["description"], "category": n["category"], "role": n["role"],
            "metric": n["metric"], "by_len": n["by_len"], "heldout_greedy": n["heldout_greedy"],
            "frontier": n.get("frontier"), "source": n.get("source")}


def frontier_rows(st):
    rows = []
    for n in sorted(frontier_nodes(st), key=lambda n: (ROLES.index(n["role"]) if n["role"] in ROLES else 9, -n["metric"], n["id"])):
        rows.append({"id": n["id"], "role": n["role"], "metric": n["metric"], "by_len": n["by_len"],
                     "metric_seeds": n.get("metric_seeds") or [], "frontier": n.get("frontier"),
                     "frontier_by_len": n.get("frontier_by_len"), "source": n.get("source"),
                     "heldout_greedy": n["heldout_greedy"], "n_children": n["n_children"],
                     "n_uses": n["n_uses"], "mean_child_improvement": r4(n["mean_child_improvement"]),
                     "last_used_step": n["last_used_step"], "category": n["category"],
                     "description": n["description"]})
    return rows


def fmt_num(x):
    if x is None:
        return "-"
    return ("%d" % x) if float(x) == int(x) else ("%.1f" % x)


def fmt_len(bl):
    if not isinstance(bl, dict):
        return "-"
    key = lambda k: (int(k) if str(k).isdigit() else 10 ** 9, str(k))
    return "/".join(fmt_num(bl[k]) for k in sorted(bl, key=key))


def frontier_table(st):
    lines = ["%-5s %-8s %6s %5s %-20s %5s %-12s %7s %4s %4s %7s %4s %-10s %s" % (
        "id", "role", "metric", "seeds", "by_len", "front", "front_by_len", "heldout", "kids", "uses", "meanimp",
        "last", "category", "description [source]")]
    for r in frontier_rows(st):
        lines.append("%-5s %-8s %6s %5s %-20s %5s %-12s %7s %4d %4d %7s %4s %-10s %s [%s]" % (
            r["id"], r["role"], fmt_num(r["metric"]), max(1, len(r["metric_seeds"])), fmt_len(r["by_len"]),
            fmt_num(r["frontier"]), fmt_len(r["frontier_by_len"]),
            "-" if r["heldout_greedy"] is None else "%.3f" % r["heldout_greedy"],
            r["n_children"], r["n_uses"],
            "-" if r["mean_child_improvement"] is None else "%+.1f" % r["mean_child_improvement"],
            "-" if r["last_used_step"] is None else r["last_used_step"], r["category"], r["description"],
            r["source"] or "-"))
    for pb in pending_rows(st):
        lines.append("%-5s %-8s %6s %5s %-20s %5s %-12s %7s  awaiting replicate (attempts %d%s) %s [%s]" % (
            pb["id"], "PENDING", fmt_num(pb["metric"]), 1, fmt_len(pb["by_len"]), fmt_num(pb["frontier"]),
            fmt_len(pb["frontier_by_len"]), "-" if pb["heldout_greedy"] is None else "%.3f" % pb["heldout_greedy"],
            pb["attempts"], ", assigned" if pb["assigned"] else "", pb["description"], pb["source"] or "-"))
    return lines


def pending_rows(st):
    """v9: PENDING_BEST nodes, oldest first, with the auditor's assignment state."""
    out = []
    job = st.get("auditor") or {}
    for p in st.get("pending", []):
        n = st["nodes"][p["node"]]
        out.append({"id": n["id"], "metric": n["metric"], "by_len": n["by_len"], "frontier": n.get("frontier"),
                    "frontier_by_len": n.get("frontier_by_len"), "heldout_greedy": n["heldout_greedy"],
                    "first_seed": (n.get("seeds") or [None])[0], "best_at_record": p.get("best_at_record"),
                    "attempts": n.get("replicate_attempts", 0), "assigned": job.get("node") == n["id"],
                    "since": p.get("since"), "worker": n["worker"], "description": n["description"],
                    "source": n.get("source")})
    return out


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
           "margins": margins(st), "frontier": frontier_rows(st),
           "pending_best": [p["id"] for p in pending_rows(st)], "notes": notes}
    if extra:
        out.update(extra)
    return out


def auditor_next(store, st, args):
    """v9: the auditor's job: replicate the oldest PENDING_BEST node with another seed, or idle."""
    notes = []
    job = st.get("auditor")
    pend = [p["node"] for p in st["pending"]]
    if job and job["node"] in pend:
        notes.append("already assigned; returning the same replicate job (use `release --worker auditor` to drop it)")
    elif not pend:
        return {"action": "idle", "worker": AUDITOR, "pending_best": [], "margins": margins(st),
                "notes": ["no PENDING_BEST node awaits a replicate"]}
    else:
        nid = pend[0]
        n = st["nodes"][nid]
        first = (n.get("seeds") or [0])[0]
        first = first if isinstance(first, int) else 0
        job = {"node": nid, "seed": first + 1 + n.get("replicate_attempts", 0), "issued": now()}
        st["auditor"] = job
        if not args.get("dry_run"):
            store.save(st)
        else:
            notes.append("dry run: nothing written")
    n = st["nodes"][job["node"]]
    b = best_node(st)
    rl = os.path.join(store.node_dir(n["id"]), "rl.py")
    run = "rep_%s_s%d" % (n["id"], job["seed"])
    notes.append("run the node's rl.py unmodified with seed %d, then `replicate --node %s --summary runs/%s/summary.json`"
                 % (job["seed"], n["id"], run))
    return {"action": "replicate", "worker": AUDITOR, "node": n["id"], "node_rl": rl, "seed": job["seed"],
            "first_seed": (n.get("seeds") or [None])[0], "first_metric": n["metric"],
            "first_heldout_greedy": n["heldout_greedy"], "first_frontier": n.get("frontier"),
            "description": n["description"], "source": n.get("source"),
            "current_best": b["id"] if b else None, "current_best_metric": b["metric"] if b else None,
            "margins": margins(st), "attempts": n.get("replicate_attempts", 0),
            "command": "python frozen/harness.py --out runs/%s --seed %d --rl %s" % (run, job["seed"], rl),
            "pending_best": pend, "issued": job["issued"], "notes": notes}


def cmd_next(store, args):
    st = store.load()
    w = args["worker"]
    if w == AUDITOR:
        return auditor_next(store, st, args)
    if w not in st["inflight"] and len(st["inflight"]) >= N_WORKERS:
        raise GearError("%d workers already hold assignments (%s); N_WORKERS is %d. Use your own worker name, "
                        "or `release` a lost assignment" % (len(st["inflight"]), sorted(st["inflight"]), N_WORKERS))
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
        # v4: a fresh elite forces a crossover only if it did not score below its parent1
        # (a DIVERSE entry whose change lost to its parent has no idea worth transplanting)
        def _beat_parent(f):
            ps = nodes[f].get("parents") or []
            pm = nodes[ps[0]]["metric"] if ps and ps[0] in nodes else None
            return pm is None or nodes[f]["metric"] is None or nodes[f]["metric"] >= pm
        skipped = [f for f in fresh if not _beat_parent(f)]
        if skipped:
            notes.append("fresh elite(s) %s not recombined: scored below their parent1" % skipped)
        fresh = [f for f in fresh if _beat_parent(f)]
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
            # v6: if the only partners are the fresh elite's own lineage, a crossover has nothing to
            # transplant (the child already carries every ancestor idea): mutate the fresh elite instead
            if p1["id"] == f["id"] and not unrelated(st, p1, fr):
                a = {"action": "mutate", "p1": p1["id"], "p2": None}
                notes.append("v6: every other elite is an ancestor/descendant of fresh elite %s or lost to its parent (v7); mutate it instead" % f["id"])
                extra = {"p1_scores": p1_scores}
        elif len(fr) >= 2 and leader_exploitable(st, w, fr, notes):
            # v8: a clear leader (ahead of every other elite by > the share margin) is exploited: mutate it,
            # instead of a due crossover or an Eq. 1 pick among clearly weaker elites
            # (v9: only while fewer than LEADER_MAX_SHARE other workers already mutate it)
            p1 = sorted(fr, key=lambda n: (-n["metric"], n["id"]))[0]
            a = {"action": "mutate", "p1": p1["id"], "p2": None}
            notes.append("v8: %s leads every other elite by > %d; mutate the leader%s" % (
                p1["id"], margins(st)["share"], " (due crossover skipped)" if due else ""))
        elif len(fr) >= 2 and due:
            p1, p1_scores = pick_p1(st, w, fr, step, notes)
            p2, p2_scores = pick_p2(st, p1, fr)
            a = {"action": "crossover", "p1": p1["id"], "p2": p2["id"]}
            notes.append("crossover forced: %d crossover(s) in the last %d experiments %s" % (
                sum(1 for o in window if o == "crossover"), XO_WINDOW, window))
            extra = {"p1_scores": p1_scores, "p2_scores": p2_scores}
            if not unrelated(st, p1, fr):
                a = {"action": "mutate", "p1": p1["id"], "p2": None}
                notes.append("v6: due crossover dropped: every other elite is an ancestor/descendant of %s" % p1["id"])
                extra = {"p1_scores": p1_scores}
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
    if w == AUDITOR:
        job, st["auditor"] = st.get("auditor"), None
        if job and not args.get("dry_run"):
            store.save(st)
        return {"ok": True, "released": job, "note": None if job else "the auditor had no replicate job"}
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
    mg = margins(st)
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

    if m > b["metric"] + mg["improve"]:
        return "BEST", replace(w)
    if m > w["metric"] + mg["improve"]:
        return "REPLACE_WEAKEST", replace(w)
    if (m >= b["metric"] - mg["tie"] and hg is not None
            and (ref["heldout_greedy"] is None or hg >= ref["heldout_greedy"] + LEAN_HELDOUT_MARGIN - 1e-12)):
        return "LEAN", replace(lean if lean is not None else w)
    if m >= w["metric"] - mg["diverse_slack"] and all(
            jaccard_distance(child["description"], n["description"]) >= JACCARD_DIVERSE for n in fr):
        plain = [n for n in non_best if n is not lean]
        return "DIVERSE", replace(weakest(plain) if plain else w)
    return "DISCARD", None


def extract_frontier(s):
    v = s.get("frontier")
    return v if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def seed_of(s):
    v = (s or {}).get("seed")
    return v if isinstance(v, int) and not isinstance(v, bool) else None


def decide(st, child, operator, notes):
    """v9 promotion front end: held-out floor, then the two-seed rule for BEST, then promote().
    Returns (decision, evicted)."""
    m, hg = child["metric"], child["heldout_greedy"]
    if hg is None or hg < HELDOUT_FLOOR:
        notes.append("heldout_greedy %s is below the floor %.2f: evaluated only, never promoted"
                     % ("missing" if hg is None else "%.3f" % hg, HELDOUT_FLOOR))
        return "FLOOR", None
    b = best_node(st)
    mg = margins(st)
    beats = b is None or m > b["metric"] + mg["improve"]
    if b is not None and beats and child["decision_first"] is None:
        st["pending"].append({"node": child["id"], "since": now(), "best_at_record": b["id"],
                              "best_metric_at_record": b["metric"], "margin_at_record": mg["improve"]})
        notes.append("beats BEST %s (%s) by more than %d: PENDING_BEST until the auditor's second seed; "
                     "not in the frontier yet" % (b["id"], fmt_num(b["metric"]), mg["improve"]))
        return "PENDING_BEST", None
    decision, evicted = promote(st, child)
    if decision in PROMOTING and beats:
        st["best_id"] = child["id"]      # first elite, or a confirmed two-seed mean over the BEST
    relabel(st)
    if decision in PROMOTING and decision != "FILL" and operator in FRESH_FROM:
        st["fresh_elites"].append(child["id"])
    return decision, evicted


def cmd_record(store, args):
    st = store.load()
    nodes = st["nodes"]
    w = args["worker"]
    notes = []
    if w == AUDITOR:
        raise GearError("the auditor records replicates with `replicate --node ID --summary PATH`, not `record`")
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
        source = (args.get("source") or "").strip()
        if not source:
            raise GearError('--source is required: the URL the idea came from (paper, repo, issue), '
                            'or "own idea"; for a crossover, the source of the transplanted idea')
        if not args.get("node_rl") or not os.path.isfile(args["node_rl"]):
            raise GearError("--node-rl %r not found" % args.get("node_rl"))
        nid = node_id(st["next_id"])
        st["next_id"] += 1
        child = new_meta(nid, parents, operator, desc, cat, w, source=source)
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
    child["frontier"] = extract_frontier(summary or {}) if metric is not None else None
    child["frontier_by_len"] = (summary or {}).get("frontier_by_len") if metric is not None else None
    child["seeds"] = [seed_of(summary)]
    child["metric_seeds"] = [metric] if metric is not None else []
    child["heldout_seeds"] = [child["heldout_greedy"]] if metric is not None else []
    child["frontier_seeds"] = [child["frontier"]] if metric is not None else []
    child["run_dir"] = os.path.dirname(os.path.abspath(args["summary"])) if args.get("summary") else None
    if summary is not None and not args.get("dry_run"):
        shutil.copyfile(args["summary"], os.path.join(store.node_dir(child["id"]), "summary.json"))

    if metric is None:
        decision, evicted = status.upper(), None
    else:
        decision, evicted = decide(st, child, operator, notes)
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
            "frontier_count": child.get("frontier"), "heldout_greedy": child["heldout_greedy"],
            "source": child.get("source"), "status": child["status"], "operator": operator, "parents": parents,
            "margins": margins(st),
            "completed_experiments": completed_count(st), "frontier": frontier_rows(st),
            "pending_best": [p["id"] for p in pending_rows(st)], "notes": notes}


def file_sha256(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def mean(xs):
    xs = [x for x in xs if x is not None]
    return (sum(xs) / len(xs)) if xs else None


def cmd_replicate(store, args):
    """v9: record the auditor's second-seed run of a PENDING_BEST node and resolve it.
    ok run: the node's metric / heldout_greedy / frontier become the two-seed means; the held-out floor
    and then the normal promotion rules are applied to the means (BEST only if the mean still beats the
    current BEST by the improvement margin). crash/timeout: the node stays pending (the auditor gets it
    again with the next seed) until MAX_REPLICATE_ATTEMPTS, then it is REPLICATE_FAILED (not promoted)."""
    st = store.load()
    nodes = st["nodes"]
    nid = args["node"]
    notes = []
    pend = [p for p in st["pending"] if p["node"] == nid]
    if nid not in nodes:
        raise GearError("unknown node %s" % nid)
    if not pend:
        raise GearError("node %s is not PENDING_BEST (pending: %s)" % (nid, [p["node"] for p in st["pending"]]))
    n = nodes[nid]
    summary = read_summary(args.get("summary"))
    if summary is None:
        if not args.get("status"):
            raise GearError("no summary.json at %s (if the replicate died before writing one, pass --status crash)"
                            % args.get("summary"))
        notes.append("no summary.json at %s" % args.get("summary"))
        summary = {}
    status = args.get("status") or summary.get("status") or "crash"
    rl_sha = (summary.get("shas") or {}).get("rl.py")
    node_sha = file_sha256(os.path.join(store.node_dir(nid), "rl.py"))
    if rl_sha is not None and rl_sha != node_sha:
        raise GearError("the replicate ran a different rl.py (sha %s) than node %s (sha %s); run nodes/%s/rl.py unmodified"
                        % (rl_sha[:12], nid, node_sha[:12], nid))
    if rl_sha is None:
        notes.append("WARNING: summary has no rl.py hash; could not check that the node's code was run")
    s2 = seed_of(summary)
    if s2 is not None and s2 in (n.get("seeds") or []):
        raise GearError("replicate seed %d was already used for %s; use another seed" % (s2, nid))
    m2 = summary.get("metric")
    if status == "ok" and not isinstance(m2, (int, float)):
        status = "crash"
        notes.append("summary has no numeric metric; replicate recorded as crash")
    rep_dir = os.path.dirname(os.path.abspath(args["summary"])) if args.get("summary") else None
    entry = {"node": nid, "time": now(), "status": status, "seed": s2, "metric": m2 if status == "ok" else None,
             "heldout_greedy": extract_heldout(summary) if status == "ok" else None,
             "frontier": extract_frontier(summary) if status == "ok" else None, "run_dir": rep_dir,
             "first_metric": n["metric"], "first_seed": (n.get("seeds") or [None])[0]}
    decision, evicted = None, None
    if status != "ok":
        n["replicate_attempts"] = n.get("replicate_attempts", 0) + 1
        st["auditor"] = None
        if n["replicate_attempts"] >= MAX_REPLICATE_ATTEMPTS:
            st["pending"] = [p for p in st["pending"] if p["node"] != nid]
            decision = "REPLICATE_FAILED"
            n["decision_first"] = n["decision_first"] or "PENDING_BEST"
            n["decision"] = decision
            notes.append("replicate %s %d times: %s is dropped (not promoted)" % (status, n["replicate_attempts"], nid))
        else:
            notes.append("replicate %s (attempt %d of %d): %s stays PENDING_BEST; `next --worker auditor` "
                         "gives it again with the next seed" % (status, n["replicate_attempts"], MAX_REPLICATE_ATTEMPTS, nid))
    else:
        hg2, f2 = entry["heldout_greedy"], entry["frontier"]
        n["seeds"] = (n.get("seeds") or []) + [s2]
        n["metric_seeds"] = (n.get("metric_seeds") or [n["metric"]]) + [m2]
        n["heldout_seeds"] = (n.get("heldout_seeds") or [n["heldout_greedy"]]) + [hg2]
        n["frontier_seeds"] = (n.get("frontier_seeds") or [n.get("frontier")]) + [f2]
        by1, by2 = n.get("by_len") or {}, summary.get("by_len") or {}
        n["by_len_seeds"] = [by1, by2]
        n["by_len"] = {k: mean([by1.get(k), by2.get(k)]) for k in sorted(set(by1) | set(by2))}
        fb1, fb2 = n.get("frontier_by_len") or {}, summary.get("frontier_by_len") or {}
        n["frontier_by_len"] = {k: mean([fb1.get(k), fb2.get(k)]) for k in sorted(set(fb1) | set(fb2))} or None
        n["metric"] = mean(n["metric_seeds"])
        n["heldout_greedy"] = mean(n["heldout_seeds"])
        n["frontier"] = mean(n["frontier_seeds"])
        n["decision_first"] = "PENDING_BEST"
        st["pending"] = [p for p in st["pending"] if p["node"] != nid]
        st["auditor"] = None
        b = best_node(st)
        notes.append("two-seed mean %s (seeds %s: %s) vs BEST %s %s, margin %d" % (
            fmt_num(n["metric"]), n["seeds"], n["metric_seeds"], b["id"] if b else "-",
            fmt_num(b["metric"]) if b else "-", margins(st)["improve"]))
        decision, evicted = decide(st, n, n["operator"], notes)
        n["decision"] = decision
        if decision != "BEST":
            notes.append("not confirmed as BEST: demoted and promoted by the ordinary rules -> %s" % decision)
        if not args.get("dry_run") and os.path.isfile(args.get("summary") or ""):
            shutil.copyfile(args["summary"], os.path.join(store.node_dir(nid), "replicate_summary.json"))
    entry.update({"decision": decision, "evicted": evicted, "mean_metric": n["metric"] if status == "ok" else None})
    st["replicates"].append(entry)
    for c in st["completed"]:
        if c["node"] == nid and decision is not None:
            c["decision_first"] = c.get("decision_first") or c["decision"]
            c["decision"] = decision
            c["evicted"] = evicted
            c["metric_first"] = c.get("metric_first", c["metric"])
            c["metric"] = n["metric"]
            if c.get("parent1_metric") is not None and n["metric"] is not None:
                c["delta"] = n["metric"] - c["parent1_metric"]
            c["role"] = n["role"]
            c["resolved"] = now()
    if not args.get("dry_run"):
        store.save(st)
    else:
        recompute_stats(st)
        notes.append("dry run: nothing written")
    return {"ok": True, "node": nid, "replicate_status": status, "decision": decision, "role": n["role"],
            "evicted": evicted, "metric": n["metric"], "metric_seeds": n.get("metric_seeds"),
            "heldout_greedy": n["heldout_greedy"], "frontier_count": n.get("frontier"), "seeds": n.get("seeds"),
            "pending_best": [p["id"] for p in pending_rows(st)], "frontier": frontier_rows(st), "notes": notes}


def cmd_frontier(store, args):
    st = store.load()
    return {"frontier": frontier_rows(st), "table": frontier_table(st),
            "completed_experiments": completed_count(st), "step": st["assign_count"],
            "inflight": {w: {"action": a["action"], "parent1": a["p1"], "parent2": a["p2"], "step": a["step"]}
                         for w, a in st["inflight"].items()},
            "fresh_elites": st["fresh_elites"], "lambda": weights(st)[0], "gamma": weights(st)[1],
            "margins": margins(st), "pending_best": pending_rows(st), "auditor": st.get("auditor"),
            "replicates": st.get("replicates", [])[-10:]}


def cmd_history(store, args):
    st = store.load()
    by_node = {c["node"]: c for c in st["completed"]}
    rows = []
    for nid in st["order"]:
        n = st["nodes"][nid]
        c = by_node.get(nid, {})
        rows.append({"id": nid, "parents": n["parents"], "operator": n["operator"], "worker": n["worker"],
                     "category": n["category"], "status": n["status"], "metric": n["metric"],
                     "metric_seeds": n.get("metric_seeds") or [], "seeds": n.get("seeds") or [],
                     "frontier": n.get("frontier"), "heldout_greedy": n["heldout_greedy"],
                     "delta": c.get("delta"), "decision": n["decision"], "decision_first": n.get("decision_first"),
                     "role": n["role"], "evicted_by": n["evicted_by"], "in_frontier": nid in st["frontier"],
                     "description": n["description"], "source": n.get("source")})
    table = ["%-5s %-11s %-9s %6s %5s %5s %7s %-16s %s [%s]" % (
        "id", "parents", "operator", "metric", "seeds", "front", "heldout", "decision", "description", "source")]
    for r in rows:
        table.append("%-5s %-11s %-9s %6s %5s %5s %7s %-16s %s [%s]" % (
            r["id"], "+".join(r["parents"]) or "-", r["operator"], fmt_num(r["metric"]), len(r["metric_seeds"]),
            fmt_num(r["frontier"]), "-" if r["heldout_greedy"] is None else "%.3f" % r["heldout_greedy"],
            (r["decision"] or r["status"] or "-") + ("*" if r["decision_first"] == "PENDING_BEST" else ""),
            r["description"], r["source"] or "-"))
    return {"history": rows, "table": table, "completed": st["completed"], "replicates": st.get("replicates", []),
            "pending_best": pending_rows(st)}


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
    if AUDITOR in st["inflight"]:
        problems.append("the auditor holds a worker assignment")
    for p in st.get("pending", []):
        if p["node"] not in st["nodes"]:
            problems.append("pending node %s unknown" % p["node"])
        elif p["node"] in st["frontier"]:
            problems.append("pending node %s is in the frontier" % p["node"])
        elif st["nodes"][p["node"]]["metric"] is None:
            problems.append("pending node %s has no metric" % p["node"])
    for nid in st["frontier"]:
        n = st["nodes"].get(nid)
        if n and nid != node_id(0) and n.get("heldout_greedy") is not None and n["heldout_greedy"] < HELDOUT_FLOOR - 1e-12:
            problems.append("frontier node %s is below the held-out floor" % nid)
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
            "replicate": ["node", "summary"], "frontier": [], "history": [], None: []}
COMMANDS = {"init": cmd_init, "next": cmd_next, "record": cmd_record, "release": cmd_release,
            "replicate": cmd_replicate, "frontier": cmd_frontier, "history": cmd_history}


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
