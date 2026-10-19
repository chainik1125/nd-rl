#!/usr/bin/env python3
"""Simulated seven-worker + auditor test of controller.py (stdlib only), run 2 (v9).

    python3 test_controller.py -v

A fake "quality" is attached to every node: baseline 0, mutation = parent + N(0.03, 0.1),
crossover = max(parents) + N(0.05, 0.05). metric = round(400 + 300 * quality + N(0, 8)).
About 12 % of non-baseline runs crash or time out and about 10 % fall below the held-out floor.
Seven workers and the auditor run with random durations, so their `next`/`record`/`replicate`
calls interleave the way the GPUs would. TestTwoSeed drives the v9 rules (two-seed BEST, held-out
floor, --source, auditor commands) deterministically through the CLI.
"""

import hashlib
import importlib
import json
import multiprocessing as mp
import os
import random
import shutil
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import controller as C  # noqa: E402

VOCAB = {
    "optimizer": ["adam lr 3e-5", "adam beta2 0.95", "grad clip 0.5", "weight decay 0.01", "sgd momentum"],
    "sampling": ["temperature 1.0 during rollout", "k 16 samples per target", "nucleus top p 0.9"],
    "data": ["mix cap6 retention records ratio 0.3", "hindsight relabel verified proofs", "extra targets pool"],
    "curriculum": ["easy first by length 7 then 9", "focus unsolved targets", "reweight by pass rate"],
    "objective": ["grpo group baseline", "kl to init 0.05", "entropy bonus 0.01", "expert iteration sft"],
    "schedule": ["cosine lr decay", "linear warmup 50 steps", "more rounds shorter train"],
    "other": ["dedupe normalized proofs", "bigger train batch 256"],
}


def call(root, *argv):
    return C.run(list(argv) + ["--root", root])


def load_state(root):
    with open(os.path.join(root, "gear_state.json")) as f:
        return json.load(f)


def _worker_next(args):
    root, worker, barrier = args
    barrier.wait()
    return call(root, "next", "--worker", worker)


SOURCES = ["own idea", "https://arxiv.org/abs/2203.14465", "https://arxiv.org/abs/2402.03300",
           "https://github.com/karpathy/autoresearch", "https://arxiv.org/abs/2308.08998"]


def sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _proc_next(root, worker, barrier, q):
    try:
        barrier.wait()
        out = call(root, "next", "--worker", worker)
        q.put(("ok", worker, out["step"], out["parent1"], out["parent2"]))
    except Exception as e:  # pragma: no cover
        q.put(("err", worker, repr(e), None, None))


def _proc_churn(root, worker, barrier, n, q):
    try:
        barrier.wait()
        for _ in range(n):
            call(root, "next", "--worker", worker)
            call(root, "release", "--worker", worker)
        q.put(("ok", worker))
    except Exception as e:  # pragma: no cover
        q.put(("err", repr(e)))


class Sim:
    def __init__(self, root, seed=1):
        self.root = root
        self.rng = random.Random(seed)
        self.quality = {"n000": 0.0}
        self.work = os.path.join(root, "_workspaces")
        os.makedirs(self.work, exist_ok=True)
        self.log = []          # (event, payload)
        self.violations = []

    def fake_run(self, worker, out, idx):
        rng = self.rng
        ws = os.path.join(self.work, worker)
        os.makedirs(ws, exist_ok=True)
        rl = os.path.join(ws, "rl.py")
        shutil.copyfile(out["parent1_rl"], rl)
        with open(rl, "a") as f:
            f.write("# %s edit %d (%s)\n" % (worker, idx, out["action"]))
        q1 = self.quality[out["parent1"]]
        if out["action"] == "baseline":
            q, crash = 0.0, False
        elif out["action"] == "mutate":
            q, crash = q1 + rng.gauss(0.03, 0.1), rng.random() < 0.12
        else:
            q = max(q1, self.quality[out["parent2"]]) + rng.gauss(0.05, 0.05)
            crash = rng.random() < 0.12
        run_dir = os.path.join(ws, "runs", "exp%03d" % idx)
        os.makedirs(run_dir, exist_ok=True)
        if crash:
            summ = {"status": rng.choice(["crash", "timeout"]), "metric": None, "by_len": None, "seed": 0}
        else:
            summ = self.fake_summary(q, 0)
            if rng.random() < 0.10:
                summ["heldout_greedy"] = round(0.80 + 0.04 * rng.random(), 3)   # below the floor
        with open(os.path.join(run_dir, "summary.json"), "w") as f:
            json.dump(summ, f)
        cat = rng.choice(list(VOCAB))
        desc = "%s %s" % (rng.choice(VOCAB[cat]), rng.choice(["variant a", "with warm restart", "halved", "doubled", "per length"]))
        return rl, os.path.join(run_dir, "summary.json"), desc, cat, q, summ

    def fake_summary(self, q, seed):
        rng = self.rng
        m = max(0, min(4000, round(400 + 300 * q + rng.gauss(0, 8))))
        a = m // 4
        fr = max(0, round(m * 0.15 + rng.gauss(0, 3)))
        return {"status": "ok", "metric": m, "by_len": {"8": a, "9": a, "10": a, "11": m - 3 * a},
                "frontier": fr, "frontier_by_len": {"12": fr - fr // 3, "13": fr // 3 - fr // 6, "14": fr // 6},
                "heldout_greedy": round(0.86 + 0.06 * rng.random(), 3), "seed": seed}

    def fake_replicate(self, job, idx):
        """The auditor's second-seed run of job['node'] (the node's rl.py, unmodified)."""
        rng = self.rng
        run_dir = os.path.join(self.work, "auditor", "runs", "rep%03d" % idx)
        os.makedirs(run_dir, exist_ok=True)
        if rng.random() < 0.1:
            summ = {"status": "crash", "metric": None, "seed": job["seed"]}
        else:
            summ = self.fake_summary(self.quality[job["node"]], job["seed"])
        summ["shas"] = {"rl.py": sha(job["node_rl"])}
        path = os.path.join(run_dir, "summary.json")
        with open(path, "w") as f:
            json.dump(summ, f)
        return path, summ

    def check_next(self, before, worker, out):
        """Checks one `next` answer against the state it was computed from."""
        fr = before["frontier"]
        others = {w: a for w, a in before["inflight"].items() if w != worker}
        if out["action"] == "crossover":
            assert out["parent2"] and out["parent2"] != out["parent1"], out
            assert out["parent1"] in fr and out["parent2"] in fr, out
        if len(fr) == C.P:
            ops = [c["operator"] for c in before["completed"]]
            ops += [a["action"] for a in sorted(others.values(), key=lambda a: a["step"])]
            fresh = [f for f in before["fresh_elites"] if f in fr]
            due = sum(1 for o in ops[-4:] if o == "crossover") < 1
            if fresh:
                assert (out["action"] == "crossover" and fresh[0] in (out["parent1"], out["parent2"])) or any(("v6" in n or "not recombined" in n) for n in out["notes"]), (fresh, out)
            elif due:
                assert out["action"] == "crossover" or any(("v6" in n or "v8" in n) for n in out["notes"]), (ops[-4:], out)
        # pending (unconfirmed) nodes are never parents
        pend = {p["node"] for p in before.get("pending", [])}
        assert out["parent1"] not in pend and out["parent2"] not in pend, (pend, out)
        # a p1 shared with another worker must be explained by the policy (v2 share, v8 leader, all held)
        if len(fr) >= 2:
            shared_ok = any(("sharing" in n or "v8" in n or "in flight elsewhere" in n or "already assigned" in n)
                            for n in out["notes"])
            held = {a["p1"] for a in others.values()}
            free_elites = [f for f in fr if f not in held]
            for a in others.values():
                if a["p1"] == out["parent1"] and not shared_ok and free_elites and out["action"] != "crossover":
                    self.violations.append(("double p1", worker, out["parent1"], out["notes"]))

    def check_roles(self):
        st = load_state(self.root)
        fr = [st["nodes"][i] for i in st["frontier"]]
        assert len(fr) <= C.P
        if fr:
            best = [n for n in fr if n["role"] == "BEST"]
            assert len(best) == 1, [n["role"] for n in fr]
            if st.get("best_id") in st["frontier"]:     # v9: BEST is sticky (two-seed rule)
                assert best[0]["id"] == st["best_id"], (best, st["best_id"])
            else:
                assert best[0]["metric"] == max(n["metric"] for n in fr)
            assert sum(1 for n in fr if n["role"] == "LEAN") <= 1
            assert all(n["role"] in C.ROLES for n in fr)
        for nid, n in st["nodes"].items():
            if nid not in st["frontier"]:
                assert n["role"] is None, n
            if n["status"] in ("crash", "timeout"):
                assert nid not in st["frontier"] and n["metric"] is None
            if nid in st["frontier"]:
                assert n["heldout_greedy"] is not None and n["heldout_greedy"] >= C.HELDOUT_FLOOR, n
                assert n["decision"] != "PENDING_BEST", n
            if n["decision"] == "FLOOR":
                assert nid not in st["frontier"], n
        for p in st["pending"]:
            assert p["node"] not in st["frontier"], p
        assert C.validate(C.Store(self.root), st) == [], C.validate(C.Store(self.root), st)
        # n_uses equals the number of live assignments naming the node as a parent
        uses = {}
        for a in st["assignments"]:
            if a["state"] != "released" and a["action"] != "baseline":
                for p in (a["p1"], a["p2"]):
                    if p:
                        uses[p] = uses.get(p, 0) + 1
        for nid, n in st["nodes"].items():
            assert n["n_uses"] == uses.get(nid, 0), (nid, n["n_uses"], uses.get(nid, 0))
        # meta.json mirrors the state
        for nid in st["order"]:
            with open(os.path.join(self.root, "nodes", nid, "meta.json")) as f:
                assert json.load(f) == st["nodes"][nid]

    def record(self, w, out, idx):
        rl, summ_path, desc, cat, q, summ = self.fake_run(w, out, idx)
        rec_args = ["record", "--worker", w, "--node-rl", rl, "--summary", summ_path,
                    "--desc", desc, "--category", cat, "--source", self.rng.choice(SOURCES)]
        before = load_state(self.root)
        res = call(self.root, *rec_args)
        self.quality[res["node"]] = q
        if summ["status"] != "ok":
            assert res["decision"] == summ["status"].upper() and res["role"] is None, res
        elif summ["heldout_greedy"] < C.HELDOUT_FLOOR:
            assert res["decision"] == "FLOOR" and res["role"] is None, res
        else:
            b = C.best_node(before)
            if b is not None:
                best = b["metric"]
                if summ["metric"] > best + C.margins(before, best)["improve"]:
                    assert res["decision"] == "PENDING_BEST" and res["role"] is None, (res, best)
                else:
                    assert res["decision"] != "PENDING_BEST", res
        self.log.append(("record", w, res))
        self.check_roles()
        return res

    def replicate(self, job, idx):
        path, summ = self.fake_replicate(job, idx)
        before = load_state(self.root)
        res = call(self.root, "replicate", "--node", job["node"], "--summary", path)
        self.log.append(("replicate", "auditor", res))
        if summ["status"] == "ok":
            n0 = before["nodes"][job["node"]]
            mean = (n0["metric"] + summ["metric"]) / 2
            b = C.best_node(before)
            best = b["metric"] if b is not None else None
            hg = (n0["heldout_greedy"] + summ["heldout_greedy"]) / 2
            assert res["metric"] == mean, (res, mean)
            if hg < C.HELDOUT_FLOOR:
                assert res["decision"] == "FLOOR", res
            elif best is not None and mean > best + C.margins(before, best)["improve"]:
                assert res["role"] == "BEST", (res, best)
            else:
                assert res["decision"] != "BEST", (res, best)
        self.check_roles()
        return res

    def drive(self, n_experiments=40, workers=("w1", "w2", "w3", "w4", "w5", "w6", "w7")):
        rng = self.rng
        clock = 0.0
        busy = {}   # worker -> (finish_time, next output, idx)
        issued = 0
        done = 0
        first = True
        rep_idx = 0
        for w in workers:
            before = load_state(self.root)
            out = call(self.root, "next", "--worker", w)
            if first:
                assert out["action"] == "baseline" and out["parent1"] == "n000", out
                first = False
            else:
                assert out["action"] == "mutate" and out["parent1"] == "n000", out
                assert any("baseline still running" in n for n in out["notes"]), out["notes"]
            self.check_next(before, w, out)
            self.log.append(("next", w, out))
            busy[w] = (clock + rng.uniform(10, 14), out, issued)
            issued += 1

        def poll_auditor():
            if "auditor" in busy:
                return
            job = call(self.root, "next", "--worker", "auditor")
            self.log.append(("next", "auditor", job))
            if job["action"] == "replicate":
                again = call(self.root, "next", "--worker", "auditor")
                assert again["node"] == job["node"] and any("already assigned" in n for n in again["notes"]), again
                busy["auditor"] = (clock + rng.uniform(10, 14), job, None)
            else:
                assert job["action"] == "idle", job

        poll_auditor()
        while done < n_experiments or "auditor" in busy:
            w = min(busy, key=lambda k: busy[k][0])
            clock, out, idx = busy.pop(w)
            if w == "auditor":
                self.replicate(out, rep_idx)
                rep_idx += 1
                poll_auditor()
                continue
            self.record(w, out, idx)
            done += 1
            if issued < n_experiments:
                before = load_state(self.root)
                nxt = call(self.root, "next", "--worker", w)
                self.check_next(before, w, nxt)
                if len(before["completed"]) >= C.EXPLORATION_PHASE:
                    assert nxt["lambda"] == C.LAMBDA0 / 2 and nxt["gamma"] == C.GAMMA0 / 2
                else:
                    assert nxt["lambda"] == C.LAMBDA0
                self.log.append(("next", w, nxt))
                busy[w] = (clock + rng.uniform(10, 14), nxt, issued)
                issued += 1
            poll_auditor()
        # drain whatever is still in flight, then every pending replicate
        for w, (_, out, idx) in sorted(busy.items()):
            self.record(w, out, idx)
        for _ in range(20):
            job = call(self.root, "next", "--worker", "auditor")
            if job["action"] == "idle":
                break
            self.replicate(job, rep_idx)
            rep_idx += 1


class TestController(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="gear_test_")
        cls.root = os.path.join(cls.tmp, "work")
        seed = os.path.join(cls.tmp, "seed_rl.py")
        with open(seed, "w") as f:
            f.write("def run(model, tok, targets, budget_secs, device, seed, log, metrics, paths):\n    return model\n")
        cls.seed = seed
        out = call(cls.root, "init", "--seed-rl", seed)
        assert out["ok"] and out["node"] == "n000"
        cls.sim = Sim(cls.root, seed=7)
        cls.sim.drive(n_experiments=40)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    # --- simulation outcomes
    def test_01_baseline_first(self):
        ev = self.sim.log[0]
        self.assertEqual(ev[0], "next")
        self.assertEqual(ev[2]["action"], "baseline")
        st = load_state(self.root)
        self.assertEqual(st["completed"][0]["operator"] in ("baseline", "mutate"), True)
        self.assertEqual(st["nodes"]["n000"]["status"], "evaluated")

    def test_02_frontier_fills_and_roles(self):
        st = load_state(self.root)
        self.assertEqual(len(st["frontier"]), C.P)
        fills = [c for c in st["completed"] if c["decision"] == "FILL"]
        self.assertEqual(len(fills), C.P)
        roles = sorted(st["nodes"][i]["role"] for i in st["frontier"])
        self.assertIn("BEST", roles)
        self.assertIn("DIVERSE", roles)
        decisions = {c["decision"] for c in st["completed"]}
        self.assertTrue(decisions & {"BEST", "REPLACE_WEAKEST", "LEAN", "DIVERSE"}, decisions)

    def test_03_crossovers_happen(self):
        st = load_state(self.root)
        xo = [c for c in st["completed"] if c["operator"] == "crossover"]
        self.assertGreaterEqual(len(xo), 3, [c["operator"] for c in st["completed"]])
        forced = [e for e in self.sim.log if e[0] == "next" and e[2]["action"] == "crossover"]
        self.assertTrue(any("fresh elite" in " ".join(e[2]["notes"]) for e in forced))
        self.assertTrue(any("in the last 4 experiments" in " ".join(e[2]["notes"]) for e in forced))
        # window invariant over completed experiments after the frontier filled
        pairs = {tuple(sorted(c["parents"])) for c in xo}
        self.assertGreaterEqual(len(pairs), 2)

    def test_04_no_double_p1(self):
        self.assertEqual(self.sim.violations, [])

    def test_05_crashes_recorded(self):
        st = load_state(self.root)
        crashes = [c for c in st["completed"] if c["decision"] in ("CRASH", "TIMEOUT")]
        self.assertGreaterEqual(len(crashes), 1)
        for c in crashes:
            self.assertIsNone(c["metric"])
            self.assertNotIn(c["node"], st["frontier"])
        # every recorded node has its rl.py and summary copy
        for c in st["completed"]:
            d = os.path.join(self.root, "nodes", c["node"])
            self.assertTrue(os.path.isfile(os.path.join(d, "rl.py")))
            self.assertTrue(os.path.isfile(os.path.join(d, "summary.json")))

    def test_06_child_stats(self):
        st = load_state(self.root)
        nodes = st["nodes"]
        for nid, n in nodes.items():
            kids = [c for c in nodes.values() if nid in c["parents"]]
            self.assertEqual(n["n_children"], len(kids))
            d = [c["metric"] - n["metric"] for c in kids if c["metric"] is not None and n["metric"] is not None]
            if d:
                self.assertAlmostEqual(n["mean_child_improvement"], sum(d) / len(d))
            else:
                self.assertIsNone(n["mean_child_improvement"])

    def test_07_history_and_frontier_cli(self):
        h = call(self.root, "history")
        st = load_state(self.root)
        self.assertEqual(len(h["history"]), len(st["order"]))
        self.assertEqual(len(h["completed"]), len(st["completed"]))
        f = call(self.root, "frontier")
        self.assertEqual(len(f["frontier"]), C.P)
        self.assertEqual(len(f["table"]), C.P + 1 + len(f["pending_best"]))
        self.assertIn("front", f["table"][0])
        self.assertIn("margins", f)
        # sources and frontier counts are shown
        for r in f["frontier"]:
            self.assertIsNotNone(r["source"])
            self.assertIsNotNone(r["frontier"])
        self.assertTrue(all("[" in line for line in f["table"][1:]))
        self.assertEqual(len(h["table"]), len(st["order"]) + 1)

    def test_07b_two_seed_rule_exercised(self):
        st = load_state(self.root)
        firsts = [n for n in st["nodes"].values() if n.get("decision_first") == "PENDING_BEST"]
        self.assertGreaterEqual(len(firsts), 1, "the simulation produced no PENDING_BEST node")
        self.assertEqual(st["pending"], [])
        for n in firsts:
            if n["decision"] == "REPLICATE_FAILED":
                continue
            self.assertEqual(len(n["metric_seeds"]), 2, n)
            self.assertEqual(n["metric"], sum(n["metric_seeds"]) / 2)
            self.assertNotEqual(n["seeds"][0], n["seeds"][1])
        # every BEST promotion over an existing frontier went through a replicate
        for c in st["completed"]:
            if c["decision"] == "BEST":
                self.assertEqual(c.get("decision_first"), "PENDING_BEST", c)
        floors = [c for c in st["completed"] if c["decision"] == "FLOOR"]
        self.assertGreaterEqual(len(floors), 1)
        self.assertGreaterEqual(len(st["replicates"]), len(firsts))

    def test_08_reload_survives(self):
        before = call(self.root, "frontier")
        importlib.reload(C)
        after = C.run(["frontier", "--root", self.root])
        self.assertEqual(before, after)
        p = subprocess.run([sys.executable, os.path.join(HERE, "controller.py"), "frontier", "--root", self.root],
                           capture_output=True, text=True, check=True)
        self.assertEqual(json.loads(p.stdout)["frontier"], before["frontier"])
        p = subprocess.run([sys.executable, os.path.join(HERE, "controller.py"), "--dry-run", "--root", self.root],
                           capture_output=True, text=True)
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        self.assertTrue(json.loads(p.stdout)["ok"])

    def test_09_dry_run_writes_nothing(self):
        path = os.path.join(self.root, "gear_state.json")
        def digest():
            with open(path, "rb") as f:
                return hashlib.sha256(f.read()).hexdigest()
        h0 = digest()
        out = call(self.root, "next", "--worker", "Z", "--dry-run")
        self.assertIn(out["action"], ("mutate", "crossover"))
        self.assertEqual(h0, digest())

    def test_10_release(self):
        st0 = load_state(self.root)
        out = call(self.root, "next", "--worker", "R")
        again = call(self.root, "next", "--worker", "R")
        self.assertEqual(again["step"], out["step"])
        self.assertTrue(any("already assigned" in n for n in again["notes"]))
        st1 = load_state(self.root)
        self.assertIn("R", st1["inflight"])
        rel = call(self.root, "release", "--worker", "R")
        self.assertEqual(rel["released"]["step"], out["step"])
        st2 = load_state(self.root)
        self.assertNotIn("R", st2["inflight"])
        for nid in st0["nodes"]:
            self.assertEqual(st0["nodes"][nid]["n_uses"], st2["nodes"][nid]["n_uses"])
            self.assertEqual(st0["nodes"][nid]["last_used_step"], st2["nodes"][nid]["last_used_step"])
        self.assertEqual(st0["co_use"], st2["co_use"])
        self.assertEqual(st0["fresh_elites"], st2["fresh_elites"])
        self.assertEqual(call(self.root, "release", "--worker", "R")["released"], None)
        # the released worker can be assigned again
        out2 = call(self.root, "next", "--worker", "R")
        self.assertIn(out2["action"], ("mutate", "crossover"))
        call(self.root, "release", "--worker", "R")

    def test_11_record_override_missing_summary(self):
        rl = os.path.join(self.tmp, "manual_rl.py")
        shutil.copyfile(self.seed, rl)
        res = call(self.root, "record", "--worker", "M", "--node-rl", rl, "--summary",
                   os.path.join(self.tmp, "nope", "summary.json"), "--desc", "manual crash test",
                   "--category", "other", "--parents", "n000", "--operator", "mutate", "--source", "own idea")
        self.assertEqual(res["decision"], "CRASH")
        self.assertEqual(res["status"], "crash")
        with self.assertRaises(C.GearError):
            call(self.root, "record", "--worker", "nobody", "--node-rl", rl, "--summary", "x",
                 "--desc", "d", "--category", "other")
        with self.assertRaises(C.GearError):
            call(self.root, "record", "--worker", "M", "--node-rl", rl, "--summary", "x",
                 "--desc", "d", "--category", "architecture", "--parents", "n000")

    def test_12_concurrent_next_under_lock(self):
        root = os.path.join(self.tmp, "lockroot")
        shutil.copytree(self.root, root)
        st0 = load_state(root)
        for w in list(st0["inflight"]):
            call(root, "release", "--worker", w)
        st0 = load_state(root)
        n = C.N_WORKERS
        ctx = mp.get_context("fork")
        barrier = ctx.Barrier(n)
        q = ctx.Queue()
        procs = [ctx.Process(target=_proc_next, args=(root, "w%d" % i, barrier, q)) for i in range(n)]
        for p in procs:
            p.start()
        res = [q.get(timeout=60) for _ in procs]
        for p in procs:
            p.join(timeout=60)
        self.assertTrue(all(r[0] == "ok" for r in res), res)
        st = load_state(root)
        self.assertEqual(len(st["inflight"]), n)
        with self.assertRaises(C.GearError):   # an eighth worker is refused while seven are in flight
            call(root, "next", "--worker", "w_extra")
        aud = call(root, "next", "--worker", "auditor")   # the auditor is not a worker slot
        self.assertIn(aud["action"], ("idle", "replicate"))
        call(root, "release", "--worker", "auditor")
        steps = sorted(r[2] for r in res)
        self.assertEqual(steps, list(range(st0["assign_count"] + 1, st0["assign_count"] + n + 1)))
        added = sum(1 + (1 if r[4] else 0) for r in res)
        self.assertEqual(sum(x["n_uses"] for x in st["nodes"].values()) -
                         sum(x["n_uses"] for x in st0["nodes"].values()), added)
        # two real workers: their concurrent assignments must use different p1s
        for w in list(st["inflight"]):
            call(root, "release", "--worker", w)
        barrier2 = ctx.Barrier(2)
        procs = [ctx.Process(target=_proc_next, args=(root, w, barrier2, q)) for w in ("A", "B")]
        for p in procs:
            p.start()
        res2 = [q.get(timeout=60) for _ in procs]
        for p in procs:
            p.join(timeout=60)
        self.assertNotEqual(res2[0][3], res2[1][3], res2)
        for w in ("A", "B"):
            call(root, "release", "--worker", w)
        # churn: two processes doing next/release 25 times each, concurrently
        st1 = load_state(root)
        barrier3 = ctx.Barrier(2)
        procs = [ctx.Process(target=_proc_churn, args=(root, w, barrier3, 25, q)) for w in ("A", "B")]
        for p in procs:
            p.start()
        res3 = [q.get(timeout=120) for _ in procs]
        for p in procs:
            p.join(timeout=60)
        self.assertTrue(all(r[0] == "ok" for r in res3), res3)
        st2 = load_state(root)
        self.assertEqual(st2["assign_count"], st1["assign_count"] + 50)
        self.assertEqual(st2["inflight"], {})
        for nid in st1["nodes"]:
            self.assertEqual(st1["nodes"][nid]["n_uses"], st2["nodes"][nid]["n_uses"])
        self.assertEqual(st1["co_use"], st2["co_use"])
        self.assertEqual(C.validate(C.Store(root), st2), [])


class TestUnits(unittest.TestCase):
    def test_jaccard(self):
        self.assertEqual(C.jaccard_distance("Adam lr 3e-5", "adam LR 3e-5"), 0.0)
        self.assertEqual(C.jaccard_distance("a b", "c d"), 1.0)
        self.assertAlmostEqual(C.jaccard_distance("a b c", "a b d"), 0.5)

    def _state_with(self, members):
        st = {"nodes": {}, "order": [], "frontier": [], "fresh_elites": [], "co_use": {}}
        for nid, m, hg, desc in members:
            n = C.new_meta(nid, [], "mutate", desc, "other", "A")
            n.update({"metric": m, "heldout_greedy": hg, "status": "evaluated"})
            st["nodes"][nid] = n
            st["order"].append(nid)
            st["frontier"].append(nid)
        C.relabel(st)
        return st

    def _child(self, st, nid, m, hg, desc):
        n = C.new_meta(nid, ["x1"], "mutate", desc, "other", "A")
        n.update({"metric": m, "heldout_greedy": hg, "status": "evaluated"})
        st["nodes"][nid] = n
        st["order"].append(nid)
        return n

    def test_promotion_rules(self):
        base = [("x1", 300, 0.80, "adam lr 3e-5 grpo"), ("x2", 290, 0.85, "adam lr 1e-5 grpo"),
                ("x3", 280, 0.80, "cosine schedule warmup"), ("x4", 250, 0.70, "retention mix cap6 data")]
        st = self._state_with(base)
        self.assertEqual(st["nodes"]["x1"]["role"], "BEST")
        self.assertEqual(st["nodes"]["x2"]["role"], "LEAN")  # within the tie margin 20, heldout +0.05
        mg = C.margins(st)
        self.assertEqual((mg["improve"], mg["tie"]), (10, 20))   # best 300: max(8, round(0.6 * 17.3)) = 10
        cases = [
            (311, 0.5, "adam lr 3e-5 grpo", "BEST", "x4"),
            (309, 0.5, "adam lr 3e-5 grpo", "REPLACE_WEAKEST", "x4"),   # not > best+10 but > weakest+10
            (259, 0.5, "adam lr 3e-5 grpo", "DISCARD", None),           # weakest +9, similar desc
            (291, 0.87, "adam lr 3e-5 grpo", "REPLACE_WEAKEST", "x4"),  # > weakest+10 first
            (232, 0.5, "entropy bonus curriculum by length", "DIVERSE", "x4"),
            (229, 0.5, "entropy bonus curriculum by length", "DISCARD", None),  # 21 below weakest
        ]
        for i, (m, hg, desc, want, ev) in enumerate(cases):
            s = json.loads(json.dumps(st))
            c = self._child(s, "c%d" % i, m, hg, desc)
            d, e = C.promote(s, c)
            self.assertEqual((d, e), (want, ev), (m, hg, desc))
        # LEAN: frontier where weakest is close to the best, so only the LEAN rule applies
        st2 = self._state_with([("y1", 300, 0.80, "a b"), ("y2", 298, 0.83, "c d"),
                                ("y3", 297, 0.80, "e f"), ("y4", 296, 0.80, "g h")])
        self.assertEqual(st2["nodes"]["y2"]["role"], "LEAN")
        c = self._child(st2, "cl", 293, 0.845, "a b")
        self.assertEqual(C.promote(st2, c), ("LEAN", "y2"))
        C.relabel(st2)
        self.assertEqual(st2["nodes"]["cl"]["role"], "LEAN")
        c = self._child(st2, "cm", 293, 0.85, "a b")   # only +0.005 over the new LEAN
        self.assertEqual(C.promote(st2, c), ("DISCARD", None))

    def test_margins(self):
        st = self._state_with([("x1", 100, 0.9, "a")])
        self.assertEqual(C.margins(st)["improve"], 8)          # floor of 8
        self.assertEqual(C.margins(st, 400)["improve"], 12)    # round(0.6 * 20)
        self.assertEqual(C.margins(st, 1000)["improve"], 19)   # round(0.6 * 31.6)
        self.assertEqual(C.margins(st, 1000)["tie"], 38)
        self.assertEqual(C.margins(st, 400)["share"], 48)
        self.assertEqual(C.margins(st, 400)["floor_gap"], 84)
        self.assertEqual(C.margins(st, 400)["loss_band"], 24)

    def test_score_components(self):
        st = self._state_with([("x1", 300, 0.80, "adam lr"), ("x2", 290, 0.85, "grpo kl")])
        st.update({"assignments": [{"action": "mutate", "p1": "x1", "p2": None, "state": "done", "step": 1}],
                   "completed": [], "inflight": {}, "assign_count": 1})
        st["nodes"]["x1"].update({"n_uses": 1, "last_used_step": 1, "mean_child_improvement": 10.0})
        s1 = C.score_components(st, st["nodes"]["x1"], 2, 0.3, 0.3)
        s2 = C.score_components(st, st["nodes"]["x2"], 2, 0.3, 0.3)
        self.assertEqual(s1["recency"], -0.2)
        self.assertEqual(s1["novelty"], 0.0)
        self.assertEqual(s2["novelty"], 1.0)
        self.assertAlmostEqual(s1["productivity"], 0.5)
        self.assertGreater(s2["ucb"], s1["ucb"])


class TestTwoSeed(unittest.TestCase):
    """v9 rules through the CLI on a fresh root: two-seed BEST, demotion, held-out floor, --source, auditor."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="gear_2seed_")
        self.root = os.path.join(self.tmp, "work")
        self.seed_rl = os.path.join(self.tmp, "seed_rl.py")
        with open(self.seed_rl, "w") as f:
            f.write("def run(*a):\n    return a[0]\n")
        call(self.root, "init", "--seed-rl", self.seed_rl)
        self.k = 0

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def summ(self, metric, hg=0.9, seed=0, frontier=30, status="ok", rl=None):
        self.k += 1
        d = os.path.join(self.tmp, "runs", "r%03d" % self.k)
        os.makedirs(d)
        s = {"status": status, "metric": metric if status == "ok" else None, "seed": seed,
             "by_len": {"8": metric // 2, "9": metric - metric // 2} if status == "ok" else None,
             "frontier": frontier if status == "ok" else None, "frontier_by_len": {"12": frontier} if status == "ok" else None,
             "heldout_greedy": hg}
        if rl:
            s["shas"] = {"rl.py": sha(rl)}
        p = os.path.join(d, "summary.json")
        with open(p, "w") as f:
            json.dump(s, f)
        return p

    def child(self, w, metric, hg=0.9, desc=None, source="own idea", seed=0):
        out = call(self.root, "next", "--worker", w)
        rl = os.path.join(self.tmp, "rl_%s_%d.py" % (w, self.k))
        with open(rl, "w") as f:
            f.write("# %s %d\n" % (w, self.k))
        return call(self.root, "record", "--worker", w, "--node-rl", rl, "--summary", self.summ(metric, hg, seed=seed),
                    "--desc", desc or "change %d lr %d" % (self.k, metric), "--category", "optimizer",
                    "--source", source), out

    def replicate(self, metric, hg=0.9, status="ok", frontier=30):
        job = call(self.root, "next", "--worker", "auditor")
        self.assertEqual(job["action"], "replicate", job)
        p = self.summ(metric, hg, seed=job["seed"], frontier=frontier, status=status, rl=job["node_rl"])
        return call(self.root, "replicate", "--node", job["node"], "--summary", p), job

    def fill(self):
        out = call(self.root, "next", "--worker", "w1")
        self.assertEqual(out["action"], "baseline")
        r = call(self.root, "record", "--worker", "w1", "--node-rl", os.path.join(self.root, "nodes", "n000", "rl.py"),
                 "--summary", self.summ(400))
        self.assertEqual((r["decision"], r["role"]), ("FILL", "BEST"))
        for i, m in enumerate((395, 390, 385)):
            r, _ = self.child("w%d" % (i + 2), m, desc="distinct idea number %d kl %d" % (i, m))
            self.assertEqual(r["decision"], "FILL", r)

    def test_pending_then_confirmed_best(self):
        self.fill()
        self.assertEqual(call(self.root, "next", "--worker", "auditor")["action"], "idle")
        r, _ = self.child("w5", 430, source="https://arxiv.org/abs/2203.14465")   # > 400 + 12
        self.assertEqual(r["decision"], "PENDING_BEST")
        self.assertIsNone(r["role"])
        st = load_state(self.root)
        self.assertNotIn(r["node"], st["frontier"])
        self.assertEqual(st["nodes"][r["node"]]["source"], "https://arxiv.org/abs/2203.14465")
        f = call(self.root, "frontier")
        self.assertEqual([p["id"] for p in f["pending_best"]], [r["node"]])
        self.assertTrue(any("PENDING" in line and "arxiv" in line for line in f["table"]))
        # workers never get the pending node as a parent
        nxt = call(self.root, "next", "--worker", "w6", "--dry-run")
        self.assertNotIn(r["node"], (nxt["parent1"], nxt["parent2"]))
        self.assertIn(r["node"], nxt["pending_best"])
        res, job = self.replicate(426)
        self.assertEqual(job["seed"], 1)
        self.assertEqual(job["node"], r["node"])
        self.assertEqual((res["decision"], res["role"]), ("BEST", "BEST"))
        self.assertEqual(res["metric"], 428)
        self.assertEqual(res["metric_seeds"], [430, 426])
        self.assertEqual(res["evicted"], "n003")
        st = load_state(self.root)
        self.assertIn(r["node"], st["frontier"])
        self.assertEqual(st["pending"], [])
        c = [c for c in st["completed"] if c["node"] == r["node"]][0]
        self.assertEqual((c["decision_first"], c["decision"], c["metric_first"], c["metric"]),
                         ("PENDING_BEST", "BEST", 430, 428))
        if st["nodes"][r["node"]]["operator"] == "mutate":   # a promoted mutation still forces a crossover
            self.assertIn(r["node"], st["fresh_elites"])
        self.assertEqual(call(self.root, "next", "--worker", "auditor")["action"], "idle")
        h = call(self.root, "history")
        row = [x for x in h["history"] if x["id"] == r["node"]][0]
        self.assertEqual((row["source"], row["metric_seeds"], row["frontier"]),
                         ("https://arxiv.org/abs/2203.14465", [430, 426], 30))

    def test_pending_demoted(self):
        self.fill()
        r, _ = self.child("w5", 440)
        self.assertEqual(r["decision"], "PENDING_BEST")
        res, _ = self.replicate(380)                     # mean 410: not > 400 + 12, but > weakest 385 + 12
        self.assertEqual(res["decision"], "REPLACE_WEAKEST", res)
        self.assertNotEqual(res["role"], "BEST")          # 410 > 400, but not by the margin: n000 stays BEST
        st = load_state(self.root)
        self.assertEqual(st["nodes"]["n000"]["role"], "BEST")
        self.assertEqual(C.best_node(st)["id"], "n000")
        self.assertIn(r["node"], st["frontier"])
        # a second pending node whose mean falls to DISCARD
        r2, _ = self.child("w6", 450, desc="change lr 450 again")
        self.assertEqual(r2["decision"], "PENDING_BEST")
        res2, _ = self.replicate(330)                    # mean 390: below weakest (390) + 12, similar desc
        self.assertIn(res2["decision"], ("DISCARD", "DIVERSE"))
        self.assertNotIn(res2["decision"], ("BEST", "PENDING_BEST"))

    def test_non_best_promotions_single_seed(self):
        self.fill()
        r, _ = self.child("w5", 405)                     # beats weakest by > 12, not best by > 12
        self.assertEqual(r["decision"], "REPLACE_WEAKEST")
        self.assertNotEqual(r["role"], "BEST")           # single seed, above 400 but within the margin
        self.assertEqual(C.best_node(load_state(self.root))["id"], "n000")
        self.assertEqual(call(self.root, "next", "--worker", "auditor")["action"], "idle")

    def test_floor(self):
        self.fill()
        r, _ = self.child("w5", 600, hg=0.84)
        self.assertEqual((r["decision"], r["role"]), ("FLOOR", None))
        st = load_state(self.root)
        self.assertNotIn(r["node"], st["frontier"])
        self.assertEqual(st["pending"], [])
        self.assertEqual(st["nodes"][r["node"]]["status"], "evaluated")
        # floor also applies to the two-seed mean
        r2, _ = self.child("w6", 450, hg=0.86)
        self.assertEqual(r2["decision"], "PENDING_BEST")
        res, _ = self.replicate(450, hg=0.82)            # mean heldout 0.84
        self.assertEqual(res["decision"], "FLOOR")
        self.assertNotIn(r2["node"], load_state(self.root)["frontier"])

    def test_replicate_failures_and_guards(self):
        self.fill()
        r, _ = self.child("w5", 450)
        job = call(self.root, "next", "--worker", "auditor")
        # wrong code
        other = os.path.join(self.tmp, "other.py")
        with open(other, "w") as f:
            f.write("# different\n")
        with self.assertRaises(C.GearError):
            call(self.root, "replicate", "--node", r["node"], "--summary", self.summ(450, seed=1, rl=other))
        # same seed as the first run
        with self.assertRaises(C.GearError):
            call(self.root, "replicate", "--node", r["node"], "--summary", self.summ(450, seed=0, rl=job["node_rl"]))
        # not pending
        with self.assertRaises(C.GearError):
            call(self.root, "replicate", "--node", "n001", "--summary", self.summ(450, seed=1))
        # two crashes -> REPLICATE_FAILED
        res1, j1 = self.replicate(0, status="crash")
        self.assertEqual(res1["decision"], None)
        self.assertEqual(load_state(self.root)["pending"][0]["node"], r["node"])
        res2, j2 = self.replicate(0, status="crash")
        self.assertEqual(j2["seed"], j1["seed"] + 1)
        self.assertEqual(res2["decision"], "REPLICATE_FAILED")
        st = load_state(self.root)
        self.assertEqual(st["pending"], [])
        self.assertNotIn(r["node"], st["frontier"])
        self.assertEqual(call(self.root, "next", "--worker", "auditor")["action"], "idle")

    def test_auditor_and_source_commands(self):
        self.fill()
        with self.assertRaises(C.GearError):             # --source is required
            call(self.root, "next", "--worker", "w5")
            rl = os.path.join(self.tmp, "x.py")
            shutil.copyfile(self.seed_rl, rl)
            call(self.root, "record", "--worker", "w5", "--node-rl", rl, "--summary", self.summ(300),
                 "--desc", "d", "--category", "other")
        with self.assertRaises(C.GearError):
            call(self.root, "record", "--worker", "auditor", "--node-rl", self.seed_rl, "--summary", self.summ(300),
                 "--desc", "d", "--category", "other", "--source", "own idea")
        self.assertEqual(call(self.root, "release", "--worker", "auditor")["released"], None)
        call(self.root, "release", "--worker", "w5")
        r, _ = self.child("w5", 450)
        job = call(self.root, "next", "--worker", "auditor")
        self.assertEqual(job["action"], "replicate")
        self.assertIn("harness.py", job["command"])
        self.assertIn("--seed 1", job["command"])
        rel = call(self.root, "release", "--worker", "auditor")
        self.assertEqual(rel["released"]["node"], r["node"])
        self.assertEqual(call(self.root, "next", "--worker", "auditor")["node"], r["node"])
        self.assertTrue(C.run(["--dry-run", "--root", self.root])["ok"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
