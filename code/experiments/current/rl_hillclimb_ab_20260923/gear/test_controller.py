#!/usr/bin/env python3
"""Simulated two-worker test of controller.py (stdlib only).

    python3 test_controller.py -v

A fake "quality" is attached to every node: baseline 0, mutation = parent + N(0.03, 0.1),
crossover = max(parents) + N(0.05, 0.05). metric = round(100 + 200 * quality + N(0, 3)).
About 12 % of non-baseline runs crash or time out. Two workers run with random durations, so
their `next`/`record` calls interleave the way two GPUs would.
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
            summ = {"status": rng.choice(["crash", "timeout"]), "metric": None, "by_len": None}
        else:
            m = max(0, min(1610, round(100 + 200 * q + rng.gauss(0, 3))))
            a = m * 300 // 1610
            summ = {"status": "ok", "metric": m, "by_len": {"7": a, "8": a // 2, "9": m - a - a // 2},
                    "rl_stats": {"heldout_greedy": round(0.87 - 0.12 * rng.random(), 3)}}
        with open(os.path.join(run_dir, "summary.json"), "w") as f:
            json.dump(summ, f)
        cat = rng.choice(list(VOCAB))
        desc = "%s %s" % (rng.choice(VOCAB[cat]), rng.choice(["variant a", "with warm restart", "halved", "doubled", "per length"]))
        return rl, os.path.join(run_dir, "summary.json"), desc, cat, q, summ

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
                assert out["action"] == "crossover" and fresh[0] in (out["parent1"], out["parent2"]), (fresh, out)
            elif due:
                assert out["action"] == "crossover", (ops[-4:], out)
        if len(fr) >= 2:
            for a in others.values():
                if a["p1"] == out["parent1"]:
                    self.violations.append(("double p1", worker, out["parent1"]))

    def check_roles(self):
        st = load_state(self.root)
        fr = [st["nodes"][i] for i in st["frontier"]]
        assert len(fr) <= C.P
        if fr:
            best = [n for n in fr if n["role"] == "BEST"]
            assert len(best) == 1, [n["role"] for n in fr]
            assert best[0]["metric"] == max(n["metric"] for n in fr)
            assert sum(1 for n in fr if n["role"] == "LEAN") <= 1
            assert all(n["role"] in C.ROLES for n in fr)
        for nid, n in st["nodes"].items():
            if nid not in st["frontier"]:
                assert n["role"] is None, n
            if n["status"] in ("crash", "timeout"):
                assert nid not in st["frontier"] and n["metric"] is None
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

    def drive(self, n_experiments=22, workers=("A", "B")):
        rng = self.rng
        clock = 0.0
        busy = {}   # worker -> (finish_time, pending record args, next output)
        issued = 0
        done = 0
        first = True
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
        while done < n_experiments:
            w = min(busy, key=lambda k: busy[k][0])
            clock, out, idx = busy.pop(w)
            rl, summ_path, desc, cat, q, summ = self.fake_run(w, out, idx)
            rec_args = ["record", "--worker", w, "--node-rl", rl, "--summary", summ_path,
                        "--desc", desc, "--category", cat]
            res = call(self.root, *rec_args)
            self.quality[res["node"]] = q
            if summ["status"] != "ok":
                assert res["decision"] == summ["status"].upper() and res["role"] is None, res
            self.log.append(("record", w, res))
            done += 1
            self.check_roles()
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
        # drain whatever is still in flight
        for w, (_, out, idx) in sorted(busy.items()):
            rl, summ_path, desc, cat, q, summ = self.fake_run(w, out, idx)
            res = call(self.root, "record", "--worker", w, "--node-rl", rl, "--summary", summ_path,
                       "--desc", desc, "--category", cat)
            self.quality[res["node"]] = q
            self.log.append(("record", w, res))
            self.check_roles()


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
        cls.sim.drive(n_experiments=22)

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
        self.assertEqual(len(f["table"]), C.P + 1)

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
                   "--category", "other", "--parents", "n000", "--operator", "mutate")
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
        n = 8
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
        self.assertEqual(st["nodes"]["x2"]["role"], "LEAN")  # within 10, heldout +0.05
        cases = [
            (306, 0.5, "adam lr 3e-5 grpo", "BEST", "x4"),
            (304, 0.5, "adam lr 3e-5 grpo", "REPLACE_WEAKEST", "x4"),   # not > best+5 but > weakest+5
            (254, 0.5, "adam lr 3e-5 grpo", "DISCARD", None),           # weakest +4, similar desc
            (291, 0.87, "adam lr 3e-5 grpo", "REPLACE_WEAKEST", "x4"),  # > weakest+5 first
            (242, 0.5, "entropy bonus curriculum by length", "DIVERSE", "x4"),
            (239, 0.5, "entropy bonus curriculum by length", "DISCARD", None),  # 11 below weakest
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
