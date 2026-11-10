"""CPU checks of fr.common on x: replay, usefulness marks, hindsight lemmas, Lean."""

import gzip, json, os, random, sys, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fr.common import *
from fr.common import patterns
from state_env import split_actions, canonicalise

X = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
rows = []
for l in gzip.open(X, "rt"):
    rows.append(json.loads(l))
random.Random(0).shuffle(rows)
rows = rows[:N]
t0 = time.time()
items = []
n_steps = 0
lem = []
maxlen = 0
maxact = 0
for r in rows:
    assert thm_to_prompt(r["thm"]) == r["prompt"], (r["thm"], r["prompt"])
    prem, concl = parse_prompt(r["prompt"])
    assert make_prompt(prem, concl) == r["prompt"]
    pairs, text = nd_pairs(r["prompt"], r["proof"])
    acts = [a for _, a in pairs]
    # every step passes the step checker on a fresh env
    env = Env(r["prompt"], canon=True)
    for s, a in pairs:
        assert env.state_tokens() == s
        assert try_step(env, a) == "ok", (r["prompt"], a)
        maxlen = max(maxlen, len(s) + len(a))
        maxact = max(maxact, len(a))
    assert env.done
    tr = Trace(r["prompt"], acts, 0)
    assert tr.done and tr.text() == text
    u = tr.useful_for_proof()
    # x proofs: are all steps useful? count
    n_steps += len(acts)
    pn = pruned_nd(r["proof"])
    n_have_pruned = sum(1 for ln in patterns.parse(pn) if ln["rule"] != "AS")
    n_have_useful = sum(1 for i in u if tr.steps[i]["kind"] != "exact")
    assert n_have_pruned == n_have_useful, (n_have_pruned, n_have_useful, r["proof"])
    items.append((r["prompt"], text))
    # hindsight lemmas on every top-level have: replay, prune, and queue for Lean
    for k in range(len(tr.top_haves)):
        p2, acts2, useful2, end = tr.lemma(k)
        t2 = Trace(p2, acts2, 0)
        assert t2.done, (p2, acts2)
        nd2 = pruned_nd(t2.env.nd())
        pairs2, text2 = nd_pairs(p2, nd2)
        nh = sum(1 for _, a in pairs2 if a[0] != "exact")
        nu = sum(1 for i in useful2 if tr.steps[i]["kind"] != "exact")
        assert nh == nu, (nh, nu)
        lem.append((p2, text2))
        lem.append((p2, t2.text()))
print(
    "replay ok",
    len(rows),
    "proofs",
    n_steps,
    "steps",
    f"{time.time() - t0:.1f}s",
    "max pair len",
    maxlen,
    "max act",
    maxact,
    "lemmas",
    len(lem),
)
# a wrong step is flagged invalid
env = Env("THM P , ( P > Q ) SEQ Q PRF", canon=True)
assert try_step(env, "have n1 : P := h1 ;".split()) == "ok"
assert try_step(env, "have n2 : ( P → Q ) := h2 ;".split()) == "ok"
assert try_step(env, "have n3 : Q := n1 n2 ;".split()) == "invalid"
assert try_step(env, "have n3 : Q := n2 n9 ;".split()) == "syntax"
assert try_step(env, "have n3 : Q := n2 n1 ;".split()) == "ok"
assert try_step(env, "exact n3".split()) == "ok" and env.done
t0 = time.time()
ok = lean_check(items)
print("lean on x proofs:", sum(ok), "/", len(ok), f"{time.time() - t0:.1f}s")
assert all(ok)
t0 = time.time()
random.Random(1).shuffle(lem)
lem = lem[:4000]
ok = lean_check(lem)
print("lean on hindsight lemmas:", sum(ok), "/", len(ok), f"{time.time() - t0:.1f}s")
assert all(ok)
bad = lean_check([("THM P SEQ Q PRF", "have n1 : P := h1 ; have n2 : Q := n1 ; exact n2")])
assert bad == [False]
print("TEST_OK")
