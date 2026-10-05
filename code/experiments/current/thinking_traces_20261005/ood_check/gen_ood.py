"""CPU-only generator of candidate OOD/TTS theorems in Dan's vocabulary (4 atoms, <=3 premises, small formula depth),
each with a constructed reference ND proof. Families:
  or_reorder(n): one premise, a binary v-tree over n distinct literals; goal: another v-tree over a permutation.
                 Proof: nested Or.elim over the premise tree, Or.inl/Or.inr chain at each leaf.
  case_product(k, w): k disjunctive premises (a_i v b_i); goal: a v-tree over all 2^k conjunctions of one disjunct per
                 premise, shuffled. Proof: k nested Or.elim, an &-intro tree and an Or-intro chain per case.
Output: jsonl rows {name, family, knob, prompt, proof}."""

import itertools
import json
import random
import sys

LITS = ["P", "Q", "R", "S", "( ~ P )", "( ~ Q )", "( ~ R )", "( ~ S )"]


class Leaf:
    def __init__(self, s):
        self.s = s

    def fs(self):
        return self.s

    def depth(self):
        return 1 if self.s.startswith("(") else 0

    def leaves(self):
        return [self]


class Node:
    def __init__(self, op, a, b):
        self.op, self.a, self.b = op, a, b

    def fs(self):
        return f"( {self.a.fs()} {self.op} {self.b.fs()} )"

    def depth(self):
        return 1 + max(self.a.depth(), self.b.depth())

    def leaves(self):
        return self.a.leaves() + self.b.leaves()


def rand_tree(items, op, rng):
    """random binary tree over `items` (Leaf/Node objects) in order; split point uniform among balanced-ish cuts."""
    if len(items) == 1:
        return items[0]
    n = len(items)
    k = rng.choice(sorted({n // 2, (n + 1) // 2}))
    return Node(op, rand_tree(items[:k], op, rng), rand_tree(items[k:], op, rng))


class LB:
    def __init__(self):
        self.lines = []

    def add(self, depth, f, rule, refs=()):
        self.lines.append((depth, f, rule, list(refs)))
        return len(self.lines)

    def body(self):
        out = []
        for i, (d, f, r, refs) in enumerate(self.lines, 1):
            out.append(f"N{i} " + "| " * d + f"{f} : {r}" + "".join(f" N{x}" for x in refs) + " ;")
        return " ".join(out) + " QED"


def path_to(goal, target_fs):
    """list of (node, side) from root to the leaf whose string is target_fs (side 'a' or 'b')."""
    if isinstance(goal, Leaf):
        return [] if goal.s == target_fs else None
    for side in ("a", "b"):
        p = path_to(getattr(goal, side), target_fs)
        if p is not None:
            return [(goal, side)] + p
    return None


def ori_chain(lb, line, leaf_fs, goal, depth):
    p = path_to(goal, leaf_fs)
    assert p is not None, (leaf_fs, goal.fs())
    cur = line
    for node, side in reversed(p):
        cur = lb.add(depth, node.fs(), "ORI1" if side == "a" else "ORI2", [cur])
    return cur


def derive_or(lb, line, T, goal, depth):
    if isinstance(T, Leaf) or T.op != "v":
        return ori_chain(lb, line, T.fs(), goal, depth)
    s1 = lb.add(depth + 1, T.a.fs(), "AS")
    e1 = derive_or(lb, s1, T.a, goal, depth + 1)
    s2 = lb.add(depth + 1, T.b.fs(), "AS")
    e2 = derive_or(lb, s2, T.b, goal, depth + 1)
    return lb.add(depth, goal.fs(), "ORE", [line, s1, e1, s2, e2])


def prompt_of(prems, concl):
    return (
        "THM " + " , ".join(prems) + " SEQ " + concl + " PRF" if prems else f"THM SEQ {concl} PRF"
    )


def or_reorder(n, rng):
    lits = rng.sample(LITS, n)
    prem = rand_tree([Leaf(x) for x in lits], "v", rng)
    perm = lits[:]
    while True:
        rng.shuffle(perm)
        goal = rand_tree([Leaf(x) for x in perm], "v", rng)
        if goal.fs() != prem.fs():
            break
    lb = LB()
    l1 = lb.add(0, prem.fs(), "PR")
    derive_or(lb, l1, prem, goal, 0)
    return prompt_of([prem.fs()], goal.fs()), lb.body()


def and_tree(lb, lines_fs, depth, rng):
    """&-intro tree over (line, formula) pairs at `depth`; returns (line, tree)."""
    items = [(l, Leaf(f)) for l, f in lines_fs]

    def build(it):
        if len(it) == 1:
            return it[0]
        k = len(it) // 2
        (la, ta), (lb_, tb) = build(it[:k]), build(it[k:])
        t = Node("&", ta, tb)
        return lb.add(depth, t.fs(), "ANDI", [la, lb_]), t

    return build(items)


def case_product(k, rng):
    # k disjunctive premises over distinct literal pairs (atoms may repeat across premises only via negation)
    for _ in range(100):
        lits = rng.sample(LITS, 2 * k)
        atoms = [x.replace("( ~ ", "").replace(" )", "") for x in lits]
        if (
            len(set(atoms)) == len(atoms) or k == 3
        ):  # k=3 needs 6 literals over 4 atoms: allow sharing
            # forbid complementary literals inside one case (keeps every case consistent)
            break
    pairs = [(lits[2 * i], lits[2 * i + 1]) for i in range(k)]
    prems = [f"( {a} v {b} )" for a, b in pairs]
    cases = list(itertools.product(*pairs))

    def conj_fs(c):
        t = Leaf(c[0]) if len(c) == 1 else None
        items = [Leaf(x) for x in c]

        def build(it):
            if len(it) == 1:
                return it[0]
            h = len(it) // 2
            return Node("&", build(it[:h]), build(it[h:]))

        return build(items)

    case_trees = [conj_fs(c) for c in cases]
    order = case_trees[:]
    rng.shuffle(order)
    goal = rand_tree(
        [Leaf(t.fs()) for t in order], "v", rng
    )  # leaves are whole conjunction strings
    lb = LB()
    pl = [lb.add(0, p, "PR") for p in prems]

    def rec(i, depth, chosen):
        if i == k:
            line, t = and_tree(lb, chosen, depth, rng)
            return ori_chain(lb, line, t.fs(), goal, depth)
        a, b = pairs[i]
        s1 = lb.add(depth + 1, a, "AS")
        e1 = rec(i + 1, depth + 1, chosen + [(s1, a)])
        s2 = lb.add(depth + 1, b, "AS")
        e2 = rec(i + 1, depth + 1, chosen + [(s2, b)])
        return lb.add(depth, goal.fs(), "ORE", [pl[i], s1, e1, s2, e2])

    rec(0, 0, [])
    return prompt_of(prems, goal.fs()), lb.body()


def and_reorder(n, rng):
    lits = rng.sample(LITS, n)
    prem = rand_tree([Leaf(x) for x in lits], "&", rng)
    perm = lits[:]
    while True:
        rng.shuffle(perm)
        goal = rand_tree([Leaf(x) for x in perm], "&", rng)
        if goal.fs() != prem.fs():
            break
    lb = LB()
    l1 = lb.add(0, prem.fs(), "PR")
    have = {}

    def extract(line, T):
        if isinstance(T, Leaf):
            have[T.s] = line
            return
        extract(lb.add(0, T.a.fs(), "ANDE1", [line]), T.a)
        extract(lb.add(0, T.b.fs(), "ANDE2", [line]), T.b)

    extract(l1, prem)

    def build(G):
        if isinstance(G, Leaf):
            return have[G.s]
        la = build(G.a)
        lb_ = build(G.b)
        return lb.add(0, G.fs(), "ANDI", [la, lb_])

    build(goal)
    return prompt_of([prem.fs()], goal.fs()), lb.body()


def mixed_tree(items, rng):
    if len(items) == 1:
        return items[0]
    n = len(items)
    k = rng.choice(sorted({n // 2, (n + 1) // 2}))
    return Node(rng.choice(["&", "v"]), mixed_tree(items[:k], rng), mixed_tree(items[k:], rng))


def commute(n, rng):
    """premise: a mixed &/v tree over n literals; goal: the same tree with children swapped at a random subset of
    nodes (at least one). Proof follows the tree: &E/&I at & nodes, Or.elim + Or-intro at v nodes."""
    lits = rng.sample(LITS, n)
    T = mixed_tree([Leaf(x) for x in lits], rng)
    while True:
        swaps = {}

        def sw(t):
            if isinstance(t, Leaf):
                return t
            s = rng.random() < 0.5
            swaps[id(t)] = s
            a, b = sw(t.a), sw(t.b)
            return Node(t.op, b, a) if s else Node(t.op, a, b)

        G = sw(T)
        if G.fs() != T.fs():
            break
    lb = LB()
    l1 = lb.add(0, T.fs(), "PR")

    def derive(line, t, g, depth):
        """line holds t; returns a line (at depth) holding g = commuted t."""
        if isinstance(t, Leaf):
            return line
        s = swaps[id(t)]
        ga, gb = (g.b, g.a) if s else (g.a, g.b)  # images of t.a, t.b inside g
        if t.op == "&":
            la = derive(lb.add(depth, t.a.fs(), "ANDE1", [line]), t.a, ga, depth)
            lb_ = derive(lb.add(depth, t.b.fs(), "ANDE2", [line]), t.b, gb, depth)
            return lb.add(depth, g.fs(), "ANDI", [lb_, la] if s else [la, lb_])
        s1 = lb.add(depth + 1, t.a.fs(), "AS")
        a1 = derive(s1, t.a, ga, depth + 1)
        e1 = lb.add(depth + 1, g.fs(), "ORI2" if s else "ORI1", [a1])
        s2 = lb.add(depth + 1, t.b.fs(), "AS")
        b1 = derive(s2, t.b, gb, depth + 1)
        e2 = lb.add(depth + 1, g.fs(), "ORI1" if s else "ORI2", [b1])
        return lb.add(depth, g.fs(), "ORE", [line, s1, e1, s2, e2])

    derive(l1, T, G, 0)
    return prompt_of([T.fs()], G.fs()), lb.body()


if __name__ == "__main__":
    rng = random.Random(20261005)
    out = open(sys.argv[1], "w")
    n_per = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    rows = []
    for n in range(3, 9):
        seen = set()
        for i in range(n_per * 5):
            p, b = or_reorder(n, rng)
            if p in seen:
                continue
            seen.add(p)
            rows.append(
                dict(
                    name=f"or_reorder_n{n}_{len(seen):03d}",
                    family="or_reorder",
                    knob=n,
                    prompt=p,
                    proof=b,
                )
            )
            if len(seen) >= n_per:
                break
    for fam, fn, knobs in (
        ("and_reorder", and_reorder, range(3, 9)),
        ("commute", commute, range(3, 9)),
    ):
        for n in knobs:
            seen = set()
            for i in range(n_per * 20):
                p, b = fn(n, rng)
                if p in seen:
                    continue
                seen.add(p)
                rows.append(
                    dict(name=f"{fam}_n{n}_{len(seen):03d}", family=fam, knob=n, prompt=p, proof=b)
                )
                if len(seen) >= n_per:
                    break
    for k in (2,):
        seen = set()
        for i in range(n_per * 5):
            p, b = case_product(k, rng)
            if p in seen:
                continue
            seen.add(p)
            rows.append(
                dict(
                    name=f"case_product_k{k}_{len(seen):03d}",
                    family="case_product",
                    knob=k,
                    prompt=p,
                    proof=b,
                )
            )
            if len(seen) >= n_per:
                break
    for r in rows:
        out.write(json.dumps(r) + "\n")
    print(len(rows), "rows")
