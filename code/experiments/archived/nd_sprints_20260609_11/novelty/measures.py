"""Formal proof/theorem 'novelty / interest' measures.

- genus_proxy(proof_text): first Betti number (cyclomatic complexity) of the
  proof's citation+discharge flow graph. Tree-shaped (no lemma reuse) -> 0;
  reusing a line in several inferences -> higher. Carbone-style cheap proxy.
- mdl_bits(proof_text): gzip length of the proof token string (compressed-
  length MDL surrogate).
- min_lines / provable: via the bounded prover.
- lemma_value(thm, target_set): how many OTHER targets become provable, or get
  a strictly shorter min proof, when `thm` is added as a premise. The
  "how many things does proving this unlock" measure.
"""
import gzip
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nd.verify import parse_proof_tokens, parse_formula
from novelty.prover import Prover


def _betti(n_nodes, edges):
    """First Betti number E - N + C of an undirected graph (C = components)."""
    if n_nodes == 0:
        return 0
    parent = list(range(n_nodes))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    uniq = set()
    for a, b in edges:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    comps = len({find(i) for i in range(n_nodes)})
    return len(edges) - n_nodes + comps


def genus_proxy(proof_text):
    """First Betti number of the citation graph (lines as nodes, each cited
    ref an edge). Includes box-start (discharged-assumption) edges since those
    are in the rule's ref list."""
    toks = proof_text.split()
    if 'PRF' in toks:
        toks = toks[toks.index('PRF') + 1:]
    try:
        lines = parse_proof_tokens(toks)
    except Exception:
        return None
    if not lines:
        return None
    idx0 = lines[0]['idx']
    n = len(lines)
    edges = []
    for ln in lines:
        for r in ln['refs']:
            a = ln['idx'] - idx0
            b = r - idx0
            if 0 <= b < n:
                edges.append((a, b))
    return _betti(n, edges)


def mdl_bits(proof_text):
    """Compressed-length surrogate: gzip byte length of the proof body tokens."""
    toks = proof_text.split()
    if 'PRF' in toks:
        toks = toks[toks.index('PRF') + 1:]
    body = ' '.join(toks).encode()
    return len(gzip.compress(body, 9))


def parse_seq(prompt):
    toks = prompt.split()
    i = 1
    prem = []
    if toks[i] != 'SEQ':
        while True:
            f, i = parse_formula(toks, i)
            prem.append(f)
            if toks[i] == ',':
                i += 1
                continue
            break
    concl, _ = parse_formula(toks, i + 1)
    return prem, concl


def lemma_value(thm_premises, thm_concl, targets, prover=None, max_depth=6,
                budget_depth=4):
    """How many of `targets` (list of (premises, concl)) get unlocked or
    shortened when `thm_concl` is added as a premise.

    Uses a SHALLOW budget (budget_depth) so 'value' = makes targets reachable
    within a small search — i.e. the lemma does real work, not 'eventually
    provable anyway at huge depth'.

    Returns dict: unlocked (newly provable), shortened (min-lines drop),
    n_targets_scored.
    """
    P = prover or Prover(max_depth=max_depth, max_fsize=20)
    unlocked = 0
    shortened = 0
    scored = 0
    for (tp, tc) in targets:
        if tc == thm_concl:
            continue
        base = P.min_proof(tp, tc, max_depth=budget_depth)
        P.memo = {}
        withl = P.min_proof(list(tp) + [thm_concl], tc, max_depth=budget_depth)
        P.memo = {}
        scored += 1
        if base is None and withl is not None:
            unlocked += 1
        elif base is not None and withl is not None and withl[0] < base[0]:
            shortened += 1
    return {'unlocked': unlocked, 'shortened': shortened,
            'n_targets_scored': scored}
