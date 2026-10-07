"""Stage 2: compose pieces into long theorems by cut (lemma chaining), with format filters.

A composite replaces one premise p of a "main" piece (Delta |- C) by a "lemma" piece (Gamma |- p)
and its minimal derivation: the theorem Gamma u (Delta - p) |- C, proved by the lemma's minimal
proof followed by the main's minimal proof (p cited from the lemma's last line), then
`ndproof.simplify` (duplicate / unused lines removed). The lemma's atoms are renamed so that its
conclusion is literally p (its other atoms are mapped injectively onto atoms of {P, Q, R, S},
uniformly at random). A lemma may itself be a composite (chains of 3+ pieces).

Format filters (the model's interface): <= 3 premises, every premise used by the reference proof, <=
4 atoms (by construction), formula depth <= --max_fdepth, prompt <= --max_prompt_tok tokens, box
depth <= --max_box, state_env.decompose(canon=True) succeeds with every state <= --max_state tokens,
and the largest canonical name <= --max_name (so that a name base of up to 32 stays inside the 64
names).
"""

from __future__ import annotations

import random
import sys

# The bytecode flag must be set before ndproof is imported, so the import stays below it.
sys.dont_write_bytecode = True
from ndproof import (  # noqa: E402
    ATOMS,
    Proof,
    atoms_of,
    cut,
    fclass,
    fdepth,
    max_box_depth,
    parse,
    rename,
    rename_proof,
    simplify,
    sink_all,
    used_premises,
)

FORK = "/scratch/cowifornia/nd-rl/think/nd-takehome"
if FORK not in sys.path:
    sys.path.insert(0, FORK)
from gen import canon_key  # noqa: E402


def lemma_renamings(lemma: Proof, target) -> list[dict]:
    """All atom maps sending lemma.concl onto `target` (same class) with the lemma's other atoms
    mapped injectively onto atoms outside the conclusion's image."""
    import itertools

    c1, a1 = fclass(lemma.concl)
    c2, a2 = fclass(target)
    if c1 != c2:
        return []
    m0 = dict(zip(a1, a2, strict=False))
    others = [a for a in _proof_atoms(lemma) if a not in m0]
    free = [a for a in ATOMS if a not in m0.values()]
    if len(others) > len(free):
        return []
    out = []
    for img in itertools.permutations(free, len(others)):
        m = dict(m0)
        m.update(zip(others, img, strict=True))
        out.append(m)
    return out


def lemma_renaming(lemma: Proof, target, rng: random.Random) -> dict | None:
    ms = lemma_renamings(lemma, target)
    return rng.choice(ms) if ms else None


def _proof_atoms(p: Proof) -> list:
    acc = []
    for f in p.prem + [p.concl]:
        atoms_of(f, acc)
    for ln in p.lines:
        atoms_of(ln.f, acc)
    return acc


def merged_premises(lem_prem: list, main_prem: list, j: int) -> list:
    prem = []
    for x in lem_prem + [q for i, q in enumerate(main_prem) if i != j]:
        if x not in prem:
            prem.append(x)
    return prem


def compose(
    lemma: Proof,
    main: Proof,
    j: int,
    rng: random.Random,
    shuffle_prem: bool = True,
    overlap: bool = False,
) -> Proof | None:
    """Cut `lemma` into premise j of `main`. The lemma's free atoms are renamed at random, or with
    overlap=True to a renaming that minimises the number of premises of the result (lemma premises
    that coincide with premises of the main are shared), ties broken at random."""
    ms = lemma_renamings(lemma, main.prem[j])
    if not ms:
        return None
    if overlap:
        scored = [
            (
                len(merged_premises([rename(x, m) for x in lemma.prem], main.prem, j)),
                rng.random(),
                k,
            )
            for k, m in enumerate(ms)
        ]
        m = ms[min(scored)[2]]
    else:
        m = rng.choice(ms)
    lem = rename_proof(lemma, m)
    prem = merged_premises(lem.prem, main.prem, j)
    if len(prem) > 3:
        return None
    if main.concl in prem:
        return None
    order = list(range(len(prem)))
    if shuffle_prem:
        rng.shuffle(order)
    c = sink_all(simplify(cut(lem, main, j, order)))
    return c


def tok_len(prompt: str) -> int:
    return len(prompt.split())


def shape(p: Proof, decompose=None) -> dict:
    """Format features of a composite (None fields when decomposition fails)."""
    prompt = p.prompt()
    out = dict(
        n_prem=len(p.prem),
        n_atoms=len(_theorem_atoms(p)),
        lines=len(p.lines),
        prompt_tok=tok_len(prompt),
        fdepth=max(fdepth(f) for f in p.prem + [p.concl]),
        proof_fdepth=max(fdepth(ln.f) for ln in p.lines),
        box_depth=max_box_depth(p),
        n_ore=sum(ln.rule == "ORE" for ln in p.lines),
        rules=sorted({ln.rule for ln in p.lines}),
        all_prem_used=len(used_premises(p)) == len(p.prem),
    )
    if decompose is not None:
        try:
            steps, toks, env = decompose(prompt, p.body(), canon=True)
            out["n_actions"] = len(steps)
            out["max_state"] = max(len(s) for s, a, h in steps)
            out["max_action"] = max(len(a) for s, a, h in steps)
            names = [int(t[1:]) for t in toks if t.startswith("n") and t[1:].isdigit()]
            out["max_name"] = max(names) if names else 0
        except Exception as e:  # noqa: BLE001
            out["decomp_err"] = f"{type(e).__name__}: {e}"[:80]
    return out


def _theorem_atoms(p: Proof) -> list:
    acc = []
    for f in p.prem + [p.concl]:
        atoms_of(f, acc)
    return acc


def class_key(p: Proof) -> str:
    return canon_key(p.thm())


def load_piece(r: dict) -> Proof:
    return parse(r["prompt"], r["proof"])
